# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Query-independent scorer with a *gated recurrent state* over the past:
``s_j = w_out . phi(W_in norm(h_j) + W_a norm(S_{j-1})) + j * eps``, where

    ``S_j = g_j * S_{j-1} + (1 - g_j) * u_j``,   ``g_j = sigmoid(W_g norm(h_j))``

i.e. a per-channel, **input-dependent** forget gate over a value stream ``u_j = W_u norm(h_j)``.

Why this arm exists, and what is new about it
---------------------------------------------
This is the second *bracketing* arm for "the scorer's structure is not the bottleneck"
(:mod:`~.conv_indexer` is the other, with a bounded receptive field; this one is unbounded).

A recurrent state has already been measured on this problem and did **not** survive its own
control. ``proxy_exp/diag_state_probe.py`` fitted a readout on cached hidden states with the
state's features included, against a position-shuffled twin: pooled over 12 (width, layer) pairs
the gap was ``+0.0132, se 0.0168, t = 0.79``, sign test ``p = 0.388`` -- indistinguishable from
zero, and at L14 the *shuffled* control won at every width. A 9-parameter ``pos_only`` feature beat
every ``h+z`` configuration including a 2.1M-parameter one.

Two things that result does **not** settle, which is why this module exists:

* **It tested one parameterization.** That state was ``S <- decay * S + beta * u x^T`` with
  ``decay`` and ``beta`` **fixed hyperparameters** -- the delta rule with no learned gating. A
  per-channel input-dependent gate is a different hypothesis class: it can hold a channel open
  across thousands of tokens for one document and close it immediately for another, which a
  scalar constant cannot express. ``gate_mode="fixed"`` recovers the probed form as an ablation,
  so the two sit in one table.
* **It was never trained end-to-end.** A probe bounds what a *readout* can extract from the
  state; it does not bound what a state trained *through the LM/KL objective* will learn to carry.
  Given that this repo's headline finding is that the objective matters more than the
  architecture, refuting an architecture with a probe alone would be inconsistent.

The prior is still strongly negative, and the bracket is what makes that useful: the strictly more
expressive :class:`~.prefix_indexer.PrefixIndexer` (softmax attention over the *entire* prefix)
lost to the token-local scalar arm at a matched objective (RULER 8K 73.45 vs 73.71, 2.5x params).
An unbounded gated state failing *as well*, when trained rather than probed, turns one
architecture's negative result into a property of the problem.

The scan, and why it is not a Python loop
-----------------------------------------
``S_j = g_j S_{j-1} + (1 - g_j) u_j`` is a first-order linear recurrence, so it is an **associative
scan** and can be evaluated in ``O(log L)`` sequential steps by Blelloch doubling rather than
``L``. At 8K a per-token Python loop is ~8192 kernel launches per layer per forward -- across 36
layers that dominates the step time and would make the arm untrainable for reasons that have
nothing to do with its hypothesis class. :func:`gated_scan` does it in 13 doubling steps.

The doubling identity, for the affine map ``S_j = a_j S_{j-1} + b_j``: composing two adjacent maps
gives ``(a, b) o (a', b') = (a a', a b' + b)``, which is what the loop below applies at stride
``2^k``. Verified against an explicit sequential reference in
``tests/presses/test_gqa_indexer_conv_rnn.py``.

**Numerical note.** The recurrence is run in fp32 regardless of module dtype. With ``g`` near 1 a
bf16 product of thousands of gates loses the distinction between "held for 4000 tokens" and "held
forever" -- the exact signal the arm is meant to test -- and the scan's cumulative products are
where that would bite first.

Strictly past, by construction
------------------------------
The readout uses ``S_{j-1}``, **not** ``S_j``: the state is shifted right by one before it enters
the trunk, so the branch's contribution is entirely *past* information and ``W_in norm(h_j)``
remains the only path carrying ``h_j``. This is the same rule the probe enforced (its docstring:
"using the post-update state would leak the label through the token itself") and it is what makes
``||w_a||`` after training a direct readout on whether the past earned its place.

Irreversibility is preserved: ``S_{j-1}`` depends only on ``h_{<j}``, so ``s_j`` is fixed when
``j`` arrives and no later query revises it -- the property the eviction path and
:mod:`~.qi_flex_attention`'s deadlines rest on.

Cost
----
``O(L * state_dim)`` compute and, at decode, **one** ``state_dim`` vector per layer -- ``O(1)`` per
step, unlike the prefix arm's ``O(t)``. So this arm stays inside the decode budget
:mod:`~.scalar_indexer` exists to defend.

Superset by construction
------------------------
Subclasses :class:`~.scalar_indexer.ScalarIndexer` and adds only the state branch. With ``w_a``
zero-initialized (the default) the score is **bit-identical** to the scalar arm's, so the state is
the only variable in the comparison. ``decay`` works unchanged for the same reason as in the other
subclasses: :meth:`RNNIndexer._trunk` overrides the trunk, and ``w_decay`` consumes the trunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.presses.gqa_indexer.indexer import IndexerNorm
from kvpress.presses.gqa_indexer.scalar_indexer import ScalarIndexer, ScalarIndexerConfig

#: Default width of the recurrent state. Matches the scalar arm's ``mid_dim`` default so the
#: branch is neither starved nor dominant relative to the ``W_in`` path it is added to.
DEFAULT_STATE_DIM = 256

#: Default ``log`` of the fixed retention factor for ``gate_mode="fixed"``, chosen so the state's
#: half-life is 512 tokens -- the EMA half-life the original probe used, so the ablation is
#: matched to it rather than to a round number.
DEFAULT_FIXED_HALF_LIFE = 512.0

#: Initial bias on the forget gate. ``sigmoid(2.0) = 0.88`` gives a half-life of ~5 tokens at
#: init: short enough that the state starts as a local feature and has to *learn* to hold, rather
#: than starting saturated at ``g = 1`` where the gate's own gradient nearly vanishes.
DEFAULT_GATE_BIAS = 2.0

#: Doubling steps are capped so a pathological sequence length cannot spin forever. 2^24 tokens.
_MAX_SCAN_STEPS = 24


def gated_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solve ``S_j = a_j * S_{j-1} + b_j`` with ``S_{-1} = 0``, in ``O(log L)`` steps.

    Blelloch-style doubling on the affine maps ``(a, b)``, whose composition is
    ``(a, b) o (a', b') = (a a', a b' + b)``. Both inputs are ``(B, L, D)`` and the result is
    ``(B, L, D)`` holding ``S_0 .. S_{L-1}``; the recurrence is elementwise in the last axis, so
    every channel is an independent scalar recurrence.

    Runs in **at least** fp32: the partial products ``prod g`` are exactly where low precision
    would erase the difference between a long finite memory and an infinite one. A higher-precision
    input (fp64, as the tests use) is preserved rather than downcast -- otherwise the chunked and
    one-pass paths would differ at fp32 epsilon and the exactness test could not be written.

    Parameters
    ----------
    a : torch.Tensor
        Per-step multipliers (the forget gates), ``(B, L, D)``.
    b : torch.Tensor
        Per-step additive terms, ``(B, L, D)``.
    """
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape, got {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.dim() != 3:
        raise ValueError(f"a and b must be (B, L, D), got {tuple(a.shape)}")

    acc = torch.promote_types(torch.promote_types(a.dtype, b.dtype), torch.float32)
    a = a.to(acc)
    x = b.to(acc)
    length = a.shape[1]
    stride = 1
    steps = 0
    while stride < length:
        if steps > _MAX_SCAN_STEPS:  # pragma: no cover - guards a pathological length
            raise RuntimeError(f"gated_scan exceeded {_MAX_SCAN_STEPS} doubling steps")
        # Shift by `stride`, zero-filling the head: positions before `stride` have no partner to
        # compose with yet, and (a=0, b=0) is the identity for "nothing to add".
        a_sh = nn.functional.pad(a[:, :-stride], (0, 0, stride, 0))
        x_sh = nn.functional.pad(x[:, :-stride], (0, 0, stride, 0))
        x = x + a * x_sh
        a = a * a_sh
        stride *= 2
        steps += 1
    return x


@dataclass
class RNNIndexerConfig(ScalarIndexerConfig):
    """
    Shape configuration for :class:`RNNIndexer`.

    Attributes
    ----------
    state_dim : int
        Width of the recurrent state, and of the decode-time carry.
    gate_mode : str
        ``"learned"`` -- per-channel input-dependent gate ``g_j = sigmoid(W_g norm(h_j) + bias)``.
        This is the arm's actual hypothesis.

        ``"fixed"`` -- a single learnable scalar retention shared by every channel and every
        token, initialized to :data:`DEFAULT_FIXED_HALF_LIFE`. This is the **ablation matched to
        the probed form** (``proxy_exp/diag_state_probe.py``'s EMA, whose half-life was 512), so
        "learned gating changes the answer" and "recurrence per se does not help" can be
        separated rather than confounded.
    gate_bias : float
        Initial bias on the learned gate's logit. See :data:`DEFAULT_GATE_BIAS` on why this is
        small rather than the large value a "start by remembering everything" init would use.
    fixed_half_life : float
        Initial half-life in tokens for ``gate_mode="fixed"``.
    zero_init_state : bool
        Zero-initialize ``w_a``, so training starts exactly at the scalar arm and the state branch
        is the only variable.
    """

    state_dim: int = DEFAULT_STATE_DIM
    gate_mode: str = "learned"
    gate_bias: float = DEFAULT_GATE_BIAS
    fixed_half_life: float = DEFAULT_FIXED_HALF_LIFE
    zero_init_state: bool = True

    #: Inherited as ``init=False``; restated only so this class's field order is well-defined.
    rope_dim: int = field(default=0, init=False)

    def __post_init__(self):
        super().__post_init__()
        if self.state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {self.state_dim}")
        if self.gate_mode not in ("learned", "fixed"):
            raise ValueError(
                f"gate_mode must be 'learned' or 'fixed', got {self.gate_mode!r}"
            )
        if self.fixed_half_life <= 0:
            raise ValueError(
                f"fixed_half_life must be positive, got {self.fixed_half_life}: it is a half-life "
                "in tokens and sets the initial retention factor 0.5 ** (1 / half_life)"
            )


class RNNIndexer(ScalarIndexer):
    """
    One score per key, from that key plus a gated recurrent summary of everything before it.

    ``score_keys`` returns ``(B, n_heads, Sk)`` in fp32 -- the base contract -- and every other
    protocol method is inherited, because all of them route through the trunk.
    """

    #: No query enters the state: it is a causal function of ``h_{<j}`` alone. This is the
    #: attribute that routes callers onto the deadline path in :mod:`~.qi_flex_attention`.
    is_query_independent = True

    def __init__(self, config: RNNIndexerConfig):
        super().__init__(config)
        self.state_dim = config.state_dim
        self.gate_mode = config.gate_mode

        self._cache_enabled = False
        #: The carried state ``S_{j-1}``, ``(B, state_dim)`` in the scan's accumulation dtype.
        #: ``O(1)`` in the sequence length, which is the property that keeps this arm inside the
        #: decode budget.
        self._cache_state: torch.Tensor | None = None
        self._cache_len = 0

        self.w_u = nn.Linear(config.hidden_size, config.state_dim, bias=False)
        if config.gate_mode == "learned":
            self.w_g = nn.Linear(config.hidden_size, config.state_dim, bias=True)
            # Zero weights, biased init: every channel starts at the same half-life and
            # differentiates from there. A random init would spread retention across channels
            # before the state carries anything meaningful.
            nn.init.zeros_(self.w_g.weight)
            nn.init.constant_(self.w_g.bias, config.gate_bias)
            self.logit_retain = None
        else:
            self.w_g = None
            # One scalar, stored as a logit so the retention factor stays in (0, 1) under any
            # optimizer step -- the same reason the decay head uses -softplus rather than a clamp.
            retain = 0.5 ** (1.0 / config.fixed_half_life)
            logit = torch.logit(torch.tensor(retain, dtype=torch.float32))
            self.logit_retain = nn.Parameter(logit)

        self.a_norm = IndexerNorm(config.state_dim, eps=config.norm_eps)

        readout_width = config.mid_dim if config.mid_dim else config.hidden_size
        self.w_a = nn.Linear(config.state_dim, readout_width, bias=False)
        if config.zero_init_state:
            nn.init.zeros_(self.w_a.weight)

    # ------------------------------------------------------------------
    # Decode-time carry
    # ------------------------------------------------------------------
    def enable_cache(self) -> None:
        """
        Start carrying ``S`` across calls, for decode and chunked prefill.

        Required for the same reason as in the other history arms: without it a suffix would be
        scored against a state reset to zero, so the score would depend on how the prefill was
        split. Unlike the prefix arm this carry is a single ``state_dim`` vector.
        """
        self._cache_enabled = True
        self._cache_state = None
        self._cache_len = 0

    def disable_cache(self) -> None:
        self._cache_enabled = False
        self._cache_state = None
        self._cache_len = 0

    @property
    def cached_length(self) -> int:
        """Absolute number of keys consumed so far, i.e. the next valid ``key_offset``."""
        return self._cache_len

    def state_readout(self, x: torch.Tensor) -> torch.Tensor:
        """
        The shifted recurrent state ``S_{j-1}``, ``(B, Sk, state_dim)``.

        Parameters
        ----------
        x : torch.Tensor
            ``in_norm(h)``, ``(B, Sk, hidden_size)`` -- already normalized and shared with the
            readout's own ``W_in`` path.

        Notes
        -----
        Returns ``S_{j-1}``, not ``S_j``. The shift is applied *after* the scan, and the incoming
        carry (or zero, at a sequence start) fills row 0 -- so the boundary is exact rather than
        approximate, and row 0 of a fresh sequence correctly sees no history.
        """
        bsz = x.shape[0]
        # At least fp32 for the recurrence, but never below the module's own precision -- see
        # gated_scan. .float() here would make the chunked path differ from the one-pass path at
        # fp32 epsilon in an fp64 module.
        acc = torch.promote_types(x.dtype, torch.float32)
        u = self.w_u(x).to(acc)
        if self.gate_mode == "learned":
            g = torch.sigmoid(self.w_g(x).to(acc))
        else:
            g = torch.sigmoid(self.logit_retain.to(acc)).expand_as(u)

        # Convex form: a channel that forgets (g -> 0) takes the new input outright, and one that
        # holds (g -> 1) ignores it. Keeping the two coefficients tied means the state's scale
        # cannot drift with the gate, which is what made the probe's unnormalized outer-product
        # state diverge within ~25 steps.
        states = gated_scan(g, (1.0 - g) * u)  # (B, Sk, state_dim), fp32

        carry = self._cache_state
        if carry is not None:
            # Fold the incoming carry in: S_j gains (prod_{i<=j} g_i) * S_in. Taken with cumprod
            # rather than a second scan -- gated_scan with b = 0 returns zeros, so it cannot
            # supply this factor.
            states = states + torch.cumprod(g, dim=1) * carry.unsqueeze(1)

        if self._cache_enabled:
            self._cache_state = states[:, -1, :].detach()
            self._cache_len += x.shape[1]

        prev = torch.zeros_like(states[:, :1, :]) if carry is None else carry.unsqueeze(1)
        return torch.cat([prev, states[:, :-1, :]], dim=1).to(x.dtype)

    def _trunk(
        self,
        hidden_states: torch.Tensor,
        *,
        key_offset: int = 0,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        The scalar trunk plus this arm's state term, ``(B, Sk, readout_width)``.

        Overriding the trunk rather than ``score_keys`` keeps ``gate_key``, ``score_at`` and the
        deadline readouts consistent with ``score_keys`` -- see
        :meth:`~.scalar_indexer.ScalarIndexer._trunk`. The ``key_offset`` guards therefore live
        here too, so the gate path is protected even though it bypasses ``score_keys``.

        ``mask`` is rejected when non-trivial, for the same reason as in
        :mod:`~.conv_indexer`: a padded token would enter the state that every later token reads,
        and the training path packs documents rather than padding.
        """
        if key_offset != 0 and not self._cache_enabled:
            raise ValueError(
                f"RNNIndexer needs key_offset=0, got {key_offset}. The score reads a recurrent "
                "state over the whole past, so a suffix scored from a zero state would differ "
                "silently depending on how the prefill was chunked. Call enable_cache() to carry "
                "the state across calls."
            )
        if self._cache_enabled and key_offset != self._cache_len:
            raise ValueError(
                f"key_offset={key_offset} but the state cache has consumed {self._cache_len} "
                "keys. The offset is the absolute position of the first new key, so a mismatch "
                "means the cache and the sequence have diverged."
            )
        if mask is not None:
            keep = mask if mask.dtype == torch.bool else mask != 0
            if not bool(keep.all()):
                raise ValueError(
                    "RNNIndexer does not support a padding mask: a padded token would enter the "
                    "recurrent state that every later key reads. Pack sequences instead."
                )

        x = self.in_norm(hidden_states)
        a = self.a_norm(self.state_readout(x))
        if self.w_in is not None:
            return nn.functional.gelu(self.mid_norm(self.w_in(x) + self.w_a(a)))
        return x + self.w_a(a)

    def project_k(
        self,
        hidden_states: torch.Tensor,
        cos=None,
        sin=None,
        *,
        value_states: torch.Tensor | None = None,
        key_offset: int | None = None,
    ) -> torch.Tensor:
        """The per-key score as an indexer key, ``(B, Sk, Di)``. Keeps the base signature in full."""
        self._reject_rope(cos, sin)
        return self.gate_key(
            hidden_states,
            key_offset=self._cache_len if key_offset is None else key_offset,
            dtype=hidden_states.dtype,
        )

    def extra_repr(self) -> str:
        shape = f"hidden={self.config.hidden_size}, n_heads={self.n_heads}"
        shape += f", mid_dim={self.mid_dim}" if self.mid_dim else " (linear)"
        state = f"state(dim={self.state_dim}, gate={self.gate_mode})"
        out = f"{shape}, {state}, pos_slope={self.pos_slope:g}"
        if self.decay:
            out += f", decay(ref={self.decay_ref:g}, init={self.config.decay_init:g})"
        return f"{out}, Di={self.idx_dim}"
