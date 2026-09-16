# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Query-independent scorer with a *causal depthwise-convolution* readout over the recent past:
``s_j = w_out . phi(W_in norm(h_j) + W_a norm(conv(z)_j)) + j * eps``, where ``z = W_cin norm(h)``
and ``conv`` reads ``z_{j-K} .. z_{j-1}`` only.

Why this arm exists
-------------------
It is one of two *bracketing* arms for the claim that the scorer's **structure** is not the
bottleneck (:mod:`~.rnn_indexer` is the other). The question all three answer is: does letting a
per-key score read the tokens *before* ``j``, rather than only ``h_j``, buy anything?

The space is already bounded from above. :class:`~.prefix_indexer.PrefixIndexer` gives the score
softmax attention over ``j``'s **entire** prefix -- strictly more expressive than a ``K``-tap
convolution -- and at a matched objective it *lost* to the token-local scalar arm (RULER 8K
73.45 against 73.71, with 2.5x the parameters). So this arm is not expected to win. It is here to
establish that the negative result is not an artifact of prefix attention's own failure mode:

* **Variance collapse.** ``softmax(...)V`` is a convex combination of ``{v_i}_{i<=j}``, so the
  prefix readout drifts toward ``mean(v)`` and its spread *across* ``j`` shrinks with position
  (see :mod:`~.prefix_indexer`'s docstring). A convolution has learnable **signed** taps and a
  fixed receptive field, so it has no such attractor -- nothing forces ``a_j`` into a convex hull
  and nothing makes late keys mutually less distinguishable.
* **Locality.** The one prefix-neighbour signal that measured *positive* in the audit was local
  and sign-flipped (``nn_novelty_neg``: keep the keys whose earlier neighbours already point the
  same way). A ``K``-tap conv is the natural hypothesis class for exactly that signal, and the
  cheapest one that contains it.

So: if a local, signed, variance-stable history readout also fails, "structure does not matter"
stops being one architecture's negative result and becomes a property of the problem.

Strictly past, by construction
------------------------------
The conv reads ``z_{j-K} .. z_{j-1}`` and **excludes ``z_j``**. This is deliberate and it is what
makes the A/B interpretable: ``W_in norm(h_j)`` already carries the current token, so a conv that
also read tap ``0`` would be partly re-deriving a feature the baseline has. Excluding it makes the
branch's entire contribution *past* information, so ``||w_a||`` after training is a direct readout
on whether past information earned its place. ``exclude_self=False`` restores the ordinary
``t-K+1 .. t`` window as an ablation.

Cost, and why this is the cheap end of the bracket
--------------------------------------------------
``O(L * K * conv_dim)`` compute, **no state that grows with the sequence**. Decode needs only the
last ``K`` projected inputs -- ``conv_dim * K`` numbers per layer, a ring buffer, against the
prefix arm's ``O(L)`` K/V cache and ``O(t)`` per-step attention. That keeps the arm inside the
``O(1)``-decode budget :mod:`~.scalar_indexer` exists to defend, which the prefix arm does not.

Superset by construction, so the A/B is single-variable
-------------------------------------------------------
Subclasses :class:`~.scalar_indexer.ScalarIndexer`, reuses its parameters under the same names,
and adds only the conv branch. With ``w_a`` zero-initialized (the default) the score is
**bit-identical** to the scalar arm's, so "read the recent past" is the only variable. The
zero-init point is an escapable one-step saddle for the same reason it is in the prefix arm:
``dL/dW_a = dL/dz (x) norm(a)`` is nonzero even where ``dL/da = W_a^T dL/dz`` vanishes.

``decay`` works unchanged, because :meth:`ConvIndexer._trunk` overrides the trunk rather than
``score_keys``: ``w_decay`` consumes the trunk, so ``log_beta = -softplus(w_decay(trunk))`` is the
same per-(token, KV head) lifetime read off a representation that now also sees the recent past.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.presses.gqa_indexer.indexer import IndexerNorm
from kvpress.presses.gqa_indexer.scalar_indexer import ScalarIndexer, ScalarIndexerConfig

#: Default receptive field, in tokens. 8 is deliberately short: the signal this arm is built to
#: contain (``nn_novelty_neg``) is a *local* neighbour effect, and a short kernel keeps the branch
#: cheap enough that a negative result cannot be blamed on the decode budget.
DEFAULT_CONV_KERNEL = 8

#: Default width of the conv's channel space. Matches the scalar arm's ``mid_dim`` default so the
#: branch is neither starved nor dominant relative to the ``W_in`` path it is added to.
DEFAULT_CONV_DIM = 256


@dataclass
class ConvIndexerConfig(ScalarIndexerConfig):
    """
    Shape configuration for :class:`ConvIndexer`.

    Extends :class:`~.scalar_indexer.ScalarIndexerConfig` with the conv branch's geometry and
    inherits everything else unchanged -- the shared parameters are the same modules under the
    same names, which is what makes the scalar arm a reachable special case.

    Attributes
    ----------
    conv_kernel : int
        Receptive field in tokens. The branch reads ``K`` taps; with :attr:`exclude_self` those
        are ``j-K .. j-1``, otherwise ``j-K+1 .. j``. Also the size of the decode ring buffer.
    conv_dim : int
        Channel width of the conv. The conv is **depthwise** (``groups=conv_dim``), so it holds
        ``conv_dim * conv_kernel`` weights rather than ``conv_dim^2 * conv_kernel``; capacity is
        meant to come from ``conv_dim`` and the surrounding pointwise maps, which is the same
        division of labour as in a Mamba-style block.
    exclude_self : bool
        Drop tap ``0`` so the branch reads *only* positions before ``j``. On by default; see the
        module docstring on why this is what makes ``||w_a||`` interpretable.
    zero_init_conv : bool
        Zero-initialize ``w_a`` so training starts exactly at the scalar arm and the conv branch
        is the only variable.
    """

    conv_kernel: int = DEFAULT_CONV_KERNEL
    conv_dim: int = DEFAULT_CONV_DIM
    exclude_self: bool = True
    zero_init_conv: bool = True

    #: Inherited as ``init=False``; restated only so this class's field order is well-defined.
    #: A convolution over hidden states has no rotary width -- ``h`` already carries the
    #: backbone's rotary signal and the recency prior is ``pos_slope``'s job.
    rope_dim: int = field(default=0, init=False)

    def __post_init__(self):
        super().__post_init__()
        for name in ("conv_kernel", "conv_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")


class ConvIndexer(ScalarIndexer):
    """
    One score per key, from that key and a ``K``-tap causal window before it.

    ``score_keys`` returns ``(B, n_heads, Sk)`` in fp32 -- the same contract as
    :class:`~.scalar_indexer.ScalarIndexer` -- and ``forward``, ``expand_to_pairs``, ``gate_key``,
    ``gate_query``, ``project_q``, ``project_k`` and ``require_gate_scale`` are all inherited
    unchanged, because every one of them is written against the trunk.

    Irreversibility is preserved, which is what makes eviction safe: ``a_j`` depends only on
    ``z_{<j}``, so ``s_j`` is fixed the moment ``j`` arrives and no later query can revise it.
    """

    #: The conv reads only the key's own causal window -- no query enters it. This is the
    #: attribute that routes callers onto the deadline path in :mod:`~.qi_flex_attention`, so it
    #: is restated on the class where a reader will look for it.
    is_query_independent = True

    def __init__(self, config: ConvIndexerConfig):
        super().__init__(config)
        self.conv_kernel = config.conv_kernel
        self.conv_dim = config.conv_dim
        self.exclude_self = config.exclude_self

        self._cache_enabled = False
        #: Trailing ``conv_kernel`` columns of ``z``, ``(B, conv_dim, <=K)``. Bounded, unlike the
        #: prefix arm's K/V cache: this is the whole reason the arm keeps ``O(1)`` decode.
        self._cache_z: torch.Tensor | None = None
        self._cache_len = 0

        self.w_cin = nn.Linear(config.hidden_size, config.conv_dim, bias=False)
        # Depthwise: one length-K filter per channel. bias=False because a_norm would absorb it.
        self.conv = nn.Conv1d(
            config.conv_dim,
            config.conv_dim,
            kernel_size=config.conv_kernel,
            groups=config.conv_dim,
            bias=False,
        )

        # Normalizes a_j across channels before the readout, so the branch's magnitude stays
        # comparable to the W_in path regardless of how h's scale drifts with depth.
        self.a_norm = IndexerNorm(config.conv_dim, eps=config.norm_eps)

        readout_width = config.mid_dim if config.mid_dim else config.hidden_size
        self.w_a = nn.Linear(config.conv_dim, readout_width, bias=False)
        if config.zero_init_conv:
            nn.init.zeros_(self.w_a.weight)

    # ------------------------------------------------------------------
    # Decode-time ring buffer
    # ------------------------------------------------------------------
    def enable_cache(self) -> None:
        """
        Start carrying the trailing ``conv_kernel`` columns of ``z`` across calls.

        Required for decode and chunked prefill: the branch reads ``K`` taps before ``j``, so a
        suffix scored without the previous chunk's tail would silently see zeros there and the
        score would depend on how the prefill was split. Unlike the prefix arm's cache this is
        ``O(K)``, not ``O(L)``.
        """
        self._cache_enabled = True
        self._cache_z = None
        self._cache_len = 0

    def disable_cache(self) -> None:
        self._cache_enabled = False
        self._cache_z = None
        self._cache_len = 0

    @property
    def cached_length(self) -> int:
        """Absolute number of keys consumed so far, i.e. the next valid ``key_offset``."""
        return self._cache_len

    def conv_readout(self, x: torch.Tensor) -> torch.Tensor:
        """
        The causal depthwise convolution, ``(B, Sk, conv_dim)``.

        Parameters
        ----------
        x : torch.Tensor
            ``in_norm(h)``, ``(B, Sk, hidden_size)`` -- already normalized and shared with the
            readout's own ``W_in`` path.

        Notes
        -----
        Exactness across a chunk boundary is the whole job here. Writing ``T`` for the number of
        cached tail columns, the conv runs over ``cat([tail, z])`` and only the last ``Sk``
        outputs are kept, so every kept output sees its true taps whenever ``T >= K - 1``. At a
        genuine sequence start ``T = 0`` and the left zero-padding is *correct* rather than
        approximate -- there is no history to see. The intermediate case ``0 < T < K`` only arises
        when the cache holds the entire history, so it is exact too.

        With :attr:`exclude_self` the stream is shifted right by one before the convolution, so
        tap ``0`` lands on ``z_{j-1}``. The shift is applied to the *concatenated* stream, which
        is why the tail cache keeps ``K`` columns rather than ``K - 1``.
        """
        bsz, k_len, _ = x.shape
        z = self.w_cin(x).transpose(1, 2)  # (B, conv_dim, Sk)

        tail = self._cache_z if self._cache_enabled else None
        full = z if tail is None else torch.cat([tail, z], dim=2)

        if self._cache_enabled:
            # Keep the last K columns for the next call, detached: the cache is inference state,
            # and holding it in the graph across calls would retain the whole decode history.
            self._cache_z = full[:, :, -self.conv_kernel :].detach()
            self._cache_len += k_len

        stream = full
        if self.exclude_self:
            # Shift right by one so the newest tap is z_{j-1}. The leading zero is only ever read
            # by outputs we discard, except at a true sequence start where it is correct.
            stream = torch.cat(
                [torch.zeros_like(full[:, :, :1]), full[:, :, :-1]], dim=2
            )

        padded = nn.functional.pad(stream, (self.conv_kernel - 1, 0))
        out = self.conv(padded)  # (B, conv_dim, stream_len)
        return out[:, :, -k_len:].transpose(1, 2)  # (B, Sk, conv_dim)

    def _trunk(
        self,
        hidden_states: torch.Tensor,
        *,
        key_offset: int = 0,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        The scalar trunk plus this arm's conv term, ``(B, Sk, readout_width)``.

        Overriding the trunk rather than ``score_keys`` is what keeps the whole indexer
        consistent: :meth:`~.scalar_indexer.ScalarIndexer._score_and_decay` is the single path
        behind ``score_keys``, ``score_at``, ``gate_key`` and the deadline readouts, so an
        override placed only on ``score_keys`` would leave every other caller computing the
        *scalar* score for a conv checkpoint.

        The ``key_offset`` guards live here, not in ``score_keys``, so that ``gate_key`` -- which
        the gate path calls directly, bypassing ``score_keys`` -- is protected too.

        ``mask`` is applied by the caller (:meth:`~.scalar_indexer.ScalarIndexer._score_and_decay`
        masks scores to ``MASK_NEG``) and is *not* used to exclude padded taps from the
        convolution. That is a deliberate scope limit: the training path packs full-length
        documents rather than padding, and a masked depthwise conv would need a renormalization
        per tap that has no counterpart at decode. Passing a padding mask with a non-trivial
        pattern therefore raises rather than scoring something subtly different from what decode
        would produce.
        """
        if key_offset != 0 and not self._cache_enabled:
            raise ValueError(
                f"ConvIndexer needs key_offset=0, got {key_offset}. The score reads a "
                f"{self.conv_kernel}-tap window before each key, so a suffix scored without the "
                "previous chunk's tail would see zeros there -- a silent dependence on how the "
                "prefill was chunked. Call enable_cache() to carry the tail across calls."
            )
        if self._cache_enabled and key_offset != self._cache_len:
            raise ValueError(
                f"key_offset={key_offset} but the conv cache has consumed {self._cache_len} "
                "keys. The offset is the absolute position of the first new key, so a mismatch "
                "means the cache and the sequence have diverged."
            )
        if mask is not None:
            keep = mask if mask.dtype == torch.bool else mask != 0
            if not bool(keep.all()):
                raise ValueError(
                    "ConvIndexer does not support a padding mask: padded positions would enter "
                    "their neighbours' conv window, and masking taps would need a per-tap "
                    "renormalization with no counterpart at decode. Pack sequences instead."
                )

        x = self.in_norm(hidden_states)
        a = self.a_norm(self.conv_readout(x))
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
        """The per-key score as an indexer key, ``(B, Sk, Di)``.

        Keeps the base signature in full -- ``value_states`` is accepted for the scorer protocol
        and unused, and ``key_offset`` defaults to :attr:`cached_length` so training and prefill
        (where the two agree) need not pass it. Narrowing this signature is what previously broke
        the prefix arm on both the e2e trainer and the eval path.
        """
        self._reject_rope(cos, sin)
        return self.gate_key(
            hidden_states,
            key_offset=self._cache_len if key_offset is None else key_offset,
            dtype=hidden_states.dtype,
        )

    def extra_repr(self) -> str:
        shape = f"hidden={self.config.hidden_size}, n_heads={self.n_heads}"
        shape += f", mid_dim={self.mid_dim}" if self.mid_dim else " (linear)"
        conv = f"conv(K={self.conv_kernel}, dim={self.conv_dim}, exclude_self={self.exclude_self})"
        out = f"{shape}, {conv}, pos_slope={self.pos_slope:g}"
        if self.decay:
            out += f", decay(ref={self.decay_ref:g}, init={self.config.decay_init:g})"
        return f"{out}, Di={self.idx_dim}"
