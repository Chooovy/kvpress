# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Query-independent scalar indexer: ``s_t = w_out . phi(W_in h_t) + t * eps``.

The baseline arm for the O(L)-decode question. :class:`~.indexer.GQAIndexer` scores every
``(query, key)`` pair, so a decode step costs ``O(t)`` -- at 128K that is 134M MACs and 33 MB
of cache reads per layer per step, against 4M MACs for the sparse attention it feeds, i.e. the
router becomes 32x the thing it was meant to accelerate. This module drops the query axis: a
key is scored once, when it arrives, from its own hidden state alone. Decode then costs
``O(1)`` amortised and the indexer's KV cache is one scalar per token instead of ``head_dim``.

What is given up, and why the comparison is worth running
--------------------------------------------------------
Query-independence is not a weaker scorer of the same kind -- it cannot express "this query
needs *this* fact". A needle matters only to the query that asks for it, so a frozen per-key
score must either keep it always or lose it, which is the known weakness of the eviction family
(H2O, SnapKV, SparseK, DMA) against query-aware routing (Quest, DSA). The point of this arm is
to *measure* that cost on the same objective and budget as the pairwise indexer, rather than
assume it either way.

Structure
---------
Deliberately linear-or-MLP rather than the bilinear ``(W_q h) . (W_k h)`` that would mirror the
pairwise indexer's shape. With both sides fed by the same token, that form collapses to a
quadratic ``h' M h``, which is worse in three measurable ways:

* only the symmetric part of ``M`` survives (the antisymmetric part contributes 1.5e-13, i.e.
  about half the parameters are dead);
* ``score(-h) == score(h)`` exactly -- a quadratic form cannot tell a direction from its
  reverse, while ``w . h`` can. A norm *with bias* breaks the symmetry, so the bilinear form
  silently depends on that detail;
* neither reference method uses it. SparseK scores ``w . h_t + t*eps``; DMA samples the value
  vector linearly and applies a scalar gate.

Capacity, when it is wanted, comes from ``mid_dim`` (a two-layer MLP) where every parameter is
live and the count is easy to match against a competing arm.

The position slope
------------------
``t * eps`` is fixed, tiny, and not learnable. Two independent reasons:

* SparseK's (Sec. 3.2): without it the scorer is pushed to predict ever-larger values so new
  tokens can outrank old ones, which hurts training stability and length generalisation. The
  slope carries that duty so the learned part does not have to.
* It must be **absolute**, not normalised by sequence length. Verified: with ``t * eps`` a key
  that leaves the top-k never re-enters it over 1500 steps (0 returns), which is the
  irreversibility SparseK relies on to prune a key the moment it is dropped. Dividing by the
  current length makes every old key's score move as the sequence grows and irreversibility
  breaks (27 returns) -- the eviction would no longer be safe.

Relation to the gate path
-------------------------
A per-key score is the small-``Di`` case of the existing gate with the indexer query pinned to
a constant. Per head, ``Di = n_heads`` and the query is the one-hot selector that routes each
KV head to its own column; shared, ``Di = 1`` and the query is all ones. Either way
:mod:`~.gated_attention`, its Triton kernel, and the ``sink`` pin in :mod:`~.gate_pin` apply
unchanged, and end-to-end training needs no new machinery.

Pinning is still required and still works: a flat gate is a no-op for a query-independent
score just as much as for a pairwise one (measured no-op distance 0.44 with a sink pin,
5.6e-17 without).

Granularity
-----------
One score per KV head by default, because that is where GQA evicts. The eight KV heads of
Llama-3-8B agree on only 14-17% of their top-k, so they genuinely want different keys -- but
they disagree in the low-mass tail, and a single shared score still recovers 0.649 of each
head's attention mass against 0.659 for per-head at a 5% budget. Sharing therefore costs about
0.01 and saves ``n_heads``x on the score cache, which is why ``n_heads=1`` is kept as a
first-class ablation rather than removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.presses.gqa_indexer.indexer import MASK_NEG, IndexerNorm

#: Default slope. Small enough to leave content ranking intact over a 128K context
#: (total tilt 0.13 against a score of order 1), large enough to break ties by recency.
DEFAULT_POS_SLOPE = 1e-6


@dataclass
class ScalarIndexerConfig:
    """
    Shape configuration for :class:`ScalarIndexer`.

    Attributes
    ----------
    hidden_size : int
        Model hidden size; input dim of the projection.
    n_heads : int
        Scores emitted per token, one per KV head. Defaults to ``num_key_value_heads``
        semantics: the caller passes the model's KV head count and each head gets its own
        ranking, which is the granularity GQA can actually evict at.

        ``1`` shares a single ranking across every KV head, which is what SparseK and DMA do.
        That is the cheaper ablation, and its cost is small but real: measured on
        Llama-3-8B layer 16, the eight KV heads agree on only 14-17% of their top-k, yet a
        shared score still recovers 0.649 of each head's attention mass against 0.659 for
        per-head at a 5% budget. The heads disagree mostly in the low-mass tail, so sharing
        loses ~0.01 -- worth knowing before paying ``n_heads`` times the score cache.
    mid_dim : int
        Hidden width of the scoring MLP. ``0`` is the plain linear ``w . h`` of SparseK.

        This is the arm's main capacity knob, not just a parameter-matching one. In the probe
        study a nonlinear readout of ``h`` beat a linear one by +0.12 (sum) and +0.09 (late)
        held-out Spearman, which is larger than anything the recurrent state added (+0.012 to
        +0.016, and negative on two targets). Worth sweeping ``{0, 256}``; at
        ``mid_dim = pairwise_params / hidden_size`` the two arms have equal parameter counts,
        which is the only configuration that isolates query-dependence from capacity.
    norm_eps : float
        Epsilon for the input and pre-activation norms.
    pos_slope : float
        Coefficient of the fixed recency tilt ``t * eps``. ``0`` disables it, which is an
        ablation rather than a default -- see the module docstring.
    gate_scale : bool
        Create the learnable gate multiplier used by end-to-end training, mirroring
        :class:`~.indexer.GQAIndexerConfig`.
    """

    hidden_size: int
    n_heads: int
    mid_dim: int = 0
    norm_eps: float = 1e-5
    pos_slope: float = DEFAULT_POS_SLOPE
    gate_scale: bool = False

    #: Always ``0``. A per-key score has no rotary width -- there is no query to be rotated
    #: relative to, and the recency prior is carried explicitly by ``pos_slope`` instead. Kept
    #: as a field so this config answers ``rope_dim`` like :class:`~.indexer.GQAIndexerConfig`
    #: does, which is what lets the press's RoPE plumbing and the end-to-end trainer treat the
    #: two scorers through one code path.
    rope_dim: int = field(default=0, init=False)

    def __post_init__(self):
        for name in ("hidden_size", "n_heads"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.mid_dim < 0:
            raise ValueError(f"mid_dim must be non-negative, got {self.mid_dim}")
        if self.pos_slope < 0:
            raise ValueError(
                f"pos_slope must be non-negative, got {self.pos_slope}: a negative tilt "
                "favours old keys over new ones, which inverts the recency prior"
            )


class ScalarIndexer(nn.Module):
    """
    One score per key, from that key's own hidden state.

    ``forward`` returns ``(B, n_heads, Sk)`` -- no query axis, which is the whole point. The
    press and the gate both consume it by broadcasting over queries, so it slots into the
    existing ``(B, n_heads, Sq, Sk)`` interface via :meth:`expand_to_pairs` without either
    having to know which scorer produced it.

    Scores are returned in fp32 regardless of module dtype. At 32K keys a bf16 score resolves
    only ~200 distinct values out of 8192 measured, so top-k would be deciding large blocks of
    ties by index order.
    """

    #: Natural gate magnitude. The input norm leaves ``h`` at unit variance per channel, so a
    #: linear map into one output has score std of order 1 -- the same scale as a real
    #: ``q @ k / sqrt(head_dim)`` attention logit, which is what the gate has to sit alongside.
    #: No ``1/sqrt(d)`` correction is needed here because there is no ``head_dim``-long dot
    #: product to shrink, unlike :attr:`~.indexer.GQAIndexer.GATE_SCALE_INIT`. Kept as a
    #: staticmethod with the same name so callers can treat the two scorers alike.
    GATE_SCALE_INIT = staticmethod(lambda _n_heads=None: 1.0)

    #: The score does not depend on the query -- that is this module's entire premise. Callers use
    #: this to take an asymptotically cheaper selection path: because a key's score is fixed and the
    #: eligible pool only grows, each key is selected by one *contiguous interval* of query rows, so
    #: the whole support is expressible as a per-key deadline instead of a
    #: ``(B, h, Sq, topk)`` index tensor. See
    #: :mod:`~kvpress.presses.gqa_indexer.qi_flex_attention`.
    is_query_independent = True

    def __init__(self, config: ScalarIndexerConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.mid_dim = config.mid_dim
        self.pos_slope = config.pos_slope
        # Mirrors GQAIndexer.rope_dim, which the press reads off the module (not the config)
        # to decide whether to narrow the layer's RoPE tables. Always 0 here, so it passes
        # (None, None) and never looks for position_embeddings.
        self.rope_dim = config.rope_dim

        # Normalise the input, not just the MLP's pre-activation. Hidden-state norms vary by
        # two orders of magnitude across depth and carry a dominant outlier direction, so a
        # raw linear map gives a score whose std drifts with the layer: measured 0.009 on a
        # unit-norm stream against 0.887 on a norm-100 one, while the attention logits it is
        # added to stay at std ~1. Without this, GATE_SCALE_INIT would be wrong by ~100x in
        # one direction or the other and every layer would need its own value.
        self.in_norm = IndexerNorm(config.hidden_size, eps=config.norm_eps)

        if config.mid_dim:
            # phi = GELU. The second norm keeps the pre-activation where the nonlinearity is
            # informative rather than wherever the first projection's scale happens to put it.
            self.w_in = nn.Linear(config.hidden_size, config.mid_dim, bias=False)
            self.mid_norm = IndexerNorm(config.mid_dim, eps=config.norm_eps)
            self.w_out = nn.Linear(config.mid_dim, config.n_heads, bias=False)
        else:
            self.w_in = None
            self.mid_norm = None
            self.w_out = nn.Linear(config.hidden_size, config.n_heads, bias=False)

        # See GQAIndexer.gate_scale: deliberately not zero, since dL/dscore is proportional to
        # it and a zero start gives the router no gradient to leave that point with.
        self.gate_scale = (
            nn.Parameter(torch.tensor(self.GATE_SCALE_INIT())) if config.gate_scale else None
        )

    @property
    def weight_dtype(self) -> torch.dtype:
        return self.w_out.weight.dtype

    def require_gate_scale(self) -> torch.Tensor:
        """The gate multiplier, raising when this indexer was not built with one.

        Mirrors :meth:`~.indexer.GQAIndexer.require_gate_scale`, and raises for the same
        reason: silently substituting the init constant would let an end-to-end run report a
        healthy loss while training a fixed-scale ablation nobody asked for.
        """
        if self.gate_scale is None:
            raise RuntimeError(
                "this ScalarIndexer has no gate_scale parameter, so it cannot be used as an "
                "attention gate. Build it with ScalarIndexerConfig(gate_scale=True)."
            )
        return self.gate_scale

    def score_keys(
        self,
        hidden_states: torch.Tensor,
        *,
        key_offset: int = 0,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Score each key once -> ``(B, n_heads, Sk)`` in fp32.

        The natural interface for this scorer, and the ``O(L)`` one: no query axis is ever
        formed. Callers that only need per-key importance -- the press's eviction path, an
        incremental top-k during decode -- should use this rather than :meth:`forward`, which
        exists to satisfy the pairwise protocol.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Key-side hidden states, ``(B, Sk, hidden_size)``. There is no separate query-side
            input: the score does not depend on the query, which is what buys the ``O(1)``
            decode step.
        key_offset : int
            Absolute position of the first key, for the recency tilt. Non-zero during decode
            and chunked prefill, where ``hidden_states`` is a suffix of the sequence. Getting
            this wrong would restart the tilt at every chunk and make the score depend on how
            the prefill happened to be split.
        mask : torch.Tensor, optional
            Keep-mask over keys, broadcastable to ``(B, Sk)`` (``True``/non-zero = real
            token). Padding positions are set to ``MASK_NEG`` so they rank last. Causality
            needs no mask here: a key is scored from its own state, so it cannot see the
            future in the first place.

        Returns
        -------
        torch.Tensor
            ``(B, n_heads, Sk)`` fp32 scores.
        """
        if hidden_states.dim() != 3:
            raise ValueError(
                f"hidden_states must be (B, Sk, hidden_size), got {tuple(hidden_states.shape)}"
            )
        if hidden_states.dtype != self.weight_dtype:
            hidden_states = hidden_states.to(self.weight_dtype)

        x = self.in_norm(hidden_states)
        if self.w_in is not None:
            x = nn.functional.gelu(self.mid_norm(self.w_in(x)))
        scores = self.w_out(x).float()  # (B, Sk, n_heads)
        scores = scores.transpose(1, 2)  # (B, n_heads, Sk)

        if self.pos_slope:
            k_len = hidden_states.shape[1]
            pos = torch.arange(
                key_offset, key_offset + k_len, device=scores.device, dtype=scores.dtype
            )
            scores = scores + self.pos_slope * pos

        if mask is not None:
            keep = mask if mask.dtype == torch.bool else mask != 0
            scores = scores.masked_fill(~keep.view(keep.shape[0], 1, -1), MASK_NEG)
        return scores

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        key_hidden_states: torch.Tensor | None = None,
        key_cos: torch.Tensor | None = None,
        key_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Pairwise-protocol view of the score: ``(B, n_heads, Sq, Sk)``, matching
        :meth:`~.indexer.GQAIndexer.forward` argument for argument.

        Every query row is identical -- that is what query-independence means -- so this is a
        broadcast **view** of :meth:`score_keys`, not an ``O(Sq * Sk)`` computation. It exists
        so the press, the query reductions and the loss helpers run over either scorer through
        one code path; the additive ``mask`` is applied here because those callers pass the
        press's ``(B, 1, Sq, Sk)`` causal mask, which only makes sense in this layout.

        Prefer :meth:`score_keys` when the query axis is not actually needed: expanding and
        then reducing it back is wasted work, and at inference it is the whole cost this
        scorer exists to avoid.
        """
        self._reject_rope(cos, sin)
        self._reject_rope(key_cos, key_sin)
        keys = hidden_states if key_hidden_states is None else key_hidden_states
        scores = self.expand_to_pairs(self.score_keys(keys), hidden_states.shape[1])
        if mask is not None:
            scores = scores + mask.to(scores.dtype)
        return scores

    def expand_to_pairs(self, scores: torch.Tensor, q_len: int) -> torch.Tensor:
        """
        Broadcast ``(B, n_heads, Sk)`` to the ``(B, n_heads, Sq, Sk)`` pairwise layout.

        A view, not a copy -- the whole content of query-independence is that every row is the
        same. Provided so downstream code written against the pairwise indexer (query
        reduction, the loss helpers) can consume this scorer unchanged; anything that only
        needs the per-key vector should use it directly rather than expanding and reducing.
        """
        bsz, n_heads, k_len = scores.shape
        return scores.unsqueeze(2).expand(bsz, n_heads, q_len, k_len)

    def gate_key(
        self, hidden_states: torch.Tensor, *, key_offset: int = 0, dtype=None
    ) -> torch.Tensor:
        """
        The score shaped as an indexer key, ``(B, Sk, Di)`` with ``Di = n_heads``.

        Pairs with :meth:`gate_query` to drive :mod:`~.gated_attention` unchanged: the gate
        computes ``qi . ki`` over a width-``Di`` axis, and a per-key score is that product with
        the query side pinned to a constant selector.

        ``dtype`` casts the result, which :meth:`forward` deliberately returns in fp32 for
        top-k resolution. The gate wants it in the attention's dtype instead -- pass the
        model's, or the einsum against a non-fp32 query raises.
        """
        k = self.score_keys(hidden_states, key_offset=key_offset).transpose(1, 2)
        return k if dtype is None else k.to(dtype)

    def gate_query(
        self, q_len: int, bsz: int, n_kv_heads: int, *, device=None, dtype=None
    ) -> torch.Tensor:
        """
        The constant indexer query for the gate path, ``(B, n_kv_heads, Sq, Di)``.

        With per-head scores (the default) this is the one-hot selector that routes KV head
        ``h`` to its own column: ``Di = n_heads`` and each head reads only its own score. With
        a shared score it is all ones over ``Di = 1``, so the dot product just picks the score
        up.

        Not a learnable query -- that is the whole point. The gate's ``qi . ki`` becomes a
        pure lookup, which is what makes the score query-independent while still travelling
        through the existing gated-attention path unchanged.
        """
        di = self.n_heads
        if di == 1:
            return torch.ones(bsz, n_kv_heads, q_len, 1, device=device, dtype=dtype)
        if di != n_kv_heads:
            raise ValueError(
                f"per-head ScalarIndexer has n_heads={di} but the model has "
                f"{n_kv_heads} KV heads; they must match for the gate to route each head "
                f"to its own score. Build it with ScalarIndexerConfig(n_heads=<KV heads>), "
                f"or n_heads=1 for the shared-score ablation."
            )
        # expand, not repeat: the selector is the same for every batch element and query, so
        # this stays a view. At Sq = 32K and Di = 8 a materialised copy would be 8 GB in fp32.
        eye = torch.eye(di, device=device, dtype=dtype)
        return eye.view(1, di, 1, di).expand(bsz, di, q_len, di)

    # ------------------------------------------------------------------
    # GQAIndexer protocol
    # ------------------------------------------------------------------
    # The press and the end-to-end trainer reach the scorer through project_q / project_k /
    # require_gate_scale and read .rope_dim. Satisfying that protocol here -- rather than
    # teaching each caller about a second scorer type -- is what lets the two arms train and
    # evict through exactly one code path, which is the property the A/B comparison rests on.
    # The RoPE arguments are accepted and ignored: rope_dim is 0, so the press passes
    # (None, None), and a caller that passes real tables is asking for something this scorer
    # cannot do.
    def project_q(
        self, hidden_states: torch.Tensor, cos=None, sin=None, *, n_kv_heads: int | None = None
    ) -> torch.Tensor:
        """The constant gate selector, ``(B, n_heads, Sq, Di)``. Not a function of the input.

        Shaped like :meth:`~.indexer.GQAIndexer.project_q` so the gate path is shared, but it
        carries no information: query-independence means the query side is a lookup, and the
        whole score lives in :meth:`project_k`.
        """
        self._reject_rope(cos, sin)
        bsz, q_len, _ = hidden_states.shape
        return self.gate_query(
            q_len,
            bsz,
            n_kv_heads if n_kv_heads is not None else self.n_heads,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    def project_k(
        self,
        hidden_states: torch.Tensor,
        cos=None,
        sin=None,
        *,
        value_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The per-key score as an indexer key, ``(B, Sk, Di)`` in the input's dtype.

        ``key_offset`` is deliberately not exposed here: this is the training/prefill entry
        point, where ``hidden_states`` starts at position 0. Decode and chunked prefill must
        call :meth:`gate_key` with the right offset, or the recency tilt restarts per chunk.

        ``value_states`` is accepted for the scorer protocol and intentionally unused. This
        scorer is defined on hidden states; value-based scorers consume it instead.
        """
        self._reject_rope(cos, sin)
        return self.gate_key(hidden_states, dtype=hidden_states.dtype)

    def _reject_rope(self, cos, sin) -> None:
        if cos is not None or sin is not None:
            raise ValueError(
                "ScalarIndexer scores a key from its own hidden state, so there is no q/k pair "
                "to rotate and RoPE tables cannot be applied. Its config reports rope_dim=0, so "
                "the press passes (None, None); a caller supplying tables is expecting a "
                "positional signal this scorer carries through pos_slope instead."
            )

    def extra_repr(self) -> str:
        shape = f"hidden={self.config.hidden_size}, n_heads={self.n_heads}"
        shape += f", mid_dim={self.mid_dim}" if self.mid_dim else " (linear)"
        return f"{shape}, pos_slope={self.pos_slope:g}"
