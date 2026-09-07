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

import math
from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.presses.gqa_indexer.indexer import MASK_NEG, IndexerNorm


def _inv_softplus(y: float) -> float:
    """``x`` such that ``softplus(x) == y``, for seeding the decay bias.

    ``y == 0`` is the inert-decay ablation (``decay_init=0``) and is exactly the point where the
    true inverse diverges, so it is clamped to a large negative bias instead: ``softplus(-30)`` is
    9e-14, i.e. zero to every dtype in play, and unlike ``-inf`` it still has a finite gradient so
    the head can train away from it. Refusing here would reject a documented ablation.
    """
    if y < 0:
        raise ValueError(f"inverse softplus needs a non-negative target, got {y}")
    if y == 0:
        return -30.0
    return float(math.log(math.expm1(y))) if y < 20 else float(y)

#: Default slope. Small enough to leave content ranking intact over a 128K context
#: (total tilt 0.13 against a score of order 1), large enough to break ties by recency.
DEFAULT_POS_SLOPE = 1e-6

#: Age normalizer for the learned decay, in tokens. Matches the 16K training stage, so
#: ``log_beta`` reads as "nats lost over one training window of age". A CONSTANT by design --
#: see :attr:`ScalarIndexerConfig.decay_ref`.
DEFAULT_DECAY_REF = 16384.0

#: Initial ``log_beta``, in nats per :data:`DEFAULT_DECAY_REF` of age. Puts the decay term's range
#: at the score's own std (~1) so the lifetime is live from step 0, unlike TrimKV's near-zero init
#: which relies on a retention hinge this port does not have.
DEFAULT_DECAY_INIT = -1.0


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
    decay : bool
        Give each key a learned *lifetime* on top of its magnitude, TrimKV's core mechanism::

            gate_j(i) = s_j + log_beta_j * (i - j) / decay_ref

        with ``log_beta_j <= 0``, so a key's contribution decays geometrically in its age
        ``i - j``. This is a strictly larger hypothesis class than the frozen score: ``s_j``
        alone fixes a key's rank for all time, while ``log_beta_j`` lets rank *evolve* -- a
        slowly-decaying key overtakes a faster-decaying older one as the sequence grows.

        Folds into the existing bilinear gate exactly, at ``Di = 2 * n_heads`` instead of
        ``n_heads`` (see :meth:`gate_key`), so no attention kernel changes: the gate, its ``lse``
        normalizer and the whole backward run unmodified. Verified to 2.4e-07 against an explicit
        per-pair reference.
    decay_ref : float
        Age normalizer, in tokens. **A fixed constant, never the live sequence length.**

        Two independent reasons, and they point the same way:

        * *Gradient scale.* ``d(gate)/d(log_beta_j) = (i - j)``, which reaches 16384 at 16K
          against ``d(gate)/d(s_j) = 1``. Unnormalized, the lifetime head trains at ~1e4 times
          the score head's effective learning rate off one shared LR. Dividing by a constant puts
          both at O(1) and makes ``log_beta`` read in *nats per ``decay_ref`` of age*.
        * *Irreversibility.* Dividing by the **live** length instead would make every old key's
          score move as the sequence grows, which breaks the property that a key dropped from the
          top-k never returns -- the property the eviction path and
          :mod:`~.qi_flex_attention`'s deadlines rest on. Already measured on the plain tilt:
          absolute gives 0 returns over 1500 steps, length-normalized gives 27.
    decay_init : float
        Initial ``log_beta``, in the same nats-per-``decay_ref`` units. ``-1.0`` puts the decay
        term's range (0 at age 0, ``-1`` at age ``decay_ref``) at the score's own std of ~1, so
        the lifetime is a live feature from step 0.

        Deliberately **not** TrimKV's near-zero init. There, ``bias_init=18`` gives
        ``log_beta = logsigmoid(18) ~ -1.5e-8`` -- decay starts inert and the *retention hinge
        loss* is the only thing driving it down. This port replaces that hinge with the gate's
        ``lse`` normalizer, which supplies a budget but exerts no pressure toward shorter
        lifetimes, so an inert init risks ``log_beta`` never leaving 0 and the arm silently
        collapsing back to the plain scalar indexer.
    """

    hidden_size: int
    n_heads: int
    mid_dim: int = 0
    norm_eps: float = 1e-5
    pos_slope: float = DEFAULT_POS_SLOPE
    gate_scale: bool = False
    decay: bool = False
    decay_ref: float = DEFAULT_DECAY_REF
    decay_init: float = DEFAULT_DECAY_INIT

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
        if self.decay_ref <= 0:
            raise ValueError(
                f"decay_ref must be positive, got {self.decay_ref}: it divides the age, and it "
                "must be a fixed constant rather than the live sequence length -- see the config "
                "docstring on irreversibility"
            )
        if self.decay_init > 0:
            raise ValueError(
                f"decay_init must be <= 0, got {self.decay_init}: log_beta is the log of a "
                "retention factor in (0, 1], so a positive value makes a key GROW with age and "
                "inverts the lifetime prior"
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

        # TrimKV's lifetime head. Emits log_beta <= 0 per (token, KV head) in nats per decay_ref
        # of age, so the gate carries s_j + log_beta_j * (i - j) / decay_ref.
        #
        # Parameterized as -softplus(raw) rather than logsigmoid(raw): both are smooth maps onto
        # (-inf, 0], but softplus is ~linear once raw > 0, so the whole useful range of log_beta is
        # reachable at O(1) raw values. logsigmoid saturates the other way -- it needs raw ~ -1 to
        # reach log_beta ~ -1 and then compresses hard, which is exactly why TrimKV must init its
        # bias at 18 to sit near zero. Here the init is a plain inverse-softplus of |decay_init|.
        self.decay = config.decay
        self.decay_ref = config.decay_ref
        if config.decay:
            self.w_decay = nn.Linear(
                config.mid_dim if config.mid_dim else config.hidden_size,
                config.n_heads,
                bias=True,
            )
            # Zero weights, biased init: every key starts at exactly decay_init and differentiates
            # from there. A random init would spread lifetimes before the score means anything.
            nn.init.zeros_(self.w_decay.weight)
            nn.init.constant_(self.w_decay.bias, _inv_softplus(-config.decay_init))
        else:
            self.w_decay = None

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

        With :attr:`decay` enabled this returns the score at **age zero** (``log_beta``
        contributes nothing at ``i == j``), which is the magnitude term only. That is a genuine
        per-key quantity but it is *not* the ranking any real query sees. Use :meth:`score_at`
        when a frozen ranking is needed, or :meth:`gate_key` for the exact age-dependent gate.

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
        scores, _ = self._score_and_decay(
            hidden_states, key_offset=key_offset, mask=mask
        )
        return scores

    def _score_and_decay(
        self,
        hidden_states: torch.Tensor,
        *,
        key_offset: int = 0,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        The trunk shared by every entry point: ``(scores, log_beta)``, both fp32 ``(B, h, Sk)``.

        ``log_beta`` is ``None`` when :attr:`decay` is off, and ``<= 0`` otherwise, in nats per
        :attr:`decay_ref` of age. One trunk so the score head and the lifetime head cannot be
        computed from different normalizations of the same hidden state, and so the MLP is
        evaluated once rather than once per head.
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

        log_beta = None
        if self.w_decay is not None:
            # -softplus keeps log_beta <= 0: a retention factor beta = exp(log_beta) in (0, 1].
            log_beta = -nn.functional.softplus(self.w_decay(x).float())
            log_beta = log_beta.transpose(1, 2)  # (B, n_heads, Sk)

        if mask is not None:
            keep = mask if mask.dtype == torch.bool else mask != 0
            keep = ~keep.view(keep.shape[0], 1, -1)
            scores = scores.masked_fill(keep, MASK_NEG)
            if log_beta is not None:
                # Padding must not also decay: MASK_NEG already ranks it last, and a nonzero
                # log_beta there would make the masked value drift with age instead of staying
                # pinned at the bottom.
                log_beta = log_beta.masked_fill(keep, 0.0)
        return scores, log_beta

    def score_at(
        self,
        hidden_states: torch.Tensor,
        query_pos: float,
        *,
        key_offset: int = 0,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        The per-key score **as seen by a query at absolute position** ``query_pos``.

        ``(B, n_heads, Sk)`` fp32. Without decay this is :meth:`score_keys` and ``query_pos`` is
        irrelevant. With decay the score is no longer a property of the key alone, so any caller
        that wants a single per-key ranking has to say *when* -- there is no query-free answer.

        This is the honest interface for the eviction and deadline paths, which need one frozen
        ranking. They pick a representative ``query_pos``; the resulting selection is an
        approximation whose error is bounded by how much ``log_beta`` varies across keys.
        """
        scores, log_beta = self._score_and_decay(
            hidden_states, key_offset=key_offset, mask=mask
        )
        if log_beta is None:
            return scores
        k_len = hidden_states.shape[1]
        pos = torch.arange(
            key_offset, key_offset + k_len, device=scores.device, dtype=scores.dtype
        )
        age = (float(query_pos) - pos).clamp(min=0.0) / self.decay_ref
        return scores + log_beta * age

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        key_hidden_states: torch.Tensor | None = None,
        key_cos: torch.Tensor | None = None,
        key_sin: torch.Tensor | None = None,
        query_offset: int | None = None,
    ) -> torch.Tensor:
        """
        Pairwise-protocol view of the score: ``(B, n_heads, Sq, Sk)``, matching
        :meth:`~.indexer.GQAIndexer.forward` argument for argument.

        Without decay every query row is identical -- that is what query-independence means --
        so this is a broadcast **view** of :meth:`score_keys`, not an ``O(Sq * Sk)`` computation.
        It exists so the press, the query reductions and the loss helpers run over either scorer
        through one code path; the additive ``mask`` is applied here because those callers pass
        the press's ``(B, 1, Sq, Sk)`` causal mask, which only makes sense in this layout.

        With decay the rows genuinely differ, so this **materializes** ``(B, h, Sq, Sk)``. That
        is correct but expensive, and it is why the gate path uses the :meth:`gate_key` fold
        instead of calling this. ``query_offset`` defaults to bottom-right alignment
        (``Sk - Sq``), matching the rest of the codebase.

        Prefer :meth:`score_keys` when the query axis is not actually needed: expanding and
        then reducing it back is wasted work, and at inference it is the whole cost this
        scorer exists to avoid.
        """
        self._reject_rope(cos, sin)
        self._reject_rope(key_cos, key_sin)
        keys = hidden_states if key_hidden_states is None else key_hidden_states
        q_len = hidden_states.shape[1]
        base, log_beta = self._score_and_decay(keys)
        if log_beta is None:
            scores = self.expand_to_pairs(base, q_len)
        else:
            k_len = keys.shape[1]
            if query_offset is None:
                query_offset = k_len - q_len
            q_pos = torch.arange(q_len, device=base.device, dtype=base.dtype) + query_offset
            k_pos = torch.arange(k_len, device=base.device, dtype=base.dtype)
            # Deliberately NOT clamped at 0. A future key (j > i) gets a positive age term here,
            # which is meaningless -- but those pairs are masked out by causality in every caller,
            # and leaving them unclamped makes this exactly equal to the gate_key/gate_query fold
            # the kernel computes. Clamping would make the two paths disagree off-causal and turn
            # any fold regression test into a false negative.
            age = (q_pos.view(-1, 1) - k_pos.view(1, -1)) / self.decay_ref
            scores = base.unsqueeze(2) + log_beta.unsqueeze(2) * age
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

    #: Indexer width. ``n_heads`` for the plain score; ``2 * n_heads`` with decay, where each head
    #: contributes an adjacent ``[magnitude, lifetime]`` pair. Consumed by the gate purely as
    #: ``q_idx.shape[-1]``, which is unconstrained -- the Triton kernel pads it to the next power
    #: of two (floor 16) and masks the tail -- so widening it needs no kernel change.
    @property
    def idx_dim(self) -> int:
        return 2 * self.n_heads if self.decay else self.n_heads

    def gate_key(
        self, hidden_states: torch.Tensor, *, key_offset: int = 0, dtype=None
    ) -> torch.Tensor:
        """
        The score shaped as an indexer key, ``(B, Sk, Di)`` with ``Di = ``:attr:`idx_dim`.

        Pairs with :meth:`gate_query` to drive :mod:`~.gated_attention` unchanged: the gate
        computes ``qi . ki`` over a width-``Di`` axis, and a per-key score is that product with
        the query side pinned to a constant selector.

        **With decay, this is where TrimKV's lifetime enters -- as an exact algebraic identity,
        not an approximation.** The target gate is

            ``s_j + log_beta_j * (i - j) / ref``

        which is bilinear in (query position, key), so it folds into the same dot product at
        twice the width. Head ``h`` occupies columns ``2h`` and ``2h+1``::

            ki[j, 2h]   = s_j - log_beta_j * j / ref        qi[h, i, 2h]   = 1
            ki[j, 2h+1] = log_beta_j                        qi[h, i, 2h+1] = i / ref

        so ``qi . ki = s_j + log_beta_j * (i - j) / ref``. Verified to 2.4e-07 against an explicit
        per-pair reference. Because it is the *same* bilinear form, the fused kernel, the ``lse``
        normalizer and the entire backward pass are untouched -- the gradient reaches ``log_beta``
        through the existing ``dKI`` path.

        The key side absorbs ``-log_beta_j * j / ref``, so ``ki`` is a function of ``j`` alone and
        stays cacheable: at decode the whole history's ``ki`` is read from the cache and only the
        new token is scored, exactly as without decay.

        ``dtype`` casts the result, which :meth:`score_keys` deliberately returns in fp32 for
        top-k resolution. The gate wants it in the attention's dtype instead -- pass the
        model's, or the einsum against a non-fp32 query raises.

        One precision note: ``s_j - log_beta_j * j / ref`` is O(1) here *because* the position is
        normalized by ``ref``. Folding a raw age instead would put ``j`` itself in a bf16 column,
        where 16383 rounds to 16384 and an age of 83 becomes 64 -- the age term would be
        destroyed. This is why :attr:`decay_ref` is load-bearing for correctness, not just for
        gradient scale.
        """
        scores, log_beta = self._score_and_decay(hidden_states, key_offset=key_offset)
        if log_beta is None:
            k = scores.transpose(1, 2)
        else:
            k_len = hidden_states.shape[1]
            pos = torch.arange(
                key_offset, key_offset + k_len, device=scores.device, dtype=scores.dtype
            )
            magnitude = scores - log_beta * (pos / self.decay_ref)  # (B, h, Sk)
            # interleave to [mag_0, beta_0, mag_1, beta_1, ...] so head h reads columns 2h, 2h+1
            k = torch.stack([magnitude, log_beta], dim=-1)  # (B, h, Sk, 2)
            k = k.permute(0, 2, 1, 3).reshape(scores.shape[0], k_len, 2 * self.n_heads)
        return k if dtype is None else k.to(dtype)

    def gate_query(
        self,
        q_len: int,
        bsz: int,
        n_kv_heads: int,
        *,
        device=None,
        dtype=None,
        query_offset: int = 0,
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

        With decay the query side stops being constant along ``Sq``: column ``2h+1`` carries
        ``(row + query_offset) / decay_ref``, the query's absolute position. It is still not
        *learnable* and still carries no content -- it is a position, so the score remains a
        function of (key content, age) with no query content in it. ``query_offset`` is the
        absolute position of row 0 and **must** be supplied whenever the queries are not a
        full-sequence prefill; getting it wrong makes every age wrong by a constant.
        """
        di = self.n_heads
        if di != 1 and di != n_kv_heads:
            raise ValueError(
                f"per-head ScalarIndexer has n_heads={di} but the model has "
                f"{n_kv_heads} KV heads; they must match for the gate to route each head "
                f"to its own score. Build it with ScalarIndexerConfig(n_heads=<KV heads>), "
                f"or n_heads=1 for the shared-score ablation."
            )
        if not self.decay:
            if di == 1:
                return torch.ones(bsz, n_kv_heads, q_len, 1, device=device, dtype=dtype)
            # expand, not repeat: the selector is the same for every batch element and query, so
            # this stays a view. At Sq = 32K and Di = 8 a materialised copy would be 8 GB in fp32.
            eye = torch.eye(di, device=device, dtype=dtype)
            return eye.view(1, di, 1, di).expand(bsz, di, q_len, di)

        # Decay: the selector picks head h's magnitude column, and the age column carries i/ref --
        # gated by the SAME selector, so head h reads only its own log_beta. Without that gating,
        # head h's age column would multiply head h'-s lifetime.
        # Built in fp32 then cast, so i/ref is rounded once at the end rather than accumulated in
        # low precision.
        q_pos = (
            torch.arange(q_len, device=device, dtype=torch.float32) + float(query_offset)
        ) / self.decay_ref
        if di == 1:
            sel = torch.ones(n_kv_heads, 1, device=device, dtype=torch.float32)
        else:
            sel = torch.eye(di, device=device, dtype=torch.float32)
        # (h, 1, di) * (1, Sq, 1) -> the pair (selector, selector * age) per head/query/column
        qi = torch.stack(
            [
                sel.unsqueeze(1).expand(n_kv_heads, q_len, di),
                sel.unsqueeze(1) * q_pos.view(1, q_len, 1),
            ],
            dim=-1,
        )  # (h, Sq, di, 2)
        qi = qi.reshape(n_kv_heads, q_len, 2 * di).to(dtype)
        return qi.unsqueeze(0).expand(bsz, n_kv_heads, q_len, 2 * di)

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
        self,
        hidden_states: torch.Tensor,
        cos=None,
        sin=None,
        *,
        n_kv_heads: int | None = None,
        query_offset: int = 0,
    ) -> torch.Tensor:
        """The gate selector, ``(B, n_kv_heads, Sq, Di)``. Carries no content.

        Shaped like :meth:`~.indexer.GQAIndexer.project_q` so the gate path is shared. Without
        decay it is a pure constant lookup: query-independence means the whole score lives in
        :meth:`project_k`. With decay it additionally carries the query's *position*, which is
        still not content -- see :meth:`gate_query`.

        ``query_offset`` is the absolute position of query row 0. It defaults to 0, which is
        correct for full-sequence training and prefill; decode and chunked prefill must pass the
        real offset or every age is wrong by a constant. Ignored entirely when decay is off.
        """
        self._reject_rope(cos, sin)
        bsz, q_len, _ = hidden_states.shape
        return self.gate_query(
            q_len,
            bsz,
            n_kv_heads if n_kv_heads is not None else self.n_heads,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            query_offset=query_offset,
        )

    def project_k(
        self,
        hidden_states: torch.Tensor,
        cos=None,
        sin=None,
        *,
        value_states: torch.Tensor | None = None,
        key_offset: int = 0,
    ) -> torch.Tensor:
        """The per-key score as an indexer key, ``(B, Sk, Di)`` in the input's dtype.

        ``key_offset`` defaults to 0 -- correct for the training/prefill entry point, where
        ``hidden_states`` starts at position 0. Decode and chunked prefill must pass the real
        offset, or the recency tilt restarts per chunk and (with decay) the folded ``-log_beta *
        j / ref`` term is computed at the wrong ``j``.

        ``value_states`` is accepted for the scorer protocol and intentionally unused. This
        scorer is defined on hidden states; value-based scorers consume it instead.
        """
        self._reject_rope(cos, sin)
        return self.gate_key(
            hidden_states, key_offset=key_offset, dtype=hidden_states.dtype
        )

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
        shape += f", pos_slope={self.pos_slope:g}"
        if self.decay:
            shape += f", decay(ref={self.decay_ref:g}, init={self.config.decay_init:g})"
        return f"{shape}, Di={self.idx_dim}"
