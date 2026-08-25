# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end indexer training through an exact-``K`` chunk subset.

The third objective in this package, and the one that closes the loophole the other two work
around:

* :mod:`~kvpress.presses.gqa_indexer.fused_trainer` **distills** -- the router matches the frozen
  model's attention weights and never touches the forward pass, so the supervision is a surrogate.
* :mod:`~kvpress.presses.gqa_indexer.e2e_trainer` **gates** -- the score is added inside the
  softmax, so the LM loss reaches it. But a gate that is flat along the key axis is inert, and the
  model then reverts to the frozen backbone with no ranking learned;
  :mod:`~kvpress.presses.gqa_indexer.gate_pin` patches that by exempting some keys from the
  normalizer.
* **This trainer** replaces attention with a genuine ``K``-of-``M`` chunk subset. There is no
  configuration of the scores under which "do nothing" recovers dense attention, so nothing needs
  pinning -- the scarcity is structural. See ``ROUTER_LEARNABILITY.md`` §7: this is the
  "train-time forward is already sparse" column, which is why DMA / SparseK / STE never needed the
  machinery SAS needs.

Shares everything else with :class:`~.e2e_trainer.E2EIndexerTrainer` on purpose: the same
``ALL_ATTENTION_FUNCTIONS`` swap, the same forward pre-hook to capture ``hidden_states`` (the
attention interface never receives them), the same ``freeze_backbone``, the same checkpoint format.
So a run of this against a run of that differs in the objective and nothing else.

No ``gate_scale``
-----------------
The additive path multiplies its score by a learnable per-layer scalar, because the score is added
to attention logits and the two have to be brought onto a comparable scale. Here the score is a
*routing* logit -- it goes through ``sigmoid`` into a Bernoulli probability -- so a positive scalar
multiplier is a temperature, not a scale match. Left out: it would be one more thing that differs
between arms, and the marginals already have a well-defined scale of their own.

The press is still built with ``gate_scale=True`` so the checkpoint is byte-compatible with the
e2e arm's and ``--init-from`` works in both directions; the parameter simply is not read. That
matters because starting from the gated run's step-600 checkpoint is the natural warm start.

Train/inference consistency
---------------------------
Unusually for this package, there is **none of the usual gap**. Training attends to exactly the
subset a hard top-K would pick (verified 1.1e-16 in fp64 against a masked softmax), and inference
via ``GQAIndexerPress(chunk_size=...)`` or ``SparseAttentionContext`` takes a plain top-K on the
same score. The one deliberate difference is that training *samples* the subset while inference
takes the argmax -- which is what makes the router explore, and is ProbMoE's own arrangement
(``probmoe_routing`` vs ``deterministic_routing``). :attr:`ExactKIndexerTrainer.jaccard` measures
whether that sampling makes the selection unstable.

What to watch, because the loss will not tell you
-------------------------------------------------
``marginal_entropy`` is the readout ``gate_sparsity`` is for the additive arm. At init the marginals
are uniform at ``K/M``, so entropy sits at its maximum; it falls as the router commits. A run whose
loss descends while entropy stays flat has learned to *use* whatever random subset it is handed
rather than to *choose* one -- which is the exact-K analogue of the flat-gate no-op, and the only
failure mode this design does not rule out structurally.

``effective_topk`` catches the other silent problem: near the diagonal a query block cannot see
``M`` chunks, and when it cannot see ``K`` either the budget is unreachable. That is correct
behaviour but it means the realized budget is below the configured one, and only this number says
so.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from kvpress.presses.gqa_indexer.exact_k_attention import (
    PAD_SCORE,
    build_candidates,
    chunk_visibility,
    exact_k_chunk_attention,
    gather_candidate_scores,
    pool_scores_to_chunks,
    selection_jaccard,
    share_over_query_blocks,
)
from kvpress.presses.gqa_indexer.indexer import build_indexer_mask
from kvpress.presses.gqa_indexer.press import GQAIndexerPress, get_language_model

logger = logging.getLogger(__name__)

#: Default byte budget for one score tile's ``(B, Hkv, dq, Sk)`` token logits.
#:
#: 64 MiB. Deliberately small: at 16K the frozen Qwen3-8B backbone's own activations already occupy
#: ~92 of the 95 GiB available, so what is left for this op is single-GiB, not tens. A 1 GiB tile --
#: which is what a 2048-query tile comes to at ``Sk=16384, Hkv=8`` -- OOM'd on the transient
#: allocation alone, with the checkpoint already in place. See
#: :attr:`ExactKIndexerTrainer.score_tile_bytes` for the three attempts this number came out of.
SCORE_TILE_BYTES = 64 * 1024**2


def _score_tile(
    indexer,
    q_hidden: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    mask: torch.Tensor,
    k_hidden: torch.Tensor,
    key_cos: torch.Tensor | None,
    key_sin: torch.Tensor | None,
    chunk_size: int,
    chunk_aggregate: str,
    query_block: int,
    query_aggregate: str,
) -> torch.Tensor:
    """
    Score one query tile and pool it to ``(B, Hkv, n_qblock_tile, n_chunk)``.

    Split out so :meth:`ExactKIndexerTrainer.chunk_scores` can wrap it in
    ``torch.utils.checkpoint``. The tensor being eliminated is the tile's
    ``(B, Hkv, dq, Sk)`` token logits -- 1 GiB per tile at 16K, and there are 8 tiles per layer and
    36 layers, so retaining them is 288 GiB for a result that is 0.5 MiB. Tiling alone does not fix
    that: without the checkpoint every tile stays in the graph and the sum is unchanged, which is
    what OOM'd the second attempt at exactly one tile's 1024 MiB.

    The recompute is one extra ``q @ k^T`` over the *indexer's* ``head_dim``, which is small beside
    the attention this feeds.
    """
    logits = indexer(
        q_hidden,
        cos=cos,
        sin=sin,
        mask=mask,
        # The key side always spans the full sequence; only the query rows are tiled.
        key_hidden_states=k_hidden,
        key_cos=key_cos,
        key_sin=key_sin,
    )
    chunks = pool_scores_to_chunks(logits, chunk_size, chunk_aggregate)
    return share_over_query_blocks(chunks, query_block, query_aggregate)


@dataclass
class ExactKIndexerTrainer:
    """
    Train the indexer end-to-end by routing attention through an exact-``K`` chunk subset.

    Register with :meth:`hooks`, then run the model and call ``.backward()`` on its LM loss. Like
    :class:`~.e2e_trainer.E2EIndexerTrainer` there is no auxiliary objective, so no ``total_loss()``.

    Parameters
    ----------
    press : GQAIndexerPress
        Supplies the per-layer indexers and, crucially, :meth:`~.press.GQAIndexerPress.get_rope_tables`
        -- so the training-time score is computed by exactly the function the press scores with at
        inference. A divergence there trains the router for a scoring function it never runs under,
        and nothing downstream would flag it.
    chunk_size : int
        Tokens per chunk. Should match the ``chunk_size`` the press will use at eval.
    query_block : int
        Queries sharing one chunk subset. A real modelling concession (see
        :mod:`~.exact_k_attention`), though **not** a performance requirement -- the GPU measurement
        contradicts the CPU analysis that claimed it was. 1 restores per-query selection.
    topk_chunk : int
        ``K``: chunks each query block commits to. ``0`` derives it from :attr:`keep_ratio`.
    n_candidate : int
        ``M``: the candidate pool. **The one parameter that sets the step time** -- the DP is
        ``O(M)`` sequential launches, and the attention is over ``M * chunk_size`` keys. ``K`` is
        free by comparison.
    keep_ratio : float
        Used when ``topk_chunk`` is 0. Set to ``1 - compression_ratio`` so training runs at the
        budget eval evicts at.
    explore_frac : float
        Fraction of the candidate pool drawn at random rather than by score. Without it a chunk
        outside top-M receives exactly zero gradient and can never enter the pool -- the same dead
        end the selected-gate proxy has, one level up. Sweep it; 0 is the ablation.
    n_sink_chunk, n_local_chunk : int
        Candidate slots reserved for the leading chunks and for the block's own diagonal.
    chunk_aggregate, query_aggregate : str
        ``mean`` or ``max`` pooling over a chunk's tokens / a block's queries. ``mean`` matches the
        press's default ``query_reduce``.
    checkpoint : bool
        Recompute the marginals' DP in the backward. Leave True -- 11.13 GiB vs 0.28 across 36
        layers.
    checkpoint_attention : bool
        Recompute each query tile's attention in the backward. Leave True -- 69 GiB vs 3.3 for one
        layer at 16K.
    hard : bool
        Take the deterministic top-K instead of sampling. The ablation for "does stochastic
        exploration matter"; not for a real run, since a deterministic selection cannot explore.
    freeze : bool
        Freeze every non-indexer parameter on :meth:`hooks` entry. Leave True, so the comparison
        against the other objectives is about the gradient rather than about the trainable set.
    """

    press: GQAIndexerPress
    chunk_size: int = 64
    query_block: int = 256
    topk_chunk: int = 0
    n_candidate: int = 32
    keep_ratio: float = 0.25

    explore_frac: float = 0.10
    n_sink_chunk: int = 1
    n_local_chunk: int = 1
    chunk_aggregate: str = "mean"
    query_aggregate: str = "mean"

    checkpoint: bool = True
    checkpoint_attention: bool = True
    hard: bool = False
    freeze: bool = True

    #: Cut the score's gradient path back into ``hidden_states``.
    #:
    #: **Leave True.** The router's score is a function of ``hidden_states``, so ``dL/d(hidden)``
    #: has two paths: the ordinary one through the attention output, and a second one through the
    #: routing decision. That second path is a genuine derivative -- perturbing ``hidden`` does
    #: change the score, which changes the subset, which changes the output -- but it is also a
    #: **per-layer feedback loop**, because the gradient it deposits in the residual stream is then
    #: re-amplified by every router below.
    #:
    #: It diverges. Measured on the real model, ``grad_norm`` at the end of one backward:
    #:
    #: ============  ===========  ==============
    #: layers        attached     detached
    #: ============  ===========  ==============
    #: 4             2.1e3        --
    #: 12            8.6e13       --
    #: 24            **inf**      **1.1e6**
    #: 36            **nan**      --
    #: ============  ===========  ==============
    #:
    #: The per-layer ratio ``|g out of hidden| / |g into attn_out|`` measured 10-50x at every layer,
    #: against 1.1x for the same truncated backbone running dense attention with no router at all --
    #: so the amplification is this feedback path, not the backbone and not the op (the op's own
    #: score-gradient gain is 0.04, measured in isolation).
    #:
    #: What detaching gives up, stated plainly: a layer's router no longer receives the second-order
    #: term "my routing changed the hidden states that a *lower* layer's router scores from". Each
    #: router is still trained by the full LM loss through its own layer's attention, and the
    #: gradient still flows through the frozen backbone to get there -- which is the property that
    #: distinguishes this from distillation. Distillation severs the same path (its loss is computed
    #: from detached teacher tensors inside a per-layer hook), so this arm is no worse off than that
    #: one on this axis.
    #:
    #: Set False to reproduce the divergence, or to test it under a smaller ``route_scale``.
    detach_score_input: bool = True

    #: Multiplier on the pooled chunk score before it becomes a Bernoulli logit. ``None`` uses
    #: ``head_dim ** -0.5``.
    #:
    #: **Required, and it is a temperature rather than a scale match.** ``IndexerNorm`` leaves q and
    #: k at unit variance per channel, so the raw ``qi . ki`` dot product has std ``~sqrt(head_dim)``
    #: -- 11.3 at ``head_dim = 128``. That is fine as an attention *addend* (which is why
    #: :attr:`~.indexer.GQAIndexer.GATE_SCALE_INIT` divides by the same factor) and it is far too
    #: loud as a *routing logit*, because ``sigmoid`` saturates. Measured, on 256 rows at ``M=32,
    #: K=8``:
    #:
    #: =======  ===========================  ==============
    #: std      fraction of mu saturated     mean |d mu/ds|
    #: =======  ===========================  ==============
    #: 1.0      0.000                        1.30e-01
    #: 5.0      0.023                        4.92e-02
    #: **11.3** **0.306**                    **2.53e-02**
    #: =======  ===========================  ==============
    #:
    #: At the indexer's natural scale **31% of the marginals are pinned at 0 or 1**, so 31% of
    #: candidates get no usable gradient -- which is the boundary-credit property this whole method
    #: exists for, lost to a units mismatch. Unscaled, the first real run showed ``grad_norm`` at
    #: 6.5e4 and the loss failing to descend.
    #:
    #: Fixed rather than learnable, matching ``gate_scale``'s default treatment: a learnable
    #: temperature can reduce its own gradient by growing, which is a loophole of the kind this
    #: package is careful about. Sweep it as a hyperparameter instead.
    route_scale: float | None = None

    #: Bytes budgeted for one score tile's token logits. ``0`` uses :data:`SCORE_TILE_BYTES`.
    #:
    #: **Not a query count**, because the right query count depends on ``Sk`` and the head count,
    #: and a fixed count that fits at 8K does not fit at 16K. This is the same byte-budget form
    #: :data:`~.exact_k_attention.TILE_LOGIT_BYTES` uses, and for the same reason.
    #:
    #: Why any of this is needed. The router's ``(B, Hkv, Sq, Sk)`` token logits are **8 GiB in fp32
    #: at Sq = Sk = 16384**, while the ``(B, Hkv, n_qblock, n_chunk)`` tensor they reduce to is
    #: 0.5 MiB -- a factor of 16000. Three separate attempts at 16K failed here:
    #:
    #: 1. no tiling: OOM building the 8 GiB tensor, in ``pool_scores_to_chunks``'s ``valid.sum(-1)``;
    #: 2. tiling without :attr:`checkpoint_scores`: every tile stayed in the graph, so the retained
    #:    total was unchanged -- OOM at exactly one tile's 1024 MiB;
    #: 3. both, at a 2048-query tile: the *transient* 1 GiB still did not fit, because the frozen
    #:    backbone's own activations already occupy ~92 of the 95 GiB at 16K.
    #:
    #: So the budget has to be small in absolute terms, not merely smaller than the whole. 64 MiB
    #: leaves the headroom the attention tile then needs.
    score_tile_bytes: int = 0

    #: Recompute each score tile's logits in the backward instead of retaining them.
    #:
    #: Leave True. Retaining them costs ``Sq * Sk * Hkv * 4`` bytes per layer -- 8 GiB at 16K, 288
    #: GiB across 36 layers -- for a tensor that reduces to 0.5 MiB. The recompute is one extra
    #: ``q @ k^T`` over the indexer's ``head_dim``, which is small beside the attention it feeds.
    checkpoint_scores: bool = True

    #: Layer index -> mean Bernoulli entropy of the marginals, in nats. **The** diagnostic here:
    #: ``log 2`` per item at maximum indecision, 0 when the router has committed. A loss that falls
    #: while this stays flat means the router learned to use a random subset rather than to choose
    #: one.
    marginal_entropy: dict[int, float] = field(default_factory=dict)
    #: Layer index -> mean number of selected slots holding a real chunk. Below ``topk_chunk`` means
    #: some query blocks cannot see ``K`` chunks yet, so the realized budget is smaller than the
    #: configured one.
    effective_topk: dict[int, float] = field(default_factory=dict)
    #: Layer index -> Jaccard overlap of this step's selection against the previous step's. The
    #: forward is stochastic (``torch.bernoulli``), and this is what says whether that makes the
    #: selection unstable. ProbMoE does not report it as a problem, but their row count is not ours.
    jaccard: dict[int, float] = field(default_factory=dict)
    #: Layer index -> number of layers that actually ran, as a wiring check.
    layers_routed: int = field(default=0, init=False)

    _hidden_states: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _kwargs: dict[int, dict] = field(default_factory=dict, init=False, repr=False)
    _previous_selection: dict[int, torch.Tensor] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.query_block <= 0:
            raise ValueError(f"query_block must be positive, got {self.query_block}")
        if self.n_candidate <= 0:
            raise ValueError(f"n_candidate must be positive, got {self.n_candidate}")
        if self.topk_chunk and self.topk_chunk > self.n_candidate:
            raise ValueError(
                f"topk_chunk {self.topk_chunk} exceeds the candidate pool n_candidate "
                f"{self.n_candidate}: the subset is drawn from the pool, so K <= M by construction."
            )
        if not 0 < self.keep_ratio <= 1:
            raise ValueError(f"keep_ratio must be in (0, 1], got {self.keep_ratio}")
        if not 0 <= self.explore_frac <= 1:
            raise ValueError(f"explore_frac must be in [0, 1], got {self.explore_frac}")
        if self.explore_frac == 0:
            logger.warning(
                "explore_frac=0 makes the candidate pool a plain top-M, so a chunk outside it "
                "receives exactly zero gradient and can never be promoted into the pool. This is "
                "the ablation baseline; the failure it reproduces is the one that took the "
                "selected-gate proxy from 0.0%% to 0.0%% recall."
            )
        if self.hard:
            logger.warning(
                "hard=True takes the deterministic top-K, so the forward no longer explores: a "
                "chunk's score can only be compared against chunks that were already selected. "
                "This is the sampling ablation, not a training configuration."
            )

    def resolved_topk(self, n_chunk: int) -> int:
        """
        ``K``, from :attr:`topk_chunk` or derived from :attr:`keep_ratio`.

        Clamped to the pool: ``K > M`` is not reachable, since the subset is drawn from the pool.
        Also clamped to at least 1 -- a budget of 0 would make every row's output zero, which the
        LM loss would report as a very bad number without saying why.
        """
        if self.topk_chunk:
            k = self.topk_chunk
        else:
            k = max(1, int(round(self.keep_ratio * n_chunk)))
        return max(1, min(k, min(self.n_candidate, n_chunk)))

    def resolved_route_scale(self, indexer) -> float:
        """
        :attr:`route_scale`, defaulting to ``head_dim ** -0.5``.

        Reuses :attr:`~.indexer.GQAIndexer.GATE_SCALE_INIT` rather than restating the exponent, so
        the two consumers of the score cannot drift apart on what its natural magnitude is. A scorer
        with no ``head_dim`` (:class:`~.scalar_indexer.ScalarIndexer` emits one number per key, not a
        dot product) needs no correction and gets 1.
        """
        if self.route_scale is not None:
            return float(self.route_scale)
        head_dim = getattr(indexer, "head_dim", None)
        if not head_dim:
            return 1.0
        return float(type(indexer).GATE_SCALE_INIT(head_dim))

    def reset(self) -> None:
        """Drop the per-pass state. Keeps ``_previous_selection``, which spans steps by design."""
        self.marginal_entropy = {}
        self.effective_topk = {}
        self.jaccard = {}
        self.layers_routed = 0
        self._hidden_states.clear()
        self._kwargs.clear()

    # ------------------------------------------------------------------
    # Parameters -- identical to E2EIndexerTrainer, so the arms are comparable
    # ------------------------------------------------------------------
    def indexer_parameters(self, model: nn.Module) -> list[nn.Parameter]:
        """Every indexer parameter, in layer order -- what the optimizer should be given."""
        params = []
        for layer in get_language_model(model).layers:
            indexer = getattr(layer.self_attn, self.press.scorer_attr, None)
            if indexer is not None:
                params.extend(indexer.parameters())
        return params

    def freeze_backbone(self, model: nn.Module) -> None:
        """
        Put every non-indexer parameter at ``requires_grad=False``.

        Identifies the indexers by **module identity**, not by parameter-name substring: a name
        filter would also catch a backbone parameter that happens to contain the attribute name and
        would silently train it.

        Note the gradient still *flows through* the frozen backbone to reach the router, which is
        the whole point. So this is ``requires_grad=False``, never ``torch.no_grad()`` -- the latter
        would sever the path and the router would receive nothing.

        No ``gate_scale`` upcast here, unlike :meth:`~.e2e_trainer.E2EIndexerTrainer.freeze_backbone`:
        this objective does not read that scalar (see the module docstring), so there is no bf16
        step-size problem to fix.
        """
        indexer_params = {id(p) for p in self.indexer_parameters(model)}
        if not indexer_params:
            raise RuntimeError(
                f"no {self.press.scorer_attr!r} modules found on the model; call "
                "press.post_init_from_model(model) first"
            )
        for param in model.parameters():
            param.requires_grad = id(param) in indexer_params

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def chunk_scores(
        self, module: nn.Module, hidden_states: torch.Tensor, kwargs: dict, k_len: int
    ) -> torch.Tensor:
        """
        Router scores per ``(query block, chunk)``: ``(B, Hkv, n_qblock, n_chunk)``.

        Built from the same ``hidden_states`` and the same RoPE tables the layer itself uses, via
        the press's own :meth:`~.press.GQAIndexerPress.get_rope_tables`. That shared path is the
        invariant worth protecting -- see :attr:`press`.

        The token-level logits are pooled twice: over the key axis into chunks, then over the query
        axis into blocks. Both poolings discount the ``MASK_NEG`` sentinel that
        :func:`~.indexer.build_indexer_mask` writes on forbidden pairs; folding it into a mean would
        make the pooled score a monotone ramp in position and reduce selection to "keep the oldest
        chunks".

        **Tiled over queries**, because the intermediate is enormous relative to the result: 8 GiB
        of ``(B, Hkv, Sq, Sk)`` fp32 logits at 16K against 0.5 MiB of output. See
        :attr:`score_query_tile`. The indexer's q/k projections are computed once and shared across
        tiles -- it is the ``q @ k^T`` outer product that has to be tiled, not the projections.
        """
        indexer = self.press.get_indexer(module)
        cos, sin = self.press.get_rope_tables(indexer, kwargs)
        q_len = hidden_states.shape[1]
        attention_mask = kwargs.get("attention_mask")
        if self.detach_score_input:
            # Severs d(score)/d(hidden) while leaving d(score)/d(indexer weights) intact -- see
            # :attr:`detach_score_input` for the divergence this prevents and what it costs.
            hidden_states = hidden_states.detach()

        # Rounded to a multiple of query_block, so no block is split across tiles: a split block
        # would be pooled over half its queries in each tile, which is a different reduction. That
        # rounding is also why the budget can be exceeded -- one query_block's logits are the
        # smallest unit, and at query_block=256 with Sk=16384, Hkv=8 that is already 128 MiB.
        budget = self.score_tile_bytes or SCORE_TILE_BYTES
        n_kv = self.press.get_indexer(module).n_heads
        per_block = hidden_states.shape[0] * n_kv * self.query_block * k_len * 4
        tile = max(1, budget // max(per_block, 1)) * self.query_block

        blocks = []
        for start in range(0, q_len, tile):
            stop = min(start + tile, q_len)
            mask = build_indexer_mask(
                stop - start,
                k_len,
                hidden_states.device,
                attention_mask=attention_mask,
                # The absolute position of this tile's first query, so causality is anchored to the
                # full sequence rather than restarting at each tile. Defaulting it (k_len - q_len)
                # would let every tile see the whole prefix, which is exactly the leak
                # build_indexer_mask exists to prevent -- and it would be invisible: attention would
                # still be causal (the attention's own mask enforces that), while the ROUTER would
                # have been trained on scores that peeked ahead. Pinned by
                # ``test_tiled_scores_stay_causal``.
                query_offset=k_len - q_len + start,
            )
            args = (
                indexer,
                hidden_states[:, start:stop],
                None if cos is None else cos[:, start:stop],
                None if sin is None else sin[:, start:stop],
                mask,
                hidden_states,
                cos,
                sin,
                self.chunk_size,
                self.chunk_aggregate,
                self.query_block,
                self.query_aggregate,
            )
            if self.checkpoint_scores and torch.is_grad_enabled():
                blocks.append(torch_checkpoint(_score_tile, *args, use_reentrant=False))
            else:
                blocks.append(_score_tile(*args))
        scores = torch.cat(blocks, dim=2)

        # Bring the score onto a scale sigmoid can resolve -- see :attr:`route_scale`. Applied AFTER
        # pooling rather than inside it so the MASK_NEG sentinel keeps its magnitude relative to real
        # scores through the pooling comparisons; it is replaced by PAD_SCORE downstream anyway.
        scale = self.resolved_route_scale(indexer)
        if scale != 1.0:
            scores = scores * scale
        # Chunks with no valid token pooled to MASK_NEG (-1e4). Scaled, that is still ~-88, which is
        # far enough below a real score to lose every ranking -- but it must NOT reach the DP as a
        # Bernoulli logit, where log(1 - sigmoid(-88)) underflows and the "not selected" branch loses
        # all resolution. build_candidates never picks such a chunk (they are invisible), so this is
        # a floor for the pathological case where a row has nothing else, not an expected path.
        return scores.clamp_min(PAD_SCORE)

    def routed_forward(
        self,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask,
        scaling: float | None,
    ) -> torch.Tensor:
        """
        Replacement attention for one layer: route through the exact-``K`` chunk subset.

        ``query``/``key``/``value`` arrive post-RoPE and post-cache-update, so this sees exactly the
        tensors the original attention would have.
        """
        layer_idx = int(module.layer_idx)
        # POPPED, not read, for the reason E2EIndexerTrainer.gated_forward documents: leaving the
        # entry in the dict pins that layer's (B, L, hidden) tensor for the whole forward AND
        # backward, which also stops autograd releasing each layer's activations as it walks up.
        hidden_states = self._hidden_states.pop(layer_idx, None)
        if hidden_states is None:
            raise RuntimeError(
                f"layer {layer_idx} reached the exact-K attention without its hidden_states being "
                "captured. The pre-hook that records them must be registered on the same modules "
                "as the attention swap -- use ExactKIndexerTrainer.hooks(). A second call for the "
                "same layer in one forward (gradient checkpointing recomputes a block) also lands "
                "here, because the entry is consumed on first use."
            )
        kwargs = self._kwargs.pop(layer_idx, {})

        k_len, q_len = key.shape[2], query.shape[2]
        scores = self.chunk_scores(module, hidden_states, kwargs, k_len)
        n_qblock, n_chunk = scores.shape[-2:]
        topk = self.resolved_topk(n_chunk)

        visible = chunk_visibility(
            n_qblock,
            n_chunk,
            query_block=self.query_block,
            chunk_size=self.chunk_size,
            q_len=q_len,
            k_len=k_len,
            device=scores.device,
        )
        # The pool is chosen from DETACHED scores: which chunks are candidates is a discrete
        # decision that carries no useful derivative, and building the graph for it would retain the
        # full (B, Hkv, n_qblock, n_chunk) ranking for nothing. The scores that enter the estimator
        # are re-gathered WITH gradient on the next line.
        candidates = build_candidates(
            scores.detach(),
            self.n_candidate,
            visible=visible,
            n_sink_chunk=self.n_sink_chunk,
            n_local_chunk=self.n_local_chunk,
            explore_frac=self.explore_frac,
            training=not self.hard,
        )
        out, stats = exact_k_chunk_attention(
            query,
            key,
            value,
            gather_candidate_scores(scores, candidates),
            candidates,
            topk_chunk=topk,
            chunk_size=self.chunk_size,
            query_block=self.query_block,
            scaling=scaling,
            checkpoint=self.checkpoint,
            checkpoint_attention=self.checkpoint_attention,
            hard=self.hard,
        )

        self.marginal_entropy[layer_idx] = stats["marginal_entropy"]
        self.effective_topk[layer_idx] = stats["effective_topk"]
        previous = self._previous_selection.get(layer_idx)
        if previous is not None:
            self.jaccard[layer_idx] = selection_jaccard(previous, stats["selected"])
        # Stored for the NEXT step's comparison. One (B, Hkv, n_qblock, M) bool-valued fp tensor
        # per layer, which is ~1 MiB total at 16K -- affordable, unlike keeping the scores.
        self._previous_selection[layer_idx] = stats["selected"]
        self.layers_routed += 1

        # The attention interface contract is (B, Sq, H, D) -- the layer reshapes to (B, Sq, H*D)
        # itself. Our ops return (B, H, Sq, D).
        return out.transpose(1, 2).contiguous()

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def _capture_hook(self, module: nn.Module, args, kwargs: dict):
        """
        Stash this layer's ``hidden_states`` and kwargs before its attention runs.

        Needed because the attention *interface* is called with q/k/v only -- it never sees
        ``hidden_states``, which is what the indexer projects from.
        """
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None:
            return None
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        self._hidden_states[int(layer_idx)] = hidden_states
        self._kwargs[int(layer_idx)] = kwargs
        return None

    @contextmanager
    def hooks(self, model: nn.Module):
        """
        Route every attention layer through the exact-``K`` subset for the duration of the block.

        Installs a pre-hook per attention module (to capture ``hidden_states``) and a temporary
        entry in ``ALL_ATTENTION_FUNCTIONS`` that the model's ``config._attn_implementation`` is
        pointed at. Both are removed on exit, including on exception.

        The registry cleanup goes through ``_global_mapping`` directly for the reason
        :func:`~.teacher_lse.capture_teacher_lse` documents: ``register()`` writes there while
        ``pop()`` only touches the instance mapping, so the naive removal leaks the entry forever.
        """
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self.press.post_init_from_model(model)
        self.reset()
        if self.freeze:
            self.freeze_backbone(model)

        impl_name = "kvpress_gqa_indexer_exact_k"

        def exact_k_attention_impl(
            module, query, key, value, attention_mask, scaling=None, dropout=0.0, **_
        ):
            return self.routed_forward(
                module, query, key, value, attention_mask, scaling
            ), None

        configs = [model.config]
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None:
            configs.append(text_config)
        previous_impls = [cfg._attn_implementation for cfg in configs]

        global_mapping = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
        had_previous = impl_name in global_mapping
        previous_fn = global_mapping.get(impl_name)
        ALL_ATTENTION_FUNCTIONS.register(impl_name, exact_k_attention_impl)

        handles = []
        try:
            for layer in get_language_model(model).layers:
                handles.append(
                    layer.self_attn.register_forward_pre_hook(self._capture_hook, with_kwargs=True)
                )
            for cfg in configs:
                cfg._attn_implementation = impl_name
            yield self
        finally:
            for handle in handles:
                handle.remove()
            for cfg, previous in zip(configs, previous_impls):
                cfg._attn_implementation = previous
            if had_previous:
                global_mapping[impl_name] = previous_fn
            else:
                global_mapping.pop(impl_name, None)
            self._hidden_states.clear()
            self._kwargs.clear()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def mean_marginal_entropy(self) -> float | None:
        """Mean marginal entropy over the layers that ran, or ``None`` before the first pass."""
        return _mean(self.marginal_entropy)

    def mean_effective_topk(self) -> float | None:
        """Mean realized budget -- below ``topk_chunk`` when blocks cannot see ``K`` chunks."""
        return _mean(self.effective_topk)

    def mean_jaccard(self) -> float | None:
        """Mean selection stability against the previous step. ``None`` on the first step."""
        return _mean(self.jaccard)


def _mean(values: dict[int, float]) -> float | None:
    finite = [v for v in values.values() if v is not None and v == v]  # v == v drops NaN
    if not finite:
        return None
    return sum(finite) / len(finite)


def exact_k_indexer_training_step(
    model: nn.Module,
    trainer: ExactKIndexerTrainer,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    skip_logits: bool | None = None,
) -> torch.Tensor:
    """
    One routed forward pass, returning the model's own LM loss.

    ``labels`` defaults to ``input_ids`` (ordinary next-token prediction; the model shifts them
    internally). ``use_cache=False`` -- nothing here reads the cache, and building one during
    training only costs memory.

    ``skip_logits=True`` asks a Liger-patched model to fuse ``lm_head`` into the cross-entropy so
    the ``(L, vocab)`` logits are never materialized -- 7.0 GiB at ``L=8192`` on Qwen3-8B, scaling
    with ``L``. It must be passed **explicitly**: Liger's own default is
    ``self.training and labels is not None``, and this backbone is deliberately left in ``eval()``
    to keep dropout off, so the default resolves to ``False`` and the patch silently saves nothing.
    ``None`` (rather than ``False``) because an unpatched model does not accept the kwarg.

    Returns
    -------
    torch.Tensor
        The LM loss, ready for ``backward()``. The router's gradient arrives through the subset
        mask's marginals rather than from an auxiliary term, so there is no second loss to add.
    """
    extra = {} if skip_logits is None else {"skip_logits": skip_logits}
    with trainer.hooks(model):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids if labels is None else labels,
            use_cache=False,
            **extra,
        )
        if trainer.layers_routed == 0:
            raise RuntimeError(
                "no layer ran the exact-K attention: the model kept its own attention "
                "implementation. This usually means the model's config is not the one "
                f"ExactKIndexerTrainer pointed at {model.config._attn_implementation!r}."
            )
        return out.loss
