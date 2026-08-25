# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end indexer training through **two-level (HSA) chunk attention**.

The fourth objective in this package. See :mod:`~.hsa_attention` for the operator and the three
structural properties that motivate it; this module is the training wiring, and it deliberately
mirrors :class:`~.e2e_trainer.E2EIndexerTrainer` and
:class:`~.exact_k_trainer.ExactKIndexerTrainer` field for field so a run of any of the three
differs in the objective and nothing else.

What is *absent* here, relative to the other two, is the point:

* **No ``pin_mode``.** The additive arm needs it because a flat gate is a no-op; here a flat router
  gives uniform mixing, which is 0.44 from dense. Nothing to close.
* **No candidate pool, no ``n_candidate``, no ``explore_frac``.** The chunk softmax is over every
  chunk, so every chunk gets a content-dependent gradient every step. The exact-K arm's measured
  bottleneck -- 11-15% of oracle-best chunks never entering its ``M=32`` pool -- cannot arise.
* **No ``route_scale`` temperature and no saturation problem.** ``sigmoid`` was the exact-K arm's
  hazard (31% of marginals pinned at the indexer's natural scale). A softmax has no such floor or
  ceiling; it is shift-invariant, so the score's absolute magnitude only sets how peaked ``w`` is,
  and that is a quantity the loss can and should control. :attr:`score_scale` exists anyway,
  defaulting to ``head_dim ** -0.5``, because the *initialization* matters: an unscaled ``qi . ki``
  has std ~11.3, which starts the router at a nearly one-hot ``w`` and therefore at an output far
  from the frozen backbone's.

What is present and shared: :attr:`detach_score_input`. The reason is unchanged from the exact-K
arm and was measured there -- the score's gradient path back into ``hidden_states`` is a per-layer
feedback loop that amplified 10-50x per layer and drove ``grad_norm`` to ``nan`` at 36 layers. That
mechanism has nothing to do with which operator consumes the score, so it applies here too. It
defaults True for that reason, and :attr:`detach_score_input` records what it gives up.

The diagnostic that matters
---------------------------
``chunk_entropy`` (normalized, in ``[0, 1]``). This objective's single failure mode is a router that
learns to *use* a near-uniform mixture rather than to *choose*: the LM loss can descend that way,
because uniform mixing over chunks is a legitimate if blunt operator. Entropy stuck near 1.0 while
the loss falls is that failure, and nothing else reports it.

``score_lse_corr`` is the second, and it is stronger than any diagnostic the other arms have.
``ROUTER_LEARNABILITY.md`` §6 establishes that for a frozen backbone the *optimal* score is exactly
the chunk's own log-sum-exp, up to a per-query constant. So the target is known in closed form and
the correlation against it can be measured directly on real text -- no oracle, no swap experiment,
no second forward pass. It costs one extra reduction over logits the forward already computed.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from kvpress.presses.gqa_indexer.exact_k_attention import pool_scores_to_chunks
from kvpress.presses.gqa_indexer.hsa_attention import chunk_lse, hsa_chunk_attention
from kvpress.presses.gqa_indexer.indexer import build_indexer_mask
from kvpress.presses.gqa_indexer.press import GQAIndexerPress, get_language_model

logger = logging.getLogger(__name__)

#: Default byte budget for one score tile's ``(B, Hkv, dq, Sk)`` token logits.
#:
#: 64 MiB, the number three OOMs produced for the exact-K arm at 16K -- see
#: :attr:`~.exact_k_trainer.ExactKIndexerTrainer.score_tile_bytes`. The constraint is the same here
#: (a frozen Qwen3-8B already occupies ~92 of 95 GiB at 16K), so the same budget applies.
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
    score_scale: float,
) -> torch.Tensor:
    """
    Score one query tile and aggregate it to ``(B, Hkv, dq, n_chunk)``.

    Split out so :meth:`HSAIndexerTrainer.chunk_scores` can checkpoint it. The tensor eliminated is
    the tile's ``(B, Hkv, dq, Sk)`` token logits; the result is smaller by ``chunk_size``, and
    retaining every tile across 36 layers is what OOM'd the exact-K arm twice.

    Unlike that arm there is **no query-block pooling**: HSA selects per query. The exact-K arm
    shared a subset over ``query_block`` queries as a memory concession that the GPU measurement
    later showed was unnecessary, and here the per-query score matrix is only 34 MiB at 8K, so there
    is nothing to concede.

    ``score_scale`` is applied to the **token** scores, i.e. *inside* the aggregation. For ``lse``
    that is load-bearing rather than cosmetic -- see :attr:`HSAIndexerTrainer.score_scale`.
    """
    logits = indexer(
        q_hidden,
        cos=cos,
        sin=sin,
        mask=mask,
        key_hidden_states=k_hidden,
        key_cos=key_cos,
        key_sin=key_sin,
    )
    if score_scale != 1.0:
        # BEFORE the reduction. `logsumexp` is not scale-equivariant -- LSE(c*x) != c*LSE(x) -- so
        # for the `lse` mode this multiplier is a temperature that decides where the aggregation sits
        # between mean-like and max-like. Applying it after would leave `logsumexp` operating on the
        # raw dot's std ~= sqrt(head_dim) = 11.3, where it degenerates to `max`; measured Spearman
        # against the true chunk LSE was 0.65 that way against 1.00 this way.
        logits = logits * score_scale
    mode = (chunk_aggregate or "lse").lower()
    if mode in ("lse", "logsumexp"):
        # MASK_NEG (-1e4) on causally-forbidden pairs exponentiates to exactly 0, so LSE discounts
        # them for free -- no valid-count division, and the ragged tail needs no special case either.
        # A chunk with no visible token returns INVISIBLE_SCORE rather than -inf, so the downstream
        # softmax cannot produce NaN.
        return chunk_lse(logits, chunk_size)
    return pool_scores_to_chunks(logits, chunk_size, mode)


@dataclass
class HSAIndexerTrainer:
    """
    Train the indexer end-to-end through two-level chunk attention.

    Register with :meth:`hooks`, run the model, call ``.backward()`` on its LM loss. No auxiliary
    objective, so no ``total_loss()``.

    Parameters
    ----------
    press : GQAIndexerPress
        Supplies the per-layer indexers and :meth:`~.press.GQAIndexerPress.get_rope_tables`, so the
        training-time score is computed by the same function the press scores with at inference.
    chunk_size : int
        Tokens per chunk. Must be ``> 1``: at ``chunk_size = 1`` the within-chunk softmax degenerates
        to ``softmax(single element) = 1``, the ``q . k`` term vanishes entirely and the router would
        have to learn the whole attention distribution from scratch, discarding the frozen backbone's
        prior. ``ROUTER_LEARNABILITY.md`` §6 flags this as structural, not a tuning matter, so it is
        rejected rather than allowed.
    chunk_aggregate : str
        How token scores become a chunk score. ``lse`` (default), ``mean`` or ``max``.

        **``lse`` is the principled choice, not a tuning knob.** ``ROUTER_LEARNABILITY.md`` §6
        verifies that for a frozen backbone the true chunk mass is ``softmax_c(LSE_c)`` exactly
        (5.55e-17), and :func:`~.hsa_attention.hsa_chunk_attention` makes ``w = softmax_c(s_chunk)``
        *be* the realized mass (5.6e-17). Equating the two fixes what the router must compute:
        ``s_chunk = LSE_c``. Since the indexer's token score imitates the backbone's attention
        *logit*, the aggregation has to be the same functional -- logsumexp -- or the chunk level
        cannot match even with a perfect token scorer.

        Measured with an exact token scorer (``s_token == a_j``), Spearman against the true chunk
        LSE: **lse 1.000, mean 0.756, max 0.631**. In a needle regime (one high-logit token in a
        64-token chunk) the gap is starker -- needle recall at top-4 chunks: **lse 1.000, mean
        0.533**, max 1.000 but Spearman only 0.630 (``max`` finds the needle yet cannot rank, having
        thrown away how many mid-sized tokens a chunk holds). ``mean`` dilutes a lone needle ~64x.

        Cost is the same order either way -- both are a reduction over an already-computed
        ``(B, Hkv, dq, Sk)`` tensor; measured 1.44x on that one reduce, immaterial beside the
        attention it feeds.
    checkpoint_attention : bool
        Recompute each query tile's attention in the backward. Leave True.
    checkpoint_scores : bool
        Recompute each score tile's token logits in the backward. Leave True -- 8 GiB per layer at
        16K for a result that reduces by 64x.
    score_tile_bytes : int
        Bytes for one score tile's token logits; ``0`` uses :data:`SCORE_TILE_BYTES`. A byte budget
        rather than a query count, because a count that fits at 8K does not at 16K.
    attention_tile_bytes : int
        Bytes for one attention tile's logits; ``0`` uses
        :data:`~.hsa_attention.TILE_LOGIT_BYTES`.
    freeze : bool
        Freeze every non-indexer parameter on entry. Leave True.
    """

    press: GQAIndexerPress
    chunk_size: int = 64
    chunk_aggregate: str = "lse"

    checkpoint_attention: bool = True
    checkpoint_scores: bool = True
    score_tile_bytes: int = 0
    attention_tile_bytes: int = 0
    freeze: bool = True

    #: Cut the score's gradient path back into ``hidden_states``.
    #:
    #: **Leave True.** Measured on the exact-K arm, whose mechanism is identical here: the router's
    #: score is a function of ``hidden_states``, so ``dL/d(hidden)`` gains a second path through the
    #: routing decision, and the gradient that path deposits in the residual stream is re-amplified
    #: by every router below. ``grad_norm`` at the end of one backward went 2.1e3 (4 layers) ->
    #: 8.6e13 (12) -> ``inf`` (24) -> ``nan`` (36) with it attached, against 1.1e6 at 24 detached.
    #: The per-layer amplification measured 10-50x, against 1.1x for the same backbone running dense
    #: attention with no router, so it is the feedback path rather than the backbone.
    #:
    #: What it gives up: a layer's router no longer receives "my routing changed the hidden states a
    #: *lower* layer's router scores from". Each router is still trained by the full LM loss through
    #: its own layer's attention, with the gradient flowing through the frozen backbone to get there
    #: -- the property that distinguishes this from distillation.
    detach_score_input: bool = True

    #: Multiplier on the **token** scores, applied *inside* the chunk aggregation. ``None`` uses
    #: ``head_dim ** -0.5``.
    #:
    #: **Under ``chunk_aggregate="lse"`` this is load-bearing, and its position matters.**
    #: ``logsumexp`` is not scale-equivariant (``LSE(c*x) != c*LSE(x)``), so the multiplier acts as a
    #: **temperature** choosing where the aggregation sits between mean-like and max-like. Measured
    #: Spearman against the backbone's true chunk LSE, with an exact token scorer:
    #:
    #: ==========================================  ==========
    #: where the scale goes                         Spearman
    #: ==========================================  ==========
    #: ``LSE(r * head_dim**-0.5)``  (inside)        **1.0000**
    #: ``mean(r) * head_dim**-0.5``                 0.7589
    #: ``LSE(r) * head_dim**-0.5``  (outside)       0.6495
    #: ==========================================  ==========
    #:
    #: Outside, ``logsumexp`` operates on the raw dot's std ``~sqrt(head_dim)`` = 11.3 and degenerates
    #: to ``max``; multiplying afterwards is then just a constant and cannot restore the ranking. So
    #: the scale is applied in :func:`_score_tile` before the reduction, and it is *not* optional.
    #:
    #: ``head_dim ** -0.5`` is the right default for a reason rather than by analogy: it is the
    #: backbone's own attention scale, so it puts the indexer's dot product on the scale of the
    #: quantity being imitated (``a_j = q k^T / sqrt(head_dim)``), which is exactly what makes the
    #: measured correlation 1.0000 rather than merely good.
    #:
    #: It also still fixes the initialization it was originally introduced for: ``IndexerNorm`` leaves
    #: q and k at unit variance per channel, so an unscaled score would start ``w`` nearly one-hot on
    #: a *randomly* chosen chunk -- far from the frozen backbone's output, which is the prior this
    #: objective exists to preserve. Reuses :attr:`~.indexer.GQAIndexer.GATE_SCALE_INIT` so the
    #: consumers of the score cannot drift apart on what its natural magnitude is.
    score_scale: float | None = None

    #: Also measure ``corr(score, chunk LSE)`` per layer. Costs one extra reduction over the
    #: attention logits inside the tile, plus retaining nothing.
    #:
    #: Worth having on: ``ROUTER_LEARNABILITY.md`` §6 proves the optimal score for a frozen backbone
    #: **is** the chunk log-sum-exp up to a per-query constant, so this measures progress against a
    #: closed-form target on real text. No other arm in this package has that.
    measure_lse_corr: bool = True

    #: Layer index -> normalized chunk-weight entropy, in ``[0, 1]``. 1.0 = uniform mixing (the
    #: router has learned no ranking), 0 = fully committed. **The** diagnostic here: a loss that
    #: descends while this stays at 1.0 means the router learned to use a blunt average rather than
    #: to choose, which is this objective's analogue of the flat-gate no-op.
    chunk_entropy: dict[int, float] = field(default_factory=dict)
    #: Layer index -> mean of ``max_c w_c``. How concentrated the mass is, i.e. how much a top-k
    #: truncation at inference would retain.
    mass_top1: dict[int, float] = field(default_factory=dict)
    #: Layer index -> mean mass on the top quarter of chunks. The closest training-time proxy for
    #: what the eval's ``topk`` budget will keep.
    mass_topquarter: dict[int, float] = field(default_factory=dict)
    #: Layer index -> Spearman correlation between the router's score and the chunk LSE, over
    #: visible chunks. The closed-form optimum is correlation 1.
    score_lse_corr: dict[int, float] = field(default_factory=dict)
    #: Number of layers that ran, as a wiring check.
    layers_routed: int = field(default=0, init=False)

    _hidden_states: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _kwargs: dict[int, dict] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        if self.chunk_size <= 1:
            raise ValueError(
                f"chunk_size must be > 1, got {self.chunk_size}. At chunk_size=1 the within-chunk "
                "softmax is softmax(one element) = 1, so the q.k term drops out entirely and the "
                "router would have to learn the whole attention distribution itself -- losing both "
                "the two-level decomposition and the frozen backbone's prior. See "
                "ROUTER_LEARNABILITY.md section 6; this is structural, not a tuning choice."
            )
        if self.chunk_aggregate.lower() not in (
            "lse", "logsumexp", "mean", "avg", "max", "amax"
        ):
            raise ValueError(
                f"chunk_aggregate must be lse, mean or max, got {self.chunk_aggregate!r}"
            )

    def resolved_score_scale(self, indexer) -> float:
        """:attr:`score_scale`, defaulting to ``head_dim ** -0.5``. See that attribute."""
        if self.score_scale is not None:
            return float(self.score_scale)
        head_dim = getattr(indexer, "head_dim", None)
        if not head_dim:
            return 1.0
        return float(type(indexer).GATE_SCALE_INIT(head_dim))

    def reset(self) -> None:
        self.chunk_entropy = {}
        self.mass_top1 = {}
        self.mass_topquarter = {}
        self.score_lse_corr = {}
        self.layers_routed = 0
        self._hidden_states.clear()
        self._kwargs.clear()

    # ------------------------------------------------------------------
    # Parameters -- identical to the other two trainers, so the arms are comparable
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

        Identifies indexers by **module identity**, not by a parameter-name substring: a name filter
        would also catch a backbone parameter containing the attribute name and silently train it.

        ``requires_grad=False``, never ``torch.no_grad()``: the gradient must still *flow through*
        the frozen backbone to reach the router, which is the whole point of an end-to-end objective.
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
        Router scores per ``(query, chunk)``: ``(B, Hkv, Sq, n_chunk)``.

        Built from the same ``hidden_states`` and RoPE tables the layer itself uses, via the press's
        own :meth:`~.press.GQAIndexerPress.get_rope_tables` -- the invariant that keeps the trained
        score and the scored-at-inference score the same function.

        Tiled over queries and checkpointed, because the intermediate is ``chunk_size`` times the
        result: ``(B, Hkv, Sq, Sk)`` fp32 logits are 8 GiB at 16K against 128 MiB of output.
        """
        indexer = self.press.get_indexer(module)
        cos, sin = self.press.get_rope_tables(indexer, kwargs)
        q_len = hidden_states.shape[1]
        attention_mask = kwargs.get("attention_mask")
        if self.detach_score_input:
            hidden_states = hidden_states.detach()

        budget = self.score_tile_bytes or SCORE_TILE_BYTES
        n_kv = indexer.n_heads
        per_query = hidden_states.shape[0] * n_kv * k_len * 4
        tile = max(1, budget // max(per_query, 1))

        blocks = []
        for start in range(0, q_len, tile):
            stop = min(start + tile, q_len)
            mask = build_indexer_mask(
                stop - start,
                k_len,
                hidden_states.device,
                attention_mask=attention_mask,
                # The absolute position of this tile's first query. Defaulting it would let every
                # tile see the whole prefix -- attention would still be causal (its own mask
                # enforces that) while the ROUTER trained on scores that peeked ahead. Silent.
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
                self.resolved_score_scale(indexer),
            )
            if self.checkpoint_scores and torch.is_grad_enabled():
                blocks.append(torch_checkpoint(_score_tile, *args, use_reentrant=False))
            else:
                blocks.append(_score_tile(*args))
        # The scale was applied to the TOKEN scores inside _score_tile, before the aggregation --
        # required for `lse`, harmless for `mean`/`max` (both are scale-equivariant).
        return torch.cat(blocks, dim=2)

    def routed_forward(
        self,
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask,
        scaling: float | None,
    ) -> torch.Tensor:
        """Replacement attention for one layer: two-level chunk attention over every chunk."""
        layer_idx = int(module.layer_idx)
        # POPPED, not read: leaving the entry in the dict pins that layer's (B, L, hidden) tensor for
        # the whole forward AND backward, and also stops autograd releasing each layer's activations
        # as it walks up.
        hidden_states = self._hidden_states.pop(layer_idx, None)
        if hidden_states is None:
            raise RuntimeError(
                f"layer {layer_idx} reached the HSA attention without its hidden_states being "
                "captured. The pre-hook that records them must be registered on the same modules as "
                "the attention swap -- use HSAIndexerTrainer.hooks(). A second call for the same "
                "layer in one forward (gradient checkpointing recomputes a block) also lands here, "
                "because the entry is consumed on first use."
            )
        kwargs = self._kwargs.pop(layer_idx, {})

        k_len = key.shape[2]
        scores = self.chunk_scores(module, hidden_states, kwargs, k_len)
        out, stats = hsa_chunk_attention(
            query,
            key,
            value,
            scores,
            chunk_size=self.chunk_size,
            scaling=scaling,
            checkpoint=self.checkpoint_attention,
            query_tile=0,
        )

        self.chunk_entropy[layer_idx] = stats["chunk_entropy"]
        self.mass_top1[layer_idx] = stats["mass_top1"]
        self.mass_topquarter[layer_idx] = stats["mass_topquarter"]
        if self.measure_lse_corr:
            self.score_lse_corr[layer_idx] = self._lse_correlation(
                query, key, scores, scaling, k_len
            )
        self.layers_routed += 1

        # The attention interface contract is (B, Sq, H, D); our op returns (B, H, Sq, D).
        return out.transpose(1, 2).contiguous()

    @torch.no_grad()
    def _lse_correlation(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        scores: torch.Tensor,
        scaling: float | None,
        k_len: int,
    ) -> float:
        """
        Spearman correlation between the router's score and the chunk log-sum-exp.

        The closed-form target -- ``ROUTER_LEARNABILITY.md`` §6: for a frozen backbone the optimal
        ``s`` is ``LSE_c`` up to a per-query constant. **Spearman**, not Pearson, precisely because of
        that constant: only the ranking is determined, and the eval takes a top-k, so rank is the
        quantity that matters. Computed per query row and averaged, so a per-row offset cannot
        contribute.

        Subsampled to a few query rows: this is a diagnostic, and computing it for every row would
        cost a second full ``q @ k^T``. ``no_grad`` throughout -- it must not enter the graph.
        """
        from kvpress.presses.gqa_indexer.hsa_attention import chunk_lse

        b, n_heads, q_len, head_dim = query.shape
        n_kv = key.shape[1]
        group = n_heads // n_kv
        scale = head_dim**-0.5 if scaling is None else float(scaling)
        query_offset = k_len - q_len
        # Later rows see more chunks, so a correlation over the first rows would be measured on
        # 1-2 chunks. Sample from the second half, where a rank statistic has something to rank.
        n_probe = min(8, max(1, q_len // 2))
        rows = torch.linspace(q_len // 2, q_len - 1, n_probe, device=query.device).long().unique()

        q_probe = query.view(b, n_kv, group, q_len, head_dim)[:, :, 0, rows]  # (B, Hkv, r, D)
        logits = torch.einsum("bhrd,bhsd->bhrs", q_probe.float(), key.float()) * scale
        q_pos = rows + query_offset
        causal = torch.arange(k_len, device=query.device).view(1, k_len) <= q_pos.view(-1, 1)
        lse = chunk_lse(logits, self.chunk_size, valid=causal.view(1, 1, len(rows), k_len))
        s = scores[:, :, rows].float()

        n_chunk = s.shape[-1]
        chunk_start = torch.arange(n_chunk, device=query.device) * self.chunk_size
        vis = chunk_start.view(1, n_chunk) <= q_pos.view(-1, 1)  # (r, n_chunk)

        total, count = 0.0, 0
        for r in range(len(rows)):
            m = vis[r]
            if int(m.sum()) < 3:  # a rank correlation over 2 points is +-1 by construction
                continue
            a = s[..., r, :][..., m].reshape(-1, int(m.sum()))
            c = lse[..., r, :][..., m].reshape(-1, int(m.sum()))
            ra = a.argsort(-1).argsort(-1).float()
            rc = c.argsort(-1).argsort(-1).float()
            ra = ra - ra.mean(-1, keepdim=True)
            rc = rc - rc.mean(-1, keepdim=True)
            denom = (ra.norm(dim=-1) * rc.norm(dim=-1)).clamp_min(1e-12)
            total += float(((ra * rc).sum(-1) / denom).mean())
            count += 1
        return total / count if count else float("nan")

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def _capture_hook(self, module: nn.Module, args, kwargs: dict):
        """Stash this layer's ``hidden_states`` and kwargs -- the attention interface never sees them."""
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
        Route every attention layer through two-level chunk attention for the block's duration.

        The registry cleanup goes through ``_global_mapping`` directly for the reason
        :func:`~.teacher_lse.capture_teacher_lse` documents: ``register()`` writes there while
        ``pop()`` only touches the instance mapping, so the naive removal leaks the entry forever.
        """
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self.press.post_init_from_model(model)
        self.reset()
        if self.freeze:
            self.freeze_backbone(model)

        impl_name = "kvpress_gqa_indexer_hsa"

        def hsa_attention_impl(
            module, query, key, value, attention_mask, scaling=None, dropout=0.0, **_
        ):
            return self.routed_forward(module, query, key, value, attention_mask, scaling), None

        configs = [model.config]
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None:
            configs.append(text_config)
        previous_impls = [cfg._attn_implementation for cfg in configs]

        global_mapping = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
        had_previous = impl_name in global_mapping
        previous_fn = global_mapping.get(impl_name)
        ALL_ATTENTION_FUNCTIONS.register(impl_name, hsa_attention_impl)

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
    def mean_chunk_entropy(self) -> float | None:
        """Mean normalized entropy over the layers that ran. 1.0 = uniform, 0 = committed."""
        return _mean(self.chunk_entropy)

    def mean_mass_top1(self) -> float | None:
        return _mean(self.mass_top1)

    def mean_mass_topquarter(self) -> float | None:
        return _mean(self.mass_topquarter)

    def mean_score_lse_corr(self) -> float | None:
        """Mean Spearman against the closed-form optimum. Rising means the router is learning."""
        return _mean(self.score_lse_corr)


def _mean(values: dict[int, float]) -> float | None:
    finite = [v for v in values.values() if v is not None and v == v]  # v == v drops NaN
    if not finite:
        return None
    return sum(finite) / len(finite)


def hsa_indexer_training_step(
    model: nn.Module,
    trainer: HSAIndexerTrainer,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    skip_logits: bool | None = None,
) -> torch.Tensor:
    """
    One routed forward pass, returning the model's own LM loss.

    ``skip_logits=True`` asks a Liger-patched model to fuse ``lm_head`` into the cross-entropy so the
    ``(L, vocab)`` logits are never materialized -- 7.0 GiB at ``L=8192`` on Qwen3-8B. It must be
    passed **explicitly**: Liger's own default is ``self.training and labels is not None``, and this
    backbone is deliberately left in ``eval()`` to keep dropout off, so the default resolves to
    ``False`` and the patch silently saves nothing. ``None`` because an unpatched model does not
    accept the kwarg.
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
                "no layer ran the HSA attention: the model kept its own attention implementation. "
                "This usually means the model's config is not the one HSAIndexerTrainer pointed at "
                f"{model.config._attn_implementation!r}."
            )
        return out.loss
