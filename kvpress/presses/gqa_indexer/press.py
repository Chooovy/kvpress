# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV cache compression driven by the GQA lightning indexer."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import cast

import torch
from torch import nn

from kvpress.presses.gqa_indexer.aggregate import aggregate_chunk_scores, reduce_queries
from kvpress.presses.gqa_indexer.dma_indexer import DMAIndexer, DMAIndexerConfig
from kvpress.presses.gqa_indexer.indexer import (
    MASK_NEG,
    GQAIndexer,
    GQAIndexerConfig,
    build_indexer_mask,
    slice_rope_tables,
)
from kvpress.presses.gqa_indexer.memory import MemoryConfig, MemoryKernel
from kvpress.presses.gqa_indexer.prefix_indexer import PrefixIndexer, PrefixIndexerConfig
from kvpress.presses.gqa_indexer.scalar_indexer import (
    DEFAULT_DECAY_INIT,
    DEFAULT_DECAY_REF,
    DEFAULT_POS_SLOPE,
    ScalarIndexer,
    ScalarIndexerConfig,
)
from kvpress.presses.gqa_indexer.kvzip_indexer import (
    DEFAULT_KVZIP_BASE,
    DEFAULT_KVZIP_DIM,
    KVzipIndexer,
    KVzipIndexerConfig,
)
from kvpress.presses.scorer_press import ScorerPress

logger = logging.getLogger(__name__)

#: ``scorer`` name -> module class. ``"prefix"`` must map to :class:`PrefixIndexer` before any
#: ``isinstance`` reasoning is applied elsewhere: it *is* a :class:`ScalarIndexer`, so an
#: ``isinstance`` dispatch would resolve the wrong way round. Nothing in this package does that
#: today -- the scorers are consumed through the duck-typed protocol -- and this table exists so
#: it stays that way.
_SCORER_CLASSES = {
    "pairwise": GQAIndexer,
    "scalar": ScalarIndexer,
    "prefix": PrefixIndexer,
    "dma": DMAIndexer,
    "kvzip": KVzipIndexer,
}


def get_language_model(model: nn.Module) -> nn.Module:
    """Return the text backbone, unwrapping the VL nesting used by multimodal models."""
    return model.model.language_model if hasattr(model.model, "language_model") else model.model


@dataclass
class GQAIndexerPress(ScorerPress):
    """
    Prune the KV cache using a learned per-KV-head lightning indexer.

    The indexer scores every (query, key) pair once per KV head, those scores are reduced
    over the query axis into one importance value per key, and the lowest-scoring keys are
    evicted -- independently for each KV head, which is what GQA's separate caches allow.

    Because scores are per-KV-head, this press relies on kvpress's head-wise compression
    path. Set ``mean_head=True`` to collapse to a single shared selection instead (useful
    as an ablation against the head-uniform DSA-style behaviour).

    Parameters
    ----------
    compression_ratio : float
        Fraction of KV pairs to remove.
    n_heads, head_dim : int, optional
        Indexer geometry. Left as ``None`` they default to ``num_key_value_heads`` and the
        attention ``head_dim``: one indexer query head per KV head, so each KV head scores
        and evicts independently.
    rope_dim : int, optional
        Channels to rotate. ``None`` rotates the full ``head_dim`` (capped by the model's
        rotary dim); ``0`` disables RoPE.
    query_reduce : str
        How the query axis collapses: ``mean``, ``max``, ``last`` or ``recency``.
    last_n_query : int, optional
        Restrict query reduction to the final N queries.
    recency_half_life : float
        Half-life for ``query_reduce="recency"``.
    n_sink : int
        Always-kept tokens at the start of the sequence.
    n_local : int
        Always-kept most recent tokens. Sink+local protection is standard for streaming
        eviction and is applied on top of whatever the indexer scores.
    chunk_size : int
        ``0`` selects at token granularity. ``>0`` pools token scores into chunks of this
        size and selects whole chunks, which preserves local contiguity.
    chunk_aggregate : str
        ``mean`` or ``max`` pooling over each chunk's token scores.
    use_vnorm : bool
        Scale scores by the value-vector norm, as several kvpress scorers do.
    gate_scale : bool
        Give each indexer the learnable ``gate_scale`` scalar that end-to-end (gated-attention)
        training needs. Off by default: distillation never reads it, so a distillation
        checkpoint stays free of the extra parameter. Required by
        :class:`~kvpress.presses.gqa_indexer.e2e_trainer.E2EIndexerTrainer`. DMA instead uses
        the per-head ``A`` in its score and fixes this outer multiplier to one.
    scorer : str
        Which scorer to attach. ``"pairwise"`` is :class:`~.indexer.GQAIndexer`, which scores
        every ``(query, key)`` pair -- query-aware, and ``O(t)`` per decode step.
        ``"scalar"`` is :class:`~.scalar_indexer.ScalarIndexer`, which scores each key once from
        its own hidden state -- ``O(1)`` per decode step and one score per token of cache
        instead of ``head_dim``, at the cost of query-awareness.
        ``"prefix"`` is :class:`~.prefix_indexer.PrefixIndexer`, which scores each key once from
        its whole *prefix* via the indexer's own causal attention. Still query-independent, so it
        can evict, but its view of a key is the prefix rather than the single vector ``h_j``. It
        is a strict superset of ``"scalar"`` -- with ``prefix_zero_init`` (the default) the score
        starts bit-identical -- so the A/B against it is single-variable. Prefill-time only.
        ``"dma"`` is :class:`~.dma_indexer.DMAIndexer`, which applies DMA's query-independent
        sampling formula directly to the cached value vectors.
    scalar_mid_dim : int
        MLP width for ``scorer="scalar"`` and ``scorer="prefix"``. ``0`` is SparseK's plain linear
        score; the arm's capacity knob otherwise. See
        :class:`~.scalar_indexer.ScalarIndexerConfig`.
    scalar_pos_slope : float
        Recency tilt for ``scorer="scalar"`` and ``scorer="prefix"``. Keeps top-k irreversible,
        which is what makes the dropped keys safe to free.
    prefix_head_dim, prefix_value_dim : int
        ``scorer="prefix"`` only: width of the prefix attention's ``q``/``k`` and of its ``v``
        readout. ``prefix_value_dim`` is what a decode-time indexer cache would cost per token
        per layer, alongside ``prefix_head_dim`` for the keys.
    prefix_zero_init : bool
        ``scorer="prefix"`` only: zero-initialize the prefix branch's output projection so
        training starts exactly at the scalar arm. On by default, which is what makes "read the
        prefix" the single variable in the comparison.
    scorer_attr : str
        Attribute name the indexer is registered under on each attention module.
    memory : bool
        Attach a :class:`~.memory.MemoryKernel` per layer, so the keys this press evicts are first
        compressed into a constant-size linear state and folded back into the same softmax at
        inference (see :mod:`~.memory`). Off by default: the eviction baseline is the thing the
        memory arm is measured against, and a distillation or router-only checkpoint must stay free
        of the extra parameters.

        Requires ``scorer="scalar"``. Query-independence is not a convenience here but the
        precondition: ``E_t`` has to be a function of ``t`` alone for one accumulated state to
        represent it, and under a pairwise scorer the evicted set varies per query.
    memory_rank, memory_mid_dim : int
        State rank ``R`` and trunk width ``R'``. ``memory_rank=0`` is the rank-0 ablation (one
        learned vector per head times the mass), which measured 78% relative residual against the
        exact evicted output -- so it is the cheap comparison, expected to lose.
    memory_per_head : bool
        Give each KV head its own kernel readout, sharing the trunk.
    memory_attr : str
        Attribute name the memory module is registered under.
    """

    compression_ratio: float = 0.0
    mean_head: bool = False

    # Indexer geometry (None -> derive from model config)
    n_heads: int | None = None
    head_dim: int | None = None
    rope_dim: int | None = None
    gate_scale: bool = False

    # Which scorer, and the scalar arm's own knobs
    scorer: str = "pairwise"
    scalar_mid_dim: int = 256
    scalar_pos_slope: float = DEFAULT_POS_SLOPE
    #: TrimKV's per-key lifetime. Widens the indexer to ``2 * n_heads`` and makes the gate
    #: ``s_j + log_beta_j * (i - j) / decay_ref``. See :class:`~.scalar_indexer.ScalarIndexerConfig`.
    scalar_decay: bool = False
    scalar_decay_ref: float = DEFAULT_DECAY_REF
    scalar_decay_init: float = DEFAULT_DECAY_INIT

    # scorer="prefix" only: the prefix-attention branch's geometry
    prefix_head_dim: int = 128
    prefix_value_dim: int = 128
    prefix_zero_init: bool = True

    # scorer="kvzip" only: Fast-KVzip's gate geometry. kvzip_ngroup=0 means "derive it
    # from the model", i.e. n_q_heads / n_kv_heads, which is what upstream uses.
    kvzip_dim: int = DEFAULT_KVZIP_DIM
    kvzip_base: int = DEFAULT_KVZIP_BASE
    kvzip_ngroup: int = 0

    # Query-axis reduction
    query_reduce: str = "mean"
    last_n_query: int | None = None
    recency_half_life: float = 32.0

    # Protection
    n_sink: int = 4
    n_local: int = 0

    # Chunk-level selection (applied to token scores)
    chunk_size: int = 0
    chunk_aggregate: str = "mean"

    use_vnorm: bool = False
    scorer_attr: str = "indexer"

    # Linear memory over the evicted keys
    memory: bool = False
    memory_rank: int = 16
    memory_mid_dim: int = 256
    memory_per_head: bool = True
    memory_attr: str = "kv_memory"

    _initialized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if self.n_sink < 0 or self.n_local < 0:
            raise ValueError("n_sink and n_local must be non-negative")
        if self.chunk_size < 0:
            raise ValueError("chunk_size must be non-negative")
        if self.scorer not in ("pairwise", "scalar", "prefix", "dma", "kvzip"):
            raise ValueError(
                "scorer must be 'pairwise', 'scalar', 'prefix', 'dma' or 'kvzip', "
                f"got {self.scorer!r}"
            )
        if self.memory and self.scorer != "scalar":
            # Structural, not a missing feature. The memory holds one state per KV head standing
            # for the whole evicted set, which only exists if the evicted set is a function of the
            # query position alone. A pairwise scorer's E_t varies per query, so no constant-size
            # state represents it -- the module would run and quietly summarize the wrong keys.
            raise ValueError(
                f"memory=True requires scorer='scalar', got {self.scorer!r}. The memory keeps one "
                "constant-size state per KV head for the evicted set, which presupposes that the "
                "set depends only on the query's position; a query-dependent scorer evicts a "
                "different set for every query and no single state can stand for it."
            )
        if self.memory and self.memory_mid_dim <= 0:
            raise ValueError(f"memory_mid_dim must be positive, got {self.memory_mid_dim}")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def build_indexer_config(self, model: nn.Module, module: nn.Module):
        """Derive indexer geometry from the model, honouring any explicit overrides.

        Returns a config for whichever scorer :attr:`scorer` names. All scorer configs report
        ``rope_dim`` and ``n_heads``, which is all the press and the trainers read off them.
        """
        config = model.config
        text_config = getattr(config, "text_config", config)

        # One indexer head per KV head: that is the granularity at which GQA holds
        # physically separate caches, so it is the granularity at which eviction can differ.
        n_heads = self.n_heads or text_config.num_key_value_heads

        if self.scorer == "dma":
            model_n_heads = text_config.num_key_value_heads
            model_head_dim = getattr(
                module, "head_dim", text_config.hidden_size // text_config.num_attention_heads
            )
            if n_heads != model_n_heads or self.head_dim not in (None, model_head_dim):
                raise ValueError(
                    "scorer='dma' is defined on the model's actual value tensor, so n_heads and "
                    "head_dim must match its KV geometry"
                )
            if self.rope_dim not in (None, 0):
                raise ValueError("scorer='dma' scores values directly and does not use RoPE")
            return DMAIndexerConfig(n_heads=model_n_heads, head_dim=model_head_dim)

        if self.scorer in ("scalar", "prefix", "kvzip"):
            # No head_dim and no rope_dim: the score is one number per (key, head), derived from
            # that key (or its prefix) alone, so there is nothing to rotate. The prefix arm's own
            # attention is deliberately NoPE -- h_j already carries the backbone's rotary signal
            # and pos_slope carries recency -- and its q/k width is prefix_head_dim, not this
            # head_dim. Overrides here would silently do nothing, so reject them.
            for name in ("head_dim", "rope_dim"):
                if getattr(self, name) is not None:
                    hint = (
                        " Use prefix_head_dim for the prefix attention's q/k width."
                        if name == "head_dim" and self.scorer == "prefix"
                        else ""
                    )
                    raise ValueError(
                        f"{name} was set but scorer={self.scorer!r} has no q/k geometry to apply "
                        f"it to; the score has no per-head dimension and no rotary width. "
                        f"Drop {name}, or use scorer='pairwise'.{hint}"
                    )
            common = dict(
                hidden_size=text_config.hidden_size,
                n_heads=n_heads,
                mid_dim=self.scalar_mid_dim,
                pos_slope=self.scalar_pos_slope,
                gate_scale=self.gate_scale,
            )
            if self.scorer == "prefix":
                if self.scalar_decay:
                    # PrefixIndexer overrides score_keys with its own prefix-attention readout and
                    # has no lifetime head; silently dropping the flag would train the wrong arm.
                    raise ValueError(
                        "scalar_decay is only implemented for scorer='scalar'. The prefix arm "
                        "computes its score through prefix attention, which the decay fold does "
                        "not cover."
                    )
                return PrefixIndexerConfig(
                    **common,
                    head_dim=self.prefix_head_dim,
                    value_dim=self.prefix_value_dim,
                    zero_init_prefix=self.prefix_zero_init,
                )
            decay_kwargs = dict(
                decay=self.scalar_decay,
                decay_ref=self.scalar_decay_ref,
                decay_init=self.scalar_decay_init,
            )
            if self.scorer == "kvzip":
                # No MLP in this head, so mid_dim is dropped rather than passed -- the config
                # rejects a nonzero one, which is what keeps a swept --scalar-mid-dim from
                # silently meaning nothing here.
                common.pop("mid_dim", None)
                ngroup = self.kvzip_ngroup or (
                    text_config.num_attention_heads // text_config.num_key_value_heads
                )
                return KVzipIndexerConfig(
                    **common,
                    **decay_kwargs,
                    kvzip_dim=self.kvzip_dim,
                    kvzip_base=self.kvzip_base,
                    kvzip_ngroup=ngroup,
                )
            return ScalarIndexerConfig(**common, **decay_kwargs)

        head_dim = self.head_dim or getattr(
            module, "head_dim", text_config.hidden_size // text_config.num_attention_heads
        )

        if self.rope_dim is None:
            # Default to rotating everything, but never claim more rotary channels than
            # the model's tables actually provide.
            model_rotary = getattr(text_config, "head_dim", head_dim)
            partial = getattr(text_config, "partial_rotary_factor", 1.0) or 1.0
            rope_dim = min(head_dim, int(model_rotary * partial))
            rope_dim -= rope_dim % 2
        else:
            rope_dim = self.rope_dim

        return GQAIndexerConfig(
            hidden_size=text_config.hidden_size,
            n_heads=n_heads,
            head_dim=head_dim,
            rope_dim=rope_dim,
            gate_scale=self.gate_scale,
        )

    def build_memory_config(self, model: nn.Module, module: nn.Module) -> MemoryConfig:
        """Derive the memory geometry from the model's KV geometry.

        The state lives on ``(n_kv_heads, head_dim)`` because that is the shape of what is evicted:
        GQA holds physically separate caches per KV head, so each head's evicted set -- and hence
        its summary -- is its own.
        """
        text_config = getattr(model.config, "text_config", model.config)
        head_dim = getattr(
            module,
            "head_dim",
            text_config.hidden_size // text_config.num_attention_heads,
        )
        return MemoryConfig(
            n_kv_heads=text_config.num_key_value_heads,
            head_dim=head_dim,
            rank=self.memory_rank,
            mid_dim=self.memory_mid_dim,
            per_head_readout=self.memory_per_head,
        )

    def post_init_from_model(self, model: nn.Module, force_reinit: bool = False) -> None:
        """
        Attach a :class:`GQAIndexer` to every attention layer.

        Idempotent when the attached indexer already has the geometry this press would
        build, and a **hard error** when it does not. Re-attaching is skipped rather than
        redone so a loaded checkpoint survives a second call, but silently accepting a
        mismatched indexer would mean the press scores with geometry other than the one it
        was configured for -- and ``rope_dim`` in particular is not a parameter shape, so
        ``load_indexer_state_dict`` cannot catch it either. Pass ``force_reinit=True`` to
        deliberately replace the existing indexers (discarding their weights).
        """
        if self._initialized and not force_reinit:
            return

        language_model = get_language_model(model)
        created = 0
        created_memory = 0
        for layer in language_model.layers:
            attn = layer.self_attn
            indexer_config = self.build_indexer_config(model, attn)
            existing = getattr(attn, self.scorer_attr, None)
            if existing is not None and not force_reinit:
                if existing.config != indexer_config:
                    raise ValueError(
                        f"{type(attn).__name__} already has a {self.scorer_attr!r} with geometry "
                        f"{existing.config}, but this press is configured for {indexer_config}. "
                        "Scoring would silently use the attached geometry rather than the "
                        "requested one. Use a fresh model, or force_reinit=True to replace it "
                        "(which discards the existing weights)."
                    )
            else:
                cls = _SCORER_CLASSES[self.scorer]
                indexer = cls(indexer_config).to(device=model.device, dtype=model.dtype)
                attn.register_module(self.scorer_attr, indexer)
                created += 1

            if not self.memory:
                continue
            memory_config = self.build_memory_config(model, attn)
            existing_memory = getattr(attn, self.memory_attr, None)
            if existing_memory is not None and not force_reinit:
                if existing_memory.config != memory_config:
                    raise ValueError(
                        f"{type(attn).__name__} already has a {self.memory_attr!r} with geometry "
                        f"{existing_memory.config}, but this press is configured for "
                        f"{memory_config}. Use a fresh model, or force_reinit=True."
                    )
                continue
            memory = MemoryKernel(memory_config).to(device=model.device, dtype=model.dtype)
            # The scalars are created in fp32 and .to(dtype=...) just cast them back down. Undo it
            # here rather than leaving it to the trainer: in bf16 a warmup learning rate rounds
            # every step back and they stay frozen at initialization, which is the failure
            # E2EIndexerTrainer.upcast_gate_scales documents (30 steps, all 36 layers, invisible).
            memory.upcast_scalars()
            attn.register_module(self.memory_attr, memory)
            created_memory += 1

        self._initialized = True
        if created:
            logger.info("Initialized %d %s modules", created, self.scorer_attr)
        if created_memory:
            logger.info("Initialized %d %s modules", created_memory, self.memory_attr)

    def get_indexer(self, module: nn.Module) -> GQAIndexer:
        indexer = getattr(module, self.scorer_attr, None)
        if indexer is None:
            raise RuntimeError(
                f"No {self.scorer_attr!r} found on {type(module).__name__}. "
                "Call post_init_from_model(model) before using this press."
            )
        return indexer

    def get_memory(self, module: nn.Module) -> MemoryKernel:
        """This layer's :class:`~.memory.MemoryKernel`, raising when the press has none.

        Raises rather than returning ``None`` so a caller cannot silently skip the memory term and
        report numbers for the plain eviction baseline under the memory arm's name.
        """
        memory = getattr(module, self.memory_attr, None)
        if memory is None:
            raise RuntimeError(
                f"No {self.memory_attr!r} found on {type(module).__name__}. Build the press with "
                "memory=True and call post_init_from_model(model)."
            )
        return memory

    # ------------------------------------------------------------------
    # RoPE plumbing
    # ------------------------------------------------------------------
    def get_rope_tables(self, indexer: GQAIndexer, kwargs: dict) -> tuple:
        """
        Narrow the layer's RoPE tables to the indexer's rotary width.

        Returns ``(None, None)`` only when the indexer has RoPE *disabled*
        (``rope_dim == 0``). If it wants RoPE and the tables are missing, this **raises**
        rather than quietly scoring without positions: at inference the press is hooked onto
        ``self_attn`` and so always receives ``position_embeddings``, so an absent table means
        a caller (usually training code) is about to build a student that differs from the one
        the press runs -- and nothing downstream would flag it.

        The narrowing itself is subtle -- see
        :func:`~kvpress.presses.gqa_indexer.indexer.slice_rope_tables`.
        """
        if indexer.rope_dim == 0:
            return None, None
        position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is None:
            raise ValueError(
                f"the indexer has rope_dim={indexer.rope_dim} but no position_embeddings were "
                "supplied, so it would score without any positional signal -- a different "
                "student than the one the press runs at inference. Pass position_embeddings, "
                "or set rope_dim=0 to opt into NoPE deliberately."
            )
        cos, sin = position_embeddings
        if cos.dim() == 4:  # some models emit (B, 1, S, D)
            cos, sin = cos.squeeze(1), sin.squeeze(1)
        return slice_rope_tables(cos, sin, indexer.rope_dim)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def indexer_logits(
        self, module: nn.Module, hidden_states: torch.Tensor, kwargs: dict, k_len: int | None = None
    ) -> torch.Tensor:
        """
        Run the indexer and return causally-masked token logits (B, n_kv_heads, Sq, Sk).

        Shared by the press and the training code so both see identical logits -- the
        single most important invariant here, since any divergence between the two silently
        trains the indexer for a scoring function it never uses at inference.
        """
        indexer = self.get_indexer(module)
        cos, sin = self.get_rope_tables(indexer, kwargs)
        q_len = hidden_states.shape[1]
        k_len = k_len if k_len is not None else q_len
        mask = build_indexer_mask(
            q_len,
            k_len,
            hidden_states.device,
            attention_mask=kwargs.get("attention_mask"),
        )
        return indexer(hidden_states, cos=cos, sin=sin, mask=mask)

    def token_scores(
        self, module: nn.Module, hidden_states: torch.Tensor, kwargs: dict, k_len: int
    ) -> torch.Tensor:
        """Per-KV-head, per-token importance -> (B, n_kv_heads, k_len)."""
        logits = self.indexer_logits(module, hidden_states, kwargs, k_len=k_len)
        return reduce_queries(
            logits,
            self.query_reduce,
            # The logits carry MASK_NEG on forbidden pairs; without this the averaging
            # modes fold the sentinel into the mean and rank keys by position instead of
            # content. Recovering validity from the logits themselves (rather than
            # rebuilding the mask) keeps the two in step by construction.
            valid_mask=logits > (MASK_NEG / 2),
            last_n_query=self.last_n_query,
            recency_half_life=self.recency_half_life,
        )

    def protect_boundaries(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Force sink and local tokens to outrank every scored token.

        A finite sentinel (``max of the finite scores + 1``) is used rather than ``+inf``:
        infinities poison later arithmetic (``inf * 0`` is NaN) and would leak through
        reductions. Protected tokens all share the sentinel, so among themselves top-k falls
        back to index order -- which is fine, since they are all kept.

        If protection alone exceeds the keep budget, ScorerPress's top-k will silently drop
        some protected tokens -- warn rather than let that pass unnoticed.
        """
        k_len = scores.shape[-1]
        n_sink = min(self.n_sink, k_len)
        n_local = min(self.n_local, max(k_len - n_sink, 0))
        if n_sink == 0 and n_local == 0:
            return scores

        n_protected = n_sink + n_local
        n_kept = int(k_len * (1 - self.compression_ratio))
        if n_protected > n_kept > 0:
            logger.warning(
                "n_sink + n_local = %d exceeds the keep budget of %d at compression_ratio=%.2f; "
                "some protected tokens will be evicted.",
                n_protected,
                n_kept,
                self.compression_ratio,
            )

        finite_max = torch.nan_to_num(scores, neginf=0.0, posinf=0.0).amax()
        sentinel = finite_max + 1.0
        scores = scores.clone()
        if n_sink:
            scores[..., :n_sink] = sentinel
        if n_local:
            scores[..., k_len - n_local :] = sentinel
        return scores

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        """
        Importance scores, (B, num_kv_heads, k_len).

        Chunk-level selection is expressed here as a score transform rather than a custom
        ``compress`` override: every token in a kept chunk inherits the chunk's pooled
        score, so ScorerPress's ordinary top-k picks up whole chunks while sink/local
        protection and the ragged tail keep working unchanged.
        """
        k_len = keys.shape[2]
        if self.scorer == "dma":
            key_mask = None
            if kwargs.get("attention_mask") is not None:
                key_mask = build_indexer_mask(
                    1,
                    k_len,
                    values.device,
                    attention_mask=kwargs["attention_mask"],
                    query_offset=k_len - 1,
                )[:, 0, 0] > (MASK_NEG / 2)
            indexer = cast(DMAIndexer, self.get_indexer(module))
            scores = indexer.score_values(values, mask=key_mask)
        else:
            scores = self.token_scores(module, hidden_states, kwargs, k_len)

        if scores.shape[-1] != k_len:
            raise RuntimeError(
                f"indexer scored {scores.shape[-1]} keys but the cache holds {k_len}. "
                "This press currently supports prefill-time compression only."
            )

        if self.use_vnorm:
            # Applied to token scores, before any chunk pooling, so a chunk's pooled score
            # reflects value magnitude too.
            scores = scores * values.norm(dim=-1).to(scores.dtype)

        if self.chunk_size > 0:
            scores = self.broadcast_chunk_scores(scores)

        if self.mean_head:
            # Ablation back to DSA-style head-uniform selection. Implemented as a score
            # transform so ScorerPress's ordinary per-head top-k then picks identical
            # positions for every head -- main's ScorerPress has no mean_head of its own.
            scores = scores.mean(dim=1, keepdim=True).expand_as(scores).contiguous()

        return self.protect_boundaries(scores)

    def broadcast_chunk_scores(self, token_scores: torch.Tensor) -> torch.Tensor:
        """
        Replace each token's score with its chunk's pooled score.

        The ragged tail (fewer than ``chunk_size`` tokens) keeps its token-level scores: a
        short chunk pooled by ``mean`` would compete unfairly against full chunks, and by
        ``max`` it would be indistinguishable from one.
        """
        chunk_scores, complete_end = aggregate_chunk_scores(
            token_scores, self.chunk_size, self.chunk_aggregate
        )
        if complete_end == 0:
            return token_scores
        broadcast = chunk_scores.repeat_interleave(self.chunk_size, dim=-1)
        if complete_end == token_scores.shape[-1]:
            return broadcast.to(token_scores.dtype)
        tail = token_scores[..., complete_end:]
        return torch.cat([broadcast.to(token_scores.dtype), tail], dim=-1)
