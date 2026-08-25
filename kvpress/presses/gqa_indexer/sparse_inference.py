# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Run the GQA indexer as *sparse attention* at inference.

The eviction press (:class:`~kvpress.presses.gqa_indexer.press.GQAIndexerPress`) drops keys from
the cache. This does the other thing the indexer enables, and the thing DSA ships: keep the whole
cache and let **each query attend only to its own top-k keys**, via
:func:`~kvpress.presses.gqa_indexer.triton_sparse_attention.sparse_gqa_attention`. Selection only --
the softmax is plain over the selected keys, with **no gate term** (that is the end-to-end training
path, :class:`~kvpress.presses.gqa_indexer.e2e_trainer.E2EIndexerTrainer`).

Mechanism
---------
The wiring mirrors ``E2EIndexerTrainer.hooks()`` exactly: a forward pre-hook on every attention
module stashes the ``hidden_states`` the indexer projects from (the attention *interface* only sees
q/k/v), and a temporary entry in ``ALL_ATTENTION_FUNCTIONS`` that ``config._attn_implementation``
is pointed at replaces the attention itself. Both are removed on exit.

The one thing training does not need and this does: an **indexer key-cache**. During training a
forward sees the whole sequence at once, so ``project_k`` produces every key. At inference the
attention interface only gets the *new* tokens' ``hidden_states`` each step, so we accumulate the
per-layer post-RoPE indexer keys ourselves -- initialize on prefill, append on each decode step --
and assert the cache length stays in lockstep with the model's own ``key.shape[2]``. The cache is
tiny (MQA: one ``head_dim`` per token per layer) and lives only for the duration of the ``with``
block, so a fresh context per generation gives a fresh cache.

Batch-1 assumption
------------------
Eval processes one context at a time, so there is no padding and the causal-arithmetic path of
:func:`~kvpress.presses.gqa_indexer.sparse_support.streaming_topk_support` (``mask=None``) is both
correct and cheap. For a padded batch the selector would need the additive mask built by
``build_indexer_mask``; that is deliberately not wired here.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from kvpress.presses.gqa_indexer.chunk_support import chunk_topk_support
from kvpress.presses.gqa_indexer.press import GQAIndexerPress, get_language_model
from kvpress.presses.gqa_indexer.qi_flex_attention import HAS_FLEX, qi_sparse_attention
from kvpress.presses.gqa_indexer.sparse_support import streaming_topk_support
from kvpress.presses.gqa_indexer.triton_sparse_attention import sparse_gqa_attention

logger = logging.getLogger(__name__)

#: The name the sparse attention is registered under in ``ALL_ATTENTION_FUNCTIONS``.
IMPL_NAME = "kvpress_gqa_indexer_sparse"


class SparseAttentionContext:
    """
    Replace every attention layer with indexer-driven sparse attention for the block's duration.

    Parameters
    ----------
    model : nn.Module
        The causal LM. Its attention layers are swapped on entry and restored on exit.
    press : GQAIndexerPress
        Holds the per-layer indexers (via ``post_init_from_model``) and the RoPE narrowing. The
        press's ``compression_ratio`` is irrelevant here -- nothing evicts; the indexer is used
        only to *select* per-query top-k keys.
    topk : int
        Support size per query, including the forced sink/local slots. Fixed, not a ratio: a ratio
        makes the retained support ``O(L^2)``.
    force_sink, force_local : int
        Slots reserved per row for the leading keys and the row's own most-recent keys, matching
        what the sparse training stage used.
    block_k : int
        Triton topk-tile; power of two, ``>= 16``. Memory/throughput knob only.
    causal : bool
        Mask slots past each query's diagonal. Redundant for causally-selected indices and cheap.
    chunk_size : int
        ``0`` (default) selects the top-``topk`` **tokens** per query -- correct for a router trained
        on per-token scores. ``>0`` selects whole **chunks** of this size instead, via
        :func:`~.chunk_support.chunk_topk_support`.

        Set this to the ``chunk_size`` the router trained with. A chunk-trained score (the exact-K
        arm) is close to piecewise-constant -- measured within/across-chunk variance ratio 0.16 at
        layer 4, against 0.99 for the token-trained gated arm -- so ranking its tokens resolves a
        near-tie inside every chunk and the token path measures a resolution it never learned. See
        :mod:`~.chunk_support`.
    chunk_aggregate : str
        ``lse``, ``mean`` or ``max`` aggregation of a chunk's token scores when ``chunk_size > 0``.
        Must match what the router trained with; the checkpoint records it. ``lse`` is the HSA arm's,
        ``mean`` the exact-K arm's.
    chunk_score_scale : float
        Token-score multiplier applied *inside* the aggregation. Only matters for ``lse`` (which is
        not scale-equivariant, so it acts as a temperature); must match the trainer's ``score_scale``.
    precision : str
        ``tl.dot`` precision. ``"tf32"`` by default here, unlike the kernel's own ``"ieee"``
        default, because at inference the operands are the model's own bf16 q/k/v: every bf16
        value is exactly representable in tf32 (10 mantissa bits against bf16's 8), so the QK
        dot is *bit-identical* either way, and the PV dot's only genuine fp32 operand -- the
        softmax weights -- rounds to ~2e-4 relative, some 30x below the bf16 epsilon the output
        is stored at. Measured on an H20 at ``L=8192, topk=2048``: 67.0 s per prefill under
        ``"ieee"`` against 9.4 s under ``"tf32"``, for identical error against the fp32
        reference (7.52e-3 both, i.e. bf16 output rounding alone).

        The reason the gap is so large is that ``"ieee"`` fp32 does not use tensor cores at all,
        so ``BLOCK_G`` -- padded up to 16 on Triton 3.3, which requires ``M >= 16`` -- becomes
        real work rather than lanes the hardware was going to occupy regardless: M scales the
        kernel ~linearly under ``"ieee"`` (1.89x for 2x M, measured) and barely at all under
        ``"tf32"`` (1.09x). Pass ``"ieee"`` to reproduce the fp32 reference exactly, which is
        what the tests do; it is the wrong default for a bf16 model at length.

    Usage
    -----
    >>> with SparseAttentionContext(model, press, topk=512, force_sink=4, force_local=64):
    ...     model.model(input_ids=context_ids, past_key_values=cache)   # sparse prefill
    ...     out = model(input_ids=question_ids, past_key_values=cache)   # sparse decode
    """

    def __init__(
        self,
        model: nn.Module,
        press: GQAIndexerPress,
        *,
        topk: int,
        force_sink: int = 0,
        force_local: int = 0,
        block_k: int = 64,
        causal: bool = True,
        precision: str = "tf32",
        query_independent: bool | None = None,
        chunk_size: int = 0,
        chunk_aggregate: str = "mean",
        chunk_score_scale: float = 1.0,
    ):
        if topk <= 0:
            raise ValueError(f"topk must be positive, got {topk}")
        if force_sink < 0 or force_local < 0:
            raise ValueError("force_sink and force_local must be non-negative")
        if force_sink + force_local > topk:
            raise ValueError(
                f"force_sink + force_local = {force_sink + force_local} exceeds topk = {topk}; "
                "the forced keys would be truncated. Lower them or raise topk."
            )
        self.model = model
        self.press = press
        self.topk = int(topk)
        self.force_sink = int(force_sink)
        self.force_local = int(force_local)
        self.block_k = int(block_k)
        self.causal = bool(causal)
        if precision not in ("ieee", "tf32"):
            raise ValueError(f"precision must be 'ieee' or 'tf32', got {precision!r}")
        self.precision = precision
        if chunk_size < 0:
            raise ValueError(f"chunk_size must be non-negative, got {chunk_size}")
        self.chunk_size = int(chunk_size)
        self.chunk_aggregate = chunk_aggregate
        self.chunk_score_scale = float(chunk_score_scale)
        if self.chunk_size and self.chunk_size > topk - force_sink - force_local:
            raise ValueError(
                f"chunk_size {self.chunk_size} exceeds the selectable budget "
                f"topk - force_sink - force_local = {topk - force_sink - force_local}: not even one "
                "chunk would fit, so every row would attend to the forced slots alone. Raise topk."
            )
        # Resolved lazily in __enter__: it depends on the scorer the press holds, which
        # post_init_from_model attaches there. None means "use the fast path when the scorer
        # declares itself query-independent and this torch has flex_attention".
        self._query_independent = query_independent
        self._use_qi = False

        # Per-layer state, all keyed by layer_idx and reset on entry.
        self._hidden_states: dict[int, torch.Tensor] = {}
        self._kwargs: dict[int, dict] = {}
        self._k_idx: dict[int, torch.Tensor] = {}  # the indexer key-cache, (B, Sk, D) post-RoPE

        self._handles: list = []
        self._configs: list = []
        self._previous_impls: list = []
        self._registry_restore = None  # (had_previous, previous_fn)

    # ------------------------------------------------------------------
    # Hooks (mirrors E2EIndexerTrainer._capture_hook)
    # ------------------------------------------------------------------
    def _capture_hook(self, module: nn.Module, args, kwargs: dict):
        """Stash this layer's hidden_states + kwargs before its attention runs."""
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None:
            return None
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        self._hidden_states[int(layer_idx)] = hidden_states
        self._kwargs[int(layer_idx)] = kwargs
        return None

    # ------------------------------------------------------------------
    # The sparse attention itself
    # ------------------------------------------------------------------
    def _attend(self, module, query, key, value, scaling):
        """
        Select each query's top-k keys with the indexer, then attend only to them.

        ``query``/``key``/``value`` arrive post-RoPE and post-cache-update from the layer, so
        ``key.shape[2]`` is the full ``k_len`` and the KV tensors carry ``n_kv_heads`` heads --
        exactly what :func:`sparse_gqa_attention` expects.
        """
        layer_idx = int(module.layer_idx)
        hidden_states = self._hidden_states.get(layer_idx)
        if hidden_states is None:
            raise RuntimeError(
                f"layer {layer_idx} reached sparse attention without its hidden_states captured. "
                "The pre-hook must be installed on the same modules as the attention swap."
            )
        kwargs = self._kwargs.get(layer_idx, {})

        indexer = self.press.get_indexer(module)
        cos, sin = self.press.get_rope_tables(indexer, kwargs)
        # Same q/k the training path builds (E2EIndexerTrainer.indexer_qk), so selection at
        # inference matches selection at train time.
        q_idx = indexer.project_q(hidden_states, cos, sin)  # (B, h, Sq, D)
        k_idx_new = indexer.project_k(hidden_states, cos, sin)  # (B, Sq, D), post-RoPE

        # Indexer key-cache: initialize on prefill, append on each decode step. Post-RoPE keys are
        # appended in position order, so the cache mirrors the model's own KV cache.
        previous = self._k_idx.get(layer_idx)
        k_idx = k_idx_new if previous is None else torch.cat([previous, k_idx_new], dim=1)
        self._k_idx[layer_idx] = k_idx

        k_len = key.shape[2]
        if k_idx.shape[1] != k_len:
            raise RuntimeError(
                f"layer {layer_idx}: indexer key-cache has {k_idx.shape[1]} keys but the model "
                f"cache holds {k_len}. The two must stay in lockstep -- use one "
                "SparseAttentionContext per generation (its cache resets on entry)."
            )

        # Query-independent scorers take the flex_attention path: the score is a fixed per-key
        # vector, so each key is selected by one contiguous interval of query rows and the whole
        # support is a per-key deadline instead of a (B, h, Sq, topk) index tensor. Same selection,
        # block-sparse contiguous reads instead of gathers -- measured 2.31x at L=8030 and 4.25x at
        # L=4096 for the select+attend pair. Only worth it when there is a query axis to amortize
        # the block-mask build over, so decode (Sq == 1) stays on the gather path.
        if self._use_qi and self.chunk_size:
            raise RuntimeError(
                "chunk_size is set but the query-independent flex path selects tokens by per-key "
                "deadline, which cannot express whole-chunk selection. Pass query_independent=False "
                "to force the gather path."
            )
        if self._use_qi and q_idx.shape[2] > 1:
            out = qi_sparse_attention(
                query,
                key,
                value,
                # The per-key score IS one row of the score matrix; take row 0 rather than calling
                # score_keys again, so this path cannot drift from what the gather path would score.
                torch.einsum("bhqd,bkd->bhk", q_idx[:, :, :1], k_idx),
                self.topk,
                force_sink=self.force_sink,
                force_local=self.force_local,
                scaling=scaling,
            )  # (B, H, Sq, Dv)
            return out.transpose(1, 2).contiguous()

        # query_offset defaults to k_len - Sq in both calls (bottom-right), correct for prefill
        # (Sq == k_len) and decode (Sq == 1) alike -- so it is never passed explicitly.
        if self.chunk_size:
            # Whole-chunk selection, for a router trained on chunk-mean scores. Same output
            # convention as the token path, so sparse_gqa_attention is unchanged.
            support, _ = chunk_topk_support(
                q_idx,
                k_idx,
                self.topk,
                chunk_size=self.chunk_size,
                chunk_aggregate=self.chunk_aggregate,
                score_scale=self.chunk_score_scale,
                force_sink=self.force_sink,
                force_local=self.force_local,
            )
        else:
            support, _ = streaming_topk_support(
                q_idx,
                k_idx,
                self.topk,
                mask=None,
                force_sink=self.force_sink,
                force_local=self.force_local,
            )  # (B, h, Sq, topk) int32, ascending, -1 empty
        out, _ = sparse_gqa_attention(
            query,
            key,
            value,
            support,
            scaling=scaling,
            causal=self.causal,
            block_k=self.block_k,
            precision=self.precision,
        )  # (B, H, Sq, Dv)
        # The attention interface contract is (B, Sq, H, D); our op returns (B, H, Sq, D).
        return out.transpose(1, 2).contiguous()

    # ------------------------------------------------------------------
    # Context management (mirrors E2EIndexerTrainer.hooks)
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._hidden_states.clear()
        self._kwargs.clear()
        self._k_idx.clear()

    def __enter__(self) -> "SparseAttentionContext":
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self.press.post_init_from_model(self.model)
        self.reset()

        # Decide the selection path now that the indexers exist. Keyed off a declared capability
        # rather than isinstance, so a third scorer only has to set the attribute.
        layers = get_language_model(self.model).layers
        scorer_is_qi = bool(
            getattr(self.press.get_indexer(layers[0].self_attn), "is_query_independent", False)
        )
        if self._query_independent is None:
            self._use_qi = scorer_is_qi and HAS_FLEX
            if scorer_is_qi and not HAS_FLEX:
                logger.warning(
                    "scorer is query-independent but this torch has no flex_attention; falling "
                    "back to the gather path (correct, just slower)."
                )
        else:
            self._use_qi = bool(self._query_independent)
            if self._use_qi and not scorer_is_qi:
                # The fast path reads one row of the score matrix and applies it to every query. For
                # a pairwise scorer that is a different (wrong) support, not a slower one.
                raise ValueError(
                    "query_independent=True but the indexer's score depends on the query "
                    f"({type(self.press.get_indexer(layers[0].self_attn)).__name__}). The flex path "
                    "would attend over the wrong keys."
                )
            if self._use_qi and not HAS_FLEX:
                raise RuntimeError("query_independent=True but this torch has no flex_attention")
        logger.info("sparse selection path: %s", "flex (query-independent)" if self._use_qi else "gather")

        def sparse_attention_impl(
            module, query, key, value, attention_mask, scaling=None, dropout=0.0, **_
        ):
            return self._attend(module, query, key, value, scaling), None

        self._configs = [self.model.config]
        text_config = getattr(self.model.config, "text_config", None)
        if text_config is not None:
            self._configs.append(text_config)
        self._previous_impls = [cfg._attn_implementation for cfg in self._configs]

        # Register through the mapping, and remember what to restore. register() writes to the
        # class-level _global_mapping while pop() only touches the instance mapping, so the naive
        # removal would leak the entry -- the same care capture_teacher_lse documents.
        global_mapping = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
        self._registry_restore = (IMPL_NAME in global_mapping, global_mapping.get(IMPL_NAME))
        ALL_ATTENTION_FUNCTIONS.register(IMPL_NAME, sparse_attention_impl)

        self._handles = []
        for layer in get_language_model(self.model).layers:
            self._handles.append(
                layer.self_attn.register_forward_pre_hook(self._capture_hook, with_kwargs=True)
            )
        for cfg in self._configs:
            cfg._attn_implementation = IMPL_NAME
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        for handle in self._handles:
            handle.remove()
        self._handles = []
        for cfg, previous in zip(self._configs, self._previous_impls):
            cfg._attn_implementation = previous
        if self._registry_restore is not None:
            global_mapping = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
            had_previous, previous_fn = self._registry_restore
            if had_previous:
                global_mapping[IMPL_NAME] = previous_fn
            else:
                global_mapping.pop(IMPL_NAME, None)
            self._registry_restore = None
        self.reset()
