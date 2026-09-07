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
module stashes the ``hidden_states`` used by hidden-state scorers, while the attention interface
provides the actual projected values used by value scorers. A temporary entry in
``ALL_ATTENTION_FUNCTIONS`` that ``config._attn_implementation`` is pointed at replaces the
attention itself. Both are removed on exit.

The one thing training does not need and this does: an **indexer key-cache**. During training a
forward sees the whole sequence at once, so ``project_k`` produces every key. At inference each
step contributes only the *new* hidden states or projected values, so we accumulate the per-layer
indexer keys ourselves -- initialize on prefill, append on each decode step -- and assert the cache
length stays in lockstep with the model's own ``key.shape[2]``. The cache is tiny (one projected
key, or one score per KV head, per token and layer) and lives only for the duration of the ``with``
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

from kvpress.presses.gqa_indexer.press import GQAIndexerPress, get_language_model
from kvpress.presses.gqa_indexer.qi_flex_attention import (
    FLEX_BLOCK,
    HAS_FLEX,
    deadlines,
    qi_sparse_attention,
)
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
    memory : bool
        Fold the linear memory over the evicted keys into the same softmax as the retained ones
        (:mod:`~kvpress.presses.gqa_indexer.memory`). Requires a press built with ``memory=True``,
        its weights loaded, and ``flex_attention`` -- the fusion reads the retained branch's ``lse``,
        which the gather kernel does not return, so this forces the flex path for decode rows too.

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
        memory: bool = False,
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
        self.memory = bool(memory)
        if self.memory and not HAS_FLEX:
            raise RuntimeError(
                "memory=True needs flex_attention: the fusion reads the retained branch's lse, "
                "which the gather kernel does not return."
            )
        self.topk = int(topk)
        self.force_sink = int(force_sink)
        self.force_local = int(force_local)
        self.block_k = int(block_k)
        self.causal = bool(causal)
        if precision not in ("ieee", "tf32"):
            raise ValueError(f"precision must be 'ieee' or 'tf32', got {precision!r}")
        self.precision = precision
        # Resolved lazily in __enter__: it depends on the scorer the press holds, which
        # post_init_from_model attaches there. None means "use the fast path when the scorer
        # declares itself query-independent and this torch has flex_attention".
        self._query_independent = query_independent
        self._use_qi = False
        self._decay_active = False

        # Per-layer state, all keyed by layer_idx and reset on entry.
        self._hidden_states: dict[int, torch.Tensor] = {}
        self._kwargs: dict[int, dict] = {}
        self._k_idx: dict[int, torch.Tensor] = {}  # the indexer key-cache, (B, Sk, Di)

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
        previous = self._k_idx.get(layer_idx)
        previous_len = 0 if previous is None else previous.shape[1]
        # Absolute position of the first NEW token: 0 at prefill, the cache length at each decode
        # step. A decay-carrying scorer folds -log_beta_j * j / ref into its key, so an offset of 0
        # at decode would score every generated token as if it sat at position 0 -- its whole
        # history would look infinitely old to it.
        q_kwargs = {}
        if self._decay_active:
            q_kwargs["query_offset"] = previous_len
            q_kwargs["n_kv_heads"] = key.shape[1]
        # Same inputs the training path uses: hidden-state scorers ignore value_states, while DMA
        # consumes the actual post-projection values for only this step's new tokens.
        q_idx = indexer.project_q(hidden_states, cos, sin, **q_kwargs)  # (B, h, Sq, Di)
        value_states_new = value[:, :, previous_len:, :]
        if value_states_new.shape[2] != hidden_states.shape[1]:
            raise RuntimeError(
                f"layer {layer_idx}: expected {hidden_states.shape[1]} newly appended values but "
                f"found {value_states_new.shape[2]}. SparseAttentionContext requires an "
                "append-only KV cache."
            )
        k_idx_new = indexer.project_k(
            hidden_states,
            cos,
            sin,
            value_states=value_states_new,
            **({"key_offset": previous_len} if self._decay_active else {}),
        )  # (B, Sq, Di)

        # Indexer key-cache: initialize on prefill, append on each decode step. Entries stay in
        # position order, so the cache mirrors the model's own KV cache.
        k_idx = k_idx_new if previous is None else torch.cat([previous, k_idx_new], dim=1)
        self._k_idx[layer_idx] = k_idx

        k_len = key.shape[2]
        if k_idx.shape[1] != k_len:
            raise RuntimeError(
                f"layer {layer_idx}: indexer key-cache has {k_idx.shape[1]} keys but the model "
                f"cache holds {k_len}. The two must stay in lockstep -- use one "
                "SparseAttentionContext per generation (its cache resets on entry)."
            )

        # A cache shorter than the forced slots has nothing to select: force_sink + force_local
        # already covers every key, so the support is the whole sequence and sparse attention IS
        # dense attention. Short-circuiting is a correctness requirement, not an optimization --
        # streaming_topk_support clamps topk down to k_len and *then* rejects
        # force_sink + force_local > topk, so this case raises instead of trivially keeping
        # everything. Reasoning benchmarks hit it constantly rather than as a corner case:
        # math500's context is a single space (4 tokens once the chat template is applied), so 20
        # of the 50 rows sampled at fraction 0.1 begin decoding at k_len < 68 = 4 + 64 and the run
        # dies on its first question instead of producing a number.
        #
        # Deliberately narrow: k_len <= topk is also a no-op selection, but the gather path handles
        # it correctly (topk is clamped to k_len), so it stays on the kernel and the precision
        # plumbing that test_precision_reaches_the_kernel pins remains observable.
        if k_len < self.force_sink + self.force_local:
            return self._attend_dense(query, key, value, scaling)

        if self.memory:
            return self._attend_with_memory(
                module, query, key, value, scaling, q_idx, k_idx, k_len
            )

        # Query-independent scorers take the flex_attention path: the score is a fixed per-key
        # vector, so each key is selected by one contiguous interval of query rows and the whole
        # support is a per-key deadline instead of a (B, h, Sq, topk) index tensor. Same selection,
        # block-sparse contiguous reads instead of gathers -- measured 2.31x at L=8030 and 4.25x at
        # L=4096 for the select+attend pair. Only worth it when there is a query axis to amortize
        # the block-mask build over, so decode (Sq == 1) stays on the gather path.
        if self._use_qi and q_idx.shape[2] > 1:
            # The deadline path needs ONE frozen per-key ranking. Without decay any query row
            # gives it, since every row is identical. With decay the rows genuinely differ, so a
            # row has to be chosen and the selection becomes an approximation of the exact
            # per-row top-k the gather path below computes.
            #
            # The MIDDLE row, not row 0. Row 0 sees every key at age ~0, which is the one position
            # where the lifetime term contributes nothing -- it throws the feature away and ranks
            # purely by magnitude. Measured deadline-mask error against the exact per-row top-k
            # (log_beta ~ U(-1,0) nats/ref, Sk=4096, take=512): row 0 4.52%, middle 1.64%, last
            # 2.59%. The middle row minimizes the worst-case age error over the row range.
            ref_row = q_idx.shape[2] // 2 if self._decay_active else 0
            out = qi_sparse_attention(
                query,
                key,
                value,
                # The per-key score IS one row of the score matrix; take it from q_idx/k_idx
                # rather than calling score_keys again, so this path cannot drift from what the
                # gather path would score.
                torch.einsum(
                    "bhqd,bkd->bhk", q_idx[:, :, ref_row : ref_row + 1], k_idx
                ),
                self.topk,
                force_sink=self.force_sink,
                force_local=self.force_local,
                scaling=scaling,
            )  # (B, H, Sq, Dv)
            return out.transpose(1, 2).contiguous()

        # query_offset defaults to k_len - Sq in both calls (bottom-right), correct for prefill
        # (Sq == k_len) and decode (Sq == 1) alike -- so it is never passed explicitly.
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

    def _attend_dense(self, query, key, value, scaling) -> torch.Tensor:
        """
        Plain causal attention, for a cache that already fits inside the support budget.

        This is the *exact* answer, not an approximation: when ``k_len <= topk`` every key is
        selected, so the sparse softmax and the dense one run over the same set. Routed through
        SDPA rather than the gather kernel because the kernel's ``force_sink + force_local <= topk``
        precondition is violated exactly when the sequence is this short (see the caller).

        The mask is built explicitly and aligned BOTTOM-RIGHT (query row ``i`` sees key ``j`` iff
        ``j <= k_len - q_len + i``), which is what ``is_causal=True`` does *not* do: SDPA aligns its
        built-in mask top-left, so any call with ``Sq != Sk`` gets the wrong triangle. That case is
        the common one here, not a corner -- the pipeline prefills the context and the question in
        two separate forwards, so answering a 28-token question over a 4-token context arrives as
        ``Sq=28, Sk=32``. Under top-left alignment its first row would attend to key 0 alone, and the
        model degenerates into repeating text instead of answering. ``streaming_topk_support``
        defaults to the same bottom-right convention, so this matches the path it replaces.
        """
        group = query.shape[1] // key.shape[1]
        q_len, k_len = query.shape[2], key.shape[2]
        key_idx = torch.arange(k_len, device=query.device)
        q_pos = torch.arange(q_len, device=query.device).unsqueeze(-1) + (k_len - q_len)
        attn_mask = key_idx <= q_pos  # (Sq, Sk), True = visible
        out = torch.nn.functional.scaled_dot_product_attention(
            query,
            key.repeat_interleave(group, dim=1),
            value.repeat_interleave(group, dim=1),
            attn_mask=attn_mask,
            scale=scaling,
        )  # (B, H, Sq, Dv)
        return out.transpose(1, 2).contiguous()

    def _attend_with_memory(
        self, module, query, key, value, scaling, q_idx, k_idx, k_len: int
    ) -> torch.Tensor:
        """
        Sparse attention plus the linear memory over the keys this row would have evicted.

        Prefill and decode take different state constructions, and the difference is causality
        rather than performance:

        * **Prefill** (``Sq > 1``) builds a per-query-block state with
          :func:`~.memory_schedule.block_memory_states`, the same function training uses. A single
          state shared by every row would let row ``t`` read keys at positions ``> t``. That is not a
          mild approximation -- measured at ``L=8192``, row 0 saw a state built from 6208 future
          keys, and it took RULER 8K from 73.71 to **4.00** while the (causal) training curve looked
          healthy throughout.
        * **Decode** (``Sq == 1``) uses one state over the whole evicted set, which is causal by
          construction: the single query sits after every key in the cache.

        Either way the state is rebuilt from the current cache each call rather than carried across
        steps. That is deliberate at eval scale: ``psi`` is ``O(L)``, and maintaining ``(H, z, W)``
        incrementally means tracking exactly which keys crossed their deadline since the last step --
        a second bookkeeping path that has to agree with ``deadlines()`` exactly or the two branches
        double-count. The incremental form is a decode optimization, not a different model.
        """
        from kvpress.presses.gqa_indexer.memory import fuse_memory
        from kvpress.presses.gqa_indexer.qi_flex_attention import _flex, qi_block_mask

        memory = self.press.get_memory(module)
        bsz, n_q_heads, q_len, _ = query.shape
        n_kv_heads = key.shape[1]
        group = n_q_heads // n_kv_heads
        if bsz != 1:
            raise NotImplementedError(
                f"memory=True supports batch 1, got {bsz}: qi_block_mask is built with B=None."
            )

        # One row of the score matrix IS the per-key score for a query-independent scorer. Taken
        # from q_idx/k_idx rather than by calling score_keys again, so this path cannot drift from
        # what the selection uses -- the same reason the non-memory branch does it this way.
        scores = torch.einsum("bhqd,bkd->bhk", q_idx[:, :, :1], k_idx)[0].float()
        dl = deadlines(
            scores, self.topk, force_sink=self.force_sink, force_local=self.force_local
        )
        block_mask = qi_block_mask(
            dl,
            q_len=q_len,
            k_len=k_len,
            n_q_heads=n_q_heads,
            force_sink=self.force_sink,
            force_local=self.force_local,
            device=query.device,
        )
        o_s, lse_s = _flex()(
            query,
            key.repeat_interleave(group, dim=1),
            value.repeat_interleave(group, dim=1),
            block_mask=block_mask,
            scale=scaling,
            return_lse=True,
        )

        q_kv = query.view(bsz, n_kv_heads, group, q_len, query.shape[-1]).mean(2)
        offset = k_len - q_len
        if q_len > 1:
            # PREFILL: a per-query-block state, exactly as training builds it. A single state shared
            # by every row is **not** valid here -- it would let row t read keys at positions > t.
            # Measured: at L=8192 row 0 saw a state built from 6208 future keys, and RULER 8K went
            # from 73.71 to 4.00 while the training curve (which is causal) looked healthy. The
            # streaming construction is the same code training uses, so the two cannot drift.
            from kvpress.presses.gqa_indexer.memory_schedule import block_memory_states
            from kvpress.presses.gqa_indexer.memory_trainer import MemoryTrainer

            H, z, W, counts = block_memory_states(
                memory,
                key,
                value,
                dl,
                q_len=q_len,
                block=FLEX_BLOCK,
                n_local=self.force_local,
                scores=scores,
            )
            n, d = MemoryTrainer._read_per_block(
                memory, q_kv, H, z, W, counts, block=FLEX_BLOCK
            )
        else:
            # DECODE: one row, and it sits after every key in the cache, so a single state over the
            # whole evicted set is causal by construction -- there is no future to leak.
            enter = dl.to(torch.int64) + 1
            weights = memory.ingest_weights(enter, k_len, scores=scores)
            state = memory.ingest(key, value, weights)
            counts = (enter <= k_len - 1).sum(-1)  # (Hkv,)
            n_evicted = counts.view(1, -1, 1).expand(bsz, -1, q_len).float()
            n, d = memory.read(q_kv, state, n_evicted)

        out = fuse_memory(o_s, lse_s, n, d, group=group)
        return out.transpose(1, 2).contiguous()

    # ------------------------------------------------------------------
    # Context management (mirrors E2EIndexerTrainer.hooks)
    # ------------------------------------------------------------------
    def _set_prefix_cache(self, enabled: bool) -> None:
        for layer in get_language_model(self.model).layers:
            indexer = self.press.get_indexer(layer.self_attn)
            setter = getattr(indexer, "enable_cache" if enabled else "disable_cache", None)
            if setter is not None:
                setter()

    def reset(self) -> None:
        self._hidden_states.clear()
        self._kwargs.clear()
        self._k_idx.clear()
        self._set_prefix_cache(True)

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
        # A decay-carrying scalar scorer is still query-INDEPENDENT in the sense that matters
        # (the query side carries only position, no content), so the flex path stays available.
        # But its score is no longer constant along the query axis, which the deadline
        # construction assumes, so the selection becomes approximate and the offsets have to be
        # threaded. Both are gated on this flag.
        self._decay_active = bool(
            getattr(self.press.get_indexer(layers[0].self_attn), "decay", False)
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
        self._hidden_states.clear()
        self._kwargs.clear()
        self._k_idx.clear()
        self._set_prefix_cache(False)
