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

from kvpress.presses.gqa_indexer.press import GQAIndexerPress, get_language_model
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

        # query_offset defaults to k_len - Sq in both calls (bottom-right), correct for prefill
        # (Sq == k_len) and decode (Sq == 1) alike -- so it is never passed explicitly.
        support, _ = streaming_topk_support(
            q_idx,
            k_idx,
            self.topk,
            mask=None,
            force_sink=self.force_sink,
            force_local=self.force_local,
        )  # (B, h, Sq, topk) int64, ascending, -1 empty
        out, _ = sparse_gqa_attention(
            query,
            key,
            value,
            support.to(torch.int32),
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
