# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Generator, List

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, QuantizedCache
from transformers import Gemma3ForConditionalGeneration

from kvpress.presses.base_press import SUPPORTED_MODELS, BasePress
from kvpress.presses.indexer_score_query_press import QueryIndexerScorePress
from kvpress.utils import extract_keys_and_values, get_prerope_query_states

logger = logging.getLogger(__name__)


@dataclass
class QueryIndexer_KVzipScorePress(BasePress):
    """
    KVzip-style reconstruction-driven scoring, but with QueryIndexer proxy scores
    (repeat prompt + chunked context reconstruction) and **real** KV cache shortening
    via gather.
    """

    compression_ratio: float = 0.0
    n_sink: int = 4
    # KVzip reconstruction chunking
    chunk_size: int = 2048
    prev_postfix_size: int = 8
    # Optional block-aligned selection for multi-token entities
    block_size: int | None = None
    # How to aggregate query->key scores from repeat-time queries into per-key importance.
    #
    # Notation: qk_scores: (B, q_len, K_prefill) are QueryIndexer logits.
    # - "amax_prob" (default, current behavior): p(q,k)=softmax_k(qk_scores); score(k)=max_q p(q,k)
    # - "amax_logit_softmax": s(k)=max_q qk_scores; score(k)=softmax_k(s(k))   (matches fused-loss max idea)
    # - "mean_prob": score(k)=mean_q softmax_k(qk_scores)
    # - "mean_logit_softmax": s(k)=mean_q qk_scores; score(k)=softmax_k(s(k))
    score_reduce: str = "amax_prob"
    # Optionally restrict aggregation to the last N repeat queries (helps reduce noise/compute).
    last_n_query: int | None = None

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "compression_ratio must be in [0, 1)"
        self._reset_internal_parameters()

    def _reset_internal_parameters(self):
        self.context_length = 0
        self.prefix_length = 0

        self._suffix_ids: torch.Tensor | None = None
        self._context_ids: torch.Tensor | None = None
        self._cache = None

        self.start_idx = 0
        self.end_idx = 0

        self._prefill_hidden_states_cpu: dict[int, torch.Tensor] = {}
        self._capture_prefill_hs = True

        self.score_val: torch.Tensor | None = None  # (n_layer, ctx_len) on model.device

    def post_init_from_model(self, model: PreTrainedModel):
        # Ensure QueryIndexer modules exist on every layer.
        QueryIndexerScorePress(compression_ratio=0.0).post_init_from_model(model)

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        if not isinstance(model, SUPPORTED_MODELS):
            logger.warning(f"Model {type(model)} not tested, supported models: {SUPPORTED_MODELS}")

        if isinstance(model, Gemma3ForConditionalGeneration):
            logger.warning_once("Compression in Gemma3 is only applied to layer without sliding window attention")

        self.post_init_from_model(model)
        tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)

        # --- KVzip chat-template suffix extraction (reuse KVzipPress logic) ---
        if tokenizer.chat_template is None:
            prefix_text = ""
            suffix_text = "\n"
        else:
            dummy_context = "dummy context"
            separator = "\n" + "#" * len(dummy_context)
            temp_context = tokenizer.apply_chat_template(
                [{"role": "user", "content": dummy_context + separator}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            context, suffix_text = temp_context.split(separator)
            prefix_text = context.split(dummy_context)[0]

        self.prefix_length = tokenizer.encode(prefix_text, return_tensors="pt", add_special_tokens=False).shape[-1]
        self._suffix_ids = tokenizer.encode(suffix_text, return_tensors="pt", add_special_tokens=False)

        # Wrap model forward to capture the prefill `input_ids` and cache pointer.
        original_forward = model.model.forward

        def wrapped_forward(model_self, *args, **kwargs):
            self._context_ids = kwargs.get("input_ids", None)
            self._cache = kwargs.get("past_key_values", None)
            return original_forward(*args, **kwargs)

        model.model.forward = MethodType(wrapped_forward, model.model)

        hooks: list = []
        try:
            # Phase 1: during the user's prefill forward, capture per-layer hidden_states.
            language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
            for layer in language_model.layers:
                if isinstance(model, Gemma3ForConditionalGeneration) and layer.self_attn.is_sliding:
                    continue
                hooks.append(layer.self_attn.register_forward_hook(self._prefill_capture_hook, with_kwargs=True))

            yield

            # Phase 2/3: reconstruction scoring and real gather-compression.
            model.model.forward = original_forward
            for h in hooks:
                h.remove()
            hooks = []

            if self.compression_ratio > 0 and self._context_ids is not None and self._cache is not None:
                # Build indexer key caches once from captured prefill hidden states.
                self._build_indexer_key_caches(model)

                # Register scoring hooks (also truncates KV to avoid cache growth during repeats).
                for layer in language_model.layers:
                    if isinstance(model, Gemma3ForConditionalGeneration) and layer.self_attn.is_sliding:
                        continue
                    hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))

                self._perform_reconstruction_scoring(model, tokenizer)
                self._gather_compress_all_layers(model)
        finally:
            # Best-effort cleanup.
            try:
                model.model.forward = original_forward
            except Exception:
                pass
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass
            self._reset_internal_parameters()

    def _prefill_capture_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        if not self._capture_prefill_hs:
            return output

        hidden_states = kwargs.get("hidden_states", None)
        cache_position = kwargs.get("cache_position", None)
        if hidden_states is None or cache_position is None:
            return output

        q_len = int(hidden_states.shape[1])
        # Only capture prefill (not decoding): same criterion as BasePress.forward_hook.
        if cache_position[-1] > q_len:
            return output
        if q_len <= 1:
            return output

        layer_idx = int(getattr(module, "layer_idx", -1))
        if layer_idx < 0 or layer_idx in self._prefill_hidden_states_cpu:
            return output

        # Store on CPU to minimize GPU footprint.
        self._prefill_hidden_states_cpu[layer_idx] = hidden_states.detach().to("cpu")
        return output

    def _chunk_fn(self, ctx_ids: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
        ctx_len = ctx_ids.shape[1]
        if ctx_len > chunk_size:
            chunk_num = (ctx_len - 1) // chunk_size + 1
            out = []
            for i in range(chunk_num):
                s = i * chunk_size
                e = (i + 1) * chunk_size
                a_ids = ctx_ids[:, s:e]
                if a_ids.shape[1] > 0:
                    out.append(a_ids)
            return out
        return [ctx_ids]

    def prepare(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> List[tuple[torch.Tensor, torch.Tensor]]:
        assert self._context_ids is not None
        assert self._suffix_ids is not None

        ctx_ids = self._context_ids[:, self.prefix_length :].to("cpu")
        chunked_input_ids = self._chunk_fn(ctx_ids, self.chunk_size)

        chunked_context_pairs = []
        for i, a_ids in enumerate(chunked_input_ids):
            if i == 0:
                prompt = "\n\nRepeat the previous context exactly."
                q_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
            else:
                prompt = "\n\nRepeat the part of the previous context exactly, starting with"
                q_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
                postfix_prev = chunked_input_ids[i - 1][:, -self.prev_postfix_size :]
                q_ids = torch.cat([q_ids, postfix_prev], dim=1)

            chunked_context_pairs.append((a_ids, torch.cat([q_ids, self._suffix_ids, a_ids], dim=1)))

        return chunked_context_pairs

    def _build_indexer_key_caches(self, model: PreTrainedModel):
        """
        Populate each layer's `module.indexer.k_cache` from captured prefill hidden states.
        This makes repeat-time scoring cheap and avoids storing full prefill hidden states on GPU.
        """
        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        device = model.device
        dtype = model.dtype

        for layer in language_model.layers:
            module = layer.self_attn
            layer_idx = int(getattr(module, "layer_idx", -1))
            if layer_idx < 0:
                continue
            hs_cpu = self._prefill_hidden_states_cpu.get(layer_idx)
            if hs_cpu is None:
                continue
            indexer = getattr(module, "indexer", None)
            if indexer is None:
                continue

            with torch.no_grad():
                indexer.reset_cache()
                hs = hs_cpu.to(device=device, dtype=dtype)
                _ = indexer.forward_cache(hs)

    def _perform_reconstruction_scoring(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        assert self._context_ids is not None
        assert self._cache is not None

        self.context_length = int(self._context_ids.shape[1])
        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        n_layer = int(len(language_model.layers))

        # Per-layer per-token importance; reduced later across layers.
        self.score_val = torch.zeros((n_layer, self.context_length), device=model.device, dtype=torch.float32)
        self.score_val[:, : min(self.n_sink, self.context_length)] = 1.0

        chunked_context_pairs = self.prepare(model, tokenizer)
        self.start_idx = int(self.prefix_length)

        for prefill_ids, repeat_ids in chunked_context_pairs:
            self.end_idx = int(self.start_idx + int(prefill_ids.shape[1]))
            model(
                input_ids=repeat_ids.to(model.device),
                past_key_values=self._cache,
                num_logits_to_keep=1,
            )
            self.start_idx = int(self.end_idx)

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        During KVzip-style repeat forwards:
        - compute proxy scores from repeat queries -> prefill keys using QueryIndexer cache
        - update per-token importance using a configurable query aggregation (KVzip-style)
        - truncate KV cache back to prefill-only to avoid cache growth
        """
        if self.score_val is None:
            return output

        hidden_states = kwargs["hidden_states"]
        cache = kwargs.get("past_key_values", None) or kwargs.get("past_key_value", None)
        if cache is None:
            return output

        layer_idx = int(getattr(module, "layer_idx", -1))
        if layer_idx < 0:
            return output

        indexer = getattr(module, "indexer", None)
        if indexer is None:
            return output

        # Score repeat queries against prefill key cache.
        with torch.no_grad():
            query_states = kwargs.get("query_states")
            if query_states is None:
                query_states = get_prerope_query_states(module, hidden_states)
            # Use the cached prefill keys inside the QueryIndexer. We intentionally pass freqs=None to
            # avoid RoPE shape/position mismatches between repeat tokens and cached prefill keys.
            qk_scores = indexer.get_cache_score(hidden_states, query_states, freqs_cis=None)  # (B, q_len, K_prefill)
            if self.last_n_query is not None:
                n = max(1, int(self.last_n_query))
                qk_scores = qk_scores[:, -n:, :]

            s = int(max(0, self.start_idx))
            e = int(min(self.end_idx, self.context_length))
            if e > s:
                mode = (self.score_reduce or "amax_prob").lower()
                if mode in ("amax_prob", "max_prob"):
                    # p(q,k)=softmax_k(logits); score(k)=max_q p(q,k)
                    probs = F.softmax(qk_scores.float(), dim=-1)  # (B, q_len, K_prefill)
                    imp = probs[:, :, s:e].amax(dim=1)  # (B, chunk_len)
                elif mode in ("amax_logit_softmax", "max_logit_softmax"):
                    # score(k)=softmax_k(max_q logits(q,k))
                    # NOTE: Tensor.amax returns a Tensor (no .values). Use amax() directly.
                    s_key = qk_scores.float().amax(dim=1)  # (B, K_prefill)
                    p_key = F.softmax(s_key, dim=-1)  # (B, K_prefill)
                    imp = p_key[:, s:e]  # (B, chunk_len)
                elif mode in ("mean_prob",):
                    probs = F.softmax(qk_scores.float(), dim=-1)
                    imp = probs[:, :, s:e].mean(dim=1)  # (B, chunk_len)
                elif mode in ("mean_logit_softmax",):
                    s_key = qk_scores.float().mean(dim=1)  # (B, K_prefill)
                    p_key = F.softmax(s_key, dim=-1)
                    imp = p_key[:, s:e]
                else:
                    raise ValueError(f"Unknown score_reduce={self.score_reduce!r}")

                imp = imp.squeeze(0)  # only support batch=1 for now
                self.score_val[layer_idx, s:e] = torch.maximum(self.score_val[layer_idx, s:e], imp)

        # Truncate cache to prefill-only tokens so KV doesn't grow during repeats.
        cache_layer = cache.layers[layer_idx]
        keys, values = extract_keys_and_values(cache, layer_idx)
        keys = keys[:, :, : self.context_length]
        values = values[:, :, : self.context_length]

        if isinstance(cache, QuantizedCache):
            cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
            cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
            cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)  # type: ignore[index]
            cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)  # type: ignore[index]
            cache_layer.cumulative_length = keys.shape[2]
        else:
            cache_layer.keys = keys
            cache_layer.values = values

        return output

    def _select_keep_positions(self, scores: torch.Tensor, ctx_len: int) -> torch.Tensor:
        """
        scores: (ctx_len,) higher is better. Returns (1, n_kept) indices, sorted in-order.
        """
        n_kept = int(math.ceil((1.0 - float(self.compression_ratio)) * ctx_len))
        n_kept = max(1, min(n_kept, ctx_len))

        sink = int(min(self.n_sink, ctx_len))
        if n_kept <= sink:
            keep = torch.arange(n_kept, device=scores.device, dtype=torch.long)
            return keep.view(1, -1)

        nonsink_len = ctx_len - sink
        n_keep_nonsink = n_kept - sink

        if not self.block_size or int(self.block_size) <= 1:
            nonsink_scores = scores[sink:]
            top = nonsink_scores.topk(n_keep_nonsink, dim=0).indices + sink
            keep = torch.cat([torch.arange(sink, device=scores.device, dtype=torch.long), top], dim=0).sort().values
            return keep.view(1, -1)

        bs = int(self.block_size)
        n_blocks = int(math.ceil(nonsink_len / float(bs)))
        n_blocks_keep = int(math.ceil(n_keep_nonsink / float(bs)))
        n_blocks_keep = min(n_blocks_keep, n_blocks)

        block_scores = []
        for bi in range(n_blocks):
            s = sink + bi * bs
            e = min(s + bs, ctx_len)
            block_scores.append(scores[s:e].mean())
        block_scores_t = torch.stack(block_scores, dim=0)  # (n_blocks,)
        kept_blocks = block_scores_t.topk(n_blocks_keep, dim=0).indices.tolist()

        pos = list(range(sink))
        for bi in sorted(kept_blocks):
            s = sink + bi * bs
            e = min(s + bs, ctx_len)
            pos.extend(range(s, e))
        keep = torch.tensor(pos, device=scores.device, dtype=torch.long)
        return keep.view(1, -1)

    def _gather_compress_all_layers(self, model: PreTrainedModel):
        assert self._cache is not None
        assert self.score_val is not None

        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        ctx_len = int(self.context_length)
        if ctx_len <= 0:
            return

        # Aggregate across layers (mean).
        agg = self.score_val.mean(dim=0)  # (ctx_len,)
        keep_pos = self._select_keep_positions(agg, ctx_len=ctx_len)  # (1, n_kept)
        n_kept = int(keep_pos.size(1))

        for layer in language_model.layers:
            module = layer.self_attn
            layer_idx = int(getattr(module, "layer_idx", -1))
            if layer_idx < 0:
                continue

            cache_layer = self._cache.layers[layer_idx]
            keys, values = extract_keys_and_values(self._cache, layer_idx)
            keys = keys[:, :, :ctx_len]
            values = values[:, :, :ctx_len]

            gather_idx = keep_pos[:, None, :, None].expand(-1, keys.size(1), -1, keys.size(-1))
            keys = keys.gather(2, gather_idx).contiguous()
            values = values.gather(2, gather_idx).contiguous()

            if isinstance(self._cache, QuantizedCache):
                cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
                cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
                cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)  # type: ignore[index]
                cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)  # type: ignore[index]
                cache_layer.cumulative_length = n_kept
            else:
                cache_layer.keys = keys
                cache_layer.values = values

