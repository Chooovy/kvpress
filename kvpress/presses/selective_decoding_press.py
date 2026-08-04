# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.cache_utils import QuantizedCache

from kvpress.presses.decode_press import DecodePress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.indexer_score_press_cache import CacheIndexerScorePress
from kvpress.utils import extract_keys_and_values

logger = logging.getLogger(__name__)



@dataclass
class SelectiveDecodingPress(DecodePress):
    base_press: ScorerPress
    compression_interval: int = 50
    target_size: int = 2048
    hidden_states_buffer_size: int = 256
    
    def __post_init__(self):
        self.hidden_states_buffer = defaultdict(list)
        self.layer_step_counts = defaultdict(int)
        self.prefill_cache_size = {}
        # Avoid spamming logs if a scorer cannot return full-length KV scores (e.g., IndexerScorePress).
        self._warned_score_len_mismatch = set()
    
    def post_init_from_model(self, model):
        self.base_press.post_init_from_model(model)

    def _maybe_get_indexer(self, module: nn.Module):
        if not isinstance(self.base_press, CacheIndexerScorePress):
            return None
        scorer_attr = getattr(self.base_press, "scorer_attr", None)
        if not scorer_attr:
            return None
        return getattr(module, scorer_attr, None)

    def _update_indexer_cache(self, module: nn.Module, hidden_states: torch.Tensor):
        indexer = self._maybe_get_indexer(module)
        if indexer is None:
            return
        with torch.no_grad():
            _ = indexer.forward_cache(hidden_states, freqs_cis=None, mask=None)

    def _compress_with_indexer_cache(self, module: nn.Module, keys: torch.Tensor, values: torch.Tensor, attentions, kwargs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        layer_idx = module.layer_idx
        prefill_size = self.prefill_cache_size.get(layer_idx, 0)
        total_len = keys.shape[2]
        decode_len = total_len - prefill_size

        if decode_len <= 0: return keys, values

        target_decode_size = max(0, self.target_size - prefill_size)
        n_kept = min(decode_len, target_decode_size)

        if n_kept >= decode_len: return keys, values
        if n_kept <= 0:
            if prefill_size > 0:
                return keys[:, :, :prefill_size, :], values[:, :, :prefill_size, :]
            # Keep at least one token to avoid empty cache tensors.
            return keys[:, :, -1:, :], values[:, :, -1:, :]

        indexer = self._maybe_get_indexer(module)
        if indexer is None or getattr(indexer, "k_cache", None) is None:
            return keys, values

        cache_len = indexer.k_cache.shape[1]
        if cache_len != total_len: 
            logger.warning(f"SelectiveDecodingPress(CacheIndexer): layer {layer_idx} indexer cache len ({cache_len}) != KV cache len ({total_len}); skipping compression.")
            return keys, values

        # Compute scores from indexer cache (full length)
        local_kwargs = dict(kwargs)
        local_kwargs["is_decoding"] = True
        scores = self.base_press.score(
            module, hidden_states=None, keys=keys, values=values, attentions=attentions, kwargs=local_kwargs
        )

        if scores is None or scores.numel() == 0:
            return keys, values

        if scores.shape[-1] != total_len: 
            logger.warning(f"SelectiveDecodingPress(CacheIndexer): layer {layer_idx} score_len ({scores.shape[-1]}) != total_len ({total_len}); skipping compression.")
            return keys, values

        decode_scores = scores[:, :, prefill_size:]  # (bsz, n_kv_heads, decode_len)
        # top-k within decode part (indices are relative to decode slice)
        topk = decode_scores.topk(n_kept, dim=-1)
        rel_sorted = topk.indices.sort(dim=-1).values
        abs_sorted = rel_sorted + prefill_size  # absolute token positions

        bsz = abs_sorted.shape[0]
        n_kv_heads = abs_sorted.shape[1]

        # Always keep all prefill tokens
        if prefill_size > 0:
            prefill_idx = torch.arange(prefill_size, device=keys.device, dtype=abs_sorted.dtype)
            prefill_idx = prefill_idx.view(1, 1, -1).expand(bsz, n_kv_heads, -1)
            final_idx = torch.cat([prefill_idx, abs_sorted], dim=-1)
        else:
            final_idx = abs_sorted

        gather_idx = final_idx.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        compressed_keys = keys.gather(2, gather_idx).contiguous()
        compressed_values = values.gather(2, gather_idx).contiguous()

        token_indices = final_idx[:, 0, :]
        indexer.compress_cache_by_indices(token_indices)

        return compressed_keys, compressed_values
    
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if isinstance(self.base_press, CacheIndexerScorePress):
            return self._compress_with_indexer_cache(module, keys, values, attentions, kwargs)

        layer_idx = module.layer_idx
        prefill_size = self.prefill_cache_size.get(layer_idx, 0)
        total_len = keys.shape[2]
        decode_len = total_len - prefill_size
        
        if decode_len <= 0: return keys, values
        
        target_decode_size = max(0, self.target_size - prefill_size)
        n_kept = min(decode_len, target_decode_size)
        
        if n_kept >= decode_len: return keys, values
        if n_kept <= 0:
            if prefill_size > 0:
                return keys[:, :, :prefill_size, :], values[:, :, :prefill_size, :]
            # Keep at least one token to avoid empty cache tensors.
            return keys[:, :, -1:, :], values[:, :, -1:, :]
        
        decode_keys = keys[:, :, prefill_size:, :]
        decode_values = values[:, :, prefill_size:, :]
        
        decode_scores = self.base_press.score(
            module, hidden_states, decode_keys, decode_values, attentions, kwargs
        )

        if decode_scores is None or decode_scores.numel() == 0:
            return keys, values

        score_len = int(decode_scores.shape[-1])
        if score_len <= 0:
            return keys, values

        # Some scorers (notably IndexerScorePress) can only score the most-recent query window
        # and therefore return scores shorter than the decode KV length. In that case we:
        #   - assume scores correspond to the most-recent `score_len` decode tokens
        #   - keep tokens from the unscored prefix by recency (most-recent first) to fill the budget
        if score_len > decode_len:
            warn_key = (layer_idx, "score_len_gt_decode_len", type(self.base_press).__name__)
            if warn_key not in self._warned_score_len_mismatch:
                logger.warning(
                    "SelectiveDecodingPress: base_press %s returned score_len %d > decode_len %d; "
                    "trimming scores to the last decode_len positions.",
                    type(self.base_press).__name__,
                    score_len,
                    decode_len,
                )
                self._warned_score_len_mismatch.add(warn_key)
            decode_scores = decode_scores[..., -decode_len:]
            score_len = decode_len

        if score_len != decode_len:
            # score_len < decode_len: map scores onto the decode tail window.
            warn_key = (layer_idx, "score_len_lt_decode_len", type(self.base_press).__name__)
            if warn_key not in self._warned_score_len_mismatch:
                logger.warning(
                    "SelectiveDecodingPress: base_press %s returned score_len %d != decode_len %d. "
                    "Assuming scores correspond to the most-recent %d decode tokens; "
                    "using recency fallback for older tokens.",
                    type(self.base_press).__name__,
                    score_len,
                    decode_len,
                    score_len,
                )
                self._warned_score_len_mismatch.add(warn_key)

            window_len = score_len
            window_start = decode_len - window_len
            bsz, n_kv_heads = decode_keys.shape[0], decode_keys.shape[1]

            if n_kept <= window_len:
                rel = decode_scores.topk(n_kept, dim=-1).indices.sort(dim=-1).values
                sorted_indices = rel + window_start
            else:
                # Keep all scored window tokens, and fill the remaining budget from the unscored prefix by recency.
                keep_prefix = n_kept - window_len
                if keep_prefix > 0:
                    prefix_start = window_start - keep_prefix
                    prefix_idx = torch.arange(
                        prefix_start, window_start, device=keys.device, dtype=torch.long
                    )
                    prefix_idx = prefix_idx.view(1, 1, -1).expand(bsz, n_kv_heads, -1)
                else:
                    prefix_idx = None

                window_idx = torch.arange(window_start, decode_len, device=keys.device, dtype=torch.long)
                window_idx = window_idx.view(1, 1, -1).expand(bsz, n_kv_heads, -1)
                sorted_indices = torch.cat([prefix_idx, window_idx], dim=-1) if prefix_idx is not None else window_idx
        else:
            topk = decode_scores.topk(n_kept, dim=-1)
            sorted_indices = topk.indices.sort(dim=-1).values

        gather_idx = sorted_indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        
        compressed_decode_keys = decode_keys.gather(2, gather_idx).contiguous()
        compressed_decode_values = decode_values.gather(2, gather_idx).contiguous()
        
        if prefill_size > 0:
            prefill_keys = keys[:, :, :prefill_size, :]
            prefill_values = values[:, :, :prefill_size, :]
            
            final_keys = torch.cat([prefill_keys, compressed_decode_keys], dim=2)
            final_values = torch.cat([prefill_values, compressed_decode_values], dim=2)
        else:
            final_keys = compressed_decode_keys
            final_values = compressed_decode_values
        
        return final_keys, final_values
    
    def forward_hook(self, module, input, kwargs, output):
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        q_len = hidden_states.shape[1]
        layer_idx = module.layer_idx
        
        if kwargs["cache_position"][-1] <= q_len:
            if q_len > 1:
                self.prefill_cache_size[layer_idx] = cache.get_seq_length(layer_idx)
                self._update_indexer_cache(module, hidden_states)
            return output
        
        self.hidden_states_buffer[layer_idx].append(hidden_states.detach().clone())
        self.layer_step_counts[layer_idx] += 1
        
        # 每 compression_interval 步压缩一次
        if self.layer_step_counts[layer_idx] >= self.compression_interval:
            cache_layer = cache.layers[layer_idx]
            keys, values = extract_keys_and_values(cache, layer_idx)
            attentions = output[1] if len(output) > 1 and output[1] is not None else None
            
            # 拼接 buffered hidden states
            buffered_hidden_states = torch.cat(self.hidden_states_buffer[layer_idx], dim=1)
            # Update indexer cache with buffered decode hidden states (if supported)
            self._update_indexer_cache(module, buffered_hidden_states)
            
            # 压缩（只压缩 decode 部分）
            keys, values = self.compress(module, buffered_hidden_states, keys, values, attentions, kwargs)
            
            # 更新 cache
            if isinstance(cache, QuantizedCache):
                cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
                cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
                cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)
                cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)
                cache_layer.cumulative_length = keys.shape[2]
            else:
                cache_layer.keys = keys
                cache_layer.values = values
            
            # 清空 buffer
            self.layer_step_counts[layer_idx] = 0
            self.hidden_states_buffer[layer_idx] = []
        
        # 限制 buffer 大小
        self.hidden_states_buffer[layer_idx] = (
            self.hidden_states_buffer[layer_idx][-self.hidden_states_buffer_size:]
            if self.hidden_states_buffer_size > 0
            else []
        )
        
        return output

    def reset(self):
        """Reset internal buffers and cached prefill sizes."""
        self.hidden_states_buffer = defaultdict(list)
        self.layer_step_counts = defaultdict(int)
        self.prefill_cache_size = {}
        self._warned_score_len_mismatch = set()
        # Reset underlying base press state if it maintains per-sequence cache/state.
        if hasattr(self.base_press, "reset") and callable(getattr(self.base_press, "reset")):
            self.base_press.reset()
        elif hasattr(self.base_press, "_reset_cache") and callable(getattr(self.base_press, "_reset_cache")):
            self.base_press._reset_cache()