# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.cache_utils import QuantizedCache

from kvpress.presses.decode_press import DecodePress
from kvpress.presses.indexer_score_press_cache import CacheIndexerScorePress
from kvpress.utils import extract_keys_and_values

logger = logging.getLogger(__name__)


@dataclass
class CacheIndexerDecodingPress(DecodePress):
    base_press: CacheIndexerScorePress
    compression_interval: int = 128
    target_size: int = 2048
    hidden_states_buffer_size: int = 128
    use_torch_compile: bool = False
    
    def __post_init__(self):
        assert isinstance(self.base_press, CacheIndexerScorePress)
        
        self.hidden_states_buffer = defaultdict(list)
        self.layer_step_counts = defaultdict(int)
        
        if self.use_torch_compile and hasattr(torch, 'compile'):
            try:
                # 使用 "default" mode 避免 CUDA Graphs
                self._compress_core = torch.compile(self._compress_core_impl, mode="default")
                logger.info("torch.compile enabled for compress_core (mode=default)")
            except:
                logger.warning("torch.compile not available, falling back to eager mode")
                self._compress_core = self._compress_core_impl
        else:
            self._compress_core = self._compress_core_impl
    
    def post_init_from_model(self, model):
        self.base_press.post_init_from_model(model)
    
    def _compress_core_impl(self, scores, keys, values, n_kept, head_dim):
        indices = scores.topk(n_kept, dim=-1).indices  # (bsz, n_heads, n_kept)
        sorted_indices = indices.sort(dim=-1).values
        gather_idx = sorted_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        
        compressed_keys = keys.gather(2, gather_idx).contiguous()
        compressed_values = values.gather(2, gather_idx).contiguous()
        
        return compressed_keys, compressed_values, sorted_indices
    
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        layer_idx = module.layer_idx
        indexer = getattr(module, self.base_press.scorer_attr)
        k_len = keys.shape[2]
        
        # 1. indexer forward一次累积 buffered hidden states 到 indexer cache
        with torch.no_grad():
            _ = indexer.forward_cache(hidden_states, freqs_cis=None, mask=None)
        
        # 2. 检查长度是否匹配
        indexer_len = indexer.k_cache.shape[1]
        if indexer_len != k_len:
            logger.error(f"Layer {layer_idx}: indexer cache len ({indexer_len}) != KV cache len ({k_len}). This should not happen!")
            return keys, values
        
        if k_len <= self.target_size: return keys, values
        
        n_kept = self.target_size
        target_ratio = 1.0 - (n_kept / k_len)

        
        # 3. 计算 scores, 设置标志让 score 方法知道要直接使用 indexer cache
        kwargs["is_decoding"] = True
        original_ratio = self.base_press.compression_ratio
        self.base_press.compression_ratio = target_ratio
        
        # score() 方法会直接访问 indexer cache，不需要传入真实的 hidden_states
        scores = self.base_press.score(
            module, hidden_states=None, keys=keys, values=values, 
            attentions=attentions, kwargs=kwargs
        )
        
        self.base_press.compression_ratio = original_ratio
        
        compressed_keys, compressed_values, sorted_indices = self._compress_core(scores, keys, values, n_kept, module.head_dim)
        
        bsz = sorted_indices.shape[0]
        token_indices = sorted_indices[:, 0, :]  # (bsz, n_kept)
        
        indexer.compress_cache_by_indices(token_indices)
        
        return compressed_keys, compressed_values
    
    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], 
                     kwargs: dict, output: list):
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        q_len = hidden_states.shape[1]
        layer_idx = module.layer_idx
        
        # Prefill 阶段：初始化 indexer cache
        if kwargs["cache_position"][-1] <= q_len:
            if q_len > 1:  # 确保是 prefill（多个 tokens）
                indexer = getattr(module, self.base_press.scorer_attr, None)
                if indexer is not None:
                    with torch.no_grad():
                        _ = indexer.forward_cache(hidden_states, freqs_cis=None, mask=None)
            return output
        
        self.hidden_states_buffer[layer_idx].append(hidden_states.detach().clone())
        self.layer_step_counts[layer_idx] += 1
        
        current_seq_len = cache.get_seq_length(layer_idx)
        should_compress = (
            self.layer_step_counts[layer_idx] >= self.compression_interval
            or current_seq_len >= self.target_size
        )
        
        if should_compress:
            cache_layer = cache.layers[layer_idx]
            keys, values = extract_keys_and_values(cache, layer_idx)
            attentions = output[1] if len(output) > 1 and output[1] is not None else None
            
            # 优化：使用torch.cat的out参数来减少内存分配（如果buffer不为空）
            if len(self.hidden_states_buffer[layer_idx]) > 1:
                buffered_hidden_states = torch.cat(
                    self.hidden_states_buffer[layer_idx], dim=1
                )
            else:
                buffered_hidden_states = self.hidden_states_buffer[layer_idx][0]
            
            # 压缩（会自动累积 buffer 到 indexer cache）
            keys, values = self.compress(
                module, buffered_hidden_states, keys, values, attentions, kwargs
            )
            
            # 更新 KV cache
            if isinstance(cache, QuantizedCache):
                cache_layer._quantized_keys = cache_layer._quantize(
                    keys, axis=cache_layer.axis_key
                )
                cache_layer._quantized_values = cache_layer._quantize(
                    values, axis=cache_layer.axis_value
                )
                cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)
                cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)
                cache_layer.cumulative_length = keys.shape[2]
            else:
                cache_layer.keys = keys
                cache_layer.values = values
            
            # 清空 buffer 和计数
            self.layer_step_counts[layer_idx] = 0
            self.hidden_states_buffer[layer_idx] = []
        
        elif self.hidden_states_buffer_size > 0 and len(self.hidden_states_buffer[layer_idx]) > self.hidden_states_buffer_size:
            self.hidden_states_buffer[layer_idx] = (
                self.hidden_states_buffer[layer_idx][-self.hidden_states_buffer_size:]
            )
        
        return output
    
    def reset(self):
        self.hidden_states_buffer = defaultdict(list)
        self.layer_step_counts = defaultdict(int)
        # Reset underlying base press state if it maintains per-sequence cache/state.
        if hasattr(self.base_press, "reset") and callable(getattr(self.base_press, "reset")):
            self.base_press.reset()
        elif hasattr(self.base_press, "_reset_cache") and callable(getattr(self.base_press, "_reset_cache")):
            self.base_press._reset_cache()