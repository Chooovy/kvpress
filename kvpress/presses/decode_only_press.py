# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.cache_utils import QuantizedCache

from kvpress.presses.base_press import BasePress
from kvpress.presses.decode_press import DecodePress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.adakv_press import AdaKVPress
from kvpress.utils import extract_keys_and_values

logger = logging.getLogger(__name__)


@dataclass
class DecodeOnlyPress(DecodePress):
    base_press: ScorerPress | AdaKVPress
    compression_interval: int = 512
    target_size: int = 2048
    hidden_states_buffer_size: int = 256
    
    def __post_init__(self):
        # Track the prefill cache size for each layer
        self.prefill_cache_size = {}
        self.hidden_states_buffer = defaultdict(list)
        self.layer_step_counts = defaultdict(int)
    
    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compress only the decode portion of KV cache while preserving prefill tokens.
        
        Args:
            module: The transformer module being compressed
            hidden_states: Buffered hidden states from recent decoding steps
            keys: Full key cache (shape: [batch, n_heads, seq_len, head_dim])
            values: Full value cache (shape: [batch, n_heads, seq_len, head_dim])
            attentions: Attention weights
            kwargs: Additional keyword arguments
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Compressed (keys, values) with prefill preserved
        """
        layer_idx = module.layer_idx
        prefill_size = self.prefill_cache_size.get(layer_idx, 0)
        total_len = keys.shape[2]
        decode_len = total_len - prefill_size
        
        # If no decode tokens yet, return as is
        if decode_len <= 0:
            return keys, values
        
        # Calculate target size for decode portion
        # Note: target_size here refers to decode tokens only, not total
        target_decode_size = self.target_size
        # target_size means the number of decode tokens to keep after compression, not include prefill tokens
        n_kept = min(decode_len, target_decode_size)
        
        # If decode portion is already within target, no compression needed
        if n_kept >= decode_len:
            return keys, values
        
        logger.debug(
            f"DecodeOnlyPress: layer={layer_idx}, total={total_len}, "
            f"prefill={prefill_size}, decode={decode_len}, keeping={n_kept} decode tokens, "
            f"ratio={n_kept/decode_len:.2%}"
        )
        
        # Extract decode portion
        decode_keys = keys[:, :, prefill_size:, :]
        decode_values = values[:, :, prefill_size:, :]
        
        # Score only the decode portion
        decode_scores = self.base_press.score(
            module, hidden_states, decode_keys, decode_values, attentions, kwargs
        )
        
        # Select top-k decode tokens based on scores
        topk = decode_scores.topk(n_kept, dim=-1)
        sorted_indices = topk.indices.sort(dim=-1).values
        
        # Gather the selected decode tokens
        gather_idx = sorted_indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        compressed_decode_keys = decode_keys.gather(2, gather_idx).contiguous()
        compressed_decode_values = decode_values.gather(2, gather_idx).contiguous()
        
        # Concatenate preserved prefill with compressed decode
        if prefill_size > 0:
            prefill_keys = keys[:, :, :prefill_size, :]
            prefill_values = values[:, :, :prefill_size, :]
            
            final_keys = torch.cat([prefill_keys, compressed_decode_keys], dim=2)
            final_values = torch.cat([prefill_values, compressed_decode_values], dim=2)
        else:
            final_keys = compressed_decode_keys
            final_values = compressed_decode_values
        
        logger.debug(
            f"Compressed: {keys.shape[2]} -> {final_keys.shape[2]} "
            f"(prefill={prefill_size}, decode={decode_len}->{n_kept})"
        )
        
        return final_keys, final_values
    
    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Forward hook that manages decoding-specific compression logic with prefill preservation.
        
        This hook:
        1. Detects and records prefill phase completion
        2. Accumulates hidden states during decode phase
        3. Applies segmented compression every N steps
        4. Clears the buffer after compression
        """
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        q_len = hidden_states.shape[1]
        layer_idx = module.layer_idx
        
        # Detect prefill phase (cache_position[-1] <= q_len means we're still prefilling)
        if kwargs["cache_position"][-1] <= q_len:
            # Record prefill cache size when prefill completes (q_len > 1 means it's actual prefill, not single token)
            if q_len > 1:
                self.prefill_cache_size[layer_idx] = cache.get_seq_length(layer_idx)
                logger.debug(f"Recorded prefill size for layer {layer_idx}: {self.prefill_cache_size[layer_idx]}")
            return output
        
        # Decoding phase - accumulate hidden states
        self.hidden_states_buffer[layer_idx].append(hidden_states.detach().clone())
        self.layer_step_counts[layer_idx] += 1
        
        # Apply compression at specified intervals or when decode cache is too large
        current_cache_len = cache.get_seq_length(layer_idx)
        prefill_size = self.prefill_cache_size.get(layer_idx, 0)
        decode_size = current_cache_len - prefill_size
        
        should_compress = (
            self.layer_step_counts[layer_idx] >= self.compression_interval
            or decode_size >= self.target_size
        )
        
        if should_compress:
            logger.debug(
                f"Applying segmented decoding compression: layer={layer_idx}, "
                f"step_count={self.layer_step_counts[layer_idx]}, "
                f"decode_size={decode_size}, target={self.target_size}"
            )
            
            cache_layer = cache.layers[layer_idx]
            keys, values = extract_keys_and_values(cache, layer_idx)
            
            # Get attention weights from output
            attentions = output[1] if len(output) > 1 and output[1] is not None else None
            
            # Apply segmented compression using buffered hidden states
            buffered_hidden_states = torch.cat(self.hidden_states_buffer[layer_idx], dim=1)
            keys, values = self.compress(module, buffered_hidden_states, keys, values, attentions, kwargs)
            
            logger.debug(
                f"Applied segmented compression: keys.shape={keys.shape}, values.shape={values.shape}"
            )
            
            # Update cache with compressed keys and values
            if isinstance(cache, QuantizedCache):
                cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
                cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
                cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)
                cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)
                cache_layer.cumulative_length = keys.shape[2]
            else:
                cache_layer.keys = keys
                cache_layer.values = values
            
            # Reset step count and clear buffer for this layer
            self.layer_step_counts[layer_idx] = 0
            self.hidden_states_buffer[layer_idx] = []
        
        # Maintain buffer size limit
        self.hidden_states_buffer[layer_idx] = (
            self.hidden_states_buffer[layer_idx][-self.hidden_states_buffer_size:]
            if self.hidden_states_buffer_size > 0
            else []
        )
        
        return output
    
    def reset(self):
        """Reset the decoding press state including prefill cache size tracking."""
        self.hidden_states_buffer = defaultdict(list)
        self.layer_step_counts = defaultdict(int)
        self.prefill_cache_size = {}
        # Reset underlying base press state if it maintains per-sequence cache/state.
        if hasattr(self.base_press, "reset") and callable(getattr(self.base_press, "reset")):
            self.base_press.reset()
        elif hasattr(self.base_press, "_reset_cache") and callable(getattr(self.base_press, "_reset_cache")):
            self.base_press._reset_cache()

