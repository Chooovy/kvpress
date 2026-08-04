# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import get_prerope_query_states


@dataclass
class RkvPress(ScorerPress):
    """
    R-KV: Redundancy-aware KV cache compression.

    Combines attention-based importance with key similarity to identify and remove
    redundant key-value pairs. The method computes a final score by mixing:
    1. Attention scores from recent queries (smoothed with max pooling)
    2. Cosine similarity between keys (to identify redundancy)

    Based on R-KV compression approach that balances attention importance and redundancy.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    window_size : int, default=8
        Number of recent tokens to exclude from compression (recent window).
        These tokens are always kept as they are most relevant for decoding.
    kernel_size : int, default=7
        Size of the max pooling kernel applied to attention weights for smoothing.
        Helps capture local importance patterns in attention scores.
    mix_lambda : float, default=0.07
        Weight for mixing attention and similarity scores.
        final_score = attn_score * mix_lambda - similarity * (1 - mix_lambda)
        Higher values emphasize attention over redundancy reduction.
    retain_ratio : float, default=0.1
        Ratio of similar keys to retain when computing similarity.
        Controls how aggressively redundant keys are penalized.
    retain_direction : str, default="last"
        Direction for retaining similar keys. Options:
        - "last": Keep the last similar key
        - "first": Keep the first similar key
        - "last_percent": Keep the last retain_ratio% of similar keys
        - "first_percent": Keep the first retain_ratio% of similar keys
    similarity_threshold : float, default=0.5
        Cosine similarity threshold for considering keys as similar.
        Keys with similarity above this threshold are considered redundant.
    """

    compression_ratio: float = 0.0
    window_size: int = 8
    kernel_size: int = 7
    mix_lambda: float = 0.07
    retain_ratio: float = 0.1 # computing similarity
    retain_direction: str = "last"
    similarity_threshold: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        assert self.window_size >= 0, "window_size must be non-negative"
        assert self.kernel_size > 0, "kernel_size must be positive"
        assert 0.0 <= self.mix_lambda <= 1.0, "mix_lambda must be between 0 and 1"
        assert 0.0 <= self.retain_ratio <= 1.0, "retain_ratio must be between 0 and 1"
        assert self.retain_direction in ["last", "first", "last_percent", "first_percent"], \
            "retain_direction must be one of: last, first, last_percent, first_percent"

    @staticmethod
    def compute_attention_scores(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        pooling: str = "max"
    ) -> torch.Tensor:
        """
        Compute attention scores between queries and keys.

        Handles both single-head and grouped query attention (GQA).

        Parameters
        ----------
        query_states : torch.Tensor
            Query tensor with shape (batch_size, num_heads, q_len, head_dim).
        key_states : torch.Tensor
            Key tensor with shape (batch_size, num_kv_heads, kv_len, head_dim).
        pooling : str, default="max"
            Pooling method for grouped query attention. Options: "mean" or "max".

        Returns
        -------
        torch.Tensor
            Attention weights with shape (batch_size, num_kv_heads, q_len, kv_len).
        """
        batch_size, q_heads, q_len, head_dim = query_states.shape
        kv_heads = key_states.shape[1]
        query_group_size = q_heads // kv_heads

        if query_group_size == 1:
            # Standard attention: no grouping
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(head_dim)
        else:
            # Grouped query attention (GQA)
            # Reshape queries: [batch_size, kv_heads, query_group_size, q_len, head_dim]
            query_states = query_states.view(
                batch_size, kv_heads, query_group_size, q_len, head_dim
            )

            # Expand keys: [batch_size, kv_heads, 1, kv_len, head_dim]
            key_states = key_states.unsqueeze(2)

            # Compute attention: [batch_size, kv_heads, query_group_size, q_len, kv_len]
            attn_weights = torch.matmul(
                query_states, key_states.transpose(3, 4)
            ) / math.sqrt(head_dim)

            # Pool across query groups
            if pooling == "mean":
                attn_weights = attn_weights.mean(dim=2)
            elif pooling == "max":
                attn_weights = attn_weights.max(dim=2).values
            else:
                raise ValueError(f"Pooling method '{pooling}' not supported. Use 'mean' or 'max'.")

        return attn_weights

    @staticmethod
    def cal_similarity(
        key_states: torch.Tensor,
        threshold: float = 0.5,
        retain_ratio: float = 0.1,
        retain_direction: str = "last",
    ) -> torch.Tensor:
        """
        Calculate cosine similarity between keys to identify redundancy.

        Computes pairwise cosine similarity between all keys and produces a similarity
        score that penalizes redundant keys.

        Parameters
        ----------
        key_states : torch.Tensor
            Key tensor with shape (batch_size, num_kv_heads, kv_len, head_dim).
        threshold : float, default=0.5
            Cosine similarity threshold for considering keys as similar.
        retain_ratio : float, default=0.1
            Ratio of similar keys to retain.
        retain_direction : str, default="last"
            Direction for retaining similar keys.

        Returns
        -------
        torch.Tensor
            Similarity scores with shape (batch_size, num_kv_heads, kv_len).
            Higher scores indicate more redundancy (will be subtracted in final score).
        """
        # Extract first batch (assuming batch_size=1 for KV cache compression)
        k = key_states[0]  # [num_kv_heads, kv_len, head_dim]
        num_heads = k.shape[0]

        # Normalize keys for cosine similarity
        k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Compute pairwise cosine similarity: [num_kv_heads, kv_len, kv_len]
        similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2))

        # Zero out diagonal (self-similarity)
        for h in range(num_heads):
            similarity_cos[h].fill_diagonal_(0.0)

        # Create similarity mask: [num_kv_heads, kv_len, kv_len]
        similarity_mask = similarity_cos > threshold

        # Compute indices of similar tokens
        seq_len = similarity_mask.size(-1)
        k_retain = max(1, int(seq_len * retain_ratio))  # At least 1 to avoid empty topk

        indices = torch.where(
            similarity_mask,
            torch.arange(similarity_mask.size(-1), device=similarity_mask.device).unsqueeze(0).unsqueeze(0).expand_as(similarity_mask),
            torch.zeros_like(similarity_mask, dtype=torch.long),
        )

        # Select tokens to retain based on direction
        if retain_direction == "last":
            # Find the last True index in each row
            similarity_retain = torch.max(indices, dim=-1)[0]
        elif retain_direction == "first":
            # Find the first True index in each row (excluding zeros from non-similar tokens)
            # Replace zeros with max value, then find min
            temp_indices = torch.where(similarity_mask, indices, torch.full_like(indices, seq_len))
            similarity_retain = torch.min(temp_indices, dim=-1)[0]
        elif retain_direction == "last_percent":
            # Keep the last retain_ratio% of similar keys
            if k_retain < indices.size(-1):
                similarity_retain = torch.topk(indices, k=k_retain, dim=-1)[0][:, :, 0]
            else:
                similarity_retain = torch.max(indices, dim=-1)[0]
        elif retain_direction == "first_percent":
            # Keep the first retain_ratio% of similar keys
            if k_retain < indices.size(-1):
                similarity_retain = torch.topk(indices, k=k_retain, dim=-1, largest=False)[0][:, :, -1]
            else:
                temp_indices = torch.where(similarity_mask, indices, torch.full_like(indices, seq_len))
                similarity_retain = torch.min(temp_indices, dim=-1)[0]
        else:
            raise ValueError(f"retain_direction '{retain_direction}' not supported")

        # Zero out retained positions in similarity scores
        batch_idx = torch.arange(num_heads, device=key_states.device).unsqueeze(1).expand(-1, similarity_retain.size(1))
        seq_idx = torch.arange(similarity_retain.size(1), device=key_states.device).unsqueeze(0).expand(num_heads, -1)
        similarity_cos[batch_idx, seq_idx, similarity_retain] = 0

        # Average similarity across the sequence dimension (mean over columns)
        # Then apply softmax to normalize
        similarity_scores = similarity_cos.mean(dim=-1).softmax(dim=-1)

        # Add batch dimension back: [1, num_kv_heads, kv_len]
        return similarity_scores.unsqueeze(0)

    @staticmethod
    def compute_window_attention(
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        window_size: int,
        position_embeddings: tuple
    ) -> torch.Tensor:
        """
        Compute attention weights for the recent window of queries.

        Parameters
        ----------
        module : nn.Module
            The attention module.
        hidden_states : torch.Tensor
            Input hidden states with shape (batch_size, seq_len, hidden_dim).
        keys : torch.Tensor
            Key tensor with shape (batch_size, num_kv_heads, kv_len, head_dim).
        window_size : int
            Size of the recent window.
        position_embeddings : tuple
            Tuple of (cos, sin) position embeddings for RoPE.

        Returns
        -------
        torch.Tensor
            Attention weights with shape (batch_size, num_heads, window_size, kv_len - window_size).
        """
        bsz, _, k_len, _ = keys.shape
        num_heads = module.config.num_attention_heads
        num_kv_heads = module.config.num_key_value_heads
        head_dim = module.head_dim
        num_key_value_groups = num_heads // num_kv_heads

        # Get last window_size queries
        query_states = get_prerope_query_states(module, hidden_states[:, -window_size:])

        # Apply RoPE
        cos, sin = position_embeddings
        cos, sin = cos[:, -window_size:], sin[:, -window_size:]
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        # Compute attention weights
        key_states = repeat_kv(keys, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        
        # Apply causal mask to prevent attending to future tokens in the window
        attention_mask = torch.ones_like(attn_weights) * float("-inf")
        attention_mask = torch.triu(attention_mask, diagonal=k_len - window_size + 1)
        attn_weights += attention_mask
        
        # Softmax and extract attention to non-window keys
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights[..., :-window_size]

        return attn_weights

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        """
        Compute importance scores for each key-value pair using R-KV method.

        Combines attention-based scores with similarity-based redundancy detection.

        Parameters
        ----------
        module : nn.Module
            The transformer attention layer.
        hidden_states : torch.Tensor
            Input embeddings with shape (batch_size, seq_len, hidden_dim).
        keys : torch.Tensor
            Key tensors with shape (batch_size, num_kv_heads, seq_len, head_dim).
        values : torch.Tensor
            Value tensors with shape (batch_size, num_kv_heads, seq_len, head_dim).
        attentions : torch.Tensor
            Attention weights with shape (batch_size, num_heads, seq_len, seq_len).
            May be None if not computed.
        kwargs : dict
            Additional arguments including position_embeddings.

        Returns
        -------
        torch.Tensor
            Importance scores with shape (batch_size, num_kv_heads, seq_len).
        """
        bsz, num_kv_heads, k_len, head_dim = keys.shape
        num_heads = module.config.num_attention_heads
        num_key_value_groups = num_heads // num_kv_heads

        # Ensure we have enough tokens for the window
        if hidden_states.shape[1] <= self.window_size:
            # If sequence is too short, return uniform scores (no compression)
            return torch.ones(bsz, num_kv_heads, k_len, device=keys.device, dtype=keys.dtype)

        # Step 1: Compute attention scores from recent window
        if attentions is not None:
            # Use precomputed attention weights: last window_size queries, first k_len - window_size keys
            # Shape: [batch, num_heads, window_size, k_len - window_size]
            attn_weights = attentions[..., -self.window_size:, :-self.window_size]
            
            # Softmax over keys and average over queries in the window
            attn_weights_sum = F.softmax(attn_weights, dim=-1, dtype=torch.float32).mean(dim=-2).to(keys.dtype)
            # Shape: [batch, num_heads, k_len - window_size]
            
            # Average across query groups to get per-KV-head scores
            attn_weights_sum = attn_weights_sum.view(bsz, num_kv_heads, num_key_value_groups, -1)
            attn_weights_sum = attn_weights_sum.mean(dim=2)
            # Shape: [batch, num_kv_heads, k_len - window_size]
        else:
            # Compute attention weights for the window (already softmaxed)
            # Shape: [batch, num_heads, window_size, k_len - window_size]
            attn_weights = self.compute_window_attention(
                module, hidden_states, keys, self.window_size, kwargs["position_embeddings"]
            )
            
            # Average over queries in the window
            attn_weights_sum = attn_weights.mean(dim=-2)
            # Shape: [batch, num_heads, k_len - window_size]
            
            # Average across query groups to get per-KV-head scores
            attn_weights_sum = attn_weights_sum.view(bsz, num_kv_heads, num_key_value_groups, -1)
            attn_weights_sum = attn_weights_sum.mean(dim=2)
            # Shape: [batch, num_kv_heads, k_len - window_size]

        # Step 2: Apply max pooling to smooth attention scores
        attn_cache = F.max_pool1d(
            attn_weights_sum,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1
        )

        # Step 3: Compute key similarity (redundancy score)
        # Compute for ALL keys, then slice to exclude window
        similarity_cos = self.cal_similarity(
            keys,
            threshold=self.similarity_threshold,
            retain_ratio=self.retain_ratio,
            retain_direction=self.retain_direction,
        )
        # Remove the window tokens from similarity scores
        similarity_cos = similarity_cos[:, :, :-self.window_size]

        # Step 4: Combine attention and similarity scores
        # final_score = attention * mix_lambda - similarity * (1 - mix_lambda)
        # Higher attention = keep, higher similarity = redundant (remove)
        final_score = attn_cache * self.mix_lambda - similarity_cos * (1 - self.mix_lambda)

        # Step 5: Add back the window tokens with maximum score to ensure they're kept 
        final_score = F.pad(final_score, (0, self.window_size), value=final_score.max().item()) # 0：左边不填充，self.window_size：右边填充 self.window_size 个value元素

        return final_score

