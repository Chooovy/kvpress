# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn
from transformers.models.llama.modeling_llama import rotate_half

from kvpress.presses.base_press import BasePress
from kvpress.utils import get_prerope_query_states


@dataclass
class ThinKPress(BasePress):
    """
    ThinK: Channel-wise key compression for transformer attention.

    ThinK compresses the dimensions of the keys, and not the sequence length.
    Hence it can be combined with any other press that compresses the sequence length, e.g.
    press = ComposedPress([SnapKVPress(0.5), ThinKPress(0.5)])

    Here, we zero out the pruned dimensions resulting in no memory gain (the shape of the keys remains the same).
    To achieve memory savings, several options can be considered (see https://github.com/NVIDIA/kvpress/pull/18/),
    we might implement them in the future, especially if other similar presses are requested.

    This press has been reviewed by Yuhui Xu, first author of the ThinK paper.

    Based on ThinK (https://arxiv.org/pdf/2407.21018).

    Parameters
    ----------
    key_channel_compression_ratio : float, default=0.0
        Fraction of key channels (dimensions) to remove during compression.
    window_size : int, default=32
        Number of recent tokens to use for computing key channel importance.
    """

    key_channel_compression_ratio: float = 0.0
    window_size: int = 32
    # Pruning controls
    # - pairwise_prune: prune RoPE dimension pairs (2i, 2i+1) together to avoid breaking RoPE structure.
    # - sync_kv_prune: apply the same dimension pruning mask to both K and V (recommended for high ratios).
    pairwise_prune: bool = False
    sync_kv_prune: bool = False
    # If we prune key dimensions (set to 0) but keep using the original 1/sqrt(d_k) scaling
    # inside softmax attention, logits variance drops and attention becomes too uniform.
    # Enable this to compensate by scaling pruned keys by sqrt(d_k / d_eff),
    # where d_eff = d_k - n_pruned.
    qk_scale_correction: bool = True

    def compute_window_queries(self, module, hidden_states, position_embeddings):
        """
        Re-compute the last window_size query states
        """
        # Get last self.window_size queries
        query_states = get_prerope_query_states(module, hidden_states[:, -self.window_size :])

        # Apply RoPE
        cos, sin = position_embeddings
        cos, sin = cos[:, -self.window_size :], sin[:, -self.window_size :]
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        return query_states

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
        If other similar presses are requested, we might create a generic compress method for dimension pruning
        to avoid code duplication.
        """

        if self.key_channel_compression_ratio == 0:
            return keys, values

        # Compute scores per dimension
        bsz, num_key_value_heads, k_len, head_dim = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        queries = self.compute_window_queries(module, kwargs["hidden_states"], kwargs["position_embeddings"])
        queries_norm = torch.pow(queries, 2).mean(dim=2)  # (bsz, num_heads, head_dim)
        queries_norm = queries_norm.view(bsz, num_key_value_heads, num_key_value_groups, module.head_dim).mean(2)
        keys_norm = torch.pow(keys, 2).mean(dim=2)
        key_scores = queries_norm * keys_norm  # (bsz, num_key_value_heads, head_dim)

        # Prune dimensions with the lowest scores by setting them to 0
        ratio = float(self.key_channel_compression_ratio)
        n_pruned_dims = int(head_dim * ratio)
        n_pruned_dims = min(max(1, n_pruned_dims), head_dim - 1)

        if self.pairwise_prune and head_dim % 2 == 0 and n_pruned_dims >= 2:
            # Convert per-dim scores -> per-pair scores, then prune whole pairs.
            pair_scores = 0.5 * (key_scores[..., 0::2] + key_scores[..., 1::2])  # (bsz, kv_heads, head_dim/2)
            n_pairs = head_dim // 2
            n_pruned_pairs = min(max(1, n_pruned_dims // 2), n_pairs - 1)
            pair_idx = pair_scores.topk(n_pruned_pairs, dim=-1, largest=False).indices  # (bsz, kv_heads, n_pruned_pairs)

            dim_idx = torch.stack((pair_idx * 2, pair_idx * 2 + 1), dim=-1).reshape(bsz, num_key_value_heads, 2 * n_pruned_pairs)
            dim_idx = dim_idx.unsqueeze(2).expand(-1, -1, k_len, -1)
            keys = keys.scatter_(-1, dim_idx, 0)
            if self.sync_kv_prune:
                values = values.scatter_(-1, dim_idx, 0)
            n_pruned_eff = 2 * n_pruned_pairs
        else:
            dim_idx = key_scores.topk(n_pruned_dims, dim=-1, largest=False).indices
            dim_idx = dim_idx.unsqueeze(2).expand(-1, -1, k_len, -1)
            keys = keys.scatter_(-1, dim_idx, 0)
            if self.sync_kv_prune:
                values = values.scatter_(-1, dim_idx, 0)
            n_pruned_eff = n_pruned_dims

        # QK scaling correction: scale K so that Var(q^T k) stays roughly stable after pruning.
        # Equivalent to using 1/sqrt(d_eff) instead of 1/sqrt(d_k) in attention.
        if self.qk_scale_correction and n_pruned_eff > 0 and head_dim - n_pruned_eff > 0:
            d_eff = head_dim - n_pruned_eff
            scale = (head_dim / d_eff) ** 0.5
            keys = keys * scale

        return keys, values

    @property
    def compression_ratio(self):
        return self.key_channel_compression_ratio / 2

    @compression_ratio.setter
    def compression_ratio(self, value):
        raise AttributeError(f"compression ratio cannot be set for {type(self).__name__}")
