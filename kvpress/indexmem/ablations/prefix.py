# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.indexmem.scorer import RetentionScorer, RetentionScorerConfig, ScorerNorm


@dataclass
class PrefixRetentionScorerConfig(RetentionScorerConfig):
    head_dim: int = 128
    value_dim: int = 128
    zero_init_prefix: bool = True
    rope_dim: int = field(default=0, init=False)


class PrefixRetentionScorer(RetentionScorer):
    is_query_independent = True

    def __init__(self, config: PrefixRetentionScorerConfig):
        super().__init__(config)
        self.head_dim = config.head_dim
        self.value_dim = config.value_dim
        self._cache_enabled = False
        self._cache_k: torch.Tensor | None = None
        self._cache_v: torch.Tensor | None = None
        self.prefix_query_proj = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.prefix_key_proj = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.prefix_value_proj = nn.Linear(config.hidden_size, config.value_dim, bias=False)
        self.history_norm = ScorerNorm(config.value_dim, eps=config.norm_eps)
        readout_width = config.mid_dim if config.mid_dim else config.hidden_size
        self.history_proj = nn.Linear(config.value_dim, readout_width, bias=False)
        if config.zero_init_prefix:
            nn.init.zeros_(self.history_proj.weight)

    def enable_cache(self) -> None:
        self._cache_enabled = True
        self._cache_k = None
        self._cache_v = None

    def disable_cache(self) -> None:
        self._cache_enabled = False
        self._cache_k = None
        self._cache_v = None

    @property
    def cached_length(self) -> int:
        return 0 if self._cache_k is None else int(self._cache_k.shape[2])

    def prefix_readout(self, x: torch.Tensor, *, keep: torch.Tensor | None = None) -> torch.Tensor:
        (bsz, k_len, _) = x.shape
        q = self.prefix_query_proj(x).view(bsz, k_len, 1, self.head_dim).transpose(1, 2)
        k = self.prefix_key_proj(x).view(bsz, k_len, 1, self.head_dim).transpose(1, 2)
        v = self.prefix_value_proj(x).view(bsz, k_len, 1, self.value_dim).transpose(1, 2)
        if self._cache_enabled:
            fresh = self._cache_k is None
            if not fresh:
                k = torch.cat([self._cache_k, k], dim=2)
                v = torch.cat([self._cache_v, v], dim=2)
            (self._cache_k, self._cache_v) = (k, v)
            if fresh:
                a = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                mask = None
                if k_len > 1:
                    total = k.shape[2]
                    offset = total - k_len
                    q_pos = torch.arange(k_len, device=x.device).unsqueeze(-1) + offset
                    k_pos = torch.arange(total, device=x.device).unsqueeze(0)
                    mask = (k_pos <= q_pos).view(1, 1, k_len, total)
                a = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        elif keep is None:
            a = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            causal = torch.ones((k_len, k_len), device=x.device, dtype=torch.bool).tril_()
            allowed = causal.view(1, 1, k_len, k_len) & keep.view(bsz, 1, 1, k_len)
            allowed = allowed | torch.eye(k_len, device=x.device, dtype=torch.bool).view(1, 1, k_len, k_len)
            a = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        return a.transpose(1, 2).reshape(bsz, k_len, self.value_dim)

    def _trunk(
        self, hidden_states: torch.Tensor, *, key_offset: int = 0, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        keep = None
        if mask is not None:
            keep = (mask if mask.dtype == torch.bool else mask != 0).view(hidden_states.shape[0], -1)
            if bool(keep.all()):
                keep = None
        x = self.in_norm(hidden_states)
        a = self.history_norm(self.prefix_readout(x, keep=keep))
        if self.input_proj is not None:
            return nn.functional.gelu(self.mid_norm(self.input_proj(x) + self.history_proj(a)))
        return x + self.history_proj(a)

    def project_k(
        self,
        hidden_states: torch.Tensor,
        cos=None,
        sin=None,
        *,
        value_states: torch.Tensor | None = None,
        key_offset: int | None = None,
    ) -> torch.Tensor:
        return self.gate_key(
            hidden_states,
            key_offset=self.cached_length if key_offset is None else key_offset,
            dtype=hidden_states.dtype,
        )
