# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.indexmem.scorer import RetentionScorer, RetentionScorerConfig, ScorerNorm

DEFAULT_CONV_KERNEL = 8
DEFAULT_CONV_DIM = 256


@dataclass
class ConvRetentionScorerConfig(RetentionScorerConfig):
    conv_kernel: int = DEFAULT_CONV_KERNEL
    conv_dim: int = DEFAULT_CONV_DIM
    exclude_self: bool = True
    zero_init_conv: bool = True
    rope_dim: int = field(default=0, init=False)


class ConvRetentionScorer(RetentionScorer):
    is_query_independent = True

    def __init__(self, config: ConvRetentionScorerConfig):
        super().__init__(config)
        self.conv_kernel = config.conv_kernel
        self.conv_dim = config.conv_dim
        self.exclude_self = config.exclude_self
        self._cache_enabled = False
        self._cache_z: torch.Tensor | None = None
        self._cache_len = 0
        self.history_input_proj = nn.Linear(config.hidden_size, config.conv_dim, bias=False)
        self.conv = nn.Conv1d(
            config.conv_dim, config.conv_dim, kernel_size=config.conv_kernel, groups=config.conv_dim, bias=False
        )
        self.history_norm = ScorerNorm(config.conv_dim, eps=config.norm_eps)
        readout_width = config.mid_dim if config.mid_dim else config.hidden_size
        self.history_proj = nn.Linear(config.conv_dim, readout_width, bias=False)
        if config.zero_init_conv:
            nn.init.zeros_(self.history_proj.weight)

    def enable_cache(self) -> None:
        self._cache_enabled = True
        self._cache_z = None
        self._cache_len = 0

    def disable_cache(self) -> None:
        self._cache_enabled = False
        self._cache_z = None
        self._cache_len = 0

    @property
    def cached_length(self) -> int:
        return self._cache_len

    def conv_readout(self, x: torch.Tensor) -> torch.Tensor:
        (bsz, k_len, _) = x.shape
        z = self.history_input_proj(x).transpose(1, 2)
        tail = self._cache_z if self._cache_enabled else None
        full = z if tail is None else torch.cat([tail, z], dim=2)
        if self._cache_enabled:
            self._cache_z = full[:, :, -self.conv_kernel :].detach()
            self._cache_len += k_len
        stream = full
        if self.exclude_self:
            stream = torch.cat([torch.zeros_like(full[:, :, :1]), full[:, :, :-1]], dim=2)
        padded = nn.functional.pad(stream, (self.conv_kernel - 1, 0))
        out = self.conv(padded)
        return out[:, :, -k_len:].transpose(1, 2)

    def _trunk(
        self, hidden_states: torch.Tensor, *, key_offset: int = 0, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is not None:
            keep = mask if mask.dtype == torch.bool else mask != 0
        x = self.in_norm(hidden_states)
        a = self.history_norm(self.conv_readout(x))
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
            hidden_states, key_offset=self._cache_len if key_offset is None else key_offset, dtype=hidden_states.dtype
        )
