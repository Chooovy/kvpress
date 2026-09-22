# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.indexmem.scorer import RetentionScorer, RetentionScorerConfig, ScorerNorm

DEFAULT_STATE_DIM = 256
DEFAULT_FIXED_HALF_LIFE = 512.0
DEFAULT_GATE_BIAS = 2.0


def gated_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    acc = torch.promote_types(torch.promote_types(a.dtype, b.dtype), torch.float32)
    a = a.to(acc)
    x = b.to(acc)
    length = a.shape[1]
    stride = 1
    while stride < length:
        a_sh = nn.functional.pad(a[:, :-stride], (0, 0, stride, 0))
        x_sh = nn.functional.pad(x[:, :-stride], (0, 0, stride, 0))
        x = x + a * x_sh
        a = a * a_sh
        stride *= 2
    return x


@dataclass
class RecurrentRetentionScorerConfig(RetentionScorerConfig):
    state_dim: int = DEFAULT_STATE_DIM
    gate_mode: str = "learned"
    gate_bias: float = DEFAULT_GATE_BIAS
    fixed_half_life: float = DEFAULT_FIXED_HALF_LIFE
    zero_init_state: bool = True
    rope_dim: int = field(default=0, init=False)


class RecurrentRetentionScorer(RetentionScorer):
    is_query_independent = True

    def __init__(self, config: RecurrentRetentionScorerConfig):
        super().__init__(config)
        self.state_dim = config.state_dim
        self.gate_mode = config.gate_mode
        self._cache_enabled = False
        self._cache_state: torch.Tensor | None = None
        self._cache_len = 0
        self.state_input_proj = nn.Linear(config.hidden_size, config.state_dim, bias=False)
        if config.gate_mode == "learned":
            self.state_gate_proj = nn.Linear(config.hidden_size, config.state_dim, bias=True)
            nn.init.zeros_(self.state_gate_proj.weight)
            nn.init.constant_(self.state_gate_proj.bias, config.gate_bias)
            self.logit_retain = None
        else:
            self.state_gate_proj = None
            retain = 0.5 ** (1.0 / config.fixed_half_life)
            logit = torch.logit(torch.tensor(retain, dtype=torch.float32))
            self.logit_retain = nn.Parameter(logit)
        self.history_norm = ScorerNorm(config.state_dim, eps=config.norm_eps)
        readout_width = config.mid_dim if config.mid_dim else config.hidden_size
        self.history_proj = nn.Linear(config.state_dim, readout_width, bias=False)
        if config.zero_init_state:
            nn.init.zeros_(self.history_proj.weight)

    def enable_cache(self) -> None:
        self._cache_enabled = True
        self._cache_state = None
        self._cache_len = 0

    def disable_cache(self) -> None:
        self._cache_enabled = False
        self._cache_state = None
        self._cache_len = 0

    @property
    def cached_length(self) -> int:
        return self._cache_len

    def state_readout(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        acc = torch.promote_types(x.dtype, torch.float32)
        u = self.state_input_proj(x).to(acc)
        if self.gate_mode == "learned":
            g = torch.sigmoid(self.state_gate_proj(x).to(acc))
        else:
            g = torch.sigmoid(self.logit_retain.to(acc)).expand_as(u)
        states = gated_scan(g, (1.0 - g) * u)
        carry = self._cache_state
        if carry is not None:
            states = states + torch.cumprod(g, dim=1) * carry.unsqueeze(1)
        if self._cache_enabled:
            self._cache_state = states[:, -1, :].detach()
            self._cache_len += x.shape[1]
        prev = torch.zeros_like(states[:, :1, :]) if carry is None else carry.unsqueeze(1)
        return torch.cat([prev, states[:, :-1, :]], dim=1).to(x.dtype)

    def _trunk(
        self, hidden_states: torch.Tensor, *, key_offset: int = 0, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is not None:
            keep = mask if mask.dtype == torch.bool else mask != 0
        x = self.in_norm(hidden_states)
        a = self.history_norm(self.state_readout(x))
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
