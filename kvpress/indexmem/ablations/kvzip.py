# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from kvpress.indexmem.scorer import MASK_NEG, RetentionScorer, RetentionScorerConfig, _inv_softplus

DEFAULT_KVZIP_DIM = 16
DEFAULT_KVZIP_BASE = 16


@dataclass
class KVzipRetentionScorerConfig(RetentionScorerConfig):
    kvzip_dim: int = DEFAULT_KVZIP_DIM
    kvzip_base: int = DEFAULT_KVZIP_BASE
    kvzip_ngroup: int = 1


class KVzipRetentionScorer(RetentionScorer):

    def __init__(self, config: KVzipRetentionScorerConfig):
        super().__init__(config)
        self.config = config
        self.kvzip_dim = config.kvzip_dim
        self.kvzip_base = config.kvzip_base
        self.ngroup = config.kvzip_ngroup
        self.input_proj = None
        self.mid_norm = None
        self.magnitude_proj = None
        n_out = self.n_heads * self.ngroup * config.kvzip_dim
        self.q_proj = nn.Linear(config.hidden_size, n_out, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.n_heads * config.kvzip_dim, bias=False)
        self.q_norm = nn.RMSNorm(config.kvzip_dim, eps=1e-06)
        self.k_norm = nn.RMSNorm(config.kvzip_dim, eps=1e-06)
        self.b = nn.Parameter(torch.zeros(self.n_heads, 1, self.ngroup))
        if config.kvzip_base:
            self.k_base = nn.Parameter(torch.zeros(self.n_heads, config.kvzip_base, config.kvzip_dim))
        else:
            self.k_base = None
        self.inv_sqrt_d = 1.0 / math.sqrt(config.kvzip_dim)
        if config.decay:
            self.retention_proj = nn.Linear(config.hidden_size, config.n_heads, bias=True)
            nn.init.zeros_(self.retention_proj.weight)
            nn.init.constant_(self.retention_proj.bias, _inv_softplus(-config.decay_init))

    @property
    def weight_dtype(self) -> torch.dtype:
        return self.k_proj.weight.dtype

    def _score_and_decay(
        self, hidden_states: torch.Tensor, *, key_offset: int = 0, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if hidden_states.dtype != self.weight_dtype:
            hidden_states = hidden_states.to(self.weight_dtype)
        (bsz, k_len, _) = hidden_states.shape
        x = self.in_norm(hidden_states)
        q = self.q_norm(self.q_proj(x).view(bsz, k_len, self.n_heads, self.ngroup, self.kvzip_dim))
        k = self.k_norm(self.k_proj(x).view(bsz, k_len, self.n_heads, 1, self.kvzip_dim))
        logit = (q * k).sum(-1).float() * self.inv_sqrt_d
        logit = logit + self.b.permute(1, 0, 2).float()
        if self.k_base is not None:
            base = torch.einsum("bshgd,hnd->bshgn", q.float(), self.k_base.float()) * self.inv_sqrt_d
            joint = torch.cat([logit.unsqueeze(-1), base], dim=-1)
            log_score = logit - torch.logsumexp(joint, dim=-1)
        else:
            log_score = logit
        scores = log_score.mean(-1).permute(0, 2, 1)
        if self.pos_slope:
            pos = torch.arange(key_offset, key_offset + k_len, device=scores.device, dtype=scores.dtype)
            scores = scores + self.pos_slope * pos
        log_retention_rate = None
        if self.retention_proj is not None:
            log_retention_rate = -nn.functional.softplus(self.retention_proj(x).float()).transpose(1, 2)
        if mask is not None:
            keep = mask if mask.dtype == torch.bool else mask != 0
            keep = ~keep.view(keep.shape[0], 1, -1)
            scores = scores.masked_fill(keep, MASK_NEG)
            if log_retention_rate is not None:
                log_retention_rate = log_retention_rate.masked_fill(keep, 0.0)
        return (scores, log_retention_rate)
