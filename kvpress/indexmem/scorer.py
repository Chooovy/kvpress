# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for IndexMem++: extracted and renamed retention scoring.

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn

MASK_NEG = -10000.0


class ScorerNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-05):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype.itemsize >= 4:
            return nn.functional.layer_norm(x, (x.shape[-1],), self.weight.to(x.dtype), self.bias.to(x.dtype), self.eps)
        return _Fp32LayerNorm.apply(x, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class _Fp32LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        x32 = x.float()
        mean = x32.mean(-1, keepdim=True)
        var = x32.var(-1, unbiased=False, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        x_hat = (x32 - mean) * rstd
        out = x_hat * weight.float() + bias.float()
        ctx.save_for_backward(x, weight, mean, rstd)
        return out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (x, weight, mean, rstd) = ctx.saved_tensors
        g = grad_out.float()
        x_hat = (x.float() - mean) * rstd
        grad_weight = grad_bias = None
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            reduce_dims = tuple(range(g.dim() - 1))
            if ctx.needs_input_grad[1]:
                grad_weight = (g * x_hat).sum(reduce_dims).to(weight.dtype)
            if ctx.needs_input_grad[2]:
                grad_bias = g.sum(reduce_dims).to(weight.dtype)
        grad_x = None
        if ctx.needs_input_grad[0]:
            n = x.shape[-1]
            gw = g * weight.float()
            grad_x = rstd / n * (n * gw - gw.sum(-1, keepdim=True) - x_hat * (gw * x_hat).sum(-1, keepdim=True))
            grad_x = grad_x.to(x.dtype)
        return (grad_x, grad_weight, grad_bias, None)


def _inv_softplus(y: float) -> float:
    if y == 0:
        return -30.0
    return float(math.log(math.expm1(y))) if y < 20 else float(y)


DEFAULT_POS_SLOPE = 1e-06
DEFAULT_AGE_SCALE = 16384.0
DEFAULT_DECAY_INIT = -1.0


@dataclass
class RetentionScorerConfig:
    hidden_size: int
    n_heads: int
    mid_dim: int = 0
    norm_eps: float = 1e-05
    pos_slope: float = DEFAULT_POS_SLOPE
    gate_scale: bool = False
    decay: bool = False
    age_scale: float = DEFAULT_AGE_SCALE
    decay_init: float = DEFAULT_DECAY_INIT
    rope_dim: int = field(default=0, init=False)


class RetentionScorer(nn.Module):
    GATE_SCALE_INIT = staticmethod(lambda _n_heads=None: 1.0)
    is_query_independent = True

    def __init__(self, config: RetentionScorerConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.mid_dim = config.mid_dim
        self.pos_slope = config.pos_slope
        self.rope_dim = config.rope_dim
        self.in_norm = ScorerNorm(config.hidden_size, eps=config.norm_eps)
        if config.mid_dim:
            self.input_proj = nn.Linear(config.hidden_size, config.mid_dim, bias=False)
            self.mid_norm = ScorerNorm(config.mid_dim, eps=config.norm_eps)
            self.magnitude_proj = nn.Linear(config.mid_dim, config.n_heads, bias=False)
        else:
            self.input_proj = None
            self.mid_norm = None
            self.magnitude_proj = nn.Linear(config.hidden_size, config.n_heads, bias=False)
        self.gate_scale = nn.Parameter(torch.tensor([self.GATE_SCALE_INIT()])) if config.gate_scale else None
        self.decay = config.decay
        self.age_scale = config.age_scale
        if config.decay:
            self.retention_proj = nn.Linear(
                config.mid_dim if config.mid_dim else config.hidden_size, config.n_heads, bias=True
            )
            nn.init.zeros_(self.retention_proj.weight)
            nn.init.constant_(self.retention_proj.bias, _inv_softplus(-config.decay_init))
        else:
            self.retention_proj = None

    @property
    def weight_dtype(self) -> torch.dtype:
        return self.magnitude_proj.weight.dtype

    def require_gate_scale(self) -> torch.Tensor:
        return self.gate_scale

    def score_keys(
        self, hidden_states: torch.Tensor, *, key_offset: int = 0, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        (scores, _) = self._score_and_decay(hidden_states, key_offset=key_offset, mask=mask)
        return scores

    def _score_and_decay(
        self, hidden_states: torch.Tensor, *, key_offset: int = 0, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if hidden_states.dtype != self.weight_dtype:
            hidden_states = hidden_states.to(self.weight_dtype)
        x = self._trunk(hidden_states, key_offset=key_offset, mask=mask)
        scores = self.magnitude_proj(x).float()
        scores = scores.transpose(1, 2)
        if self.pos_slope:
            k_len = hidden_states.shape[1]
            pos = torch.arange(key_offset, key_offset + k_len, device=scores.device, dtype=scores.dtype)
            scores = scores + self.pos_slope * pos
        log_retention_rate = None
        if self.retention_proj is not None:
            log_retention_rate = -nn.functional.softplus(self.retention_proj(x).float())
            log_retention_rate = log_retention_rate.transpose(1, 2)
        if mask is not None:
            keep = mask if mask.dtype == torch.bool else mask != 0
            keep = ~keep.view(keep.shape[0], 1, -1)
            scores = scores.masked_fill(keep, MASK_NEG)
            if log_retention_rate is not None:
                log_retention_rate = log_retention_rate.masked_fill(keep, 0.0)
        return (scores, log_retention_rate)

    def _trunk(
        self, hidden_states: torch.Tensor, *, key_offset: int = 0, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.in_norm(hidden_states)
        if self.input_proj is not None:
            x = nn.functional.gelu(self.mid_norm(self.input_proj(x)))
        return x

    def score_at(
        self, hidden_states: torch.Tensor, query_pos: float, *, key_offset: int = 0, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        (scores, log_retention_rate) = self._score_and_decay(hidden_states, key_offset=key_offset, mask=mask)
        if log_retention_rate is None:
            return scores
        k_len = hidden_states.shape[1]
        pos = torch.arange(key_offset, key_offset + k_len, device=scores.device, dtype=scores.dtype)
        age = (float(query_pos) - pos).clamp(min=0.0) / self.age_scale
        return scores + log_retention_rate * age

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        key_hidden_states: torch.Tensor | None = None,
        key_cos: torch.Tensor | None = None,
        key_sin: torch.Tensor | None = None,
        query_offset: int | None = None,
    ) -> torch.Tensor:
        keys = hidden_states if key_hidden_states is None else key_hidden_states
        q_len = hidden_states.shape[1]
        (base, log_retention_rate) = self._score_and_decay(keys)
        if log_retention_rate is None:
            scores = self.expand_to_pairs(base, q_len)
        else:
            k_len = keys.shape[1]
            if query_offset is None:
                query_offset = k_len - q_len
            q_pos = torch.arange(q_len, device=base.device, dtype=base.dtype) + query_offset
            k_pos = torch.arange(k_len, device=base.device, dtype=base.dtype)
            age = (q_pos.view(-1, 1) - k_pos.view(1, -1)) / self.age_scale
            scores = base.unsqueeze(2) + log_retention_rate.unsqueeze(2) * age
        if mask is not None:
            scores = scores + mask.to(scores.dtype)
        return scores

    def expand_to_pairs(self, scores: torch.Tensor, q_len: int) -> torch.Tensor:
        (bsz, n_heads, k_len) = scores.shape
        return scores.unsqueeze(2).expand(bsz, n_heads, q_len, k_len)

    @property
    def idx_dim(self) -> int:
        return 2 * self.n_heads if self.decay else self.n_heads

    def gate_key(self, hidden_states: torch.Tensor, *, key_offset: int = 0, dtype=None) -> torch.Tensor:
        (scores, log_retention_rate) = self._score_and_decay(hidden_states, key_offset=key_offset)
        if log_retention_rate is None:
            k = scores.transpose(1, 2)
        else:
            k_len = hidden_states.shape[1]
            pos = torch.arange(key_offset, key_offset + k_len, device=scores.device, dtype=scores.dtype)
            score_intercept = scores - log_retention_rate * (pos / self.age_scale)
            k = torch.stack([score_intercept, log_retention_rate], dim=-1)
            k = k.permute(0, 2, 1, 3).reshape(scores.shape[0], k_len, 2 * self.n_heads)
        return k if dtype is None else k.to(dtype)

    def gate_query(
        self, q_len: int, bsz: int, n_kv_heads: int, *, device=None, dtype=None, query_offset: int = 0
    ) -> torch.Tensor:
        di = self.n_heads
        if not self.decay:
            if di == 1:
                return torch.ones(bsz, n_kv_heads, q_len, 1, device=device, dtype=dtype)
            eye = torch.eye(di, device=device, dtype=dtype)
            return eye.view(1, di, 1, di).expand(bsz, di, q_len, di)
        q_pos = (torch.arange(q_len, device=device, dtype=torch.float32) + float(query_offset)) / self.age_scale
        if di == 1:
            sel = torch.ones(n_kv_heads, 1, device=device, dtype=torch.float32)
        else:
            sel = torch.eye(di, device=device, dtype=torch.float32)
        qi = torch.stack(
            [sel.unsqueeze(1).expand(n_kv_heads, q_len, di), sel.unsqueeze(1) * q_pos.view(1, q_len, 1)], dim=-1
        )
        qi = qi.reshape(n_kv_heads, q_len, 2 * di).to(dtype)
        return qi.unsqueeze(0).expand(bsz, n_kv_heads, q_len, 2 * di)

    def project_q(
        self, hidden_states: torch.Tensor, cos=None, sin=None, *, n_kv_heads: int | None = None, query_offset: int = 0
    ) -> torch.Tensor:
        (bsz, q_len, _) = hidden_states.shape
        return self.gate_query(
            q_len,
            bsz,
            n_kv_heads if n_kv_heads is not None else self.n_heads,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            query_offset=query_offset,
        )

    def project_k(
        self,
        hidden_states: torch.Tensor,
        cos=None,
        sin=None,
        *,
        value_states: torch.Tensor | None = None,
        key_offset: int = 0,
    ) -> torch.Tensor:
        return self.gate_key(hidden_states, key_offset=key_offset, dtype=hidden_states.dtype)

    def extra_repr(self) -> str:
        shape = f"hidden={self.config.hidden_size}, n_heads={self.n_heads}"
        shape += f", mid_dim={self.mid_dim}" if self.mid_dim else " (linear)"
        shape += f", pos_slope={self.pos_slope:g}"
        if self.decay:
            shape += f", decay(ref={self.age_scale:g}, init={self.config.decay_init:g})"
        return f"{shape}, Di={self.idx_dim}"
