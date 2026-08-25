# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DMA's query-independent value-vector scorer."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

from kvpress.presses.gqa_indexer.indexer import MASK_NEG


@dataclass
class DMAIndexerConfig:
    """KV-head count and value-head width consumed by :class:`DMAIndexer`."""

    n_heads: int
    head_dim: int
    rope_dim: int = field(default=0, init=False)

    def __post_init__(self):
        for name in ("n_heads", "head_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")


class DMAIndexer(nn.Module):
    """Score cached tokens from their value vectors with DMA's sampling formula.

    For values shaped ``(B, Hkv, K, D)``, all KV heads at a token are concatenated before
    ``dt_proj`` maps ``Hkv * D`` channels back to one score per KV head. The returned score is
    ``exp(A * softplus(V Delta))`` with shape ``(B, Hkv, K)``.
    """

    is_query_independent = True

    def __init__(self, config: DMAIndexerConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.rope_dim = config.rope_dim

        self.dt_proj = nn.Linear(config.n_heads * config.head_dim, config.n_heads, bias=False)
        self.A = nn.Parameter(torch.empty(config.n_heads))
        nn.init.normal_(self.A, mean=0.0, std=0.02)

    def score_values(self, values: torch.Tensor, *, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply ``exp(A * softplus(V Delta))`` -> ``(B, Hkv, K)`` in fp32."""
        if values.dim() != 4:
            raise ValueError(f"values must be (B, Hkv, K, D), got {tuple(values.shape)}")
        if values.shape[1] != self.n_heads or values.shape[-1] != self.head_dim:
            raise ValueError(f"values must have Hkv={self.n_heads} and D={self.head_dim}, got {tuple(values.shape)}")

        bsz, _, k_len, _ = values.shape
        flat_values = values.transpose(1, 2).reshape(bsz, k_len, -1)
        sampled = F.linear(flat_values.float(), self.dt_proj.weight.float(), bias=None)
        scores = torch.exp(self.A.float() * F.softplus(sampled)).transpose(1, 2)

        if mask is not None:
            keep = mask if mask.dtype == torch.bool else mask != 0
            scores = scores.masked_fill(~keep.view(keep.shape[0], 1, -1), MASK_NEG)
        return scores

    def gate_key(self, values: torch.Tensor) -> torch.Tensor:
        """Return the fp32 per-key DMA score as ``(B, K, Hkv)`` for the gate path."""
        return self.score_values(values).transpose(1, 2)

    def gate_query(self, q_len: int, bsz: int, n_kv_heads: int, *, device=None, dtype=None) -> torch.Tensor:
        """Return the constant one-hot selector ``(B, Hkv, Q, Hkv)``."""
        if n_kv_heads != self.n_heads:
            raise ValueError(f"DMAIndexer has {self.n_heads} score heads but the model has {n_kv_heads} KV heads")
        eye = torch.eye(self.n_heads, device=device, dtype=dtype)
        return eye.view(1, self.n_heads, 1, self.n_heads).expand(bsz, self.n_heads, q_len, self.n_heads)

    def project_q(
        self, hidden_states: torch.Tensor, cos=None, sin=None, *, n_kv_heads: int | None = None
    ) -> torch.Tensor:
        """Project the query side to DMA's constant per-KV-head selector."""
        self._reject_rope(cos, sin)
        bsz, q_len, _ = hidden_states.shape
        return self.gate_query(
            q_len,
            bsz,
            n_kv_heads if n_kv_heads is not None else self.n_heads,
            device=hidden_states.device,
            dtype=torch.float32,
        )

    def project_k(
        self,
        hidden_states: torch.Tensor,
        cos=None,
        sin=None,
        *,
        value_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project actual cached values to DMA gate keys ``(B, K, Hkv)``."""
        self._reject_rope(cos, sin)
        if value_states is None:
            raise ValueError("DMAIndexer.project_k requires value_states")
        return self.gate_key(value_states)

    def require_gate_scale(self) -> torch.Tensor:
        """DMA's formula already contains ``A``; its outer gate multiplier is fixed at one."""
        return torch.ones((), device=self.A.device, dtype=torch.float32)

    @staticmethod
    def _reject_rope(cos, sin) -> None:
        if cos is not None or sin is not None:
            raise ValueError("DMAIndexer scores values directly and does not use RoPE")

    def extra_repr(self) -> str:
        return f"n_heads={self.n_heads}, head_dim={self.head_dim}"
