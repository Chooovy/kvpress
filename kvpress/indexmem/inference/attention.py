# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for the IndexMem++ public implementation.

from __future__ import annotations

import torch

_MAX_FUSE_EXP = 80.0


def _merge_lse(branches: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    outs = [o for (o, _) in branches]
    lses = [torch.where(torch.isfinite(l), l, torch.full_like(l, -float("inf"))) for (_, l) in branches]
    m = lses[0]
    for l in lses[1:]:
        m = torch.maximum(m, l)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    weights = [(l - m).exp().unsqueeze(-1) for l in lses]
    num = sum((o * w for (o, w) in zip(outs, weights)))
    den = sum(weights)
    return num / den.clamp(min=torch.finfo(den.dtype).tiny)


def fuse_attention(
    o_s: torch.Tensor, lse_s: torch.Tensor, n: torch.Tensor, d: torch.Tensor, *, group: int = 1
) -> torch.Tensor:
    bsz, n_q_heads, q_len, head_dim = o_s.shape
    n_kv_heads = n.shape[1]
    o_g = o_s.float().view(bsz, n_kv_heads, group, q_len, head_dim)
    lse_g = lse_s.float().view(bsz, n_kv_heads, group, q_len)
    n_g = n.float().unsqueeze(2)
    d_g = d.float().unsqueeze(2)
    scale = (-lse_g).clamp(max=_MAX_FUSE_EXP).exp().unsqueeze(-1)
    fused = (o_g + scale * n_g) / (1.0 + (scale.squeeze(-1) * d_g).unsqueeze(-1))
    return fused.view(bsz, n_q_heads, q_len, head_dim).to(o_s.dtype)
