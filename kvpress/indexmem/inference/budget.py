# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for the IndexMem++ public implementation.

from __future__ import annotations

import torch


def head_mass_curve(
    query: torch.Tensor,
    key: torch.Tensor,
    scores: torch.Tensor,
    *,
    ref_row: int,
    scaling: float,
    sink_size: int,
    window_size: int,
) -> tuple[torch.Tensor, int]:
    n_q_heads = query.shape[1]
    n_kv, k_len = (key.shape[1], key.shape[2])
    q_len = query.shape[2]
    group = n_q_heads // n_kv
    device = query.device
    query_offset = k_len - q_len
    q_row = min(max(int(ref_row) - query_offset, 0), q_len - 1)
    qr = query[0, :, q_row, :].float()
    kf = key[0].float().repeat_interleave(group, 0)
    logits = torch.einsum("hd,hsd->hs", qr, kf) * scaling
    key_idx = torch.arange(k_len, device=device)
    causal = key_idx <= ref_row
    logits = logits.masked_fill(~causal, torch.finfo(torch.float32).min)
    p_kv = torch.softmax(logits, dim=-1).view(n_kv, group, k_len).mean(1)
    sink = key_idx < sink_size
    local = (key_idx > ref_row - window_size) & (key_idx <= ref_row) & ~sink
    pool = causal & ~(sink | local)
    pinned_mass = p_kv[:, sink | local].sum(-1)
    neg = torch.finfo(torch.float32).min
    pooled = torch.where(pool.unsqueeze(0), scores.float(), torch.tensor(neg, device=device))
    order = torch.argsort(pooled, dim=-1, descending=True, stable=True)
    n_pool = int(pool.sum())
    cum = p_kv.gather(-1, order)[:, :n_pool].cumsum(-1) + pinned_mass.unsqueeze(-1)
    return (cum, n_pool)


def allocate_by_mass(cum: torch.Tensor, total: int, *, iters: int = 60) -> torch.Tensor:
    n_heads, n_pool = cum.shape
    device = cum.device
    if total <= n_heads:
        base = torch.zeros(n_heads, dtype=torch.int64, device=device)
        if total > 0:
            base[:total] = 1
        return base

    def k_for(target: float) -> torch.Tensor:
        reached = cum >= target
        first = reached.to(torch.uint8).argmax(-1) + 1
        return torch.where(reached.any(-1), first, torch.full_like(first, n_pool))

    lo, hi = (0.0, 1.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if int(k_for(mid).sum()) > total:
            hi = mid
        else:
            lo = mid
    k = k_for(lo).clamp(min=1, max=n_pool)
    residual = total - int(k.sum())
    while residual != 0:
        if residual > 0:
            gain = torch.where(
                k < n_pool,
                cum.gather(-1, k.clamp(max=n_pool - 1).unsqueeze(-1)).squeeze(-1)
                - cum.gather(-1, (k - 1).unsqueeze(-1)).squeeze(-1),
                torch.full((n_heads,), -1.0, device=device),
            )
            if float(gain.max()) < 0:
                break
            k[int(gain.argmax())] += 1
            residual -= 1
        else:
            loss = torch.where(
                k > 1,
                cum.gather(-1, (k - 1).unsqueeze(-1)).squeeze(-1)
                - cum.gather(-1, (k - 2).clamp(min=0).unsqueeze(-1)).squeeze(-1),
                torch.full((n_heads,), float("inf"), device=device),
            )
            if not bool(torch.isfinite(loss).any()):
                break
            k[int(loss.argmin())] -= 1
            residual += 1
    return k


def mass_head_budgets(
    query: torch.Tensor,
    key: torch.Tensor,
    scores: torch.Tensor,
    *,
    cache_budget: int,
    sink_size: int,
    window_size: int,
    scaling: float,
    ref_row: int | None = None,
    floor: int = 0,
) -> torch.Tensor:
    n_kv = key.shape[1]
    take = cache_budget - sink_size - window_size
    row = key.shape[2] - 1 if ref_row is None else int(ref_row)
    cum, n_pool = head_mass_curve(
        query, key, scores, ref_row=row, scaling=scaling, sink_size=sink_size, window_size=window_size
    )
    total = take * n_kv
    if n_pool <= 0 or total >= n_pool * n_kv:
        return torch.full((n_kv,), cache_budget, dtype=torch.int64, device=key.device)
    floor = max(0, min(int(floor), take, max(n_pool - 1, 0)))
    if floor:
        k = allocate_by_mass(cum[:, floor:], total - floor * n_kv) + floor
    else:
        k = allocate_by_mass(cum, total)
    return (k + sink_size + window_size).to(torch.int64)
