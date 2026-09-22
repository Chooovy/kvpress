# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for the IndexMem++ public implementation.

from __future__ import annotations

import torch

MASK_NEG = -10000.0
TOPK_SCRATCH_BUDGET = 16000000


def resolve_cache_budget(k_len: int, cache_budget: int | None, keep_ratio: float) -> int:
    if cache_budget is None:
        cache_budget = max(1, int(k_len * keep_ratio))
    return max(1, min(int(cache_budget), k_len))


def forced_support_positions(
    q_index: torch.Tensor, *, sink_size: int, window_size: int, query_offset: int, k_len: int
) -> torch.Tensor:
    device = q_index.device
    limit = (q_index + query_offset).clamp(max=k_len - 1)
    blocks = []
    if sink_size > 0:
        sink = torch.arange(sink_size, device=device).expand(q_index.shape[0], sink_size)
        blocks.append(torch.where(sink <= limit.unsqueeze(-1), sink, torch.full_like(sink, -1)))
    if window_size > 0:
        back = torch.arange(window_size - 1, -1, -1, device=device)
        local = limit.unsqueeze(-1) - back
        usable = (local >= 0) & (local >= sink_size)
        blocks.append(torch.where(usable, local, torch.full_like(local, -1)))
    if not blocks:
        return torch.zeros((q_index.shape[0], 0), dtype=torch.long, device=device)
    return torch.cat(blocks, dim=-1)


def excluded_key_mask(
    q_index: torch.Tensor, k_index: torch.Tensor, *, sink_size: int, window_size: int, query_offset: int, k_len: int
) -> torch.Tensor | None:
    if sink_size <= 0 and window_size <= 0:
        return None
    limit = (q_index + query_offset).clamp(max=k_len - 1).unsqueeze(-1)
    keys = k_index.unsqueeze(0)
    excluded = torch.zeros((q_index.shape[0], k_index.shape[0]), dtype=torch.bool, device=q_index.device)
    if sink_size > 0:
        excluded |= keys < sink_size
    if window_size > 0:
        excluded |= keys > limit - window_size
    return excluded


def causal_keep(q_index: torch.Tensor, k_index: torch.Tensor, *, query_offset: int) -> torch.Tensor:
    return k_index.unsqueeze(0) <= (q_index + query_offset).unsqueeze(-1)


def sort_support(support: torch.Tensor, k_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    filled = torch.where(support >= 0, support, torch.full_like(support, k_len))
    filled, _ = filled.sort(dim=-1)
    valid = filled < k_len
    support = torch.where(valid, filled, torch.full_like(filled, -1))
    return (support.to(torch.int32), valid)


def topk_tiles(take: int, k_len: int, q_len: int, budget: int = TOPK_SCRATCH_BUDGET) -> tuple[int, int]:
    take = max(1, int(take))
    k_len = max(int(k_len), 1)
    min_query_tile = 256
    if min_query_tile * (take + k_len) <= budget:
        key_tile = k_len
    else:
        key_tile = min(max(512, 1 << (2 * take - 1).bit_length()), k_len)
    query_tile = min(max(min_query_tile, budget // max(take + key_tile, 1)), max(q_len, 1))
    return (key_tile, query_tile)


@torch.no_grad()
def streaming_topk_support(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    cache_budget: int,
    *,
    mask: torch.Tensor | None = None,
    query_offset: int | None = None,
    sink_size: int = 0,
    window_size: int = 0,
    key_tile: int | None = None,
    query_tile: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, n_heads, q_len, _ = q_idx.shape
    k_len = k_idx.shape[1]
    device = q_idx.device
    cache_budget = max(1, min(int(cache_budget), k_len))
    if query_offset is None:
        query_offset = k_len - q_len
    n_forced = sink_size + window_size
    take = cache_budget - n_forced
    default_key_tile, default_query_tile = topk_tiles(take, k_len, q_len)
    if key_tile is None:
        key_tile = default_key_tile
    if query_tile is None:
        query_tile = default_query_tile
    support = torch.full((bsz, n_heads, q_len, cache_budget), -1, dtype=torch.int32, device=device)
    k_index_all = torch.arange(k_len, device=device)
    neg_inf = torch.tensor(-float("inf"), device=device, dtype=q_idx.dtype)
    for q_start in range(0, q_len, query_tile):
        q_stop = min(q_start + query_tile, q_len)
        dq = q_stop - q_start
        q_index = torch.arange(q_start, q_stop, device=device)
        q_view = q_idx[:, :, q_start:q_stop]
        slots = []
        if n_forced > 0:
            forced = forced_support_positions(
                q_index, sink_size=sink_size, window_size=window_size, query_offset=query_offset, k_len=k_len
            )
            forced = forced.to(torch.int32).expand(bsz, n_heads, dq, n_forced).clone()
            if mask is not None:
                keep = mask[..., q_start:q_stop, :] > MASK_NEG / 2
                keep = keep.expand(bsz, n_heads, dq, k_len)
                allowed = keep.gather(-1, forced.clamp_min(0).long())
                forced = torch.where(allowed, forced, torch.full_like(forced, -1))
            slots.append(forced)
        if take > 0:
            best_v = None
            best_i = None
            for start in range(0, k_len, key_tile):
                stop = min(start + key_tile, k_len)
                k_index = k_index_all[start:stop]
                logits = torch.einsum("bhqd,bkd->bhqk", q_view, k_idx[:, start:stop])
                if mask is not None:
                    allowed = mask[..., q_start:q_stop, start:stop] > MASK_NEG / 2
                else:
                    allowed = causal_keep(q_index, k_index, query_offset=query_offset)
                    allowed = allowed.unsqueeze(0).unsqueeze(0)
                skip = excluded_key_mask(
                    q_index,
                    k_index,
                    sink_size=sink_size,
                    window_size=window_size,
                    query_offset=query_offset,
                    k_len=k_len,
                )
                if skip is not None:
                    allowed = allowed & ~skip
                logits = torch.where(allowed, logits, neg_inf)
                if best_v is None:
                    width = min(take, logits.shape[-1])
                    best_v, order = logits.topk(width, dim=-1, sorted=False)
                    best_i = order.to(torch.int32) + start
                else:
                    prev = best_v.shape[-1]
                    cand_v = torch.cat([best_v, logits], dim=-1)
                    width = min(take, cand_v.shape[-1])
                    best_v, order = cand_v.topk(width, dim=-1, sorted=False)
                    best_i = torch.where(
                        order < prev,
                        best_i.gather(-1, order.clamp(max=max(prev - 1, 0))),
                        (order - prev).to(torch.int32) + start,
                    )
            best_i = torch.where(torch.isfinite(best_v), best_i, torch.full_like(best_i, -1))
            slots.append(best_i)
        support[:, :, q_start:q_stop] = torch.cat(slots, dim=-1)
    return sort_support(support, k_len)
