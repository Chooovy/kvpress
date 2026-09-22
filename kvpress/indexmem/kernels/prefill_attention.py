# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for the IndexMem++ public implementation.

from __future__ import annotations

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

FLEX_BLOCK = 128
_flex_compiled = None
_block_mask_compiled = None
_BLOCK_MASK_DYNAMIC = True


def _flex():
    global _flex_compiled
    if _flex_compiled is None:
        _flex_compiled = torch.compile(flex_attention, dynamic=None)
    return _flex_compiled


def _block_mask():
    global _block_mask_compiled
    if _block_mask_compiled is None:
        _block_mask_compiled = torch.compile(create_block_mask, dynamic=_BLOCK_MASK_DYNAMIC)
    return _block_mask_compiled


def deadlines(
    scores: torch.Tensor, cache_budget: int | torch.Tensor, *, sink_size: int = 0, window_size: int = 0
) -> torch.Tensor:
    n_heads, k_len = scores.shape
    device = scores.device
    if isinstance(cache_budget, torch.Tensor):
        take_h = cache_budget.to(device=device, dtype=torch.int64).reshape(n_heads) - int(sink_size) - int(window_size)
    else:
        take_h = torch.full(
            (n_heads,), int(cache_budget) - int(sink_size) - int(window_size), dtype=torch.int64, device=device
        )
    if k_len == 0 or int(take_h.max()) <= 0:
        return torch.full((n_heads, k_len), -1, dtype=torch.int32, device=device)
    empty_head = take_h <= 0
    take_h = take_h.clamp(min=1)
    key_idx = torch.arange(k_len, device=device)
    in_pool = key_idx >= sink_size
    neg = torch.tensor(-float("inf"), device=device, dtype=scores.dtype)
    pooled = torch.where(in_pool, scores, neg)
    order = torch.argsort(pooled, dim=-1, descending=True, stable=True)
    BS = 64
    n_blocks = (k_len + BS - 1) // BS
    blk_of_rank = torch.div(order, BS, rounding_mode="floor").clamp(max=n_blocks - 1)
    block = torch.empty((n_heads, k_len), dtype=torch.int64, device=device)
    unreached = torch.empty((n_heads, k_len), dtype=torch.bool, device=device)
    need = torch.empty((n_heads, k_len), dtype=torch.int64, device=device)
    carry = torch.zeros((n_heads, 1, n_blocks), dtype=torch.int32, device=device)
    rank_chunk_size = 1024
    for start in range(0, k_len, rank_chunk_size):
        end = min(start + rank_chunk_size, k_len)
        chunk_blocks = blk_of_rank[:, start:end].unsqueeze(-1)
        hist = torch.zeros((n_heads, end - start, n_blocks), dtype=torch.int16, device=device)
        hist.scatter_(2, chunk_blocks, torch.ones_like(chunk_blocks, dtype=torch.int16))
        inclusive = hist.cumsum(1, dtype=torch.int32)
        del hist
        inclusive.add_(carry)
        cum_rank = torch.cat([carry, inclusive[:, :-1]], dim=1)
        carry = inclusive[:, -1:].clone()
        del inclusive
        prefix_blk = cum_rank.cumsum(2)
        del cum_rank
        reached = prefix_blk >= take_h.view(n_heads, 1, 1)
        chunk_block = reached.to(torch.uint8).argmax(2)
        block[:, start:end] = chunk_block
        unreached[:, start:end] = ~reached.any(2)
        before = torch.where(
            chunk_block > 0,
            prefix_blk.gather(2, (chunk_block - 1).clamp(min=0).unsqueeze(-1)).squeeze(-1),
            torch.zeros_like(chunk_block),
        )
        need[:, start:end] = take_h.view(n_heads, 1) - before
        del prefix_blk, reached, chunk_block, before, chunk_blocks
    del carry
    arrival = torch.empty_like(order)
    arrival.scatter_(-1, order, torch.arange(k_len, device=device).expand_as(order))
    ranks = torch.arange(k_len, device=device)
    offsets = torch.arange(BS, device=device)
    threshold = torch.empty((n_heads, k_len), dtype=torch.int64, device=device)
    for h in range(n_heads):
        keys = block[h].unsqueeze(-1) * BS + offsets.view(1, BS)
        valid = keys < k_len
        safe = keys.clamp(max=k_len - 1)
        alive = (arrival[h][safe] < ranks.unsqueeze(-1)) & in_pool[safe] & valid
        hit = alive.cumsum(-1) >= need[h].unsqueeze(-1)
        pos = hit.to(torch.uint8).argmax(-1)
        found = safe.gather(-1, pos.unsqueeze(-1)).squeeze(-1)
        threshold[h] = torch.where(hit.any(-1), found, torch.full_like(found, k_len + 1))
    threshold = torch.where(unreached, torch.full_like(threshold, k_len + 1), threshold)
    per_rank = torch.where(threshold > k_len, torch.full_like(threshold, k_len - 1), (threshold - 1).clamp(min=-1))
    out = torch.empty((n_heads, k_len), dtype=torch.int64, device=device)
    out.scatter_(-1, order, per_rank)
    out = torch.where(empty_head.view(n_heads, 1), torch.full_like(out, -1), out)
    return torch.where(in_pool.view(1, -1), out, torch.full_like(out, k_len - 1)).to(torch.int32)


def qi_block_mask(
    deadline: torch.Tensor,
    *,
    q_len: int,
    k_len: int,
    n_q_heads: int,
    sink_size: int,
    window_size: int,
    query_offset: int | None = None,
    device: torch.device | None = None,
):
    n_kv_heads = deadline.shape[0]
    group = n_q_heads // n_kv_heads
    offset = k_len - q_len if query_offset is None else int(query_offset)
    device = device or deadline.device
    fs, fl = (int(sink_size), int(window_size))
    last_key = k_len - 1

    def mask_mod(b, h, q_i, k_j):
        limit = torch.clamp(q_i + offset, max=last_key)
        horizon = limit - fl
        sink = k_j < fs
        local = (k_j > limit - fl) & (k_j >= fs)
        alive = horizon <= deadline[h // group, k_j]
        chosen = (k_j >= fs) & (k_j <= horizon) & alive
        return (k_j <= limit) & (sink | local | chosen)

    return _block_mask()(mask_mod, B=None, H=n_q_heads, Q_LEN=q_len, KV_LEN=k_len, device=device)


def qi_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scores: torch.Tensor,
    cache_budget: int,
    *,
    sink_size: int = 0,
    window_size: int = 0,
    scaling: float | None = None,
    query_offset: int | None = None,
) -> torch.Tensor:
    bsz, n_q_heads, q_len, _ = query.shape
    n_kv_heads, k_len = (key.shape[1], key.shape[2])
    group = n_q_heads // n_kv_heads
    dl = deadlines(scores[0].float(), cache_budget, sink_size=sink_size, window_size=window_size)
    block_mask = qi_block_mask(
        dl,
        q_len=q_len,
        k_len=k_len,
        n_q_heads=n_q_heads,
        sink_size=sink_size,
        window_size=window_size,
        query_offset=query_offset,
        device=query.device,
    )
    return _flex()(
        query,
        key.repeat_interleave(group, dim=1),
        value.repeat_interleave(group, dim=1),
        block_mask=block_mask,
        scale=scaling,
    )
