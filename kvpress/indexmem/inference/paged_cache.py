# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for the IndexMem++ public implementation.

from __future__ import annotations

import torch
from transformers import Cache, CacheLayerMixin

from kvpress.indexmem.inference.attention import _merge_lse

PAGE_BLOCK = 256
POS_BITS = 21
MAX_POSITION = (1 << POS_BITS) - 1


def rank_key(score: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    bits = score.float().contiguous().view(torch.int32).to(torch.int64)
    mono = torch.where(bits >= 0, bits, -(bits & 2147483647))
    return (mono << POS_BITS) - pos.to(torch.int64)


class PagedKVPool:

    def __init__(
        self,
        budgets: torch.Tensor,
        *,
        batch_size: int,
        n_layers: int,
        n_kv_heads: int,
        n_sink: int,
        n_local: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        age_scale: float | None = None,
    ):
        self.budgets = budgets.to(device=device, dtype=torch.int64)
        self.batch_size = int(batch_size)
        self.n_layers = int(n_layers)
        self.n_kv_heads = int(n_kv_heads)
        self.n_sink = int(n_sink)
        self.n_local = int(n_local)
        self.head_dim = int(head_dim)
        self.device = device
        self.dtype = dtype
        self.age_scale = age_scale
        rows = n_layers * batch_size * n_kv_heads
        self.rows = rows
        self.row_budget = (
            self.budgets.view(n_layers, 1, n_kv_heads)
            .expand(n_layers, batch_size, n_kv_heads)
            .reshape(rows)
            .contiguous()
        )
        blocks_per_row = (self.row_budget + PAGE_BLOCK - 1) // PAGE_BLOCK
        self.max_blocks = int(blocks_per_row.max())
        n_blocks = int(blocks_per_row.sum())
        self.k_pool = torch.zeros((n_blocks, PAGE_BLOCK, 1, head_dim), device=device, dtype=dtype)
        self.v_pool = torch.zeros_like(self.k_pool)
        self.block_table = torch.zeros((rows, self.max_blocks), device=device, dtype=torch.int32)
        starts = torch.cumsum(blocks_per_row, 0) - blocks_per_row
        slot_ax = torch.arange(self.max_blocks, device=device)
        valid = slot_ax.view(1, -1) < blocks_per_row.view(-1, 1)
        self.block_table = torch.where(
            valid, (starts.view(-1, 1) + slot_ax.view(1, -1)).to(torch.int32), torch.zeros_like(self.block_table)
        )
        self.filled = torch.zeros(rows, device=device, dtype=torch.int32)
        self.take = (self.row_budget - n_sink - n_local).clamp(min=0)
        self.width = int(self.take.max())
        self.pool_pos = torch.full((rows, self.width), -1, device=device, dtype=torch.int64)
        self.pool_key = torch.full((rows, self.width), torch.iinfo(torch.int64).min, device=device, dtype=torch.int64)
        self._pad = torch.arange(self.width, device=device).view(1, -1) >= self.take.view(-1, 1)
        self.pool_key.masked_fill_(self._pad, torch.iinfo(torch.int64).max)
        self.pool_score_intercept = torch.zeros((rows, self.width), device=device, dtype=torch.float32)
        self.pool_log_retention_rate = (
            torch.zeros((rows, self.width), device=device, dtype=torch.float32) if age_scale is not None else None
        )
        self.ring_slot = torch.zeros((rows, n_local), device=device, dtype=torch.int64)
        self.ring_pos = torch.full((rows, n_local), -1, device=device, dtype=torch.int64)
        self.ring_score_intercept = torch.zeros((rows, n_local), device=device, dtype=torch.float32)
        self.ring_log_retention_rate = (
            torch.zeros((rows, n_local), device=device, dtype=torch.float32) if age_scale is not None else None
        )
        self.ring_head = torch.zeros(rows, device=device, dtype=torch.int64)
        self.ring_count = torch.zeros(rows, device=device, dtype=torch.int64)
        self.pool_live = torch.zeros(rows, device=device, dtype=torch.int64)
        self.seen = torch.zeros(batch_size, device=device, dtype=torch.int64)

    def rows_for(self, layer_idx: int, seq: int | None = None) -> slice:
        base = layer_idx * self.batch_size * self.n_kv_heads
        if seq is None:
            return slice(base, base + self.batch_size * self.n_kv_heads)
        start = base + seq * self.n_kv_heads
        return slice(start, start + self.n_kv_heads)

    @torch.no_grad()
    def replicate_seq(self, src: int, dst: int) -> None:
        if src == dst:
            return
        for layer_idx in range(self.n_layers):
            (s, d) = (self.rows_for(layer_idx, src), self.rows_for(layer_idx, dst))
            for offset in range(self.n_kv_heads):
                (sr, dr) = (s.start + offset, d.start + offset)
                n = int(self.filled[sr])
                if n:
                    slots = torch.arange(n, device=self.device)
                    sb = self.block_table[sr, slots // PAGE_BLOCK].to(torch.int64)
                    db = self.block_table[dr, slots // PAGE_BLOCK].to(torch.int64)
                    self.k_pool[db, slots % PAGE_BLOCK, 0, :] = self.k_pool[sb, slots % PAGE_BLOCK, 0, :]
                    self.v_pool[db, slots % PAGE_BLOCK, 0, :] = self.v_pool[sb, slots % PAGE_BLOCK, 0, :]
            for buf in (
                self.filled,
                self.pool_pos,
                self.pool_key,
                self.pool_score_intercept,
                self.pool_live,
                self.ring_slot,
                self.ring_pos,
                self.ring_score_intercept,
                self.ring_head,
                self.ring_count,
            ):
                buf[d] = buf[s]
            if self.pool_log_retention_rate is not None:
                self.pool_log_retention_rate[d] = self.pool_log_retention_rate[s]
                self.ring_log_retention_rate[d] = self.ring_log_retention_rate[s]
        self.seen[dst] = self.seen[src]

    def _write_kv(self, rows: torch.Tensor, slots: torch.Tensor, key, value) -> None:
        block = self.block_table[rows, slots // PAGE_BLOCK].to(torch.int64)
        self.k_pool[block, slots % PAGE_BLOCK, 0, :] = key.to(self.dtype)
        self.v_pool[block, slots % PAGE_BLOCK, 0, :] = value.to(self.dtype)

    @torch.no_grad()
    def commit(
        self,
        layer_idx: int,
        seq: int,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        score_intercept: torch.Tensor,
        log_retention_rate: torch.Tensor | None,
        k_len: int,
    ) -> None:
        from kvpress.indexmem.kernels.prefill_attention import deadlines

        rows = self.rows_for(layer_idx, seq)
        budget = self.row_budget[rows]
        (n_local, n_sink) = (self.n_local, self.n_sink)
        if k_len <= int(budget.min()):
            self._commit_whole(
                layer_idx,
                seq,
                key=key,
                value=value,
                score_intercept=score_intercept,
                log_retention_rate=log_retention_rate,
                k_len=k_len,
            )
            return
        scores = (
            score_intercept
            if log_retention_rate is None
            else score_intercept + log_retention_rate * (float(k_len - 1) / self.age_scale)
        )
        dl = deadlines(scores, budget, sink_size=n_sink, window_size=n_local)
        horizon = max(k_len - 1 - n_local, 0)
        pos_ax = torch.arange(k_len, device=self.device)
        in_pool = (pos_ax >= n_sink) & (pos_ax <= horizon)
        keep_pool = in_pool.view(1, -1) & (horizon <= dl.to(torch.int64))
        is_sink = pos_ax < n_sink
        is_local = pos_ax > horizon
        take = self.take[rows]
        flat_row = torch.arange(rows.start, rows.stop, device=self.device)
        for h in range(self.n_kv_heads):
            row = int(flat_row[h])
            n_s = int(is_sink.sum())
            if n_s:
                slots = torch.arange(n_s, device=self.device)
                self._write_kv(torch.full_like(slots, row), slots, key[h, :n_s], value[h, :n_s])
            sel = keep_pool[h].nonzero().flatten()
            n_p = int(sel.numel())
            if n_p:
                slots = torch.arange(n_s, n_s + n_p, device=self.device)
                self._write_kv(torch.full_like(slots, row), slots, key[h, sel], value[h, sel])
                pool_slot = torch.arange(n_p, device=self.device)
                self.pool_pos[row, pool_slot] = sel
                self.pool_score_intercept[row, pool_slot] = score_intercept[h, sel]
                if self.pool_log_retention_rate is not None:
                    self.pool_log_retention_rate[row, pool_slot] = log_retention_rate[h, sel]
                self.pool_key[row, pool_slot] = rank_key(scores[h, sel], sel)
            self.pool_live[row] = n_p
            loc = is_local.nonzero().flatten()
            if loc.numel():
                base = int(budget[h]) - n_local
                n_l = int(loc.numel())
                slots = base + torch.arange(n_l, device=self.device)
                self._write_kv(torch.full_like(slots, row), slots, key[h, loc], value[h, loc])
                self.ring_slot[row, :n_l] = slots
                self.ring_pos[row, :n_l] = loc
                self.ring_score_intercept[row, :n_l] = score_intercept[h, loc]
                if self.ring_log_retention_rate is not None:
                    self.ring_log_retention_rate[row, :n_l] = log_retention_rate[h, loc]
                self.ring_head[row] = 0
                self.ring_count[row] = n_l
            self.filled[row] = int(budget[h])
        self.seen[seq] = k_len

    @torch.no_grad()
    def _commit_whole(self, layer_idx, seq, *, key, value, score_intercept, log_retention_rate, k_len) -> None:
        rows = self.rows_for(layer_idx, seq)
        flat_row = torch.arange(rows.start, rows.stop, device=self.device)
        (n_sink, n_local) = (self.n_sink, self.n_local)
        slots = torch.arange(k_len, device=self.device)
        for h in range(self.n_kv_heads):
            row = int(flat_row[h])
            self._write_kv(torch.full_like(slots, row), slots, key[h], value[h])
            n_pool = max(k_len - n_local - n_sink, 0)
            if n_pool:
                keys = torch.arange(n_sink, n_sink + n_pool, device=self.device)
                self.pool_pos[row, :n_pool] = keys
                self.pool_score_intercept[row, :n_pool] = score_intercept[h, keys]
                if self.pool_log_retention_rate is not None:
                    self.pool_log_retention_rate[row, :n_pool] = log_retention_rate[h, keys]
                sc = (
                    score_intercept[h, keys]
                    if log_retention_rate is None
                    else score_intercept[h, keys] + log_retention_rate[h, keys] * (float(k_len - 1) / self.age_scale)
                )
                self.pool_key[row, :n_pool] = rank_key(sc, keys)
            self.pool_live[row] = n_pool
            n_l = max(min(k_len - n_sink, n_local), 0)
            if n_l:
                w = torch.arange(k_len - n_l, k_len, device=self.device)
                self.ring_slot[row, :n_l] = w
                self.ring_pos[row, :n_l] = w
                self.ring_score_intercept[row, :n_l] = score_intercept[h, w]
                if self.ring_log_retention_rate is not None:
                    self.ring_log_retention_rate[row, :n_l] = log_retention_rate[h, w]
                self.ring_head[row] = 0
                self.ring_count[row] = n_l
            self.filled[row] = k_len
        self.seen[seq] = k_len

    @torch.no_grad()
    def ingest(
        self,
        layer_idx: int | list[int],
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        score_intercept: torch.Tensor,
        log_retention_rate: torch.Tensor | None,
        positions: torch.Tensor,
        seqs: torch.Tensor | None = None,
    ) -> None:
        (n_kv, n_local) = (self.n_kv_heads, self.n_local)
        if seqs is None:
            seqs = torch.arange(self.batch_size, device=self.device)
        seqs = seqs.to(self.device)
        layers = [layer_idx] if isinstance(layer_idx, int) else list(layer_idx)
        lay = torch.as_tensor(layers, device=self.device, dtype=torch.int64)
        rows = (
            lay.view(-1, 1, 1) * (self.batch_size * n_kv)
            + seqs.view(1, -1, 1) * n_kv
            + torch.arange(n_kv, device=self.device).view(1, 1, -1)
        ).reshape(-1)
        pos = positions.to(self.device).view(1, -1, 1).expand(len(layers), seqs.numel(), n_kv).reshape(-1)
        k_flat = key.reshape(-1, self.head_dim)
        v_flat = value.reshape(-1, self.head_dim)
        score_intercept_flat = score_intercept.reshape(-1).float()
        log_retention_rate_flat = None if log_retention_rate is None else log_retention_rate.reshape(-1).float()
        budget = self.row_budget[rows]
        filled = self.filled[rows].to(torch.int64)
        head = self.ring_head[rows]
        count = self.ring_count[rows]
        rolls = count >= n_local
        demoted_slot = self.ring_slot[rows, head]
        demoted_pos = self.ring_pos[rows, head]
        demoted_score_intercept = self.ring_score_intercept[rows, head]
        demoted_log_retention_rate = (
            None if self.ring_log_retention_rate is None else self.ring_log_retention_rate[rows, head]
        )
        growing = filled < budget
        arrive_slot = torch.where(growing, filled, demoted_slot)
        d_score = demoted_score_intercept
        if demoted_log_retention_rate is not None:
            d_score = d_score + demoted_log_retention_rate * (pos.to(torch.float32) / self.age_scale)
        d_key = rank_key(d_score, demoted_pos.clamp(min=0))
        live_n = self.pool_live[rows]
        take_r = self.take[rows]
        has_free = live_n < take_r
        first_free = live_n.clamp(max=max(self.width - 1, 0))
        pool_key_r = self.pool_key[rows]
        if self.pool_log_retention_rate is not None:
            live = self.pool_pos[rows] >= 0
            resc = self.pool_score_intercept[rows] + self.pool_log_retention_rate[rows] * (
                pos.to(torch.float32).view(-1, 1) / self.age_scale
            )
            pool_key_r = torch.where(
                live & ~self._pad[rows], rank_key(resc, self.pool_pos[rows].clamp(min=0)), pool_key_r
            )
        j_min = pool_key_r.argmin(-1)
        k_min = pool_key_r.gather(-1, j_min.unsqueeze(-1)).squeeze(-1)
        promote = rolls & (demoted_pos >= self.n_sink) & (has_free | (d_key > k_min))
        target = torch.where(has_free, first_free, j_min)
        dst = self.n_sink + target
        loser_slot = torch.where(promote, dst, demoted_slot.clamp(min=0))
        lost = rolls & (demoted_pos >= self.n_sink) & ~(promote & has_free)
        lb = self.block_table[rows, loser_slot // PAGE_BLOCK].to(torch.int64)
        dropped_k = self.k_pool[lb, loser_slot % PAGE_BLOCK, 0, :].clone()
        dropped_v = self.v_pool[lb, loser_slot % PAGE_BLOCK, 0, :].clone()
        src = torch.where(promote, demoted_slot.clamp(min=0), dst)
        self._copy_slot(rows, src, dst)
        self.pool_pos[rows, target] = torch.where(promote, demoted_pos, self.pool_pos[rows, target])
        self.pool_score_intercept[rows, target] = torch.where(
            promote, demoted_score_intercept, self.pool_score_intercept[rows, target]
        )
        if self.pool_log_retention_rate is not None:
            self.pool_log_retention_rate[rows, target] = torch.where(
                promote, demoted_log_retention_rate, self.pool_log_retention_rate[rows, target]
            )
        self.pool_key[rows, target] = torch.where(promote, d_key, self.pool_key[rows, target])
        self.pool_live[rows] = live_n + (promote & has_free).to(torch.int64)
        self._write_kv(rows, arrive_slot, k_flat, v_flat)
        ring_idx = torch.where(rolls, head, (head + count) % n_local)
        self.ring_slot[rows, ring_idx] = arrive_slot
        self.ring_pos[rows, ring_idx] = pos
        self.ring_score_intercept[rows, ring_idx] = score_intercept_flat
        if self.ring_log_retention_rate is not None:
            self.ring_log_retention_rate[rows, ring_idx] = log_retention_rate_flat
        self.ring_head[rows] = torch.where(rolls, (head + 1) % n_local, head)
        self.ring_count[rows] = torch.where(rolls, count, count + 1)
        self.filled[rows] = torch.where(growing, filled + 1, filled).to(self.filled.dtype)
        return {"rows": rows, "key": dropped_k, "value": dropped_v, "evicted": lost}

    def _copy_slot(self, rows: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> None:
        sb = self.block_table[rows, src // PAGE_BLOCK].to(torch.int64)
        db = self.block_table[rows, dst // PAGE_BLOCK].to(torch.int64)
        self.k_pool[db, dst % PAGE_BLOCK, 0, :] = self.k_pool[sb, src % PAGE_BLOCK, 0, :]
        self.v_pool[db, dst % PAGE_BLOCK, 0, :] = self.v_pool[sb, src % PAGE_BLOCK, 0, :]

    def attend(
        self,
        layer_idx: int,
        query: torch.Tensor,
        *,
        seqs: torch.Tensor | None = None,
        scaling: float | None = None,
        new_key: torch.Tensor | None,
        new_value: torch.Tensor | None,
        extra: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        from flash_attn import flash_attn_func, flash_attn_with_kvcache

        (bsz, n_q_heads, q_len, head_dim) = query.shape
        n_kv = self.n_kv_heads
        group = n_q_heads // n_kv
        if seqs is None:
            seqs = torch.arange(bsz, device=self.device)
        rows = (
            layer_idx * self.batch_size * n_kv
            + seqs.view(-1, 1).to(self.device) * n_kv
            + torch.arange(n_kv, device=self.device).view(1, -1)
        ).reshape(-1)
        q_flash = (
            query.view(bsz, n_kv, group, q_len, head_dim)
            .permute(0, 1, 3, 2, 4)
            .reshape(bsz * n_kv, q_len, group, head_dim)
        )
        (out, lse) = flash_attn_with_kvcache(
            q_flash.to(self.dtype),
            self.k_pool,
            self.v_pool,
            cache_seqlens=self.filled[rows],
            block_table=self.block_table[rows],
            softmax_scale=scaling,
            causal=False,
            return_softmax_lse=True,
        )
        o_cache = (
            out.view(bsz, n_kv, q_len, group, head_dim)
            .permute(0, 1, 3, 2, 4)
            .reshape(bsz, n_q_heads, q_len, head_dim)
            .float()
        )
        empty_kv = (self.filled[rows] == 0).view(bsz, n_kv)
        l_cache = lse.view(bsz, n_kv, group, q_len).reshape(bsz, n_q_heads, q_len).float()
        if bool(empty_kv.any()):
            mask_q = empty_kv.repeat_interleave(group, 1).view(bsz, n_q_heads, 1)
            l_cache = torch.where(mask_q, torch.full_like(l_cache, -float("inf")), l_cache)
            o_cache = torch.where(mask_q.unsqueeze(-1), torch.zeros_like(o_cache), o_cache)
        (o_new, l_new) = flash_attn_func(
            query.transpose(1, 2).to(self.dtype),
            new_key.transpose(1, 2).to(self.dtype),
            new_value.transpose(1, 2).to(self.dtype),
            causal=True,
            softmax_scale=scaling,
            return_attn_probs=True,
        )[:2]
        o_new = o_new.permute(0, 2, 1, 3).float()
        l_new = l_new.float()
        if extra is not None:
            (o_x, l_x) = extra
            merged = _merge_lse([(o_cache, l_cache), (o_new, l_new), (o_x, l_x)])
            return merged.transpose(1, 2).contiguous().to(query.dtype)
        merged = _merge_lse([(o_cache, l_cache), (o_new, l_new)])
        return merged.transpose(1, 2).contiguous().to(query.dtype)

    def memory_bytes(self) -> int:
        return self.k_pool.numel() * self.k_pool.element_size() * 2


class PagedKVCacheLayer(CacheLayerMixin):
    is_sliding = False

    def __init__(self, pool: PagedKVPool, layer_idx: int):
        super().__init__()
        self.pool = pool
        self.layer_idx = layer_idx

    def lazy_initialization(self, key_states: torch.Tensor):
        (self.dtype, self.device) = (key_states.dtype, key_states.device)

    def update(self, key_states, value_states, cache_kwargs=None):
        return (key_states, value_states)

    def get_seq_length(self) -> int:
        return int(self.pool.seen.max())

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        return (int(self.pool.filled.max()) + cache_position.shape[0], 0)

    def get_max_cache_shape(self) -> int:
        return int(self.pool.row_budget.max())

    def reset(self) -> None:
        raise NotImplementedError("Create a new pool for a new context.")

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        raise NotImplementedError("Use greedy or sampling decoding.")


class PagedKVCache(Cache):

    def __init__(self, pool: PagedKVPool):
        super().__init__(layers=[PagedKVCacheLayer(pool, i) for i in range(pool.n_layers)])
        self.pool = pool

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        return int(self.pool.seen.max())
