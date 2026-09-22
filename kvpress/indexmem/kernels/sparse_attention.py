# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for the IndexMem++ public implementation.

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_gqa_attn_fwd(
    Q,
    K,
    V,
    IDX,
    OUT,
    LSE,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ib,
    stride_ih,
    stride_im,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lm,
    k_len_static,
    query_offset_static,
    scale,
    n_kv_heads,
    topk,
    GROUP: tl.constexpr,
    BLOCK_G: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CAUSAL: tl.constexpr,
    PRECISION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // n_kv_heads
    off_kvh = pid_bh % n_kv_heads
    k_start = 0
    k_len = k_len_static
    query_offset = query_offset_static
    q_local = pid_m
    offs_g = tl.arange(0, BLOCK_G)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    g_valid = offs_g < GROUP
    d_valid = offs_d < D
    dv_valid = offs_dv < DV
    head = off_kvh * GROUP + offs_g
    q_ptrs = Q + off_b * stride_qb + head[:, None] * stride_qh + pid_m * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=g_valid[:, None] & d_valid[None, :], other=0.0).to(tl.float32)
    run_max = tl.full([BLOCK_G], float("-inf"), dtype=tl.float32)
    run_sum = tl.zeros([BLOCK_G], dtype=tl.float32)
    acc = tl.zeros([BLOCK_G, BLOCK_DV], dtype=tl.float32)
    last_key = q_local + query_offset
    idx_base = IDX + off_b * stride_ib + off_kvh * stride_ih + pid_m * stride_im
    for start_k in range(0, topk, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        slot_valid = offs_k < topk
        idx = tl.load(idx_base + offs_k, mask=slot_valid, other=-1).to(tl.int32)
        valid = slot_valid & (idx >= 0) & (idx < k_len)
        if CAUSAL:
            valid = valid & (idx <= last_key)
        row = tl.cast(k_start, tl.int64) + tl.cast(tl.where(valid, idx, 0), tl.int64)
        k_ptrs = K + off_b * stride_kb + off_kvh * stride_kh + row[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k_tile = tl.load(k_ptrs, mask=valid[:, None] & d_valid[None, :], other=0.0).to(tl.float32)
        logits = tl.dot(q, tl.trans(k_tile), input_precision=PRECISION) * scale
        logits = tl.where(valid[None, :], logits, float("-inf"))
        new_max = tl.maximum(run_max, tl.max(logits, 1))
        safe_max = tl.where(new_max == float("-inf"), 0.0, new_max)
        rescale = tl.where(run_max == float("-inf"), 0.0, tl.exp(run_max - safe_max))
        p = tl.exp(logits - safe_max[:, None])
        v_ptrs = V + off_b * stride_vb + off_kvh * stride_vh + row[:, None] * stride_vn + offs_dv[None, :] * stride_vd
        v_tile = tl.load(v_ptrs, mask=valid[:, None] & dv_valid[None, :], other=0.0).to(tl.float32)
        run_sum = run_sum * rescale + tl.sum(p, 1)
        acc = acc * rescale[:, None] + tl.dot(p.to(v_tile.dtype), v_tile, input_precision=PRECISION)
        run_max = new_max
    alive = run_sum > 0.0
    out = tl.where(alive[:, None], acc / tl.where(alive[:, None], run_sum[:, None], 1.0), 0.0)
    lse = tl.where(alive, tl.log(tl.where(alive, run_sum, 1.0)) + run_max, float("-inf"))
    out_ptrs = OUT + off_b * stride_ob + head[:, None] * stride_oh + pid_m * stride_om + offs_dv[None, :] * stride_od
    tl.store(out_ptrs, out.to(OUT.dtype.element_ty), mask=g_valid[:, None] & dv_valid[None, :])
    tl.store(LSE + off_b * stride_lb + head * stride_lh + pid_m * stride_lm, lse, mask=g_valid)


@functools.lru_cache(maxsize=1)
def min_dot_m():
    from triton.compiler.compiler import make_backend
    from triton.runtime import driver

    backend = make_backend(driver.active.get_current_target())
    minimum = backend.get_codegen_implementation(backend.parse_options({}))["min_dot_size"]
    operand = tl.block_type(tl.float32, [16, 16])
    return int(minimum(operand, operand)[0])


def sparse_gqa_attention(
    query,
    key,
    value,
    support,
    *,
    scaling=None,
    query_offset=None,
    causal=True,
    block_k=64,
    precision="tf32",
    num_warps=4,
):
    batch, query_heads, query_length, head_dim = query.shape
    kv_heads, key_length = key.shape[1:3]
    group = query_heads // kv_heads
    value_dim = value.shape[-1]
    output = torch.empty((batch, query_heads, query_length, value_dim), dtype=query.dtype, device=query.device)
    logsumexp = torch.empty((batch, query_heads, query_length), dtype=torch.float32, device=query.device)
    support = support.contiguous()
    offset = key_length - query_length if query_offset is None else query_offset
    scale = head_dim**-0.5 if scaling is None else scaling
    _sparse_gqa_attn_fwd[query_length, batch * kv_heads](
        query,
        key,
        value,
        support,
        output,
        logsumexp,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *support.stride()[:-1],
        *output.stride(),
        *logsumexp.stride(),
        key_length,
        offset,
        scale,
        kv_heads,
        support.shape[-1],
        GROUP=group,
        BLOCK_G=triton.next_power_of_2(max(group, min_dot_m())),
        D=head_dim,
        DV=value_dim,
        BLOCK_D=triton.next_power_of_2(max(16, head_dim)),
        BLOCK_DV=triton.next_power_of_2(max(16, value_dim)),
        BLOCK_K=block_k,
        CAUSAL=causal,
        PRECISION=precision,
        num_warps=num_warps,
    )
    return output, logsumexp
