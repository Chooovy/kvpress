# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


def block_pow2(value):
    return max(16, triton.next_power_of_2(value))


@triton.jit
def _gated_attn_fwd(
    gQ,
    gK,
    gV,
    gQI,
    gKI,
    gLSE,
    gGateScale,
    gOut,
    gRowLSE,
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
    stride_qib,
    stride_qih,
    stride_qim,
    stride_qid,
    stride_kib,
    stride_kin,
    stride_kid,
    stride_lb,
    stride_lh,
    stride_lm,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_rb,
    stride_rh,
    stride_rm,
    q_len,
    k_len,
    query_offset,
    n_sink,
    sm_scale,
    PIN_LOCAL: tl.constexpr,
    N_LOCAL: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DI: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IDX_DIM: tl.constexpr,
    DIM_V: tl.constexpr,
    PRECISION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    head_kv = pid_h // GROUP
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_di = tl.arange(0, BLOCK_DI)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_m = offs_m < q_len
    mask_d = offs_d < HEAD_DIM
    mask_di = offs_di < IDX_DIM
    mask_dv = offs_dv < DIM_V
    q = tl.load(
        gQ + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )
    q_idx = tl.load(
        gQI + pid_b * stride_qib + head_kv * stride_qih + offs_m[:, None] * stride_qim + offs_di[None, :] * stride_qid,
        mask=mask_m[:, None] & mask_di[None, :],
        other=0.0,
    )
    lse = tl.load(gLSE + pid_b * stride_lb + head_kv * stride_lh + offs_m * stride_lm, mask=mask_m, other=0.0)
    gate_scale = tl.load(gGateScale).to(tl.float32)
    run_max = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    run_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    q_pos = offs_m + query_offset
    n_end = tl.minimum(k_len, pid_m * BLOCK_M + BLOCK_M - 1 + query_offset + 1)
    for start_n in range(0, n_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < k_len
        k = tl.load(
            gK + pid_b * stride_kb + head_kv * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd,
            mask=mask_n[None, :] & mask_d[:, None],
            other=0.0,
        )
        k_idx = tl.load(
            gKI + pid_b * stride_kib + offs_n[None, :] * stride_kin + offs_di[:, None] * stride_kid,
            mask=mask_n[None, :] & mask_di[:, None],
            other=0.0,
        )
        logits = tl.dot(q, k, input_precision=PRECISION) * sm_scale
        score = tl.dot(q_idx, k_idx, input_precision=PRECISION) * gate_scale
        pinned = offs_n[None, :] < n_sink
        if PIN_LOCAL:
            age = q_pos[:, None] - offs_n[None, :]
            pinned = pinned | (age >= 0) & (age < N_LOCAL)
        logits = logits + tl.where(pinned, 0.0, score - lse[:, None])
        causal = (offs_n[None, :] <= q_pos[:, None]) & mask_n[None, :] & mask_m[:, None]
        logits = tl.where(causal, logits, float("-inf"))
        new_max = tl.maximum(run_max, tl.max(logits, 1))
        alive = new_max > float("-inf")
        safe_max = tl.where(alive, new_max, 0.0)
        rescale = tl.where(run_max > float("-inf"), tl.exp(run_max - safe_max), 0.0)
        p = tl.where(causal, tl.exp(logits - safe_max[:, None]), 0.0)
        v = tl.load(
            gV + pid_b * stride_vb + head_kv * stride_vh + offs_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd,
            mask=mask_n[:, None] & mask_dv[None, :],
            other=0.0,
        )
        acc = acc * rescale[:, None] + tl.dot(p.to(v.dtype), v, input_precision=PRECISION)
        run_sum = run_sum * rescale + tl.sum(p, 1)
        run_max = safe_max
    out = acc / run_sum[:, None]
    tl.store(
        gOut + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_om + offs_dv[None, :] * stride_od,
        out.to(gOut.dtype.element_ty),
        mask=mask_m[:, None] & mask_dv[None, :],
    )
    tl.store(
        gRowLSE + pid_b * stride_rb + pid_h * stride_rh + offs_m * stride_rm, run_max + tl.log(run_sum), mask=mask_m
    )


@triton.jit
def _gated_attn_bwd(
    gQ,
    gK,
    gV,
    gQI,
    gKI,
    gLSE,
    gGateScale,
    gOut,
    gRowLSE,
    gDOut,
    gDelta,
    gDQ,
    gDK,
    gDV,
    gDQI,
    gDKI,
    gDGateScale,
    gDLSE,
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
    stride_qib,
    stride_qih,
    stride_qim,
    stride_qid,
    stride_kib,
    stride_kin,
    stride_kid,
    stride_lb,
    stride_lh,
    stride_lm,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_rb,
    stride_rh,
    stride_rm,
    q_len,
    k_len,
    query_offset,
    n_sink,
    sm_scale,
    PIN_LOCAL: tl.constexpr,
    N_LOCAL: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DI: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IDX_DIM: tl.constexpr,
    DIM_V: tl.constexpr,
    PRECISION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)
    head_kv = pid_h // GROUP
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_di = tl.arange(0, BLOCK_DI)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_m = offs_m < q_len
    mask_d = offs_d < HEAD_DIM
    mask_di = offs_di < IDX_DIM
    mask_dv = offs_dv < DIM_V
    q = tl.load(
        gQ + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )
    q_idx = tl.load(
        gQI + pid_b * stride_qib + head_kv * stride_qih + offs_m[:, None] * stride_qim + offs_di[None, :] * stride_qid,
        mask=mask_m[:, None] & mask_di[None, :],
        other=0.0,
    )
    lse = tl.load(gLSE + pid_b * stride_lb + head_kv * stride_lh + offs_m * stride_lm, mask=mask_m, other=0.0)
    row_lse = tl.load(gRowLSE + pid_b * stride_rb + pid_h * stride_rh + offs_m * stride_rm, mask=mask_m, other=0.0)
    delta = tl.load(gDelta + pid_b * stride_rb + pid_h * stride_rh + offs_m * stride_rm, mask=mask_m, other=0.0)
    dout = tl.load(
        gDOut + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_om + offs_dv[None, :] * stride_od,
        mask=mask_m[:, None] & mask_dv[None, :],
        other=0.0,
    )
    gate_scale = tl.load(gGateScale).to(tl.float32)
    dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    dq_idx = tl.zeros([BLOCK_M, BLOCK_DI], dtype=tl.float32)
    d_gate_scale = tl.zeros([1], dtype=tl.float32)
    d_lse = tl.zeros([BLOCK_M], dtype=tl.float32)
    q_pos = offs_m + query_offset
    n_end = tl.minimum(k_len, pid_m * BLOCK_M + BLOCK_M - 1 + query_offset + 1)
    for start_n in range(0, n_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < k_len
        k = tl.load(
            gK + pid_b * stride_kb + head_kv * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd,
            mask=mask_n[None, :] & mask_d[:, None],
            other=0.0,
        )
        k_idx = tl.load(
            gKI + pid_b * stride_kib + offs_n[None, :] * stride_kin + offs_di[:, None] * stride_kid,
            mask=mask_n[None, :] & mask_di[:, None],
            other=0.0,
        )
        v = tl.load(
            gV + pid_b * stride_vb + head_kv * stride_vh + offs_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd,
            mask=mask_n[:, None] & mask_dv[None, :],
            other=0.0,
        )
        raw_gate = tl.dot(q_idx, k_idx, input_precision=PRECISION)
        logits = tl.dot(q, k, input_precision=PRECISION) * sm_scale
        pinned = offs_n[None, :] < n_sink
        if PIN_LOCAL:
            age = q_pos[:, None] - offs_n[None, :]
            pinned = pinned | (age >= 0) & (age < N_LOCAL)
        logits = logits + tl.where(pinned, 0.0, raw_gate * gate_scale - lse[:, None])
        causal = (offs_n[None, :] <= q_pos[:, None]) & mask_n[None, :] & mask_m[:, None]
        p = tl.where(causal, tl.exp(logits - row_lse[:, None]), 0.0)
        tl.atomic_add(
            gDV + pid_b * stride_vb + head_kv * stride_vh + offs_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd,
            tl.dot(tl.trans(p).to(dout.dtype), dout, input_precision=PRECISION),
            mask=mask_n[:, None] & mask_dv[None, :],
        )
        dp = tl.dot(dout, tl.trans(v), input_precision=PRECISION)
        ds = p * (dp - delta[:, None])
        ds_attn = ds * sm_scale
        ds_gate = tl.where(pinned, 0.0, ds)
        dq += tl.dot(ds_attn.to(k.dtype), tl.trans(k), input_precision=PRECISION)
        tl.atomic_add(
            gDK + pid_b * stride_kb + head_kv * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            tl.dot(tl.trans(ds_attn).to(q.dtype), q, input_precision=PRECISION),
            mask=mask_n[:, None] & mask_d[None, :],
        )
        dq_idx += tl.dot(ds_gate.to(k_idx.dtype), tl.trans(k_idx), input_precision=PRECISION) * gate_scale
        tl.atomic_add(
            gDKI + pid_b * stride_kib + offs_n[:, None] * stride_kin + offs_di[None, :] * stride_kid,
            tl.dot(tl.trans(ds_gate).to(q_idx.dtype), q_idx, input_precision=PRECISION) * gate_scale,
            mask=mask_n[:, None] & mask_di[None, :],
        )
        d_gate_scale += tl.sum(tl.sum(ds_gate * raw_gate, 1), 0)
        d_lse += -tl.sum(ds_gate, 1)
    tl.store(
        gDQ + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        dq.to(gDQ.dtype.element_ty),
        mask=mask_m[:, None] & mask_d[None, :],
    )
    tl.atomic_add(
        gDQI + pid_b * stride_qib + head_kv * stride_qih + offs_m[:, None] * stride_qim + offs_di[None, :] * stride_qid,
        dq_idx,
        mask=mask_m[:, None] & mask_di[None, :],
    )
    tl.atomic_add(gDGateScale, tl.sum(d_gate_scale, 0))
    tl.atomic_add(gDLSE + pid_b * stride_lb + head_kv * stride_lh + offs_m * stride_lm, d_lse, mask=mask_m)


class _GatedAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        q_idx,
        k_idx,
        lse,
        gate_scale,
        sm_scale,
        query_offset,
        n_sink,
        n_local,
        block_m,
        block_n,
        precision,
    ):
        (bsz, n_heads, q_len, head_dim) = q.shape
        (n_kv_heads, k_len) = (k.shape[1], k.shape[2])
        dim_v = v.shape[-1]
        group = n_heads // n_kv_heads
        out = torch.empty((bsz, n_heads, q_len, dim_v), device=q.device, dtype=q.dtype)
        row_lse = torch.empty((bsz, n_heads, q_len), device=q.device, dtype=torch.float32)
        shapes = dict(
            GROUP=group,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_pow2(head_dim),
            BLOCK_DI=block_pow2(q_idx.shape[-1]),
            BLOCK_DV=block_pow2(dim_v),
            HEAD_DIM=head_dim,
            IDX_DIM=q_idx.shape[-1],
            DIM_V=dim_v,
            PRECISION=precision,
        )
        grid = (triton.cdiv(q_len, block_m), n_heads, bsz)
        _gated_attn_fwd[grid](
            q,
            k,
            v,
            q_idx,
            k_idx,
            lse,
            gate_scale,
            out,
            row_lse,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *q_idx.stride(),
            *k_idx.stride(),
            *lse.stride(),
            *out.stride(),
            *row_lse.stride(),
            q_len,
            k_len,
            query_offset,
            n_sink,
            sm_scale,
            PIN_LOCAL=n_local > 0,
            N_LOCAL=n_local,
            **shapes,
        )
        ctx.save_for_backward(q, k, v, q_idx, k_idx, lse, gate_scale, out, row_lse)
        ctx.meta = (sm_scale, query_offset, n_sink, n_local, block_m, block_n, precision, group)
        return (out, row_lse)

    @staticmethod
    def backward(ctx, d_out, _d_row_lse):
        (q, k, v, q_idx, k_idx, lse, gate_scale, out, row_lse) = ctx.saved_tensors
        (sm_scale, query_offset, n_sink, n_local, block_m, block_n, precision, group) = ctx.meta
        (bsz, n_heads, q_len, head_dim) = q.shape
        (k_len, dim_v) = (k.shape[2], v.shape[-1])
        d_out = d_out.contiguous()
        delta = (out.float() * d_out.float()).sum(-1)
        d_q = torch.zeros_like(q, dtype=torch.float32)
        d_k = torch.zeros_like(k, dtype=torch.float32)
        d_v = torch.zeros_like(v, dtype=torch.float32)
        d_q_idx = torch.zeros_like(q_idx, dtype=torch.float32)
        d_k_idx = torch.zeros_like(k_idx, dtype=torch.float32)
        d_gate_scale = torch.zeros(1, device=q.device, dtype=torch.float32)
        d_lse = torch.zeros_like(lse, dtype=torch.float32)
        shapes = dict(
            GROUP=group,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_pow2(head_dim),
            BLOCK_DI=block_pow2(q_idx.shape[-1]),
            BLOCK_DV=block_pow2(dim_v),
            HEAD_DIM=head_dim,
            IDX_DIM=q_idx.shape[-1],
            DIM_V=dim_v,
            PRECISION=precision,
        )
        grid = (triton.cdiv(q_len, block_m), n_heads, bsz)
        _gated_attn_bwd[grid](
            q,
            k,
            v,
            q_idx,
            k_idx,
            lse,
            gate_scale,
            out,
            row_lse,
            d_out,
            delta,
            d_q,
            d_k,
            d_v,
            d_q_idx,
            d_k_idx,
            d_gate_scale,
            d_lse,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *q_idx.stride(),
            *k_idx.stride(),
            *lse.stride(),
            *d_out.stride(),
            *row_lse.stride(),
            q_len,
            k_len,
            query_offset,
            n_sink,
            sm_scale,
            PIN_LOCAL=n_local > 0,
            N_LOCAL=n_local,
            **shapes,
        )
        return (
            d_q.to(q.dtype),
            d_k.to(k.dtype),
            d_v.to(v.dtype),
            d_q_idx.to(q_idx.dtype),
            d_k_idx.to(k_idx.dtype),
            d_lse.to(lse.dtype),
            d_gate_scale.to(gate_scale.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def triton_gated_attention(
    q, k, v, q_idx, k_idx, lse, *, gate_scale, scaling, query_offset, sink_size=0, window_size=0
):
    out, _ = _GatedAttention.apply(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        q_idx.contiguous(),
        k_idx.contiguous(),
        lse.contiguous(),
        gate_scale,
        scaling,
        query_offset,
        sink_size,
        window_size,
        64,
        64,
        "ieee",
    )
    return out
