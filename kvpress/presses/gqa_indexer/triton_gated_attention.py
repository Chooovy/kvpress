# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fused Triton kernel for gated attention: the indexer score added inside the softmax.

The operation is defined by
:func:`~kvpress.presses.gqa_indexer.gated_attention.gated_attention_reference`::

    out = softmax(scale * q @ k^T + gate) @ v
    gate = gate_scale * qi @ ki^T - lse    on history keys
         = 0                               on pinned keys

This module is the path that makes it *runnable*. Two earlier attempts on
``scaled_dot_product_attention`` both OOM'd, and the reasons are worth recording because they
are what the kernel exists to avoid:

1. **The concat trick needs ``Dqk != Dv``.** Folding the bilinear gate into the QK product
   widens Q and K to ``D + Di`` while V stays at ``Dv``. Flash attention requires
   ``Q.size(-1) == V.size(-1)``, so SDPA silently chose the **math** backend, which
   materializes ``(B, H, Sq, Sk)`` attention weights *and keeps them for backward*: 144 GiB
   across 36 layers at ``L=8192, Hq=32`` in bf16.
2. **Padding V to match did not rescue it.** ``Dqk = Dv = 256`` restores the shape rule, but a
   256-wide head is beyond what flash and mem-efficient support for the *backward* pass on
   these architectures, and an explicit ``attn_mask`` disqualifies flash outright. So the math
   backend was chosen again.

Both are structural: nothing about the concat formulation lets a stock kernel keep ``O(L)``
memory here. Computing the gate inside the tile loop removes the need for it entirely -- ``Dqk``
and ``Dv`` both stay at their true 128, and the gate is a second ``tl.dot`` on ``Di``-wide
operands rather than an extra 128 columns on every Q/K/V load.

Design
------
One program per ``(query tile, KV head, batch)``. Per key tile:

* ``scale * q @ k^T`` -- the attention logits, ``D``-wide contraction.
* ``gate_scale * qi @ ki^T`` -- the gate, ``Di``-wide, on the *same* key tile.
* ``- lse`` on history keys, ``0`` on pinned keys, then the standard online softmax update.

``lse`` is passed in rather than computed here. It is a per-query scalar that
:func:`~kvpress.presses.gqa_indexer.gate_pin.history_lse` already produces in one streaming
pass, and folding it in would force this kernel to make two passes over the keys (one to
normalize, one to attend) for no memory saving.

Pinning is expressed as a **bias table over the pinned set**, not as a mode flag. Both
``sink`` (the first ``n_sink`` keys, same for every query) and ``self`` (each query's own
diagonal) reduce to "these ``(query, key)`` pairs take gate ``0`` instead of ``score - lse``",
and the kernel decides that from ``n_sink`` and the diagonal position arithmetically. So
``self`` is no longer the expensive mode -- it costs one extra comparison per element, and the
``O(Sq * Sk)`` two-branch fallback it needed is gone.

Numerics
--------
fp32 accumulation, ``input_precision="ieee"`` by default, matching
:mod:`~.triton_sparse_attention` and :mod:`~.triton_fused_loss`: the kernel's first job is to
be trusted against the fp64 reference, so it should not start out TF32's ~1e-3 away from it.

Backward
--------
Recomputes the logits per tile rather than storing them -- the same trade
:class:`~.gate_pin._HistoryLSE` makes, and for the same reason. Five gradients come out:
``q``, ``k``, ``v``, ``q_idx``, ``k_idx``, plus the scalar ``gate_scale``. ``dq``/``dq_idx``
accumulate within one program (a query tile owns its row), while ``dk``/``dv``/``dk_idx`` are
written with ``atomic_add`` because every query tile touches every key tile.
"""

from __future__ import annotations

import logging

import torch

from kvpress.presses.gqa_indexer.triton_fused_loss import (
    HAS_TRITON,
    block_pow2,
    triton_interpret_enabled,
)

logger = logging.getLogger(__name__)

if HAS_TRITON:
    import triton
    import triton.language as tl


def gated_kernels_available(*tensors: torch.Tensor) -> bool:
    """
    Whether the Triton gated-attention path can run on these tensors.

    Requires Triton, and CUDA unless the interpreter is on. fp16/bf16/fp32 only: ``tl.dot`` has
    no fp64, and silently demoting an fp64 caller would defeat the point of asking for it --
    which is also why the fp64 reference tests must keep routing to the torch path.
    """
    if not HAS_TRITON:
        return False
    if not triton_interpret_enabled() and not all(t.is_cuda for t in tensors):
        return False
    return all(t.dtype in (torch.float16, torch.bfloat16, torch.float32) for t in tensors)


if HAS_TRITON:

    @triton.jit
    def _gated_attn_fwd(
        gQ, gK, gV, gQI, gKI, gLSE, gGateScale, gOut, gRowLSE,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_qib, stride_qih, stride_qim, stride_qid,
        stride_kib, stride_kin, stride_kid,
        stride_lb, stride_lh, stride_lm,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_rb, stride_rh, stride_rm,
        q_len, k_len, query_offset, n_sink,
        sm_scale,
        PIN_SELF: tl.constexpr,
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
        """
        One ``(query tile, query head, batch)`` program; streams key tiles with online softmax.

        ``GROUP`` maps a query head to its KV head (``head // GROUP``), so the indexer tensors --
        which are per *KV* head -- are read with the same index the K/V loads use. The gate is
        therefore shared by the query heads of a group, matching the reference.
        """
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
            gQ + pid_b * stride_qb + pid_h * stride_qh
            + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=mask_m[:, None] & mask_d[None, :], other=0.0,
        )
        q_idx = tl.load(
            gQI + pid_b * stride_qib + head_kv * stride_qih
            + offs_m[:, None] * stride_qim + offs_di[None, :] * stride_qid,
            mask=mask_m[:, None] & mask_di[None, :], other=0.0,
        )
        lse = tl.load(
            gLSE + pid_b * stride_lb + head_kv * stride_lh + offs_m * stride_lm,
            mask=mask_m, other=0.0,
        )
        gate_scale = tl.load(gGateScale).to(tl.float32)

        run_max = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        run_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)

        # Bottom-right alignment: query i sits at absolute position i + query_offset, so it may
        # see keys up to that index. Matches causal_mask_bottom_right and build_indexer_mask.
        q_pos = offs_m + query_offset
        # Only tiles at or before the last query's diagonal can contribute.
        n_end = tl.minimum(k_len, (pid_m * BLOCK_M + BLOCK_M - 1) + query_offset + 1)

        for start_n in range(0, n_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < k_len

            k = tl.load(
                gK + pid_b * stride_kb + head_kv * stride_kh
                + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd,
                mask=mask_n[None, :] & mask_d[:, None], other=0.0,
            )
            k_idx = tl.load(
                gKI + pid_b * stride_kib
                + offs_n[None, :] * stride_kin + offs_di[:, None] * stride_kid,
                mask=mask_n[None, :] & mask_di[:, None], other=0.0,
            )

            logits = tl.dot(q, k, input_precision=PRECISION) * sm_scale
            score = tl.dot(q_idx, k_idx, input_precision=PRECISION) * gate_scale

            # Pinned pairs take gate 0; history takes score - lse. Derived arithmetically so
            # both pin modes share this line -- `self` costs one comparison, not a second pass.
            pinned = offs_n[None, :] < n_sink
            if PIN_SELF:
                pinned = pinned | (offs_n[None, :] == q_pos[:, None])
            logits = logits + tl.where(pinned, 0.0, score - lse[:, None])

            causal = (offs_n[None, :] <= q_pos[:, None]) & mask_n[None, :] & mask_m[:, None]
            logits = tl.where(causal, logits, float("-inf"))

            new_max = tl.maximum(run_max, tl.max(logits, 1))
            # A fully masked tile leaves new_max at -inf and exp(-inf - -inf) is NaN, so the
            # rescale is forced to 0 there. Same guard as the torch paths.
            alive = new_max > float("-inf")
            safe_max = tl.where(alive, new_max, 0.0)
            rescale = tl.where(run_max > float("-inf"), tl.exp(run_max - safe_max), 0.0)
            p = tl.where(causal, tl.exp(logits - safe_max[:, None]), 0.0)

            v = tl.load(
                gV + pid_b * stride_vb + head_kv * stride_vh
                + offs_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd,
                mask=mask_n[:, None] & mask_dv[None, :], other=0.0,
            )
            acc = acc * rescale[:, None] + tl.dot(p.to(v.dtype), v, input_precision=PRECISION)
            run_sum = run_sum * rescale + tl.sum(p, 1)
            run_max = safe_max

        # Padding lanes (offs_m >= q_len) never accumulate, so run_sum is 0 there and this
        # divide -- and the log below -- produce inf/nan for them. Both are discarded by the
        # masked stores, so the guard is the mask rather than a clamp; clamping would instead
        # write a plausible-looking wrong number into a lane nobody should read. Verified
        # tile-invariant: block_m with and without padding lanes agree to 1e-6.
        out = acc / run_sum[:, None]
        tl.store(
            gOut + pid_b * stride_ob + pid_h * stride_oh
            + offs_m[:, None] * stride_om + offs_dv[None, :] * stride_od,
            out.to(gOut.dtype.element_ty),
            mask=mask_m[:, None] & mask_dv[None, :],
        )
        # The row logsumexp, saved so backward can rebuild p without storing it.
        tl.store(
            gRowLSE + pid_b * stride_rb + pid_h * stride_rh + offs_m * stride_rm,
            run_max + tl.log(run_sum),
            mask=mask_m,
        )

    @triton.jit
    def _gated_attn_bwd(
        gQ, gK, gV, gQI, gKI, gLSE, gGateScale, gOut, gRowLSE, gDOut, gDelta,
        gDQ, gDK, gDV, gDQI, gDKI, gDGateScale, gDLSE,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_qib, stride_qih, stride_qim, stride_qid,
        stride_kib, stride_kin, stride_kid,
        stride_lb, stride_lh, stride_lm,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_rb, stride_rh, stride_rm,
        q_len, k_len, query_offset, n_sink,
        sm_scale,
        PIN_SELF: tl.constexpr,
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
        """
        Backward for one query tile: recompute the logits, then accumulate all five gradients.

        ``dq``/``dq_idx`` are private to this tile and written once. ``dk``/``dv``/``dk_idx`` are
        accumulated with ``atomic_add``: every query tile touches every key tile, so the
        alternative is a second kernel that re-tiles by key, which is more code for the same
        traffic at these sizes.

        ``delta = rowsum(out * dout)`` is precomputed by the caller (the standard flash trick),
        which is what lets ``dS = p * (dp - delta)`` be formed without a second pass over V.
        """
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
            gQ + pid_b * stride_qb + pid_h * stride_qh
            + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=mask_m[:, None] & mask_d[None, :], other=0.0,
        )
        q_idx = tl.load(
            gQI + pid_b * stride_qib + head_kv * stride_qih
            + offs_m[:, None] * stride_qim + offs_di[None, :] * stride_qid,
            mask=mask_m[:, None] & mask_di[None, :], other=0.0,
        )
        lse = tl.load(
            gLSE + pid_b * stride_lb + head_kv * stride_lh + offs_m * stride_lm,
            mask=mask_m, other=0.0,
        )
        row_lse = tl.load(
            gRowLSE + pid_b * stride_rb + pid_h * stride_rh + offs_m * stride_rm,
            mask=mask_m, other=0.0,
        )
        delta = tl.load(
            gDelta + pid_b * stride_rb + pid_h * stride_rh + offs_m * stride_rm,
            mask=mask_m, other=0.0,
        )
        dout = tl.load(
            gDOut + pid_b * stride_ob + pid_h * stride_oh
            + offs_m[:, None] * stride_om + offs_dv[None, :] * stride_od,
            mask=mask_m[:, None] & mask_dv[None, :], other=0.0,
        )
        gate_scale = tl.load(gGateScale).to(tl.float32)

        dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        dq_idx = tl.zeros([BLOCK_M, BLOCK_DI], dtype=tl.float32)
        d_gate_scale = tl.zeros([1], dtype=tl.float32)
        # The gate is (score - lse) on history, so d/d(lse) = -sum over history of dS.
        # Verified against autograd; equals +sum over PINNED of dS, since a softmax row's dS
        # sums to zero. Accumulated here because ds_gate is already the history-only dS.
        d_lse = tl.zeros([BLOCK_M], dtype=tl.float32)

        q_pos = offs_m + query_offset
        n_end = tl.minimum(k_len, (pid_m * BLOCK_M + BLOCK_M - 1) + query_offset + 1)

        for start_n in range(0, n_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < k_len

            k = tl.load(
                gK + pid_b * stride_kb + head_kv * stride_kh
                + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd,
                mask=mask_n[None, :] & mask_d[:, None], other=0.0,
            )
            k_idx = tl.load(
                gKI + pid_b * stride_kib
                + offs_n[None, :] * stride_kin + offs_di[:, None] * stride_kid,
                mask=mask_n[None, :] & mask_di[:, None], other=0.0,
            )
            v = tl.load(
                gV + pid_b * stride_vb + head_kv * stride_vh
                + offs_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd,
                mask=mask_n[:, None] & mask_dv[None, :], other=0.0,
            )

            raw_gate = tl.dot(q_idx, k_idx, input_precision=PRECISION)
            logits = tl.dot(q, k, input_precision=PRECISION) * sm_scale
            pinned = offs_n[None, :] < n_sink
            if PIN_SELF:
                pinned = pinned | (offs_n[None, :] == q_pos[:, None])
            logits = logits + tl.where(pinned, 0.0, raw_gate * gate_scale - lse[:, None])

            causal = (offs_n[None, :] <= q_pos[:, None]) & mask_n[None, :] & mask_m[:, None]
            p = tl.where(causal, tl.exp(logits - row_lse[:, None]), 0.0)

            # dV = p^T @ dout, and dp = dout @ V^T -> dS = p * (dp - delta).
            tl.atomic_add(
                gDV + pid_b * stride_vb + head_kv * stride_vh
                + offs_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd,
                tl.dot(tl.trans(p).to(dout.dtype), dout, input_precision=PRECISION),
                mask=mask_n[:, None] & mask_dv[None, :],
            )
            dp = tl.dot(dout, tl.trans(v), input_precision=PRECISION)
            ds = p * (dp - delta[:, None])

            # The attention term carries sm_scale; the gate term carries gate_scale, and only
            # on the non-pinned entries (a pinned entry's gate is the constant 0).
            ds_attn = ds * sm_scale
            ds_gate = tl.where(pinned, 0.0, ds)

            dq += tl.dot(ds_attn.to(k.dtype), tl.trans(k), input_precision=PRECISION)
            tl.atomic_add(
                gDK + pid_b * stride_kb + head_kv * stride_kh
                + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                tl.dot(tl.trans(ds_attn).to(q.dtype), q, input_precision=PRECISION),
                mask=mask_n[:, None] & mask_d[None, :],
            )
            dq_idx += tl.dot(ds_gate.to(k_idx.dtype), tl.trans(k_idx), input_precision=PRECISION) * gate_scale
            tl.atomic_add(
                gDKI + pid_b * stride_kib
                + offs_n[:, None] * stride_kin + offs_di[None, :] * stride_kid,
                tl.dot(tl.trans(ds_gate).to(q_idx.dtype), q_idx, input_precision=PRECISION) * gate_scale,
                mask=mask_n[:, None] & mask_di[None, :],
            )
            d_gate_scale += tl.sum(tl.sum(ds_gate * raw_gate, 1), 0)
            d_lse += -tl.sum(ds_gate, 1)

        tl.store(
            gDQ + pid_b * stride_qb + pid_h * stride_qh
            + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            dq.to(gDQ.dtype.element_ty),
            mask=mask_m[:, None] & mask_d[None, :],
        )
        tl.atomic_add(
            gDQI + pid_b * stride_qib + head_kv * stride_qih
            + offs_m[:, None] * stride_qim + offs_di[None, :] * stride_qid,
            dq_idx,
            mask=mask_m[:, None] & mask_di[None, :],
        )
        tl.atomic_add(gDGateScale, tl.sum(d_gate_scale, 0))
        # Summed over the query heads of a group, matching lse's per-KV-head layout.
        tl.atomic_add(
            gDLSE + pid_b * stride_lb + head_kv * stride_lh + offs_m * stride_lm,
            d_lse, mask=mask_m,
        )


class _GatedAttention(torch.autograd.Function):
    """
    Autograd wrapper: forward saves ``O(L)``, backward recomputes the logits per tile.

    ``-lse`` enters the gate as a per-query bias rather than being recomputed here; see the
    module docstring. Note the gradient w.r.t. ``lse`` is *deliberately* propagated by the
    caller through :func:`~.gate_pin.history_lse`, not by this kernel: ``lse`` is an input, and
    its own dependence on ``q_idx``/``k_idx`` is that function's business.
    """

    @staticmethod
    def forward(ctx, q, k, v, q_idx, k_idx, lse, gate_scale, sm_scale,
                query_offset, n_sink, pin_self, block_m, block_n, precision):
        bsz, n_heads, q_len, head_dim = q.shape
        n_kv_heads, k_len = k.shape[1], k.shape[2]
        dim_v = v.shape[-1]
        group = n_heads // n_kv_heads

        out = torch.empty((bsz, n_heads, q_len, dim_v), device=q.device, dtype=q.dtype)
        row_lse = torch.empty((bsz, n_heads, q_len), device=q.device, dtype=torch.float32)
        shapes = dict(
            GROUP=group, BLOCK_M=block_m, BLOCK_N=block_n,
            BLOCK_D=block_pow2(head_dim), BLOCK_DI=block_pow2(q_idx.shape[-1]),
            BLOCK_DV=block_pow2(dim_v),
            HEAD_DIM=head_dim, IDX_DIM=q_idx.shape[-1], DIM_V=dim_v,
            PRECISION=precision,
        )
        grid = (triton.cdiv(q_len, block_m), n_heads, bsz)
        _gated_attn_fwd[grid](
            q, k, v, q_idx, k_idx, lse, gate_scale, out, row_lse,
            *q.stride(), *k.stride(), *v.stride(), *q_idx.stride(), *k_idx.stride(),
            *lse.stride(), *out.stride(), *row_lse.stride(),
            q_len, k_len, query_offset, n_sink,
            sm_scale, PIN_SELF=pin_self, **shapes,
        )

        ctx.save_for_backward(q, k, v, q_idx, k_idx, lse, gate_scale, out, row_lse)
        ctx.meta = (sm_scale, query_offset, n_sink, pin_self, block_m, block_n, precision, group)
        return out, row_lse

    @staticmethod
    def backward(ctx, d_out, _d_row_lse):
        q, k, v, q_idx, k_idx, lse, gate_scale, out, row_lse = ctx.saved_tensors
        sm_scale, query_offset, n_sink, pin_self, block_m, block_n, precision, group = ctx.meta
        bsz, n_heads, q_len, head_dim = q.shape
        k_len, dim_v = k.shape[2], v.shape[-1]

        d_out = d_out.contiguous()
        # delta = rowsum(out * dout): the flash trick that turns dS into p * (dp - delta) and
        # removes a second pass over V.
        delta = (out.float() * d_out.float()).sum(-1)

        # fp32 accumulators: atomic_add into a reduced-precision buffer would lose the small
        # contributions that make up dk/dv, and there are Sq/BLOCK_M of them per key.
        d_q = torch.zeros_like(q, dtype=torch.float32)
        d_k = torch.zeros_like(k, dtype=torch.float32)
        d_v = torch.zeros_like(v, dtype=torch.float32)
        d_q_idx = torch.zeros_like(q_idx, dtype=torch.float32)
        d_k_idx = torch.zeros_like(k_idx, dtype=torch.float32)
        d_gate_scale = torch.zeros((), device=q.device, dtype=torch.float32)
        d_lse = torch.zeros_like(lse, dtype=torch.float32)

        shapes = dict(
            GROUP=group, BLOCK_M=block_m, BLOCK_N=block_n,
            BLOCK_D=block_pow2(head_dim), BLOCK_DI=block_pow2(q_idx.shape[-1]),
            BLOCK_DV=block_pow2(dim_v),
            HEAD_DIM=head_dim, IDX_DIM=q_idx.shape[-1], DIM_V=dim_v,
            PRECISION=precision,
        )
        grid = (triton.cdiv(q_len, block_m), n_heads, bsz)
        _gated_attn_bwd[grid](
            q, k, v, q_idx, k_idx, lse, gate_scale, out, row_lse, d_out, delta,
            d_q, d_k, d_v, d_q_idx, d_k_idx, d_gate_scale, d_lse,
            *q.stride(), *k.stride(), *v.stride(), *q_idx.stride(), *k_idx.stride(),
            *lse.stride(), *d_out.stride(), *row_lse.stride(),
            q_len, k_len, query_offset, n_sink,
            sm_scale, PIN_SELF=pin_self, **shapes,
        )

        return (
            d_q.to(q.dtype), d_k.to(k.dtype), d_v.to(v.dtype),
            d_q_idx.to(q_idx.dtype), d_k_idx.to(k_idx.dtype),
            d_lse.to(lse.dtype),
            d_gate_scale.to(gate_scale.dtype),
            None, None, None, None, None, None, None,
        )


def triton_gated_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    lse: torch.Tensor,
    *,
    gate_scale: torch.Tensor,
    scaling: float,
    query_offset: int,
    n_sink: int = 0,
    pin_self: bool = False,
    block_m: int = 64,
    block_n: int = 64,
    precision: str = "ieee",
) -> torch.Tensor:
    """
    Gated attention with the gate computed inside the tile loop, ``O(L)`` memory.

    Parameters mirror
    :func:`~kvpress.presses.gqa_indexer.gated_attention.gated_attention_reference`, with the
    pinning expressed as ``(n_sink, pin_self)`` -- the two shapes every supported ``pin_mode``
    reduces to.

    ``lse`` is the per-``(batch, kv head, query)`` history normalizer from
    :func:`~.gate_pin.history_lse`; pass zeros to gate without a budget (``pin_mode="none"``,
    where the normalizer is provably inert).

    Returns ``(B, H, Sq, Dv)`` in ``q``'s dtype.
    """
    if not HAS_TRITON:
        raise RuntimeError("triton_gated_attention needs Triton")
    out, _ = _GatedAttention.apply(
        q.contiguous(), k.contiguous(), v.contiguous(),
        q_idx.contiguous(), k_idx.contiguous(), lse.contiguous(),
        gate_scale, float(scaling), int(query_offset), int(n_sink), bool(pin_self),
        int(block_m), int(block_n), precision,
    )
    return out
