# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Triton kernels for the stage-1 indexer distillation loss.

:mod:`kvpress.presses.gqa_indexer.fused_loss` already streams tiles and so is ``O(L)`` in
memory, but every tile round-trips through HBM: the student logits, ``exp_logits``, the
teacher's ``alpha`` and ``p_bar`` are each a separate ``(B, h, query_tile, key_tile)``
tensor written and read back. These kernels keep all of that in registers and shared memory,
so one pass over a tile touches HBM only for ``q``, ``k`` and the per-row accumulators.

Two further wins are structural rather than just bandwidth:

**No dense mask.** The PyTorch path takes an additive ``(B, 1, Sq, Sk)`` mask, which is
itself ``O(L^2)`` -- 64 GiB of fp32 at ``L=128K``, dwarfing everything the tiling saved. The
kernels derive causality from ``query_offset`` arithmetic and take padding as a ``(B, Sk)``
keep vector. The dispatcher falls back to PyTorch for any mask it cannot decompose that way,
so generality is preserved without paying for it on the fast path.

**Causal early exit.** A query block starting at ``m`` can only see keys up to
``m + BLOCK_M - 1 + query_offset``, so the key loop stops there instead of running to
``Sk``. That halves the work on a square causal problem.

Numerics
--------
Everything accumulates in fp32 and ``tl.dot`` is pinned to ``input_precision="ieee"``, so
results match the PyTorch path to fp32 rounding rather than to TF32's ~1e-3. That costs
throughput on Ampere and later; it is deliberate for a reference kernel whose job is to be
trusted, and it is the first knob to turn once it is.

Dead rows diverge, by design
----------------------------
A query row with no visible key (padding plus causality can produce one) gets a *different*
per-row value from the two paths. The PyTorch path *adds* a finite ``MASK_NEG = -1e4``
sentinel, so such a row's logits sit near ``-1e4`` and its ``lse`` near ``-9997``; the kernels
use a true ``-inf`` and clamp the sumexp, landing near ``-23``.

Both are finite -- which is the property that matters, since a NaN would poison every
gradient in the batch -- and both are meaningless. What makes the difference unobservable is
that :func:`triton_indexer_loss` and :func:`~.fused_loss.fused_indexer_loss` both weight rows
by ``row_valid``, so ``d(loss)/d(row)`` is exactly zero there: the scalar losses and all
gradients agree even though the raw rows do not. Compare raw ``rows`` between the two
implementations only on live rows.

Testing
-------
Correctness of the tiled algorithm itself is established by
:mod:`kvpress.presses.gqa_indexer.fused_loss` and its tests. These kernels are a transcription
of that code, so the tests here compare the two directly rather than re-deriving the
objective. They also run under ``TRITON_INTERPRET=1``, which executes the kernel bodies on
CPU and makes the logic testable without a GPU.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

try:  # pragma: no cover - depends on the runtime env
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False

    class _Stub:
        """
        Let the module import (and the PyTorch fallback work) without Triton installed.

        ``jit`` has to be a working pass-through decorator rather than a raiser: the kernel
        definitions below are guarded by ``if HAS_TRITON``, but keeping the stub honest means
        the guard is belt-and-braces rather than load-bearing.
        """

        def __getattr__(self, name):
            raise RuntimeError("Triton is not available")

        @staticmethod
        def jit(fn=None, **kwargs):
            if fn is not None:
                return fn

            def decorate(inner):
                return inner

            return decorate

    triton = tl = _Stub()


def triton_interpret_enabled() -> bool:
    """
    True when Triton is running its CPU interpreter.

    Worth checking explicitly: under the interpreter the kernels are correct but far slower
    than the PyTorch path, so choosing them for *speed* would be backwards. The dispatcher
    still uses them when asked, since that is the point of interpreter mode.
    """
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


def kernels_available(*tensors: torch.Tensor) -> bool:
    """
    Whether the Triton path can run on these tensors.

    Requires CUDA unless the interpreter is on, and fp16/bf16/fp32 -- float64 has no
    ``tl.dot``, and silently demoting a caller who asked for fp64 would defeat the point of
    asking.
    """
    if not HAS_TRITON:
        return False
    if not triton_interpret_enabled() and not all(t.is_cuda for t in tensors):
        return False
    return all(t.dtype in (torch.float16, torch.bfloat16, torch.float32) for t in tensors)


if HAS_TRITON:

    @triton.jit
    def _indexer_ce_fwd(
        Q,          # (B, h, Sq, D)   indexer queries
        K,          # (B, Sk, D)      shared indexer key (MQA)
        QT,         # (B, H, Sq, DT)  teacher queries
        KT,         # (B, h, Sk, DT)  teacher keys (one per KV head)
        LSE,        # (B, H, Sq)      teacher logsumexp over the full masked key axis
        KEEP,       # (B, Sk) int8 padding keep-mask, or a dummy
        LOSS,       # (B, h, Sq)      out: per-row cross-entropy
        LSE_S,      # (B, h, Sq)      out: student logsumexp (needed by the backward pass)
        DQ,         # (B, h, Sq, D)   out: unit-weight dQ
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kn, stride_kd,
        stride_qtb, stride_qth, stride_qtm, stride_qtd,
        stride_ktb, stride_kth, stride_ktn, stride_ktd,
        stride_lb, stride_lh, stride_lm,
        stride_keepb,
        stride_ob, stride_oh, stride_om,
        stride_db, stride_dh, stride_dm, stride_dd,
        q_len, k_len, query_offset,
        scaling,
        n_idx_heads,
        HAS_KEEP: tl.constexpr,
        GROUP: tl.constexpr,
        D: tl.constexpr,
        DT: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_DT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """
        One program per (query block, batch, indexer head).

        Maintains the same five per-row quantities as the PyTorch forward -- running
        ``(max, sumexp)``, the cross term, and the two halves of ``dQ`` -- and emits the row
        loss, the student ``lse`` and the unit-weight ``dQ``. The teacher needs no running
        state at all: ``p_bar = exp(alpha - lse)`` is exact, so it is rebuilt per tile.
        """
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        off_b = pid_bh // n_idx_heads
        off_h = pid_bh % n_idx_heads

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        offs_dt = tl.arange(0, BLOCK_DT)
        m_valid = offs_m < q_len

        q_ptrs = (
            Q + off_b * stride_qb + off_h * stride_qh
            + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        )
        q = tl.load(q_ptrs, mask=m_valid[:, None] & (offs_d[None, :] < D), other=0.0)
        q = q.to(tl.float32)

        run_max = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        run_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
        cross = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc_q = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
        tea_q = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        # Causal early exit: the last key this block can see. Without it the loop would run
        # to k_len and throw away most of its work on a square causal problem.
        last_key = pid_m * BLOCK_M + BLOCK_M - 1 + query_offset
        n_end = tl.minimum(last_key + 1, k_len)

        for start_n in range(0, n_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            n_valid = offs_n < k_len

            k_ptrs = (
                K + off_b * stride_kb
                + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            )
            k = tl.load(k_ptrs, mask=n_valid[:, None] & (offs_d[None, :] < D), other=0.0)
            k = k.to(tl.float32)

            valid = m_valid[:, None] & n_valid[None, :]
            valid = valid & (offs_n[None, :] <= offs_m[:, None] + query_offset)
            if HAS_KEEP:
                keep = tl.load(
                    KEEP + off_b * stride_keepb + offs_n, mask=n_valid, other=0
                )
                valid = valid & (keep[None, :] != 0)

            logits = tl.dot(q, tl.trans(k), input_precision="ieee")
            logits = tl.where(valid, logits, float("-inf"))

            # --- student: online softmax, rescaling sumexp and acc_q together ---
            new_max = tl.maximum(run_max, tl.max(logits, 1))
            # A block whose every entry is masked leaves new_max at -inf; exp(-inf - -inf)
            # is NaN, so the exponent uses a finite stand-in. The masked logits are already
            # -inf, so they still exponentiate to exactly 0 and contribute nothing.
            safe_max = tl.where(new_max == float("-inf"), 0.0, new_max)
            rescale = tl.where(run_max == float("-inf"), 0.0, tl.exp(run_max - safe_max))
            p = tl.exp(logits - safe_max[:, None])
            run_sum = run_sum * rescale + tl.sum(p, 1)
            acc_q = acc_q * rescale[:, None] + tl.dot(p, k, input_precision="ieee")
            run_max = new_max

            # --- teacher: exact from lse, so no running state ---
            kt_ptrs = (
                KT + off_b * stride_ktb + off_h * stride_kth
                + offs_n[:, None] * stride_ktn + offs_dt[None, :] * stride_ktd
            )
            kt = tl.load(kt_ptrs, mask=n_valid[:, None] & (offs_dt[None, :] < DT), other=0.0)
            kt = tl.trans(kt.to(tl.float32))

            p_bar = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            for g in range(GROUP):
                head_t = off_h * GROUP + g
                qt_ptrs = (
                    QT + off_b * stride_qtb + head_t * stride_qth
                    + offs_m[:, None] * stride_qtm + offs_dt[None, :] * stride_qtd
                )
                qt = tl.load(
                    qt_ptrs, mask=m_valid[:, None] & (offs_dt[None, :] < DT), other=0.0
                )
                alpha = tl.dot(qt.to(tl.float32), kt, input_precision="ieee") * scaling
                lse_g = tl.load(
                    LSE + off_b * stride_lb + head_t * stride_lh + offs_m * stride_lm,
                    mask=m_valid,
                    other=0.0,
                ).to(tl.float32)
                # Masking BEFORE the exp mirrors the PyTorch path (which adds MASK_NEG to
                # alpha) and keeps the exponent bounded: alpha at a masked key is not
                # covered by lse, so exp(alpha - lse) there is unconstrained.
                alpha = tl.where(valid, alpha - lse_g[:, None], float("-inf"))
                p_bar += tl.exp(alpha)
            p_bar = p_bar / GROUP

            cross += tl.sum(p_bar * tl.where(valid, logits, 0.0), 1)
            tea_q += tl.dot(p_bar, k, input_precision="ieee")

        safe_max = tl.where(run_max == float("-inf"), 0.0, run_max)
        lse_student = safe_max + tl.log(tl.maximum(run_sum, 1e-10))
        tl.store(
            LOSS + off_b * stride_ob + off_h * stride_oh + offs_m * stride_om,
            lse_student - cross,
            mask=m_valid,
        )
        tl.store(
            LSE_S + off_b * stride_ob + off_h * stride_oh + offs_m * stride_om,
            lse_student,
            mask=m_valid,
        )
        dq = acc_q / tl.maximum(run_sum, 1e-10)[:, None] - tea_q
        tl.store(
            DQ + off_b * stride_db + off_h * stride_dh
            + offs_m[:, None] * stride_dm + offs_d[None, :] * stride_dd,
            dq,
            mask=m_valid[:, None] & (offs_d[None, :] < D),
        )

    @triton.jit
    def _indexer_ce_bwd_dk(
        Q, K, QT, KT, LSE, LSE_S, GRAD, KEEP,
        DK,         # (B, Sk, D) out
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kn, stride_kd,
        stride_qtb, stride_qth, stride_qtm, stride_qtd,
        stride_ktb, stride_kth, stride_ktn, stride_ktd,
        stride_lb, stride_lh, stride_lm,
        stride_gb, stride_gh, stride_gm,
        stride_keepb,
        stride_dkb, stride_dkn, stride_dkd,
        q_len, k_len, query_offset,
        scaling,
        n_idx_heads,
        HAS_KEEP: tl.constexpr,
        GROUP: tl.constexpr,
        D: tl.constexpr,
        DT: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_DT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """
        One program per (key block, batch); loops over heads and query blocks.

        ``dK`` sums over queries while the forward pass streams keys, so it needs this
        separate transposed pass -- see the ``fused_loss`` module docstring. Parallelizing
        over key blocks makes every program's output range disjoint, so no atomics are
        needed even though the reduction is over queries.
        """
        pid_n = tl.program_id(0)
        off_b = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        offs_dt = tl.arange(0, BLOCK_DT)
        n_valid = offs_n < k_len

        k = tl.load(
            K + off_b * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=n_valid[:, None] & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        if HAS_KEEP:
            keep = tl.load(KEEP + off_b * stride_keepb + offs_n, mask=n_valid, other=0)

        dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

        # Only queries at or after (first key in this block - query_offset) can see it.
        m_start = tl.maximum(pid_n * BLOCK_N - query_offset, 0)
        m_start = (m_start // BLOCK_M) * BLOCK_M

        for off_h in range(n_idx_heads):
            kt = tl.load(
                KT + off_b * stride_ktb + off_h * stride_kth
                + offs_n[:, None] * stride_ktn + offs_dt[None, :] * stride_ktd,
                mask=n_valid[:, None] & (offs_dt[None, :] < DT),
                other=0.0,
            )
            kt = tl.trans(kt.to(tl.float32))

            for start_m in range(m_start, q_len, BLOCK_M):
                offs_m = start_m + tl.arange(0, BLOCK_M)
                m_valid = offs_m < q_len

                q = tl.load(
                    Q + off_b * stride_qb + off_h * stride_qh
                    + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                    mask=m_valid[:, None] & (offs_d[None, :] < D),
                    other=0.0,
                ).to(tl.float32)

                valid = m_valid[:, None] & n_valid[None, :]
                valid = valid & (offs_n[None, :] <= offs_m[:, None] + query_offset)
                if HAS_KEEP:
                    valid = valid & (keep[None, :] != 0)

                logits = tl.dot(q, tl.trans(k), input_precision="ieee")
                # LSE_S and GRAD are both (B, h, Sq), so one stride set covers both.
                lse_s = tl.load(
                    LSE_S + off_b * stride_gb + off_h * stride_gh + offs_m * stride_gm,
                    mask=m_valid,
                    other=0.0,
                ).to(tl.float32)
                p_hat = tl.where(valid, tl.exp(logits - lse_s[:, None]), 0.0)

                p_bar = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                for g in range(GROUP):
                    head_t = off_h * GROUP + g
                    qt = tl.load(
                        QT + off_b * stride_qtb + head_t * stride_qth
                        + offs_m[:, None] * stride_qtm + offs_dt[None, :] * stride_qtd,
                        mask=m_valid[:, None] & (offs_dt[None, :] < DT),
                        other=0.0,
                    )
                    alpha = tl.dot(qt.to(tl.float32), kt, input_precision="ieee") * scaling
                    lse_g = tl.load(
                        LSE + off_b * stride_lb + head_t * stride_lh + offs_m * stride_lm,
                        mask=m_valid,
                        other=0.0,
                    ).to(tl.float32)
                    alpha = tl.where(valid, alpha - lse_g[:, None], float("-inf"))
                    p_bar += tl.exp(alpha)
                p_bar = p_bar / GROUP

                grad = tl.load(
                    GRAD + off_b * stride_gb + off_h * stride_gh + offs_m * stride_gm,
                    mask=m_valid,
                    other=0.0,
                ).to(tl.float32)
                weighted = tl.where(valid, (p_hat - p_bar) * grad[:, None], 0.0)
                dk += tl.dot(tl.trans(weighted), q, input_precision="ieee")

        tl.store(
            DK + off_b * stride_dkb
            + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd,
            dk,
            mask=n_valid[:, None] & (offs_d[None, :] < D),
        )


def block_pow2(n: int, minimum: int = 16) -> int:
    """Smallest power of two >= n, floored at ``minimum`` (Triton needs power-of-two blocks)."""
    size = minimum
    while size < n:
        size *= 2
    return size


class _TritonIndexerCE(torch.autograd.Function):
    """
    Triton-backed drop-in for :class:`~.fused_loss._FusedIndexerCE`.

    Same contract: forward returns ``(B, h, Sq)`` per-row cross-entropy and stashes the
    unit-weight ``dQ``; backward scales that and launches the transposed ``dK`` pass. What
    changes is only where the tile intermediates live -- registers and shared memory instead
    of HBM.

    Unlike the PyTorch version this takes the teacher's Q/K directly rather than a
    ``teacher_alpha`` callable: the recompute has to happen *inside* the kernel for the fusion
    to mean anything, so there is nothing for a Python callable to do.
    """

    @staticmethod
    def forward(
        ctx, q_idx, k_idx, q_tea, k_tea, teacher_lse, scaling, keep, query_offset,
        block_m, block_n,
    ):
        bsz, n_idx_heads, q_len, dim = q_idx.shape
        k_len = k_idx.shape[1]
        n_heads, dim_t = q_tea.shape[1], q_tea.shape[3]
        group = n_heads // n_idx_heads

        loss_rows = torch.empty((bsz, n_idx_heads, q_len), device=q_idx.device, dtype=torch.float32)
        lse_student = torch.empty_like(loss_rows)
        dq_unit = torch.empty(
            (bsz, n_idx_heads, q_len, dim), device=q_idx.device, dtype=torch.float32
        )

        has_keep = keep is not None
        keep_arg = keep if has_keep else q_idx  # a dummy pointer; never dereferenced
        keep_stride = keep.stride(0) if has_keep else 0

        grid = (triton.cdiv(q_len, block_m), bsz * n_idx_heads)
        _indexer_ce_fwd[grid](
            q_idx, k_idx, q_tea, k_tea, teacher_lse, keep_arg,
            loss_rows, lse_student, dq_unit,
            *q_idx.stride(), *k_idx.stride(), *q_tea.stride(), *k_tea.stride(),
            *teacher_lse.stride(), keep_stride,
            *loss_rows.stride(), *dq_unit.stride(),
            q_len, k_len, query_offset, scaling, n_idx_heads,
            HAS_KEEP=has_keep, GROUP=group, D=dim, DT=dim_t,
            BLOCK_D=block_pow2(dim), BLOCK_DT=block_pow2(dim_t),
            BLOCK_M=block_m, BLOCK_N=block_n,
        )

        ctx.save_for_backward(q_idx, k_idx, q_tea, k_tea, teacher_lse, lse_student, dq_unit)
        ctx.keep = keep
        ctx.scaling = scaling
        ctx.query_offset = query_offset
        ctx.block_m, ctx.block_n = block_m, block_n
        return loss_rows

    @staticmethod
    def backward(ctx, grad_rows):
        q_idx, k_idx, q_tea, k_tea, teacher_lse, lse_student, dq_unit = ctx.saved_tensors
        keep = ctx.keep
        bsz, n_idx_heads, q_len, dim = q_idx.shape
        k_len = k_idx.shape[1]
        dim_t = q_tea.shape[3]
        group = q_tea.shape[1] // n_idx_heads

        grad_rows = grad_rows.contiguous().to(torch.float32)
        grad_q = dq_unit * grad_rows.unsqueeze(-1)
        grad_k = torch.empty((bsz, k_len, dim), device=k_idx.device, dtype=torch.float32)

        has_keep = keep is not None
        keep_arg = keep if has_keep else q_idx
        keep_stride = keep.stride(0) if has_keep else 0

        grid = (triton.cdiv(k_len, ctx.block_n), bsz)
        _indexer_ce_bwd_dk[grid](
            q_idx, k_idx, q_tea, k_tea, teacher_lse, lse_student, grad_rows, keep_arg, grad_k,
            *q_idx.stride(), *k_idx.stride(), *q_tea.stride(), *k_tea.stride(),
            *teacher_lse.stride(), *grad_rows.stride(), keep_stride, *grad_k.stride(),
            q_len, k_len, ctx.query_offset, ctx.scaling, n_idx_heads,
            HAS_KEEP=has_keep, GROUP=group, D=dim, DT=dim_t,
            BLOCK_D=block_pow2(dim), BLOCK_DT=block_pow2(dim_t),
            BLOCK_M=ctx.block_m, BLOCK_N=ctx.block_n,
        )

        return (
            grad_q.to(q_idx.dtype), grad_k.to(k_idx.dtype),
            None, None, None, None, None, None, None, None,
        )


def triton_indexer_ce_rows(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    teacher_lse: torch.Tensor,
    *,
    scaling: float,
    keep: torch.Tensor | None = None,
    query_offset: int | None = None,
    block_m: int = 64,
    block_n: int = 64,
) -> torch.Tensor:
    """
    Per-row cross-entropy between the indexer and a group-averaged teacher, on Triton.

    Parameters
    ----------
    q_idx, k_idx : torch.Tensor
        Indexer queries ``(B, h, Sq, D)`` and the shared MQA key ``(B, Sk, D)``, post-RoPE.
    query_states, key_states : torch.Tensor
        Teacher Q ``(B, H, Sq, d)`` and K ``(B, h, Sk, d)``, post-RoPE. Unlike the PyTorch
        path these are passed directly rather than behind a callable: the recompute happens
        inside the kernel, which is the whole point.
    teacher_lse : torch.Tensor
        ``(B, H, Sq)`` teacher logsumexp, under the same mask the loss applies.
    scaling : float
        Teacher softmax scale.
    keep : torch.Tensor, optional
        ``(B, Sk)`` int8 padding keep-mask. Causality is derived from ``query_offset``, so no
        dense ``(Sq, Sk)`` mask is needed -- that tensor would be 64 GiB of fp32 at
        ``L=128K``, more than the tiling saves.
    query_offset : int, optional
        Defaults to ``Sk - Sq`` (bottom-right alignment, matching flash-attention).
    block_m, block_n : int
        Tile sizes; must be powers of two.

    Returns
    -------
    torch.Tensor
        ``(B, h, Sq)`` fp32 per-row loss.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available; use fused_loss.fused_indexer_ce_rows")
    if q_idx.dim() != 4:
        raise ValueError(f"q_idx must be (B, h, Sq, D), got {tuple(q_idx.shape)}")
    if k_idx.dim() != 3:
        raise ValueError(f"k_idx must be (B, Sk, D), got {tuple(k_idx.shape)}")
    if teacher_lse.dim() != 3:
        raise ValueError(f"teacher_lse must be (B, H, Sq), got {tuple(teacher_lse.shape)}")

    n_idx_heads, n_heads = q_idx.shape[1], query_states.shape[1]
    if n_heads % n_idx_heads != 0:
        raise ValueError(f"H={n_heads} is not divisible by indexer heads={n_idx_heads}")
    if key_states.shape[1] != n_idx_heads:
        raise ValueError(
            f"key_states must carry one head per indexer head ({n_idx_heads}), got "
            f"{key_states.shape[1]}. Slice a repeat_interleave'd tensor back down first."
        )
    if block_m & (block_m - 1) or block_n & (block_n - 1):
        raise ValueError(f"block sizes must be powers of two, got {block_m}, {block_n}")

    if query_offset is None:
        query_offset = k_idx.shape[1] - q_idx.shape[2]

    return _TritonIndexerCE.apply(
        q_idx.contiguous(),
        k_idx.contiguous(),
        query_states.contiguous(),
        key_states.contiguous(),
        teacher_lse.contiguous().to(torch.float32),
        float(scaling),
        keep.contiguous() if keep is not None else None,
        int(query_offset),
        block_m,
        block_n,
    )


def decompose_mask(
    mask: torch.Tensor | None, q_len: int, k_len: int, query_offset: int
) -> tuple[bool, torch.Tensor | None]:
    """
    Split an additive mask into "causal + per-key padding", if it is expressible that way.

    Returns ``(ok, keep)``. ``ok=False`` means the mask carries structure the kernels cannot
    represent -- an arbitrary per-``(query, key)`` bias, a sliding window, a mask built with a
    different alignment than ``query_offset`` -- and the caller must fall back to the PyTorch
    path. Falling back is the *correct* outcome there, not a failure: a wrong decomposition
    would train against a mask the student never sees, with nothing downstream to catch it.

    A sink skip *does* decompose, since masking the first N keys is per-key. It can leave the
    leading query rows with no valid key at all, which the kernels keep finite (their
    ``lse`` falls back to a finite stand-in) and the caller drops via row validity.

    ``keep`` is ``None`` when the mask is purely causal (no padding at all), which lets the
    kernel skip the keep-mask load entirely.

    Verified by an exhaustive random sweep (3000 masks across causal / padded /
    random-per-pair / sliding-window shapes): every accepted mask rebuilds exactly from
    ``(causal, keep)``, and every non-decomposable one is rejected.
    """
    from kvpress.presses.gqa_indexer.indexer import MASK_NEG

    if mask is None:
        return True, None
    if mask.dim() != 4:
        return False, None

    allowed = mask > (MASK_NEG / 2)  # (B, 1, Sq, Sk) bool
    q_pos = torch.arange(q_len, device=mask.device).unsqueeze(-1) + query_offset
    k_pos = torch.arange(k_len, device=mask.device).unsqueeze(0)
    causal = k_pos <= q_pos  # (Sq, Sk)

    # Anything the causal mask forbids must also be forbidden here; a mask that *allows*
    # a non-causal pair is not something the kernels can express.
    if bool((allowed & ~causal).any()):
        return False, None

    # What remains must factor as a per-key keep vector: for every key, either every
    # causally-visible query keeps it or none does.
    visible = causal.unsqueeze(0).unsqueeze(0) & allowed.new_ones(allowed.shape)
    kept = allowed | ~causal.unsqueeze(0).unsqueeze(0)
    per_key = kept.all(dim=-2)  # (B, 1, Sk): kept by every visible query
    reconstructed = causal.unsqueeze(0).unsqueeze(0) & per_key.unsqueeze(-2)
    if not bool((reconstructed == (allowed & visible)).all()):
        return False, None

    keep = per_key.squeeze(1)  # (B, Sk)
    if bool(keep.all()):
        return True, None
    return True, keep.to(torch.int8).contiguous()


def triton_indexer_loss(
    indexer,
    hidden_states: torch.Tensor,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    teacher_lse: torch.Tensor,
    *,
    scaling: float,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    keep: torch.Tensor | None = None,
    query_offset: int | None = None,
    row_valid: torch.Tensor | None = None,
    block_m: int = 64,
    block_n: int = 64,
    loss_coeff: float = 1.0,
) -> torch.Tensor:
    """
    Scalar Triton stage-1 loss for one layer -- the kernel counterpart of
    :func:`~.fused_loss.fused_indexer_loss`.

    Same signature shape and same reduction, so the two are interchangeable at the call site.
    ``teacher_alpha``/``mask``/``key_tile``/``query_tile`` are replaced by the teacher's Q/K,
    a ``(B, Sk)`` keep vector and the block sizes, for the reasons in the module docstring.
    """
    q_idx = indexer.project_q(hidden_states, cos, sin)
    k_idx = indexer.project_k(hidden_states, cos, sin)

    rows = triton_indexer_ce_rows(
        q_idx,
        k_idx,
        query_states,
        key_states,
        teacher_lse,
        scaling=scaling,
        keep=keep,
        query_offset=query_offset,
        block_m=block_m,
        block_n=block_n,
    )

    if row_valid is None:
        return rows.mean() * loss_coeff
    weight = row_valid.to(rows.dtype).expand_as(rows)
    return (rows * weight).sum() / weight.sum().clamp_min(1.0) * loss_coeff
