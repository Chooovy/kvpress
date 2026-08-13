# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Triton kernel for indexer-driven sparse attention (GQA DSA), forward / inference only.

The operation is defined and referenced in
:mod:`kvpress.presses.gqa_indexer.sparse_attention`; this module is the fast path for it.
Each ``(query, KV head)`` row reads its own ``topk`` slice of the cache, so the kernel is a
gather feeding a flash-attention-style online softmax: never materialize ``(Sq, Sk)``, never
touch a key the index list did not name.

Why the tiles look like this
----------------------------
The MLA sparse kernels (tilelang's ``sparse_mla_fwd``, FlashMLA) put 64-128 query heads in
the ``M`` dimension of every GEMM, because MLA's single latent KV cache means *all* heads
share one index list and therefore one gathered tile. Under GQA the selection is per KV head,
so only the ``group_size = H // Hkv`` query heads of one group -- typically 4 or 8 -- share a
list.

That cannot be widened by also tiling over query tokens, the usual fix for a short ``M``:
adjacent queries hold *different* index lists, so they cannot share a gathered tile. It is
intrinsic to per-query selection, not an artifact here. So the ``Q @ K^T`` GEMM runs with
``M = group_size`` and is bandwidth-bound on the gather rather than compute-bound -- which is
the regime that makes the whole approach worthwhile, since the win comes from reading ``topk``
keys instead of ``Sk``. A configuration where ``topk`` approaches ``Sk`` will lose to a dense
kernel; that is a property of the configuration, so :func:`sparse_gqa_attention` does not try
to hide it.

``tl.dot`` shape floors
-----------------------
Triton's NVIDIA backend reports ``min_dot_size = (M=1, N=1, K=16)``: only the **contraction**
dimension is constrained. That has two consequences worth stating, because guessing wrong in
either direction costs something real:

- ``M`` needs no padding. ``BLOCK_G`` is ``group_size`` rounded to a power of two, not to 16 --
  at ``group_size=4`` that is a 4x smaller ``[BLOCK_G, DV]`` fp32 accumulator and 4x fewer
  wasted QK lanes.
- ``BLOCK_K`` **must** be >= 16, because the ``P @ V`` dot contracts over it. This is enforced
  in the wrappers rather than left to fail at compile time, since Triton's CPU interpreter does
  *not* apply the floor: a smaller ``block_k`` runs fine under ``TRITON_INTERPRET=1`` and then
  fails on the first real GPU. Validating eagerly keeps the interpreter tests honest about
  hardware.

Varlen
------
The batched and packed-varlen layouts share one kernel body, differing only in how the
per-row ``(k_start, k_len, query_offset)`` triple is obtained -- read from ``cu_seqlens`` for
varlen, passed as scalars for batched. The batched path is expressed as varlen's degenerate
case (``stride_qb = 0`` and a zero base), so there is one code path to be correct rather than
two that agree.

Varlen indices are **sequence-local**: slot values are positions within that sequence's own
KV, and the kernel adds ``cu_seqlens_k[seq]`` to reach the physical row. This is the
deliberately simple choice -- no page tables, no block tables. It means the indexer's output
needs no remapping, and a sequence can be moved in the packed buffer without rewriting its
indices.

Numerics
--------
Accumulation is fp32 and ``tl.dot`` defaults to ``input_precision="ieee"``, matching
:mod:`~.triton_fused_loss` and for the same reason: the kernel's job is first to be trusted
against the fp32 reference, so it should not be off by TF32's ~1e-3 before it starts. Pass
``precision="tf32"`` for throughput once that trust exists.

Empty rows produce ``out = 0`` and ``lse = -inf`` rather than ``0/0``; see the module
docstring of :mod:`~.sparse_attention` for why those particular values.
"""

from __future__ import annotations

import logging

import torch

from kvpress.presses.gqa_indexer.sparse_attention import (
    check_sparse_shapes,
    resolve_scaling,
    sparse_gqa_attention_reference,
)
from kvpress.presses.gqa_indexer.triton_fused_loss import (
    HAS_TRITON,
    block_pow2,
    triton_interpret_enabled,
)

logger = logging.getLogger(__name__)

if HAS_TRITON:
    import triton
    import triton.language as tl


def sparse_kernels_available(*tensors: torch.Tensor) -> bool:
    """
    Whether the Triton sparse-attention path can run on these tensors.

    Requires Triton, and CUDA unless the interpreter is on. fp16/bf16/fp32 only: ``tl.dot``
    has no fp64, and silently demoting an fp64 caller would defeat the point of asking for it.
    """
    if not HAS_TRITON:
        return False
    if not triton_interpret_enabled() and not all(t.is_cuda for t in tensors):
        return False
    return all(t.dtype in (torch.float16, torch.bfloat16, torch.float32) for t in tensors)


#: Floor on ``BLOCK_K``. ``tl.dot``'s contraction dimension must be at least 16 on NVIDIA
#: (``min_dot_size`` is ``(1, 1, 16)``), and ``P @ V`` contracts over the topk tile. Triton's
#: CPU interpreter does not enforce it, so validating here is what stops an interpreter-green
#: ``block_k`` from failing on the first real GPU.
MIN_BLOCK_K = 16


def check_block_k(block_k: int) -> None:
    """Reject a ``block_k`` that ``tl.dot`` could not accept on hardware."""
    if block_k <= 0 or block_k & (block_k - 1):
        raise ValueError(f"block_k must be a power of two, got {block_k}")
    if block_k < MIN_BLOCK_K:
        raise ValueError(
            f"block_k must be >= {MIN_BLOCK_K}, got {block_k}: it is the contraction dimension "
            "of the P @ V dot, and Triton requires K >= 16 on NVIDIA. Smaller values run under "
            "TRITON_INTERPRET=1 (which skips the check) and then fail to compile on a GPU."
        )


if HAS_TRITON:

    @triton.jit
    def _sparse_gqa_attn_fwd(
        Q,            # (B, H, Sq, D)      or packed (total_q, H, D)
        K,            # (B, Hkv, Sk, D)    or packed (total_k, Hkv, D)
        V,            # (B, Hkv, Sk, Dv)   or packed (total_k, Hkv, Dv)
        IDX,          # (B, Hkv, Sq, topk) or packed (total_q, Hkv, topk), int32, -1 = empty
        OUT,          # (B, H, Sq, Dv)     or packed (total_q, H, Dv)
        LSE,          # (B, H, Sq)         or packed (total_q, H)
        SEQ_ID,       # (total_q,) int32 token -> sequence, varlen only
        CU_Q,         # (n_seq + 1,) int32, varlen only
        CU_K,         # (n_seq + 1,) int32, varlen only
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ib, stride_ih, stride_im,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_lb, stride_lh, stride_lm,
        k_len_static,        # Sk; ignored when IS_VARLEN
        query_offset_static, # Sk - Sq; ignored when IS_VARLEN
        scale,
        n_kv_heads,
        topk,
        GROUP: tl.constexpr,      # real query heads per KV head
        BLOCK_G: tl.constexpr,    # GROUP rounded up to a power of two (no 16 floor; M is free)
        D: tl.constexpr,
        DV: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_DV: tl.constexpr,
        BLOCK_K: tl.constexpr,    # slots of the topk axis per iteration
        IS_VARLEN: tl.constexpr,
        CAUSAL: tl.constexpr,
        PRECISION: tl.constexpr,
    ):
        """
        One program per ``(query token, KV head[, batch])``; the group's query heads are the
        ``M`` dimension, so they share every gathered KV tile.

        The program owns exactly one query row of the index tensor, which is what makes the
        gather sound: all ``BLOCK_G`` lanes read the same ``topk`` list.
        """
        pid_m = tl.program_id(0)   # query token (within batch item, or within the packed axis)
        pid_bh = tl.program_id(1)  # batch * n_kv_heads + kv_head
        off_b = pid_bh // n_kv_heads
        off_kvh = pid_bh % n_kv_heads

        # --- per-row key window and causal diagonal -------------------------------------
        if IS_VARLEN:
            seq = tl.load(SEQ_ID + pid_m).to(tl.int32)
            q_start = tl.load(CU_Q + seq).to(tl.int32)
            q_stop = tl.load(CU_Q + seq + 1).to(tl.int32)
            k_start = tl.load(CU_K + seq).to(tl.int32)
            k_stop = tl.load(CU_K + seq + 1).to(tl.int32)
            k_len = k_stop - k_start
            # Bottom-right alignment within the sequence, matching the batched path.
            query_offset = k_len - (q_stop - q_start)
            q_local = pid_m - q_start
        else:
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

        # --- load this row's queries for the whole group --------------------------------
        head = off_kvh * GROUP + offs_g
        q_ptrs = (
            Q + off_b * stride_qb + head[:, None] * stride_qh
            + pid_m * stride_qm + offs_d[None, :] * stride_qd
        )
        q = tl.load(q_ptrs, mask=g_valid[:, None] & d_valid[None, :], other=0.0).to(tl.float32)

        run_max = tl.full([BLOCK_G], float("-inf"), dtype=tl.float32)
        run_sum = tl.zeros([BLOCK_G], dtype=tl.float32)
        acc = tl.zeros([BLOCK_G, BLOCK_DV], dtype=tl.float32)

        # Last key this query may legally see. Used only to mask, never to bound the loop:
        # the index list is unordered as far as the kernel is concerned.
        last_key = q_local + query_offset

        idx_base = IDX + off_b * stride_ib + off_kvh * stride_ih + pid_m * stride_im

        for start_k in range(0, topk, BLOCK_K):
            offs_k = start_k + tl.arange(0, BLOCK_K)
            slot_valid = offs_k < topk
            idx = tl.load(idx_base + offs_k, mask=slot_valid, other=-1).to(tl.int32)

            valid = slot_valid & (idx >= 0) & (idx < k_len)
            if CAUSAL:
                valid = valid & (idx <= last_key)

            # Clamp before use so every gather address is in range; `valid` is what actually
            # removes the slot, via a -inf logit. Reading a wrong-but-valid row is harmless,
            # reading out of bounds is not.
            #
            # Widen to int64 *before* adding k_start, not after: the row index then feeds
            # `row * stride`, and that product is what realistically overflows int32 (a packed
            # buffer of 4M tokens at stride 1024 already exceeds 2^31). Casting after the add
            # would leave both the add and nothing else protected, which is the wrong half.
            row = tl.cast(k_start, tl.int64) + tl.cast(tl.where(valid, idx, 0), tl.int64)

            k_ptrs = (
                K + off_b * stride_kb + off_kvh * stride_kh
                + row[:, None] * stride_kn + offs_d[None, :] * stride_kd
            )
            k_tile = tl.load(
                k_ptrs, mask=valid[:, None] & d_valid[None, :], other=0.0
            ).to(tl.float32)

            logits = tl.dot(q, tl.trans(k_tile), input_precision=PRECISION) * scale
            logits = tl.where(valid[None, :], logits, float("-inf"))

            # --- online softmax ---------------------------------------------------------
            new_max = tl.maximum(run_max, tl.max(logits, 1))
            # A tile with no valid slot leaves new_max at -inf, and exp(-inf - -inf) is NaN.
            # A finite stand-in avoids that; the -inf logits still exponentiate to exactly 0,
            # so the tile contributes nothing, which is the intent.
            safe_max = tl.where(new_max == float("-inf"), 0.0, new_max)
            rescale = tl.where(run_max == float("-inf"), 0.0, tl.exp(run_max - safe_max))
            p = tl.exp(logits - safe_max[:, None])

            v_ptrs = (
                V + off_b * stride_vb + off_kvh * stride_vh
                + row[:, None] * stride_vn + offs_dv[None, :] * stride_vd
            )
            v_tile = tl.load(
                v_ptrs, mask=valid[:, None] & dv_valid[None, :], other=0.0
            ).to(tl.float32)

            run_sum = run_sum * rescale + tl.sum(p, 1)
            acc = acc * rescale[:, None] + tl.dot(p.to(v_tile.dtype), v_tile, input_precision=PRECISION)
            run_max = new_max

        # --- epilogue: normalize, and define the empty row as (0, -inf) -----------------
        # `run_sum` needs no small-epsilon clamp. Whichever tile held the running max
        # contributed exp(max - max) = 1 to it, so a live row always has run_sum >= 1 and
        # cannot underflow; a row with run_sum == 0 had no valid slot at all, and the
        # `alive` gate replaces the divisor with 1 there so the division is never 0/0. The
        # inner `where` is what keeps the *untaken* branch finite too -- on a GPU both sides
        # of a vectorized select are evaluated, so dividing by a raw 0 would still produce
        # an inf that the outer `where` would then have to discard.
        alive = run_sum > 0.0
        out = tl.where(alive[:, None], acc / tl.where(alive[:, None], run_sum[:, None], 1.0), 0.0)
        lse = tl.where(alive, tl.log(tl.where(alive, run_sum, 1.0)) + run_max, float("-inf"))

        out_ptrs = (
            OUT + off_b * stride_ob + head[:, None] * stride_oh
            + pid_m * stride_om + offs_dv[None, :] * stride_od
        )
        tl.store(out_ptrs, out.to(OUT.dtype.element_ty), mask=g_valid[:, None] & dv_valid[None, :])
        tl.store(LSE + off_b * stride_lb + head * stride_lh + pid_m * stride_lm, lse, mask=g_valid)


def _launch(
    q, k, v, indices, out, lse,
    *, seq_id, cu_q, cu_k, k_len_static, query_offset_static,
    scale, n_kv_heads, group_size, topk, causal, block_k, precision, num_warps,
):
    """
    Shared launch for both layouts; ``stride_*b == 0`` collapses the batched axis away.

    The two layouts order their axes differently -- batched is ``(B, H, S, D)`` while packed
    is ``(S, H, D)`` -- so the packed case is *not* the batched case with a leading axis
    dropped: its token and head strides come in the opposite order. Both are normalized here
    to the ``(batch, head, token, dim)`` tuple the kernel indexes with, which is why the
    kernel body needs no layout switch of its own.
    """
    is_varlen = seq_id is not None
    dim, dim_v = q.shape[-1], v.shape[-1]

    def strides_bhnd(t):
        """Normalize to (batch, head, token, dim) strides for either layout."""
        if t.dim() == 4:  # batched (B, H, S, D)
            return t.stride()
        s = t.stride()  # packed (S, H, D): token first, no batch
        return (0, s[1], s[0], s[2])

    def strides_bhn(t):
        """Same, for a tensor whose trailing axis is not indexed by stride (indices, lse)."""
        if t.dim() == 4:  # batched indices (B, Hkv, Sq, topk)
            return t.stride()[:-1]
        if t.dim() == 3:
            s = t.stride()
            if is_varlen:  # packed indices (Sq, Hkv, topk)
                return (0, s[1], s[0])
            return s  # batched lse (B, H, Sq)
        s = t.stride()  # packed lse (Sq, H)
        return (0, s[1], s[0])

    sq_b, sq_h, sq_m, sq_d = strides_bhnd(q)
    sk_b, sk_h, sk_n, sk_d = strides_bhnd(k)
    sv_b, sv_h, sv_n, sv_d = strides_bhnd(v)
    so_b, so_h, so_m, so_d = strides_bhnd(out)
    si_b, si_h, si_m = strides_bhn(indices)
    sl_b, sl_h, sl_m = strides_bhn(lse)

    n_prog_m = q.shape[0] if is_varlen else q.shape[2]
    n_bh = n_kv_heads if is_varlen else q.shape[0] * n_kv_heads
    dummy = indices  # any tensor; unused pointers must still be valid arguments

    _sparse_gqa_attn_fwd[(n_prog_m, n_bh)](
        q, k, v, indices, out, lse,
        seq_id if is_varlen else dummy,
        cu_q if is_varlen else dummy,
        cu_k if is_varlen else dummy,
        sq_b, sq_h, sq_m, sq_d,
        sk_b, sk_h, sk_n, sk_d,
        sv_b, sv_h, sv_n, sv_d,
        si_b, si_h, si_m,
        so_b, so_h, so_m, so_d,
        sl_b, sl_h, sl_m,
        k_len_static, query_offset_static,
        scale, n_kv_heads, topk,
        GROUP=group_size,
        # M has no floor (min_dot_size is (1, 1, 16) -- only K is constrained), so this pads
        # to a power of two rather than to 16: at group_size=4 that is a 4x smaller fp32
        # accumulator and 4x fewer wasted QK lanes. See the module docstring.
        BLOCK_G=block_pow2(group_size, minimum=1),
        D=dim, DV=dim_v,
        BLOCK_D=block_pow2(dim), BLOCK_DV=block_pow2(dim_v),
        BLOCK_K=block_k,
        IS_VARLEN=is_varlen,
        CAUSAL=causal,
        PRECISION=precision,
        num_warps=num_warps,
    )


def triton_sparse_gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    *,
    scaling: float | None = None,
    query_offset: int | None = None,
    causal: bool = True,
    block_k: int = 64,
    precision: str = "ieee",
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Indexer-driven sparse attention on Triton, batched layout.

    Parameters
    ----------
    q : torch.Tensor
        Queries ``(B, H, Sq, D)``.
    k, v : torch.Tensor
        Keys ``(B, Hkv, Sk, D)`` and values ``(B, Hkv, Sk, Dv)``.
    indices : torch.Tensor
        ``(B, Hkv, Sq, topk)`` int32 key positions, ``-1`` for empty slots -- exactly what
        :func:`~.sparse_support.streaming_topk_support` returns. Order does not matter.
    scaling : float, optional
        Softmax scale; defaults to ``D ** -0.5``.
    query_offset : int, optional
        Key index of query 0's diagonal; defaults to ``Sk - Sq`` (bottom-right).
    causal : bool
        Mask slots past each query's diagonal. Redundant for indices that were selected
        causally, and cheap; see :func:`~.sparse_attention.slot_validity`.
    block_k : int
        Slots of the ``topk`` axis per iteration; power of two.
    precision : str
        ``tl.dot`` precision: ``"ieee"`` (default, matches the fp32 reference) or ``"tf32"``.
    num_warps : int
        Warps per program. ``M`` is only 16, so the default is deliberately small.

    Returns
    -------
    out : torch.Tensor
        ``(B, H, Sq, Dv)`` in ``q``'s dtype.
    lse : torch.Tensor
        ``(B, H, Sq)`` fp32; ``-inf`` where a row had no valid slot.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available; use sparse_gqa_attention_reference")
    bsz, n_heads, n_kv_heads, group_size, q_len, k_len, topk = check_sparse_shapes(q, k, v, indices)
    check_block_k(block_k)
    if query_offset is None:
        query_offset = k_len - q_len

    out = torch.empty((bsz, n_heads, q_len, v.shape[-1]), dtype=q.dtype, device=q.device)
    lse = torch.empty((bsz, n_heads, q_len), dtype=torch.float32, device=q.device)
    _launch(
        q, k, v, indices.contiguous(), out, lse,
        seq_id=None, cu_q=None, cu_k=None,
        k_len_static=k_len, query_offset_static=query_offset,
        scale=resolve_scaling(q.shape[-1], scaling),
        n_kv_heads=n_kv_heads, group_size=group_size, topk=topk,
        causal=causal, block_k=block_k, precision=precision, num_warps=num_warps,
    )
    return out, lse


def seq_ids_from_cu_seqlens(cu_seqlens: torch.Tensor, total: int) -> torch.Tensor:
    """
    Map each packed token to its sequence, ``(total,)`` int32.

    ``searchsorted`` rather than ``repeat_interleave`` so it stays correct for empty
    sequences (``cu[i] == cu[i + 1]``), which a padded batch produces routinely and which
    ``repeat_interleave`` would handle only by accident.
    """
    if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
        raise ValueError(f"cu_seqlens must be 1D with >= 2 entries, got {tuple(cu_seqlens.shape)}")
    if int(cu_seqlens[-1]) != total:
        raise ValueError(f"cu_seqlens ends at {int(cu_seqlens[-1])} but the packed axis is {total}")
    pos = torch.arange(total, device=cu_seqlens.device)
    ids = torch.searchsorted(cu_seqlens[1:].contiguous(), pos, right=True)
    return ids.to(torch.int32)


def triton_sparse_gqa_attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    scaling: float | None = None,
    causal: bool = True,
    block_k: int = 64,
    precision: str = "ieee",
    num_warps: int = 4,
    seq_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Indexer-driven sparse attention on Triton, packed varlen layout.

    Sequences are concatenated along one axis with no padding, so a batch of mixed lengths
    costs what it weighs. Per-sequence query and key lengths may differ (prefill a long
    prompt while decoding another), and each sequence is bottom-right aligned within itself.

    Parameters
    ----------
    q : torch.Tensor
        Packed queries ``(total_q, H, D)``.
    k, v : torch.Tensor
        Packed keys ``(total_k, Hkv, D)`` and values ``(total_k, Hkv, Dv)``.
    indices : torch.Tensor
        ``(total_q, Hkv, topk)`` int32, **sequence-local** positions with ``-1`` for empty
        slots: slot value ``j`` means row ``cu_seqlens_k[seq] + j``. Local rather than global
        so the indexer needs no remapping and a sequence can be relocated in the buffer
        without rewriting its indices.
    cu_seqlens_q, cu_seqlens_k : torch.Tensor
        ``(n_seq + 1,)`` int32 cumulative lengths, starting at 0.
    seq_ids : torch.Tensor, optional
        Precomputed token -> sequence map from :func:`seq_ids_from_cu_seqlens`. Pass it to
        skip the ``searchsorted`` when the layout is reused across layers.

    Returns
    -------
    out : torch.Tensor
        ``(total_q, H, Dv)`` in ``q``'s dtype.
    lse : torch.Tensor
        ``(total_q, H)`` fp32; ``-inf`` where a row had no valid slot.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available; use sparse_gqa_attention_varlen_reference")
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(
            "varlen expects packed 3D q (total_q, H, D), k/v (total_k, Hkv, D); got "
            f"{tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}"
        )
    if indices.dim() != 3:
        raise ValueError(f"varlen indices must be (total_q, Hkv, topk), got {tuple(indices.shape)}")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError(
            f"cu_seqlens_q has {cu_seqlens_q.numel()} entries, cu_seqlens_k has "
            f"{cu_seqlens_k.numel()}; they must describe the same sequences"
        )
    check_block_k(block_k)

    total_q, n_heads, dim = q.shape
    total_k, n_kv_heads, dim_k = k.shape
    if dim_k != dim:
        raise ValueError(f"q head_dim {dim} != k head_dim {dim_k}")
    if v.shape[0] != total_k or v.shape[1] != n_kv_heads:
        raise ValueError(f"v must be (total_k={total_k}, Hkv={n_kv_heads}, Dv), got {tuple(v.shape)}")
    if n_heads % n_kv_heads != 0:
        raise ValueError(f"n_heads {n_heads} is not a multiple of n_kv_heads {n_kv_heads}")
    if indices.shape[:2] != (total_q, n_kv_heads):
        raise ValueError(
            f"indices must be (total_q={total_q}, Hkv={n_kv_heads}, topk), got {tuple(indices.shape)}"
        )

    cu_q = cu_seqlens_q.to(device=q.device, dtype=torch.int32).contiguous()
    cu_k = cu_seqlens_k.to(device=q.device, dtype=torch.int32).contiguous()
    if seq_ids is None:
        seq_ids = seq_ids_from_cu_seqlens(cu_q, total_q)
    seq_ids = seq_ids.to(device=q.device, dtype=torch.int32).contiguous()

    out = torch.empty((total_q, n_heads, v.shape[-1]), dtype=q.dtype, device=q.device)
    lse = torch.empty((total_q, n_heads), dtype=torch.float32, device=q.device)
    _launch(
        q, k, v, indices.contiguous(), out, lse,
        seq_id=seq_ids, cu_q=cu_q, cu_k=cu_k,
        k_len_static=0, query_offset_static=0,
        scale=resolve_scaling(dim, scaling),
        n_kv_heads=n_kv_heads, group_size=n_heads // n_kv_heads, topk=indices.shape[-1],
        causal=causal, block_k=block_k, precision=precision, num_warps=num_warps,
    )
    return out, lse


def sparse_gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    *,
    scaling: float | None = None,
    query_offset: int | None = None,
    causal: bool = True,
    force_reference: bool = False,
    **kernel_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Indexer-driven sparse attention, Triton where possible and PyTorch otherwise.

    The fallback is the gather reference, which is ``O(topk * D)`` per row in *memory* as
    well as in FLOPs -- correct, but not a substitute at real lengths. ``force_reference``
    exists for tests and for isolating a suspected kernel bug.
    """
    if force_reference or not sparse_kernels_available(q, k, v):
        return sparse_gqa_attention_reference(
            q, k, v, indices, scaling=scaling, query_offset=query_offset, causal=causal
        )
    return triton_sparse_gqa_attention(
        q, k, v, indices,
        scaling=scaling, query_offset=query_offset, causal=causal, **kernel_kwargs,
    )
