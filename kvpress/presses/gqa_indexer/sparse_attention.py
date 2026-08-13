# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Indexer-driven sparse attention: the *inference* counterpart of the indexer.

The press evicts -- it drops keys from the cache, so every query afterwards sees the same
reduced cache. This module does the other thing the indexer enables, and the thing DSA
actually ships: keep the cache whole and let **each query attend to its own top-k keys**.
Nothing is discarded, so a key one query ignored is still there for the next one; the saving
is in the attention FLOPs and the score-matrix bandwidth rather than in cache residency.

That difference is why this path needs a kernel at all. Eviction ends with a smaller dense
cache, which the existing dense attention handles unchanged. Per-query selection instead
produces a *gather* -- every (query, kv-head) row reads a different ``topk`` slice of the
cache -- and no dense kernel expresses that.

Head granularity
----------------
:class:`~.indexer.GQAIndexer` emits one score per **KV head** (``n_heads ==
num_key_value_heads``), so the selection is per KV head, not shared across the model. This is
the substantive difference from MLA DSA, where a single shared latent cache forces one top-k
list for all heads. In GQA the KV caches are physically separate, so each KV head selects
freely; the ``group_size = num_heads // num_key_value_heads`` query heads that read a given
KV head share that head's list, because they share the cache it indexes.

Practically this sets the kernel's tile shape. The MLA kernels put 64-128 heads in the ``M``
dimension of every GEMM, all sharing one gathered KV tile. Here only ``group_size`` query
heads (typically 4-8) share a tile, so ``M`` is small and the arithmetic intensity of the
``Q @ K^T`` GEMM is correspondingly lower. It cannot be fixed by tiling over query tokens:
adjacent queries hold *different* index lists, so they cannot share a gathered tile. That is
inherent to per-query selection under GQA, not a limitation of this implementation.

Index convention
----------------
Deliberately the one :func:`~.sparse_support.sort_support` already emits, so
:func:`~.sparse_support.streaming_topk_support` feeds this directly with no adapter:

``(B, n_kv_heads, Sq, topk)`` int32, ascending within a row, ``-1`` in unused slots.

``-1`` rather than ``Sk``-as-sentinel (tilelang's choice) because ``-1`` is checkable without
knowing ``Sk`` and cannot be confused with a real position; it matches Megatron's DSA and
sglang. A row may be *entirely* ``-1`` (padding can produce one); see "Empty rows".

Empty rows
----------
A row with no valid slot has ``sumexp == 0``, and the natural ``out = acc / sumexp`` is
``0/0 = NaN`` -- which would then propagate through the whole model rather than staying local.
Both paths here define such a row as ``out = 0`` and ``lse = -inf`` instead. ``-inf`` is the
honest value (the log of an empty sum) and is what a downstream combine needs to see in order
to give the row zero weight; ``0`` for the output keeps the tensor finite.

Duplicate indices are summed with multiplicity by both the reference and the kernel, so they
agree. ``sort_support`` cannot emit duplicates -- the forced sink/local block and the top-k
pool are disjoint by construction -- so this is a statement about robustness, not a case that
arises in practice.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def resolve_scaling(head_dim: int, scaling: float | None) -> float:
    """Softmax scale, defaulting to ``head_dim ** -0.5``."""
    return head_dim**-0.5 if scaling is None else float(scaling)


def check_sparse_shapes(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, indices: torch.Tensor
) -> tuple[int, int, int, int, int, int, int]:
    """
    Validate the batched (non-varlen) shapes and return the geometry.

    Returns ``(bsz, n_heads, n_kv_heads, group_size, q_len, k_len, topk)``.

    The checks are eager and specific because every one of these mismatches is otherwise
    silent: a wrong ``group_size`` mixes up which KV head a query head reads, and a
    ``k_len`` disagreement between ``k`` and ``v`` reads values from the wrong rows. Neither
    crashes; both just return wrong numbers.
    """
    if q.dim() != 4:
        raise ValueError(f"q must be (B, H, Sq, D), got {tuple(q.shape)}")
    if k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"k and v must be (B, Hkv, Sk, D), got {tuple(k.shape)}, {tuple(v.shape)}")
    if indices.dim() != 4:
        raise ValueError(f"indices must be (B, Hkv, Sq, topk), got {tuple(indices.shape)}")

    bsz, n_heads, q_len, head_dim = q.shape
    bsz_k, n_kv_heads, k_len, head_dim_k = k.shape
    bsz_v, n_kv_heads_v, k_len_v, _ = v.shape

    if bsz_k != bsz or bsz_v != bsz:
        raise ValueError(f"batch mismatch: q={bsz}, k={bsz_k}, v={bsz_v}")
    if n_kv_heads_v != n_kv_heads:
        raise ValueError(f"k has {n_kv_heads} heads, v has {n_kv_heads_v}")
    if k_len_v != k_len:
        raise ValueError(f"k has {k_len} keys, v has {k_len_v}")
    if head_dim_k != head_dim:
        raise ValueError(f"q head_dim {head_dim} != k head_dim {head_dim_k}")
    if n_heads % n_kv_heads != 0:
        raise ValueError(f"n_heads {n_heads} is not a multiple of n_kv_heads {n_kv_heads}")

    idx_b, idx_h, idx_q, topk = indices.shape
    if (idx_b, idx_h, idx_q) != (bsz, n_kv_heads, q_len):
        raise ValueError(
            f"indices must be (B={bsz}, Hkv={n_kv_heads}, Sq={q_len}, topk), "
            f"got {tuple(indices.shape)}"
        )

    return bsz, n_heads, n_kv_heads, n_heads // n_kv_heads, q_len, k_len, topk


def slot_validity(
    indices: torch.Tensor,
    k_len: int,
    *,
    q_start: int = 0,
    query_offset: int = 0,
    causal: bool = True,
) -> torch.Tensor:
    """
    Which slots of ``indices`` may contribute, ``(..., dq, topk)`` bool.

    A slot is valid when it holds a real position (``>= 0``), that position is in range
    (``< k_len``), and -- when ``causal`` -- it does not reach past the query's own diagonal
    at ``q + query_offset``.

    The causal test is redundant for indices from
    :func:`~.sparse_support.streaming_topk_support`, which already applies it during
    selection. It is kept because it is nearly free and turns "the caller handed us an index
    list that peeks at the future" from a silent correctness bug into no effect at all. Pass
    ``causal=False`` for a genuinely bidirectional use.

    ``q_start`` offsets the query rows of this slice within the full query axis, so the
    function works on a tile.
    """
    valid = (indices >= 0) & (indices < k_len)
    if causal:
        dq = indices.shape[-2]
        q_pos = torch.arange(q_start, q_start + dq, device=indices.device) + query_offset
        valid = valid & (indices <= q_pos.unsqueeze(-1))
    return valid


def sparse_gqa_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    *,
    scaling: float | None = None,
    query_offset: int | None = None,
    causal: bool = True,
    query_tile: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gather-based reference for indexer-driven sparse attention.

    The straightforward reading of the operation: gather each row's ``topk`` keys and values,
    then run an ordinary dense softmax over that gathered axis. Correctness is meant to be
    obvious by inspection, so this is the thing the kernel is checked against.

    Parameters
    ----------
    q : torch.Tensor
        Queries ``(B, H, Sq, D)``.
    k, v : torch.Tensor
        Keys ``(B, Hkv, Sk, D)`` and values ``(B, Hkv, Sk, Dv)``. ``Dv`` may differ from ``D``.
    indices : torch.Tensor
        ``(B, Hkv, Sq, topk)`` int32/int64 selected key positions, ``-1`` for empty slots.
    scaling : float, optional
        Softmax scale; defaults to ``D ** -0.5``.
    query_offset : int, optional
        Key index of query 0's diagonal. Defaults to ``Sk - Sq`` (bottom-right alignment,
        matching :func:`~.indexer.build_indexer_mask` and flash-attention).
    causal : bool
        Apply the causal check to each slot; see :func:`slot_validity`.
    query_tile : int
        Queries processed at once. The gathered tensors are ``O(B * Hkv * query_tile * topk *
        D)``, which is the reference's dominant cost -- this is the knob that bounds it.

    Returns
    -------
    out : torch.Tensor
        ``(B, H, Sq, Dv)`` in ``q``'s dtype.
    lse : torch.Tensor
        ``(B, H, Sq)`` fp32 log-sum-exp of the scaled logits over the selected keys; ``-inf``
        for a row with no valid slot.
    """
    bsz, n_heads, n_kv_heads, group_size, q_len, k_len, topk = check_sparse_shapes(q, k, v, indices)
    scale = resolve_scaling(q.shape[-1], scaling)
    if query_offset is None:
        query_offset = k_len - q_len
    if query_tile <= 0:
        raise ValueError(f"query_tile must be positive, got {query_tile}")

    dim_v = v.shape[-1]
    # fp32 throughout: the softmax and the P@V accumulation are exactly what the kernel keeps
    # in fp32, so a bf16 reference would be measuring its own rounding, not the kernel's.
    q_f32 = q.float().view(bsz, n_kv_heads, group_size, q_len, q.shape[-1])
    k_f32, v_f32 = k.float(), v.float()

    out = torch.empty((bsz, n_kv_heads, group_size, q_len, dim_v), dtype=torch.float32, device=q.device)
    lse = torch.empty((bsz, n_kv_heads, group_size, q_len), dtype=torch.float32, device=q.device)

    for start in range(0, q_len, query_tile):
        stop = min(start + query_tile, q_len)
        idx = indices[:, :, start:stop]  # (B, Hkv, dq, topk)
        valid = slot_validity(
            idx, k_len, q_start=start, query_offset=query_offset, causal=causal
        )  # (B, Hkv, dq, topk)

        # Route every invalid slot to row 0 so the gather is in range -- this must cover
        # `idx >= k_len` too, not just the -1 sentinel, or an out-of-range index raises here
        # while the kernel (which masks it) returns a number. Whatever is read at a masked
        # slot is irrelevant: `valid` drives it to -inf below. Same expression the kernel
        # uses, so the two agree on which row an invalid slot nominally touches.
        safe = torch.where(valid, idx, torch.zeros_like(idx))
        flat = safe.long().reshape(bsz, n_kv_heads, -1, 1)
        k_sel = k_f32.gather(2, flat.expand(-1, -1, -1, k.shape[-1]))
        k_sel = k_sel.reshape(bsz, n_kv_heads, stop - start, topk, k.shape[-1])
        v_sel = v_f32.gather(2, flat.expand(-1, -1, -1, dim_v))
        v_sel = v_sel.reshape(bsz, n_kv_heads, stop - start, topk, dim_v)

        logits = torch.einsum("bhgqd,bhqkd->bhgqk", q_f32[:, :, :, start:stop], k_sel) * scale
        logits = logits.masked_fill(~valid.unsqueeze(2), -float("inf"))

        # A fully masked row would make softmax emit NaN (it divides by a zero sumexp), so
        # the row max is taken with a finite stand-in and the empty rows are zeroed after.
        row_max = logits.amax(dim=-1, keepdim=True)
        alive = torch.isfinite(row_max)
        safe_max = torch.where(alive, row_max, torch.zeros_like(row_max))
        p = torch.exp(logits - safe_max)
        sumexp = p.sum(dim=-1, keepdim=True)
        p = torch.where(alive, p / sumexp.clamp_min(torch.finfo(torch.float32).tiny), p * 0.0)

        out[:, :, :, start:stop] = torch.einsum("bhgqk,bhqkd->bhgqd", p, v_sel)
        row_lse = torch.log(sumexp) + safe_max
        lse[:, :, :, start:stop] = torch.where(alive, row_lse, torch.full_like(row_lse, -float("inf"))).squeeze(-1)

    return (
        out.reshape(bsz, n_heads, q_len, dim_v).to(q.dtype),
        lse.reshape(bsz, n_heads, q_len),
    )


def sparse_gqa_attention_dense_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    *,
    scaling: float | None = None,
    query_offset: int | None = None,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Second reference, via a scattered dense ``(B, Hkv, Sq, Sk)`` mask.

    Independent of :func:`sparse_gqa_attention_reference` in the way that matters: the
    selection becomes a boolean mask over the *full* key axis and the attention is an ordinary
    dense one, so nothing about the gather indexing is shared between the two. Agreement
    between them is therefore evidence about the operation's definition rather than about one
    implementation of it -- and the gather's index arithmetic (the part easiest to get subtly
    wrong, e.g. a reshape that transposes ``dq`` and ``topk``) is exactly what is not shared.

    ``O(Sq * Sk)`` in memory, so this is for tests, not for real sequence lengths.

    A duplicated index collapses under ``scatter`` and so is counted **once** here, against
    the gather path's multiplicity. Compare the two only on duplicate-free indices --
    :func:`~.sparse_support.sort_support` never emits duplicates.
    """
    bsz, n_heads, n_kv_heads, group_size, q_len, k_len, _ = check_sparse_shapes(q, k, v, indices)
    scale = resolve_scaling(q.shape[-1], scaling)
    if query_offset is None:
        query_offset = k_len - q_len

    valid = slot_validity(indices, k_len, query_offset=query_offset, causal=causal)
    # Route every invalid slot into a scratch column at k_len, then drop that column. This
    # avoids having to pick a real column to absorb them, which would corrupt it.
    target = torch.where(valid, indices.long(), torch.full_like(indices, k_len, dtype=torch.long))
    keep = torch.zeros((bsz, n_kv_heads, q_len, k_len + 1), dtype=torch.bool, device=q.device)
    keep.scatter_(3, target, True)
    keep = keep[..., :k_len]

    q_f32 = q.float().view(bsz, n_kv_heads, group_size, q_len, q.shape[-1])
    logits = torch.einsum("bhgqd,bhkd->bhgqk", q_f32, k.float()) * scale
    logits = logits.masked_fill(~keep.unsqueeze(2), -float("inf"))

    row_max = logits.amax(dim=-1, keepdim=True)
    alive = torch.isfinite(row_max)
    safe_max = torch.where(alive, row_max, torch.zeros_like(row_max))
    p = torch.exp(logits - safe_max)
    sumexp = p.sum(dim=-1, keepdim=True)
    p = torch.where(alive, p / sumexp.clamp_min(torch.finfo(torch.float32).tiny), p * 0.0)

    out = torch.einsum("bhgqk,bhkd->bhgqd", p, v.float())
    lse = torch.log(sumexp) + safe_max
    lse = torch.where(alive, lse, torch.full_like(lse, -float("inf"))).squeeze(-1)

    return out.reshape(bsz, n_heads, q_len, v.shape[-1]).to(q.dtype), lse.reshape(bsz, n_heads, q_len)


def sparse_gqa_attention_varlen_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    scaling: float | None = None,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference for the packed varlen layout: loop over sequences, reuse the batched reference.

    Deliberately a Python loop over sequences. It makes the varlen contract legible -- each
    sequence is an independent, bottom-right-aligned problem over its own KV slice -- and it
    reuses :func:`sparse_gqa_attention_reference` rather than restating the attention, so the
    only thing under test here is the packing arithmetic.

    ``indices`` are **sequence-local**: slot value ``j`` refers to row ``cu_seqlens_k[s] + j``
    of the packed KV. See :func:`~.triton_sparse_attention.triton_sparse_gqa_attention_varlen`.

    Parameters
    ----------
    q : torch.Tensor
        Packed queries ``(total_q, H, D)``.
    k, v : torch.Tensor
        Packed keys ``(total_k, Hkv, D)`` and values ``(total_k, Hkv, Dv)``.
    indices : torch.Tensor
        ``(total_q, Hkv, topk)`` int32, sequence-local, ``-1`` for empty slots.
    cu_seqlens_q, cu_seqlens_k : torch.Tensor
        ``(n_seq + 1,)`` cumulative lengths starting at 0.

    Returns
    -------
    out : torch.Tensor
        ``(total_q, H, Dv)`` in ``q``'s dtype.
    lse : torch.Tensor
        ``(total_q, H)`` fp32.
    """
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3 or indices.dim() != 3:
        raise ValueError(
            "varlen expects packed 3D tensors: q (total_q, H, D), k/v (total_k, Hkv, D), "
            f"indices (total_q, Hkv, topk); got {tuple(q.shape)}, {tuple(k.shape)}, "
            f"{tuple(v.shape)}, {tuple(indices.shape)}"
        )
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError(
            f"cu_seqlens_q has {cu_seqlens_q.numel()} entries, cu_seqlens_k has "
            f"{cu_seqlens_k.numel()}; they must describe the same sequences"
        )

    total_q, n_heads, _ = q.shape
    dim_v = v.shape[-1]
    out = torch.zeros((total_q, n_heads, dim_v), dtype=q.dtype, device=q.device)
    lse = torch.full((total_q, n_heads), -float("inf"), dtype=torch.float32, device=q.device)
    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()

    for s in range(len(cu_q) - 1):
        q0, q1, k0, k1 = cu_q[s], cu_q[s + 1], cu_k[s], cu_k[s + 1]
        if q1 == q0:  # an empty sequence contributes no rows
            continue
        # (dq, H, D) -> (1, H, dq, D); the batched reference owns the attention itself.
        o_s, l_s = sparse_gqa_attention_reference(
            q[q0:q1].permute(1, 0, 2).unsqueeze(0),
            k[k0:k1].permute(1, 0, 2).unsqueeze(0),
            v[k0:k1].permute(1, 0, 2).unsqueeze(0),
            indices[q0:q1].permute(1, 0, 2).unsqueeze(0),
            scaling=scaling,
            causal=causal,
        )
        out[q0:q1] = o_s[0].permute(1, 0, 2)
        lse[q0:q1] = l_s[0].permute(1, 0)

    return out, lse


def pack_varlen(
    tensors: list[torch.Tensor], dim: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Concatenate per-sequence tensors and return ``(packed, cu_seqlens)``.

    ``cu_seqlens`` is int32 on the tensors' device, starting at 0 -- the flash-attention
    convention the varlen kernel expects.
    """
    if not tensors:
        raise ValueError("pack_varlen needs at least one tensor")
    lengths = torch.tensor([t.shape[dim] for t in tensors], dtype=torch.int32)
    cu = torch.zeros(len(tensors) + 1, dtype=torch.int32, device=tensors[0].device)
    cu[1:] = lengths.cumsum(0).to(cu.device)
    return torch.cat(tensors, dim=dim), cu


def unpack_varlen(packed: torch.Tensor, cu_seqlens: torch.Tensor, dim: int = 0) -> list[torch.Tensor]:
    """Inverse of :func:`pack_varlen`: split a packed tensor back into per-sequence views."""
    cu = cu_seqlens.tolist()
    return [packed.narrow(dim, cu[s], cu[s + 1] - cu[s]) for s in range(len(cu) - 1)]
