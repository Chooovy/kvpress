# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Stage-2 support selection: which keys the sparse objective is defined over.

This is pass 1 of the two-pass stage-2 scheme. It runs the indexer under ``no_grad`` to
decide each query row's top-k support, then hands the *indices* to
:mod:`kvpress.presses.gqa_indexer.fused_sparse_loss`, which recomputes with gradients on
only those ``topk`` keys. Because the output is ``(B, h, Sq, topk)`` int64 and nothing else,
pass 1 contributes no autograd state: 8 MiB at ``L=32K, topk=512, h=8``.

Top-k is not differentiable, so treating the support as a constant is not an approximation
-- it is the only thing available. DSA and AngelPTM do the same.

Streaming
---------
:func:`streaming_topk_support` merges a running best-``take`` against each key tile
(tournament merge), so it never materializes an ``(Sq, Sk)`` score matrix. The selected
*values* are identical to a dense ``topk`` for every tile size; index tie-breaks among
exactly-equal logits may differ, which is harmless.

Forced positions
----------------
``force_sink`` / ``force_local`` reserve slots for the leading keys and each row's most
recent keys, mirroring MiniMax MSA's always-selected local block. They are handled by
*excluding* those positions from the top-k pool and concatenating them back, rather than by
biasing their logits: exclusion is magnitude-free, so it cannot be defeated by a large
logit and needs no sentinel that might overflow.

Note this is the per-query-row analogue of the press's ``n_sink``/``n_local``, not the same
operation. The press protects keys *globally* after the query axis has been reduced; here
every query row protects its own recent window, which is what a row-wise objective needs.
"""

from __future__ import annotations

import logging

import torch

from kvpress.presses.gqa_indexer.indexer import MASK_NEG

logger = logging.getLogger(__name__)


def resolve_topk(k_len: int, topk: int | None, keep_ratio: float) -> int:
    """Support size for a key axis of length ``k_len``, from an explicit ``topk`` or a ratio."""
    if topk is None:
        topk = max(1, int(k_len * keep_ratio))
    return max(1, min(int(topk), k_len))


def forced_support_positions(
    q_index: torch.Tensor, *, force_sink: int, force_local: int, query_offset: int, k_len: int
) -> torch.Tensor:
    """
    Positions every query row must keep, ``(dq, force_sink + force_local)`` int64.

    ``-1`` marks a slot the row cannot use: a sink beyond its causal horizon, a local
    position before the start of the sequence, or a local position the sink block already
    owns. Deduplication against the sink is exact because the sink block is precisely
    ``[0, force_sink)``, so ``pos >= force_sink`` is a complete test.

    Parameters
    ----------
    q_index : torch.Tensor
        Absolute query positions, ``(dq,)`` int64.
    force_sink, force_local : int
        Slot counts to reserve at the start of the sequence and at each row's own end.
    query_offset : int
        Key index of the diagonal for query 0; ``k_len - q_len`` for bottom-right causal
        alignment, matching :func:`~.indexer.build_indexer_mask` and flash-attention.
    k_len : int
        Key axis length, used to clamp the causal horizon.
    """
    device = q_index.device
    limit = (q_index + query_offset).clamp(max=k_len - 1)  # last visible key per row

    blocks = []
    if force_sink > 0:
        sink = torch.arange(force_sink, device=device).expand(q_index.shape[0], force_sink)
        blocks.append(torch.where(sink <= limit.unsqueeze(-1), sink, torch.full_like(sink, -1)))
    if force_local > 0:
        back = torch.arange(force_local - 1, -1, -1, device=device)
        local = limit.unsqueeze(-1) - back  # (dq, force_local), ascending
        usable = (local >= 0) & (local >= force_sink)
        blocks.append(torch.where(usable, local, torch.full_like(local, -1)))

    if not blocks:
        return torch.zeros((q_index.shape[0], 0), dtype=torch.long, device=device)
    return torch.cat(blocks, dim=-1)


def excluded_key_mask(
    q_index: torch.Tensor,
    k_index: torch.Tensor,
    *,
    force_sink: int,
    force_local: int,
    query_offset: int,
    k_len: int,
) -> torch.Tensor | None:
    """
    Keys the top-k pool must skip because the forced block already owns them, ``(dq, tile)``.

    Returns ``None`` when nothing is forced, letting the caller skip the mask entirely.
    """
    if force_sink <= 0 and force_local <= 0:
        return None
    limit = (q_index + query_offset).clamp(max=k_len - 1).unsqueeze(-1)
    keys = k_index.unsqueeze(0)
    excluded = torch.zeros((q_index.shape[0], k_index.shape[0]), dtype=torch.bool, device=q_index.device)
    if force_sink > 0:
        excluded |= keys < force_sink
    if force_local > 0:
        excluded |= keys > limit - force_local
    return excluded


def causal_keep(
    q_index: torch.Tensor, k_index: torch.Tensor, *, query_offset: int
) -> torch.Tensor:
    """Bottom-right causal validity for a (query tile, key tile) pair, ``(dq, tile)`` bool."""
    return k_index.unsqueeze(0) <= (q_index + query_offset).unsqueeze(-1)


def sort_support(support: torch.Tensor, k_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sort each row's support ascending, pushing empty slots to the end.

    Ascending key order makes the downstream gathers contiguous-ish and the result
    independent of the order top-k happened to emit -- the same determinism argument
    AngelPTM makes for sorting ``topk_indices`` before its sparse kernel.

    Returns ``(support, valid)`` with ``-1`` in the empty slots.
    """
    filled = torch.where(support >= 0, support, torch.full_like(support, k_len))
    filled, _ = filled.sort(dim=-1)
    valid = filled < k_len
    return torch.where(valid, filled, torch.full_like(filled, -1)), valid


@torch.no_grad()
def streaming_topk_support(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    topk: int,
    *,
    mask: torch.Tensor | None = None,
    query_offset: int | None = None,
    force_sink: int = 0,
    force_local: int = 0,
    key_tile: int = 512,
    query_tile: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pick each query row's top-``topk`` keys without materializing the score matrix.

    Parameters
    ----------
    q_idx : torch.Tensor
        Indexer queries after norm and RoPE, ``(B, h, Sq, D)``.
    k_idx : torch.Tensor
        Shared indexer key after norm and RoPE, ``(B, Sk, D)``.
    topk : int
        Support size, including the forced slots.
    mask : torch.Tensor, optional
        Additive ``(B, 1, Sq, Sk)`` mask. Pass ``None`` for pure causal masking, which
        avoids the ``O(Sq * Sk)`` mask tensor entirely -- the causal structure is then
        derived from ``query_offset`` arithmetic per tile.
    query_offset : int, optional
        Defaults to ``Sk - Sq`` (bottom-right alignment).
    force_sink, force_local : int
        Slots reserved for the leading and most-recent keys.
    key_tile, query_tile : int
        Tile sizes; peak scratch is ``O(query_tile * key_tile)``.

    Returns
    -------
    support : torch.Tensor
        ``(B, h, Sq, topk)`` int64 key indices, ascending, ``-1`` for empty slots.
    valid : torch.Tensor
        ``(B, h, Sq, topk)`` bool, ``support >= 0``.
    """
    if q_idx.dim() != 4:
        raise ValueError(f"q_idx must be (B, h, Sq, D), got {tuple(q_idx.shape)}")
    if k_idx.dim() != 3:
        raise ValueError(f"k_idx must be (B, Sk, D), got {tuple(k_idx.shape)}")
    if key_tile <= 0 or query_tile <= 0:
        raise ValueError(f"tile sizes must be positive, got key_tile={key_tile}, query_tile={query_tile}")

    bsz, n_heads, q_len, _ = q_idx.shape
    k_len = k_idx.shape[1]
    device = q_idx.device
    topk = max(1, min(int(topk), k_len))
    if query_offset is None:
        query_offset = k_len - q_len

    n_forced = force_sink + force_local
    if n_forced > topk:
        raise ValueError(
            f"force_sink + force_local = {n_forced} exceeds topk = {topk}; the forced keys "
            "would be silently truncated. Lower them, or raise topk / keep_ratio."
        )
    take = topk - n_forced

    support = torch.full((bsz, n_heads, q_len, topk), -1, dtype=torch.long, device=device)
    k_index_all = torch.arange(k_len, device=device)

    for q_start in range(0, q_len, query_tile):
        q_stop = min(q_start + query_tile, q_len)
        dq = q_stop - q_start
        q_index = torch.arange(q_start, q_stop, device=device)
        q_view = q_idx[:, :, q_start:q_stop]

        slots = []
        if n_forced > 0:
            forced = forced_support_positions(
                q_index,
                force_sink=force_sink,
                force_local=force_local,
                query_offset=query_offset,
                k_len=k_len,
            )  # (dq, n_forced)
            forced = forced.expand(bsz, n_heads, dq, n_forced).clone()
            if mask is not None:
                # A forced position can still be padding; read the mask only at those
                # positions, which is O(Sq * n_forced) rather than O(Sq * Sk).
                keep = mask[..., q_start:q_stop, :] > (MASK_NEG / 2)  # (B, 1, dq, Sk)
                keep = keep.expand(bsz, n_heads, dq, k_len)
                allowed = keep.gather(-1, forced.clamp_min(0))
                forced = torch.where(allowed, forced, torch.full_like(forced, -1))
            slots.append(forced)

        if take > 0:
            best_v = torch.full((bsz, n_heads, dq, 0), -float("inf"), device=device, dtype=q_idx.dtype)
            best_i = torch.zeros((bsz, n_heads, dq, 0), dtype=torch.long, device=device)

            for start in range(0, k_len, key_tile):
                stop = min(start + key_tile, k_len)
                k_index = k_index_all[start:stop]
                logits = torch.einsum("bhqd,bkd->bhqk", q_view, k_idx[:, start:stop])

                if mask is not None:
                    allowed = mask[..., q_start:q_stop, start:stop] > (MASK_NEG / 2)
                else:
                    allowed = causal_keep(q_index, k_index, query_offset=query_offset)
                    allowed = allowed.unsqueeze(0).unsqueeze(0)
                skip = excluded_key_mask(
                    q_index,
                    k_index,
                    force_sink=force_sink,
                    force_local=force_local,
                    query_offset=query_offset,
                    k_len=k_len,
                )
                if skip is not None:
                    allowed = allowed & ~skip
                logits = logits.masked_fill(~allowed.expand_as(logits), -float("inf"))

                cand_v = torch.cat([best_v, logits], dim=-1)
                cand_i = torch.cat([best_i, k_index.expand(bsz, n_heads, dq, stop - start)], dim=-1)
                width = min(take, cand_v.shape[-1])
                best_v, order = cand_v.topk(width, dim=-1)
                best_i = cand_i.gather(-1, order)

            # -inf survives only where the pool ran out of eligible keys.
            best_i = torch.where(torch.isfinite(best_v), best_i, torch.full_like(best_i, -1))
            slots.append(best_i)

        support[:, :, q_start:q_stop] = torch.cat(slots, dim=-1)

    return sort_support(support, k_len)


def gather_support_keys(keys: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """
    Gather per-row support keys.

    ``keys`` is ``(B, Sk, D)`` (the indexer's single MQA key) or ``(B, h, Sk, D)`` (teacher
    keys); ``support`` is ``(B, h, dq, tk)``. Returns ``(B, h, dq, tk, D)``.

    The ``keys`` tensor is ``expand``-ed rather than repeated, so only the gathered output
    is materialized -- ``O(query_tile * topk_tile)`` per tile. A real kernel does this gather
    into shared memory; here it is the dominant term in stage 2's footprint, which is why
    stage 2 wants smaller tiles than stage 1.
    """
    bsz, n_heads, dq, tk = support.shape
    if keys.dim() == 3:
        keys = keys.unsqueeze(1)
    if keys.shape[1] == 1:
        keys = keys.expand(bsz, n_heads, keys.shape[2], keys.shape[3])
    elif keys.shape[1] != n_heads:
        raise ValueError(f"keys has {keys.shape[1]} heads, support has {n_heads}")
    dim = keys.shape[-1]
    flat = support.clamp_min(0).reshape(bsz, n_heads, dq * tk, 1).expand(-1, -1, -1, dim)
    return keys.gather(2, flat).reshape(bsz, n_heads, dq, tk, dim)
