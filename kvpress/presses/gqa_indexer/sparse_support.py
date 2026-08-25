# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Stage-2 support selection: which keys the sparse objective is defined over.

This is pass 1 of the two-pass stage-2 scheme. It runs the indexer under ``no_grad`` to
decide each query row's top-k support, then hands the *indices* to
:mod:`kvpress.presses.gqa_indexer.fused_sparse_loss`, which recomputes with gradients on
only those ``topk`` keys. Because the output is ``(B, h, Sq, topk)`` int32 and nothing else,
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

    Returns ``(support, valid)`` with ``-1`` in the empty slots. ``support`` is **int32**:
    it only has to address ``Sk``, and it is the single largest retained tensor in stage 2
    (83% of retained bytes at ``topk=2048``), so int64 doubles the dominant term for no reach.
    ``valid`` is exactly ``support >= 0`` and is returned for convenience -- callers that
    keep tensors alive across ``backward()`` should recompute it instead of storing it.
    """
    if k_len > torch.iinfo(torch.int32).max:
        raise ValueError(
            f"k_len={k_len} exceeds int32; the support tensor would silently wrap. "
            "Widen sort_support to int64 if sequences this long are real."
        )
    filled = torch.where(support >= 0, support, torch.full_like(support, k_len))
    filled, _ = filled.sort(dim=-1)
    valid = filled < k_len
    support = torch.where(valid, filled, torch.full_like(filled, -1))
    return support.to(torch.int32), valid


#: Scratch budget for the running top-k, in candidate elements per ``(batch, head)``. The
#: default tiling is derived from this rather than fixed, so ``key_tile`` grows with ``take``
#: instead of leaving a constant 512 to be re-sorted against a much larger buffer.
#:
#: Raised from 2M to 16M after measuring what the scratch actually costs: the candidate buffer is
#: ``query_tile * (take + key_tile)`` elements of bf16, so even the single-pass extreme
#: (``key_tile = k_len = 32768``, ``query_tile = 512``) is ~0.3 GiB -- immaterial next to the model
#: and KV cache, while the old 2M forced ``query_tile`` down to 57 at 32K and thus 575 sequential
#: launches per layer. The budget exists to bound scratch, not to be the binding constraint on
#: launch count; 16M puts the knee where throughput wants it and still caps a pathological shape.
TOPK_SCRATCH_BUDGET = 16_000_000


def topk_tiles(take: int, k_len: int, q_len: int, budget: int = TOPK_SCRATCH_BUDGET) -> tuple[int, int]:
    """
    Choose ``(key_tile, query_tile)`` for the running top-k.

    The running buffer is re-sorted against every key tile, so total work is
    ``Sq * Sk * (1 + take / key_tile)`` -- note that ``query_tile`` **cancels** out of that
    expression. It costs scratch, not work. That asymmetry is the basis of this function: push
    ``key_tile`` up to shrink the ``take / key_tile`` redundancy, and pay for the resulting
    ``query_tile * (take + key_tile)`` scratch out of the budget.

    ``query_tile`` cancelling out of the *work* model does not make it free in *time*, though:
    each query tile is a separate ``topk`` launch, and at ``L=8030, topk=2048`` single-pass
    selection measured 61.2 ms at ``query_tile=64`` against 22.7 ms at 1598 -- a 2.7x spread over
    identical work. Hence ``min_query_tile`` below, and hence the budget being spent on a *wide*
    query tile rather than shaved to the minimum.

    A fixed ``key_tile = 512`` against ``take = 1980`` carries a 4.87x redundancy factor;
    measured on an H20 at ``L=16384``, raising it to 4096 cut selection from 26.7 s to 6.5 s
    (4.1x). Taking that argument to its conclusion, ``key_tile`` is now sized to cover the
    **whole key axis** when the budget allows: the tournament merge then disappears entirely
    (redundancy exactly 1.0x, one ``topk`` per query tile instead of ``ceil(Sk / key_tile)``).
    Re-measured per layer at ``take=1980``: ``L=8030`` 29.7 ms -> 22.8 ms (1.30x), ``L=16384``
    111.8 ms -> 65.9 ms (**1.70x**). Only when a single pass cannot afford ``min_query_tile`` rows
    does it fall back to ~2x ``take`` (~1.5x redundancy), the previous knee.

    Returns tiles capped at the real extents, so short sequences do not allocate for length they
    do not have.
    """
    take = max(1, int(take))
    k_len = max(int(k_len), 1)
    #: Query rows per tile below which launch overhead starts to dominate. ``query_tile`` cancels
    #: out of the *work* model, but not out of the measured time: at ``L=8030, topk=2048``,
    #: single-pass selection measured 61.2 ms at ``query_tile=64`` against 22.7 ms at 1598 (2.7x),
    #: because each tile is a separate topk launch over a narrow tensor. So a tiling is only
    #: worth choosing if it can afford a reasonably wide query tile.
    min_query_tile = 256
    # Prefer covering the whole key axis in one pass -- redundancy exactly 1.0x and a single topk
    # per query tile -- but only if the budget still affords a wide enough query tile. Otherwise
    # fall back to ~2x take (~1.5x redundancy), which buys a much wider query_tile per element.
    if min_query_tile * (take + k_len) <= budget:
        key_tile = k_len
    else:
        # Round 2*take up to a power of two: ~1.5x redundancy, the previous measured knee.
        key_tile = min(max(512, 1 << (2 * take - 1).bit_length()), k_len)
    query_tile = min(max(min_query_tile, budget // max(take + key_tile, 1)), max(q_len, 1))
    return key_tile, query_tile


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
    key_tile: int | None = None,
    query_tile: int | None = None,
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
    key_tile, query_tile : int, optional
        Tile sizes; peak scratch is ``O(query_tile * (take + key_tile))``. Both default to
        :func:`topk_tiles`, which sizes ``key_tile`` against ``take`` -- a constant ``key_tile``
        makes the running buffer dominate the sort as ``topk`` grows. Pass them to override.

    Returns
    -------
    support : torch.Tensor
        ``(B, h, Sq, topk)`` **int32** key indices, ascending, ``-1`` for empty slots.
    valid : torch.Tensor
        ``(B, h, Sq, topk)`` bool, ``support >= 0``.
    """
    if q_idx.dim() != 4:
        raise ValueError(f"q_idx must be (B, h, Sq, D), got {tuple(q_idx.shape)}")
    if k_idx.dim() != 3:
        raise ValueError(f"k_idx must be (B, Sk, D), got {tuple(k_idx.shape)}")
    if (key_tile is not None and key_tile <= 0) or (query_tile is not None and query_tile <= 0):
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

    # Resolved here rather than in the signature because the useful tiling depends on `take`,
    # which is only known after topk has been clamped to k_len.
    default_key_tile, default_query_tile = topk_tiles(take, k_len, q_len)
    if key_tile is None:
        key_tile = default_key_tile
    if query_tile is None:
        query_tile = default_query_tile

    # int32 throughout: these only have to address Sk (guarded in sort_support), and the
    # (B, h, Sq, topk) buffer is the largest tensor here -- 0.98 GiB at the eval's
    # (8 heads, Sq=8030, topk=2048) in int64 against 0.49 in int32. The `.to(torch.int32)` that
    # sparse_inference used to pay on the int64 result then disappears too.
    support = torch.full((bsz, n_heads, q_len, topk), -1, dtype=torch.int32, device=device)
    k_index_all = torch.arange(k_len, device=device)
    # Scalar -inf of the score dtype, so `torch.where` broadcasts it instead of the caller
    # allocating a full_like per tile.
    neg_inf = torch.tensor(-float("inf"), device=device, dtype=q_idx.dtype)

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
            forced = forced.to(torch.int32).expand(bsz, n_heads, dq, n_forced).clone()
            if mask is not None:
                # A forced position can still be padding; read the mask only at those
                # positions, which is O(Sq * n_forced) rather than O(Sq * Sk).
                keep = mask[..., q_start:q_stop, :] > (MASK_NEG / 2)  # (B, 1, dq, Sk)
                keep = keep.expand(bsz, n_heads, dq, k_len)
                # gather requires an int64 index, so widen just this (Sq x n_forced) read.
                allowed = keep.gather(-1, forced.clamp_min(0).long())
                forced = torch.where(allowed, forced, torch.full_like(forced, -1))
            slots.append(forced)

        if take > 0:
            # The running tournament: `best_v` holds the best `take` scores seen so far and
            # `best_i` their absolute key indices. Both are None until the first tile, so the
            # single-pass case (key_tile == k_len) never builds a candidate buffer at all.
            best_v = None
            best_i = None

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
                # `where` against a broadcast mask, rather than masked_fill on an expanded one:
                # expand_as materializes the full (B, h, dq, tile) bool before the fill.
                logits = torch.where(allowed, logits, neg_inf)

                # `sorted=False`: the intermediate order is never read -- the merge only needs the
                # top `take` as a *set*, and sort_support orders the survivors at the end. Measured
                # 1.5x cheaper than sorted=True at (dq=329, width=6076, take=1980).
                if best_v is None:
                    width = min(take, logits.shape[-1])
                    best_v, order = logits.topk(width, dim=-1, sorted=False)
                    # Tile-local rank -> absolute key index.
                    best_i = order.to(torch.int32) + start
                else:
                    prev = best_v.shape[-1]
                    cand_v = torch.cat([best_v, logits], dim=-1)
                    width = min(take, cand_v.shape[-1])
                    best_v, order = cand_v.topk(width, dim=-1, sorted=False)
                    # Recover the winners' key indices by *arithmetic* on `order` instead of
                    # gathering from a materialized candidate-index tensor: `order < prev` came
                    # from the running buffer, and the rest is the new tile's (order - prev)-th
                    # key. That drops a (take + key_tile)-wide int64 cat + gather per tile, which
                    # measured 24% of the loop -- and it is exact, not an approximation.
                    best_i = torch.where(
                        order < prev,
                        best_i.gather(-1, order.clamp(max=max(prev - 1, 0))),
                        (order - prev).to(torch.int32) + start,
                    )

            # -inf survives only where the pool ran out of eligible keys.
            best_i = torch.where(torch.isfinite(best_v), best_i, torch.full_like(best_i, -1))
            slots.append(best_i)

        support[:, :, q_start:q_stop] = torch.cat(slots, dim=-1)

    return sort_support(support, k_len)


def gather_support_keys(keys: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """
    Gather per-row support keys.

    ``keys`` is ``(B, Sk, D)`` (the indexer's single MQA key) or ``(B, h, Sk, D)`` (teacher
    keys); ``support`` is ``(B, h, dq, tk)``, int32 or int64. Returns ``(B, h, dq, tk, D)``.

    The ``keys`` tensor is ``expand``-ed rather than repeated, so only the gathered output
    is materialized. That output is ``O(query_tile * topk_tile * D)`` -- note the ``D``, which
    makes it multi-GiB at 512x512x128 and is stage 2's dominant transient. A real kernel does
    this gather into shared memory; here shrinking ``topk_tile`` is the lever.

    ``support`` is stored as int32 to halve the largest *retained* tensor, but ``gather``
    requires int64, so the cast happens here -- once, on a tile-sized slice, rather than on the
    full ``(B, h, L, topk)`` tensor.
    """
    bsz, n_heads, dq, tk = support.shape
    if keys.dim() == 3:
        keys = keys.unsqueeze(1)
    if keys.shape[1] == 1:
        keys = keys.expand(bsz, n_heads, keys.shape[2], keys.shape[3])
    elif keys.shape[1] != n_heads:
        raise ValueError(f"keys has {keys.shape[1]} heads, support has {n_heads}")
    dim = keys.shape[-1]
    flat = support.clamp_min(0).long().reshape(bsz, n_heads, dq * tk, 1).expand(-1, -1, -1, dim)
    return keys.gather(2, flat).reshape(bsz, n_heads, dq, tk, dim)
