# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Query-independent sparse attention as a **block-sparse mask** on ``flex_attention``.

The gather path (:mod:`~.triton_sparse_attention`) materializes each query row's ``topk`` key
indices and reads ``k``/``v`` through them. That is necessary when the scorer is pairwise, but it is
the wrong shape for a *query-independent* scorer, and measurably so: the gather is
bandwidth-bound -- 62.7 GiB per layer at ``L=8030, topk=2048``, running at ~1370 GiB/s, i.e. already
at the hardware limit -- so no amount of batching or tiling helps (measured: 1.00x per sequence at
batch 1/2/4).

The structural fact this module exploits
----------------------------------------
When the score is a fixed per-key vector (``ScalarIndexer.is_query_independent``), each key is
selected by **one contiguous interval of query rows**. Verified directly against
:func:`~.sparse_support.streaming_topk_support`'s output: 0 keys with a gap, over every shape tried.

The reason is the irreversibility :mod:`~.scalar_indexer` documents (citing SparseK). Query row
``t``'s top-k pool is ``[force_sink, hi_t]`` with ``hi_t = min(t + query_offset, Sk-1) -
force_local``, which only *grows* with ``t``. Key ``j`` is taken iff fewer than ``take`` pool keys
beat it, and that count is non-decreasing in ``t`` -- so once ``j`` drops out it never returns. One
scalar per key therefore describes its whole row set:

    ``deadline[j]`` -- the last ``hi`` at which ``j`` is still inside the top-``take``.

Selection then collapses from an ``O(Sq * Sk)`` score matrix plus an ``O(Sq * Sk log)`` top-k into
an ``O(L log L)`` precomputation plus a handful of integer comparisons inside ``mask_mod``. No
support tensor (0.49 GiB at ``L=8030``), no final sort of it, no gathers -- ``flex_attention`` reads
``k``/``v`` contiguously and ``create_block_mask`` skips whole 128x128 blocks (measured 48.8%
skipped).

Measured, H20, Qwen3-8B geometry (H=32, Hkv=8, D=128), ``topk=2048``, per layer:

===========  ==================  ===================  =========
``L``        current path        this module          speedup
===========  ==================  ===================  =========
4096         32.96 ms            7.76 ms              4.25x
8030         68.90 ms            29.82 ms             2.31x
===========  ==================  ===================  =========

``flex_attention`` itself is 6.52 ms of that 29.82 at ``L=8030``, against 46.24 ms for the gather
kernel -- the rest is the deadline precompute and the block mask.

Exactness
---------
The *selection* is exact, not an approximation. The mask reproduces
:func:`~.sparse_support.streaming_topk_support`'s selection entry for entry -- verified both in
``tests/presses/test_gqa_indexer_qi_flex.py`` and on the real model, where a 36-layer 7504-token
prefill agreed on **every (query, key) entry of every layer**. That includes the tie-break:
``deadlines`` orders equal scores by ascending key index, matching a stable descending sort. Ties are
not a corner case here -- a bf16 score resolves only ~12% distinct values at ``L=8030``, so 95% of
keys share a score with another key, and a different convention would select a different set.

The *arithmetic* is not bit-identical to the gather kernel, and cannot be: the two accumulate the
same softmax in a different order. Measured against an fp32 dense reference over the identical
support, both are equally accurate -- ``flex`` 2.07e-3 relative, gather ``tf32`` 2.04e-3, gather
``ieee`` 2.04e-3, all at or below bf16's 3.9e-3 epsilon -- while differing from each other by
5.0e-3. Under greedy decoding that is enough to change a token when two candidates are within
rounding, which on RULER's uuid/needle tasks showed up as 2 of 5 sampled rows producing a different
(not worse) answer. **So numbers from this path are comparable to the gather path's in aggregate but
not row-identical**; when A/B-ing the two, compare task scores, not individual generations.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

try:  # flex_attention landed in torch 2.5 and the block-mask API firmed up after
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    HAS_FLEX = True
except ImportError:  # pragma: no cover - depends on the installed torch
    create_block_mask = flex_attention = None
    HAS_FLEX = False

#: ``create_block_mask``'s granularity. A block is skipped only when *every* entry in it is masked,
#: so this is the resolution at which sparsity turns into saved work, not a correctness knob --
#: ``mask_mod`` is still evaluated per element inside a surviving block.
FLEX_BLOCK = 128

#: Compiled once per process. ``flex_attention`` is a torch.compile target; calling it eagerly
#: gives up the fused kernel that is the entire point.
#:
#: ``dynamic=None`` (torch's automatic mode) is deliberate and load-bearing **for
#: ``flex_attention``**. Every RULER context has a different token count, and the choice is not a
#: micro-optimization -- measured over 12 distinct lengths spanning 5905..11094, total wall clock for
#: the select+attend call:
#:
#: * ``dynamic=False``: **73.2 s**. Recompiles per length, hits dynamo's ``recompile_limit`` (8),
#:   then falls back to eager for the rest of the process -- so a sweep silently loses the speedup
#:   partway through, with per-call times climbing to 10-20 s.
#: * ``dynamic=True``: 4.93 s total, 86 ms steady state. One compile, but a slower kernel.
#: * ``dynamic=None``: **4.85 s** total, **45 ms** steady state. A couple of early recompiles, then
#:   the fastest of the three.
#:
#: So ``None`` wins on both total and steady state; ``False`` is a trap that looks fastest on a
#: single-length microbenchmark and is 15x slower on the real length mix.
_flex_compiled = None
_block_mask_compiled = None

#: ``create_block_mask`` is compiled with ``dynamic=True``, NOT ``None``, and that is a correctness
#: fix rather than a tuning choice. Under ``dynamic=None`` a multi-length run dies with
#:
#:     CUDA error: an illegal memory access was encountered
#:
#: inside an inductor-generated reduction (``triton_per_fused__to_copy_slice_sum_transpose_*``, i.e.
#: the block-wise reduction this call compiles to). Reproduced on RULER 8K at ``fraction=0.02`` and
#: on all 4 shards of a full run; the shards died at *different* contexts (45/163, 121/163, 117/162,
#: 109/162), which is the signature of a shape-sequence-dependent recompile rather than a bad index.
#:
#: It is specifically the specialise-then-generalise transition of automatic mode: ``deadlines`` and
#: :func:`qi_block_mask` both pass standalone at every RULER length (including non-multiples of
#: :data:`FLEX_BLOCK`) and across 60 sequential decode steps. Pinning **only this** callable to
#: ``dynamic=True`` makes the crashing run complete, and leaves ``flex_attention`` -- where the 45 ms
#: steady state lives -- untouched. Verified equal: ``block_mask=True/flex=None`` and
#: ``block_mask=True/flex=True`` produce identical RULER numbers on all 13 tasks, so the mode does
#: not change which keys are selected.
#:
#: Suspected upstream bug (torch 2.10.0-rc6). Recheck on a release build before reverting; if it is
#: fixed there, ``None`` is worth ~nothing here anyway -- the mask build is not the 45 ms.
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
    scores: torch.Tensor, topk: int, *, force_sink: int = 0, force_local: int = 0
) -> torch.Tensor:
    """
    Per-key eviction deadline for a query-independent score, ``(n_heads, Sk)`` int32.

    ``deadline[h, j]`` is the largest pool horizon ``hi`` for which key ``j`` still sits inside the
    top-``take`` of ``[force_sink, hi]``; ``Sk - 1`` means "never evicted" and ``-1`` means "never
    selected by the top-k" (which is the whole array when ``take <= 0``, i.e. the forced slots have
    consumed the entire budget). A query row with horizon ``hi_t`` keeps ``j`` iff
    ``hi_t <= deadline[h, j]`` -- so both sentinels fall out of the same comparison, since
    ``hi_t >= 0`` always.

    Parameters
    ----------
    scores : torch.Tensor
        ``(n_heads, Sk)`` per-key scores, position tilt already applied, padding already set to a
        very negative value. This is what
        :meth:`~.scalar_indexer.ScalarIndexer.score_keys` returns, minus the batch axis.
    topk, force_sink, force_local : int
        The support budget, matching the gather path's arguments. ``take = topk - force_sink -
        force_local`` is what the deadline actually governs.

    Notes
    -----
    Let ``ord`` be the pool sorted by score descending (ties by ascending key index -- a *stable*
    sort, which is what makes ``ord[0..r-1]`` exactly the set of keys that beat ``ord[r]``). Then

        ``deadline[ord[r]] = T(r) - 1``, where ``T(r)`` is the ``take``-th smallest key index in
        ``ord[0..r-1]``

because key ``ord[r]`` is squeezed out precisely when the pool horizon reaches the ``take``-th
smallest index among the keys that beat it. ``T`` is a prefix order statistic, which torch has no
primitive for, so it is evaluated with a two-level count: a cumulative histogram over
``(rank, key-block)`` locates ``T(r)``'s block, then a ``BS``-wide scan resolves it exactly. That
replaces the naive ``(Sk, Sk)`` comparison with ``(Sk, Sk / BS)`` -- measured 4.9x at ``L=8030``
and 10.2x at ``L=16384``, both bit-identical to the naive form.
    """
    if scores.dim() != 2:
        raise ValueError(f"scores must be (n_heads, Sk), got {tuple(scores.shape)}")
    n_heads, k_len = scores.shape
    device = scores.device
    take = int(topk) - int(force_sink) - int(force_local)
    if take <= 0 or k_len == 0:
        # No top-k budget at all: the support is exactly the forced slots. The sentinel must
        # therefore select NOTHING, not "never evicted" -- returning k_len-1 here would make
        # `horizon <= deadline` true for every key and the mask would keep the whole pool. (Caught
        # by test_deadline_mask_matches_streaming_topk at topk == force_sink + force_local, where it
        # kept 528 entries instead of 275.)
        return torch.full((n_heads, k_len), -1, dtype=torch.int32, device=device)

    key_idx = torch.arange(k_len, device=device)
    in_pool = key_idx >= force_sink
    neg = torch.tensor(-float("inf"), device=device, dtype=scores.dtype)
    pooled = torch.where(in_pool, scores, neg)
    # Stable, so equal scores are beaten by the lower key index -- the tie-break the gather path's
    # top-k arrives at, and the one sort_support's ascending output makes observable.
    order = torch.argsort(pooled, dim=-1, descending=True, stable=True)

    #: Key-index block width for locating ``T(r)``. Trades the histogram's second extent
    #: (``Sk / BS``) against the width of the exact scan inside the located block.
    BS = 64
    n_blocks = (k_len + BS - 1) // BS

    # hist[h, r, b] = 1 where rank r's key lives in block b; cumsum over r (exclusive) then over b
    # gives "how many of the first r arrivals have block <= b".
    blk_of_rank = torch.div(order, BS, rounding_mode="floor").clamp(max=n_blocks - 1)
    hist = torch.zeros((n_heads, k_len, n_blocks), dtype=torch.int16, device=device)
    hist.scatter_(2, blk_of_rank.unsqueeze(-1), torch.ones_like(blk_of_rank.unsqueeze(-1), dtype=torch.int16))
    cum_rank = hist.cumsum(1, dtype=torch.int32)
    cum_rank = torch.cat([torch.zeros_like(cum_rank[:, :1]), cum_rank[:, :-1]], dim=1)  # exclusive
    prefix_blk = cum_rank.cumsum(2)
    del hist, cum_rank

    reached = prefix_blk >= take
    block = reached.to(torch.uint8).argmax(2)  # first block whose running total hits `take`
    unreached = ~reached.any(2)  # fewer than `take` keys beat this rank => never evicted
    before = torch.where(
        block > 0,
        prefix_blk.gather(2, (block - 1).clamp(min=0).unsqueeze(-1)).squeeze(-1),
        torch.zeros_like(block),
    )
    need = take - before  # how many more are needed from within `block`
    del prefix_blk, reached

    arrival = torch.empty_like(order)
    arrival.scatter_(-1, order, torch.arange(k_len, device=device).expand_as(order))

    ranks = torch.arange(k_len, device=device)
    offsets = torch.arange(BS, device=device)
    threshold = torch.empty((n_heads, k_len), dtype=torch.int64, device=device)
    for h in range(n_heads):
        keys = (block[h].unsqueeze(-1) * BS + offsets.view(1, BS))  # (Sk, BS) candidate keys
        valid = keys < k_len
        safe = keys.clamp(max=k_len - 1)
        # a candidate counts for rank r only if it is in the pool and arrived strictly before r
        alive = (arrival[h][safe] < ranks.unsqueeze(-1)) & in_pool[safe] & valid
        hit = alive.cumsum(-1) >= need[h].unsqueeze(-1)
        pos = hit.to(torch.uint8).argmax(-1)
        found = safe.gather(-1, pos.unsqueeze(-1)).squeeze(-1)
        threshold[h] = torch.where(hit.any(-1), found, torch.full_like(found, k_len + 1))

    threshold = torch.where(unreached, torch.full_like(threshold, k_len + 1), threshold)
    per_rank = torch.where(
        threshold > k_len,
        torch.full_like(threshold, k_len - 1),  # never evicted
        (threshold - 1).clamp(min=-1),
    )
    out = torch.empty((n_heads, k_len), dtype=torch.int64, device=device)
    out.scatter_(-1, order, per_rank)
    # Keys outside the pool (the sink) are unconditional; give them the "never" value so a caller
    # that forgets the sink rule still cannot evict them.
    return torch.where(in_pool.view(1, -1), out, torch.full_like(out, k_len - 1)).to(torch.int32)


def qi_block_mask(
    deadline: torch.Tensor,
    *,
    q_len: int,
    k_len: int,
    n_q_heads: int,
    force_sink: int,
    force_local: int,
    query_offset: int | None = None,
    device: torch.device | None = None,
):
    """
    Build the ``BlockMask`` for the query-independent support described by ``deadline``.

    ``deadline`` is ``(n_kv_heads, Sk)``; query heads in the same GQA group share it, which is what
    lets one mask serve all ``n_q_heads``.
    """
    if not HAS_FLEX:
        raise RuntimeError("this torch has no torch.nn.attention.flex_attention")
    n_kv_heads = deadline.shape[0]
    if n_q_heads % n_kv_heads != 0:
        raise ValueError(f"n_q_heads {n_q_heads} is not a multiple of n_kv_heads {n_kv_heads}")
    group = n_q_heads // n_kv_heads
    offset = k_len - q_len if query_offset is None else int(query_offset)
    device = device or deadline.device
    fs, fl = int(force_sink), int(force_local)
    last_key = k_len - 1

    def mask_mod(b, h, q_i, k_j):
        limit = torch.clamp(q_i + offset, max=last_key)  # this row's causal horizon
        horizon = limit - fl  # top-k pool's upper end
        sink = k_j < fs
        local = (k_j > limit - fl) & (k_j >= fs)
        # `deadline` is indexed by KV head; query heads in a group share it.
        alive = horizon <= deadline[h // group, k_j]
        chosen = (k_j >= fs) & (k_j <= horizon) & alive
        return (k_j <= limit) & (sink | local | chosen)

    return _block_mask()(
        mask_mod, B=None, H=n_q_heads, Q_LEN=q_len, KV_LEN=k_len, device=device
    )


def qi_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
    *,
    force_sink: int = 0,
    force_local: int = 0,
    scaling: float | None = None,
    query_offset: int | None = None,
) -> torch.Tensor:
    """
    Sparse attention over a query-independent support, via ``flex_attention``.

    Parameters
    ----------
    query : torch.Tensor
        ``(B, H, Sq, D)``.
    key, value : torch.Tensor
        ``(B, Hkv, Sk, D)`` -- GQA layout, expanded to ``H`` here.
    scores : torch.Tensor
        ``(B, Hkv, Sk)`` per-key scores from a query-independent scorer.
    topk, force_sink, force_local : int
        Support budget, matching :func:`~.sparse_support.streaming_topk_support`.

    Returns
    -------
    torch.Tensor
        ``(B, H, Sq, D)``, the same layout the gather kernel returns.
    """
    if not HAS_FLEX:
        raise RuntimeError("this torch has no torch.nn.attention.flex_attention")
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("query/key/value must be 4-D (B, heads, S, D)")
    if scores.dim() != 3:
        raise ValueError(f"scores must be (B, Hkv, Sk), got {tuple(scores.shape)}")
    if scores.shape[0] != 1:
        # The mask is shared across the batch (B=None below), so a per-sequence score would be
        # silently ignored. Eval runs one context at a time; refuse rather than mis-attend.
        raise NotImplementedError(
            f"qi_sparse_attention supports batch 1, got {scores.shape[0]}. The block mask is built "
            "with B=None, so per-sequence deadlines would be ignored."
        )
    bsz, n_q_heads, q_len, _ = query.shape
    n_kv_heads, k_len = key.shape[1], key.shape[2]
    if scores.shape[1:] != (n_kv_heads, k_len):
        raise ValueError(
            f"scores {tuple(scores.shape)} does not match key (Hkv={n_kv_heads}, Sk={k_len})"
        )
    group = n_q_heads // n_kv_heads

    dl = deadlines(scores[0].float(), topk, force_sink=force_sink, force_local=force_local)
    block_mask = qi_block_mask(
        dl,
        q_len=q_len,
        k_len=k_len,
        n_q_heads=n_q_heads,
        force_sink=force_sink,
        force_local=force_local,
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
