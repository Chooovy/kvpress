# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
**Hard eviction** for a query-independent router: keys are thrown away, not masked.

:mod:`~kvpress.presses.gqa_indexer.qi_flex_attention` keeps the whole KV cache and hides the
unselected keys behind a block mask. That is the right shape for prefill -- it skips 48.8% of the
128x128 blocks -- but it saves no memory and it leaves decode on the gather path, reading ``O(L)``
per step. This module takes the other half of what the deadline buys: once a key is provably out of
every future row's top-k, it can be **deleted**.

The two are the same selection, and that is a set identity rather than an approximation
---------------------------------------------------------------------------------------
``deadlines`` rests on irreversibility: a query row's pool horizon only grows, the count of keys
beating ``j`` is non-decreasing in it, so once ``j`` leaves the top-k it never returns. Written
incrementally, that statement *is* this module's decode step:

* each new token rolls the local window, pushing **exactly one** key out of it and into the
  evictable pool;
* the pool, already at its budget, sheds its current minimum -- unless the arriving key is worse,
  in which case the arrival is dropped instead.

Replaying that to the end of a sequence retains exactly the set the deadline mask would have shown
the final row. ``tests/presses/test_gqa_indexer_evict_cache.py`` asserts it as a symmetric
difference of **0**, over uniform and ragged budgets, on bf16-rounded scores where ties are dense.

What eviction buys
------------------
* **Decode reads the budget, not the context.** No mask, no gather: every slot in the cache is
  causally past and already selected, so a decode step is a dense GQA attention over ``budget``
  keys. Measured on an H20 over 36 layers at ``topk=2048``: 0.76 ms/token at batch 1 against
  7.11 ms/token for dense 8K at batch 4.
* **The cache decouples from the context length.** At ``topk=2048`` with the fitted static table
  the whole 36-layer cache is 0.298 GiB whether the context is 8K (3.77x smaller than dense) or
  128K (**60.3x**).
* **A ragged budget becomes real.** ``head_budget`` currently only changes which keys are *visible*
  -- the storage stays rectangular, so it saves nothing. Here each (layer, batch, head) row holds
  its own ``budget_h`` slots.

Why paged, and why ``BS = 256``
-------------------------------
Per-head budgets are wildly uneven: the fitted table spans 645..6968 within a layer, so padding
every row out to ``max_h`` costs **2.12x** the logical slots (measured; 2.39x for the per-document
``mass`` allocator, worst case 4.16x) -- which would hand back most of the compression. Paging at
``BS=256`` costs 6.1% instead.

256 is forced, not tuned: this flash-attn build rejects anything else with ``Paged KV cache block
size must be divisible by 256``. Verified at 64 and 128, both raise.

The layout, and the property that makes it simple
-------------------------------------------------
Every ``(layer, batch, head)`` row is flattened into the paged **batch** axis, in that order, so
one layer's rows are the contiguous slice ``[layer * B * H, (layer+1) * B * H)`` and need no gather
(verified: the slice is contiguous, and the attention matches an fp32 reference to 6.7e-4).

Within a row of ``budget`` slots, the local window is pinned to the **top**, at
``[budget - n_local, budget)``, and the evictable pool takes ``[0, budget - n_local)``. That choice
buys three things at once:

* occupied slots are always the prefix ``[0, filled)``, so ``cache_seqlens = filled`` is exactly
  what flash-attn wants and there are never holes;
* while a row is still **growing** (a context shorter than the budget, or the short sequences in a
  batch) nothing moves: the arriving token lands at slot ``filled``, and the key leaving the local
  window is already sitting in the pool region, because ``filled - n_local < budget - n_local``;
* at the moment ``filled`` reaches ``budget``, "the last ``n_local`` occupied slots" coincides with
  the pinned window region exactly -- so there is no relayout, ever.

Only a full row does any work: one ``argmin`` and one slot overwrite. There is no free list, no
compaction, no block release and no growth, which is what separates this from the paged caches it
is modelled on (DMS's ``cache_paged.py``, TrimKV's ``PagedTrimKVCache``).

Ties are not a corner case
--------------------------
A bf16 score resolves only ~12% distinct values at ``L=8030``, so 95% of keys share a score with
another. ``deadlines`` breaks ties by ascending key index (a stable descending sort), i.e. **among
equals the later position loses**. :func:`rank_key` reproduces that exactly by packing the score
and the position into one monotone int64, so the ``argmin`` is a total order with no float epsilon
anywhere.
"""

from __future__ import annotations

import logging

import torch
from transformers import Cache, CacheLayerMixin

logger = logging.getLogger(__name__)

#: flash-attn's paged block size. **Not a tuning knob**: this build raises ``Paged KV cache block
#: size must be divisible by 256`` for anything else (verified at 64 and 128). It sets the padding
#: waste, measured at 6.1% of the logical slots for the fitted static table against 2.12x for a
#: rectangular layout.
PAGE_BLOCK = 256

#: Bits reserved for the position in :func:`rank_key`'s packed key. 21 bits covers contexts up to
#: 2,097,152 tokens while leaving the 32-bit monotone score mapping room inside int64
#: (32 + 21 = 53 bits, plus sign).
POS_BITS = 21

#: Largest position :func:`rank_key` can encode without the score/position packing colliding.
MAX_POSITION = (1 << POS_BITS) - 1


def _merge_lse(
    branches: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """
    Merge N already-normalized attention branches by their log-sum-exps. NaN-safe by construction.

    Each branch is ``(o, lse)`` with ``o`` shaped ``(B, H, Sq, D)`` and ``lse`` ``(B, H, Sq)``, and
    ``o`` already divided by its own denominator. Returns the combined ``(B, H, Sq, D)``.

    **Two ways this produced NaN, both guarded here rather than at the call sites.** Both cost
    multi-hour 8-shard runs to localize, because the symptom is an async device-side assert inside
    ``multinomial`` blamed on ``evict_runner.py:640``, and because greedy decoding cannot detect it
    at all -- ``argmax`` has no NaN check and silently returns the NaN's index, so only sampled
    decoding ever reports the corruption.

    1. **A branch with ``lse = +inf``.** ``flash_attn_with_kvcache`` returns ``+inf`` (not ``-inf``)
       for a row whose ``cache_seqlens`` is 0. Then ``m = +inf`` and ``exp(lse - m) = exp(inf-inf)``
       is NaN for *every* branch, so one empty row poisons the whole merge. Non-finite ``lse`` is
       therefore forced to ``-inf`` -- "contributes nothing" -- before the max is taken.
    2. **Every branch empty, so ``m = -inf``.** Then ``exp(-inf - -inf)`` is NaN again, and the
       denominator is NaN rather than 0. ``m`` is clamped to a finite floor for those rows, which
       makes the weights exactly 0 and the output exactly 0 -- the correct answer when there is
       nothing to attend to. :meth:`~.streaming_cmp.StreamingCMP.read` already does this internally
       for the same reason; the merge simply has to do it too.

    A denominator floor is NOT sufficient on its own: ``0 * NaN`` is NaN, so a NaN weight
    contaminates the numerator no matter how the division is protected. The fix has to be upstream
    of the exponential, which is what both steps above do.
    """
    outs = [o for o, _ in branches]
    lses = [torch.where(torch.isfinite(l), l, torch.full_like(l, -float("inf"))) for _, l in branches]
    m = lses[0]
    for l in lses[1:]:
        m = torch.maximum(m, l)
    # Rows where every branch is empty: m is -inf. Substitute a finite max so `l - m` is -inf (not
    # NaN) and every weight comes out exactly 0.
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    weights = [(l - m).exp().unsqueeze(-1) for l in lses]
    num = sum(o * w for o, w in zip(outs, weights))
    den = sum(weights)
    return num / den.clamp(min=torch.finfo(den.dtype).tiny)


def rank_key(score: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """
    Pack ``(score, position)`` into one monotone int64 whose ``argmin`` is the key to evict.

    Ascending in this key == ascending in "how much this key deserves to be dropped", matching
    :func:`~.qi_flex_attention.deadlines`' ordering **including its tie-break**: equal scores are
    ranked by ascending key index there, so among equals the *larger* position is the loser.

    The score half is the standard IEEE-754 monotone map -- an fp32's bit pattern is already
    ordered for non-negatives, and negating the magnitude fixes the negatives -- so no epsilon or
    float comparison is involved and the resulting order is total. That matters because ties are
    the common case, not a corner: 95% of keys share a score with another at ``L=8030``.

    Parameters
    ----------
    score : torch.Tensor
        Any shape, cast to fp32. For a decaying router this is the score *at the current query
        position*, i.e. ``mag + log_beta * (i / decay_ref)``.
    pos : torch.Tensor
        Absolute key positions, broadcastable to ``score``. Must be ``<= MAX_POSITION``.

    Returns
    -------
    torch.Tensor
        int64, same shape as the broadcast of the inputs.
    """
    bits = score.float().contiguous().view(torch.int32).to(torch.int64)
    # Non-negative floats are already correctly ordered by their bit pattern; for negatives the
    # magnitude ordering runs backwards, so negate it.
    mono = torch.where(bits >= 0, bits, -(bits & 0x7FFFFFFF))
    return (mono << POS_BITS) - pos.to(torch.int64)


try:  # the decode kernel; absent on a CPU-only box, where the torch fallback runs instead
    from flash_attn import flash_attn_func, flash_attn_with_kvcache

    HAS_FLASH = True
except ImportError:  # pragma: no cover - depends on the installed flash-attn
    flash_attn_func = flash_attn_with_kvcache = None
    HAS_FLASH = False


class EvictPagedPool:
    """
    The physical cache: one paged pool holding every ``(layer, batch, head)`` row.

    Rows are flattened in that order -- layer outermost -- so one layer's rows are the contiguous
    slice ``[layer * B * H, (layer + 1) * B * H)``. That is what lets a decode step hand flash-attn
    a plain slice of ``block_table`` and ``cache_seqlens`` with no gather.

    Each row holds ``budget[layer, head]`` slots and **never grows past them**. Within a row:

    * ``[0, budget - n_local)`` is the evictable pool, held in no particular order (attention does
      not care, and post-RoPE keys carry their own position, so there is nothing to sort);
    * ``[budget - n_local, budget)`` is the local window, a ring indexed by position modulo
      ``n_local``.

    Occupied slots are always the prefix ``[0, filled)``, so ``cache_seqlens = filled`` needs no
    correction and the cache never has holes.

    Parameters
    ----------
    budgets : torch.Tensor
        ``(n_layers, n_kv_heads)`` int64 per-head slot counts. Shared by every sequence in the
        batch -- a per-sequence budget would make the block allocation sequence-dependent and
        destroy the contiguous per-layer slice this design rests on.
    batch_size : int
        Sequences shar[ing] this pool. Each gets its own rows and its own ``seen`` counter.
    n_local, n_sink : int
        The pin geometry, which must match what the prefill selected under.
    head_dim : int
        KV head width.
    decay_ref : float or None
        ``ScalarIndexer.decay_ref``. ``None`` means the router has no decay, in which case a key's
        score is fixed and ``log_beta`` is not stored at all.
    """

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
        decay_ref: float | None = None,
    ):
        if budgets.shape != (n_layers, n_kv_heads):
            raise ValueError(
                f"budgets must be (n_layers={n_layers}, n_kv_heads={n_kv_heads}), "
                f"got {tuple(budgets.shape)}"
            )
        take = budgets - n_sink - n_local
        if int(take.min()) <= 0:
            layer, head = (take <= 0).nonzero()[0].tolist()
            raise ValueError(
                f"layer {layer} head {head} has budget {int(budgets[layer, head])} against "
                f"n_sink={n_sink} + n_local={n_local}, leaving {int(take.min())} evictable slots. "
                "A head whose pins already consume its budget can never retain anything the "
                "router selected -- raise its budget or --head_budget_floor."
            )
        self.budgets = budgets.to(device=device, dtype=torch.int64)
        self.batch_size = int(batch_size)
        self.n_layers = int(n_layers)
        self.n_kv_heads = int(n_kv_heads)
        self.n_sink = int(n_sink)
        self.n_local = int(n_local)
        self.head_dim = int(head_dim)
        self.device = device
        self.dtype = dtype
        self.decay_ref = decay_ref

        rows = n_layers * batch_size * n_kv_heads
        self.rows = rows
        # Row-major (layer, batch, head): a per-layer slice is contiguous, which is the whole
        # point of this ordering.
        self.row_budget = (
            self.budgets.view(n_layers, 1, n_kv_heads)
            .expand(n_layers, batch_size, n_kv_heads)
            .reshape(rows)
            .contiguous()
        )

        blocks_per_row = (self.row_budget + PAGE_BLOCK - 1) // PAGE_BLOCK
        self.max_blocks = int(blocks_per_row.max())
        n_blocks = int(blocks_per_row.sum())
        # One global pool. Allocated once; there is no free list and no growth, because every row's
        # size is fixed at construction and eviction replaces rather than appends.
        self.k_pool = torch.zeros(
            (n_blocks, PAGE_BLOCK, 1, head_dim), device=device, dtype=dtype
        )
        self.v_pool = torch.zeros_like(self.k_pool)
        # Block table, built once and then frozen.
        self.block_table = torch.zeros((rows, self.max_blocks), device=device, dtype=torch.int32)
        starts = torch.cumsum(blocks_per_row, 0) - blocks_per_row
        slot_ax = torch.arange(self.max_blocks, device=device)
        valid = slot_ax.view(1, -1) < blocks_per_row.view(-1, 1)
        self.block_table = torch.where(
            valid, (starts.view(-1, 1) + slot_ax.view(1, -1)).to(torch.int32), torch.zeros_like(self.block_table)
        )

        self.filled = torch.zeros(rows, device=device, dtype=torch.int32)
        # Pool metadata, rectangular at the widest `take`. Rectangular is deliberate: it is what
        # makes the eviction step one vectorized argmin. Slots past a row's own `take` are pinned
        # to int64.max so the argmin can never select them -- without that a narrow row would
        # "evict" an unused padding slot and grow past its budget (caught by
        # test_ragged_per_head_budgets_match_too, where a budget-150 head kept 151).
        self.take = (self.row_budget - n_sink - n_local).clamp(min=0)
        self.width = int(self.take.max())
        self.pool_pos = torch.full((rows, self.width), -1, device=device, dtype=torch.int64)
        self.pool_key = torch.full(
            (rows, self.width), torch.iinfo(torch.int64).min, device=device, dtype=torch.int64
        )
        self._pad = torch.arange(self.width, device=device).view(1, -1) >= self.take.view(-1, 1)
        self.pool_key.masked_fill_(self._pad, torch.iinfo(torch.int64).max)
        # The router state each pool slot carries, so its score can be re-evaluated at any future
        # query position. `mag` is ScalarIndexer's folded magnitude (s_j - log_beta_j * j / ref),
        # which is exactly what gate_key packs, so a slot's score at position i is
        # mag + log_beta * (i / ref) with no reference to j.
        self.pool_mag = torch.zeros((rows, self.width), device=device, dtype=torch.float32)
        self.pool_beta = (
            torch.zeros((rows, self.width), device=device, dtype=torch.float32)
            if decay_ref is not None
            else None
        )
        if n_local < 1:
            raise ValueError(
                f"n_local must be at least 1, got {n_local}: the local window is what makes a key "
                "become evictable one step at a time, which is the whole incremental rule."
            )
        # The local window, as an explicit circular buffer. `ring_slot` records the PHYSICAL slot
        # of each entry rather than deriving it from the position, and that is a correctness fix
        # rather than a convenience: a `pos % n_local` convention does not survive the transition
        # from growing to full (during growth a token sits at slot == position, once full it sits
        # wherever the key it displaced was), and the mismatch silently mis-ranks ~30% of the pool
        # at the boundary. Storing the slot makes both phases record where they actually wrote.
        self.ring_slot = torch.zeros((rows, n_local), device=device, dtype=torch.int64)
        self.ring_pos = torch.full((rows, n_local), -1, device=device, dtype=torch.int64)
        self.ring_mag = torch.zeros((rows, n_local), device=device, dtype=torch.float32)
        self.ring_beta = (
            torch.zeros((rows, n_local), device=device, dtype=torch.float32)
            if decay_ref is not None
            else None
        )
        #: Index of the OLDEST window entry -- the one that ages out next.
        self.ring_head = torch.zeros(rows, device=device, dtype=torch.int64)
        self.ring_count = torch.zeros(rows, device=device, dtype=torch.int64)
        #: Live entries in each row's pool. Maintained as a counter rather than recovered by
        #: scanning ``pool_pos``, and that is a throughput fix: the scan is a reduction over the
        #: rectangular width (6836 at the fitted table), which measured 0.12 ms per layer for the
        #: ``any``/``argmax`` pair -- 4.3 ms/token over 36 layers, against 3.3 ms/token for the
        #: attention itself. The counter is exact because the pool never develops holes: slots are
        #: filled in order while growing, and eviction *replaces* an occupied slot rather than
        #: freeing one. So ``pool_live`` is both the count and the index of the first free slot.
        self.pool_live = torch.zeros(rows, device=device, dtype=torch.int64)
        # Logical tokens seen per sequence -- NOT the physical slot count, which stops growing at
        # the budget. Every position-dependent quantity (the recency tilt, the decay age, RoPE)
        # keys off this, so conflating the two silently deforms the score.
        self.seen = torch.zeros(batch_size, device=device, dtype=torch.int64)
        self._committed = torch.zeros((n_layers, batch_size), dtype=torch.bool)

    # ------------------------------------------------------------------
    def rows_for(self, layer_idx: int, seq: int | None = None) -> slice:
        """The pool rows for one layer, or for one (layer, sequence) pair. Always a slice."""
        base = layer_idx * self.batch_size * self.n_kv_heads
        if seq is None:
            return slice(base, base + self.batch_size * self.n_kv_heads)
        start = base + seq * self.n_kv_heads
        return slice(start, start + self.n_kv_heads)

    def is_committed(self, layer_idx: int, seq: int) -> bool:
        return bool(self._committed[layer_idx, seq])

    @torch.no_grad()
    def replicate_seq(self, src: int, dst: int) -> None:
        """Copy sequence ``src``'s whole compressed state onto row-set ``dst``, for every layer.

        A benchmark's questions share one context, so committing it once and replicating is what
        makes batched decode available without re-prefilling. It is cheap *because* the cache is
        compressed -- at ``topk=2048`` a whole 36-layer sequence is 0.298 GiB against 1.125 GiB
        dense at 8K, and the copy replaces a full prefill (measured ~5 s per 8K context).

        Everything that describes the state is copied, not just k/v: the pool metadata, the ring,
        and the logical ``seen``. Missing any one of them would leave the replica attending over
        the right keys while ranking them with another sequence's scores.
        """
        if src == dst:
            return
        for layer_idx in range(self.n_layers):
            s, d = self.rows_for(layer_idx, src), self.rows_for(layer_idx, dst)
            # k/v live in the paged pool, which each row addresses through its OWN block table --
            # so this is a real copy through both tables, not a tensor slice assignment.
            for offset in range(self.n_kv_heads):
                sr, dr = s.start + offset, d.start + offset
                n = int(self.filled[sr])
                if n:
                    slots = torch.arange(n, device=self.device)
                    sb = self.block_table[sr, slots // PAGE_BLOCK].to(torch.int64)
                    db = self.block_table[dr, slots // PAGE_BLOCK].to(torch.int64)
                    self.k_pool[db, slots % PAGE_BLOCK, 0, :] = self.k_pool[
                        sb, slots % PAGE_BLOCK, 0, :
                    ]
                    self.v_pool[db, slots % PAGE_BLOCK, 0, :] = self.v_pool[
                        sb, slots % PAGE_BLOCK, 0, :
                    ]
            for buf in (
                self.filled, self.pool_pos, self.pool_key, self.pool_mag, self.pool_live,
                self.ring_slot, self.ring_pos, self.ring_mag, self.ring_head, self.ring_count,
            ):
                buf[d] = buf[s]
            if self.pool_beta is not None:
                self.pool_beta[d] = self.pool_beta[s]
                self.ring_beta[d] = self.ring_beta[s]
            self._committed[layer_idx, dst] = bool(self._committed[layer_idx, src])
        self.seen[dst] = self.seen[src]

    def pool_scores(self, rows: slice, query_pos: torch.Tensor) -> torch.Tensor:
        """Each live pool slot's score as seen from ``query_pos``, ``(n_rows, width)`` fp32.

        Without decay the score is a property of the key alone and ``query_pos`` is irrelevant.
        With decay it is ``mag + log_beta * (query_pos / decay_ref)`` -- the same fold
        :meth:`~.scalar_indexer.ScalarIndexer.gate_key` uses, which is why the stored magnitude
        already has ``-log_beta * j / ref`` absorbed into it and no key position is needed here.
        """
        if self.pool_beta is None:
            return self.pool_mag[rows]
        age = (query_pos.to(torch.float32) / float(self.decay_ref)).view(-1, 1)
        return self.pool_mag[rows] + self.pool_beta[rows] * age

    # ------------------------------------------------------------------
    # Writing slots
    # ------------------------------------------------------------------
    def _write_kv(self, rows: torch.Tensor, slots: torch.Tensor, key, value) -> None:
        """Scatter ``key``/``value`` into physical ``(row, slot)`` addresses.

        ``rows``/``slots`` are 1-D and equal length; ``key``/``value`` are ``(n, head_dim)``.
        """
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
        mag: torch.Tensor,
        log_beta: torch.Tensor | None,
        k_len: int,
    ) -> None:
        """
        Compress one prefilled ``(layer, sequence)`` into the pool: the one-shot eviction.

        Called once per layer at the end of a sequence's context prefill, with the layer's full
        dense cache. Selection comes from :func:`~.qi_flex_attention.deadlines` evaluated at the
        **last** prefill row, and that choice is not free: eviction is monotone in the horizon, so
        the final row's keep-set is the one every subsequent decode step inherits. Committing an
        earlier row's set would leave keys in the cache that the very first decode step wants to
        evict -- and since a step can only shed one key, it would take hundreds of steps to catch
        up.

        Parameters
        ----------
        key, value : torch.Tensor
            ``(n_kv_heads, k_len, head_dim)`` -- this layer's dense cache for this sequence.
        mag : torch.Tensor
            ``(n_kv_heads, k_len)`` fp32. The router's per-key score with the decay fold applied,
            i.e. ``s_j - log_beta_j * j / decay_ref`` when decaying and plain ``s_j`` otherwise --
            exactly what :meth:`~.scalar_indexer.ScalarIndexer.gate_key` packs.
        log_beta : torch.Tensor or None
            ``(n_kv_heads, k_len)`` fp32, ``<= 0``. Required iff the pool was built with
            ``decay_ref``.
        k_len : int
            The prefilled context length, which becomes this sequence's ``seen``.
        """
        from kvpress.presses.gqa_indexer.qi_flex_attention import deadlines

        if self._committed[layer_idx, seq]:
            raise RuntimeError(
                f"layer {layer_idx} sequence {seq} is already committed. The pool holds one "
                "compressed context per (layer, sequence); re-committing would evict against a "
                "cache that has already been compressed."
            )
        if (log_beta is None) != (self.pool_beta is None):
            raise ValueError(
                "log_beta must be supplied exactly when the pool was built with decay_ref "
                f"(log_beta={'None' if log_beta is None else 'given'}, "
                f"decay_ref={self.decay_ref})"
            )
        if k_len > MAX_POSITION:
            raise ValueError(
                f"context length {k_len} exceeds the {MAX_POSITION} positions rank_key can pack; "
                "raise POS_BITS."
            )
        rows = self.rows_for(layer_idx, seq)
        budget = self.row_budget[rows]
        n_local, n_sink = self.n_local, self.n_sink

        if k_len <= int(budget.min()):
            # The whole context fits: nothing is evicted, and the rows are simply filled in
            # position order. This is the growing path, and it must not run `deadlines` -- with
            # k_len below the budget there is nothing to rank, and the pool's invariant ("occupied
            # slots are the prefix [0, filled)") is satisfied by a straight copy.
            self._commit_whole(layer_idx, seq, key=key, value=value, mag=mag,
                               log_beta=log_beta, k_len=k_len)
            return

        # The score the ranking is taken at: the LAST row, so the committed set is the one decode
        # inherits. `deadlines` wants (n_heads, Sk) with the decay already resolved at that row.
        scores = mag if log_beta is None else mag + log_beta * (float(k_len - 1) / self.decay_ref)
        dl = deadlines(scores, budget, force_sink=n_sink, force_local=n_local)
        horizon = max(k_len - 1 - n_local, 0)
        pos_ax = torch.arange(k_len, device=self.device)
        # The keep-set, exactly as qi_block_mask's mask_mod would evaluate it at the last row.
        in_pool = (pos_ax >= n_sink) & (pos_ax <= horizon)
        keep_pool = in_pool.view(1, -1) & (horizon <= dl.to(torch.int64))
        is_sink = pos_ax < n_sink
        is_local = pos_ax > horizon

        take = self.take[rows]
        flat_row = torch.arange(rows.start, rows.stop, device=self.device)
        for h in range(self.n_kv_heads):
            row = int(flat_row[h])
            # --- sinks: slots [0, n_sink) -----------------------------------------------------
            n_s = int(is_sink.sum())
            if n_s:
                slots = torch.arange(n_s, device=self.device)
                self._write_kv(
                    torch.full_like(slots, row), slots, key[h, :n_s], value[h, :n_s]
                )
            # --- pool: the retained middle, packed into [n_sink, n_sink + take) ---------------
            sel = keep_pool[h].nonzero().flatten()
            if sel.numel() > int(take[h]):
                raise RuntimeError(
                    f"layer {layer_idx} seq {seq} head {h}: deadline kept {sel.numel()} pool keys "
                    f"against take={int(take[h])}. The budget the deadline was computed at and "
                    "the pool's own width have diverged."
                )
            n_p = int(sel.numel())
            if n_p:
                slots = torch.arange(n_s, n_s + n_p, device=self.device)
                self._write_kv(
                    torch.full_like(slots, row), slots, key[h, sel], value[h, sel]
                )
                pool_slot = torch.arange(n_p, device=self.device)
                self.pool_pos[row, pool_slot] = sel
                self.pool_mag[row, pool_slot] = mag[h, sel]
                if self.pool_beta is not None:
                    self.pool_beta[row, pool_slot] = log_beta[h, sel]
                self.pool_key[row, pool_slot] = rank_key(scores[h, sel], sel)
            self.pool_live[row] = n_p
            # --- local window: pinned to the TOP of the row, [budget - n_local, budget) -------
            loc = is_local.nonzero().flatten()
            if loc.numel():
                # Written in age order, oldest first, so ring_head = 0 and the next step ages out
                # exactly the oldest entry. The physical slot is RECORDED rather than derived:
                # see the ring_slot comment in __init__.
                base = int(budget[h]) - n_local
                n_l = int(loc.numel())
                slots = base + torch.arange(n_l, device=self.device)
                self._write_kv(
                    torch.full_like(slots, row), slots, key[h, loc], value[h, loc]
                )
                self.ring_slot[row, :n_l] = slots
                self.ring_pos[row, :n_l] = loc
                self.ring_mag[row, :n_l] = mag[h, loc]
                if self.ring_beta is not None:
                    self.ring_beta[row, :n_l] = log_beta[h, loc]
                self.ring_head[row] = 0
                self.ring_count[row] = n_l
            self.filled[row] = int(budget[h])

        self.seen[seq] = k_len
        self._committed[layer_idx, seq] = True

    @torch.no_grad()
    def _commit_whole(self, layer_idx, seq, *, key, value, mag, log_beta, k_len) -> None:
        """A context that fits inside every head's budget: copy it, evict nothing.

        Slots are written in position order, so the pool's prefix invariant holds and the local
        window lands exactly where the ring expects it once the row eventually fills.
        """
        rows = self.rows_for(layer_idx, seq)
        flat_row = torch.arange(rows.start, rows.stop, device=self.device)
        n_sink, n_local = self.n_sink, self.n_local
        slots = torch.arange(k_len, device=self.device)
        for h in range(self.n_kv_heads):
            row = int(flat_row[h])
            self._write_kv(torch.full_like(slots, row), slots, key[h], value[h])
            # Everything below the local window is already evictable and must be tracked, so a
            # later decode step can rank it. The newest n_local keys are the window itself.
            n_pool = max(k_len - n_local - n_sink, 0)
            if n_pool:
                keys = torch.arange(n_sink, n_sink + n_pool, device=self.device)
                self.pool_pos[row, :n_pool] = keys
                self.pool_mag[row, :n_pool] = mag[h, keys]
                if self.pool_beta is not None:
                    self.pool_beta[row, :n_pool] = log_beta[h, keys]
                sc = (
                    mag[h, keys]
                    if log_beta is None
                    else mag[h, keys] + log_beta[h, keys] * (float(k_len - 1) / self.decay_ref)
                )
                self.pool_key[row, :n_pool] = rank_key(sc, keys)
            self.pool_live[row] = n_pool
            # The newest min(k_len - n_sink, n_local) keys are the live window. Recorded at the
            # slots they were actually written to (== their positions, on this path), oldest first.
            n_l = max(min(k_len - n_sink, n_local), 0)
            if n_l:
                w = torch.arange(k_len - n_l, k_len, device=self.device)
                self.ring_slot[row, :n_l] = w
                self.ring_pos[row, :n_l] = w
                self.ring_mag[row, :n_l] = mag[h, w]
                if self.ring_beta is not None:
                    self.ring_beta[row, :n_l] = log_beta[h, w]
                self.ring_head[row] = 0
                self.ring_count[row] = n_l
            self.filled[row] = k_len
        self.seen[seq] = k_len
        self._committed[layer_idx, seq] = True

    # ------------------------------------------------------------------
    # The incremental step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def ingest(
        self,
        layer_idx: int | list[int],
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        mag: torch.Tensor,
        log_beta: torch.Tensor | None,
        positions: torch.Tensor,
        seqs: torch.Tensor | None = None,
    ) -> None:
        """
        Absorb one new token per sequence, evicting at most one key per row.

        This is the incremental form of ``deadlines``' irreversibility, and it is exact: replaying
        it over a whole sequence retains precisely the set the deadline mask would show the final
        row (``tests/presses/test_gqa_indexer_evict_cache.py``, symmetric difference 0).

        The order of operations is forced. The arriving token enters the **local window**, which
        pushes the key that has just aged out of it into the evictable pool; only then can the pool
        shed anything. Doing it the other way round would let the new token compete against the
        pool while the aged-out key is still unaccounted for.

        ``layer_idx`` may be a **list of layers**, in which case the tensors carry a leading layer
        axis and every layer is absorbed in one shot. That is the form decode should use, and the
        reason is launch overhead rather than elegance: one call issues ~90 small kernels whose
        total *device* time is only 204 us, so at 36 separate calls per token the step is bound by
        CPU dispatch (measured 0.66 ms/layer wall against 0.006 ms of work). Batching the layer
        axis in amortizes those launches over 36x the rows.

        Parameters
        ----------
        key, value : torch.Tensor
            ``(n_seq, n_kv_heads, head_dim)``, or ``(n_layers, n_seq, n_kv_heads, head_dim)`` when
            ``layer_idx`` is a list.
        mag, log_beta : torch.Tensor
            Matching router state, without the ``head_dim`` axis.
        positions : torch.Tensor
            ``(n_seq,)`` absolute positions of the arriving tokens. Shared across layers.
        seqs : torch.Tensor, optional
            Which sequences these belong to; defaults to all of them.
        """
        n_kv, n_local = self.n_kv_heads, self.n_local
        if seqs is None:
            seqs = torch.arange(self.batch_size, device=self.device)
        seqs = seqs.to(self.device)
        layers = (
            [layer_idx] if isinstance(layer_idx, int) else list(layer_idx)
        )
        lay = torch.as_tensor(layers, device=self.device, dtype=torch.int64)
        # Row index for every (layer, sequence, head) triple being written, in the pool's layout.
        rows = (
            lay.view(-1, 1, 1) * (self.batch_size * n_kv)
            + seqs.view(1, -1, 1) * n_kv
            + torch.arange(n_kv, device=self.device).view(1, 1, -1)
        ).reshape(-1)
        pos = (
            positions.to(self.device)
            .view(1, -1, 1)
            .expand(len(layers), seqs.numel(), n_kv)
            .reshape(-1)
        )
        k_flat = key.reshape(-1, self.head_dim)
        v_flat = value.reshape(-1, self.head_dim)
        m_flat = mag.reshape(-1).float()
        b_flat = None if log_beta is None else log_beta.reshape(-1).float()
        if k_flat.shape[0] != rows.numel():
            raise ValueError(
                f"got {k_flat.shape[0]} arriving tokens for {rows.numel()} rows "
                f"({len(layers)} layers x {seqs.numel()} sequences x {n_kv} heads)"
            )

        # NOTHING below reads a tensor on the host. That is a hard performance requirement, not a
        # style preference: a single `bool(mask.any())` costs 0.94 ms here because it stalls the
        # CPU on the GPU, and at 36 layers per token that alone was 34 ms/token -- ten times the
        # attention it serves. Measured: the tensor math in this function totals ~0.15 ms, while
        # the two `.any()` guards it used to have took it to 3.3 ms/layer (117 ms/token). So every
        # branch here is expressed as masked arithmetic over all rows, and the writes are made
        # idempotent by pointing the inactive ones at a slot that is then written with its own
        # current contents.
        budget = self.row_budget[rows]
        filled = self.filled[rows].to(torch.int64)
        head = self.ring_head[rows]
        count = self.ring_count[rows]

        # The window is full exactly when it holds n_local entries, and only then does a token age
        # OUT of it. `oldest` is that token's ring index; its physical slot and router state are
        # read from the ring rather than recomputed, which is what makes this correct across the
        # growing/full transition (a `pos % n_local` convention is not).
        rolls = count >= n_local
        demoted_slot = self.ring_slot[rows, head]
        demoted_pos = self.ring_pos[rows, head]
        demoted_mag = self.ring_mag[rows, head]
        demoted_beta = None if self.ring_beta is None else self.ring_beta[rows, head]

        # Where the arriving token goes. A growing row appends at `filled`; a full row takes the
        # slot the aged-out token is vacating -- but only AFTER that token has been rescued, which
        # is why the pool write below happens first.
        growing = filled < budget
        arrive_slot = torch.where(growing, filled, demoted_slot)

        # ---- step 1: the aged-out token competes for a pool slot ---------------------------
        # Scored at the CURRENT query position, so a decaying router ranks it against the pool on
        # equal footing.
        d_score = demoted_mag
        if demoted_beta is not None:
            d_score = d_score + demoted_beta * (pos.to(torch.float32) / self.decay_ref)
        d_key = rank_key(d_score, demoted_pos.clamp(min=0))

        # A row with a free pool slot takes it; a full row must beat its weakest entry.
        # `pool_live` doubles as the index of the first free slot (the pool never develops holes),
        # so this needs no scan over the rectangular width -- see its definition.
        #
        # The `< take` comparison is load-bearing: padding slots past this row's own `take` also
        # carry pool_pos == -1, so treating "not occupied" as "free" would let a narrow row fill
        # them and grow its retained set one key per step up to the rectangular width. Measured
        # before that fix: a budget-100 head went from 80 live pool slots to 120 over 120 decode
        # steps, holding 40 keys it had no budget for, while `filled` still read a reassuring 100.
        live_n = self.pool_live[rows]
        take_r = self.take[rows]
        has_free = live_n < take_r
        first_free = live_n.clamp(max=max(self.width - 1, 0))

        pool_key_r = self.pool_key[rows]
        if self.pool_beta is not None:
            # With decay the stored key was packed at an older query position and is stale, so the
            # ranking is recomputed. Without decay the score is fixed and the stored key is still
            # exact -- which is why the plain arm pays nothing for this.
            live = self.pool_pos[rows] >= 0
            resc = self.pool_mag[rows] + self.pool_beta[rows] * (
                pos.to(torch.float32).view(-1, 1) / self.decay_ref
            )
            pool_key_r = torch.where(
                live & ~self._pad[rows],
                rank_key(resc, self.pool_pos[rows].clamp(min=0)),
                pool_key_r,
            )
        j_min = pool_key_r.argmin(-1)
        k_min = pool_key_r.gather(-1, j_min.unsqueeze(-1)).squeeze(-1)

        # Promote only when a token actually aged out of the window, it is not a sink, and it
        # either found a free slot or beat the weakest entry.
        promote = rolls & (demoted_pos >= self.n_sink) & (has_free | (d_key > k_min))
        target = torch.where(has_free, first_free, j_min)
        dst = self.n_sink + target

        # --- capture what LEAVES the cache, before anything is overwritten -----------------
        # Exactly one key per row can leave, and which one depends on the contest: if the demoted
        # key won, the pool member it displaced leaves; if it lost, the demoted key itself leaves.
        # Either way it must be read here -- a step later both slots have been written.
        #
        # A row that promoted into a FREE slot displaced nothing, and a row whose window has not
        # rolled yet has nothing to lose, so `lost` is false for both.
        loser_slot = torch.where(promote, dst, demoted_slot.clamp(min=0))
        lost = rolls & (demoted_pos >= self.n_sink) & ~(promote & has_free)
        lb = self.block_table[rows, loser_slot // PAGE_BLOCK].to(torch.int64)
        dropped_k = self.k_pool[lb, loser_slot % PAGE_BLOCK, 0, :].clone()
        dropped_v = self.v_pool[lb, loser_slot % PAGE_BLOCK, 0, :].clone()

        # Inactive rows are redirected to copy their target slot onto ITSELF, which makes the
        # unconditional scatter a no-op for them. Cheaper than a gather-scatter pair on the active
        # subset, and it keeps the kernel count fixed regardless of how many rows promote.
        src = torch.where(promote, demoted_slot.clamp(min=0), dst)
        self._copy_slot(rows, src, dst)
        self.pool_pos[rows, target] = torch.where(
            promote, demoted_pos, self.pool_pos[rows, target]
        )
        self.pool_mag[rows, target] = torch.where(
            promote, demoted_mag, self.pool_mag[rows, target]
        )
        if self.pool_beta is not None:
            self.pool_beta[rows, target] = torch.where(
                promote, demoted_beta, self.pool_beta[rows, target]
            )
        self.pool_key[rows, target] = torch.where(promote, d_key, self.pool_key[rows, target])
        # A promotion into a FREE slot grows the live count; one that displaced a loser does not.
        self.pool_live[rows] = live_n + (promote & has_free).to(torch.int64)

        # ---- step 2: the arriving token takes its window slot ------------------------------
        self._write_kv(rows, arrive_slot, k_flat, v_flat)
        # Append to the ring, or overwrite the entry that just aged out.
        ring_idx = torch.where(rolls, head, (head + count) % n_local)
        self.ring_slot[rows, ring_idx] = arrive_slot
        self.ring_pos[rows, ring_idx] = pos
        self.ring_mag[rows, ring_idx] = m_flat
        if self.ring_beta is not None:
            self.ring_beta[rows, ring_idx] = b_flat
        self.ring_head[rows] = torch.where(rolls, (head + 1) % n_local, head)
        self.ring_count[rows] = torch.where(rolls, count, count + 1)
        self.filled[rows] = torch.where(growing, filled + 1, filled).to(self.filled.dtype)

        # What actually left the cache this step, for a caller that wants to summarize it
        # (:mod:`~.streaming_cmp`). Returned rather than logged because the k/v have to be read
        # BEFORE the arriving token overwrites the slot -- which is why `dropped_k` is gathered
        # above, not here.
        return {
            "rows": rows,
            "key": dropped_k,
            "value": dropped_v,
            "evicted": lost,
        }

    def _copy_slot(self, rows: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Move one slot's k/v within a row, by physical address."""
        sb = self.block_table[rows, src // PAGE_BLOCK].to(torch.int64)
        db = self.block_table[rows, dst // PAGE_BLOCK].to(torch.int64)
        self.k_pool[db, dst % PAGE_BLOCK, 0, :] = self.k_pool[sb, src % PAGE_BLOCK, 0, :]
        self.v_pool[db, dst % PAGE_BLOCK, 0, :] = self.v_pool[sb, src % PAGE_BLOCK, 0, :]

    # ------------------------------------------------------------------
    # Attention over the compressed cache
    # ------------------------------------------------------------------
    def attend(
        self,
        layer_idx: int,
        query: torch.Tensor,
        *,
        seqs: torch.Tensor | None = None,
        scaling: float | None = None,
        new_key: torch.Tensor | None = None,
        new_value: torch.Tensor | None = None,
        extra: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Attention over the compressed cache, returning ``(B, Sq, n_q_heads, head_dim)``.

        **No mask.** Every slot in the pool is causally in the past and was already selected, so a
        decode row attends densely over its row's ``filled`` slots. That is the structural payoff
        of real eviction over masking: the block mask, the deadline evaluation and the gather all
        disappear from the decode path.

        ``new_key``/``new_value`` carry a multi-row forward's own tokens (the question forward,
        where ``Sq`` is ~28). Those cannot simply be written into the ring first and then attended
        over: the rows must not see each other's futures. They are instead attended as a **second,
        causal branch** and merged with the cache branch through their log-sum-exps, which is an
        identity rather than an approximation (verified to 1.6e-4 against an fp32 reference).

        Parameters
        ----------
        query : torch.Tensor
            ``(B, n_q_heads, Sq, head_dim)``, the model's own layout.
        new_key, new_value : torch.Tensor, optional
            ``(B, n_kv_heads, Sq, head_dim)``. Required iff ``Sq > 1``.
        """
        bsz, n_q_heads, q_len, head_dim = query.shape
        n_kv = self.n_kv_heads
        group = n_q_heads // n_kv
        if seqs is None:
            seqs = torch.arange(bsz, device=self.device)
        rows = (
            layer_idx * self.batch_size * n_kv
            + seqs.view(-1, 1).to(self.device) * n_kv
            + torch.arange(n_kv, device=self.device).view(1, -1)
        ).reshape(-1)
        if new_key is None:
            raise ValueError(
                "new_key/new_value are required: the arriving tokens are not in the pool yet "
                "(it is updated after the forward), so without them the query at position t would "
                "not attend to its own key. That costs cwe 95.71 -> 24.29 on RULER 4096."
            )
        if not HAS_FLASH:
            return self._attend_torch(
                rows, query, scaling=scaling, new_key=new_key, new_value=new_value
            )

        # flash-attn wants the KV head folded into the batch axis, with the GQA group as its head
        # axis: (B*Hkv, Sq, group, D). The pool's rows are laid out to match.
        q_flash = (
            query.view(bsz, n_kv, group, q_len, head_dim)
            .permute(0, 1, 3, 2, 4)
            .reshape(bsz * n_kv, q_len, group, head_dim)
        )
        out, lse = flash_attn_with_kvcache(
            q_flash.to(self.dtype),
            self.k_pool,
            self.v_pool,
            cache_seqlens=self.filled[rows],
            block_table=self.block_table[rows],
            softmax_scale=scaling,
            causal=False,  # every cached slot is already in the past
            return_softmax_lse=True,
        )
        o_cache = (
            out.view(bsz, n_kv, q_len, group, head_dim)
            .permute(0, 1, 3, 2, 4)
            .reshape(bsz, n_q_heads, q_len, head_dim)
            .float()
        )
        # An empty cache row (`cache_seqlens == 0`) must contribute NOTHING to the merge. `_merge_lse`
        # already neutralizes its `lse` -- flash-attn reports `+inf` there, not `-inf` -- but the
        # `out` it returns for that row is unconstrained, and `0 * garbage` is only 0 if the garbage
        # is finite. Zero it explicitly so the merge cannot depend on that.
        #
        # Per-row, and for every `q_len`. The guard this replaced ran only when `q_len == 1` and
        # tested `filled.min() == 0`, so it missed both (a) the multi-row question forward -- the
        # path that actually crashed -- and (b) MIXED batches, where the batch-wide `min() == 0`
        # discarded the cache branch of rows that DID have a cache. Under `--decode_batch 4` mixed
        # batches are the norm, not an edge case.
        empty_kv = (self.filled[rows] == 0).view(bsz, n_kv)

        # Second branch: this forward's own rows, causal among themselves -- and for a single
        # decode token that is the causal DIAGONAL, i.e. the query attending to its own key.
        l_cache = lse.view(bsz, n_kv, group, q_len).reshape(bsz, n_q_heads, q_len).float()
        if bool(empty_kv.any()):
            mask_q = empty_kv.repeat_interleave(group, 1).view(bsz, n_q_heads, 1)
            l_cache = torch.where(mask_q, torch.full_like(l_cache, -float("inf")), l_cache)
            o_cache = torch.where(mask_q.unsqueeze(-1), torch.zeros_like(o_cache), o_cache)
        o_new, l_new = flash_attn_func(
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
            # Three branches -- cache, this forward's own tokens, and the CMP slots. `extra` is
            # (o_cmp, lse_cmp) with o_cmp already normalized by its own denominator, so nothing
            # here exponentiates an absolute logit: a slot holding thousands of keys reaches
            # q.k + log n_r ~ 1600 on the deep layers' large-norm keys, and the unnormalized form
            # saturated fp32 to inf, producing inf/inf = nan.
            o_x, l_x = extra
            merged = _merge_lse([(o_cache, l_cache), (o_new, l_new), (o_x, l_x)])
            return merged.transpose(1, 2).contiguous().to(query.dtype)
        merged = _merge_lse([(o_cache, l_cache), (o_new, l_new)])
        return merged.transpose(1, 2).contiguous().to(query.dtype)

    def _attend_torch(self, rows, query, *, scaling, new_key, new_value) -> torch.Tensor:
        """Pure-torch reference, for a box without flash-attn (and for the tests).

        Reads each row's live slots through the block table and runs an ordinary softmax, so it
        exercises the same addressing the kernel does without needing CUDA.
        """
        bsz, n_q_heads, q_len, head_dim = query.shape
        n_kv = self.n_kv_heads
        group = n_q_heads // n_kv
        scale = scaling if scaling is not None else head_dim ** -0.5
        out = torch.zeros(bsz, n_q_heads, q_len, head_dim, dtype=torch.float32, device=query.device)
        for b in range(bsz):
            for h in range(n_kv):
                row = int(rows[b * n_kv + h])
                n = int(self.filled[row])
                slots = torch.arange(n, device=self.device)
                block = self.block_table[row, slots // PAGE_BLOCK].to(torch.int64)
                k = self.k_pool[block, slots % PAGE_BLOCK, 0, :].float()
                v = self.v_pool[block, slots % PAGE_BLOCK, 0, :].float()
                if new_key is not None:
                    k = torch.cat([k, new_key[b, h].float()])
                    v = torch.cat([v, new_value[b, h].float()])
                for g in range(group):
                    hq = h * group + g
                    logits = query[b, hq].float() @ k.T * scale
                    if new_key is not None:
                        # The cached part is unconditionally visible; the new part is causal.
                        j = torch.arange(q_len, device=self.device)
                        vis = torch.cat(
                            [
                                torch.ones(q_len, n, dtype=torch.bool, device=self.device),
                                j.view(-1, 1) >= j.view(1, -1),
                            ],
                            dim=1,
                        )
                        logits = logits.masked_fill(~vis, float("-inf"))
                    out[b, hq] = torch.softmax(logits, dim=-1) @ v
        return out.transpose(1, 2).contiguous().to(query.dtype)

    # ------------------------------------------------------------------
    def memory_bytes(self) -> int:
        """Bytes held by the paged k/v pool -- what the compression claim is measured on."""
        return self.k_pool.numel() * self.k_pool.element_size() * 2

    def summary(self) -> str:
        logical = int(self.row_budget.sum())
        physical = self.k_pool.shape[0] * PAGE_BLOCK
        return (
            f"{self.rows} rows ({self.n_layers}L x {self.batch_size}B x {self.n_kv_heads}H), "
            f"budget {int(self.row_budget.min())}..{int(self.row_budget.max())}, "
            f"{logical} logical slots in {physical} paged ({physical / max(logical, 1) - 1:.1%} "
            f"padding), {self.memory_bytes() / 2 ** 30:.3f} GiB"
        )


class EvictCacheLayer(CacheLayerMixin):
    """
    One layer's view of an :class:`EvictPagedPool`, as a transformers cache layer.

    ``update`` is a deliberate **pass-through**: it records the new tokens and hands them straight
    back. Nothing is written to the pool here, and that is not laziness -- the eviction step needs
    the router's score for the arriving token, which is computed from the *hidden state*, and a
    cache layer never sees one. The attention implementation does (via the pre-hook that
    :class:`~.sparse_inference.SparseAttentionContext` installs), so it owns both the ingest and
    the attend, and this class exists only to satisfy the ``Cache`` protocol and to keep the
    logical length correct.
    """

    is_sliding = False

    def __init__(self, pool: EvictPagedPool, layer_idx: int):
        super().__init__()
        self.pool = pool
        self.layer_idx = layer_idx

    def lazy_initialization(self, key_states: torch.Tensor):  # pragma: no cover - pool preallocates
        self.dtype, self.device = key_states.dtype, key_states.device

    def update(self, key_states, value_states, cache_kwargs=None):
        """Return the new tokens unchanged; the attention implementation does the real work."""
        return key_states, value_states

    def get_seq_length(self) -> int:
        """The **logical** number of tokens seen, not the physical slot count.

        These two diverge the moment anything is evicted, and the distinction is load-bearing:
        ``GenerationMixin._get_initial_cache_position`` slices ``cache_position`` by this value, so
        returning the (smaller) physical length would rewind every subsequent token's RoPE
        position. Reported as the max over the batch, which is what a single scalar can mean when
        sequences are at different lengths.
        """
        return int(self.pool.seen.max())

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Never actually consulted -- see :class:`EvictCache` -- but answered honestly anyway."""
        return int(self.pool.filled.max()) + cache_position.shape[0], 0

    def get_max_cache_shape(self) -> int:
        return int(self.pool.row_budget.max())

    def reset(self) -> None:  # pragma: no cover - a fresh pool is built per context instead
        raise NotImplementedError(
            "an EvictPagedPool holds one compressed context; build a new one rather than "
            "resetting, so a stale block table can never be read against new keys."
        )

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        raise NotImplementedError(
            "beam search would have to permute the block table across the (layer, batch, head) "
            "row layout; use greedy or sampling decoding."
        )


class EvictCache(Cache):
    """
    A transformers :class:`~transformers.Cache` backed by one :class:`EvictPagedPool`.

    Exists so the hard-eviction path can be driven by ordinary ``model(...)`` calls. One detail
    makes that work without any masking support: the sparse attention implementation is registered
    in ``ALL_ATTENTION_FUNCTIONS`` but **not** in ``ALL_MASK_ATTENTION_FUNCTIONS``, so
    ``_preprocess_mask_arguments`` takes its "custom attention backend" early exit and hands every
    forward ``attention_mask=None``. That is exactly right here -- an evicted cache needs no mask --
    but it also means padding information never reaches the attention, which is why sequences are
    prefilled and committed one at a time rather than as a padded batch.
    """

    def __init__(self, pool: EvictPagedPool):
        super().__init__(layers=[EvictCacheLayer(pool, i) for i in range(pool.n_layers)])
        self.pool = pool

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        return int(self.pool.seen.max())





