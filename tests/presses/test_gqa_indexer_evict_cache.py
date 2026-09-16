# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
The foundation of the hard-eviction path: **the incremental rule selects the same keys**.

:mod:`~kvpress.presses.gqa_indexer.qi_flex_attention` keeps the whole cache and masks it, deriving
each query row's support from a per-key ``deadline``. The eviction path instead *throws keys away*,
one per step, and can never get them back. Those two are only interchangeable if the incremental
"ring rolls, argmin loses" rule retains exactly the set the deadline mask would have shown the last
row -- so that is what these tests assert, as a **set identity with symmetric difference 0**, not a
tolerance.

Why an exact identity is available at all
-----------------------------------------
``deadlines`` documents the irreversibility: query row ``t``'s pool horizon only grows, and the
count of keys beating ``j`` is non-decreasing in ``t``, so once ``j`` drops out it never returns.
The incremental form of that same statement is the rule implemented here: each new token pushes
exactly one key out of the local window into the evictable pool, and the pool then sheds its
current minimum. Replaying it to the end must land on the deadline's survivors.

The tie-break is load-bearing, not a detail
-------------------------------------------
A bf16 score resolves only ~12% distinct values at ``L=8030``, so 95% of keys share a score with
another key (measured -- see ``qi_flex_attention``'s module docstring). ``deadlines`` orders equal
scores by ascending key index, matching a stable descending sort, which means **among equals the
LARGER position is the loser**. The tests therefore feed bf16-rounded scores rather than fp32
noise: with fp32 scores ties essentially never occur and a wrong tie-break would pass silently.
"""

from __future__ import annotations

import pytest
import torch

from kvpress.presses.gqa_indexer.evict_cache import POS_BITS, rank_key
from kvpress.presses.gqa_indexer.qi_flex_attention import deadlines


def deadline_survivors(dl: torch.Tensor, *, k_len: int, force_sink: int, force_local: int):
    """The set each head keeps at the LAST row, read off ``deadlines``' own contract.

    ``mask_mod`` keeps key ``j`` for a row with horizon ``hi`` iff ``j`` is a sink, is inside the
    row's local window, or (is in the pool, has arrived, and ``hi <= deadline[j]``). Evaluated at
    the final row, whose horizon is ``k_len - 1 - force_local``, this is the committed keep-set
    that every subsequent decode step inherits.
    """
    horizon = max(k_len - 1 - force_local, 0)
    out = []
    for h in range(dl.shape[0]):
        keep = set(range(min(force_sink, k_len)))
        keep |= set(range(max(k_len - force_local, force_sink), k_len))
        alive = (horizon <= dl[h]).nonzero().flatten().tolist()
        keep |= {j for j in alive if force_sink <= j <= horizon}
        out.append(keep)
    return out


def replay_incremental(scores: torch.Tensor, budgets, *, force_sink: int, force_local: int):
    """Replay the pool's decode rule from an empty cache, vectorized over heads.

    This mirrors what :class:`~kvpress.presses.gqa_indexer.evict_cache.EvictPagedPool` does one
    token at a time, and is deliberately written against ``budgets`` as a **vector** so the ragged
    (per-head) case exercises the same code as the uniform one.
    """
    n_heads, k_len = scores.shape
    budgets = torch.as_tensor(budgets, dtype=torch.int64).reshape(n_heads)
    take = (budgets - force_sink - force_local).clamp(min=0)
    width = int(take.max()) if int(take.max()) > 0 else 1
    NEG = torch.iinfo(torch.int64).min

    pool_pos = torch.full((n_heads, width), -1, dtype=torch.int64)
    pool_key = torch.full((n_heads, width), NEG, dtype=torch.int64)
    # Slots past a head's own budget are pinned to +inf so the argmin can never land on them. The
    # pool buffer is rectangular at `max_h take_h` (that is what makes this vectorizable), so
    # without this a narrow head would "evict" an unused padding slot and grow past its budget.
    slot_ax = torch.arange(width).view(1, width)
    pool_key = pool_key.masked_fill(slot_ax >= take.view(n_heads, 1), torch.iinfo(torch.int64).max)
    filled = torch.zeros(n_heads, dtype=torch.int64)
    ring: list[int] = []

    for i in range(k_len):
        if i < force_sink:
            continue
        ring.append(i)
        if len(ring) <= force_local:
            continue
        demoted = ring.pop(0)
        d_pos = torch.full((n_heads,), demoted, dtype=torch.int64)
        d_key = rank_key(scores[:, demoted], d_pos)

        # A head still below its own budget simply appends: nothing is evicted while growing.
        growing = filled < take
        rows = growing.nonzero().flatten()
        if rows.numel():
            slot = filled.clamp(max=width - 1)
            pool_pos[rows, slot[rows]] = d_pos[rows]
            pool_key[rows, slot[rows]] = d_key[rows]
            filled = filled + growing.to(torch.int64)

        # A head at its budget sheds its current minimum -- but only if the arriving key beats it.
        full = (~growing) & (take > 0)
        if bool(full.any()):
            j_min = pool_key.argmin(-1)
            k_min = pool_key.gather(-1, j_min.unsqueeze(-1)).squeeze(-1)
            win = full & (d_key > k_min)
            rows = win.nonzero().flatten()
            if rows.numel():
                pool_pos[rows, j_min[rows]] = d_pos[rows]
                pool_key[rows, j_min[rows]] = d_key[rows]

    out = []
    for h in range(n_heads):
        live = pool_pos[h][pool_pos[h] >= 0].tolist()
        out.append(set(range(min(force_sink, k_len))) | set(live) | set(ring))
    return out


def bf16_scores(n_heads: int, k_len: int, seed: int) -> torch.Tensor:
    """Scores with REALISTIC tie density: bf16-rounded, so many keys share a value."""
    torch.manual_seed(seed)
    return torch.randn(n_heads, k_len).to(torch.bfloat16).float()


# ----------------------------------------------------------------------------------------------
# The load-bearing identity
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "k_len, n_heads, force_sink, force_local, take",
    [
        (344, 4, 4, 29, 80),
        (479, 4, 0, 33, 103),
        (616, 3, 4, 27, 102),
        (739, 2, 2, 25, 104),
        (513, 4, 4, 29, 70),
        (128, 2, 4, 16, 32),
    ],
)
def test_incremental_keeps_exactly_the_deadline_survivors(
    k_len, n_heads, force_sink, force_local, take
):
    """Replaying the decode rule lands on ``deadlines``' keep-set, entry for entry."""
    scores = bf16_scores(n_heads, k_len, seed=k_len)
    topk = take + force_sink + force_local

    dl = deadlines(scores, topk, force_sink=force_sink, force_local=force_local)
    want = deadline_survivors(dl, k_len=k_len, force_sink=force_sink, force_local=force_local)
    got = replay_incremental(
        scores, [topk] * n_heads, force_sink=force_sink, force_local=force_local
    )

    for h in range(n_heads):
        assert want[h] == got[h], (
            f"head {h}: {len(want[h] ^ got[h])} keys differ "
            f"(incremental kept {len(got[h])}, deadline kept {len(want[h])})"
        )


@pytest.mark.parametrize(
    "budgets, force_sink, force_local",
    [
        ([200, 90, 340, 120], 4, 24),
        ([64, 300, 64, 500], 2, 16),
        ([150, 150, 151, 149], 4, 32),
    ],
)
def test_ragged_per_head_budgets_match_too(budgets, force_sink, force_local):
    """The per-head (head_budget) case: ``deadlines`` takes a vector, and so must the replay.

    This is the configuration hard eviction exists to serve -- a ragged budget is what makes the
    physical cache smaller than ``n_heads * topk`` -- so it gets its own identity check rather
    than riding on the uniform one.
    """
    k_len = 640
    scores = bf16_scores(len(budgets), k_len, seed=7)
    budget_t = torch.tensor(budgets, dtype=torch.int64)

    dl = deadlines(scores, budget_t, force_sink=force_sink, force_local=force_local)
    want = deadline_survivors(dl, k_len=k_len, force_sink=force_sink, force_local=force_local)
    got = replay_incremental(scores, budgets, force_sink=force_sink, force_local=force_local)

    for h in range(len(budgets)):
        assert want[h] == got[h], f"head {h} (budget {budgets[h]}): {len(want[h] ^ got[h])} differ"
        # And the head really is holding its own budget, not the widest one.
        assert len(got[h]) <= budgets[h], f"head {h} kept {len(got[h])} > budget {budgets[h]}"


def test_cache_shorter_than_budget_keeps_everything():
    """A context that never fills the budget must evict nothing -- the growing path."""
    k_len, n_heads = 100, 3
    scores = bf16_scores(n_heads, k_len, seed=3)
    got = replay_incremental(scores, [512] * n_heads, force_sink=4, force_local=16)
    for h in range(n_heads):
        assert got[h] == set(range(k_len)), f"head {h} dropped {k_len - len(got[h])} keys early"


# ----------------------------------------------------------------------------------------------
# The tie-break, isolated
# ----------------------------------------------------------------------------------------------
def test_rank_key_is_monotonic_in_score():
    """The packed key must order exactly as the float does, including negatives and zero."""
    torch.manual_seed(0)
    scores = torch.cat([torch.randn(4000), torch.zeros(8), torch.tensor([-0.0, 1e-30, -1e-30])])
    pos = torch.zeros_like(scores, dtype=torch.int64)
    keys = rank_key(scores, pos)
    order_float = torch.argsort(scores, stable=True)
    order_key = torch.argsort(keys, stable=True)
    assert torch.equal(order_float, order_key)


def test_equal_scores_evict_the_later_position():
    """Among equal scores the LARGER index must lose, matching a stable descending sort.

    ``deadlines`` ranks equal scores by ascending key index, so the *newer* of two tied keys is
    the one squeezed out. Getting this backwards changes the selected set on real inputs, where
    95% of keys share a score with another.
    """
    tied = torch.full((1, 4), 0.5)
    pos = torch.arange(4).view(1, 4)
    keys = rank_key(tied, pos)
    assert int(keys.argmin(-1)) == 3, "the largest position should be the loser among equals"
    # strictly decreasing in position, so the order is total (no accidental equalities)
    assert torch.all(keys[0, 1:] < keys[0, :-1])


def test_score_dominates_position_in_the_packed_key():
    """A better score must win regardless of position -- the pack must not let position leak up."""
    scores = torch.tensor([[1.0, 1.0009766]])  # adjacent bf16 values
    pos = torch.tensor([[0, (1 << POS_BITS) - 1]])  # worst case: the winner is the newest key
    keys = rank_key(scores, pos)
    assert int(keys.argmax(-1)) == 1, "the higher score must win even at the maximum position gap"


# ----------------------------------------------------------------------------------------------
# The pool itself: commit, then decode, against the same identity
# ----------------------------------------------------------------------------------------------
CUDA_ONLY = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

POOL_SINK, POOL_LOCAL = 4, 16


def make_pool(budgets, *, n_layers=1, batch_size=1, head_dim=16, decay=False, device="cuda"):
    from kvpress.presses.gqa_indexer.evict_cache import EvictPagedPool

    table = torch.tensor([list(budgets)] * n_layers)
    return EvictPagedPool(
        table,
        batch_size=batch_size,
        n_layers=n_layers,
        n_kv_heads=len(budgets),
        n_sink=POOL_SINK,
        n_local=POOL_LOCAL,
        head_dim=head_dim,
        device=torch.device(device),
        dtype=torch.bfloat16,
        decay_ref=16384.0 if decay else None,
    )


def router_state(n_heads, total, device, *, decay, seed=1):
    torch.manual_seed(seed)
    mag = torch.randn(n_heads, total, device=device).to(torch.bfloat16).float()
    log_beta = -torch.rand(n_heads, total, device=device) if decay else None
    return mag, log_beta


def pool_held_positions(pool, layer, seq, n_heads):
    """Every position the pool physically holds: sinks + tracked pool + the live window."""
    rows = pool.rows_for(layer, seq)
    out = []
    for h in range(n_heads):
        row = rows.start + h
        p = pool.pool_pos[row]
        held = set(p[p >= 0].tolist())
        r = pool.ring_pos[row]
        held |= set(r[r >= 0].tolist())
        out.append(set(range(POOL_SINK)) | held)
    return out


@CUDA_ONLY
@pytest.mark.parametrize("decay", [False, True])
@pytest.mark.parametrize("budgets", [[100, 140], [150, 220, 151, 300]])
def test_commit_retains_the_deadline_keepset(decay, budgets):
    """The one-shot compression must land on exactly what the mask path would have shown."""
    n_heads = len(budgets)
    k_len = 640
    dev = torch.device("cuda")
    pool = make_pool(budgets, decay=decay)
    key = torch.randn(n_heads, k_len, 16, device=dev, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    mag, log_beta = router_state(n_heads, k_len, dev, decay=decay)

    pool.commit(0, 0, key=key, value=value, mag=mag, log_beta=log_beta, k_len=k_len)

    scores = mag if log_beta is None else mag + log_beta * ((k_len - 1) / 16384.0)
    dl = deadlines(
        scores, torch.tensor(budgets, device=dev),
        force_sink=POOL_SINK, force_local=POOL_LOCAL,
    )
    want = deadline_survivors(dl, k_len=k_len, force_sink=POOL_SINK, force_local=POOL_LOCAL)
    got = pool_held_positions(pool, 0, 0, n_heads)
    for h in range(n_heads):
        assert want[h] == got[h], f"head {h}: {len(want[h] ^ got[h])} keys differ"
        # and the row is exactly at its budget, not over it
        row = pool.rows_for(0, 0).start + h
        assert int(pool.filled[row]) == budgets[h]


@CUDA_ONLY
@pytest.mark.parametrize("decay", [False, True])
def test_decode_tracks_the_deadline_over_many_steps(decay):
    """After the commit the pool only sees one token at a time, and must still agree.

    This is the property hard eviction lives or dies on: a decode step can shed at most one key,
    so if the incremental rule drifted from the deadline the two arms would diverge further with
    every generated token.
    """
    budgets = [100, 140, 101, 180]
    n_heads, ctx, steps = len(budgets), 400, 120
    total = ctx + steps
    dev = torch.device("cuda")
    pool = make_pool(budgets, decay=decay)
    key = torch.randn(n_heads, total, 16, device=dev, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    mag, log_beta = router_state(n_heads, total, dev, decay=decay)

    pool.commit(
        0, 0, key=key[:, :ctx], value=value[:, :ctx], mag=mag[:, :ctx],
        log_beta=None if log_beta is None else log_beta[:, :ctx], k_len=ctx,
    )
    for t in range(ctx, total):
        pool.ingest(
            0,
            key=key[:, t].unsqueeze(0), value=value[:, t].unsqueeze(0),
            mag=mag[:, t].unsqueeze(0),
            log_beta=None if log_beta is None else log_beta[:, t].unsqueeze(0),
            positions=torch.tensor([t], device=dev),
        )
        pool.seen[0] = t + 1

    scores = mag if log_beta is None else mag + log_beta * ((total - 1) / 16384.0)
    dl = deadlines(
        scores, torch.tensor(budgets, device=dev),
        force_sink=POOL_SINK, force_local=POOL_LOCAL,
    )
    want = deadline_survivors(dl, k_len=total, force_sink=POOL_SINK, force_local=POOL_LOCAL)
    got = pool_held_positions(pool, 0, 0, n_heads)
    for h in range(n_heads):
        assert want[h] == got[h], (
            f"head {h} (budget {budgets[h]}) drifted after {steps} steps: "
            f"{len(want[h] ^ got[h])} keys differ"
        )


@CUDA_ONLY
def test_budget_never_grows_during_decode():
    """A narrow head must not creep up to the rectangular pool width.

    The pool metadata is rectangular at ``max_h take_h`` so the eviction step can be one
    vectorized argmin, which means a narrow row has padding slots carrying ``pool_pos == -1`` --
    indistinguishable from "free" unless the padding mask is consulted. Measured before that fix:
    a budget-100 head grew from 80 to 120 tracked keys over 120 steps, holding 40 it had no
    budget for, while ``filled`` still read a reassuring 100.
    """
    budgets = [100, 300]  # a wide sibling, so the rectangular width is much larger than take[0]
    n_heads, ctx, steps = 2, 400, 150
    total = ctx + steps
    dev = torch.device("cuda")
    pool = make_pool(budgets)
    key = torch.randn(n_heads, total, 16, device=dev, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    mag, _ = router_state(n_heads, total, dev, decay=False)

    pool.commit(0, 0, key=key[:, :ctx], value=value[:, :ctx], mag=mag[:, :ctx],
                log_beta=None, k_len=ctx)
    for t in range(ctx, total):
        pool.ingest(0, key=key[:, t].unsqueeze(0), value=value[:, t].unsqueeze(0),
                    mag=mag[:, t].unsqueeze(0), log_beta=None,
                    positions=torch.tensor([t], device=dev))
        pool.seen[0] = t + 1
        for h in range(n_heads):
            row = pool.rows_for(0, 0).start + h
            live = int((pool.pool_pos[row] >= 0).sum())
            assert live <= int(pool.take[row]), (
                f"t={t} head {h}: {live} tracked keys against take={int(pool.take[row])}"
            )
            assert int(pool.filled[row]) == budgets[h]


@CUDA_ONLY
def test_short_context_commits_whole_and_then_grows():
    """A context inside every budget evicts nothing, and keeps not evicting as it grows."""
    budgets = [200, 260]
    n_heads, ctx, steps = 2, 60, 80
    total = ctx + steps
    dev = torch.device("cuda")
    pool = make_pool(budgets)
    key = torch.randn(n_heads, total, 16, device=dev, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    mag, _ = router_state(n_heads, total, dev, decay=False)

    pool.commit(0, 0, key=key[:, :ctx], value=value[:, :ctx], mag=mag[:, :ctx],
                log_beta=None, k_len=ctx)
    assert pool_held_positions(pool, 0, 0, n_heads)[0] == set(range(ctx))

    for t in range(ctx, total):
        pool.ingest(0, key=key[:, t].unsqueeze(0), value=value[:, t].unsqueeze(0),
                    mag=mag[:, t].unsqueeze(0), log_beta=None,
                    positions=torch.tensor([t], device=dev))
        pool.seen[0] = t + 1
    # Still below every budget, so nothing may have been dropped.
    for h in range(n_heads):
        assert pool_held_positions(pool, 0, 0, n_heads)[h] == set(range(total))
        assert int(pool.filled[pool.rows_for(0, 0).start + h]) == total


@CUDA_ONLY
def test_pool_rejects_a_budget_the_pins_already_consume():
    """A head whose sink+local already fill its budget can retain nothing and must be refused."""
    with pytest.raises(ValueError, match="evictable slots"):
        make_pool([POOL_SINK + POOL_LOCAL, 200])


# ----------------------------------------------------------------------------------------------
# Attention over the compressed cache
# ----------------------------------------------------------------------------------------------
def _held_kv(pool, layer, seq, h, device):
    """Read one row's live slots back through the block table, as fp32."""
    from kvpress.presses.gqa_indexer.evict_cache import PAGE_BLOCK

    row = pool.rows_for(layer, seq).start + h
    n = int(pool.filled[row])
    slots = torch.arange(n, device=device)
    blk = pool.block_table[row, slots // PAGE_BLOCK].to(torch.int64)
    return (
        pool.k_pool[blk, slots % PAGE_BLOCK, 0, :].float(),
        pool.v_pool[blk, slots % PAGE_BLOCK, 0, :].float(),
    )


def _committed_pool(budgets, *, head_dim, k_len, device, group):
    pool = make_pool(budgets, head_dim=head_dim)
    n_heads = len(budgets)
    key = (torch.randn(n_heads, k_len, head_dim, device=device) * 0.3).to(torch.bfloat16)
    value = (torch.randn(n_heads, k_len, head_dim, device=device) * 0.3).to(torch.bfloat16)
    mag, _ = router_state(n_heads, k_len, device, decay=False)
    pool.commit(0, 0, key=key, value=value, mag=mag, log_beta=None, k_len=k_len)
    return pool


@CUDA_ONLY
def test_decode_attention_matches_fp32_over_the_held_keys():
    """A decode row attends over its row's slots **plus its own key** -- the causal diagonal.

    The pool is updated only after the forward, so the arriving token is never in it yet and has
    to be supplied as the second branch. Omitting it lets the query at position t attend to its
    history but not to itself, which measured cwe 95.71 -> 24.29 and vt 100.00 -> 40.00 on
    RULER 4096 while leaving the pure-needle tasks untouched.
    """
    budgets, head_dim, group = [300, 420, 301, 560], 128, 4
    n_heads = len(budgets)
    n_q = n_heads * group
    dev = torch.device("cuda")
    scale = head_dim ** -0.5
    pool = _committed_pool(budgets, head_dim=head_dim, k_len=700, device=dev, group=group)

    q = (torch.randn(1, n_q, 1, head_dim, device=dev) * 0.3).to(torch.bfloat16)
    nk = (torch.randn(1, n_heads, 1, head_dim, device=dev) * 0.3).to(torch.bfloat16)
    nv = (torch.randn(1, n_heads, 1, head_dim, device=dev) * 0.3).to(torch.bfloat16)
    out = pool.attend(0, q, scaling=scale, new_key=nk, new_value=nv)
    assert out.shape == (1, 1, n_q, head_dim)

    worst = 0.0
    for h in range(n_heads):
        k, v = _held_kv(pool, 0, 0, h, dev)
        assert k.shape[0] == budgets[h]
        # The reference includes the arriving key, exactly as a dense causal forward would.
        kc = torch.cat([k, nk[0, h].float()])
        vc = torch.cat([v, nv[0, h].float()])
        for g in range(group):
            hq = h * group + g
            w = torch.softmax(q[0, hq, 0].float() @ kc.T * scale, -1)
            worst = max(worst, (w @ vc - out[0, 0, hq].float()).abs().max().item())
    assert worst < 1e-2, f"decode attention off by {worst:.2e}"


@CUDA_ONLY
def test_attend_refuses_a_forward_without_its_own_keys():
    """Silently dropping the diagonal is the costliest bug found here, so it must raise."""
    pool = _committed_pool([300, 420], head_dim=64, k_len=500, device=torch.device("cuda"), group=2)
    q = torch.randn(1, 4, 1, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="new_key"):
        pool.attend(0, q, scaling=0.125)


@CUDA_ONLY
def test_multi_row_forward_merges_the_two_branches_exactly():
    """The question forward: cache branch (all visible) + its own rows (causal), merged by lse.

    The new tokens cannot simply be written into the window and then attended over -- row ``t``
    would see rows ``> t``. Merging two softmaxes through their log-sum-exps is an identity, so
    this must match a single fp32 softmax over the concatenation.
    """
    budgets, head_dim, group, q_len = [300, 420, 301, 560], 128, 4, 28
    n_heads = len(budgets)
    n_q = n_heads * group
    dev = torch.device("cuda")
    scale = head_dim ** -0.5
    pool = _committed_pool(budgets, head_dim=head_dim, k_len=700, device=dev, group=group)

    q = (torch.randn(1, n_q, q_len, head_dim, device=dev) * 0.3).to(torch.bfloat16)
    nk = (torch.randn(1, n_heads, q_len, head_dim, device=dev) * 0.3).to(torch.bfloat16)
    nv = (torch.randn(1, n_heads, q_len, head_dim, device=dev) * 0.3).to(torch.bfloat16)
    out = pool.attend(0, q, scaling=scale, new_key=nk, new_value=nv)
    assert out.shape == (1, q_len, n_q, head_dim)

    worst = 0.0
    for h in range(n_heads):
        k, v = _held_kv(pool, 0, 0, h, dev)
        n = k.shape[0]
        kc = torch.cat([k, nk[0, h].float()])
        vc = torch.cat([v, nv[0, h].float()])
        for g in range(group):
            hq = h * group + g
            for t in range(q_len):
                logits = q[0, hq, t].float() @ kc.T * scale
                logits[n + t + 1 :] = float("-inf")
                ref = torch.softmax(logits, -1) @ vc
                worst = max(worst, (ref - out[0, t, hq].float()).abs().max().item())
    assert worst < 1e-2, f"two-branch merge off by {worst:.2e}"


@CUDA_ONLY
def test_multi_row_forward_requires_its_own_keys():
    """Refuse rather than silently attend a multi-row forward against the cache alone."""
    pool = _committed_pool([300, 420], head_dim=64, k_len=500, device=torch.device("cuda"), group=2)
    q = torch.randn(1, 4, 8, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="new_key"):
        pool.attend(0, q, scaling=0.125)


@CUDA_ONLY
@pytest.mark.parametrize("q_len", [1, 28])
def test_torch_fallback_agrees_with_the_kernel(q_len, monkeypatch):
    """The pure-torch reference exists for CPU boxes and must not be a different model."""
    import kvpress.presses.gqa_indexer.evict_cache as ec

    budgets, head_dim, group = [300, 420], 64, 2
    n_q = len(budgets) * group
    dev = torch.device("cuda")
    scale = head_dim ** -0.5
    pool = _committed_pool(budgets, head_dim=head_dim, k_len=500, device=dev, group=group)

    q = (torch.randn(1, n_q, q_len, head_dim, device=dev) * 0.3).to(torch.bfloat16)
    kw = {
        "new_key": (torch.randn(1, 2, q_len, head_dim, device=dev) * 0.3).to(torch.bfloat16),
        "new_value": (torch.randn(1, 2, q_len, head_dim, device=dev) * 0.3).to(torch.bfloat16),
    }

    fast = pool.attend(0, q, scaling=scale, **kw)
    monkeypatch.setattr(ec, "HAS_FLASH", False)
    slow = pool.attend(0, q, scaling=scale, **kw)
    assert (fast - slow).abs().max().item() < 1e-2


@CUDA_ONLY
@pytest.mark.parametrize("decay", [False, True])
def test_all_layers_ingest_equals_per_layer_ingest(decay):
    """Absorbing every layer in one call must be identical to 36 separate calls.

    Decode uses the batched form for throughput -- one call issues ~90 small kernels totalling
    only 204 us of *device* time, so per-layer calls are bound by CPU launch dispatch and cost
    36x more (measured 32 ms/token against 0.9 ms). This test is what makes that an optimization
    rather than a second implementation.
    """
    budgets = [100, 140, 101, 180]
    n_layers, n_heads, ctx, steps = 4, len(budgets), 300, 40
    total = ctx + steps
    dev = torch.device("cuda")
    head_dim = 16

    pools = [
        make_pool(budgets, n_layers=n_layers, head_dim=head_dim, decay=decay) for _ in range(2)
    ]
    key = torch.randn(n_layers, n_heads, total, head_dim, device=dev, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    torch.manual_seed(5)
    mag = torch.randn(n_layers, n_heads, total, device=dev).to(torch.bfloat16).float()
    log_beta = -torch.rand(n_layers, n_heads, total, device=dev) if decay else None

    for pool in pools:
        for layer in range(n_layers):
            pool.commit(
                layer, 0, key=key[layer, :, :ctx], value=value[layer, :, :ctx],
                mag=mag[layer, :, :ctx],
                log_beta=None if log_beta is None else log_beta[layer, :, :ctx], k_len=ctx,
            )

    every = list(range(n_layers))
    for t in range(ctx, total):
        pos = torch.tensor([t], device=dev)
        # pools[0]: one call for every layer. pools[1]: one call per layer.
        pools[0].ingest(
            every,
            key=key[:, :, t].unsqueeze(1), value=value[:, :, t].unsqueeze(1),
            mag=mag[:, :, t].unsqueeze(1),
            log_beta=None if log_beta is None else log_beta[:, :, t].unsqueeze(1),
            positions=pos,
        )
        for layer in range(n_layers):
            pools[1].ingest(
                layer,
                key=key[layer, :, t].unsqueeze(0), value=value[layer, :, t].unsqueeze(0),
                mag=mag[layer, :, t].unsqueeze(0),
                log_beta=None if log_beta is None else log_beta[layer, :, t].unsqueeze(0),
                positions=pos,
            )

    for layer in range(n_layers):
        a = pool_held_positions(pools[0], layer, 0, n_heads)
        b = pool_held_positions(pools[1], layer, 0, n_heads)
        for h in range(n_heads):
            assert a[h] == b[h], f"layer {layer} head {h}: {len(a[h] ^ b[h])} keys differ"
    assert torch.equal(pools[0].filled, pools[1].filled)
    assert torch.equal(pools[0].pool_live, pools[1].pool_live)


@CUDA_ONLY
def test_ingest_rejects_a_token_count_that_does_not_match_its_rows():
    """A layer-list ingest with per-layer-shaped tensors would silently write the wrong rows."""
    budgets = [100, 140]
    pool = make_pool(budgets, n_layers=3)
    dev = torch.device("cuda")
    key = torch.randn(1, len(budgets), 16, device=dev, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="arriving tokens"):
        pool.ingest(
            [0, 1, 2], key=key, value=key,
            mag=torch.randn(1, len(budgets), device=dev), log_beta=None,
            positions=torch.tensor([50], device=dev),
        )


# ----------------------------------------------------------------------------------------------
# End to end, on a real model
# ----------------------------------------------------------------------------------------------
transformers = pytest.importorskip("transformers")
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM  # noqa: E402

E2E_SINK, E2E_LOCAL, E2E_TOPK = 4, 32, 192


def _e2e_model_and_press():
    from kvpress import GQAIndexerPress

    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=256, hidden_size=128, intermediate_size=256, num_hidden_layers=3,
        num_attention_heads=8, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=4096, attn_implementation="sdpa",
    )
    model = Qwen3ForCausalLM(cfg).to("cuda").to(torch.bfloat16).eval()
    press = GQAIndexerPress(
        compression_ratio=0.0, gate_scale=True, scorer="scalar", scalar_decay=True
    )
    press.post_init_from_model(model)
    budgets = torch.full((cfg.num_hidden_layers, cfg.num_key_value_heads), E2E_TOPK)
    return model, press, budgets, cfg


def _decode_evict(model, press, budgets, id_list, steps):
    """Prefill + commit each sequence, then decode them as one batch."""
    from kvpress.presses.gqa_indexer.evict_runner import EvictInferenceContext

    batch = len(id_list)
    sparse_kwargs = dict(topk=E2E_TOPK, force_sink=E2E_SINK, force_local=E2E_LOCAL)
    with EvictInferenceContext(
        model, press, budgets=budgets, n_sink=E2E_SINK, n_local=E2E_LOCAL, batch_size=batch
    ) as ec:
        for s, ids in enumerate(id_list):
            ec.prefill_and_commit(ids, s, sparse_kwargs)
        ec.activate()
        cache = ec.new_cache()
        seen = ec.pool.seen[:batch].clone()
        nxt = torch.cat([x[:, -1:] for x in id_list], 0)
        out = [[] for _ in range(batch)]
        for step in range(steps):
            logits = model(
                input_ids=nxt, past_key_values=cache,
                position_ids=(seen + step).view(-1, 1),
            ).logits
            ec.finish_step()
            nxt = logits[:, -1].argmax(-1).view(-1, 1)
            for b in range(batch):
                out[b].append(int(nxt[b]))
        filled = ec.pool.filled.clone()
    return out, filled


@CUDA_ONLY
def test_end_to_end_decode_equals_dense_when_nothing_is_evicted():
    """At a budget above the whole sequence, eviction is a no-op and must reproduce DENSE decode.

    Dense is used as the reference rather than the masking arm deliberately. Comparing two
    approximate paths cannot say which one is wrong, and that ambiguity hid a real bug: with the
    causal diagonal dropped, this test still passed against the masking arm for six steps before
    diverging, while the eval had already lost 18.6 RULER points. Against dense there is a single
    right answer.
    """
    from kvpress.presses.gqa_indexer.evict_runner import EvictInferenceContext

    model, press, _, cfg = _e2e_model_and_press()
    ctx, steps = 256, 12
    big = 4096  # > ctx + steps, so no key is ever evicted
    ids = torch.randint(0, cfg.vocab_size, (1, ctx), device="cuda")
    kw = dict(topk=big, force_sink=E2E_SINK, force_local=E2E_LOCAL)

    def greedy(cache, finish=None):
        out, nxt = [], ids[:, -1:]
        for step in range(steps):
            logits = model(
                input_ids=nxt, past_key_values=cache,
                position_ids=torch.tensor([[ctx + step]], device="cuda"),
            ).logits
            if finish is not None:
                finish()
            nxt = logits[0, -1].argmax().view(1, 1)
            out.append(int(nxt))
        return out

    cache = DynamicCache()
    model.model(input_ids=ids, past_key_values=cache)
    dense = greedy(cache)

    budgets = torch.full((cfg.num_hidden_layers, cfg.num_key_value_heads), big)
    with EvictInferenceContext(
        model, press, budgets=budgets, n_sink=E2E_SINK, n_local=E2E_LOCAL, batch_size=1
    ) as ec:
        ec.prefill_and_commit(ids, 0, kw)
        ec.activate()
        evict = greedy(ec.new_cache(), finish=ec.finish_step)
    assert evict == dense, f"evict {evict} != dense {dense}"


@CUDA_ONLY
def test_end_to_end_decode_tracks_the_masking_path():
    """Under real compression the two arms select the same keys, so the FIRST token must match.

    Only the first token is asserted, and the reason is that this model has random weights: at
    ``topk=192`` of a 512-token context an untrained network is in a chaotic regime where both arms
    wander away from dense decode (measured 3/12 and 1/12 agreement with dense), so a
    token-sequence comparison here measures chaos, not correctness.

    The first token is different in kind: it is produced from the committed cache alone, before any
    eviction bookkeeping has run, so a mismatch there is a wiring bug -- a stale position, a
    mis-scored arrival, a slot written to the wrong row. Real quality is settled by the benchmark,
    where the two arms agree to within the bf16 accumulation-order noise that
    ``qi_flex_attention`` already documents for flex vs gather.
    """
    from kvpress import SparseAttentionContext

    model, press, budgets, cfg = _e2e_model_and_press()
    ctx, steps = 512, 12
    ids = torch.randint(0, cfg.vocab_size, (1, ctx), device="cuda")

    evict, filled = _decode_evict(model, press, budgets, [ids], steps)
    assert bool((filled == E2E_TOPK).all()), "eviction let the cache leave its budget"

    with SparseAttentionContext(
        model, press, topk=E2E_TOPK, force_sink=E2E_SINK, force_local=E2E_LOCAL
    ):
        cache = DynamicCache()
        model.model(input_ids=ids, past_key_values=cache)
        ref, nxt = [], ids[:, -1:]
        for step in range(steps):
            logits = model(
                input_ids=nxt, past_key_values=cache,
                position_ids=torch.tensor([[ctx + step]], device="cuda"),
            ).logits
            nxt = logits[0, -1].argmax().view(1, 1)
            ref.append(int(nxt))

    assert evict[0][0] == ref[0], (
        f"the FIRST generated token differs: evict {evict[0]} vs mask {ref}. That token comes "
        "from the committed cache alone, so this is a wiring bug rather than drift."
    )




@CUDA_ONLY
def test_batched_decode_matches_one_sequence_at_a_time():
    """Batching must not couple sequences -- including when their contexts differ in length.

    Each row carries its own logical position, so a batch whose sequences sit at different
    absolute positions is the normal case rather than a corner one: the router's recency tilt and
    decay age are both position-dependent, and sharing one scalar offset across the batch would
    mis-score every sequence but the first.
    """
    model, press, budgets, cfg = _e2e_model_and_press()
    lens, steps = [512, 300, 640], 10
    ids = [torch.randint(0, cfg.vocab_size, (1, n), device="cuda") for n in lens]

    batched, _ = _decode_evict(model, press, budgets, ids, steps)
    for b, n in enumerate(lens):
        single, _ = _decode_evict(model, press, budgets, [ids[b]], steps)
        assert batched[b] == single[0], (
            f"sequence {b} (context {n}) decoded differently in a batch: "
            f"{batched[b]} vs {single[0]}"
        )


@CUDA_ONLY
def test_query_aware_scorer_is_refused():
    """A pairwise router has no frozen rank, so eviction would delete keys a later query needs."""
    from kvpress import GQAIndexerPress
    from kvpress.presses.gqa_indexer.evict_runner import EvictInferenceContext

    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=256, attn_implementation="sdpa",
    )
    model = Qwen3ForCausalLM(cfg).to("cuda").to(torch.bfloat16).eval()
    press = GQAIndexerPress(compression_ratio=0.0, gate_scale=True, scorer="pairwise")
    budgets = torch.full((2, 2), 64)
    with pytest.raises(ValueError, match="query-independent|frozen|Hard eviction|rank"):
        with EvictInferenceContext(
            model, press, budgets=budgets, n_sink=E2E_SINK, n_local=E2E_LOCAL
        ):
            pass


@CUDA_ONLY
def test_generate_batches_answers_across_different_questions():
    """The full pipeline: context prefill -> per-sequence question -> batched answer.

    Contexts AND questions differ in length here, which is the normal case for a benchmark and
    the reason the question forwards are not themselves batched: left-padding them would put
    padding tokens inside the local window, which is pinned and so can never be evicted.
    """
    model, press, budgets, cfg = _e2e_model_and_press()
    from kvpress.presses.gqa_indexer.evict_runner import EvictInferenceContext

    lens, q_lens, steps = [512, 300, 640], [28, 15, 40], 16
    ctxs = [torch.randint(0, cfg.vocab_size, (1, n), device="cuda") for n in lens]
    qs = [torch.randint(0, cfg.vocab_size, (1, n), device="cuda") for n in q_lens]
    sparse_kwargs = dict(topk=E2E_TOPK, force_sink=E2E_SINK, force_local=E2E_LOCAL)

    def run(ctx_list, q_list):
        with EvictInferenceContext(
            model, press, budgets=budgets, n_sink=E2E_SINK, n_local=E2E_LOCAL,
            batch_size=len(ctx_list),
        ) as ec:
            for s, c in enumerate(ctx_list):
                ec.prefill_and_commit(c, s, sparse_kwargs)
            # eos_token_ids=[-1] disables early stopping, so every sequence runs the full length
            # and the comparison is over the same number of tokens.
            return ec.generate(q_list, max_new_tokens=steps, eos_token_ids=[-1])

    batched = run(ctxs, qs)
    assert all(len(a) == steps for a in batched)
    for b in range(len(lens)):
        single = run([ctxs[b]], [qs[b]])[0]
        assert batched[b] == single, (
            f"sequence {b} (context {lens[b]}, question {q_lens[b]}) differs in a batch"
        )


@CUDA_ONLY
def test_memory_is_refused():
    """`memory=True` needs the retained branch's lse, which the paged decode kernel never returns.

    Refused rather than ignored: silently dropping the flag would publish an eviction number under
    the name of a compensated run, and nothing in the shapes or the score would reveal it.
    (`cmp_slots` is *supported* here, through the streaming centroid update -- see
    `test_streaming_cmp_slots_participate_in_decode`.)
    """
    from kvpress.presses.gqa_indexer.evict_runner import EvictInferenceContext

    model, press, budgets, cfg = _e2e_model_and_press()
    ids = torch.randint(0, cfg.vocab_size, (1, 256), device="cuda")
    kwargs = dict(topk=E2E_TOPK, force_sink=E2E_SINK, force_local=E2E_LOCAL, memory=True)
    with EvictInferenceContext(
        model, press, budgets=budgets, n_sink=E2E_SINK, n_local=E2E_LOCAL
    ) as ec:
        with pytest.raises(ValueError, match="lse|deleted"):
            ec.prefill_and_commit(ids, 0, kwargs)


@CUDA_ONLY
def test_streaming_cmp_slots_participate_in_decode():
    """CMP slots must accumulate the CoT's own evicted keys and change the output.

    Both halves are asserted. The populations must GROW during generation -- that is the streaming
    property frozen CMP cannot have, and on a CoT benchmark (math500's context is a single space)
    the prefill evicts nothing at all, so growth during decode is the only way a slot ever exists.
    And the fused output must actually differ from the no-slot run, since a slot set that is
    present but inert would score identically and look like "CMP does not help".
    """
    from kvpress.presses.gqa_indexer.evict_runner import EvictInferenceContext

    model, press, budgets, cfg = _e2e_model_and_press()
    ctx, steps, R = 512, 24, 8
    ids = torch.randint(0, cfg.vocab_size, (1, ctx), device="cuda")
    kw = dict(topk=E2E_TOPK, force_sink=E2E_SINK, force_local=E2E_LOCAL, cmp_slots=R)

    def run(slots):
        with EvictInferenceContext(
            model, press, budgets=budgets, n_sink=E2E_SINK, n_local=E2E_LOCAL,
            batch_size=1, cmp_slots=slots,
        ) as ec:
            k = dict(kw) if slots else dict(topk=E2E_TOPK, force_sink=E2E_SINK,
                                            force_local=E2E_LOCAL)
            ec.prefill_and_commit(ids, 0, k)
            ec.activate()
            cache = ec.new_cache()
            after_commit = None if ec.cmp is None else ec.cmp.pop.sum().item()
            out, nxt = [], ids[:, -1:]
            for step in range(steps):
                logits = model(
                    input_ids=nxt, past_key_values=cache,
                    position_ids=torch.tensor([[ctx + step]], device="cuda"),
                ).logits
                ec.finish_step()
                nxt = logits[0, -1].argmax().view(1, 1)
                out.append(int(nxt))
            grew = None if ec.cmp is None else ec.cmp.pop.sum().item()
            return out, after_commit, grew

    with_slots, at_commit, after = run(R)
    without, _, _ = run(0)

    assert after > at_commit, (
        f"CMP populations did not grow during decode ({at_commit} -> {after}): the evicted CoT "
        "keys are not reaching the slots, which is the entire streaming mechanism."
    )
    assert with_slots != without, (
        "the slots are present but inert -- the fused output is identical to the no-slot run, so "
        "any benchmark A/B would be measuring nothing."
    )


@CUDA_ONLY
def test_cmp_slots_are_funded_out_of_the_budget():
    """R slots must come OUT OF the per-head budget, so the A/B is budget-matched.

    Subtracted from the resolved budgets rather than from `topk`, because a
    `head_budget=static` table's rows sum to the topk it was fitted at and the loader refuses any
    other value -- lowering topk would make the table and the run disagree about the total.
    """
    from kvpress.presses.gqa_indexer.evict_runner import EvictInferenceContext

    model, press, _, cfg = _e2e_model_and_press()
    R = 16
    ids = torch.randint(0, cfg.vocab_size, (1, 512), device="cuda")
    with EvictInferenceContext(
        model, press, n_sink=E2E_SINK, n_local=E2E_LOCAL, batch_size=1, cmp_slots=R
    ) as ec:
        ec.prefill_and_commit(ids, 0, dict(
            topk=E2E_TOPK, force_sink=E2E_SINK, force_local=E2E_LOCAL, cmp_slots=R,
        ))
        # The exact branch holds topk - R keys; the slots make up the difference.
        assert int(ec.pool.budgets[0, 0]) == E2E_TOPK - R, (
            f"budget {int(ec.pool.budgets[0, 0])} != topk - R = {E2E_TOPK - R}"
        )


@CUDA_ONLY
def test_cmp_slots_coexist_with_a_static_budget_table(tmp_path):
    """`--cmp_slots` and `--head_budget static` together: R comes off each ROW of the table.

    Regression test for a real conflict: funding the slots by lowering `topk` made the table's
    loader refuse the run outright ("fitted at topk=512 but this run uses topk=448"), because a
    table's row sums *are* its budget. Both must hold at once, and more than one document must
    resolve the same thing or the pool's fixed block table is invalid.
    """
    from kvpress.presses.gqa_indexer.evict_runner import EvictInferenceContext

    model, press, _, cfg = _e2e_model_and_press()
    n_layers, n_kv = cfg.num_hidden_layers, cfg.num_key_value_heads
    R = 16
    row = [E2E_TOPK // 2, E2E_TOPK + E2E_TOPK // 2] + [E2E_TOPK] * (n_kv - 2)
    table = torch.tensor([row] * n_layers)
    path = tmp_path / "t.pt"
    torch.save({"table": table, "topk": E2E_TOPK}, path)

    kw = dict(
        topk=E2E_TOPK, force_sink=E2E_SINK, force_local=E2E_LOCAL, cmp_slots=R,
        head_budget="static", head_budget_table=str(path),
    )
    with EvictInferenceContext(
        model, press, n_sink=E2E_SINK, n_local=E2E_LOCAL, batch_size=2, cmp_slots=R
    ) as ec:
        ec.prefill_and_commit(
            torch.randint(0, cfg.vocab_size, (1, 512), device="cuda"), 0, kw
        )
        assert torch.equal(
            ec.pool.budgets.cpu(), (table - R).to(ec.pool.budgets.dtype).cpu()
        ), f"expected the table minus R, got {ec.pool.budgets[0].tolist()}"
        # A second document must resolve the SAME budget -- i.e. the subtraction happens before
        # the consistency check, not after it.
        ec.prefill_and_commit(
            torch.randint(0, cfg.vocab_size, (1, 480), device="cuda"), 1, kw
        )




@CUDA_ONLY
def test_pool_reads_the_static_head_budget_table(tmp_path):
    """A ragged `head_budget=static` table must reach the POOL, not just the mask.

    This is the configuration hard eviction exists for: under the mask path a ragged budget only
    changes which keys are visible and saves nothing, while here each row is physically that size.
    The budget is resolved during the prefill, so the pool is allocated from it afterwards.
    """
    from kvpress.presses.gqa_indexer.evict_runner import EvictInferenceContext

    model, press, _, cfg = _e2e_model_and_press()
    n_layers, n_kv = cfg.num_hidden_layers, cfg.num_key_value_heads
    # Ragged, but conserving the per-layer total exactly -- otherwise the run would buy its score
    # with extra cache.
    row = [E2E_TOPK // 2, E2E_TOPK + E2E_TOPK // 2] + [E2E_TOPK] * (n_kv - 2)
    table = torch.tensor([row] * n_layers)
    assert int(table.sum(1)[0]) == E2E_TOPK * n_kv
    path = tmp_path / "table.pt"
    torch.save({"table": table, "topk": E2E_TOPK}, path)

    ctx = torch.randint(0, cfg.vocab_size, (1, 1024), device="cuda")
    with EvictInferenceContext(
        model, press, n_sink=E2E_SINK, n_local=E2E_LOCAL, batch_size=1
    ) as ec:
        ec.prefill_and_commit(ctx, 0, dict(
            topk=E2E_TOPK, force_sink=E2E_SINK, force_local=E2E_LOCAL,
            head_budget="static", head_budget_table=str(path),
        ))
        assert torch.equal(ec.pool.budgets.cpu(), table.to(ec.pool.budgets.dtype).cpu())
        for h, want in enumerate(row):
            assert int(ec.pool.filled[h]) == want


@CUDA_ONLY
@pytest.mark.parametrize("ctx_len", [512, 2048])
def test_topk_ratio_sizes_the_pool_per_document(ctx_len):
    """`--topk_ratio` resolves the budget from the document\'s own length, during the prefill."""
    from kvpress.presses.gqa_indexer.evict_runner import EvictInferenceContext

    model, press, _, cfg = _e2e_model_and_press()
    ctx = torch.randint(0, cfg.vocab_size, (1, ctx_len), device="cuda")
    with EvictInferenceContext(
        model, press, n_sink=E2E_SINK, n_local=E2E_LOCAL, batch_size=1
    ) as ec:
        ec.prefill_and_commit(ctx, 0, dict(
            topk=E2E_TOPK, topk_ratio=0.25, force_sink=E2E_SINK, force_local=E2E_LOCAL,
        ))
        want = max(int(0.25 * ctx_len), E2E_SINK + E2E_LOCAL + 1)
        assert int(ec.pool.budgets[0, 0]) == want


@CUDA_ONLY
def test_a_second_document_with_a_different_budget_is_refused():
    """One pool serves one budget: the paged block table is fixed at allocation.

    Under `--topk_ratio` a batch of differing context lengths resolves differing budgets, which
    cannot share a pool. Refused loudly, because the shapes would still line up and the run would
    quietly evict against the wrong capacity.
    """
    from kvpress.presses.gqa_indexer.evict_runner import EvictInferenceContext

    model, press, _, cfg = _e2e_model_and_press()
    kw = dict(topk=E2E_TOPK, topk_ratio=0.25, force_sink=E2E_SINK, force_local=E2E_LOCAL)
    with EvictInferenceContext(
        model, press, n_sink=E2E_SINK, n_local=E2E_LOCAL, batch_size=2
    ) as ec:
        ec.prefill_and_commit(torch.randint(0, 256, (1, 2048), device="cuda"), 0, kw)
        with pytest.raises(ValueError, match="different per-head budget"):
            ec.prefill_and_commit(torch.randint(0, 256, (1, 512), device="cuda"), 1, kw)


@CUDA_ONLY
def test_attend_neutralizes_an_empty_cache_branch_at_realistic_scale():
    """An empty cache row must contribute NOTHING to the merge, not NaN.

    THE REGRESSION THIS GUARDS. `flash_attn_with_kvcache` returns ``lse = +inf`` -- not ``-inf`` --
    for a row whose ``cache_seqlens`` is 0. The log-sum-exp merge then takes
    ``m = max(+inf, l_new) = +inf`` and evaluates ``exp(l_cache - m) = exp(inf - inf) = NaN``, so
    the empty row's weight is NaN instead of 0. The NaN travels through the hidden state and only
    surfaces when ``multinomial`` rejects the probability vector, as a device-side assert blamed on
    ``evict_runner.py:640`` -- which is why this cost several 8-shard runs to localize.

    Deliberately at the scale the failure occurred at, not on a toy tensor:
      * a **mixed** batch (row 0 empty, row 1 filled), the normal state under ``--decode_batch 4``,
        which the previous ``filled.min() == 0`` guard got wrong in BOTH directions -- it fired for
        the whole batch and discarded the filled row's cache branch too;
      * ``q_len > 1`` (the question forward), which the previous guard skipped entirely because it
        tested ``q_len == 1``. That is the path that actually crashed.
    """
    torch.manual_seed(0)
    n_kv, group, head_dim, q_len = 2, 2, 16, 3
    budget = POOL_SINK + POOL_LOCAL + 8
    pool = make_pool([budget] * n_kv, batch_size=2, head_dim=head_dim)
    n_q = n_kv * group

    # Row 1 gets a real cache; row 0 stays empty -- `filled` is what `attend` keys off.
    mag, log_beta = router_state(n_kv, budget, "cuda", decay=False)
    k = torch.randn(n_kv, budget, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(n_kv, budget, head_dim, device="cuda", dtype=torch.bfloat16)
    pool.commit(0, 1, key=k, value=v, mag=mag, log_beta=log_beta, k_len=budget)
    assert int(pool.filled[pool.rows_for(0, seq=0)].max()) == 0, "row 0 must be empty"
    assert int(pool.filled[pool.rows_for(0, seq=1)].min()) > 0, "row 1 must be filled"

    query = torch.randn(2, n_q, q_len, head_dim, device="cuda", dtype=torch.bfloat16)
    new_key = torch.randn(2, n_kv, q_len, head_dim, device="cuda", dtype=torch.bfloat16)
    new_value = torch.randn(2, n_kv, q_len, head_dim, device="cuda", dtype=torch.bfloat16)

    # seqs=[0, 1] -> the MIXED batch: sequence 0 empty, sequence 1 filled.
    out = pool.attend(
        0, query, seqs=torch.tensor([0, 1], device="cuda"),
        scaling=head_dim ** -0.5, new_key=new_key, new_value=new_value,
    )
    assert torch.isfinite(out).all(), "an empty cache row produced non-finite attention output"

    # The filled row must NOT have been reduced to its new-token branch: that was the other half of
    # the old guard's bug, and a finiteness check alone cannot see it.
    alone = pool.attend(
        0, query[1:2], seqs=torch.tensor([1], device="cuda"),
        scaling=head_dim ** -0.5, new_key=new_key[1:2], new_value=new_value[1:2],
    )
    assert torch.isfinite(alone).all()
    assert torch.allclose(out[1], alone[0], atol=2e-2), (
        "the filled row's own result changed when an empty row shared the batch -- the mask leaked"
    )


def test_merge_lse_is_nan_safe_for_both_degenerate_cases():
    """`_merge_lse` must never emit NaN, for either way the merge produced one.

    Pure CPU arithmetic, no CUDA: these are the two exact failure modes that cost multi-hour
    8-shard runs, and both are one line of algebra to trigger.

    Case 1 -- **a `+inf` lse**. `flash_attn_with_kvcache` returns `+inf` (not `-inf`) for a row with
    `cache_seqlens == 0`. `m` becomes `+inf` and `exp(lse - m) = exp(inf - inf)` is NaN for EVERY
    branch, so one empty row poisons the entire merge including the healthy branches.

    Case 2 -- **every branch empty, `m = -inf`**. `exp(-inf - -inf)` is NaN again and the denominator
    is NaN rather than 0. This one only became reachable after case 1 was fixed by mapping `+inf` to
    `-inf`, which is why fixing case 1 alone did not stop the crash.

    Also asserts the merge is CORRECT, not merely finite: a two-branch merge where one branch is
    empty must equal the surviving branch exactly, and equal-lse branches must average.
    """
    from kvpress.presses.gqa_indexer.evict_cache import _merge_lse

    B, H, S, D = 1, 2, 3, 4
    o1 = torch.randn(B, H, S, D)
    o2 = torch.randn(B, H, S, D)

    # Case 1: branch 1 is the flash-attn empty row (+inf lse, garbage output).
    l_inf = torch.full((B, H, S), float("inf"))
    l_ok = torch.zeros(B, H, S)
    out = _merge_lse([(o1, l_inf), (o2, l_ok)])
    assert torch.isfinite(out).all(), "a +inf lse produced NaN in the merge"
    assert torch.allclose(out, o2, atol=1e-6), (
        "an empty branch must contribute nothing, so the merge must equal the surviving branch"
    )

    # Case 2: every branch empty -> m = -inf.
    l_neg = torch.full((B, H, S), -float("inf"))
    out = _merge_lse([(o1, l_neg), (o2, l_neg)])
    assert torch.isfinite(out).all(), "an all-empty merge produced NaN"
    assert torch.allclose(out, torch.zeros_like(out)), "nothing to attend to must give exactly 0"

    # Correctness sanity: equal lses average the branches.
    out = _merge_lse([(o1, l_ok), (o2, l_ok)])
    assert torch.allclose(out, (o1 + o2) / 2, atol=1e-6)

    # NaN in an lse is treated as empty rather than propagated.
    l_nan = torch.full((B, H, S), float("nan"))
    out = _merge_lse([(o1, l_nan), (o2, l_ok)])
    assert torch.isfinite(out).all() and torch.allclose(out, o2, atol=1e-6)
