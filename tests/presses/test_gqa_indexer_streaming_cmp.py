# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for :mod:`~kvpress.presses.gqa_indexer.streaming_cmp`.

The load-bearing claim is that the running mean is **exact**, not an approximation: absorbing a
cluster's members one at a time must land on the same centroid as averaging them in one batch.
That is what lets hard eviction keep CMP slots at all, since it has deleted the keys a batch
re-clustering would need.
"""

from __future__ import annotations

import pytest
import torch

from kvpress.presses.gqa_indexer.cmp_slots import cluster_reduce
from kvpress.presses.gqa_indexer.streaming_cmp import StreamingCMP


def make(n_rows=4, n_slots=8, head_dim=16, reseed_below=0, device="cpu"):
    return StreamingCMP(
        n_rows, n_slots, head_dim, device=torch.device(device), reseed_below=reseed_below
    )


def test_running_mean_equals_the_batch_mean():
    """One key at a time == `cluster_reduce` over the same membership.

    This is the identity the whole module rests on. Checked against the *production* batch
    reducer rather than a hand-rolled mean, so a change to its weighting convention would fail
    here rather than drift silently.
    """
    torch.manual_seed(0)
    dim, n = 16, 300
    cmp = make(n_rows=1, n_slots=1, head_dim=dim)
    keys = torch.randn(n, dim)
    values = torch.randn(n, dim)
    rows = torch.zeros(1, dtype=torch.int64)
    for i in range(n):
        cmp.ingest(rows, keys[i : i + 1], values[i : i + 1])

    # Everything landed in the single slot, so the batch mean is over all of it.
    want_k, pop = cluster_reduce(
        keys.unsqueeze(0), torch.zeros(1, n, dtype=torch.int64), 1, torch.ones(1, n)
    )
    want_v, _ = cluster_reduce(
        values.unsqueeze(0), torch.zeros(1, n, dtype=torch.int64), 1, torch.ones(1, n)
    )
    assert float(pop[0, 0]) == n
    assert float(cmp.pop[0, 0]) == n
    assert torch.allclose(cmp.k_cmp[0, 0], want_k[0, 0], atol=1e-5), (
        f"running mean drifted by {(cmp.k_cmp[0, 0] - want_k[0, 0]).abs().max():.2e}"
    )
    assert torch.allclose(cmp.v_cmp[0, 0], want_v[0, 0], atol=1e-5)


def test_cold_start_seeds_slots_in_order_then_clusters():
    """With no prefill batch to seed from, the first R arrivals each take a slot.

    That is sequential k-means' cold start, and it is the CoT case: generation begins with an
    empty evicted set, so there is nothing for k-means to initialize on.
    """
    torch.manual_seed(1)
    dim, R = 8, 5
    cmp = make(n_rows=1, n_slots=R, head_dim=dim)
    rows = torch.zeros(1, dtype=torch.int64)
    # R well-separated arrivals: each must claim its own slot.
    for i in range(R):
        k = torch.zeros(1, dim)
        k[0, i] = 100.0
        cmp.ingest(rows, k, torch.randn(1, dim))
        assert int((cmp.pop[0] > 0).sum()) == i + 1, f"arrival {i} did not claim a fresh slot"
    # The next arrival, close to slot 2's centroid, must JOIN it rather than displace anything.
    near = torch.zeros(1, dim)
    near[0, 2] = 99.0
    cmp.ingest(rows, near, torch.randn(1, dim))
    assert int((cmp.pop[0] > 0).sum()) == R, "a sixth slot appeared out of R=5"
    assert float(cmp.pop[0, 2]) == 2.0, f"joined the wrong slot: pop={cmp.pop[0].tolist()}"


def test_nearest_centroid_wins_not_a_dead_slot():
    """A zero-vector (unused) centroid must not attract arrivals once R is exhausted."""
    dim, R = 8, 3
    cmp = make(n_rows=1, n_slots=R, head_dim=dim, reseed_below=0)
    rows = torch.zeros(1, dtype=torch.int64)
    for i in range(R):
        k = torch.zeros(1, dim)
        k[0, i] = 10.0
        cmp.ingest(rows, k, torch.zeros(1, dim))
    assert int((cmp.pop[0] > 0).sum()) == R
    # An arrival near slot 1. With R exhausted it must join slot 1, not sit at the origin.
    k = torch.zeros(1, dim)
    k[0, 1] = 11.0
    cmp.ingest(rows, k, torch.zeros(1, dim))
    assert float(cmp.pop[0, 1]) == 2.0
    assert torch.allclose(cmp.pop[0], torch.tensor([1.0, 2.0, 1.0]))


def test_inactive_rows_are_untouched():
    """A row that evicted nothing this step must not absorb anything.

    The common case, not an edge case: a row still growing into its budget evicts nothing, and a
    row whose demoted key lost the pool contest drops that key instead.
    """
    torch.manual_seed(2)
    dim = 8
    cmp = make(n_rows=3, n_slots=4, head_dim=dim)
    rows = torch.arange(3)
    keys = torch.randn(3, dim)
    cmp.ingest(rows, keys, torch.randn(3, dim),
               active=torch.tensor([True, False, True]))
    assert float(cmp.pop[0].sum()) == 1.0
    assert float(cmp.pop[1].sum()) == 0.0, "an inactive row absorbed a key"
    assert float(cmp.pop[2].sum()) == 1.0
    assert torch.all(cmp.k_cmp[1] == 0)


def test_reseed_replaces_a_dead_slot():
    """With `reseed_below`, a slot nobody joined is taken over rather than left anchoring nothing.

    The cold-start seeds are the LOWEST-SCORING keys at the moment the budget first binds, so some
    of them are a poor basis; re-seeding is what stops a bad seed from owning a slot for the whole
    generation.
    """
    dim, R = 8, 3
    cmp = make(n_rows=1, n_slots=R, head_dim=dim, reseed_below=1)
    rows = torch.zeros(1, dtype=torch.int64)
    for i in range(R):
        k = torch.zeros(1, dim)
        k[0, i] = 10.0
        cmp.ingest(rows, k, torch.zeros(1, dim))
    # Every slot has pop == 1, i.e. all are "dead" at reseed_below=1, so the next arrival takes
    # slot 0 over instead of joining it.
    fresh = torch.full((1, dim), -5.0)
    cmp.ingest(rows, fresh, torch.zeros(1, dim))
    assert float(cmp.pop[0, 0]) == 1.0
    assert torch.allclose(cmp.k_cmp[0, 0], fresh[0]), "the slot was blended, not replaced"


def test_read_matches_an_explicit_softmax():
    """`read` must return the slots' own normalized output and its exact lse."""
    torch.manual_seed(3)
    dim, R, group = 16, 4, 2
    n_kv, q_len = 2, 3
    cmp = make(n_rows=n_kv, n_slots=R, head_dim=dim)
    rows = torch.arange(n_kv)
    for _ in range(12):
        cmp.ingest(rows, torch.randn(n_kv, dim), torch.randn(n_kv, dim))

    query = torch.randn(1, n_kv * group, q_len, dim)
    scaling = dim ** -0.5
    o, lse = cmp.read(rows, query, group=group, scaling=scaling)

    for h in range(n_kv * group):
        kv = h // group
        kc, vc, pop = cmp.k_cmp[kv], cmp.v_cmp[kv], cmp.pop[kv]
        for t in range(q_len):
            z = query[0, h, t] @ kc.T * scaling + torch.log(pop.clamp(min=1e-30))
            z = torch.where(pop > 0, z, torch.full_like(z, -float("inf")))
            assert torch.allclose(lse[0, h, t], z.logsumexp(-1), atol=1e-4)
            assert torch.allclose(o[0, h, t], z.softmax(-1) @ vc, atol=1e-3)


def test_read_does_not_overflow_on_large_logits():
    """The bug that killed 5 of 8 shards: unnormalized exp(q.k + log n_r) saturates fp32.

    A slot holding thousands of CoT keys carries log n_r ~ 8, and the deep layers' keys have
    norm ~30, so the joint logit reaches ~1600 -- `exp` gives inf, the fused output becomes
    inf/inf = nan, and the nan surfaces as a device-side assert inside `multinomial`. The
    subtract-the-max form must stay finite at any scale.
    """
    dim, R = 128, 64
    cmp = make(n_rows=1, n_slots=R, head_dim=dim)
    torch.manual_seed(7)
    cmp.k_cmp[0] = torch.randn(R, dim) * 30.0      # deep-layer key norms
    cmp.v_cmp[0] = torch.randn(R, dim)
    cmp.pop[0] = torch.full((R,), 3000.0)          # a 32k-token CoT over 64 slots
    query = torch.randn(1, 1, 1, dim) * 30.0

    o, lse = cmp.read(torch.zeros(1, dtype=torch.int64), query, group=1, scaling=dim ** -0.5)
    assert torch.isfinite(o).all(), "output overflowed"
    assert torch.isfinite(lse).all(), "lse overflowed"
    # The normalized output must be a convex combination of the slot values, so it cannot
    # exceed their range -- a cheap check that the normalization really happened.
    assert float(o.abs().max()) <= float(cmp.v_cmp[0].abs().max()) + 1e-3


def test_read_on_all_empty_slots_is_inert():
    """Before anything has been evicted the slots must contribute EXACTLY nothing.

    Every logit is -inf there, so a naive max-subtraction gives -inf - -inf = nan. The merge has
    to receive lse = -inf instead, which drops the branch out exactly.
    """
    dim, R = 32, 8
    cmp = make(n_rows=2, n_slots=R, head_dim=dim)
    query = torch.randn(1, 2, 2, dim)
    o, lse = cmp.read(torch.arange(2), query, group=1, scaling=dim ** -0.5)
    assert torch.isfinite(o).all(), f"empty slots produced {o}"
    assert bool((lse == -float("inf")).all()), f"expected lse=-inf, got {lse}"


def test_unused_slots_contribute_exactly_zero():
    """An empty slot must be -inf in the log domain, not a small finite logit.

    Anything finite would let R - live slots vote with a zero-vector key, which is a real logit of
    0 -- the slot would claim mass it has no members for.
    """
    torch.manual_seed(4)
    dim, R = 8, 6
    cmp = make(n_rows=1, n_slots=R, head_dim=dim)
    rows = torch.zeros(1, dtype=torch.int64)
    cmp.ingest(rows, torch.randn(1, dim), torch.randn(1, dim))  # exactly ONE live slot
    query = torch.randn(1, 1, 1, dim)
    o, lse = cmp.read(rows, query, group=1, scaling=dim ** -0.5)
    # With one live slot the lse is that slot's logit alone, and the output IS its value --
    # the five empty slots must not dilute either.
    logit = (query[0, 0, 0] @ cmp.k_cmp[0, 0]) * dim ** -0.5 + torch.log(cmp.pop[0, 0])
    assert torch.allclose(lse[0, 0, 0], logit, atol=1e-4)
    assert torch.allclose(o[0, 0, 0], cmp.v_cmp[0, 0], atol=1e-4)
    assert torch.isfinite(o).all() and torch.isfinite(lse).all()


def test_load_batch_round_trips_through_b_cmp():
    """Seeding from a batch `cluster_evicted` result must recover its populations.

    A long context should still be summarized by real k-means at the commit and only then kept up
    to date by streaming -- the two compose rather than compete.
    """
    torch.manual_seed(5)
    dim, R = 8, 4
    cmp = make(n_rows=2, n_slots=R, head_dim=dim)
    k_cmp = torch.randn(2, R, dim)
    v_cmp = torch.randn(2, R, dim)
    pop = torch.tensor([[7.0, 3.0, 0.0, 11.0], [1.0, 0.0, 0.0, 5.0]])
    b_cmp = torch.where(pop > 0, pop.clamp(min=1e-30).log(), torch.full_like(pop, -float("inf")))

    cmp.load_batch(slice(0, 2), k_cmp, v_cmp, b_cmp)
    assert torch.allclose(cmp.pop, pop), f"populations did not round-trip: {cmp.pop.tolist()}"
    assert torch.allclose(cmp.k_cmp, k_cmp)
    # A silenced slot came back free, so the next arrival may claim it.
    rows = torch.zeros(1, dtype=torch.int64)
    cmp.ingest(rows, torch.randn(1, dim), torch.randn(1, dim))
    assert float(cmp.pop[0, 2]) == 1.0


def test_streaming_beats_frozen_on_drifted_arrivals():
    """The reason this module exists: frozen centroids go stale, streaming tracks the drift.

    Deliberately an inequality rather than a threshold -- the absolute error depends on the
    synthetic geometry, but the ORDERING (streaming closer than frozen) is the claim. Measured on
    this shape: frozen 1.047 relative error (worse than a zero vector, since the centroid points
    the wrong way) against 0.550 streaming.
    """
    torch.manual_seed(6)
    dim, R = 32, 8
    cmp_frozen = make(n_rows=1, n_slots=R, head_dim=dim)
    cmp_stream = make(n_rows=1, n_slots=R, head_dim=dim)
    rows = torch.zeros(1, dtype=torch.int64)

    # Both see the same first phase, so they start identical.
    early = torch.randn(60, dim)
    for i in range(60):
        cmp_frozen.ingest(rows, early[i : i + 1], early[i : i + 1])
        cmp_stream.ingest(rows, early[i : i + 1], early[i : i + 1])
    assert torch.allclose(cmp_frozen.k_cmp, cmp_stream.k_cmp)

    # Second phase is OFF-DISTRIBUTION -- the CoT moving to a new topic. Only streaming absorbs it.
    late = torch.randn(120, dim) + 4.0
    for i in range(120):
        cmp_stream.ingest(rows, late[i : i + 1], late[i : i + 1])

    def err(state):
        c = state.k_cmp[0]
        live = state.pop[0] > 0
        d = torch.cdist(late, c[live]).min(-1).values
        return float((d / late.norm(dim=-1)).mean())

    assert err(cmp_stream) < err(cmp_frozen), (
        f"streaming {err(cmp_stream):.3f} did not beat frozen {err(cmp_frozen):.3f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_ingest_makes_no_host_synchronization():
    """The per-token path must not read a tensor on the host.

    One `.any()` costs 0.94 ms on this box, which at 36 layers a token would dominate the step --
    the same constraint `EvictPagedPool.ingest` documents.
    """
    dim = 64
    cmp = make(n_rows=64, n_slots=16, head_dim=dim, device="cuda")
    rows = torch.arange(64, device="cuda")
    key = torch.randn(64, dim, device="cuda")
    active = torch.rand(64, device="cuda") > 0.5
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")  # raises on any implicit device->host copy
    try:
        cmp.ingest(rows, key, key, active=active)
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_read_survives_an_overflowing_slot_logit():
    """A live slot whose logit overflows fp32 must not produce NaN.

    THE THIRD NaN SOURCE, independent of the two in `evict_cache._merge_lse`. `q . k_cmp * scaling`
    is an unbounded dot product; measured centroid norms reached 549 against real key norms ~340, and
    a single fp32 overflow makes the row's max `+inf`. The original guard tested only
    `isfinite(m)` and substituted 0 for BOTH infinities, so `+inf` became
    `exp(+inf - 0) = inf`, `denom = inf`, `w / denom = inf / inf` = **NaN**.

    Also asserts the row is not silently DISCARDED: the old `lse = where(isfinite(m), lse, -inf)`
    threw away an overflowing slot's contribution entirely, which is a wrong answer rather than a
    crash -- the failure mode greedy decoding would have hidden.
    """
    from kvpress.presses.gqa_indexer.streaming_cmp import StreamingCMP

    dev = torch.device("cpu")
    cmp = StreamingCMP(n_rows=1, n_slots=4, head_dim=8, device=dev)
    # One live slot, and a centroid huge enough that q . k_cmp overflows fp32.
    cmp.pop[0, 0] = 100.0
    cmp.k_cmp[0, 0] = 1e20
    cmp.v_cmp[0, 0] = 1.0
    rows = torch.tensor([0], device=dev)
    query = torch.full((1, 1, 1, 8), 1e20, device=dev)

    o, lse = cmp.read(rows, query, group=1, scaling=8 ** -0.5)
    assert torch.isfinite(o).all(), "an overflowing slot logit produced NaN in the CMP output"
    assert torch.isfinite(lse).all(), "an overflowing slot produced a non-finite lse"
    assert not bool((lse == -float("inf")).any()), (
        "an overflowing LIVE slot was discarded as if empty -- silently wrong, not just unstable"
    )
    # It should dominate its row: weight saturates to 1, so the output is that slot's value.
    assert torch.allclose(o, torch.ones_like(o), atol=1e-5)

    # The genuinely-empty row must still drop out exactly.
    cmp2 = StreamingCMP(n_rows=1, n_slots=4, head_dim=8, device=dev)
    o2, lse2 = cmp2.read(rows, torch.randn(1, 1, 1, 8), group=1, scaling=8 ** -0.5)
    assert torch.isfinite(o2).all()
    assert bool((lse2 == -float("inf")).all()), "an all-empty row must still be lse = -inf"
