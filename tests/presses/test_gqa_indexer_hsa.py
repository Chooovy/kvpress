# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Two-level (HSA) chunk attention.

The tests that justify the module existing are the three structural ones, because they are the
claims ``ROUTER_LEARNABILITY.md`` §5-§6 makes and the reason this objective was chosen over the
additive gate:

* ``test_router_weight_is_exactly_the_realized_chunk_mass`` -- the router outputs the quantity
  inference ranks on, not a correction to it.
* ``test_a_flat_router_is_not_a_no_op`` -- there is no zero-cost setting, so no pinning is needed.
* ``test_true_chunk_mass_reproduces_dense_attention`` -- the ceiling is dense, so the objective is
  achievable rather than a compromise.

Everything else checks the implementation against the reference, in fp64 where the tolerance should
measure floating-point noise and nothing else.
"""

import pytest
import torch

from kvpress.presses.gqa_indexer.indexer import MASK_NEG
from kvpress.presses.gqa_indexer.hsa_attention import (
    INVISIBLE_SCORE,
    chunk_lse,
    hsa_chunk_attention,
    hsa_chunk_attention_reference,
)

DT = torch.float64


def make_qkv(b=1, hq=4, hkv=2, sq=16, sk=32, d=8, dv=8, cs=8, seed=0, dtype=DT, requires_grad=False):
    torch.manual_seed(seed)
    n_chunk = -(-sk // cs)
    t = lambda *shape: torch.randn(*shape, dtype=dtype, requires_grad=requires_grad)  # noqa: E731
    return (
        t(b, hq, sq, d),
        t(b, hkv, sk, d),
        t(b, hkv, sk, dv),
        t(b, hkv, sq, n_chunk),
    )


def dense_reference(q, k, v, *, query_offset=None):
    """Plain causal attention, for the ceiling tests."""
    b, hq, sq, d = q.shape
    hkv, sk = k.shape[1], k.shape[2]
    group = hq // hkv
    if query_offset is None:
        query_offset = sk - sq
    logits = torch.einsum(
        "bhgqd,bhsd->bhgqs", q.view(b, hkv, group, sq, d), k
    ) * d**-0.5
    q_pos = torch.arange(sq) + query_offset
    causal = torch.arange(sk).view(1, sk) <= q_pos.view(sq, 1)
    logits = logits.masked_fill(~causal.view(1, 1, 1, sq, sk), -float("inf"))
    p = torch.softmax(logits, dim=-1)
    return torch.einsum("bhgqs,bhsd->bhgqd", p, v).reshape(b, hq, sq, v.shape[-1])


# ------------------------------------------------------- the structural claims


def test_router_weight_is_exactly_the_realized_chunk_mass():
    """
    ``softmax(s)_c`` **is** chunk ``c``'s share of the output -- the property the whole objective
    rests on.

    Why it matters: inference takes ``argtop-k`` on ``s``. If ``s`` ranked something other than the
    mass, the router would be trained on one quantity and deployed on another. The additive arm has
    exactly that gap (its optimum is ``log(mass) - LSE_c``, a constant, so it carries no ranking);
    this test is the statement that this arm does not.

    Measured by decomposing the output onto per-chunk outputs, which is only possible because the
    within-chunk softmax normalizes: ``out = sum_c w_c o_c`` with each ``o_c`` a convex combination
    of ``v``, so the mass is ``w_c`` by construction. Verified numerically against the equivalent
    dense attention's actual row sums.
    """
    q, k, v, s = make_qkv(hq=2, hkv=1, sq=8, sk=24, cs=8)
    b, hq, sq, d = q.shape
    sk, n_chunk = k.shape[2], s.shape[-1]

    # The additive form: out == softmax(qk^T + g) @ v with g_c = s_c - LSE_c. Its row sums over a
    # chunk ARE the realized mass, so this is a genuine measurement rather than a restatement.
    logits = torch.einsum("bhqd,bhsd->bhqs", q, k.expand(b, hq, sk, d)) * d**-0.5
    q_pos = torch.arange(sq) + (sk - sq)
    causal = torch.arange(sk).view(1, sk) <= q_pos.view(sq, 1)
    lse = chunk_lse(logits, 8, valid=causal.view(1, 1, sq, sk).expand(b, hq, sq, sk))
    vis = (torch.arange(n_chunk) * 8).view(1, n_chunk) <= q_pos.view(sq, 1)
    w = torch.softmax(s.masked_fill(~vis.view(1, 1, sq, n_chunk), INVISIBLE_SCORE), -1)

    g = (w.log() - lse).repeat_interleave(8, dim=-1)[..., :sk]
    A = torch.softmax((logits + g).masked_fill(~causal.view(1, 1, sq, sk), -float("inf")), -1)
    realized = A.reshape(b, hq, sq, n_chunk, 8).sum(-1)

    gap = (realized - w.expand_as(realized)).abs().max()
    assert gap < 1e-12, f"router weight is not the realized mass: max gap {gap:.2e}"
    # And the distribution is still valid -- the two-level form does not leak mass.
    assert (A.sum(-1) - 1).abs().max() < 1e-12


def test_a_flat_router_is_not_a_no_op():
    """
    A constant score gives uniform mixing, which is **far** from dense attention.

    This is why no :mod:`~.gate_pin` machinery is needed. The additive gate can be switched off by
    going flat along the key axis (the addend cancels in the softmax), recovering the strong frozen
    backbone at zero cost -- so the LM loss is satisfied with no ranking learned, and SAS reports
    18.8 vs 54.4 on that difference. Here the within-chunk softmax has already normalized, so a flat
    ``w`` averages chunks equally instead of disappearing.
    """
    q, k, v, _ = make_qkv(sq=8, sk=32, cs=8, seed=3)
    n_chunk = 4
    flat = torch.zeros(q.shape[0], k.shape[1], q.shape[2], n_chunk, dtype=DT)
    uniform = hsa_chunk_attention_reference(q, k, v, flat, chunk_size=8)
    dense = dense_reference(q, k, v)
    gap = (uniform - dense).abs().max()
    assert gap > 1e-2, (
        f"a flat router is only {gap:.2e} from dense attention -- if this is small the objective has "
        f"the same no-op loophole the additive gate has, and would need pinning"
    )
    # Scaling the score to zero is the other route to a no-op the additive arm has (gate_scale -> 0).
    # It is the SAME point here, so it is equally not a no-op -- checked so the claim covers both.
    scaled = hsa_chunk_attention_reference(q, k, v, flat * 0.0, chunk_size=8)
    assert torch.equal(scaled, uniform)


def test_true_chunk_mass_reproduces_dense_attention():
    """
    ``s = LSE_c`` recovers dense attention exactly, so the objective's ceiling is dense.

    Together with the previous test this is the elegant part: reaching the optimum *requires*
    learning the ranking, because the optimum's score IS the ranking. There is no lazy solution to
    fall into and no gap between "scores well" and "ranks correctly".
    """
    q, k, v, _ = make_qkv(hq=4, hkv=2, sq=12, sk=32, cs=8, seed=4)
    b, hq, sq, d = q.shape
    hkv, sk = k.shape[1], k.shape[2]
    group = hq // hkv
    logits = torch.einsum("bhgqd,bhsd->bhgqs", q.view(b, hkv, group, sq, d), k) * d**-0.5
    q_pos = torch.arange(sq) + (sk - sq)
    causal = torch.arange(sk).view(1, sk) <= q_pos.view(sq, 1)

    # LSE is per (Hkv, query) once the group is reduced -- but the scores are shared across a group,
    # so a group whose heads disagree cannot be exact. Use hq == hkv * 1 by averaging is WRONG;
    # instead check per-group-of-one by taking group 0's LSE and comparing that head only.
    lse = chunk_lse(logits[:, :, 0], 8, valid=causal.view(1, 1, sq, sk).expand(b, hkv, sq, sk))
    out = hsa_chunk_attention_reference(q, k, v, lse, chunk_size=8)
    dense = dense_reference(q, k, v)
    # Head 0 of each KV group is the one whose LSE was used.
    got = out.view(b, hkv, group, sq, -1)[:, :, 0]
    want = dense.view(b, hkv, group, sq, -1)[:, :, 0]
    gap = (got - want).abs().max()
    assert gap < 1e-12, f"w = true chunk mass did not reproduce dense attention: {gap:.2e}"


def test_matches_the_additive_gate_form_including_gradients():
    """
    ``two-level(s) == softmax(qk^T + s - LSE) @ v``, gradients included.

    An independent cross-check of the whole forward, written from a different formula. It is also
    the identity that makes the no-op argument rigorous: a flat ``s`` still leaves ``-LSE_c``, which
    is not constant along the key axis, so it cannot cancel.

    ``LSE`` must stay **attached**. This test asserts the exactness that detaching breaks -- which is
    why the implementation computes per-chunk softmaxes directly instead of taking this cheaper
    route.
    """
    cs = 8
    q, k, v, s = make_qkv(hq=2, hkv=2, sq=8, sk=24, cs=cs, seed=5, requires_grad=True)
    b, hq, sq, d = q.shape
    sk, n_chunk = k.shape[2], s.shape[-1]
    target = torch.randn(b, hq, sq, v.shape[-1], dtype=DT)

    def two_level():
        return ((hsa_chunk_attention_reference(q, k, v, s, chunk_size=cs) - target) ** 2).mean()

    def additive():
        logits = torch.einsum("bhqd,bhsd->bhqs", q, k) * d**-0.5
        q_pos = torch.arange(sq) + (sk - sq)
        causal = torch.arange(sk).view(1, sk) <= q_pos.view(sq, 1)
        cmask = causal.view(1, 1, sq, sk).expand(b, hq, sq, sk)
        lse = chunk_lse(logits, cs, valid=cmask)  # ATTACHED
        vis = (torch.arange(n_chunk) * cs).view(1, n_chunk) <= q_pos.view(sq, 1)
        ss = s.masked_fill(~vis.view(1, 1, sq, n_chunk), INVISIBLE_SCORE)
        g = (ss - lse).repeat_interleave(cs, dim=-1)[..., :sk]
        A = torch.softmax((logits + g).masked_fill(~cmask, -float("inf")), -1)
        return ((torch.einsum("bhqs,bhsd->bhqd", A, v) - target) ** 2).mean()

    la, lb = two_level(), additive()
    assert (la - lb).abs() < 1e-12, f"losses differ: {float(la)} vs {float(lb)}"

    grads = []
    for fn in (two_level, additive):
        for t in (q, k, v, s):
            t.grad = None
        fn().backward()
        grads.append([t.grad.clone() for t in (q, k, v, s)])
    for name, a, bb in zip("qkvs", grads[0], grads[1]):
        rel = (a - bb).norm() / a.norm().clamp_min(1e-30)
        assert rel < 1e-10, f"d{name} differs by {rel:.2e} relative"


def test_every_chunk_receives_gradient():
    """
    No candidate pool: **every** chunk gets a nonzero gradient every step.

    This is the exact-K arm's measured bottleneck removed by construction -- there, 11-15% of
    oracle-best chunks never entered the M=32 pool, and a chunk outside the pool appears nowhere in
    the graph so no backward estimator can reach it. The softmax here is over all chunks.
    """
    cs = 8
    q, k, v, s = make_qkv(hq=2, hkv=1, sq=8, sk=64, cs=cs, seed=6)
    s = s.detach().requires_grad_(True)
    out = hsa_chunk_attention_reference(q, k, v, s, chunk_size=cs)
    (out**2).sum().backward()
    n_chunk = s.shape[-1]
    q_pos = torch.arange(q.shape[2]) + (k.shape[2] - q.shape[2])
    vis = (torch.arange(n_chunk) * cs).view(1, n_chunk) <= q_pos.view(q.shape[2], 1)
    got = (s.grad.abs() > 0)[0, 0]
    # Every VISIBLE (query, chunk) pair -- an invisible one correctly gets none.
    assert bool((got == vis).all()), (
        f"{int((vis & ~got).sum())} visible (query, chunk) pairs received no gradient"
    )


# ------------------------------------------------------- implementation vs reference


@pytest.mark.parametrize("cs", [1, 4, 8, 16])
def test_tiled_matches_reference(cs):
    q, k, v, s = make_qkv(hq=4, hkv=2, sq=16, sk=32, cs=cs, seed=7)
    want = hsa_chunk_attention_reference(q, k, v, s, chunk_size=cs)
    got, _ = hsa_chunk_attention(q, k, v, s, chunk_size=cs, checkpoint=False)
    gap = (got - want).abs().max()
    assert gap < 1e-12, f"chunk_size={cs}: tiled implementation differs by {gap:.2e}"


@pytest.mark.parametrize("tile", [1, 3, 8, 64])
def test_query_tile_is_invariant(tile):
    """The tile is a memory knob; the result must not depend on it."""
    q, k, v, s = make_qkv(sq=16, sk=32, cs=8, seed=8)
    ref, _ = hsa_chunk_attention(q, k, v, s, chunk_size=8, checkpoint=False, query_tile=1024)
    got, _ = hsa_chunk_attention(q, k, v, s, chunk_size=8, checkpoint=False, query_tile=tile)
    assert torch.equal(ref, got), f"query_tile={tile} changed the output"


def test_ragged_tail_matches_reference():
    """``Sk`` not divisible by ``chunk_size``: the tail is a short chunk, not a padded full one."""
    q, k, v, s = make_qkv(sq=8, sk=30, cs=8, seed=9)  # 30 = 3*8 + 6
    assert s.shape[-1] == 4
    want = hsa_chunk_attention_reference(q, k, v, s, chunk_size=8)
    got, _ = hsa_chunk_attention(q, k, v, s, chunk_size=8, checkpoint=False)
    assert (got - want).abs().max() < 1e-12


def test_gradients_match_reference():
    cs = 8
    q, k, v, s = make_qkv(hq=4, hkv=2, sq=12, sk=32, cs=cs, seed=10, requires_grad=True)
    target = torch.randn(q.shape[0], q.shape[1], q.shape[2], v.shape[-1], dtype=DT)
    grads = []
    for fn in (
        lambda: hsa_chunk_attention_reference(q, k, v, s, chunk_size=cs),
        lambda: hsa_chunk_attention(q, k, v, s, chunk_size=cs, checkpoint=False)[0],
    ):
        for t in (q, k, v, s):
            t.grad = None
        ((fn() - target) ** 2).mean().backward()
        grads.append([t.grad.clone() for t in (q, k, v, s)])
    for name, a, b in zip("qkvs", grads[0], grads[1]):
        rel = (a - b).norm() / a.norm().clamp_min(1e-30)
        assert rel < 1e-10, f"d{name} differs by {rel:.2e}"


def test_checkpointing_changes_memory_not_values():
    """
    ``checkpoint=True`` must be numerically transparent, and must actually retain less.

    The memory half is measured by counting *live retained elements* via ``saved_tensors_hooks``
    rather than by inspecting graph depth: both paths have a shallow ``grad_fn`` chain, so depth
    cannot discriminate them (2 nodes vs 3), and a test that passed for that reason would not be
    testing anything.
    """
    import weakref

    cs = 8
    q, k, v, s = make_qkv(hq=4, hkv=2, sq=16, sk=32, cs=cs, seed=11, requires_grad=True)

    def retained(flag):
        live: list = []

        def pack(t):
            live.append(weakref.ref(t))
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            out, _ = hsa_chunk_attention(q, k, v, s, chunk_size=cs, checkpoint=flag, query_tile=4)
        n = sum(r().numel() for r in live if r() is not None)
        return out, n

    off, n_off = retained(False)
    on, n_on = retained(True)
    assert (off - on).abs().max() < 1e-12, "checkpointing changed the output"
    assert n_on < n_off, f"checkpointing retained no less: {n_on} vs {n_off} elements"


# ------------------------------------------------------- causality and edge cases


def test_no_information_from_beyond_the_diagonal():
    """
    Perturbing a key past a query's diagonal must not change that query's output.

    Both levels have to be causal, and only one of them is enforced by the attention mask: if the
    chunk *weights* were computed over invisible chunks the softmax denominator would include them,
    which is a leak that leaves the output still looking like valid attention.
    """
    cs = 8
    q, k, v, s = make_qkv(sq=32, sk=32, cs=cs, seed=12)
    a = hsa_chunk_attention_reference(q, k, v, s, chunk_size=cs)
    k2, v2 = k.clone(), v.clone()
    k2[0, :, 24:] += 50.0
    v2[0, :, 24:] += 50.0
    s2 = s.clone()
    s2[..., 3] += 50.0  # the chunk covering keys 24-31
    b = hsa_chunk_attention_reference(q, k2, v2, s2, chunk_size=cs)
    early = slice(0, 24)
    assert (a[:, :, early] - b[:, :, early]).abs().max() < 1e-12, (
        "an early query's output changed when a key beyond its horizon changed"
    )
    assert (a[:, :, 24:] - b[:, :, 24:]).abs().max() > 1e-3, (
        "late queries should have noticed, or the test proves nothing"
    )


def test_first_query_sees_only_its_own_chunk():
    """A query in chunk 0 has exactly one visible chunk, so its output is that chunk's alone."""
    cs = 8
    q, k, v, s = make_qkv(sq=32, sk=32, cs=cs, seed=13)
    out = hsa_chunk_attention_reference(q, k, v, s, chunk_size=cs)
    # Query 0 can see key 0 only, so its output must be v[0] regardless of the score.
    for shift in (0.0, 100.0, -100.0):
        got = hsa_chunk_attention_reference(q, k, v, s + shift, chunk_size=cs)
        assert (got[:, :, 0] - v[:, :, 0].repeat_interleave(2, dim=1)).abs().max() < 1e-12
    assert torch.isfinite(out).all()


def test_no_nan_when_scores_saturate():
    """
    An extreme score must not produce NaN.

    ``softmax`` over a row containing ``INVISIBLE_SCORE`` is why that constant is finite: with
    ``-inf`` a query in chunk 0 -- whose only visible chunk is its own -- would have an all-``-inf``
    row and softmax would give NaN, taking the whole model with it.
    """
    cs = 8
    q, k, v, s = make_qkv(sq=16, sk=32, cs=cs, seed=14)
    for scale in (1e3, -1e3, 1e6):
        out, stats = hsa_chunk_attention(q, k, v, s * scale, chunk_size=cs, checkpoint=False)
        assert torch.isfinite(out).all(), f"score scale {scale} produced non-finite output"
        assert stats["chunk_entropy"] == stats["chunk_entropy"]


def test_chunk_lse_matches_a_masked_logsumexp():
    """``chunk_lse`` against the obvious per-chunk loop, including the ragged tail."""
    torch.manual_seed(15)
    sq, sk, cs = 8, 30, 8
    logits = torch.randn(1, 2, sq, sk, dtype=DT)
    q_pos = torch.arange(sq) + (sk - sq)
    valid = (torch.arange(sk).view(1, sk) <= q_pos.view(sq, 1)).view(1, 1, sq, sk).expand(1, 2, sq, sk)
    got = chunk_lse(logits, cs, valid=valid)
    n_chunk = -(-sk // cs)
    assert got.shape == (1, 2, sq, n_chunk)
    for c in range(n_chunk):
        lo, hi = c * cs, min((c + 1) * cs, sk)
        # NOT `.float()`: that would downcast an fp64 reference and the tolerance would then measure
        # the downcast (2.1e-7) rather than chunk_lse. The same mistake in the implementation is what
        # `chunk_lse`'s accumulation_dtype comment records.
        x = logits[..., lo:hi].masked_fill(~valid[..., lo:hi], -float("inf"))
        want = torch.logsumexp(x, -1)
        want = torch.where(torch.isfinite(want), want, want.new_full((), INVISIBLE_SCORE))
        assert (got[..., c] - want).abs().max() < 1e-12, f"chunk {c}"


def test_chunk_lse_gradient_is_the_within_chunk_softmax_and_skips_invisible_chunks():
    """
    The gradient through ``chunk_lse``, which is what the LSE retrain actually depends on.

    Two properties, and the second is the one that could have been a real bug:

    1. For a **partially** visible chunk, ``d LSE / d x_j`` is the softmax over that chunk's
       *visible* keys -- exact to 8.3e-17. Masked keys sit at ``MASK_NEG``, whose ``exp`` is 0, so
       they are discounted with no valid-count bookkeeping.
    2. For a **fully** invisible chunk the gradient is exactly **0**. This matters: a naive softmax
       over a row of all-``MASK_NEG`` slots yields a spurious *uniform* ``1/chunk_size``, which would
       train the router on chunks the query cannot see. (I initially mis-read this as a 0.125
       discrepancy in the implementation; the implementation was right and the naive reference was
       wrong.)

    Also pins that an all-masked chunk's score, ``MASK_NEG * scale + log(chunk_size)``, stays far
    below any real score -- the ``log(n)`` term is a ~4.85 offset on a ~-884 value, so such a chunk
    is never selected and gets exactly 0 softmax weight.
    """
    cs, n_chunk = 8, 4
    sk = cs * n_chunk
    torch.manual_seed(0)
    x = torch.randn(1, 1, 3, sk, dtype=DT)
    forbidden = torch.zeros(1, 1, 3, sk, dtype=torch.bool)
    forbidden[0, 0, 0, 20:] = True  # row 0 sees keys 0..19: chunk 2 partial, chunk 3 fully hidden

    xa = x.clone().requires_grad_(True)
    scores = chunk_lse(torch.where(forbidden, torch.full_like(xa, MASK_NEG), xa), cs)
    scores.sum().backward()
    grad = xa.grad[0, 0, 0]

    # (1) partially visible chunk 2 (keys 16-23, only 16-19 visible)
    want = torch.softmax(x[0, 0, 0, 16:20], -1)
    assert (grad[16:20] - want).abs().max() < 1e-12, (
        f"partial chunk gradient is not the softmax over its visible keys: {grad[16:20]} vs {want}"
    )
    assert grad[20:24].abs().max() == 0.0, "masked keys received gradient"

    # (2) fully invisible chunk 3 gets NO gradient, not a uniform 1/cs
    assert grad[24:32].abs().max() == 0.0, (
        f"a fully invisible chunk received gradient {grad[24:32].abs().max():.4f} -- a naive softmax "
        f"over all-MASK_NEG slots would give a spurious uniform {1 / cs:.4f}"
    )

    # An all-masked chunk stays far below any real score, so it can never be selected. Note the
    # value is MASK_NEG * scale + log(chunk_size), NOT INVISIBLE_SCORE: logsumexp of finite
    # sentinels is finite, so the isfinite() guard is inert on this path. The log(n) term is a
    # ~4.85 offset on a ~-884 value, which is why that is harmless.
    scale = 128**-0.5
    invisible = float(chunk_lse(torch.full((1, 1, 1, cs), MASK_NEG, dtype=DT) * scale, cs))
    assert invisible < -800, invisible
    w = torch.softmax(torch.tensor([2.0, invisible], dtype=DT), 0)
    assert float(w[1]) == 0.0, f"an invisible chunk got softmax weight {float(w[1]):.3e}"

    # The guard IS live on the `valid=` path, which masks with true -inf. There it must return the
    # FINITE INVISIBLE_SCORE: an -inf would make the downstream softmax NaN for any row whose only
    # chunk is invisible -- the very first query is exactly that row.
    ok = torch.zeros(1, 1, 1, 2 * cs, dtype=torch.bool)
    ok[..., :cs] = True
    guarded = chunk_lse(torch.randn(1, 1, 1, 2 * cs, dtype=DT), cs, valid=ok)
    assert float(guarded[0, 0, 0, 1]) == INVISIBLE_SCORE, (
        f"the valid= path returned {float(guarded[0, 0, 0, 1])} for a fully-invalid chunk; it must be "
        f"the finite INVISIBLE_SCORE, or a softmax over it yields NaN"
    )
    assert torch.isfinite(guarded).all()


def test_stats_report_commitment():
    """
    ``chunk_entropy`` is 1.0 for a uniform router and near 0 for a committed one.

    The diagnostic exists because the loss cannot distinguish them: a router that learned to *use*
    whatever mixture it produces looks the same in the loss as one that learned to *choose*. Both
    ends are asserted, since a metric that only moves in one direction is not a readout.
    """
    cs = 8
    q, k, v, s = make_qkv(sq=16, sk=64, cs=cs, seed=16)
    flat = torch.zeros_like(s)
    _, uniform = hsa_chunk_attention(q, k, v, flat, chunk_size=cs, checkpoint=False)
    assert uniform["chunk_entropy"] > 0.99, uniform
    assert uniform["mass_top1"] < 0.5, uniform

    peaked = torch.zeros_like(s)
    peaked[..., 0] = 50.0
    _, committed = hsa_chunk_attention(q, k, v, peaked, chunk_size=cs, checkpoint=False)
    assert committed["chunk_entropy"] < 0.05, committed
    assert committed["mass_top1"] > 0.99, committed
    assert committed["effective_chunks"] < 1.1, committed


def test_shift_is_per_chunk_not_per_row_in_fp32():
    """
    The softmax shift must be each **chunk's** max, not the row's.

    A global row max is *algebraically* identical -- the shift cancels -- so an fp64 test cannot see
    the difference, and every other test here passed with the mutation in place. It is a real bug in
    fp32: a far-past chunk whose logits sit ~120 below the recent chunk's has every weight underflow
    to 0 under a global shift, so its ``sum`` is 0 and its whole contribution vanishes **while its
    ``w_c`` is unchanged**. The output silently loses that chunk -- exactly the long-range term the
    router exists to bring back.

    Measured on the shifted values directly: per-chunk shift gives the far chunk a valid distribution
    summing to 1.0; global shift gives 0.0.

    Constructed rather than sampled, because random logits do not span 120 nats. Runs in fp32 on
    purpose -- this is the one property whose test must NOT be in fp64.
    """
    cs, n_chunk = 8, 4
    sk = cs * n_chunk
    torch.manual_seed(20)
    # A key whose logit against every query is hugely negative for chunk 0, ~0 elsewhere. Built by
    # making q point along a direction and chunk 0's keys point far the other way.
    d = 4
    direction = torch.zeros(d, dtype=torch.float32)
    direction[0] = 1.0
    q = direction.view(1, 1, 1, d).expand(1, 1, sk, d).contiguous() * 2.0
    coef = torch.zeros(sk, dtype=torch.float32)
    coef[:cs] = -120.0 * d**0.5 / 2.0  # chunk 0: logits ~ -120 after scaling
    coef[cs:] = 0.0
    k = (coef.unsqueeze(-1) * direction).view(1, 1, sk, d)
    # Distinguishable values so a lost chunk changes the output.
    v = torch.zeros(1, 1, sk, d, dtype=torch.float32)
    v[0, :, :cs] = 5.0
    v[0, :, cs:] = 0.0
    # Give chunk 0 almost all the mass: if it is dropped the output goes to ~0 instead of ~5.
    s = torch.full((1, 1, sk, n_chunk), -20.0, dtype=torch.float32)
    s[..., 0] = 20.0

    out, _ = hsa_chunk_attention(q, k, v, s, chunk_size=cs, checkpoint=False)
    assert torch.isfinite(out).all()
    # The last query sees every chunk and puts ~all its mass on chunk 0, whose values are 5.0.
    got = float(out[0, 0, -1, 0])
    assert got > 4.0, (
        f"the far-past chunk's contribution was lost (output {got:.3f}, expected ~5.0). The softmax "
        f"shift is being taken over the whole row rather than per chunk, so that chunk's weights "
        f"underflowed to 0 while its w_c stayed large."
    )


def test_rejects_mismatched_scores():
    q, k, v, s = make_qkv(sq=16, sk=32, cs=8)
    with pytest.raises(ValueError, match="chunk_scores must be"):
        hsa_chunk_attention(q, k, v, s[..., :-1], chunk_size=8)
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_lse(torch.randn(1, 1, 4, 8, dtype=DT), 0)
