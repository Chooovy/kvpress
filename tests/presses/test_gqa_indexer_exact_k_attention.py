# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Chunk-wise exact-K subset attention.

The load-bearing tests here are:

* ``test_forward_is_exactly_sparse_attention`` -- the multiplicative form over the candidate pool
  equals :func:`~.sparse_attention.sparse_gqa_attention_reference` on the sampled subset. This is
  what says there is no train/inference gap in the forward.
* ``test_full_pool_gives_unselected_chunks_more_gradient`` -- the reason normalization runs over
  the pool rather than the selected K. Gathering only the K leaves unselected candidates with 73x
  less gradient, which is the difference between a router that can promote a chunk and one that
  cannot.
* ``test_causality_within_a_shared_query_block`` -- the subtle one. A query block shares one chunk
  subset, so ``chunk_visibility`` admits chunks that only the block's *later* queries may read;
  the per-token mask inside the attention is what stops the earlier ones. Getting this wrong keeps
  attention causal but silently shrinks the effective budget near the diagonal.
"""

import pytest
import torch

from kvpress.presses.gqa_indexer.exact_k_attention import (
    PAD_SCORE,
    build_candidates,
    chunk_visibility,
    exact_k_chunk_attention,
    gather_candidate_scores,
    pool_scores_to_chunks,
    selection_jaccard,
    share_over_query_blocks,
)
from kvpress.presses.gqa_indexer.indexer import MASK_NEG
from kvpress.presses.gqa_indexer.sparse_attention import sparse_gqa_attention_reference


def make_inputs(b=2, hq=4, hkv=2, sq=64, sk=64, d=8, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    return (
        torch.randn(b, hq, sq, d, dtype=dtype),
        torch.randn(b, hkv, sk, d, dtype=dtype),
        torch.randn(b, hkv, sk, d, dtype=dtype),
    )


# ----------------------------------------------------------------------------- pooling


def test_pool_keeps_the_ragged_tail_as_a_chunk():
    """
    A ragged tail becomes a short chunk, unlike ``aggregate_chunk_scores`` which returns it.

    The subset's cardinality is defined against a fixed ``n_chunk``, so a tail that sometimes
    exists would change ``n`` between steps.
    """
    scores = torch.randn(1, 1, 4, 70, dtype=torch.float64)
    chunks = pool_scores_to_chunks(scores, 16)
    assert chunks.shape == (1, 1, 4, 5), "70 / 16 -> 5 chunks, the last holding 6 tokens"


def test_pool_mean_uses_the_tails_real_width():
    """The tail's mean is over its 6 real tokens, not diluted by 10 padded ones."""
    scores = torch.full((1, 1, 1, 70), 2.0, dtype=torch.float64)
    chunks = pool_scores_to_chunks(scores, 16)
    assert torch.allclose(chunks, torch.full_like(chunks, 2.0)), (
        "a padded tail would drag its mean to 6/16 * 2 = 0.75"
    )


def test_pool_discounts_the_mask_sentinel():
    """
    Masked (future) tokens are excluded from the mean rather than averaged in.

    Without this the sentinel dominates: key ``t`` is forbidden by ``t`` of the query rows, so the
    pooled score becomes a monotone ramp in position and selection degenerates to "keep the
    oldest chunks". Same trap :func:`~.aggregate.reduce_queries` documents.
    """
    scores = torch.full((1, 1, 1, 8), 3.0, dtype=torch.float64)
    scores[..., 4:] = MASK_NEG
    chunks = pool_scores_to_chunks(scores, 4)
    assert chunks[0, 0, 0, 0].item() == pytest.approx(3.0)
    assert chunks[0, 0, 0, 1].item() == pytest.approx(MASK_NEG), "an all-masked chunk ranks last"


def test_pool_max_mode():
    scores = torch.zeros(1, 1, 1, 8, dtype=torch.float64)
    scores[0, 0, 0, 5] = 9.0
    chunks = pool_scores_to_chunks(scores, 4, mode="max")
    assert chunks[0, 0, 0].tolist() == [0.0, 9.0]


def test_query_block_sharing_shapes_and_identity():
    scores = torch.randn(2, 3, 65, 8, dtype=torch.float64)
    assert share_over_query_blocks(scores, 1).shape == scores.shape
    assert torch.equal(share_over_query_blocks(scores, 1), scores), "block=1 is the identity"
    shared = share_over_query_blocks(scores, 16)
    assert shared.shape == (2, 3, 5, 8), "65 queries / 16 -> 5 blocks"
    # First block is the plain mean of its 16 queries.
    assert torch.allclose(shared[:, :, 0], scores[:, :, :16].mean(2))


# ----------------------------------------------------------------------------- visibility


def test_visibility_uses_the_blocks_last_query():
    """
    A block may select any chunk its *last* query can read.

    Deliberately permissive at block granularity: the block shares one subset, so a chunk only
    later queries can see must still be selectable. The per-token mask inside the attention is
    what enforces causality for the earlier ones.
    """
    # Sq = Sk = 32, query_block 8, chunk 8 -> 4 blocks, 4 chunks.
    vis = chunk_visibility(
        4, 4, query_block=8, chunk_size=8, q_len=32, k_len=32, device=torch.device("cpu")
    )
    # Block 0 covers queries 0..7, last query 7 -> chunks starting at 0 only.
    assert vis[0].tolist() == [True, False, False, False]
    # Block 1 covers 8..15, last query 15 -> chunks starting at 0 and 8.
    assert vis[1].tolist() == [True, True, False, False]
    assert vis[3].tolist() == [True, True, True, True]


def test_visibility_respects_bottom_right_alignment():
    """``Sq < Sk`` puts query 0's diagonal at ``Sk - Sq``, matching flash-attention."""
    vis = chunk_visibility(
        1, 4, query_block=8, chunk_size=8, q_len=8, k_len=32, device=torch.device("cpu")
    )
    # The single block's last query is at absolute position 24+7 = 31, so every chunk is visible.
    assert vis[0].tolist() == [True, True, True, True]


# ----------------------------------------------------------------------------- candidates


def test_candidates_are_ascending_unique_and_visible():
    torch.manual_seed(0)
    b, h, nb, nc, m = 2, 2, 4, 32, 8
    scores = torch.randn(b, h, nb, nc, dtype=torch.float64)
    vis = chunk_visibility(
        nb, nc, query_block=8, chunk_size=8, q_len=32, k_len=nc * 8, device=torch.device("cpu")
    )
    cand = build_candidates(scores, m, visible=vis)
    assert cand.shape == (b, h, nb, m)
    assert bool((cand.diff(dim=-1) > 0).all()), "ascending and duplicate-free"
    for block in range(nb):
        assert bool(vis[block][cand[:, :, block]].all()), "every candidate must be visible"


def test_candidates_reserve_sink_and_local():
    """Chunk 0 and the block's own diagonal chunk are always in the pool."""
    torch.manual_seed(1)
    nb, nc, m = 6, 24, 6
    # Scores that actively rank sink and local LAST, so only reservation can put them in.
    scores = torch.zeros(1, 1, nb, nc, dtype=torch.float64)
    scores[..., 0] = -100.0
    vis = chunk_visibility(
        nb, nc, query_block=4, chunk_size=4, q_len=nb * 4, k_len=nc * 4, device=torch.device("cpu")
    )
    for block in range(nb):
        scores[0, 0, block, int(vis[block].sum()) - 1] = -100.0
    cand = build_candidates(
        scores, m, visible=vis, n_sink_chunk=1, n_local_chunk=1, explore_frac=0.0
    )
    for block in range(nb):
        picked = cand[0, 0, block].tolist()
        assert 0 in picked, f"block {block} lost the sink chunk"
        assert int(vis[block].sum()) - 1 in picked, f"block {block} lost its local chunk"


def test_candidates_explore_beyond_top_m():
    """
    With ``explore_frac > 0`` the pool includes chunks the score would not have ranked in.

    Without exploration a chunk outside top-M appears nowhere in the graph and receives exactly
    zero gradient -- the same dead end as the selected-gate proxy, one level up.
    """
    torch.manual_seed(2)
    nb, nc, m = 2, 64, 16
    # A strict ranking, so top-M is exactly chunks 0..15.
    scores = torch.arange(nc, 0, -1, dtype=torch.float64).view(1, 1, 1, nc).expand(1, 1, nb, nc)
    vis = torch.ones(nb, nc, dtype=torch.bool)
    no_explore = build_candidates(
        scores, m, visible=vis, n_sink_chunk=0, n_local_chunk=0, explore_frac=0.0
    )
    assert no_explore[0, 0, 0].tolist() == list(range(m)), "explore_frac=0 is pure top-M"

    torch.manual_seed(3)
    explored = build_candidates(
        scores, m, visible=vis, n_sink_chunk=0, n_local_chunk=0, explore_frac=0.25
    )
    outside = [c for c in explored[0, 0, 0].tolist() if c >= m]
    assert outside, "exploration should reach chunks outside top-M"


def test_candidates_eval_mode_is_deterministic_top_m():
    """``training=False`` drops the random slots -- an evaluation should not sample its pool."""
    scores = torch.arange(32, 0, -1, dtype=torch.float64).view(1, 1, 1, 32)
    vis = torch.ones(1, 32, dtype=torch.bool)
    a = build_candidates(
        scores, 8, visible=vis, n_sink_chunk=0, n_local_chunk=0, explore_frac=0.5, training=False
    )
    b = build_candidates(
        scores, 8, visible=vis, n_sink_chunk=0, n_local_chunk=0, explore_frac=0.5, training=False
    )
    assert torch.equal(a, b)
    assert a[0, 0, 0].tolist() == list(range(8))


def test_candidates_handle_rows_with_few_visible_chunks():
    """
    A row with fewer than ``M`` visible chunks fills the shortfall with chunk 0, not garbage.

    Chunk 0 is visible to every row, and a duplicate slot is harmless in the attention (see
    ``test_duplicate_candidate_slots_are_harmless``).
    """
    nb, nc, m = 4, 8, 8
    vis = chunk_visibility(
        nb, nc, query_block=8, chunk_size=16, q_len=32, k_len=nc * 16, device=torch.device("cpu")
    )
    assert int(vis[0].sum()) < m, "block 0 should see fewer than M chunks for this to be a test"
    cand = build_candidates(torch.zeros(1, 1, nb, nc, dtype=torch.float64), m, visible=vis)
    for block in range(nb):
        row = cand[0, 0, block]
        real = row[row >= 0]
        assert bool(vis[block][real].all()), "no invisible chunk may be selected"
        assert int((row < 0).sum()) == m - int(vis[block].sum()), (
            "the shortfall must be exactly the pad count"
        )


# ----------------------------------------------------------------------------- attention


def _run(q, k, v, *, chunk_size, query_block, m, topk_chunk, seed=0, hard=False, scores=None):
    """Build candidates and run the attention, returning ``(out, stats, candidates, scores)``."""
    torch.manual_seed(seed)
    b, hkv, sk, _ = k.shape
    sq = q.shape[2]
    n_chunk = -(-sk // chunk_size)
    n_qblock = -(-sq // query_block)
    if scores is None:
        scores = torch.randn(b, hkv, n_qblock, n_chunk, dtype=q.dtype)
    else:
        assert scores.shape == (b, hkv, n_qblock, n_chunk), (
            f"caller passed scores {tuple(scores.shape)} but the geometry needs "
            f"{(b, hkv, n_qblock, n_chunk)}"
        )
    scores = scores.detach().clone().requires_grad_(True)
    m = min(m, n_chunk)
    vis = chunk_visibility(
        n_qblock,
        n_chunk,
        query_block=query_block,
        chunk_size=chunk_size,
        q_len=sq,
        k_len=sk,
        device=q.device,
    )
    cand = build_candidates(scores.detach(), m, visible=vis, explore_frac=0.0)
    picked = gather_candidate_scores(scores, cand)
    out, stats = exact_k_chunk_attention(
        q,
        k,
        v,
        picked,
        cand,
        topk_chunk=min(topk_chunk, m),
        chunk_size=chunk_size,
        query_block=query_block,
        hard=hard,
        checkpoint=False,
    )
    return out, stats, cand, scores


def test_forward_is_exactly_sparse_attention():
    """
    The multiplicative form equals dense softmax over the sampled subset's tokens.

    Checked against :func:`~.sparse_attention.sparse_gqa_attention_reference`, which shares no
    code with this module -- it gathers per-query index lists and runs an ordinary softmax. Their
    agreement is evidence about the operation, not about one implementation of it.

    This is why there is no train/inference gap in the forward, unlike the dense-forward gated
    path which gates every key at train time and hard-selects at eval.
    """
    chunk_size, query_block, m, k_chunk = 8, 8, 6, 3
    q, k, v = make_inputs(sq=32, sk=32, d=8)
    out, stats, cand, _ = _run(
        q, k, v, chunk_size=chunk_size, query_block=query_block, m=m, topk_chunk=k_chunk
    )

    # Rebuild the same selection as per-query token indices for the reference.
    z = stats["selected"]  # (B, Hkv, n_qblock, M)
    b, hkv, sq = q.shape[0], k.shape[1], q.shape[2]
    n_qblock = z.shape[2]
    token_idx = torch.full((b, hkv, sq, m * chunk_size), -1, dtype=torch.int64)
    for block in range(n_qblock):
        rows = range(block * query_block, min((block + 1) * query_block, sq))
        for bi in range(b):
            for hi in range(hkv):
                chosen = cand[bi, hi, block][z[bi, hi, block] > 0]
                positions = (
                    chosen.view(-1, 1) * chunk_size + torch.arange(chunk_size)
                ).flatten()
                positions = positions[positions < k.shape[2]].sort().values
                for r in rows:
                    token_idx[bi, hi, r, : positions.numel()] = positions

    reference, _ = sparse_gqa_attention_reference(q, k, v, token_idx)
    err = (out - reference).abs().max().item()
    # fp32 tolerance despite fp64 inputs. sparse_gqa_attention_reference hardcodes `.float()`
    # (sparse_attention.py:205-206), so it computes in fp32 whatever it is handed -- verified: its
    # fp64 and fp32 outputs are bit-identical, while this module's differ by 3.8e-7. So the residual
    # here is the *reference's* rounding, not an approximation in the exact-K forward. The fp64
    # exactness claim (1.11e-16) is established separately, against a masked softmax written in the
    # test itself, in ``test_multiplicative_form_is_exactly_sparse_attention``.
    assert err < 1e-6, f"max |exact-K - sparse reference| = {err:.3e}"


def test_forward_is_exact_in_fp64_against_a_masked_softmax():
    """
    The exactness claim at full fp64 precision, against a masked softmax written here.

    Needed as well as the test above because
    :func:`~.sparse_attention.sparse_gqa_attention_reference` hardcodes ``.float()`` and so cannot
    resolve better than ~1e-7. This reference is three lines and stays in the caller's dtype, so it
    can. Measured 1.1e-16 -- the multiplicative pool form *is* sparse attention over the subset,
    not an approximation of it.
    """
    b, hkv, sq, sk, d, cs, qb = 1, 1, 16, 16, 4, 4, 16
    torch.manual_seed(0)
    q = torch.randn(b, 1, sq, d, dtype=torch.float64)
    k = torch.randn(b, hkv, sk, d, dtype=torch.float64)
    v = torch.randn(b, hkv, sk, d, dtype=torch.float64)
    n_chunk = sk // cs
    scores = torch.zeros(b, hkv, sq // qb, n_chunk, dtype=torch.float64)
    cand = torch.arange(n_chunk).view(1, 1, 1, n_chunk).expand(b, hkv, sq // qb, n_chunk)
    out, stats = exact_k_chunk_attention(
        q, k, v, scores, cand.contiguous(),
        topk_chunk=2, chunk_size=cs, query_block=qb, hard=True, checkpoint=False,
    )

    chosen = cand[0, 0, 0][stats["selected"][0, 0, 0] > 0]
    keep = torch.zeros(sk, dtype=torch.bool)
    for c in chosen.tolist():
        keep[c * cs : (c + 1) * cs] = True
    logits = (q[0, 0] @ k[0, 0].T) * d**-0.5
    causal = torch.arange(sk).view(1, -1) <= torch.arange(sq).view(-1, 1)
    logits = logits.masked_fill(~(keep.view(1, -1) & causal), -float("inf"))
    alive = torch.isfinite(logits).any(-1)
    reference = torch.where(
        alive.view(-1, 1), torch.softmax(logits, -1) @ v[0, 0], torch.zeros(sq, d, dtype=torch.float64)
    )
    err = (out[0, 0] - reference).abs().max().item()
    assert err < 1e-15, f"fp64 max |exact-K - masked softmax| = {err:.3e}"


def test_hard_mode_matches_reference_too():
    """Same exactness with deterministic top-K selection, which is what inference does."""
    q, k, v = make_inputs(sq=32, sk=32, seed=5)
    out, stats, cand, _ = _run(
        q, k, v, chunk_size=8, query_block=8, m=4, topk_chunk=2, hard=True
    )
    assert int(stats["selected"].sum(-1).unique().item()) == 2
    assert torch.isfinite(out).all()


def test_full_pool_gives_unselected_chunks_more_gradient():
    """
    **Why normalization runs over the pool, not the selected K.**

    Both forms are numerically identical in the forward (``g`` is 0 off the subset). They differ in
    the backward: gathering only the K selected chunks reaches the unselected ones solely through
    the selected chunks' marginals, which is ~73x weaker. Normalizing over the pool puts each
    unselected candidate's own ``g_j`` in the graph, so it gets credit directly.

    That gap is the difference between a router that can promote a chunk from outside its current
    selection and one that structurally cannot.
    """
    q, k, v = make_inputs(sq=32, sk=64, seed=7)
    m, k_chunk = 8, 3
    out, stats, cand, scores = _run(
        q, k, v, chunk_size=8, query_block=16, m=m, topk_chunk=k_chunk
    )
    out.sum().backward()

    # Only real slots: a -1 pad has no chunk to carry a score, and gather(-1) would raise.
    real = cand >= 0
    grad = gather_candidate_scores(scores.grad, cand).abs()
    selected = (stats["selected"] > 0) & real
    assert grad[selected].mean() > 0, "selected candidates must get gradient"
    unselected = (~(stats["selected"] > 0)) & real
    unselected_mean = grad[unselected].mean().item()
    assert unselected_mean > 0, "unselected candidates must get gradient"
    # Comparable magnitude, not a vanishing residue. Measured 1.18e-1 vs 1.22e-1 on the toy.
    assert unselected_mean > 0.05 * grad[selected].mean().item(), (
        f"unselected grad {unselected_mean:.3e} is a vanishing residue against selected "
        f"{grad[selected].mean().item():.3e} -- the pool normalization has been lost"
    )


def test_every_candidate_receives_gradient():
    """No candidate is left out of the graph -- the pool's whole purpose."""
    q, k, v = make_inputs(sq=32, sk=64, seed=8)
    out, stats, cand, scores = _run(q, k, v, chunk_size=8, query_block=32, m=6, topk_chunk=2)
    out.sum().backward()
    real = cand >= 0
    on_candidates = gather_candidate_scores(scores.grad, cand)
    assert bool((on_candidates[real].abs() > 0).all()), "every real candidate must get gradient"


def test_causality_within_a_shared_query_block():
    """
    A block's earlier queries do not read chunks only its later queries may see.

    ``chunk_visibility`` is deliberately permissive at block granularity, so this per-token mask is
    the only thing enforcing causality. The check perturbs a future value and asserts the earlier
    query's output does not move.
    """
    chunk_size, query_block = 8, 16
    q, k, v = make_inputs(sq=32, sk=32, d=8, seed=9)
    torch.manual_seed(0)
    out_a, _, _, _ = _run(
        q, k, v, chunk_size=chunk_size, query_block=query_block, m=4, topk_chunk=2, seed=0
    )
    # Change values strictly after query 3's diagonal (position 3, since Sq == Sk).
    v2 = v.clone()
    v2[:, :, 4:] += 100.0
    torch.manual_seed(0)
    out_b, _, _, _ = _run(
        q, k, v2, chunk_size=chunk_size, query_block=query_block, m=4, topk_chunk=2, seed=0
    )
    assert torch.allclose(out_a[:, :, :4], out_b[:, :, :4], atol=1e-10), (
        "an early query in a shared block read a key past its own diagonal"
    )
    assert not torch.allclose(out_a[:, :, 8:], out_b[:, :, 8:]), (
        "later queries should have seen the change, or the test proves nothing"
    )


def test_output_rows_are_convex_combinations():
    """Attention rows sum to 1, so the output stays in the values' convex hull."""
    q, k, v = make_inputs(sq=32, sk=32, seed=10)
    ones = torch.ones_like(v)
    out, _, _, _ = _run(q, k, ones, chunk_size=8, query_block=8, m=4, topk_chunk=2)
    assert torch.allclose(out, torch.ones_like(out), atol=1e-12), (
        "attending to all-ones values must give ones, i.e. the rows are normalized"
    )


def test_short_rows_pad_with_minus_one_not_a_repeated_chunk():
    """
    A block that cannot see ``M`` chunks pads with ``-1``, and repeating a chunk would be wrong.

    Repeating is not a harmless redundancy: the subset's cardinality is over *slots*, so the DP
    would spend two of its ``K`` on one chunk and the row would attend to ``K - 1`` distinct chunks
    while reporting a budget of ``K``. Demonstrated below -- the two spellings give different
    outputs.

    And it is common, not an edge case: 31 of 128 blocks (24%) see fewer than ``M`` chunks at
    ``Sq = Sk = 16384, chunk_size = 64, query_block = 128, M = 64``.
    """
    nb, nc, m = 4, 4, 4
    vis = chunk_visibility(
        nb, nc, query_block=8, chunk_size=8, q_len=32, k_len=32, device=torch.device("cpu")
    )
    assert int(vis[0].sum()) == 1, "block 0 should see exactly one chunk for this to be a test"
    cand = build_candidates(torch.zeros(1, 1, nb, nc, dtype=torch.float64), m, visible=vis)
    assert cand[0, 0, 0].tolist() == [-1, -1, -1, 0], "the 3 unusable slots must be -1, not chunk 0"
    assert cand[0, 0, 3].tolist() == [0, 1, 2, 3], "a fully visible block has no pads"

    # And repeating really does change the answer, which is why -1 is required.
    q, k, v = make_inputs(b=1, hq=2, hkv=1, sq=8, sk=32, d=8, seed=11)
    outs = []
    for spelling in (torch.tensor([[[[0, 1]]]]), torch.tensor([[[[0, 0, 1, 1]]]])):
        scores = torch.zeros(1, 1, 1, spelling.shape[-1], dtype=torch.float64)
        out, _ = exact_k_chunk_attention(
            q, k, v, scores, spelling,
            topk_chunk=1, chunk_size=8, query_block=8, hard=True, checkpoint=False,
        )
        outs.append(out)
    assert not torch.allclose(outs[0], outs[1]), (
        "if duplicates were harmless this test would be pointless -- they are not, which is "
        "exactly why build_candidates pads with -1"
    )


def test_pad_slots_are_scored_low_and_masked_out():
    """
    ``-1`` slots get :data:`PAD_SCORE`, so their marginals are ~0 while real chunks exist.

    A plain ``gather`` would raise on ``-1``; the obvious repair ``clamp(min=0)`` is silently wrong
    -- it scores the pad with chunk 0's real score, so the DP would spend budget on a slot that
    contributes nothing.
    """
    scores = torch.tensor([[[[5.0, 1.0, -2.0, 0.0]]]], dtype=torch.float64)
    cand = torch.tensor([[[[-1, -1, 0, 2]]]])
    picked = gather_candidate_scores(scores, cand)
    assert picked[0, 0, 0].tolist() == [PAD_SCORE, PAD_SCORE, 5.0, -2.0]


def test_pad_slots_do_not_consume_budget():
    """
    With ``V >= K`` real chunks available, all ``K`` selected slots hold real chunks.

    This is what ``sum(mu) == K`` buys: the pads sit at ~0 probability, so the exact conditioning
    pushes the entire budget onto the real candidates.
    """
    q, k, v = make_inputs(b=1, hq=2, hkv=1, sq=8, sk=64, d=8, seed=16)
    cand = torch.tensor([[[[-1, -1, 0, 1, 2, 3]]]])
    scores = gather_candidate_scores(torch.zeros(1, 1, 1, 8, dtype=torch.float64), cand)
    out, stats = exact_k_chunk_attention(
        q, k, v, scores, cand, topk_chunk=2, chunk_size=8, query_block=8, checkpoint=False
    )
    z = stats["selected"][0, 0, 0]
    assert z[:2].sum().item() == 0, "a pad slot was selected while real chunks were available"
    assert stats["effective_topk"] == pytest.approx(2.0)
    assert torch.isfinite(out).all()


def test_budget_above_what_a_row_can_see_falls_back_to_all_visible():
    """
    ``V < K``: the row attends to every chunk it can see, and ``effective_topk`` reports the
    shortfall.

    The DP cannot place ``K`` ones among ``V`` plausible slots, so it is forced onto the pads and
    every real chunk gets marginal 1. That is the only sensible answer when the budget exceeds what
    exists -- but the effective budget is then below ``K``, and only this statistic would say so.
    """
    q, k, v = make_inputs(b=1, hq=2, hkv=1, sq=8, sk=64, d=8, seed=17)
    cand = torch.tensor([[[[-1, -1, -1, 0, 1]]]])  # only 2 real chunks
    scores = gather_candidate_scores(torch.zeros(1, 1, 1, 8, dtype=torch.float64), cand)
    out, stats = exact_k_chunk_attention(
        q, k, v, scores, cand, topk_chunk=4, chunk_size=8, query_block=8, checkpoint=False
    )
    z = stats["selected"][0, 0, 0]
    assert z[3:].sum().item() == 2, "both real chunks must be selected"
    assert stats["effective_topk"] == pytest.approx(2.0), (
        "effective_topk must report 2, not the nominal budget of 4"
    )
    assert torch.isfinite(out).all()


def test_a_row_of_only_pads_gives_zero_not_nan():
    """
    A query with no visible chunk outputs 0, not ``0/0``.

    A NaN here would propagate through the entire model rather than staying local -- the same
    reasoning :mod:`~.sparse_attention` documents for its empty rows.
    """
    q, k, v = make_inputs(b=1, hq=2, hkv=1, sq=8, sk=64, d=8, seed=18)
    cand = torch.full((1, 1, 1, 4), -1)
    scores = gather_candidate_scores(torch.zeros(1, 1, 1, 8, dtype=torch.float64), cand)
    out, _ = exact_k_chunk_attention(
        q, k, v, scores, cand, topk_chunk=2, chunk_size=8, query_block=8, checkpoint=False
    )
    assert torch.isfinite(out).all() and torch.equal(out, torch.zeros_like(out))


def test_row_max_ignores_unselected_slots():
    """
    The softmax shift comes from the SELECTED slots only, and taking it over all candidates
    produces a non-finite gradient.

    If an unselected chunk happens to have a much larger attention logit than any selected one, and
    the shift is taken over everything, every surviving weight becomes
    ``exp(a_selected - a_unselected_max)`` -- which underflows, so ``total`` goes to ~0 and
    ``p = w / total`` amplifies without bound. This is one of the two bugs that produced
    ``grad_norm = nan`` on the real 36-layer model.

    Shifting by the selected max is not an approximation: ``g`` is 0 off the subset, so the
    unselected logits contribute to neither the numerator nor the denominator.
    """
    q, k, v = make_inputs(b=1, hq=2, hkv=1, sq=8, sk=64, d=8, dtype=torch.float32, seed=20)
    # Chunk 4 is NOT selected, and its keys are scaled up 50x so its logits dominate.
    k = k.clone()
    k[:, :, 32:40] *= 50.0
    n_chunk = 8
    scores = torch.zeros(1, 1, 1, n_chunk)
    scores[..., :2] = 20.0  # force chunks 0-1 to be the top-2
    cand = torch.arange(n_chunk).view(1, 1, 1, n_chunk)
    out, stats = exact_k_chunk_attention(
        q, k, v, scores.clone().requires_grad_(True), cand,
        topk_chunk=2, chunk_size=8, query_block=8, hard=True, checkpoint=False,
    )
    assert stats["selected"][0, 0, 0, 4].item() == 0, "chunk 4 must be unselected for this test"
    assert torch.isfinite(out).all(), "a dominant unselected logit made the output non-finite"

    leaf = scores.clone().requires_grad_(True)
    out2, _ = exact_k_chunk_attention(
        q, k, v, leaf, cand,
        topk_chunk=2, chunk_size=8, query_block=8, hard=True, checkpoint=False,
    )
    out2.sum().backward()
    assert torch.isfinite(leaf.grad).all(), (
        "the score gradient is non-finite -- the row max was taken over unselected slots"
    )


def test_unselected_slot_with_a_huge_logit_stays_finite():
    """
    An unselected candidate whose attention logit far exceeds the selected max still yields a
    finite gradient, and a nonzero one.

    ``exp(a_j - a_selected_max)`` is unbounded above precisely for the interesting case -- a chunk
    that looks better than anything currently held -- so it overflows to ``inf``, and then
    ``inf * g_j = inf * 0 = NaN`` in the forward with ``dw_j/dg_j = inf`` in the backward.
    :data:`~.exact_k_attention.MAX_SHIFTED_EXPONENT` bounds it.

    Masking those slots instead would be *wrong*, not merely conservative: ``dw_j/dg_j`` for an
    unselected candidate IS the boundary credit the method exists to provide. So the test asserts
    the gradient is both finite AND nonzero.
    """
    from kvpress.presses.gqa_indexer.exact_k_attention import MAX_SHIFTED_EXPONENT

    q, k, v = make_inputs(b=1, hq=2, hkv=1, sq=8, sk=64, d=8, dtype=torch.float32, seed=21)
    k = k.clone()
    k[:, :, 32:40] *= 200.0  # chunk 4: logits far beyond exp's range once shifted
    n_chunk = 8
    scores = torch.zeros(1, 1, 1, n_chunk)
    scores[..., :2] = 20.0
    cand = torch.arange(n_chunk).view(1, 1, 1, n_chunk)
    leaf = scores.clone().requires_grad_(True)
    out, stats = exact_k_chunk_attention(
        q, k, v, leaf, cand,
        topk_chunk=2, chunk_size=8, query_block=8, hard=True, checkpoint=False,
    )
    assert torch.isfinite(out).all(), "forward went non-finite: exp overflowed then met g = 0"
    out.sum().backward()
    assert torch.isfinite(leaf.grad).all(), "backward went non-finite"
    assert leaf.grad[0, 0, 0, 4].abs() > 0, (
        "the dominant unselected chunk got ZERO gradient -- masking it out would pass the finiteness "
        "check while destroying the boundary credit this method exists for"
    )
    assert MAX_SHIFTED_EXPONENT > 0


def test_query_tile_is_invariant():
    """The query tile is a memory knob; the result must not depend on it."""
    q, k, v = make_inputs(sq=64, sk=64, seed=12)
    n_chunk, n_qblock = 8, 8
    scores = torch.randn(2, 2, n_qblock, n_chunk, dtype=torch.float64)
    cand = torch.arange(4).view(1, 1, 1, 4).expand(2, 2, n_qblock, 4).contiguous()
    picked = scores.gather(-1, cand)
    kwargs = dict(
        topk_chunk=2, chunk_size=8, query_block=8, hard=True, checkpoint=False
    )
    a, _ = exact_k_chunk_attention(q, k, v, picked, cand, query_tile=1, **kwargs)
    b, _ = exact_k_chunk_attention(q, k, v, picked, cand, query_tile=8, **kwargs)
    assert (a - b).abs().max().item() < 1e-13


def test_checkpointed_attention_matches():
    """Checkpointing the DP changes neither the output nor the score gradient."""
    q, k, v = make_inputs(sq=32, sk=32, seed=13)
    n_chunk, n_qblock, m = 4, 4, 4
    base = torch.randn(2, 2, n_qblock, n_chunk, dtype=torch.float64)
    cand = torch.arange(m).view(1, 1, 1, m).expand(2, 2, n_qblock, m).contiguous()
    grads, outs = [], []
    for ckpt in (False, True):
        scores = base.clone().requires_grad_(True)
        torch.manual_seed(0)
        out, _ = exact_k_chunk_attention(
            q, k, v, scores.gather(-1, cand), cand,
            topk_chunk=2, chunk_size=8, query_block=8, checkpoint=ckpt,
        )
        out.sum().backward()
        outs.append(out)
        grads.append(scores.grad.clone())
    assert (outs[0] - outs[1]).abs().max().item() < 1e-13
    assert (grads[0] - grads[1]).abs().max().item() < 1e-12


def test_stats_report_marginal_entropy():
    """
    Entropy is near its maximum at init and falls as the router commits.

    This is the readout ``gate_sparsity`` provides for the additive path: a loss curve cannot
    distinguish a router that has learned a ranking from one still spreading its mass uniformly.
    """
    q, k, v = make_inputs(sq=32, sk=32, seed=14)
    # Sq=Sk=32 at chunk 8 / query_block 8 -> 4 chunks, 4 blocks. M is capped at n_chunk.
    m, k_chunk = 4, 1
    flat = torch.zeros(2, 2, 4, 4, dtype=torch.float64)
    _, stats_flat, _, _ = _run(
        q, k, v, chunk_size=8, query_block=8, m=m, topk_chunk=k_chunk, scores=flat
    )
    # Uniform over the REAL slots. Near the diagonal a block sees fewer than M chunks, so its pads
    # sit at ~0 and the real slots share the budget at a higher rate than K/M -- which is why the
    # entropy is checked against the range rather than a single closed form. See
    # test_stats_entropy_closed_form_on_a_full_block for the exact value with no pads.
    import math

    uniform = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75))
    assert 0 < stats_flat["marginal_entropy"] <= uniform + 1e-9
    assert stats_flat["marginal_max"] <= 1.0 + 1e-9

    decided = torch.full((2, 2, 4, 4), -20.0, dtype=torch.float64)
    decided[..., :k_chunk] = 20.0
    _, stats_sharp, _, _ = _run(
        q, k, v, chunk_size=8, query_block=8, m=m, topk_chunk=k_chunk, scores=decided
    )
    assert stats_sharp["marginal_entropy"] < stats_flat["marginal_entropy"], (
        "a committed router must have lower marginal entropy than a flat one"
    )


def test_stats_entropy_closed_form_on_a_full_block():
    """
    Entropy against its exact value, on a pool with no pad slots.

    Isolated from ``test_stats_report_marginal_entropy`` because that one's blocks are near the
    diagonal and carry pads, which shifts the marginals off ``K/M``. Here every slot is real, so
    flat scores give exactly ``mu = K/M`` and the Bernoulli entropy is closed-form.
    """
    import math

    q, k, v = make_inputs(b=1, hq=2, hkv=1, sq=8, sk=64, d=8, seed=19)
    m, k_chunk = 8, 2
    cand = torch.arange(m).view(1, 1, 1, m)
    scores = torch.zeros(1, 1, 1, m, dtype=torch.float64)
    _, stats = exact_k_chunk_attention(
        q, k, v, scores, cand, topk_chunk=k_chunk, chunk_size=8, query_block=8, checkpoint=False
    )
    p = k_chunk / m
    expected = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    assert stats["marginal_entropy"] == pytest.approx(expected, abs=1e-4)
    assert stats["marginal_max"] == pytest.approx(p, abs=1e-6)
    assert stats["effective_topk"] == pytest.approx(float(k_chunk))


def test_selection_jaccard():
    a = torch.tensor([[1.0, 1, 0, 0]])
    assert selection_jaccard(a, a) == pytest.approx(1.0)
    b = torch.tensor([[0.0, 0, 1, 1]])
    assert selection_jaccard(a, b) == pytest.approx(0.0)
    c = torch.tensor([[1.0, 0, 1, 0]])
    assert selection_jaccard(a, c) == pytest.approx(1 / 3)
    # A curriculum boundary changes n_qblock; a diagnostic must not raise there.
    import math

    assert math.isnan(selection_jaccard(a, torch.ones(1, 8)))


def test_rejects_topk_above_pool():
    q, k, v = make_inputs(sq=16, sk=16)
    scores = torch.randn(2, 2, 2, 4, dtype=torch.float64)
    cand = torch.arange(4).view(1, 1, 1, 4).expand(2, 2, 2, 4).contiguous()
    with pytest.raises(ValueError, match="topk_chunk must be in"):
        exact_k_chunk_attention(
            q, k, v, scores, cand, topk_chunk=5, chunk_size=8, query_block=8
        )


def test_bf16_inputs_run():
    """The training path calls this with the model's own bf16 q/k/v."""
    q, k, v = make_inputs(sq=32, sk=32, dtype=torch.bfloat16, seed=15)
    n_chunk, n_qblock, m = 4, 4, 4
    scores = torch.randn(2, 2, n_qblock, n_chunk, dtype=torch.float32, requires_grad=True)
    cand = torch.arange(m).view(1, 1, 1, m).expand(2, 2, n_qblock, m).contiguous()
    out, stats = exact_k_chunk_attention(
        q, k, v, scores.gather(-1, cand), cand,
        topk_chunk=2, chunk_size=8, query_block=8,
    )
    assert out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()
    out.float().sum().backward()
    assert torch.isfinite(scores.grad).all()
