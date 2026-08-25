# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
The swap oracle and its supporting diagnostics.

``HANDOFF_exact_k_subset.md`` §7 names the swap oracle **the single most informative diagnostic**:
it directly answers "does our gradient recover the discrete marginal utility", which no loss curve
and no entropy trace can.

The tests here do two distinct things, and it is worth separating them:

1. **Validate the instrument** -- on a synthetic problem whose true swap utility is known
   analytically, the measurement must report near-perfect agreement, and must report *chance* for a
   deliberately uninformative gradient. Without both halves a diagnostic that always says "0.9" is
   indistinguishable from a working one.
2. **Apply it to the real estimator** -- exact-K's own gradient, measured against a genuine
   double-forward oracle. Measured: Spearman **+0.69**, Pearson **+0.82**, centered sign **0.58**,
   raw sign **0.25**. The last of those looks like a failure and is not; see
   ``test_exact_k_gradient_recovers_the_swap_ranking`` for why the rank statistics are the ones that
   bear on the question, and ``exact_k_diagnostics`` for the measured population offset that
   explains the discrepancy.
"""

import math

import pytest
import torch

from kvpress.presses.gqa_indexer.exact_k_attention import (
    build_candidates,
    chunk_visibility,
    exact_k_chunk_attention,
    gather_candidate_scores,
)
from kvpress.presses.gqa_indexer.exact_k_diagnostics import (
    lm_loss_regret,
    router_recall_at_k,
    swap_oracle_correlation,
)


# ----------------------------------------------------------------- validating the instrument


def test_swap_oracle_detects_a_perfect_gradient():
    """
    A gradient that *is* the true utility scores ~1.0 on all three measures.

    The synthetic problem: each item has a fixed value ``v_i`` and the loss is ``-sum(v_i)`` over the
    selected set, so swapping ``i`` for ``j`` changes the loss by exactly ``v_i - v_j``. The exact
    gradient wrt a score gating item ``i`` is ``-v_i``, so a perfect estimator's
    ``g_j - g_i = -v_j + v_i = v_i - v_j`` -- identical to the truth.

    **This test earned its place**: the first version of the diagnostic used ``-(g_j - g_i)``, having
    reasoned about which way descent moves a score rather than about how the loss changes. Those have
    opposite signs, so a perfect gradient scored 0.0 and a perfectly wrong one scored 1.0 -- an
    inversion that looks like a plausible result in isolation. Hence 1.0 exactly, not "above
    chance".
    """
    torch.manual_seed(0)
    n, k = 16, 5
    values = torch.randn(n, dtype=torch.float64)
    selected = torch.zeros(1, n, dtype=torch.float64)
    selected[0, values.topk(k).indices] = 1.0

    def loss_fn(mask):
        return -(mask[0] * values).sum()

    perfect_grad = (-values).unsqueeze(0)
    result = swap_oracle_correlation(loss_fn, selected, perfect_grad, max_pairs=48)

    assert result.n_pairs > 30, f"too few pairs to conclude anything: {result.n_pairs}"
    assert result.sign_accuracy == pytest.approx(1.0), str(result)
    assert result.pearson == pytest.approx(1.0, abs=1e-9), str(result)
    assert result.spearman == pytest.approx(1.0, abs=1e-9), str(result)


def test_swap_oracle_detects_an_inverted_gradient():
    """A sign-flipped gradient scores 0.0 sign accuracy and -1 correlation, not ~0.5."""
    torch.manual_seed(1)
    n, k = 16, 5
    values = torch.randn(n, dtype=torch.float64)
    selected = torch.zeros(1, n, dtype=torch.float64)
    selected[0, values.topk(k).indices] = 1.0

    result = swap_oracle_correlation(
        lambda mask: -(mask[0] * values).sum(), selected, values.unsqueeze(0), max_pairs=48
    )
    assert result.sign_accuracy == pytest.approx(0.0), str(result)
    assert result.pearson == pytest.approx(-1.0, abs=1e-9), str(result)


def test_swap_oracle_reports_chance_for_a_random_gradient():
    """
    An uninformative gradient must score ~0.5, not something that looks like a result.

    This is the half that makes the diagnostic falsifiable. A measurement that reports high agreement
    for noise cannot be used as evidence for anything.
    """
    torch.manual_seed(2)
    n, k = 24, 8
    values = torch.randn(n, dtype=torch.float64)
    selected = torch.zeros(1, n, dtype=torch.float64)
    selected[0, values.topk(k).indices] = 1.0

    accuracies = []
    for seed in range(8):
        gen = torch.Generator().manual_seed(seed)
        noise = torch.randn(1, n, dtype=torch.float64, generator=gen)
        accuracies.append(
            swap_oracle_correlation(
                lambda mask: -(mask[0] * values).sum(), selected, noise,
                max_pairs=48, generator=gen,
            ).sign_accuracy
        )
    mean = sum(accuracies) / len(accuracies)
    assert 0.3 < mean < 0.7, f"random gradient scored {mean:.3f}, expected ~0.5: {accuracies}"


def test_swap_oracle_is_reproducible_under_a_generator():
    torch.manual_seed(3)
    n, k = 20, 6
    values = torch.randn(n, dtype=torch.float64)
    selected = torch.zeros(1, n, dtype=torch.float64)
    selected[0, values.topk(k).indices] = 1.0
    grad = (-values).unsqueeze(0)

    def run(seed):
        return swap_oracle_correlation(
            lambda mask: -(mask[0] * values).sum(), selected, grad,
            max_pairs=20, generator=torch.Generator().manual_seed(seed),
        )

    a, b = run(7), run(7)
    assert (a.sign_accuracy, a.n_pairs) == (b.sign_accuracy, b.n_pairs)


def test_swap_oracle_excludes_zero_effect_pairs_from_sign_accuracy():
    """
    A swap with no effect has no direction to get right, so it must not be scored as a coin flip.

    Counting them would drag any result toward 0.5 in proportion to how many interchangeable items
    the problem has -- i.e. the metric would depend on the problem's degeneracy rather than on the
    estimator.
    """
    n, k = 12, 4
    values = torch.zeros(n, dtype=torch.float64)  # every swap is worth exactly 0
    selected = torch.zeros(1, n, dtype=torch.float64)
    selected[0, :k] = 1.0
    result = swap_oracle_correlation(
        lambda mask: -(mask[0] * values).sum(), selected,
        torch.randn(1, n, dtype=torch.float64), max_pairs=16,
    )
    assert math.isnan(result.sign_accuracy), "all-zero utilities should give NaN, not a number"
    assert result.mean_abs_true == 0.0


def test_swap_oracle_rejects_a_degenerate_mask():
    """A fully-selected mask has no boundary, and that is an error rather than an empty result."""
    selected = torch.ones(1, 8, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="no boundary pairs"):
        swap_oracle_correlation(lambda m: m.sum(), selected, torch.zeros(1, 8), max_pairs=4)


def test_spearman_handles_ties():
    """Tied predictions must not make Spearman depend on sort order."""
    torch.manual_seed(4)
    n, k = 16, 5
    values = torch.randn(n, dtype=torch.float64)
    selected = torch.zeros(1, n, dtype=torch.float64)
    selected[0, values.topk(k).indices] = 1.0
    # A constant gradient: every prediction ties, so no ranking exists and Spearman is undefined.
    result = swap_oracle_correlation(
        lambda mask: -(mask[0] * values).sum(), selected,
        torch.full((1, n), 0.5, dtype=torch.float64), max_pairs=24,
    )
    assert math.isnan(result.spearman), "a constant predictor should give NaN, not 0.0"


# ----------------------------------------------------------------- applying it to exact-K


def test_exact_k_gradient_recovers_the_swap_ranking():
    """
    **The finding.** Exact-K's gradient recovers the *ranking* of true swap utility:
    Spearman +0.69, Pearson +0.82, centered sign accuracy 0.58 (chance 0.5).

    The oracle is genuine -- for each boundary pair the attention is re-run with the swapped mask
    forced, so ``dL_true`` is a double-forward measurement rather than a linearization.

    **Why the assertion is on rank, not on raw sign.** Raw sign accuracy is **0.25**, which looks
    like a failure and is not. The two gradient populations are offset (mean ``g`` on selected
    chunks ``+2.8e-3`` against ``-9.2e-4`` on unselected), so every prediction carries a systematic
    ``-3.7e-3`` shift; meanwhile 85% of real swaps *hurt*, so ``dL_true`` is mostly positive. Signs
    therefore disagree almost everywhere while the ordering is largely right. A constant added to
    every score's gradient cannot change which chunk wins a comparison, and comparisons are all the
    router makes -- so the rank statistics are the ones that bear on whether the estimator works.

    Asserted with margin above chance rather than against a tight threshold: the honest claim is
    that the ranking is recovered, not that it is recovered to a particular number. The measured
    values are printed so a regression appears as a number rather than a pass/fail flip.
    """
    torch.manual_seed(10)
    b, hq, hkv, sq, sk, d = 1, 2, 1, 32, 128, 16
    cs, qb, m, k_chunk = 8, 32, 16, 4
    q = torch.randn(b, hq, sq, d, dtype=torch.float64)
    key = torch.randn(b, hkv, sk, d, dtype=torch.float64)
    value = torch.randn(b, hkv, sk, d, dtype=torch.float64)
    target = torch.randn(b, hq, sq, d, dtype=torch.float64)

    n_chunk, n_qblock = sk // cs, sq // qb
    vis = chunk_visibility(
        n_qblock, n_chunk, query_block=qb, chunk_size=cs, q_len=sq, k_len=sk, device=q.device
    )
    base_scores = torch.randn(b, hkv, n_qblock, n_chunk, dtype=torch.float64)
    cand = build_candidates(base_scores, m, visible=vis, explore_frac=0.0)

    def forced_loss(mask):
        """
        Loss with the subset forced to ``mask``.

        The +1e4 offset plus ``hard=True`` reproduces any target subset exactly. Forcing is what
        makes the oracle deterministic: ``exact_k_chunk_attention`` samples, so an unforced re-run
        would measure sampling noise rather than the swap.
        """
        picked = gather_candidate_scores(base_scores, cand)
        out, _ = exact_k_chunk_attention(
            q, key, value, picked + 1e4 * mask, cand,
            topk_chunk=k_chunk, chunk_size=cs, query_block=qb,
            hard=True, checkpoint=False, checkpoint_attention=False,
        )
        return ((out - target) ** 2).mean().item()

    leaf = base_scores.clone().requires_grad_(True)
    out, stats = exact_k_chunk_attention(
        q, key, value, gather_candidate_scores(leaf, cand), cand,
        topk_chunk=k_chunk, chunk_size=cs, query_block=qb,
        checkpoint=False, checkpoint_attention=False,
    )
    ((out - target) ** 2).mean().backward()

    result = swap_oracle_correlation(
        forced_loss, stats["selected"], gather_candidate_scores(leaf.grad, cand),
        max_pairs=40, generator=torch.Generator().manual_seed(0),
    )
    print(f"\nexact-K {result}")

    assert result.n_pairs >= 30, f"too few pairs: {result.n_pairs}"
    assert result.mean_abs_true > 0, "no swap changed the loss -- the oracle has no signal to recover"
    assert result.spearman > 0.35, (
        f"exact-K's gradient does not recover the swap RANKING: {result}. That is a finding about "
        f"the estimator, not necessarily a bug -- but it must not pass silently."
    )
    assert result.centered_sign_accuracy > 0.5, (
        f"centered sign accuracy is at or below chance: {result}"
    )


def test_swap_oracle_bias_is_reported_not_hidden():
    """
    The offset that makes raw sign accuracy misleading is surfaced as a field, and centering it out
    does not damage a gradient that was already correct.

    Without ``bias``, a reader seeing raw sign 0.25 on the real estimator would conclude it is
    anti-correlated, when Spearman on the same pairs is +0.69. Reporting the offset is what makes the
    two numbers reconcilable rather than contradictory.

    Note ``bias`` is ``median(dL_hat)`` over the sampled pairs, so it is generally **nonzero even for
    a perfect estimator** -- it tracks the median true utility, which is nonzero whenever swaps are
    asymmetrically good or bad. That is why it is reported as context for the raw sign accuracy
    rather than treated as an error signal: the test asserts that centering leaves a perfect gradient
    perfect, not that the bias vanishes.
    """
    torch.manual_seed(11)
    n, k = 20, 6
    values = torch.randn(n, dtype=torch.float64)
    selected = torch.zeros(1, n, dtype=torch.float64)
    selected[0, values.topk(k).indices] = 1.0

    result = swap_oracle_correlation(
        lambda mask: -(mask[0] * values).sum(), selected, (-values).unsqueeze(0), max_pairs=40
    )
    # Swapping a top-k item out for a non-top-k item almost always hurts, so the median utility --
    # and hence the median prediction -- is positive rather than 0.
    assert result.bias > 0, f"expected a positive median utility for a good subset: {result}"
    # Centering must not damage a gradient that was already exactly right.
    assert result.sign_accuracy == pytest.approx(1.0)
    assert result.centered_sign_accuracy == pytest.approx(1.0), (
        "centering broke a perfect estimator -- it subtracts the median from BOTH sides, so a "
        "monotone predictor must survive it"
    )
    assert result.spearman == pytest.approx(1.0, abs=1e-9)


# ----------------------------------------------------------------- supporting measures


def test_router_recall_at_k():
    """Recall is 1.0 for a router that agrees with the oracle, and ~k/n for a random one."""
    oracle = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0]])
    assert router_recall_at_k(oracle.clone(), oracle, 3) == pytest.approx(1.0)
    # Exactly reversed: the top-3 and bottom-3 are disjoint.
    assert router_recall_at_k(-oracle, oracle, 3) == pytest.approx(0.0)
    # Partial overlap: top-2 of the router is {0, 7}, oracle's is {0, 1} -> 1 of 2.
    router = torch.tensor([[9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.0]])
    assert router_recall_at_k(router, oracle, 2) == pytest.approx(0.5)


def test_router_recall_respects_validity():
    """An invalid (invisible) chunk cannot be credited or blamed."""
    oracle = torch.tensor([[5.0, 4.0, 3.0, 2.0]])
    router = torch.tensor([[0.0, 0.0, 9.0, 8.0]])
    valid = torch.tensor([[False, False, True, True]])
    # Restricted to the last two, both agree on their ordering.
    assert router_recall_at_k(router, oracle, 2, valid=valid) == pytest.approx(1.0)


def test_lm_loss_regret_is_zero_for_the_oracle_subset():
    """Regret against oneself is 0, and positive for a worse subset."""
    values = torch.tensor([3.0, 2.0, 1.0, 0.0])

    def loss_fn(mask):
        return -(mask * values).sum()

    oracle = torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert lm_loss_regret(loss_fn, oracle, oracle) == pytest.approx(0.0)
    worse = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert lm_loss_regret(loss_fn, worse, oracle) == pytest.approx(4.0)
