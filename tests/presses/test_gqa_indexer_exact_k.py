# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Correctness of the exact-K subset router.

Everything reference-checked here runs in fp64, so the tolerances measure floating-point noise
rather than the DP's approximation error -- there is none to measure, the DP is exact.

The two tests worth reading first are ``test_unselected_items_receive_gradient`` (the property
that motivates the whole method: an item outside the sampled subset still gets credit, which the
selected-set-only proxy cannot do) and ``test_clamp_is_required`` (which fails if the ``-1e-7``
clamp in :func:`~.exact_k_subset.log_sigmoid` is removed -- it is not defensive coding).
"""

from itertools import combinations

import pytest
import torch
import torch.nn.functional as F

import kvpress.presses.gqa_indexer.exact_k_subset as exact_k_module
from kvpress.presses.gqa_indexer.exact_k_subset import (
    LOG_P_MAX,
    NEG_INF,
    exact_k_marginals,
    log1mexp,
    log_pr_exactly_k,
    log_sigmoid,
    sample_k_subset,
    straight_through_mask,
    subset_indices,
)


def brute_force_marginals(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Marginals by enumerating all ``C(n, k)`` subsets. Only tractable for small ``n``.

    Deliberately shares no code with the DP: it works in probability space over explicit subsets,
    while the DP works in log space over counts. Agreement is therefore evidence about the
    *definition* of ``P(z_i = 1 | sum z = k)`` rather than about one implementation of it.
    """
    n = logits.shape[-1]
    p = torch.sigmoid(logits.double())
    total = torch.zeros((), dtype=torch.float64)
    per_item = torch.zeros(n, dtype=torch.float64)
    for subset in combinations(range(n), k):
        weight = torch.ones((), dtype=torch.float64)
        chosen = set(subset)
        for i in range(n):
            weight = weight * (p[i] if i in chosen else 1 - p[i])
        total = total + weight
        for i in subset:
            per_item[i] = per_item[i] + weight
    return per_item / total


@pytest.mark.parametrize("n,k", [(9, 4), (12, 3), (8, 1), (7, 7), (6, 0)])
def test_marginals_match_brute_force(n, k):
    """The DP reproduces enumeration to floating-point noise, in fp64."""
    torch.manual_seed(0)
    logits = torch.randn(n, dtype=torch.float64) * 2.0
    mu = exact_k_marginals(logits.unsqueeze(0), k).squeeze(0)
    if k == 0:
        assert torch.allclose(mu, torch.zeros_like(mu), atol=1e-14)
        return
    reference = brute_force_marginals(logits, k)
    err = (mu - reference).abs().max().item()
    assert err < 1e-14, f"max |DP - enumeration| = {err:.3e}"


def test_marginals_sum_to_k():
    """``sum(mu) == k``: the conditioning is on the cardinality, so this is not approximate."""
    torch.manual_seed(1)
    logits = torch.randn(4, 5, 32, dtype=torch.float64) * 3.0
    for k in (1, 4, 16, 31):
        mu = exact_k_marginals(logits, k)
        totals = mu.sum(-1)
        assert torch.allclose(totals, torch.full_like(totals, float(k)), atol=1e-11)


def test_equal_scores_give_uniform_marginals():
    """
    No-op immunity. Equal scores -> ``mu_i = k/n`` for every ``i``.

    This is the structural claim the module exists for. An additive gate with equal scores is
    *inert* (softmax is shift-invariant), so the model falls back to the frozen backbone having
    learned nothing. Here equal scores give a well-defined uniform marginal and the forward still
    commits to exactly ``k`` items, so "flat" is not a free pass.
    """
    n, k = 24, 6
    for value in (-5.0, 0.0, 5.0):
        logits = torch.full((3, n), value, dtype=torch.float64)
        mu = exact_k_marginals(logits, k)
        assert torch.allclose(mu, torch.full_like(mu, k / n), atol=1e-12)


def test_unselected_items_receive_gradient():
    """
    **The property that motivates exact-K.** An item NOT in the sampled subset still gets
    gradient, so a key outside the current top-k can be promoted.

    The comparison is against the selected-gate proxy, whose gradient on unselected items is
    identically zero -- structurally stuck at whatever it started with. Measured on an adversarial
    retrieval toy (needle outside top-k, strong frozen backbone) that difference is 0.0% vs 93.8%
    final recall.
    """
    torch.manual_seed(2)
    n, k = 20, 5
    logits = (torch.randn(n, dtype=torch.float64) * 2).requires_grad_(True)
    g, z, _ = straight_through_mask(logits.unsqueeze(0), k)
    # Any scalar function of g that touches every coordinate; the loss shape is irrelevant to the
    # claim, only which coordinates receive nonzero grad.
    weights = torch.arange(1, n + 1, dtype=torch.float64)
    (g.squeeze(0) * weights).sum().backward()

    selected = z.squeeze(0) > 0
    grad = logits.grad
    assert selected.sum().item() == k
    assert (grad[selected].abs() > 0).all(), "selected items must get gradient"
    assert (grad[~selected].abs() > 0).all(), (
        "UNSELECTED items must get gradient too -- this is the boundary-credit property; "
        "if it fails, the estimator has degenerated to the selected-gate proxy"
    )


def test_gradient_magnitudes_are_comparable_across_the_boundary():
    """
    Unselected gradient is the same order as selected, not a vanishing residue.

    Nonzero-but-1e-12 would satisfy the test above while being useless in practice, so the
    magnitudes are compared directly. The reference measurement was 1.14e-01 unselected against
    1.06e-01 selected.
    """
    torch.manual_seed(3)
    n, k = 32, 8
    logits = (torch.randn(64, n, dtype=torch.float64)).requires_grad_(True)
    g, z, _ = straight_through_mask(logits, k)
    g.sum().backward()
    grad = logits.grad.abs()
    selected = z > 0
    mean_in = grad[selected].mean().item()
    mean_out = grad[~selected].mean().item()
    assert mean_out > 0.1 * mean_in, f"unselected grad {mean_out:.3e} vs selected {mean_in:.3e}"


@pytest.mark.parametrize("k", [1, 3, 8, 16])
def test_sampled_subsets_have_exact_cardinality(k):
    """Every sampled row holds exactly ``k`` ones -- the forward's whole guarantee."""
    torch.manual_seed(4)
    logits = torch.randn(7, 11, 16, dtype=torch.float64) * 4.0
    z = sample_k_subset(logits, k)
    counts = z.sum(-1)
    assert torch.equal(counts, torch.full_like(counts, float(k)))
    assert set(z.unique().tolist()) <= {0.0, 1.0}


def test_sample_frequencies_match_marginals():
    """
    The sampler draws from the distribution the marginals describe.

    Checks the two halves of the estimator agree with each other: the empirical selection
    frequency over many draws should converge to ``mu``. Without this, a sampler that happened to
    draw from a *different* exactly-k distribution would pass every other test here while the
    forward and backward silently disagreed.
    """
    torch.manual_seed(5)
    n, k, draws = 12, 4, 8000
    logits = torch.randn(n, dtype=torch.float64) * 1.5
    mu = exact_k_marginals(logits.unsqueeze(0), k).squeeze(0)
    batched = logits.unsqueeze(0).expand(draws, n)
    freq = sample_k_subset(batched, k).mean(0)
    # Binomial standard error at 8000 draws is ~0.006; 5 sigma with a small floor for the
    # near-0/near-1 coordinates where the normal approximation is loose.
    assert (freq - mu).abs().max().item() < 0.03


def test_straight_through_forward_is_the_hard_mask():
    """``g`` equals ``z`` numerically -- the forward is genuinely discrete, not a relaxation."""
    torch.manual_seed(6)
    logits = torch.randn(5, 20, dtype=torch.float64) * 2
    g, z, mu = straight_through_mask(logits, 6)
    assert torch.allclose(g, z, atol=1e-12)
    assert not torch.allclose(mu, z), "mu should differ from z, or nothing is being estimated"


def test_multiplicative_form_is_exactly_sparse_attention():
    """
    ``g * exp(a) / sum(g * exp(a))`` **is** softmax over the sampled subset.

    This is why there is no train/inference gap in the forward, unlike the dense-forward gated
    path. Checked against an independently written masked softmax rather than against a
    rearrangement of the same expression.
    """
    torch.manual_seed(7)
    rows, n, k = 16, 24, 7
    logits = torch.randn(rows, n, dtype=torch.float64) * 2
    attn = torch.randn(rows, n, dtype=torch.float64) * 3

    g, z, _ = straight_through_mask(logits, k)
    weighted = g * torch.exp(attn)
    alpha = weighted / weighted.sum(-1, keepdim=True)

    reference = F.softmax(attn.masked_fill(z == 0, -float("inf")), dim=-1)
    assert (alpha - reference).abs().max().item() < 1e-14


def test_extreme_logits_stay_finite():
    """
    +/-80 logits produce finite marginals and a valid sample.

    sigmoid saturates well before this, so the clamp and the sentinel are both exercised. A run
    that reaches saturation is not exotic: nothing bounds the indexer's score, and a router that
    becomes confident is exactly a router that is working.
    """
    logits = torch.tensor(
        [[80.0, -80.0, 80.0, 0.0, -80.0, 80.0, 12.0, -3.0]], dtype=torch.float64
    )
    mu = exact_k_marginals(logits, 3)
    assert torch.isfinite(mu).all()
    assert torch.allclose(mu.sum(-1), torch.tensor([3.0], dtype=torch.float64), atol=1e-9)
    z = sample_k_subset(logits, 3)
    assert torch.isfinite(z).all() and z.sum().item() == 3


def test_clamp_is_required():
    """
    Removing the ``-1e-7`` clamp on ``log_sigmoid`` silently corrupts the marginals **and** NaNs
    the gradient. Reproduced here by monkeypatching the clamp away.

    :data:`LOG_P_MAX` is load-bearing, not stylistic. When a score saturates, ``log p == 0`` so
    ``log(1 - p) = -inf``, and every "not selected" DP transition is dead: the only reachable
    terminal state is "all n selected", the marginals come out ranked by *position* rather than by
    score, they sum to ``k + 1`` instead of ``k``, and ``d mu / d s`` is NaN throughout.

    Note what does **not** happen: ProbMoE's reported symptom is a ``torch.bernoulli`` range
    error, and that does not reproduce -- :func:`sample_k_subset`'s ``remaining == 0`` guard
    replaces the degenerate ratio with the sentinel first. Verified over 2000 random draws with
    saturated scores mixed in: cardinality stayed exact and nothing raised. So the *sampler*
    survives and the *marginals* break, which is the worse of the two failures because it has no
    traceback attached.
    """
    n, k = 8, 3
    saturated = torch.full((1, n), 110.0, dtype=torch.float32)
    assert (F.logsigmoid(saturated) == 0.0).all(), "fp32 sigmoid saturates by s~=104"
    assert torch.isinf(log1mexp(F.logsigmoid(saturated))).all()

    original = exact_k_module.log_sigmoid
    exact_k_module.log_sigmoid = F.logsigmoid  # drop the clamp
    try:
        bad = exact_k_module.exact_k_marginals(saturated.clone().requires_grad_(True), k)
        assert bad.sum().item() == pytest.approx(k + 1), (
            f"unclamped marginals should sum to k+1={k + 1}, got {bad.sum().item()}"
        )
        # Ranked by position, not by score -- every score here is identical, so a correct
        # computation would give the uniform k/n.
        assert not torch.allclose(bad, torch.full_like(bad, k / n))

        leaf = saturated.clone().requires_grad_(True)
        exact_k_module.exact_k_marginals(leaf, k).sum().backward()
        assert torch.isnan(leaf.grad).all(), "unclamped gradient should be NaN"
    finally:
        exact_k_module.log_sigmoid = original

    # With the clamp: uniform marginals summing to k, and a finite gradient.
    leaf = saturated.clone().requires_grad_(True)
    good = exact_k_marginals(leaf, k)
    assert good.sum().item() == pytest.approx(k, abs=1e-4)
    assert torch.allclose(good, torch.full_like(good, k / n), atol=1e-5)
    good.sum().backward()
    assert torch.isfinite(leaf.grad).all()
    assert sample_k_subset(saturated, k).sum().item() == k


def test_sampler_survives_saturation_without_the_clamp():
    """
    The sampler alone does **not** need the clamp -- documented because ProbMoE's comment says it
    does, and a future reader chasing a bernoulli error would look in the wrong place.

    The ``remaining == 0`` guard in :func:`~.exact_k_subset.sample_k_subset` forces the sentinel
    before :func:`log1mexp` can produce a NaN probability, so cardinality stays exact.
    """
    torch.manual_seed(11)
    original = exact_k_module.log_sigmoid
    exact_k_module.log_sigmoid = F.logsigmoid
    try:
        for _ in range(50):
            s = torch.randn(16, 12) * 5
            s[torch.rand_like(s) > 0.6] = 200.0
            z = exact_k_module.sample_k_subset(s, 4)
            assert torch.equal(z.sum(-1), torch.full((16,), 4.0))
    finally:
        exact_k_module.log_sigmoid = original


def test_sentinel_is_finite_not_neg_inf():
    """
    The DP sentinel must be a finite number: ``logaddexp(-inf, -inf)`` NaNs in the shifted form.

    Checked on the DP's own output rather than on the constant, so the test tracks behaviour: an
    unreachable state (more than ``i`` selected out of the first ``i``) stays at the sentinel and
    the reachable ones stay finite.
    """
    assert NEG_INF > -float("inf")
    log_p = log_sigmoid(torch.zeros(1, 6, dtype=torch.float64))
    table = log_pr_exactly_k(log_p, log1mexp(log_p), 4)
    assert torch.isfinite(table).all(), "a -inf sentinel would have produced NaN here"
    # State after 1 item cannot have 3 selected (j axis is shifted by one, so column 4).
    assert table[0, 1, 4].item() == pytest.approx(NEG_INF, abs=1e-6)


@pytest.mark.parametrize("n,k", [(16, 4), (64, 16), (33, 9)])
def test_checkpointed_matches_retained(n, k):
    """
    The checkpointed and retained paths agree on both value and gradient.

    Checkpointing here is a hand-written double-backward, not ``torch.utils.checkpoint`` (which
    cannot express a forward whose output is itself a gradient), so it is genuinely a second
    implementation and this is a real cross-check.
    """
    torch.manual_seed(8)
    base = torch.randn(6, n, dtype=torch.float64) * 2

    a = base.clone().requires_grad_(True)
    b = base.clone().requires_grad_(True)
    mu_a = exact_k_marginals(a, k, checkpoint=False)
    mu_b = exact_k_marginals(b, k, checkpoint=True)
    assert (mu_a - mu_b).abs().max().item() < 1e-13

    weights = torch.randn(6, n, dtype=torch.float64)
    (mu_a * weights).sum().backward()
    (mu_b * weights).sum().backward()
    assert (a.grad - b.grad).abs().max().item() < 1e-12


def test_checkpointed_retains_less():
    """
    The checkpointed path does not keep the DP table alive after the forward.

    Measured as **live** retained elements: ``saved_tensors_hooks`` records every tensor autograd
    saves, and a weakref then tells which of those are still alive once the call returns. Counting
    the raw save events would not discriminate -- the checkpointed forward also runs the probe's
    inner ``autograd.grad``, which saves and immediately frees a full DP -- and graph *depth* does
    not either (3 nodes vs 2 on both paths; the difference is in what they hold, not how many).

    On CUDA the same comparison at ``rows=512, n=128, k=32`` reads 78.6 MiB against 0.2 MiB, and
    over 36 layers at ``rows=2048`` it is 11.13 GiB against 0.28. See
    :class:`~.exact_k_subset._CheckpointedMarginals`.
    """
    import weakref

    n, k, rows = 64, 8, 4
    logits = torch.randn(rows, n, dtype=torch.float64, requires_grad=True)

    def live_retained(checkpoint):
        refs = []

        def pack(t):
            refs.append((weakref.ref(t), t.numel()))
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            out = exact_k_marginals(logits, k, checkpoint=checkpoint)
        alive = sum(count for ref, count in refs if ref() is not None)
        return alive, out

    retained, _keep_a = live_retained(False)
    checkpointed, _keep_b = live_retained(True)

    # Scale against the DP table itself, rows x (n+1) x (k+2), since that is the term being
    # eliminated. Measured: 38844 retained vs 1792 checkpointed, DP = 2600. The checkpointed
    # residue is a handful of (rows, n) tensors -- O(n) per row, not O(n k).
    dp_elements = rows * (n + 1) * (k + 2)
    assert checkpointed < dp_elements, (
        f"checkpointed holds {checkpointed} live elements, more than the {dp_elements}-element "
        "DP table it is supposed to have freed"
    )
    assert retained > 5 * checkpointed, f"retained {retained} vs checkpointed {checkpointed}"


def test_batched_over_leading_dims():
    """``(..., n)`` works for any leading shape, and matches the flattened computation."""
    torch.manual_seed(9)
    logits = torch.randn(2, 3, 4, 16, dtype=torch.float64)
    mu = exact_k_marginals(logits, 5)
    assert mu.shape == logits.shape
    flat = exact_k_marginals(logits.reshape(-1, 16), 5).reshape(2, 3, 4, 16)
    assert torch.equal(mu, flat)


def test_hard_mode_is_topk():
    """``hard=True`` reproduces plain top-k -- what inference does."""
    logits = torch.tensor([[0.1, 5.0, -2.0, 3.0, 1.0]], dtype=torch.float64)
    g, z, _ = straight_through_mask(logits, 2, hard=True)
    assert z.squeeze(0).nonzero().flatten().tolist() == [1, 3]
    assert torch.allclose(g, z)


def test_subset_indices_are_ascending():
    """Indices come out ascending, the convention the sparse-attention path expects."""
    torch.manual_seed(10)
    logits = torch.randn(3, 5, 20, dtype=torch.float64)
    z = sample_k_subset(logits, 6)
    idx = subset_indices(z, 6)
    assert idx.shape == (3, 5, 6)
    assert bool((idx.diff(dim=-1) > 0).all()), "indices must be strictly ascending"
    # And they really are the selected positions.
    assert torch.equal(z.gather(-1, idx), torch.ones_like(idx, dtype=z.dtype))


def test_subset_indices_rejects_wrong_cardinality():
    """A mask with the wrong count raises rather than silently under-attending."""
    z = torch.zeros(1, 8)
    z[0, :3] = 1.0
    with pytest.raises(RuntimeError, match="exactly k=4"):
        subset_indices(z, 4)


def test_bf16_input_runs_in_fp32():
    """
    A bf16 caller gets an fp32 DP.

    ``n`` sequential ``logaddexp`` steps in 8 mantissa bits do not survive; the accumulation dtype
    rule (never narrower than fp32, never narrower than the input) is what makes the training path
    safe to call with the model's own dtype.
    """
    logits = torch.randn(4, 24, dtype=torch.bfloat16)
    mu = exact_k_marginals(logits, 6)
    assert mu.dtype == torch.float32
    assert torch.allclose(mu.sum(-1), torch.full((4,), 6.0), atol=1e-4)


def test_k_equals_n_selects_everything():
    """``k == n`` gives all marginals 1 and a full mask -- the degenerate end of the range."""
    logits = torch.randn(3, 9, dtype=torch.float64)
    mu = exact_k_marginals(logits, 9)
    assert torch.allclose(mu, torch.ones_like(mu), atol=1e-11)
    assert torch.equal(sample_k_subset(logits, 9), torch.ones_like(logits))


def test_invalid_k_rejected():
    logits = torch.randn(2, 8, dtype=torch.float64)
    with pytest.raises(ValueError, match="k must be in"):
        exact_k_marginals(logits, 9)
    with pytest.raises(ValueError, match="k must be in"):
        sample_k_subset(logits, -1)
