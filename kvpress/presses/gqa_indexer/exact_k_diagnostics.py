# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Diagnostics that answer "is the estimator *correct*", not "did the loss go down".

``HANDOFF_exact_k_subset.md`` §7 is blunt about this: accuracy alone cannot tell you whether the
gradient means anything, and it names the swap oracle as **the single most informative diagnostic**.
This module implements it, plus the two supporting measurements that interpret its result.

The swap oracle
---------------
The router's job is a discrete choice: which ``K`` chunks. So the quantity that matters is the
**marginal utility of a swap** -- for ``i`` currently selected and ``j`` not, how much would the loss
change if we exchanged them::

    dL_true(i, j) = L(S - i + j) - L(S)

That is computable exactly, by running the forward twice. The estimator's *prediction* of the same
quantity is the first-order expansion of that swap::

    dL_hat(i, j) = g_j - g_i             where g = dL/d(score)

**The sign is derivable, and I got it backwards first.** The swap moves ``z_i`` from 1 to 0 and
``z_j`` from 0 to 1, so to first order ``dL ~= g_i·(-1) + g_j·(+1) = g_j - g_i``. Checked against a
problem with a known answer: for ``L = -sum_{i in S} v_i`` the true utility is ``v_i - v_j`` and the
exact gradient is ``g = -v``, giving ``g_j - g_i = v_i - v_j`` -- an identity.

The trap is that "which way does descent move this score" and "how does the loss change" have
*opposite* signs, and it is easy to reach for the former because that is what the optimizer does.
Writing ``-(g_j - g_i)`` makes the diagnostic report the exact negation of the truth, so a perfect
estimator scores **0.0** sign accuracy and a perfectly wrong one scores 1.0. Nothing about that looks
like an error in isolation -- which is why
``test_swap_oracle_detects_a_perfect_gradient`` and its inverted twin both exist, and why they assert
1.0 and 0.0 rather than merely "better than chance".

**What a result means, and the bias that makes the naive reading wrong.** Measured on exact-K at
init: raw sign accuracy **0.25** with Spearman **+0.69**. Those look contradictory and are not. The
gradient populations are offset -- mean ``g`` on selected chunks was ``+2.8e-3`` against ``-9.2e-4``
on unselected, so ``g_j - g_i`` carries a systematic ``-3.7e-3`` shift. Meanwhile 85% of real swaps
*hurt* (the router's picks are already decent), so ``dL_true`` is mostly positive and the shifted
prediction is mostly negative: the signs disagree almost everywhere while the *ordering* is largely
right.

That offset is a property of the loss landscape, not a defect in the ranking: a constant added to
every score's gradient cannot change which chunk wins a comparison, and comparisons are all the
router ever makes. So:

* **Read the rank statistics** (Spearman, and :attr:`SwapOracleResult.centered_sign_accuracy`) as the
  measure of whether the estimator recovers marginal utility.
* **Read the raw sign accuracy together with** :attr:`SwapOracleResult.bias`. If the bias is large
  relative to the spread of the predictions, the raw number is reporting the offset.

An earlier hypothesis for the offset -- marginal saturation making ``d mu/d score`` smaller on
selected chunks -- was **tested and rejected**: the measured ratio of unselected to selected
self-derivative is 0.89, i.e. the wrong direction. Recorded because it is the intuitive explanation
and it is false.

Pearson is the weakest of the three: ``dL_true`` has a heavy tail (most swaps barely matter, a few
matter a lot) so one outlier pair moves it, which is why Spearman is reported too.

Why this is worth the two extra forwards
----------------------------------------
Every other signal in this package is consistent with a router that has learned nothing useful. The
loss falls because the frozen backbone is strong. Marginal entropy falls because *any* score
separation reduces it, including a wrong one. Only this diagnostic compares the estimator against
ground truth on the decision it actually makes.

It also scores the **existing** paths on the same axis, which is what makes it a comparison rather
than a self-report: pass a different ``score_grad_fn`` and the gated arm's additive gate goes through
the identical measurement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class SwapOracleResult:
    """
    Outcome of :func:`swap_oracle_correlation`.

    Attributes
    ----------
    sign_accuracy : float
        Fraction of boundary pairs whose swap direction the estimator gets right, judged against
        zero. 0.5 is chance. **Read this together with** :attr:`centered_sign_accuracy` and
        :attr:`bias`: a systematic offset between the selected and unselected gradient populations
        wrecks this number while leaving the ranking intact, and that offset is real (see the class
        docstring's note on bias).
    centered_sign_accuracy : float
        Sign accuracy after subtracting the **median** prediction, i.e. does the estimator order
        swaps correctly *relative to each other*. This is the number that reflects what the router
        can actually use, because a per-step constant added to every score's gradient does not
        change which chunk wins a comparison.
    bias : float
        ``median(dL_hat)``, the offset that was removed. Large relative to the spread of ``dL_hat``
        means the raw sign accuracy is measuring the offset rather than the ranking.
    pearson, spearman : float
        Correlation between predicted and true swap utility. Spearman is the more trustworthy of
        the two -- ``dL_true`` is heavy-tailed, so Pearson is outlier-sensitive.
    n_pairs : int
        Pairs actually evaluated. Small ``n`` makes all three numbers noisy; ``< 30`` is not
        evidence of anything.
    mean_abs_true : float
        Mean ``|dL_true|``. Context for the correlations: if every swap is worth ~0 then there is no
        signal to recover and a poor correlation says nothing about the estimator.
    """

    sign_accuracy: float
    centered_sign_accuracy: float
    bias: float
    pearson: float
    spearman: float
    n_pairs: int
    mean_abs_true: float

    def __str__(self) -> str:
        return (
            f"swap oracle over {self.n_pairs} pairs: sign {self.sign_accuracy:.3f} "
            f"/ centered {self.centered_sign_accuracy:.3f} (chance 0.5), "
            f"pearson {self.pearson:+.3f}, spearman {self.spearman:+.3f}, "
            f"bias {self.bias:+.2e}, mean |dL_true| {self.mean_abs_true:.2e}"
        )


def _rank(x: torch.Tensor) -> torch.Tensor:
    """Average ranks, so Spearman is correct in the presence of ties."""
    n = x.numel()
    order = x.argsort()
    ranks = torch.empty(n, dtype=torch.float64, device=x.device)
    ranks[order] = torch.arange(n, dtype=torch.float64, device=x.device)
    # Average the ranks within each tied group. Without this, ties get arbitrary distinct ranks and
    # Spearman silently reports a correlation that depends on sort order.
    sorted_x = x[order]
    start = 0
    for stop in range(1, n + 1):
        if stop == n or sorted_x[stop] != sorted_x[start]:
            if stop - start > 1:
                ranks[order[start:stop]] = ranks[order[start:stop]].mean()
            start = stop
    return ranks


def _correlate(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """``(pearson, spearman)`` between two 1-D tensors, in fp64. NaN when a side is constant."""
    a, b = a.double().flatten(), b.double().flatten()
    if a.numel() < 2:
        return float("nan"), float("nan")

    def pearson(x, y):
        xc, yc = x - x.mean(), y - y.mean()
        denom = xc.norm() * yc.norm()
        # A constant side has no correlation defined -- return NaN rather than 0, which would read
        # as "measured, and no relationship".
        if denom == 0:
            return float("nan")
        return float((xc @ yc) / denom)

    return pearson(a, b), pearson(_rank(a), _rank(b))


def swap_oracle_correlation(
    loss_fn,
    selected: torch.Tensor,
    score_grad: torch.Tensor,
    *,
    max_pairs: int = 64,
    generator: torch.Generator | None = None,
) -> SwapOracleResult:
    """
    Does the estimator's gradient recover the true utility of a discrete swap?

    Parameters
    ----------
    loss_fn : callable
        ``loss_fn(mask) -> float``, where ``mask`` is a 0/1 tensor shaped like ``selected``. Called
        once per pair plus once for the baseline, so ``max_pairs + 1`` forwards total. Must be
        deterministic given the mask -- see the note below.
    selected : torch.Tensor
        ``(..., n)`` 0/1 mask, the subset the estimator actually chose.
    score_grad : torch.Tensor
        ``(..., n)`` ``dL/d(score)`` from one backward through the estimator.
    max_pairs : int
        Boundary pairs to sample. Each costs one forward.
    generator : torch.Generator, optional
        For reproducible pair sampling.

    Returns
    -------
    SwapOracleResult

    Notes
    -----
    **``loss_fn`` must be deterministic.** The estimator's forward samples its subset, so a naive
    ``loss_fn`` that re-samples would measure sampling noise instead of the swap. Callers should
    force the mask (see ``forced_mask_loss_fn`` in the test module for the pattern).

    Pairs are drawn from the **boundary** -- one selected item, one not -- because that is where the
    router's decision is actually contested, and where a wrong sign costs something. Sampling
    uniformly over all ``(i, j)`` would mostly draw pairs whose answer is obvious and inflate the
    sign accuracy.
    """
    if selected.shape != score_grad.shape:
        raise ValueError(
            f"selected {tuple(selected.shape)} and score_grad {tuple(score_grad.shape)} must agree"
        )
    flat_sel = selected.reshape(-1, selected.shape[-1])
    flat_grad = score_grad.reshape(-1, score_grad.shape[-1])
    n_rows = flat_sel.shape[0]

    baseline = float(loss_fn(selected))
    predicted, truth = [], []

    for _ in range(max_pairs):
        row = int(torch.randint(n_rows, (1,), generator=generator))
        chosen = (flat_sel[row] > 0).nonzero().flatten()
        others = (flat_sel[row] == 0).nonzero().flatten()
        if chosen.numel() == 0 or others.numel() == 0:
            continue
        i = int(chosen[torch.randint(chosen.numel(), (1,), generator=generator)])
        j = int(others[torch.randint(others.numel(), (1,), generator=generator)])

        swapped = flat_sel.clone()
        swapped[row, i] = 0.0
        swapped[row, j] = 1.0
        delta_true = float(loss_fn(swapped.reshape(selected.shape))) - baseline

        # g_j - g_i: the first-order change in the loss when z_i goes 1->0 and z_j goes 0->1.
        # NOT the negation -- see the module docstring; that reports the exact opposite of the truth.
        delta_hat = float(flat_grad[row, j]) - float(flat_grad[row, i])

        predicted.append(delta_hat)
        truth.append(delta_true)

    if not predicted:
        raise RuntimeError(
            "no boundary pairs found: every row is either fully selected or fully unselected, so "
            "there is no swap to evaluate. Check that `selected` holds a genuine k-subset."
        )

    pred = torch.tensor(predicted, dtype=torch.float64)
    true = torch.tensor(truth, dtype=torch.float64)
    pearson, spearman = _correlate(pred, true)
    # Pairs whose true effect is exactly 0 have no direction to get right, so they are excluded from
    # the sign accuracy rather than counted as a coin flip either way.
    decisive = true != 0
    if bool(decisive.any()):
        pd, td = pred[decisive], true[decisive]
        sign_accuracy = float((torch.sign(pd) == torch.sign(td)).double().mean())
        # Median, not mean: the prediction distribution inherits dL_true's heavy tail, so the mean
        # is dragged by the same outliers the centering is meant to be robust to.
        bias = float(pd.median())
        centered = float((torch.sign(pd - bias) == torch.sign(td - td.median())).double().mean())
    else:
        sign_accuracy = centered = bias = float("nan")
    return SwapOracleResult(
        sign_accuracy=sign_accuracy,
        centered_sign_accuracy=centered,
        bias=bias,
        pearson=pearson,
        spearman=spearman,
        n_pairs=int(pred.numel()),
        mean_abs_true=float(true.abs().mean()),
    )


def router_recall_at_k(
    score: torch.Tensor, oracle: torch.Tensor, k: int, *, valid: torch.Tensor | None = None
) -> float:
    """
    Fraction of the oracle's top-``k`` that the router's top-``k`` also picks.

    ``oracle`` is any ground-truth importance -- total attention mass over each chunk is the natural
    choice, which is what the distillation arm's teacher already computes.

    Reported as a fraction of ``k``, so it is comparable across budgets. Chance level is ``k / n``,
    which is worth stating alongside any number: recall 0.5 at ``k/n = 0.5`` is nothing.
    """
    if score.shape != oracle.shape:
        raise ValueError(f"score {tuple(score.shape)} and oracle {tuple(oracle.shape)} must agree")
    n = score.shape[-1]
    k = min(k, n)
    if k == 0:
        return float("nan")
    if valid is not None:
        score = score.masked_fill(~valid, -float("inf"))
        oracle = oracle.masked_fill(~valid, -float("inf"))
    got = score.topk(k, dim=-1).indices
    want = oracle.topk(k, dim=-1).indices
    hit = (got.unsqueeze(-1) == want.unsqueeze(-2)).any(-1).double().sum(-1)
    return float((hit / k).mean())


def lm_loss_regret(loss_fn, router_mask: torch.Tensor, oracle_mask: torch.Tensor) -> float:
    """
    ``L(router's subset) - L(oracle's subset)``, in nats.

    The end-to-end version of recall: 0 means the router's choice is as good as the oracle's, and a
    large value means it is leaving loss on the table. Complements recall because the two can
    disagree -- a router can miss half the oracle's chunks and still pay almost nothing, if the ones
    it missed were interchangeable.
    """
    return float(loss_fn(router_mask)) - float(loss_fn(oracle_mask))
