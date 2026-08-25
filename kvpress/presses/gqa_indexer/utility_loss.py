# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
The LM-gradient utility, and the ranking loss that distills it into the router.

``differentiable_topk_for_sparse_attention.md`` §31. Let ``b_j`` be an additive bias on key ``j``'s
attention logit. Then

    u_j = -dL/db_j = -alpha_j * <dL/do, v_j - o>

is the **first-order marginal utility** of key ``j``: how much the loss falls per unit of attention
mass moved onto it. One backward assigns a utility to *every* key, with no key having to be selected
first -- which is exactly the dead end the exact-K arm hit, where 11-15% of oracle-best chunks never
entered the candidate pool and a chunk outside the pool appears nowhere in the graph.

Why this is a distillation arm and not an end-to-end one
--------------------------------------------------------
The forward pass is untouched -- plain dense attention, no gate, no routing. So the router is *not*
on the forward path and ``dL/dtheta_router`` from the LM loss is identically **None**, not merely
small. Verified: after ``L_LM.backward()`` the router's ``.grad`` is ``None``; only the ranking loss
produces a gradient (norm 8.31 in the same check). ``loss = loss_rank`` is therefore the whole
objective, and this arm belongs in the same class as
:mod:`~kvpress.presses.gqa_indexer.fused_trainer` rather than beside the gated one.

What it buys over attention-KL distillation is the **teacher**, and the gap is large. Against the
true single-key drop effect, measured on real text:

=================================  ==================
teacher                            Spearman vs truth
=================================  ==================
``alpha`` (what fused_loss uses)   **+0.037**
``u``                              **+0.991**
=================================  ==================

``alpha`` is nearly *uninformative*: a key can hold a lot of attention and still be worthless,
because ``v_j`` may already sit at ``o`` -- and ``u``'s ``v_j - o`` factor is precisely that
correction. This is the mechanism behind SAS's 96.8% attention mass at 79.5% accuracy.

The measured limitation, which this module does not fix
------------------------------------------------------
``u``'s ranking is dominated by a factor the router **cannot observe**. It factors as ``alpha_j``
(a function of ``q . k`` -- reachable) times ``<dL/do, v_j - o>`` (a function of ``v_j`` and of the
loss direction -- a ``q . k`` scorer sees neither). Measured on Qwen3-8B:

* ``spearman(u, alpha)`` = **+0.11 to +0.32** (4-layer truncation), **+0.025** (8-layer)
* ``spearman(u, value term)`` = **+0.752**
* best construction using ``v`` magnitude as well = **+0.24** -- so giving the router values
  does *not* rescue it.

So there is a **ceiling** on :func:`pairwise_rank_loss` well below 1, and it is a property of the
hypothesis class rather than of the loss. :func:`score_utility_correlation` reports progress against
that ceiling every logged step, which is the number to watch: if it plateaus near the probe's value
the objective has converged to what this router can represent, and the next move is architectural.

Recorded in the session memory as ``lm-gradient-utility-is-not-router-reachable``. Implemented anyway
because the alternative -- reasoning about it further -- has been less informative than the two
measurements that overturned earlier conclusions in this same investigation.

One more caveat on the target, worth stating because it looks like a result
--------------------------------------------------------------------------
``u`` contains ``g = dL/do``, which is computed from the **label**. Selecting top-K by ``u`` beats
*dense attention itself* -- measured 15.3 against 18.66 row loss at K=32 of 511 keys. That is not a
better attention operator; it is the target leaking through. It is legitimate for a teacher (a
teacher is allowed privileged information) but it means ``u``'s absolute quality is not an
achievable bound, and any part of ``u``'s ranking that exists only because it knows the answer is
unlearnable in principle.
"""

from __future__ import annotations

import torch

#: Utility assigned to a key that is not visible to a query, or to a padded pair slot. Must sort
#: below every real utility so :func:`sample_boundary_pairs` never draws it, and must be finite so a
#: weight built from it cannot produce NaN.
INVALID_UTILITY = -1e30


def lm_gradient_utility(
    alpha: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
) -> torch.Tensor:
    """
    ``u_j = -alpha_j * <dL/do, v_j - o>``, the first-order marginal utility of every key.

    Parameters
    ----------
    alpha : torch.Tensor
        ``(B, H, Sq, Sk)`` attention probabilities. Rows must sum to 1 over visible keys.
    value : torch.Tensor
        ``(B, H, Sk, Dv)``, already repeated to the query head count if the model is GQA.
    out : torch.Tensor
        ``(B, H, Sq, Dv)``, the attention output ``alpha @ value``.
    grad_out : torch.Tensor
        ``(B, H, Sq, Dv)``, ``dL/do`` at that same output.

    Returns
    -------
    torch.Tensor
        ``(B, H, Sq, Sk)`` utilities. **Higher is better**: ``u_j > 0`` means moving mass onto ``j``
        lowers the loss. Sign errors here are silent -- the loss still descends, it just trains the
        router to rank the *worst* keys first -- so :func:`sample_boundary_pairs` is written against
        this convention and ``test_utility_sign_is_loss_decreasing`` pins it against a finite
        difference rather than against the algebra.

    Notes
    -----
    The ``- o`` term is the whole content of the correction and is what ``alpha`` alone misses: a key
    whose value already equals the row's output moves the output nowhere, so its utility is zero no
    matter how much attention it holds. It also makes ``sum_j u_j = 0`` exactly, since
    ``sum_j alpha_j (v_j - o) = o - o``, which is the identity
    ``test_utilities_sum_to_zero_over_a_row`` checks: a softmax cannot add mass without taking it
    from somewhere, so utility is inherently *relative* and only differences are meaningful. That is
    why the loss below is pairwise and why the diagnostic is a rank correlation.

    Computed in the caller's dtype. The trainer promotes to fp32 first -- in bf16 the
    ``<g, v_j> - <g, o>`` difference is between two numbers of similar magnitude, and its sign is the
    signal.
    """
    if alpha.shape[:-1] != out.shape[:-1] or alpha.shape[-1] != value.shape[-2]:
        raise ValueError(
            f"shape mismatch: alpha {tuple(alpha.shape)} against value {tuple(value.shape)} and "
            f"out {tuple(out.shape)}. Expected alpha (B, H, Sq, Sk), value (B, H, Sk, Dv), "
            "out (B, H, Sq, Dv), with value already repeat_interleave'd to H if the model is GQA."
        )
    # <g, v_j> for every key, minus the row-constant <g, o>. Written as one matmul plus a reduction
    # rather than materializing (v_j - o), which would be (B, H, Sq, Sk, Dv) -- 4 orders of magnitude
    # larger and the reason the naive form is unaffordable at length.
    projection = torch.matmul(grad_out, value.transpose(-1, -2))
    projection = projection - (grad_out * out).sum(-1, keepdim=True)
    return -(alpha * projection)


def sample_boundary_pairs(
    scores: torch.Tensor,
    utility: torch.Tensor,
    *,
    n_pairs: int,
    band: int = 32,
    budget: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample ``(i, j)`` key pairs concentrated where mis-ranking changes the selected set.

    Returns ``(idx_win, idx_lose)``, each ``(..., n_pairs)`` indices into the key axis, with
    ``utility[idx_win] >= utility[idx_lose]`` by construction.

    Why sampling at all, and why not uniform
    ----------------------------------------
    A row of 8192 keys has 34M pairs, so the full pairwise loss is out of reach. But uniform sampling
    spends nearly all of its budget on pairs that **cannot matter**: top-k selection depends only on
    the order *across the K-th boundary*, so a pair at ranks 3 and 7000 is already ordered correctly
    by any usable router and its gradient is wasted. §23.3. The pairs that decide the selected set
    are the ones whose *predicted* ranks straddle ``K``.

    So the band is drawn around the boundary of the **router's own current ranking**, not the
    teacher's: a pair the router already places far from ``K`` cannot flip the selection this step,
    whatever the teacher thinks of it. This makes the sampler self-correcting -- as the router
    improves, the band tracks where it is still uncertain.

    Parameters
    ----------
    scores : torch.Tensor
        ``(..., Sk)`` router scores. Only their *ranking* is read.
    utility : torch.Tensor
        ``(..., Sk)`` teacher utilities, with :data:`INVALID_UTILITY` on invisible keys.
    n_pairs : int
        Pairs per row.
    band : int
        Half-width, in ranks, of the window around ``budget`` that ``i`` and ``j`` are drawn from.
        Small (~8-64) is the point; a band as wide as the row degenerates to uniform sampling.
    budget : int, optional
        The rank the boundary sits at. ``None`` uses the midpoint of each row's valid keys, which is
        the right default when the eval budget is a *ratio* rather than a count.
    generator : torch.Generator, optional
        For reproducibility in tests.

    Notes
    -----
    Pairs are ordered by the **teacher**, so the loss always knows which of the two *should* win --
    ``idx_win`` is the higher-utility one. A pair whose two utilities are equal (both invalid, in a
    row with fewer valid keys than the band) contributes weight 0 and so drops out on its own; no
    masking is needed downstream.
    """
    if n_pairs < 1:
        raise ValueError(f"n_pairs must be >= 1, got {n_pairs}")
    if band < 1:
        raise ValueError(f"band must be >= 1, got {band}")

    n_keys = scores.shape[-1]
    device = scores.device
    valid = utility > INVALID_UTILITY / 2
    n_valid = valid.sum(-1, keepdim=True)  # (..., 1)

    # Rank keys by the ROUTER, best first. Invalid keys are pushed to the back so they only appear in
    # the band for rows too short to fill it, where they cost weight 0 anyway.
    ranked = torch.where(valid, scores, torch.full_like(scores, -float("inf")))
    order = ranked.argsort(-1, descending=True)  # (..., Sk) -> key index at each rank

    if budget is None:
        centre = n_valid // 2
    else:
        centre = torch.minimum(
            torch.full_like(n_valid, int(budget)), n_valid
        )
    low = (centre - band).clamp(min=0)
    # The window must stay inside the valid prefix, or the band would be dominated by the invisible
    # keys that were sorted to the back -- silently turning the sampler into a no-op on short rows.
    high = torch.minimum(centre + band, n_valid).clamp(min=1)
    width = (high - low).clamp(min=1)

    shape = scores.shape[:-1] + (n_pairs,)
    draw = lambda: low + (  # noqa: E731
        torch.rand(shape, device=device, generator=generator) * width
    ).long().clamp(max=n_keys - 1)
    rank_a, rank_b = draw(), draw()

    idx_a = order.gather(-1, rank_a)
    idx_b = order.gather(-1, rank_b)
    u_a = utility.gather(-1, idx_a)
    u_b = utility.gather(-1, idx_b)

    # Orient by the teacher: idx_win is whichever the teacher prefers.
    a_wins = u_a >= u_b
    return (
        torch.where(a_wins, idx_a, idx_b),
        torch.where(a_wins, idx_b, idx_a),
    )


def pairwise_rank_loss(
    scores: torch.Tensor,
    utility: torch.Tensor,
    idx_win: torch.Tensor,
    idx_lose: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Utility-weighted pairwise logistic ranking loss.

        L = mean over pairs of  w_ij * softplus(s_j - s_i)     where u_i >= u_j
        w_ij = |u_i - u_j|,  rescaled per row to mean 1 when ``normalize``

    Parameters
    ----------
    scores : torch.Tensor
        ``(..., Sk)`` router scores, **with gradients** -- this is the only tensor the loss trains
        through.
    utility : torch.Tensor
        ``(..., Sk)`` teacher utilities. Detached by the caller; a graph here would push gradient
        back into the frozen backbone's attention.
    idx_win, idx_lose : torch.Tensor
        ``(..., n_pairs)`` from :func:`sample_boundary_pairs`.
    normalize : bool
        Rescale each row's weights to mean 1. **Leave True** -- see below; ``False`` exists to
        reproduce the un-normalized behaviour, which does not train.

    Returns
    -------
    torch.Tensor
        Scalar loss.

    Notes
    -----
    **Why a ranking loss and not a regression onto ``u``.** Top-k reads only the *order* of the
    scores, so ``s`` and ``a*s + b`` select identically for any ``a > 0`` (§24). Regressing onto
    ``u``'s values would spend the router's capacity fitting a scale and an offset that the operator
    discards.

    **Why weighted by ``|u_i - u_j|``.** Two keys of nearly equal utility can be swapped at nearly no
    cost, so forcing the router to resolve them wastes capacity on noise -- and ``u`` is itself a noisy
    first-order estimate, so those pairs are where its own sign is least reliable. The weight makes the
    loss track the *regret* of a mis-ranking rather than its count, and removes any need to filter
    near-ties: their weight goes to 0 continuously.

    **Why the weights must be normalized per row, which is not cosmetic.** ``u`` is proportional to
    ``alpha_j`` (``~1/Sq``) times ``dL/do`` (which carries the LM loss's ``1/(B*Sq)`` mean), so
    ``|u| ~ 1/Sq**2``. Measured on a real model: mean ``|u|`` falls almost exactly 4x per doubling of
    the sequence -- 5.2e-7 at 256 tokens, 8.3e-9 at 2048, and ~3.5e-10 at 8192 on Qwen3-8B. Two things
    break at that magnitude, and neither is visible in the loss curve:

    * **AdamW stops being scale-invariant.** Its denominator is ``sqrt(v_hat) + eps`` with
      ``eps = 1e-8``, so once the gradient falls below ``eps`` the update degenerates to being
      *proportional* to the gradient again. Measured realized step size against the scale-invariant
      ideal: 100% at gradient 1e-3, 96.6% at 1e-6, **42.9% at 1e-8, 8.8% at 1e-9, 1.0% at 1e-10**.
    * **``grad_clip`` is an absolute threshold.** A run whose gradient norm is 1e-9 is never clipped
      while one at 1e-3 is, so the same ``--grad-clip 1.0`` means different things at different
      lengths.

    Together those make the *effective learning rate a function of the curriculum stage* -- a 16x
    change between 8K and 32K, silently. Normalizing is the right fix rather than raising the LR to
    compensate, because only *relative* weights within a row carry information: the loss is asking
    "which of these two keys matters more", and multiplying every weight in a row by a constant does
    not change that question. It is exactly the same argument that makes this a ranking loss rather
    than a regression, applied to the weights instead of the scores.

    ``softplus`` rather than a hinge: it keeps a gradient on pairs that are already correctly ordered
    but only just, which is where the boundary sampler is deliberately concentrating the draws.
    """
    s_win = scores.gather(-1, idx_win)
    s_lose = scores.gather(-1, idx_lose)
    weight = (utility.gather(-1, idx_win) - utility.gather(-1, idx_lose)).abs()
    if normalize:
        # Per row, over the drawn pairs. clamp_min keeps a row whose pairs are all ties (a row shorter
        # than the band, where every drawn utility is INVALID_UTILITY) from dividing by zero; such a
        # row's weights are 0, so it contributes nothing either way.
        weight = weight / weight.mean(-1, keepdim=True).clamp_min(1e-30)
    # Pairs drawn from a row's invalid tail have equal (invalid) utilities, so weight 0 -- they
    # neither contribute loss nor gradient, and no explicit mask is needed.
    return (weight * torch.nn.functional.softplus(s_lose - s_win)).mean()


def _rank(x: torch.Tensor) -> torch.Tensor:
    """Ranks along the last axis, as floats. Ties broken arbitrarily but consistently."""
    return x.argsort(-1).argsort(-1).to(torch.float32)


@torch.no_grad()
def score_utility_correlation(
    scores: torch.Tensor, utility: torch.Tensor, *, min_valid: int = 8
) -> float:
    """
    Spearman correlation between the router's score and ``u``, over visible keys, per row.

    **The diagnostic for this arm.** The loss value alone cannot say whether the router is learning,
    because it is a weighted average whose weights (``|u_i - u_j|``) change every step with
    ``dL/do``'s magnitude -- the loss can fall simply because the batch got easier. This is measured
    against a fixed quantity.

    It also reads directly against the measured ceiling. Probes on Qwen3-8B put
    ``spearman(u, alpha)`` at +0.03 to +0.32, and ``alpha`` is what a ``q . k`` scorer can represent,
    so a run whose correlation plateaus there has converged to the limit of the hypothesis class
    rather than of the optimizer. That distinction decides whether to keep tuning the loss or to
    change the architecture, and no other number in the metrics separates the two.

    **Spearman, not Pearson**, and computed per row then averaged: only the ranking is used
    downstream, and ``u``'s scale varies per row with ``||dL/do||`` so a pooled Pearson would be
    dominated by whichever rows happen to have large gradients.
    """
    valid = utility > INVALID_UTILITY / 2
    flat_scores = scores.reshape(-1, scores.shape[-1])
    flat_utility = utility.reshape(-1, utility.shape[-1])
    flat_valid = valid.reshape(-1, valid.shape[-1])

    total, count = 0.0, 0
    for row in range(flat_scores.shape[0]):
        mask = flat_valid[row]
        if int(mask.sum()) < min_valid:
            continue
        a = _rank(flat_scores[row][mask].unsqueeze(0))
        b = _rank(flat_utility[row][mask].unsqueeze(0))
        a = a - a.mean(-1, keepdim=True)
        b = b - b.mean(-1, keepdim=True)
        denom = (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)
        total += float((a * b).sum(-1) / denom)
        count += 1
    return total / count if count else float("nan")


@torch.no_grad()
def utility_recall_at_k(scores: torch.Tensor, utility: torch.Tensor, k: int) -> float:
    """
    Fraction of the top-``k`` keys by ``u`` that the router's own top-``k`` also keeps.

    Closer to what the eval does than :func:`score_utility_correlation` -- inference takes a top-k,
    so a router can have mediocre global rank correlation and still retain everything that matters,
    or vice versa. Reported alongside rather than instead: recall alone cannot distinguish "ranks the
    survivors correctly" from "ranks nothing but happens to include them".
    """
    valid = utility > INVALID_UTILITY / 2
    take = min(k, int(valid.sum(-1).min()))
    if take < 1:
        return float("nan")
    masked = torch.where(valid, scores.float(), torch.full_like(scores, -float("inf"), dtype=torch.float32))
    router_top = masked.topk(take, dim=-1).indices
    teacher_top = utility.float().topk(take, dim=-1).indices
    hits = (router_top.unsqueeze(-1) == teacher_top.unsqueeze(-2)).any(-1).float()
    return float(hits.mean())
