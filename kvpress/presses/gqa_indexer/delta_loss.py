# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Delta-weighted LM loss: spend the router's gradient where routing can actually help.

The ordinary objective is ``(1/N) sum_t L_t``, which weights every position equally. That is not
what the router needs, because a high ``L_t`` has at least three causes and only one of them is the
router's fault:

1. **irreducible entropy** -- the next token was genuinely open ("he walked into the ___"). No
   support set lowers this.
2. **missing knowledge** -- the model does not know the fact, and the context does not contain it.
3. **retrieval failure** -- the answer *is* in the context but was not in the selected support.

Only (3) is routing. Reweighting by ``L_t`` itself (a power mean, ``sum L_t^p``, or an LSE softmax)
promotes (1) hardest of all, since irreducible-entropy positions sit at the top of the loss
distribution permanently and their loss does not move with the support. Their gradient is noise
with respect to the router.

The quantity that isolates (3) is the **gap against the same model run densely**:

    ``delta_t = L_t^dense - L_t^sparse``

``L^dense`` is produced by a frozen backbone with the gate removed, so it is a constant with respect
to the router. Large ``delta_t`` means "this position is one the routing is already deciding";
``delta_t ~ 0`` means dense and sparse agree, which is exactly the (1)/(2) case. So the objective is

    ``loss = sum_t w_t * L_t^sparse / sum_t w_t``,    ``w_t = clamp(delta_t, min=0) + lambda``

with ``w_t`` **detached**.

Why delta cannot be the loss itself
-----------------------------------
``d/dtheta (-delta_t) = d/dtheta (L_t^sparse - const) = d L_t^sparse / dtheta``. Subtracting a
constant does not change a gradient, so "train on ``dense - sparse``" is *bit-identical* to the
present objective and merely shifts the reported number towards zero -- while paying for a second
forward pass. The delta has to enter as a weight, not as the target.

Why ``clamp(min=0)`` and why ``+ lambda``
-----------------------------------------
* ``clamp``: ``delta_t < 0`` means the sparse run beat the dense one at that position, which happens
  and is not a signal to push on. A negative weight would actively *raise* that token's loss.
* ``+ lambda``: without it, a position the router has already brought up to dense quality gets
  ``w_t -> 0`` and stops receiving gradient at all, so nothing maintains it. ``lambda`` is the floor
  that keeps every valid position in the objective; at ``lambda -> inf`` this reduces continuously to
  the ordinary mean, which makes it the knob that interpolates between the two objectives.

Normalizing by ``sum_t w_t`` rather than by ``N`` keeps the loss on the same scale as the mean it
replaces, so ``--peak-lr`` transfers and the number stays readable against an existing run. It also
makes the objective invariant to a uniform rescaling of ``w``, i.e. only the *relative* weighting
carries.

Cost
----
One extra forward per step. It runs under ``no_grad`` with no gate hooks, so it stores no activation
graph -- the increment is compute (~50% of a step), not memory. Deliberately recomputed per step
rather than cached per corpus: ``delta_t`` is a function of the router's *current* state, so a cache
would go stale after the first optimizer step, and the alignment between a cached row and the row the
loader actually drew is the kind of silent mismatch this package has been bitten by before.

Interaction with Liger
----------------------
Per-token loss needs the logits, and ``--liger``'s fused linear+CE exists precisely so they are never
materialized (7.0 GiB at ``L=8192``). :func:`per_token_ce` therefore computes ``lm_head`` and the
cross-entropy in chunks, keeping only the ``(chunk,)`` losses: 0.3 GiB per 1024-row chunk at Qwen3's
151936-wide vocabulary, against 2.5 GiB for the whole 8K sequence at once. This is sound because
``sum_t w_t L_t`` is **linear** in ``L_t`` and so decomposes across chunks -- unlike a power mean,
which would need every ``L_t`` in hand before its outer exponent and hence a two-pass scheme.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

#: Rows the loss ignores, matching ``CrossEntropyLoss``'s convention and HF's label padding.
IGNORE_INDEX = -100

#: Rows per ``lm_head`` chunk. At Qwen3-8B's 151936 vocabulary this is 0.3 GiB of bf16 logits per
#: chunk; the whole 8K sequence at once would be 2.5 GiB, and it grows with ``L``.
DEFAULT_LOGIT_CHUNK = 1024


def shift_for_next_token(
    hidden_states: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Align hidden states with the labels they predict, flattened to ``(N, ...)``.

    Position ``t``'s hidden state predicts token ``t+1``, so the states lose their last row and the
    labels lose their first. Done here rather than relying on the model's internal shift because
    this module computes the loss itself -- and an off-by-one would produce a *plausible* loss curve
    trained against the wrong targets.
    """
    if hidden_states.dim() != 3:
        raise ValueError(f"hidden_states must be (B, L, H), got {tuple(hidden_states.shape)}")
    if labels.shape[:2] != hidden_states.shape[:2]:
        raise ValueError(
            f"labels {tuple(labels.shape)} do not match hidden_states "
            f"{tuple(hidden_states.shape[:2])}"
        )
    states = hidden_states[:, :-1].reshape(-1, hidden_states.shape[-1])
    targets = labels[:, 1:].reshape(-1)
    return states, targets


def per_token_ce(
    lm_head: nn.Module,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = DEFAULT_LOGIT_CHUNK,
) -> torch.Tensor:
    """
    Cross-entropy at every position, ``(N,)`` in fp32, computed a chunk of rows at a time.

    Parameters
    ----------
    lm_head : nn.Module
        The output projection. Applied here rather than by the model so the logits can be released
        per chunk; a Liger-patched model never exposes them at all.
    hidden_states : torch.Tensor
        ``(B, L, H)`` final-layer states, **unshifted**.
    labels : torch.Tensor
        ``(B, L)`` token ids, **unshifted**. ``IGNORE_INDEX`` marks positions to skip.
    chunk_size : int
        Rows per ``lm_head`` call. Bounds peak memory at ``chunk_size * vocab``.

    Returns
    -------
    torch.Tensor
        ``(N,)`` with ``N = B * (L - 1)``. Ignored positions hold ``0.0`` and must be excluded via
        :func:`valid_mask` rather than by testing against zero -- a real loss can be ~0 too.

    Notes
    -----
    fp32 for the accumulation, as the reference kernels do: the losses are summed over thousands of
    positions and a bf16 accumulation over ``L=8192`` has only 8 significant bits to carry it.

    The gradient flows: this is used for the *sparse* pass, where ``hidden_states`` carries the
    router's graph. Autograd sees each chunk's ``lm_head`` matmul separately, which is what bounds
    the retained logits to one chunk instead of the whole sequence.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    states, targets = shift_for_next_token(hidden_states, labels)

    losses = []
    for start in range(0, states.shape[0], chunk_size):
        rows = states[start : start + chunk_size]
        target_rows = targets[start : start + chunk_size]
        logits = lm_head(rows)
        losses.append(
            nn.functional.cross_entropy(
                logits.float(),
                target_rows,
                reduction="none",
                ignore_index=IGNORE_INDEX,
            )
        )
        del logits  # the point of chunking: do not let 151936-wide tiles accumulate
    return torch.cat(losses)


def valid_mask(labels: torch.Tensor) -> torch.Tensor:
    """
    ``(N,)`` boolean over the shifted positions: ``True`` where a real label is predicted.

    Derived from ``labels`` rather than from the losses, because ``cross_entropy`` reports ``0.0``
    for an ignored row and a genuinely confident position can also be ~0. Testing the loss would
    silently drop the model's best predictions from the objective.
    """
    return (labels[:, 1:].reshape(-1) != IGNORE_INDEX)


def delta_weights(
    dense_loss: torch.Tensor,
    sparse_loss: torch.Tensor,
    *,
    lam: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    ``w_t = clamp(dense_t - sparse_t, min=0) + lambda``, detached, ``(N,)``.

    Detached deliberately and not as an optimization: ``w`` is a *weighting*, and letting the
    gradient reach it would add a term that rewards making ``sparse_t`` large wherever the weight
    is large -- the objective would be optimizable by getting worse.

    Parameters
    ----------
    dense_loss, sparse_loss : torch.Tensor
        ``(N,)`` per-token losses from the ungated and gated passes.
    lam : float
        The weight floor. ``0`` drops already-solved positions from the objective entirely (an
        ablation, not a default); large values recover the ordinary mean.
    mask : torch.Tensor, optional
        ``(N,)`` boolean of valid positions. Invalid ones get weight ``0`` so they cannot
        contribute through either term.
    """
    if lam < 0:
        raise ValueError(f"lam must be non-negative, got {lam}")
    if dense_loss.shape != sparse_loss.shape:
        raise ValueError(
            f"dense {tuple(dense_loss.shape)} and sparse {tuple(sparse_loss.shape)} disagree"
        )
    weights = (dense_loss.detach() - sparse_loss.detach()).clamp_(min=0.0) + lam
    if mask is not None:
        weights = weights * mask.to(weights.dtype)
    return weights


def delta_weighted_loss(
    sparse_loss: torch.Tensor,
    weights: torch.Tensor,
    *,
    delta: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    ``sum_t w_t L_t / sum_t w_t``, plus diagnostics.

    Normalized by ``sum w`` rather than ``N`` so the value stays on the same scale as the mean it
    replaces: ``--peak-lr`` carries over, and the logged number is comparable to an existing run's.

    Parameters
    ----------
    sparse_loss : torch.Tensor
        ``(N,)`` per-token loss from the gated pass, carrying the router's graph.
    weights : torch.Tensor
        ``(N,)`` from :func:`delta_weights`, already detached and masked.
    delta, mask : torch.Tensor, optional
        Supplied only for the diagnostics: ``delta_positive_frac`` needs the raw gap and the
        validity mask to report a meaningful denominator.

    Returns
    -------
    (loss, stats)
        ``stats`` reports what the weighting actually did, which is the thing to watch rather than
        the loss:

        * ``weight_participation`` -- ``(sum w)^2 / (n sum w^2)``, the effective *fraction* of
          weighted positions carrying the objective. ``1.0`` means the weighting is uniform and did
          nothing, whatever the loss says; falling towards ``0`` means it concentrated. This is the
          number that separates "reweighted" from "reweighted-looking", and it is the direct
          analogue of the gate's own participation diagnostic.
        * ``delta_positive_frac`` -- fraction of valid positions where the dense run is genuinely
          ahead. **If this is near zero the weighting has nothing to work with**: every position
          falls back to ``lambda`` and the objective is the ordinary mean with a second forward pass
          paid for nothing. Watch this before the loss.
        * ``delta_mean_positive`` -- mean gap over those positions, i.e. how much headroom the
          routing actually has where it has any.
    """
    total = weights.sum()
    if not torch.isfinite(total) or total <= 0:
        raise RuntimeError(
            f"delta weights sum to {float(total)}, so the loss is undefined. With lam=0 this "
            "happens when the sparse pass matches or beats the dense one everywhere; pass a "
            "positive --delta-lambda."
        )
    loss = (weights * sparse_loss).sum() / total

    with torch.no_grad():
        active = weights > 0
        n_active = int(active.sum())
        sum_sq = float((weights * weights).sum())
        participation = (
            float(total) ** 2 / (n_active * sum_sq) if n_active and sum_sq > 0 else 0.0
        )
        stats = {
            "weight_sum": float(total),
            "weight_participation": participation,
            "n_weighted": n_active,
        }
        if delta is not None:
            valid = mask if mask is not None else torch.ones_like(delta, dtype=torch.bool)
            n_valid = int(valid.sum())
            positive = (delta > 0) & valid
            n_positive = int(positive.sum())
            stats["delta_positive_frac"] = n_positive / n_valid if n_valid else 0.0
            stats["delta_mean_positive"] = (
                float(delta[positive].mean()) if n_positive else 0.0
            )
    return loss, stats
