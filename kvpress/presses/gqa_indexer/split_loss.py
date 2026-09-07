# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
The C1/C2 (split-context) objective: train the router on *future utility*.

Split a document into a prefix ``C1`` and a suffix ``C2``. Gate only ``C1`` -- so its keys are
under eviction pressure -- read ``C2`` densely, and take the loss on ``C2`` alone::

    context:   [ C1 ................ | C2 ......... ]
    gate:      [ soft-evicted        | pinned (dense) ]
    loss:      [ ignored             | counted        ]

What this fixes, concretely
---------------------------
The plain objective gates every key and scores every position. But a position's loss is
dominated by its **immediate neighbours** -- attention is locally concentrated -- and at
inference those neighbours are retained *for free* by ``force_local`` (64 by default). So a
large share of the router's gradient goes into ranking keys whose fate was never the router's
to decide, and the decision it actually faces at inference -- "is this key still worth keeping
thousands of tokens later?" -- is supervised only weakly, by whatever long-range signal
survives in the average.

A split removes the confound by construction. Every ``C2`` query sits at least ``|C2|`` tokens
after every ``C1`` key, so **no ``C1`` key can be rescued by a local window**: whether it is
available to ``C2`` is decided by the gate and nothing else. The gradient that arrives at a
``C1`` key therefore measures exactly the quantity the router is deployed to predict.

This is the same intuition as the ``--sft-ruler`` mode (mask the loss to the tokens that need
retrieval) but without needing labelled needles: any long document supplies the supervision,
because predicting ``C2`` genuinely requires ``C1``.

Why C2 must be pinned rather than ungated
-----------------------------------------
The obvious reading of "C2 doesn't get the gate" is to drop the gate term on every key ``C2``
attends to. That silently destroys the objective: if a ``(C2 query, C1 key)`` pair carries no
gate term, then ``dL/d(score)`` for that ``C1`` key is **identically zero** and the router
receives no signal about the one thing this objective exists to teach.

So the split is a **pin**, in the precise sense of :mod:`~.gate_pin`: ``C2``'s own keys are
exempt from the gate (they read at dense weight, gate ``= log 1 = 0``), while ``C1``'s keys
stay gated *including when ``C2`` reads them*. Concretely, for a query in ``C2``:

* key in ``C1`` -> ``score - lse``, gated, gradient flows to the router ✓
* key in ``C2`` -> ``0``, dense ✓

Two consequences worth naming, both of which fall out rather than being designed in:

* The budget normalizer ``lse`` is now taken over ``C1`` only, so "a fixed multiplier shared
  across history" means a fixed budget *over the compressible region* -- the domain that is
  actually being compressed. Under the unsplit objective the same budget is diluted across the
  local window, which is never evicted.
* ``C2``-internal attention is dense and ungated, which is precisely the training-time image of
  ``force_local`` at inference. The two now agree instead of being mismatched.

Relation to stage 2 (sparse scope)
----------------------------------
``stage="sparse"`` also removes the dense fallback, by restricting the forward to the router's
top-k. The difference is the gradient: under a hard top-k an **unselected** key's gradient is
identically zero (``test_full_scope_gradients_are_independent``), so a router that currently
misses the needle gets no signal to start selecting it. This objective keeps the forward dense
over ``C1``, so every ``C1`` key -- selected or not -- receives a gradient. It is the continuous
relaxation of the same idea, and that is its advantage.

What is NOT claimed
-------------------
That this beats LongCE. They are different axes and compose: LongCE reweights *which positions*
count (``w_t`` from a short-vs-long context gap), while the split changes *which keys are under
pressure* and *where the loss is taken*. ``--longce-weights`` may be combined with
``--split-frac``; the weights are simply restricted to the ``C2`` positions.

The failure mode to watch is **signal sparsity**: with ``|C2|`` too small, few positions carry
loss while many ``C1`` keys need ranking, which is the noise problem ``--sft-ruler``'s docstring
records (~0.1% of positions carrying gradient). ``split_frac=0.5`` is the default for that
reason; treat anything below ~0.25 as an experiment rather than a setting.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from kvpress.presses.gqa_indexer.delta_loss import (
    DEFAULT_LOGIT_CHUNK,
    IGNORE_INDEX,
    per_token_ce,
)

logger = logging.getLogger(__name__)

#: Fraction of the sequence given to ``C1`` (the gated, compressible prefix). ``0.5`` splits
#: evenly, which keeps both the number of loss-carrying positions and the number of pressured
#: keys at half the sequence -- the balanced point of the trade named in the module docstring.
DEFAULT_SPLIT_FRAC = 0.5


def resolve_split(seq_len: int, split_frac: float, *, min_side: int = 64) -> int:
    """
    Absolute split index for a sequence of ``seq_len`` tokens.

    Both sides need to be non-trivial: ``C1`` must hold enough keys for a ranking to mean
    anything, and ``C2`` must hold enough positions to average the loss over. ``min_side``
    enforces that and this raises rather than clamping silently, because a caller who asked for
    a 1% split on a short sequence is expressing a misconception about the objective, not a
    preference to be honoured.
    """
    if not 0.0 < split_frac < 1.0:
        raise ValueError(f"split_frac must be in (0, 1), got {split_frac}")
    if seq_len < 2 * min_side:
        raise ValueError(
            f"seq_len={seq_len} is too short to split with min_side={min_side}: the C1/C2 "
            f"objective needs both a prefix worth ranking and a suffix worth averaging over"
        )
    split = int(round(seq_len * split_frac))
    return max(min_side, min(split, seq_len - min_side))


def split_labels(
    labels: torch.Tensor, split: int, *, gap: int = 0
) -> torch.Tensor:
    """
    ``labels`` with everything before the split (and an optional gap) set to ``IGNORE_INDEX``.

    The loss is taken on ``C2`` only. ``C1``'s own positions are excluded because their loss is
    explained by ``C1``-local context, which is exactly the confound this objective removes --
    including them would reintroduce it at full strength.

    ``gap`` additionally drops the first ``gap`` positions of ``C2``. The token right after the
    split still has genuinely local dependencies reaching back into ``C1``'s tail, so its loss
    partly measures "did the gate keep my immediate predecessor" -- the same short-range decision
    ``force_local`` makes for free at inference. A gap of one local window (64) removes that
    contamination at the cost of that many supervised positions.

    Note the off-by-one this deliberately does NOT hide: ``per_token_ce`` predicts token ``t+1``
    from position ``t``, so masking label index ``i`` removes the prediction *of* token ``i``.
    Position ``split - 1`` predicts token ``split``, which is the first ``C2`` token, and that
    prediction is made from a ``C1`` position -- it is masked here, since a query inside ``C1``
    has no long-range dependency to speak of.
    """
    if split < 0 or split > labels.shape[1]:
        raise ValueError(f"split={split} out of range for labels of length {labels.shape[1]}")
    out = labels.clone()
    stop = min(split + max(gap, 0), labels.shape[1])
    out[:, :stop] = IGNORE_INDEX
    return out


def split_context_loss(
    lm_head: nn.Module,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    split: int,
    *,
    gap: int = 0,
    weights: torch.Tensor | None = None,
    logit_chunk: int = DEFAULT_LOGIT_CHUNK,
) -> tuple[torch.Tensor, dict]:
    """
    Mean cross-entropy over the ``C2`` positions, optionally LongCE-weighted.

    Parameters
    ----------
    lm_head, hidden_states, labels, logit_chunk
        As :func:`~.delta_loss.per_token_ce`. ``hidden_states`` comes from a forward pass whose
        gate was split at the same index -- this function does not verify that, and cannot.
    split, gap
        Passed to :func:`split_labels`.
    weights
        Optional ``(B, L-1)`` or ``(N,)`` LongCE weights. Composed multiplicatively with the C2
        mask, so the two mechanisms stack rather than one overriding the other.

    Returns
    -------
    (loss, stats)
        ``stats`` reports ``c2_positions`` (how many positions actually carried gradient) and
        ``c2_frac``. Watch the former: it is the signal-sparsity failure mode from the module
        docstring, and it is the number that explains a noisy curve.
    """
    masked = split_labels(labels, split, gap=gap)
    losses = per_token_ce(lm_head, hidden_states, masked, chunk_size=logit_chunk)
    keep = (masked[:, 1:].reshape(-1) != IGNORE_INDEX).to(losses.dtype)

    n_c2 = int(keep.sum())
    if n_c2 == 0:
        raise RuntimeError(
            f"the C2 region carries no supervised position (split={split}, gap={gap}, "
            f"len={labels.shape[1]}). Every label after the split is IGNORE_INDEX, so this step "
            "would divide by zero; check split_frac and the label masking."
        )

    w = keep
    stats = {"c2_positions": n_c2, "c2_frac": n_c2 / max(keep.numel(), 1)}
    if weights is not None:
        flat = weights.reshape(-1).to(losses.device, dtype=losses.dtype)
        if flat.shape != losses.shape:
            raise ValueError(
                f"weights has {tuple(flat.shape)} entries but the loss has {tuple(losses.shape)}; "
                "an off-by-one here still broadcasts and would train the wrong positions"
            )
        w = keep * flat.detach()
        denom = w.sum()
        if denom <= 0:
            raise RuntimeError(
                "LongCE weights are zero across the whole C2 region, so the objective is "
                "undefined. A cache miss yields 1.0, not 0.0, so this points at a corrupt "
                "weight file rather than a missing one."
            )
        stats["c2_weight_mean"] = float((w.sum() / keep.sum()).detach())

    loss = (losses * w).sum() / w.sum()
    # The unweighted C2 mean, so the curve stays comparable across weightings.
    stats["c2_loss"] = float(((losses * keep).sum() / keep.sum()).detach())
    return loss, stats


def e2e_indexer_split_step(
    model: nn.Module,
    trainer,
    *,
    input_ids: torch.Tensor,
    split: int,
    gap: int = 0,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    logit_chunk: int = DEFAULT_LOGIT_CHUNK,
) -> tuple[torch.Tensor, dict]:
    """
    One step of the C1/C2 objective. Mirrors :func:`~.e2e_trainer.e2e_indexer_longce_step`.

    **The caller must set ``trainer.split = split`` before this runs**, which is what pins C2 out
    of the gate; this function asserts it rather than setting it, because the trainer is the
    object the attention hooks read and silently mutating it here would make the forward pass
    depend on argument order.

    One forward pass, like the LongCE step and unlike the delta step. There is no second
    (dense-reference) pass: the split is a property of the gate geometry, not a comparison.
    """
    from kvpress.presses.gqa_indexer.e2e_trainer import _final_hidden_states

    if trainer.split != split:
        raise RuntimeError(
            f"trainer.split is {trainer.split!r} but this step was called with split={split}. "
            "The gate's pin geometry comes from the trainer (the attention hooks read it), so a "
            "mismatch would gate the whole sequence while scoring only C2 -- which is a "
            "different, unvalidated objective."
        )
    target = input_ids if labels is None else labels
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("model exposes no output embeddings, so per-token CE cannot be formed")

    with trainer.hooks(model):
        hidden = _final_hidden_states(
            model, input_ids=input_ids, attention_mask=attention_mask
        )
        if trainer.layers_gated == 0:
            raise RuntimeError(
                "no layer ran the gated attention: the model kept its own attention "
                "implementation. This usually means the model's config is not the one "
                f"E2EIndexerTrainer pointed at {model.config._attn_implementation!r}."
            )
        loss, stats = split_context_loss(
            lm_head, hidden, target, split,
            gap=gap, weights=weights, logit_chunk=logit_chunk,
        )

    stats["split"] = split
    stats["gap"] = gap
    return loss, stats


__all__ = [
    "DEFAULT_SPLIT_FRAC",
    "e2e_indexer_split_step",
    "resolve_split",
    "split_context_loss",
    "split_labels",
]
