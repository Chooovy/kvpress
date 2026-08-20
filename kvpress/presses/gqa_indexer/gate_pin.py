# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pinned keys: what stops the gate from learning to do nothing.

The gate ``softmax(scale * q.k + lam * s)`` has a hole. Adding the *same* number to every key
of a row cancels in the softmax, so a gate that is **flat along the key axis** is a no-op --
the model falls back to the frozen dense backbone, which is already strong. The LM loss is
then satisfied without the router having learned any ranking at all. Reaching that point costs
the router nothing: ``s = 0`` does it.

Every failure row of SAS's Table 1 is this one hole, reached by a different route -- verified
in ``test_flat_gate_variants_all_reach_the_no_op``:

============================  ===========================  ==============  ==========
gate                          how it flattens              distance        SAS 1-epoch
============================  ===========================  ==============  ==========
``s`` (raw)                   ``s = 0``                    0.0             18.8
``log sigmoid(s)``            ``s -> +inf`` (saturates)    2e-16           17.0
``log softmax(s)``, no pin    ``s = const``                2e-16           --
``log softmax(s)`` + pin      **unreachable**              1.2             **54.4**
============================  ===========================  ==============  ==========

(dense baseline: 56.1)

Two conditions are needed together, and either one alone leaves the hole open:

1. **Normalize** over the key axis (``log softmax``), which fixes the *total* multiplier the
   gated keys may spend at 1.
2. **Exempt some keys** from that normalizer, pinning their gate to ``1`` (log-space ``0``).

With both, a flat gate is arithmetically impossible: the pinned keys sit at multiplier 1, and
matching them on all ``N`` gated keys would need the gated multipliers to sum to ``N``, not 1.
The router cannot make the gate vanish -- it can only choose *which* keys receive the fixed
budget. That choice is the ranking, and it is now the only thing the router can express.

Normalizing without pinning is inert, not merely weaker: ``logsumexp(s)`` is one constant per
row, so it cancels along with everything else and ``log softmax(s)`` is *exactly*
interchangeable with raw ``s`` (forward and ``d/ds``, to 1e-15). The pin is what makes the
normalizer load-bearing.

Query-independent vs query-dependent pins
-----------------------------------------
This distinction decides whether a kernel is needed, and it is the only reason ``self`` and
``sink`` are implemented by different code paths here.

The gate can be folded into the ``QK`` dot product -- the trick that lets
:mod:`~kvpress.presses.gqa_indexer.gated_attention` run on plain SDPA -- only if a *static*
key matrix can express it. Folding requires zeroing the indexer key on pinned positions:

* ``sink`` pins the first ``n_sink`` keys, the same set for every query. The indicator is a
  fixed per-key vector, so it folds: one extra concatenated dimension, one SDPA call. Verified
  to 1.1e-15.
* ``self`` pins each query's own diagonal key -- a *different* key per query. No static ``K``
  can zero a per-query position, and the naive attempt is wrong by 2.5 (not a rounding error).

So ``self`` takes a two-branch route instead: history-only attention (which *does* fold) plus
the single self key, merged by their log-sum-exps. Exact, verified to 6.7e-16, at the cost of
one extra attention call.

Both modes need the history ``logsumexp``, an extra streaming pass over ``qi . ki``. It
recomputes its score tiles in the backward pass rather than retaining them. The common
plain-causal sink path also rebuilds its mask per tile from ``n_sink`` and ``query_offset``, so
its retention is ``O(L)`` per layer -- see :class:`_HistoryLSE`. Inside the history branch the
``-LSE`` term cancels as a per-row constant, so it does not need to be applied there -- it only
sets the weight *between* the two branches.

Empty history
-------------
A causal row whose only visible key is its own diagonal has **no** history at all -- true for
the first token of every sequence under ``self`` pinning. Its ``logsumexp`` is ``-inf``, so
``s - LSE`` would be ``+inf`` and the softmax would return NaN, which then spreads through the
whole model. Such rows are given an inert gate instead (see :func:`history_lse`); they have
exactly one key to attend to, so no ranking is being suppressed.
"""

from __future__ import annotations

import logging

import torch

from kvpress.presses.gqa_indexer.fused_loss import accumulation_dtype

logger = logging.getLogger(__name__)

#: ``none`` leaves the no-op reachable and exists as the ablation baseline -- it is the
#: behaviour before pinning was added. The other three close the hole.
PIN_MODES = ("none", "sink", "self", "self+sink")


def check_pin_mode(pin_mode: str) -> None:
    """Reject an unknown pin mode, naming the alternatives."""
    if pin_mode not in PIN_MODES:
        raise ValueError(f"pin_mode must be one of {PIN_MODES}, got {pin_mode!r}")


def pins_self(pin_mode: str) -> bool:
    """Whether ``pin_mode`` exempts each query's own diagonal key."""
    return pin_mode in ("self", "self+sink")


def pins_sink(pin_mode: str) -> bool:
    """Whether ``pin_mode`` exempts the leading keys."""
    return pin_mode in ("sink", "self+sink")


def is_query_dependent(pin_mode: str) -> bool:
    """
    Whether the pinned set differs per query row -- i.e. whether the concat fold is unavailable.

    ``sink`` is query-independent and folds. ``self`` is not: the pinned column moves with the
    row, and no shared key matrix can represent that.
    """
    return pins_self(pin_mode)


def pinned_mask(
    pin_mode: str,
    q_len: int,
    k_len: int,
    device: torch.device,
    *,
    n_sink: int = 0,
    query_offset: int | None = None,
) -> torch.Tensor | None:
    """
    Which ``(query, key)`` pairs are exempt from the gate, ``(q_len, k_len)`` bool.

    ``None`` for ``pin_mode="none"``, so callers can skip the whole pinning path rather than
    carrying an all-False tensor through it.

    ``query_offset`` is the absolute position of the first query (default ``k_len - q_len``,
    bottom-right alignment), matching :func:`~.indexer.build_indexer_mask` and
    :func:`~.gated_attention.causal_mask_bottom_right`. A ``self`` pin is meaningless without
    it: at ``Sq < Sk`` the diagonal is not at column ``i``.
    """
    check_pin_mode(pin_mode)
    if pin_mode == "none":
        return None
    if query_offset is None:
        query_offset = k_len - q_len

    pinned = torch.zeros((q_len, k_len), dtype=torch.bool, device=device)
    if pins_sink(pin_mode) and n_sink > 0:
        pinned[:, : min(n_sink, k_len)] = True
    if pins_self(pin_mode):
        q_pos = torch.arange(q_len, device=device).unsqueeze(-1) + query_offset
        k_pos = torch.arange(k_len, device=device).unsqueeze(0)
        pinned |= k_pos == q_pos
    return pinned


def history_mask(
    pinned: torch.Tensor,
    causal_keep: torch.Tensor | None,
    q_len: int,
    k_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    ``(Sq, Sk)`` bool: visible and not pinned. Plain causal when ``causal_keep`` is None.

    Public because the gate's own diagnostics need the SAME notion of "history" the normalizer
    uses -- see ``E2EIndexerTrainer._gate_participation``, which divides by this mask's row sums.
    A second definition there could drift from this one and would silently rescale the metric.
    """
    if causal_keep is None:
        q_pos = torch.arange(q_len, device=device).unsqueeze(-1) + (k_len - q_len)
        k_pos = torch.arange(k_len, device=device).unsqueeze(0)
        causal_keep = k_pos <= q_pos
    return causal_keep & ~pinned


class _HistoryLSE(torch.autograd.Function):
    """
    Streaming history ``logsumexp`` that **recomputes** its score tiles in the backward pass.

    The obvious implementation -- a Python loop of differentiable tile ops -- bounds the forward
    peak but not the memory that matters. Autograd retains every tile's intermediates until
    backward, so total retention is ``O(Sq * Sk)`` per layer *and gets worse as the tile shrinks*
    (more tiles, more saved intermediates). Measured at ``Sq=Sk=512, h=8``: 28.8 MiB retained
    against an 8.0 MiB score matrix -- 3.6x -- which at Qwen3-8B, 36 layers, ``L=8192``
    extrapolates to ~259 GiB and OOMs on the first step.

    A custom Function fixes it because the gradient has a closed form. With
    ``lse = log sum_k exp(s_k)`` over the history keys,

        d(lse) / d(s_k) = softmax(s)_k = exp(s_k - lse)

    so given ``lse`` the per-tile weights can be rebuilt from ``q_idx``/``k_idx`` alone. On the
    plain-causal sink path, visibility is rebuilt from two integers as well, so only
    ``(q_idx, k_idx, gate_scale, lse)`` are saved -- ``O(L)`` -- and the score tiles are formed
    twice (once per pass) instead of being kept. An explicit arbitrary mask remains an input and
    is retained unchanged so its exact pattern is preserved.

    That trade is the right one here: the extra pass is one ``Di``-wide GEMM over the same tiles
    the forward already walked, against a retention that made the configuration impossible to
    run at all.
    """

    @staticmethod
    def forward(
        ctx, q_idx, k_idx, gate_scale, history, key_tile, acc, n_sink, query_offset
    ):
        bsz, n_heads, q_len, _ = q_idx.shape
        k_len = k_idx.shape[1]
        with torch.no_grad():
            run_max = torch.full(
                (bsz, n_heads, q_len), -float("inf"), device=q_idx.device, dtype=acc
            )
            run_sum = torch.zeros_like(run_max)
            q_pos = torch.arange(q_len, device=q_idx.device).unsqueeze(-1) + query_offset
            for start in range(0, k_len, key_tile):
                stop = min(start + key_tile, k_len)
                logits = torch.einsum(
                    "bhqd,bkd->bhqk", q_idx.to(acc), k_idx[:, start:stop].to(acc)
                ) * gate_scale.to(acc)
                if history is None:
                    k_pos = torch.arange(start, stop, device=q_idx.device).unsqueeze(0)
                    tile_history = (k_pos >= n_sink) & (k_pos <= q_pos)
                else:
                    tile_history = history[:, start:stop]
                logits = logits.masked_fill(~tile_history, -float("inf"))
                new_max = torch.maximum(run_max, logits.amax(dim=-1))
                # A fully masked tile leaves new_max at -inf, and exp(-inf - -inf) is NaN, so
                # the rescale is taken as 0 there. Same guard as teacher_lse_from_qk.
                rescale = torch.where(
                    torch.isfinite(run_max),
                    torch.exp(run_max - new_max),
                    torch.zeros_like(run_max),
                )
                safe_max = torch.where(
                    torch.isfinite(new_max), new_max, torch.zeros_like(new_max)
                )
                run_sum = run_sum * rescale + torch.exp(
                    logits - safe_max.unsqueeze(-1)
                ).sum(dim=-1)
                run_max = new_max

            empty = ~torch.isfinite(run_max)
            lse = torch.where(
                empty, torch.zeros_like(run_max), run_max + torch.log(run_sum.clamp_min(1e-30))
            )

        ctx.save_for_backward(q_idx, k_idx, gate_scale, lse)
        ctx.history, ctx.key_tile, ctx.acc, ctx.empty = history, key_tile, acc, empty
        ctx.n_sink, ctx.query_offset = n_sink, query_offset
        return lse

    @staticmethod
    def backward(ctx, grad_lse):
        q_idx, k_idx, gate_scale, lse = ctx.saved_tensors
        acc, history, key_tile = ctx.acc, ctx.history, ctx.key_tile
        k_len = k_idx.shape[1]
        scale = gate_scale.to(acc)

        d_q = torch.zeros_like(q_idx, dtype=acc)
        d_k = torch.zeros_like(k_idx, dtype=acc)
        d_scale = torch.zeros((), device=q_idx.device, dtype=acc)

        # An empty-history row returned a constant 0, so it has no gradient path; zeroing the
        # incoming cotangent there keeps its (all -inf) weights from producing NaN below.
        grad = (grad_lse.to(acc) * ~ctx.empty).unsqueeze(-1)
        q_pos = (
            torch.arange(q_idx.shape[2], device=q_idx.device).unsqueeze(-1) + ctx.query_offset
        )

        for start in range(0, k_len, key_tile):
            stop = min(start + key_tile, k_len)
            k_tile = k_idx[:, start:stop].to(acc)
            raw = torch.einsum("bhqd,bkd->bhqk", q_idx.to(acc), k_tile)
            if history is None:
                k_pos = torch.arange(start, stop, device=q_idx.device).unsqueeze(0)
                tile_history = (k_pos >= ctx.n_sink) & (k_pos <= q_pos)
            else:
                tile_history = history[:, start:stop]
            logits = (raw * scale).masked_fill(~tile_history, -float("inf"))
            # softmax over the FULL history axis, recovered from the saved lse -- which is why
            # the tiles do not need to be retained.
            weights = torch.nan_to_num(torch.exp(logits - lse.unsqueeze(-1)), 0.0) * grad
            d_q += torch.einsum("bhqk,bkd->bhqd", weights, k_tile) * scale
            d_k[:, start:stop] += torch.einsum("bhqk,bhqd->bkd", weights, q_idx.to(acc)) * scale
            d_scale += (weights * raw).sum()

        return (
            d_q.to(q_idx.dtype),
            d_k.to(k_idx.dtype),
            d_scale.to(gate_scale.dtype) if ctx.needs_input_grad[2] else None,
            None,
            None,
            None,
            None,
            None,
        )


def history_lse(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    *,
    gate_scale: torch.Tensor | float,
    pinned: torch.Tensor | None,
    causal_keep: torch.Tensor | None = None,
    key_tile: int = 1024,
    return_history_count: bool = False,
    n_sink: int | None = None,
    query_offset: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    ``logsumexp`` of the gate score over each row's **history**, ``(B, h, Sq)``.

    History means: visible (causal, and unmasked) but *not* pinned. This is the normalizer that
    turns the gate into a fixed budget -- see the module docstring.

    Streamed over key tiles and **recomputed in the backward pass**. With ``pinned=None`` and
    ``n_sink`` set, the plain-causal sink geometry is also recomputed per tile, so retention is
    ``O(L)`` rather than ``O(Sq * Sk)``. See :class:`_HistoryLSE` for why the naive
    differentiable loop is not enough and what the closed-form gradient buys. Differentiable, unlike
    :func:`~.fused_loss.teacher_lse_from_qk` which serves a frozen teacher under ``no_grad`` --
    the gradient path through this term is part of what trains the router.

    Rows with **no** history get ``0`` rather than ``-inf``. ``-inf`` would make the gate
    ``s - (-inf) = +inf`` and the attention NaN; ``0`` leaves the gate inert for that row, which
    is right because such a row has a single visible key and nothing to rank. Returning
    ``-inf`` and expecting callers to notice would be the silent-NaN trap this codebase avoids
    elsewhere.

    Parameters
    ----------
    q_idx : torch.Tensor
        Indexer queries ``(B, h, Sq, Di)``.
    k_idx : torch.Tensor
        Shared indexer key ``(B, Sk, Di)``.
    gate_scale : torch.Tensor or float
        Multiplier applied to the score before the logsumexp; must match what the attention
        applies, or the budget normalizes the wrong quantity.
    pinned : torch.Tensor, optional
        ``(Sq, Sk)`` bool from :func:`pinned_mask`; pinned pairs are excluded. ``None`` selects
        the compact plain-causal sink path described by ``n_sink``.
    causal_keep : torch.Tensor, optional
        ``(Sq, Sk)`` bool of visible pairs. ``None`` derives plain causal visibility from the
        shapes (bottom-right aligned).
    key_tile : int
        Keys per streaming step. A pure memory/speed knob: the result is tile-invariant, and
        retention no longer grows as the tile shrinks.
    return_history_count : bool
        Also return each query row's number of visible, non-pinned history keys, computed from
        the exact same mask as the logsumexp.
    n_sink : int, optional
        Number of leading pinned keys for the compact plain-causal path. This avoids retaining
        an ``(Sq, Sk)`` history mask in every layer.
    query_offset : int, optional
        Absolute position of query row 0; defaults to bottom-right alignment.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        The ``(B, h, Sq)`` logsumexp in at least fp32, optionally paired with ``(Sq,)`` history
        counts.
    """
    q_len, k_len = q_idx.shape[2], k_idx.shape[1]
    acc = accumulation_dtype(q_idx, k_idx)
    if n_sink is None:
        history = history_mask(pinned, causal_keep, q_len, k_len, q_idx.device)
        sink_stop = -1
    else:
        history = None
        sink_stop = min(n_sink, k_len)
    if query_offset is None:
        query_offset = k_len - q_len
    scale = (
        gate_scale
        if isinstance(gate_scale, torch.Tensor)
        else torch.tensor(gate_scale, device=q_idx.device, dtype=acc)
    )
    lse = _HistoryLSE.apply(
        q_idx, k_idx, scale, history, key_tile, acc, sink_stop, query_offset
    )
    if return_history_count:
        if history is not None:
            return lse, history.sum(dim=-1)
        q_pos = torch.arange(q_len, device=q_idx.device) + query_offset
        return lse, (q_pos - sink_stop + 1).clamp(min=0, max=k_len - sink_stop)
    return lse


def gate_from_score(
    score: torch.Tensor,
    lse: torch.Tensor,
    pinned: torch.Tensor,
) -> torch.Tensor:
    """
    Assemble the additive gate: ``score - lse`` on history, ``0`` on pinned pairs.

    The reference form, used to define the operation and to check the fast paths against. Takes
    ``(B, h, Sq, Sk)`` and so materializes the gate -- for tests and short sequences.
    """
    return torch.where(pinned, torch.zeros_like(score), score - lse.unsqueeze(-1))
