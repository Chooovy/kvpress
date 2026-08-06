# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Stage-2 sparse indexer distillation, following AngelPTM's cost model.

Stage 1 (:mod:`kvpress.presses.gqa_indexer.fused_loss`) is ``O(L^2)`` in *compute* even
though it is ``O(L)`` in memory: every key still has to be visited to normalize the
softmax. Stage 2 restricts the objective to each query row's own top-``topk`` support, which
turns the teacher recompute from ``H * L^2 * d`` into ``H * L * topk * d``. That is
AngelPTM's central stage-2 optimization -- ``lighting_indexer`` returns
``(topk_scores, topk_indices)`` and the teacher is only ever recovered *at those positions*,
from sparse-MLA's own ``lse``:

===========  =========  =============  ============
``L``        ``topk``   dense TFLOP    sparse
===========  =========  =============  ============
32K          512        8.80           0.14  (64x)
128K         512        140.74         0.55  (256x)
128K         2048       140.74         2.20  (64x)
===========  =========  =============  ============

Two passes, and why the split is free
-------------------------------------
Pass 1 (:mod:`kvpress.presses.gqa_indexer.sparse_support`) runs under ``no_grad`` and emits
only int64 indices, so it adds nothing to the autograd graph. Pass 2 recomputes the
indexer's own logits at the support *with* gradients. Top-k is not differentiable, so
treating the support as a constant is not an approximation -- it is the only option, and it
is what DSA and AngelPTM do.

Full KL, not cross-entropy
--------------------------
Stage 1 optimizes cross-entropy because ``KL = CE - H(pbar)`` and a *fixed* teacher makes
the entropy a constant with identical gradients. **Stage 2 must not take that shortcut.**
The teacher here is restricted to the student's own support, so ``H(pbar)`` moves as the
support moves -- measured drift of 1.166 to 1.232 nats across supports on the same teacher.
A CE curve would then be uninterpretable, mixing objective progress with support churn.

The entropy is also cheap now: the support is ``topk`` wide, not ``L`` wide, and
``sum p log p`` streams over top-k tiles as easily as the cross term. So the whole reason
stage 1 drops it (no online form for a mixture teacher over ``L`` keys) does not apply.

Teacher normalization
---------------------
Two defensible teachers, and they are genuinely different objectives (measured max
elementwise gap 0.238):

``teacher_mode="global"``
    Softmax over the **full** key axis, group-averaged, then restricted to the support and
    renormalized. Needs the dense ``teacher_lse``. Its normalizer ``Z`` is the teacher's
    probability mass the support actually captured -- a direct recall metric, returned for
    logging. This is what :func:`~.loss.build_sparse_indexer_target` computes, so stage 2
    stays comparable with the dense reference path.

``teacher_mode="support"``
    Softmax over the **support only**, per head, then group-averaged. Needs no dense
    ``teacher_lse`` at all, so stage 2 becomes ``O(L * topk)`` end to end. This matches
    sparse-MLA, whose ``lse`` is by construction over the selected keys.

The two coincide only when every head in a group captures the same support mass; measured
per-head mass within one group spread from 0.005 to 0.995, so they do not coincide in
practice. ``global`` is the default because it keeps the teacher fixed across steps, which
makes the loss curve mean the same thing at step 1 and step 10000.

Both run through one code path: the mode only chooses which ``lse`` feeds
``exp(alpha - lse)``. In support mode ``Z == 1`` identically (verified to 2.2e-16), so the
renormalization is a no-op rather than a special case.
"""

from __future__ import annotations

import logging

import torch

from kvpress.presses.gqa_indexer.fused_loss import EPS, accumulation_dtype, group_view
from kvpress.presses.gqa_indexer.indexer import GQAIndexer
from kvpress.presses.gqa_indexer.sparse_support import gather_support_keys

logger = logging.getLogger(__name__)

TEACHER_MODES = ("global", "support")


def support_teacher_lse(
    alpha: torch.Tensor, valid: torch.Tensor, *, group_size: int = 1, topk_tile: int | None = None
) -> torch.Tensor:
    """
    Per-head logsumexp over the support only, ``(B, H, dq)``.

    This is ``teacher_mode="support"``'s replacement for a dense ``teacher_lse``: it reads
    ``topk`` entries per row instead of ``L``, so the teacher never touches the full key
    axis. Rows with no valid slot return ``0`` rather than ``-inf``, keeping every
    downstream ``exp`` finite; the caller drops those rows via row validity.

    Parameters
    ----------
    alpha : torch.Tensor
        Scaled teacher logits at the support, ``(B, H, dq, tk)``.
    valid : torch.Tensor
        Bool, ``(B, H, dq, tk)`` or ``(B, h, dq, tk)`` with ``H = h * group_size``.
    group_size : int
        Attention heads per KV group, used only to expand a per-KV-head ``valid``.
    topk_tile : int, optional
        Stream the support axis in tiles of this width. ``None`` reads it at once.
    """
    acc = accumulation_dtype(alpha)
    valid = expand_to_heads(valid, alpha.shape[1], group_size)
    if topk_tile is None or topk_tile >= alpha.shape[-1]:
        masked = alpha.to(acc).masked_fill(~valid, -float("inf"))
        lse = torch.logsumexp(masked, dim=-1)
        return torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))

    run_max = torch.full(alpha.shape[:-1], -float("inf"), device=alpha.device, dtype=acc)
    run_sum = torch.zeros_like(run_max)
    for start in range(0, alpha.shape[-1], topk_tile):
        stop = min(start + topk_tile, alpha.shape[-1])
        tile = alpha[..., start:stop].to(acc).masked_fill(~valid[..., start:stop], -float("inf"))
        new_max = torch.maximum(run_max, tile.amax(dim=-1))
        rescale = torch.where(
            torch.isfinite(run_max), torch.exp(run_max - new_max), torch.zeros_like(run_max)
        )
        contrib = torch.where(
            torch.isfinite(new_max).unsqueeze(-1),
            torch.exp(tile - new_max.unsqueeze(-1)),
            torch.zeros_like(tile),
        )
        run_sum = run_sum * rescale + contrib.nan_to_num(0.0).sum(dim=-1)
        run_max = new_max
    lse = run_max + torch.log(run_sum.clamp_min(EPS))
    return torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))


def expand_to_heads(x: torch.Tensor, n_heads: int, group_size: int) -> torch.Tensor:
    """
    Broadcast a per-KV-head tensor to per-attention-head, ``(B, h, ...) -> (B, H, ...)``.

    ``expand`` cannot do this: ``h`` and ``H`` are both non-singleton, so it raises. The
    unsqueeze/expand/reshape below produces exactly the layout
    ``view(B, h, group_size, ...)`` assumes, which is what makes the group mean in
    :func:`sparse_teacher_probs` line up with the head ordering GQA uses (head ``i`` reads
    KV head ``i // group_size``).
    """
    if x.shape[1] == n_heads:
        return x
    bsz, kv_heads = x.shape[0], x.shape[1]
    if kv_heads * group_size != n_heads:
        raise ValueError(f"cannot expand {kv_heads} KV heads by {group_size} to reach {n_heads}")
    return x.unsqueeze(2).expand(bsz, kv_heads, group_size, *x.shape[2:]).reshape(
        bsz, n_heads, *x.shape[2:]
    )


def sparse_teacher_probs(
    alpha: torch.Tensor, lse: torch.Tensor, valid: torch.Tensor, group_size: int
) -> torch.Tensor:
    """
    Group-averaged, *unnormalized* teacher weights at the support, ``(B, h, dq, tk)``.

    ``exp(alpha - lse)`` per head, zeroed at empty slots, then averaged within each KV
    group. The result is deliberately left unnormalized: the caller accumulates the row sum
    ``Z`` across support tiles and divides once at the end, since ``Z`` is not known until
    every tile has been seen. In ``teacher_mode="support"`` ``Z`` is identically 1.

    ``valid`` may be per-KV-head ``(B, h, dq, tk)`` or per-head ``(B, H, dq, tk)``; the
    former is expanded, since a KV group shares one support.
    """
    bsz, n_heads, dq, tk = alpha.shape
    if n_heads % group_size != 0:
        raise ValueError(f"H={n_heads} is not divisible by group_size={group_size}")
    acc = accumulation_dtype(alpha, lse)
    probs = torch.exp(alpha.to(acc) - lse.to(acc).unsqueeze(-1))
    probs = probs.masked_fill(~expand_to_heads(valid, n_heads, group_size), 0.0)
    return probs.view(bsz, n_heads // group_size, group_size, dq, tk).mean(dim=2)


class _FusedSparseIndexerKL(torch.autograd.Function):
    """
    Tiled full KL over a fixed per-row support.

    Streams ``(query_tile, topk_tile)`` blocks, keeping five per-row accumulators:

    ==========  ================================================
    ``m, ell``  student online-softmax max and sumexp
    ``Z``       teacher mass on the support (``1`` in support mode)
    ``A``       ``sum pbar_raw * log pbar_raw`` -- the entropy term
    ``C``       ``sum pbar_raw * I`` -- the cross term
    ==========  ================================================

    then assembles ``KL = A/Z - log Z - C/Z + lse``. Every accumulator is linear in the
    teacher weights, which is what lets the row sum ``Z`` be divided out *after* the fact
    instead of being needed up front.

    As in stage 1, ``dQ`` is accumulated during the forward pass (the row-wise gradient
    ``qhat - pbar`` is separable, so backward only scales it) while ``dK`` needs a second
    transposed pass. Here ``dK`` scatters with ``index_add`` rather than a dense einsum,
    since each query row touches only its own ``topk`` keys.
    """

    @staticmethod
    def forward(
        ctx,
        q_idx,          # (B, h, Sq, D) indexer queries
        k_idx,          # (B, Sk, D) shared indexer key (MQA)
        support,        # (B, h, Sq, tk) int64, -1 = empty
        valid,          # (B, h, Sq, tk) bool
        teacher_alpha,  # callable (q0, q1, support_tile, valid_tile) -> (alpha, lse)
        group_size,
        query_tile,
        topk_tile,
        stats,          # dict to receive 'recall', or None
    ):
        bsz, n_idx_heads, q_len, dim = q_idx.shape
        k_len, topk = k_idx.shape[1], support.shape[-1]
        device = q_idx.device
        acc_dtype = accumulation_dtype(q_idx, k_idx)

        loss_rows = torch.empty((bsz, n_idx_heads, q_len), device=device, dtype=acc_dtype)
        recall = torch.empty_like(loss_rows)
        lse_student = torch.empty_like(loss_rows)
        teacher_mass = torch.empty_like(loss_rows)
        dq_unit = torch.empty((bsz, n_idx_heads, q_len, dim), device=device, dtype=acc_dtype)

        for q_start in range(0, q_len, query_tile):
            q_stop = min(q_start + query_tile, q_len)
            dq = q_stop - q_start
            q_acc = q_idx[:, :, q_start:q_stop].to(acc_dtype)

            run_max = torch.full((bsz, n_idx_heads, dq), -float("inf"), device=device, dtype=acc_dtype)
            run_sum = torch.zeros_like(run_max)
            mass = torch.zeros_like(run_max)     # Z
            entropy_acc = torch.zeros_like(run_max)  # A
            cross = torch.zeros_like(run_max)    # C
            acc_q = torch.zeros((bsz, n_idx_heads, dq, dim), device=device, dtype=acc_dtype)
            tea_q = torch.zeros_like(acc_q)

            for start in range(0, topk, topk_tile):
                stop = min(start + topk_tile, topk)
                sup_tile = support[:, :, q_start:q_stop, start:stop]
                val_tile = valid[:, :, q_start:q_stop, start:stop]

                k_tile = gather_support_keys(k_idx, sup_tile).to(acc_dtype)  # (B,h,dq,t,D)
                logits = torch.einsum("bhqd,bhqtd->bhqt", q_acc, k_tile)
                logits = logits.masked_fill(~val_tile, -float("inf"))

                # --- student: online softmax over the support ---
                new_max = torch.maximum(run_max, logits.amax(dim=-1))
                rescale = torch.where(
                    torch.isfinite(run_max), torch.exp(run_max - new_max), torch.zeros_like(run_max)
                )
                exp_logits = torch.where(
                    torch.isfinite(new_max).unsqueeze(-1),
                    torch.exp(logits - new_max.unsqueeze(-1)),
                    torch.zeros_like(logits),
                )
                exp_logits = exp_logits.masked_fill(~val_tile, 0.0)
                run_sum = run_sum * rescale + exp_logits.sum(dim=-1)
                acc_q = acc_q * rescale.unsqueeze(-1) + torch.einsum(
                    "bhqt,bhqtd->bhqd", exp_logits, k_tile
                )
                run_max = new_max

                # --- teacher: unnormalized weights, three linear accumulators ---
                alpha, lse_tile = teacher_alpha(q_start, q_stop, sup_tile, val_tile)
                p_raw = sparse_teacher_probs(alpha, lse_tile, val_tile, group_size).to(acc_dtype)
                safe_logits = logits.masked_fill(~val_tile, 0.0)
                mass = mass + p_raw.sum(dim=-1)
                entropy_acc = entropy_acc + (p_raw * torch.log(p_raw.clamp_min(EPS))).sum(dim=-1)
                cross = cross + (p_raw * safe_logits).sum(dim=-1)
                tea_q = tea_q + torch.einsum("bhqt,bhqtd->bhqd", p_raw, k_tile)

            safe_mass = mass.clamp_min(EPS)
            tile_lse = torch.where(
                torch.isfinite(run_max), run_max, torch.zeros_like(run_max)
            ) + torch.log(run_sum.clamp_min(EPS))
            # KL(pbar || qhat) with pbar = p_raw / Z:
            #   sum (p/Z) log(p/Z) - sum (p/Z) (I - lse) = A/Z - log Z - C/Z + lse
            loss_rows[:, :, q_start:q_stop] = (
                entropy_acc / safe_mass - torch.log(safe_mass) - cross / safe_mass + tile_lse
            )
            lse_student[:, :, q_start:q_stop] = tile_lse
            teacher_mass[:, :, q_start:q_stop] = safe_mass
            recall[:, :, q_start:q_stop] = mass
            dq_unit[:, :, q_start:q_stop] = (
                acc_q / run_sum.clamp_min(EPS).unsqueeze(-1) - tea_q / safe_mass.unsqueeze(-1)
            )

        # `valid` is deliberately NOT saved: it is exactly `support >= 0`, so storing it would
        # retain a bool of the same shape as the largest tensor here for no information --
        # 576 KiB/token across 36 layers at topk=2048. Backward recomputes it per tile.
        ctx.save_for_backward(q_idx, k_idx, support, lse_student, teacher_mass, dq_unit)
        ctx.teacher_alpha = teacher_alpha
        ctx.group_size = group_size
        ctx.query_tile = query_tile
        ctx.topk_tile = topk_tile
        ctx.k_len = k_len
        if stats is not None:
            # Teacher mass the support captured, per row. In global mode this is the
            # recall the selection achieved -- the number that says whether topk is big
            # enough -- and it is free here. In support mode it is identically 1.
            stats["recall"] = recall.detach()
        return loss_rows

    @staticmethod
    def backward(ctx, grad_rows):
        q_idx, k_idx, support, lse_student, teacher_mass, dq_unit = ctx.saved_tensors
        query_tile, topk_tile, k_len = ctx.query_tile, ctx.topk_tile, ctx.k_len
        group_size = ctx.group_size
        acc_dtype = dq_unit.dtype
        grad_rows = grad_rows.to(acc_dtype)

        grad_q = dq_unit * grad_rows.unsqueeze(-1)

        # dK scatters: each query row touches only its own topk keys, so a flat index_add
        # over (B * Sk) rows plus one trash row for the empty slots replaces stage 1's dense
        # einsum. The trash row is dropped afterwards, so invalid slots need no branching.
        bsz, n_idx_heads, q_len, dim = q_idx.shape
        topk = support.shape[-1]
        flat_grad_k = torch.zeros((bsz * k_len + 1, dim), device=k_idx.device, dtype=acc_dtype)
        batch_base = (torch.arange(bsz, device=k_idx.device) * k_len).view(bsz, 1, 1, 1)

        for q_start in range(0, q_len, query_tile):
            q_stop = min(q_start + query_tile, q_len)
            q_acc = q_idx[:, :, q_start:q_stop].to(acc_dtype)
            tile_lse = lse_student[:, :, q_start:q_stop]
            tile_mass = teacher_mass[:, :, q_start:q_stop]
            tile_grad = grad_rows[:, :, q_start:q_stop]

            for start in range(0, topk, topk_tile):
                stop = min(start + topk_tile, topk)
                sup_tile = support[:, :, q_start:q_stop, start:stop]
                val_tile = sup_tile >= 0  # cheaper than retaining it from the forward pass

                k_tile = gather_support_keys(k_idx, sup_tile).to(acc_dtype)
                logits = torch.einsum("bhqd,bhqtd->bhqt", q_acc, k_tile)
                p_hat = torch.exp(logits - tile_lse.unsqueeze(-1)).masked_fill(~val_tile, 0.0)

                alpha, lse_t = ctx.teacher_alpha(q_start, q_stop, sup_tile, val_tile)
                p_bar = sparse_teacher_probs(alpha, lse_t, val_tile, group_size).to(acc_dtype)
                p_bar = p_bar / tile_mass.unsqueeze(-1)

                weighted = ((p_hat - p_bar) * tile_grad.unsqueeze(-1)).masked_fill(~val_tile, 0.0)
                contrib = weighted.unsqueeze(-1) * q_acc.unsqueeze(-2)  # (B,h,dq,t,D)
                # int64 both because index_add_ requires it and because batch_base + sup
                # reaches bsz * k_len, which can exceed int32 even when k_len alone does not.
                sup_long = sup_tile.long()
                flat_index = torch.where(
                    val_tile, batch_base + sup_long, torch.full_like(sup_long, bsz * k_len)
                )
                flat_grad_k.index_add_(0, flat_index.reshape(-1), contrib.reshape(-1, dim))

        grad_k = flat_grad_k[: bsz * k_len].view(bsz, k_len, dim)

        return (
            grad_q.to(q_idx.dtype),
            grad_k.to(k_idx.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def make_sparse_recompute_teacher(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    scaling: float,
    group_size: int,
    *,
    teacher_lse: torch.Tensor | None = None,
    teacher_mode: str = "global",
    support: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
    topk_tile: int | None = None,
) -> object:
    """
    Build the ``teacher_alpha`` callable the sparse loss streams against.

    Returns ``(q_start, q_stop, support_tile, valid_tile) -> (alpha, lse)`` where ``alpha``
    is ``(B, H, dq, tk_tile)`` scaled teacher logits *at the support* and ``lse`` is
    ``(B, H, dq)``. Gathering the teacher's keys per row is what makes the recompute
    ``O(L * topk * d)`` rather than ``O(L^2 * d)``.

    Parameters
    ----------
    query_states, key_states : torch.Tensor
        Teacher Q ``(B, H, Sq, d)`` and K ``(B, H | h, Sk, d)``, post-RoPE.
    scaling : float
        Softmax scale.
    group_size : int
        Attention heads per KV group.
    teacher_lse : torch.Tensor, optional
        Dense ``(B, H, Sq)`` logsumexp. Required for ``teacher_mode="global"``.
    teacher_mode : str
        ``global`` (normalize over the full key axis) or ``support`` (normalize over the
        support). See the module docstring; they are different objectives.
    support, valid : torch.Tensor, optional
        Full ``(B, h, Sq, tk)`` support. Required for ``teacher_mode="support"``, which must
        see every slot of a row to normalize it, even when the loss streams the row in tiles.
    topk_tile : int, optional
        Tile width for the support-mode logsumexp.
    """
    if teacher_mode not in TEACHER_MODES:
        raise ValueError(f"teacher_mode must be one of {TEACHER_MODES}, got {teacher_mode!r}")

    n_heads = query_states.shape[1]
    n_kv = key_states.shape[1]
    if n_kv != n_heads and n_heads % n_kv != 0:
        raise ValueError(f"H={n_heads} is not divisible by key heads={n_kv}")

    bsz = query_states.shape[0]
    dim = key_states.shape[-1]
    acc = accumulation_dtype(query_states, key_states)
    kv_heads = n_heads // group_size
    grouped = n_kv != n_heads
    # Neither tensor is upcast here. This closure is stored on the autograd ctx, so an fp32
    # copy would be retained per layer -- 720 KiB/token across 36 layers on Qwen3-8B. The
    # gather reads the caller's own dtype (for the keys, the resident KV cache) and only the
    # gathered tile is widened, which is bit-identical since widening is exact and elementwise.
    if grouped:
        # View the query as (B, h, g, Sq, d) and leave the keys at h heads, so the gather
        # below runs once per KV head instead of once per attention head. No
        # repeat_interleave: that copy would be group_size times the size for identical
        # arithmetic, and it is the gathered tile -- the largest transient in stage 2.
        q_view = group_view(query_states, n_kv, n_heads)
    else:
        q_view = query_states

    def alpha_at(q_start: int, q_stop: int, sup: torch.Tensor) -> torch.Tensor:
        tile_q, tile_k = q_stop - q_start, sup.shape[-1]
        if not grouped:
            sup_h = expand_to_heads(sup, n_heads, group_size)
            flat = sup_h.clamp_min(0).long().reshape(bsz, n_heads, -1, 1).expand(-1, -1, -1, dim)
            k_gathered = key_states.gather(2, flat).reshape(*sup_h.shape, dim).to(acc)
            q_tile = q_view[:, :, q_start:q_stop].to(acc)
            return torch.einsum("bhqd,bhqtd->bhqt", q_tile, k_gathered) * scaling

        # A whole KV group shares one support, so gather (B, h, dq, tk, d) -- group_size
        # times smaller than the (B, H, ...) tensor the expanded form would build.
        sup_kv = sup if sup.shape[1] == n_kv else sup[:, ::group_size]
        flat = sup_kv.clamp_min(0).long().reshape(bsz, n_kv, -1, 1).expand(-1, -1, -1, dim)
        k_gathered = key_states.gather(2, flat).reshape(bsz, n_kv, tile_q, tile_k, dim).to(acc)
        q_tile = q_view[:, :, :, q_start:q_stop].to(acc)
        alpha = torch.einsum("bhgqd,bhqtd->bhgqt", q_tile, k_gathered)
        return alpha.reshape(bsz, n_heads, tile_q, tile_k) * scaling

    if teacher_mode == "global":
        if teacher_lse is None:
            raise ValueError("teacher_mode='global' needs teacher_lse; pass one or use 'support'")
        lse_acc = teacher_lse.to(acc)

        def teacher_alpha(q_start, q_stop, sup, val):
            return alpha_at(q_start, q_stop, sup), lse_acc[:, :, q_start:q_stop]

        return teacher_alpha

    if support is None or valid is None:
        raise ValueError(
            "teacher_mode='support' needs the full support/valid tensors: a row's normalizer "
            "spans all of its slots, so it cannot be derived from one tile"
        )
    if support.shape[1] not in (kv_heads, n_heads):
        raise ValueError(f"support has {support.shape[1]} heads, expected {kv_heads} or {n_heads}")

    # One logsumexp per row over the full support, cached across tiles. This is the whole
    # teacher cost in support mode: O(L * topk * d), never touching the full key axis.
    cached: dict[tuple[int, int], torch.Tensor] = {}

    def teacher_alpha(q_start, q_stop, sup, val):
        key = (q_start, q_stop)
        if key not in cached:
            full_alpha = alpha_at(q_start, q_stop, support[:, :, q_start:q_stop])
            cached.clear()  # only the current query tile is ever needed
            cached[key] = support_teacher_lse(
                full_alpha,
                valid[:, :, q_start:q_stop],
                group_size=group_size,
                topk_tile=topk_tile,
            )
        return alpha_at(q_start, q_stop, sup), cached[key]

    return teacher_alpha


def fused_sparse_indexer_kl_rows(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    support: torch.Tensor,
    valid: torch.Tensor,
    teacher_alpha,
    *,
    group_size: int,
    query_tile: int = 512,
    topk_tile: int = 512,
    stats: dict | None = None,
) -> torch.Tensor:
    """
    Per-row full KL between the indexer and the teacher, both restricted to ``support``.

    Parameters
    ----------
    q_idx : torch.Tensor
        Indexer queries after norm and RoPE, ``(B, h, Sq, D)``.
    k_idx : torch.Tensor
        Shared indexer key after norm and RoPE, ``(B, Sk, D)``.
    support : torch.Tensor
        ``(B, h, Sq, topk)`` int64 key indices, ``-1`` for empty slots.
    valid : torch.Tensor
        ``(B, h, Sq, topk)`` bool.
    teacher_alpha : callable
        From :func:`make_sparse_recompute_teacher`.
    group_size : int
        Attention heads per KV group.
    query_tile, topk_tile : int
        Tile sizes. Peak scratch is ``O(query_tile * topk_tile * D)``; the result is
        identical for every combination.
    stats : dict, optional
        Receives ``"recall"``: the ``(B, h, Sq)`` teacher mass the support captured. Worth
        logging in ``global`` mode -- a low value means ``topk`` is too small for the
        teacher's actual spread, which no loss value alone reveals.

    Returns
    -------
    torch.Tensor
        ``(B, h, Sq)`` per-row KL. Reduce with the caller's row-validity mask.
    """
    if q_idx.dim() != 4:
        raise ValueError(f"q_idx must be (B, h, Sq, D), got {tuple(q_idx.shape)}")
    if k_idx.dim() != 3:
        raise ValueError(f"k_idx must be (B, Sk, D), got {tuple(k_idx.shape)}")
    if support.shape != valid.shape:
        raise ValueError(f"support {tuple(support.shape)} vs valid {tuple(valid.shape)}")
    if support.shape[:3] != q_idx.shape[:3]:
        raise ValueError(
            f"support {tuple(support.shape)} does not match q_idx {tuple(q_idx.shape)} on (B, h, Sq)"
        )
    if query_tile <= 0 or topk_tile <= 0:
        raise ValueError(f"tile sizes must be positive, got query_tile={query_tile}, topk_tile={topk_tile}")

    return _FusedSparseIndexerKL.apply(
        q_idx, k_idx, support, valid, teacher_alpha, group_size, query_tile, topk_tile, stats
    )


def fused_sparse_indexer_loss(
    indexer: GQAIndexer,
    hidden_states: torch.Tensor,
    support: torch.Tensor,
    valid: torch.Tensor,
    teacher_alpha,
    *,
    group_size: int,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    row_valid: torch.Tensor | None = None,
    query_tile: int = 512,
    topk_tile: int = 512,
    loss_coeff: float = 1.0,
    stats: dict | None = None,
) -> torch.Tensor:
    """
    Scalar stage-2 loss for one layer.

    Note there is no ``mask`` argument, unlike stage 1: the support was already filtered
    through the mask in pass 1, so every slot in it is valid by construction. Re-applying a
    dense ``(Sq, Sk)`` mask here would reintroduce exactly the ``O(L^2)`` tensor stage 2
    exists to avoid.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    q_idx = indexer.project_q(hidden_states, cos, sin)
    k_idx = indexer.project_k(hidden_states, cos, sin)

    rows = fused_sparse_indexer_kl_rows(
        q_idx,
        k_idx,
        support,
        valid,
        teacher_alpha,
        group_size=group_size,
        query_tile=query_tile,
        topk_tile=topk_tile,
        stats=stats,
    )

    if row_valid is None:
        row_valid = valid.any(dim=-1)
    weight = row_valid.to(rows.dtype).expand_as(rows)
    return (rows * weight).sum() / weight.sum().clamp_min(1.0) * loss_coeff
