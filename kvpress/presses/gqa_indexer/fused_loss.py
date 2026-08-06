# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tiled indexer distillation loss: O(L) memory instead of O(L^2).

The naive objective in :mod:`kvpress.presses.gqa_indexer.train` materializes the full
``(B, h, Sq, Sk)`` student logits and the dense teacher attention, which caps usable
sequence length at a few thousand tokens. This module streams tiles instead and keeps only
``O(L * h)`` state, following FlashKL (Liu et al., "FlashKL", 2026) with the engineering
details from StreamKL (arXiv:2606.20005).

**Both** axes are tiled. Tiling keys alone still leaves each tile at
``(B, H, Sq, key_tile)`` -- 20 GiB at ``L=128K`` on Llama-3.1-8B -- which is the
``O(L * tile)`` trap FlashKL's own reference implementation falls into. With the query axis
tiled too, peak tile memory is ``O(query_tile * key_tile)``: flat in sequence length.

Objective
---------
The trained quantity is the cross-entropy, not the full KL::

    loss[j, t] = -sum_s pbar[j, t, s] * I[j, t, s] + logsumexp_s(I[j, t, :])

``KL = CE - H(pbar)`` and the teacher is frozen, so the entropy term is a constant with
**identical gradients** -- but the reported number sits above the true KL by ``H(pbar)``,
so it is not comparable with a KL curve from :func:`~.train.indexer_layer_loss`. The
entropy of a *mixture* teacher has no online form (``x log x`` is non-linear, so the
running rescale trick does not apply), which is exactly why it is dropped; recovering it
would need a second teacher pass.

Why this streams in one pass
----------------------------
``I[j, t, s]`` depends only on that ``(t, s)`` pair -- there is no activation and no
cross-head reduction to entangle keys -- so ``sum_s pbar * I`` is *linear* in the teacher
probabilities and accumulates exactly like FlashAttention's ``sum_s p * V``. The student
side needs the usual online-softmax running ``(max, sumexp)``.

The row-wise gradient is ``dloss/dI = phat - pbar``, which depends only on that row, so
the forward pass can accumulate a unit-weight ``dQ`` and the backward pass merely scales
it by the upstream gradient. Each query row is self-contained, which is also why query
tiles never interact for the loss or ``dQ``.

What still needs a second pass
------------------------------
``dK[s] = sum_j sum_t (phat - pbar)[j, t, s] * q[j, t]`` sums over *queries* while the
forward streams *keys*. Its student half carries ``1 / ell[j, t]``, which is final only
after every key tile, and a single key tile's contribution mixes many different per-query
normalizers -- so no scalar can rescale it afterwards. ``dK`` therefore runs in a second
pass, where every query tile accumulates into it, matching StreamKL's separate
``dL``/``dN`` kernels. This is a layout constraint, not an artifact of any activation.
"""

from __future__ import annotations

import torch

from kvpress.presses.gqa_indexer.indexer import GQAIndexer

# Rows with no valid key would divide by zero; their loss is masked out anyway.
EPS = 1e-10


def accumulation_dtype(*tensors: torch.Tensor) -> torch.dtype:
    """
    Pick the accumulation dtype: fp32 for low-precision inputs, otherwise the promoted one.

    Unconditionally calling ``.float()`` would silently *downcast* float64, which both
    loses the precision a caller explicitly asked for and breaks gradient checks against a
    float64 reference.

    ``torch.result_type`` is strictly binary, so promotion is folded pairwise with
    ``torch.promote_types`` instead of being splatted over all inputs.
    """
    if not tensors:
        raise ValueError("accumulation_dtype needs at least one tensor")
    dtype = tensors[0].dtype
    for other in tensors[1:]:
        dtype = torch.promote_types(dtype, other.dtype)
    return torch.float32 if dtype.itemsize < 4 else dtype


def teacher_probs_from_lse(
    alpha: torch.Tensor, lse: torch.Tensor, group_size: int
) -> torch.Tensor:
    """
    Rebuild the group-averaged teacher probabilities from logits and their logsumexp.

    ``p[i, t, s] = exp(alpha[i, t, s] - lse[i, t])`` is exact, so a frozen model's
    attention distribution can be recovered from any tile of logits plus the ``O(L * H)``
    logsumexp that flash-attention already computes -- no ``(H, L, L)`` matrix required.

    IMPORTANT: ``lse`` must have been computed under the *same* mask the caller applies to
    ``alpha``. Masking after the fact does not work: the rows would no longer sum to one
    (a masked key's probability mass is simply lost), silently down-weighting exactly the
    rows with the most padding. Callers therefore fold every mask into ``alpha`` before
    taking the logsumexp -- see :func:`teacher_lse_from_qk`.

    Parameters
    ----------
    alpha : torch.Tensor
        Scaled teacher logits for a key tile, (B, H, Sq, tile), already masked.
    lse : torch.Tensor
        Per-head teacher logsumexp over the *full* masked key axis, (B, H, Sq).
    group_size : int
        Attention heads per KV group (``H // h``).

    Returns
    -------
    torch.Tensor
        (B, h, Sq, tile) teacher probabilities, averaged within each KV group.
    """
    bsz, n_heads, q_len, tile = alpha.shape
    if n_heads % group_size != 0:
        raise ValueError(f"H={n_heads} is not divisible by group_size={group_size}")
    acc = accumulation_dtype(alpha, lse)
    probs = torch.exp(alpha.to(acc) - lse.to(acc).unsqueeze(-1))
    return probs.view(bsz, n_heads // group_size, group_size, q_len, tile).mean(dim=2)


class _FusedIndexerCE(torch.autograd.Function):
    """
    Tiled cross-entropy against a group-averaged teacher.

    Both axes are tiled. Every query row keeps its own running ``(max, sumexp)`` and its own
    ``dQ`` accumulator, so query tiles are fully independent for the loss and ``dQ`` -- the
    outer loop could be parallel. ``dK``, by contrast, sums over queries, so each query tile
    *adds* into it.

    Tiling the query axis is what makes the footprint flat in ``L``. With only key tiling, a
    tile is ``(B, H, Sq, key_tile)``, i.e. 20 GiB at ``L=128K`` -- the same ``O(L * tile)``
    trap that FlashKL's reference implementation falls into.

    ``forward`` streams tiles once, producing the per-row loss and the unit-weight ``dQ``;
    ``backward`` scales that ``dQ`` and runs a transposed pass for ``dK``. Saved state is
    ``O(L * h)``: no ``(Sq, Sk)`` tensor is ever held.
    """

    @staticmethod
    def forward(
        ctx,
        q_idx,          # (B, h, Sq, D)   indexer queries, post-norm/RoPE
        k_idx,          # (B, Sk, D)      shared indexer key (MQA)
        teacher_alpha,  # callable (q0, q1, k0, k1) -> (B, H, q1-q0, k1-k0)
        teacher_lse,    # (B, H, Sq)
        group_size,
        mask,           # (B, 1, Sq, Sk) additive, or None
        key_tile,
        query_tile,
    ):
        bsz, n_idx_heads, q_len, dim = q_idx.shape
        k_len = k_idx.shape[1]
        device = q_idx.device
        acc_dtype = accumulation_dtype(q_idx, k_idx, teacher_lse)

        loss_rows = torch.empty(
            (bsz, n_idx_heads, q_len), device=device, dtype=acc_dtype
        )
        lse_student = torch.empty_like(loss_rows)
        dq_unit = torch.empty((bsz, n_idx_heads, q_len, dim), device=device, dtype=acc_dtype)

        for q_start in range(0, q_len, query_tile):
            q_stop = min(q_start + query_tile, q_len)
            q_acc = q_idx[:, :, q_start:q_stop].to(acc_dtype)
            tile_q = q_stop - q_start

            run_max = torch.full(
                (bsz, n_idx_heads, tile_q), -float("inf"), device=device, dtype=acc_dtype
            )
            run_sum = torch.zeros((bsz, n_idx_heads, tile_q), device=device, dtype=acc_dtype)
            # Student half of dQ, in the running-max frame; rescaled alongside run_sum.
            acc_q = torch.zeros(
                (bsz, n_idx_heads, tile_q, dim), device=device, dtype=acc_dtype
            )
            # Teacher half of dQ and the cross term: pbar is already normalized, so these
            # accumulate directly with no rescaling.
            tea_q = torch.zeros_like(acc_q)
            cross = torch.zeros_like(run_sum)

            for start in range(0, k_len, key_tile):
                stop = min(start + key_tile, k_len)
                k_tile = k_idx[:, start:stop].to(acc_dtype)
                logits = torch.einsum("bhqd,bkd->bhqk", q_acc, k_tile)
                if mask is not None:
                    logits = logits + mask[..., q_start:q_stop, start:stop].to(acc_dtype)

                # --- student: online softmax, rescaling both sumexp and acc_q together ---
                new_max = torch.maximum(run_max, logits.amax(dim=-1))
                rescale = torch.where(
                    torch.isfinite(run_max),
                    torch.exp(run_max - new_max),
                    torch.zeros_like(run_max),
                )
                exp_logits = torch.exp(logits - new_max.unsqueeze(-1))
                run_sum = run_sum * rescale + exp_logits.sum(dim=-1)
                acc_q = acc_q * rescale.unsqueeze(-1) + torch.einsum(
                    "bhqk,bkd->bhqd", exp_logits, k_tile
                )
                run_max = new_max

                # --- teacher: probabilities are exact from lse, so no running state ---
                # The mask is folded into alpha (never applied to p_bar afterwards) so the
                # rows stay normalized; teacher_lse is required to use the same mask.
                alpha = teacher_alpha(q_start, q_stop, start, stop)
                if mask is not None:
                    alpha = alpha + mask[..., q_start:q_stop, start:stop].to(alpha.dtype)
                p_bar = teacher_probs_from_lse(
                    alpha, teacher_lse[:, :, q_start:q_stop], group_size
                ).to(acc_dtype)
                cross = cross + (p_bar * logits).sum(dim=-1)
                tea_q = tea_q + torch.einsum("bhqk,bkd->bhqd", p_bar, k_tile)

            tile_lse = run_max + torch.log(run_sum.clamp_min(EPS))
            lse_student[:, :, q_start:q_stop] = tile_lse
            loss_rows[:, :, q_start:q_stop] = tile_lse - cross
            # acc_q/run_sum turns the running-frame sum into sum_s phat * k, the student
            # half of (phat - pbar) @ k.
            dq_unit[:, :, q_start:q_stop] = (
                acc_q / run_sum.clamp_min(EPS).unsqueeze(-1) - tea_q
            )

        ctx.save_for_backward(q_idx, k_idx, lse_student, teacher_lse, dq_unit)
        ctx.teacher_alpha = teacher_alpha
        ctx.group_size = group_size
        ctx.mask = mask
        ctx.key_tile = key_tile
        ctx.query_tile = query_tile
        return loss_rows

    @staticmethod
    def backward(ctx, grad_rows):
        q_idx, k_idx, lse_student, teacher_lse, dq_unit = ctx.saved_tensors
        mask, key_tile, query_tile = ctx.mask, ctx.key_tile, ctx.query_tile
        group_size = ctx.group_size
        acc_dtype = dq_unit.dtype
        grad_rows = grad_rows.to(acc_dtype)

        # dQ was already accumulated in the forward pass; the row-wise loss makes the
        # gradient separable, so it only needs scaling by the upstream gradient.
        grad_q = dq_unit * grad_rows.unsqueeze(-1)

        # dK needs the final normalizers, hence a second pass. It also sums over queries, so
        # every query tile ADDS into it -- unlike dQ, which each tile owns outright.
        grad_k = torch.zeros_like(k_idx, dtype=acc_dtype)
        q_len, k_len = q_idx.shape[2], k_idx.shape[1]
        for q_start in range(0, q_len, query_tile):
            q_stop = min(q_start + query_tile, q_len)
            q_acc = q_idx[:, :, q_start:q_stop].to(acc_dtype)
            tile_lse = lse_student[:, :, q_start:q_stop]
            tile_grad = grad_rows[:, :, q_start:q_stop]

            for start in range(0, k_len, key_tile):
                stop = min(start + key_tile, k_len)
                k_tile = k_idx[:, start:stop].to(acc_dtype)
                logits = torch.einsum("bhqd,bkd->bhqk", q_acc, k_tile)
                if mask is not None:
                    logits = logits + mask[..., q_start:q_stop, start:stop].to(acc_dtype)
                p_hat = torch.exp(logits - tile_lse.unsqueeze(-1))

                alpha = ctx.teacher_alpha(q_start, q_stop, start, stop)
                if mask is not None:
                    alpha = alpha + mask[..., q_start:q_stop, start:stop].to(alpha.dtype)
                p_bar = teacher_probs_from_lse(
                    alpha, teacher_lse[:, :, q_start:q_stop], group_size
                ).to(acc_dtype)

                weighted = (p_hat - p_bar) * tile_grad.unsqueeze(-1)
                grad_k[:, start:stop] += torch.einsum("bhqk,bhqd->bkd", weighted, q_acc)

        return (
            grad_q.to(q_idx.dtype),
            grad_k.to(k_idx.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fused_indexer_ce_rows(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    teacher_alpha,
    teacher_lse: torch.Tensor,
    *,
    group_size: int,
    mask: torch.Tensor | None = None,
    key_tile: int = 512,
    query_tile: int = 512,
) -> torch.Tensor:
    """
    Per-row cross-entropy between the indexer and a group-averaged teacher.

    Parameters
    ----------
    q_idx : torch.Tensor
        Indexer queries after norm and RoPE, (B, h, Sq, D).
    k_idx : torch.Tensor
        Shared indexer key after norm and RoPE, (B, Sk, D).
    teacher_alpha : callable
        ``(q_start, q_stop, k_start, k_stop) -> (B, H, q_stop - q_start, k_stop - k_start)``
        scaled teacher logits for one tile. A callable (rather than a tensor) is what keeps
        the teacher off HBM: the caller recomputes each tile from Q/K on the fly.
    teacher_lse : torch.Tensor
        Teacher logsumexp over the full key axis, (B, H, Sq).
    group_size : int
        Attention heads per KV group (``H // h``).
    mask : torch.Tensor, optional
        Additive (B, 1, Sq, Sk) mask; ``0`` allowed, ``MASK_NEG`` disallowed.
    key_tile, query_tile : int
        Tile sizes. Peak tile memory is ``O(query_tile * key_tile)``, so both must be bounded
        for the footprint to stay flat in sequence length. The result is mathematically
        identical for every combination.

    Returns
    -------
    torch.Tensor
        (B, h, Sq) per-row loss. Reduce with the caller's row-validity mask.
    """
    if q_idx.dim() != 4:
        raise ValueError(f"q_idx must be (B, h, Sq, D), got {tuple(q_idx.shape)}")
    if k_idx.dim() != 3:
        raise ValueError(f"k_idx must be (B, Sk, D), got {tuple(k_idx.shape)}")
    if teacher_lse.dim() != 3:
        raise ValueError(f"teacher_lse must be (B, H, Sq), got {tuple(teacher_lse.shape)}")
    if key_tile <= 0:
        raise ValueError(f"key_tile must be positive, got {key_tile}")
    if query_tile <= 0:
        raise ValueError(f"query_tile must be positive, got {query_tile}")
    return _FusedIndexerCE.apply(
        q_idx, k_idx, teacher_alpha, teacher_lse, group_size, mask, key_tile, query_tile
    )


def make_recompute_teacher(
    query_states: torch.Tensor, key_states: torch.Tensor, scaling: float, group_size: int
):
    """
    Build a ``teacher_alpha`` callable that recomputes logits per tile from Q/K.

    Slicing both axes is what keeps the teacher's contribution to peak memory at
    ``O(query_tile * key_tile)`` rather than ``O(L * key_tile)``.

    ``key_states`` may carry either ``H`` heads or ``H // group_size`` KV heads. The latter is
    **not** expanded: a ``repeat_interleave`` would materialize a real ``group_size``-fold copy
    that then lives as long as this closure, which the autograd graph holds until ``backward()``.
    Instead the query is viewed as ``(B, h, group_size, Sq, d)`` and the einsum broadcasts over
    the group axis, which is exact (verified to 0.0) and keeps the head mapping
    ``i -> i // group_size``.

    Nor is either tensor upcast here. ``.to(float32)`` on entry would retain an fp32 copy of the
    whole teacher **per layer** -- 720 KiB/token across 36 layers on Qwen3-8B, the single largest
    term in the dense stage. The upcast happens per *tile* instead, inside the closure, so what
    is retained is the caller's own bf16 tensors: for the keys that is the KV cache, which is
    resident anyway, making the copy free rather than merely smaller.

    The per-tile upcast is bit-identical to upcasting once. bf16 and fp16 have no more mantissa
    bits than fp32 and the same or fewer exponent bits, so widening is exact, and ``.to()`` is
    elementwise -- slicing then widening equals widening then slicing.

    Parameters
    ----------
    query_states : torch.Tensor
        (B, H, Sq, d) teacher queries, post-RoPE.
    key_states : torch.Tensor
        (B, H, Sk, d) or (B, H // group_size, Sk, d) teacher keys, post-RoPE.
    scaling : float
        Softmax scale, normally ``head_dim ** -0.5``.
    group_size : int
        Attention heads per KV group.

    Returns
    -------
    callable
        ``(q_start, q_stop, k_start, k_stop) -> (B, H, q_stop - q_start, k_stop - k_start)``.
    """
    bsz, n_heads = query_states.shape[0], query_states.shape[1]
    n_kv = key_states.shape[1]
    if n_kv != n_heads and n_heads % n_kv != 0:
        raise ValueError(f"H={n_heads} is not divisible by key heads={n_kv}")

    acc = accumulation_dtype(query_states, key_states)
    grouped = n_kv != n_heads
    q_view = group_view(query_states, n_kv, n_heads) if grouped else query_states

    def teacher_alpha(q_start: int, q_stop: int, k_start: int, k_stop: int) -> torch.Tensor:
        k_tile = key_states[:, :, k_start:k_stop].to(acc)
        if not grouped:
            q_tile = q_view[:, :, q_start:q_stop].to(acc)
            return torch.einsum("bhqd,bhkd->bhqk", q_tile, k_tile) * scaling
        q_tile = q_view[:, :, :, q_start:q_stop].to(acc)
        alpha = torch.einsum("bhgqd,bhkd->bhgqk", q_tile, k_tile)
        return alpha.reshape(bsz, n_heads, q_stop - q_start, k_stop - k_start) * scaling

    return teacher_alpha


def group_view(query_states: torch.Tensor, n_kv: int, n_heads: int) -> torch.Tensor:
    """
    View ``(B, H, Sq, d)`` as ``(B, h, g, Sq, d)`` without copying when possible.

    ``view`` needs compatible strides. A non-contiguous teacher query forces a copy, but in the
    caller's own dtype rather than fp32 -- and :func:`~.fused_trainer.teacher_query_states`
    produces a contiguous tensor, so the copy does not normally happen.
    """
    if not query_states.is_contiguous():
        query_states = query_states.contiguous()
    bsz, _, q_len, dim = query_states.shape
    return query_states.view(bsz, n_kv, n_heads // n_kv, q_len, dim)


def teacher_lse_from_qk(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    scaling: float,
    *,
    mask: torch.Tensor | None = None,
    key_tile: int = 512,
    query_tile: int = 512,
) -> torch.Tensor:
    """
    Streaming teacher logsumexp, for when flash-attention's own value is unavailable.

    Prefer capturing it from the base model's forward (see
    :func:`~.teacher_lse.capture_teacher_lse`) -- that value is free. This fallback costs a
    second ``H * L^2 * d`` pass but never materializes more than
    ``O(query_tile * key_tile)``, and is exact.

    Like :func:`make_recompute_teacher`, KV-head keys are broadcast over a group axis rather
    than expanded with ``repeat_interleave``, so no ``group_size``-fold copy is made. This
    function runs under ``no_grad`` so the copy would be transient, but it is still
    ``group_size`` times the traffic for the same arithmetic.

    Returns
    -------
    torch.Tensor
        (B, H, Sq) logsumexp over the key axis.
    """
    bsz, n_heads, q_len = query_states.shape[0], query_states.shape[1], query_states.shape[2]
    k_len = key_states.shape[2]
    n_kv = key_states.shape[1]
    if n_kv != n_heads and n_heads % n_kv != 0:
        raise ValueError(f"H={n_heads} is not divisible by key heads={n_kv}")

    acc = accumulation_dtype(query_states, key_states)
    grouped = n_kv != n_heads
    # Upcast per tile, not up front: this runs under no_grad so an fp32 copy would be
    # transient, but it is still O(L) of extra peak for the same arithmetic.
    q_view = group_view(query_states, n_kv, n_heads) if grouped else query_states

    lse = torch.empty((bsz, n_heads, q_len), device=query_states.device, dtype=acc)
    for q_start in range(0, q_len, query_tile):
        q_stop = min(q_start + query_tile, q_len)
        tile_q = q_stop - q_start
        run_max = torch.full(
            (bsz, n_heads, tile_q), -float("inf"), device=query_states.device, dtype=acc
        )
        run_sum = torch.zeros_like(run_max)
        for start in range(0, k_len, key_tile):
            stop = min(start + key_tile, k_len)
            k_tile = key_states[:, :, start:stop].to(acc)
            if grouped:
                logits = torch.einsum(
                    "bhgqd,bhkd->bhgqk",
                    q_view[:, :, :, q_start:q_stop].to(acc),
                    k_tile,
                ).reshape(bsz, n_heads, tile_q, stop - start)
            else:
                logits = torch.einsum(
                    "bhqd,bhkd->bhqk", q_view[:, :, q_start:q_stop].to(acc), k_tile
                )
            logits = logits * scaling
            if mask is not None:
                logits = logits + mask[..., q_start:q_stop, start:stop].to(logits.dtype)
            new_max = torch.maximum(run_max, logits.amax(dim=-1))
            rescale = torch.where(
                torch.isfinite(run_max),
                torch.exp(run_max - new_max),
                torch.zeros_like(run_max),
            )
            run_sum = run_sum * rescale + torch.exp(logits - new_max.unsqueeze(-1)).sum(dim=-1)
            run_max = new_max
        lse[:, :, q_start:q_stop] = run_max + torch.log(run_sum.clamp_min(EPS))
    return lse


def fused_indexer_loss(
    indexer: GQAIndexer,
    hidden_states: torch.Tensor,
    teacher_alpha,
    teacher_lse: torch.Tensor,
    *,
    group_size: int,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    row_valid: torch.Tensor | None = None,
    key_tile: int = 512,
    query_tile: int = 512,
    loss_coeff: float = 1.0,
) -> torch.Tensor:
    """
    Scalar tiled indexer loss for one layer.

    Projects q/k through the indexer (the only place the indexer's parameters enter, so
    autograd reaches them through ``q_idx``/``k_idx``), runs the tiled cross-entropy, and
    averages over valid rows.

    Parameters
    ----------
    indexer : GQAIndexer
        The layer's indexer.
    hidden_states : torch.Tensor
        Attention-layer input, (B, Sq, hidden_size).
    teacher_alpha : callable
        ``(start, stop) -> (B, H, Sq, tile)`` teacher logits; see
        :func:`make_recompute_teacher`.
    teacher_lse : torch.Tensor
        (B, H, Sq) teacher logsumexp.
    group_size : int
        Attention heads per KV group.
    cos, sin : torch.Tensor, optional
        RoPE tables already narrowed to ``indexer.rope_dim``.
    mask : torch.Tensor, optional
        Additive (B, 1, Sq, Sk) mask.
    row_valid : torch.Tensor, optional
        Bool (B, h, Sq) or (B, 1, Sq); False rows leave the average.
    key_tile, query_tile : int
        Tile sizes; peak tile memory is ``O(query_tile * key_tile)``.
    loss_coeff : float
        Scalar multiplier.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    q_idx = indexer.project_q(hidden_states, cos, sin)
    k_idx = indexer.project_k(hidden_states, cos, sin)

    rows = fused_indexer_ce_rows(
        q_idx,
        k_idx,
        teacher_alpha,
        teacher_lse,
        group_size=group_size,
        mask=mask,
        key_tile=key_tile,
        query_tile=query_tile,
    )

    if row_valid is None:
        return rows.mean() * loss_coeff
    weight = row_valid.to(rows.dtype).expand_as(rows)
    return (rows * weight).sum() / weight.sum().clamp_min(1.0) * loss_coeff
