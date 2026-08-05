# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
KL objective for the GQA indexer.

Structure and semantics follow Megatron-LM's
``megatron/core/transformer/experimental_attention_variant/dsa_indexer_loss.py`` and
``dsa_masking.py``: the teacher is the true attention distribution, the student is
``softmax(indexer_logits)``, and the loss is ``KL(teacher || student)`` reduced over keys
then averaged over valid query rows.

The one substantive departure from DSA concerns head grouping. DSA sums attention over
*all* heads into a single target because MLA has a single shared KV cache. GQA has one
cache per KV head, so :func:`build_dense_indexer_target` groups the attention heads by
KV group and builds an independent target per group. Averaging across groups instead
would train every indexer head toward the same thing and throw away the per-head
capacity that motivates the design.
"""

from __future__ import annotations

import torch

INDEXER_LOSS_EPS = 1e-10


def to_accum(x: torch.Tensor) -> torch.Tensor:
    """Upcast to fp32 for accumulation, but never downcast a higher-precision input."""
    return x.float() if x.dtype.itemsize < 4 else x


def normalize_indexer_target(target: torch.Tensor) -> torch.Tensor:
    """L1-normalize a non-negative target along the key dimension."""
    return target / target.sum(dim=-1, keepdim=True).clamp_min(INDEXER_LOSS_EPS)


def masked_softmax(logits: torch.Tensor, valid_mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax that zeroes invalid entries and keeps fully-masked rows finite.

    A row with no valid key would otherwise produce NaN (``exp(-inf - -inf)``); the
    row_max is forced to 0 for such rows and the output stays exactly zero, letting the
    caller drop the row via the row-validity mask instead.
    """
    if logits.shape != valid_mask.shape:
        raise ValueError(f"shape mismatch: logits {tuple(logits.shape)} vs mask {tuple(valid_mask.shape)}")
    masked = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
    row_has_valid = valid_mask.any(dim=dim, keepdim=True)
    row_max = masked.max(dim=dim, keepdim=True).values
    row_max = torch.where(row_has_valid, row_max, torch.zeros_like(row_max))
    probs = torch.exp(masked - row_max).masked_fill(~valid_mask, 0.0)
    probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(INDEXER_LOSS_EPS)
    return probs.masked_fill(~valid_mask, 0.0)


def masked_log_softmax(logits: torch.Tensor, valid_mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Log-softmax that zeroes invalid entries and keeps fully-masked rows finite.

    Note the convention, inherited from the Megatron reference: invalid entries are set to
    ``0.0`` *in log space*, not to ``-inf``. Log-probabilities over the valid entries are
    correct and sum to one, but ``exp()`` of this tensor reads ``1.0`` at masked slots, so
    it is not a probability distribution. Always pair it with the same ``valid_mask`` when
    consuming the result -- :func:`indexer_kl_per_row` does exactly that, which is why the
    bogus entries never reach the loss.
    """
    if logits.shape != valid_mask.shape:
        raise ValueError(f"shape mismatch: logits {tuple(logits.shape)} vs mask {tuple(valid_mask.shape)}")
    masked = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
    row_has_valid = valid_mask.any(dim=dim, keepdim=True)
    safe = torch.where(row_has_valid, masked, torch.zeros_like(masked))
    return torch.log_softmax(safe, dim=dim).masked_fill(~valid_mask, 0.0)


def indexer_kl_per_row(
    target: torch.Tensor,
    predict_log_probs: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Elementwise ``KL(target || predict)`` reduced over the key dimension.

    Computed as ``sum_k p * (log p - log q)`` directly, rather than via ``F.kl_div``, so
    that ``p == 0`` entries contribute exactly zero without relying on ``0 * -inf``.
    """
    kl = target * (torch.log(target.clamp_min(INDEXER_LOSS_EPS)) - predict_log_probs)
    if valid_mask is not None:
        kl = kl.masked_fill(~valid_mask, 0.0)
    return kl.sum(dim=-1)


def indexer_loss_from_target(
    target: torch.Tensor,
    predict_log_probs: torch.Tensor,
    *,
    loss_coeff: float = 1.0,
    valid_mask: torch.Tensor | None = None,
    row_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Scalar indexer KL loss from an already-normalized target.

    Parameters
    ----------
    target : torch.Tensor
        Normalized teacher probabilities, (..., Sk); must sum to 1 over the last dim.
    predict_log_probs : torch.Tensor
        Student log-probabilities, same shape as ``target``.
    loss_coeff : float
        Scalar multiplier on the final loss.
    valid_mask : torch.Tensor, optional
        Bool, same shape as ``target``; False entries are excluded from the key sum.
    row_valid : torch.Tensor, optional
        Bool, shape ``target.shape[:-1]``; False rows are excluded from the row average.
        Rows with no valid key must be excluded here, since their KL is meaningless.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    kl_per_row = indexer_kl_per_row(target, predict_log_probs, valid_mask)
    if row_valid is not None:
        row_w = row_valid.to(dtype=kl_per_row.dtype, device=kl_per_row.device)
        kl_per_row = kl_per_row * row_w
        denom = row_w.sum().clamp_min(1.0)
    else:
        denom = torch.tensor(
            float(max(kl_per_row.numel(), 1)), dtype=kl_per_row.dtype, device=kl_per_row.device
        )
    return (kl_per_row.sum() / denom) * loss_coeff


def group_attention_by_kv_head(
    attentions: torch.Tensor, n_kv_heads: int, reduce: str = "mean"
) -> torch.Tensor:
    """
    Group attention heads by KV group: (B, H, Sq, Sk) -> (B, n_kv_heads, Sq, Sk).

    In GQA, attention head ``i`` reads KV head ``i // group_size``, so heads are grouped
    with a plain reshape. ``reduce`` controls how the ``group_size`` heads of a group
    combine into that group's target:

    ``mean`` -- the group's average demand (a well-conditioned probability distribution).
    ``amax`` -- a key is wanted if any head in the group wants it. Needs renormalizing
    afterwards since the elementwise max of distributions does not sum to 1.
    """
    bsz, n_heads, q_len, k_len = attentions.shape
    if n_heads % n_kv_heads != 0:
        raise ValueError(f"n_heads {n_heads} is not divisible by n_kv_heads {n_kv_heads}")
    group_size = n_heads // n_kv_heads
    grouped = attentions.view(bsz, n_kv_heads, group_size, q_len, k_len)
    reduce = (reduce or "mean").lower()
    if reduce in ("mean", "avg"):
        return grouped.mean(dim=2)
    if reduce in ("max", "amax"):
        return grouped.amax(dim=2)
    if reduce == "sum":
        return grouped.sum(dim=2)
    raise ValueError(f"Unknown group reduce {reduce!r}; use mean, amax or sum.")


def build_dense_indexer_target(
    attentions: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    n_kv_heads: int,
    head_reduce: str = "mean",
) -> torch.Tensor:
    """
    Stage-1 (dense) teacher: the true attention distribution, grouped per KV head.

    Parameters
    ----------
    attentions : torch.Tensor
        Attention probabilities from the frozen model, (B, H, Sq, Sk).
    valid_mask : torch.Tensor
        Bool (B, n_kv_heads, Sq, Sk) or broadcastable; False entries are zeroed before
        normalization so masked keys carry no target mass.
    n_kv_heads : int
        Number of KV heads / indexer heads.
    head_reduce : str
        How to combine heads within a KV group (see :func:`group_attention_by_kv_head`).

    Returns
    -------
    torch.Tensor
        (B, n_kv_heads, Sq, Sk) normalized, non-negative target.
    """
    target = group_attention_by_kv_head(to_accum(attentions), n_kv_heads, head_reduce)
    target = target.masked_fill(~valid_mask.expand_as(target), 0.0)
    return normalize_indexer_target(target)


def build_sparse_indexer_target(
    attentions: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    n_kv_heads: int,
    head_reduce: str = "mean",
) -> torch.Tensor:
    """
    Stage-2 (sparse) teacher: attention mass restricted to the indexer's own top-k.

    Mirrors DSA's ``sparse_loss=True`` path, where both teacher and student are
    renormalized over the selected support. This sharpens the ranking *within* the kept
    set once stage 1 has already taught the indexer roughly where to look.

    Parameters
    ----------
    attentions : torch.Tensor
        Attention probabilities, (B, H, Sq, Sk).
    topk_indices : torch.Tensor
        Indices selected by the indexer, (B, n_kv_heads, Sq, topk). Negative entries mark
        empty slots and receive zero target mass.
    n_kv_heads : int
        Number of KV heads / indexer heads.
    head_reduce : str
        How to combine heads within a KV group.

    Returns
    -------
    torch.Tensor
        (B, n_kv_heads, Sq, topk) normalized target aligned with ``topk_indices``.
    """
    grouped = group_attention_by_kv_head(to_accum(attentions), n_kv_heads, head_reduce)
    valid = topk_indices >= 0
    gathered = grouped.gather(-1, topk_indices.clamp_min(0))
    gathered = gathered.masked_fill(~valid, 0.0)
    return normalize_indexer_target(gathered)
