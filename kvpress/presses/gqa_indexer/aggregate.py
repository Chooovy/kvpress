# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Post-processing of indexer token scores.

Two independent reductions live here, applied in this order:

1. :func:`reduce_queries` collapses the query axis, turning per-(query, key) logits into
   one importance value per key: (B, h, Sq, Sk) -> (B, h, Sk).
2. :func:`aggregate_chunk_scores` optionally pools those token scores into fixed-size
   chunks, so selection can happen at chunk granularity: (B, h, Sk) -> (B, h, n_chunks).

Keeping chunking here -- rather than inside the indexer -- means the indexer always
produces token-level scores and chunking stays a swappable policy, which is what makes
mean-vs-max ablations cheap.
"""

from __future__ import annotations

import math

import torch


def reduce_queries(
    scores: torch.Tensor,
    mode: str = "mean",
    *,
    last_n_query: int | None = None,
    recency_half_life: float = 32.0,
) -> torch.Tensor:
    """
    Collapse the query axis of ``(B, h, Sq, Sk)`` into ``(B, h, Sk)``.

    Parameters
    ----------
    scores : torch.Tensor
        Token-level logits from the indexer, (B, h, Sq, Sk).
    mode : str
        ``mean``   -- average over queries (default; matches the dense KL objective).
        ``max``    -- a key is important if *any* query wants it.
        ``last``   -- average over the final ``last_n_query`` queries only.
        ``recency``-- exponentially recency-weighted average over queries.
    last_n_query : int, optional
        Window size for ``last``; also restricts ``mean``/``max``/``recency`` when set.
    recency_half_life : float
        Half-life in tokens for ``recency``. Larger is flatter.

    Returns
    -------
    torch.Tensor
        (B, h, Sk) importance per key, per KV head.
    """
    if scores.dim() != 4:
        raise ValueError(f"expected (B, h, Sq, Sk), got shape {tuple(scores.shape)}")

    if last_n_query is not None:
        n = min(int(last_n_query), scores.shape[2])
        scores = scores[:, :, -n:, :]

    mode = (mode or "mean").lower()
    if mode in ("mean", "avg", "last"):
        return scores.mean(dim=2)
    if mode in ("max", "amax"):
        return scores.amax(dim=2)
    if mode in ("recency", "recency_weighted"):
        q_len = scores.shape[2]
        if q_len == 1:
            return scores.squeeze(2)
        decay = math.log(2.0) / max(float(recency_half_life), 1e-3)
        ages = torch.arange(q_len - 1, -1, -1, device=scores.device, dtype=torch.float32)
        w = torch.exp(-decay * ages)
        w = w / w.sum().clamp_min(1e-8)
        return (scores.float() * w.view(1, 1, q_len, 1)).sum(dim=2).to(scores.dtype)
    raise ValueError(f"Unknown query reduce mode {mode!r}; use mean, max, last or recency.")


def aggregate_chunk_scores(
    token_scores: torch.Tensor, chunk_size: int, mode: str = "mean"
) -> tuple[torch.Tensor, int]:
    """
    Pool token scores into fixed-size chunks.

    Only whole chunks are pooled; the ragged tail is returned to the caller via
    ``complete_end`` so it can be handled explicitly (the press keeps the tail at token
    granularity rather than letting a short chunk compete on unequal terms).

    Parameters
    ----------
    token_scores : torch.Tensor
        (B, h, L) per-token importance.
    chunk_size : int
        Tokens per chunk.
    mode : str
        ``mean`` (average importance) or ``max`` (a chunk is as good as its best token).

    Returns
    -------
    chunk_scores : torch.Tensor
        (B, h, n_chunks) pooled scores.
    complete_end : int
        Number of tokens covered by whole chunks; ``token_scores[..., complete_end:]``
        is the leftover tail.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    length = token_scores.shape[-1]
    n_chunks = length // chunk_size
    complete_end = n_chunks * chunk_size
    if n_chunks == 0:
        empty = token_scores.new_zeros((*token_scores.shape[:-1], 0))
        return empty, 0

    reshaped = token_scores[..., :complete_end].float().reshape(
        *token_scores.shape[:-1], n_chunks, chunk_size
    )
    mode = (mode or "mean").lower()
    if mode in ("mean", "avg"):
        chunk_scores = reshaped.mean(dim=-1)
    elif mode in ("max", "amax"):
        chunk_scores = reshaped.amax(dim=-1)
    else:
        raise ValueError(f"Unknown chunk aggregate mode {mode!r}; use mean or max.")
    return chunk_scores, complete_end


def expand_chunk_indices(chunk_indices: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Expand chunk indices into the token indices they cover.

    (B, h, n_kept) -> (B, h, n_kept * chunk_size), following the same offset-broadcast
    pattern used by the chunk presses in this repo.
    """
    offsets = torch.arange(chunk_size, device=chunk_indices.device).view(1, 1, 1, -1)
    return (chunk_indices.unsqueeze(-1) * chunk_size + offsets).flatten(-2)
