# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Chunk-wise support selection for inference: pick whole chunks, then expand them to tokens.

Why this exists
---------------
:func:`~.sparse_support.streaming_topk_support` selects the top-``topk`` **tokens** per query. That
is the right thing for a router trained on per-token scores (the gated arm), and the **wrong** thing
for one trained on chunk-mean scores over query blocks (the exact-K arm).

Measured on the two trained checkpoints -- fraction of score variance lying *within* a 64-token chunk
versus *between* chunks, on real text:

======  =====================  ===================
layer   exact-K within/across  gated within/across
======  =====================  ===================
0       **0.17**               0.70
4       **0.16**               0.99
7       0.69                   0.74
======  =====================  ===================

exact-K learned an almost **piecewise-constant** score: at early layers the between-chunk variance is
~6x the within-chunk variance, because chunk-mean scores are the only thing its loss ever saw, so
within-chunk structure is unconstrained and stays near initialization. Ranking its *tokens* therefore
resolves a near-tie inside every chunk, and a token-budget top-k spends most of its decisions in
exactly the region where the score carries no information.

That is not a bug in the router; it is an evaluation that measures a resolution the router was never
trained to have. This module supplies the matched alternative, so the exact-K objective can be scored
on the operator it actually trains.

What it does
------------
1. pool the per-``(query, key)`` indexer logits to chunks, exactly as
   :func:`~.exact_k_attention.pool_scores_to_chunks` does at train time;
2. take the top ``topk // chunk_size`` **chunks** per query;
3. expand each chunk to its ``chunk_size`` token positions and merge in the forced sink/local slots.

The result is the same ``(B, Hkv, Sq, topk)`` ascending int32 support tensor with ``-1`` in empty
slots that :func:`~.sparse_attention.sparse_gqa_attention` already consumes, so nothing downstream
changes.

Deliberately per-query, not per-query-block
-------------------------------------------
Training shares one subset across ``query_block`` queries. This selects per query. That is a
*generous* mismatch rather than a harmful one -- per-query selection is strictly finer, and each
query gets at least the chunks its block would have agreed on -- and keeping it per-query means the
comparison against the gated arm holds the *selection granularity along the query axis* fixed and
varies only the key axis, which is the axis the score's structure actually differs on. A
block-shared variant would confound the two.
"""

from __future__ import annotations

import logging

import torch

from kvpress.presses.gqa_indexer.indexer import MASK_NEG
from kvpress.presses.gqa_indexer.sparse_support import forced_support_positions, sort_support

logger = logging.getLogger(__name__)


def chunk_topk_support(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    topk: int,
    *,
    chunk_size: int,
    chunk_aggregate: str = "mean",
    score_scale: float = 1.0,
    query_offset: int | None = None,
    force_sink: int = 0,
    force_local: int = 0,
    query_tile: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select whole chunks per query and expand them to a token support tensor.

    Parameters
    ----------
    q_idx : torch.Tensor
        Indexer queries after norm and RoPE, ``(B, h, Sq, D)``.
    k_idx : torch.Tensor
        Shared indexer key after norm and RoPE, ``(B, Sk, D)``.
    topk : int
        **Token** budget, including the forced slots -- the same units
        :func:`~.sparse_support.streaming_topk_support` takes, so the two are swappable and a budget
        comparison stays honest. The chunk budget is ``(topk - forced) // chunk_size``.
    chunk_size : int
        Tokens per chunk. Must match the ``chunk_size`` the router trained with; it is recorded in
        the checkpoint config for exactly this reason.
    chunk_aggregate : str
        ``lse``, ``mean`` or ``max`` aggregation of a chunk's token scores. **Must match what the
        router trained with** -- it is recorded in the checkpoint for that reason. ``lse`` is what the
        HSA arm trains, and it is not interchangeable with ``mean``: with an exact token scorer the
        Spearman against the true chunk log-sum-exp is 1.000 for ``lse`` against 0.756 for ``mean``,
        and in a needle regime ``mean`` dilutes a lone high-logit token ~64x (needle recall at top-4
        chunks 0.533 against 1.000).
    score_scale : float
        Multiplier applied to the **token** scores before aggregation. Only meaningful for ``lse``,
        which is not scale-equivariant: ``LSE(c*x) != c*LSE(x)``, so this is a temperature that
        decides whether the aggregation behaves like a mean or like a max. Must match the trainer's
        ``score_scale`` (default ``head_dim ** -0.5``); scoring with a different one silently ranks on
        a different functional than training optimized.
    query_offset : int, optional
        Key index of query 0's diagonal. Defaults to ``Sk - Sq`` (bottom-right), matching
        flash-attention and :func:`~.indexer.build_indexer_mask`.
    force_sink, force_local : int
        Token slots reserved for the leading keys and each row's own most recent keys. Applied at
        **token** granularity, not chunk, so they mean the same thing as on the token path.
    query_tile : int
        Query rows scored at once. The scratch is ``O(query_tile * Sk)``, which at ``Sk=16384`` is
        1024 x 16384 x 4 B = 64 MiB per head -- the knob that bounds it.

    Returns
    -------
    support : torch.Tensor
        ``(B, h, Sq, topk)`` int32, ascending, ``-1`` in empty slots.
    valid : torch.Tensor
        ``support >= 0``.
    """
    if q_idx.dim() != 4:
        raise ValueError(f"q_idx must be (B, h, Sq, D), got {tuple(q_idx.shape)}")
    if k_idx.dim() != 3:
        raise ValueError(f"k_idx must be (B, Sk, D), got {tuple(k_idx.shape)}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if query_tile <= 0:
        raise ValueError(f"query_tile must be positive, got {query_tile}")

    bsz, n_heads, q_len, _ = q_idx.shape
    k_len = k_idx.shape[1]
    device = q_idx.device
    if query_offset is None:
        query_offset = k_len - q_len

    n_forced = int(force_sink) + int(force_local)
    if n_forced > topk:
        raise ValueError(
            f"force_sink + force_local = {n_forced} exceeds topk = {topk}; the forced keys would "
            "be silently truncated. Lower them, or raise topk."
        )
    n_chunk = -(-k_len // chunk_size)
    # Chunk budget from the TOKEN budget, so `topk` means the same thing on both paths. Clamped to
    # at least 1: a budget smaller than one chunk still has to select something, and returning an
    # empty support would make every row attend to the forced slots alone -- a silent budget change.
    chunk_budget = max(1, min((topk - n_forced) // chunk_size, n_chunk))

    mode = (chunk_aggregate or "mean").lower()
    if mode not in ("lse", "logsumexp", "mean", "avg", "max", "amax"):
        raise ValueError(f"chunk_aggregate must be lse, mean or max, got {chunk_aggregate!r}")

    token_offsets = torch.arange(chunk_size, device=device)
    chunk_starts = torch.arange(n_chunk, device=device) * chunk_size
    supports = []

    for start in range(0, q_len, query_tile):
        stop = min(start + query_tile, q_len)
        dq = stop - start
        q_tile = q_idx[:, :, start:stop]  # (B, h, dq, D)
        q_index = torch.arange(start, stop, device=device)
        limit = (q_index + query_offset).clamp(max=k_len - 1)  # last visible key per row

        # (B, h, dq, Sk) token scores. This is the term that makes the whole thing O(Sq * Sk), hence
        # the query tiling; it is transient (never retained, no autograd here) so only the tile
        # matters for peak memory.
        scores = torch.einsum("bhqd,bkd->bhqk", q_tile.float(), k_idx.float())
        # Causally-forbidden keys get the sentinel BEFORE pooling. Without this a chunk straddling
        # the diagonal would be scored partly on keys the row cannot see, so its rank would depend
        # on future content -- silent, and it would inflate exactly the near-diagonal chunks.
        causal = torch.arange(k_len, device=device).view(1, 1, 1, k_len) <= limit.view(1, 1, dq, 1)
        scores = torch.where(causal, scores, scores.new_full((), MASK_NEG))

        padded = n_chunk * chunk_size
        if padded != k_len:
            pad = scores.new_full((bsz, n_heads, dq, padded - k_len), MASK_NEG)
            scores = torch.cat([scores, pad], dim=-1)
        blocks = scores.reshape(bsz, n_heads, dq, n_chunk, chunk_size)

        if mode in ("lse", "logsumexp"):
            # logsumexp over the chunk's tokens, scale applied FIRST (it is inside the reduction --
            # see `score_scale`). MASK_NEG exponentiates to exactly 0, so forbidden and padded slots
            # are discounted for free, with no valid-count division. An all-sentinel chunk would give
            # -inf, so it is floored back to MASK_NEG to keep the downstream topk finite.
            lse = torch.logsumexp(blocks * score_scale, dim=-1)
            chunk_scores = torch.where(
                torch.isfinite(lse), lse, lse.new_full((), MASK_NEG)
            )
        elif mode in ("max", "amax"):
            chunk_scores = blocks.amax(dim=-1)
        else:
            # Mean over the chunk's VALID tokens only. Folding the sentinel in would make the pooled
            # score a monotone ramp in position -- the trap reduce_queries documents -- and reduce
            # selection to "keep the oldest chunks". Matches pool_scores_to_chunks.
            ok = blocks > (MASK_NEG / 2)
            count = ok.sum(-1)
            total = torch.where(ok, blocks, torch.zeros_like(blocks)).sum(-1)
            chunk_scores = torch.where(
                count > 0, total / count.clamp(min=1), total.new_full((), MASK_NEG)
            )

        # A chunk is selectable only if it starts at or before the row's own diagonal. Chunks
        # entirely in the future are already at the sentinel, but masking explicitly keeps the
        # selection from depending on sentinel ties when a row has fewer visible chunks than budget.
        visible = chunk_starts.view(1, 1, 1, n_chunk) <= limit.view(1, 1, dq, 1)
        chunk_scores = chunk_scores.masked_fill(~visible, -float("inf"))

        picked = chunk_scores.topk(chunk_budget, dim=-1).indices  # (B, h, dq, chunk_budget)
        # Drop chunks that were not actually visible (a row with fewer visible chunks than the
        # budget gets -inf slots, which topk still returns).
        picked_ok = torch.gather(chunk_scores, -1, picked) > -float("inf")

        # Chunk -> its token positions. (B, h, dq, chunk_budget * chunk_size)
        tokens = (picked.unsqueeze(-1) * chunk_size + token_offsets).reshape(
            bsz, n_heads, dq, chunk_budget * chunk_size
        )
        ok = (
            picked_ok.unsqueeze(-1).expand(-1, -1, -1, -1, chunk_size).reshape(tokens.shape)
            & (tokens < k_len)
            & (tokens <= limit.view(1, 1, dq, 1))
        )
        tokens = torch.where(ok, tokens, torch.full_like(tokens, -1))

        forced = forced_support_positions(
            q_index,
            force_sink=int(force_sink),
            force_local=int(force_local),
            query_offset=query_offset,
            k_len=k_len,
        )  # (dq, n_forced)
        if forced.numel():
            forced = forced.view(1, 1, dq, -1).expand(bsz, n_heads, dq, forced.shape[-1])
            tokens = torch.cat([tokens, forced], dim=-1)

        # sort_support pushes -1 to the end and returns int32 ascending -- the convention
        # sparse_gqa_attention already consumes.
        support, _ = sort_support(tokens, k_len)

        # DEDUPLICATE. A forced sink/local token that also falls inside a selected chunk would
        # otherwise occupy two slots, and that is not merely wasteful: sparse_gqa_attention sums
        # duplicate indices **with multiplicity**, so the key would get double weight in the softmax
        # -- silently wrong attention. The token path avoids this structurally (excluded_key_mask
        # keeps forced positions out of the top-k pool); here the two sets are chosen independently,
        # so the overlap has to be removed after the fact.
        #
        # Cheap because `support` is already ascending: a duplicate is exactly a value equal to its
        # left neighbour. Blanked to -1 and re-sorted so the survivors stay ascending with the holes
        # at the end.
        dup = torch.zeros_like(support, dtype=torch.bool)
        dup[..., 1:] = (support[..., 1:] == support[..., :-1]) & (support[..., 1:] >= 0)
        if bool(dup.any()):
            support, _ = sort_support(support.masked_fill(dup, -1).long(), k_len)
        # Trim or pad to exactly `topk`.
        #
        # THE TRIM MUST NOT TAKE THE TAIL. `support` is ascending by key index, so
        # `support[..., :topk]` keeps the *smallest* indices and discards the *largest* -- the NEWEST
        # keys, the query's own diagonal among them. Measured with the old tail-trim at
        # `Sk=192, topk=128, chunk_size=64, force_sink=4, force_local=64`: the row held 128 slots
        # whose max index was 187 while the query sat at 191, so keys 188-191 were silently gone. It
        # bit 39% of lengths in a 190-330 scan. The trigger is
        # `chunk_budget * chunk_size + n_forced > topk`, reachable whenever `topk` is small relative
        # to one chunk plus the forced slots (`chunk_budget` is clamped up to 1, so it can overflow).
        #
        # Dropping the OLDEST keys instead is the graceful degradation: a selected chunk keeps its
        # most recent tokens, the diagonal survives, and the sink -- being forced AND lowest-indexed --
        # is protected explicitly rather than by luck.
        if support.shape[-1] > topk:
            overflow = support.shape[-1] - topk
            protected = torch.zeros_like(support, dtype=torch.bool)
            if forced.numel():
                f = forced.to(support.dtype)
                for slot in range(f.shape[-1]):
                    protected |= support == f[..., slot : slot + 1]
            # Rank each real, unprotected key by age (ascending index == oldest first) and blank the
            # `overflow` oldest. `argsort` on the ascending support gives that order directly.
            droppable = (support >= 0) & ~protected
            age_rank = droppable.to(torch.int32).cumsum(-1)  # 1-based among droppable, oldest first
            drop = droppable & (age_rank <= overflow)
            support, _ = sort_support(support.masked_fill(drop, -1).long(), k_len)
            support = support[..., :topk]
        elif support.shape[-1] < topk:
            fill = support.new_full((*support.shape[:-1], topk - support.shape[-1]), -1)
            support = torch.cat([support, fill], dim=-1)
        supports.append(support)

    support = torch.cat(supports, dim=2)
    return support, support >= 0
