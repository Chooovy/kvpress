# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Two-level (HSA) chunk attention: the router's weight **is** the chunk's attention mass.

    out = sum_c  w_c * softmax_within-chunk-c(q k^T) @ v_c        w = softmax(s)

where ``s`` is the indexer's per-``(query, chunk)`` score. This is the fourth objective in the
package and the one ``ROUTER_LEARNABILITY.md`` §5-§6 singles out as structurally sound.

Why this shape, against the three that came before
--------------------------------------------------
====================  ===================  ==========================  ==================
objective             flat router is       what the router learns      candidate pool?
====================  ===================  ==========================  ==================
distillation          n/a                  the teacher's weights       no
gated (additive)      **a no-op**          ``log(mass) - LSE``         no
exact-K subset        not a no-op          a Bernoulli routing logit   **yes, M of n**
**this**              **not a no-op**      **the mass itself**         **no**
====================  ===================  ==========================  ==================

Three consequences, each verified in ``tests/presses/test_gqa_indexer_hsa.py``:

1. **No pinning.** The within-chunk softmax already sums to 1, so ``w_c`` *is* chunk ``c``'s share
   of the output. A flat ``w`` is therefore not the frozen backbone -- it is uniform mixing, which
   is 0.44 away from dense. There is no zero-cost setting for the router to fall into, so none of
   :mod:`~.gate_pin`'s machinery is needed. Contrast the additive gate, whose optimum
   ``g* = log(mass) - LSE`` is a *constant* and therefore carries no ranking at all.

2. **What is learned is what inference ranks on.** ``w = softmax(s)`` and the realized mass agree
   to 1.1e-16, so ``argtop-k`` on ``s`` is ``argtop-k`` on the mass. The additive arm's score is a
   *correction* to the backbone and needs ``LSE_c`` to be turned back into a mass; this one does
   not.

3. **No candidate pool.** The softmax is over every chunk, so every chunk gets a gradient every
   step. That is the bottleneck the exact-K arm actually hit -- measured 11-15% of oracle-best
   chunks never entered its ``M=32`` pool, which no backward estimator can repair. Here the
   question does not arise. Affordable because the score matrix is *pooled*:
   ``(B, Hkv, Sq, n_chunk)`` is 34 MiB at 8K, against the ``(B, Hkv, Sq, Sk)`` token logits' 2 GiB.

Equivalent additive form, and why it is not used
------------------------------------------------
Verified to 3.9e-16 in fp64, gradients included::

    two-level(s)  ==  softmax(q k^T + g) @ v      with  g_c = s_c - LSE_c

so this op *is* a gated attention whose gate happens to subtract the chunk's own log-sum-exp. That
identity is the proof of point 1 above (a flat ``s`` leaves a non-constant ``-LSE_c`` behind, which
is why it cannot flatten), and it is a useful cross-check -- ``test_matches_the_additive_gate_form``
is written against it.

It is **not** the implementation, for one reason: ``LSE_c`` must stay attached to the graph.
Detaching it leaves ``ds`` and ``dv`` exact but puts ``dq`` and ``dk`` 3.3e-2 / 4.4e-2 off --
and ``q``/``k`` are the path the LM loss takes to every layer below this one. Computing ``LSE_c``
with gradient means a first pass over the keys and a second to attend, i.e. strictly more work than
computing the per-chunk softmax once and weighting it, which is what :func:`hsa_chunk_attention`
does.

Memory
------
The retained tensor is the same one every attention in this package fights: per query tile,
``(B, Hq, tile, Sk)`` logits. Tiling alone does not help -- without a checkpoint every tile stays
in the graph and the sum is unchanged, which is the trap
:attr:`~.exact_k_trainer.ExactKIndexerTrainer.score_tile_bytes` records three OOMs for. So the tile
is checkpointed and sized by **bytes**, not by a query count: a count that fits at 8K does not at
16K.
"""

from __future__ import annotations

import logging

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from kvpress.presses.gqa_indexer.exact_k_subset import accumulation_dtype
from kvpress.presses.gqa_indexer.indexer import MASK_NEG

logger = logging.getLogger(__name__)

#: Target bytes for one query tile's ``(B, Hq, tile, Sk)`` attention logits.
#:
#: 256 MiB, matching :data:`~.exact_k_attention.TILE_LOGIT_BYTES` and for the same measured reason:
#: the logits are the first of ~4 same-shaped intermediates the softmax creates, so the forward peak
#: is 4-5x this number. At 8K on Qwen3-8B geometry (Hq=32, Sk=8192) that is a 256-query tile.
TILE_LOGIT_BYTES = 256 * 1024**2

#: Score written on a chunk no query in the tile can see.
#:
#: Finite, unlike ``-inf``: an all-``-inf`` row would make ``softmax`` return NaN rather than zero,
#: and a query in the very first chunk legitimately has no *earlier* chunk. ``-1e4`` is
#: :data:`~.indexer.MASK_NEG`, already this package's convention, and ``exp(-1e4 - max)`` is 0 in
#: fp32 for any plausible max -- so an invisible chunk contributes exactly nothing while the row
#: still normalizes against whatever it *can* see.
INVISIBLE_SCORE = MASK_NEG


def chunk_lse(
    logits: torch.Tensor, chunk_size: int, *, valid: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Per-chunk log-sum-exp of attention logits: ``(..., Sq, Sk)`` -> ``(..., Sq, n_chunk)``.

    The quantity the router is implicitly competing with -- ``ROUTER_LEARNABILITY.md`` §6 shows the
    additive gate's optimum is ``log(mass) - LSE_c``, and that for a frozen backbone
    ``mass_c = softmax(LSE_c)`` exactly. Exposed because it is what makes the *diagnostics*
    possible: the router's job here is to predict ``LSE_c`` up to a per-query constant, so
    ``corr(s, LSE)`` is a direct readout of whether it is learning, available without any oracle.

    ``valid`` masks causally-forbidden ``(query, key)`` pairs; a chunk with no valid key returns
    :data:`INVISIBLE_SCORE` rather than ``-inf``, so a downstream softmax cannot produce NaN.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    *lead, k_len = logits.shape
    n_chunk = -(-k_len // chunk_size)
    padded = n_chunk * chunk_size

    x = logits
    if valid is not None:
        x = x.masked_fill(~valid, -float("inf"))
    if padded != k_len:
        x = torch.cat([x, x.new_full((*lead, padded - k_len), -float("inf"))], dim=-1)
    # accumulation_dtype, not a hardcoded .float(): fp32 is the FLOOR, not the target. Forcing fp32
    # here makes an fp64 caller measure this downcast (5e-8) instead of the property it is testing,
    # which is exactly how `sparse_gqa_attention_reference`'s hardcoded `.float()` produced a
    # misleading 4.5e-7 in the exact-K tests.
    acc = accumulation_dtype(x.dtype)
    lse = torch.logsumexp(x.reshape(*lead, n_chunk, chunk_size).to(acc), dim=-1)
    # An all -inf chunk gives -inf; replace it so a softmax over these cannot see -inf - (-inf).
    return torch.where(torch.isfinite(lse), lse, lse.new_full((), INVISIBLE_SCORE))


def hsa_chunk_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_scores: torch.Tensor,
    *,
    chunk_size: int,
    scaling: float | None = None,
    query_offset: int | None = None,
) -> torch.Tensor:
    """
    Two-level chunk attention, written the obvious way: one explicit loop over chunks.

    ``O(Sq * Sk)`` memory and no checkpointing, so this is the definition the tiled implementation
    is tested against -- not a path to run at length. Deliberately a separate function rather than a
    flag: a reference that shares code with the thing it checks cannot catch a shared mistake.

    Parameters
    ----------
    q : torch.Tensor
        ``(B, Hq, Sq, D)`` post-RoPE queries.
    k, v : torch.Tensor
        ``(B, Hkv, Sk, D)`` / ``(B, Hkv, Sk, Dv)``.
    chunk_scores : torch.Tensor
        ``(B, Hkv, Sq, n_chunk)`` router scores, **with gradient**. Shared across a KV group's
        query heads, matching the indexer's MQA key and the reference HSA implementation's
        per-group block gate.
    chunk_size : int
        Tokens per chunk.
    scaling : float, optional
        Softmax scale; defaults to ``D ** -0.5``.
    query_offset : int, optional
        Key index of query 0's diagonal. Defaults to ``Sk - Sq``.
    """
    b, n_heads, q_len, head_dim = q.shape
    _, n_kv, k_len, _ = k.shape
    if n_heads % n_kv:
        raise ValueError(f"Hq {n_heads} is not a multiple of Hkv {n_kv}")
    group = n_heads // n_kv
    n_chunk = -(-k_len // chunk_size)
    if chunk_scores.shape != (b, n_kv, q_len, n_chunk):
        raise ValueError(
            f"chunk_scores must be (B={b}, Hkv={n_kv}, Sq={q_len}, n_chunk={n_chunk}), got "
            f"{tuple(chunk_scores.shape)}"
        )
    scale = head_dim**-0.5 if scaling is None else float(scaling)
    if query_offset is None:
        query_offset = k_len - q_len

    acc = accumulation_dtype(q.dtype)
    q_pos = torch.arange(q_len, device=q.device) + query_offset
    k_pos = torch.arange(k_len, device=q.device)
    causal = k_pos.view(1, k_len) <= q_pos.view(q_len, 1)  # (Sq, Sk)

    # A chunk is visible to a query if ANY of its keys is. Computed from the causal mask rather than
    # from arithmetic so the two cannot disagree.
    vis = causal.reshape(q_len, n_chunk, chunk_size).any(-1) if k_len % chunk_size == 0 else None
    if vis is None:
        pad = torch.zeros(q_len, n_chunk * chunk_size - k_len, dtype=torch.bool, device=q.device)
        vis = torch.cat([causal, pad], dim=-1).reshape(q_len, n_chunk, chunk_size).any(-1)

    w = torch.softmax(
        chunk_scores.to(acc).masked_fill(~vis.view(1, 1, q_len, n_chunk), INVISIBLE_SCORE), dim=-1
    )

    out = q.new_zeros(b, n_kv, group, q_len, v.shape[-1], dtype=acc)
    q_view = q.view(b, n_kv, group, q_len, head_dim).to(acc)
    for c in range(n_chunk):
        lo, hi = c * chunk_size, min((c + 1) * chunk_size, k_len)
        logits = torch.einsum("bhgqd,bhsd->bhgqs", q_view, k[:, :, lo:hi].to(acc)) * scale
        keep = causal[:, lo:hi]
        logits = logits.masked_fill(~keep.view(1, 1, 1, q_len, hi - lo), -float("inf"))
        # A query that can see no key of this chunk has w_c ~ 0, but its softmax would still be
        # 0/0. Zero the row explicitly instead.
        alive = keep.any(-1).view(1, 1, 1, q_len, 1)
        p = torch.where(alive, torch.softmax(logits, dim=-1), torch.zeros_like(logits))
        o_c = torch.einsum("bhgqs,bhsd->bhgqd", p, v[:, :, lo:hi].to(acc))
        out = out + w[:, :, None, :, c, None] * o_c
    return out.reshape(b, n_heads, q_len, v.shape[-1]).to(q.dtype)


def _attend_tile(
    q_tile: torch.Tensor,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    w_tile: torch.Tensor,
    causal: torch.Tensor,
    scale: float,
    chunk_size: int,
    n_chunk: int,
) -> torch.Tensor:
    """
    One query tile's two-level attention. Returns ``(B, Hkv, group, tile, Dv)``.

    Split out so :func:`hsa_chunk_attention` can wrap it in ``torch.utils.checkpoint``: the
    ``(B, Hq, tile, Sk)`` logits plus the ``exp``/``p`` intermediates are the dominant retained term,
    and no tile size fixes that on its own once the graph spans every tile.

    ``k_full``/``v_full`` are passed whole and sliced *inside*, so the checkpoint's saved inputs are
    tensors the model already holds rather than fresh per-tile copies.
    """
    b, n_kv, group, tile, head_dim = q_tile.shape
    k_len = k_full.shape[2]
    acc = accumulation_dtype(q_tile.dtype)
    padded = n_chunk * chunk_size

    logits = torch.einsum(
        "bhgqd,bhsd->bhgqs", q_tile.to(acc), k_full.to(acc)
    ) * scale  # (B, Hkv, group, tile, Sk)
    logits = logits.masked_fill(~causal.view(1, 1, 1, tile, k_len), -float("inf"))
    if padded != k_len:
        logits = torch.cat(
            [logits, logits.new_full((b, n_kv, group, tile, padded - k_len), -float("inf"))],
            dim=-1,
        )
    blocks = logits.reshape(b, n_kv, group, tile, n_chunk, chunk_size)

    # Within-chunk softmax. Shift by the chunk's own max -- that is the level the normalization
    # happens at, and using a global row max instead would leave a far-past chunk's weights
    # underflowed to 0 while its w_c is not, silently deleting its contribution.
    chunk_max = blocks.amax(dim=-1, keepdim=True)
    alive = torch.isfinite(chunk_max)
    chunk_max = torch.where(alive, chunk_max, torch.zeros_like(chunk_max))
    e = torch.exp(blocks - chunk_max)
    # masked_fill above put -inf on forbidden keys, so exp already gave them 0; the shift keeps
    # that exact. A chunk with no visible key sums to 0 and is zeroed rather than divided.
    total = e.sum(-1, keepdim=True)
    p = torch.where(total > 0, e / total.clamp_min(torch.finfo(acc).tiny), e * 0.0)
    # Fold the chunk weight in before contracting with V, so the (..., n_chunk, chunk_size)
    # intermediate is never materialized a second time at Dv width.
    p = p * w_tile[:, :, None, :, :, None]
    p = p.reshape(b, n_kv, group, tile, padded)
    if padded != k_len:
        p = p[..., :k_len]
    return torch.einsum("bhgqs,bhsd->bhgqd", p, v_full.to(acc))


def hsa_chunk_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_scores: torch.Tensor,
    *,
    chunk_size: int,
    scaling: float | None = None,
    query_offset: int | None = None,
    checkpoint: bool = True,
    query_tile: int = 0,
) -> tuple[torch.Tensor, dict]:
    """
    Two-level chunk attention over **every** chunk, differentiable in ``chunk_scores``.

    The training forward for :class:`~.hsa_trainer.HSAIndexerTrainer`. Full scope on purpose: the
    softmax runs over all ``n_chunk`` chunks, so every chunk receives a content-dependent gradient
    each step. ``ROUTER_LEARNABILITY.md`` §8 argues for that over a sparse-scope forward (SAS's own
    ablation, 47.4 -> 55.6), and unlike the additive arm it costs nothing here -- there is no
    flat-gate loophole to close, so full scope needs no pinning to be safe.

    Parameters
    ----------
    q, k, v, chunk_scores, chunk_size, scaling, query_offset
        As :func:`hsa_chunk_attention_reference`.
    checkpoint : bool
        Recompute each query tile's attention in the backward. Leave True.
    query_tile : int
        Queries per tile. ``0`` sizes it from :data:`TILE_LOGIT_BYTES`.

    Returns
    -------
    out : torch.Tensor
        ``(B, Hq, Sq, Dv)`` in ``q``'s dtype.
    stats : dict
        ``chunk_entropy`` -- the mean entropy of ``w`` in nats, normalized by ``log(n_visible)`` so
        it is comparable across lengths. **The** diagnostic: 1.0 means the router is still mixing
        uniformly and has learned no ranking, which is this objective's analogue of the flat-gate
        no-op and the only failure it does not rule out structurally. Also ``mass_top1`` /
        ``mass_topk`` (how concentrated the mass is, i.e. how much a top-k inference truncation
        would keep) and ``effective_chunks``.
    """
    b, n_heads, q_len, head_dim = q.shape
    _, n_kv, k_len, _ = k.shape
    if n_heads % n_kv:
        raise ValueError(f"Hq {n_heads} is not a multiple of Hkv {n_kv}")
    group = n_heads // n_kv
    n_chunk = -(-k_len // chunk_size)
    if chunk_scores.shape != (b, n_kv, q_len, n_chunk):
        raise ValueError(
            f"chunk_scores must be (B={b}, Hkv={n_kv}, Sq={q_len}, n_chunk={n_chunk}), got "
            f"{tuple(chunk_scores.shape)}"
        )
    scale = head_dim**-0.5 if scaling is None else float(scaling)
    if query_offset is None:
        query_offset = k_len - q_len

    acc = accumulation_dtype(q.dtype)
    itemsize = torch.finfo(acc).bits // 8
    if query_tile <= 0:
        per_query = b * n_heads * k_len * itemsize
        query_tile = max(1, min(q_len, TILE_LOGIT_BYTES // max(per_query, 1)))

    device = q.device
    k_pos = torch.arange(k_len, device=device)
    chunk_start = torch.arange(n_chunk, device=device) * chunk_size

    q_view = q.view(b, n_kv, group, q_len, head_dim)
    tiles: list[torch.Tensor] = []
    ent_sum, ent_n = 0.0, 0
    top1_sum, topk_sum, eff_sum = 0.0, 0.0, 0.0

    for start in range(0, q_len, query_tile):
        stop = min(start + query_tile, q_len)
        tile = stop - start
        q_pos = torch.arange(start, stop, device=device) + query_offset
        causal = k_pos.view(1, k_len) <= q_pos.view(tile, 1)
        # A chunk is visible iff its FIRST key is -- chunk_start <= q_pos. Equivalent to
        # "any key visible" and cheaper than reducing the mask.
        vis = chunk_start.view(1, n_chunk) <= q_pos.view(tile, 1)

        s_tile = chunk_scores[:, :, start:stop].to(acc)
        w = torch.softmax(s_tile.masked_fill(~vis.view(1, 1, tile, n_chunk), INVISIBLE_SCORE), -1)

        args = (
            q_view[:, :, :, start:stop].contiguous(),
            k, v, w, causal, scale, chunk_size, n_chunk,
        )
        if checkpoint and torch.is_grad_enabled():
            tile_out = torch_checkpoint(_attend_tile, *args, use_reentrant=False)
        else:
            tile_out = _attend_tile(*args)
        # Narrowed per tile, not at the end: the concatenated fp32 output is retained for the whole
        # backward, and the tile's arithmetic already ran in fp32.
        tiles.append(tile_out.to(q.dtype))

        with torch.no_grad():
            wd = w.detach()
            n_vis = vis.sum(-1).clamp(min=1).view(1, 1, tile)
            ent = -(wd.clamp_min(1e-12) * wd.clamp_min(1e-12).log()).sum(-1)
            # Normalized by log(n_visible): a query near the start has few chunks to choose from,
            # so raw entropy would drift with position and confound "committed" with "early".
            denom = n_vis.float().log().clamp_min(1e-6)
            ent_sum += float((ent / denom).mean()) * tile
            ent_n += tile
            top1_sum += float(wd.amax(-1).mean()) * tile
            kk = min(n_chunk, max(1, n_chunk // 4))
            topk_sum += float(wd.topk(kk, dim=-1).values.sum(-1).mean()) * tile
            eff_sum += float(torch.exp(ent).mean()) * tile

    out = torch.cat(tiles, dim=3) if len(tiles) > 1 else tiles[0]
    stats = {
        "chunk_entropy": ent_sum / max(ent_n, 1),
        "mass_top1": top1_sum / max(ent_n, 1),
        "mass_topquarter": topk_sum / max(ent_n, 1),
        "effective_chunks": eff_sum / max(ent_n, 1),
        "n_chunk": int(n_chunk),
    }
    return out.reshape(b, n_heads, q_len, v.shape[-1]), stats
