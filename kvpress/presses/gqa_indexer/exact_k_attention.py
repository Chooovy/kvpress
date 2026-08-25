# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Chunk-wise exact-K subset attention: the router picks ``K`` chunks per query block, and the
forward attends to exactly those.

This is the attention-side half of :mod:`~kvpress.presses.gqa_indexer.exact_k_subset`. That module
supplies the estimator (discrete ``k``-subset forward, exact-marginal backward); this one decides
*what* the subset is over, and wires the resulting mask into an attention that the LM loss can
differentiate.

Why this exists at all is the same argument as :mod:`~kvpress.presses.gqa_indexer.gate_pin`, taken
one step further. An additive gate can go flat along the key axis, and a flat gate is inert
(softmax is shift-invariant), so the model reverts to the frozen backbone and the LM loss is
satisfied with no ranking learned. ``gate_pin`` patches that by exempting some keys from the
normalizer. Here the forward *commits* to exactly ``K`` chunks, so there is no configuration of
the scores under which "do nothing" reproduces dense attention -- the hole is closed structurally
rather than patched.

Three granularity decisions, and what each costs
------------------------------------------------
1. **Chunks, not tokens.** Selection is over ``ceil(Sk / chunk_size)`` chunks. Also what the press
   already does at inference (``GQAIndexerPress(chunk_size=...)``), so training and eval agree on
   the unit of selection.
2. **Query blocks, not queries.** One subset is shared by ``query_block`` consecutive queries.
   This is a genuine modelling concession, not just an optimization -- 128 queries agreeing on one
   chunk set is coarser than per-query selection. NSA and HSA both do it, so there is precedent.
   See "What the GPU actually costs" for why it is *not* required for speed here, which was a
   surprise.
3. **A candidate pool of size ``M``, not every chunk.** The DP is ``O(M)`` sequential steps, so
   ``M`` is the one parameter that sets the cost. :func:`build_candidates` restricts to ``M``
   chunks per query block.

What the GPU actually costs -- and where the handoff analysis was wrong
----------------------------------------------------------------------
``HANDOFF_exact_k_subset.md`` §4 extrapolated from CPU timings that the row count
``B * Hkv * Sq`` = 262144 would cost ~1690 s per layer per step, "dead on arrival", and concluded
that query-block sharing was **required**. Measured on an H20 (fp32, ``exact_k_marginals``
with ``create_graph`` + sample + backward):

======  =========  =========  =========  ==========
rows    M          K          ms         peak
======  =========  =========  =========  ==========
1024    64         8          95.1       27 MiB
8192    64         8          95.6       217 MiB
65536   64         8          97.4       1.75 GiB
131072  64         8          114.2      3.56 GiB
======  =========  =========  =========  ==========

**Cost is essentially independent of the row count**, and independent of ``K`` as well (81-85 ms
across ``K`` = 8..128 at ``M=256``). It is linear in ``M`` alone: ~1.5 ms per DP step at
``rows <= 65536``, of which ~310 us is CUDA launch latency for the handful of elementwise kernels
each step issues. The DP is **launch-bound, not FLOP-bound** -- the arithmetic per step is a
``(rows, K+2)`` elementwise pass, which an H20 finishes in the noise.

So the CPU extrapolation was wrong by roughly four orders of magnitude, and it was wrong
*structurally*: on CPU the per-step work is real work, so it scales with rows; on GPU it is a
launch, so it does not. Consequences for the design:

* **Query-block sharing is not needed for throughput.** It is retained (default 128) because it
  reduces the *memory* of the candidate/mask tensors, which do scale with rows, and because
  sharing is defensible on modelling grounds. But ``query_block=1`` is affordable, which the
  handoff concluded it was not -- if per-query granularity turns out to matter, it is available.
* **``M`` is the only knob that matters for time.** Budget ``M`` first; ``K`` is free.
* ``torch.compile`` on the DP would fuse the launches, and does not work: the probe trick needs a
  double-backward and inductor raises ``element 0 of tensors does not require grad``. Measured, not
  assumed. A hand-written Triton scan is the remaining option and was not attempted.

The multiplicative form, over the whole pool
--------------------------------------------
``g`` multiplies the exponentiated attention logits, and the normalization runs over the **entire
candidate pool**, not over the selected ``K``::

    alpha_j = g_j * exp(a_j) / sum_{i in pool} g_i * exp(a_i)

Since ``g`` is 0 on unselected candidates, this is *numerically* identical to softmax over the
selected subset -- verified 1.11e-16 in fp64, so there is no train/inference gap in the forward.
But it is **not** identical in the backward, and the difference is the point of the whole method:

=============  =====================  =======================
form           mean grad, selected    mean grad, unselected
=============  =====================  =======================
gather the K   7.05e-02               1.61e-03
**full pool**  1.22e-01               **1.18e-01**
=============  =====================  =======================

Gathering only the ``K`` selected chunks leaves the unselected ones with **73x less** gradient --
they are reached only indirectly, through the selected chunks' marginals. Normalizing over the
pool puts each unselected candidate's own ``g_j`` in the graph, so it receives credit directly and
at comparable magnitude. That is the property that lets a chunk *outside* the current selection be
promoted, which is the difference between 0.0% and 93.8% recall on the adversarial toy in
``HANDOFF_exact_k_subset.md`` §3.

The cost is that the forward computes attention over ``M`` chunks rather than ``K``, so the
candidate pool -- not the budget -- sets the attention FLOPs at training time. At inference
nothing is gathered: the press takes a plain top-K.

Candidate pool needs exploration
--------------------------------
If the pool were just ``TopM(chunk_scores)``, chunks outside it would receive no gradient at all
-- the same disease as the gather-K form above, one level up. :func:`build_candidates` therefore
mixes top-M with random and structural (sink / local) slots; see its docstring for the split and
for how to measure whether the pool is missing the chunks that matter.
"""

from __future__ import annotations

import logging

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from kvpress.presses.gqa_indexer.exact_k_subset import accumulation_dtype, straight_through_mask
from kvpress.presses.gqa_indexer.indexer import MASK_NEG

logger = logging.getLogger(__name__)

#: Score assigned to a ``-1`` pad slot in the candidate pool.
#:
#: Low enough that ``sigmoid(PAD_SCORE) ~= 3e-14``, so a pad's exact marginal is ~0 whenever the row
#: has a real chunk to spend budget on -- but **finite**, for the reason
#: :data:`~.exact_k_subset.NEG_INF` is finite: an ``-inf`` here would reach the DP's ``logaddexp``
#: and turn ``-inf - (-inf)`` into NaN.
#:
#: Not ``MASK_NEG`` (-1e4): that is so far down that ``log(1 - p)`` underflows to 0 and the DP's
#: "not selected" branch loses all resolution between pads. -31 is comfortably saturated in fp32
#: while both branches stay representable.
PAD_SCORE = -31.0

#: Target bytes for one query tile's attention logits, when ``query_tile`` is left at 0.
#:
#: 256 MiB, not 2 GiB. The logits are only the *first* of ~4 same-shaped intermediates the softmax
#: creates (``exp``, ``weights``, ``p``), so the forward peak lands at roughly 4-5x this number --
#: measured 11.5 GiB at a 2 GiB target, which does not fit beside a frozen 8B backbone whose own
#: activations already reach ~90 GiB at 16K. At 256 MiB the whole op peaks at ~2 GiB.
TILE_LOGIT_BYTES = 256 * 1024**2

#: Cap on the shifted attention exponent ``a_j - a_selected_max``, before ``exp``.
#:
#: A **selected** slot's exponent is ``<= 0`` by construction, so this never touches the
#: distribution being computed. It bounds only the *unselected* slots, whose exponent is positive
#: exactly when the router has found a chunk that looks better than anything it currently holds --
#: which is the useful case, and also the unbounded one.
#:
#: 30 gives ``exp(30) ~ 1e13``: comfortably inside fp32 (max 3.4e38) with room for the ``P @ V``
#: accumulation that follows, and large enough that a genuine promotion signal is not flattened
#: against its neighbours until it is 13 orders of magnitude ahead. Clamping preserves the sign and
#: the ordering of the boundary credit and saturates only its magnitude.
MAX_SHIFTED_EXPONENT = 30.0


def pool_scores_to_chunks(
    token_scores: torch.Tensor, chunk_size: int, mode: str = "mean"
) -> torch.Tensor:
    """
    Pool ``(B, Hkv, Sq, Sk)`` token logits into ``(B, Hkv, Sq, n_chunk)`` chunk scores.

    Unlike :func:`~.aggregate.aggregate_chunk_scores`, the **ragged tail is kept as a short
    chunk** rather than handed back to the caller. The press can afford to leave the tail at token
    granularity because it is choosing which tokens to evict; here the subset is over a fixed
    ``n_chunk`` axis that the DP's cardinality is defined against, so a tail chunk that sometimes
    exists and sometimes does not would change ``n`` between steps.

    ``mean`` pools over the tail's *real* width, so a 7-token tail is not diluted by 57 zeros --
    which would make it lose every comparison against a full chunk for reasons having nothing to
    do with its content.

    Parameters
    ----------
    token_scores : torch.Tensor
        ``(B, Hkv, Sq, Sk)`` indexer logits, carrying ``MASK_NEG`` on causally-forbidden pairs.
    chunk_size : int
        Tokens per chunk.
    mode : str
        ``mean`` or ``max``.

    Returns
    -------
    torch.Tensor
        ``(B, Hkv, Sq, n_chunk)`` where ``n_chunk = ceil(Sk / chunk_size)``.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if token_scores.dim() != 4:
        raise ValueError(f"expected (B, Hkv, Sq, Sk), got {tuple(token_scores.shape)}")

    *lead, k_len = token_scores.shape
    n_chunk = -(-k_len // chunk_size)
    padded = n_chunk * chunk_size
    mode = (mode or "mean").lower()

    scores = token_scores.to(accumulation_dtype(token_scores.dtype))
    if padded != k_len:
        # Pad with the mask sentinel, not with 0: 0 is a plausible score, so a padded slot would
        # compete against real ones. MASK_NEG loses `max` outright, and is discounted from `mean`
        # by the valid-count division below.
        pad = scores.new_full((*lead, padded - k_len), MASK_NEG)
        scores = torch.cat([scores, pad], dim=-1)
    blocks = scores.reshape(*lead, n_chunk, chunk_size)

    if mode in ("max", "amax"):
        return blocks.amax(dim=-1)
    if mode not in ("mean", "avg"):
        raise ValueError(f"Unknown chunk aggregate mode {mode!r}; use mean or max.")

    valid = blocks > (MASK_NEG / 2)
    count = valid.sum(-1)
    total = torch.where(valid, blocks, torch.zeros_like(blocks)).sum(-1)
    # A chunk with no valid token (entirely in the query's future) scores the sentinel: ranked
    # last, but finite, since MASK_NEG is deliberately finite throughout this package.
    return torch.where(count > 0, total / count.clamp(min=1), torch.full_like(total, MASK_NEG))


def share_over_query_blocks(
    chunk_scores: torch.Tensor, query_block: int, mode: str = "mean"
) -> torch.Tensor:
    """
    Collapse the query axis into blocks: ``(B, Hkv, Sq, n_chunk)`` ->
    ``(B, Hkv, n_qblock, n_chunk)``.

    One subset per block of ``query_block`` consecutive queries. ``mean`` over the block is the
    default and matches ``query_reduce="mean"``, the press's own default -- so what the router is
    trained to score is what the press ranks on.

    A ragged final block (``Sq % query_block != 0``) is pooled over its real width for the same
    reason :func:`pool_scores_to_chunks` does.

    Note this is a *modelling* choice, not a performance one -- see the module docstring's
    measurement. ``query_block=1`` restores per-query selection and is affordable.
    """
    if query_block <= 0:
        raise ValueError(f"query_block must be positive, got {query_block}")
    b, h, q_len, n_chunk = chunk_scores.shape
    if query_block == 1:
        return chunk_scores

    n_block = -(-q_len // query_block)
    padded = n_block * query_block
    scores = chunk_scores
    if padded != q_len:
        pad = scores.new_full((b, h, padded - q_len, n_chunk), MASK_NEG)
        scores = torch.cat([scores, pad], dim=2)
    blocks = scores.reshape(b, h, n_block, query_block, n_chunk)

    mode = (mode or "mean").lower()
    if mode in ("max", "amax"):
        return blocks.amax(dim=3)
    if mode not in ("mean", "avg"):
        raise ValueError(f"Unknown query share mode {mode!r}; use mean or max.")
    valid = blocks > (MASK_NEG / 2)
    count = valid.sum(3)
    total = torch.where(valid, blocks, torch.zeros_like(blocks)).sum(3)
    return torch.where(count > 0, total / count.clamp(min=1), torch.full_like(total, MASK_NEG))


def chunk_visibility(
    n_qblock: int,
    n_chunk: int,
    *,
    query_block: int,
    chunk_size: int,
    q_len: int,
    k_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Which ``(query block, chunk)`` pairs a causal model may select, ``(n_qblock, n_chunk)`` bool.

    A block may select a chunk if **any** query in the block can see **any** token of it, i.e. if
    the chunk starts at or before the block's *last* query. Using the last query rather than the
    first is what makes the mask a valid superset: the block shares one subset, so a chunk that
    only the block's later queries can see must still be selectable, and the per-token causal mask
    inside :func:`exact_k_chunk_attention` then stops the earlier queries from actually reading it.

    Getting this backwards is silent -- attention would still be causal (the token mask enforces
    that) but the router would be forbidden from selecting chunks it is allowed to use, so the
    effective budget would shrink near the diagonal and only long-context recall would show it.
    """
    block_last_query = (
        torch.arange(n_qblock, device=device) * query_block + query_block - 1
    ).clamp(max=q_len - 1)
    query_offset = k_len - q_len
    block_deadline = block_last_query + query_offset  # highest key index the block may read
    chunk_start = torch.arange(n_chunk, device=device) * chunk_size
    return chunk_start.unsqueeze(0) <= block_deadline.unsqueeze(1)


def build_candidates(
    chunk_scores: torch.Tensor,
    n_candidate: int,
    *,
    visible: torch.Tensor,
    n_sink_chunk: int = 1,
    n_local_chunk: int = 1,
    explore_frac: float = 0.10,
    generator: torch.Generator | None = None,
    training: bool = True,
) -> torch.Tensor:
    """
    The candidate pool: ``(B, Hkv, n_qblock, M)`` ascending chunk indices.

    **Why the pool cannot just be top-M.** A chunk outside the pool appears nowhere in the graph,
    so it receives *exactly* zero gradient and can never be promoted -- the same structural dead
    end as the selected-gate proxy, moved up one level. Whatever the estimator does inside the
    pool, the pool itself has to explore.

    The split, and it is a hyperparameter worth sweeping:

    * **structural** -- ``n_sink_chunk`` leading chunks and ``n_local_chunk`` chunks ending at the
      block's own diagonal. These are the slots eviction methods reserve as a matter of course
      (attention sinks; recency), and reserving them here means the router is never *spending*
      budget to rediscover them.
    * **random** -- ``explore_frac`` of the remaining slots, drawn uniformly from the visible
      chunks not already taken. This is what gives an unranked chunk a route into the graph.
    * **top-M** -- everything left, by score.

    At the default ``explore_frac=0.10`` with ``M=64`` and one sink + one local chunk that is
    ~6 random, 2 structural, 56 ranked.

    ``training=False`` drops the random slots entirely and takes pure top-M, which is what an
    evaluation of *this* module should use (the press's own inference path takes a plain top-K and
    does not call this at all).

    Diagnostics worth collecting, per the handoff's §6: the **candidate miss rate** -- how often
    the chunk an oracle would pick is absent from the pool. If that is high, no backward estimator
    can compensate, and the fix is a bigger ``M`` or a different exploration policy rather than
    anything about the gradient.

    Returns
    -------
    torch.Tensor
        ``(B, Hkv, n_qblock, M)`` int64, ascending within a row. Every entry is a *visible* chunk,
        except that a row with fewer than ``M`` visible chunks pads the shortfall with ``-1``.

    Padding is ``-1``, not a repeated chunk
    --------------------------------------
    This matters more than it looks, and repeating a chunk is **wrong**. A duplicated candidate
    occupies two slots of the pool, and the subset's cardinality is over *slots*: the DP would
    happily spend two of its ``K`` on the same chunk, so the row would attend to ``K - 1`` distinct
    chunks while reporting a budget of ``K``. Verified -- the two spellings of "select chunk 0" give
    different outputs, so this is not a harmless redundancy.

    It is also not a rare edge case. Near the diagonal a query block simply cannot see ``M`` chunks
    yet. At ``Sq = Sk = 16384``, ``chunk_size = 64``, ``query_block = 128``, ``M = 64``:
    **31 of 128 blocks (24%)** have fewer than ``M`` visible chunks, and 7 have fewer than ``K``.
    At 8K it is 48%.

    ``-1`` slots are scored at :data:`PAD_SCORE` by :func:`gather_candidate_scores` and masked out
    of the attention entirely, which makes both regimes come out right *because* ``sum(mu) == K``
    is exact:

    * ``V >= K`` -- the pads' marginals are ~0, so the real chunks absorb the whole budget and
      exactly ``K`` distinct chunks are selected.
    * ``V < K`` -- the DP cannot place ``K`` ones among ``V`` plausible slots, so it is forced onto
      the pads; every real chunk gets marginal 1 and is selected. The row attends to all ``V``
      chunks it can see, which is the only sensible answer when the budget exceeds what exists.
    """
    b, h, n_qblock, n_chunk = chunk_scores.shape
    m = min(int(n_candidate), n_chunk)
    if m <= 0:
        raise ValueError(f"n_candidate must be positive, got {n_candidate}")
    device = chunk_scores.device

    # Rank only among visible chunks. -inf rather than MASK_NEG so an invisible chunk cannot be
    # picked even by a row whose visible chunks all sit at the sentinel.
    ranked = chunk_scores.masked_fill(~visible.view(1, 1, n_qblock, n_chunk), -float("inf"))

    n_explore = int(round(explore_frac * m)) if training else 0
    sink = min(max(int(n_sink_chunk), 0), m)
    local = min(max(int(n_local_chunk), 0), max(m - sink, 0))
    n_explore = min(n_explore, max(m - sink - local, 0))

    # A large finite boost, not +inf: `forced` entries must outrank every real score, but the
    # priority tensor is also what breaks ties among the forced ones, and inf ties are unordered.
    boost = 1e6
    priority = ranked.clone()

    if sink:
        # The leading chunks. Visible to every query block, so no mask interaction.
        priority[..., :sink] = boost + torch.arange(sink, 0, -1, device=device, dtype=priority.dtype)

    if local:
        # The `local` chunks ending at each block's own deadline. Per-row, so scatter rather than
        # slice: the diagonal moves with the block.
        last_visible = visible.to(torch.int64).sum(-1) - 1  # (n_qblock,)
        offsets = torch.arange(local, device=device)
        local_idx = (last_visible.view(n_qblock, 1) - offsets.view(1, local)).clamp(min=0)
        local_idx = local_idx.view(1, 1, n_qblock, local).expand(b, h, n_qblock, local)
        priority.scatter_(
            -1, local_idx, torch.full_like(local_idx, boost, dtype=priority.dtype)
        )

    if n_explore:
        # Uniform over visible chunks, then boosted above the ranked pool but below structural.
        noise = torch.rand(
            (b, h, n_qblock, n_chunk), device=device, dtype=priority.dtype, generator=generator
        )
        noise = noise.masked_fill(~visible.view(1, 1, n_qblock, n_chunk), -1.0)
        # Exclude what structural already took, so exploration does not spend its slots there.
        noise = noise.masked_fill(priority >= boost, -1.0)
        explore_idx = noise.topk(n_explore, dim=-1).indices
        priority.scatter_(
            -1, explore_idx, torch.full_like(explore_idx, boost / 2, dtype=priority.dtype)
        )

    picked = priority.topk(m, dim=-1).indices
    # Rows with fewer than m visible chunks: topk resolved those slots onto arbitrary invisible
    # indices, so mark them -1. Sorting puts the pads first, which is harmless -- the attention
    # masks them by value, not by position.
    picked = torch.where(
        priority.gather(-1, picked) > -float("inf"), picked, torch.full_like(picked, -1)
    )
    return picked.sort(dim=-1).values


def gather_candidate_scores(
    chunk_scores: torch.Tensor, candidates: torch.Tensor
) -> torch.Tensor:
    """
    Score the candidate pool, sending ``-1`` pad slots to :data:`PAD_SCORE`.

    Kept as its own function rather than a ``gather`` at each call site, because a plain
    ``chunk_scores.gather(-1, candidates)`` **raises** on the ``-1`` slots, and the obvious repair
    (``candidates.clamp(min=0)``) is silently wrong: it would score the pad with chunk 0's real
    score, so the DP would treat pads as strong candidates and spend budget on them.

    :data:`PAD_SCORE` is low enough that ``sigmoid`` gives ~0 -- so a pad's marginal is ~0 whenever
    a real chunk is available -- but finite, so it cannot produce the ``-inf`` that
    :data:`~.exact_k_subset.NEG_INF` exists to keep out of the DP.
    """
    valid = candidates >= 0
    gathered = chunk_scores.gather(-1, candidates.clamp(min=0))
    return torch.where(valid, gathered, gathered.new_full((), PAD_SCORE))


def _attend_tile(
    q_tile: torch.Tensor,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    g_tile: torch.Tensor,
    safe: torch.Tensor,
    valid: torch.Tensor,
    scale: float,
    tiny: float,
) -> torch.Tensor:
    """
    One query tile's attention: gather this tile's keys/values, then attend. Returns
    ``(B, Hkv, group, nb, qb, Dv)``.

    Split out so :func:`exact_k_chunk_attention` can wrap it in ``torch.utils.checkpoint``. That is
    not a micro-optimization -- the retained ``(B, Hq, nb, qb, M * chunk_size)`` logits are the
    dominant term by a wide margin, measured at **64.4 GiB for a single layer** at
    ``Sq=16384, M=64, chunk_size=64`` on Qwen3-8B geometry. No tile size fixes that on its own,
    because once the graph spans every tile the logits are retained for the *whole* backward.

    **The gather is inside**, deliberately. Passing pre-gathered ``k_sel``/``v_sel`` in would make
    them *inputs* to the checkpointed region, and checkpoint saves its inputs -- so the
    ``(B, Hkv, nb, M * chunk_size, D)`` gathers would be retained across the backward anyway, which
    is the second-largest term. Passing the full ``k_acc``/``v_acc`` plus integer indices instead
    retains only tensors the model already holds.

    Unlike the DP's checkpoint (:class:`~.exact_k_subset._CheckpointedMarginals`, a hand-written
    double-backward), plain ``torch.utils.checkpoint`` works here: this is an ordinary function
    whose output is a value, not a gradient.
    """
    b, n_kv, nb, slots = safe.shape
    head_dim, dim_v = k_full.shape[-1], v_full.shape[-1]
    # Gather in the CALLER's dtype and upcast the gathered slice, not the whole cache. Upcasting
    # k/v/q up front instead costs 640 MiB per layer at Sq=16384 on Qwen3-8B geometry -- 22.5 GiB
    # across 36 layers, retained for the entire backward, which does not fit beside a frozen 8B
    # backbone that already peaks at 89.6 GiB of 95 at that length. The gathered slice is
    # O(M * chunk_size) instead of O(Sk) and dies with the tile.
    acc = accumulation_dtype(q_tile.dtype)
    flat = safe.reshape(b, n_kv, nb * slots, 1)
    k_sel = k_full.gather(2, flat.expand(-1, -1, -1, head_dim)).reshape(
        b, n_kv, nb, slots, head_dim
    ).to(acc)
    v_sel = v_full.gather(2, flat.expand(-1, -1, -1, dim_v)).reshape(
        b, n_kv, nb, slots, dim_v
    ).to(acc)

    logits = torch.einsum("bhgnqd,bhnsd->bhgnqs", q_tile.to(acc), k_sel) * scale
    logits = logits.masked_fill(~valid.unsqueeze(2), -float("inf"))

    # Multiplicative, normalized over the WHOLE pool. Two things here are load-bearing.
    #
    # 1. The row max is over the SELECTED slots only. Taking it over every candidate lets the max
    #    land on a slot with ``g = 0``, whose weight is then discarded -- so every surviving weight
    #    becomes ``exp(a_selected - a_unselected_max)``, which underflows whenever an unselected
    #    chunk has a much larger attention logit than any selected one. ``total`` goes to ~0 and
    #    ``p = w / total`` amplifies without bound. Shifting by the selected max is the standard
    #    softmax shift *for the distribution being computed*: ``g`` is 0 off the subset, so the
    #    unselected logits contribute nothing to the numerator or denominator, and letting one of
    #    them set the shift measures a term that is then multiplied by zero.
    #
    # 2. The shifted exponent is clamped above. An unselected slot's exponent is
    #    ``a_j - a_selected_max``, which is *positive* precisely when chunk ``j`` looks better than
    #    anything currently selected -- and unbounded, so ``exp`` overflows to ``inf``. Then
    #    ``inf * g_j = inf * 0 = NaN`` in the forward, and ``dw_j/dg_j = exp(...) = inf`` in the
    #    backward.
    #
    #    Masking those slots out instead would be wrong, not merely conservative: ``dw_j/dg_j`` for
    #    an unselected candidate **is** the boundary credit this whole method exists to provide (see
    #    the module docstring's 73x table). Zeroing it reproduces the gather-K form's dead end. So
    #    the term is kept and bounded: a selected slot's exponent is ``<= 0`` and is never touched,
    #    while an unselected slot's is capped, which preserves the *sign* and the ranking of "this
    #    chunk should be promoted" and only saturates its magnitude.
    #
    # Both were found the same way. Unfixed, ``grad_norm`` on the real 36-layer model read 2.1e3 at
    # 4 layers, 8.6e13 at 12, ``inf`` at 24 and ``nan`` at 36 -- the indexer parameters are bf16,
    # whose largest finite value is 3.4e38. See ``test_row_max_ignores_unselected_slots`` and
    # ``test_unselected_slot_with_a_huge_logit_stays_finite``.
    selected = (g_tile.detach() > 0).unsqueeze(2).unsqueeze(-2)  # (B, Hkv, 1, nb, 1, slots)
    row_max = logits.masked_fill(~selected, -float("inf")).amax(dim=-1, keepdim=True)
    alive = torch.isfinite(row_max)
    row_max = torch.where(alive, row_max, torch.zeros_like(row_max))
    weights = torch.exp((logits - row_max).clamp(max=MAX_SHIFTED_EXPONENT))
    weights = weights * g_tile.unsqueeze(2).unsqueeze(-2)
    total = weights.sum(-1, keepdim=True)
    # A row whose entire subset is masked (only possible for a query before the first chunk)
    # gets 0 rather than 0/0 = NaN, which would then propagate through the whole model.
    p = torch.where(total > 0, weights / total.clamp_min(tiny), weights * 0.0)
    return torch.einsum("bhgnqs,bhnsd->bhgnqd", p, v_sel)


def gather_candidate_gradient(
    grad: torch.Tensor, candidates: torch.Tensor
) -> torch.Tensor:
    """
    Gather a **gradient** onto the candidate pool, sending ``-1`` pad slots to **zero**.

    Deliberately separate from :func:`gather_candidate_scores`, which sends pads to
    :data:`PAD_SCORE` (-31). That sentinel is correct for a *score* -- it means "this slot is not a
    real candidate, rank it last" -- and **wrong for a gradient**, where the honest value is 0: a pad
    slot has no chunk, so no chunk's parameters receive anything through it.

    Using the score helper on a gradient is a real trap, not a hypothetical: it put a constant -31
    into a swap-oracle prediction and moved the reported median (``bias``) to exactly -31.0, which is
    what exposed it. A diagnostic contaminated that way still produces plausible correlations.
    """
    valid = candidates >= 0
    gathered = grad.gather(-1, candidates.clamp(min=0))
    return torch.where(valid, gathered, torch.zeros_like(gathered))


def exact_k_chunk_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidates: torch.Tensor,
    *,
    topk_chunk: int,
    chunk_size: int,
    query_block: int,
    scaling: float | None = None,
    query_offset: int | None = None,
    checkpoint: bool = True,
    checkpoint_attention: bool = True,
    generator: torch.Generator | None = None,
    hard: bool = False,
    query_tile: int = 0,
) -> tuple[torch.Tensor, dict]:
    """
    Attention over an exact-``K`` chunk subset, differentiable in ``candidate_scores``.

    Parameters
    ----------
    q : torch.Tensor
        ``(B, Hq, Sq, D)`` post-RoPE queries.
    k, v : torch.Tensor
        ``(B, Hkv, Sk, D)`` / ``(B, Hkv, Sk, Dv)``.
    candidate_scores : torch.Tensor
        ``(B, Hkv, n_qblock, M)`` router scores for the candidate chunks, **with gradient**. This
        is what the LM loss trains.
    candidates : torch.Tensor
        ``(B, Hkv, n_qblock, M)`` ascending chunk indices from :func:`build_candidates`.
    topk_chunk : int
        ``K``: chunks each query block commits to. Must be ``<= M``.
    chunk_size, query_block : int
        Geometry, as used to build ``candidate_scores``.
    scaling : float, optional
        Softmax scale; defaults to ``D ** -0.5``.
    query_offset : int, optional
        Key index of query 0's diagonal. Defaults to ``Sk - Sq``, matching flash-attention and
        :func:`~.indexer.build_indexer_mask`.
    checkpoint : bool
        Recompute the marginals' DP in the backward. Leave True; see
        :class:`~.exact_k_subset._CheckpointedMarginals` for the 25x memory difference.
    checkpoint_attention : bool
        Recompute each query tile's attention in the backward. Leave True: the retained
        ``(B, Hq, Sq, M * chunk_size)`` logits are 64.4 GiB for **one** layer at
        ``Sq=16384, M=64, chunk_size=64``, which no tile size fixes by itself. See
        :func:`_attend_tile`.
    generator : torch.Generator, optional
        RNG for the subset sample.
    hard : bool
        Deterministic top-K instead of sampling. For measuring the sampling variance, not for
        training -- a deterministic selection does not explore.
    query_tile : int
        Query blocks processed at once, bounding the gathered ``(tile, M * chunk_size, D)``
        tensors. ``0`` picks a tile that keeps that under ~2 GiB.

    Returns
    -------
    out : torch.Tensor
        ``(B, Hq, Sq, Dv)`` in ``q``'s dtype.
    stats : dict
        Diagnostics that a loss curve cannot give you:
        ``marginal_entropy`` (how undecided the router is; ``log K`` at init, falling as it
        commits), ``selected_chunks`` (K, as a wiring check), and ``jaccard`` -- left absent here
        and filled by the trainer, which is the only thing that sees consecutive steps.

    Notes
    -----
    The normalization runs over the whole candidate pool rather than the selected ``K``. Those are
    numerically identical in the forward (``g`` is 0 off the subset) and very different in the
    backward -- 73x more gradient on unselected candidates. See the module docstring.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, v must be 4D (B, H, S, D)")
    b, n_heads, q_len, head_dim = q.shape
    _, n_kv, k_len, _ = k.shape
    if n_heads % n_kv:
        raise ValueError(f"Hq {n_heads} is not a multiple of Hkv {n_kv}")
    if candidates.shape != candidate_scores.shape:
        raise ValueError(
            f"candidates {tuple(candidates.shape)} and candidate_scores "
            f"{tuple(candidate_scores.shape)} must agree"
        )
    n_qblock, m = candidates.shape[-2:]
    if candidates.shape[:2] != (b, n_kv):
        raise ValueError(f"candidates must lead with (B={b}, Hkv={n_kv}), got {tuple(candidates.shape)}")
    if not 0 < topk_chunk <= m:
        raise ValueError(f"topk_chunk must be in (0, M={m}], got {topk_chunk}")

    group = n_heads // n_kv
    scale = head_dim**-0.5 if scaling is None else float(scaling)
    if query_offset is None:
        query_offset = k_len - q_len
    dim_v = v.shape[-1]

    # The straight-through mask, once for the whole pool. This is the only place the DP runs.
    g, z, mu = straight_through_mask(
        candidate_scores, topk_chunk, checkpoint=checkpoint, generator=generator, hard=hard
    )

    acc = accumulation_dtype(q.dtype)
    itemsize = torch.finfo(acc).bits // 8

    if query_tile <= 0:
        # Sized against the LOGITS, not the gathered keys. The logits are
        # (B, Hq, nb, qb, M * chunk_size) -- a factor `group * query_block` larger than
        # (B, Hkv, nb, M * chunk_size, D) whenever query_block > D / group, which it is at the
        # default 128. Sizing off the keys was the first attempt and it left the tile large enough
        # that checkpointing bought nothing: peak stayed at 69 GiB because one tile's logits were
        # already the whole problem.
        per_block = b * n_heads * query_block * m * chunk_size * itemsize
        query_tile = max(1, min(n_qblock, TILE_LOGIT_BYTES // max(per_block, 1)))

    # q/k/v stay in the CALLER's dtype here; the per-tile upcast happens inside _attend_tile, which
    # is what keeps the fp32 copies from being retained for the whole backward. (The arithmetic is
    # still at least fp32, and never narrower than the input -- the reference tests run in fp64 so
    # their tolerances measure floating-point noise rather than a silent downcast.)
    token_offset = torch.arange(chunk_size, device=q.device)
    q_view = q.view(b, n_kv, group, q_len, head_dim)
    tiny = torch.finfo(acc).tiny
    slots = m * chunk_size

    # Collected and concatenated rather than written into a preallocated `out`. An in-place write
    # into a tensor that requires grad is a graph mutation, and under checkpointing the recomputed
    # tile would write into an already-consumed buffer.
    tiles: list[torch.Tensor] = []

    for start in range(0, n_qblock, query_tile):
        stop = min(start + query_tile, n_qblock)
        nb = stop - start
        q_start = start * query_block
        q_stop = min(stop * query_block, q_len)
        if q_stop <= q_start:
            continue
        # Queries of the last tile may not fill their blocks; qb is the per-block query count and
        # the tail is trimmed after the einsum rather than special-cased inside it.
        qb = query_block
        q_pad = nb * qb - (q_stop - q_start)

        cand = candidates[:, :, start:stop]  # (B, Hkv, nb, M)
        # Chunk index -> the token positions it covers. (B, Hkv, nb, M * chunk_size)
        # A -1 pad slot has no chunk at all; clamp it for the gather and drop it via `real` below.
        real = (cand >= 0).unsqueeze(-1).expand(-1, -1, -1, -1, chunk_size).reshape(
            b, n_kv, nb, slots
        )
        pos = (cand.clamp(min=0).unsqueeze(-1) * chunk_size + token_offset).reshape(
            b, n_kv, nb, slots
        )
        usable = real & (pos < k_len)
        # Out-of-range and pad positions are clamped for the gather and masked afterwards.
        # Gathering a real row and masking after is what sparse_gqa_attention_reference does too,
        # and for the same reason: an out-of-bounds gather raises, while masking is free.
        safe = torch.where(usable, pos, torch.zeros_like(pos))

        # Queries reshaped to (B, Hkv, group, nb, qb, D). Padding the tail with zeros is safe: its
        # rows are dropped below and a zero query cannot make another row's output wrong.
        q_tile = q_view[:, :, :, q_start:q_stop]
        if q_pad:
            q_tile = torch.cat([q_tile, q_tile.new_zeros(b, n_kv, group, q_pad, head_dim)], dim=3)
        q_tile = q_tile.reshape(b, n_kv, group, nb, qb, head_dim)

        # Per-(query, slot) causal validity. The block shares one subset, so this is where the
        # block's earlier queries are stopped from reading a chunk only its later queries can see
        # -- chunk_visibility deliberately admits those chunks at block granularity.
        q_pos = (
            torch.arange(nb * qb, device=q.device) + q_start + query_offset
        ).reshape(nb, qb)
        valid = (q_pos.view(1, 1, nb, qb, 1) >= safe.view(b, n_kv, nb, 1, slots)) & usable.view(
            b, n_kv, nb, 1, slots
        )

        # g broadcast from chunk to the tokens it covers.
        g_tile = g[:, :, start:stop].repeat_interleave(chunk_size, dim=-1)  # (B, Hkv, nb, slots)

        args = (q_tile, k, v, g_tile, safe, valid, scale, tiny)
        if checkpoint_attention and torch.is_grad_enabled():
            tile_out = torch_checkpoint(_attend_tile, *args, use_reentrant=False)
        else:
            tile_out = _attend_tile(*args)
        tile_out = tile_out.reshape(b, n_kv, group, nb * qb, dim_v)
        # Narrowed to the caller's dtype per tile rather than at the end: the concatenated fp32
        # output is 256 MiB per layer at Sq=16384 on Qwen3-8B geometry, and it is retained for the
        # whole backward. The tile's own arithmetic already ran in fp32, so this only decides what
        # the graph holds -- the same trade _attend_tile's gather makes.
        tiles.append(tile_out[:, :, :, : q_stop - q_start].to(q.dtype))

    stats = {
        "selected_chunks": int(topk_chunk),
        "candidate_pool": int(m),
        # How many of the K selected slots hold a REAL chunk rather than a -1 pad. Near the
        # diagonal a query block cannot see M chunks yet (24% of blocks at Sq=16K, chunk 64,
        # query_block 128, M=64), and when it cannot see K either the DP is forced to place some of
        # its budget on pads. That is the correct behaviour -- the row attends to every chunk it
        # can see -- but it means the effective budget is below K for those rows, and nothing else
        # would tell you. Report it rather than let a run silently train at a budget it did not ask
        # for.
        "effective_topk": float((z.detach() * (candidates >= 0)).sum(-1).float().mean()),
        # How undecided the router is. At init the marginals are uniform at K/M, so this is at its
        # maximum; falling means the router is committing. This is the readout `gate_sparsity` is
        # for the additive path -- a loss curve cannot distinguish a router that ranks from one
        # that is still spreading its mass evenly.
        "marginal_entropy": float(_marginal_entropy(mu)),
        "marginal_max": float(mu.detach().amax()),
        "selected_fraction": float(topk_chunk) / float(m),
        # Kept detached and on-device: the trainer compares it against the previous step's to
        # report selection stability (Jaccard), which needs the actual set, not a summary.
        "selected": z.detach(),
    }
    out = torch.cat(tiles, dim=3) if len(tiles) > 1 else tiles[0]
    return out.reshape(b, n_heads, q_len, dim_v), stats


def _marginal_entropy(mu: torch.Tensor) -> torch.Tensor:
    """
    Mean Bernoulli entropy of the marginals, in nats.

    Per-item Bernoulli rather than a categorical entropy over the pool, because the marginals do
    not sum to 1 -- they sum to ``K``. ``log 2`` per item at ``mu = 0.5``, 0 when every item is
    decided. This is the quantity that says whether the router has *committed*, which neither the
    loss nor ``sum(mu) == K`` can tell you.
    """
    p = mu.detach().float().clamp(1e-7, 1 - 1e-7)
    return -(p * p.log() + (1 - p) * (1 - p).log()).mean()


def selection_jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Mean Jaccard overlap between two 0/1 selection masks of the same shape.

    The handoff's §6 asks for this: ``torch.bernoulli`` makes the forward stochastic, and the
    question is whether adjacent steps select wildly different chunk sets. Both masks hold exactly
    ``K`` ones, so ``|A ∩ B| / |A ∪ B| = i / (2K - i)`` -- reported per row and averaged.

    Returns ``float('nan')`` for mismatched shapes rather than raising: the shape changes
    legitimately at a curriculum boundary (``Sq`` changes, so ``n_qblock`` does), and a diagnostic
    should not be able to stop a training run.
    """
    if a.shape != b.shape:
        return float("nan")
    inter = (a * b).sum(-1)
    union = a.sum(-1) + b.sum(-1) - inter
    return float((inter / union.clamp(min=1)).mean())
