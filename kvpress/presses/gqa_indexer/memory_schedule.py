# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
When each key enters the memory, and the per-query-block state that follows.

At inference the answer is trivial: the press compresses once at the end of prefill, so there is
one evicted set and one ``(H, z, W)``. Training wants more than that. Under a one-shot schedule
every supervised row shares a single evicted set, which gives ``psi`` -- the deeper of the two
kernels, the one deciding *what to store* -- exactly **one** eviction event per sequence to learn
from.

The streaming schedule instead reuses the fact that :func:`~.qi_flex_attention.deadlines` already
computed the ingestion time. Its contract is that a row with horizon ``hi_t`` retains key ``j`` iff
``hi_t <= deadline[j]``, so

    ``j in E_t  <=>  horizon_t > deadline[j]``     i.e. ``j`` enters at ``horizon = deadline[j]+1``

and every query row gets its own evicted set, nested and growing. Three things fall out:

* ``L`` supervised positions instead of a handful.
* ``psi`` sees ~``L`` distinct evicted sets rather than one.
* **length generalisation is trained inside the sequence.** At a fixed budget ``b``, row ``t`` has
  ``|E_t| = t - b``, so a single 32K sequence covers ``|E|`` from 0 to ``32K - b`` continuously.
  That is what makes the explicit ``|E|`` normalization (invariant 2 in :mod:`~.memory`) trainable
  rather than merely correct.

Sink and local protection need no special handling: :func:`deadlines` gives a sink
``deadline = k_len - 1`` so it never enters, and a local key sits outside the horizon.

Block granularity, and why it is free
-------------------------------------
The state is held per **query block** of :data:`~.qi_flex_attention.FLEX_BLOCK` (128) rows, with a
key alive for the block iff ``horizon(block_start) <= deadline[j]``. This is the same granularity
``create_block_mask`` already quantizes the mask to, so quantizing the partition to match costs
nothing *and* makes "no key double-counted, none lost" true by construction rather than by
arithmetic that has to be checked. Row-exact partitioning would need an intra-block correction
term (the ``<= BT`` keys whose status changes mid-block) inside the kernel; deliberately not done
in this version.

Because the evicted set grows monotonically with the block index, the per-block state is a plain
prefix sum::

    enter[j] = deadline[j] + 1
    entry[j] = first block whose horizon reaches max(enter[j], j)
    P[b]     = sum_{entry[j] == b} w_j psi(k_j)^T v_j            (scatter-add, one pass over keys)
    H_b      = cumsum(P)[b]

The ``max(enter[j], j)`` is not defensive padding -- see :func:`entry_block`. A key the router ranks
last gets ``deadline = -1`` and so ``enter = 0``, i.e. "evicted as of block 0", but at block 0 that
key has not been reached yet; ingesting it there folds a future value vector into a state every
earlier row reads. The same term also keeps keys inside the protected local window out of the
memory, since the mask retains those unconditionally. ``test_partition_is_exact`` caught both.

``psi`` is evaluated once per key -- every key is eventually evicted, so there is nothing to skip
-- at ``O(L)``, and its gradient accumulates over every query block after the one it entered. At
``L=32K`` with ``BT=128`` there are ``NT=256`` blocks, so the materialized state is
``256 x 8 x 16 x 128`` fp32 ~ **17 MB/layer**, which is small beside the 1.38 GiB/layer the
block-sparse attention backward itself retains at 16K.
"""

from __future__ import annotations

import logging

import torch

from kvpress.presses.gqa_indexer.qi_flex_attention import FLEX_BLOCK

logger = logging.getLogger(__name__)


def ingestion_horizon(deadline: torch.Tensor) -> torch.Tensor:
    """
    ``enter[h, j] = deadline[h, j] + 1`` -- the horizon at which key ``j`` enters the memory.

    A one-line function on purpose: it is the single place the sign convention linking
    :func:`~.qi_flex_attention.deadlines` to this module is written down, and getting it off by one
    would silently shift every key's ingestion by one query row -- a mistake that changes no shape
    and raises nothing.

    ``deadline = k_len - 1`` (never evicted, which is what a sink gets) maps to ``enter = k_len``,
    which no horizon ever reaches, so such keys are never ingested. ``deadline = -1`` (never
    selected at all) maps to ``enter = 0``: ingested from the very first block, which is right.
    """
    return deadline.to(torch.int64) + 1


def block_horizons(
    q_len: int, k_len: int, *, block: int = FLEX_BLOCK, n_local: int = 0, device=None
) -> torch.Tensor:
    """
    Each query block's horizon, ``(n_blocks,)`` int64 -- taken at the block's **first** row.

    Matching :func:`~.qi_flex_attention.qi_block_mask`'s ``horizon = min(q_i + offset, k_len-1) -
    force_local``, evaluated at ``q_i = block_start``. The first row is the conservative choice: it
    is the smallest horizon in the block, hence the smallest evicted set, so a key is counted as
    ingested for a block only when it has been evicted for *every* row of that block. The
    alternative (last row) would credit the memory with keys some rows still hold in the exact
    branch, which is the double-counting invariant 1 of :mod:`~.memory` cares about.
    """
    offset = k_len - q_len
    starts = torch.arange(0, q_len, block, device=device, dtype=torch.int64)
    return (starts + offset).clamp(max=k_len - 1) - int(n_local)


def entry_block(enter: torch.Tensor, horizons: torch.Tensor) -> torch.Tensor:
    """
    Which query block each key enters the memory at, ``(n_heads, Sk)`` int64.

    The first block whose horizon reaches **both** of two thresholds, and both are load-bearing:

    * ``enter[j] = deadline[j] + 1`` -- the router has evicted ``j``;
    * ``j`` itself -- ``j`` has left the protected local window and entered the ranked pool.

    Taking only the first is wrong in a way that produces no error and no shape mismatch, and it
    fails twice over. A key the router ranks last gets ``deadline = -1``, hence ``enter = 0``, hence
    "evicted as of block 0" -- but at block 0 that key is in the sequence's *future*. Ingesting it
    there folds a value vector the model has not reached into a state that every earlier row reads,
    i.e. future information flowing backwards: training loss improves, evaluation does not, nothing
    raises. The same comparison also covers the local window, since
    ``qi_block_mask`` retains ``j`` whenever ``j > horizon`` regardless of its deadline -- so a
    freshly-arrived key is held by the exact branch and must not also be in the memory. Both were
    caught by ``test_partition_is_exact`` (the second only in the ``n_local > 0`` case).

    ``n_blocks`` means "never, within this sequence" -- where sinks land (``deadline = k_len - 1``
    gives an unreachable ``enter``) and keys still retained at the end. Callers index prefix sums of
    length ``n_blocks + 1`` with this, so the sentinel is a valid slot rather than a filter.

    One ``searchsorted`` over the monotone horizons rather than an ``(n_blocks, Sk)`` comparison: at
    ``L=32K`` that matrix is 256 x 32768 per head.
    """
    if enter.dim() != 2:
        raise ValueError(f"enter must be (n_heads, Sk), got {tuple(enter.shape)}")
    n_heads, k_len = enter.shape
    keys = torch.arange(k_len, device=enter.device, dtype=torch.int64)
    threshold = torch.maximum(enter, keys.view(1, -1).expand(n_heads, -1))
    # horizons is non-decreasing by construction (clamped affine in the block index), which is what
    # searchsorted requires. `left` gives the first block reaching the value, matching the strict
    # `horizon > deadline` test.
    return torch.searchsorted(
        horizons.contiguous(), threshold.reshape(-1).contiguous(), right=False
    ).view_as(enter)


def evicted_counts(entry: torch.Tensor, n_blocks: int) -> torch.Tensor:
    """
    ``|E_b|`` per KV head and query block, ``(n_heads, n_blocks)`` int64.

    How many keys have entered the memory by block ``b``, which is the ``|E_t|``
    :meth:`~.memory.MemoryKernel.read` needs -- and needs *explicitly*, because ``H`` and ``z``
    grow linearly in it while the exact branch's ``D_S`` is bounded by the budget.

    A histogram over :func:`entry_block` followed by a prefix sum, which is ``O(Sk)`` and exact
    because a key enters exactly once and never leaves. The "never leaves" half is the
    irreversibility :mod:`~.scalar_indexer` measures (0 re-entries over 1500 steps with the
    position tilt on); without it this count would have to subtract and no single accumulated state
    could represent the evicted set at all.

    What matters is agreement with the attention mask, not the count in isolation: a key counted
    here must be absent from block ``b``'s block-sparse mask and vice versa. Both derive from the
    same ``deadline`` and the same per-block limits, so they agree by construction --
    ``test_partition_is_exact`` still checks it per (head, block), because leaked or
    double-counted mass changes no shape and raises nothing.
    """
    n_heads = entry.shape[0]
    # +1 column absorbs the "never enters" sentinel, then is dropped.
    hist = torch.zeros((n_heads, n_blocks + 1), dtype=torch.int64, device=entry.device)
    hist.scatter_add_(1, entry.clamp(max=n_blocks), torch.ones_like(entry))
    return hist[:, :n_blocks].cumsum(-1)


def block_memory_states(
    kernel,
    k: torch.Tensor,
    v: torch.Tensor,
    deadline: torch.Tensor,
    *,
    q_len: int,
    block: int = FLEX_BLOCK,
    n_local: int = 0,
    scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-query-block memory state for the streaming schedule.

    Returns ``(H, z, W, counts)`` with shapes ``(B, Hkv, n_blocks, R, D)``,
    ``(B, Hkv, n_blocks, R)``, ``(B, Hkv, n_blocks)`` and ``(Hkv, n_blocks)``.

    Parameters
    ----------
    kernel : MemoryKernel
        Supplies ``psi``, the ingestion weights and the rank.
    k, v : torch.Tensor
        ``(B, Hkv, Sk, D)`` post-RoPE keys and values -- the cache contents.
    deadline : torch.Tensor
        ``(Hkv, Sk)`` from :func:`~.qi_flex_attention.deadlines`. **Must be detached**: the
        argsort/top-k inside it carries no gradient, and leaving it attached would retain that
        graph for nothing. Enforced here rather than trusted.
    q_len, block, n_local : int
        Query length, block width, and the local window the mask reserves -- all three must match
        what :func:`~.qi_flex_attention.qi_block_mask` is built with, or the partition and the mask
        will disagree.
    scores : torch.Tensor, optional
        ``(Hkv, Sk)`` router scores, for the ``g(s~)`` ingestion weighting. Unused at ``a = 0``.

    Notes
    -----
    The prefix sum is over the block axis, exclusive-shifted so block ``b`` sees keys ingested
    *strictly before* it -- which is what "``j`` enters at ``horizon = deadline[j]+1``" means: a key
    entering at block ``b`` is already absent from block ``b``'s attention mask, so it belongs to
    block ``b``'s memory. Hence ``cumsum`` inclusive of ``b`` here, and ``block_of`` being the
    first block for which the key is *already* evicted.
    """
    if deadline.requires_grad:
        raise ValueError(
            "deadline must be detached: it comes from an argsort/top-k that carries no gradient, "
            "so keeping the graph retains the selection pass's activations for a term that can "
            "never contribute one."
        )
    bsz, n_kv_heads, k_len, head_dim = k.shape
    if deadline.shape != (n_kv_heads, k_len):
        raise ValueError(
            f"deadline {tuple(deadline.shape)} does not match k (Hkv={n_kv_heads}, Sk={k_len})"
        )
    device = k.device
    n_blocks = (q_len + block - 1) // block

    enter = ingestion_horizon(deadline)
    horizons = block_horizons(q_len, k_len, block=block, n_local=n_local, device=device)
    entry = entry_block(enter, horizons)  # (Hkv, Sk), n_blocks = never
    counts = evicted_counts(entry, n_blocks)  # (Hkv, n_blocks)

    w = kernel.ingest_weights(enter, k_len, scores=scores)  # (Hkv, Sk) fp32

    if kernel.rank_zero:
        feat = torch.ones(bsz, n_kv_heads, k_len, 1, device=device, dtype=torch.float32)
    else:
        feat = kernel.psi(k).float()  # (B, Hkv, Sk, R)
    rank = feat.shape[-1]
    # w folded into psi ONCE, so H, z and W all carry it identically -- invariant 1 of ~.memory.
    feat_w = feat * w.unsqueeze(0).unsqueeze(-1)
    # RAW v, deliberately -- see MemoryKernel.ingest. `n/d` is a convex combination of these
    # vectors, so it lands in the same space as `oE*` only if they are unnormalized; a rescaling here
    # cannot be undone by gamma, which cancels out of the ratio.
    v_n = v.float()

    slot = entry.clamp(max=n_blocks)  # (Hkv, Sk); n_blocks column absorbs "never", then dropped
    P, Pz = _bucket_sums(feat_w, v_n, slot, n_blocks)
    Pw = w.new_zeros((n_kv_heads, n_blocks + 1))
    Pw.scatter_add_(1, slot, w)
    Pw = Pw[:, :n_blocks].unsqueeze(0).expand(bsz, -1, -1)

    # Inclusive cumsum: a key entering at block b is already absent from block b's attention mask,
    # so it belongs to block b's memory -- which is what "enters at horizon deadline[j]+1" means.
    return P.cumsum(2), Pz.cumsum(2), Pw.cumsum(2), counts


#: Keys per chunk in :func:`_bucket_sums`. Bounds the transient outer product to
#: ``chunk * R * D`` per head instead of ``Sk * R * D``.
BUCKET_KEY_CHUNK = 2048


def _bucket_sums(
    feat_w: torch.Tensor,
    v: torch.Tensor,
    slot: torch.Tensor,
    n_blocks: int,
    *,
    chunk: int = BUCKET_KEY_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sum ``w_j psi(k_j) (x) v_j`` and ``w_j psi(k_j)`` into per-entry-block buckets.

    Returns ``(B, Hkv, n_blocks, R, D)`` and ``(B, Hkv, n_blocks, R)``.

    Structured this way because the obvious one-liner does not fit. Forming the outer product for
    every key at once -- ``feat_w.unsqueeze(-1) * v.unsqueeze(-2)``, shape
    ``(B, Hkv, Sk, R, D)`` -- is minimal *work*, but autograd **retains** it for the backward: at
    ``Sk=8192, Hkv=8, R=16, D=128`` that is 0.5 GiB per layer in fp32, ~19 GiB across 36 layers on
    top of the attention's own state, and it OOM'd an H20 at 8K (the traceback lands in an unrelated
    ``mlp`` forward, which is what makes this kind of retention annoying to attribute).

    Two things fix it together:

    * **chunking over keys**, so the transient is ``chunk * R * D`` rather than ``Sk * R * D``. The
      bucket sums are additive, so a chunked accumulation is exact rather than approximate.
    * **``checkpoint``**, so no chunk's outer product is retained at all -- they are recomputed
      during backward. Without this, chunking alone changes nothing: the retained total is the same
      whether it arrives in one tensor or sixteen.

    Total work stays ``O(Sk R D)``, i.e. the minimum. The alternative formulations trade that away:
    a per-block masked matmul is ``O(n_blocks Sk R)``, which at ``L=32K`` is ~30x more arithmetic
    *and* a bigger transient.
    """
    from torch.utils.checkpoint import checkpoint

    bsz, n_kv_heads, k_len, rank = feat_w.shape
    head_dim = v.shape[-1]

    def build(feat_w: torch.Tensor, v: torch.Tensor):
        P = v.new_zeros((bsz, n_kv_heads, n_blocks + 1, rank, head_dim))
        Pz = v.new_zeros((bsz, n_kv_heads, n_blocks + 1, rank))
        for start in range(0, k_len, chunk):
            stop = min(start + chunk, k_len)
            f = feat_w[:, :, start:stop]  # (B, Hkv, c, R)
            idx = slot[:, start:stop]  # (Hkv, c)
            contrib = f.unsqueeze(-1) * v[:, :, start:stop].unsqueeze(-2)  # (B,Hkv,c,R,D)
            P.scatter_add_(
                2,
                idx.view(1, n_kv_heads, -1, 1, 1).expand(bsz, -1, -1, rank, head_dim),
                contrib,
            )
            Pz.scatter_add_(
                2, idx.view(1, n_kv_heads, -1, 1).expand(bsz, -1, -1, rank), f
            )
        return P[:, :, :n_blocks], Pz[:, :, :n_blocks]

    if not torch.is_grad_enabled() or not feat_w.requires_grad:
        return build(feat_w, v)
    # use_reentrant=False: the reentrant implementation cannot handle a function whose inputs do not
    # all require grad (v comes off a frozen backbone), and it is the deprecated path regardless.
    return checkpoint(build, feat_w, v, use_reentrant=False)


def expand_blocks_to_rows(
    per_block: torch.Tensor, q_len: int, *, block: int = FLEX_BLOCK, dim: int = 2
) -> torch.Tensor:
    """
    Broadcast a per-block tensor to per-row along ``dim``, trimming the ragged tail.

    ``repeat_interleave`` rather than a view: the row axis is what the fusion indexes, and a
    non-multiple ``q_len`` (every RULER context) leaves a partial final block that has to be cut
    rather than wrapped.
    """
    expanded = per_block.repeat_interleave(block, dim=dim)
    return expanded.narrow(dim, 0, q_len)
