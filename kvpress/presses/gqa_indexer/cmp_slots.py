# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Training-free CMP slots: cluster the keys the router drops, keep one pseudo-KV per cluster.

The router throws away ~22% of every row's softmax mass (``rho`` 0.213/0.219/0.234 at 4K/8K/32K,
flat in length). This module spends a small slice of the read budget buying some of it back, with
**zero learned parameters**::

    prefill -> router score -> top-k     the ``topk - R`` keys the row reads exactly
                            \\-> rest -> k-means into R clusters
                                         slot r = (kbar_r, vbar_r, b_r)

and the slots join the *same* softmax as the retained keys, so there is no second branch, no fusion
gate and no ``lse`` bookkeeping -- ``[K_topk ; K_cmp]`` is one attention call.

Why clustering rather than positional chunks
--------------------------------------------
The earlier arm gave each *positional* chunk one slot and failed: measured cos to the true evicted
output 0.702 (L0) / 0.680 (L18) against 0.861 / 0.846 for k-means at the same slot count. A
positional span mixes unrelated value vectors, so its mean is a poor summary at any slot count,
while a content cluster is coherent -- whichever member a query wanted, the cluster mean is close.
At layers 0/18/35 k-means CMP **reaches the ceiling** set by an oracle that sees the target and the
future queries (0.861 vs 0.861, 0.846 vs 0.842, 0.955 vs 0.965).

Two facts make the summary cheap rather than lossy, and both were measured rather than assumed:

* Each row's evicted mass is spread over **60-466 keys** (median participation ratio), and 24-601
  even on the ``rho > 0.3`` rows. The target is already an average over hundreds of vectors, so
  replacing them with R centroids loses little. Had it been few-key associative recall, no
  compressed summary could work.
* Rank is not the bottleneck and neither is nonlinearity: the best *linear* map from ``q`` to the
  target ties the best nonlinear one (cos 0.83-0.95 either way), and a rank-64 random-feature state
  scores the same as rank-16 (0.590 vs 0.575). Capacity was never what was missing.

post-RoPE, and why there is no choice about it
----------------------------------------------
Clustering and averaging happen on **post-RoPE keys**, i.e. the cache contents. The logit lives in
that space, so

    ``q . mean_j(R_j k_j) = mean_j(q . R_j k_j)`` = the cluster's true mean logit,

first-order exact. A pre-RoPE centroid would have to be rotated to enter attention, which requires a
position -- and a cluster drawn from across the sequence has none to give, while
``R_jbar kbar != mean(R_j k_j)`` anyway. Averaging post-RoPE is safe here because RoPE does not
decohere over a span: measured ``||kbar|| / mean||k_j|| = 0.84-0.94``. As a side effect the distance
metric partly encodes position, so clusters come out position-local without being told to.

The mass term: ``log n_r`` is the answer, and this is settled
------------------------------------------------------------
``b_r`` is the log of how many tokens the slot stands for. The Jensen argument says it must be biased
*low* -- ``log sum_j exp(q.k_j) = log n_r + q.kbar_r + 1/2 Var_j + ...`` -- so the obvious move is to
add the variance term back. **Measured, that is wrong in the opposite direction.** RULER 8K at R=64,
budget-matched against 76.65:

======================================  =======  =======
 ``b_r``                                 MEAN     delta
======================================  =======  =======
 ``log n_r``                             77.15    +0.49
 ``log n_r + 1/2 Var`` (analytic)        76.82    +0.16
 ``log n_r + 1``                         75.89    -0.76
 ``log n_r + 2``                         74.92    -1.73
 learned (3 scalars/head, 864 params)    76.73    +0.08
======================================  =======  =======

Monotonically decreasing: ``log n_r`` sits at or past the optimum and **every richer mass model
loses**. The per-layer reconstruction check agrees independently -- ``count`` beats eviction on 36/36
layers while the analytic ``count+var`` loses on 7 and is worse than adding no slot at all in total
L2. The Laplace lower bound is real; the higher cumulants it drops dominate it in practice.

So :class:`CMPMassHead` exists for the record rather than for use, and the "check ``cos >= 0.8``
before letting the slot speak" caution is subsumed: at ``log n_r`` the slot is quiet enough that no
layer is hurt, which is why the zero-parameter version never loses one.

Decode only, in this version
----------------------------
The slots are built once, after prefill, and read only by rows that come *after* every clustered
key. That is causal by construction. It is deliberately **not** applied to prefill rows, because
under ``qi_flex_attention`` eviction is per-row -- row ``t`` drops key ``j`` iff
``horizon_t > deadline[j]`` -- so a single slot set shared by every prefill row would let early rows
read centroids built from their own future. That exact mistake took RULER 8K from 73.71 to **4.00**
with a training curve that looked healthy throughout. The causal prefill variant is a prefix-sum
over :func:`~.memory_schedule.entry_block` and is a separate step; :func:`cluster_evicted` already
accepts the per-row ``deadline`` so it can be reused there unchanged.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

#: Default slot count. 64 slots against the ~6K keys a 8K/topk-2048 row drops is a 96x summary, and
#: it costs 64 of the 2048 reads (3.1%) since the slots are paid for out of ``topk`` rather than
#: added to it.
#:
#: **Measured optimum, and non-monotonic**, RULER 8K budget-matched against a 76.65 baseline:
#: R=16 gives 76.69, R=64 **77.15**, R=256 76.67. Note this *disagrees* with the per-layer
#: reconstruction probe, which had R=256 clearly best (layer 35: 4.81x against 3.14x at R=64) -- more
#: slots reconstruct the evicted output better while scoring no better end-to-end, because they are
#: funded out of the exact budget and because each slot's ``log n_r`` is a noisier claim when clusters
#: are small.
#:
#: The per-task split shows what R actually trades: ``cwe`` (aggregation) rises monotonically
#: 83.72 -> 87.21 -> 93.02, while ``niah_multikey_2`` (retrieval) peaks at R=64 and collapses at
#: R=256 (35.14 -> 21.62). So pick R per workload; the mean over RULER's 13 tasks partly cancels the
#: effect it is measuring.
DEFAULT_N_SLOTS = 64

#: Lloyd iterations. The assignment stops moving well before this on real key distributions; it is
#: cheap (``O(iters * |E| * R * D)`` once per layer per prefill) so there is no reason to skimp.
DEFAULT_KMEANS_ITERS = 15

#: Floor on a cluster's population before its slot is emitted at all. A singleton cluster's
#: "summary" is the token itself, which the row already decided not to read; emitting it spends a
#: slot to re-add one evicted key, which is strictly worse than giving that slot to the top-k.
MIN_CLUSTER = 1


def unrotate(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Recover pre-RoPE keys from the cache's post-RoPE ones. ``(H, S, D)`` with ``cos/sin`` ``(S, D)``.

    Exact, not an approximation: RoPE is an orthogonal rotation, so its inverse is the rotation by
    the negated angle. HF applies ``k' = k*cos + rot(k)*sin`` with
    ``rot(x) = cat(-x[d/2:], x[:d/2])``, and ``cos(-t)=cos(t), sin(-t)=-sin(t)`` gives

        ``k = k'*cos - rot(k')*sin``.

    Needed because the KV cache stores keys *after* rotation, so a content-only clustering space is
    not directly available -- and recomputing ``k_norm(k_proj(h))`` instead would require the layer's
    weights and the right hidden states, which is more coupling for the same tensor.
    """
    if k.dim() != 3:
        raise ValueError(f"k must be (H, S, D), got {tuple(k.shape)}")
    if cos.shape[-1] != k.shape[-1]:
        raise ValueError(
            f"rope tables have width {cos.shape[-1]} but keys have head_dim {k.shape[-1]}; "
            "a narrowed rope_dim would need the un-rotation restricted to the same channels"
        )
    half = k.shape[-1] // 2
    rot = torch.cat([-k[..., half:], k[..., :half]], dim=-1)
    return k * cos.unsqueeze(0) - rot * sin.unsqueeze(0)


def position_locality(assign: torch.Tensor, weights: torch.Tensor, n_slots: int) -> torch.Tensor:
    """
    How position-spread each cluster is, ``(H, R)``, normalized so ``1.0`` = as spread as uniform.

    ``std(member positions) / std(uniform over the evicted range)``. This is the diagnostic that
    decides whether post-RoPE clustering is doing what it claims: RoPE makes a key's post-rotation
    direction depend strongly on its position, so a k-means over post-RoPE keys can silently
    degenerate into *positional* chunking -- and positional chunking is the variant that was measured
    to lose badly (cos 0.68-0.70 against 0.85-0.86). A value near 0 means the clusters are contiguous
    spans, i.e. the "content" clustering is really a position clustering wearing a different name.
    """
    n_heads = assign.shape[0]
    pos = torch.arange(assign.shape[1], device=assign.device, dtype=torch.float32)
    oh = torch.nn.functional.one_hot(assign, n_slots).to(torch.float32) * weights.unsqueeze(-1)
    pop = oh.sum(1).clamp(min=1e-9)
    m1 = torch.einsum("hsr,s->hr", oh, pos) / pop
    m2 = torch.einsum("hsr,s->hr", oh, pos.square()) / pop
    std = (m2 - m1.square()).clamp(min=0).sqrt()
    # reference: a uniform spread over each head's own evicted positions
    ref = []
    for h in range(n_heads):
        live = weights[h] > 0
        ref.append(pos[live].std() if live.sum() > 1 else torch.tensor(1.0, device=pos.device))
    return std / torch.stack(ref).clamp(min=1e-9).view(-1, 1)


def kmeans_assign(
    x: torch.Tensor,
    n_slots: int,
    *,
    iters: int = DEFAULT_KMEANS_ITERS,
    weights: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched Lloyd's over the head axis. ``x`` is ``(H, S, D)`` -> ``(centroids (H, R, D), assign (H, S))``.

    ``weights`` optionally weights each point when recomputing centroids, which is how a *masked*
    call is expressed: pass 0 for points that are not in the evicted set and they influence neither
    the centroids nor (via :func:`cluster_reduce`) the slots, while the tensor keeps its full ``S``
    extent so every head can share one kernel despite having a different number of evicted keys.

    Empty clusters keep their previous centroid rather than being re-seeded. Re-seeding would make
    the routine non-idempotent across calls at decode, where it is re-run as the cache grows -- a
    slot's meaning would then change identity between steps for reasons unrelated to the data.
    """
    if x.dim() != 3:
        raise ValueError(f"x must be (H, S, D), got {tuple(x.shape)}")
    n_heads, n_pts, dim = x.shape
    if n_slots <= 0:
        raise ValueError(f"n_slots must be positive, got {n_slots}")
    n_slots = min(n_slots, n_pts)

    w = torch.ones(n_heads, n_pts, device=x.device, dtype=x.dtype) if weights is None else weights
    # Seed from the weighted points only: seeding on a masked-out (retained) key would place a
    # centroid where no evicted key lives and waste a slot for the whole run.
    probs = w.clamp(min=0)
    probs = torch.where(probs.sum(-1, keepdim=True) > 0, probs, torch.ones_like(probs))
    seed = torch.multinomial(probs, n_slots, replacement=False, generator=generator)  # (H, R)
    c = x.gather(1, seed.unsqueeze(-1).expand(n_heads, n_slots, dim)).clone()

    assign = torch.zeros(n_heads, n_pts, dtype=torch.int64, device=x.device)
    for _ in range(iters):
        # (H, S, R); cdist rather than the -2<x,c> expansion so a fp32 cache does not lose the
        # ||x||^2 term's precision on the large-norm keys of the late layers (||k|| ~ 30 at L35).
        assign = torch.cdist(x, c).argmin(-1)
        oh = torch.nn.functional.one_hot(assign, n_slots).to(x.dtype) * w.unsqueeze(-1)
        pop = oh.sum(1)  # (H, R)
        new_c = torch.einsum("hsr,hsd->hrd", oh, x) / pop.clamp(min=1e-9).unsqueeze(-1)
        c = torch.where(pop.unsqueeze(-1) > 0, new_c, c)
    return c, assign


def cluster_reduce(
    values: torch.Tensor, assign: torch.Tensor, n_slots: int, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Weighted per-cluster mean of ``values`` ``(H, S, D)`` plus each cluster's population.

    Returns ``(mean (H, R, D), pop (H, R))``. The mean is the right operator here and that is worth
    stating, because pooling was measured to be the *wrong* one in a neighbouring experiment: mean-
    pooling KVzip's queries lost at every rate because ``<mean(q), k> = mean(<q, k>)`` silently
    replaced an ``amax`` over queries with an average. Here the target
    ``sum_j softmax(q.k_j) v_j`` **is** an average of value vectors, so averaging is exact in form
    and only the weights are approximated. Picking a real medoid instead would discard the other
    members' contribution, which is information the mean keeps.
    """
    oh = torch.nn.functional.one_hot(assign, n_slots).to(values.dtype) * weights.unsqueeze(-1)
    pop = oh.sum(1)
    mean = torch.einsum("hsr,hsd->hrd", oh, values) / pop.clamp(min=1e-9).unsqueeze(-1)
    return mean, pop


def slot_mass(
    pop: torch.Tensor,
    *,
    logit_var: torch.Tensor | None = None,
    delta: float = 0.0,
    count_coef: torch.Tensor | float = 1.0,
    var_coef: torch.Tensor | float = 1.0,
    bias: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """
    ``b_r``: the log-mass a slot claims in the shared softmax. ``(H, R)``.

    A slot replaces ``n_r`` keys with one, so the denominator loses their count and ``b_r`` puts it
    back. The exact quantity a slot should carry is the cluster's own log-sum-exp,

        ``log sum_{j in r} exp(q.k_j) = log n_r + q.kbar_r + 1/2 Var_j(q.k_j) + O(kappa_3)``,

    a second-order (Laplace) expansion. The slot's logit ``q.kbar_r`` already supplies the middle
    term, so what is missing from a bare ``log n_r`` is the **within-cluster logit variance** --
    strictly positive, so ``log n_r`` alone is *biased low*. Measured on the positional variant that
    bias was ~2.0 nats, i.e. the slot claimed ``e^-2 ~ 1/7`` of its due.

    That bias is why a bare ``log n_r`` is the *safe* starting point rather than the good one: it is
    safe precisely because it silences the slot. Fixing the mass while leaving the direction
    mediocre is actively harmful -- with ``v_r`` held fixed and only ``b_r`` improved, the fused
    error went from 1.02x (mass off by 2.0 nats) to **0.29x** (off by 0.7), against 4.32x once the
    direction was also right. A confident slot pointing the wrong way takes softmax mass away from
    retained keys that were pointing the right way.

    The full form is

        ``b_r = count_coef * log n_r + var_coef * (1/2 Var_r) + bias``

    and **``count_coef`` is the load-bearing learnable one.** ``log n_r`` is only the right mass if a
    slot standing for ``n_r`` keys really carries ``n_r`` times the weight of a singleton -- which is
    false in general, because the members have different logits and the count says nothing about
    them. Measured: each row's evicted mass is spread over 60-466 keys, but the *distribution* over
    those keys is far from flat (the participation ratio is well below the cluster size), so the
    effective multiplicity a query sees is ``n_r^c`` for some ``c < 1`` rather than ``n_r``. Learning
    ``c`` per head is a one-parameter correction to exactly that, and it is strictly more expressive
    than any constant offset: ``bias`` shifts every slot equally, while ``count_coef`` changes how
    the mass *scales with cluster size*, which is what a wrongly-flat count gets wrong.

    All three may be **per-(layer, head) learned scalars** (:class:`CMPMassHead`) -- per head rather
    than per slot, because ``R`` slots are built at inference from whatever the document evicted, so
    a per-slot parameter has no identity to carry across documents. The learned form is a strict
    generalization of both fixed modes: ``(1, 1, 0)`` is ``count+var`` and ``(1, 0, 0)`` is
    ``count``, so the ablation is nested by construction and a learned run starts *at* whichever
    fixed mode it is initialized to.

    Parameters
    ----------
    pop : torch.Tensor
        ``(H, R)`` cluster populations. Zero-population slots get ``-inf`` so they drop out of the
        softmax exactly rather than by luck of the fp path.
    logit_var : torch.Tensor, optional
        ``(H, R)`` within-cluster variance of ``q.k_j``, estimated from the queries already seen
        during prefill (see :func:`prefill_logit_var`). Adds the ``1/2 Var`` term. Using *past*
        queries keeps this training-free and leak-free: they are all at positions before the slot is
        built.
    delta : float
        Constant nats added to every slot, for sweeping the correction by hand instead.
    count_coef, var_coef, bias : torch.Tensor or float
        Per-head vectors (broadcast over the slot axis) or scalars. ``count_coef`` scales
        ``log n_r`` -- see above, it is the one that matters. Scalars reproduce the fixed modes.
    """
    def _bcast(x):
        # A per-head vector broadcasts over the slot axis; a python float passes through. One helper
        # so the fixed and learned modes cannot take different code paths.
        return x.reshape(-1, 1) if isinstance(x, torch.Tensor) and x.numel() > 1 else x

    b = _bcast(count_coef) * torch.log(pop.clamp(min=1e-30))
    if logit_var is not None:
        b = b + _bcast(var_coef) * 0.5 * logit_var
    b = b + _bcast(bias) + float(delta)
    # `torch.where` rather than masked_fill so this stays differentiable in var_coef/bias. The
    # clamp above keeps the masked branch's `b` FINITE (log 1e-30 = -69), which matters: `where`
    # computes both branches, and an inf there would put NaN into the gradient of every live slot.
    return torch.where(pop > 0, b, torch.full_like(b, -float("inf")))


class CMPMassHead(torch.nn.Module):
    """
    Learned per-KV-head scalars for a layer's slot mass::

        b_r = count_coef * log n_r + var_coef * (1/2 Var_r) + bias

    **``count_coef`` is the point of this module.** The training-free version trusts ``log n_r`` at
    face value -- a slot standing for ``n_r`` evicted keys claims ``n_r`` times a singleton's mass.
    That is only right if the members' logits are interchangeable, and they are not: a cluster's
    participation ratio sits well below its size, so the multiplicity a query actually experiences
    grows like ``n_r^c`` with ``c < 1``. One scalar per head learns ``c``.

    Why per (layer, head) and not per slot: the slots are built at inference from whatever the
    document evicted, so a per-slot parameter has nothing to attach to across documents. ``log n_r``
    and ``Var_r`` are the document-dependent quantities; these scalars only say how much to trust
    each. 3 x 8 x 36 = **864 parameters** for Qwen3-8B, which is what keeps this honest about being
    nearly-training-free.

    Initialized at ``(1, 1, 0)`` = the analytic ``count+var`` mode, so training starts from the best
    closed-form answer rather than from zero, and both fixed modes stay exactly reachable. ``bias``
    is unconstrained downward, which matters: the layers where the *direction* is bad (L27/L32
    measured 0.25-0.42x) are ones where the correct move is to shut the slot up, and a freely-negative
    bias lets the objective discover that instead of needing a hand-set per-layer gate.
    """

    def __init__(
        self,
        n_kv_heads: int,
        *,
        count_init: float = 1.0,
        var_init: float = 1.0,
        bias_init: float = 0.0,
    ):
        super().__init__()
        self.count_coef = torch.nn.Parameter(torch.full((n_kv_heads,), float(count_init)))
        self.var_coef = torch.nn.Parameter(torch.full((n_kv_heads,), float(var_init)))
        self.bias = torch.nn.Parameter(torch.full((n_kv_heads,), float(bias_init)))

    def forward(
        self, pop: torch.Tensor, logit_var: torch.Tensor | None = None
    ) -> torch.Tensor:
        return slot_mass(
            pop,
            logit_var=logit_var,
            count_coef=self.count_coef,
            var_coef=self.var_coef,
            bias=self.bias,
        )

    def extra_repr(self) -> str:
        return (
            f"heads={self.count_coef.numel()}, "
            f"count={self.count_coef.detach().mean():.3f}, "
            f"var={self.var_coef.detach().mean():.3f}, "
            f"bias={self.bias.detach().mean():+.3f}"
        )


def prefill_logit_var(
    q: torch.Tensor, k: torch.Tensor, assign: torch.Tensor, n_slots: int,
    weights: torch.Tensor, *, scaling: float, max_queries: int = 256,
) -> torch.Tensor:
    """
    Within-cluster variance of ``q.k_j``, averaged over a sample of the prefill's own queries.

    ``(H, R)``. Estimated from queries the model has *already* processed, which is what keeps the
    whole scheme training-free without leaking: every sampled query sits at a position before the
    slots are built, so nothing about the future enters. Whether this transfers to the queries that
    arrive later is an empirical question -- the query distribution is the single largest lever
    measured in this repo (2.2x on the achievable ceiling), so it is also the term most worth
    checking rather than trusting.

    Queries are subsampled (evenly, not from the tail) because the estimate is a per-cluster second
    moment over thousands of keys and converges long before all rows are used.
    """
    n_heads, q_len, dim = q.shape
    step = max(1, q_len // max_queries)
    qs = q[:, ::step]  # (H, T, D)
    logits = torch.einsum("htd,hsd->hts", qs, k) * scaling  # (H, T, S)
    oh = torch.nn.functional.one_hot(assign, n_slots).to(logits.dtype) * weights.unsqueeze(-1)
    pop = oh.sum(1).clamp(min=1e-9)  # (H, R)
    s1 = torch.einsum("hts,hsr->htr", logits, oh) / pop.unsqueeze(0).squeeze(0).unsqueeze(1)
    s2 = torch.einsum("hts,hsr->htr", logits.square(), oh) / pop.unsqueeze(1)
    return (s2 - s1.square()).clamp(min=0).mean(1)  # (H, R)


def cluster_evicted(
    keys: torch.Tensor,
    values: torch.Tensor,
    evicted: torch.Tensor,
    n_slots: int,
    *,
    cluster_keys: torch.Tensor | None = None,
    queries: torch.Tensor | None = None,
    scaling: float = 1.0,
    delta: float = 0.0,
    iters: int = DEFAULT_KMEANS_ITERS,
    generator: torch.Generator | None = None,
    diagnostics: bool = False,
    mass_head: "CMPMassHead | None" = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build the CMP slots for one layer: ``(k_cmp (H, R, D), v_cmp (H, R, D), b_cmp (H, R))``.

    Parameters
    ----------
    keys, values : torch.Tensor
        ``(H, S, D)`` **post-RoPE** cache contents for one KV head axis.
    evicted : torch.Tensor
        ``(H, S)`` boolean, ``True`` for the keys this row set drops. Per-head, because GQA evicts
        per KV head and the heads agree on only 14-17% of their top-k.
    n_slots : int
        ``R``. Paid for out of the read budget by the caller, not added to it.
    cluster_keys : torch.Tensor, optional
        ``(H, S, D)`` the space to run k-means *in*, when it should differ from ``keys``. Pass
        :func:`unrotate`'s output to cluster on **content alone**.

        **Only the assignment changes.** ``k_cmp`` stays the mean of the *post-RoPE* keys, because
        the slot's logit is ``q . k_cmp`` and the quantity it must approximate is the cluster's mean
        post-RoPE logit ``mean_j(q . R_j k_j) = q . mean_j(R_j k_j)``. Averaging pre-RoPE keys and
        rotating afterwards would be a different (and wrong) quantity, since
        ``R_jbar kbar != mean_j(R_j k_j)`` and a cross-position cluster has no single ``jbar``.

        There is a real tension to measure rather than assume, in both directions. Clustering
        post-RoPE risks degenerating into *positional* chunking, because RoPE makes a key's direction
        depend strongly on its position -- and positional chunking is the variant that lost (cos
        0.68-0.70 vs 0.85-0.86). Clustering pre-RoPE removes that confound, but its clusters span
        arbitrary positions, so the post-RoPE mean they must still produce suffers more phase
        cancellation; in the limit ``k_cmp -> 0`` and the slot becomes a content-blind constant that
        can only be addressed by ``b_r``. :func:`position_locality` and the returned norm ratio are
        what decide which effect dominates.
    queries : torch.Tensor, optional
        ``(H, S, D)`` prefill queries for the ``1/2 Var`` mass correction. Omitted, ``b_r`` is a
        bare ``log n_r`` -- safe but quiet (see :func:`slot_mass`).
    diagnostics : bool
        Also return a dict with ``position_locality`` (0 = contiguous spans, 1 = as spread as
        uniform) and ``norm_ratio`` (``||k_cmp|| / mean||k_j||``, i.e. how much the centroid survived
        phase cancellation).
    """
    if keys.shape != values.shape:
        raise ValueError(f"keys {tuple(keys.shape)} and values {tuple(values.shape)} must match")
    if evicted.shape != keys.shape[:2]:
        raise ValueError(f"evicted {tuple(evicted.shape)} must be {tuple(keys.shape[:2])}")
    space = keys if cluster_keys is None else cluster_keys
    if space.shape != keys.shape:
        raise ValueError(
            f"cluster_keys {tuple(space.shape)} must match keys {tuple(keys.shape)}"
        )

    w = evicted.to(keys.dtype)
    centroids, assign = kmeans_assign(
        space, n_slots, iters=iters, weights=w, generator=generator
    )
    n_eff = centroids.shape[1]
    # Reduce the POST-RoPE keys regardless of which space the assignment came from -- see above.
    k_cmp, pop = cluster_reduce(keys, assign, n_eff, w)
    v_cmp, _ = cluster_reduce(values, assign, n_eff, w)

    # Slots below the population floor are silenced, not dropped: a fixed R keeps every head's slot
    # block the same width, which is what lets the caller concatenate one tensor per layer.
    pop = torch.where(pop >= MIN_CLUSTER, pop, torch.zeros_like(pop))

    logit_var = None
    if queries is not None:
        logit_var = prefill_logit_var(
            queries, keys, assign, n_eff, w, scaling=scaling
        )
    if mass_head is not None:
        b_cmp = mass_head(pop, logit_var)
    else:
        b_cmp = slot_mass(pop, logit_var=logit_var, delta=delta)
    if not diagnostics:
        return k_cmp, v_cmp, b_cmp
    live = pop > 0
    ref = (keys.norm(dim=-1) * w).sum(-1) / w.sum(-1).clamp(min=1e-9)  # (H,)
    return (
        k_cmp,
        v_cmp,
        b_cmp,
        {
            "position_locality": position_locality(assign, w, n_eff),
            "norm_ratio": k_cmp.norm(dim=-1) / ref.unsqueeze(-1).clamp(min=1e-9),
            "live": live,
            "pop": pop,
        },
    )


def evicted_from_deadline(deadline: torch.Tensor, horizon: int) -> torch.Tensor:
    """
    ``(H, S)`` eviction mask at a given pool horizon, from :func:`~.qi_flex_attention.deadlines`.

    Its contract is that a row with horizon ``hi`` keeps key ``j`` iff ``hi <= deadline[j]``, so the
    evicted set is the strict complement. Deriving the mask here -- rather than re-running a top-k --
    is what keeps the slots and the sparse attention mask consistent by construction: a key must be
    in exactly one of the two, and both come from the same ``deadline``.

    Keys at positions ``> horizon`` have not arrived yet and are not evicted, they are simply absent;
    including them would fold the future into a slot, which is the failure mode recorded in the
    module docstring.
    """
    if deadline.dim() != 2:
        raise ValueError(f"deadline must be (H, S), got {tuple(deadline.shape)}")
    pos = torch.arange(deadline.shape[1], device=deadline.device)
    arrived = pos.view(1, -1) <= int(horizon)
    return arrived & (int(horizon) > deadline.to(torch.int64))
