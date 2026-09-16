# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CMP slots maintained **incrementally**, for a cache that physically evicts.

:func:`~.cmp_slots.cluster_evicted` runs k-means over the evicted keys, which needs those keys in
hand. Under hard eviction (:mod:`~.evict_cache`) they are gone -- deleted to buy the compression --
so the batch form can only run once, at the commit, and then never again.

That is not a small loss on the benchmarks that generate long chains of thought. Walk the code on
math500: the context is a single space, so at the question forward ``take`` far exceeds ``k_len``,
the evicted set is **empty**, ``_build_cmp`` records ``_cmp_at`` and returns without writing a
slot, and every later step skips the CMP branch. Frozen CMP is therefore *completely inert* on
math500 and aime25 -- exactly the two benchmarks where the mass eviction discards is worst
(measured on Qwen3-4B at ``topk=1024``: ``rho`` 0.028 at a 1649-token CoT, 0.071 at 2873, **0.141
at 7198**, and 0.261 for rows whose history reaches 4x their budget; aime25 generates 32000 tokens,
i.e. 28x). So for CoT, streaming is not an improvement to CMP -- it is what makes CMP exist.

What is exact here, and what is given up
----------------------------------------
The centroid update is a **running mean**, which is algebraically identical to re-averaging the
same set (measured 6.3e-8, pure fp32 rounding)::

    n_r <- n_r + 1
    k_r <- k_r + (k_new - k_r) / n_r

and ``b_r = log n_r`` is an integer counter. So the state is ``O(R * D)`` per row and **nothing is
recomputed**: 18.9 MB for a 36-layer Qwen3-8B at ``R=64``, about 6% of the compressed cache it
rides along with, and ``R * D`` work per row per step (~0.8% of the attention it feeds).

What is given up is **reassignment**. Once a key is deleted it cannot be moved to a centroid that
has since drifted closer, so this is online k-means with frozen membership rather than Lloyd's.
Measured on clustered data (40 true centres, R=64, points drawn from 5 unseen centres): frozen
prefill centroids reconstruct at 1.047 relative error -- *worse than a zero vector*, because the
centroid points the wrong way -- streaming at 0.550, and a full re-clustering, which is
unavailable, at 0.406. Streaming recovers 74% of that gap without damaging the summary of the
context it started from (0.2023 -> 0.2092).

Cold start
----------
A document gives k-means a whole batch of evicted keys to seed from. A CoT gives none: generation
starts with an empty evicted set. The first ``R`` keys to be evicted therefore become the seeds,
one each, which is textbook sequential k-means -- but note *which* keys those are. They are the
lowest-scoring keys at the moment the budget first binds, so they are a biased basis, and a slot
seeded on one of them may stay a poor summary for the whole run. :attr:`StreamingCMP.reseed_below`
exists for that: a slot whose population stays at or below it is treated as dead and re-seeded by
the next arrival rather than left to anchor a cluster nobody joins.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class StreamingCMP:
    """
    Per-row CMP slots that absorb evicted keys one at a time.

    Rows are the same ``(layer, batch, head)`` flattening
    :class:`~.evict_cache.EvictPagedPool` uses, so a row index means the same thing in both and
    the two can be updated from one set of indices.

    Parameters
    ----------
    n_rows : int
        ``n_layers * batch_size * n_kv_heads``.
    n_slots : int
        ``R``. Funded out of the read budget by the caller, never added to it.
    head_dim : int
    reseed_below : int
        A slot whose population is at or below this is considered dead and may be taken over by a
        new arrival instead of being updated. ``0`` disables re-seeding, so the first ``R`` evicted
        keys own their slots permanently. See the cold-start note above.
    """

    def __init__(
        self,
        n_rows: int,
        n_slots: int,
        head_dim: int,
        *,
        device: torch.device,
        reseed_below: int = 0,
    ):
        if n_slots <= 0:
            raise ValueError(f"n_slots must be positive, got {n_slots}")
        self.n_rows = int(n_rows)
        self.n_slots = int(n_slots)
        self.head_dim = int(head_dim)
        self.device = device
        self.reseed_below = int(reseed_below)
        # fp32 throughout: these are running means updated thousands of times, and the increment
        # (k - k_r)/n_r becomes small enough that bf16 would stop accumulating it. At n_r = 500 a
        # bf16 centroid of norm ~30 cannot represent a 0.06 update at all.
        self.k_cmp = torch.zeros((n_rows, n_slots, head_dim), device=device, dtype=torch.float32)
        self.v_cmp = torch.zeros_like(self.k_cmp)
        #: Population per slot. The sole source of ``b_r = log n_r``, and 0 means "never used",
        #: which :meth:`read` turns into an exact ``-inf`` rather than a small finite logit.
        self.pop = torch.zeros((n_rows, n_slots), device=device, dtype=torch.float32)

    # ------------------------------------------------------------------
    def load_batch(self, rows: slice | torch.Tensor, k_cmp, v_cmp, b_cmp) -> None:
        """Seed from a batch :func:`~.cmp_slots.cluster_evicted` result.

        Lets a long context be summarized properly by k-means at the commit and *then* kept up to
        date by streaming -- the two are not alternatives. ``b_cmp`` is inverted back into a
        population, since that is what the running mean needs; silenced slots (``-inf``) come back
        as 0 and are free for re-seeding.
        """
        self.k_cmp[rows] = k_cmp.to(self.k_cmp.dtype)
        self.v_cmp[rows] = v_cmp.to(self.v_cmp.dtype)
        pop = b_cmp.to(torch.float32).exp()
        self.pop[rows] = torch.where(torch.isfinite(b_cmp), pop, torch.zeros_like(pop)).round()

    @torch.no_grad()
    def ingest(
        self,
        rows: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        active: torch.Tensor | None = None,
    ) -> None:
        """
        Absorb one evicted key per row.

        Parameters
        ----------
        rows : torch.Tensor
            ``(n,)`` row indices. **Each row may appear at most once**, which holds for the
            eviction path because a step rolls the local window by exactly one key per row. Two
            arrivals for one row in a single call would race in the scatter and one would be lost.
        key, value : torch.Tensor
            ``(n, head_dim)`` the evicted key/value.
        active : torch.Tensor, optional
            ``(n,)`` bool. ``False`` rows are skipped -- which is the normal case rather than an
            edge case: a row still growing into its budget evicts nothing, and a row whose demoted
            key lost the pool contest discards *that* key instead of a pool member.

        No host synchronization: the whole update is masked arithmetic, for the same reason
        :meth:`~.evict_cache.EvictPagedPool.ingest` avoids it -- one ``.any()`` costs 0.94 ms here,
        which at 36 layers a token would dominate everything else in the step.
        """
        if active is None:
            active = torch.ones(rows.numel(), dtype=torch.bool, device=self.device)
        k = key.to(torch.float32)
        v = value.to(torch.float32)

        cent = self.k_cmp[rows]                      # (n, R, D)
        pop_r = self.pop[rows]                       # (n, R)
        dead = pop_r <= self.reseed_below             # free or dead slots
        # A dead slot is taken over rather than blended into: `first_dead` is the lowest-indexed
        # one, so the cold start fills slots 0..R-1 in order and a later re-seed reuses the same
        # deterministic choice.
        has_dead = dead.any(-1)
        first_dead = dead.to(torch.uint8).argmax(-1)
        # Squared distance to each live centroid; dead slots are pushed out of the argmin so a
        # zero-vector centroid cannot attract every arrival.
        d2 = (cent - k.unsqueeze(1)).pow(2).sum(-1)   # (n, R)
        d2 = torch.where(pop_r > 0, d2, torch.full_like(d2, float("inf")))
        nearest = d2.argmin(-1)
        target = torch.where(has_dead, first_dead, nearest)
        # Taking over a dead slot REPLACES it (n_r goes 0 -> 1); joining a live one updates its
        # mean. Expressed as one blend so there is a single scatter.
        takeover = has_dead & active
        new_pop = torch.where(takeover, torch.ones_like(pop_r[:, 0]),
                              pop_r.gather(-1, target.unsqueeze(-1)).squeeze(-1) + 1.0)
        cur_k = cent.gather(1, target.view(-1, 1, 1).expand(-1, 1, self.head_dim)).squeeze(1)
        cur_v = self.v_cmp[rows].gather(
            1, target.view(-1, 1, 1).expand(-1, 1, self.head_dim)
        ).squeeze(1)
        upd_k = torch.where(takeover.unsqueeze(-1), k, cur_k + (k - cur_k) / new_pop.unsqueeze(-1))
        upd_v = torch.where(takeover.unsqueeze(-1), v, cur_v + (v - cur_v) / new_pop.unsqueeze(-1))

        keep = active.unsqueeze(-1)
        self.k_cmp[rows, target] = torch.where(keep, upd_k, cur_k)
        self.v_cmp[rows, target] = torch.where(keep, upd_v, cur_v)
        self.pop[rows, target] = torch.where(
            active, new_pop, pop_r.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        )

    # ------------------------------------------------------------------
    def read(
        self,
        rows: torch.Tensor,
        query: torch.Tensor,
        *,
        group: int,
        scaling: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The slots' contribution as ``(weighted values, lse)``, both per QUERY head.

        Returns ``(o_cmp (B, H, Sq, D), lse_cmp (B, H, Sq))`` where ``o_cmp`` is already
        *normalized* by the slots' own denominator and ``lse_cmp`` is that denominator in the log
        domain -- exactly the shape a log-sum-exp merge consumes.

        **Not the raw ``(numerator, denominator)`` pair.** Returning ``sum exp(q.k_r + log n_r)``
        unnormalized overflows fp32: a slot holding 3000 keys contributes ``log n_r = 8``, and the
        logit itself reaches ~1600 on the large-norm keys of the deep layers (``||k|| ~ 30`` at
        L35), so ``exp`` saturates to ``inf``, the fused output becomes ``inf/inf = nan``, and the
        nan reaches ``multinomial`` as a probability -- a device-side assert, which is how this was
        found (5 of 8 shards of the CMP arm died mid-run while the no-CMP arm was clean). The
        subtract-the-max form below is bounded by construction, and it is what
        :func:`~.memory.fuse_memory` does for the same reason.

        Per **query** head, not per KV head. The slot is a KV entry so its ``k``/``v`` are shared
        across the group, but the logit ``q . k_r`` is not -- each query head brings its own query.
        Cheap because ``R`` is 64: the weight tensor is ``(B, H, Sq, 64)``, not
        ``(B, H, Sq, k_len)``.
        """
        bsz, n_q_heads, q_len, head_dim = query.shape
        kc = self.k_cmp[rows].repeat_interleave(group, 0)     # (B*H, R, D)
        vc = self.v_cmp[rows].repeat_interleave(group, 0)
        pop = self.pop[rows].repeat_interleave(group, 0)       # (B*H, R)
        kc = kc.view(bsz, n_q_heads, self.n_slots, head_dim)
        vc = vc.view(bsz, n_q_heads, self.n_slots, head_dim)
        pop = pop.view(bsz, n_q_heads, 1, self.n_slots)

        logits = torch.einsum("bhqd,bhrd->bhqr", query.float(), kc) * scaling
        # b_r = log n_r. An unused slot gets exactly -inf, so it cannot contribute a spurious
        # logit -- `slot_mass` makes the same choice for the same reason.
        b = torch.where(
            pop > 0, torch.log(pop.clamp(min=1e-30)), torch.full_like(pop, -float("inf"))
        )
        z = logits + b
        # Two degenerate maxima, and they need OPPOSITE treatment. The original code handled only
        # the first and mapped both to a finite 0, which turned the second into NaN.
        #
        # * `m = -inf` -- every slot empty. Substituting 0 makes `z - m` stay -inf, so every weight
        #   is 0 and the row contributes nothing. Correct, and what line ~242 then records as
        #   `lse = -inf`.
        # * `m = +inf` -- a LIVE slot whose logit overflowed fp32. `q . k_cmp * scaling` is an
        #   unbounded dot product: a centroid of large-norm deep-layer keys reached |k_cmp| ~ 520
        #   here against real key norms ~340, and a single fp32 overflow makes the whole row +inf.
        #   Substituting 0 gives `exp(+inf - 0) = inf`, `denom = inf`, and `w / denom = inf / inf`
        #   = **NaN** -- a third, independent source of the device-side assert at
        #   `evict_runner.py:640`, distinct from the two in `evict_cache._merge_lse`.
        #
        # Clamping `m` to the largest finite fp32 instead of 0 fixes the `+inf` case without
        # touching the `-inf` case: `z - m` becomes <= 0 for every finite entry, the exponential is
        # bounded, and the overflowed slots saturate to weight 1 rather than NaN -- i.e. an
        # overflowing slot dominates its row, which is the arithmetically honest outcome and is
        # exactly what an un-overflowed computation in higher precision would have produced.
        m = z.amax(-1, keepdim=True)
        m_pos_inf = m == float("inf")
        m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        m_safe = torch.where(m_pos_inf, torch.full_like(m, torch.finfo(m.dtype).max), m_safe)
        z = torch.where(
            m_pos_inf.expand_as(z) & (z == float("inf")),
            torch.full_like(z, torch.finfo(z.dtype).max),
            z,
        )
        w = (z - m_safe).exp()                                  # <= 1 by construction
        denom = w.sum(-1, keepdim=True)
        o = torch.einsum("bhqr,bhrd->bhqd", w / denom.clamp(min=1e-30), vc)
        lse = (m_safe.squeeze(-1) + denom.clamp(min=1e-30).log().squeeze(-1))
        # Empty-slot rows (m = -inf) drop out of the merge exactly. A `+inf` row is NOT empty -- it
        # is an overflowed live slot -- so it must keep its (now finite) lse rather than be
        # discarded, which is what testing `isfinite(m)` alone used to do.
        lse = torch.where(
            torch.isfinite(m.squeeze(-1)) | m_pos_inf.squeeze(-1),
            lse,
            torch.full_like(lse, -float("inf")),
        )
        return o, lse

    def summary(self) -> str:
        live = (self.pop > 0).sum(-1).float()
        total = self.pop.sum(-1)
        return (
            f"{self.n_rows} rows x R={self.n_slots}: live slots "
            f"{float(live.min()):.0f}..{float(live.max()):.0f} (mean {float(live.mean()):.1f}), "
            f"summarized keys mean {float(total.mean()):.0f}, "
            f"{self.k_cmp.numel() * 4 * 2 / 2 ** 20:.1f} MiB"
        )
