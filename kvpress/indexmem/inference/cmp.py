# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for the IndexMem++ public implementation.

from __future__ import annotations

import torch

DEFAULT_KMEANS_ITERS = 15


def kmeans_assign(
    x: torch.Tensor,
    n_slots: int,
    *,
    iters: int = DEFAULT_KMEANS_ITERS,
    weights: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_heads, n_pts, dim = x.shape
    n_slots = min(n_slots, n_pts)
    w = torch.ones(n_heads, n_pts, device=x.device, dtype=x.dtype) if weights is None else weights
    probs = w.clamp(min=0)
    probs = torch.where(probs.sum(-1, keepdim=True) > 0, probs, torch.ones_like(probs))
    seed = torch.multinomial(probs, n_slots, replacement=False, generator=generator)
    c = x.gather(1, seed.unsqueeze(-1).expand(n_heads, n_slots, dim)).clone()
    assign = torch.zeros(n_heads, n_pts, dtype=torch.int64, device=x.device)
    for _ in range(iters):
        assign = torch.cdist(x, c).argmin(-1)
        oh = torch.nn.functional.one_hot(assign, n_slots).to(x.dtype) * w.unsqueeze(-1)
        cluster_counts = oh.sum(1)
        new_c = torch.einsum("hsr,hsd->hrd", oh, x) / cluster_counts.clamp(min=1e-09).unsqueeze(-1)
        c = torch.where(cluster_counts.unsqueeze(-1) > 0, new_c, c)
    return (c, assign)


def cluster_reduce(
    values: torch.Tensor, assign: torch.Tensor, n_slots: int, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    oh = torch.nn.functional.one_hot(assign, n_slots).to(values.dtype) * weights.unsqueeze(-1)
    cluster_counts = oh.sum(1)
    mean = torch.einsum("hsr,hsd->hrd", oh, values) / cluster_counts.clamp(min=1e-09).unsqueeze(-1)
    return (mean, cluster_counts)


def evicted_from_deadline(deadline: torch.Tensor, horizon: int) -> torch.Tensor:
    pos = torch.arange(deadline.shape[1], device=deadline.device)
    arrived = pos.view(1, -1) <= int(horizon)
    return arrived & (int(horizon) > deadline.to(torch.int64))


class CentroidMemory:

    def __init__(self, n_rows: int, n_slots: int, head_dim: int, *, device: torch.device):
        self.n_rows = int(n_rows)
        self.n_slots = int(n_slots)
        self.head_dim = int(head_dim)
        self.device = device
        self.centroid_keys = torch.zeros((n_rows, n_slots, head_dim), device=device, dtype=torch.float32)
        self.centroid_values = torch.zeros_like(self.centroid_keys)
        self.cluster_counts = torch.zeros((n_rows, n_slots), device=device, dtype=torch.float32)

    def load_batch(self, rows: slice | torch.Tensor, centroid_keys, centroid_values, log_cluster_mass) -> None:
        self.centroid_keys[rows] = centroid_keys.to(self.centroid_keys.dtype)
        self.centroid_values[rows] = centroid_values.to(self.centroid_values.dtype)
        cluster_counts = log_cluster_mass.to(torch.float32).exp()
        self.cluster_counts[rows] = torch.where(
            torch.isfinite(log_cluster_mass), cluster_counts, torch.zeros_like(cluster_counts)
        ).round()

    @torch.no_grad()
    def ingest(
        self, rows: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, active: torch.Tensor | None = None
    ) -> None:
        if active is None:
            active = torch.ones(rows.numel(), dtype=torch.bool, device=self.device)
        k = key.to(torch.float32)
        v = value.to(torch.float32)
        cent = self.centroid_keys[rows]
        pop_r = self.cluster_counts[rows]
        dead = pop_r <= 0
        has_dead = dead.any(-1)
        first_dead = dead.to(torch.uint8).argmax(-1)
        d2 = (cent - k.unsqueeze(1)).pow(2).sum(-1)
        d2 = torch.where(pop_r > 0, d2, torch.full_like(d2, float("inf")))
        nearest = d2.argmin(-1)
        target = torch.where(has_dead, first_dead, nearest)
        takeover = has_dead & active
        new_pop = torch.where(
            takeover, torch.ones_like(pop_r[:, 0]), pop_r.gather(-1, target.unsqueeze(-1)).squeeze(-1) + 1.0
        )
        cur_k = cent.gather(1, target.view(-1, 1, 1).expand(-1, 1, self.head_dim)).squeeze(1)
        cur_v = self.centroid_values[rows].gather(1, target.view(-1, 1, 1).expand(-1, 1, self.head_dim)).squeeze(1)
        upd_k = torch.where(takeover.unsqueeze(-1), k, cur_k + (k - cur_k) / new_pop.unsqueeze(-1))
        upd_v = torch.where(takeover.unsqueeze(-1), v, cur_v + (v - cur_v) / new_pop.unsqueeze(-1))
        keep = active.unsqueeze(-1)
        self.centroid_keys[rows, target] = torch.where(keep, upd_k, cur_k)
        self.centroid_values[rows, target] = torch.where(keep, upd_v, cur_v)
        self.cluster_counts[rows, target] = torch.where(
            active, new_pop, pop_r.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        )

    def read(
        self, rows: torch.Tensor, query: torch.Tensor, *, group: int, scaling: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n_q_heads, q_len, head_dim = query.shape
        kc = self.centroid_keys[rows].repeat_interleave(group, 0)
        vc = self.centroid_values[rows].repeat_interleave(group, 0)
        cluster_counts = self.cluster_counts[rows].repeat_interleave(group, 0)
        kc = kc.view(bsz, n_q_heads, self.n_slots, head_dim)
        vc = vc.view(bsz, n_q_heads, self.n_slots, head_dim)
        cluster_counts = cluster_counts.view(bsz, n_q_heads, 1, self.n_slots)
        logits = torch.einsum("bhqd,bhrd->bhqr", query.float(), kc) * scaling
        b = torch.where(
            cluster_counts > 0,
            torch.log(cluster_counts.clamp(min=1e-30)),
            torch.full_like(cluster_counts, -float("inf")),
        )
        z = logits + b
        m = z.amax(-1, keepdim=True)
        m_pos_inf = m == float("inf")
        m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        m_safe = torch.where(m_pos_inf, torch.full_like(m, torch.finfo(m.dtype).max), m_safe)
        z = torch.where(m_pos_inf.expand_as(z) & (z == float("inf")), torch.full_like(z, torch.finfo(z.dtype).max), z)
        w = (z - m_safe).exp()
        denom = w.sum(-1, keepdim=True)
        o = torch.einsum("bhqr,bhrd->bhqd", w / denom.clamp(min=1e-30), vc)
        lse = m_safe.squeeze(-1) + denom.clamp(min=1e-30).log().squeeze(-1)
        lse = torch.where(
            torch.isfinite(m.squeeze(-1)) | m_pos_inf.squeeze(-1), lse, torch.full_like(lse, -float("inf"))
        )
        return (o, lse)


@torch.no_grad()
def cluster_evicted(keys, values, evicted, n_slots, *, iters=DEFAULT_KMEANS_ITERS, generator=None):
    weights = evicted.to(keys.dtype)
    centroids, assignments = kmeans_assign(keys, n_slots, iters=iters, weights=weights, generator=generator)
    key_centroids, population = cluster_reduce(keys, assignments, centroids.shape[1], weights)
    value_centroids, _ = cluster_reduce(values, assignments, centroids.shape[1], weights)
    log_mass = torch.where(
        population >= 1, population.clamp(min=1e-30).log(), torch.full_like(population, -float("inf"))
    )
    return (key_centroids, value_centroids, log_mass)
