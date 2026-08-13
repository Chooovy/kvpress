# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress

logger = logging.getLogger(__name__)


def facility_location_order(
    demand: torch.Tensor,
    budget: int,
    *,
    initial_cover: torch.Tensor | None = None,
    forced: torch.Tensor | None = None,
    candidate_pool: int | None = None,
) -> torch.Tensor:
    """
    Greedily maximize the facility-location objective over the key axis.

    The objective is::

        f(S) = sum_i  max_{j in S} demand[i, j]

    Read it as coverage: every *demand* row ``i`` is served by whichever selected key suits
    it best, and only that one counts. The inner ``max`` is what makes the function see
    redundancy -- once a key is selected, a second key that serves the same rows adds almost
    nothing, so its marginal gain collapses. A per-key scalar score cannot express that,
    because it is computed without reference to what else is being kept.

    The objective is monotone submodular, so this greedy is within ``1 - 1/e`` of the optimum
    (Nemhauser et al. 1978). ``test_greedy_respects_the_submodular_bound`` checks that against
    brute force.

    Parameters
    ----------
    demand : torch.Tensor
        ``(..., n_demand, n_keys)`` non-negative affinities. Leading dims are batched over.
    budget : int
        Number of keys to select. Clamped to ``n_keys``.
    initial_cover : torch.Tensor, optional
        ``(..., n_demand)`` coverage already provided by keys kept outside this selection
        (e.g. forced sink/local). Passing it makes the greedy account for them rather than
        re-covering rows they already serve.
    forced : torch.Tensor, optional
        ``(..., n_keys)`` bool mask of keys to exclude from selection because they are kept
        unconditionally. Their coverage belongs in ``initial_cover``.
    candidate_pool : int, optional
        Restrict the greedy to the ``candidate_pool`` keys with the largest total demand,
        selected once up front. This bounds the cost at the price of the submodular guarantee
        (the optimum may lie outside the pool). ``None`` considers every key.

    Returns
    -------
    torch.Tensor
        ``(..., budget)`` int64 key indices, in the order greedy selected them (most
        valuable first).
    """
    if demand.dim() < 2:
        raise ValueError(f"demand must be at least 2D (n_demand, n_keys), got {tuple(demand.shape)}")
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")

    *batch, n_demand, n_keys = demand.shape
    budget = min(budget, n_keys)

    cover = (
        torch.zeros(*batch, n_demand, dtype=demand.dtype, device=demand.device)
        if initial_cover is None
        else initial_cover.clone()
    )
    available = torch.ones(*batch, n_keys, dtype=torch.bool, device=demand.device)
    if forced is not None:
        available &= ~forced

    if candidate_pool is not None and candidate_pool < n_keys:
        # Keep the pool at least as large as the budget, else the greedy cannot fill it.
        pool = max(candidate_pool, budget)
        if pool < n_keys:
            total = demand.sum(dim=-2).masked_fill(~available, -float("inf"))
            pool_idx = total.topk(pool, dim=-1).indices.sort(dim=-1).values
            # Physically narrow `demand` to the pool. Masking alone would leave the inner
            # loop reducing over all n_keys and save nothing -- measured 482.5s vs 486.1s
            # before this narrowing, i.e. the knob did not work at all.
            demand = demand.gather(-1, pool_idx.unsqueeze(-2).expand(*batch, n_demand, pool))
            available = available.gather(-1, pool_idx)
            n_keys = pool
    else:
        pool_idx = None

    picks = []
    for _ in range(budget):
        # Marginal gain of key j = the coverage it adds beyond what is already covered.
        # relu, not a plain difference: a key that serves a row *worse* than the current
        # best subtracts nothing -- the row simply keeps its existing server.
        gain = (demand - cover.unsqueeze(-1)).clamp_min(0).sum(dim=-2)
        gain = gain.masked_fill(~available, -float("inf"))
        chosen = gain.argmax(dim=-1)
        picks.append(chosen)

        available.scatter_(-1, chosen.unsqueeze(-1), False)
        # Rows served better by the new key switch to it; the rest are unchanged.
        cover = torch.maximum(cover, demand.gather(-1, chosen[..., None, None].expand(*batch, n_demand, 1)).squeeze(-1))

    order = torch.stack(picks, dim=-1)
    # Map pool-local indices back to original key positions.
    return order if pool_idx is None else pool_idx.gather(-1, order)


@dataclass
class FacilityLocationPress(ScorerPress):
    """
    Set-level KV cache compression via facility location on the prefill attention.

    Every score-based press picks keys by a **per-key scalar**, which is blind to redundancy:
    two keys carrying the same information both score highly and both get kept, spending two
    slots on one piece of information. This press instead maximizes a **set** objective over
    the attention already computed during prefill::

        f(S) = sum_over_(query_head, query_token)  max_{j in S} attention[.., j]

    i.e. "every query row must still find something it attended to among the kept keys".
    Selection is greedy, which for this (monotone submodular) objective is within ``1 - 1/e``
    of optimal.

    Unlike ``KVzipPress`` this needs **no second pass over the context**: the demands are the
    prefill attention rows, which the forward pass has already produced. The trade-off is
    what those demands represent -- see "Scope" below.

    Requires ``attn_implementation="eager"`` (it needs the attention matrix, like
    :class:`~kvpress.presses.observed_attention_press.ObservedAttentionPress`).

    Scope: this is query-agnostic in the sense of never reading a question, but its demands
    come from tokens *already in the cache*, whose needs were satisfied during prefill. So it
    optimizes "the context can re-derive itself", a proxy for "a future query can be
    answered". KVzip's second pass instead elicits demands from tokens that are **not** cached
    and that can see every key -- a stronger claim to query-agnosticism. Prefer this press for
    single-query prefill compression; consider KVzip when one cache is reused across many
    unrelated queries.

    Causal skew: because attention is causal, key ``j`` is visible to only ``k_len - j`` query
    rows, so late keys attract far less total demand -- measured column sums of ~5.2 for the
    first keys against ~0.02 for the last. Left uncorrected the greedy puts **0%** of its
    budget in the final quarter of the sequence, which is where a follow-up question usually
    looks. ``n_local`` is the direct fix and defaults to a non-zero value here for that
    reason; ``normalize_by_visibility`` is a softer alternative.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove.
    n_sink : int, default=4
        Leading keys always kept (attention sinks). Excluded from the greedy but their
        coverage is credited, so the greedy does not re-cover rows they already serve.
    n_local : int, default=128
        Most recent keys always kept. Non-zero by default to offset the causal skew above;
        set to 0 to see the uncorrected behaviour.
    normalize_by_visibility : bool, default=False
        Divide each key's marginal gain by the number of query rows that can see it. Corrects
        the same count bias that ``ObservedAttentionPress`` handles with ``/n_tokens_in_sum``,
        but it breaks the coverage interpretation (and hence the submodular guarantee), so it
        is off by default in favour of ``n_local``.
    n_demand_sample : int, optional
        Subsample this many query rows as demands. The objective is a *sum* over demands, so a
        uniform subsample is an unbiased estimate of it -- the standard way to cut the cost
        below. ``None`` uses every row.
    candidate_pool : int, optional
        Restrict the greedy to this many keys, pre-filtered by total demand. Bounds the cost
        (which is linear in the number of candidates) at the price of the submodular guarantee.
        ``None`` considers every key.
    group_reduce : str, default="stack"
        How the GQA query heads sharing a KV head are folded in. ``stack`` treats each
        ``(query_head, token)`` pair as its own demand, which is the natural reading and
        introduces no arbitrary reduction. ``max`` first maxes over the group, matching
        KVzip's collapse and cutting demand rows by ``group_size``.

    Cost
    ----
    Greedy is inherently sequential: ``n_kept - n_protected`` steps, each reducing the whole
    demand tensor. Per layer that is ``O(n_kept * n_demand * k_len)`` **elementwise** work --
    memory-bound, not matmul, so FLOP comparisons against the model's own forward flatter it.

    Worked example, Llama-3-8B (32 layers, ``Hkv=8``, ``group_size=4``) at a **4K context** and
    ``compression_ratio=0.5`` (so ``n_kept=2048``), measured on CPU and extrapolated per layer:

    ==========================  ==========  ===========  ==============
    config                      demand      1 layer      all 32 layers
    ==========================  ==========  ===========  ==============
    ``stack``, no knobs         16384x4096  ~1710 s      ~15 h
    ``max``, no knobs           4096x4096   ~470 s       ~4.2 h
    ``max`` + ``pool=1024``     4096x4096   ~157 s       ~1.4 h
    ``max`` + ``sample=512``    512x4096    ~59 s        ~31 min
    ``max`` + both              512x4096    ~19 s        ~10 min
    ``max`` + ``sample=128``,
    ``pool=256``                128x4096    ~0.4 s       ~13 s
    ==========================  ==========  ===========  ==============

    For scale, the model's own 4K prefill is ~74 TFLOPs, and the ``stack`` demand tensor alone
    is 2 GB in fp32 per layer. **Defaults are unusable at 4K** -- treat the knobs as required
    rather than optional, and prefer a GPU:

    - ``group_reduce="max"`` cuts demand rows by ``group_size`` (4x here) for free;
    - ``n_demand_sample`` shrinks each step's reduction. Unbiased (the objective is a *sum* over
      demands), so the guarantee survives;
    - ``candidate_pool`` narrows the key axis the greedy searches. Biased -- the optimum may lie
      outside the pool -- so the submodular guarantee is lost.

    Caveat: the returned "score" is a **selection rank**, not a comparable importance value
    (a set objective has no per-key score). It orders keys correctly *within* a layer and KV
    head, which is all ``ScorerPress`` needs, but it is not meaningful across heads or layers
    -- so do not combine this press with anything that reallocates budget between them by
    comparing scores.
    """

    compression_ratio: float = 0.0
    n_sink: int = 4
    n_local: int = 128
    normalize_by_visibility: bool = False
    n_demand_sample: int | None = None
    candidate_pool: int | None = None
    group_reduce: str = "stack"

    def __post_init__(self):
        super().__post_init__()
        if self.n_sink < 0 or self.n_local < 0:
            raise ValueError("n_sink and n_local must be non-negative")
        if self.group_reduce not in ("stack", "max"):
            raise ValueError(f"group_reduce must be 'stack' or 'max', got {self.group_reduce!r}")
        if self.n_demand_sample is not None and self.n_demand_sample <= 0:
            raise ValueError(f"n_demand_sample must be positive, got {self.n_demand_sample}")
        if self.candidate_pool is not None and self.candidate_pool <= 0:
            raise ValueError(f"candidate_pool must be positive, got {self.candidate_pool}")

    def build_demand(self, attentions: torch.Tensor, n_kv_heads: int, kwargs: dict) -> torch.Tensor:
        """
        Turn ``(B, n_heads, Sq, Sk)`` attention into ``(B, n_kv_heads, n_demand, Sk)`` demands.

        Padded query rows are dropped rather than zeroed: they are not real demands, and
        leaving them in would let a key earn coverage for serving padding.
        """
        bsz, n_heads, q_len, k_len = attentions.shape
        group_size = n_heads // n_kv_heads
        demand = attentions.view(bsz, n_kv_heads, group_size, q_len, k_len)

        if self.group_reduce == "max":
            demand = demand.amax(dim=2)
        else:
            demand = demand.reshape(bsz, n_kv_heads, group_size * q_len, k_len)

        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None and attention_mask.dim() == 2:
            keep = attention_mask[:, -q_len:].bool()
            if not keep.all():
                if self.group_reduce == "max":
                    demand = demand * keep.view(bsz, 1, q_len, 1)
                else:
                    demand = demand * keep.view(bsz, 1, 1, q_len, 1).expand(
                        bsz, 1, group_size, q_len, 1
                    ).reshape(bsz, 1, group_size * q_len, 1)

        return demand

    def subsample_demand(self, demand: torch.Tensor) -> torch.Tensor:
        """Uniformly subsample demand rows, an unbiased estimate of the summed objective."""
        n_demand = demand.shape[-2]
        if self.n_demand_sample is None or self.n_demand_sample >= n_demand:
            return demand
        idx = torch.randperm(n_demand, device=demand.device)[: self.n_demand_sample]
        return demand.index_select(-2, idx.sort().values)

    def protected(self, k_len: int, device: torch.device) -> torch.Tensor:
        """``(k_len,)`` bool mask of unconditionally kept keys (sink + local)."""
        mask = torch.zeros(k_len, dtype=torch.bool, device=device)
        n_sink = min(self.n_sink, k_len)
        n_local = min(self.n_local, max(k_len - n_sink, 0))
        if n_sink:
            mask[:n_sink] = True
        if n_local:
            mask[k_len - n_local :] = True
        return mask

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        assert attentions is not None, 'Set attn_implementation="eager" to use this press'

        bsz, n_kv_heads, k_len, _ = keys.shape
        # Mirrors ScorerPress.compress's budget exactly; the greedy has to select the same
        # number of keys that compress() will then keep, or the two disagree on the set.
        n_kept = int(k_len * (1 - self.compression_ratio))
        if n_kept <= 0:
            return torch.zeros(bsz, n_kv_heads, k_len, device=keys.device, dtype=torch.float32)

        demand = self.build_demand(attentions, n_kv_heads, kwargs).float()
        demand = self.subsample_demand(demand)

        if self.normalize_by_visibility:
            # Each key is visible to (k_len - j) query rows under causal masking; dividing by
            # that removes the count bias, at the cost of the coverage interpretation.
            visible = torch.arange(k_len, 0, -1, device=demand.device, dtype=demand.dtype)
            demand = demand / visible

        protected = self.protected(k_len, keys.device)
        n_protected = int(protected.sum())
        forced = protected.view(1, 1, k_len).expand(bsz, n_kv_heads, k_len)

        if n_protected >= n_kept:
            # Protection alone fills the budget. Warn rather than silently letting the greedy
            # pick nothing and ScorerPress's top-k break the tie by index order.
            logger.warning(
                "n_sink + n_local = %d fills the keep budget of %d at compression_ratio=%.2f; "
                "facility location has no slots left to allocate.",
                n_protected,
                n_kept,
                self.compression_ratio,
            )
            return protected.view(1, 1, k_len).expand(bsz, n_kv_heads, k_len).float()

        # Credit the protected keys' coverage before selecting, so the greedy spends its
        # slots on rows the protected set does NOT already serve.
        initial_cover = demand[..., protected].amax(dim=-1) if n_protected else None

        order = facility_location_order(
            demand,
            n_kept - n_protected,
            initial_cover=initial_cover,
            forced=forced,
            candidate_pool=self.candidate_pool,
        )

        # Encode the selection as a score: rank 0 (picked first) scores highest, and every
        # protected key outranks every greedy pick. ScorerPress's top-k then reproduces
        # exactly this set. Values are ranks, not importances -- see the class docstring.
        scores = torch.zeros(bsz, n_kv_heads, k_len, device=keys.device, dtype=torch.float32)
        budget = order.shape[-1]
        ranks = torch.arange(budget, 0, -1, device=keys.device, dtype=torch.float32)
        scores.scatter_(-1, order, ranks.expand(bsz, n_kv_heads, budget))
        scores = scores.masked_fill(protected.view(1, 1, k_len), float(budget + 1))
        return scores
