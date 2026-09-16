# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Split a layer's KV budget across heads so each retains the same **attention mass**.

Why mass, and not score
-----------------------
The trained gate is ``score - lse + log B`` on history and ``0`` on pinned keys
(:func:`~.gate_pin.gate_from_score`, :func:`~.gated_attention._gate_lse`). Add a constant ``c`` to
one (layer, head)'s whole score vector: ``lse`` shifts by the same ``c``, so ``score - lse`` is
unchanged, and the pinned branch never saw the score. **The forward is bitwise invariant to a
per-(layer, head) additive constant**, and eval's per-row top-k is invariant to it as well, since
it only ranks within a head.

So that constant is a gauge freedom -- nothing in training or inference determines it. (The model
cannot even represent it: ``ScalarIndexer``'s ``w_out`` is ``bias=False``.) A rule that pools raw
scores across heads and takes one top-k over the union -- AdaKV's
``scores.reshape(bsz, -1)``, and the same pattern in ``CriticalKVPress`` -- is ranking heads by an
unidentifiable quantity. ``facility_location_press.py`` warns about exactly this for its own
scores.

Retained softmax mass ``rho_h = sum_{j kept} p_j`` **is** invariant: it is computed from the
attention distribution, not from the router's score, so shifting the score changes nothing about
it.

What this does and does not fix
-------------------------------
Mass is used to *rank and size* the per-head allocation. The deployed constraint is still a
**count** -- each head keeps ``budget_h`` keys. So this does NOT reconcile the train-time mass
budget (``gate_budget`` constrains ``sum_history exp(gate) = B``) with the eval-time cardinality
budget; that mismatch, which ``hard_evict.py`` measured, is untouched. Mass enters only as the
currency in which "how much does this head need" is legally comparable across heads.

Equalizing mass is also the right *objective*, not just the only legal one: the marginal value of
one more key is its softmax weight, so "every head at the same mass" is the first-order condition
for "no slot can be moved to a head that would use it better" -- water-filling, with ``rho*``
playing the role of the price.

Evidence that the head<->budget correspondence is what matters
--------------------------------------------------------------
A **budget-shuffle control** (``head_budget="shuffle"``) permutes the same budget multiset across
heads: identical total, identical ragged shape, only the head assignment destroyed. RULER 8K goes
88.73 -> **80.49**, i.e. *below* the uniform baseline's 82.18, and ``niah_multikey_2`` /
``niah_multikey_3`` land back on uniform's 59.46 / 23.91 exactly. So raggedness by itself is worth
nothing (slightly harmful); the entire +6.55 comes from giving the budget to the right heads.


Measured headroom (``scratch/diag_head_budget.py``, fwkl_ce01 router, 8K, topk=2048): under the
uniform budget, retained mass spreads **0.268** across heads within a layer on average (max 0.606
at layer 4); matching lifts the worst head by **+0.154** mass on average (max +0.314). Several
heads in layers 0-4 are saturated by sink+local alone, while their siblings want 3.5x the uniform
budget.
"""

from __future__ import annotations

import torch


def head_mass_curve(
    query: torch.Tensor,
    key: torch.Tensor,
    scores: torch.Tensor,
    *,
    ref_row: int,
    scaling: float,
    force_sink: int,
    force_local: int,
) -> tuple[torch.Tensor, int]:
    """
    Cumulative retained attention mass per KV head along the router's own ranking.

    Returns ``(cum, n_pool)`` where ``cum[h, r]`` is the mass head ``h`` retains with the top
    ``r+1`` evictable keys **plus** its pinned mass -- so ``cum[h, take-1]`` is what today's
    uniform budget retains, and the curve is the head's demand function.

    Ordered by the **router's** score restricted to the evictable pool, not by mass, so the curve
    describes the policy that will actually run. An oracle ordering would overstate every head's
    achievable mass and would allocate against a selection the press cannot make.

    Query heads are folded into their KV head by averaging: the budget is per KV head, but the
    attention that has to survive belongs to the whole group it serves.
    """
    n_q_heads = query.shape[1]
    n_kv, k_len = key.shape[1], key.shape[2]
    q_len = query.shape[2]
    group = n_q_heads // n_kv
    device = query.device

    # `ref_row` is an ABSOLUTE key position, but `query` holds only this forward's rows. The two
    # coincide only when the forward starts at position 0 (a single whole-context prefill, which
    # is all RULER ever sends). A split prefill -- LongBench's context-then-question pipeline --
    # has `k_len > q_len`, and indexing the query axis with an absolute position then reads out of
    # bounds: observed `IndexError: index 17185 is out of bounds for dimension 2 with size 1067`
    # on triviaqa. Convert to a row index in this forward's frame.
    query_offset = k_len - q_len
    q_row = min(max(int(ref_row) - query_offset, 0), q_len - 1)

    qr = query[0, :, q_row, :].float()                         # (Hq, D)
    kf = key[0].float().repeat_interleave(group, 0)            # (Hq, Sk, D)
    logits = torch.einsum("hd,hsd->hs", qr, kf) * scaling
    key_idx = torch.arange(k_len, device=device)
    causal = key_idx <= ref_row
    logits = logits.masked_fill(~causal, torch.finfo(torch.float32).min)
    p_kv = torch.softmax(logits, dim=-1).view(n_kv, group, k_len).mean(1)   # (Hkv, Sk)

    sink = key_idx < force_sink
    local = (key_idx > ref_row - force_local) & (key_idx <= ref_row) & ~sink
    pool = causal & ~(sink | local)
    pinned_mass = p_kv[:, sink | local].sum(-1)

    neg = torch.finfo(torch.float32).min
    pooled = torch.where(pool.unsqueeze(0), scores.float(), torch.tensor(neg, device=device))
    # Stable descending, matching `deadlines`' tie-break so the curve is indexed by the same
    # ranking the mask will apply.
    order = torch.argsort(pooled, dim=-1, descending=True, stable=True)
    n_pool = int(pool.sum())
    cum = p_kv.gather(-1, order)[:, :n_pool].cumsum(-1) + pinned_mass.unsqueeze(-1)
    return cum, n_pool


def allocate_by_mass(cum: torch.Tensor, total: int, *, iters: int = 60) -> torch.Tensor:
    """
    Per-head key counts reaching a common mass target, with ``sum == total`` **exactly**.

    Bisects the target ``rho*`` on the monotone map ``rho* -> sum_h k_h(rho*)``, then settles the
    integer residual by marginal value: leftover slots go to the head whose next key is worth the
    most mass, and an overspend is reclaimed from the head whose last key is worth the least.

    Exact conservation is the load-bearing property. A vector that summed even slightly high would
    buy its win with extra cache, and the A/B against the uniform budget would measure the budget
    instead of the allocation.
    """
    n_heads, n_pool = cum.shape
    device = cum.device

    # `k` is clamped to >= 1 per head below, so a total under `n_heads` is unsatisfiable: the
    # reclaim loop would drive some head to 0 and then index `k - 1 == -1`. On GPU that appears as
    # a bare `device-side assert triggered` from a gather, with no hint of the cause. Hand back a
    # near-even split instead -- at this size the allocation carries no information anyway, since
    # every head gets 0 or 1 evictable slots.
    if total <= n_heads:
        base = torch.zeros(n_heads, dtype=torch.int64, device=device)
        if total > 0:
            base[:total] = 1
        return base

    def k_for(target: float) -> torch.Tensor:
        reached = cum >= target
        first = reached.to(torch.uint8).argmax(-1) + 1
        # A head that never reaches the target wants everything it can get.
        return torch.where(reached.any(-1), first, torch.full_like(first, n_pool))

    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if int(k_for(mid).sum()) > total:
            hi = mid
        else:
            lo = mid
    k = k_for(lo).clamp(min=1, max=n_pool)

    residual = total - int(k.sum())
    while residual != 0:
        if residual > 0:
            gain = torch.where(
                k < n_pool,
                cum.gather(-1, k.clamp(max=n_pool - 1).unsqueeze(-1)).squeeze(-1)
                - cum.gather(-1, (k - 1).unsqueeze(-1)).squeeze(-1),
                torch.full((n_heads,), -1.0, device=device),
            )
            if float(gain.max()) < 0:
                break  # every head is already at n_pool; nothing left to buy
            k[int(gain.argmax())] += 1
            residual -= 1
        else:
            loss = torch.where(
                k > 1,
                cum.gather(-1, (k - 1).unsqueeze(-1)).squeeze(-1)
                - cum.gather(-1, (k - 2).clamp(min=0).unsqueeze(-1)).squeeze(-1),
                torch.full((n_heads,), float("inf"), device=device),
            )
            if not bool(torch.isfinite(loss).any()):
                # Every head is already at its 1-slot floor, so there is nothing to reclaim.
                # Breaking leaves `sum(k) > total`, which the caller must not silently accept --
                # but it is strictly better than the alternative, which was to decrement a head to
                # 0 and then gather at index -1 (a device-side assert with no diagnosable cause).
                # Unreachable now that `total <= n_heads` returns early; kept as a guard because
                # the failure mode is invisible rather than loud.
                break
            k[int(loss.argmin())] -= 1
            residual += 1
    return k


def mass_head_budgets(
    query: torch.Tensor,
    key: torch.Tensor,
    scores: torch.Tensor,
    *,
    topk: int,
    force_sink: int,
    force_local: int,
    scaling: float,
    ref_row: int | None = None,
    floor: int = 0,
) -> torch.Tensor:
    """
    ``(n_kv_heads,)`` per-head ``topk`` values summing to ``n_kv_heads * topk``.

    ``floor`` reserves a minimum evictable budget per head before the split, the role
    TrimKV's ``min_tokens_per_head`` plays: a pooled allocation can otherwise starve a head to its
    pins, and a head that looks cheap on this one reference row may not be cheap on the rows that
    follow. 0 disables it.

    The reference row defaults to the **last** row of the prefill -- the position closest to where
    the question will be asked, and the row whose history is longest.
    """
    n_kv = key.shape[1]
    take = topk - force_sink - force_local
    if take <= 0:
        raise ValueError(f"topk={topk} leaves no evictable budget after the pins")
    row = key.shape[2] - 1 if ref_row is None else int(ref_row)

    cum, n_pool = head_mass_curve(
        query, key, scores,
        ref_row=row, scaling=scaling, force_sink=force_sink, force_local=force_local,
    )
    total = take * n_kv
    if n_pool <= 0 or total >= n_pool * n_kv:
        # Nothing to allocate: every head can keep its whole pool.
        return torch.full((n_kv,), topk, dtype=torch.int64, device=key.device)

    # `floor` is a per-head reservation, so it must leave a NON-NEGATIVE remainder to allocate:
    # `floor * n_kv <= total`, i.e. `floor <= take`. Clamping to `total // n_kv` is the same bound
    # only when the pool is large; with a small pool the reservation could exceed the budget and
    # `total - floor * n_kv` went NEGATIVE, after which `allocate_by_mass`'s residual loop
    # decrements past zero and indexes `k - 2 == -1`. On GPU that surfaces as a bare
    # `CUDA error: device-side assert triggered` from inside `allocate_by_mass` -- observed on
    # LongBench `multi_news` (context ~1900 tokens) once `--topk_ratio` made the budget scale with
    # the document: take=343 against floor=512.
    #
    # Also bounded by `n_pool - 1` so at least one column is left for the allocator to rank over.
    floor = max(0, min(int(floor), take, max(n_pool - 1, 0)))
    if floor:
        # Reserve, allocate the remainder, then add the reservation back. Bisecting on the
        # remainder rather than clamping afterwards is what keeps the total exact.
        k = allocate_by_mass(cum[:, floor:], total - floor * n_kv) + floor
    else:
        k = allocate_by_mass(cum, total)
    return (k + force_sink + force_local).to(torch.int64)


def fit_static_table(
    model,
    press,
    input_ids_list,
    *,
    topk: int,
    force_sink: int,
    force_local: int,
    floor: int = 0,
) -> torch.Tensor:
    """
    One ``(n_layers, n_kv_heads)`` budget table, fitted offline and shipped as a constant.

    The measurement that motivates this: the per-head demand ranking is as stable **across
    documents** as it is across reference rows within one document (Spearman 0.59 for the mass
    allocator, 0.64 for participation ratio -- ``scratch/diag_skeleton.py``). So a large part of
    "which heads need keys" is a property of the *model*, not of the input, and does not have to
    be measured at inference time at all.

    That matters for three reasons a per-document fit cannot claim:

    * **No prefill-time cost.** No cumulative mass curve, no bisection, no attention distribution
      to capture. The table is read.
    * **No decode question.** A budget that never depends on the input cannot drift as generation
      proceeds, which is the one real weakness of the per-document version (budgets refitted at
      L=3072 vs L=8192 disagree by ~0.53 of the total).
    * **It is a statement about the architecture**, testable and reportable: these heads are
      streaming heads, those are retrieval heads.

    Averaging is over the mass **curve position**, not over budgets: each document's allocation is
    fitted independently and the resulting counts are averaged, then re-integerized so the total
    is exact. Averaging the budgets rather than the curves keeps this cheap and matches how the
    table will be used.
    """
    from kvpress.presses.gqa_indexer.press import get_language_model

    layers = get_language_model(model).layers
    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    group = n_q // n_kv
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_q)
    scaling = head_dim ** -0.5

    grabbed: dict[int, tuple] = {}
    hidden: dict[int, torch.Tensor] = {}

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    import torch.nn.functional as F

    def impl(module, query, key, value, attention_mask, scaling_=None, dropout=0.0, **kw):
        grabbed[int(module.layer_idx)] = (query.detach(), key.detach())
        out = F.scaled_dot_product_attention(
            query, key.repeat_interleave(group, 1), value.repeat_interleave(group, 1),
            is_causal=True, scale=scaling_,
        )
        return out.transpose(1, 2).contiguous(), None

    name = "head_budget_fit"
    gmap = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
    ALL_ATTENTION_FUNCTIONS.register(name, impl)

    def pre_hook(module, args, kwargs):
        hs = kwargs.get("hidden_states")
        if hs is None and args:
            hs = args[0]
        hidden[int(module.layer_idx)] = hs.detach()
        return None

    handles = [
        layer.self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True) for layer in layers
    ]
    configs = [cfg] + ([cfg.text_config] if getattr(cfg, "text_config", None) else [])
    prev = [c._attn_implementation for c in configs]
    for c in configs:
        c._attn_implementation = name

    try:
        acc = torch.zeros((len(layers), n_kv), dtype=torch.float64)
        with torch.no_grad():
            for input_ids in input_ids_list:
                grabbed.clear()
                hidden.clear()
                model(input_ids=input_ids, use_cache=False)
                for layer_idx, layer in enumerate(layers):
                    query, key = grabbed[layer_idx]
                    h = hidden[layer_idx]
                    indexer = press.get_indexer(layer.self_attn)
                    ref = key.shape[2] - 1
                    if getattr(indexer, "decay", False):
                        scores = indexer.score_at(h, float(ref))[0]
                    else:
                        scores = indexer.score_keys(h)[0]
                    acc[layer_idx] += mass_head_budgets(
                        query, key, scores,
                        topk=topk, force_sink=force_sink, force_local=force_local,
                        scaling=scaling, floor=floor,
                    ).double().cpu()
    finally:
        for handle in handles:
            handle.remove()
        for c, p in zip(configs, prev):
            c._attn_implementation = p
        gmap.pop(name, None)

    acc /= max(len(input_ids_list), 1)
    # Re-integerize per layer with the total exact (largest remainder).
    total = topk * n_kv
    table = torch.zeros((len(layers), n_kv), dtype=torch.int64)
    for layer_idx in range(len(layers)):
        ideal = acc[layer_idx] * (total / acc[layer_idx].sum().clamp(min=1e-9))
        base = ideal.floor().long()
        residual = int(total - base.sum())
        if residual > 0:
            base[(ideal - base).argsort(descending=True)[:residual]] += 1
        elif residual < 0:
            base[(ideal - base).argsort()[: -residual]] -= 1
        floor_total = force_sink + force_local + floor
        table[layer_idx] = base.clamp(min=floor_total)
    return table
