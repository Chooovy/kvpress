# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Per-row top-k score boundaries, for training under **hard eviction**.

Soft-gate training and hard-evict inference are different forward passes, and the first joint run
showed what that gap costs. With the router free to move, the backbone lowered the LM loss by
re-spreading attention through q/k until the gate stopped binding -- ``gate_sparsity`` went
0.282 -> 0.359 (the frozen-backbone arm reached 0.267) while the loss fell 1.75 -> 1.53. RULER 8K
then dropped 77.62 -> 69.17 with the entire loss confined to needle retrieval, and
``topk=8192`` scored 93.71 against the dense model's 93.69, i.e. **the backbone was unharmed**.
It had simply learned to route around a constraint that is soft during training and absolute at
inference. Training on the hard geometry removes that option.

Why a threshold and not an index set
------------------------------------
The obvious implementation gathers each row's top-k keys, which is what ``--stage sparse`` does.
Its backward is unaffordable: the gather reference has no fused kernel, so autograd retains every
tile's gathered ``k``/``v``, totalling ``O(B * Hkv * L * topk * D)`` -- **576 GiB across 36 layers
at topk=512, 2304 GiB at topk=2048**, and independent of ``query_tile``, which bounds only one
tile's peak rather than the retention. Against ~1 GiB/layer for the fused dense path, that is
three orders of magnitude.

A threshold costs the same as ``lse``: ``O(Sq)`` per (batch, KV head). It is *exactly* equivalent
to a per-row top-k, but only under two conditions, and both have to hold:

1. **The router is frozen.** A key's score does not change within the step, so a boundary
   computed once stays correct for the whole forward and backward.
2. **The router is query-independent** (the scalar/kvzip arms). A key has ONE score rather than
   one per query, so "key j is in query i's top-k" reduces to "score_j >= thresh_i".

Neither holds for the pairwise indexer or for a training router, and
:func:`topk_threshold` refuses those cases rather than silently computing a boundary that
means something else.

What the gradient does under a hard gate
----------------------------------------
An evicted key contributes nothing to the softmax, so it receives no gradient -- which is the
documented weakness of sparse-scope training and the reason ``--sft-ruler`` rejects it. Here that
is **not** a problem, and the reason is specific: the router is frozen, so nothing needs to learn
to start selecting a key it currently misses. The only thing training is the backbone, and what
it has to learn is how to use the keys the router does select. Those keys are exactly the ones
carrying gradient.
"""

from __future__ import annotations

import torch


def topk_threshold(
    scores: torch.Tensor,
    topk: int,
    *,
    n_sink: int = 0,
    n_local: int = 0,
    query_offset: int = 0,
    q_len: int | None = None,
) -> torch.Tensor:
    """
    Each query row's top-k score boundary over its *evictable* history, ``(B, h, Sq)`` fp32.

    A non-pinned key is kept iff its score is ``>=`` the returned threshold, so the kernel's
    comparison reproduces a per-row top-k without materializing one.

    Parameters
    ----------
    scores : torch.Tensor
        Per-key scores ``(B, h, Sk)`` from a query-independent scorer's ``score_keys`` /
        ``score_at``. Must already carry whatever position term the gate applies.
    topk : int
        Keys to retain per row, counted over the evictable set only -- pinned keys (sinks and the
        local window) are kept regardless and are excluded from both the ranking and the budget,
        matching ``pinned_mask`` and the eviction path's ``force_sink``/``force_local``.
    n_sink, n_local : int
        The pin geometry, which must match training's. Getting these wrong shifts every threshold
        and silently trains against a different keep-set.
    query_offset : int
        Absolute position of query row 0.
    q_len : int, optional
        Number of query rows; defaults to ``Sk - query_offset``.

    Returns
    -------
    torch.Tensor
        ``(B, h, Sq)`` fp32 thresholds. A row with no evictable key, or with fewer than ``topk``
        of them, gets ``-inf`` so that every visible key passes -- the honest answer, since there
        is nothing to evict.
    """
    if scores.dim() != 3:
        raise ValueError(f"scores must be (B, h, Sk), got {tuple(scores.shape)}")
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    bsz, n_heads, k_len = scores.shape
    if q_len is None:
        q_len = k_len - query_offset
    if q_len <= 0:
        raise ValueError(f"q_len must be positive, got {q_len}")

    scores = scores.float()
    device = scores.device
    k_pos = torch.arange(k_len, device=device)
    q_pos = torch.arange(q_len, device=device) + query_offset

    # Evictable = causal AND not pinned. Built per row because the local window moves with the
    # query; O(Sq * Sk) bool, which is why this runs once per layer under no_grad rather than
    # inside the tile loop.
    causal = k_pos.view(1, -1) <= q_pos.view(-1, 1)
    pinned = k_pos.view(1, -1) < n_sink
    if n_local > 0:
        age = q_pos.view(-1, 1) - k_pos.view(1, -1)
        pinned = pinned | ((age >= 0) & (age < n_local))
    evictable = causal & ~pinned  # (Sq, Sk)

    # -inf on non-evictable slots so they cannot enter the ranking. Done with masked_fill on an
    # expanded view rather than by gathering, to keep one (Sq, Sk) buffer instead of Sq of them.
    masked = scores.unsqueeze(2).expand(bsz, n_heads, q_len, k_len).clone()
    masked.masked_fill_(~evictable.view(1, 1, q_len, k_len), -float("inf"))

    n_evictable = evictable.sum(-1)  # (Sq,)
    # kthvalue/topk need k <= the row's count, and rows differ. Clamping to the row count and
    # then overwriting the short rows with -inf is simpler than a ragged gather, and gives the
    # right answer: a row with <= topk evictable keys evicts nothing.
    k_eff = int(min(topk, k_len))
    boundary = masked.topk(k_eff, dim=-1, largest=True, sorted=True).values[..., -1]
    return torch.where(
        (n_evictable >= topk).view(1, 1, q_len),
        boundary,
        torch.full_like(boundary, -float("inf")),
    )


def assert_router_frozen(model, scorer_attr: str = "indexer") -> int:
    """
    Verify every router parameter is frozen. Call **before** FSDP wraps the model.

    Returns the number of router parameters checked.

    Timing is the whole point. Inside a forward, FSDP marks its all-gathered parameters
    ``requires_grad=True`` regardless of the user's setting, so the same assertion made there
    fires on a correctly-frozen run -- observed exactly that. Before wrapping, the flags are the
    ones ``split_trainable`` set.
    """
    from kvpress.presses.gqa_indexer.press import get_language_model

    trainable, total = [], 0
    for layer in get_language_model(model).layers:
        indexer = getattr(layer.self_attn, scorer_attr, None)
        if indexer is None:
            continue
        for name, param in indexer.named_parameters():
            total += 1
            if param.requires_grad:
                trainable.append(name)
    if trainable:
        raise ValueError(
            f"hard-evict training needs a FROZEN router, but {len(trainable)} of {total} indexer "
            f"parameter(s) require grad (e.g. {sorted(set(trainable))[:3]}). A moving score makes "
            "the per-row threshold stale within the step, and an evicted key gets no gradient to "
            "re-enter with. Pass --freeze-router."
        )
    return total


def hard_threshold_for_layer(
    indexer,
    hidden_states: torch.Tensor,
    topk: int,
    *,
    n_sink: int,
    n_local: int,
    query_offset: int = 0,
    q_len: int | None = None,
) -> torch.Tensor:
    """
    Score one layer's keys with a **frozen, query-independent** indexer and return its thresholds.

    Refuses a scorer that is not query-independent, and refuses one whose parameters still require
    gradients. Both would break the equivalence between a threshold and a per-row top-k, and
    neither is visible in the resulting tensor -- the run would simply train against a keep-set
    that drifts inside the step.

    Runs under ``no_grad``: the boundary is a selection, not a differentiable quantity, and
    retaining its graph would defeat the memory argument for using a threshold at all.
    """
    if not getattr(indexer, "is_query_independent", False):
        raise ValueError(
            f"{type(indexer).__name__} is not query-independent, so a per-key score is not "
            "defined and a per-row threshold cannot stand in for a top-k. Hard-evict training "
            "needs --scorer scalar / kvzip (or the sparse scope, which is unaffordable here -- "
            "see this module's docstring)."
        )
    # NO requires_grad check here, deliberately. This runs INSIDE the forward, and FSDP sets
    # requires_grad=True on the parameters it has just all-gathered so that the recomputation
    # graph can be built -- so a frozen router reads as trainable at exactly this point, which
    # made the first version of this guard fire on a correctly-frozen run. The real check is
    # :func:`assert_router_frozen`, called once before FSDP wraps anything, where the flags still
    # mean what they say.

    with torch.no_grad():
        if getattr(indexer, "decay", False):
            # With a learned lifetime the score is not a property of the key alone, so the
            # ranking has to be taken at a stated query position. The LAST query is the
            # conservative choice for a full-sequence forward: it is the position whose view is
            # most decayed, so a key kept there is kept by every earlier query too.
            k_len = hidden_states.shape[1]
            scores = indexer.score_at(hidden_states, float(k_len - 1), key_offset=query_offset)
        else:
            scores = indexer.score_keys(hidden_states, key_offset=query_offset)
        return topk_threshold(
            scores,
            topk,
            n_sink=n_sink,
            n_local=n_local,
            query_offset=query_offset,
            q_len=q_len,
        )
