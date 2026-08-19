# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Gated attention: the indexer score added *inside* the attention softmax.

This is the end-to-end training counterpart of distillation. Distillation trains the indexer
against the teacher's attention weights and the score never touches the model's own forward,
so the LM loss cannot reach it. Here the score enters the logits::

    out = softmax(scale * q @ k^T + lam * qi @ ki^T) @ v

and ``dL/dscore`` follows from the ordinary attention backward. Nothing is distilled and no
teacher is needed -- the objective is the model's own loss.

Why inside the softmax
----------------------
The alternative -- ``softmax(q @ k^T) * g`` -- looks equivalent and is not. Writing ``p`` for
the ungated attention probability and ``p~`` for the gated one, the two gate gradients are::

    outer:  dg_m = sum_{i in m} p_i  * do^T v_i
    inner:  dg_m = sum_{i in m} p~_i / g_m * do^T (v_i - out)

Only the inner form carries ``(v_i - out)``: a *relative* signal saying whether key ``i``
pulls the output somewhere better than where it already is, which is what lets the gate
reallocate attention mass. The outer form multiplies a fixed probability, so it can scale a
key's contribution but never reorder one against another. SAS ablates both and the outer form
is markedly worse (42.0 vs 55.6 on MATH500).

Pinning: what stops the gate doing nothing
------------------------------------------
A gate that is **flat along the key axis** is a no-op -- a per-key constant cancels in the
softmax -- so the model reverts to the frozen pretrained backbone and earns a good LM loss with
no ranking learned. With a raw score that point sits at ``s = 0``, reachable from anywhere at
zero cost. This is SAS Table 1 row (6): 18.8 against 54.4 for the pinned form.

With a positive budget, ``pin_mode`` exempts some keys from the gate's normalizer, which removes
the flat solution: a pinned key sits at ``log 1 = 0`` while history shares a fixed total
multiplier, so history cannot recover the raw dense gate. ``gate_budget=0`` deliberately restores
that raw-score ablation. See :mod:`~kvpress.presses.gqa_indexer.gate_pin` for the pin geometry.

``scope="sparse"`` needs none of this: a sparse training forward removes the dense fallback
outright, which is also why DMA / SparseK / STE never needed it.

Implementations, and why the kernel exists
------------------------------------------
Three paths, in preference order:

1. :mod:`~kvpress.presses.gqa_indexer.triton_gated_attention` -- the fused kernel. Computes the
   gate as a second ``tl.dot`` inside the tile loop, so ``Dqk`` and ``Dv`` stay at their true
   widths and nothing ``(Sq, Sk)`` is materialized. **The only path with O(L) memory**, and it
   handles every pin mode including the query-dependent ones.
2. The concat form on SDPA (below) -- correct, and ``O(L)`` *only* when a fused SDPA backend is
   eligible. It frequently is not; see the next paragraph.
3. :func:`gated_attention_pinned_self` -- the two-branch fallback for a query-dependent pin
   without Triton. ``O(Sq * Sk)``, for tests and short sequences.

The SDPA route was tried first and OOM'd twice, which is what the kernel is for. Folding the gate
into QK needs ``Dqk = D + Di`` while ``Dv`` stays ``D``; flash requires ``Dqk == Dv``, so SDPA
chose the **math** backend, which retains the whole attention matrix -- 144 GiB across 36 layers
at ``L=8192, Hq=32``. Padding V to match made the heads 256 wide, past what flash and
mem-efficient support in the *backward* pass, so it landed on math again.

The concat identity
-------------------
The gate is *bilinear*, and that is what makes this cheap. For a query head reading KV head
``h``::

    scale * (q . k) + lam * (qi . ki)  ==  scale * ([q, (lam/scale) qi] . [k, ki])

so gated attention **is** ordinary attention at ``head_dim = D + Di``, and any dense kernel
computes it -- with the gradient reaching ``qi``/``ki`` for free. Verified to ~1e-15 in
``test_concat_matches_explicit``.

What actually decides whether folding is available is **not** block-vs-token granularity -- a
block gate folds too, as long as it is bilinear (broadcast the pooled block key back to its
tokens on the key side). Two things decide it:

1. **The score must be bilinear.** ``qi . ki`` folds. Any score that puts a nonlinearity or a
   projection *after* the query-key interaction does not, and must be materialized.
2. **The gate must be uniform over keys.** SAS pins its self-block gate to 1, which breaks the
   fold -- see "No normalizer" below, since it is the same property that breaks the normalizer's
   inertness.

This indexer satisfies both, so it needs no gate table and no kernel. SAS satisfies neither and
therefore needs one. Where granularity *does* matter is how expensive that table is when you do
need it: at ``L=32K, H=4`` the ``(query, block)`` table is 0.12-0.25 GiB at block size 128-64,
but a ``(query, token)`` table is 16 GiB -- so a design that must materialize the gate is
affordable at block granularity and not at token granularity.

Scope
-----
``full`` (stage 1) gates every key. ``sparse`` (stage 2) gates only each row's own top-k.
These are different objectives, not two implementations of one -- see :func:`gated_attention`.
"""

from __future__ import annotations

import logging
import math

import torch

from kvpress.presses.gqa_indexer.fused_loss import accumulation_dtype
from kvpress.presses.gqa_indexer.gate_pin import (
    check_pin_mode,
    gate_from_score,
    history_lse,
    is_query_dependent,
    pinned_mask,
    pins_self,
    pins_sink,
)
from kvpress.presses.gqa_indexer.sparse_attention import sparse_gqa_attention_reference
from kvpress.presses.gqa_indexer.triton_gated_attention import (
    gated_kernels_available,
    triton_gated_attention,
)

logger = logging.getLogger(__name__)

SCOPES = ("full", "sparse")


def causal_mask_bottom_right(
    q_len: int, k_len: int, device: torch.device, dtype: torch.dtype, query_offset: int | None = None
) -> torch.Tensor:
    """
    Additive ``(1, 1, Sq, Sk)`` causal mask, **bottom-right** aligned.

    Needed because SDPA's ``is_causal=True`` is **top-left** aligned: at ``Sq=3, Sk=5`` it lets
    query 0 see key 0 only, where bottom-right alignment lets it see keys 0-2. The two coincide
    exactly when ``Sq == Sk`` and disagree otherwise -- silently, since both produce a
    well-formed lower-triangular attention.

    Bottom-right is the convention used by flash-attention and by
    :func:`~.indexer.build_indexer_mask`, so it is the one that has to win: a decode step or a
    chunked prefill (both ``Sq < Sk``) would otherwise train the gate against a different key
    set than the indexer scored.
    """
    if query_offset is None:
        query_offset = k_len - q_len
    if query_offset < 0:
        # Bottom-right alignment puts query 0 at absolute position `k_len - q_len`, so a
        # negative offset means the leading queries sit before every key and see nothing at
        # all: their logits are uniformly -inf and the softmax returns NaN, which then spreads
        # through the model. Causally this cannot happen -- the keys of a causal layer include
        # every past position, so `k_len >= q_len` always -- which makes it a caller error
        # (usually q/k swapped, or a decode step given the prefill's query length) rather than a
        # case to paper over.
        raise ValueError(
            f"q_len={q_len} exceeds k_len={k_len} (query_offset={query_offset}): the first "
            f"{-query_offset} queries would have no visible key and produce NaN. A causal layer "
            "always has k_len >= q_len; check that q and k are not swapped."
        )
    q_pos = torch.arange(q_len, device=device).unsqueeze(-1) + query_offset
    k_pos = torch.arange(k_len, device=device).unsqueeze(0)
    mask = torch.zeros((1, 1, q_len, k_len), device=device, dtype=dtype)
    return mask.masked_fill_((k_pos > q_pos).view(1, 1, q_len, k_len), -float("inf"))



def check_gate_shapes(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
) -> tuple[int, int, int, int, int, int, int]:
    """
    Validate the gated-attention shapes and return the geometry.

    Returns ``(bsz, n_heads, n_kv_heads, group_size, q_len, k_len, idx_dim)``.

    ``q_idx`` carries ``n_kv_heads`` heads and ``k_idx`` a single shared one (the indexer is
    MQA on the key side), so the gate is per KV head -- the granularity at which GQA holds
    separate caches, and the same granularity the press evicts at.
    """
    if q.dim() != 4:
        raise ValueError(f"q must be (B, H, Sq, D), got {tuple(q.shape)}")
    if k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"k and v must be (B, Hkv, Sk, D), got {tuple(k.shape)}, {tuple(v.shape)}")
    if q_idx.dim() != 4:
        raise ValueError(f"q_idx must be (B, Hkv, Sq, Di), got {tuple(q_idx.shape)}")
    if k_idx.dim() != 3:
        raise ValueError(f"k_idx must be (B, Sk, Di), got {tuple(k_idx.shape)}")

    bsz, n_heads, q_len, _ = q.shape
    bsz_k, n_kv_heads, k_len, _ = k.shape
    if bsz_k != bsz or v.shape[0] != bsz:
        raise ValueError(f"batch mismatch: q={bsz}, k={bsz_k}, v={v.shape[0]}")
    if v.shape[1] != n_kv_heads:
        raise ValueError(f"k has {n_kv_heads} heads, v has {v.shape[1]}")
    if v.shape[2] != k_len:
        raise ValueError(f"k has {k_len} keys, v has {v.shape[2]}")
    if k.shape[-1] != q.shape[-1]:
        raise ValueError(f"q head_dim {q.shape[-1]} != k head_dim {k.shape[-1]}")
    if n_heads % n_kv_heads != 0:
        raise ValueError(f"n_heads {n_heads} is not a multiple of n_kv_heads {n_kv_heads}")

    idx_dim = q_idx.shape[-1]
    if q_idx.shape[:3] != (bsz, n_kv_heads, q_len):
        raise ValueError(
            f"q_idx must be (B={bsz}, Hkv={n_kv_heads}, Sq={q_len}, Di), got {tuple(q_idx.shape)}"
        )
    if k_idx.shape != (bsz, k_len, idx_dim):
        raise ValueError(
            f"k_idx must be (B={bsz}, Sk={k_len}, Di={idx_dim}), got {tuple(k_idx.shape)}"
        )

    return bsz, n_heads, n_kv_heads, n_heads // n_kv_heads, q_len, k_len, idx_dim


def _gate_lse(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    gate_scale: torch.Tensor | float,
    gate_budget: float,
    gate_budget_ratio: float | None,
    pinned: torch.Tensor,
    causal_keep: torch.Tensor | None,
    key_tile: int,
) -> torch.Tensor:
    """Return the row shift for raw, fixed-budget, or row-wise ratio-budget gating."""
    if gate_budget_ratio is None and gate_budget == 0:
        return torch.zeros(
            q_idx.shape[:3], device=q_idx.device, dtype=accumulation_dtype(q_idx, k_idx)
        )

    if gate_budget_ratio is None:
        return history_lse(
            q_idx, k_idx, gate_scale=gate_scale, pinned=pinned, causal_keep=causal_keep,
            key_tile=key_tile,
        ) - math.log(gate_budget)

    lse, n_history = history_lse(
        q_idx, k_idx, gate_scale=gate_scale, pinned=pinned, causal_keep=causal_keep,
        key_tile=key_tile, return_history_count=True,
    )
    budget = n_history.to(lse.dtype) * gate_budget_ratio
    log_budget = torch.where(n_history > 0, budget.log(), torch.zeros_like(budget))
    return lse - log_budget


def build_concat_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    *,
    scale: float,
    gate_scale: torch.Tensor | float,
    group_size: int,
    lse: torch.Tensor | None = None,
    history: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fold the gate into the QK dot product -- see "The concat identity" above.

    Returns ``(Q, K)`` such that ``scale * Q @ K^T`` equals the gated logits.

    Without pinning (``lse is None``) the width is ``D + Di`` and the identity is
    ``scale * q.k + gate_scale * qi.ki``.

    With a **query-independent** pin (``lse`` and ``history`` given) the width is
    ``D + Di + 1``. The extra dimension carries the ``-LSE`` normalizer, which is rank one --
    a per-query scalar times a per-key indicator -- so a single dimension suffices::

        Q = [q, (gate_scale/scale) * qi, -lse/scale]
        K = [k, ki * 1_history,           1_history ]

    On a history key this yields ``scale*q.k + gate_scale*qi.ki - lse``; on a pinned key both
    gate terms vanish and it yields ``scale*q.k`` -- exactly a gate of ``0``. Verified to
    1.1e-15 in ``test_sink_pin_folds_into_concat``.

    ``history`` must be a per-key mask (broadcastable to ``(1, k_len)``). A query-dependent
    pin cannot be expressed here at all: zeroing ``k_idx`` at a *different* position per query
    is not something a shared key matrix can do, which is why ``self`` pinning takes the
    two-branch path in :func:`gated_attention_pinned` instead.

    The ``gate_scale / scale`` factor rides on the *query* side. It could equally go on the
    key side, but ``q_idx`` is the tensor already being broadcast per query head, and keeping
    ``k_idx`` untouched means the shared indexer key can be cached across steps verbatim.
    """
    # gate_scale is a 0-dim Parameter on the e2e path, so this stays inside autograd and the
    # scalar gets its gradient like any other leaf.
    q_gate = q_idx * (gate_scale / scale)
    if group_size != 1:
        q_gate = q_gate.repeat_interleave(group_size, dim=1)
    query_parts = [q, q_gate.to(q.dtype)]

    n_kv_heads, k_len = k.shape[1], k.shape[2]
    key_gate = k_idx.unsqueeze(1).expand(-1, n_kv_heads, k_len, -1)
    key_parts = [k, key_gate.to(k.dtype)]

    if lse is not None:
        if history is None:
            raise ValueError("build_concat_qk needs `history` alongside `lse` to fold a pin")
        keep = history.reshape(1, 1, k_len, 1).to(k.dtype)
        # Zero the indexer key on pinned positions so their gate term drops out entirely.
        key_parts[1] = key_parts[1] * keep
        key_parts.append(keep.expand(k.shape[0], n_kv_heads, k_len, 1))
        norm = (-lse / scale).unsqueeze(-1)
        if group_size != 1:
            norm = norm.repeat_interleave(group_size, dim=1)
        query_parts.append(norm.to(q.dtype))

    return torch.cat(query_parts, dim=-1), torch.cat(key_parts, dim=-1)


def gated_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    *,
    scaling: float | None = None,
    gate_scale: torch.Tensor | float = 1.0,
    gate_budget: float = 1.0,
    gate_budget_ratio: float | None = None,
    mask: torch.Tensor | None = None,
    query_offset: int | None = None,
    pin_mode: str = "none",
    n_sink: int = 0,
    key_tile: int = 1024,
) -> torch.Tensor:
    """
    Explicit reference: build the gate, add it to the logits, softmax densely.

    Materializes the full ``(B, H, Sq, Sk)`` logits, so this is for tests -- but it is the
    literal reading of the formula, with the gate visible as its own term, and is what the
    folded and two-branch paths are checked against.

    Parameters
    ----------
    q : torch.Tensor
        Queries ``(B, H, Sq, D)``, post-RoPE.
    k, v : torch.Tensor
        Keys ``(B, Hkv, Sk, D)`` and values ``(B, Hkv, Sk, Dv)``, post-RoPE.
    q_idx : torch.Tensor
        Indexer queries ``(B, Hkv, Sq, Di)``, after norm and RoPE.
    k_idx : torch.Tensor
        Shared indexer key ``(B, Sk, Di)``, after norm and RoPE.
    scaling : float, optional
        Attention softmax scale; defaults to ``D ** -0.5``.
    gate_scale : torch.Tensor or float
        Multiplier on the gate. A 0-dim Parameter keeps it learnable.
    gate_budget : float
        ``0`` uses the raw history gate. A positive value fixes history's total multiplier.
    gate_budget_ratio : float, optional
        Fix each row's history budget to this ratio times its visible history length.
    mask : torch.Tensor, optional
        Additive ``(B, 1, Sq, Sk)`` mask. ``None`` applies plain causal masking.
    query_offset : int, optional
        Key index of query 0's diagonal; defaults to ``Sk - Sq``.
    pin_mode : str
        Which keys are exempt from the gate; see
        :mod:`~kvpress.presses.gqa_indexer.gate_pin`. ``"none"`` reproduces the ungated-budget
        behaviour, where the no-op is reachable.
    n_sink : int
        Leading keys to pin under ``sink`` / ``self+sink``.

    Returns
    -------
    torch.Tensor
        ``(B, H, Sq, Dv)`` in ``q``'s dtype.
    """
    bsz, n_heads, n_kv_heads, group_size, q_len, k_len, _ = check_gate_shapes(q, k, v, q_idx, k_idx)
    check_pin_mode(pin_mode)
    scale = q.shape[-1] ** -0.5 if scaling is None else float(scaling)

    # At least fp32 for the softmax, matching every kernel path here -- but not narrower than
    # the caller's dtype, or an fp64 reference test would measure its own rounding.
    acc = accumulation_dtype(q, k, v, q_idx, k_idx)
    logits = torch.einsum(
        "bhqd,bhkd->bhqk", q.to(acc), _expand_kv(k.to(acc), group_size)
    ) * scale
    score = torch.einsum("bhqd,bkd->bhqk", q_idx.to(acc), k_idx.to(acc)) * _as_acc(gate_scale, acc)

    pinned = pinned_mask(
        pin_mode, q_len, k_len, q.device, n_sink=n_sink, query_offset=query_offset
    )
    if pinned is None:
        gate = score
    else:
        lse = _gate_lse(
            q_idx, k_idx, gate_scale, gate_budget, gate_budget_ratio, pinned,
            causal_keep=_visible(mask, q_len, k_len, q.device, query_offset),
            key_tile=key_tile,
        )
        gate = gate_from_score(score, lse, pinned)
    logits = logits + _expand_kv(gate, group_size)

    if mask is None:
        logits = logits + causal_mask_bottom_right(q_len, k_len, q.device, acc, query_offset)
    else:
        logits = logits + mask.to(logits.dtype)

    p = logits.softmax(dim=-1)
    out = torch.einsum("bhqk,bhkd->bhqd", p, _expand_kv(v.to(acc), group_size))
    return out.to(q.dtype)


def pad_value_to_width(v: torch.Tensor, width: int) -> tuple[torch.Tensor, int]:
    """
    Widen ``v``'s head dim to ``width`` with zeros, returning ``(padded, original_dim)``.

    **This is what keeps the memory-efficient SDPA backends available.** Flash attention requires
    ``Q.size(-1) == V.size(-1)``, and the concat trick deliberately widens Q and K to ``D + Di``
    while V stays at ``Dv``. That mismatch makes SDPA fall back to the **math** backend, which
    materializes the full ``(B, H, Sq, Sk)`` attention weights *and retains them for backward* --
    4.0 GiB per layer at ``L=8192, Hq=32`` in bf16, so 144 GiB across 36 layers, which is what
    OOM'd the first real run.

    The fallback is silent: SDPA returns correct numbers either way, and the only symptom is the
    memory. That is why the first version of this module shipped with it -- the check performed
    was "does SDPA accept these shapes", not "which backend does it pick".

    Padding costs one extra ``O(L * (width - Dv))`` transient and doubles the ``P @ V`` GEMM at
    ``Di == D``, against a two-orders-of-magnitude memory reduction. The output's extra columns
    are sliced off by the caller; they are exactly zero, since every row of the pad is zero.

    A fused kernel would avoid the padding entirely by keeping ``Dv`` at its true width (two
    ``tl.dot``s per tile, one for the attention term and one for the gate) -- the remaining reason
    to write one.
    """
    dim_v = v.shape[-1]
    if dim_v == width:
        return v, dim_v
    if dim_v > width:
        raise ValueError(f"value head_dim {dim_v} exceeds the query width {width}")
    pad = v.new_zeros((*v.shape[:-1], width - dim_v))
    return torch.cat([v, pad], dim=-1), dim_v


def gated_attention_full(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    *,
    scaling: float | None = None,
    gate_scale: torch.Tensor | float = 1.0,
    gate_budget: float = 1.0,
    gate_budget_ratio: float | None = None,
    mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    query_offset: int | None = None,
    pin_mode: str = "none",
    n_sink: int = 0,
    key_tile: int = 1024,
    block_m: int = 64,
    block_n: int = 64,
    return_row_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    """
    Stage-1 gated attention over the **full** key axis.

    Every key participates, so every key's gate gets its own content-dependent gradient. That
    is the property stage 2 gives up, and the reason SAS finds full scope worth its
    ``O(n^2)``: under sparse scope an unselected key has no ``dg`` of its own and moves only
    through the softmax normalizer, so the whole unselected set is dragged up or down together
    rather than judged individually. Their Figure 5 shows it as a perfect line
    (``R^2 = 1.00``) through the unselected points; ``test_full_scope_gradients_are_independent``
    reproduces the diagnostic here.

    No memory is saved relative to dense attention -- the point is the gradient, not the FLOPs.
    Inference is where the saving happens, via :func:`gated_attention_sparse` or the press.

    ``mask=None`` builds a **bottom-right** aligned causal mask rather than using SDPA's
    ``is_causal``, which is top-left aligned -- see :func:`causal_mask_bottom_right` for why
    that distinction is load-bearing rather than cosmetic. When ``Sq == Sk`` the two agree and
    ``is_causal`` is taken instead, keeping the fast path for the common training shape.

    With a positive budget, ``pin_mode`` other than ``"none"`` exempts some keys from the gate so
    a flat gate can no longer be a no-op -- see :mod:`~kvpress.presses.gqa_indexer.gate_pin`. A
    query-independent pin (``sink``) folds into one extra concatenated dimension and stays a
    single SDPA call; a query-dependent one (``self``) cannot fold and is routed to
    :func:`gated_attention_pinned_self`.

    ``gate_budget=0`` disables history normalization and uses the raw gate score. A positive
    value fixes the history gate mass relative to unit-gated pinned keys. Alternatively,
    ``gate_budget_ratio`` sets each row's budget to its visible, non-pinned history length times
    that ratio. Neither budget changes the ranking within history, and neither has an effect
    without pinning.

    ``return_row_lse`` exposes the fused kernel's attention log-normalizer for lightweight
    diagnostics. Non-fused paths return ``None`` in its place rather than materializing an
    attention matrix solely to reconstruct it.
    """
    _, _, _, group_size, q_len, k_len, _ = check_gate_shapes(q, k, v, q_idx, k_idx)
    check_pin_mode(pin_mode)
    scale = q.shape[-1] ** -0.5 if scaling is None else float(scaling)
    if query_offset is None:
        query_offset = k_len - q_len

    pinned = pinned_mask(
        pin_mode, q_len, k_len, q.device, n_sink=n_sink, query_offset=query_offset
    )
    lse = None
    if pinned is not None:
        lse = _gate_lse(
            q_idx, k_idx, gate_scale, gate_budget, gate_budget_ratio, pinned,
            causal_keep=_visible(mask, q_len, k_len, q.device, query_offset),
            key_tile=key_tile,
        )

    # The fused kernel is the only path with O(L) memory. SDPA cannot provide it here: the
    # concat needs Dqk != Dv, which disqualifies flash, and padding V to match pushes the head
    # width to 256, which is past what flash/mem-efficient support in the backward pass -- so
    # SDPA lands on the math backend and retains the whole (Sq, Sk) matrix. See
    # triton_gated_attention's module docstring; both failure modes were real OOMs.
    if mask is None and gated_kernels_available(q, k, v, q_idx, k_idx):
        zeros_lse = lse is None
        return triton_gated_attention(
            q, k, v, q_idx, k_idx,
            torch.zeros(
                (q.shape[0], k.shape[1], q_len), device=q.device, dtype=torch.float32
            ) if zeros_lse else lse,
            gate_scale=_as_tensor(gate_scale, q.device),
            scaling=scale,
            query_offset=query_offset,
            n_sink=n_sink if pins_sink(pin_mode) else 0,
            pin_self=pins_self(pin_mode),
            block_m=block_m,
            block_n=block_n,
            return_row_lse=return_row_lse,
        )

    if is_query_dependent(pin_mode):
        out = gated_attention_pinned_self(
            q, k, v, q_idx, k_idx,
            scaling=scale, gate_scale=gate_scale, gate_budget=gate_budget,
            gate_budget_ratio=gate_budget_ratio, mask=mask,
            query_offset=query_offset,
            pin_mode=pin_mode, n_sink=n_sink, key_tile=key_tile,
        )
        return (out, None) if return_row_lse else out

    query, key = build_concat_qk(
        q, k, q_idx, k_idx, scale=scale, gate_scale=gate_scale, group_size=group_size,
        lse=lse, history=None if pinned is None else ~pinned[0],
    )

    is_causal = False
    if mask is None:
        if q_len == k_len:
            is_causal = True  # top-left == bottom-right here, so take SDPA's own fast path
        else:
            mask = causal_mask_bottom_right(q_len, k_len, q.device, query.dtype)

    # V is widened to the concatenated query width so flash/mem-efficient stay eligible; without
    # it SDPA drops to the math backend and retains the whole attention matrix. See
    # pad_value_to_width -- this line is the difference between O(L) and O(L^2) per layer.
    value, dim_v = pad_value_to_width(v, query.shape[-1])

    # `scale` is passed explicitly: SDPA would otherwise default to (D + Di) ** -0.5 from the
    # concatenated width, silently rescaling both the attention term and the gate.
    out = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask.to(query.dtype) if mask is not None else None,
        is_causal=is_causal,
        scale=scale,
        dropout_p=dropout_p,
        enable_gqa=group_size != 1,
    )
    out = out[..., :dim_v]
    return (out, None) if return_row_lse else out


def gated_attention_pinned_self(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    *,
    scaling: float | None = None,
    gate_scale: torch.Tensor | float = 1.0,
    gate_budget: float = 1.0,
    gate_budget_ratio: float | None = None,
    mask: torch.Tensor | None = None,
    query_offset: int | None = None,
    pin_mode: str = "self",
    n_sink: int = 0,
    key_tile: int = 1024,
) -> torch.Tensor:
    """
    Gated attention with a **query-dependent** pin, as two merged attentions.

    A ``self`` pin exempts each query's own diagonal key, so the pinned column moves with the
    row and no shared key matrix can express it -- the concat fold is unavailable (the naive
    attempt is wrong by 2.5, not by a rounding error). Instead:

    1. **History branch.** Attention over the non-pinned keys with the gate applied. The
       ``-LSE`` normalizer is a per-row constant *within this branch*, so it cancels in the
       branch's own softmax and does not need to be applied -- the branch is therefore a plain
       concat attention at width ``D + Di``, with no extra dimension.
    2. **Pinned branch.** Attention over the pinned keys with gate ``0``, i.e. ungated.
    3. **Merge** by log-sum-exp weights. This is where ``LSE`` re-enters: it sets how much mass
       the history branch wins against the pinned keys, which is the whole point of the budget.

    Exact, verified to 6.7e-16 against :func:`gated_attention_reference`.

    Implemented with explicit logits rather than two SDPA calls: SDPA does not return the
    log-sum-exp needed for the merge, and recovering it would cost a third pass over the same
    logits. That makes this path ``O(Sq * Sk)`` in memory, so it is currently the short-sequence
    / correctness path. A fused kernel (one pass, gate applied per tile) is the way to make
    ``self`` pinning viable at 32K; ``sink`` pinning needs none of this.
    """
    bsz, n_heads, n_kv_heads, group_size, q_len, k_len, _ = check_gate_shapes(q, k, v, q_idx, k_idx)
    check_pin_mode(pin_mode)
    scale = q.shape[-1] ** -0.5 if scaling is None else float(scaling)
    acc = accumulation_dtype(q, k, v, q_idx, k_idx)

    pinned = pinned_mask(
        pin_mode, q_len, k_len, q.device, n_sink=n_sink, query_offset=query_offset
    )
    if pinned is None:
        raise ValueError(f"gated_attention_pinned_self needs a pinning mode, got {pin_mode!r}")

    visible = _visible(mask, q_len, k_len, q.device, query_offset)
    lse = _gate_lse(
        q_idx, k_idx, gate_scale, gate_budget, gate_budget_ratio, pinned, visible, key_tile
    )
    score = torch.einsum("bhqd,bkd->bhqk", q_idx.to(acc), k_idx.to(acc)) * _as_acc(gate_scale, acc)
    gate = gate_from_score(score, lse, pinned)

    logits = torch.einsum(
        "bhqd,bhkd->bhqk", q.to(acc), _expand_kv(k.to(acc), group_size)
    ) * scale + _expand_kv(gate, group_size)
    logits = logits.masked_fill(~visible, -float("inf"))
    if mask is not None and mask.dtype != torch.bool:
        # `visible` already folded in which pairs are allowed; anything else the mask carries
        # (a soft bias, say) still has to be added.
        logits = logits + mask.to(logits.dtype).clamp_min(torch.finfo(acc).min / 4)

    p = logits.softmax(dim=-1)
    out = torch.einsum("bhqk,bhkd->bhqd", p, _expand_kv(v.to(acc), group_size))
    return out.to(q.dtype)


def gated_attention_sparse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    indices: torch.Tensor,
    *,
    scaling: float | None = None,
    gate_scale: torch.Tensor | float = 1.0,
    query_offset: int | None = None,
    query_tile: int = 128,
) -> torch.Tensor:
    """
    Stage-2 gated attention restricted to each row's own ``topk`` support.

    Same concat trick, dispatched to :func:`~.sparse_attention.sparse_gqa_attention_reference`
    -- which is differentiable in q/k/v, so the gate's gradient survives the gather.

    This is the scope that matches inference: the selected set is what the model will actually
    attend to, and the gate values inside it still tell it how to weight them. Its gradient is
    weaker than stage 1's by construction (an unselected key contributes nothing, so it gets no
    signal of its own), which is exactly the trade the two stages exist to measure.

    ``indices`` follows the convention :func:`~.sparse_support.sort_support` emits:
    ``(B, Hkv, Sq, topk)`` int32, ascending, ``-1`` in unused slots.
    """
    _, _, _, group_size, _, k_len, _ = check_gate_shapes(q, k, v, q_idx, k_idx)
    scale = q.shape[-1] ** -0.5 if scaling is None else float(scaling)
    query, key = build_concat_qk(
        q, k, q_idx, k_idx, scale=scale, gate_scale=gate_scale, group_size=group_size
    )
    out, _ = sparse_gqa_attention_reference(
        query,
        key,
        v,
        indices,
        scaling=scale,
        query_offset=query_offset,
        query_tile=query_tile,
    )
    return out


def gated_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    *,
    scope: str = "full",
    indices: torch.Tensor | None = None,
    scaling: float | None = None,
    gate_scale: torch.Tensor | float = 1.0,
    gate_budget: float = 1.0,
    gate_budget_ratio: float | None = None,
    mask: torch.Tensor | None = None,
    query_offset: int | None = None,
    dropout_p: float = 0.0,
    pin_mode: str = "none",
    n_sink: int = 0,
    key_tile: int = 1024,
    return_row_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    """
    Dispatch to :func:`gated_attention_full` or :func:`gated_attention_sparse`.

    ``scope="sparse"`` requires ``indices``; it is not derived here, because selection needs
    the ``topk`` budget and the sink/local reservations that live on the trainer, and picking a
    default would quietly train at a budget nobody chose.

    ``pin_mode`` applies to the full scope only. Under the sparse scope the forward pass is
    *already* restricted to the router's own top-k, so a flat gate does not recover dense
    attention and there is no no-op to block -- the same reason DMA and SparseK need no pin. The
    combination is rejected rather than silently ignored.

    Gate budgets are likewise full-scope quantities: ``gate_budget=0`` selects the raw gate, a
    positive fixed value sets history's total multiplier, and ``gate_budget_ratio`` instead sets
    it row-wise from the visible history length.
    """
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
    check_pin_mode(pin_mode)
    if scope == "full":
        if indices is not None:
            raise ValueError(
                "scope='full' gates every key, so `indices` would be ignored -- pass "
                "scope='sparse' to restrict the gate to a support"
            )
        return gated_attention_full(
            q, k, v, q_idx, k_idx,
            scaling=scaling, gate_scale=gate_scale, gate_budget=gate_budget,
            gate_budget_ratio=gate_budget_ratio, mask=mask, dropout_p=dropout_p,
            query_offset=query_offset, pin_mode=pin_mode, n_sink=n_sink, key_tile=key_tile,
            return_row_lse=return_row_lse,
        )

    if indices is None:
        raise ValueError("scope='sparse' needs `indices` (B, Hkv, Sq, topk)")
    if mask is not None:
        # The support was selected under a mask already; re-applying one here would have no
        # way to act (masked keys are simply absent from `indices`) and pretending otherwise
        # would hide a caller's misconception about where masking happens.
        raise ValueError(
            "scope='sparse' takes its masking from `indices`, which the selection step already "
            "built under the mask; pass mask=None"
        )
    if pin_mode != "none":
        raise ValueError(
            f"pin_mode={pin_mode!r} is meaningless under scope='sparse': the forward pass is "
            "already restricted to the selected keys, so a flat gate cannot recover dense "
            "attention and there is no no-op to pin against. Use pin_mode='none' here, and pin "
            "in the full-scope stage."
        )
    out = gated_attention_sparse(
        q, k, v, q_idx, k_idx, indices,
        scaling=scaling, gate_scale=gate_scale, query_offset=query_offset,
    )
    return (out, None) if return_row_lse else out


def _expand_kv(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Broadcast a per-KV-head tensor up to the query-head count along dim 1."""
    return x if group_size == 1 else x.repeat_interleave(group_size, dim=1)


def _visible(
    mask: torch.Tensor | None,
    q_len: int,
    k_len: int,
    device: torch.device,
    query_offset: int | None,
) -> torch.Tensor:
    """
    ``(Sq, Sk)`` bool of which pairs the attention can see, for the history normalizer.

    The gate's budget must be normalized over exactly the keys the row actually attends to. A
    key that is masked out but counted in the ``logsumexp`` would consume budget that no key
    receives, silently shrinking the gate on the rows with the most masking.

    An additive float mask is thresholded against a large negative value, which covers both
    ``-inf`` and the finite ``MASK_NEG`` sentinel this package uses. A batched mask is reduced
    with ``all`` over the batch: the normalizer is shared across the batch here, so the
    conservative choice is to count a key only where every sequence can see it.
    """
    causal = causal_mask_bottom_right(q_len, k_len, device, torch.float32, query_offset)
    keep = causal[0, 0] == 0
    if mask is not None:
        if mask.dtype == torch.bool:
            extra = mask
        else:
            extra = mask > (torch.finfo(mask.dtype).min / 2)
        extra = extra.reshape(-1, extra.shape[-2], extra.shape[-1]).all(dim=0)
        if extra.shape[-2] == 1:
            extra = extra.expand(q_len, k_len)
        keep = keep & extra
    return keep


def _as_tensor(gate_scale: torch.Tensor | float, device: torch.device) -> torch.Tensor:
    """The kernel loads gate_scale from memory, so a python float needs materializing."""
    if isinstance(gate_scale, torch.Tensor):
        return gate_scale
    return torch.tensor(float(gate_scale), device=device, dtype=torch.float32)


def _as_acc(gate_scale: torch.Tensor | float, acc: torch.dtype) -> torch.Tensor | float:
    """Cast a tensor gate_scale to the accumulation dtype, keeping it in the graph."""
    return gate_scale.to(acc) if isinstance(gate_scale, torch.Tensor) else gate_scale
