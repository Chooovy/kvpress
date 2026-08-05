# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The GQA lightning indexer module."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

# Sentinel for "this (query, key) pair is not allowed". Kept finite so a fully masked row
# still produces finite logits; the loss helpers track validity with a boolean mask rather
# than relying on -inf arithmetic.
MASK_NEG = -1e4


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the halves of the last dimension, matching HuggingFace's RoPE convention."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def slice_rope_tables(cos: torch.Tensor, sin: torch.Tensor, rope_dim: int) -> tuple:
    """
    Narrow HuggingFace RoPE tables from their full width down to ``rope_dim`` channels.

    HF builds the tables as ``cat([freqs, freqs], -1)``, so a table of width ``W`` holds
    frequencies ``f[0..W/2-1]`` twice. ``rotate_half`` pairs channel ``j`` with
    ``j + rope_dim/2``, and that pair must be driven by ``f[j]``. The narrowed table
    therefore has to be ``[f[0..r/2-1], f[0..r/2-1]]`` -- the first ``r/2`` entries of
    *each* half.

    A plain contiguous prefix ``cos[..., :rope_dim]`` is NOT equivalent: it yields
    ``[f[0..r-1]]``, which drives the two halves of each pair with different frequencies.
    Striding (``cos[..., ::2]``) is likewise wrong -- it samples every other frequency.
    Both are silently wrong: no crash, just a degraded position signal.
    """
    width = cos.shape[-1]
    if rope_dim == width:
        return cos, sin
    if rope_dim > width:
        raise ValueError(f"rope_dim {rope_dim} exceeds RoPE table width {width}")
    half, r2 = width // 2, rope_dim // 2
    cos = torch.cat([cos[..., :r2], cos[..., half : half + r2]], dim=-1)
    sin = torch.cat([sin[..., :r2], sin[..., half : half + r2]], dim=-1)
    return cos, sin


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to ``x`` of shape (..., S, D) with ``cos``/``sin`` of shape (B, S, R).

    Only the leading ``R`` channels are rotated; the remaining ``D - R`` pass through
    untouched (NoPE), mirroring HF's ``apply_rotary_pos_emb`` and DSA, which rotates only
    ``qk_pos_emb_head_dim`` of ``index_head_dim``.

    ``cos``/``sin`` must already be narrowed to ``R`` via :func:`slice_rope_tables`, and
    are unsqueezed at ``dim=1`` to broadcast over the head axis.
    """
    rotary_dim = cos.shape[-1]
    if rotary_dim == 0:
        return x
    if rotary_dim % 2 != 0:
        raise ValueError(f"RoPE rotary_dim must be even, got {rotary_dim}")
    if rotary_dim > x.shape[-1]:
        raise ValueError(f"RoPE rotary_dim {rotary_dim} exceeds head_dim {x.shape[-1]}")

    cos = cos.unsqueeze(1).to(x.dtype)  # (B, 1, S, R)
    sin = sin.unsqueeze(1).to(x.dtype)
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_rot = x_rot * cos + rotate_half(x_rot) * sin
    return torch.cat((x_rot, x_pass), dim=-1) if x_pass.numel() else x_rot


@dataclass
class GQAIndexerConfig:
    """
    Shape configuration for :class:`GQAIndexer`.

    Attributes
    ----------
    hidden_size : int
        Model hidden size; input dim of both projections.
    n_heads : int
        Number of indexer heads. Set to ``num_key_value_heads`` so the indexer emits one
        score per KV head, which is the granularity at which GQA can actually evict.
    head_dim : int
        Per-head dimension of the indexer's q/k.
    rope_dim : int
        Number of leading channels to rotate. ``0`` disables RoPE; must be even.
    norm_eps : float
        Epsilon for the q/k LayerNorms.
    """

    hidden_size: int
    n_heads: int
    head_dim: int
    rope_dim: int = 0
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {self.rope_dim}")
        if self.rope_dim > self.head_dim:
            raise ValueError(f"rope_dim {self.rope_dim} cannot exceed head_dim {self.head_dim}")
        for name in ("hidden_size", "n_heads", "head_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")


class GQAIndexer(nn.Module):
    """
    Lightning indexer for grouped-query attention.

    Projects ``n_heads`` indexer queries and a single shared key (MQA), scores every
    (query, key) pair, and returns one score per indexer head. With
    ``n_heads == num_key_value_heads`` each KV head gets its own score and therefore
    selects its own top-k -- the arrangement MiniMax M3 uses, and the one GQA's physically
    independent KV caches make free.

    Deliberately minimal: no activation, no cross-head reduction, no per-head weighting.
    Each of those exists in DSA only to collapse many indexer heads into a *single* shared
    score, which MLA needs because it has one shared latent KV cache. Without that
    collapse they are dead weight or worse:

    - An activation cannot change a per-head top-k, since top-k is invariant to strictly
      increasing maps. ReLU is not strictly increasing -- it ties every negative score at
      0 and randomises selection among them, which matters precisely at the moderate
      compression ratios where the keep boundary falls in the negative region.
    - A per-``(token, head)`` scalar weight is constant along the key axis, so it cannot
      reorder a row: it is a no-op when positive and reverses the ranking when negative.

    For Llama-3.1-8B (32 attention heads, 8 KV heads, head_dim 128) this is 8 indexer
    query heads, 1 key head, head_dim 128: 4.7M params/layer against 21M for a design that
    mirrors the full query-head count, and an 8x smaller indexer key cache.

    ``forward`` returns ``(B, n_heads, Sq, Sk)`` token-level logits. Query reduction,
    chunk pooling and sink/local protection all live downstream in the press, so this
    module stays a pure scorer.
    """

    def __init__(self, config: GQAIndexerConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.rope_dim = config.rope_dim

        # Bias-free, like DSA and MiniMax M3. A bias on w_k would also be partly absorbed
        # by the LayerNorm that immediately follows it.
        self.w_q = nn.Linear(config.hidden_size, config.n_heads * config.head_dim, bias=False)
        self.w_k = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.q_norm = nn.LayerNorm(config.head_dim, eps=config.norm_eps)
        self.k_norm = nn.LayerNorm(config.head_dim, eps=config.norm_eps)

    def project_q(self, hidden_states: torch.Tensor, cos=None, sin=None) -> torch.Tensor:
        """Project and rotate indexer queries -> (B, n_heads, Sq, D)."""
        bsz, q_len, _ = hidden_states.shape
        q = self.w_q(hidden_states).view(bsz, q_len, self.n_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        if cos is not None:
            q = apply_rotary(q, cos, sin)
        return q

    def project_k(self, hidden_states: torch.Tensor, cos=None, sin=None) -> torch.Tensor:
        """
        Project and rotate the single shared indexer key -> (B, Sk, D).

        One key head (MQA) keeps the indexer's own KV cache at ``head_dim`` per token
        instead of ``n_heads * head_dim``; the heads differ on the query side only.
        """
        bsz, k_len, _ = hidden_states.shape
        k = self.w_k(hidden_states).view(bsz, k_len, 1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)  # (B, 1, Sk, D) so RoPE broadcasts over heads
        if cos is not None:
            k = apply_rotary(k, cos, sin)
        return k.squeeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        key_hidden_states: torch.Tensor | None = None,
        key_cos: torch.Tensor | None = None,
        key_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Score every (query, key) pair, once per indexer head.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Query-side hidden states, (B, Sq, hidden_size).
        cos, sin : torch.Tensor, optional
            Query RoPE tables, (B, Sq, rope_dim). ``None`` disables RoPE.
        mask : torch.Tensor, optional
            Additive mask broadcastable to (B, 1, Sq, Sk); ``0`` allowed, ``MASK_NEG``
            disallowed. Built by the press so causality and padding live in one place.
        key_hidden_states : torch.Tensor, optional
            Key-side hidden states, (B, Sk, hidden_size). Defaults to ``hidden_states``
            (self-scoring, the prefill case).
        key_cos, key_sin : torch.Tensor, optional
            Key RoPE tables, (B, Sk, rope_dim). Default to ``cos``/``sin``.

        Returns
        -------
        torch.Tensor
            Token-level logits, (B, n_heads, Sq, Sk), in fp32.
        """
        if hidden_states.dtype != self.w_q.weight.dtype:
            hidden_states = hidden_states.to(self.w_q.weight.dtype)
        if key_hidden_states is None:
            key_hidden_states, key_cos, key_sin = hidden_states, cos, sin
        elif key_hidden_states.dtype != self.w_k.weight.dtype:
            key_hidden_states = key_hidden_states.to(self.w_k.weight.dtype)

        q = self.project_q(hidden_states, cos, sin)  # (B, h, Sq, D)
        k = self.project_k(key_hidden_states, key_cos, key_sin)  # (B, Sk, D)

        # fp32 accumulation, as in the reference kernels. The single key head broadcasts
        # across the head axis.
        scores = torch.einsum("bhqd,bkd->bhqk", q.float(), k.float())

        if mask is not None:
            scores = scores + mask.to(scores.dtype)
        return scores


def build_indexer_mask(
    q_len: int,
    k_len: int,
    device: torch.device,
    *,
    attention_mask: torch.Tensor | None = None,
    query_offset: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build the additive (B, 1, q_len, k_len) indexer mask: causal + padding.

    Causality is mandatory. Without it a query scores keys that come after it, and those
    scores leak into whatever query reduction the press applies -- while the training
    target *is* causally masked, so student and teacher would disagree on a growing
    fraction of entries.

    ``attention_mask`` may be a 2D (B, k_len) keep-mask (1 = keep), a 4D additive float
    mask, or a 4D boolean keep-mask; all three are normalized here.

    ``query_offset`` is the absolute position of the first query (defaults to
    ``k_len - q_len``, correct for both prefill and a decode step appended at the end).
    """
    if query_offset is None:
        query_offset = k_len - q_len

    q_pos = torch.arange(q_len, device=device).unsqueeze(-1) + query_offset  # (q_len, 1)
    k_pos = torch.arange(k_len, device=device).unsqueeze(0)  # (1, k_len)
    causal = k_pos > q_pos  # True where the key is in the query's future
    mask = torch.zeros((1, 1, q_len, k_len), device=device, dtype=dtype)
    mask.masked_fill_(causal.view(1, 1, q_len, k_len), MASK_NEG)

    if attention_mask is None:
        return mask

    if attention_mask.dim() == 2:
        keep = attention_mask[:, -k_len:].to(device).bool().view(-1, 1, 1, k_len)
    elif attention_mask.dim() == 4:
        am = attention_mask[..., -k_len:].to(device)
        if am.dtype == torch.bool:
            keep = am
        else:
            # Additive float mask: finite/~0 means allowed. Comparing against a large
            # negative threshold covers both -inf and dtype-min conventions.
            keep = am > (torch.finfo(am.dtype).min / 2)
        if keep.shape[-2] not in (1, q_len):
            keep = keep[..., -q_len:, :]
    else:
        raise ValueError(f"attention_mask must be 2D or 4D, got {attention_mask.dim()}D")

    return mask.masked_fill(~keep, MASK_NEG)
