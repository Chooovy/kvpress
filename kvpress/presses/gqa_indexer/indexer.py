# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The GQA lightning indexer module."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

# Sentinel used for "this (query, key) pair is not allowed". Kept finite so that a fully
# masked row still produces finite logits; ``masked_*_softmax`` in loss.py handles the
# all-invalid case explicitly via a boolean mask rather than relying on -inf arithmetic.
MASK_NEG = -1e4


def apply_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    """
    Activation applied to the raw q.k logits before group reduction.

    DeepSeek-V3.2 uses ReLU to keep per-head contributions non-negative, which matters
    because the contributions are summed. GLM5 and MiniMax M3 use no activation at all.
    ``none`` is the right choice when the group reduction cannot change sign (e.g. amax,
    or a non-negative weighting).
    """
    act = (activation or "relu").lower()
    if act in ("none", "identity", "linear"):
        return x
    if act == "relu":
        return F.relu(x)
    if act == "leaky_relu":
        return F.leaky_relu(x, negative_slope=0.01)
    if act == "softplus":
        return F.softplus(x)
    raise ValueError(f"Unknown activation {activation!r}; use relu, leaky_relu, softplus or none.")


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
    therefore has to be ``[f[0..r/2-1], f[0..r/2-1]]`` -- i.e. the first ``r/2`` entries of
    *each* half.

    A plain contiguous prefix ``cos[..., :rope_dim]`` is NOT equivalent: it yields
    ``[f[0..r-1]]``, which pairs channel ``j`` with frequency ``f[j]`` for the first half
    but ``f[j + r/2]`` for the second, rotating each pair by two different angles.
    Striding (``cos[..., ::2]``) is likewise wrong -- it samples every other frequency.
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
    Apply RoPE to ``x`` of shape (B, H, S, D) with ``cos``/``sin`` of shape (B, S, R).

    Only the leading ``R`` channels are rotated; the remaining ``D - R`` pass through
    untouched (NoPE). This mirrors both HF's ``apply_rotary_pos_emb`` and DSA, which
    rotates only ``qk_pos_emb_head_dim`` of ``index_head_dim``.

    ``cos``/``sin`` must already be narrowed to ``R`` via :func:`slice_rope_tables`.
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
    Shape and behaviour configuration for :class:`GQAIndexer`.

    Attributes
    ----------
    hidden_size : int
        Model hidden size; input dim of both projections.
    n_kv_heads : int
        Number of KV heads (``h``). The indexer emits one score per KV head, and
        ``w_k`` produces exactly this many key heads.
    group_size : int
        Query heads per KV head (``g``). ``w_q`` produces ``h * g`` heads, so each KV
        head is scored by ``g`` independent queries which are then reduced.
    head_dim : int
        Per-head dimension of the indexer's q/k.
    rope_dim : int
        Number of leading channels to rotate. ``0`` disables RoPE; must be even.
    activation : str
        Activation on the logits before group reduction.
    group_reduce : str
        How to collapse the ``g`` per-group query scores into one score per KV head:
        ``weights_proj`` (learned, DSA-style), ``sum``, ``mean``, or ``amax``.
    bias : bool
        Whether the projections carry a bias. DSA and MiniMax M3 both use bias-free
        projections, so this defaults to False.
    norm_eps : float
        Epsilon for the q/k LayerNorms.
    """

    hidden_size: int
    n_kv_heads: int
    group_size: int
    head_dim: int
    rope_dim: int = 0
    activation: str = "relu"
    group_reduce: str = "weights_proj"
    bias: bool = False
    norm_eps: float = 1e-5

    VALID_GROUP_REDUCE = ("weights_proj", "sum", "mean", "amax")

    def __post_init__(self):
        if self.group_reduce not in self.VALID_GROUP_REDUCE:
            raise ValueError(
                f"group_reduce must be one of {self.VALID_GROUP_REDUCE}, got {self.group_reduce!r}"
            )
        if self.rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even, got {self.rope_dim}")
        if self.rope_dim > self.head_dim:
            raise ValueError(f"rope_dim {self.rope_dim} cannot exceed head_dim {self.head_dim}")
        for name in ("hidden_size", "n_kv_heads", "group_size", "head_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")


class GQAIndexer(nn.Module):
    """
    Lightning indexer for grouped-query attention.

    Projects ``h * g`` query heads and ``h`` key heads from ``hidden_states``, scores
    every (query, key) pair within each KV group, and reduces the ``g`` scores of a group
    into a single per-KV-head token score.

    For Llama-3.1-8B (32 attention heads, 8 KV heads, head_dim 128) the natural setting
    is ``h=8, g=4``: ``w_q`` emits 32 heads, ``w_k`` emits 8, and the output carries one
    score per KV head -- matching the 8 physically independent KV caches.

    Shapes
    ------
    ``forward`` returns ``(B, h, Sq, Sk)`` token-level logits. Chunk-level aggregation,
    query reduction and sink/local protection all happen downstream in the press, so this
    module stays a pure scorer.
    """

    def __init__(self, config: GQAIndexerConfig):
        super().__init__()
        self.config = config
        self.n_kv_heads = config.n_kv_heads
        self.group_size = config.group_size
        self.head_dim = config.head_dim
        self.rope_dim = config.rope_dim
        self.activation = config.activation
        self.group_reduce = config.group_reduce

        n_q_heads = config.n_kv_heads * config.group_size
        self.n_q_heads = n_q_heads

        self.w_q = nn.Linear(config.hidden_size, n_q_heads * config.head_dim, bias=config.bias)
        self.w_k = nn.Linear(config.hidden_size, config.n_kv_heads * config.head_dim, bias=config.bias)
        self.q_norm = nn.LayerNorm(config.head_dim, eps=config.norm_eps)
        self.k_norm = nn.LayerNorm(config.head_dim, eps=config.norm_eps)

        # Learned per-(token, query-head) pooling weight, used only when
        # group_reduce == "weights_proj". Under the other reductions it would be dead
        # weight, so it is not allocated -- keeps state_dicts honest about what trains.
        if self.group_reduce == "weights_proj":
            self.weights_proj = nn.Linear(config.hidden_size, n_q_heads, bias=config.bias)
        else:
            self.weights_proj = None

        self.softmax_scale = config.head_dim**-0.5

    def project_q(self, hidden_states: torch.Tensor, cos=None, sin=None) -> torch.Tensor:
        """Project and rotate indexer queries -> (B, h, g, Sq, D)."""
        bsz, q_len, _ = hidden_states.shape
        q = self.w_q(hidden_states).view(bsz, q_len, self.n_q_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)  # (B, h*g, Sq, D)
        if cos is not None:
            q = apply_rotary(q, cos, sin)
        return q.view(bsz, self.n_kv_heads, self.group_size, q_len, self.head_dim)

    def project_k(self, hidden_states: torch.Tensor, cos=None, sin=None) -> torch.Tensor:
        """Project and rotate indexer keys -> (B, h, Sk, D)."""
        bsz, k_len, _ = hidden_states.shape
        k = self.w_k(hidden_states).view(bsz, k_len, self.n_kv_heads, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)  # (B, h, Sk, D)
        if cos is not None:
            k = apply_rotary(k, cos, sin)
        return k

    def group_weights(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Per-(token, query-head) pooling weights -> (B, h, g, Sq, 1), or None."""
        if self.weights_proj is None:
            return None
        bsz, q_len, _ = hidden_states.shape
        # fp32: these weights multiply summed per-head logits, and the reference
        # implementation (AngelPTM) explicitly forces fp32 here for numerical stability.
        w = self.weights_proj(hidden_states.float())
        w = w * (self.group_size**-0.5) * self.softmax_scale
        w = w.view(bsz, q_len, self.n_kv_heads, self.group_size)
        return w.permute(0, 2, 3, 1).unsqueeze(-1)  # (B, h, g, Sq, 1)

    def reduce_group(self, logits: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
        """Collapse (B, h, g, Sq, Sk) -> (B, h, Sq, Sk)."""
        if self.group_reduce == "amax":
            return logits.amax(dim=2)
        if self.group_reduce == "mean":
            return logits.mean(dim=2)
        if self.group_reduce == "sum":
            return logits.sum(dim=2)
        # weights_proj: learned, content-dependent pooling (DSA-style).
        return (logits * weights).sum(dim=2)

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
        Score every (query, key) pair, per KV head.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Query-side hidden states, (B, Sq, hidden_size).
        cos, sin : torch.Tensor, optional
            Query RoPE tables, (B, Sq, rope_dim). ``None`` disables RoPE.
        mask : torch.Tensor, optional
            Additive mask broadcastable to (B, 1, Sq, Sk); ``0`` allowed, ``MASK_NEG``
            disallowed. Built by the press so causality and padding are handled in one
            place.
        key_hidden_states : torch.Tensor, optional
            Key-side hidden states, (B, Sk, hidden_size). Defaults to ``hidden_states``
            (self-scoring, the prefill case).
        key_cos, key_sin : torch.Tensor, optional
            Key RoPE tables, (B, Sk, rope_dim). Default to ``cos``/``sin``.

        Returns
        -------
        torch.Tensor
            Token-level logits, (B, n_kv_heads, Sq, Sk), in fp32.
        """
        if hidden_states.dtype != self.w_q.weight.dtype:
            hidden_states = hidden_states.to(self.w_q.weight.dtype)
        if key_hidden_states is None:
            key_hidden_states, key_cos, key_sin = hidden_states, cos, sin
        elif key_hidden_states.dtype != self.w_k.weight.dtype:
            key_hidden_states = key_hidden_states.to(self.w_k.weight.dtype)

        q = self.project_q(hidden_states, cos, sin)  # (B, h, g, Sq, D)
        k = self.project_k(key_hidden_states, key_cos, key_sin)  # (B, h, Sk, D)

        # fp32 accumulation: the reference kernels upcast before the einsum because the
        # activation + group reduction sums many terms per (query, key) pair.
        logits = torch.einsum("bhgqd,bhkd->bhgqk", q.float(), k.float())
        logits = apply_activation(logits, self.activation)

        weights = self.group_weights(hidden_states)
        scores = self.reduce_group(logits, weights)  # (B, h, Sq, Sk)

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

    Causality is mandatory here. Without it a query scores keys that come after it, and
    those scores then leak into whatever query reduction the press applies -- while the
    training target is causally masked, so student and teacher would disagree on a
    growing fraction of entries.

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
        # (B, k_len) keep-mask -> additive
        keep = attention_mask[:, -k_len:].to(device).bool().view(-1, 1, 1, k_len)
    elif attention_mask.dim() == 4:
        am = attention_mask[..., -k_len:].to(device)
        if am.dtype == torch.bool:
            keep = am
        else:
            # Additive float mask: finite/≈0 means allowed. Comparing against a large
            # negative threshold covers both -inf and dtype-min conventions.
            keep = am > (torch.finfo(am.dtype).min / 2)
        if keep.shape[-2] not in (1, q_len):
            keep = keep[..., -q_len:, :]
    else:
        raise ValueError(f"attention_mask must be 2D or 4D, got {attention_mask.dim()}D")

    return mask.masked_fill(~keep, MASK_NEG)
