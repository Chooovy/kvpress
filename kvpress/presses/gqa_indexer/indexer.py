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


class IndexerNorm(nn.Module):
    """
    LayerNorm whose statistics are always computed in fp32, then cast back.

    ``nn.LayerNorm`` on a bf16 module reduces in bf16, and the mean/variance over ``head_dim``
    channels is a long accumulation with only 8 significant bits: measured relative error
    against an fp32 reduction is ~7e-2 median, 1e-1 worst case over 200 draws. That lands
    directly on ``q``/``k``, and the ``head_dim``-long dot product that follows amplifies it,
    so the score -- the quantity every reference implementation is careful to keep in fp32 --
    would inherit the error before its own GEMM even starts.

    This is not a novel precaution. MiniMax M3's indexer norm is documented as "Gemma-style
    RMSNorm: normalizes in fp32", and Megatron's DSA exposes it as a config flag
    (``dsa_indexer_k_norm_fp32``, which does ``self.k_norm(k.float()).to(dtype=k_dtype)``). M3
    builds it in unconditionally rather than making it optional, which is the choice taken
    here: the cost is one cast on a ``head_dim``-wide tensor, and there is no regime where
    reducing in bf16 is preferable.

    Parameter shapes and names match ``nn.LayerNorm``, so existing checkpoints load unchanged.

    Memory
    ------
    The naive form -- ``layer_norm(x.to(fp32), ...)`` -- makes autograd save the **upcast**
    tensor, so it retains an fp32 copy of the input rather than referring to the bf16 one the
    caller already holds. On the pairwise indexer that is invisible: the norm sits after
    ``w_k``, so the tensor is ``head_dim``-wide. On :class:`~.scalar_indexer.ScalarIndexer` it
    is the *first* op and therefore runs at full ``hidden_size``, where the extra fp32 copy is
    256 MiB per layer at ``L=16384`` -- 9 GiB across 36 layers, which is what made the scalar
    arm OOM at a length the pairwise arm trains at comfortably.

    So the statistics are taken from an fp32 view without handing that view to autograd:
    :class:`_Fp32LayerNorm` saves the original ``x`` and re-derives what it needs in the
    backward. Identical arithmetic in the forward, one wide fp32 tensor instead of two.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast only for low-precision inputs: an fp64 caller asked for fp64, and .float()
        # would silently narrow it -- the same trap accumulation_dtype exists to avoid.
        if x.dtype.itemsize >= 4:
            return nn.functional.layer_norm(
                x, (x.shape[-1],), self.weight.to(x.dtype), self.bias.to(x.dtype), self.eps
            )
        return _Fp32LayerNorm.apply(x, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class _Fp32LayerNorm(torch.autograd.Function):
    """
    LayerNorm with fp32 statistics that saves the *input's own* dtype, not the upcast copy.

    Equivalent to ``layer_norm(x.float(), ...).to(x.dtype)`` in the forward, to the bit. The
    difference is only what survives into the backward: ``x`` as the caller passed it, plus the
    two ``(…, 1)`` statistics, rather than a full-width fp32 tensor. See :class:`IndexerNorm`
    for why that matters at ``hidden_size`` width.

    The backward is the standard LayerNorm gradient, recomputed in fp32 from ``x`` and the saved
    mean/rstd -- the recompute is one pass over ``x`` and replaces a tensor twice its size.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        x32 = x.float()
        mean = x32.mean(-1, keepdim=True)
        var = x32.var(-1, unbiased=False, keepdim=True)
        rstd = torch.rsqrt(var + eps)
        x_hat = (x32 - mean) * rstd
        out = x_hat * weight.float() + bias.float()
        # x in its ORIGINAL dtype; mean/rstd are (..., 1) so they are free at any width.
        ctx.save_for_backward(x, weight, mean, rstd)
        return out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, mean, rstd = ctx.saved_tensors
        g = grad_out.float()
        x_hat = (x.float() - mean) * rstd

        grad_weight = grad_bias = None
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            reduce_dims = tuple(range(g.dim() - 1))
            if ctx.needs_input_grad[1]:
                grad_weight = (g * x_hat).sum(reduce_dims).to(weight.dtype)
            if ctx.needs_input_grad[2]:
                grad_bias = g.sum(reduce_dims).to(weight.dtype)

        grad_x = None
        if ctx.needs_input_grad[0]:
            n = x.shape[-1]
            gw = g * weight.float()
            grad_x = rstd / n * (
                n * gw - gw.sum(-1, keepdim=True) - x_hat * (gw * x_hat).sum(-1, keepdim=True)
            )
            grad_x = grad_x.to(x.dtype)
        return grad_x, grad_weight, grad_bias, None


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
    gate_scale : bool
        Create the learnable ``gate_scale`` scalar used by end-to-end (gated-attention)
        training. Left ``False`` for distillation, which never reads it, so a
        distillation-trained checkpoint carries no extra parameter. See
        :attr:`GQAIndexer.gate_scale`.
    """

    hidden_size: int
    n_heads: int
    head_dim: int
    rope_dim: int = 0
    norm_eps: float = 1e-5
    gate_scale: bool = False

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

    #: Natural magnitude for the score when it is used as an additive attention gate.
    #:
    #: ``IndexerNorm`` leaves q and k at unit variance per channel, so the ``head_dim``-long
    #: dot product has standard deviation ``~sqrt(head_dim)``: measured 11.4 against 1.0 for a
    #: real ``q @ k / sqrt(head_dim)`` attention logit at ``head_dim=128``. Added raw, the gate
    #: would be an 11x louder term than the attention it is supposed to modulate. Dividing by
    #: ``sqrt(head_dim)`` -- the same correction, and for the same reason, as attention's own
    #: scaling -- brings it to std 1.0.
    #:
    #: Note this is exactly the attention softmax scale when the indexer's ``head_dim`` equals
    #: the model's, which is the default: the concatenated form in
    #: :mod:`~kvpress.presses.gqa_indexer.gated_attention` then needs no rescale at all.
    GATE_SCALE_INIT = staticmethod(lambda head_dim: head_dim**-0.5)

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
        # fp32 statistics regardless of the module dtype -- see IndexerNorm.
        self.q_norm = IndexerNorm(config.head_dim, eps=config.norm_eps)
        self.k_norm = IndexerNorm(config.head_dim, eps=config.norm_eps)

        # Multiplier on the score when it acts as an additive attention gate. One scalar per
        # layer, so each layer learns how much to lean on its own router -- and the trained
        # value is a readout on whether the router earns its place at all.
        #
        # Deliberately NOT initialized to 0. A zero gate would start end-to-end training from
        # exactly the frozen dense model, which is tempting, but `dL/dscore` is proportional to
        # this scalar: at 0 the router receives no gradient and the run never leaves that point.
        # Starting at the natural scale means accepting a std-1 perturbation of the attention
        # logits at step 0, which is the price of having a gradient at all.
        self.gate_scale = (
            nn.Parameter(torch.tensor(self.GATE_SCALE_INIT(config.head_dim)))
            if config.gate_scale
            else None
        )

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

    def require_gate_scale(self) -> torch.Tensor:
        """
        The gate multiplier, raising when this indexer was not built with one.

        Raises rather than defaulting to the init constant: an indexer without the parameter
        is one that will not *learn* the gate strength, and end-to-end training would then
        report a healthy loss while silently running the fixed-scale ablation instead of the
        configured one.
        """
        if self.gate_scale is None:
            raise RuntimeError(
                "this indexer has no gate_scale parameter, so it cannot be used as an "
                "attention gate. Build it with GQAIndexerConfig(gate_scale=True) -- or, from "
                "the press, GQAIndexerPress(gate_scale=True)."
            )
        return self.gate_scale

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

        # Accumulate in at least fp32, as the reference kernels do -- but never below the
        # input precision, so a float64 caller keeps it. The single key head broadcasts
        # across the head axis.
        acc = torch.float32 if q.dtype.itemsize < 4 else q.dtype
        scores = torch.einsum("bhqd,bkd->bhqk", q.to(acc), k.to(acc))

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
