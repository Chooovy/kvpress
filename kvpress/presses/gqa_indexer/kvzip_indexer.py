# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fast-KVzip's gate as a scoring head: a token's strength against learnable reference keys.

A structural ablation of :class:`~.scalar_indexer.ScalarIndexer`. Both emit one score per
``(key, KV head)`` from that key's own hidden state -- so both are query-independent and both
travel the existing gate path unchanged -- but they read the hidden state differently:

* ``ScalarIndexer``: ``w_out . phi(W_in h)``, a plain MLP readout.
* here: ``h`` is projected to a per-token query AND a per-token key, and the score is that
  token's *self* dot product measured against a bank of ``sink`` learnable reference vectors.

Ported from :class:`~kvpress.presses.fastkvzip_press.FastKVzipGate`
(https://arxiv.org/abs/2601.17668). In the original it is distilled against KVzip scores on a
frozen model; here it is trained from the LM loss like every other arm, so the comparison is
about the *architecture* and not about the supervision.

What the softmax actually computes
----------------------------------
Both projections read the same token, so ``logit_j = k_j . q_j / sqrt(d) + b`` is **not** a
pairwise interaction -- it is one number per token per head. The published forward then does::

    score_j = 1 / (1 + sum_i exp(logit_base_i - logit_j))

which is the same thing as::

    score_j = exp(logit_j) / (exp(logit_j) + sum_i exp(logit_base_i))

i.e. ``logit_j``'s softmax share against ``sink`` learnable reference logits. ``k_base`` is
therefore an **adaptive threshold**: a token is important to the extent that it beats the bank.
That is the one real structural idea here, and it is what the ``mid_dim`` MLP has no analogue of.

Why the score is returned as ``log(score)``
-------------------------------------------
The published head emits ``(0, 1)``, but this package's gate consumes ``gate_scale * s - lse``,
where ``s`` sits alongside attention logits of std ~1. Feeding a probability in directly caps the
spread between the best and worst key at ``1``, so at ``gate_scale=1`` the router could express
less than one nat of preference across the whole cache and would have to grow ``gate_scale`` by
an order of magnitude before it could rank anything.

Taking the log fixes the range without touching the parameterization::

    s_j = log(score_j) = logit_j - logsumexp([logit_j, base_1..base_sink])

which is in ``(-inf, 0]``, keeps every gradient path through ``k_base`` intact, and is the same
form SP-KV's ``log u`` gate uses. It composes with the history normalizer rather than competing
with it: ``k_base`` normalizes each token against a *fixed bank* (a per-token, content-based
threshold), while ``lse`` normalizes *across keys* (a budget). The two are orthogonal.

Computed with ``logsumexp`` rather than as ``log(sigmoid-ish ratio)`` so a saturated token cannot
produce ``log(0) = -inf``; the published implementation has no epsilon because it never takes a
log.

What is deliberately kept from the scalar arm
---------------------------------------------
``decay`` (TrimKV's per-key lifetime) and ``pos_slope`` are inherited untouched, because they live
in :meth:`~.scalar_indexer.ScalarIndexer._score_and_decay`'s callers rather than in the scoring
head. Only the magnitude term is replaced. That keeps a run against the scalar arm at exactly one
variable -- the shape of the scoring head -- instead of three.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.gqa_indexer.indexer import MASK_NEG
from kvpress.presses.gqa_indexer.scalar_indexer import (
    ScalarIndexer,
    ScalarIndexerConfig,
    _inv_softplus,
)

#: Width of the per-token q/k projections, and the ``1/sqrt(d)`` softmax scale. Fast-KVzip's
#: published Qwen3 gates use 16.
DEFAULT_KVZIP_DIM = 16

#: Number of learnable reference keys the token is scored against. Named ``sink`` upstream after
#: the attention-sink rows it stands in for; unrelated to this package's ``n_sink`` pin.
DEFAULT_KVZIP_BASE = 16


@dataclass
class KVzipIndexerConfig(ScalarIndexerConfig):
    """
    Shape configuration for :class:`KVzipIndexer`.

    Inherits every field of :class:`~.scalar_indexer.ScalarIndexerConfig` so the decay head, the
    position tilt and the gate multiplier are configured identically across the two arms.
    ``mid_dim`` is inherited but **unused** -- this head has no MLP -- and is rejected if set, so
    a swept configuration cannot silently mean nothing.

    Attributes
    ----------
    kvzip_dim : int
        Width of the per-token ``q``/``k`` projections, and the ``1/sqrt(d)`` scale. 16 upstream.
    kvzip_base : int
        Size of the learnable reference bank the token's logit is scored against. 16 upstream.
        ``0`` removes the bank, which reduces the score to a bare per-token logit and throws away
        the only thing this architecture adds -- allowed as an ablation, not as a default.
    kvzip_ngroup : int
        Query groups per KV head. Upstream sets this to the model's ``n_q_heads / n_kv_heads``
        and averages the resulting scores, so one KV head's score pools ``ngroup`` opinions.
    """

    kvzip_dim: int = DEFAULT_KVZIP_DIM
    kvzip_base: int = DEFAULT_KVZIP_BASE
    kvzip_ngroup: int = 1

    def __post_init__(self):
        super().__post_init__()
        if self.mid_dim:
            raise ValueError(
                f"mid_dim={self.mid_dim} does not apply to the kvzip scorer, which replaces the "
                "MLP readout with a q/k self-interaction against a learnable key bank. Set "
                "kvzip_dim (the projection width) instead, and leave mid_dim at 0."
            )
        for name in ("kvzip_dim", "kvzip_ngroup"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.kvzip_base < 0:
            raise ValueError(f"kvzip_base must be non-negative, got {self.kvzip_base}")


class KVzipIndexer(ScalarIndexer):
    """
    Fast-KVzip's gate, emitting ``log(score)`` per ``(key, KV head)``.

    Subclasses :class:`~.scalar_indexer.ScalarIndexer` and overrides **only**
    :meth:`_score_and_decay`, the single trunk every entry point routes through
    (``score_keys``, ``score_at``, ``forward``, ``gate_key``). Everything downstream of the
    magnitude term -- the decay fold, the position tilt, the padding mask, the one-hot gate
    selector, the concat identity and the Triton kernel -- is inherited unchanged, which is what
    makes this a single-variable swap against the scalar arm.
    """

    def __init__(self, config: KVzipIndexerConfig):
        super().__init__(config)
        self.config = config
        self.kvzip_dim = config.kvzip_dim
        self.kvzip_base = config.kvzip_base
        self.ngroup = config.kvzip_ngroup

        # The MLP trunk the parent built is dead weight here: this head reads the normalized
        # hidden state directly. Dropped so the parameter count reports what actually trains.
        self.w_in = None
        self.mid_norm = None
        self.w_out = None

        n_out = self.n_heads * self.ngroup * config.kvzip_dim
        self.q_proj = nn.Linear(config.hidden_size, n_out, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.n_heads * config.kvzip_dim, bias=False)
        # Upstream normalizes q and k with the model's own RMSNorm before the dot product, which
        # is what keeps the logit at O(1) regardless of how the projections drift.
        # eps 1e-6, matching Qwen3RMSNorm (what the published gate uses) rather than this
        # package's norm_eps of 1e-5 -- at width 16 the two differ by ~1e-5 in the normalized
        # vector, which is far above fp32 noise and would make a port check fail for a reason
        # that has nothing to do with the architecture.
        self.q_norm = nn.RMSNorm(config.kvzip_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(config.kvzip_dim, eps=1e-6)
        self.b = nn.Parameter(torch.zeros(self.n_heads, 1, self.ngroup))
        if config.kvzip_base:
            self.k_base = nn.Parameter(torch.zeros(self.n_heads, config.kvzip_base, config.kvzip_dim))
        else:
            self.k_base = None
        self.inv_sqrt_d = 1.0 / math.sqrt(config.kvzip_dim)

        # The decay head reads the trunk's output in the parent; here there is no MLP, so it must
        # read the hidden state. Rebuilt at the right width, keeping the parent's zero-weight /
        # biased init so every key still starts at exactly `decay_init`.
        if config.decay:
            self.w_decay = nn.Linear(config.hidden_size, config.n_heads, bias=True)
            nn.init.zeros_(self.w_decay.weight)
            nn.init.constant_(self.w_decay.bias, _inv_softplus(-config.decay_init))

    @property
    def weight_dtype(self) -> torch.dtype:
        return self.k_proj.weight.dtype

    def _score_and_decay(
        self,
        hidden_states: torch.Tensor,
        *,
        key_offset: int = 0,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        ``(log_score, log_beta)``, both fp32 ``(B, h, Sk)`` -- the parent's contract exactly.

        ``log_score = logit - logsumexp([logit, bank])``, i.e. the log of the token's softmax
        share against the learnable reference bank. Everything after this point (the tilt, the
        decay fold, the mask) is the parent's code.
        """
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be (B, Sk, hidden_size), got {tuple(hidden_states.shape)}")
        if hidden_states.dtype != self.weight_dtype:
            hidden_states = hidden_states.to(self.weight_dtype)
        bsz, k_len, _ = hidden_states.shape

        x = self.in_norm(hidden_states)
        q = self.q_norm(self.q_proj(x).view(bsz, k_len, self.n_heads, self.ngroup, self.kvzip_dim))
        k = self.k_norm(self.k_proj(x).view(bsz, k_len, self.n_heads, 1, self.kvzip_dim))

        # Both sides read the SAME token, so this contracts the width axis only -- one logit per
        # (token, head, group). Not a pairwise interaction despite the q/k naming.
        logit = (q * k).sum(-1).float() * self.inv_sqrt_d  # (B, Sk, h, g)
        logit = logit + self.b.permute(1, 0, 2).float()  # b is (h, 1, g)

        if self.k_base is not None:
            # (B, Sk, h, g, base): the token's query against every reference key.
            base = torch.einsum("bshgd,hnd->bshgn", q.float(), self.k_base.float()) * self.inv_sqrt_d
            # log softmax-share of `logit` against {logit} u bank, computed with logsumexp so a
            # saturated token gives a large negative number rather than log(0) = -inf.
            joint = torch.cat([logit.unsqueeze(-1), base], dim=-1)
            log_score = logit - torch.logsumexp(joint, dim=-1)
        else:
            # Bank ablated: the bare logit, which is no longer bounded above by 0.
            log_score = logit

        # Upstream averages the group opinions to one score per KV head. Averaged in log space,
        # which is the geometric mean of the shares -- the arithmetic mean of the probabilities
        # would be dominated by whichever group saturates first.
        scores = log_score.mean(-1).permute(0, 2, 1)  # (B, h, Sk)

        if self.pos_slope:
            pos = torch.arange(key_offset, key_offset + k_len, device=scores.device, dtype=scores.dtype)
            scores = scores + self.pos_slope * pos

        log_beta = None
        if self.w_decay is not None:
            log_beta = -nn.functional.softplus(self.w_decay(x).float()).transpose(1, 2)

        if mask is not None:
            keep = mask if mask.dtype == torch.bool else mask != 0
            keep = ~keep.view(keep.shape[0], 1, -1)
            scores = scores.masked_fill(keep, MASK_NEG)
            if log_beta is not None:
                log_beta = log_beta.masked_fill(keep, 0.0)
        return scores, log_beta

    def extra_repr(self) -> str:
        shape = (
            f"hidden={self.config.hidden_size}, n_heads={self.n_heads}, "
            f"kvzip_dim={self.kvzip_dim}, base={self.kvzip_base}, ngroup={self.ngroup}"
        )
        shape += f", pos_slope={self.pos_slope:g}"
        if self.decay:
            shape += f", decay(ref={self.decay_ref:g}, init={self.config.decay_init:g})"
        return f"{shape}, Di={self.idx_dim}"
