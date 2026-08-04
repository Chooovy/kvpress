# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class GTScorePress(ScorerPress):
    """
    Ground-truth score press based on *actual* attention weights.

    It assumes `attentions` is provided to `score()` with shape:
        (bsz, num_heads, q_len, k_len)

    It reduces query dimension -> per-head key importance, then maps
    num_heads -> num_kv_heads (for GQA) to produce:
        (bsz, num_kv_heads, k_len)

    Notes:
    - You MUST run the model with output_attentions=True.
    - Many backends (flash/sdpa) may not return attentions; eager is usually required.
    """

    # keep these tokens always (set to max score)
    n_sink: int = 4

    # optionally keep the most recent tokens always (like SnapKV window behavior)
    keep_last: int = 0

    # reduce queries to key-importance
    # - "mean": average over queries
    # - "causal_mean": average over *valid* queries per key under standard causal mask
    # - "max": max over queries
    # - "last": only the last query row
    # - "last_n": average over last_n_query queries
    query_reduce: str = "mean"
    last_n_query: int = 128

    # reduce heads
    # - "kv_group_mean": reshape (num_kv_heads, num_groups) and mean over groups
    # - "all_head_mean": mean over all heads then broadcast to kv heads
    head_reduce: str = "kv_group_mean"

    # optional smoothing along key axis
    smooth_kernel: int = 0  # 0 disables; >0 uses avg_pool1d with this kernel size

    # optionally weight by value norm (like some other presses)
    use_vnorm: bool = False
    #获取并校验注意力权重
    def _require_attn(self, attentions: Optional[Union[torch.Tensor, list, tuple]]) -> torch.Tensor:
        if attentions is None:
            raise ValueError(
                "GTScorePress needs `attentions` but got None. "
                "Run with output_attentions=True, and usually attn_implementation='eager' "
                "(flash/sdpa often returns None)."
            )
        attn = attentions[0] if isinstance(attentions, (tuple, list)) else attentions
        if not isinstance(attn, torch.Tensor) or attn.dim() != 4:
            raise ValueError(f"Unexpected attentions type/shape: {type(attn)} {getattr(attn,'shape',None)}")
        return attn
    #将 Query 维度 Q 聚合掉，得到形状 (B, H, K)
    def _reduce_queries(self, attn: torch.Tensor) -> torch.Tensor:
        # attn: (B, H, Q, K) -> (B, H, K)
        mode = (self.query_reduce or "mean").lower()
        if mode == "last":
            return attn[:, :, -1, :]
        if mode == "last_n":
            n = max(1, int(self.last_n_query))
            return attn[:, :, -n:, :].mean(dim=-2)
        if mode in ("causal_mean", "mean_causal"):
            # For standard causal attention in prefill, Q == K and key position j
            # is only visible to the last (Q-j) queries. A naive mean over Q
            # will down-weight late keys due to masked zeros. Correct by
            # averaging over the number of valid queries per key.
            q_len = attn.size(-2)
            k_len = attn.size(-1)
            if q_len == k_len:
                counts = torch.arange(k_len, 0, -1, device=attn.device, dtype=attn.dtype).view(1, 1, k_len)
                return attn.sum(dim=-2) / counts.clamp_min(1)
            # Fallback: if Q != K (unusual here), default to naive mean
            return attn.mean(dim=-2)
        if mode == "max":
            return attn.max(dim=-2).values
        # default mean
        return attn.mean(dim=-2)
    #Query Heads 的数量 (H) 多于 KV Heads 的数量 (H_{kv})。KV Cache 是按 H_{kv} 存储的，因此必须将分数从 H 映射回 H_{kv}。
    def _reduce_heads_to_kv(self, module: nn.Module, per_head_key: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
        # per_head_key: (B, num_heads, K)
        bsz, num_heads, k_len = per_head_key.shape
        mode = (self.head_reduce or "kv_group_mean").lower()

        if mode in ("all_head_mean", "all-head-mean"):
            x = per_head_key.mean(dim=1, keepdim=True)  # (B,1,K)
            return x.expand(bsz, num_kv_heads, k_len)

        # default: kv_group_mean (for GQA)
        # num_heads should be divisible by num_kv_heads
        if num_heads % num_kv_heads != 0:
            # fallback: just mean over heads then broadcast
            x = per_head_key.mean(dim=1, keepdim=True)
            return x.expand(bsz, num_kv_heads, k_len)

        num_groups = num_heads // num_kv_heads
        x = per_head_key.view(bsz, num_kv_heads, num_groups, k_len).mean(dim=2)  # (B, kv, K)
        return x

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: Optional[torch.Tensor],
        kwargs,
    ) -> torch.Tensor:
        # keys: (B, kv, K, D)
        bsz, num_kv_heads, k_len, _ = keys.shape

        attn = self._require_attn(attentions)  # (B, H, Q, K_attn)
        # align key length if needed
        if attn.size(-1) != k_len:
            attn = attn[..., -k_len:]

        per_head_key = self._reduce_queries(attn)  # (B, H, K)

        # optional smoothing along key axis
        if self.smooth_kernel and self.smooth_kernel > 1:
            per_head_key = F.avg_pool1d(
                per_head_key,
                kernel_size=self.smooth_kernel,
                stride=1,
                padding=self.smooth_kernel // 2,
            )

        scores = self._reduce_heads_to_kv(module, per_head_key, num_kv_heads)  # (B, kv, K)

        # optional value-norm reweight
        if self.use_vnorm:
            vnorm = values.norm(dim=-1)  # (B, kv, K)
            scores = (scores + 1e-6) * vnorm

        # enforce sink tokens always kept
        if self.n_sink > 0 and k_len > 0:
            sink = min(self.n_sink, k_len)
            maxv = scores.max().detach()
            scores[:, :, :sink] = maxv

        # enforce keep_last tokens always kept (like "recent window")
        if self.keep_last > 0 and k_len > 0:
            keep = min(self.keep_last, k_len)
            maxv = scores.max().detach()
            scores[:, :, -keep:] = maxv

        return scores
