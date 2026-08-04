# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from dataclasses import dataclass, field

import os
import torch
from torch import nn

from kvpress.presses.base_press import BasePress
import math
logger = logging.getLogger(__name__)


@dataclass
class ScorerPress(BasePress):
    """
    Base class for score-based KV cache compression methods.

    This class assigns scores to key-value pairs and prune those with the lowest scores.
    Subclasses then implement the `score` method to define how importance is calculated.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    """

    compression_ratio: float = 0.0
    mean_head: bool = False
    layer_running_mean: bool = False
    running_mean_start_layer: int = 0

    # Internal state (reset at the beginning of each forward when layer_idx == 0)
    _running_layer_score_sum: torch.Tensor | None = field(default=None, init=False, repr=False)

    # ---- entropy-gated running mean options ----
    entropy_gate: bool = False
    entropy_gate_level: str = "head"   # "head" | "layer" | "head_layer"
    entropy_skip: str = "low"          # "low"=skip low-entropy (尖峰/专门化); "high"=skip high-entropy (分散)
    entropy_threshold_mode: str = "mean"   # "mean" | "fixed" | "quantile"
    entropy_threshold: float = 0.5     # used when mode="fixed" (normalized entropy in [0,1])
    entropy_quantile: float = 0.5      # used when mode="quantile"
    entropy_eps: float = 1e-12
    entropy_debug: bool = False
    # ---- entropy normalization variants ----
    entropy_shift_mode: str = "always_min"     # "always_min" | "neg_only"
    entropy_prob_mode: str = "l1"             # "l1" | "softmax"
    entropy_softmax_tau: float = 1.0          # only used when prob_mode="softmax"

    # Internal states
    # --- new knobs ---
    layer_running_alpha: float = 1.0          # α=1 -> 纯 mean（保持旧行为）
    layer_running_use_max: bool = False       # True -> 启用 mean/max 混合
    layer_running_skip_spiky: bool = False    # True -> 尖峰不进 mean，只进 max
    layer_spiky_scope: str = "head"           # "head" 或 "layer"
    layer_spiky_metric: str = "entropy"       # "entropy"（推荐）或 "peak"
    layer_spiky_entropy_thresh: float = 0.4   # 归一化熵阈值（0~1）；低于它算“尖”
    layer_spiky_peak_thresh: float = 0.0      # peak 阈值（如果用 peak，自己调）
    meanmax_gate: bool = False
    meanmax_gate_keep_high: bool = False
    # --- new internal states ---
    _running_layer_score_max: torch.Tensor | None = field(default=None, init=False, repr=False)
    _running_layer_score_cnt: torch.Tensor | None = field(default=None, init=False, repr=False)  # 用于“跳过尖峰”的计数

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"
        assert self.running_mean_start_layer >= 0, "running_mean_start_layer must be >= 0"

    def dump_layer_scores(self, module: nn.Module, kwargs: dict, scores: torch.Tensor) -> None:
        """
        Dump per-layer per-head per-token scores to disk for analysis.

        Enable by setting env var KVPRESS_SCORE_DUMP_PATH to a directory path.
        Each call writes one file: layer{layer_idx}_step{n}_pos{cache_pos}.pt

        Saved object is a dict with keys:
        - layer_idx: int
        - cache_pos: Optional[int]
        - scores: torch.Tensor with shape (B, num_kv_heads, seq_len)
        """
        out_dir = os.environ.get("KVPRESS_SCORE_DUMP_PATH", "")
        if not out_dir:
            return

        os.makedirs(out_dir, exist_ok=True)

        layer_idx = int(getattr(module, "layer_idx", -1))
        cache_pos = None
        try:
            cache_position = kwargs.get("cache_position", None)
            if cache_position is not None and len(cache_position) > 0:
                cache_pos = int(cache_position[-1])
        except Exception:
            cache_pos = None

        step = int(getattr(self, "_score_dump_step", 0))
        setattr(self, "_score_dump_step", step + 1)

        filename = f"layer{layer_idx}_step{step}_pos{cache_pos if cache_pos is not None else 'na'}.pt"
        path = os.path.join(out_dir, filename)

        payload = {
            "layer_idx": layer_idx,
            "cache_pos": cache_pos,
            "scores": scores.detach().cpu(),
        }
        torch.save(payload, path)

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        """
        Compute importance scores for each key-value pair.

        This method must be implemented by subclasses to define how the importance
        of each token position is calculated. Higher scores indicate more important
        tokens that should be kept during compression.

        Parameters
        ----------
        module : nn.Module
            The transformer attention layer where scoring is applied.
        hidden_states : torch.Tensor
            Input embeddings with shape (batch_size, seq_len, hidden_dim).
        keys : torch.Tensor
            Key tensors with shape (batch_size, num_kv_heads, seq_len, head_dim).
        values : torch.Tensor
            Value tensors with shape (batch_size, num_kv_heads, seq_len, head_dim).
        attentions : torch.Tensor
            Attention weights with shape (batch_size, num_heads, seq_len, seq_len).
            May be None if not computed or needed by the scoring method.
        kwargs : dict
            Additional arguments from the forward pass, including cache and position info.

        Returns
        -------
        torch.Tensor
            Importance scores with shape (batch_size, num_kv_heads, seq_len).
            Higher scores indicate more important tokens. The tokens with the
            lowest scores will be pruned during compression.
        """
        raise NotImplementedError
    def _peakiness(self, scores: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """
        scores: (B, H, L)
        return: peak (B, H) where peak = max/mean over token dimension
        """
        x = scores.detach().float()
        x = x - x.min(dim=-1, keepdim=True).values   # shift to >=0
        x = x.clamp_min(0.0) + eps
        peak = x.max(dim=-1).values / x.mean(dim=-1).clamp_min(eps)
        return peak

    def _normalized_entropy(self, scores: torch.Tensor) -> torch.Tensor:
        """
        scores:  (B, H, L) 任意实数都行；内部会转成分布
        return: (B, H) normalized entropy in [0,1] (approx)
        """
        x = scores.detach().float()

        # ---------- Branch (1): shift mode ----------
        m = x.min(dim=-1, keepdim=True).values  # (B,H,1)
        if self.entropy_shift_mode == "always_min":
            x = x - m
        elif self.entropy_shift_mode == "neg_only":
            # only shift when min < 0, otherwise keep x unchanged
            x = torch.where(m < 0, x - m, x)
        else:
            raise ValueError(f"Unknown entropy_shift_mode={self.entropy_shift_mode}")

        # ---------- Branch (2): probability mode ----------
        if self.entropy_prob_mode == "l1":
            x = x.clamp_min(0.0) + self.entropy_eps
            p = x / x.sum(dim=-1, keepdim=True).clamp_min(self.entropy_eps)

        elif self.entropy_prob_mode == "softmax":
            tau = float(self.entropy_softmax_tau)
            tau = max(tau, float(self.entropy_eps))

            z = x / tau
            # stable softmax: subtract max (output is unchanged due to translation invariance)
            z = z - z.max(dim=-1, keepdim=True).values
            p = torch.softmax(z, dim=-1)

            # avoid log(0) in entropy
            p = p.clamp_min(self.entropy_eps)
            p = p / p.sum(dim=-1, keepdim=True).clamp_min(self.entropy_eps)

        else:
            raise ValueError(f"Unknown entropy_prob_mode={self.entropy_prob_mode}")

        H = -(p * p.log()).sum(dim=-1)  # (B,H)
        L = p.size(-1)
        return H / max(math.log(L), self.entropy_eps)

    def _entropy_include_mask(self, ent: torch.Tensor) -> torch.Tensor:
        """
        ent: (B, H) or (B, 1)
        returns include_mask (B, H) / (B,1): True means "参与 running-mean 的累加与平均"
        """
        if self.entropy_threshold_mode == "mean":
            thr = ent.mean(dim=1, keepdim=True)
        elif self.entropy_threshold_mode == "fixed":
            thr = torch.full((ent.size(0), 1), float(self.entropy_threshold), device=ent.device, dtype=ent.dtype)
        elif self.entropy_threshold_mode == "quantile":
            thr = torch.quantile(ent, float(self.entropy_quantile), dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown entropy_threshold_mode={self.entropy_threshold_mode}")

        # include 的逻辑：skip 哪一侧，就 include 另一侧
        if self.entropy_skip == "low":
            # skip 低熵 => include 高熵
            include = ent >= thr
        elif self.entropy_skip == "high":
            # skip 高熵 => include 低熵
            include = ent <= thr
        else:
            raise ValueError(f"Unknown entropy_skip={self.entropy_skip}")

        return include

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.compression_ratio == 0 and not os.environ.get("KVPRESS_SCORE_DUMP_PATH"):
            return keys, values

        # Compute scores
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        if self.compression_ratio == 0:
            if os.environ.get("KVPRESS_SCORE_DUMP_PATH"):
                self.dump_layer_scores(module, kwargs, scores)
            return keys, values

        if self.layer_running_mean:
            layer_idx = int(getattr(module, "layer_idx", 0))

            # reset per-forward state at beginning
            if layer_idx == 0:
                self._running_layer_score_sum = None
                self._running_layer_score_cnt = None  # ★ 新增：gated RM 必须要 cnt

            if layer_idx >= self.running_mean_start_layer:
                # ---------- entropy-gated running mean ----------
                if getattr(self, "entropy_gate", False):

                    # init buffers
                    if (self._running_layer_score_sum is None
                        or self._running_layer_score_sum.shape != scores.shape):
                        self._running_layer_score_sum = torch.zeros_like(scores).detach()
                        self._running_layer_score_cnt = torch.zeros(
                            (scores.size(0), scores.size(1), 1),
                            device=scores.device,
                            dtype=scores.dtype,
                        ).detach()

                    # decide include mask
                    ent_h = self._normalized_entropy(scores)  # (B,H)

                    include_h = torch.ones_like(ent_h, dtype=torch.bool)
                    include_l = torch.ones((scores.size(0), 1), device=scores.device, dtype=torch.bool)

                    if self.entropy_gate_level in ("head", "head_layer"):
                        include_h = self._entropy_include_mask(ent_h)  # (B,H)

                    if self.entropy_gate_level in ("layer", "head_layer"):
                        ent_l = ent_h.mean(dim=1, keepdim=True)        # (B,1)
                        include_l = self._entropy_include_mask(ent_l)  # (B,1)

                    include = include_h & include_l  # (B,H)
                    w = include.to(scores.dtype).unsqueeze(-1)  # (B,H,1)

                    # update running buffers only for included
                    self._running_layer_score_sum = (self._running_layer_score_sum + scores.detach() * w).detach()
                    self._running_layer_score_cnt = (self._running_layer_score_cnt + w).detach()

                    # compute mean for included heads
                    mean_scores = self._running_layer_score_sum / self._running_layer_score_cnt.clamp_min(1.0)

                    # IMPORTANT: excluded heads keep raw (no-mean)
                    scores = torch.where(include.unsqueeze(-1), mean_scores, scores)

                    if getattr(self, "entropy_debug", False):
                        keep_frac = include.float().mean().item()
                        print(f"[entropy-gated RM] layer={layer_idx} keep_frac={keep_frac:.3f} "
                            f"ent_mean={ent_h.mean().item():.3f} ent_std={ent_h.std().item():.3f}")
                # ---------- peak-based mean/max gate ----------
                elif getattr(self, "meanmax_gate", False):
                    # reset per-forward state
                    if layer_idx == 0:
                        self._running_layer_score_sum = None
                        self._running_layer_score_max = None

                    # init buffers
                    if (self._running_layer_score_sum is None
                        or self._running_layer_score_sum.shape != scores.shape):
                        self._running_layer_score_sum = torch.zeros_like(scores).detach()
                        self._running_layer_score_max = scores.detach()
                    else:
                        self._running_layer_score_max = torch.maximum(self._running_layer_score_max, scores.detach())

                    # update running sum
                    self._running_layer_score_sum = (self._running_layer_score_sum + scores.detach()).detach()

                    denom = float(layer_idx - self.running_mean_start_layer + 1)
                    mean_scores = self._running_layer_score_sum / denom
                    max_scores = self._running_layer_score_max

                    # compute peak per head
                    peak = self._peakiness(scores)  # (B,H)

                    # threshold: if layer_spiky_peak_thresh <= 0, use per-sample mean(peak) as threshold
                    if getattr(self, "layer_spiky_peak_thresh", 0.0) and self.layer_spiky_peak_thresh > 0:
                        thr = torch.full((peak.size(0), 1), float(self.layer_spiky_peak_thresh),
                                        device=peak.device, dtype=peak.dtype)
                    else:
                        thr = peak.mean(dim=1, keepdim=True)  # (B,1)

                    spiky = peak > thr  # (B,H)

                    # alpha: flat heads use 1.0 (pure mean), spiky heads use layer_running_alpha (mix toward max)
                    alpha_spiky = float(getattr(self, "layer_running_alpha", 0.3))
                    if getattr(self, "meanmax_gate_keep_high", False):
                        # keep-high version:
                        # peak 大(spiky) -> alpha=1.0 (更信 mean)
                        # peak 小(flat)  -> alpha=alpha_spiky (更信 max)
                        alpha = torch.full_like(peak, alpha_spiky, dtype=mean_scores.dtype)
                        alpha = torch.where(spiky, torch.ones_like(alpha), alpha)
                    else:
                        # default version:
                        # peak 大(spiky) -> alpha=alpha_spiky (更信 max)
                        # peak 小(flat)  -> alpha=1.0 (更信 mean)
                        alpha = torch.ones_like(peak, dtype=mean_scores.dtype)
                        alpha = torch.where(spiky, torch.full_like(alpha, alpha_spiky), alpha)  # (B,H)         
                    alpha = alpha.unsqueeze(-1)  # (B,H,1)

                    scores = alpha * mean_scores + (1.0 - alpha) * max_scores

                # ---------- ORIGINAL running mean (unchanged) ----------
                else:
                    if self._running_layer_score_sum is None or self._running_layer_score_sum.shape != scores.shape:
                        self._running_layer_score_sum = scores.detach()
                    else:
                        self._running_layer_score_sum = (self._running_layer_score_sum + scores.detach()).detach()

                    denom = float(layer_idx - self.running_mean_start_layer + 1)
                    scores = self._running_layer_score_sum / denom

        selection_scores = scores
        if self.mean_head:
            selection_scores = scores.mean(dim=1, keepdim=True)

        # Get indices of KV pairs with the lowest scores
        k_len = keys.shape[2]
        n_kept = max(1, int(k_len * (1 - self.compression_ratio)))
        indices = selection_scores.topk(n_kept, dim=-1).indices  # (B, H or 1, n_kept)
        if self.mean_head:
            indices = indices.expand(-1, keys.shape[1], -1)  # (B, num_kv_heads, n_kept)
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        # Prune keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values
