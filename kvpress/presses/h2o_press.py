# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import torch
import torch.nn.functional as F
import math
from torch import nn

from kvpress.presses.scorer_press import ScorerPress

def compute_attention_scores(query_states: torch.Tensor, key_states: torch.Tensor) -> torch.Tensor:
    """
    简单版的 attention score 计算：
    - query_states: [B, H, T_q, D]
    - key_states  : [B, H, T_k, D]

    返回:
    - attn_scores : [B, H, T_q, T_k]
    """
    # [B, H, T_q, D] @ [B, H, D, T_k] -> [B, H, T_q, T_k]
    attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1))
    dim = query_states.size(-1)
    attn_scores = attn_scores / math.sqrt(dim)
    return attn_scores

@dataclass
class H2OPress(ScorerPress):
    """
    Decode-only 版本的 H2O。

    现在支持：
    - budget:      保留 token 总数（≈ target_size）
    - window_size: 最近不压缩的窗口（≈ hidden_states_buffer_size）
    - compression_interval: 每多少个 decode step 执行一次压缩
        * =1  时：每步压一次，对齐原始 H2O 行为
        * =128 时：每 128 步压一次，对齐你 decode-press 的设置
    """

    compression_ratio: float = 0.0  # 只是为了兼容基类，不参与 budget 计算
    budget: int = 128
    window_size: int = 1
    # ★ 新增：压缩间隔（decode step 粒度）
    compression_interval: int = 1
    record_kept_token_indices: bool = False

    # 用来记录 decode 过程中累计的 kept indices（和你原来 H2O 的逻辑一致）
    evicted_token_num: int = 0
    kept_token_indices: List[torch.Tensor] = field(default_factory=list, repr=False)
    # ★ 新增：当前 context 内的 decode 步数计数器
    decode_step: int = 0
    def __post_init__(self):
        super().__post_init__()
        assert self.budget > self.window_size, "budget must be greater than window_size"
        assert self.compression_interval >= 1, "compression_interval must be >= 1"

    def reset_global_scores(self):
        """
        kvpress 在每个 context 开头会调用这个（你已经在 EvaluationRunner 里处理了），
        这里我们顺手把 H2O 自己的状态也清掉。
        """
        # ScorerPress 自己的状态（sample_idx 等）
        super().reset_global_scores()

        # H2O 自己的记录
        self.evicted_token_num = 0
        self.kept_token_indices.clear()
        self.decode_step = 0  # ★ 重置 decode 步数

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: Dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # keys, values: [B, H_kv, T_kv, D]
        bsz, num_kv_heads, kv_cache_len, head_dim = keys.shape

        # hidden_states: [B, T_q, hidden_dim]
        # 通常：
        #   - prefill：T_q > 1，且 kv_cache_len == T_q
        #   - decode：T_q == 1，且 kv_cache_len >= T_q + 已生成长度
        T_q = hidden_states.shape[1]

        # ==== 这里是我们自己的“prefill 判定” ====
        is_prefill = (T_q > 1) or (kv_cache_len == T_q)
        if is_prefill:
            # 不压缩，prefill 完全保留 full KV
            # 同时重置 decode_step，后面 decode 从 1 开始计
            self.decode_step = 0
            return keys, values

        # ==== 下面就是 decode-only H2O 逻辑 ====
        # ★ 每个 decode step 递增计数
        self.decode_step += 1
        # 长度不足 budget，没必要压缩
        if kv_cache_len < self.budget:
            return keys, values
        # ★ 压缩间隔控制：只有在到达 interval 的步上才压
        #    compression_interval = 1 → 每步都会走压缩逻辑（对齐原版 H2O）
        if (self.compression_interval > 1) and (self.decode_step % self.compression_interval != 0):
            # 不到压缩步：直接返回原始 KV
            return keys, values
        # 取最后一个 token 当 query
        q_last = hidden_states[:, -1:, :]  # [B, 1, hidden_dim]
        assert q_last.shape[-1] == num_kv_heads * head_dim, (
            f"hidden_dim = {q_last.shape[-1]} 不等于 num_heads * head_dim = "
            f"{num_kv_heads} * {head_dim}"
        )
        query_states = q_last.view(bsz, num_kv_heads, 1, head_dim)  # [B, H, 1, D]

        # === 计算对历史 KV 的注意力（和你原实现一样） ===
        # attn_weights: [B, H, T]
        attn_weights = compute_attention_scores(query_states, keys).squeeze(2)

        # 对前 (T - window_size) 段做 softmax，然后对 head 求平均
        # 形状: [B, H, T - window_size] -> softmax -> mean over heads -> [B, 1, T - window_size]
        prefix = attn_weights[:, :, : -self.window_size]
        attn_weights_sum = (
            F.softmax(prefix, dim=-1, dtype=torch.float32)
            .mean(dim=-2)  # over heads
            .to(query_states.dtype)
        )  # [B, T - window_size]

        # === 从 prefix 中选出 budget - window_size 个历史 token ===
        n_keep_prefix = self.budget - self.window_size
        # indices_prefix: [B, n_keep_prefix]
        indices_prefix = attn_weights_sum.topk(n_keep_prefix, dim=-1).indices

        # ③ —— 记录 kept_token_indices（完全照你原来的逻辑，只是多了 batch 维）
        if self.record_kept_token_indices:
            indices_cl = indices_prefix.clone().to("cpu")  # [B, n_keep_prefix]

            # 最近 window 的原始位置：[T - window_size, ..., T-1]
            recent_window_indices = torch.arange(
                kv_cache_len - self.window_size,
                kv_cache_len,
                device="cpu",
            ).unsqueeze(0).expand(bsz, -1)  # [B, window_size]

            # 当前 step 的 indices（历史部分 + 最近 window）
            cur_indices = torch.cat([indices_cl, recent_window_indices], dim=-1)  # [B, budget]

            if self.evicted_token_num > 0 and len(self.kept_token_indices) > 0:
                prev_indices = self.kept_token_indices[-1]  # [B, budget_prev]
                mask = cur_indices < self.budget  # [B, budget]

                for i in range(bsz):
                    positions = torch.where(mask[i])[0]
                    for pos in positions:
                        val = cur_indices[i, pos].item()
                        cur_indices[i, pos] = prev_indices[i, val]

                cur_indices[~mask] += self.evicted_token_num

            self.kept_token_indices.append(cur_indices)
            self.evicted_token_num += kv_cache_len - self.budget

        # === 用选中的 prefix indices + window 做真正的 gather ===
        # indices_prefix: [B, n_keep_prefix] -> [B, 1, n_keep_prefix, 1] -> broadcast 到 [B, H, n_keep_prefix, D]
        gather_idx = indices_prefix.unsqueeze(1).unsqueeze(-1).expand(
            bsz, num_kv_heads, n_keep_prefix, head_dim
        )

        k_past_compress = keys[:, :, : -self.window_size, :].gather(dim=2, index=gather_idx)
        v_past_compress = values[:, :, : -self.window_size, :].gather(dim=2, index=gather_idx)

        k_cur = keys[:, :, -self.window_size :, :]
        v_cur = values[:, :, -self.window_size :, :]

        new_keys = torch.cat([k_past_compress, k_cur], dim=2)   # [B, H, budget, D]
        new_vals = torch.cat([v_past_compress, v_cur], dim=2)

        # ④ —— logging：把 “哪些 token 被 evict 了” 记下来（配合你后面的 _save_all_evicted_masks）
        if self.enable_log:
            layer_idx = getattr(module, "layer_idx", None)
            if layer_idx is not None:
                # 先构造 keep_mask: [B, T]
                keep_mask = torch.zeros(bsz, kv_cache_len, dtype=torch.bool, device=keys.device)

                # prefix 部分
                keep_mask.scatter_(1, indices_prefix, True)
                # window 部分
                recent_idx = torch.arange(
                    kv_cache_len - self.window_size, kv_cache_len, device=keys.device
                ).unsqueeze(0).expand(bsz, -1)
                keep_mask.scatter_(1, recent_idx, True)

                evicted_mask = ~keep_mask  # [B, T]

                record = {
                    "sample_idx": self._sample_idx,              # 来自 ScorerPress.reset_global_scores
                    "evicted_mask": evicted_mask.detach().cpu(), # [B, T]
                }
                if layer_idx not in self.logged_evicted:
                    self.logged_evicted[layer_idx] = []
                self.logged_evicted[layer_idx].append(record)

        return new_keys.contiguous(), new_vals.contiguous()
