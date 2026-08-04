# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import logging
import os

import torch
from torch import nn

from kvpress.presses.base_press import BasePress
import torch.nn.functional as F
import math

from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from pathlib import Path
from datasets import load_dataset

try:
    # HF Llama uses RoPE via apply_rotary_pos_emb in modeling_llama
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
except Exception:
    apply_rotary_pos_emb = None
logger = logging.getLogger(__name__)
@dataclass
class GatedPress(BasePress):
    # gate_mode:
    # - "dynamic": gate_score = gate_proj(hidden_states) (query-dependent)
    # - "static": per-layer per-KV-head per-dim trainable gates (query-independent)
    gate_mode: str = "dynamic"  # "dynamic" | "static"
    static_separate_kv: bool = True  # if True, learn separate gates for K and V; else share K gate

    gate_type: str = "elementwise"  # "headwise" or "elementwise"
    init: str = "open"
    init_open_p: float = 0.999
    bias: Optional[bool] = True  # 强烈建议直接 True，否则没 bias 没法“全开”
    record_mse: bool = False          # 训练时开 True；评测时 False
    mse_detach_target: bool = True    # 防止梯度流进“未门控的 x”
    # MSE reduction strategy:
    # - "mean": mean over all elements (B*Q*H*D). Can yield tiny gradients for large hidden sizes.
    # - "sum": sum over all elements (very large scale; usually needs normalization outside).
    # - "token_mean": sum over hidden, mean over tokens (B*Q). Good default for stable gradients.
    mse_reduction: str = "token_mean"
    aux_fp32: bool = True              
    reg_type: str = "group_lasso"
    gate_sparsity_threshold: float = 0.5  # fraction of sigmoid(gate_score) below this threshold
    # ---------- Channel pruning (ThinK-style) ----------
    # NOTE: This does not reduce tensor shapes (same as ThinKPress); it zeros out pruned dims.
    key_channel_compression_ratio: float = 0.0
    window_size: int = 32
    # Pruning controls
    # - pairwise_prune: prune RoPE dimension pairs (2i, 2i+1) together.
    # - sync_kv_prune: apply the same pruning mask to both K and V.
    # pairwise_prune: bool = False
    pairwise_prune: bool = True
    # sync_kv_prune: bool = False
    sync_kv_prune: bool = True
    # Same as ThinK: compensate attention temperature after pruning K dims.
    # Scales pruned keys by sqrt(d_k / d_eff) where d_eff = d_k - n_pruned.
    qk_scale_correction: bool = True

    _warned_channel_prune_unsupported: bool = False  # internal: warn once
    # ---------- Training controls ----------
    # To compute MSE between SDPA outputs with/without gate, we run two forwards:
    # - capture_mode="baseline": store ungated SDPA output as target (typically under no_grad)
    # - capture_mode="gated": compute MSE vs stored target and record aux losses
    capture_mode: str = "gated"  # "baseline" | "gated"
    apply_kv_gate: bool = True   # whether to apply gate to K/V projections (G3/G2)
    record_gate_stats: bool = False  # record gate_mean/bin/sparsity even if record_mse=False
    
    train_gate_mode: str = "soft"  # "soft" | "ste_topk"
    rope_pairing: str = "half"  # "half" for Llama RoPE; "even_odd" for (0,1)(2,3) style
    cache_infer_mask: bool = True
    prune_in_compress: bool = False
    # --- Retrieval-head protection (auto, DuoAttention-style) ---
    protect_retrieval_heads: bool = False
    retrieval_head_topk: int = 2

    # DuoAttention on-the-fly settings (match your _compute_duo_scores usage)
    duo_cache_dir: str = os.path.join(os.path.expanduser("~"), ".cache", "kvpress_duo")
    duo_num_samples: int = 50
    duo_q_len: int = 500
    duo_dataset: str = "kmfoda/booksum"
    duo_split: str = "train"
    duo_text_field: str = "chapter"
    duo_seed: int = 42

    # You may specify either KV-head indices or Q-head indices.
    # KV heads index range: [0, num_kv_heads)
    protected_kv_heads: Optional[List[int]] = None
    # Q heads index range: [0, num_heads)
    protected_q_heads: Optional[List[int]] = None

    # If True: keep overall key-channel ratio by pruning NON-protected heads more.
    # If False: protected heads stay full, overall compression becomes smaller (less aggressive).
    preserve_global_key_channel_ratio: bool = True

    #------debug--------
    debug_layers: Optional[List[int]] = None
    debug: bool = False

    @property
    def compression_ratio(self):
        # GatedPress does not compress KV length; keep this as 0.0 for compatibility with eval code.
        # Channel pruning is controlled via `key_channel_compression_ratio`.
        return 0.0
    def _dbg_layer_enabled(self, obj) -> bool:
        if not getattr(self, "debug", False):
            return False
        layers = getattr(self, "debug_layers", None)  # 你可以在 dataclass 里加 debug_layers: Optional[List[int]]=None
        if not layers:
            return True
        idx = getattr(obj, "_kvpress_layer_idx", None)
        return idx in set(layers)

    def _dbg_once(self, obj, tag: str) -> bool:
        key = f"_kvpress_dbg_once_{tag}"
        if getattr(obj, key, False):
            return False
        setattr(obj, key, True)
        return True

    @compression_ratio.setter
    def compression_ratio(self, value):
        # evaluation code may try to set it; ignore to stay compatible
        return
    def _build_ste_topk_mask(self, gate_prob: torch.Tensor, keep_ratio: float, pairwise: bool):
        """
        作用：
        - 给 gate_prob 做“Top-k 保留”：保留最重要的维度，其余置 0。
        - 用 STE (straight-through estimator)：前向像 hard mask，反向梯度仍走 gate_prob。
        - 可选 pairwise：按 RoPE 维度对（2i,2i+1 或 half-half）一起保留/剪枝。

        何时调用：
        - 训练时 train_gate_mode='ste_topk'
        - 或推理时 cache_infer_mask=True（prefill 算一次 mask，decode 复用）

        输入：
        - gate_prob: [B,Q,kvH,D]，sigmoid 后的 gate 概率
        - keep_ratio: 每个 head 维度保留比例（=1-prune_ratio，可能被“保护 head”逻辑改写）
        - pairwise: 是否按 RoPE pairing 做成对剪枝

        输出：
        - mask_ste: [B,Q,kvH,D]
        - k_keep_eff: 实际保留的维度数（pairwise 会变成偶数）
        """
        B, Q, H, D = gate_prob.shape
        raw_ws = int(getattr(self, "window_size", 32) or 32)
        ws = Q if raw_ws == -1 else max(1, min(raw_ws, Q))

        scores = gate_prob[:, -ws:, :, :].mean(dim=1, keepdim=True)  # [B,1,H,D]
        scores_det = scores.detach()

        if pairwise and (D % 2 == 0):
            pair_scores = self._pair_scores(scores_det)              # [B,1,H,D/2]  (half-half by default)
            k_keep_pairs = max(1, min((D // 2), int(round((D // 2) * keep_ratio))))
            keep_pair_idx = pair_scores.topk(k_keep_pairs, dim=-1, largest=True).indices
            mask_pairs = torch.zeros_like(pair_scores).scatter_(-1, keep_pair_idx, 1.0)
            mask_hard = self._expand_pair_mask(mask_pairs)           # [B,1,H,D]
            k_keep_eff = 2 * k_keep_pairs
        else:
            k_keep = max(1, min(D, int(round(D * keep_ratio))))
            keep_idx = scores_det.topk(k_keep, dim=-1, largest=True).indices
            mask_hard = torch.zeros_like(scores_det).scatter_(-1, keep_idx, 1.0)
            k_keep_eff = k_keep

        mask_hard = mask_hard.expand(-1, Q, -1, -1)                  # [B,Q,H,D]
        mask_ste = (mask_hard - gate_prob).detach() + gate_prob

        return mask_ste, k_keep_eff

    def post_init_from_model(self, model: nn.Module, attn_modules=None):
        if attn_modules is None:
            language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
            attn_modules = [layer.self_attn for layer in language_model.layers]

        # for module in attn_modules:
        #assign layer_idx so hooks can know which layer they are in
        for layer_idx, module in enumerate(attn_modules):
            module._kvpress_layer_idx = layer_idx
            num_heads, num_kv_heads, head_dim, _ = self._infer_head_info(module)
            device = module.q_proj.weight.device if hasattr(module, "q_proj") else next(module.parameters()).device
            dtype  = module.q_proj.weight.dtype  if hasattr(module, "q_proj") else next(module.parameters()).dtype

            if getattr(self, "gate_mode", "dynamic") == "static":
                self._get_or_create_static_gates(module, num_kv_heads=num_kv_heads, head_dim=head_dim, device=device, dtype=dtype)
            else:
                # For KV channel pruning / gating, we parameterize gates in KV head space.
                out_dim = num_kv_heads if self.gate_type == "headwise" else (num_kv_heads * head_dim)
                hidden_size = module.q_proj.in_features if hasattr(module, "q_proj") else (num_heads * head_dim)
                _ = self._get_or_create_gate_proj(module, hidden_size, out_dim, device, dtype)
        # ---- compute & attach retrieval head indices (optional) ----
        if getattr(self, "protect_retrieval_heads", False):
            # only do this when the model is in eval/inference mode
            any_training = any(getattr(m, "training", False) for m in attn_modules)
            if not any_training:
                self._init_retrieval_heads(model, attn_modules)
        if getattr(self, "protect_retrieval_heads", False) and getattr(self, "force_init_retrieval_heads", False):
            self._init_retrieval_heads(model, attn_modules)
    def _attn_posthook_check(self, module, inputs, output):
        """
        作用：
        - debug 用：forward 结束后检查 k_proj/v_proj 上是否还残留 _kvpress_gate_mask。
        - 正常情况下，_kvproj_forward_hook_apply_gate 会在用完后删除它，避免跨 step 污染。
        """
        if not getattr(self, "debug", False): 
            return
        if getattr(module, "_dbg_post_once", False):
            return
        has_k = hasattr(module.k_proj, "_kvpress_gate_mask") if hasattr(module, "k_proj") else False
        has_v = hasattr(module.v_proj, "_kvpress_gate_mask") if hasattr(module, "v_proj") else False
        # logger.warning(f"[GATED DBG] posthook layer={getattr(module,'_kvpress_layer_idx',None)} "
        #             f"leftover_gate_mask k={has_k} v={has_v}")
        module._dbg_post_once = True

    # ---------- hook registration ----------
    def _has_any_hooks(self, m):
        """
        作用：
        - 工具函数：判断一个 module 上是否已经有 forward hook / pre-hook。
        - 主要用于 debug 或防止重复注册。
        """
        return (len(getattr(m, "_forward_hooks", {})) +
                len(getattr(m, "_forward_pre_hooks", {}))) > 0
    def register_hooks(self, module: nn.Module):
        """
        作用：
        - 给单层 attention 安装 hooks（这是 GatedPress “真正工作的地方”）。
        - 顺序大致是：
        1) attention pre-hook：算 gate_mask（dynamic 才需要）
        2) k_proj/v_proj forward hook：把 gate_mask 乘到输出上
        3) o_proj pre-hook：训练时记录 MSE/正则（可选）

        实现细节：
        - 会先 remove 旧 handles，避免重复注册导致同一层 hook 越挂越多。
        - 会在 k_proj/v_proj/o_proj 上写入 layer_idx、static gate 引用等元信息。

        PyTorch 机制点：
        - hook 返回的 handle 需要 .remove() 才能卸载。:contentReference[oaicite:1]{index=1}
        """
        c = getattr(module, "_kvpress_reg_count", 0) + 1
        module._kvpress_reg_count = c
        # if c in (1, 2, 3, 10, 100, 1000, 10000):
        #     logger.warning("[GATED DBG] register_hooks call_count=%d layer=%s type=%s",
        #                 c, getattr(module, "_kvpress_layer_idx", None), type(module).__name__)

        # --- 如果之前装过但可能被外部 remove：先把旧 handles 全 remove，保证不会堆叠 ---
        old = getattr(module, "_kvpress_gated_handles", None) or []
        for h in old:
            try:
                h.remove()
            except Exception:
                pass

        handles = []

        # debug 时才挂观察 posthook（只挂一次）
        if getattr(self, "debug", False):
            handles.append(module.register_forward_hook(self._attn_posthook_check))

        # 防御：不是所有 attention 都叫 k_proj/v_proj/o_proj
        k_proj = getattr(module, "k_proj", None)
        v_proj = getattr(module, "v_proj", None)
        o_proj = getattr(module, "o_proj", None)

        if not (isinstance(k_proj, nn.Module) and isinstance(v_proj, nn.Module) and isinstance(o_proj, nn.Module)):
            handles += super().register_hooks(module)
            module._kvpress_gated_handles = handles
            module._kvpress_gated_hooks_installed = True
            return handles

        # 标记层号/类型
        layer_idx = getattr(module, "_kvpress_layer_idx", None)
        module.k_proj._kvpress_is_k_proj = True
        module.v_proj._kvpress_is_k_proj = False
        module.k_proj._kvpress_layer_idx = layer_idx
        module.v_proj._kvpress_layer_idx = layer_idx
        module.o_proj._kvpress_layer_idx = layer_idx

        if getattr(self, "gate_mode", "dynamic") == "static":
            module.k_proj._kvpress_static_gate_logits = getattr(module, "_kvpress_static_gate_logits_k", None)
            module.v_proj._kvpress_static_gate_logits = getattr(module, "_kvpress_static_gate_logits_v", None)
            module.o_proj._kvpress_static_gate_logits_k = getattr(module, "_kvpress_static_gate_logits_k", None)
            module.o_proj._kvpress_static_gate_logits_v = getattr(module, "_kvpress_static_gate_logits_v", None)

        if getattr(self, "gate_mode", "dynamic") == "dynamic":
            handles.append(module.register_forward_pre_hook(self._attn_prehook_capture_gate, with_kwargs=True))

        handles.append(module.k_proj.register_forward_hook(self._kvproj_forward_hook_apply_gate))
        handles.append(module.v_proj.register_forward_hook(self._kvproj_forward_hook_apply_gate))
        handles.append(module.o_proj.register_forward_pre_hook(self._oproj_prehook_capture_sdpa, with_kwargs=False))

        if getattr(self, "prune_in_compress", False):
            handles.append(module.register_forward_hook(self.forward_hook, with_kwargs=True))

        module._kvpress_gated_handles = handles
        module._kvpress_gated_hooks_installed = True
        return handles

    def _infer_head_info(self, module: nn.Module) -> Tuple[int, int, int, int]:
        def _get_attr(names):
            for name in names:
                if (val := getattr(module, name, None)) is not None:
                    return val
            if hasattr(module, "config"):
                for name in names:
                    if (val := getattr(module.config, name, None)) is not None:
                        return val
            return None

        num_heads = _get_attr(["num_heads", "n_heads", "num_attention_heads"])
        if num_heads is None:
            raise AttributeError("Cannot infer num_heads from attention module.")

        num_kv_heads = _get_attr(["num_key_value_heads", "n_kv_heads", "num_kv_heads"])
        if num_kv_heads is None:
            # fallback: assume MHA
            num_kv_heads = num_heads

        head_dim = _get_attr(["head_dim"])
        if head_dim is None:
            hidden_size = _get_attr(["hidden_size", "embed_dim"])
            if hidden_size is None:
                raise AttributeError("Cannot infer hidden_size/head_dim from attention module.")
            head_dim = hidden_size // int(num_heads)
            return int(num_heads), int(num_kv_heads), int(head_dim), int(hidden_size)
        
        return int(num_heads), int(num_kv_heads), int(head_dim), int(num_heads) * int(head_dim)
    def _resolve_protected_kv_heads(self, num_heads: int, num_kv_heads: int) -> List[int]:
        prot = set()

        if self.protected_kv_heads:
            prot |= {int(h) for h in self.protected_kv_heads}

        if self.protected_q_heads:
            q_list = [int(h) for h in self.protected_q_heads]
            if num_kv_heads == num_heads:
                prot |= {h for h in q_list}
            else:
                # HF GQA usually repeats each KV head `group = num_heads // num_kv_heads` times consecutively
                group = max(1, num_heads // num_kv_heads)
                prot |= {h // group for h in q_list}

        # clamp into valid KV-head range
        prot = sorted([h for h in prot if 0 <= h < num_kv_heads])
        return prot

    def _get_language_model(self, model: nn.Module):
        # mirror your existing logic
        return model.model.language_model if hasattr(model.model, "language_model") else model.model

    def _duo_cache_path(self, model: nn.Module) -> str:
        """
        作用：
        - 生成 duo_attention 的缓存文件路径（按模型名、采样数、q_len、topk 区分）。
        - 用来把 retrieval heads 的 index 落盘，避免每次启动都重新跑采样。
        """
        Path(self.duo_cache_dir).mkdir(parents=True, exist_ok=True)
        name = getattr(model.config, "name_or_path", "model").replace("/", "_")
        return os.path.join(
            self.duo_cache_dir,
            f"retrieval_kv_heads__{name}__ns{self.duo_num_samples}__ql{self.duo_q_len}__top{self.retrieval_head_topk}.pt"
        )

    @torch.no_grad()
    def _compute_duo_scores(self, model: nn.Module) -> torch.Tensor:
        """
        Returns: scores [num_layers, num_kv_heads] on CPU float32
        Logic mirrors the code you pasted: mean Q/K -> repeat q_len -> RoPE -> attn(last token) -> AUC(cumsum)
        作用：
        - “复刻 DuoAttentionPress 的 on_the_fly 思路”：用采样文本估计每层每个 KV head 的 retrieval 强度评分。
        - 过程：
        1) 采样文本 -> forward 拿 hidden_states
        2) 对每层：算 mean Q/mean K -> repeat 到 q_len -> RoPE -> 只看最后 token 的注意力
        3) cumsum 后求均值，得到每个 head 的 score
        4) 多文本平均得到稳定分数
        """
        language_model = self._get_language_model(model)
        tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)

        num_heads = model.config.num_attention_heads
        num_kv_heads = model.config.num_key_value_heads
        num_kv_groups = num_heads // num_kv_heads

        # sample texts
        ds = load_dataset(self.duo_dataset, split=self.duo_split)
        n = min(self.duo_num_samples, len(ds))
        rng = random.Random(self.duo_seed)
        idxs = rng.sample(range(len(ds)), n)
        texts = [ds[i][self.duo_text_field] for i in idxs]

        q_len = int(self.duo_q_len)
        position_ids = torch.arange(q_len, device=model.device).unsqueeze(0)

        scores = torch.zeros((model.config.num_hidden_layers, num_kv_heads), dtype=torch.float32)

        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)
            out = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
            hidden_states = list(out.hidden_states[:-1])  # drop final

            for layer_idx, h in enumerate(hidden_states):
                layer = language_model.layers[layer_idx]
                attn = layer.self_attn
                d = attn.head_dim

                if hasattr(layer, "input_layernorm"):
                    h = layer.input_layernorm(h)

                # mean query
                q = attn.q_proj(h)                             # [1, T, num_heads*d]
                q = q.view(1, q.shape[1], -1, d)               # [1, T, num_heads, d]
                if hasattr(attn, "q_norm"):
                    q = attn.q_norm(q)
                q = q.mean(dim=1, keepdim=True)                # [1, 1, num_heads, d]
                q = q.repeat(1, q_len, 1, 1).transpose(1, 2)   # [1, num_heads, q_len, d]

                # mean key
                k = attn.k_proj(h)                             # [1, T, num_kv_heads*d]
                k = k.view(1, k.shape[1], -1, d)               # [1, T, num_kv_heads, d]
                if hasattr(attn, "k_norm"):
                    k = attn.k_norm(k)
                k = k.mean(dim=1, keepdim=True)                # [1, 1, num_kv_heads, d]
                k = k.repeat(1, q_len, 1, 1).transpose(1, 2)   # [1, num_kv_heads, q_len, d]

                # RoPE (match HF Llama: rotary_emb + apply_rotary_pos_emb) :contentReference[oaicite:4]{index=4}
                rotary = getattr(attn, "rotary_emb", None) or getattr(language_model, "rotary_emb", None)
                if rotary is not None and apply_rotary_pos_emb is not None:
                    cos, sin = rotary(k, position_ids)  # many impls accept (x, position_ids)
                    try:
                        q, k = apply_rotary_pos_emb(q, k, cos, sin)
                    except TypeError:
                        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

                # expand kv heads to match num_heads
                k = k.repeat_interleave(num_kv_groups, dim=1)  # [1, num_heads, q_len, d]

                # attention of last token
                attn_w = (q[:, :, -1:, :] @ k.transpose(2, 3)) / (d ** 0.5)
                attn_w = attn_w.softmax(dim=-1, dtype=torch.float32).squeeze()  # [num_heads, q_len]

                s = torch.cumsum(attn_w, dim=1).mean(1)        # [num_heads]
                s = s.view(-1, num_kv_groups).mean(1)          # [num_kv_heads]

                scores[layer_idx] += s.detach().cpu() / float(n)

        return scores

    def _init_retrieval_heads(self, model: nn.Module, attn_modules: List[nn.Module]):
        """
        作用：
        - 初始化 retrieval heads：
        - 如果缓存文件存在：直接 load retrieval_idx
        - 否则：调用 _compute_duo_scores，按 topk 取每层最强的 KV heads，并写入缓存
        """
        cache_path = self._duo_cache_path(model)
        if os.path.exists(cache_path):
            obj = torch.load(cache_path, map_location="cpu")
            retrieval_idx = obj["retrieval_idx"]
        else:
            scores = self._compute_duo_scores(model)  # [L, kv_heads]
            topk = max(1, min(int(self.retrieval_head_topk), scores.shape[1]))
            retrieval_idx = []
            for l in range(scores.shape[0]):
                idx = torch.topk(scores[l], k=topk, largest=True).indices.tolist()
                retrieval_idx.append(idx)
            tmp = cache_path + f".tmp.{os.getpid()}"
            torch.save({"retrieval_idx": retrieval_idx}, tmp)
            os.replace(tmp, cache_path)

        # attach indices to each layer module
        for layer_idx, m in enumerate(attn_modules):
            m._kvpress_retrieval_kv_head_idx = retrieval_idx[layer_idx]

    def _pair_scores(self, scores: torch.Tensor) -> torch.Tensor:
        # scores: [B, 1, kv_heads, head_dim]
        d = scores.size(-1)
        assert d % 2 == 0, "head_dim must be even for pairwise pruning"
        if getattr(self, "rope_pairing", "half") == "half":
            h = d // 2
            return 0.5 * (scores[..., :h] + scores[..., h:])
        else:
            return 0.5 * (scores[..., 0::2] + scores[..., 1::2])

    def _expand_pair_mask(self, mask_pairs: torch.Tensor) -> torch.Tensor:
        # mask_pairs: [B, 1, kv_heads, head_dim/2] -> [B, 1, kv_heads, head_dim]
        if getattr(self, "rope_pairing", "half") == "half":
            return torch.cat([mask_pairs, mask_pairs], dim=-1)
        else:
            out = torch.zeros(mask_pairs.shape[:-1] + (mask_pairs.size(-1) * 2,),
                            device=mask_pairs.device, dtype=mask_pairs.dtype)
            out[..., 0::2] = mask_pairs
            out[..., 1::2] = mask_pairs
            return out

    def _get_or_create_gate_proj(self, module: nn.Module, hidden_size: int, out_dim: int, device, dtype) -> nn.Linear:
        """
        作用：
        - dynamic gate 的线性层 gate_proj：hidden_states -> gate_score。
        - 会检查已有 gate_proj 的形状是否匹配，不匹配就重建。
        - 初始化策略：
        - 权重置 0
        - bias 用 logit(init_open_p) 让 gate 默认接近全开（比如 0.999）
        """
        proj = getattr(module, "_kvpress_gate_proj", None)
        # Check if existing proj matches spec
        if isinstance(proj, nn.Linear) and proj.in_features == hidden_size and proj.out_features == out_dim:
            return proj
        
        # Infer bias: use self.bias, or follow q_proj convention
        use_bias = self.bias if self.bias is not None else hasattr(getattr(module, "q_proj", None), "bias")
        proj = nn.Linear(hidden_size, out_dim, bias=use_bias).to(device=device, dtype=dtype)
        
        # Initialize weights
        nn.init.zeros_(proj.weight)
        if proj.bias is not None:
            if self.init in ("open", "all_open"):
                # Gate = sigmoid(bias) ≈ init_open_p → bias = logit(p)
                p = getattr(self, "init_open_p", 0.999)
                proj.bias.data.fill_(math.log(p / (1.0 - p)))
            else:  # "zeros"
                nn.init.zeros_(proj.bias)
        elif self.init in ("open", "all_open"):
            raise ValueError("init='open' requires bias=True")
        
        setattr(module, "_kvpress_gate_proj", proj)
        return proj

    def _get_or_create_static_gates(self, module: nn.Module, num_kv_heads: int, head_dim: int, device, dtype):
        """
        Create per-layer static trainable gates for K/V:
          _kvpress_static_gate_logits_k: [num_kv_heads, head_dim or 1]
          _kvpress_static_gate_logits_v: [num_kv_heads, head_dim or 1] (optional / shared)
        Gate values are sigmoid(logits).
        """
        gate_dim = 1 if getattr(self, "gate_type", "elementwise") == "headwise" else head_dim
        p = float(getattr(self, "init_open_p", 0.999))
        init_logit = math.log(p / (1.0 - p)) if getattr(self, "init", "open") in ("open", "all_open") else 0.0

        def _ensure_param(name: str) -> nn.Parameter:
            existing = getattr(module, name, None)
            if isinstance(existing, nn.Parameter) and tuple(existing.shape) == (num_kv_heads, gate_dim):
                return existing
            param = nn.Parameter(torch.full((num_kv_heads, gate_dim), init_logit, device=device, dtype=dtype))
            if name in module._parameters:
                module._parameters[name] = param
            else:
                module.register_parameter(name, param)
            return param

        k_logits = _ensure_param("_kvpress_static_gate_logits_k")
        if getattr(self, "static_separate_kv", True):
            v_logits = _ensure_param("_kvpress_static_gate_logits_v")
        else:
            # share V gate with K gate
            v_logits = k_logits
            module._kvpress_static_gate_logits_v = k_logits

        # Attach parameter refs onto projection modules for hooks (no nn.Module refs to avoid cycles).
        if hasattr(module, "k_proj"):
            module.k_proj._kvpress_static_gate_logits = k_logits
        if hasattr(module, "v_proj"):
            module.v_proj._kvpress_static_gate_logits = v_logits
        if hasattr(module, "o_proj"):
            module.o_proj._kvpress_static_gate_logits_k = k_logits
            module.o_proj._kvpress_static_gate_logits_v = v_logits

    # ---------- hooks ----------
    def _attn_prehook_capture_gate(self, module, args, kwargs):
        """
        作用：
        - attention forward 之前执行：根据 hidden_states 计算 gate_mask，并存到 k_proj/v_proj 上。
        - 这是 dynamic gate 的核心：
        1) gate_proj(hidden_states) -> gate_score
        2) sigmoid -> gate_prob -> reshape 成 [B,Q,kvH,D]
        3) 如果推理且 cache_infer_mask=True：
            - prefill 计算一次 hard mask 缓存到 module._kvpress_cached_gate_mask
            - decode 复用缓存 mask
        4) 如果 protect_retrieval_heads=True：
            - 把 retrieval head 的 mask 强行设为 1（绝不剪）
        5) 可选 qk_scale_correction：对 K 做温度修正（避免剪维后 attention 变“过热”）
        """
        hidden_states = args[0] if (args and len(args) > 0) else kwargs.get("hidden_states", None)
        if hidden_states is None or hidden_states.dim() != 3:
            return

        bsz, q_len, hidden_size = hidden_states.shape
        num_heads, num_kv_heads, head_dim, _ = self._infer_head_info(module)
        out_dim = num_kv_heads if self.gate_type == "headwise" else (num_kv_heads * head_dim)

        # # ---- (A) HOOK 数量快照：每层只打一次，用来确认是否重复注册 ----
        # if self._dbg_layer_enabled(module) and self._dbg_once(module, "hook_snapshot"):
        #     k = getattr(module, "k_proj", None)
        #     v = getattr(module, "v_proj", None)
        #     o = getattr(module, "o_proj", None)
        #     logger.warning(
        #         "[GATED DBG] HOOK_SNAPSHOT layer=%s q_len=%d | attn_pre=%d attn_fwd=%d | "
        #         "k_fwd=%d v_fwd=%d | o_pre=%d",
        #         getattr(module, "_kvpress_layer_idx", None), q_len,
        #         len(getattr(module, "_forward_pre_hooks", {})),
        #         len(getattr(module, "_forward_hooks", {})),
        #         len(getattr(k, "_forward_hooks", {})) if k is not None else -1,
        #         len(getattr(v, "_forward_hooks", {})) if v is not None else -1,
        #         len(getattr(o, "_forward_pre_hooks", {})) if o is not None else -1,
        #     )

        # ---- (B) past 推断：只做一次，并且尽量把 past_seq 推断准确 ----
        past = kwargs.get("past_key_value", None) or kwargs.get("past_key_values", None)
        if past is None and args is not None and len(args) >= 4:
            past = args[3]

        past_seq = None
        if past is None:
            past_seq = 0
        elif hasattr(past, "get_seq_length"):
            try:
                past_seq = int(past.get_seq_length())
            except Exception:
                past_seq = None
        elif isinstance(past, (tuple, list)) and len(past) > 0:
            # 兼容 (k, v) 这种老式 cache
            k0 = past[0]
            if isinstance(k0, torch.Tensor) and k0.dim() == 4:
                # 常见两种： [B, kvH, S, D] 或 [B, S, kvH, D]
                if k0.shape[-1] == head_dim:
                    if k0.shape[1] == num_kv_heads:
                        past_seq = int(k0.shape[2])
                    elif k0.shape[2] == num_kv_heads:
                        past_seq = int(k0.shape[1])
                    else:
                        past_seq = int(k0.shape[-2])
                else:
                    past_seq = int(k0.shape[-2])

        # ✅ 只用 q_len 判别：HF DynamicCache 里这是最稳的
        is_decode  = (q_len == 1)
        is_prefill = (q_len > 1)

        prune_ratio = float(getattr(self, "key_channel_compression_ratio", 0.0) or 0.0)
        keep_ratio  = 1.0 - prune_ratio

        # ---- (C) gate_proj + gate_score ----
        gate_proj = self._get_or_create_gate_proj(
            module, hidden_size, out_dim, hidden_states.device, hidden_states.dtype
        )
        gate_score = gate_proj(hidden_states)  # [B,Q,out_dim]

        # if self._dbg_layer_enabled(module) and self._dbg_once(module, "gate_proj_spec"):
        #     logger.warning(
        #         "[GATED DBG] GATE_PROJ layer=%s id=%s in=%d out=%d bias=%s dtype=%s dev=%s",
        #         getattr(module, "_kvpress_layer_idx", None),
        #         id(gate_proj), gate_proj.in_features, gate_proj.out_features,
        #         (gate_proj.bias is not None), str(gate_proj.weight.dtype), str(gate_proj.weight.device),
        #     )

        # reshape -> [B,Q,kvH,D]
        if self.gate_type == "headwise":
            gate_score = gate_score.view(bsz, q_len, num_kv_heads, 1)
        else:
            gate_score = gate_score.view(bsz, q_len, num_kv_heads, head_dim)

        gate_prob = torch.sigmoid(gate_score)
        if gate_prob.shape[-1] == 1:
            gate_prob = gate_prob.expand(-1, -1, -1, head_dim)

        # ---- 你的 keep_ratio_nonprot 逻辑保持不动 ----
        if getattr(self, "protect_retrieval_heads", False) and getattr(self, "preserve_global_key_channel_ratio", True):
            retrieval_idx = getattr(module, "_kvpress_retrieval_kv_head_idx", []) or []
            P = len(retrieval_idx); H = num_kv_heads; D = head_dim
            if 0 < P < H:
                target_keep_total = int(round(H * D * keep_ratio))
                keep_prot_total   = P * D
                keep_nonprot_total = max(1, target_keep_total - keep_prot_total)
                keep_ratio_nonprot = keep_nonprot_total / float((H - P) * D)
                keep_ratio_nonprot = float(max(1.0 / D, min(1.0, keep_ratio_nonprot)))
            else:
                keep_ratio_nonprot = keep_ratio
        else:
            keep_ratio_nonprot = keep_ratio

        gate_mask = gate_prob
        k_keep_eff = None

        if prune_ratio > 0 and module.training and self.train_gate_mode == "ste_topk":
            gate_mask, k_keep_eff = self._build_ste_topk_mask(
                gate_prob, keep_ratio_nonprot, pairwise=self.pairwise_prune
            )

        elif prune_ratio > 0 and (not module.training) and getattr(self, "cache_infer_mask", True):
            if is_prefill:
                hard_mask, k_keep_eff = self._build_ste_topk_mask(
                    gate_prob, keep_ratio_nonprot, pairwise=self.pairwise_prune
                )
                hard_mask = hard_mask[:, :1, :, :].detach()
                module._kvpress_cached_gate_mask = hard_mask
                module._kvpress_cached_k_keep_eff = k_keep_eff

            base = getattr(module, "_kvpress_cached_gate_mask", None)
            if base is not None:
                gate_mask = base.expand(-1, q_len, -1, -1)
                k_keep_eff = getattr(module, "_kvpress_cached_k_keep_eff", None)
            else:
                # ---- (D) decode/推理阶段没命中缓存：每层只打一次 ----
                if is_decode and self._dbg_layer_enabled(module) and self._dbg_once(module, "decode_no_cache"):
                    logger.warning(
                        "[GATED DBG] NO_CACHED_MASK layer=%s q_len=%d past_seq=%s "
                        "=> decode but cached_mask is None (prefill没走到? 或 hooks没生效?)",
                        getattr(module, "_kvpress_layer_idx", None), q_len, str(past_seq)
                    )

        # protect retrieval heads
        if getattr(self, "protect_retrieval_heads", False):
            retrieval_idx = getattr(module, "_kvpress_retrieval_kv_head_idx", [])
            if retrieval_idx:
                gate_mask = gate_mask.clone()
                gate_mask[:, :, retrieval_idx, :] = 1.0

        # gate_summary（你已有的保留即可）
        if self._dbg_layer_enabled(module) and self._dbg_once(module, "gate_summary"):
            gm = gate_mask
            logger.warning(
                "[GATED DBG] GATE_SUM layer=%s q_len=%d prefill=%s decode=%s past_seq=%s "
                "mask=%s min=%.4f max=%.4f mean=%.4f",
                getattr(module, "_kvpress_layer_idx", None),
                q_len, is_prefill, is_decode, str(past_seq),
                tuple(gm.shape), gm.min().item(), gm.max().item(), gm.mean().item()
            )

        # qk_scale_correction（你原逻辑保持）
        scale = None
        if getattr(self, "qk_scale_correction", True) and (k_keep_eff is not None) and k_keep_eff > 0:
            base_scale = (head_dim / float(k_keep_eff)) ** 0.5
            retrieval_idx = getattr(module, "_kvpress_retrieval_kv_head_idx", []) if getattr(self, "protect_retrieval_heads", False) else []
            if retrieval_idx:
                scale = torch.full((1, 1, num_kv_heads, 1), float(base_scale),
                                device=hidden_states.device, dtype=hidden_states.dtype)
                scale[:, :, retrieval_idx, :] = 1.0
            else:
                scale = base_scale

        if hasattr(module, "k_proj"):
            if scale is None:
                if hasattr(module.k_proj, "_kvpress_qk_scale"):
                    delattr(module.k_proj, "_kvpress_qk_scale")
            else:
                module.k_proj._kvpress_qk_scale = scale

        # stash
        if hasattr(module, "k_proj"):
            module.k_proj._kvpress_gate_mask = gate_mask
        if hasattr(module, "v_proj"):
            module.v_proj._kvpress_gate_mask = gate_mask
        if hasattr(module, "o_proj"):
            module.o_proj._kvpress_gate_score = gate_score

    def _kvproj_forward_hook_apply_gate(self, proj: nn.Module, inputs, output):
        """
        作用：
        - 挂在 k_proj/v_proj 上：在它们 forward 之后，把 gate_mask 乘到投影输出上（等价“按维度置零”）。
        - 支持两种 gate_mode：
        - static：直接 sigmoid(static_logits) 得到 gate
        - dynamic：读取 proj._kvpress_gate_mask

        实现细节：
        - 对 K：如果存在 _kvpress_qk_scale，会额外乘 scale（温度修正）
        - 用完会 del proj._kvpress_gate_mask，防止跨 step 污染（非常关键）
        """
        # 0) 基本健壮性
        if (not getattr(self, "apply_kv_gate", True)) or (not isinstance(output, torch.Tensor)) or (output.dim() != 3):
            return output

        bsz, q_len, kv_hidden = output.shape

        # 1) static gate：不依赖 _kvpress_gate_mask
        if getattr(self, "gate_mode", "dynamic") == "static":
            gate_logits = getattr(proj, "_kvpress_static_gate_logits", None)
            if not isinstance(gate_logits, (torch.Tensor, nn.Parameter)):
                return output

            kv_heads = int(gate_logits.shape[0])
            if kv_hidden % kv_heads != 0:
                if self._dbg_layer_enabled(proj) and self._dbg_once(proj, "kv_hidden_mismatch_static"):
                    logger.warning(f"[GATED DBG] layer={getattr(proj,'_kvpress_layer_idx',None)} "
                                f"kv_hidden={kv_hidden} not divisible by kv_heads={kv_heads} (static)")
                return output
            head_dim = kv_hidden // kv_heads

            gate = torch.sigmoid(gate_logits)
            if gate.dim() != 2:
                return output
            if gate.shape[1] == 1:
                gate = gate.expand(kv_heads, head_dim)
            gate = gate.view(1, 1, kv_heads, head_dim)  # broadcast

            out4 = output.view(bsz, q_len, kv_heads, head_dim)
            out  = (out4 * gate).view(bsz, q_len, kv_hidden)

            if self._dbg_layer_enabled(proj) and self._dbg_once(proj, "kv_effect_static"):
                delta = (out - output).abs().mean().item()
                zfrac = (out.view_as(out4) == 0).float().mean().item()
                logger.warning(f"[GATED DBG] layer={getattr(proj,'_kvpress_layer_idx',None)} "
                            f"{'K' if getattr(proj,'_kvpress_is_k_proj',False) else 'V'} "
                            f"static_effect mean_abs_delta={delta:.6g} zero_frac={zfrac:.4f}")

            return out

        # 2) dynamic gate：依赖 _kvpress_gate_mask
        gate = getattr(proj, "_kvpress_gate_mask", None)
        if gate is None:
            return output

        kv_heads = int(gate.shape[2])
        if kv_hidden % kv_heads != 0:
            if self._dbg_layer_enabled(proj) and self._dbg_once(proj, "kv_hidden_mismatch_dynamic"):
                logger.warning(f"[GATED DBG] layer={getattr(proj,'_kvpress_layer_idx',None)} "
                            f"kv_hidden={kv_hidden} not divisible by kv_heads={kv_heads} (dynamic)")
            return output
        head_dim = kv_hidden // kv_heads

        if gate.shape[-1] == 1:
            gate = gate.expand(-1, -1, -1, head_dim)

        out4 = output.view(bsz, q_len, kv_heads, head_dim)
        out4 = out4 * gate

        # 只对 K 应用 qk_scale_correction
        if getattr(proj, "_kvpress_is_k_proj", False):
            scale = getattr(proj, "_kvpress_qk_scale", None)
            if scale is not None:
                out4 = out4 * scale

        out = out4.view(bsz, q_len, kv_hidden)

        if self._dbg_layer_enabled(proj) and self._dbg_once(proj, "kv_effect_dynamic"):
            delta = (out - output).abs().mean().item()
            zfrac = (out4 == 0).float().mean().item()
            logger.warning(f"[GATED DBG] layer={getattr(proj,'_kvpress_layer_idx',None)} "
                        f"{'K' if getattr(proj,'_kvpress_is_k_proj',False) else 'V'} "
                        f"dynamic_effect mean_abs_delta={delta:.6g} zero_frac={zfrac:.4f}")

        # 清理，避免跨 step 污染
        if hasattr(proj, "_kvpress_gate_mask"):
            delattr(proj, "_kvpress_gate_mask")

        return out

    def _oproj_prehook_capture_sdpa(self, o_proj: nn.Linear, inputs: Tuple[torch.Tensor, ...]):
        """Capture SDPA output (o_proj input) and compute aux losses vs ungated baseline."""
        """
        作用：
        - 挂在 o_proj 的 pre-hook：拿到 SDPA 输出（也就是 o_proj 的输入 x）。
        - 用于训练时的辅助损失：
        - capture_mode='baseline'：把当前 x 存成 target
        - capture_mode='gated'：用当前 x 和之前存的 target 算 MSE
        """
        if not inputs or not isinstance(x := inputs[0], torch.Tensor) or x.dim() != 3:
            return

        # Always clear gate_score on o_proj to avoid stale reuse; k/v hooks will have consumed theirs.
        gate_score = getattr(o_proj, "_kvpress_gate_score", None)
        if hasattr(o_proj, "_kvpress_gate_score"):
            delattr(o_proj, "_kvpress_gate_score")

        # Optionally record gate stats (for budget/bin regularizers) without computing MSE targets.
        if getattr(self, "record_gate_stats", False):
            self._record_gate_stats_on_oproj(o_proj, gate_score)

        if not getattr(self, "record_mse", False):
            return

        mode = getattr(self, "capture_mode", "gated")
        if mode == "baseline":
            # Store baseline SDPA output as target (detach to avoid grads into teacher forward).
            o_proj._kvpress_sdpa_target = x.detach()
            return

        # mode == "gated": compute MSE vs stored target (if any)
        target = getattr(o_proj, "_kvpress_sdpa_target", None)
        if target is None or not isinstance(target, torch.Tensor):
            return

        to_fp32 = (lambda t: t.float()) if getattr(self, "aux_fp32", False) else (lambda t: t)
        x_loss = to_fp32(x)
        tgt_loss = to_fp32(target.detach() if getattr(self, "mse_detach_target", True) else target)

        mse_reduction = getattr(self, "mse_reduction", "token_mean")
        if mse_reduction == "token_mean":
            mse = ((x_loss - tgt_loss).pow(2).sum(dim=-1).mean())
        elif mse_reduction in ("mean", "sum", "none"):
            mse = F.mse_loss(x_loss, tgt_loss, reduction=mse_reduction)
        else:
            raise ValueError(f"Unknown mse_reduction={mse_reduction}")
        o_proj._kvpress_aux_mse = mse

        # Regularization on gate (dynamic or static)
        self._record_gate_stats_on_oproj(o_proj, gate_score, write_reg=True)

        # Clear target after use to avoid accidental mixing across steps
        delattr(o_proj, "_kvpress_sdpa_target")

    def _record_gate_stats_on_oproj(self, o_proj: nn.Linear, gate_score: Optional[torch.Tensor], write_reg: bool = False):
        """
        Record gate stats and optionally a reg term on o_proj for trainer to collect.
        Supports both dynamic gate_score and static per-layer gates.
        """
        to_fp32 = (lambda t: t.float()) if getattr(self, "aux_fp32", False) else (lambda t: t)
        thr = float(getattr(self, "gate_sparsity_threshold", 0.5))
        reg_type = getattr(self, "reg_type", "l1")

        gate_tensor = None
        if isinstance(gate_score, torch.Tensor) and gate_score.dim() == 4:
            gate_tensor = torch.sigmoid(gate_score)
        elif getattr(self, "gate_mode", "dynamic") == "static":
            gate_logits_k = getattr(o_proj, "_kvpress_static_gate_logits_k", None)
            gate_logits_v = getattr(o_proj, "_kvpress_static_gate_logits_v", None)
            if isinstance(gate_logits_k, (torch.Tensor, nn.Parameter)):
                gk = torch.sigmoid(gate_logits_k).unsqueeze(0).unsqueeze(0)  # [1,1,kv_heads,D/1]
                gate_tensor = gk
                if isinstance(gate_logits_v, (torch.Tensor, nn.Parameter)) and gate_logits_v is not gate_logits_k:
                    gv = torch.sigmoid(gate_logits_v).unsqueeze(0).unsqueeze(0)
                    gate_tensor = 0.5 * (gk + gv)

        if gate_tensor is None:
            return

        gate_loss = to_fp32(gate_tensor)
        o_proj._kvpress_gate_sparsity = (gate_loss < thr).float().mean()
        o_proj._kvpress_gate_mean = gate_loss.mean()
        o_proj._kvpress_gate_bin = (gate_loss * (1.0 - gate_loss)).mean()
        if write_reg:
            if reg_type == "l1":
                reg = gate_loss.abs().mean()
            elif reg_type == "group_lasso":
                reg = torch.linalg.vector_norm(gate_loss, ord=2, dim=-1).mean()
            else:
                raise ValueError(f"Unknown reg_type={reg_type}")
            o_proj._kvpress_aux_reg = reg

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        # 如果 prune_in_compress=True：在 KV cache 上做“维度剪枝”（把某些 head_dim 维度置零）
        # 在 compress() 里区分 prefill vs decode，并缓存 dim_idx；把 compress() 里“选 dim_idx + 对 k_len 全长 scatter”的逻辑改成：
        # prefill（无 past）：算 dim_idx，存到 module._kvpress_prune_dim_idx，并对整个 cache 应用一次
        # decode（有 past）：直接取缓存的 dim_idx，只对 k_len 的最后一个位置 -1 应用（或者直接 return 不动）
        # 保护 retrieval heads：如果 protect_retrieval_heads=True：retrieval heads 的 keep mask 强制全 1（不剪）
        if not getattr(self, "prune_in_compress", False):
            return keys, values
        ratio = float(getattr(self, "key_channel_compression_ratio", 0.0) or 0.0)
        bsz, kv_heads, k_len, head_dim = keys.shape
        if ratio <= 0 or head_dim <= 1:
            return keys, values
        # retrieval head indices for this layer (may be absent)
        retrieval_idx = getattr(module, "_kvpress_retrieval_kv_head_idx", []) if getattr(self, "protect_retrieval_heads", False) else []
        # ---- detect decode ----
        # decode 通常 q_len == 1；prefill q_len > 1
        q_len = hidden_states.shape[1] if isinstance(hidden_states, torch.Tensor) and hidden_states.dim() == 3 else None
        is_decode = (q_len == 1)

        # ---- reuse precomputed prune idx on decode ----
        if is_decode:
            dim_idx = getattr(module, "_kvpress_prune_dim_idx", None)
            scale = getattr(module, "_kvpress_prune_scale", None)
            if dim_idx is None:
                return keys, values

            # only apply to the NEW token at position -1
            dim_idx_1 = dim_idx.unsqueeze(2)  # [B, kv_heads, 1, n_pruned]
            # ✅只scatter到最后一个位置
            # keys_last = keys[:, :, -1:, :]          # [B, kv_heads, 1, D]
            # keys_last.scatter_(-1, dim_idx_1, 0.0)
            # if self.sync_kv_prune:
            #     values_last = values[:, :, -1:, :]
            #     values_last.scatter_(-1, dim_idx_1, 0.0)
            # build keep mask then "un-prune" retrieval heads
            keep = torch.ones((bsz, kv_heads, 1, head_dim), device=keys.device, dtype=keys.dtype)
            keep.scatter_(-1, dim_idx_1, 0.0)
            if retrieval_idx:
                keep[:, retrieval_idx, :, :] = 1.0
            keys[:, :, -1:, :] = keys[:, :, -1:, :] * keep
            if self.sync_kv_prune:
                values[:, :, -1:, :] = values[:, :, -1:, :] * keep
            # scale 也只乘最后一个位置
            if scale is not None:
                keys[:, :, -1:, :] = keys[:, :, -1:, :] * scale
            return keys, values

        # ---- prefill: compute dim_idx ONCE and cache it ----
        # (你原来怎么计算 gate_kv / key_scores 就怎么来)
        raw_ws = int(getattr(self, "window_size", 32) or 32)
        seq_len = hidden_states.shape[1]
        ws = seq_len if raw_ws == -1 else max(1, min(raw_ws, seq_len))
        hs = hidden_states[:, -ws:, :]

        # infer kv heads
        _, num_kv_heads_infer, _, _ = self._infer_head_info(module)
        if num_kv_heads_infer != kv_heads:
            num_kv_heads_infer = kv_heads  # 保守对齐 keys 的形状

        if getattr(self, "gate_mode", "dynamic") == "static":
            gate_logits_k = getattr(module, "_kvpress_static_gate_logits_k", None)
            gate_kv = torch.sigmoid(gate_logits_k)
            if gate_kv.shape[1] == 1:
                gate_kv = gate_kv.expand(num_kv_heads_infer, head_dim)
            gate_kv = gate_kv.view(1, num_kv_heads_infer, head_dim).expand(bsz, -1, -1)
        else:
            gate_proj = getattr(module, "_kvpress_gate_proj", None)
            if gate_proj is None:
                return keys, values  # 或者 gate_kv = torch.ones(...)

            g = torch.sigmoid(gate_proj(hs))  # [B, ws, out_dim]

            if getattr(self, "gate_type", "elementwise") == "headwise":
                # out_dim = num_kv_heads_infer
                if g.shape[-1] != num_kv_heads_infer:
                    # 防御：shape 不对就直接不剪
                    return keys, values
                g = g.view(bsz, ws, num_kv_heads_infer, 1).expand(-1, -1, -1, head_dim)  # -> [B, ws, kv_heads, D]
            else:
                # elementwise: out_dim = num_kv_heads_infer * head_dim
                if g.shape[-1] != num_kv_heads_infer * head_dim:
                    return keys, values
                g = g.view(bsz, ws, num_kv_heads_infer, head_dim)  # -> [B, ws, kv_heads, D]

            gate_kv = g.mean(dim=1)  # [B, kv_heads, D]

        key_scores = gate_kv

        n_pruned = min(max(1, int(head_dim * ratio)), head_dim - 1)

        # compute dim_idx (pairwise or not)
        if getattr(self, "pairwise_prune", False) and head_dim % 2 == 0 and n_pruned >= 2:
            pair_scores = self._pair_scores(key_scores.unsqueeze(1)).squeeze(1)  # [B, kv_heads, D/2]  (half 配对)
            n_pairs = head_dim // 2
            n_pruned_pairs = min(max(1, n_pruned // 2), n_pairs - 1)
            # 选“要剪掉的pair”：largest=False
            pair_idx = pair_scores.topk(n_pruned_pairs, dim=-1, largest=False).indices  # [B, kv_heads, n_pruned_pairs]
            # 还原到 dim_idx (D 维)
            dim_idx = torch.stack((pair_idx, pair_idx + head_dim//2), dim=-1).reshape(bsz, kv_heads, 2*n_pruned_pairs)
            n_pruned_eff = 2 * n_pruned_pairs
        else:
            dim_idx = key_scores.topk(n_pruned, dim=-1, largest=False).indices
            n_pruned_eff = n_pruned

        # cache for decode reuse
        module._kvpress_prune_dim_idx = dim_idx

        # apply to FULL cache ONCE
        # dim_idx_full = dim_idx.unsqueeze(2).expand(-1, -1, k_len, -1)
        # keys.scatter_(-1, dim_idx_full, 0)
        # if getattr(self, "sync_kv_prune", False):
        #     values.scatter_(-1, dim_idx_full, 0)
        # apply to FULL cache ONCE (mask-based so we can un-prune retrieval heads)
        dim_idx_full = dim_idx.unsqueeze(2)  # [B, kv_heads, 1, n_pruned]
        keep = torch.ones((bsz, kv_heads, 1, head_dim), device=keys.device, dtype=keys.dtype)
        keep.scatter_(-1, dim_idx_full, 0.0)
        if retrieval_idx:
            keep[:, retrieval_idx, :, :] = 1.0
        keep_full = keep.expand(-1, -1, k_len, -1)
        keys.mul_(keep_full)
        if getattr(self, "sync_kv_prune", False):
            values.mul_(keep_full)
        # cache & apply scale ONCE
        if getattr(self, "qk_scale_correction", True) and n_pruned_eff > 0 and head_dim - n_pruned_eff > 0:
            d_eff = head_dim - n_pruned_eff
            base_scale = (head_dim / d_eff) ** 0.5
            scale = torch.full((1, kv_heads, 1, 1), float(base_scale), device=keys.device, dtype=keys.dtype)
            if retrieval_idx:
                scale[:, retrieval_idx, :, :] = 1.0
            module._kvpress_prune_scale = scale
            keys.mul_(scale)
        else:
            module._kvpress_prune_scale = None

        return keys, values



def load_model_with_gated_press(model_path, model_kwargs=None):
    """
    Load HF model + tokenizer, then load trained GatedPress gate_proj weights.
    Supports single-file and sharded .bin checkpoints.
    """
    model_kwargs = model_kwargs or {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def _infer_gate_spec(state_dict: dict) -> dict | None:
        """Extract (in_dim, out_dim, has_bias) from gate_proj tensors."""
        weight = next((v for k, v in state_dict.items() 
                      if "_kvpress_gate_proj.weight" in k and hasattr(v, "shape") and v.ndim == 2), None)
        if weight is None:
            return None
        has_bias = any("_kvpress_gate_proj.bias" in k for k in state_dict)
        return {"in_dim": weight.shape[1], "out_dim": weight.shape[0], "has_bias": has_bias}

    def _infer_static_gate_spec(state_dict: dict) -> dict | None:
        """Extract (kv_heads, gate_dim, separate_kv) from static gate logits tensors."""
        k_t = next((v for k, v in state_dict.items()
                    if k.endswith("_kvpress_static_gate_logits_k") and hasattr(v, "shape") and v.ndim == 2), None)
        if k_t is None:
            return None
        separate = any(k.endswith("_kvpress_static_gate_logits_v") for k in state_dict)
        return {"kv_heads": int(k_t.shape[0]), "gate_dim": int(k_t.shape[1]), "separate_kv": separate}

    def _create_gate_modules(spec: dict | None):
        """Ensure each layer has properly shaped gate_proj module."""
        if spec is None:
            GatedPress().post_init_from_model(model)
            return

        language_model = getattr(model.model, "language_model", model.model)
        device, dtype = next(model.parameters()).device, next(model.parameters()).dtype

        for layer in language_model.layers:
            attn = layer.self_attn
            existing = getattr(attn, "_kvpress_gate_proj", None)
            
            # Check if existing module matches spec
            if isinstance(existing, nn.Linear):
                if (existing.in_features == spec["in_dim"] and 
                    existing.out_features == spec["out_dim"] and 
                    (existing.bias is not None) == spec["has_bias"]):
                    continue
            
            # Create new module
            if "_kvpress_gate_proj" in attn._modules:
                del attn._modules["_kvpress_gate_proj"]
            proj = nn.Linear(spec["in_dim"], spec["out_dim"], bias=spec["has_bias"]).to(device=device, dtype=dtype)
            attn.register_module("_kvpress_gate_proj", proj)

    def _create_static_gate_params(spec: dict | None):
        """Ensure each layer has properly shaped static gate logits parameters."""
        if spec is None:
            return
        language_model = getattr(model.model, "language_model", model.model)
        device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
        for layer in language_model.layers:
            attn = layer.self_attn
            # create / replace parameters to match spec
            def _set_param(name: str):
                param = nn.Parameter(torch.zeros((spec["kv_heads"], spec["gate_dim"]), device=device, dtype=dtype))
                if name in attn._parameters:
                    attn._parameters[name] = param
                else:
                    attn.register_parameter(name, param)
            _set_param("_kvpress_static_gate_logits_k")
            if spec["separate_kv"]:
                _set_param("_kvpress_static_gate_logits_v")

    def _load_gate_state(state_dict: dict) -> bool:
        """Load gate weights into model, return success status."""
        gate_state = {k: v for k, v in state_dict.items() if ("_kvpress_gate_proj" in k or "_kvpress_static_gate_logits_" in k)}
        if not gate_state:
            return False
        
        incompatible = model.load_state_dict(gate_state, strict=False)
        logger.info("Loaded gate params: %d keys (missing=%d, unexpected=%d)",
                   len(gate_state), len(incompatible.missing_keys), len(incompatible.unexpected_keys))
        return True

    # Try sharded checkpoint first
    index_path = os.path.join(model_path, "pytorch_model.bin.index.json")
    if os.path.exists(index_path):
        import json
        with open(index_path) as f:
            weight_map = json.load(f).get("weight_map", {})
        
        gate_keys = {k: v for k, v in weight_map.items() if "_kvpress_gate_proj" in k}
        static_keys = {k: v for k, v in weight_map.items() if "_kvpress_static_gate_logits_" in k}
        if gate_keys or static_keys:
            # Load one shard to infer spec
            sample_file = next(iter((gate_keys or static_keys).values()))
            sample_shard = os.path.join(model_path, sample_file)
            sample_sd = torch.load(sample_shard, map_location="cpu")
            _create_gate_modules(_infer_gate_spec(sample_sd))
            _create_static_gate_params(_infer_static_gate_spec(sample_sd))
            
            # Load all gate weights from relevant shards
            gate_state = {}
            for shard_file in set(list(gate_keys.values()) + list(static_keys.values())):
                shard_path = os.path.join(model_path, shard_file)
                if os.path.exists(shard_path):
                    gate_state.update({k: v for k, v in torch.load(shard_path, map_location="cpu").items()
                                      if ("_kvpress_gate_proj" in k or "_kvpress_static_gate_logits_" in k)})
            
            if _load_gate_state(gate_state):
                print("✓ Loaded gate params from sharded checkpoint", flush=True)
                return model, tokenizer
            logger.info("Found gate keys in index but failed to load weights")

    # Fallback to single-file checkpoint
    checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if any(("_kvpress_gate_proj" in k or "_kvpress_static_gate_logits_" in k) for k in state_dict):
            _create_gate_modules(_infer_gate_spec(state_dict))
            _create_static_gate_params(_infer_static_gate_spec(state_dict))
            if _load_gate_state(state_dict):
                print("✓ Loaded gate params from single-file checkpoint", flush=True)

    return model, tokenizer