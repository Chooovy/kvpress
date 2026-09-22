# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for the IndexMem++ public implementation.

from __future__ import annotations

import torch

from kvpress.indexmem.config import IndexMemConfig
from kvpress.indexmem.inference.budget import mass_head_budgets
from kvpress.indexmem.inference.cmp import cluster_evicted, evicted_from_deadline
from kvpress.indexmem.inference.selection import streaming_topk_support
from kvpress.indexmem.kernels.prefill_attention import deadlines, qi_sparse_attention


class PrefillContext:

    def __init__(self, model, config: IndexMemConfig, *, build_cmp=True):
        self.model = model
        self.cache_budget = config.cache_budget
        self.sink_size = config.sink_size
        self.window_size = config.window_size
        self.cache_budget_ratio = config.cache_budget_ratio
        self.head_budget = config.head_budget
        self.head_budget_table = (
            torch.load(config.head_budget_table, map_location="cpu", weights_only=True)["table"]
            if config.head_budget == "static"
            else None
        )
        self.min_head_budget = config.min_head_budget
        self.cmp_slots = config.cmp_slots if build_cmp else 0
        self.block_k = 64
        self.precision = "tf32"
        self.causal = True
        self._decay_active = model.model.layers[0].self_attn.retention_scorer.decay
        self._hidden_states = {}
        self._k_idx = {}
        self._head_budgets = {}
        self._cmp = {}
        self._cmp_at = {}
        self._cmp_r = {}

    def _capture_hook(self, module, args, kwargs):
        self._hidden_states[module.layer_idx] = kwargs["hidden_states"]

    def _budget_for(self, module, query, key, router_scores, scaling, layer_idx):
        if self.head_budget == "uniform":
            return self.cache_budget
        if layer_idx in self._head_budgets:
            return self._head_budgets[layer_idx]
        if self.head_budget == "static":
            self._head_budgets[layer_idx] = self.head_budget_table[layer_idx].to(key.device)
            return self._head_budgets[layer_idx]
        if query.shape[2] <= 1 or key.shape[2] <= self.cache_budget:
            return self.cache_budget
        budgets = mass_head_budgets(
            query,
            key,
            router_scores[0],
            cache_budget=self.cache_budget,
            sink_size=self.sink_size,
            window_size=self.window_size,
            scaling=scaling,
            floor=self.min_head_budget,
        )
        if self.head_budget == "shuffle":
            from kvpress.indexmem.ablations.budget import permute_head_budgets

            budgets = permute_head_budgets(budgets, layer_idx)
        self._head_budgets[layer_idx] = budgets
        return budgets

    @torch.no_grad()
    def _build_cmp(self, module, query, key, value, q_idx, k_idx, k_len, scaling):
        layer_idx = module.layer_idx
        scores = self._cmp_scores(q_idx, k_idx)
        budget = self._cmp_take(module, query, key, scaling, layer_idx, scores.unsqueeze(0))
        deadline = deadlines(scores, budget, sink_size=self.sink_size, window_size=self.window_size)
        evicted = evicted_from_deadline(deadline, max(k_len - 1 - self.window_size, 0))
        self._cmp_at[layer_idx] = k_len
        num_slots = self._cmp_r.get(layer_idx, self.cmp_slots)
        if not bool(evicted.any()) or num_slots == 0:
            return
        self._cmp[layer_idx] = cluster_evicted(key[0].float(), value[0].float(), evicted, num_slots)

    def __enter__(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        def attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
            return (self._attend(module, query, key, value, scaling), None)

        self._previous_impl = self.model.config._attn_implementation
        self._registry = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
        self._previous_entry = self._registry.get("indexmem_prefill")
        self._had_entry = "indexmem_prefill" in self._registry
        ALL_ATTENTION_FUNCTIONS.register("indexmem_prefill", attention)
        self._handles = [
            layer.self_attn.register_forward_pre_hook(self._capture_hook, with_kwargs=True)
            for layer in self.model.model.layers
        ]
        self.model.config._attn_implementation = "indexmem_prefill"
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self._handles:
            handle.remove()
        self.model.config._attn_implementation = self._previous_impl
        if self._had_entry:
            self._registry["indexmem_prefill"] = self._previous_entry
        else:
            del self._registry["indexmem_prefill"]
        self._hidden_states.clear()
        self._k_idx.clear()

    def _attend(self, module, query, key, value, scaling):
        from kvpress.indexmem.kernels.sparse_attention import sparse_gqa_attention

        layer_idx = int(module.layer_idx)
        hidden_states = self._hidden_states[layer_idx]
        scorer = module.retention_scorer
        cos, sin = (None, None)
        previous = self._k_idx.get(layer_idx)
        previous_len = 0 if previous is None else previous.shape[1]
        q_kwargs = {}
        if self._decay_active:
            q_kwargs["query_offset"] = previous_len
            q_kwargs["n_kv_heads"] = key.shape[1]
        q_idx = scorer.project_q(hidden_states, cos, sin, **q_kwargs)
        value_states_new = value[:, :, previous_len:, :]
        k_idx_new = scorer.project_k(
            hidden_states,
            cos,
            sin,
            value_states=value_states_new,
            **{"key_offset": previous_len} if self._decay_active else {},
        )
        k_idx = k_idx_new if previous is None else torch.cat([previous, k_idx_new], dim=1)
        self._k_idx[layer_idx] = k_idx
        k_len = key.shape[2]
        if k_len < self.sink_size + self.window_size:
            return self._attend_dense(query, key, value, scaling)
        if self.cmp_slots:
            built_at = self._cmp_at.get(layer_idx)
            if built_at is None:
                self._build_cmp(module, query, key, value, q_idx, k_idx, k_len, scaling)
            elif k_len - q_idx.shape[2] >= built_at and layer_idx in self._cmp:
                return self._attend_with_cmp(module, query, key, value, scaling, q_idx, k_idx, k_len)
        if q_idx.shape[2] > 1:
            ref_row = q_idx.shape[2] // 2 if self._decay_active else 0
            router_scores = torch.einsum("bhqd,bkd->bhk", q_idx[:, :, ref_row : ref_row + 1], k_idx)
            out = qi_sparse_attention(
                query,
                key,
                value,
                router_scores,
                self._budget_for(module, query, key, router_scores, scaling, layer_idx),
                sink_size=self.sink_size,
                window_size=self.window_size,
                scaling=scaling,
            )
            return out.transpose(1, 2).contiguous()
        gather_scores = None
        if self.head_budget in ("mass", "shuffle") and q_idx.shape[2] > 1 and (layer_idx not in self._head_budgets):
            mid = q_idx.shape[2] // 2
            gather_scores = torch.einsum("bhqd,bkd->bhk", q_idx[:, :, mid : mid + 1], k_idx)
        budget = self._budget_for(module, query, key, gather_scores, scaling, layer_idx)
        if not torch.is_tensor(budget):
            support, _ = streaming_topk_support(
                q_idx, k_idx, self.cache_budget, mask=None, sink_size=self.sink_size, window_size=self.window_size
            )
        else:
            widest = int(budget.max())
            parts = []
            for h in range(q_idx.shape[1]):
                one, _ = streaming_topk_support(
                    q_idx[:, h : h + 1],
                    k_idx,
                    int(budget[h]),
                    mask=None,
                    sink_size=self.sink_size,
                    window_size=self.window_size,
                )
                if one.shape[-1] < widest:
                    pad = one.new_full((*one.shape[:-1], widest - one.shape[-1]), -1)
                    one = torch.cat([one, pad], dim=-1)
                parts.append(one)
            support = torch.cat(parts, dim=1)
        out, _ = sparse_gqa_attention(
            query,
            key,
            value,
            support,
            scaling=scaling,
            causal=self.causal,
            block_k=self.block_k,
            precision=self.precision,
        )
        return out.transpose(1, 2).contiguous()

    def set_context_length(self, context_length: int) -> int:
        if self.cache_budget_ratio is None:
            return self.cache_budget
        import math

        floor = self.sink_size + self.window_size + 1
        resolved = max(int(math.ceil(self.cache_budget_ratio * int(context_length))), floor)
        if resolved != self.cache_budget:
            self._head_budgets.clear()
            self._cmp.clear()
            self._cmp_at.clear()
        self.cache_budget = resolved
        return resolved

    def _cmp_take(self, module, query, key, scaling, layer_idx, router_scores):
        budget = self._budget_for(module, query, key, router_scores, scaling, layer_idx)
        if not self.cmp_slots:
            return budget
        evictable = (
            (int(budget.min()) if isinstance(budget, torch.Tensor) else int(budget)) - self.sink_size - self.window_size
        )
        if self.cmp_slots < evictable:
            return budget - self.cmp_slots
        usable = max(0, min(self.cmp_slots, (evictable - 1) // 2))
        self._cmp_r[layer_idx] = usable
        return budget - usable if usable else budget

    def _cmp_scores(self, q_idx, k_idx):
        ref = q_idx.shape[2] // 2 if self._decay_active else 0
        return torch.einsum("bhqd,bkd->bhk", q_idx[:, :, ref : ref + 1], k_idx)[0].float()

    def _attend_with_cmp(self, module, query, key, value, scaling, q_idx, k_idx, k_len) -> torch.Tensor:
        from kvpress.indexmem.inference.attention import fuse_attention
        from kvpress.indexmem.kernels.prefill_attention import _flex, qi_block_mask

        layer_idx = int(module.layer_idx)
        centroid_keys, centroid_values, log_cluster_mass = self._cmp[layer_idx]
        bsz, n_q_heads, q_len, head_dim = query.shape
        n_kv = key.shape[1]
        group = n_q_heads // n_kv
        scores = self._cmp_scores(q_idx, k_idx)
        dl = deadlines(
            scores,
            self._cmp_take(module, query, key, scaling, layer_idx, scores.unsqueeze(0)),
            sink_size=self.sink_size,
            window_size=self.window_size,
        )
        block_mask = qi_block_mask(
            dl,
            q_len=q_len,
            k_len=k_len,
            n_q_heads=n_q_heads,
            sink_size=self.sink_size,
            window_size=self.window_size,
            device=query.device,
        )
        o_s, lse_s = _flex()(
            query,
            key.repeat_interleave(group, dim=1),
            value.repeat_interleave(group, dim=1),
            block_mask=block_mask,
            scale=scaling,
            return_lse=True,
        )
        kc = centroid_keys.to(query.device).float().repeat_interleave(group, 0)
        vc = centroid_values.to(query.device).float().repeat_interleave(group, 0)
        bc = log_cluster_mass.to(query.device).float().repeat_interleave(group, 0)
        l_cmp = torch.einsum("bhqd,hrd->bhqr", query.float(), kc) * scaling + bc.view(1, n_q_heads, 1, -1)
        w = l_cmp.exp()
        n = torch.einsum("bhqr,hrd->bhqd", w, vc)
        d = w.sum(-1)
        out = fuse_attention(o_s, lse_s, n, d, group=1)
        return out.transpose(1, 2).contiguous()

    def _attend_dense(self, query, key, value, scaling) -> torch.Tensor:
        group = query.shape[1] // key.shape[1]
        q_len, k_len = (query.shape[2], key.shape[2])
        key_idx = torch.arange(k_len, device=query.device)
        q_pos = torch.arange(q_len, device=query.device).unsqueeze(-1) + (k_len - q_len)
        attn_mask = key_idx <= q_pos
        out = torch.nn.functional.scaled_dot_product_attention(
            query,
            key.repeat_interleave(group, dim=1),
            value.repeat_interleave(group, dim=1),
            attn_mask=attn_mask,
            scale=scaling,
        )
        return out.transpose(1, 2).contiguous()
