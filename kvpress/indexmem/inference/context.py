# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for the IndexMem++ public implementation.

from __future__ import annotations

import torch
from transformers import DynamicCache

from kvpress.indexmem.config import IndexMemConfig
from kvpress.indexmem.inference.cmp import CentroidMemory, cluster_evicted, evicted_from_deadline
from kvpress.indexmem.inference.generation import sample_token
from kvpress.indexmem.inference.paged_cache import PagedKVCache, PagedKVPool
from kvpress.indexmem.inference.prefill import PrefillContext
from kvpress.indexmem.kernels.prefill_attention import deadlines


def split_router_key(k_idx: torch.Tensor, decay: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    if decay:
        score_intercept = k_idx[..., 0::2].transpose(1, 2).contiguous().float()
        log_retention_rate = k_idx[..., 1::2].transpose(1, 2).contiguous().float()
        return (score_intercept, log_retention_rate)
    return (k_idx.transpose(1, 2).contiguous().float(), None)


def shift_router_key(
    score_intercept: torch.Tensor,
    log_retention_rate: torch.Tensor | None,
    *,
    offset: torch.Tensor | float,
    pos_slope: float,
    age_scale: float | None,
) -> torch.Tensor:
    off = torch.as_tensor(offset, dtype=torch.float32, device=score_intercept.device)
    if off.dim() == 2:
        off = off.unsqueeze(1)
    while off.dim() < score_intercept.dim():
        off = off.unsqueeze(0)
    shifted = score_intercept + off * pos_slope
    if log_retention_rate is not None:
        shifted = shifted - log_retention_rate * (off / float(age_scale))
    return shifted


class IndexMemInferenceContext:

    def __init__(self, model, config: IndexMemConfig, batch_size=None):
        self.model = model
        self.config = config
        self.batch_size = config.decode_batch if batch_size is None else batch_size
        self.sink_size = config.sink_size
        self.window_size = config.window_size
        self.cmp_slots = config.cmp_slots
        self.n_kv_heads = model.config.num_key_value_heads
        self.head_dim = model.config.head_dim
        scorer = model.model.layers[0].self_attn.retention_scorer
        self._decay = scorer.decay
        self._pos_slope = scorer.pos_slope
        self.pool = None
        self.cmp = None
        self._hidden = {}
        self._pending = {}
        self._active_seqs = None

    def _capture_hook(self, module, args, kwargs):
        self._hidden[module.layer_idx] = kwargs["hidden_states"]

    def _router_state(self, module, layer_idx, *, key_offset):
        scorer = module.retention_scorer
        packed = scorer.project_k(self._hidden[layer_idx], None, None, value_states=None, key_offset=0)
        score_intercept, log_retention_rate = split_router_key(packed, self._decay)
        score_intercept = shift_router_key(
            score_intercept,
            log_retention_rate,
            offset=key_offset,
            pos_slope=self._pos_slope,
            age_scale=self.pool.age_scale,
        )
        return (score_intercept, log_retention_rate)

    def _ensure_pool(self, prefill):
        if self.pool is not None:
            return
        layers = self.model.model.layers
        budgets = (
            torch.stack(
                [
                    (
                        prefill.head_budget_table[i].to("cpu", torch.int64)
                        if prefill.head_budget == "static"
                        else (
                            prefill._head_budgets[i].to("cpu", torch.int64)
                            if i in prefill._head_budgets
                            else torch.full((self.n_kv_heads,), prefill.cache_budget, dtype=torch.int64)
                        )
                    )
                    for i in range(len(layers))
                ]
            )
            - self.cmp_slots
        )
        parameter = next(self.model.parameters())
        scorer = layers[0].self_attn.retention_scorer
        self.pool = PagedKVPool(
            budgets,
            batch_size=self.batch_size,
            n_layers=len(layers),
            n_kv_heads=self.n_kv_heads,
            n_sink=self.sink_size,
            n_local=self.window_size,
            head_dim=self.head_dim,
            device=parameter.device,
            dtype=parameter.dtype,
            age_scale=scorer.age_scale if self._decay else None,
        )
        if self.cmp_slots:
            self.cmp = CentroidMemory(self.pool.rows, self.cmp_slots, self.head_dim, device=parameter.device)

    @torch.no_grad()
    def prefill_and_commit(self, input_ids, seq=0):
        cache = DynamicCache()
        layers = self.model.model.layers
        with PrefillContext(self.model, self.config, build_cmp=False) as prefill:
            prefill.set_context_length(input_ids.shape[1])

            def release_inputs(module, args, output):
                self._hidden.pop(module.layer_idx, None)
                prefill._hidden_states.pop(module.layer_idx, None)

            handles = [layer.self_attn.register_forward_hook(release_inputs) for layer in layers]
            try:
                self.model.model(input_ids=input_ids, past_key_values=cache)
            finally:
                for handle in handles:
                    handle.remove()
            self._ensure_pool(prefill)
            length = input_ids.shape[1]
            for layer_idx in range(len(layers)):
                key = cache.layers[layer_idx].keys[0]
                value = cache.layers[layer_idx].values[0]
                score_intercept, log_retention_rate = split_router_key(prefill._k_idx[layer_idx], self._decay)
                self.pool.commit(
                    layer_idx,
                    seq,
                    key=key,
                    value=value,
                    score_intercept=score_intercept[0],
                    log_retention_rate=None if log_retention_rate is None else log_retention_rate[0],
                    k_len=length,
                )
                if self.cmp is not None:
                    self._seed_cmp(layer_idx, seq, key, value, score_intercept[0], log_retention_rate, length)
        self.pool.seen[seq] = length

    def _seed_cmp(self, layer_idx, seq, key, value, score_intercept, log_retention_rate, length):
        rows = self.pool.rows_for(layer_idx, seq)
        scores = (
            score_intercept
            if log_retention_rate is None
            else score_intercept + log_retention_rate[0] * ((length - 1) / self.pool.age_scale)
        )
        deadline = deadlines(scores, self.pool.row_budget[rows], sink_size=self.sink_size, window_size=self.window_size)
        evicted = evicted_from_deadline(deadline, max(length - 1 - self.window_size, 0))
        if not bool(evicted.any()):
            return
        keys, values, log_mass = cluster_evicted(key.float(), value.float(), evicted, self.cmp_slots)
        padding = self.cmp_slots - keys.shape[1]
        if padding:
            keys = torch.nn.functional.pad(keys, (0, 0, 0, padding))
            values = torch.nn.functional.pad(values, (0, 0, 0, padding))
            log_mass = torch.nn.functional.pad(log_mass, (0, padding), value=-float("inf"))
        self.cmp.load_batch(rows, keys, values, log_mass)

    def __enter__(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        def attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
            return (self._attend(module, query, key, value, scaling), None)

        self._previous_impl = self.model.config._attn_implementation
        self._registry = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
        self._had_entry = "indexmem_decode" in self._registry
        self._previous_entry = self._registry.get("indexmem_decode")
        ALL_ATTENTION_FUNCTIONS.register("indexmem_decode", attention)
        self._handles = [
            layer.self_attn.register_forward_pre_hook(self._capture_hook, with_kwargs=True)
            for layer in self.model.model.layers
        ]
        return self

    def activate(self):
        self.model.config._attn_implementation = "indexmem_decode"

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self._handles:
            handle.remove()
        self.model.config._attn_implementation = self._previous_impl
        if self._had_entry:
            self._registry["indexmem_decode"] = self._previous_entry
        else:
            del self._registry["indexmem_decode"]
        self._hidden.clear()
        self._pending.clear()
        self.pool = None
        self.cmp = None

    def _attend(self, module, query, key, value, scaling):
        layer_idx = int(module.layer_idx)
        bsz, _, q_len, _ = query.shape
        seqs = self._active_seqs
        if seqs is None:
            seqs = torch.arange(bsz, device=self.pool.device)
        seen = self.pool.seen[seqs]
        base = seen.view(-1, 1) + torch.arange(q_len, device=seen.device).view(1, -1)
        score_intercept, log_retention_rate = self._router_state(module, layer_idx, key_offset=base)
        self._pending[layer_idx] = (key, value, score_intercept, log_retention_rate)
        extra = None
        if self.cmp is not None:
            cmp_rows = (
                layer_idx * self.batch_size * self.n_kv_heads
                + seqs.view(-1, 1) * self.n_kv_heads
                + torch.arange(self.n_kv_heads, device=self.pool.device).view(1, -1)
            ).reshape(-1)
            extra = self.cmp.read(
                cmp_rows,
                query,
                group=query.shape[1] // self.n_kv_heads,
                scaling=scaling if scaling is not None else query.shape[-1] ** (-0.5),
            )
        return self.pool.attend(layer_idx, query, seqs=seqs, scaling=scaling, new_key=key, new_value=value, extra=extra)

    def finish_step(self, seqs: torch.Tensor | None = None) -> None:
        if not self._pending:
            return
        layers = sorted(self._pending)
        keys = torch.stack([self._pending[i][0] for i in layers])
        values = torch.stack([self._pending[i][1] for i in layers])
        score_intercepts = torch.stack([self._pending[i][2] for i in layers])
        log_retention_rates = (
            torch.stack([self._pending[i][3] for i in layers]) if self._pending[layers[0]][3] is not None else None
        )
        self._pending.clear()
        q_len = keys.shape[3]
        if seqs is None:
            seqs = torch.arange(self.batch_size, device=self.pool.device)
        seqs = seqs.to(self.pool.device)
        seen = self.pool.seen[seqs].clone()
        for t in range(q_len):
            dropped = self.pool.ingest(
                layers,
                key=keys[:, :, :, t, :],
                value=values[:, :, :, t, :],
                score_intercept=score_intercepts[:, :, :, t],
                log_retention_rate=None if log_retention_rates is None else log_retention_rates[:, :, :, t],
                positions=seen + t,
                seqs=seqs,
            )
            if self.cmp is not None and dropped is not None:
                self.cmp.ingest(dropped["rows"], dropped["key"], dropped["value"], active=dropped["evicted"])
        self.pool.seen[seqs] = seen + q_len

    def replicate(self, src: int, dst: int) -> None:
        self.pool.replicate_seq(src, dst)
        if self.cmp is not None:
            for layer_idx in range(self.pool.n_layers):
                s = self.pool.rows_for(layer_idx, src)
                d = self.pool.rows_for(layer_idx, dst)
                self.cmp.centroid_keys[d] = self.cmp.centroid_keys[s]
                self.cmp.centroid_values[d] = self.cmp.centroid_values[s]
                self.cmp.cluster_counts[d] = self.cmp.cluster_counts[s]

    def new_cache(self) -> PagedKVCache:
        return PagedKVCache(self.pool)

    @torch.no_grad()
    def generate(
        self,
        question_ids: list[torch.Tensor],
        *,
        max_new_tokens: int,
        eos_token_ids: list[int] | None = None,
        sampling: dict | None = None,
    ) -> list[list[int]]:
        batch = len(question_ids)
        if eos_token_ids is None:
            eos = self.model.generation_config.eos_token_id
            eos_token_ids = eos if isinstance(eos, list) else [eos]
        device = self.pool.device
        self.activate()
        cache = self.new_cache()
        first = []
        for seq, q_ids in enumerate(question_ids):
            q_ids = q_ids.to(device)
            start = int(self.pool.seen[seq])
            pos = torch.arange(start, start + q_ids.shape[1], device=device).unsqueeze(0)
            only = torch.tensor([seq], device=device)
            self._active_seqs = only
            try:
                logits = self.model(
                    input_ids=q_ids, past_key_values=cache, position_ids=pos, num_logits_to_keep=1
                ).logits
                self.finish_step(seqs=only)
            finally:
                self._active_seqs = None
            first.append(int(sample_token(logits[:, -1], sampling)[0]))
        out: list[list[int]] = [[t] for t in first]
        done = [t in eos_token_ids for t in first]
        nxt = torch.tensor(first, device=device).view(batch, 1)
        for _ in range(max_new_tokens - 1):
            if all(done):
                break
            pos = self.pool.seen[:batch].view(batch, 1)
            logits = self.model(input_ids=nxt, past_key_values=cache, position_ids=pos).logits
            self.finish_step()
            nxt = sample_token(logits[:, -1], sampling).view(batch, 1)
            for b in range(batch):
                if done[b]:
                    continue
                token = int(nxt[b])
                out[b].append(token)
                if token in eos_token_ids:
                    done[b] = True
        return out
