# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from kvpress.indexmem.inference.budget import mass_head_budgets


def permute_head_budgets(budgets, layer_idx):
    generator = torch.Generator(device="cpu").manual_seed(1234 + layer_idx)
    order = torch.randperm(budgets.numel(), generator=generator).to(budgets.device)
    return budgets[order]


def fit_offline_head_budgets(model, documents, *, cache_budget, sink_size=4, window_size=128, min_head_budget=512):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    layers = model.model.layers
    config = model.config
    groups = config.num_attention_heads // config.num_key_value_heads
    head_dim = config.head_dim
    captured, hidden = {}, {}

    def attention(module, query, key, value, attention_mask, scaling=None, **kwargs):
        captured[module.layer_idx] = (query.detach(), key.detach())
        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key.repeat_interleave(groups, 1),
            value.repeat_interleave(groups, 1),
            is_causal=True,
            scale=scaling,
        )
        return output.transpose(1, 2).contiguous(), None

    def capture_hidden(module, args, kwargs):
        hidden[module.layer_idx] = kwargs["hidden_states"].detach()

    name = "indexmem_offline_budget"
    registry = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
    ALL_ATTENTION_FUNCTIONS.register(name, attention)
    hooks = [layer.self_attn.register_forward_pre_hook(capture_hidden, with_kwargs=True) for layer in layers]
    previous = config._attn_implementation
    config._attn_implementation = name
    accumulated = torch.zeros(len(layers), config.num_key_value_heads, dtype=torch.float64)
    try:
        with torch.no_grad():
            for input_ids in documents:
                model.model(input_ids=input_ids, use_cache=False)
                for index, layer in enumerate(layers):
                    query, key = captured[index]
                    scorer = layer.self_attn.retention_scorer
                    scores = scorer.score_at(hidden[index], key.shape[2] - 1)[0]
                    accumulated[index] += (
                        mass_head_budgets(
                            query,
                            key,
                            scores,
                            cache_budget=cache_budget,
                            sink_size=sink_size,
                            window_size=window_size,
                            scaling=head_dim**-0.5,
                            floor=min_head_budget,
                        )
                        .double()
                        .cpu()
                    )
    finally:
        for hook in hooks:
            hook.remove()
        config._attn_implementation = previous
        registry.pop(name)
    accumulated /= len(documents)
    total = cache_budget * config.num_key_value_heads
    table = torch.zeros_like(accumulated, dtype=torch.int64)
    for index, average in enumerate(accumulated):
        ideal = average * (total / average.sum())
        budget = ideal.floor().long()
        remainder = int(total - budget.sum())
        if remainder > 0:
            budget[(ideal - budget).argsort(descending=True)[:remainder]] += 1
        elif remainder < 0:
            budget[(ideal - budget).argsort()[:-remainder]] -= 1
        table[index] = budget
    return table
