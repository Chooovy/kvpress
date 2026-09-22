# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager

import torch
from torch import nn

from kvpress.indexmem.kernels.gated_attention import gated_attention
from kvpress.indexmem.training.losses import reverse_kl_loss


class RetentionTrainer:
    def __init__(self, model, *, gate_mass=256.0, sink_size=4, window_size=128):
        self.model = model
        self.gate_mass = gate_mass
        self.sink_size = sink_size
        self.window_size = window_size
        self.hidden_states = {}

    def freeze_backbone(self):
        self.model.requires_grad_(False)
        parameters = []
        for layer in self.model.model.layers:
            scorer = layer.self_attn.retention_scorer
            scorer.gate_scale = nn.Parameter(scorer.gate_scale.detach().float())
            scorer.requires_grad_(True)
            parameters.extend(scorer.parameters())
        return parameters

    def _capture(self, module, args, kwargs):
        self.hidden_states[module.layer_idx] = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]

    def _attention(self, module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
        scorer = module.retention_scorer
        hidden = self.hidden_states.pop(module.layer_idx)
        query_kwargs = (
            {"query_offset": key.shape[2] - query.shape[2], "n_kv_heads": key.shape[1]} if scorer.decay else {}
        )
        gate_queries = scorer.project_q(hidden, None, None, **query_kwargs)
        gate_keys = scorer.project_k(hidden, None, None, value_states=value)
        output = gated_attention(
            query,
            key,
            value,
            gate_queries,
            gate_keys,
            scorer.require_gate_scale(),
            scaling=scaling,
            gate_mass=self.gate_mass,
            sink_size=self.sink_size,
            window_size=self.window_size,
        )
        return output.transpose(1, 2).contiguous(), None

    @contextmanager
    def hooks(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        name = f"indexmem_training_{id(self)}"
        registry = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
        previous = self.model.config._attn_implementation
        ALL_ATTENTION_FUNCTIONS.register(name, self._attention)
        handles = [
            layer.self_attn.register_forward_pre_hook(self._capture, with_kwargs=True)
            for layer in self.model.model.layers
        ]
        self.model.config._attn_implementation = name
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()
            self.model.config._attn_implementation = previous
            del registry[name]
            self.hidden_states.clear()

    def loss(
        self, input_ids, *, objective="reverse_kl", teacher_hidden=None, weights=None, ce_weight=0.0, chunk_size=2048
    ):
        if objective in ("reverse_kl", "forward_kl") and teacher_hidden is None:
            with torch.no_grad():
                teacher_hidden = self.model.model(input_ids=input_ids, use_cache=False).last_hidden_state
        with self.hooks():
            student_hidden = self.model.model(input_ids=input_ids, use_cache=False).last_hidden_state
        if objective == "reverse_kl" and ce_weight == 0.0:
            return reverse_kl_loss(student_hidden, teacher_hidden, self.model.lm_head, chunk_size=chunk_size)
        from kvpress.indexmem.ablations.objectives import objective_loss

        return objective_loss(
            student_hidden,
            input_ids,
            self.model.lm_head,
            objective=objective,
            teacher_hidden=teacher_hidden,
            weights=weights,
            ce_weight=ce_weight,
            chunk_size=chunk_size,
        )
