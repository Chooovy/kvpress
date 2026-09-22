# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for the IndexMem++ public implementation.

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, DynamicCache
from transformers.pipelines import PIPELINE_REGISTRY

from kvpress.indexmem.checkpoint import load_scorer_checkpoint
from kvpress.indexmem.config import IndexMemConfig
from kvpress.indexmem.inference.context import IndexMemInferenceContext
from kvpress.indexmem.inference.prefill import PrefillContext
from kvpress.pipeline import KVPressTextGenerationPipeline


class IndexMemTextGenerationPipeline(KVPressTextGenerationPipeline):
    def __init__(self, model, tokenizer, scorer_checkpoint, config: IndexMemConfig, **kwargs):
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)
        load_scorer_checkpoint(model, scorer_checkpoint)
        self.indexmem_config = config
        self.sampling = None

    def _forward(self, input_tensors, max_new_tokens=50, press=None, cache=None):
        context_ids = input_tensors["context_ids"].to(self.model.device)
        questions = list(input_tensors["questions_ids"])
        config = self.indexmem_config
        answers = []
        if config.inference_mode == "mask":
            for question in questions:
                with PrefillContext(self.model, config) as context:
                    context.set_context_length(context_ids.shape[1])
                    cache = DynamicCache()
                    self.model.model(input_ids=context_ids, past_key_values=cache)
                    answers.append(
                        self.generate_answer(
                            question_ids=question.to(self.model.device),
                            cache=cache,
                            context_length=context_ids.shape[1],
                            max_new_tokens=max_new_tokens,
                        )
                    )
            return answers
        for start in range(0, len(questions), config.decode_batch):
            group = questions[start : start + config.decode_batch]
            with IndexMemInferenceContext(self.model, config, batch_size=len(group)) as context:
                context.prefill_and_commit(context_ids)
                for sequence in range(1, len(group)):
                    context.replicate(0, sequence)
                tokens = context.generate(
                    [question.to(self.model.device) for question in group],
                    max_new_tokens=max_new_tokens,
                    sampling=self.sampling,
                )
            answers.extend(
                str(self.tokenizer.decode(torch.tensor(sequence), skip_special_tokens=True)) for sequence in tokens
            )
        return answers


PIPELINE_REGISTRY.register_pipeline(
    "indexmem-text-generation",
    pipeline_class=IndexMemTextGenerationPipeline,
    pt_model=AutoModelForCausalLM,
)
