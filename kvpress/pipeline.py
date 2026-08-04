# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import contextlib
import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, Cache, DynamicCache, Pipeline, QuantizedCache
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.pipelines.base import GenericTensor

from kvpress.presses.base_press import BasePress
from kvpress.presses.decode_press import DecodePress
from kvpress.presses.decoding_press import DecodingPress
from kvpress.presses.finch_press import FinchPress
from kvpress.presses.key_rerotation_press import KeyRerotationPress
from kvpress.presses.observed_attention_press import ObservedAttentionPress
from kvpress.presses.prefill_decoding_press import PrefillDecodingPress
from kvpress.presses.gt_score_press import GTScorePress

logger = logging.getLogger(__name__)


class KVPressTextGenerationPipeline(Pipeline):
    """
    Pipeline for key-value cache compression in causal language models.

    Enables efficient processing of long contexts by applying KV cache compression
    during pre-filling, then generating answers using greedy decoding.

    Example:
    ```python
    pipeline = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer)
    press = SnapKVPress(compression_ratio=0.5)
    result = pipeline(context="Long text...", question="A question about the long context.", press=press)
    ```
    """

    def _sanitize_parameters(
        self,
        question: Optional[str] = None,
        questions: Optional[list[str]] = None,
        answer_prefix: Optional[str] = None,
        press: Optional[BasePress] = None,
        max_new_tokens: int = 50,
        max_context_length: Optional[int] = None,
        cache: Optional[Cache] = None,
        use_chunk_prefill: bool = False,
        chunk_prefill_size: Optional[int] = None,
        **kwargs,
    ):
        """
        Sanitize the input parameters for the pipeline.
        The user can either provide a single question or a list of questions to be asked about the context.

        Parameters
        ----------
        question : str, optional
            The question to be asked about the context. Exclusive with `questions`.
        questions : list[str], optional
            A list of questions to be asked about the context. Exclusive with `question`.
        answer_prefix : str, optional
            The prefix to be added to the generated answer.
        press : BasePress, optional
            The key-value cache compression method to apply during pre-filling.

            Accepts any KVPress compression method (SnapKVPress, KnormPress,
            ExpectedAttentionPress, BlockPress, AdaKVPress, ComposedPress, etc.).
            If None, no compression is applied.
        max_new_tokens : int, optional
            The maximum number of new tokens to generate for each answer.
        max_context_length : int, optional
            The maximum number of tokens in the context. By default will use the maximum length supported by the model.
        cache : Cache, optional
            The cache to use for the forward pass. Defaults to None (DynamicCache).
        **kwargs : dict
            Additional keyword arguments, currently ignored.

        Returns
        -------
        Tuple[dict, dict, dict]
            A tuple containing three dictionaries:
                - preprocess_kwargs: The keyword arguments for the preprocess function.
                - forward_kwargs: The keyword arguments for the forward function.
                - postprocess_kwargs: The keyword arguments for the postprocess function.
        """

        answer_prefix = answer_prefix or ""
        postprocess_kwargs = {"single_question": questions is None}
        assert question is None or questions is None, "Either question or questions should be provided, not both."
        questions = questions or ([question] if question else [""])
        if max_context_length is None:
            max_context_length = min(self.tokenizer.model_max_length, int(1e10))  # 1e10 to avoid overflow
        preprocess_kwargs = {
            "questions": questions,
            "answer_prefix": answer_prefix,
            "max_context_length": max_context_length,
        }
        forward_kwargs = {
            "press": press,
            "max_new_tokens": max_new_tokens,
            "cache": cache,
            "use_chunk_prefill": use_chunk_prefill,
            "chunk_prefill_size": chunk_prefill_size,
        }
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(
        self,
        context: str,
        questions: list[str],
        answer_prefix: str,
        max_context_length: int,
    ):
        """
        Apply chat template and tokenize the context and questions.

        Prepares input text for KV cache compression and generation by applying
        appropriate chat templates and tokenizing. Handles models with and without
        chat templates.

        Parameters
        ----------
        context : str
            Long context text to be compressed using the press method.
        questions : list[str]
            Questions to be asked about the context.
        answer_prefix : str
            Optional prefix for generated answers.
        max_context_length : int
            Maximum tokens allowed in context (truncated if exceeded).

        Returns
        -------
        dict[str, GenericTensor]
            Dictionary with "context_ids" and "questions_ids" tensors.
        """

        # Apply chat template if available
        if self.tokenizer.chat_template is None:
            bos_token = getattr(self.tokenizer, "bos_token", "")
            context = bos_token + context
            question_suffix = "\n"  # to separate the question from the answer
        else:
            # NOTE: `context` may be empty for some datasets (e.g. aime24 sets context="").
            # The previous separator logic (`"\n" + "#" * len(context)`) degenerates to "\n",
            # which appears many times in chat templates and breaks `split()` unpacking.
            # Use a unique sentinel and only split once to robustly recover the
            # (templated) context prefix and the question suffix.
            # Use a unique marker that survives chat-template normalization. Some
            # templates may strip/alter surrounding newlines, so avoid depending on them.
            separator = "<|KV_PRESS_SEPARATOR|>"
            context = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": context + separator}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            parts = context.split(separator, 1)
            if len(parts) != 2:
                # Fallback: if something odd happened (e.g. template transforms whitespace),
                # try splitting from the right once before giving up.
                parts = context.rsplit(separator, 1)
            if len(parts) != 2:
                raise ValueError(
                    "Failed to split chat-templated prompt into (context, question_suffix). "
                    f"Expected exactly one occurrence of separator={separator!r}, "
                    f"but found {context.count(separator)}."
                )
            context, question_suffix = parts

        # Add question_suffix and answer prefix
        # e.g. for llama3.1, question_suffix="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        questions = [question + question_suffix + answer_prefix for question in questions]

        # Tokenize the context and questions
        context_ids = self.tokenizer.encode(context, return_tensors="pt", add_special_tokens=False)
        question_ids = [
            self.tokenizer.encode(question, return_tensors="pt", add_special_tokens=False) for question in questions
        ]

        # Truncate context
        if context_ids.shape[1] > max_context_length:
            logger.warning(
                f"Context length has been truncated from {context_ids.shape[1]} to {max_context_length} tokens."
            )
            context_ids = context_ids[:, :max_context_length]

        return {"context_ids": context_ids, "questions_ids": question_ids}


    def _chunk_prefill_context(
        self,
        context_ids: torch.Tensor,
        cache: Cache,
        press: Optional[BasePress],
        chunk_prefill_size: int,
    ):
        output_attentions = self.output_attentions(press)
        num_chunks = (context_ids.shape[1] + chunk_prefill_size - 1) // chunk_prefill_size

        for chunk_idx, start in enumerate(range(0, context_ids.shape[1], chunk_prefill_size)):
            end = min(start + chunk_prefill_size, context_ids.shape[1])
            seg_ids = context_ids[:, start:end]

            # IMPORTANT:
            # After online compression, the real KV length is no longer equal to `start`.
            # We must use the ACTUAL cache length to construct a contiguous cache_position
            # that matches the physically stored KV cache.
            seg_cache_pos = torch.arange(
                start,
                end,
                device=seg_ids.device,
            )


            self.model(
                input_ids=seg_ids,
                past_key_values=cache,
                cache_position=seg_cache_pos,
                use_cache=True,
                output_attentions=output_attentions,
                kvpress_chunk_prefill=True,
                kvpress_chunk_start=start,
                kvpress_chunk_end=end,
                kvpress_chunk_idx=chunk_idx,
                kvpress_num_chunks=num_chunks,
            )
            
    def _forward(
        self,
        input_tensors: dict[str, GenericTensor],
        max_new_tokens: int = 50,
        press: Optional[BasePress] = None,
        cache: Optional[Cache] = None,
        use_chunk_prefill: bool = False,
        chunk_prefill_size: Optional[int] = None,
    ):
        """
        Execute KV cache compression and text generation pipeline.

        Performs context compression using the press method during pre-filling,
        then generates answers using greedy decoding.

        Parameters
        ----------
        input_tensors : dict[str, GenericTensor]
            Tokenized inputs with "context_ids" and "questions_ids".
        max_new_tokens : int, default=50
            Maximum tokens to generate for each answer.
        press : BasePress, optional
            Compression method for context pre-filling. If None, no compression.
        cache : Cache, optional
            Cache object for forward pass. If None, creates new DynamicCache.

        Returns
        -------
        list[str]
            Generated answers for each input question.
        """
        if isinstance(press, (DecodePress, DecodingPress, PrefillDecodingPress)) and len(input_tensors["questions_ids"]) > 1:
            raise ValueError(
                "DecodingPress is not compatible with multiple questions. Please specify a single question."
            )

        context_ids = input_tensors["context_ids"].to(self.model.device)
        context_length = context_ids.shape[1]

        # Prefilling using the press on the context
        if cache is None:
            cache = DynamicCache()

        # We only perform prefill compression if the press is a prefill press
        perform_prefill_compression = press is not None and not isinstance(press, DecodingPress)
        with press(self.model) if perform_prefill_compression else contextlib.nullcontext():
            if use_chunk_prefill and chunk_prefill_size is not None and chunk_prefill_size > 0:
                if press is not None and hasattr(press, "_reset_chunk_prefill_buffers"):
                    press._reset_chunk_prefill_buffers()

                self._chunk_prefill_context(
                    context_ids=context_ids,
                    cache=cache,
                    press=press,
                    chunk_prefill_size=int(chunk_prefill_size),
                )

                # After all chunks are processed, run deferred finalize (e.g. streaming score + block-compress).
                if press is not None and hasattr(press, "finalize_chunk_prefill"):
                    cache = press.finalize_chunk_prefill(self.model, cache)

                context_length = self._actual_cache_length(cache)
            else:
                self.model(
                    input_ids=context_ids,
                    past_key_values=cache,
                    use_cache=True,
                    output_attentions=self.output_attentions(press),
                )

            logger.debug(f"Context Length: {context_length}")
            logger.debug(f"Compressed Context Length: {self._actual_cache_length(cache)}")

        # We only perform decoding compression if the press is a decoding or prefill decoding press
        perform_decoding_compression = press is not None and isinstance(press, (DecodePress, DecodingPress, PrefillDecodingPress))
        with press(self.model) if perform_decoding_compression else contextlib.nullcontext():
            # Greedy decoding for each question
            answers = []
            for question_ids in input_tensors["questions_ids"]:
                if isinstance(press, KeyRerotationPress) or (isinstance(press, FinchPress) and press.rerotate_keys):
                    context_length = cache.get_seq_length()

                cache_seq_lengths = [cache.get_seq_length(layer_idx) for layer_idx in range(len(cache))]
                answer = self.generate_answer(
                    question_ids=question_ids.to(self.model.device),
                    cache=cache,
                    context_length=context_length,
                    max_new_tokens=max_new_tokens,
                )
                self._remove_answer_from_cache(cache, cache_seq_lengths)

                answers.append(answer)
        return answers

    def _actual_cache_length(self, cache: Cache) -> int:
        #从真实 KV 取长度，不要信 cache.get_seq_length()
        lengths = []
        for layer_idx in range(len(cache)):
            layer = cache.layers[layer_idx]
            if getattr(layer, "keys", None) is not None:
                lengths.append(layer.keys.shape[2])

        if not lengths:
            return 0

        first = lengths[0]
        for i, x in enumerate(lengths):
            if x != first:
                raise ValueError(f"Inconsistent cache lengths across layers: layer0={first}, layer{i}={x}")
        return first

    def _remove_answer_from_cache(self, cache: Cache, cache_seq_lengths: list[int]):

        for layer_idx, sequence_length in enumerate(cache_seq_lengths):
            cache.layers[layer_idx].keys = cache.layers[layer_idx].keys[:, :, :sequence_length]
            cache.layers[layer_idx].values = cache.layers[layer_idx].values[:, :, :sequence_length]

        if isinstance(cache, QuantizedCache):
            for layer_idx, sequence_length in enumerate(cache_seq_lengths):
                cache.layers[layer_idx]._quantized_keys = cache.layers[layer_idx]._quantized_keys[
                    :, :, :sequence_length
                ]
                cache.layers[layer_idx]._quantized_values = cache.layers[layer_idx]._quantized_values[
                    :, :, :sequence_length
                ]

    def generate_answer(
        self, question_ids: torch.Tensor, cache: Cache, context_length: int, max_new_tokens: int
    ) -> str:
    #后续 question 的位置和 cache 位置，必须从“压缩后真实还剩多少 token”开始算
        logger.info(f"[generate_answer] input context_length = {context_length}")
        logger.info(f"[generate_answer] actual cache length = {self._actual_cache_length(cache)}")
        real_cache_len = self._actual_cache_length(cache)
        if context_length != real_cache_len:
            logger.warning(
                f"[generate_answer] context_length ({context_length}) != actual cache length ({real_cache_len}); "
                f"using actual cache length."
            )
            context_length = real_cache_len

        question_ids = question_ids.to(self.model.device)

        # Use the REAL cache length as the starting position.
        position_ids = torch.arange(
            context_length, context_length + question_ids.shape[1], device=self.model.device
        ).unsqueeze(0)
        cache_position = position_ids.clone()

        outputs = self.model(
            input_ids=question_ids,
            past_key_values=cache,
            position_ids=position_ids,
            cache_position=cache_position,
            num_logits_to_keep=1,
        )

        next_position = torch.tensor([[context_length + question_ids.shape[1]]], device=self.model.device)
        generated_ids = [outputs.logits[0, -1].argmax()]

        should_stop_token_ids = self.model.generation_config.eos_token_id
        if not isinstance(should_stop_token_ids, list):
            should_stop_token_ids = [should_stop_token_ids]

        for _ in range(max_new_tokens - 1):
            outputs = self.model(
                input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
                past_key_values=cache,
                position_ids=next_position,
                cache_position=next_position,
            )
            new_id = outputs.logits[0, -1].argmax()
            generated_ids.append(new_id)
            if new_id.item() in should_stop_token_ids:
                break
            next_position = next_position + 1

        answer = self.tokenizer.decode(torch.stack(generated_ids), skip_special_tokens=True)
        return answer

    def output_attentions(self, press: BasePress):
        def contains(p: Optional[BasePress]) -> bool:
            if p is None:
                return False
            if isinstance(p, (ObservedAttentionPress, GTScorePress)):
                return True
            # common wrappers
            if hasattr(p, "press") and contains(getattr(p, "press")):
                return True
            if hasattr(p, "base_press") and contains(getattr(p, "base_press")):
                return True
            if hasattr(p, "presses"):
                try:
                    return any(contains(x) for x in getattr(p, "presses"))
                except Exception:
                    return False
            return False

        return contains(press)

    def postprocess(self, model_outputs, single_question):
        if single_question:
            return {"answer": model_outputs[0]}
        return {"answers": model_outputs}


PIPELINE_REGISTRY.register_pipeline(
    "kv-press-text-generation",
    pipeline_class=KVPressTextGenerationPipeline,
    pt_model=AutoModelForCausalLM,
)
