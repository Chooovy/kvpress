# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from benchmarks.needle_in_haystack.utils import insert_needle_in_haystack
from datasets import get_dataset_config_names, load_dataset
from evaluate_registry import DATASET_REGISTRY, PRESS_REGISTRY, SCORER_REGISTRY
from fire import Fire
from tqdm import tqdm
from transformers import Pipeline, pipeline

from kvpress import (
    CacheIndexerDecodingPress,
    ComposedPress,
    DecodePress,
    DecodingPress,
    DuoAttentionPress,
    FinchPress,
    ObservedAttentionPress,
    ScorerPress,
    SelectiveDecodingPress,
    ThinKPress,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Dataclass to handle all the configuration for the evaluation."""

    # Core evaluation parameters
    dataset: str = "ruler"
    data_dir: Optional[str] = None
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device: Optional[Any] = None
    press_name: str = "knorm"
    compression_ratio: float = 1.0
    key_channel_compression_ratio: Optional[float] = None
    samples: Optional[int] = None

    # Dataset and generation parameters
    fraction: float = 1.0
    max_new_tokens: Optional[int] = None
    max_context_length: Optional[int] = None
    compress_questions: bool = False
    needle_depth: Optional[int] = None

    # Decoding parameters
    compression_interval: Optional[int] = None
    target_size: Optional[int] = None
    hidden_states_buffer_size: Optional[int] = None

    # Output and logging
    output_dir: str = "./results"
    log_level: str = "INFO"

    # Model-specific parameters
    model_kwargs: Optional[Dict[str, Any]] = None

    # Press information (will be set after press setup)
    press_init_command: Optional[str] = None

    # For reproducibility
    seed: int = 42
    # --- GT score recording (eager attentions) ---
    gt_mode: bool = False
    gt_out_dir: Optional[str] = None
    gt_max_steps: Optional[int] = None

    def __post_init__(self):
        # Validate dataset
        assert self.dataset in DATASET_REGISTRY, f"No dataset found for {self.dataset}"
        assert self.dataset in SCORER_REGISTRY, f"No scorer found for {self.dataset}"

        # Validate press
        assert self.press_name in PRESS_REGISTRY, f"Press '{self.press_name}' not found in PRESS_REGISTRY"

        if self.press_name == "no_press":
            logger.info("Using 'no_press' configuration. Overriding compression_ratio to 0.0")
            self.compression_ratio = 0.0

        assert 0.0 <= self.compression_ratio <= 1.0, (
            f"compression_ratio must be between 0.0 and 1.0, got {self.compression_ratio}"
        )

        if self.key_channel_compression_ratio is not None:
            assert 0.0 <= self.key_channel_compression_ratio <= 1.0, (
                f"key_channel_compression_ratio must be between 0.0 and 1.0, got {self.key_channel_compression_ratio}"
            )

        assert 0.0 < self.fraction <= 1.0, f"fraction must be between 0.0 and 1.0, got {self.fraction}"

        if self.model_kwargs is None:
            self.model_kwargs = {}

        if self.dataset == "needle_in_haystack":
            assert self.needle_depth is not None, "needle_depth must be set for needle_in_haystack"
            assert self.max_context_length is not None, "max_context_length must be set for needle_in_haystack"

    def get_results_dir(self, output_dir: Path) -> Path:
        model_name = self.model.split("/")[-1] if "/" in self.model else self.model

        components = [self.dataset]
        if self.data_dir and str(self.data_dir) not in ["", "None", "True", "False"]:
            components.append(str(self.data_dir))

        components.extend([model_name, self.press_name, f"cr{self.compression_ratio:.2f}"])

        if self.fraction < 1.0:
            components.append(f"fraction{self.fraction:.3f}")
        if self.max_context_length is not None:
            components.append(f"max_context{self.max_context_length}")
        if self.compress_questions:
            components.append("compressed_questions")
        if self.key_channel_compression_ratio is not None:
            components.append(f"key_channel_cr{self.key_channel_compression_ratio:.2f}")
        if self.needle_depth is not None and self.dataset == "needle_in_haystack":
            components.append(f"needle_depth{self.needle_depth}")

        dir_name = "__".join(filter(None, components))
        config_dir = output_dir / dir_name

        if config_dir.exists():
            i = 1
            while (config_dir / f"{i}").exists():
                i += 1
            config_dir = config_dir / f"{i}"

        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir

    def save_config(self, config_filename: Path):
        with open(str(config_filename), "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, indent=2, sort_keys=False)


def _load_yaml_config(path: str | Path) -> dict:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"Config file not found at {path}. Using only command-line arguments and defaults.")
        return {}


class EvaluationRunner:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.pipeline: Optional[Pipeline] = None
        self.press: None | ScorerPress = None
        self.df: Optional[pd.DataFrame] = None

        self._setup_logging()
        self._setup_deterministic_seeds()
        logger.info(f"Initialized EvaluationRunner with config:\n{json.dumps(asdict(self.config), indent=2)}")

    def _setup_deterministic_seeds(self):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Set deterministic seeds to {self.config.seed}")

    def _setup_logging(self):
        log_level = self.config.log_level.upper()

        # 避免重复 add handler（多次初始化会刷屏）
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(handler)
        logger.setLevel(log_level)

    def _setup_directories(self) -> Path:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set to: {output_dir}")
        return output_dir

    def _setup_press(self):
        press_name = self.config.press_name
        compression_ratio = self.config.compression_ratio
        key_channel_compression_ratio = self.config.key_channel_compression_ratio

        press = PRESS_REGISTRY[press_name]

        if isinstance(press, DuoAttentionPress):
            press.head_compression_ratio = compression_ratio
            logger.info(f"Set DuoAttentionPress head_compression_ratio to {compression_ratio}")

        elif isinstance(press, ComposedPress):
            for ps in press.presses:
                if isinstance(ps, ThinKPress):
                    assert key_channel_compression_ratio is not None, (
                        "key_channel_compression_ratio must be set for ThinKPress in ComposedPress"
                    )
                    ps.key_channel_compression_ratio = key_channel_compression_ratio
                    logger.info(f"Set ThinKPress key_channel_compression_ratio to {key_channel_compression_ratio}")
                else:
                    if hasattr(ps, "compression_ratio"):
                        ps.compression_ratio = compression_ratio
                        logger.info(f"Set ComposedPress component compression_ratio to {compression_ratio}")
                    else:
                        logger.warning(
                            f"ComposedPress component {ps.__class__.__name__} has no 'compression_ratio' attribute."
                        )

        elif isinstance(press, ThinKPress):
            assert key_channel_compression_ratio is not None, "key_channel_compression_ratio must be set for ThinKPress"
            press.key_channel_compression_ratio = key_channel_compression_ratio
            logger.info(f"Set ThinKPress key_channel_compression_ratio to {key_channel_compression_ratio}")

        elif isinstance(press, DecodePress):
            press.compression_interval = self.config.compression_interval or press.compression_interval
            press.target_size = self.config.target_size or press.target_size
            press.hidden_states_buffer_size = self.config.hidden_states_buffer_size or press.hidden_states_buffer_size
            logger.info(
                "Set DecodePress: "
                f"compression_interval={press.compression_interval}, "
                f"target_size={press.target_size}, "
                f"hidden_states_buffer_size={press.hidden_states_buffer_size}"
            )

        else:
            if hasattr(press, "compression_ratio"):
                press.compression_ratio = compression_ratio
                logger.info(f"Set {press.__class__.__name__} compression_ratio to {compression_ratio}")
            else:
                logger.warning(
                    f"Press {press.__class__.__name__} has no 'compression_ratio' attribute. "
                    "This is expected if you use `no_press`."
                )
        # --- NEW: apply key_channel_compression_ratio to presses that support it (e.g., GatedPress) ---
        if key_channel_compression_ratio is not None and hasattr(press, "key_channel_compression_ratio"):
            press.key_channel_compression_ratio = key_channel_compression_ratio
            logger.info(f"Set {press.__class__.__name__} key_channel_compression_ratio to {key_channel_compression_ratio}")
        self.press = press
        self.config.press_init_command = str(press)
        logger.info(f"KV Press '{press_name}' setup.")

    def _load_and_prepare_dataset(self):
        dataset_name = self.config.dataset
        data_dir = str(self.config.data_dir) if self.config.data_dir else None
        fraction = self.config.fraction

        path = DATASET_REGISTRY[dataset_name]
        logger.info(f"Loading dataset: {path} (data_dir: {data_dir})")

        load_kwargs = {"split": "test"}
        # 注意：LongBench 走 hub configs；data_dir 对 hub 通常没意义，还可能造成 TypeError
        if data_dir and dataset_name not in ["math500", "aime25", "longbench", "zero_scrolls"]:
            load_kwargs["data_dir"] = data_dir
        ZERO_SCROLLS_CFGS = [
            "gov_report","summ_screen_fd","qmsum","squality","qasper",
            "narrative_qa","quality","musique","space_digest","book_sum_sort",
        ]

        if dataset_name == "zero_scrolls":
            dfs = []
            for cfg in ZERO_SCROLLS_CFGS:
                tmp = load_dataset("tau/zero_scrolls", cfg, split="test", trust_remote_code=True).to_pandas()

                # ✅ 跟 verifier 对齐：按 input 去重
                tmp = tmp.drop_duplicates(subset=["input"]).reset_index(drop=True)

                tmp["task"] = cfg

                tmp["context"] = tmp.apply(
                    lambda r: r["input"][int(r["document_start_index"]): int(r["document_end_index"])],
                    axis=1,
                )

                def _get_q(r):
                    qs, qe = r["query_start_index"], r["query_end_index"]
                    if pd.isna(qs) or pd.isna(qe):
                        return ""
                    qs, qe = int(qs), int(qe)
                    if qs < 0 or qe < 0 or qe <= qs:
                        return ""
                    return r["input"][qs:qe]

                tmp["question"] = tmp.apply(_get_q, axis=1)

                # ✅ 让 _run_inference 不炸
                tmp["answer_prefix"] = ""  # 或 "\nAnswer:\n"
                tmp["max_new_tokens"] = self.config.max_new_tokens or 256

                dfs.append(tmp)

            df = pd.concat(dfs, ignore_index=True)

        elif dataset_name == "longbench":
            cfgs = get_dataset_config_names(path)
            dfs = []
            for cfg in cfgs:
                logger.info(f"  -> config: {cfg}")
                tmp = load_dataset(path, cfg, split="test").to_pandas()
                tmp["config"] = cfg
                dfs.append(tmp)
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = load_dataset(path, **load_kwargs).to_pandas()

        if dataset_name == "math500" and self.config.samples is not None:
            original_len = len(df)
            df = df.head(self.config.samples) if len(df) >= self.config.samples else df
            logger.info(f"Limited math500 dataset to {len(df)} samples from original {original_len} samples.")

        if fraction < 1.0:
            original_len = len(df)
            df = df.sample(frac=fraction, random_state=self.config.seed)
            logger.info(f"Sampled {len(df)} samples ({fraction:.2f}) from original {original_len} samples.")

        logger.info(f"Dataset loaded with {len(df)} entries.")

        if self.config.dataset == "needle_in_haystack":
            df = insert_needle_in_haystack(
                df, self.pipeline.tokenizer, self.config.max_context_length, self.config.needle_depth
            )

        if isinstance(self.press, FinchPress):
            if not self.config.compress_questions:
                logger.error("FinchPress requires 'compress_questions' to be set to True.")
                raise ValueError("FinchPress requires compress_questions to be set to True")
            logger.info("FinchPress detected, updating model and tokenizer with delimiter token.")
            self.press.update_model_and_tokenizer(self.pipeline.model, self.pipeline.tokenizer)  # type: ignore[attr-defined]
            df["context"] = df["context"] + self.press.delimiter_token  # type: ignore[attr-defined, index]

        if self.config.compress_questions:
            logger.info("Compressing questions into context.")
            df["context"] = df["context"] + df["question"]  # type: ignore[index]
            df["question"] = ""  # type: ignore[index]

        self.df = df
        logger.info(f"Dataset processed with {len(self.df)} entries.")

    def _setup_model_pipeline(self):
        from kvpress.presses.dma_score_press import DMAScorePress, load_model_with_dma_press
        from kvpress.presses.indexer_score_press import IndexerScorePress, load_model_with_indexer_press
        from kvpress.presses.indexer_score_press_cache import CacheIndexerScorePress, load_model_with_indexer_press_cache
        from kvpress.presses.indexer_score_query_press import QueryIndexerScorePress, load_model_with_query_indexer_press
        from kvpress.presses.memory_scorer_press import MemoryScorerPress, load_model_with_memory_params
        from kvpress.pipeline import KVPressTextGenerationPipeline

        model_name = self.config.model
        device = self.config.device

        if device is None:
            device = "auto" if torch.cuda.is_available() else "cpu"
            logger.info(f"No device specified, auto-detected device: {device}")
        else:
            # Fire/cli 可能传入 "0" 这种字符串
            if isinstance(device, str) and device.isdigit():
                device = int(device)

        model_kwargs = dict(self.config.model_kwargs or {})

        actual_press = self.press
        if isinstance(self.press, DecodePress):
            actual_press = self.press.base_press

        use_dma_press = isinstance(actual_press, DMAScorePress)
        use_indexer_press = isinstance(actual_press, IndexerScorePress)
        use_indexer_press_cache = isinstance(actual_press, CacheIndexerScorePress)
        use_query_indexer_press = isinstance(actual_press, QueryIndexerScorePress)
        use_memory_press = isinstance(actual_press, MemoryScorerPress)

        # ---- MUST: GT forces eager regardless of press ----
        if getattr(self.config, "gt_mode", False):
            model_kwargs["attn_implementation"] = "eager"
            logger.info("[GT] gt_mode=True, forcing attn_implementation='eager'.")
        elif isinstance(self.press, ObservedAttentionPress):
            model_kwargs["attn_implementation"] = "eager"
            logger.info("ObservedAttentionPress detected, setting attn_implementation to 'eager'.")
        else:
            try:
                import flash_attn  # noqa: F401
                model_kwargs["attn_implementation"] = "flash_attention_2"
                model_kwargs.setdefault("dtype", torch.bfloat16)
                logger.info("Flash Attention 2 detected, using 'flash_attention_2' and default dtype=bf16.")
            except ImportError:
                logger.info("Flash Attention 2 not available, using default attn_implementation.")

        logger.info(f"Loading model pipeline for: {model_name} on device: {device} with model_kwargs: {model_kwargs}")

        if use_dma_press or use_indexer_press or use_indexer_press_cache or use_query_indexer_press or use_memory_press:
            model_kwargs["device_map"] = device if device != "auto" else "auto"
            load_func = (
                load_model_with_dma_press
                if use_dma_press
                else load_model_with_indexer_press
                if use_indexer_press
                else load_model_with_indexer_press_cache
                if use_indexer_press_cache
                else load_model_with_query_indexer_press
                if use_query_indexer_press
                else load_model_with_memory_params
                if use_memory_press
                else None
            )
            assert load_func is not None, "No load function found for the press"
            model, tokenizer = load_func(model_name, model_kwargs=model_kwargs)
            self.pipeline = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer)
        else:
            pipeline_kwargs = {"model": model_name, "model_kwargs": model_kwargs, "trust_remote_code": True}
            if device == "auto":
                pipeline_kwargs["device_map"] = "auto"
            else:
                pipeline_kwargs["device"] = device
            self.pipeline = pipeline("kv-press-text-generation", **pipeline_kwargs)

        self.pipeline.model.eval()
        logger.info("Model pipeline loaded.")

    def _reset_indexer_cache(self):
        from kvpress.presses.indexer_score_press import IndexerScorePress
        from kvpress.presses.indexer_score_press_cache import CacheIndexerScorePress
        from kvpress.presses.indexer_score_query_press import QueryIndexerScorePress

        actual_press = self.press
        if isinstance(self.press, DecodePress):
            actual_press = self.press.base_press

        if isinstance(actual_press, (IndexerScorePress, CacheIndexerScorePress, QueryIndexerScorePress)):
            language_model = (
                self.pipeline.model.model.language_model
                if hasattr(self.pipeline.model.model, "language_model")
                else self.pipeline.model.model
            )
            for layer in language_model.layers:
                if hasattr(layer.self_attn, actual_press.scorer_attr):
                    indexer = getattr(layer.self_attn, actual_press.scorer_attr)
                    if hasattr(indexer, "reset_cache") and callable(getattr(indexer, "reset_cache")):
                        indexer.reset_cache()
    @torch.inference_mode()
    def _run_gt_record(self, results_dir: Path):
        assert self.df is not None
        assert self.pipeline is not None

        model = self.pipeline.model
        tok = self.pipeline.tokenizer
        model.eval()

        from transformers import DynamicCache

        out_root = Path(self.config.gt_out_dir) if self.config.gt_out_dir else (results_dir / "gt_tokens")
        out_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"[GT] Saving gt_score_mean to {out_root}")

        df_context_grouped = self.df.groupby("context", sort=False)

        # ---- IMPORTANT: eager prefill 会 OOM，必须分块 ----
        PREFILL_CHUNK = 256  # 不稳就调小：128 / 64；H800 可以先 256

        sample_idx = 0
        for context, df_group in tqdm(df_context_grouped, total=self.df["context"].nunique(), desc="[GT] contexts"):
            sample_idx += 1

            question = df_group["question"].iloc[0] if "question" in df_group.columns else ""
            answer_prefix = df_group["answer_prefix"].iloc[0] if "answer_prefix" in df_group.columns else ""
            max_new = self.config.gt_max_steps or self.config.max_new_tokens or int(df_group["max_new_tokens"].iloc[0])

            prep = self.pipeline.preprocess(
                context=context,
                questions=[question],
                answer_prefix=answer_prefix,
                max_context_length=self.config.max_context_length or min(tok.model_max_length, int(1e10)),
            )

            context_ids = prep["context_ids"].to(model.device)        # [1, ctx_len]
            question_ids = prep["questions_ids"][0].to(model.device)  # [1, q_len]
            ctx_len = int(context_ids.shape[1])
            q_len = int(question_ids.shape[1])

            # ---- (1) Prefill: build KV cache by chunks (EAGER SAFE) ----
            cache = DynamicCache()

            # 用 model.model 更轻（LlamaModel），但如果你这边 wrapper 没有 model.model，再 fallback 用 model(...)
            core = model.model if hasattr(model, "model") else model

            for s in range(0, ctx_len, PREFILL_CHUNK):
                ids = context_ids[:, s : s + PREFILL_CHUNK]  # [1, chunk]
                pos = torch.arange(s, s + ids.shape[1], device=model.device).unsqueeze(0)

                _ = core(
                    input_ids=ids,
                    past_key_values=cache,
                    position_ids=pos,
                    use_cache=True,
                    output_attentions=False,
                    return_dict=True,
                )

            # ---- (2) step0: question forward (need attentions) ----
            pos_q = torch.arange(ctx_len, ctx_len + q_len, device=model.device).unsqueeze(0)
            out_q = model(
                input_ids=question_ids,
                past_key_values=cache,
                position_ids=pos_q,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
            if out_q.attentions is None:
                raise RuntimeError("[GT] No attentions returned. (You said eager, so it must return.)")

            for layer_idx, a in enumerate(out_q.attentions):
                # a: [B, H, q_len, ctx_len+q_len] 取最后一个 query token 对 context 的注意力
                a_ctx = a[0, :, -1, :ctx_len]      # [H, ctx_len]
                score_mean = a_ctx.mean(dim=0)     # [ctx_len]

                layer_dir = out_root / f"layer_{layer_idx}"
                layer_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "sample_idx": sample_idx,
                        "step_idx": 0,
                        "ctx_len": ctx_len,
                        "gt_score_mean": score_mean.detach().to(torch.float16).cpu(),
                    },
                    layer_dir / f"gt_sample{sample_idx}_step0.pt",
                )

            # ---- (3) decode steps: each step is 1 token (attentions small, safe) ----
            next_token = torch.argmax(out_q.logits[:, -1, :], dim=-1, keepdim=True)
            base_pos = pos_q[:, -1:] + 1

            eos_ids = model.generation_config.eos_token_id
            if not isinstance(eos_ids, list):
                eos_ids = [eos_ids]

            for step in range(1, max_new):
                out = model(
                    input_ids=next_token,
                    past_key_values=cache,
                    position_ids=base_pos + (step - 1),
                    use_cache=True,
                    output_attentions=True,
                    return_dict=True,
                )
                if out.attentions is None:
                    raise RuntimeError("[GT] No attentions returned in decode. (eager should return.)")

                for layer_idx, a in enumerate(out.attentions):
                    # a: [B, H, 1, ctx_len+...] 取这 1 个 query token 对 context 的注意力
                    a_ctx = a[0, :, 0, :ctx_len]      # [H, ctx_len]
                    score_mean = a_ctx.mean(dim=0)

                    layer_dir = out_root / f"layer_{layer_idx}"
                    layer_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "sample_idx": sample_idx,
                            "step_idx": step,
                            "ctx_len": ctx_len,
                            "gt_score_mean": score_mean.detach().to(torch.float16).cpu(),
                        },
                        layer_dir / f"gt_sample{sample_idx}_step{step}.pt",
                    )

                next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                if next_token.item() in eos_ids:
                    break

            logger.info(f"[GT] sample {sample_idx} done. ctx_len={ctx_len}")

            # 防碎片
            del cache, out_q
            torch.cuda.empty_cache()

        logger.info("[GT] Done saving gt_score_mean.")

    @torch.inference_mode()
    def _run_inference(self):
        self.df["predicted_answer"] = None  # type: ignore[index]

        record_response_length = self.config.dataset in ["math500", "aime25"]
        if record_response_length:
            self.df["response_length"] = None  # type: ignore[index]

        if isinstance(self.press, DecodePress):
            logger.info("DecodePress detected, running inference for each context-question pair.")
            for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Running Inference"):
                if hasattr(self.press, "reset") and callable(getattr(self.press, "reset")):
                    self.press.reset()
                self._reset_indexer_cache()

                context = row["context"]
                question = row["question"]
                answer_prefix = row["answer_prefix"]
                max_new_tokens = self.config.max_new_tokens or row["max_new_tokens"]

                output = self.pipeline(
                    context,
                    question=question,
                    answer_prefix=answer_prefix,
                    press=self.press,
                    max_new_tokens=max_new_tokens,
                    max_context_length=self.config.max_context_length,
                )
                answer = output["answer"]  # type: ignore[union-attr]
                self.df.loc[index, "predicted_answer"] = answer  # type: ignore[union-attr]

                if record_response_length:
                    response_tokens = self.pipeline.tokenizer.encode(answer, add_special_tokens=False)
                    self.df.loc[index, "response_length"] = len(response_tokens)  # type: ignore[union-attr]

                torch.cuda.empty_cache()

        else:
            # LongBench：按 (config, context) 分组，避免不同子任务混在一起
            if self.config.dataset == "longbench" and "config" in self.df.columns:  # type: ignore[operator]
                df_context_grouped = self.df.groupby(["config", "context"])  # type: ignore[union-attr]
                total_groups = self.df[["config", "context"]].drop_duplicates().shape[0]  # type: ignore[union-attr]
                logger.info("LongBench detected: grouping by (config, context).")
            else:
                df_context_grouped = self.df.groupby("context")  # type: ignore[union-attr]
                total_groups = self.df["context"].nunique()  # type: ignore[union-attr]

            assert all(
                df_context_grouped["answer_prefix"].nunique() == 1
            ), "Inconsistent 'answer_prefix' within the same context group detected."

            logger.info("Starting inference...")
            for key, df_group in tqdm(df_context_grouped, total=total_groups, desc="Running Inference"):
                if isinstance(key, tuple):
                    _, context = key
                else:
                    context = key

                if hasattr(self.press, "_reset_cache") and callable(getattr(self.press, "_reset_cache")):
                    self.press._reset_cache()
                self._reset_indexer_cache()

                questions = df_group["question"].to_list()
                max_new_tokens = self.config.max_new_tokens or df_group["max_new_tokens"].iloc[0]
                answer_prefix = df_group["answer_prefix"].iloc[0]

                output = self.pipeline(  # type: ignore[misc]
                    context,
                    questions=questions,
                    answer_prefix=answer_prefix,
                    press=self.press,
                    max_new_tokens=max_new_tokens,
                    max_context_length=self.config.max_context_length,
                )
                answers = output["answers"]  # type: ignore[union-attr]

                self.df.loc[df_group.index, "predicted_answer"] = answers  # type: ignore[union-attr]

                if record_response_length:
                    response_lengths = [
                        len(self.pipeline.tokenizer.encode(a, add_special_tokens=False)) for a in answers
                    ]
                    self.df.loc[df_group.index, "response_length"] = response_lengths  # type: ignore[union-attr]

                self.df.loc[df_group.index, "compression_ratio"] = (
                    self.press.compression_ratio if self.press is not None else 0.0  # type: ignore[attr-defined]
                )  # type: ignore[union-attr, attr-defined]

                torch.cuda.empty_cache()

        logger.info("Inference completed.")

    def _save_results(self, save_filename: Path):
        if save_filename.exists():
            logger.warning(f"Results CSV already exists at {save_filename}. Overwriting.")

        self.df[list(set(self.df.columns) - set(["context"]))].to_csv(str(save_filename), index=False)  # type: ignore[index]
        logger.info(f"Results saved to {save_filename}")

    def _calculate_and_save_metrics(self, save_filename: Path):
        dataset_name = self.config.dataset
        scorer = SCORER_REGISTRY[dataset_name]

        logger.info(f"Calculating metrics for dataset: {dataset_name}")
        metrics = scorer(self.df)  # type: ignore[call-arg]

        with open(str(save_filename), "w") as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Metrics saved to {save_filename}")
        logger.info(f"Metrics:\n{json.dumps(metrics, indent=2)}")

    def run_evaluation(self):
        logger.info("Starting evaluation run...")
        output_dir = self._setup_directories()

        results_dir = self.config.get_results_dir(output_dir)
        predictions_filename = results_dir / "predictions.csv"
        metrics_filename = results_dir / "metrics.json"
        config_filename = results_dir / "config.yaml"

        if predictions_filename.exists() and metrics_filename.exists():
            logger.info(
                f"Evaluation files already exist at \n {predictions_filename} \n {metrics_filename}.\nSkipping..."
            )
            return

        self._setup_press()
        self._setup_model_pipeline()
        self._load_and_prepare_dataset()
        # ---- NEW: GT-only path ----
        if getattr(self.config, "gt_mode", False):
            self._run_gt_record(results_dir)
            self.config.save_config(config_filename)
            logger.info("GT run completed successfully.")
            return
        self._run_inference()
        self._save_results(predictions_filename)
        self._calculate_and_save_metrics(metrics_filename)
        self.config.save_config(config_filename)
        logger.info("Evaluation run completed successfully.")


class CliEntryPoint:
    def __call__(self, config_file: Optional[str] = "./evaluate_config.yaml", **cli_overrides):
        final_args = asdict(EvaluationConfig())
        yaml_config = _load_yaml_config(config_file)
        final_args.update(yaml_config)

        cli_args = {k: v for k, v in cli_overrides.items() if v is not None}
        final_args.update(cli_args)

        try:
            config = EvaluationConfig(**final_args)
        except TypeError as e:
            print(f"Error: Invalid configuration argument provided. {e}", file=sys.stderr)
            sys.exit(1)

        runner = EvaluationRunner(config)
        runner.run_evaluation()


if __name__ == "__main__":
    Fire(CliEntryPoint)
