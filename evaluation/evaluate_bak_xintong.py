# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
import yaml
from benchmarks.needle_in_haystack.utils import insert_needle_in_haystack
from datasets import load_dataset
from evaluate_registry import DATASET_REGISTRY, PRESS_REGISTRY, SCORER_REGISTRY
from fire import Fire
from tqdm import tqdm
from transformers import Pipeline, pipeline

from kvpress import (
    ComposedPress,
    DecodingPress,
    DuoAttentionPress,
    FinchPress,
    ObservedAttentionPress,
    ScorerPress,
    ThinKPress,
    SelectiveDecodingPress,
    CacheIndexerDecodingPress,
    DecodePress,
    QueryIndexerScorePress,
    MemoryScorerPress
)
from kvpress.presses.gt_score_press import GTScorePress

logger = logging.getLogger(__name__)


def _coerce_scalar(value: str) -> Any:
    """Best-effort coercion for CLI-provided strings (null/bool/int/float)."""
    v = value.strip()
    if v == "":
        return ""
    low = v.lower()
    if low in {"null", "none"}:
        return None
    if low in {"true", "false"}:
        return low == "true"
    # Strip simple quotes
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1]
        low = v.lower()
        if low in {"null", "none"}:
            return None
        if low in {"true", "false"}:
            return low == "true"
    try:
        return int(v)
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return v


def _parse_model_kwargs(raw: Union[None, Dict[str, Any], str]) -> Dict[str, Any]:
    """
    Normalize model_kwargs from YAML/CLI into a dict.

    Accepted forms:
    - dict: returned as-is
    - None: {}
    - str:
      - JSON object string: '{"dtype":"auto","attn_implementation":"eager"}'
      - key=value pairs (comma/space separated): 'attn_implementation=eager dtype=auto'
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise TypeError(f"model_kwargs must be a dict, a string, or None; got {type(raw)}")

    s = raw.strip()
    if s == "":
        return {}

    # JSON dict string support
    if s.startswith("{") and s.endswith("}"):
        parsed = json.loads(s)
        if not isinstance(parsed, dict):
            raise ValueError(f"model_kwargs JSON must decode to an object/dict, got {type(parsed)}")
        return parsed

    # key=value pairs (space or comma separated)
    tokens = [t for t in s.replace(",", " ").split() if t]
    out: Dict[str, Any] = {}
    for tok in tokens:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        k = k.strip()
        if not k:
            continue
        out[k] = _coerce_scalar(v)
    return out


@dataclass
class EvaluationConfig:
    """Dataclass to handle all the configuration for the evaluation."""

    # Core evaluation parameters
    dataset: str = "ruler"
    data_dir: Optional[str] = None
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device: Optional[str] = None
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
    # For datasets with multiple splits/tasks (e.g., InfiniteBench)
    task: Optional[str] = None
    # Hard filter for too-long samples (token-based)
    filter_max_tokens: Optional[int] = 131072   # 128k = 131072 tokens
    filter_on: str = "context+question"         # or "context"

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate dataset
        assert self.dataset in DATASET_REGISTRY, f"No dataset found for {self.dataset}"
        assert self.dataset in SCORER_REGISTRY, f"No scorer found for {self.dataset}"

        # Validate press
        assert self.press_name in PRESS_REGISTRY, f"Press '{self.press_name}' not found in PRESS_REGISTRY"

        if self.press_name == "no_press":
            # override compression_ratio to 0.0
            logger.info("Using 'no_press' configuration. Overriding compression_ratio to 0.0")
            self.compression_ratio = 0.0

        # Validate compression ratios
        assert (
            0.0 <= self.compression_ratio <= 1.0
        ), f"compression_ratio must be between 0.0 and 1.0, got {self.compression_ratio}"

        # Only validate key_channel_compression_ratio if it's not None
        if self.key_channel_compression_ratio is not None:
            assert (
                0.0 <= self.key_channel_compression_ratio <= 1.0
            ), f"key_channel_compression_ratio must be between 0.0 and 1.0, got {self.key_channel_compression_ratio}"

        # Validate fraction
        assert 0.0 < self.fraction <= 1.0, f"fraction must be between 0.0 and 1.0, got {self.fraction}"

        # Normalize model_kwargs (python-fire can pass it as a raw string)
        self.model_kwargs = _parse_model_kwargs(self.model_kwargs)

        if self.dataset == "needle_in_haystack":
            assert self.needle_depth is not None, "needle_depth must be set for needle_in_haystack"
            assert self.max_context_length is not None, "max_context_length must be set for needle_in_haystack"

    def get_results_dir(self, output_dir: Path) -> Path:
        model_name = self.model.split("/")[-1] if "/" in self.model else self.model

        def _valid(x: Optional[str]) -> Optional[str]:
            if x is None:
                return None
            s = str(x).strip()
            return s if s and s not in {"None", "True", "False"} else None

        # one tag for "subtask"
        task_tag = _valid(self.task) if self.dataset == "infinitebench" else _valid(self.data_dir)

        components = [self.dataset]
        if task_tag:
            components.append(task_tag)

        components.extend([model_name, self.press_name, f"cr{self.compression_ratio:.2f}"])

        if self.fraction < 1.0:
            components.append(f"fraction{self.fraction:.3f}")
        if self.max_context_length is not None:
            components.append(f"max_context{self.max_context_length}")
        if self.compress_questions:
            components.append("compressed_questions")
        if self.key_channel_compression_ratio is not None:
            components.append(f"key_channel_cr{self.key_channel_compression_ratio:.2f}")
        if self.dataset == "needle_in_haystack" and self.needle_depth is not None:
            components.append(f"needle_depth{self.needle_depth}")

        dir_name = "__".join(components)
        config_dir = output_dir / dir_name

        if config_dir.exists():
            i = 1
            while (config_dir / f"{i}").exists():
                i += 1
            config_dir = config_dir / f"{i}"

        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir

    def save_config(self, config_filename: Path):
        """
        Saves the evaluation configuration to a YAML file.
        """
        with open(str(config_filename), "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, indent=2, sort_keys=False)


def _load_yaml_config(path: str | Path) -> dict:
    """Loads a YAML file. Returns an empty dict if it doesn't exist."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"Config file not found at {path}. Using only command-line arguments and defaults.")
        return {}


class EvaluationRunner:
    """
    EvaluationRunner class that orchestrates the entire evaluation process.

    Parameters
    ----------
    config : EvaluationConfig
        The configuration for the evaluation run.

    The final output will be predictions_<config>.csv and metrics_<config>.json in the output_dir.
    If the evaluation files already exist, evaluation will be skipped.

    """

    def __init__(self, config: EvaluationConfig):
        """
        Initializes the EvaluationRunner with a given configuration.

        Parameters
        ----------
        config : EvaluationConfig
            The configuration for the evaluation run.
        """
        self.config = config
        self.pipeline: Optional[Pipeline] = None  # Will be set by _setup_model_pipeline()
        self.press: None | ScorerPress = None  # Will be set by _setup_press()
        self.df: Optional[pd.DataFrame] = None  # Will be set by _load_dataset()
        self._setup_logging()
        self._setup_deterministic_seeds()
        logger.info(f"Initialized EvaluationRunner with config:\n{json.dumps(asdict(self.config), indent=2)}")

    def _setup_deterministic_seeds(self):
        """Set deterministic seeds for reproducible results."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Set deterministic seeds to {self.config.seed}")

    # def _setup_logging(self):
    #     """Configures the logging level based on the config."""
    #     log_level = self.config.log_level.upper()

    #     handler = logging.StreamHandler()
    #     handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    #     logger.addHandler(handler)
    #     logger.setLevel(log_level)
    #改成 root logger
    def _setup_logging(self):
        log_level = self.config.log_level.upper()
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,
        )

    def _setup_directories(self) -> Path:
        """
        Creates the output directory for saving results if it doesn't exist.

        Returns
        -------
        Path
            The path to the output directory.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set to: {output_dir}")
        return output_dir

    def _setup_press(self):
        """
        Initializes the KVPress instance and applies compression ratios based on its type.
        """
        from kvpress.presses.gated_press import GatedPress

        press_name = self.config.press_name
        compression_ratio = self.config.compression_ratio
        key_channel_compression_ratio = self.config.key_channel_compression_ratio

        press = PRESS_REGISTRY[press_name]

        # Apply compression ratios based on press type
        if isinstance(press, DuoAttentionPress):
            press.head_compression_ratio = compression_ratio
            logger.info(f"Set DuoAttentionPress head_compression_ratio to {compression_ratio}")
        elif isinstance(press, ComposedPress):
            for ps in press.presses:
                if isinstance(ps, ThinKPress):
                    assert (
                        key_channel_compression_ratio is not None
                    ), "key_channel_compression_ratio must be set for ThinKPress in ComposedPress"
                    ps.key_channel_compression_ratio = key_channel_compression_ratio
                    logger.info(f"Set ComposedPress key_channel_compression_ratio to {key_channel_compression_ratio}")
                else:
                    # Check if compression_ratio attribute exists before setting
                    if hasattr(ps, "compression_ratio"):
                        ps.compression_ratio = compression_ratio
                        logger.info(f"Set ComposedPress compression_ratio to {compression_ratio}")
                    else:
                        logger.warning(
                            f"ComposedPress component {ps.__class__.__name__} has no 'compression_ratio' attribute."
                        )
        elif isinstance(press, ThinKPress):
            assert key_channel_compression_ratio is not None, "key_channel_compression_ratio must be set for ThinKPress"
            press.key_channel_compression_ratio = key_channel_compression_ratio
            logger.info(f"Set ThinKPress key_channel_compression_ratio to {key_channel_compression_ratio}")
        elif isinstance(press, GatedPress):
            # GatedPress uses `key_channel_compression_ratio` for channel pruning (ThinK-style).
            # Do NOT map `compression_ratio` into it; eval scripts often pass both.
            if key_channel_compression_ratio is not None:
                press.key_channel_compression_ratio = key_channel_compression_ratio
                logger.info(f"Set GatedPress key_channel_compression_ratio to {key_channel_compression_ratio}")
            else:
                logger.info("GatedPress detected but key_channel_compression_ratio is None; no channel pruning will be applied.")
            press.debug = True
            # 只看少数层，避免日志爆炸
            press.debug_layers = [0, 1, 2]
        elif isinstance(press, DecodePress):
            press.compression_interval = self.config.compression_interval or press.compression_interval
            press.target_size = self.config.target_size or press.target_size
            press.hidden_states_buffer_size = self.config.hidden_states_buffer_size or press.hidden_states_buffer_size
            logger.info(
                f"Set DecodingPress compression_interval to {self.config.compression_interval}, target_size to {self.config.target_size}, hidden_states_buffer_size to {self.config.hidden_states_buffer_size}"
            )
        else:
            if hasattr(press, "compression_ratio"):
                press.compression_ratio = compression_ratio
                logger.info(f"Set {press.__class__.__name__} compression_ratio to {compression_ratio}")
            else:
                logger.warning(
                    f"Press {press.__class__.__name__} has no 'compression_ratio' attribute. This is expected is you set `no_press`."
                )

        self.press = press
        # Set the press info in the config for saving to YAML
        self.config.press_init_command = str(press)
        logger.info(f"KV Press '{press_name}' setup.")

    def _load_and_prepare_dataset(self):
        """
        Loads the dataset specified in the config and applies sampling/filtering.
        """
        dataset_name = self.config.dataset
        data_dir = str(self.config.data_dir) if self.config.data_dir else None
        fraction = self.config.fraction

        logger.info(f"Loading dataset: {DATASET_REGISTRY[dataset_name]} (data_dir: {data_dir})")
        
        load_kwargs = {"split": "test"}
        if dataset_name == "aime24":
            load_kwargs["split"] = "train"
        if data_dir and dataset_name not in ["math500", "aime25", "aime24", "infinitebench"]:
            load_kwargs["data_dir"] = data_dir

        try:
            if dataset_name == "infinitebench":
                task = self.config.task or data_dir
                if task is None:
                    raise ValueError("InfiniteBench requires --task or --data_dir")
                self.config.task = task

                # ✅ 多-config 数据集：task 就是 config name
                ds = load_dataset(DATASET_REGISTRY[dataset_name], task, split="test")
            else:
                ds = load_dataset(DATASET_REGISTRY[dataset_name], **load_kwargs)

            # ---- 下面保持你原来的 token 长度过滤逻辑 ----
            if dataset_name == "infinitebench":
                tok = self.pipeline.tokenizer
                max_tok = int(self.config.filter_max_tokens or 131072)

                def _fmt_opts(opts):
                    if not isinstance(opts, (list, tuple)) or len(opts) == 0:
                        return ""
                    lines = [f"{i+1}. {str(o)}" for i, o in enumerate(opts)]
                    return "\nOptions:\n" + "\n".join(lines)

                def _add_len(ex):
                    ctx = str(ex.get("context", ""))
                    q = str(ex.get("question", ex.get("input", "")))
                    q = q + _fmt_opts(ex.get("options", []))
                    ap = str(ex.get("answer_prefix", "Answer (only the final answer): "))
                    n = len(tok.encode(ctx + q + ap, add_special_tokens=False))
                    return {"__prompt_len": n}

                before = len(ds)
                ds = ds.map(_add_len)
                ds = ds.filter(lambda ex: ex["__prompt_len"] <= max_tok)
                after = len(ds)
                ds = ds.remove_columns(["__prompt_len"])
                logger.info(f"[InfiniteBench] filtered {before-after}/{before} samples > {max_tok} tokens")

            df = ds.to_pandas()
            
        except Exception:
            # Fallback for aime24: sometimes data_dir might be passed incorrectly or structure changed
            if dataset_name == "aime24":
                # Try loading without split first or with default args
                df = load_dataset(DATASET_REGISTRY[dataset_name], split="train").to_pandas()
            else:
                raise

        if dataset_name == "aime24":
            if "problem" in df.columns and "question" not in df.columns:
                df = df.rename(columns={"problem": "question"})
            
            if "context" not in df.columns:
                df["context"] = ""
            if "answer_prefix" not in df.columns:
                df["answer_prefix"] = "Please output your final answer within \\\\boxed{}." 
            if "max_new_tokens" not in df.columns:
                df["max_new_tokens"] = 32768
    
        if dataset_name == "math500":
            original_len = len(df)
            df = df.head(self.config.samples) if len(df) >= self.config.samples else df
            logger.info(f"Limited math500 dataset to {len(df)} samples from original {original_len} samples.")
        # ---------------- InfiniteBench schema normalization ----------------
        if dataset_name == "infinitebench":
            # Rename HF "input" -> our internal "question"
            if "question" not in df.columns and "input" in df.columns:
                df = df.rename(columns={"input": "question"})

            # Build answer_prefix + max_new_tokens if missing
            if "answer_prefix" not in df.columns:
                # keep it simple and evaluation-friendly
                df["answer_prefix"] = "Answer (only the final answer): "

            if "max_new_tokens" not in df.columns:
                # per-task defaults (user can override with --max_new_tokens)
                task2max = {
                    "passkey": 32,
                    "number_string": 32,
                    "kv_retrieval": 128,
                    "code_run": 256,
                    "code_debug": 32,
                    "math_find": 64,
                    # Math.Calc output is extremely long in the benchmark table
                    # Set high by default; you may want to override for practicality.
                    "math_calc": 60000,
                    "longdialogue_qa_eng": 64,
                    "longbook_sum_eng": 2048,
                    "longbook_qa_eng": 256,
                    "longbook_choice_eng": 64,
                    "longbook_qa_chn": 256,
                }
                df["max_new_tokens"] = task2max.get(self.config.task, 256)

            # If options exist and non-empty, append them to question (helps MC tasks like code_debug)
            if "options" in df.columns:
                def _fmt_opts(opts):
                    if not isinstance(opts, (list, tuple)) or len(opts) == 0:
                        return ""
                    lines = [f"{i+1}. {str(o)}" for i, o in enumerate(opts)]
                    return "\nOptions:\n" + "\n".join(lines)

                df["question"] = df.apply(
                    lambda r: str(r["question"]) + _fmt_opts(r.get("options", [])),
                    axis=1
                )

            # Tag which split this came from (useful for debugging/scoring)
            if "task" not in df.columns:
                df["task"] = self.config.task


        if fraction < 1.0:
            original_len = len(df)
            # For ruler dataset, sample evenly from each task category
            if dataset_name in ["ruler", "loogle"] and "task" in df.columns:
                sampled_dfs = []
                for task_name, task_df in df.groupby("task"):
                    task_sample = task_df.sample(frac=fraction, random_state=self.config.seed)
                    sampled_dfs.append(task_sample)
                    logger.info(f"Task '{task_name}': sampled {len(task_sample)} from {len(task_df)} samples ({fraction:.2f})")
                df = pd.concat(sampled_dfs, ignore_index=True)
                logger.info(f"Total sampled {len(df)} samples ({fraction:.2f}) from original {original_len} samples across all tasks.")
            # For longbench-v2 dataset, sample evenly from each domain category
            elif dataset_name == "longbench-v2" and "domain" in df.columns:
                sampled_dfs = []
                for domain_name, domain_df in df.groupby("domain"):
                    domain_sample = domain_df.sample(frac=fraction, random_state=self.config.seed)
                    sampled_dfs.append(domain_sample)
                    logger.info(
                        f"Domain '{domain_name}': sampled {len(domain_sample)} from {len(domain_df)} samples ({fraction:.2f})"
                    )
                df = pd.concat(sampled_dfs, ignore_index=True)
                logger.info(
                    f"Total sampled {len(df)} samples ({fraction:.2f}) from original {original_len} samples across all domains."
                )
            else:
                df = df.sample(frac=fraction, random_state=self.config.seed)
                logger.info(f"Sampled {len(df)} samples ({fraction:.2f}) from original {original_len} samples.")

        logger.info(f"Dataset loaded with {len(df)} entries.")

        # if we have needle in a haystack, we need to insert it in the context
        if self.config.dataset == "needle_in_haystack":
            df = insert_needle_in_haystack(
                df, self.pipeline.tokenizer, self.config.max_context_length, self.config.needle_depth
            )

        if isinstance(self.press, FinchPress):
            if not self.config.compress_questions:
                logger.error("FinchPress requires 'compress_questions' to be set to True.")
                raise ValueError("FinchPress requires compress_questions to be set to True")
            # FinchPress uses a delimiter token to separate context and question
            # So we need to update the tokenizer and the model embeddings.
            logger.info("FinchPress detected, updating model and tokenizer with delimiter token.")
            self.press.update_model_and_tokenizer(self.pipeline.model, self.pipeline.tokenizer)  # type: ignore[attr-defined]
            df["context"] = df["context"] + self.press.delimiter_token  # type: ignore[attr-defined, index]

        if self.config.compress_questions:
            logger.info("Compressing questions into context.")
            if isinstance(self.press, QueryIndexerScorePress):
                tok = self.pipeline.tokenizer
                df["question_len"] = df["question"].apply(
                    lambda q: int(len(tok.encode(str(q), add_special_tokens=False)))
                )
                self.press.query_reduce = "question"
            elif isinstance(self.press, MemoryScorerPress) and isinstance(self.press.base_press, QueryIndexerScorePress):
                tok = self.pipeline.tokenizer
                df["question_len"] = df["question"].apply(
                    lambda q: int(len(tok.encode(str(q), add_special_tokens=False)))
                )
                self.press.base_press.query_reduce = "question"

            df["context"] = df["context"] + df["question"]  # type: ignore[index]
            df["question"] = ""  # type: ignore[index]

        self.df = df
        logger.info(f"Dataset processed with {len(self.df)} entries.")

    def _setup_model_pipeline(self):
        from kvpress.presses.dma_score_press import DMAScorePress, load_model_with_dma_press
        from kvpress.presses.indexer_score_press import IndexerScorePress, load_model_with_indexer_press
        from kvpress.presses.indexer_score_press_cache import CacheIndexerScorePress, load_model_with_indexer_press_cache
        from kvpress.presses.indexer_score_query_press import QueryIndexerScorePress, load_model_with_query_indexer_press
        from kvpress.presses.query_indexer_kvzip_press import QueryIndexer_KVzipScorePress
        from kvpress.presses.memory_scorer_press import MemoryScorerPress, load_model_with_memory_params
        from kvpress.presses.gated_press import GatedPress, load_model_with_gated_press
        from kvpress.pipeline import KVPressTextGenerationPipeline
        from kvpress import FixedLayerScoreEvictPress

        model_name = self.config.model
        device = self.config.device

        if device is None:
            device = "auto" if torch.cuda.is_available() else "cpu"
            logger.info(f"No device specified, auto-detected device: {device}")

        model_kwargs = self.config.model_kwargs or {}
        actual_press = self.press
        if isinstance(self.press, DecodePress):
            actual_press = self.press.base_press
        
        use_dma_press = isinstance(actual_press, DMAScorePress)
        use_indexer_press = isinstance(actual_press, IndexerScorePress)
        use_indexer_press_cache = isinstance(actual_press, CacheIndexerScorePress)
        use_query_indexer_press = isinstance(actual_press, (QueryIndexerScorePress, QueryIndexer_KVzipScorePress))
        use_memory_press = isinstance(actual_press, MemoryScorerPress)
        use_gated_press = isinstance(actual_press, GatedPress)
        

        def _contains_attn_press(p) -> bool:
            if p is None:
                return False
            if isinstance(p, (ObservedAttentionPress, GTScorePress)):
                return True
            if hasattr(p, "press") and _contains_attn_press(getattr(p, "press")):
                return True
            if hasattr(p, "base_press") and _contains_attn_press(getattr(p, "base_press")):
                return True
            if hasattr(p, "presses"):
                try:
                    return any(_contains_attn_press(x) for x in getattr(p, "presses"))
                except Exception:
                    return False
            return False

        needs_attn = _contains_attn_press(self.press)

        # Respect user-provided attn_implementation if explicitly set (Fire/YAML merge can override this).
        user_attn_impl = model_kwargs.get("attn_implementation", None)
        if needs_attn:
            model_kwargs["attn_implementation"] = "eager"
            logger.info("Attention-dependent press detected, setting attn_implementation to 'eager'.")
        elif user_attn_impl is None:
            try:
                import flash_attn  # noqa: F401

                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 detected, setting attn_implementation to 'flash_attention_2'.")
            except ImportError:
                logger.info("Flash Attention 2 not available, using default attn_implementation.")
                pass

        logger.info(f"Loading model pipeline for: {model_name} on device: {device} with model_kwargs: {model_kwargs}")

        if use_dma_press or use_indexer_press or use_indexer_press_cache or use_query_indexer_press or use_memory_press or use_gated_press:
            model_kwargs["device_map"] = device if device != "auto" else "auto"
            load_func = (
                load_model_with_dma_press if use_dma_press else
                load_model_with_indexer_press if use_indexer_press else
                load_model_with_indexer_press_cache if use_indexer_press_cache else
                load_model_with_query_indexer_press if use_query_indexer_press else
                load_model_with_memory_params if use_memory_press else
                load_model_with_gated_press if use_gated_press else
                None
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

    @torch.inference_mode()
    def _run_inference(self):
        """
        Executes the inference process on the prepared dataset using the model pipeline.
        """

        self.df["predicted_answer"] = None  # type: ignore[index]
        
        # Initialize response_length column for math500 and aime25 datasets
        record_response_length = self.config.dataset in ["math500", "aime25"]
        if record_response_length:
            self.df["response_length"] = None  # type: ignore[index]

        if isinstance(self.press, DecodePress):
            logger.info("DecodingPress detected, running inference for each context-question pair.")
            for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Running Inference"):
                # Reset any internal state carried by decoding presses between samples
                if hasattr(self.press, "reset") and callable(getattr(self.press, "reset")):
                    self.press.reset()
                # Reset indexer cache (if applicable) to avoid cache accumulation across samples
                self._reset_indexer_cache()

                context = row["context"]
                question = row["question"]
                answer_prefix = row["answer_prefix"]
                max_new_tokens = self.config.max_new_tokens or row["max_new_tokens"]
                if "question_len" in row:
                    if isinstance(self.press, QueryIndexerScorePress):
                        self.press.question_len = int(row["question_len"])
                        self.press.query_reduce = "question"
                    elif isinstance(self.press, MemoryScorerPress) and isinstance(self.press.base_press, QueryIndexerScorePress):
                        self.press.base_press.question_len = int(row["question_len"])
                        self.press.base_press.query_reduce = "question"

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
                
                # Record response length for math500 and aime25 datasets
                if record_response_length:
                    response_tokens = self.pipeline.tokenizer.encode(answer, add_special_tokens=False)
                    self.df.loc[index, "response_length"] = len(response_tokens)  # type: ignore[union-attr]
                
                torch.cuda.empty_cache()  # Clear CUDA cache to free up memory

        else:
            df_context_grouped = self.df.groupby("context")  # type: ignore[union-attr]
            assert all(
                df_context_grouped["answer_prefix"].nunique() == 1
            ), "Inconsistent 'answer_prefix' within the same context group detected."

            logger.info("Starting inference...")
            for context, df_group in tqdm(
                df_context_grouped, total=self.df["context"].nunique(), desc="Running Inference"
            ):  # type: ignore[union-attr]
                if hasattr(self.press, "_reset_cache") and callable(getattr(self.press, "_reset_cache")):
                    self.press._reset_cache()
                if "question_len" in df_group.columns:
                    if isinstance(self.press, QueryIndexerScorePress):
                        self.press.question_len = int(df_group["question_len"].iloc[0])
                        self.press.query_reduce = "question"
                    elif isinstance(self.press, MemoryScorerPress) and isinstance(self.press.base_press, QueryIndexerScorePress):
                        self.press.base_press.question_len = int(df_group["question_len"].iloc[0])
                        self.press.base_press.query_reduce = "question"

                questions = df_group["question"].to_list()
                # Use max_new_tokens from config, or fallback to dataset's default for the task
                max_new_tokens = self.config.max_new_tokens or df_group["max_new_tokens"].iloc[0]
                answer_prefix = df_group["answer_prefix"].iloc[0]
                answers = []
                for q in questions:
                    out = self.pipeline(
                        context,
                        question=q,                 # 关键：一次只喂一个 question
                        answer_prefix=answer_prefix,
                        press=self.press,
                        max_new_tokens=max_new_tokens,
                        max_context_length=self.config.max_context_length,
                    )
                    answers.append(out["answer"])
                    torch.cuda.empty_cache()        # 可选：释放缓存，长上下文更稳

                self.df.loc[df_group.index, "predicted_answer"] = answers
                # Record response length for math500 and aime25 datasets
                if record_response_length:
                    response_lengths = [
                        len(self.pipeline.tokenizer.encode(answer, add_special_tokens=False))
                        for answer in answers
                    ]
                    self.df.loc[df_group.index, "response_length"] = response_lengths  # type: ignore[union-attr]
                
                # Store the actual compression ratio used (if the press has one)
                self.df.loc[df_group.index, "compression_ratio"] = (
                    self.press.compression_ratio if self.press is not None else 0.0  # type: ignore[attr-defined]
                )  # type: ignore[union-attr, attr-defined]
                torch.cuda.empty_cache()  # Clear CUDA cache to free up memory

        logger.info("Inference completed.")

    def _reset_indexer_cache(self):
        from kvpress.presses.indexer_score_press import IndexerScorePress
        from kvpress.presses.indexer_score_press_cache import CacheIndexerScorePress
        from kvpress.presses.indexer_score_query_press import QueryIndexerScorePress
        from kvpress.presses.memory_scorer_press import MemoryScorerPress
        actual_press = self.press
        if isinstance(self.press, DecodePress):
            actual_press = self.press.base_press
        
        if isinstance(actual_press, MemoryScorerPress):
            actual_press = actual_press.base_press

        if isinstance(actual_press, (IndexerScorePress, CacheIndexerScorePress, QueryIndexerScorePress)):
            language_model = self.pipeline.model.model.language_model if hasattr(self.pipeline.model.model, "language_model") else self.pipeline.model.model
            for layer in language_model.layers:
                if hasattr(layer.self_attn, actual_press.scorer_attr):
                    indexer = getattr(layer.self_attn, actual_press.scorer_attr)
                    # Cache indexers expose reset_cache(); non-cache indexers may not.
                    if hasattr(indexer, "reset_cache") and callable(getattr(indexer, "reset_cache")):
                        indexer.reset_cache()

    def _save_results(self, save_filename: Path):
        """
        Saves the predicted answers and compression ratios to a CSV file.

        Parameters
        ----------
        save_filename : Path
            The full path including filename to save the CSV.
        """
        if save_filename.exists():
            logger.warning(f"Results CSV already exists at {save_filename}. Overwriting.")

        self.df[list(set(self.df.columns) - set(["context"]))].to_csv(
            str(save_filename), index=False
        )  # type: ignore[index]
        logger.info(f"Results saved to {save_filename}")

    def _calculate_and_save_metrics(self, save_filename: Path):
        """
        Calculates evaluation metrics and saves them to a JSON file.

        Parameters
        ----------
        save_filename : Path
            The base filename (e.g., CSV path) to derive the JSON path from.
        """
        dataset_name = self.config.dataset
        scorer = SCORER_REGISTRY[dataset_name]

        logger.info(f"Calculating metrics for dataset: {dataset_name}")
        metrics = scorer(self.df)  # type: ignore[call-arg]

        with open(str(save_filename), "w") as f:
            json.dump(metrics, f, indent=4)  # Pretty print JSON

        logger.info(f"Metrics saved to {save_filename}")
        logger.info(f"Metrics:\n{json.dumps(metrics, indent=2)}")

    def run_evaluation(self):
        """
        Orchestrates the entire evaluation process.
        """
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

        self._run_inference()
        self._save_results(predictions_filename)
        self._calculate_and_save_metrics(metrics_filename)
        self.config.save_config(config_filename)
        logger.info("Evaluation run completed successfully.")


# --- Command-Line Interface ---
class CliEntryPoint:
    """
    CLI entry point for building configuration and running the evaluation.

    This class provides a command-line interface for running KVPress evaluations.
    Configuration can be specified via:
    1. YAML config file (default: "./evaluate_config.yaml")
    2. Command-line arguments (highest priority)
    """

    def __call__(self, config_file: Optional[str] = "./evaluate_config.yaml", **cli_overrides):
        """
        Builds the configuration and runs the evaluation.

        Configuration is built by layering:
        1. Default values from EvaluationConfig
        2. Values from YAML config file
        3. Command-line arguments (highest priority)
        """
        # 1. Start with dataclass defaults.
        final_args = asdict(EvaluationConfig())

        # 2. Layer YAML values on top.
        yaml_config = _load_yaml_config(config_file)
        final_args.update(yaml_config)

        # 3. Layer CLI arguments on top (highest priority).
        # Filter out None values from CLI overrides
        cli_args = {k: v for k, v in cli_overrides.items() if v is not None}
        # Special-case: merge model_kwargs instead of replacing the whole dict from YAML.
        if "model_kwargs" in cli_args:
            base_mk = _parse_model_kwargs(final_args.get("model_kwargs"))
            override_mk = _parse_model_kwargs(cli_args.get("model_kwargs"))
            base_mk.update(override_mk)
            cli_args["model_kwargs"] = base_mk

        final_args.update(cli_args)

        # 4. Create and validate the final config object.
        try:
            config = EvaluationConfig(**final_args)
        except TypeError as e:
            # Provide a user-friendly error for bad arguments.
            print(f"Error: Invalid configuration argument provided. {e}", file=sys.stderr)
            sys.exit(1)

        runner = EvaluationRunner(config)
        runner.run_evaluation()


if __name__ == "__main__":
    Fire(CliEntryPoint)
