# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Force transformers to avoid optional TF/Flax backends in this environment.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from evaluate_registry import DATASET_REGISTRY, PRESS_REGISTRY, SCORER_REGISTRY
from fire import Fire
from tqdm import tqdm
from transformers import DynamicCache, Pipeline, pipeline
import re
import math
# --- quantized cache (HF transformers versions differ) ---
try:
    # new versions: documented import path
    from transformers import QuantizedCacheConfig  # type: ignore
except Exception:
    try:
        # some versions expose it here
        from transformers.cache_utils import QuantizedCacheConfig  # type: ignore
    except Exception:
        QuantizedCacheConfig = None  # fallback

from kvpress import (
    ComposedPress,
    DecodePress,
    DecodingPress,
    DuoAttentionPress,
    FinchPress,
    MemoryScorerPress,
    ObservedAttentionPress,
    QueryIndexerScorePress,
    ScorerPress,
    SelectiveDecodingPress,
    CacheIndexerDecodingPress,
    ThinKPress,
)
from kvpress.presses.gt_score_press import GTScorePress

logger = logging.getLogger(__name__)


def _infer_model_device(model: Any) -> torch.device:
    # Handle sharded / device_map models best-effort
    try:
        dev = getattr(model, "device", None)
        if dev is not None:
            return torch.device(dev)
    except Exception:
        pass
    try:
        for p in model.parameters():
            if hasattr(p, "device") and p.device.type != "meta":
                return p.device
    except Exception:
        pass
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _press_needs_output_attentions(press: Any) -> bool:
    """Mirror KVPressTextGenerationPipeline.output_attentions (presses that require attention tensors)."""

    def _contains(p: Any) -> bool:
        if p is None:
            return False
        if isinstance(p, (ObservedAttentionPress, GTScorePress)):
            return True
        if hasattr(p, "press") and _contains(getattr(p, "press")):
            return True
        if hasattr(p, "base_press") and _contains(getattr(p, "base_press")):
            return True
        if hasattr(p, "presses"):
            try:
                return any(_contains(x) for x in getattr(p, "presses"))
            except Exception:
                return False
        return False

    return _contains(press)


class ForceOptionLogitsProcessor(torch.nn.Module):
    def __init__(self, prefix_ids):
        super().__init__()
        self.prefix_ids = prefix_ids
        self.step_counter = 0
        self.original_logits = []

    def forward(self, input_ids, scores):
        if self.step_counter < len(self.prefix_ids):
            self.original_logits.append(scores.detach().clone())
            forced_token_id = self.prefix_ids[self.step_counter]
            mask = torch.full_like(scores, float("-inf"))
            mask[:, forced_token_id] = 0
            scores = scores + mask
            self.step_counter += 1
        return scores


def _truncate_inputs_to_max_tokens(tokenizer, prompt: str, max_input_tokens: Optional[int]):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    if max_input_tokens is None:
        return inputs
    input_ids = inputs["input_ids"]
    if input_ids.shape[1] > int(max_input_tokens):
        inputs["input_ids"] = input_ids[:, -int(max_input_tokens) :]
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, -int(max_input_tokens) :]
    return inputs


@torch.inference_mode()
def evaluate_cloze_generate(
    model,
    tokenizer,
    prompt: str,
    options: list[str],
    *,
    max_input_tokens: Optional[int],
):
    device = _infer_model_device(model)
    option_scores: list[float] = []

    inputs = _truncate_inputs_to_max_tokens(tokenizer, prompt, max_input_tokens)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    for option in options:
        option_ids = tokenizer.encode(option, add_special_tokens=False)
        option_len = len(option_ids)
        option_processor = ForceOptionLogitsProcessor(option_ids)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            outputs = model.generate(
                **inputs,
                max_new_tokens=option_len,
                logits_processor=[option_processor],
                output_scores=True,
                return_dict_in_generate=True,
                num_beams=1,
                do_sample=False,
                use_cache=True,
            )

        generated_ids = outputs.sequences[0, inputs["input_ids"].shape[-1] :]
        if not torch.equal(generated_ids.detach().cpu(), torch.tensor(option_ids)):
            raise ValueError("Forced generation failed.")

        log_prob = 0.0
        assert len(option_processor.original_logits) == option_len
        for step in range(option_len):
            logits = option_processor.original_logits[step][0]
            prob = torch.log_softmax(logits, dim=-1)
            target_token = option_ids[step]
            log_prob += prob[target_token].item()

        option_scores.append(log_prob / max(option_len, 1))

    best_option_index = int(np.argmax(option_scores))
    return chr(65 + best_option_index)


@torch.inference_mode()
def evaluate_cloze_forward(
    model,
    tokenizer,
    prompt: str,
    options: list[str],
    *,
    max_input_tokens: Optional[int],
    pad_to_multiple: int = 1,
    press: Any = None,
):
    device = _infer_model_device(model)
    # Match pipeline: prefill hooks only for prefill presses (not decoding-only presses).
    apply_prefill_press = press is not None and not isinstance(press, DecodingPress)
    option_scores: list[float] = []

    for option in options:
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=False).input_ids[0]
        prompt_len = int(prompt_ids.shape[0])
        option_ids = tokenizer(option, return_tensors="pt", truncation=False).input_ids[0]
        option_len = int(option_ids.shape[0])
        input_ids = torch.cat([prompt_ids, option_ids], dim=0)

        if max_input_tokens is not None and int(input_ids.shape[0]) > int(max_input_tokens):
            input_ids = input_ids[-int(max_input_tokens) :]

        original_length = int(input_ids.shape[0])
        if pad_to_multiple and pad_to_multiple > 1:
            target_length = int(math.ceil(original_length / pad_to_multiple) * pad_to_multiple)
            padding_length = int(target_length - original_length)
            if padding_length > 0:
                padding_tensor = torch.tensor([-100] * padding_length, dtype=input_ids.dtype)
                input_ids = torch.cat([input_ids, padding_tensor])

        actual_prompt_len = min(prompt_len, int(input_ids.shape[0]) - option_len)
        actual_option_len = int(input_ids.shape[0]) - actual_prompt_len

        input_b = input_ids.unsqueeze(0).to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            if apply_prefill_press:
                cache = DynamicCache()
                with press(model):
                    outputs = model(
                        input_ids=input_b,
                        past_key_values=cache,
                        use_cache=True,
                        output_attentions=_press_needs_output_attentions(press),
                    )
            else:
                outputs = model(input_ids=input_b, use_cache=False)
        logits = outputs.logits[0]
        logits = logits.to(torch.float16)
        log_probs = torch.log_softmax(logits, dim=-1).detach().cpu().numpy()

        log_prob = 0.0
        input_ids_cpu = input_ids.detach().cpu().numpy()
        total_len = int(input_ids_cpu.shape[0])
        for i in range(actual_prompt_len - 1, original_length - 1):
            target_token = int(input_ids_cpu[i + 1])
            neg_index = i - total_len
            log_prob += float(log_probs[neg_index][target_token])

        option_scores.append(log_prob / max(actual_option_len, 1))

    best_option_index = int(np.argmax(option_scores))
    return chr(65 + best_option_index)


_LBV2_Q_RE = re.compile(r"^What is the correct answer to this question:\s*(.*?)\s*$", flags=re.MULTILINE)


def _extract_longbenchv2_question(templated_question: str) -> str:
    """
    The processed HF dataset replaces `question` with a template that includes Choices.
    For cloze scoring we need the raw question string.
    """
    m = _LBV2_Q_RE.search(str(templated_question))
    if m:
        return m.group(1).strip()
    # Fallback: best-effort strip everything after 'Choices:'
    s = str(templated_question)
    if "Choices:" in s:
        head = s.split("Choices:", 1)[0]
        return head.strip()
    return s.strip()


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
    use_chunk_prefill: bool = False #是否启用分段 prefill
    chunk_prefill_size: Optional[int] = None #每段多少 token（具体含义由 pipeline 里循环步长决定
    # --- LongBench-v2 specific ---
    # If set, truncate tokenized prompt from the left to this many tokens (like eval_longbench.py).
    max_input_tokens: Optional[int] = None
    # "generate" (default): use pipeline.generate text; "cloze": score A/B/C/D options and output letter.
    longbenchv2_eval: str = "generate"
    # When longbenchv2_eval == "cloze": "generate" (forced generation) or "forward" (single forward pass scoring)
    longbenchv2_cloze_impl: str = "forward"
    # If set, keep only rows with this exact `domain` value (after load, before fraction sampling).
    longbenchv2_domain: Optional[str] = None
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
    # --- KV cache quantization (optional) ---
    kv_cache_nbits: Optional[int] = None   # 4 or 8; None => no KV-cache quant
    kv_cache_backend: str = "quanto"       # "quanto" or "hqq"
    kvq_smoke_test: bool = False   # 只在 debug 时打开
    kvq_smoke_prompt_repeats: int = 512
    kvq_smoke_new_tokens: int = 256
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
        if self.dataset == "longbench-v2":
            assert self.longbenchv2_eval in {"generate", "cloze"}, "longbenchv2_eval must be 'generate' or 'cloze'"
            assert self.longbenchv2_cloze_impl in {"generate", "forward"}, "longbenchv2_cloze_impl must be 'generate' or 'forward'"
        if self.kv_cache_nbits is not None:
            assert self.kv_cache_nbits in (4, 8), "kv_cache_nbits must be 4 or 8"
            assert self.kv_cache_backend in ("quanto", "hqq"), "kv_cache_backend must be 'quanto' or 'hqq'"
        if self.use_chunk_prefill:
            assert self.chunk_prefill_size is not None and self.chunk_prefill_size > 0, (
                "chunk_prefill_size must be set and > 0 when use_chunk_prefill=True"
            )

    def get_results_dir(self, output_dir: Path) -> Path:
        """
        Generates the unique save directory and filenames based on configuration parameters.

        Parameters
        ----------
        output_dir : Path
            The output directory path
        press
            The press instance to check for ThinKPress components

        Returns
        -------
        Path
            The path to the results directory
        """
        model_name = self.model.split("/")[-1] if "/" in self.model else self.model
        
        # Build directory name components
        components = [
            self.dataset,
        ]
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
        if self.kv_cache_nbits is not None:
            components.append(f"kvq{int(self.kv_cache_nbits)}")
            components.append(f"kvq_backend{self.kv_cache_backend}")

        dir_name = "__".join(filter(None, components))  # Filter None/empty strings
        config_dir = output_dir / dir_name

        # Make sure the directory does not exist, if it does, add a number to the end
        # This is to avoid overwriting results
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
        logger.info(f"Preparing dataset: {dataset_name} (data_dir: {data_dir})")

        # ---- 1) local_ruler: read local JSONL directly ----
        if dataset_name == "local_ruler":
            if not data_dir:
                raise ValueError("data_dir must be provided for local_ruler")

            jsonl_path = Path(data_dir) / "validation.jsonl"
            if not jsonl_path.exists():
                raise FileNotFoundError(f"Local dataset file not found: {jsonl_path}")

            logger.info(f"Loading local dataset from: {jsonl_path}")
            df = pd.read_json(jsonl_path, lines=True)

            logger.info(f"Local dataset columns before normalization: {list(df.columns)}")

            # Normalize xKV-generated RULER JSONL to kvpress schema
            if "input" in df.columns and "context" not in df.columns:
                df["context"] = df["input"]

            if "question" not in df.columns:
                df["question"] = ""

            if "outputs" in df.columns and "answers" not in df.columns:
                df["answers"] = df["outputs"]

            if "answer_prefix" not in df.columns:
                df["answer_prefix"] = ""

            if "max_new_tokens" not in df.columns:
                df["max_new_tokens"] = 32
            
            # 补 task：从单任务目录名自动推断，例如 .../65536/niah_single_1 -> niah_single_1
            if "task" not in df.columns:
                df["task"] = Path(data_dir).name

            # 兼容 kvpress scorer：它默认读 answer，不是 answers
            if "answer" not in df.columns and "answers" in df.columns:
                df["answer"] = df["answers"]

            # 可选：保证 answer 最终都是 list[str] 形式
            def _normalize_answer_cell(x):
                if isinstance(x, list):
                    return [str(v) for v in x]
                if pd.isna(x):
                    return [""]
                return [str(x)]

            df["answer"] = df["answer"].apply(_normalize_answer_cell)

            required_cols = ["context", "question", "answers", "answer", "task"]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"local_ruler is missing required column after normalization: {col}")

            logger.info(f"Local dataset columns after normalization: {list(df.columns)}")

        # ---- 2) all other datasets: use HF datasets ----
        else:
            logger.info(f"Loading dataset: {DATASET_REGISTRY[dataset_name]} (data_dir: {data_dir})")

            load_kwargs = {"split": "test"}
            if dataset_name == "aime24":
                load_kwargs["split"] = "train"
            if data_dir and dataset_name not in ["math500", "aime25", "aime24"]:
                load_kwargs["data_dir"] = data_dir

            try:
                df = load_dataset(DATASET_REGISTRY[dataset_name], **load_kwargs).to_pandas()
            except Exception:
                if dataset_name == "aime24":
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

        if dataset_name == "longbench-v2" and self.config.longbenchv2_domain:
            dom = str(self.config.longbenchv2_domain).strip()
            if dom:
                if "domain" not in df.columns:
                    raise ValueError(
                        "longbench-v2 dataframe has no 'domain' column; cannot apply longbenchv2_domain filter."
                    )
                before = len(df)
                domain_choices = sorted(df["domain"].astype(str).unique().tolist()) if before else []
                df = df.loc[df["domain"] == dom].copy()
                logger.info(
                    "Filtered longbench-v2 to domain=%r: %s rows (was %s).",
                    dom,
                    len(df),
                    before,
                )
                if len(df) == 0:
                    raise ValueError(
                        f"No rows left after filtering longbench-v2 to domain={dom!r}. "
                        f"Check spelling against dataset `domain` values (sample): {domain_choices[:40]}"
                    )

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
            elif dataset_name == "longbench" and "dataset" in df.columns:
                sampled_dfs = []
                for ds_name, ds_df in df.groupby("dataset"):
                    ds_sample = ds_df.sample(frac=fraction, random_state=self.config.seed)
                    sampled_dfs.append(ds_sample)
                    logger.info(f"Dataset '{ds_name}': sampled {len(ds_sample)} from {len(ds_df)} ({fraction:.2f})")
                df = pd.concat(sampled_dfs, ignore_index=True)
                logger.info(f"Total sampled {len(df)} samples ({fraction:.2f}) from original {original_len} across all datasets.")
            elif dataset_name == "infinitebench" and "task" in df.columns:
                sampled_dfs = []
                for tname, tdf in df.groupby("task"):
                    ts = tdf.sample(frac=fraction, random_state=self.config.seed)
                    sampled_dfs.append(ts)
                    logger.info(f"Task '{tname}': sampled {len(ts)} from {len(tdf)} ({fraction:.2f})")
                df = pd.concat(sampled_dfs, ignore_index=True)
                logger.info(f"Total sampled {len(df)} samples ({fraction:.2f}) from original {original_len} across all tasks.")
            else:
                df = df.sample(frac=fraction, random_state=self.config.seed)
                logger.info(f"Sampled {len(df)} samples ({fraction:.2f}) from original {original_len} samples.")

        logger.info(f"Dataset loaded with {len(df)} entries.")

        # if we have needle in a haystack, we need to insert it in the context
        if self.config.dataset == "needle_in_haystack":
            from benchmarks.needle_in_haystack.utils import insert_needle_in_haystack
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
        if self.config.kvq_smoke_test:
            self._kvq_smoke_test()

            # 可选：把 KVQ_CHECK 也放到 debug 开关里
            if torch.cuda.is_available() and self.config.kv_cache_nbits is not None:
                try:
                    tok = self.pipeline.tokenizer
                    m = self.pipeline.model
                    ids = tok("hello", return_tensors="pt").to(next(m.parameters()).device)
                    kvq = self._kv_cache_generate_kwargs()
                    out = m.generate(**ids, max_new_tokens=2, return_dict_in_generate=True, **kvq)
                    pkv = getattr(out, "past_key_values", None)
                    logger.info(f"[KVQ_CHECK] past_key_values type = {type(pkv)}; kvq={kvq}")
                except Exception as e:
                    logger.warning(f"[KVQ_CHECK] failed: {e}")

        # ---- KVQ self-check (MUST be after pipeline is created) ----
        if torch.cuda.is_available() and self.config.kv_cache_nbits is not None:
            try:
                tok = self.pipeline.tokenizer
                m = self.pipeline.model
                ids = tok("hello", return_tensors="pt").to(next(m.parameters()).device)
                kvq = self._kv_cache_generate_kwargs()
                out = m.generate(**ids, max_new_tokens=2, return_dict_in_generate=True, **kvq)
                pkv = getattr(out, "past_key_values", None)
                logger.info(f"[KVQ_CHECK] past_key_values type = {type(pkv)}; kvq={kvq}")
            except Exception as e:
                logger.warning(f"[KVQ_CHECK] failed: {e}")
        # ------------------------------------------------------------
    @staticmethod
    def _estimate_pkv_bytes(pkv) -> int:
        # 兼容最常见的 legacy past_key_values: tuple(layer) of (k, v)
        total = 0
        if pkv is None:
            return 0
        # 有些实现是 list/tuple
        if isinstance(pkv, (list, tuple)):
            for layer in pkv:
                if isinstance(layer, (list, tuple)) and len(layer) >= 2:
                    k, v = layer[0], layer[1]
                    if torch.is_tensor(k):
                        total += k.numel() * k.element_size()
                    if torch.is_tensor(v):
                        total += v.numel() * v.element_size()
        return total

    def _kvq_smoke_test(self):
        if not torch.cuda.is_available() or self.config.kv_cache_nbits is None:
            logger.info("[KVQ_SMOKE] skip (no cuda or kv_cache_nbits is None)")
            return

        tok = self.pipeline.tokenizer
        m = self.pipeline.model
        dev = next(m.parameters()).device

        # 拉长 prompt，让 KV cache 真正长起来
        ids = tok(("hello " * 512), return_tensors="pt").to(dev)

        def run(tag, kvq):
            torch.cuda.synchronize()
            torch.cuda.memory.reset_peak_memory_stats()

            out = m.generate(
                **ids,
                max_new_tokens=256,
                use_cache=True,
                return_dict_in_generate=True,
                **kvq,
            )

            pkv = getattr(out, "past_key_values", None)
            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

            pkv_type = type(pkv)
            pkv_bytes = self._estimate_pkv_bytes(pkv)
            logger.info(
                f"[KVQ_SMOKE] {tag} peak_alloc_mb={peak_mb:.2f} pkv_type={pkv_type} pkv_bytes={pkv_bytes/1024/1024:.2f} kvq={kvq}"
            )

            # 如果你打开了 quant，但 pkv 还是老的 tuple[tensor,tensor]，基本=没生效
            if tag == "quant" and isinstance(pkv, (tuple, list)):
                # 粗判：量化 cache 通常不会还是纯 tensor tuple（取决于 transformers 版本实现）
                logger.warning("[KVQ_SMOKE] quant ON but pkv looks like legacy tuple/list -> very likely NOT active")

        run("baseline", {})
        run("quant", self._kv_cache_generate_kwargs())

    def _kv_cache_generate_kwargs(self) -> Dict[str, Any]:
        nbits = getattr(self.config, "kv_cache_nbits", None)
        if nbits is None:
            return {}
        backend = getattr(self.config, "kv_cache_backend", "quanto")
        return {
            "cache_implementation": "quantized",
            "cache_config": QuantizedCacheConfig(nbits=int(nbits), backend=str(backend)),
        }

    def _reset_peak_mem(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()  # resets peak counters :contentReference[oaicite:6]{index=6}
    def _get_device_mem_mb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        torch.cuda.synchronize()
        if hasattr(torch.cuda, "device_memory_used"):
            return float(torch.cuda.device_memory_used() / (1024 * 1024))
        # fallback: 至少给出 reserved 作为近似
        return float(torch.cuda.memory_reserved() / (1024 * 1024))

    def _get_peak_alloc_mb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated()  # peak since last reset :contentReference[oaicite:7]{index=7}
        return peak_bytes / (1024 * 1024)

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

        if self.config.dataset == "longbench-v2" and self.config.longbenchv2_eval == "cloze":
            logger.info(
                f"LongBench-v2 cloze mode enabled (impl={self.config.longbenchv2_cloze_impl}, max_input_tokens={self.config.max_input_tokens})."
            )
            m = self.pipeline.model
            tok = self.pipeline.tokenizer
            for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Running Inference (longbench-v2 cloze)"):
                # Reset any internal state carried by presses between samples
                if isinstance(self.press, DecodePress) and hasattr(self.press, "reset") and callable(getattr(self.press, "reset")):
                    self.press.reset()
                if hasattr(self.press, "_reset_cache") and callable(getattr(self.press, "_reset_cache")):
                    self.press._reset_cache()
                self._reset_indexer_cache()

                context = str(row["context"])
                q_raw = _extract_longbenchv2_question(row["question"])
                prompt = f"{context}\n\n{q_raw}\n\n"
                options = [str(row["choice_A"]), str(row["choice_B"]), str(row["choice_C"]), str(row["choice_D"])]

                if self.config.longbenchv2_cloze_impl == "generate":
                    pred = evaluate_cloze_generate(m, tok, prompt, options, max_input_tokens=self.config.max_input_tokens)
                else:
                    pred = evaluate_cloze_forward(
                        m,
                        tok,
                        prompt,
                        options,
                        max_input_tokens=self.config.max_input_tokens,
                        pad_to_multiple=1,
                        press=self.press,
                    )

                # Keep output compatible with existing longbenchv2 scorer
                self.df.loc[index, "predicted_answer"] = f"The correct answer is ({pred})."  # type: ignore[union-attr]
                torch.cuda.empty_cache()

        elif isinstance(self.press, DecodePress):
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

                kvq = self._kv_cache_generate_kwargs()
                output = self.pipeline(
                    context,
                    question=question,
                    answer_prefix=answer_prefix,
                    press=self.press,
                    max_new_tokens=max_new_tokens,
                    max_context_length=self.config.max_context_length,
                    use_chunk_prefill=self.config.use_chunk_prefill,
                    chunk_prefill_size=self.config.chunk_prefill_size,
                    **kvq,
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
                kvq = self._kv_cache_generate_kwargs()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                output = self.pipeline(
                    context,
                    questions=questions,
                    answer_prefix=answer_prefix,
                    press=self.press,
                    max_new_tokens=max_new_tokens,
                    max_context_length=self.config.max_context_length,
                    use_chunk_prefill=self.config.use_chunk_prefill,
                    chunk_prefill_size=self.config.chunk_prefill_size,
                    **kvq,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    group_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    logger.info(f"[GPU_MEM_GROUP] task={self.config.data_dir} peak_alloc_mb={group_peak:.2f} kvq={kvq}")

                answers = output["answers"]  # type: ignore[union-attr]
                self.df.loc[df_group.index, "predicted_answer"] = answers  # type: ignore[union-attr]
                
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
        mem_filename = results_dir / "gpu_mem.json"
        if predictions_filename.exists() and metrics_filename.exists():
            logger.info(
                f"Evaluation files already exist at \n {predictions_filename} \n {metrics_filename}.\nSkipping..."
            )
            return

        self._setup_press()
        self._setup_model_pipeline()
        self._load_and_prepare_dataset()
        # ====== 新增：只统计推理阶段的 peak_alloc ======
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()  # reset peak stats :contentReference[oaicite:2]{index=2}
        # ===========================================

        self._run_inference()
        # ====== 新增：推理结束后读 peak_alloc_mb 并落盘 ======
        peak_alloc_mb = 0.0
        peak_reserved_mb = 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)  # :contentReference[oaicite:3]{index=3}
            peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
        mem_report = {
            "peak_alloc_mb": float(peak_alloc_mb),
            "peak_reserved_mb": float(peak_reserved_mb),
            "device_mem_mb_end": self._get_device_mem_mb(),
            "kv_cache_nbits": getattr(self.config, "kv_cache_nbits", None),
            "kv_cache_backend": getattr(self.config, "kv_cache_backend", None),
        }
        with open(mem_filename, "w") as f:
            json.dump(mem_report, f, indent=2)

        logger.info(f"[GPU_MEM] peak_alloc_mb={peak_alloc_mb:.2f} MB (saved to {mem_filename})")
        # =====================================================

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
