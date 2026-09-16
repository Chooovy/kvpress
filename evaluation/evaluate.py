# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from fire import Fire
from tqdm import tqdm
from transformers import FineGrainedFP8Config, Pipeline, pipeline

# Run as a plain script without pip-installing the package: put the repo root on sys.path so
# `import kvpress` resolves. Must precede the evaluate_registry import too -- that module imports
# kvpress itself, so it is the line that fails first without this.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.needle_in_haystack.utils import insert_needle_in_haystack  # noqa: E402
from evaluate_registry import (  # noqa: E402
    DATASET_REGISTRY,
    PRESS_REGISTRY,
    SCORER_REGISTRY,
)
from kvpress import (  # noqa: E402
    ComposedPress,
    DecodingPress,
    DMSPress,
    DuoAttentionPress,
    FinchPress,
    ObservedAttentionPress,
    ScorerPress,
    ThinKPress,
)

logger = logging.getLogger(__name__)

#: Index stride between rollouts when ``--rollouts > 1``; must match
#: ``evaluate_sparse.ROLLOUT_STRIDE`` so a dense run and a sparse run key their rows identically.
#: Defined here rather than imported because ``evaluate_sparse`` pulls in the whole gqa_indexer
#: stack (triton kernels, flex_attention), which an eviction-only run has no reason to load.
ROLLOUT_STRIDE = 1_000_000


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
    head_compression_ratio: Optional[float] = None
    threshold: Optional[float] = None

    # Dataset and generation parameters
    fraction: float = 1.0
    max_new_tokens: Optional[int] = None
    max_context_length: Optional[int] = None
    query_aware: bool = False
    needle_depth: Optional[int] = None
    # Qwen3-style thinking mode. False (the pipeline's own default) makes the chat template emit a
    # PRE-CLOSED, empty "<think>\n\n</think>" before the answer, which suppresses the reasoning
    # block; True leaves the turn open so the model opens <think> itself. Not a cosmetic switch on
    # the math benchmarks -- non-thinking Qwen3-8B scored 0.167 on aime25 here, and thinking traces
    # are several times longer, so max_new_tokens has to be raised with it or the trace is truncated
    # before it ever reaches \boxed{}.
    enable_thinking: bool = False

    # Sampled decoding and pass@1 rollouts, mirroring evaluate_sparse.py field for field.
    #
    # Defaults (`rollouts=1`, `do_sample=False`) are the greedy single-trace path every earlier
    # result used, bitwise -- `KVPressTextGenerationPipeline._pick_token` falls back to `argmax`
    # when `sampling` is unset. They exist so the Full-KV reference for the reasoning benchmarks
    # can be measured under the SAME protocol as the sparse arm: comparing an R=2 sampled mean
    # against one greedy trace would confound the compression effect with the decoding rule, and
    # the difference is large (thinking-mode traces degenerate into repetition under argmax).
    rollouts: int = 1
    do_sample: bool = False
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20

    # Decoding parameters
    compression_interval: Optional[int] = None
    target_size: Optional[int] = None
    hidden_states_buffer_size: Optional[int] = None

    # Output and logging
    output_dir: str = "./results"
    log_level: str = "INFO"

    # Data-parallel sharding, mirroring evaluate_sparse.py. num_shards > 1 makes this process
    # evaluate only its slice of the contexts and write a parquet shard *without* scoring: a
    # per-shard metric is a per-task mean over an arbitrary subset of rows and is not comparable to
    # anything. Use evaluate_sharded.py, which launches the shards and scores their union once.
    shard_index: int = 0
    num_shards: int = 1
    # Sharding axis, mirroring evaluate_sparse.py. "context" is the default and right for every
    # benchmark with long shared contexts. "row" exists for the reasoning benchmarks: math500 and
    # aime25 put the whole problem in `question` and leave `context` a single space, so ALL rows
    # share ONE context and the round-robin over contexts hands every row to shard 0 while the other
    # GPUs idle. There is no prefill to share at 4 tokens, so splitting by row costs nothing there.
    shard_by: str = "context"
    # Set by evaluate_sharded.py so every shard writes into the one directory the driver chose.
    # get_results_dir uniquifies by appending a counter when the directory exists, so N shards each
    # calling it would race and land in N different directories.
    results_dir: Optional[str] = None

    # Model-specific parameters
    model_kwargs: Optional[Dict[str, Any]] = None

    # Attention kernel. None keeps the historical behaviour: flash_attention_2 whenever the
    # flash_attn package merely *imports*, with no check that the build actually works against the
    # installed torch/transformers. A mismatched build does not raise -- it silently returns wrong
    # logits, which reads as "the model scores 0 on everything" rather than as a crash. Set this to
    # "sdpa" (or "eager") to take that variable out of an evaluation.
    attn_implementation: Optional[str] = None

    # Press information (will be set after press setup)
    press_init_command: Optional[str] = None

    # For reproducibility
    seed: int = 42

    # Quantization
    fp8: bool = False

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

        # Only validate key_channel_compression_ratio if it's not None
        if self.key_channel_compression_ratio is not None:
            assert (
                0.0 <= self.key_channel_compression_ratio <= 1.0
            ), f"key_channel_compression_ratio must be between 0.0 and 1.0, got {self.key_channel_compression_ratio}"

        # Validate fraction
        assert 0.0 < self.fraction <= 1.0, f"fraction must be between 0.0 and 1.0, got {self.fraction}"

        # Initialize model_kwargs if None
        if self.model_kwargs is None:
            self.model_kwargs = {}

        if self.dataset == "needle_in_haystack":
            assert self.needle_depth is not None, "needle_depth must be set for needle_in_haystack"
            assert self.max_context_length is not None, "max_context_length must be set for needle_in_haystack"

        assert self.num_shards >= 1, f"num_shards must be >= 1, got {self.num_shards}"
        assert (
            0 <= self.shard_index < self.num_shards
        ), f"shard_index must be in [0, {self.num_shards}), got {self.shard_index}"
        assert self.shard_by in (
            "context",
            "row",
        ), f"shard_by must be 'context' or 'row', got {self.shard_by!r}"

    def get_results_dir(self, output_dir: Path) -> Path:
        """
        Generates the unique save directory and filenames based on configuration parameters.

        Parameters
        ----------
        output_dir : Path
            The output directory path

        Returns
        -------
        Path
            The path to the results directory
        """
        # A sharded run has its directory chosen once by the driver: every shard must write into
        # the same one, and the uniquifying branch below would otherwise give each shard a
        # different suffix.
        if self.results_dir is not None:
            config_dir = Path(self.results_dir)
            config_dir.mkdir(parents=True, exist_ok=True)
            return config_dir

        # Build directory name components
        components = [
            self.dataset,
            str(self.data_dir) if self.data_dir else "",
            self.model.replace("/", "--"),
            self.press_name,
            f"{self.compression_ratio:.2f}",
        ]

        if self.threshold is not None:
            components[-1] = f"{self.threshold:.2f}"
        elif self.head_compression_ratio is not None:
            components[-1] = f"{self.head_compression_ratio:.2f}"
        if self.fraction < 1.0:
            components.append(f"fraction{self.fraction:.3f}")
        if self.max_context_length is not None:
            components.append(f"max_context{self.max_context_length}")
        if self.query_aware:
            components.append("query_aware")
        if self.enable_thinking:
            # Part of the directory name, not just the saved config: thinking mode changes the
            # PROMPT, so a thinking run and a non-thinking one are different measurements. Without
            # this they collide in one directory, and since run_evaluation skips a directory that
            # already holds predictions.csv + metrics.json, the second run would silently report the
            # first one's numbers.
            components.append("thinking")
        if self.do_sample:
            # Same collision hazard as `thinking`, and worse here: run_evaluation SKIPS a directory
            # that already holds predictions.csv + metrics.json, so without this a sampled run
            # would not merely overwrite the greedy one -- it would decline to run and report the
            # greedy numbers as its own. R is in the name too, since an R=8 mean and an R=2 mean
            # have different variance.
            components.append(f"sample-t{self.temperature:g}-p{self.top_p:g}-k{self.top_k}")
        if self.rollouts > 1:
            components.append(f"r{self.rollouts}")
        if self.key_channel_compression_ratio is not None:
            components.append(f"key_channel_cr{self.key_channel_compression_ratio:.2f}")
        if self.needle_depth is not None and self.dataset == "needle_in_haystack":
            components.append(f"needle_depth{self.needle_depth}")

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

    def save_config(self, config_filename: Path, resumed_traces: int = 0):
        """
        Saves the evaluation configuration to a YAML file.

        ``resumed_traces`` records how many traces were replayed from a progress log rather than
        generated in this process. It is provenance, not decoration: a resumed run reseeds only
        for the traces it still has to generate, so it is NOT bit-identical to an uninterrupted
        run at the same seed, and the paired dense/sparse seed argument holds only for clean runs.
        Recording it means the difference can never surface later as an unexplained mismatch.
        """
        config_dict = asdict(self)
        if resumed_traces:
            config_dict["resumed_traces"] = int(resumed_traces)
        if self.threshold is not None or self.head_compression_ratio is not None:
            config_dict.pop("compression_ratio", None)
        if self.threshold is None:
            config_dict.pop("threshold", None)
        if self.head_compression_ratio is None:
            config_dict.pop("head_compression_ratio", None)
        with open(str(config_filename), "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=False)


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

    def _setup_logging(self):
        """Configures the logging level based on the config."""
        log_level = self.config.log_level.upper()

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(log_level)

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
        press_name = self.config.press_name
        compression_ratio = self.config.compression_ratio
        key_channel_compression_ratio = self.config.key_channel_compression_ratio

        press = PRESS_REGISTRY[press_name]

        # Apply compression ratios based on press type
        if isinstance(press, DuoAttentionPress):
            assert (
                self.config.head_compression_ratio is not None
            ), "head_compression_ratio must be set for DuoAttentionPress"
            press.head_compression_ratio = self.config.head_compression_ratio
            logger.info(f"Set DuoAttentionPress head_compression_ratio to {press.head_compression_ratio}")
        elif isinstance(press, DMSPress):
            assert self.config.threshold is not None, "threshold must be set for DMSPress"
            press.threshold = self.config.threshold
            logger.info(f"Set DMSPress threshold to {press.threshold}")
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
        elif isinstance(press, DecodingPress):
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
        df = load_dataset(DATASET_REGISTRY[dataset_name], data_dir=data_dir, split="test").to_pandas()

        if fraction < 1.0:
            original_len = len(df)
            df = df.sample(frac=fraction, random_state=self.config.seed)
            logger.info(f"Sampled {len(df)} samples ({fraction:.2f}) from original {original_len} samples.")

        logger.info(f"Dataset loaded with {len(df)} entries.")

        # if we have needle in a haystack, we need to insert it in the context
        if self.config.dataset == "needle_in_haystack":
            df = insert_needle_in_haystack(
                df, self.pipeline.tokenizer, self.config.max_context_length, self.config.needle_depth
            )

        if isinstance(self.press, FinchPress):
            if not self.config.query_aware:
                logger.error("FinchPress requires 'query_aware' to be set to True.")
                raise ValueError("FinchPress requires query_aware to be set to True")
            # FinchPress uses a delimiter token to separate context and question
            # So we need to update the tokenizer and the model embeddings.
            logger.info("FinchPress detected, updating model and tokenizer with delimiter token.")
            self.press.update_model_and_tokenizer(self.pipeline.model, self.pipeline.tokenizer)  # type: ignore[attr-defined]
            df["context"] = df["context"] + self.press.delimiter_token  # type: ignore[attr-defined, index]

        if self.config.query_aware:
            logger.info("Query-aware compression: including question in context for compression.")
            df["context"] = df["context"] + df["question"]  # type: ignore[index]
            df["question"] = ""  # type: ignore[index]

        # Shard AFTER sampling and needle insertion, so every shard derives its slice from the
        # identical full frame -- the union over shards is then exactly the unsharded row set.
        if self.config.num_shards > 1:
            full = len(df)
            contexts = df["context"].drop_duplicates()
            if self.config.shard_by == "row":
                # One context shared by every row (math500/aime25): context sharding would put the
                # whole dataset on shard 0. Nothing is lost by splitting rows here -- the "context"
                # is a single space, so there is no prefill to amortize.
                df = df.iloc[self.config.shard_index :: self.config.num_shards]
            else:
                # Round-robin over contexts (not rows): a context's questions share one prefill, so
                # splitting them across shards would re-prefill the same long context in each --
                # which matters more here than in the sparse path, since KVzip's scoring pass costs
                # 2-3x prefill on top.
                mine = set(contexts.iloc[self.config.shard_index :: self.config.num_shards])
                df = df[df["context"].isin(mine)]
            logger.info(
                "Shard %d/%d by %s: %d of %d rows (%d contexts in the full frame)",
                self.config.shard_index,
                self.config.num_shards,
                self.config.shard_by,
                len(df),
                full,
                len(contexts),
            )

        self.df = df
        logger.info(f"Dataset processed with {len(self.df)} entries.")

    def _setup_model_pipeline(self):
        model_name = self.config.model
        device = self.config.device

        if device is None:
            device = "auto" if torch.cuda.is_available() else "cpu"
            logger.info(f"No device specified, auto-detected device: {device}")

        model_kwargs = self.config.model_kwargs or {}

        if self.config.fp8:
            model_kwargs["quantization_config"] = FineGrainedFP8Config()
            logger.info("FP8 quantization enabled.")

        if isinstance(self.press, ObservedAttentionPress):
            model_kwargs["attn_implementation"] = "eager"
            logger.info("ObservedAttentionPress detected, setting attn_implementation to 'eager'.")
        elif self.config.attn_implementation:
            # Explicit wins over autodetection: a flash-attn build that imports but does not match
            # the installed torch returns wrong logits silently, so being able to pin "sdpa" is how
            # that gets ruled out without uninstalling the package.
            model_kwargs["attn_implementation"] = self.config.attn_implementation
            logger.info("Using requested attn_implementation=%r.", self.config.attn_implementation)
        else:
            # DEFAULT TO SDPA, NOT AUTODETECTED FLASH-ATTN. A flash-attn build that imports but does
            # not match the installed torch produces token garbage on long contexts and scores ~0
            # WITHOUT raising -- measured on LongBench gov_report: 0.0 under flash_attention_2 vs
            # 33.55 under sdpa, same checkpoint, same rows. Silently wrong beats loudly broken only
            # for the machine, never for the paper, so the safe implementation is the default and
            # flash-attn must be requested explicitly via --attn_implementation.
            model_kwargs["attn_implementation"] = "sdpa"
            logger.info("No attn_implementation requested; defaulting to 'sdpa' (flash-attn must be explicit).")

        logger.info(f"Loading model pipeline for: {model_name} on device: {device} with model_kwargs: {model_kwargs}")
        pipeline_kwargs = {
            "model": model_name,
            "model_kwargs": model_kwargs,
            "trust_remote_code": True,
        }
        if device == "auto":
            pipeline_kwargs["device_map"] = "auto"
        else:
            pipeline_kwargs["device"] = device
        try:
            self.pipeline = pipeline("kv-press-text-generation", **pipeline_kwargs)
        except (TypeError, ValueError) as e:
            # transformers >=4.56 dev injects `dtype=` into the model constructor when the caller
            # supplies none, and Qwen3ForCausalLM.__init__ rejects it:
            #   TypeError: Qwen3ForCausalLM.__init__() got an unexpected keyword argument 'dtype'
            # Pinning the dtype explicitly avoids the injection. Same try/except shape as
            # evaluate_sparse.py:619-621, which hit the mirror-image version of this.
            #
            # CATCHING ValueError TOO IS LOAD-BEARING. `pipeline()` does not propagate the
            # TypeError: `infer_framework_load_model` catches every per-class failure and re-raises
            # a ValueError ("Could not load model ... See the original errors:") with the original
            # traceback embedded as TEXT. So `except TypeError` never fired and this whole retry was
            # dead code -- evaluate.py simply could not load Qwen3 on this environment. The
            # substring check below is what keeps the wider catch honest: anything not about dtype
            # is re-raised untouched.
            if "dtype" not in str(e):
                raise
            logger.info("pipeline() rejected an injected dtype (%s); retrying with explicit dtype.", str(e)[:200])
            retry = dict(pipeline_kwargs)
            retry_kwargs = dict(model_kwargs)
            retry_kwargs.setdefault("dtype", torch.bfloat16)
            retry["model_kwargs"] = retry_kwargs
            try:
                self.pipeline = pipeline("kv-press-text-generation", **retry)
            except (TypeError, ValueError):
                # Same wrapping as above: an older transformers wants `torch_dtype` and rejects
                # `dtype`, and the rejection again arrives as a ValueError from
                # infer_framework_load_model rather than the underlying TypeError.
                retry_kwargs.pop("dtype", None)
                retry_kwargs["torch_dtype"] = torch.bfloat16
                retry["model_kwargs"] = retry_kwargs
                self.pipeline = pipeline("kv-press-text-generation", **retry)

        self.pipeline.model.eval()
        logger.info("Model pipeline loaded.")

    @torch.inference_mode()
    def _run_inference(self):
        """
        Executes the inference process on the prepared dataset using the model pipeline.
        """

        self.df["predicted_answer"] = None  # type: ignore[index]
        # Sampling rule and rollouts, matching evaluate_sparse.py exactly so the dense reference
        # and the sparse arm are the same measurement apart from compression.
        self.pipeline.sampling = (  # type: ignore[union-attr]
            {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
            }
            if self.config.do_sample
            else None
        )
        if self.config.rollouts > 1:
            # Replicate the FRAME, one row per trace: the scorers already average over rows, so
            # the reported accuracy becomes a pass@1 estimate with no scorer change.
            #
            # `index + r * ROLLOUT_STRIDE`, NOT `ignore_index=True` and NOT a frame-derived offset.
            # The sharded driver unions shards on their index and refuses to score when two shards
            # carry the same index value. Resetting the index made every shard start at 0; a
            # per-shard `max()+1` offset also collides, because each shard holds a different slice
            # (verified: 997 unique out of 1000). Only a constant every shard agrees on composes
            # with sharding. Row identity stays recoverable as `index % ROLLOUT_STRIDE`.
            offset = ROLLOUT_STRIDE
            if int(self.df.index.max()) >= offset:  # type: ignore[union-attr]
                raise ValueError(
                    f"row index {int(self.df.index.max())} exceeds the rollout stride {offset}"  # type: ignore[union-attr]
                )
            self.df = pd.concat(  # type: ignore[assignment]
                [
                    self.df.assign(rollout=r).set_axis(self.df.index + r * offset)  # type: ignore[union-attr]
                    for r in range(self.config.rollouts)
                ]
            )
            logger.info(
                "rollouts=%d: %d rows (%d problems x %d); accuracy is the mean over them, "
                "i.e. pass@1, NOT pass@k",
                self.config.rollouts, len(self.df), len(self.df) // self.config.rollouts,
                self.config.rollouts,
            )

        if isinstance(self.press, DecodingPress):
            logger.info("DecodingPress detected, running inference for each context-question pair.")
            for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Running Inference"):
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
                    enable_thinking=self.config.enable_thinking,
                )
                self.df.loc[index, "predicted_answer"] = output["answer"]  # type: ignore[union-attr]
                torch.cuda.empty_cache()  # Clear CUDA cache to free up memory

        else:
            df_context_grouped = self.df.groupby("context")  # type: ignore[union-attr]
            assert all(
                df_context_grouped["answer_prefix"].nunique() == 1
            ), "Inconsistent 'answer_prefix' within the same context group detected."

            logger.info("Starting inference...")
            # Incremental progress log, mirroring evaluate_sparse.py. One JSON line per completed
            # trace, fsync'd per chunk, replayed on restart so only the missing traces are
            # generated. Without it a run is all-or-nothing: math500/aime25 put every row under
            # ONE context, so a whole shard is a single pipeline() call and a kill at 99% leaves
            # nothing. That cost 6h38m when the filesystem filled mid-run.
            #
            # `_results_dir` is set by run(); when it is None (this method driven directly, as the
            # diagnostics do) the log is simply skipped.
            progress_path = (
                None
                if getattr(self, "_results_dir", None) is None
                else self._results_dir / (
                    f"progress_shard{self.config.shard_index}.jsonl"
                    if self.config.num_shards > 1
                    else "progress.jsonl"
                )
            )
            done: dict[int, str] = {}
            if progress_path is not None and progress_path.exists():
                with open(progress_path) as pfh:
                    for line in pfh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            # A torn final line is the expected result of a hard kill: redo that
                            # one trace rather than refuse to resume.
                            logger.warning("ignoring a truncated final line in %s", progress_path.name)
                            continue
                        done[int(rec["i"])] = rec["a"]
                keep = [i for i in done if i in self.df.index]  # type: ignore[union-attr]
                self.df.loc[keep, "predicted_answer"] = pd.Series({i: done[i] for i in keep})  # type: ignore[union-attr]
                logger.info(
                    "resuming from %s: %d/%d traces already generated, %d to go",
                    progress_path.name, len(keep), len(self.df), len(self.df) - len(keep),
                )
                if keep:
                    self._resumed_from = len(keep)

            pfh = open(progress_path, "a", buffering=1) if progress_path is not None else None
            try:
                for context, df_group in tqdm(
                    df_context_grouped, total=self.df["context"].nunique(), desc="Running Inference"
                ):  # type: ignore[union-attr]
                    # Use max_new_tokens from config, or fallback to dataset's default for the task
                    max_new_tokens = self.config.max_new_tokens or df_group["max_new_tokens"].iloc[0]
                    answer_prefix = df_group["answer_prefix"].iloc[0]

                    # One generate call per rollout index, reseeded from (seed, r) so the traces
                    # are reproducible AND the sparse arm draws the identical seed sequence --
                    # which is what makes the dense-vs-sparse difference paired rather than two
                    # independent samples. With rollouts=1 this is the original single call.
                    #
                    # NOTE: a RESUMED run cannot reproduce the skipped traces' RNG stream, so it is
                    # not bit-identical to an uninterrupted one. Recorded as `resumed_traces` in
                    # the saved config so the difference is never discovered as a mystery later.
                    for rollout, sub in (
                        df_group.groupby("rollout") if "rollout" in df_group else [(0, df_group)]
                    ):
                        todo = sub[sub["predicted_answer"].isna()]
                        if todo.empty:
                            continue
                        if self.config.do_sample:
                            torch.manual_seed(self.config.seed + 1000 * int(rollout))
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(self.config.seed + 1000 * int(rollout))
                        # Chunked so a kill loses at most one chunk. 8 is a compromise: small
                        # enough to bound the loss, large enough that the per-call overhead and
                        # the shared context prefill still amortize.
                        for start in range(0, len(todo), 8):
                            part = todo.iloc[start : start + 8]
                            output = self.pipeline(  # type: ignore[misc]
                                context,
                                questions=part["question"].to_list(),
                                answer_prefix=answer_prefix,
                                press=self.press,
                                max_new_tokens=max_new_tokens,
                                max_context_length=self.config.max_context_length,
                                enable_thinking=self.config.enable_thinking,
                            )
                            answers = output["answers"]
                            self.df.loc[part.index, "predicted_answer"] = answers  # type: ignore[union-attr]
                            # Store the actual compression ratio used (if the press has one)
                            self.df.loc[part.index, "compression_ratio"] = (
                                self.press.compression_ratio if self.press is not None else 0.0  # type: ignore[attr-defined]
                            )  # type: ignore[union-attr, attr-defined]
                            if pfh is not None:
                                for idx, ans in zip(part.index, answers):
                                    pfh.write(json.dumps({"i": int(idx), "a": ans}) + "\n")
                                pfh.flush()
                                os.fsync(pfh.fileno())
                    torch.cuda.empty_cache()  # Clear CUDA cache to free up memory
            finally:
                if pfh is not None:
                    pfh.close()

        logger.info("Inference completed.")

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

        # Resolved once and stashed: _run_inference writes its progress log here. Must not
        # call get_results_dir twice -- it uniquifies against the filesystem and creates the
        # directory, so a second call returns a DIFFERENT path and the log would be orphaned
        # from the run that wrote it (the exact bug hit in evaluate_sparse.py).
        results_dir = self._results_dir = self.config.get_results_dir(output_dir)
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

        if self.config.num_shards > 1:
            # Write the shard and stop. Scoring happens once, over the union, in
            # evaluate_sharded.py -- a per-shard metric would be a per-task mean over an arbitrary
            # subset of rows, which is not comparable to anything.
            #
            # Parquet, not CSV: `answer` holds an ndarray of reference strings and the scorers
            # iterate it. CSV stringifies it to "['2166941']", which then iterates CHARACTER by
            # character -- 11 phantom references -- and a genuinely wrong prediction scores 0.27
            # instead of 0.0. The corruption is silent and inflates the metric.
            shard_file = results_dir / f"predictions_shard{self.config.shard_index}.parquet"
            self.df.to_parquet(str(shard_file), index=True)  # type: ignore[union-attr]
            logger.info(
                "Shard %d wrote %d rows to %s", self.config.shard_index, len(self.df), shard_file  # type: ignore[arg-type]
            )
            return

        self._save_results(predictions_filename)
        self._calculate_and_save_metrics(metrics_filename)
        self.config.save_config(config_filename, getattr(self, "_resumed_from", 0))
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
