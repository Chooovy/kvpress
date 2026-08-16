# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evaluate a trained GQA indexer as **sparse attention** on the kvpress benchmarks.

The counterpart of :mod:`evaluate`, which measures *eviction* presses. Here nothing is evicted:
the full KV cache is kept and each query attends only to the indexer's per-query top-k keys, via
:class:`~kvpress.presses.gqa_indexer.SparseAttentionContext` (selection only, no gate). Prefill and
every decode step run sparse attention -- the indexer key-cache is maintained across steps -- so
this is the faithful inference-time picture of the method rather than a prefill-only approximation.

The datasets, scoring and answer formatting are reused verbatim from :mod:`evaluate` /
:mod:`evaluate_registry`, so a sparse number sits beside an eviction number on the same task.

    python evaluate_sparse.py --dataset ruler --data_dir 4096 \\
        --model /path/Qwen3-8B --indexer_ckpt /path/stage1/final.pt \\
        --topk 512 --force_local 64 --force_sink 4 --device cuda:0

Loading either objective's checkpoint works: an end-to-end checkpoint carries a ``gate_scale``
parameter and a distilled one does not, so the press is built with ``gate_scale`` matched to what
the checkpoint actually contains (otherwise the strict key check in ``load_indexer_state_dict``
would reject the e2e one). The gate is never read -- selection uses the indexer score only.
"""

import json
import logging
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from fire import Fire
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# Run as a plain script (`python evaluate_sparse.py`) without pip-installing the package: put the
# repo root on sys.path so `import kvpress` resolves, exactly as the training scripts do. Must
# precede the benchmarks / evaluate_registry imports too -- evaluate_registry imports kvpress.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.needle_in_haystack.utils import insert_needle_in_haystack  # noqa: E402
from evaluate_registry import DATASET_REGISTRY, SCORER_REGISTRY  # noqa: E402
from kvpress import (  # noqa: E402
    GQAIndexerPress,
    SparseAttentionContext,
    load_indexer_state_dict,
)
from kvpress.pipeline import KVPressTextGenerationPipeline  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class SparseEvaluationConfig:
    """Configuration for a sparse-attention evaluation run."""

    # What to run
    dataset: str = "ruler"
    data_dir: Optional[str] = None
    model: str = "Qwen/Qwen3-8B"
    device: Optional[str] = None
    dtype: str = "bfloat16"
    # Backbone attention kernel used for the q/k/v the sparse path then consumes. "sdpa" by default
    # rather than flash_attention_2: a flash-attn build that imports but does not match the
    # installed torch returns wrong logits *silently* (the failure looks like "the model scores 0
    # on everything", not like a crash), and this eval should not inherit that risk.
    attn_implementation: str = "sdpa"

    # The trained indexer
    indexer_ckpt: str = ""

    # Sparse-attention budget (defaults match the sparse training stage)
    topk: int = 512
    force_sink: int = 4
    force_local: int = 64
    block_k: int = 64
    # tl.dot precision. "tf32" because q/k/v here are the model's own bf16, and every bf16 value
    # is exact in tf32 -- so the QK dot is bit-identical and the whole kernel matches the fp32
    # reference to the same 7.5e-3 that bf16 output rounding costs anyway. "ieee" forgoes tensor
    # cores entirely, which measured 67.0 s vs 9.4 s per 8K prefill on an H20 for no accuracy.
    precision: str = "tf32"

    # Indexer geometry overrides. Leave None to derive from the model exactly as training did;
    # pass them only if training passed --n-heads/--head-dim/--rope-dim, since a wrong rope_dim
    # is not a parameter shape and would mis-score silently rather than fail to load.
    n_heads: Optional[int] = None
    head_dim: Optional[int] = None
    rope_dim: Optional[int] = None

    # Dataset / generation
    fraction: float = 1.0
    max_new_tokens: Optional[int] = None
    max_context_length: Optional[int] = None
    needle_depth: Optional[int] = None

    # Output
    output_dir: str = "./results_sparse"
    log_level: str = "INFO"
    seed: int = 42

    def __post_init__(self):
        assert self.dataset in DATASET_REGISTRY, f"No dataset found for {self.dataset}"
        assert self.dataset in SCORER_REGISTRY, f"No scorer found for {self.dataset}"
        assert self.indexer_ckpt, "indexer_ckpt is required (the trained indexer checkpoint)"
        assert 0.0 < self.fraction <= 1.0, f"fraction must be in (0, 1], got {self.fraction}"
        assert self.precision in ("ieee", "tf32"), (
            f"precision must be 'ieee' or 'tf32', got {self.precision!r}"
        )
        assert self.force_sink + self.force_local <= self.topk, (
            f"force_sink + force_local = {self.force_sink + self.force_local} exceeds topk="
            f"{self.topk}"
        )
        if self.dataset == "needle_in_haystack":
            assert self.needle_depth is not None, "needle_depth must be set for needle_in_haystack"
            assert (
                self.max_context_length is not None
            ), "max_context_length must be set for needle_in_haystack"

    def get_results_dir(self) -> Path:
        """Unique results directory, mirroring evaluate.py's layout so runs sit side by side."""
        components = [
            self.dataset,
            str(self.data_dir) if self.data_dir else "",
            self.model.replace("/", "--"),
            "sparse_indexer",
            f"topk{self.topk}",
            Path(self.indexer_ckpt).stem,
        ]
        if self.fraction < 1.0:
            components.append(f"fraction{self.fraction:.3f}")
        if self.max_context_length is not None:
            components.append(f"max_context{self.max_context_length}")
        if self.needle_depth is not None and self.dataset == "needle_in_haystack":
            components.append(f"needle_depth{self.needle_depth}")

        config_dir = Path(self.output_dir) / "__".join(filter(None, components))
        if config_dir.exists():  # never overwrite an existing run
            i = 1
            while (config_dir / f"{i}").exists():
                i += 1
            config_dir = config_dir / f"{i}"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir


class SparseGenerationPipeline(KVPressTextGenerationPipeline):
    """
    The kvpress generation pipeline, but attention is the indexer's sparse attention.

    Reuses ``preprocess`` / ``generate_answer`` / ``postprocess`` unchanged -- the chat template,
    ``answer_prefix`` handling and context truncation are therefore identical to the eviction eval.
    Only ``_forward`` changes: each question re-prefills the context inside a fresh
    :class:`SparseAttentionContext`, so the indexer key-cache stays trivially in lockstep with the
    model's KV cache (no cross-question cache reuse to keep synchronized). The incoming ``press``
    argument is ignored -- there is nothing to evict.
    """

    def configure_sparse(self, press: GQAIndexerPress, **sparse_kwargs) -> None:
        self._sparse_press = press
        self._sparse_kwargs = sparse_kwargs

    def _forward(self, input_tensors, max_new_tokens=50, press=None, cache=None):
        context_ids = input_tensors["context_ids"].to(self.model.device)
        context_length = context_ids.shape[1]
        answers = []
        for question_ids in input_tensors["questions_ids"]:
            with SparseAttentionContext(self.model, self._sparse_press, **self._sparse_kwargs):
                fresh = DynamicCache()
                # Prefill the context under sparse attention (no lm head, matching the base class).
                self.model.model(input_ids=context_ids, past_key_values=fresh)
                answers.append(
                    self.generate_answer(
                        question_ids=question_ids.to(self.model.device),
                        cache=fresh,
                        context_length=context_length,
                        max_new_tokens=max_new_tokens,
                    )
                )
        return answers


class SparseEvaluationRunner:
    """Load the indexer, run sparse-attention generation over a dataset, and score it."""

    def __init__(self, config: SparseEvaluationConfig):
        self.config = config
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(config.log_level.upper())

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.pipeline: Optional[SparseGenerationPipeline] = None
        self.df: Optional[pd.DataFrame] = None
        logger.info("Sparse eval config:\n%s", json.dumps(asdict(config), indent=2))

    # ------------------------------------------------------------------
    def _setup_pipeline(self):
        cfg = self.config
        device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = getattr(torch, cfg.dtype)
        logger.info("Loading %s on %s (%s)", cfg.model, device, cfg.dtype)

        tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        load_kwargs = {"attn_implementation": cfg.attn_implementation}
        try:
            model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype, **load_kwargs)
        except TypeError:  # older transformers used torch_dtype
            model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=dtype, **load_kwargs)
        model = model.to(device).eval()
        logger.info("Backbone attention: %s", model.config._attn_implementation)

        # Load the indexer. gate_scale is matched to what the checkpoint holds so an e2e checkpoint
        # (which has the parameter) and a distilled one (which does not) both load.
        ckpt = torch.load(cfg.indexer_ckpt, map_location="cpu", weights_only=False)
        indexer_sd = ckpt.get("indexer", ckpt)
        has_gate = any(str(k).endswith("gate_scale") for k in indexer_sd)
        press = GQAIndexerPress(
            compression_ratio=0.0,
            gate_scale=has_gate,
            scorer_attr="indexer",
            n_heads=cfg.n_heads,
            head_dim=cfg.head_dim,
            rope_dim=cfg.rope_dim,
        )
        press.post_init_from_model(model)
        load_indexer_state_dict(model, indexer_sd, "indexer")
        logger.info(
            "Loaded indexer from %s (gate_scale=%s, ckpt config=%s)",
            cfg.indexer_ckpt,
            has_gate,
            ckpt.get("config"),
        )

        pipeline = SparseGenerationPipeline(model=model, tokenizer=tokenizer, device=model.device)
        pipeline.configure_sparse(
            press,
            topk=cfg.topk,
            force_sink=cfg.force_sink,
            force_local=cfg.force_local,
            block_k=cfg.block_k,
            precision=cfg.precision,
        )
        self.pipeline = pipeline

    def _load_dataset(self):
        cfg = self.config
        data_dir = str(cfg.data_dir) if cfg.data_dir else None
        df = load_dataset(DATASET_REGISTRY[cfg.dataset], data_dir=data_dir, split="test").to_pandas()
        if cfg.fraction < 1.0:
            df = df.sample(frac=cfg.fraction, random_state=cfg.seed)
        if cfg.dataset == "needle_in_haystack":
            df = insert_needle_in_haystack(
                df, self.pipeline.tokenizer, cfg.max_context_length, cfg.needle_depth
            )
        self.df = df
        logger.info("Dataset %s loaded with %d entries", cfg.dataset, len(df))

    @torch.inference_mode()
    def _run_inference(self):
        cfg = self.config
        self.df["predicted_answer"] = None
        grouped = self.df.groupby("context")
        assert all(grouped["answer_prefix"].nunique() == 1), "answer_prefix varies within a context"
        for context, group in tqdm(grouped, total=self.df["context"].nunique(), desc="Sparse eval"):
            questions = group["question"].to_list()
            max_new_tokens = cfg.max_new_tokens or group["max_new_tokens"].iloc[0]
            answer_prefix = group["answer_prefix"].iloc[0]
            output = self.pipeline(
                context,
                questions=questions,
                answer_prefix=answer_prefix,
                press=None,
                max_new_tokens=max_new_tokens,
                max_context_length=cfg.max_context_length,
            )
            self.df.loc[group.index, "predicted_answer"] = output["answers"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def run(self):
        results_dir = self.config.get_results_dir()
        predictions_file = results_dir / "predictions.csv"
        metrics_file = results_dir / "metrics.json"
        config_file = results_dir / "config.yaml"

        self._setup_pipeline()
        self._load_dataset()
        self._run_inference()

        self.df[list(set(self.df.columns) - {"context"})].to_csv(str(predictions_file), index=False)
        metrics = SCORER_REGISTRY[self.config.dataset](self.df)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        with open(config_file, "w") as f:
            yaml.dump(asdict(self.config), f, default_flow_style=False, sort_keys=False)
        logger.info("Metrics:\n%s", json.dumps(metrics, indent=2))
        logger.info("Saved to %s", results_dir)


def main(config_file: Optional[str] = None, **cli_overrides):
    """Build config (dataclass defaults < YAML < CLI) and run."""
    final_args = asdict(SparseEvaluationConfig(indexer_ckpt="_placeholder_"))
    final_args.pop("indexer_ckpt")  # placeholder only satisfied the required-field assert above
    if config_file:
        with open(config_file) as f:
            final_args.update(yaml.safe_load(f) or {})
    final_args.update({k: v for k, v in cli_overrides.items() if v is not None})
    try:
        config = SparseEvaluationConfig(**final_args)
    except TypeError as e:
        print(f"Error: invalid configuration argument. {e}", file=sys.stderr)
        sys.exit(1)
    SparseEvaluationRunner(config).run()


if __name__ == "__main__":
    Fire(main)
