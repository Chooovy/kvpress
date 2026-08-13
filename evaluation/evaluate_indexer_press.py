# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evaluate a trained GQA indexer as an **eviction press** on the kvpress benchmarks.

Third of the three evals, and the one that measures the indexer the way the *press* uses it:

* ``evaluate_dense_baseline.sh``  -- no compression at all, the upper bound.
* ``evaluate_indexer_press.py``   -- THIS: the indexer scores keys, the lowest-scoring ones are
  **dropped from the cache**, and every later query sees the same reduced cache. The saving is cache
  residency (memory), and the budget knob is ``--compression_ratio``.
* ``evaluate_sparse.py``          -- the indexer selects **per query** and nothing is dropped. The
  saving is attention FLOPs/bandwidth, and the budget knob is ``--topk``.

The two indexer evals answer different questions, so a number from one is not a number from the
other: eviction is a *permanent* decision made once per key during prefill, while sparse selection
is remade for every query. A key the eviction path discards is gone; under sparse attention it is
still there for the next query. Expect eviction to be the harder setting at matched budget.

    python evaluate_indexer_press.py --dataset ruler --data_dir 8192 \\
        --model /path/Qwen3-8B --indexer_ckpt /path/stage1/step600.pt \\
        --compression_ratio 0.5 --device cuda:0

Why this is not just ``evaluate.py --press_name gqa_indexer``
------------------------------------------------------------
The press builds its indexers in ``post_init_from_model``, which ``BasePress.__call__`` invokes at
*prefill* time. A press taken from ``PRESS_REGISTRY`` therefore attaches **randomly initialized**
indexers unless the trained weights are loaded onto the model first -- and nothing downstream would
flag that, because a random scorer still produces a valid ranking and so still produces plausible
numbers. This script owns that ordering: attach the indexers, load the checkpoint into them, and
only then hand the press to the pipeline.

Sink/local protection (``--n_sink`` / ``--n_local``) is the press's own, applied on top of whatever
the indexer scores; the defaults mirror the sparse eval's reserved slots so the two budgets are
comparable.
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
from transformers import AutoModelForCausalLM, AutoTokenizer

# Run as a plain script without pip-installing the package; must precede evaluate_registry, which
# imports kvpress itself.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.needle_in_haystack.utils import insert_needle_in_haystack  # noqa: E402
from evaluate_registry import DATASET_REGISTRY, SCORER_REGISTRY  # noqa: E402
from kvpress import GQAIndexerPress, load_indexer_state_dict  # noqa: E402
from kvpress.pipeline import KVPressTextGenerationPipeline  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class IndexerPressEvaluationConfig:
    """Configuration for an indexer-as-eviction-press evaluation run."""

    # What to run
    dataset: str = "ruler"
    data_dir: Optional[str] = None
    model: str = "Qwen/Qwen3-8B"
    device: Optional[str] = None
    dtype: str = "bfloat16"
    # "sdpa" rather than flash_attention_2: a flash-attn build that imports but does not match the
    # installed torch returns wrong logits *silently*, and the symptom is every task scoring ~0 in a
    # way that looks like a bad model or a bad metric. Verify with check_attention_backend.py before
    # changing this.
    attn_implementation: str = "sdpa"

    # The trained indexer
    indexer_ckpt: str = ""

    # Eviction budget: the fraction of KV pairs removed. This is the axis to sweep.
    compression_ratio: float = 0.5

    # Press behaviour. n_sink/n_local mirror the sparse eval's force_sink/force_local so the two
    # budgets protect the same boundary tokens; query_reduce is how the (Sq, Sk) indexer logits
    # collapse to one score per key, which only the eviction path needs (a per-query selection has
    # nothing to reduce).
    n_sink: int = 4
    n_local: int = 64
    query_reduce: str = "mean"
    mean_head: bool = False
    chunk_size: int = 0

    # Indexer geometry overrides; leave None to derive from the model exactly as training did.
    n_heads: Optional[int] = None
    head_dim: Optional[int] = None
    rope_dim: Optional[int] = None

    # Dataset / generation
    fraction: float = 1.0
    max_new_tokens: Optional[int] = None
    max_context_length: Optional[int] = None
    needle_depth: Optional[int] = None

    # Output
    output_dir: str = "./results_indexer_press"
    log_level: str = "INFO"
    seed: int = 42

    def __post_init__(self):
        assert self.dataset in DATASET_REGISTRY, f"No dataset found for {self.dataset}"
        assert self.dataset in SCORER_REGISTRY, f"No scorer found for {self.dataset}"
        assert self.indexer_ckpt, "indexer_ckpt is required (the trained indexer checkpoint)"
        assert 0.0 <= self.compression_ratio < 1.0, (
            f"compression_ratio must be in [0, 1), got {self.compression_ratio}"
        )
        assert 0.0 < self.fraction <= 1.0, f"fraction must be in (0, 1], got {self.fraction}"
        if self.dataset == "needle_in_haystack":
            assert self.needle_depth is not None, "needle_depth must be set for needle_in_haystack"
            assert (
                self.max_context_length is not None
            ), "max_context_length must be set for needle_in_haystack"

    def get_results_dir(self) -> Path:
        """Unique results directory, mirroring the other evals' layout."""
        components = [
            self.dataset,
            str(self.data_dir) if self.data_dir else "",
            self.model.replace("/", "--"),
            "indexer_press",
            f"{self.compression_ratio:.2f}",
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


class IndexerPressEvaluationRunner:
    """Load the indexer into a press, generate with cache eviction, and score."""

    def __init__(self, config: IndexerPressEvaluationConfig):
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

        self.pipeline: Optional[KVPressTextGenerationPipeline] = None
        self.press: Optional[GQAIndexerPress] = None
        self.df: Optional[pd.DataFrame] = None
        logger.info("Indexer-press eval config:\n%s", json.dumps(asdict(config), indent=2))

    # ------------------------------------------------------------------
    def _setup_pipeline(self):
        cfg = self.config
        device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = getattr(torch, cfg.dtype)
        logger.info("Loading %s on %s (%s, attn=%s)", cfg.model, device, cfg.dtype, cfg.attn_implementation)

        tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        load_kwargs = {"attn_implementation": cfg.attn_implementation}
        try:
            model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype, **load_kwargs)
        except TypeError:  # older transformers used torch_dtype
            model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=dtype, **load_kwargs)
        model = model.to(device).eval()

        ckpt = torch.load(cfg.indexer_ckpt, map_location="cpu", weights_only=False)
        indexer_sd = ckpt.get("indexer", ckpt)
        # An end-to-end checkpoint carries gate_scale and a distilled one does not; the press must be
        # built to match or load_indexer_state_dict rejects the extra key. Eviction never reads the
        # gate -- only the score's ranking matters -- so this is purely about loading cleanly.
        has_gate = any(str(k).endswith("gate_scale") for k in indexer_sd)

        press = GQAIndexerPress(
            compression_ratio=cfg.compression_ratio,
            gate_scale=has_gate,
            scorer_attr="indexer",
            n_heads=cfg.n_heads,
            head_dim=cfg.head_dim,
            rope_dim=cfg.rope_dim,
            n_sink=cfg.n_sink,
            n_local=cfg.n_local,
            query_reduce=cfg.query_reduce,
            mean_head=cfg.mean_head,
            chunk_size=cfg.chunk_size,
        )
        # ORDER IS LOAD-BEARING. post_init_from_model attaches the indexer modules; the checkpoint is
        # loaded into them afterwards. Doing it the other way round -- or relying on the press's own
        # call inside BasePress.__call__ at prefill time -- would leave randomly initialized scorers
        # in place, which still rank keys and so still produce plausible-looking numbers.
        press.post_init_from_model(model)
        load_indexer_state_dict(model, indexer_sd, "indexer")
        logger.info(
            "Loaded indexer from %s (gate_scale=%s, ckpt step=%s, ckpt config=%s)",
            cfg.indexer_ckpt,
            has_gate,
            ckpt.get("step"),
            ckpt.get("config"),
        )

        self.press = press
        self.pipeline = KVPressTextGenerationPipeline(
            model=model, tokenizer=tokenizer, device=model.device
        )

    def _load_dataset(self):
        cfg = self.config
        data_dir = str(cfg.data_dir) if cfg.data_dir else None
        df = load_dataset(DATASET_REGISTRY[cfg.dataset], data_dir=data_dir, split="test").to_pandas()
        if cfg.fraction < 1.0:
            # Same (fraction, seed) as the other evals => the identical rows, so the three numbers
            # are comparable rather than three different subsets.
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
        for context, group in tqdm(grouped, total=self.df["context"].nunique(), desc="Indexer press"):
            questions = group["question"].to_list()
            max_new_tokens = cfg.max_new_tokens or group["max_new_tokens"].iloc[0]
            answer_prefix = group["answer_prefix"].iloc[0]
            output = self.pipeline(
                context,
                questions=questions,
                answer_prefix=answer_prefix,
                press=self.press,
                max_new_tokens=max_new_tokens,
                max_context_length=cfg.max_context_length,
            )
            self.df.loc[group.index, "predicted_answer"] = output["answers"]
            self.df.loc[group.index, "compression_ratio"] = cfg.compression_ratio
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def run(self):
        results_dir = self.config.get_results_dir()
        self._setup_pipeline()
        self._load_dataset()
        self._run_inference()

        self.df[list(set(self.df.columns) - {"context"})].to_csv(
            str(results_dir / "predictions.csv"), index=False
        )
        metrics = SCORER_REGISTRY[self.config.dataset](self.df)
        with open(results_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        with open(results_dir / "config.yaml", "w") as f:
            yaml.dump(asdict(self.config), f, default_flow_style=False, sort_keys=False)
        logger.info("Metrics:\n%s", json.dumps(metrics, indent=2))
        logger.info("Saved to %s", results_dir)


def main(config_file: Optional[str] = None, **cli_overrides):
    """Build config (dataclass defaults < YAML < CLI) and run."""
    defaults = asdict(IndexerPressEvaluationConfig(indexer_ckpt="_placeholder_"))
    defaults.pop("indexer_ckpt")  # the placeholder only satisfied the required-field assert
    if config_file:
        with open(config_file) as f:
            defaults.update(yaml.safe_load(f) or {})
    defaults.update({k: v for k, v in cli_overrides.items() if v is not None})
    try:
        config = IndexerPressEvaluationConfig(**defaults)
    except TypeError as e:
        print(f"Error: invalid configuration argument. {e}", file=sys.stderr)
        sys.exit(1)
    IndexerPressEvaluationRunner(config).run()


if __name__ == "__main__":
    Fire(main)
