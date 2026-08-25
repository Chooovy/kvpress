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

To split one configuration's rows across several GPUs, do not run this script directly with
``--num_shards``: a shard writes predictions and deliberately does not score, since a per-shard
metric is a per-task mean over an arbitrary subset. Use :mod:`evaluate_sparse_sharded`, which
launches the shards and scores their union once.

Loading either objective's checkpoint works: an end-to-end checkpoint carries a ``gate_scale``
parameter and a distilled one does not, so the press is built with ``gate_scale`` matched to what
the checkpoint actually contains (otherwise the strict key check in ``load_indexer_state_dict``
would reject the e2e one). The gate is never read -- selection uses the indexer score only.
"""

import json
import logging
import os
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
from kvpress.presses.gqa_indexer.press import get_language_model  # noqa: E402
from kvpress.presses.gqa_indexer.train import detect_scorer, infer_scalar_mid_dim  # noqa: E402
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
    # Whole-CHUNK selection instead of per-token. 0 = token-level (the default, and what the gated
    # arm should use). Set it to the chunk_size the router trained with -- an exact-K checkpoint
    # records that in its config, and `chunk_size=-1` reads it from there automatically.
    #
    # Why it matters: a chunk-trained score is close to piecewise-constant (measured within/across
    # chunk variance ratio 0.16 vs the gated arm's 0.99), so a token-level top-k resolves a near-tie
    # inside every kept chunk -- it measures a resolution the router never learned. See
    # kvpress/presses/gqa_indexer/chunk_support.py.
    chunk_size: int = 0
    # "auto" (default) reads it from the checkpoint. NOT hardcoded: chunk_size and
    # chunk_score_scale already resolve from the ckpt, so leaving this one behind would score an
    # lse-trained router with mean aggregation -- a silent mismatch of exactly the kind chunk_size
    # is guarded against. Pass "lse"/"mean"/"max" to override deliberately.
    chunk_aggregate: str = "auto"
    # Token-score multiplier inside the chunk aggregation. -1 resolves from the checkpoint (the HSA
    # arm records its score_scale); only meaningful for chunk_aggregate="lse", which is not
    # scale-equivariant. Ignored on the token-level path, which ranks raw scores.
    chunk_score_scale: float = -1.0

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

    # Which scorer the checkpoint holds. None (default) reads it from the checkpoint -- its
    # recorded config when present, otherwise its weight names, which are disjoint between the
    # two scorers. Set it only to override that detection.
    scorer: Optional[str] = None
    # Scalar-scorer recency tilt. None takes the checkpoint's recorded value. Worth overriding
    # only for a checkpoint written before the field existed: pos_slope is added to the score
    # and never stored as a parameter, so a mismatch cannot be caught by weight loading.
    # mid_dim is deliberately absent -- it is w_in's shape, so it is read from the weights.
    scalar_pos_slope: Optional[float] = None

    # Dataset / generation
    fraction: float = 1.0
    max_new_tokens: Optional[int] = None
    max_context_length: Optional[int] = None
    needle_depth: Optional[int] = None

    # Data-parallel sharding. num_shards > 1 makes this process evaluate only its slice of the
    # (already sampled) rows and write predictions to a parquet shard file instead of scoring:
    # a per-shard score is not a score of anything, since RULER's metric is a per-task mean over
    # whatever rows the shard happened to get. evaluate_sparse_sharded.py launches the shards and
    # scores their union. Sharding is by CONTEXT, not by row, so a context's questions stay in one
    # process and are prefilled once -- splitting them would re-prefill the same context per shard.
    shard_index: int = 0
    num_shards: int = 1

    # Output
    output_dir: str = "./results_sparse"
    # Set by the sharded driver so every shard writes into the run directory it chose. Bypasses
    # get_results_dir's uniquification, which N concurrent processes would otherwise race on --
    # each testing "does this dir exist" and some landing on `.../1`, others on `.../2`.
    results_dir: Optional[str] = None
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
        assert self.scorer in (None, "pairwise", "scalar"), (
            f"scorer must be None, 'pairwise' or 'scalar', got {self.scorer!r}"
        )
        assert self.force_sink + self.force_local <= self.topk, (
            f"force_sink + force_local = {self.force_sink + self.force_local} exceeds topk="
            f"{self.topk}"
        )
        assert self.num_shards >= 1, f"num_shards must be >= 1, got {self.num_shards}"
        assert self.chunk_size >= -1, f"chunk_size must be >= -1, got {self.chunk_size}"
        assert self.chunk_aggregate in ("auto", "lse", "mean", "max"), (
            f"chunk_aggregate must be 'auto', 'lse', 'mean' or 'max', got {self.chunk_aggregate!r}"
        )
        assert 0 <= self.shard_index < self.num_shards, (
            f"shard_index must be in [0, {self.num_shards}), got {self.shard_index}"
        )
        if self.dataset == "needle_in_haystack":
            assert self.needle_depth is not None, "needle_depth must be set for needle_in_haystack"
            assert (
                self.max_context_length is not None
            ), "max_context_length must be set for needle_in_haystack"

    def get_results_dir(self) -> Path:
        """Unique results directory, mirroring evaluate.py's layout so runs sit side by side."""
        if self.results_dir is not None:
            # Chosen by the sharded driver, which already uniquified it once for all shards.
            config_dir = Path(self.results_dir)
            config_dir.mkdir(parents=True, exist_ok=True)
            return config_dir
        components = [
            self.dataset,
            str(self.data_dir) if self.data_dir else "",
            self.model.replace("/", "--"),
            "sparse_indexer",
            f"topk{self.topk}",
            # In the directory name so a chunk-wise run is never confused with a token-wise one at
            # the same length and budget -- they are different operators, not different seeds.
            # "chunkauto" for -1: get_results_dir runs before the checkpoint is read, so the
            # resolved value is not known yet, and "chunk-1" reads like a negative chunk size. The
            # resolved value is logged and lands in config.yaml either way.
            ("chunkauto" if self.chunk_size < 0 else f"chunk{self.chunk_size}")
            if self.chunk_size else "",
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



def load_cached_dataset(repo_id: str, data_dir: Optional[str]):
    """
    Load a HuggingFace dataset from the local cache when ``data_dir`` cannot be resolved offline.

    Why this is needed. ``load_dataset(repo, data_dir="8192")`` hashes ``data_dir`` into the cache
    key, and that hash is only computed from the *remote* builder script. With no network,
    ``datasets`` raises ``Couldn't find cache for <repo> for config 'default-data_dir=8192'`` and
    lists the hashes it does have -- which are opaque, so it cannot tell which one is 8192.

    Rather than hardcode a hash (they differ per machine and per download), the length is
    **measured**: RULER's ``data_dir`` *is* the context length in tokens, so the cached config whose
    median context is closest to it is the right one. Verified on this box against a Qwen3 tokenizer
    -- the two cached configs measure 7849 and 15932 median context tokens, mapping to 8192 and
    16384. A rough char/token ratio is used here instead of a real tokenizer, since the two
    candidates differ by 2x and the decision has enormous margin.

    Raises if the match is not within 40%, rather than silently evaluating at the wrong length: a
    16K number reported as an 8K one is a result that looks fine and means something else.
    """
    import glob
    import statistics

    from datasets import Dataset

    if data_dir is None:
        raise ValueError(f"cannot resolve {repo_id} from cache without a data_dir")
    target = float(data_dir)
    cache_root = Path(
        os.environ.get("HF_DATASETS_CACHE")
        or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "datasets"
    )
    pattern = str(cache_root / repo_id.replace("/", "___") / "*" / "*" / "*" / "*.arrow")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise ValueError(
            f"no cached arrow files for {repo_id} under {cache_root}. With no network this cannot "
            f"be downloaded; copy the cache from a machine that has it."
        )

    best, best_gap, measured = None, float("inf"), {}
    for path in candidates:
        dataset = Dataset.from_file(path)
        if "context" not in dataset.column_names:
            continue
        step = max(1, len(dataset) // 20)
        # ~4 chars per token for English, which is plenty to separate configs that differ by 2x.
        tokens = statistics.median(
            len(dataset[i]["context"]) / 4.0 for i in range(0, len(dataset), step)
        )
        measured[path] = tokens
        gap = abs(tokens - target) / target
        if gap < best_gap:
            best, best_gap = dataset, gap

    if best is None or best_gap > 0.4:
        raise ValueError(
            f"no cached config of {repo_id} matches data_dir={data_dir} (closest is "
            f"{best_gap:.0%} off). Measured median context tokens per cache entry: "
            f"{ {Path(k).parent.name[:12]: int(v) for k, v in measured.items()} }. Refusing to "
            f"evaluate at a length other than the one requested."
        )
    logger.warning(
        "resolved %s data_dir=%s from the local cache by measuring context length (%.0f%% off "
        "target); `datasets` could not map data_dir to a cache hash offline.",
        repo_id, data_dir, 100 * best_gap,
    )
    return best


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

        # Load the indexer. Both the scorer and gate_scale are read from the checkpoint rather
        # than configured here: a pairwise and a scalar indexer share the parameter *prefix* but
        # agree on no weight names, so guessing wrong fails with "216 keys are absent from the
        # model" rather than anything that names the real problem. gate_scale is matched the same
        # way, so an e2e checkpoint (which has it) and a distilled one (which does not) both load.
        ckpt = torch.load(cfg.indexer_ckpt, map_location="cpu", weights_only=False)
        indexer_sd = ckpt.get("indexer", ckpt)
        ckpt_config = ckpt.get("config") or {}
        has_gate = any(str(k).endswith("gate_scale") for k in indexer_sd)
        try:
            scorer = cfg.scorer or detect_scorer(indexer_sd, ckpt_config)
        except ValueError as exc:
            raise SystemExit(f"{exc} Use --scorer pairwise or --scorer scalar.") from exc

        scorer_kwargs: dict = {}
        if scorer == "scalar":
            # mid_dim is recoverable from w_in's shape, but pos_slope is NOT a parameter -- it is
            # added to the score at :meth:`ScalarIndexer.score_keys` and never stored. So a wrong
            # value here mis-scores silently, with every weight loading cleanly. Prefer the
            # checkpoint's record, and say so when falling back to the default.
            scorer_kwargs["scalar_mid_dim"] = infer_scalar_mid_dim(indexer_sd, ckpt_config)
            if cfg.scalar_pos_slope is not None:
                scorer_kwargs["scalar_pos_slope"] = cfg.scalar_pos_slope
            elif "scalar_pos_slope" in ckpt_config:
                scorer_kwargs["scalar_pos_slope"] = float(ckpt_config["scalar_pos_slope"])
            else:
                logger.warning(
                    "checkpoint records no scalar_pos_slope; using the ScalarIndexer default. "
                    "pos_slope is not a parameter, so a mismatch against training cannot be "
                    "detected by weight loading -- pass --scalar_pos_slope if training set it."
                )
            # The scalar score has no per-head q/k geometry, and the press rejects these rather
            # than accepting and ignoring them.
            if cfg.head_dim is not None or cfg.rope_dim is not None:
                raise SystemExit(
                    "--head_dim/--rope_dim do not apply to a scalar indexer (it has no q/k pair "
                    "to shape or rotate). Drop them."
                )
        else:
            scorer_kwargs["head_dim"] = cfg.head_dim
            scorer_kwargs["rope_dim"] = cfg.rope_dim

        press = GQAIndexerPress(
            compression_ratio=0.0,
            gate_scale=has_gate,
            scorer_attr="indexer",
            scorer=scorer,
            n_heads=cfg.n_heads,
            **scorer_kwargs,
        )
        press.post_init_from_model(model)
        load_indexer_state_dict(model, indexer_sd, "indexer")
        logger.info(
            "Loaded indexer from %s (scorer=%s, gate_scale=%s, ckpt config=%s)",
            cfg.indexer_ckpt,
            scorer,
            has_gate,
            ckpt_config or None,
        )

        chunk_size = cfg.chunk_size
        if chunk_size == -1:
            # Read the training geometry from the checkpoint rather than making the caller restate
            # it. chunk_size is NOT a parameter shape, so a mismatch cannot be caught by weight
            # loading -- exactly why the trainer records it.
            chunk_size = int(ckpt_config.get("chunk_size") or 0)
            if chunk_size:
                logger.info(
                    "chunk_size=-1 resolved to %d from the checkpoint (objective=%s). Selection is "
                    "whole-chunk, matching how this router was trained.",
                    chunk_size, ckpt_config.get("objective"),
                )
            else:
                logger.warning(
                    "chunk_size=-1 but the checkpoint records no chunk_size (objective=%s); falling "
                    "back to token-level selection.", ckpt_config.get("objective"),
                )
        elif chunk_size and ckpt_config.get("chunk_size") and chunk_size != ckpt_config["chunk_size"]:
            logger.warning(
                "chunk_size=%d differs from the checkpoint's %s. That is a train/inference "
                "granularity mismatch; pass -1 to use the recorded value.",
                chunk_size, ckpt_config["chunk_size"],
            )

        chunk_aggregate = cfg.chunk_aggregate
        if chunk_aggregate == "auto":
            chunk_aggregate = str(ckpt_config.get("chunk_aggregate") or "mean")
            if chunk_size:
                logger.info(
                    "chunk_aggregate='auto' resolved to %r from the checkpoint (objective=%s). "
                    "Scoring an lse-trained router with mean aggregation would rank on a different "
                    "functional than training optimized, and nothing downstream would flag it.",
                    chunk_aggregate, ckpt_config.get("objective"),
                )
        elif chunk_size and ckpt_config.get("chunk_aggregate") and (
            chunk_aggregate != ckpt_config["chunk_aggregate"]
        ):
            logger.warning(
                "chunk_aggregate=%r differs from the checkpoint's %r -- a train/inference mismatch "
                "in the aggregation functional. Pass 'auto' to use the recorded value.",
                chunk_aggregate, ckpt_config["chunk_aggregate"],
            )

        chunk_score_scale = cfg.chunk_score_scale
        if chunk_score_scale < 0:
            # Resolve from the checkpoint. Like chunk_size this is not a parameter shape, and for the
            # `lse` aggregation it is NOT cosmetic: logsumexp is not scale-equivariant, so a wrong
            # value ranks on a different functional than training optimized (measured Spearman
            # against the true chunk LSE: 1.000 with the trained scale inside the reduction, 0.65
            # with it outside). Falls back to the indexer's own head_dim ** -0.5, which is the
            # trainer's default and the backbone's attention scale.
            recorded = ckpt_config.get("score_scale")
            if recorded:
                chunk_score_scale = float(recorded)
            else:
                indexer = press.get_indexer(get_language_model(model).layers[0].self_attn)
                head_dim = getattr(indexer, "head_dim", None)
                chunk_score_scale = float(head_dim**-0.5) if head_dim else 1.0
            if chunk_size and cfg.chunk_aggregate == "lse":
                logger.info(
                    "chunk_score_scale resolved to %.6g (%s). This is a TEMPERATURE for the lse "
                    "aggregation, not a cosmetic factor.",
                    chunk_score_scale,
                    "from the checkpoint" if recorded else "head_dim ** -0.5 fallback",
                )

        pipeline = SparseGenerationPipeline(model=model, tokenizer=tokenizer, device=model.device)
        pipeline.configure_sparse(
            press,
            topk=cfg.topk,
            force_sink=cfg.force_sink,
            force_local=cfg.force_local,
            block_k=cfg.block_k,
            precision=cfg.precision,
            chunk_size=chunk_size,
            chunk_aggregate=chunk_aggregate,
            chunk_score_scale=chunk_score_scale,
        )
        self.pipeline = pipeline

    def _load_dataset(self):
        cfg = self.config
        data_dir = str(cfg.data_dir) if cfg.data_dir else None
        try:
            df = load_dataset(
                DATASET_REGISTRY[cfg.dataset], data_dir=data_dir, split="test"
            ).to_pandas()
        except ValueError as exc:
            if "Couldn't find cache" not in str(exc):
                raise
            # Offline box: the arrow files are present but `datasets` cannot map data_dir to a
            # cache hash. Resolve it by measuring the contexts instead -- see
            # :func:`load_cached_dataset`.
            df = load_cached_dataset(DATASET_REGISTRY[cfg.dataset], data_dir).to_pandas()
        if cfg.fraction < 1.0:
            df = df.sample(frac=cfg.fraction, random_state=cfg.seed)
        if cfg.dataset == "needle_in_haystack":
            df = insert_needle_in_haystack(
                df, self.pipeline.tokenizer, cfg.max_context_length, cfg.needle_depth
            )
        # Shard AFTER sampling and needle insertion, so every shard derives its slice from the
        # identical full frame -- the union over shards is then exactly the unsharded row set.
        if cfg.num_shards > 1:
            full = len(df)
            contexts = df["context"].drop_duplicates()
            # Round-robin over contexts (not rows): a context's questions share one prefill, so
            # splitting them across shards would re-prefill the same long context in each.
            mine = set(contexts.iloc[cfg.shard_index :: cfg.num_shards])
            df = df[df["context"].isin(mine)]
            logger.info(
                "Shard %d/%d: %d of %d rows (%d of %d contexts)",
                cfg.shard_index,
                cfg.num_shards,
                len(df),
                full,
                len(mine),
                len(contexts),
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

        self._setup_pipeline()
        self._load_dataset()
        self._run_inference()

        if self.config.num_shards > 1:
            # Write the shard and stop. Scoring happens once, over the union, in
            # evaluate_sparse_sharded.py -- a per-shard metric would be a per-task mean over an
            # arbitrary subset of rows, which is not comparable to anything.
            #
            # Parquet, not CSV: `answer` holds an ndarray of reference strings and the scorers
            # iterate it. CSV stringifies it to "['2166941']", which then iterates CHARACTER by
            # character -- 11 phantom references -- and a genuinely wrong prediction scores 0.27
            # instead of 0.0. The corruption is silent and inflates the metric.
            shard_file = results_dir / f"predictions_shard{self.config.shard_index}.parquet"
            self.df.to_parquet(str(shard_file), index=True)
            logger.info("Shard %d wrote %d rows to %s", self.config.shard_index, len(self.df), shard_file)
            return

        predictions_file = results_dir / "predictions.csv"
        metrics_file = results_dir / "metrics.json"
        config_file = results_dir / "config.yaml"

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
