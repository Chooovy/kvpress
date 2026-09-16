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
from kvpress.presses.gqa_indexer.evict_runner import EvictInferenceContext  # noqa: E402
from kvpress.presses.gqa_indexer.train import press_kwargs_from_checkpoint  # noqa: E402
from kvpress.pipeline import KVPressTextGenerationPipeline  # noqa: E402

logger = logging.getLogger(__name__)

#: Index stride between rollouts when ``--rollouts > 1``. Rollout ``r`` of row ``i`` is indexed
#: ``i + r * ROLLOUT_STRIDE``, so the index stays unique across BOTH rollouts and shards, and the
#: original row is recoverable as ``index % ROLLOUT_STRIDE``. It has to be a constant rather than
#: ``frame.index.max() + 1``: every shard holds a different slice, so a frame-derived offset
#: differs per shard and the unioned indices collide -- which the sharded driver correctly refuses
#: to score ("the same row appears in more than one shard"). Larger than any benchmark's row count.
ROLLOUT_STRIDE = 1_000_000


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

    # A JOINT checkpoint's adapted BACKBONE (scripts/train_gqa_indexer_joint.py). Empty means the
    # backbone is whatever --model loads, which is right for every frozen-backbone checkpoint.
    #
    # REQUIRED for a joint checkpoint, and the reason is that nothing else catches its absence:
    # joint training moves 1.51B attention parameters, so the router was optimized against a
    # backbone that --model does not contain. The indexer would load cleanly, the eval would run,
    # and the number would describe a pairing that never existed. Point this at the same file as
    # --indexer_ckpt (a joint payload carries both under "model" and "indexer").
    backbone_ckpt: str = ""

    # The trained linear memory over the evicted keys (kvpress.presses.gqa_indexer.memory). Empty
    # runs plain sparse attention, which is the baseline this arm is measured against. The geometry
    # comes from the checkpoint's recorded config; the router it was trained against is recorded
    # there too and cross-checked against --indexer_ckpt, because a memory read over a different
    # router's support is summarizing keys that router did not evict -- and no weight shape says so.
    memory_ckpt: str = ""

    # Training-free CMP slots (kvpress.presses.gqa_indexer.cmp_slots). 0 disables. R slots are
    # funded OUT OF --topk, so the row still reads topk entries and the A/B against the plain sparse
    # run is budget-matched -- comparing at unmatched budget would measure the budget, not the idea.
    cmp_slots: int = 0
    # How each slot's log-mass is set. "count" is log n_r, which is provably biased LOW by the
    # within-cluster logit variance and therefore SAFE (the slot barely participates); "count+var"
    # adds the 1/2 Var correction estimated from the prefill's own queries. Both are training-free.
    # See cmp_slots.slot_mass: a confident mass on a mediocre direction measured 3-5x WORSE than no
    # slot at all, so "count" is the right first run.
    cmp_mass: str = "count"
    # Constant nats added to every slot, for sweeping the correction by hand.
    cmp_delta: float = 0.0
    # Space the k-means ASSIGNMENT runs in. "post_rope" clusters the cache as stored; "pre_rope"
    # un-rotates first so the partition sees content with the positional phase removed. Only the
    # partition changes -- k_cmp/v_cmp stay means of post-RoPE keys either way, since the slot's
    # logit is q.k_cmp. Measured: post-RoPE clusters are near-contiguous position spans
    # (position_locality 0.07-0.49) while pre-RoPE ones are position-blind (0.31-0.90).
    cmp_space: str = "post_rope"
    # Trained CMPMassHead checkpoint (scripts/train_cmp_mass.py). Supersedes cmp_mass/cmp_delta:
    # b_r = count_coef*log n_r + var_coef*(1/2 Var_r) + bias, 3 scalars per (layer, KV head).
    cmp_mass_ckpt: str = ""

    # Sparse-attention budget (defaults match the sparse training stage)
    topk: int = 512
    # Retain this FRACTION of each document's own context instead of a fixed count. Required for a
    # length-heterogeneous benchmark: LongBench's median context is 10156 tokens, so a fixed
    # topk=16384 ("retain 50% of 32768") leaves 80% of documents uncompressed and the 50%/25% arms
    # scored 47.07 vs 47.06. With a ratio, "retain 50%" means the same thing on every row.
    topk_ratio: Optional[float] = None
    force_sink: int = 4
    force_local: int = 64
    block_k: int = 64

    # How the layer's budget (n_kv_heads * topk) is split across KV heads.
    #   "uniform" -- today's behaviour: every head gets exactly topk.
    #   "mass"    -- every head reaches the same RETAINED ATTENTION MASS, total conserved exactly.
    # Mass is the only currency that is legal here: the trained gate is invariant to a per-(layer,
    # head) constant added to the score, so pooling raw scores across heads (AdaKV) ranks on an
    # unidentifiable quantity, while retained softmax mass is invariant to it. See
    # kvpress/presses/gqa_indexer/head_budget.py.
    head_budget: str = "uniform"
    # Minimum evictable keys per head before the split (TrimKV's min_tokens_per_head). Guards
    # against one reference row starving a head that later rows need.
    head_budget_floor: int = 0
    # Static table fitted offline by scripts/fit_head_budget_table.py. Required by
    # head_budget="static", which reads it instead of measuring anything at prefill.
    head_budget_table: str = ""

    # Physically COMPRESS the cache after the context prefill, instead of keeping it all and
    # masking. Same selection (a set identity -- tests/presses/test_gqa_indexer_evict_cache.py),
    # so the score is comparable to the masking arm; what changes is that decode reads `topk` keys
    # rather than the whole context, and the cache stops scaling with the context length (0.298 GiB
    # at topk=2048 whether the context is 8K or 128K). This is the arm to run when the claim is
    # about decode cost or memory rather than about quality.
    evict: bool = False
    # Questions decoded together under --evict. They share one committed context, so a batch costs
    # one prefill plus a replication per extra question. Decode is memory-bound, so this is where
    # the throughput comes from; raise it until the GPU saturates.
    decode_batch: int = 1


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
    # Qwen3-style thinking mode, matching evaluate.py's flag so the two arms can be compared. False
    # makes the chat template emit a PRE-CLOSED, empty "<think>\n\n</think>", suppressing the
    # reasoning block; True leaves the turn open. Raise max_new_tokens with it -- thinking traces are
    # several times longer, and a truncated trace never reaches \boxed{} at all.
    enable_thinking: bool = False

    # Sampled decoding and pass@1 rollouts, for the reasoning benchmarks.
    #
    # `rollouts=1` with `do_sample=False` (the defaults) is the greedy single-sample path every
    # earlier result used, bitwise. Set both for math500/aime25: those benchmarks cannot be run
    # greedily and stay comparable to the literature -- Qwen3 ships temperature 0.6 / top_p 0.95 /
    # top_k 20 in its own generation_config, thinking traces degenerate into repetition under
    # argmax, and "pass@1" is by definition a mean over sampled rollouts, not one greedy trace.
    #
    # Each rollout re-runs the WHOLE row (fresh prefill, fresh sparse context), so the R traces are
    # independent, and the metric is the mean of the per-rollout scores. Rollout r uses seed
    # `seed + 1000 * r`, so a run is reproducible and two arms (dense vs IndexMem++) draw the same
    # seed sequence -- which is what makes their difference a paired comparison rather than two
    # independent samples.
    rollouts: int = 1
    do_sample: bool = False
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20

    # Reuse an existing results directory instead of uniquifying into `.../1`, so a killed run can
    # pick up its own `progress.jsonl`. OFF by default, because the uniquification is what stops a
    # rerun from silently overwriting a completed result -- turning it off unconditionally would
    # trade a recoverable failure for an unrecoverable one.
    #
    # Without this flag the progress log is nearly useless: `get_results_dir` sends every restart
    # to a NEW directory, so the resume logic looks for a file that is not there and regenerates
    # everything. That is the trap this flag exists to close.
    resume: bool = False

    # Data-parallel sharding. num_shards > 1 makes this process evaluate only its slice of the
    # (already sampled) rows and write predictions to a parquet shard file instead of scoring:
    # a per-shard score is not a score of anything, since RULER's metric is a per-task mean over
    # whatever rows the shard happened to get. evaluate_sparse_sharded.py launches the shards and
    # scores their union. Sharding is by CONTEXT, not by row, so a context's questions stay in one
    # process and are prefilled once -- splitting them would re-prefill the same context per shard.
    shard_index: int = 0
    num_shards: int = 1
    # Sharding axis. "context" is the default and right for every benchmark with long shared
    # contexts. "row" exists for the reasoning benchmarks: math500 and aime25 put the whole problem
    # in `question` and leave `context` a single space, so ALL rows share ONE context and the
    # round-robin over contexts hands every row to shard 0 while the other GPUs sit idle. There is
    # no prefill to share at 4 tokens, so splitting by row costs nothing there.
    shard_by: str = "context"

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
        # Sourced from the press's own registry rather than hand-listed: this assert previously
        # rejected 'prefix' and 'kvzip', which every other line in this file supports, and the bug
        # was invisible because auto-detection passes None.
        from kvpress.presses.gqa_indexer.press import _SCORER_CLASSES

        assert self.scorer is None or self.scorer in _SCORER_CLASSES, (
            f"scorer must be None or one of {sorted(_SCORER_CLASSES)}, got {self.scorer!r}"
        )
        assert self.force_sink + self.force_local <= self.topk, (
            f"force_sink + force_local = {self.force_sink + self.force_local} exceeds topk="
            f"{self.topk}"
        )
        assert self.num_shards >= 1, f"num_shards must be >= 1, got {self.num_shards}"
        assert 0 <= self.shard_index < self.num_shards, (
            f"shard_index must be in [0, {self.num_shards}), got {self.shard_index}"
        )
        assert self.shard_by in ("context", "row"), (
            f"shard_by must be 'context' or 'row', got {self.shard_by!r}"
        )
        if self.dataset == "needle_in_haystack":
            assert self.needle_depth is not None, "needle_depth must be set for needle_in_haystack"
            assert (
                self.max_context_length is not None
            ), "max_context_length must be set for needle_in_haystack"
        if self.evict:
            # `cmp_slots` IS supported under --evict, through the streaming centroid update
            # (kvpress/presses/gqa_indexer/streaming_cmp.py). It behaves differently from the
            # masking arm and that is the point: there the slots are built once at the prefill and
            # then frozen, which on a CoT benchmark means they are never built at all (math500's
            # context is a single space, so the prefill evicts nothing). Here they accumulate the
            # generation's own evicted keys.
            assert not self.memory_ckpt, (
                "--memory_ckpt reads the retained branch's lse, which the paged decode kernel "
                "does not return, and summarizes keys --evict has deleted. Drop one of the two."
            )
            assert self.head_budget in ("uniform", "static"), (
                f"--head_budget {self.head_budget!r} measures the budget from the DOCUMENT's own "
                "attention, so a batch would resolve a different budget per sequence -- and the "
                "paged block table is fixed at allocation. Use 'uniform' or 'static' (both "
                "input-independent) with --evict, or run the masking arm."
            )
            assert self.decode_batch >= 1, (
                f"decode_batch must be >= 1, got {self.decode_batch}"
            )
            if self.topk_ratio is not None and self.decode_batch > 1:
                # Every question in a group shares ONE context, hence one resolved topk, so this
                # is safe -- but only because the grouping is by context. Spelled out because the
                # combination looks dangerous and the guard that would catch it is a raise.
                logger.info(
                    "--topk_ratio with --decode_batch %d: safe here because a batch's questions "
                    "share one context and therefore one resolved budget.",
                    self.decode_batch,
                )


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
            (f"ratio{self.topk_ratio:g}" if self.topk_ratio else f"topk{self.topk}"),
            Path(self.indexer_ckpt).stem,
        ]
        if self.memory_ckpt:
            # Part of the directory name, not just the config blob: without it a memory run and the
            # plain sparse run it is compared against land in the SAME directory, and the second one
            # silently reads as the first's numbers.
            components.append(f"memory-{Path(self.memory_ckpt).stem}")
        if self.cmp_slots:
            # Same reason, and the mass mode belongs in it too: "count" and "count+var" are
            # different measurements of the same checkpoint, so they must not share a directory.
            tag = f"cmp{self.cmp_slots}-{self.cmp_mass}"
            if self.cmp_space != "post_rope":
                tag += f"-{self.cmp_space}"
            if self.cmp_mass_ckpt:
                # In the directory name for the same reason memory-<stem> is: otherwise a learned-mass
                # run and the fixed-mass run it is compared against collide and the second reads as
                # the first's numbers.
                tag += f"-mass{Path(self.cmp_mass_ckpt).stem}"
            if self.cmp_delta:
                tag += f"-d{self.cmp_delta:g}"
            components.append(tag)
        if self.head_budget != "uniform":
            # Same collision hazard as memory-<stem> and cmp<N>: without this the mass-allocated
            # run and the uniform baseline it is compared against land in ONE directory, and the
            # second overwrites the first while looking like a completed A/B.
            tag = f"hb-{self.head_budget}"
            if self.head_budget_floor:
                tag += f"-fl{self.head_budget_floor}"
            if self.head_budget_table:
                tag += f"-{Path(self.head_budget_table).stem}"
            components.append(tag)
        if self.evict:
            # Same collision hazard as memory-<stem>, cmp<N> and hb-<mode>: the whole point of
            # this arm is to be compared against the masking run at the SAME topk, so without a
            # tag the two land in one directory and the second reads as the first's numbers.
            components.append(
                "evict" if self.decode_batch == 1 else f"evict-b{self.decode_batch}"
            )
        if self.fraction < 1.0:
            components.append(f"fraction{self.fraction:.3f}")
        if self.max_context_length is not None:
            components.append(f"max_context{self.max_context_length}")
        if self.needle_depth is not None and self.dataset == "needle_in_haystack":
            components.append(f"needle_depth{self.needle_depth}")
        if self.enable_thinking:
            # Part of the directory name for the same reason as in evaluate.py: thinking mode changes
            # the prompt, so it is a different measurement and must not share a directory with the
            # non-thinking run of the same (dataset, topk).
            components.append("thinking")
        if self.do_sample:
            # Same collision hazard as `thinking`: a sampled run and a greedy run of the same
            # (dataset, topk) are different measurements, and R matters too -- an R=8 mean and an
            # R=2 mean have different variance. Without this the second overwrites the first while
            # looking like a completed comparison.
            components.append(f"sample-t{self.temperature:g}-p{self.top_p:g}-k{self.top_k}")
        if self.rollouts > 1:
            components.append(f"r{self.rollouts}")

        config_dir = Path(self.output_dir) / "__".join(filter(None, components))
        if config_dir.exists():  # never overwrite an existing run
            if self.resume:
                # Opt-in: reuse the directory so `progress.jsonl` is found and only the missing
                # traces are generated. Refuses a COMPLETED run, because "resuming" one would
                # rewrite its metrics.json from a fresh scoring pass and there is no reason to.
                if (config_dir / "metrics.json").exists():
                    raise SystemExit(
                        f"{config_dir} already holds metrics.json -- that run is complete. "
                        "Drop --resume to write a new run beside it, or point --output_dir "
                        "somewhere else."
                    )
                return config_dir
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
            with SparseAttentionContext(self.model, self._sparse_press, **self._sparse_kwargs) as ctx:
                # Per-document budget, when --topk_ratio is set. Must happen before the prefill:
                # the head-budget split and the CMP slots are both derived from topk, and
                # set_context_length clears them when it changes.
                ctx.set_context_length(context_length)
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


class EvictGenerationPipeline(SparseGenerationPipeline):
    """
    The same pipeline, but the cache is **physically compressed** and decode reads only the budget.

    The masking arm above keeps every key and hides the unselected ones, so it measures the
    method's *quality* but not its cost. This arm measures both: after the context prefill the
    cache is compressed to ``topk`` slots per head and never grows again, so decode is a dense
    attention over the budget with no mask at all. The two select the same keys (a set identity --
    ``tests/presses/test_gqa_indexer_evict_cache.py``), so the scores are comparable.

    Two things differ from the masking arm, and both are wins here:

    * **The context is prefilled once per document, not once per question.** The masking arm has to
      re-prefill because its cache is the full context and keeping one per question would not fit;
      a compressed sequence is 0.298 GiB at ``topk=2048``, so it is committed once and *replicated*
      onto the other rows. On a 4-question RULER row that removes 3 of 4 prefills.
    * **The answers are generated as one batch**, which is where eviction pays: decode is
      memory-bound and every sequence now reads ``budget`` keys instead of the whole context.
    """

    def configure_evict(self, *, decode_batch: int) -> None:
        self._decode_batch = int(decode_batch)

    def _forward(self, input_tensors, max_new_tokens=50, press=None, cache=None):
        context_ids = input_tensors["context_ids"].to(self.model.device)
        questions = list(input_tensors["questions_ids"])
        answers: list[str] = []
        # Questions are grouped into batches; every question in a group shares the one committed
        # context, so a group costs ONE prefill plus len(group) - 1 replications.
        group_size = max(1, self._decode_batch)
        for start in range(0, len(questions), group_size):
            group = questions[start : start + group_size]
            with EvictInferenceContext(
                self.model,
                self._sparse_press,
                n_sink=self._sparse_kwargs["force_sink"],
                n_local=self._sparse_kwargs["force_local"],
                batch_size=len(group),
                cmp_slots=self._sparse_kwargs.get("cmp_slots", 0) or 0,
            ) as ec:
                ec.prefill_and_commit(context_ids, 0, self._sparse_kwargs)
                for seq in range(1, len(group)):
                    ec.replicate(0, seq)
                tokens = ec.generate(
                    [q.to(self.model.device) for q in group],
                    max_new_tokens=max_new_tokens,
                    sampling=getattr(self, "sampling", None),
                )
            answers.extend(
                str(self.tokenizer.decode(torch.tensor(t), skip_special_tokens=True))
                for t in tokens
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
        # Resolved lazily in run(), NOT here: get_results_dir() creates the directory as a side
        # effect, so calling it in __init__ would uniquify a second directory for every run and
        # reintroduce exactly the split this replaced (progress log in `.../1`, metrics in the
        # parent). None means "no progress log" -- the state the diagnostic scripts, which call
        # _run_inference without run(), should get.
        self._results_dir: Optional[Path] = None
        self._resumed_from: int = 0
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

        # --- the adapted backbone, for a joint checkpoint -----------------------------------
        # Loaded BEFORE the press is constructed. The press creates the indexer modules, so
        # injecting a backbone afterwards with strict=False would silently drop them (the joint
        # payload's "model" holds indexer keys too, and load_state_dict would overwrite the freshly
        # loaded router with the same values -- harmless here, but the ordering is only obviously
        # correct this way round).
        joint_scope = ckpt_config.get("train_scope")
        if not cfg.backbone_ckpt and joint_scope:
            # Refuse rather than warn. The router was trained against a backbone --model does not
            # contain, so the run would produce a plausible number for a pairing that never
            # existed, and no weight shape or key name would reveal it.
            raise SystemExit(
                f"{cfg.indexer_ckpt} is a JOINT checkpoint (train_scope={joint_scope!r}): its "
                f"router was trained against an ADAPTED backbone, which --model does not have. "
                f"Pass --backbone_ckpt {cfg.indexer_ckpt} to load it. Evaluating the router "
                f"against the pretrained backbone measures a model that never existed."
            )
        if cfg.backbone_ckpt:
            backbone_payload = (
                ckpt
                if Path(cfg.backbone_ckpt) == Path(cfg.indexer_ckpt)
                else torch.load(cfg.backbone_ckpt, map_location="cpu", weights_only=False)
            )
            backbone_sd = backbone_payload.get("model")
            if backbone_sd is None:
                raise SystemExit(
                    f"--backbone_ckpt {cfg.backbone_ckpt} has no 'model' key, so it carries no "
                    f"backbone (keys: {sorted(backbone_payload)[:6]}). A joint checkpoint written "
                    "before the full-model fix stored only the router -- its adapted attention "
                    "weights are unrecoverable and that run has to be redone."
                )
            # Only the backbone: the indexer is loaded separately from `indexer_sd`, after the
            # press has built the modules those keys belong to.
            backbone_only = {k: v for k, v in backbone_sd.items() if ".indexer." not in k}

            # Strip FFN sequence-parallel's wrapper prefix. Training wraps each layer's `mlp` in a
            # SequenceParallelFFN that holds the real module as `self.inner`, so the saved keys read
            # `mlp.inner.gate_proj.weight` while an unwrapped model expects `mlp.gate_proj.weight`
            # (36 layers x 3 projections = 108 of them). That wrapper is a training-time memory
            # optimization and has no business appearing at eval, so it is normalized away here
            # rather than reproduced -- and the count is asserted, because a silent miss would leave
            # the FFN at its pretrained values with only `missing_keys` to show for it.
            wrapped = [k for k in backbone_only if ".mlp.inner." in k]
            if wrapped:
                for key in wrapped:
                    backbone_only[key.replace(".mlp.inner.", ".mlp.")] = backbone_only.pop(key)
                logger.info(
                    "unwrapped %d FFN sequence-parallel key(s): mlp.inner.* -> mlp.* "
                    "(SequenceParallelFFN keeps the real module as .inner; the wrapper is a "
                    "training-time detail and the eval model has no such nesting)",
                    len(wrapped),
                )
            incompatible = model.load_state_dict(backbone_only, strict=False)
            if incompatible.unexpected_keys:
                raise SystemExit(
                    f"--backbone_ckpt has {len(incompatible.unexpected_keys)} keys the model does "
                    f"not accept (e.g. {list(incompatible.unexpected_keys)[:3]}); is it the same "
                    "architecture as --model?"
                )
            logger.info(
                "Loaded ADAPTED backbone from %s (%d tensors, train_scope=%s, step=%s). "
                "%d model keys were left at their pretrained values.",
                cfg.backbone_ckpt, len(backbone_only), joint_scope,
                backbone_payload.get("step"), len(incompatible.missing_keys),
            )
        has_gate = any(str(k).endswith("gate_scale") for k in indexer_sd)
        try:
            scorer, scorer_kwargs = press_kwargs_from_checkpoint(
                indexer_sd, ckpt_config, scorer=cfg.scorer
            )
        except ValueError as exc:
            raise SystemExit(
                f"{exc} Use --scorer with one of: pairwise, scalar, prefix, dma, kvzip, "
                "conv, rnn."
            ) from exc

        if scorer in ("scalar", "prefix", "kvzip", "conv", "rnn"):
            # pos_slope is NOT a parameter -- it is added inside score_keys and never stored -- so
            # a wrong value mis-scores silently with every weight loading cleanly. The CLI wins
            # over the checkpoint's record; if neither has it, say so rather than quietly taking
            # the module default.
            if cfg.scalar_pos_slope is not None:
                scorer_kwargs["scalar_pos_slope"] = cfg.scalar_pos_slope
            elif "scalar_pos_slope" not in scorer_kwargs:
                logger.warning(
                    "checkpoint records no scalar_pos_slope; using the module default. pos_slope "
                    "is not a parameter, so a mismatch against training cannot be detected by "
                    "weight loading -- pass --scalar_pos_slope if training set it."
                )
            # None of these scorers has per-head q/k geometry, and the press rejects these rather
            # than accepting and ignoring them. (Each history arm's own width is read from the
            # weights above -- prefix_head_dim, conv_dim, state_dim -- not from --head_dim.)
            if cfg.head_dim is not None or cfg.rope_dim is not None:
                raise SystemExit(
                    f"--head_dim/--rope_dim do not apply to a {scorer} indexer (its score has no "
                    "per-head q/k pair to shape or rotate). Drop them."
                )
        else:
            scorer_kwargs["head_dim"] = cfg.head_dim
            scorer_kwargs["rope_dim"] = cfg.rope_dim

        # The memory's geometry has to be known before the press is built, since the press is what
        # constructs the modules. Read from the checkpoint rather than exposed as flags: rank and
        # mid_dim ARE parameter shapes, so taking them from anywhere else risks a mismatch that
        # loading would then have to catch.
        memory_kwargs: dict = {}
        memory_sd = None
        memory_config: dict = {}
        if cfg.memory_ckpt:
            memory_payload = torch.load(cfg.memory_ckpt, map_location="cpu", weights_only=False)
            memory_sd = memory_payload.get("memory", memory_payload)
            memory_config = memory_payload.get("config") or {}
            memory_kwargs = {
                "memory": True,
                "memory_rank": memory_config.get("memory_rank", 16),
                "memory_mid_dim": memory_config.get("memory_mid_dim", 256),
                "memory_per_head": memory_config.get("memory_per_head", True),
            }

        press = GQAIndexerPress(
            compression_ratio=0.0,
            gate_scale=has_gate,
            scorer_attr="indexer",
            scorer=scorer,
            n_heads=cfg.n_heads,
            **scorer_kwargs,
            **memory_kwargs,
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

        if memory_sd is not None:
            from kvpress.presses.gqa_indexer.train import load_memory_state_dict

            load_memory_state_dict(model, memory_sd, press.memory_attr)
            trained_router = memory_config.get("init_router")
            if trained_router and Path(trained_router) != Path(cfg.indexer_ckpt):
                # A warning rather than an error, because a moved or copied checkpoint path is a
                # legitimate reason for these to differ. But it is worth saying loudly: the memory
                # summarizes exactly the keys ITS router evicted, so reading it over a different
                # router's support is a silent mismatch -- every tensor loads, every number looks
                # plausible, and the state stands for the wrong set of keys.
                logger.warning(
                    "the memory was trained against router %s but this run uses %s. The memory "
                    "summarizes the keys that router evicted, so a different router's support "
                    "makes it meaningless -- and nothing downstream can detect that.",
                    trained_router,
                    cfg.indexer_ckpt,
                )
            for name in ("force_sink", "force_local"):
                trained = memory_config.get(name)
                if trained is not None and int(trained) != int(getattr(cfg, name)):
                    logger.warning(
                        "the memory was trained at %s=%s but this run uses %s. That changes which "
                        "keys are evicted, so the state is read over a different partition than it "
                        "was built over.",
                        name, trained, getattr(cfg, name),
                    )
            logger.info(
                "Loaded memory from %s (rank=%s, ckpt config=%s)",
                cfg.memory_ckpt,
                memory_kwargs["memory_rank"],
                memory_config or None,
            )

        pipeline_cls = EvictGenerationPipeline if cfg.evict else SparseGenerationPipeline
        pipeline = pipeline_cls(model=model, tokenizer=tokenizer, device=model.device)
        pipeline.configure_sparse(
            press,
            topk=cfg.topk,
            force_sink=cfg.force_sink,
            force_local=cfg.force_local,
            block_k=cfg.block_k,
            precision=cfg.precision,
            memory=bool(cfg.memory_ckpt),
            cmp_slots=cfg.cmp_slots,
            cmp_mass=cfg.cmp_mass,
            cmp_delta=cfg.cmp_delta,
            cmp_space=cfg.cmp_space,
            cmp_mass_ckpt=cfg.cmp_mass_ckpt,
            head_budget=cfg.head_budget,
            head_budget_floor=cfg.head_budget_floor,
            head_budget_table=cfg.head_budget_table,
            topk_ratio=cfg.topk_ratio,
        )
        if cfg.evict:
            pipeline.configure_evict(decode_batch=cfg.decode_batch)
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
            if cfg.shard_by == "row":
                # One context shared by every row (math500/aime25): context sharding would put the
                # whole dataset on shard 0. Nothing is lost by splitting rows here -- the "context"
                # is a single space, so there is no prefill to amortize.
                df = df.iloc[cfg.shard_index :: cfg.num_shards]
            else:
                # Round-robin over contexts (not rows): a context's questions share one prefill, so
                # splitting them across shards would re-prefill the same long context in each.
                mine = set(contexts.iloc[cfg.shard_index :: cfg.num_shards])
                df = df[df["context"].isin(mine)]
            logger.info(
                "Shard %d/%d by %s: %d of %d rows (%d contexts in the full frame)",
                cfg.shard_index,
                cfg.num_shards,
                cfg.shard_by,
                len(df),
                full,
                len(contexts),
            )
        self.df = df
        logger.info("Dataset %s loaded with %d entries", cfg.dataset, len(df))

    @torch.inference_mode()
    def _run_inference(self):
        cfg = self.config
        self.df["predicted_answer"] = None
        # Sampling is configured on the pipeline, not passed per call: `generate_answer` is shared
        # with the eviction eval and its signature is part of the base class's contract.
        self.pipeline.sampling = (
            {"temperature": cfg.temperature, "top_p": cfg.top_p, "top_k": cfg.top_k}
            if cfg.do_sample
            else None
        )
        if cfg.rollouts > 1:
            # R independent traces per row, scored as a mean -- the pass@1 estimator. Replicating
            # the FRAME (rather than looping inside the row loop) keeps every downstream step
            # untouched: the scorers compute a mean over rows, `predictions.csv` keeps one line per
            # trace, and the sharded driver's union stays a plain concatenation. `rollout` is
            # carried as a column so a per-rollout breakdown is recoverable after the fact.
            #
            # The index must stay GLOBALLY UNIQUE, which is why the rollout is folded into it
            # rather than reset. `ignore_index=True` renumbered each shard's rows 0..N-1, so shards
            # 0 and 1 both produced indices 0..143 and the sharded driver's overlap guard correctly
            # refused to score their union ("the same row appears in more than one shard"). The
            # generated rows were fine -- 500 questions x 2 rollouts, verified -- but a union keyed
            # on a colliding index would double-count, and that guard exists precisely because a
            # double-counted union reads exactly like a correct metric.
            #
            # `index + r * ROLLOUT_STRIDE` keeps rollout r of row i distinct from every other row
            # while preserving row identity as `index % ROLLOUT_STRIDE`. The stride is a CONSTANT,
            # not derived from this frame: each shard holds a different row slice, so a
            # frame-derived offset (`index.max() + 1`) differs per shard and the indices collide
            # again -- verified, 997 unique out of 1000. A constant every shard agrees on is the
            # only version that composes with sharding.
            offset = ROLLOUT_STRIDE
            if int(self.df.index.max()) >= offset:
                raise ValueError(
                    f"row index {int(self.df.index.max())} exceeds the rollout stride {offset}; "
                    "raise ROLLOUT_STRIDE or rollout indices would collide with row indices."
                )
            self.df = pd.concat(
                [
                    self.df.assign(rollout=r).set_axis(self.df.index + r * offset)
                    for r in range(cfg.rollouts)
                ]
            )
            logger.info(
                "rollouts=%d: %d rows to generate (%d problems x %d). The reported accuracy is the "
                "mean over all of them, i.e. a pass@1 estimate, NOT pass@k.",
                cfg.rollouts, len(self.df), len(self.df) // cfg.rollouts, cfg.rollouts,
            )
        grouped = self.df.groupby("context")
        assert all(grouped["answer_prefix"].nunique() == 1), "answer_prefix varies within a context"

        # Incremental progress log: one JSON line per completed trace, fsync'd per chunk.
        #
        # Without it a run is all-or-nothing. math500/aime25 put every row under ONE context, so
        # the whole dataset was a single `self.pipeline(...)` call and nothing reached disk until
        # it returned -- when the shared filesystem filled at 09:56 this cost 6h38m of generation
        # with zero recoverable output. The log is append-only and keyed by the strided index, so
        # a restart replays what is already there and generates only the remainder.
        #
        # JSONL rather than parquet: appending to parquet means rewriting the file, which is both
        # slower and not crash-safe at the moment that matters. The parquet/CSV is still written
        # at the end, unchanged, so nothing downstream has to know this file exists.
        # `get_results_dir()` is STATEFUL -- it uniquifies against what exists on disk and creates
        # the directory before returning, so a second call yields a DIFFERENT path (`.../` then
        # `.../1`). Calling it here as well sent the progress log into `/1` while metrics.json
        # went to the parent, orphaning the resume file from the run that wrote it. Resolved once
        # in run(); None when _run_inference is driven directly (diagnostics), in which case the
        # progress log is simply skipped.
        progress_path = (
            None
            if self._results_dir is None
            else self._results_dir / (
                f"progress_shard{cfg.shard_index}.jsonl" if cfg.num_shards > 1 else "progress.jsonl"
            )
        )
        done: dict[int, str] = {}
        if progress_path is not None and progress_path.exists():
            with open(progress_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        # A torn final line is expected after a hard kill; ignore it and redo that
                        # trace rather than refuse to resume.
                        logger.warning("ignoring a truncated final line in %s", progress_path.name)
                        continue
                    done[int(rec["i"])] = rec["a"]
            keep = [i for i in done if i in self.df.index]
            self.df.loc[keep, "predicted_answer"] = pd.Series({i: done[i] for i in keep})
            logger.info(
                "resuming from %s: %d/%d traces already generated, %d to go",
                progress_path.name, len(keep), len(self.df), len(self.df) - len(keep),
            )
            if keep:
                # Stamped onto the saved config so a resumed number is never mistaken for a clean
                # one: the per-(context, rollout) reseed cannot reproduce the RNG stream across an
                # interruption, so the traces are valid samples but not bit-reproducible.
                self._resumed_from = len(keep)

        # Chunk size for flushing. Under --evict the pipeline groups questions into decode_batch
        # batches internally, so matching it means a flush lands on a batch boundary and costs
        # nothing; otherwise flush every question.
        chunk = max(1, int(cfg.decode_batch)) if cfg.evict else 1

        fh = open(progress_path, "a", buffering=1) if progress_path is not None else None
        try:
            for context, group in tqdm(
                grouped, total=self.df["context"].nunique(), desc="Sparse eval"
            ):
                max_new_tokens = cfg.max_new_tokens or group["max_new_tokens"].iloc[0]
                answer_prefix = group["answer_prefix"].iloc[0]
                # Generate per rollout index, reseeding before each: the traces of rollout r are
                # then a function of (seed, r) alone, so a rerun reproduces them and the dense arm
                # draws the identical seed sequence -- making the two arms' difference paired.
                #
                # NOTE ON RESUME AND THE SEED: reseeding happens once per (context, rollout), so a
                # resumed run that skips part of a rollout does NOT reproduce the skipped traces'
                # RNG stream for the remaining ones. The completed traces are replayed from the
                # log verbatim, so the result is still a valid set of samples at the right
                # temperature -- but a resumed run is not bit-identical to an uninterrupted one.
                # Recorded in the config as `resumed` so a number can never be silently attributed
                # to a clean run.
                for rollout, sub in (
                    group.groupby("rollout") if "rollout" in group else [(0, group)]
                ):
                    todo = sub[sub["predicted_answer"].isna()]
                    if todo.empty:
                        continue
                    if cfg.do_sample:
                        torch.manual_seed(cfg.seed + 1000 * int(rollout))
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(cfg.seed + 1000 * int(rollout))
                    for start in range(0, len(todo), chunk):
                        part = todo.iloc[start : start + chunk]
                        output = self.pipeline(
                            context,
                            questions=part["question"].to_list(),
                            answer_prefix=answer_prefix,
                            press=None,
                            max_new_tokens=max_new_tokens,
                            max_context_length=cfg.max_context_length,
                            enable_thinking=cfg.enable_thinking,
                        )
                        answers = output["answers"]
                        self.df.loc[part.index, "predicted_answer"] = answers
                        if fh is not None:
                            for idx, ans in zip(part.index, answers):
                                fh.write(json.dumps({"i": int(idx), "a": ans}) + "\n")
                            fh.flush()
                            os.fsync(fh.fileno())
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            if fh is not None:
                fh.close()

    def run(self):
        # Resolved exactly ONCE: get_results_dir uniquifies against the filesystem and creates the
        # directory, so a second call returns a different path. `_run_inference` reads this for
        # the progress log.
        results_dir = self._results_dir = self.config.get_results_dir()

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
            # Greedy decoding cannot detect a NaN (argmax has no NaN check), so a completed greedy
            # run is NOT evidence of a clean run. Report the count explicitly: nonzero means some
            # traces in this shard were generated from a corrupted distribution.
            try:
                from kvpress.presses.gqa_indexer.evict_runner import nonfinite_logit_count

                nf = nonfinite_logit_count()
                if nf:
                    logger.warning(
                        "NONFINITE_LOGIT_ROWS=%d in shard %d -- some traces are CORRUPT; do not "
                        "report this shard's score without disclosing it",
                        nf, self.config.shard_index,
                    )
                else:
                    logger.info("NONFINITE_LOGIT_ROWS=0 in shard %d (clean)", self.config.shard_index)
            except Exception:  # never let instrumentation break a finished shard
                pass
            return

        predictions_file = results_dir / "predictions.csv"
        metrics_file = results_dir / "metrics.json"
        config_file = results_dir / "config.yaml"

        self.df[list(set(self.df.columns) - {"context"})].to_csv(str(predictions_file), index=False)
        metrics = SCORER_REGISTRY[self.config.dataset](self.df)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        with open(config_file, "w") as f:
            saved = asdict(self.config)
            resumed = getattr(self, "_resumed_from", 0)
            if resumed:
                # Provenance, not decoration. A resumed run replays completed traces from the
                # progress log and reseeds only for the ones it still has to generate, so it is
                # NOT bit-identical to an uninterrupted run at the same seed. Recording it here
                # means the difference can never be discovered later as an unexplained mismatch.
                saved["resumed_traces"] = int(resumed)
            yaml.dump(saved, f, default_flow_style=False, sort_keys=False)
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
