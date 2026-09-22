# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from fire import Fire
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.evaluate_registry import DATASET_REGISTRY, SCORER_REGISTRY
from kvpress.indexmem import IndexMemConfig, IndexMemTextGenerationPipeline

logger = logging.getLogger(__name__)
ROLLOUT_STRIDE = 1_000_000


@dataclass
class IndexMemEvaluationConfig:
    scorer_checkpoint: str
    dataset: str = "ruler"
    data_dir: str | None = None
    model: str = "Qwen/Qwen3-8B"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    cache_budget: int = 2048
    cache_budget_ratio: float | None = None
    sink_size: int = 4
    window_size: int = 128
    cmp_slots: int = 64
    head_budget: str = "mass"
    min_head_budget: int = 512
    head_budget_table: str | None = None
    inference_mode: str = "evict"
    decode_batch: int = 1
    fraction: float = 1.0
    max_new_tokens: int | None = None
    max_context_length: int | None = None
    enable_thinking: bool = False
    rollouts: int = 1
    do_sample: bool = False
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    shard_index: int = 0
    num_shards: int = 1
    shard_by: str = "context"
    output_dir: str = "./results_indexmem"
    results_dir: str | None = None
    seed: int = 42

    def get_results_dir(self):
        if self.results_dir:
            path = Path(self.results_dir)
            path.mkdir(parents=True, exist_ok=True)
            return path
        budget = (
            f"ratio{self.cache_budget_ratio:g}" if self.cache_budget_ratio is not None else f"budget{self.cache_budget}"
        )
        name = "__".join(
            filter(
                None,
                [
                    self.dataset,
                    str(self.data_dir) if self.data_dir else "",
                    self.model.replace("/", "--"),
                    budget,
                    self.inference_mode,
                    self.head_budget,
                    f"cmp{self.cmp_slots}",
                    Path(self.scorer_checkpoint).stem,
                ],
            )
        )
        path = Path(self.output_dir) / name
        index = 1
        while path.exists():
            path = Path(self.output_dir) / f"{name}__{index}"
            index += 1
        path.mkdir(parents=True)
        return path


def shard_dataset(frame, shard_index, num_shards, shard_by):
    if shard_by == "row":
        return frame.iloc[shard_index::num_shards]
    contexts = frame["context"].drop_duplicates().iloc[shard_index::num_shards]
    return frame[frame["context"].isin(contexts)]


def expand_rollouts(frame, count):
    if count == 1:
        return frame.copy()
    return pd.concat(
        [frame.assign(rollout=rollout).set_axis(frame.index + rollout * ROLLOUT_STRIDE) for rollout in range(count)],
        verify_integrity=True,
    )


def score_predictions(frame, dataset, results_dir):
    frame.drop(columns=["context"]).to_csv(results_dir / "predictions.csv", index=False)
    metrics = SCORER_REGISTRY[dataset](frame)
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


class IndexMemEvaluationRunner:
    def __init__(self, config):
        self.config = config
        self.pipeline = None
        self.df = None

    def _setup_pipeline(self):
        cfg = self.config
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        model = (
            AutoModelForCausalLM.from_pretrained(
                cfg.model,
                torch_dtype=getattr(torch, cfg.dtype),
                attn_implementation=cfg.attn_implementation,
            )
            .to(cfg.device)
            .eval()
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        inference = IndexMemConfig(
            cache_budget=cfg.cache_budget,
            cache_budget_ratio=cfg.cache_budget_ratio,
            sink_size=cfg.sink_size,
            window_size=cfg.window_size,
            cmp_slots=cfg.cmp_slots,
            head_budget=cfg.head_budget,
            min_head_budget=cfg.min_head_budget,
            head_budget_table=cfg.head_budget_table,
            inference_mode=cfg.inference_mode,
            decode_batch=cfg.decode_batch,
        )
        self.pipeline = IndexMemTextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            scorer_checkpoint=cfg.scorer_checkpoint,
            config=inference,
        )

    def _load_dataset(self):
        cfg = self.config
        frame = load_dataset(
            DATASET_REGISTRY[cfg.dataset],
            data_dir=str(cfg.data_dir) if cfg.data_dir else None,
            split="test",
        ).to_pandas()
        if cfg.fraction < 1.0:
            frame = frame.sample(frac=cfg.fraction, random_state=cfg.seed)
        self.df = shard_dataset(frame, cfg.shard_index, cfg.num_shards, cfg.shard_by)

    @torch.inference_mode()
    def _run_inference(self, results_dir):
        cfg = self.config
        self.df = expand_rollouts(self.df, cfg.rollouts)
        self.df["predicted_answer"] = None
        self.pipeline.sampling = (
            {
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
            }
            if cfg.do_sample
            else None
        )
        groups = self.df.groupby("context")
        chunk_size = cfg.decode_batch if cfg.inference_mode == "evict" else 1
        progress_path = results_dir / f"progress_shard{cfg.shard_index}.jsonl"
        with progress_path.open("w", buffering=1) as progress:
            for context, group in tqdm(groups, total=len(groups), desc="IndexMem"):
                max_tokens = cfg.max_new_tokens or int(group["max_new_tokens"].iloc[0])
                prefix = group["answer_prefix"].iloc[0]
                rollouts = group.groupby("rollout") if "rollout" in group else [(0, group)]
                for rollout, rows in rollouts:
                    if cfg.do_sample:
                        torch.manual_seed(cfg.seed + 1000 * int(rollout))
                    for start in range(0, len(rows), chunk_size):
                        part = rows.iloc[start : start + chunk_size]
                        answers = self.pipeline(
                            context,
                            questions=part["question"].tolist(),
                            answer_prefix=prefix,
                            max_new_tokens=max_tokens,
                            max_context_length=cfg.max_context_length,
                            enable_thinking=cfg.enable_thinking,
                        )["answers"]
                        self.df.loc[part.index, "predicted_answer"] = answers
                        for index, answer in zip(part.index, answers):
                            progress.write(json.dumps({"i": int(index), "a": answer}) + "\n")

    def run(self):
        results_dir = self.config.get_results_dir()
        self._setup_pipeline()
        self._load_dataset()
        self._run_inference(results_dir)
        self.df.to_parquet(results_dir / f"predictions_shard{self.config.shard_index}.parquet", index=True)
        if self.config.num_shards == 1:
            metrics = score_predictions(self.df, self.config.dataset, results_dir)
            (results_dir / "config.yaml").write_text(yaml.safe_dump(asdict(self.config), sort_keys=False))
            logger.info("Metrics: %s", json.dumps(metrics))
        return results_dir


def main(config_file=None, **overrides):
    values = yaml.safe_load(Path(config_file).read_text()) if config_file else {}
    values.update({key: value for key, value in overrides.items() if value is not None})
    logging.basicConfig(level=logging.INFO)
    IndexMemEvaluationRunner(IndexMemEvaluationConfig(**values)).run()


def sharded_main(devices=None, ngpu=None, config_file=None, **overrides):
    values = yaml.safe_load(Path(config_file).read_text()) if config_file else {}
    values.update({key: value for key, value in overrides.items() if value is not None})
    config = IndexMemEvaluationConfig(**values)
    if devices is None:
        gpu_ids = list(range(ngpu or torch.cuda.device_count()))
    elif isinstance(devices, (tuple, list)):
        gpu_ids = list(devices)
    else:
        gpu_ids = [int(value) for value in str(devices).split(",")]
    results_dir = config.get_results_dir()
    processes, logs = [], []
    for shard, gpu in enumerate(gpu_ids):
        shard_values = asdict(config) | {
            "shard_index": shard,
            "num_shards": len(gpu_ids),
            "device": f"cuda:{gpu}",
            "results_dir": str(results_dir),
        }
        config_path = results_dir / f"config_shard{shard}.yaml"
        config_path.write_text(yaml.safe_dump(shard_values, sort_keys=False))
        log = (results_dir / f"shard{shard}.log").open("w")
        logs.append(log)
        command = [
            sys.executable,
            str(Path(__file__).resolve().parents[2] / "evaluation" / "evaluate_indexmem.py"),
            "--config_file",
            str(config_path),
        ]
        processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
    exit_codes = [process.wait() for process in processes]
    for log in logs:
        log.close()
    for process, code in zip(processes, exit_codes):
        if code:
            raise subprocess.CalledProcessError(code, process.args)
    frames = [pd.read_parquet(results_dir / f"predictions_shard{shard}.parquet") for shard in range(len(gpu_ids))]
    frame = pd.concat(frames, verify_integrity=True).sort_index()
    score_predictions(frame, config.dataset, results_dir)
    saved = asdict(config) | {"num_shards": len(gpu_ids), "sharded_devices": gpu_ids}
    (results_dir / "config.yaml").write_text(yaml.safe_dump(saved, sort_keys=False))


if __name__ == "__main__":
    Fire(main)
