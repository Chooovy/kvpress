# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data-parallel driver for :mod:`evaluate`: split ONE (dataset, length, press) configuration across
N GPUs and score the union.

The eviction-press counterpart of :mod:`evaluate_sparse_sharded`. That driver only works for the
GQA-indexer *sparse attention* path -- ``evaluate_sparse.py`` requires an ``--indexer_ckpt`` and
never builds a press -- so an ordinary eviction press (``kvzip``, ``snapkv``, ...) had no way to use
more than one GPU for a single configuration. ``evaluate_dense_baseline.sh`` parallelizes over
*lengths*, one per GPU, which leaves 7 GPUs idle whenever you want one number.

That is worth fixing specifically for KVzip: its scoring pass is a second, chunked forward over the
whole context, so it costs 2-3x prefill on top of generation. At RULER 16K a single GPU is hours.

    python evaluate_sharded.py --dataset ruler --data_dir 8192 \\
        --model /path/Qwen3-8B --press_name kvzip --compression_ratio 0.75 \\
        --fraction 0.1 --ngpu 8

Each shard is a separate ``evaluate.py`` process pinned to one GPU (subprocesses, not threads: one
CUDA context per process, and the model is loaded per GPU anyway). Shards write parquet prediction
files; this driver concatenates them and scores ONCE over the union, so the metric is identical to
what the unsharded run would have produced.

The design notes in :mod:`evaluate_sparse_sharded` apply verbatim and are the reason for each
choice here: shard by *context* rather than by row (a context's questions share one prefill),
*parquet* rather than CSV for the shard files (CSV stringifies RULER's ndarray of reference answers
into ``"['2166941']"``, which the scorers then iterate character by character -- 11 phantom
references, so a wrong prediction scores 0.27 instead of 0.0), the *driver* picks the results
directory (otherwise N processes race in ``get_results_dir`` and land on different uniquified
suffixes), and a failed shard is *fatal* (scoring the survivors' union silently reports a metric
over a subset).
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import yaml
from fire import Fire

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate import EvaluationConfig, _load_yaml_config  # noqa: E402
from evaluate_registry import SCORER_REGISTRY  # noqa: E402

HERE = Path(__file__).resolve().parent


def main(
    ngpu: Optional[int] = None,
    devices: Optional[str] = None,
    keep_shards: bool = False,
    config_file: Optional[str] = "./evaluate_config.yaml",
    **eval_kwargs,
):
    """
    Run one eviction-press eval configuration sharded over GPUs, then score the union.

    Parameters
    ----------
    ngpu : int, optional
        Number of shards / GPUs. Defaults to every visible GPU.
    devices : str, optional
        Comma-separated CUDA indices to use, e.g. ``"0,1,4,5"``. Overrides ``ngpu``. Needed because
        ``ngpu=n`` always takes GPUs ``0..n-1``, which collides with anything already running there.
    keep_shards : bool
        Keep the per-shard parquet files. They are kept on failure regardless.
    config_file : str, optional
        Same YAML layer ``evaluate.py`` reads, applied here with the same precedence so the driver
        and its shards resolve one identical configuration.
    **eval_kwargs
        Everything else is forwarded verbatim to ``evaluate.py`` (``--dataset``, ``--data_dir``,
        ``--model``, ``--press_name``, ``--compression_ratio``, ``--fraction``, ...).
    """
    if devices is not None:
        # Fire turns "0,1" into a tuple and a bare "0" into an int.
        if isinstance(devices, (list, tuple)):
            gpu_ids = [int(d) for d in devices]
        else:
            gpu_ids = [int(d) for d in str(devices).split(",") if d != ""]
    else:
        available = torch.cuda.device_count()
        if available == 0:
            raise SystemExit("no CUDA devices visible; sharded eval needs at least one GPU")
        n = int(ngpu) if ngpu is not None else available
        if n > available:
            raise SystemExit(f"ngpu={n} exceeds the {available} visible GPU(s)")
        gpu_ids = list(range(n))
    num_shards = len(gpu_ids)
    if num_shards == 0:
        raise SystemExit("no GPUs selected")

    # Build the config once, here, so that (a) invalid arguments fail before any GPU is touched
    # rather than N times in N subprocesses, and (b) the results directory is chosen exactly once.
    # Unknown keys raise TypeError, matching evaluate.py's CLI behaviour.
    #
    # The layering mirrors evaluate.py's CliEntryPoint exactly -- dataclass defaults, then the YAML,
    # then the caller's arguments -- because the shards run that CLI and would otherwise resolve a
    # *different* configuration than the one validated here (the YAML ships press_name knorm,
    # compression_ratio 0.5 and a Llama model). The fully merged result is then forwarded
    # explicitly, so the shards' own YAML layer cannot reintroduce a value the driver did not pick.
    defaults = asdict(EvaluationConfig())
    defaults.update(_load_yaml_config(config_file))
    merged = {**defaults, **{k: v for k, v in eval_kwargs.items() if v is not None}}
    try:
        config = EvaluationConfig(**merged)
    except TypeError as e:
        raise SystemExit(f"invalid configuration argument. {e}") from e

    results_dir = config.get_results_dir(Path(config.output_dir))  # uniquified once, then reused
    print(f"sharded eval over {num_shards} GPU(s) {gpu_ids} -> {results_dir}", flush=True)

    # Forward the *fully merged* configuration rather than only what the caller passed, so a shard's
    # own YAML layer cannot reintroduce a default the driver did not intend. results_dir/shard_index/
    # num_shards/device are set per shard below.
    forwarded = {
        k: v
        for k, v in merged.items()
        if v is not None and k not in {"device", "shard_index", "num_shards", "results_dir"}
    }

    procs = []
    log_files = []
    for shard_index, gpu in enumerate(gpu_ids):
        # The shard is given the SAME config_file the driver resolved, not left to find the default
        # one. Explicit flags below still win, but a key whose merged value is None is not forwarded
        # at all (there is no way to spell "None" on Fire's command line), so without this the shard
        # would fill that hole from ./evaluate_config.yaml -- a different file than the driver read.
        # Concretely: data_dir null here plus data_dir "4096" there makes every shard request a
        # RULER-shaped subdirectory of a dataset that has none.
        cmd = [sys.executable, str(HERE / "evaluate.py")]
        if config_file:
            cmd += ["--config_file", str(config_file)]
        for key, value in forwarded.items():
            cmd += [f"--{key}", str(value)]
        cmd += [
            "--shard_index",
            str(shard_index),
            "--num_shards",
            str(num_shards),
            "--results_dir",
            str(results_dir),
            "--device",
            f"cuda:{gpu}",
        ]
        log_path = results_dir / f"shard{shard_index}.log"
        log_file = open(log_path, "w")
        log_files.append(log_file)
        print(f"  shard {shard_index}/{num_shards} on cuda:{gpu} -> {log_path}", flush=True)
        procs.append(
            subprocess.Popen(cmd, cwd=str(HERE), stdout=log_file, stderr=subprocess.STDOUT)
        )

    codes = [p.wait() for p in procs]
    for f in log_files:
        f.close()

    failed = [i for i, c in enumerate(codes) if c != 0]
    if failed:
        # Do NOT score a partial union: a metric over the shards that happened to survive reads
        # exactly like a metric over the whole dataset.
        raise SystemExit(
            f"shard(s) {failed} failed with exit code(s) {[codes[i] for i in failed]}. "
            f"See {results_dir}/shard<i>.log. Shard files kept; nothing was scored."
        )

    # Score the union, exactly once.
    shard_files = [results_dir / f"predictions_shard{i}.parquet" for i in range(num_shards)]
    missing = [str(p) for p in shard_files if not p.exists()]
    if missing:
        raise SystemExit(f"shard(s) exited 0 but wrote no predictions: {missing}")

    frames = [pd.read_parquet(p) for p in shard_files]
    df = pd.concat(frames).sort_index()
    total = sum(len(f) for f in frames)
    assert len(df) == total, f"concat lost rows: {len(df)} != {total}"
    if df.index.has_duplicates:
        raise SystemExit(
            "shards overlap: the same row appears in more than one shard, so the union would "
            "double-count it. This means the shards did not derive from the identical frame."
        )
    if df["predicted_answer"].isna().any():
        n = int(df["predicted_answer"].isna().sum())
        raise SystemExit(f"{n} row(s) have no prediction; refusing to score an incomplete union")

    # Write the predictions BEFORE scoring. Several scorers (RULER's among them) mutate
    # df["predicted_answer"] in place -- it strips control characters, newlines included -- so
    # scoring first would persist a scrubbed transcript and the artifact would silently differ from
    # the unsharded run's. evaluate.py and evaluate_sparse.py write in this order for the same
    # reason.
    df[list(set(df.columns) - {"context"})].to_csv(str(results_dir / "predictions.csv"), index=False)

    metrics = SCORER_REGISTRY[config.dataset](df)

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    with open(results_dir / "config.yaml", "w") as f:
        saved = asdict(config)
        saved.update({"num_shards": num_shards, "shard_index": None, "sharded_devices": gpu_ids})
        yaml.dump(saved, f, default_flow_style=False, sort_keys=False)

    if not keep_shards:
        for p in shard_files:
            p.unlink()

    print(f"scored {len(df)} rows from {num_shards} shard(s)", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved to {results_dir}", flush=True)


if __name__ == "__main__":
    Fire(main)
