# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Data-parallel driver for :mod:`evaluate_sparse`: split ONE (dataset, length, topk) configuration
across N GPUs and score the union.

The existing shell wrappers (``evaluate_sparse.sh``, ``evaluate_sparse_scalar.sh``) parallelize over
*configurations* -- one (length, topk) pair per GPU -- because ``evaluate_sparse.py`` had no
sharding option, so running the same configuration on 8 GPUs would evaluate the identical rows 8
times. That leaves a single configuration stuck at one GPU's throughput, which is the common case
when you want one number (one length, one topk) as fast as possible.

This driver shards the ROWS instead:

    python evaluate_sparse_sharded.py --dataset ruler --data_dir 8192 \\
        --model /path/Qwen3-8B --indexer_ckpt /path/final.pt --topk 2048 --ngpu 8

Each shard is a separate ``evaluate_sparse.py`` process pinned to one GPU (subprocesses, not
threads: one CUDA context per process, and the model is loaded per GPU anyway). Shards write
parquet prediction files; this driver concatenates them and scores ONCE over the union, so the
metric is identical to what the unsharded run would have produced.

Why the pieces are the way they are
-----------------------------------
*Sharding is by context, not by row.* A context's questions share one prefill, so splitting them
across shards would re-prefill the same long context in every shard. ``evaluate_sparse.py`` does
the round-robin over unique contexts; see its ``_load_dataset``.

*Parquet, not CSV, for the shard files.* RULER's ``answer`` column holds an ndarray of reference
strings and the scorers iterate it. A CSV round-trip turns ``array(['2166941'])`` into the string
``"['2166941']"``, which then iterates character by character -- 11 phantom references -- so a
wrong prediction scores 0.27 instead of 0.0. Silent, and it inflates the metric.

*The driver picks the results directory.* Otherwise N processes race in ``get_results_dir``, each
asking "does this directory exist" and landing on different uniquified suffixes.

*Shards that fail are fatal.* Scoring the union of the surviving shards would silently report a
metric over a subset -- a number that looks like a result but is a result of less data.
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

from evaluate_registry import SCORER_REGISTRY  # noqa: E402
from evaluate_sparse import SparseEvaluationConfig  # noqa: E402

HERE = Path(__file__).resolve().parent


def main(
    ngpu: Optional[int] = None,
    devices: Optional[str] = None,
    keep_shards: bool = False,
    **eval_kwargs,
):
    """
    Run one sparse-eval configuration sharded over GPUs, then score the union.

    Parameters
    ----------
    ngpu : int, optional
        Number of shards / GPUs. Defaults to every visible GPU.
    devices : str, optional
        Comma-separated CUDA indices to use, e.g. ``"0,1,4,5"``. Overrides ``ngpu``.
    keep_shards : bool
        Keep the per-shard parquet files. They are kept on failure regardless.
    **eval_kwargs
        Everything else is forwarded verbatim to ``evaluate_sparse.py`` (``--dataset``,
        ``--data_dir``, ``--model``, ``--indexer_ckpt``, ``--topk``, ``--force_local``, ...).
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
    # Unknown keys raise TypeError, matching evaluate_sparse.main's behaviour.
    defaults = asdict(SparseEvaluationConfig(indexer_ckpt="_placeholder_"))
    defaults.pop("indexer_ckpt")
    merged = {**defaults, **{k: v for k, v in eval_kwargs.items() if v is not None}}
    try:
        config = SparseEvaluationConfig(**merged)
    except TypeError as e:
        raise SystemExit(f"invalid configuration argument. {e}") from e

    results_dir = config.get_results_dir()  # uniquified once, then reused by every shard
    print(f"sharded sparse eval over {num_shards} GPU(s) {gpu_ids} -> {results_dir}", flush=True)

    # Launch one subprocess per shard. Each gets the driver's chosen results_dir and its own
    # (shard_index, device); everything else is the shared configuration.
    forwarded = {k: v for k, v in eval_kwargs.items() if v is not None}
    forwarded.pop("device", None)  # the shard's device is assigned here, per GPU
    forwarded.pop("shard_index", None)
    forwarded.pop("num_shards", None)
    forwarded.pop("results_dir", None)

    procs = []
    log_files = []
    for shard_index, gpu in enumerate(gpu_ids):
        cmd = [sys.executable, str(HERE / "evaluate_sparse.py")]
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
    # scoring first would persist a scrubbed transcript and the artifact would silently differ
    # from the unsharded run's. evaluate_sparse.py writes in this order for the same reason.
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
