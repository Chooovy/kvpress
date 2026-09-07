# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Collect the runs written by ``evaluate_reasoning_longbench.sh`` into one table.

    python summarize_reasoning_longbench.py --output_dir ./results_reasoning_longbench

Each run directory holds a ``metrics.json`` and the ``config.yaml`` it was produced under, so the
arm (dense / sparse), the task and the top-k are read from the config rather than parsed out of the
directory name -- a name is a summary of a configuration, and the two drift.

Two columns exist because of how these benchmarks fail rather than for completeness:

*``answered``* -- the fraction of rows where ``\\boxed{}`` appeared at all. math500 and aime25 score a
missing box as wrong, which is correct but conflates two very different failures: the model reasoned
and got the wrong number, or it never finished. A sparse arm that drops from 0.90 to 0.20 answered
has had its reasoning truncated (or has begun looping), which is a different finding from a drop in
accuracy at the same answered rate. The scorers already compute it; this surfaces it.

*``vs dense``* -- accuracy minus the dense no-press run's accuracy on the same task. The absolute
number is not the result on these benchmarks; the gap to the dense upper bound is, since it is the
only column that holds the model, the sampled rows and the prompt fixed.

Row counts are printed with the metric because the reasoning sets are small enough that the count is
part of interpreting the number: aime25 is 30 problems whatever the fraction, where the 95% binomial
CI at p=0.5 is roughly +/-18 points. Treat it as a smoke test for "reasoning survives at all", not a
ranking instrument.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from fire import Fire


def _wilson_halfwidth(p: float, n: int, z: float = 1.96) -> Optional[float]:
    """Half-width of the Wilson score interval, in accuracy points.

    Wilson rather than the normal approximation: at n=30 with p near 0 or 1 the normal interval runs
    past [0, 1] and reports a precision the sample does not have. Returns None when n is 0.
    """
    if n <= 0:
        return None
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo, hi = max(0.0, centre - margin), min(1.0, centre + margin)
    return (hi - lo) / 2


def _accuracy(task: str, metrics: dict) -> Optional[float]:
    """The primary metric per task, normalized to a 0-1 accuracy."""
    if "accuracy" in metrics:  # math500, aime25
        return float(metrics["accuracy"])
    if "average" in metrics:  # longbench-v2
        return float(metrics["average"])
    # RULER and friends report per-task means; the mean over them is the headline number.
    numeric = [v for v in metrics.values() if isinstance(v, (int, float))]
    return float(sum(numeric) / len(numeric)) if numeric else None


def main(output_dir: str = "./results_reasoning_longbench"):
    """Print one row per run, dense first, then each sparse top-k, grouped by task."""
    root = Path(output_dir)
    if not root.exists():
        raise SystemExit(f"{root} does not exist -- run evaluate_reasoning_longbench.sh first")

    rows = []
    for metrics_file in sorted(root.rglob("metrics.json")):
        run_dir = metrics_file.parent
        config_file = run_dir / "config.yaml"
        if not config_file.exists():
            # A shard directory mid-run, or a run that died before scoring. Skipping is right:
            # a metrics.json with no config cannot be attributed to an arm.
            continue
        with open(config_file) as f:
            config = yaml.safe_load(f) or {}
        with open(metrics_file) as f:
            metrics = json.load(f)

        task = config.get("dataset", "?")
        # indexer_ckpt is the field only the sparse config has -- more robust than the directory name.
        is_sparse = bool(config.get("indexer_ckpt"))
        # Only the math scorers report `total`; longbench-v2's returns per-difficulty means and no
        # count. Fall back to counting the predictions actually scored, so the row count -- which is
        # what the confidence interval is computed from -- is never silently reported as 0.
        #
        # Parsed with pandas rather than by counting lines: the predictions hold model output with
        # embedded newlines, so a line count reads 534 for 50 rows and the interval it feeds comes
        # out ~3x too tight (0.037 against the true 0.118).
        total = metrics.get("total")
        if total is None:
            predictions = run_dir / "predictions.csv"
            if predictions.exists():
                total = len(pd.read_csv(predictions))
        answered = metrics.get("answered")
        rows.append(
            {
                "task": task,
                "arm": f"sparse topk{config.get('topk')}" if is_sparse else "dense no_press",
                "topk": config.get("topk") if is_sparse else None,
                "accuracy": _accuracy(task, metrics),
                "answered": (answered / total) if (answered is not None and total) else None,
                "n": total,
                "fraction": config.get("fraction"),
                "max_context": config.get("max_context_length"),
                "dir": run_dir.name,
            }
        )

    if not rows:
        raise SystemExit(f"no scored runs found under {root}")

    header = (
        f"{'task':<14} {'arm':<20} {'n':>5} {'frac':>5} {'acc':>7} {'answered':>9} {'+/-95%':>7} "
        f"{'vs dense':>9}"
    )
    print(header)
    print("-" * len(header))

    for task in sorted({r["task"] for r in rows}):
        group = [r for r in rows if r["task"] == task]
        dense = next((r for r in group if r["topk"] is None), None)
        # Dense first, then ascending top-k: the reading order is "upper bound, then how much budget
        # it takes to approach it".
        group.sort(key=lambda r: (r["topk"] is not None, r["topk"] or 0))
        for r in group:
            acc, ans, n = r["accuracy"], r["answered"], r["n"] or 0
            hw = _wilson_halfwidth(acc, n) if acc is not None else None
            gap = (
                acc - dense["accuracy"]
                if (dense and acc is not None and dense["accuracy"] is not None and r is not dense)
                else None
            )
            print(
                f"{r['task']:<14} {r['arm']:<20} {n:>5} "
                f"{(f'{r['fraction']:.2f}' if r['fraction'] is not None else '-'):>5} "
                f"{(f'{acc:.4f}' if acc is not None else '-'):>7} "
                f"{(f'{ans:.2f}' if ans is not None else '-'):>9} "
                f"{(f'{hw:.3f}' if hw is not None else '-'):>7} "
                f"{(f'{gap:+.4f}' if gap is not None else '-'):>9}"
            )
        print()

    print(
        "acc      = accuracy (math500/aime25) or average (longbench-v2)\n"
        "answered = fraction of rows containing \\boxed{} -- a drop here is truncated or looping\n"
        "           reasoning, which is a different failure from a wrong answer\n"
        "+/-95%   = Wilson half-width at this n. aime25's n=30 makes it a smoke test, not a ranking\n"
        "vs dense = accuracy minus the dense no_press run on the same task"
    )


if __name__ == "__main__":
    Fire(main)
