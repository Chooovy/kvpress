# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Rebuild a ``metrics.jsonl`` from a training log, for a run whose metrics file was lost.

The trainer writes two records of every logged step: a line on stdout and a row in
``--metrics-file``. The JSONL is the richer one -- it carries the per-layer dicts and the
weighted-objective diagnostics -- but the stdout line carries the fields the budget/pin sweeps
are actually read on: ``loss``, ``gate_scale_mean``, ``gate_sparsity_mean``,
``history_attention_mass_mean``, ``grad_norm``, ``lr``, ``peak_gib``.

So this recovers the *curves* and not the run. What is NOT recoverable and is simply absent from
the output: per-layer ``gate_scales`` / ``gate_sparsity`` breakdowns, ``weight_participation`` and
``longce_cache_miss_frac`` (LongCE), and anything a checkpoint held. Rows are marked
``"source": "stdout_log"`` so a reconstructed file can never be mistaken for a real one.

    python -m scripts.metrics_from_log run.log -o .../metrics.jsonl

Read ``-`` for stdin. Idempotent and order-preserving; duplicate steps keep the last occurrence,
which is what a resumed run's log would want.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

#: The stdout format emitted by scripts/train_gqa_indexer_e2e.py's non-SFT branch. `sparsity` and
#: `history_mass` print the literal `off` when their diagnostic is disabled, which maps to null.
LINE = re.compile(
    r"step\s+(?P<step>\d+)/(?P<total>\d+)\s+"
    r"L=(?P<seq_len>\d+)\s+"
    r"lm_loss\s+(?P<loss>[-\d.]+)\s+"
    r"\(avg\s+(?P<avg>[-\d.]+)\)\s+"
    r"\|g\|\s+(?P<grad_norm>[-\d.]+)\s+"
    r"lr\s+(?P<lr>[\d.e+-]+)\s+"
    r"gate\s+(?P<gate>[-\d.nan]+)\s+"
    r"sparsity\s+(?P<sparsity>off|[-\d.]+)\s+"
    r"history_mass\s+(?P<history_mass>off|[-\d.]+)\s+"
    r"peak\s+(?P<peak_gib>[-\d.]+)\s+GiB"
)


def _num(value: str) -> float | None:
    return None if value == "off" else float(value)


def parse(lines) -> list[dict]:
    """Every ``step`` line in the log, as metrics rows in file order (last wins per step)."""
    rows: dict[int, dict] = {}
    for line in lines:
        m = LINE.search(line)
        if m is None:
            continue
        step = int(m["step"])
        rows[step] = {
            "step": step,
            "seq_len": int(m["seq_len"]),
            "loss": float(m["loss"]),
            "grad_norm": float(m["grad_norm"]),
            "lr": float(m["lr"]),
            "gate_scale_mean": float(m["gate"]),
            "gate_sparsity_mean": _num(m["sparsity"]),
            "history_attention_mass_mean": _num(m["history_mass"]),
            "peak_gib": float(m["peak_gib"]),
            # Marked so a rebuilt file is never mistaken for one the trainer wrote: the per-layer
            # dicts and the LongCE diagnostics are absent, not zero.
            "source": "stdout_log",
        }
    return [rows[s] for s in sorted(rows)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("log", help="training log, or - for stdin")
    ap.add_argument("-o", "--out", help="write JSONL here (default: stdout)")
    args = ap.parse_args()

    stream = sys.stdin if args.log == "-" else open(args.log)
    rows = parse(stream)
    if not rows:
        raise SystemExit("no `step N/M ... lm_loss ...` lines found; is this a trainer log?")

    text = "".join(json.dumps(r) + "\n" for r in rows)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text)
        covered = sum(r["history_attention_mass_mean"] is not None for r in rows)
        print(
            f"wrote {len(rows)} rows (steps {rows[0]['step']}..{rows[-1]['step']}) to {args.out}; "
            f"{covered} carry history_attention_mass"
        )
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
