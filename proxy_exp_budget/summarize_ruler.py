# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarise RULER sparse-eval result dirs: per-task string_match plus the unweighted task mean.

RULER's headline number is the mean over tasks, and ``metrics.json`` is nested
``{task: {"string_match": v}}`` -- flattening it wrong is how a 20.43 gets read as a 44.75.
"""

import argparse
import json
from pathlib import Path

import yaml


def load(d: Path) -> tuple[dict, dict]:
    metrics = json.loads((d / "metrics.json").read_text())
    flat = {t: v["string_match"] if isinstance(v, dict) else v for t, v in metrics.items()}
    cfg_path = d / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    return flat, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--label", action="append", default=None)
    args = ap.parse_args()

    rows = []
    for i, d in enumerate(args.dirs):
        if not (d / "metrics.json").exists():
            print(f"[skip] no metrics.json in {d}")
            continue
        flat, cfg = load(d)
        label = args.label[i] if args.label and i < len(args.label) else d.name
        rows.append((label, flat, cfg))

    if not rows:
        return
    tasks = sorted({t for _, f, _ in rows for t in f})
    width = max(len(t) for t in tasks) + 2
    head = "".join(f"{lbl[:14]:>16s}" for lbl, _, _ in rows)
    print(f"{'task':<{width}}{head}")
    for t in tasks:
        line = "".join(f"{f.get(t, float('nan')):>16.2f}" for _, f, _ in rows)
        print(f"{t:<{width}}{line}")
    means = "".join(f"{sum(f.values()) / len(f):>16.2f}" for _, f, _ in rows)
    print(f"{'-' * (width - 1)} {'-' * (16 * len(rows))}")
    print(f"{'MEAN':<{width}}{means}")
    print()
    for lbl, f, cfg in rows:
        print(f"{lbl}: topk={cfg.get('topk')} n_tasks={len(f)} ckpt={cfg.get('indexer_ckpt', '?')}")


if __name__ == "__main__":
    main()
