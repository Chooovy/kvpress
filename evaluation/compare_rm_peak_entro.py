#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_runs(root: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for metrics_path in root.rglob("metrics.json"):
        run_dir = metrics_path.parent
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            pairs.append((config_path, metrics_path))
    return pairs


def extract_task_string_match_strict(metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    STRICT format:
      metrics = {
        "taskA": {"string_match": 12.3},
        "taskB": {"string_match": 45.6},
        ...
      }
    """
    out: Dict[str, float] = {}
    for task, v in metrics.items():
        if not isinstance(v, dict):
            continue
        if "string_match" not in v:
            continue
        sm = v["string_match"]
        if isinstance(sm, (int, float)):
            out[str(task)] = float(sm)
        elif isinstance(sm, str):
            try:
                out[str(task)] = float(sm)
            except Exception:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="results root dir")
    ap.add_argument("--out_prefix", type=str, default="compare_strict", help="output file prefix")
    ap.add_argument("--press", type=str, nargs="*", default=None, help="optional: only keep these press names")
    args = ap.parse_args()

    root = Path(args.root)
    assert root.exists(), f"Root not found: {root}"

    runs = find_runs(root)
    if not runs:
        raise RuntimeError(f"No runs found under {root} (need config.yaml + metrics.json)")

    rows_summary = []
    rows_task = []

    for config_path, metrics_path in runs:
        cfg = load_yaml(config_path)
        met = load_json(metrics_path)

        press = str(cfg.get("press_name", ""))
        dataset = str(cfg.get("dataset", ""))
        data_dir = cfg.get("data_dir", None)
        model = str(cfg.get("model", "")).split("/")[-1]

        cr = cfg.get("compression_ratio", None)
        key_cr = cfg.get("key_channel_compression_ratio", None)
        fraction = cfg.get("fraction", None)

        try:
            cr = float(cr) if cr is not None else None
        except Exception:
            cr = None
        try:
            key_cr = float(key_cr) if key_cr is not None else None
        except Exception:
            key_cr = None
        try:
            fraction = float(fraction) if fraction is not None else None
        except Exception:
            fraction = None

        if args.press is not None and len(args.press) > 0 and press not in args.press:
            continue

        task_scores = extract_task_string_match_strict(met)
        if not task_scores:
            continue

        overall_avg = sum(task_scores.values()) / len(task_scores)

        rows_summary.append(
            {
                "run_dir": str(metrics_path.parent),
                "dataset": dataset,
                "data_dir": data_dir,
                "model": model,
                "press": press,
                "cr": cr,
                "key_channel_cr": key_cr,
                "fraction": fraction,
                "overall_avg": overall_avg,
                "n_tasks": len(task_scores),
            }
        )

        for task, score in task_scores.items():
            rows_task.append(
                {
                    "run_dir": str(metrics_path.parent),
                    "dataset": dataset,
                    "data_dir": data_dir,
                    "model": model,
                    "press": press,
                    "cr": cr,
                    "key_channel_cr": key_cr,
                    "fraction": fraction,
                    "task": task,
                    "string_match": score,
                }
            )
    df_task = pd.DataFrame(rows_task)
    if df_task.empty:
        raise RuntimeError("No valid runs parsed. Check metrics.json format and press filter.")

    # 如果同一个 (press, cr) 你跑了多次，这里会自动取平均（更稳）
    avg_by_press_cr = (
        df_task.groupby(["cr", "press"], as_index=False)["string_match"]
        .mean()
        .rename(columns={"string_match": "avg"})
    )

    # 同一个 (cr, task) 下，不同 press 的 task 分数（多次run会平均）
    task_by_press = (
        df_task.groupby(["cr", "task", "press"], as_index=False)["string_match"]
        .mean()
    )

    # 固定打印顺序
    cr_list = sorted([c for c in avg_by_press_cr["cr"].dropna().unique()])
    task_list = sorted([t for t in task_by_press["task"].dropna().unique()])

    print("\n================ 1) Overall avg ranking within each CR ================\n")
    for cr in cr_list:
        sub = avg_by_press_cr[avg_by_press_cr["cr"] == cr].copy()
        sub = sub.sort_values(["avg", "press"], ascending=[False, True]).reset_index(drop=True)
        print(f"[cr={cr}] (avg = mean over tasks' string_match)")
        for i, row in sub.iterrows():
            print(f"  #{i+1:<2d}  {row['press']:<45s}  avg={row['avg']:.4f}")
        print()

    print("\n================ 2) Per-task ranking within each (CR, task) ================\n")
    for cr in cr_list:
        print(f"========== CR = {cr} ==========")
        for task in task_list:
            sub = task_by_press[(task_by_press["cr"] == cr) & (task_by_press["task"] == task)].copy()
            if sub.empty:
                continue
            sub = sub.sort_values(["string_match", "press"], ascending=[False, True]).reset_index(drop=True)

            # 一行标题 + 多行排序
            print(f"[task={task}]")
            for i, row in sub.iterrows():
                print(f"  #{i+1:<2d}  {row['press']:<45s}  score={float(row['string_match']):.4f}")
        print()

if __name__ == "__main__":
    main()
