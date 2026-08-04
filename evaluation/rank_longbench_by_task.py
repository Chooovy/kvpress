#!/usr/bin/env python3
# rank_longbench_by_task.py

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple


def extract_scalar(obj: Any) -> Optional[float]:
    """Extract a single scalar from various json shapes."""
    if obj is None:
        return None
    if isinstance(obj, (int, float)):
        return float(obj)
    if isinstance(obj, str):
        try:
            return float(obj.strip())
        except Exception:
            return None
    if isinstance(obj, dict):
        for k in ["score", "avg", "overall", "metric", "metrics", "result"]:
            if k in obj:
                v = extract_scalar(obj[k])
                if v is not None:
                    return v
        nums = []
        for v in obj.values():
            vv = extract_scalar(v)
            if vv is not None:
                nums.append(vv)
        if len(nums) == 1:
            return nums[0]
        if len(nums) >= 2:
            return float(mean(nums))
    if isinstance(obj, list):
        nums = []
        for v in obj:
            vv = extract_scalar(v)
            if vv is not None:
                nums.append(vv)
        if nums:
            return float(mean(nums))
    return None


def parse_run_dir(run_dir_name: str) -> Dict[str, Any]:
    """
    Expected pattern:
      longbench__<task>__<model>__<press>__cr0.10__fraction0.100
    But task/model/press may contain '_' so we use '__' as delimiter.
    """
    parts = run_dir_name.split("__")
    meta = {
        "task": None,
        "model": None,
        "press": None,
        "cr": None,
        "fraction": None,
    }

    if len(parts) >= 4 and parts[0] == "longbench":
        meta["task"] = parts[1]
        meta["model"] = parts[2]
        meta["press"] = parts[3]

    for p in parts[4:]:
        if p.startswith("cr"):
            try:
                meta["cr"] = float(p[len("cr"):])
            except Exception:
                pass
        elif p.startswith("fraction"):
            try:
                meta["fraction"] = float(p[len("fraction"):])
            except Exception:
                pass

    return meta


def canonical_family(model_name: str) -> str:
    s = (model_name or "").lower()
    if "llama" in s:
        return "llama3-8b"
    if "mistral" in s:
        return "mistral-7b"
    if "qwen" in s:
        return "qwen3-8b"
    return "unknown_family"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="e.g. /.../evaluation/results_longbench_frac0.1")
    ap.add_argument("--out_csv", type=str, required=True, help="e.g. /.../rank_by_task.csv")
    ap.add_argument("--fraction", type=float, default=None, help="Optional: only keep runs with this fraction (e.g. 0.1)")
    ap.add_argument("--cr", type=float, default=None, help="Optional: only keep this CR (e.g. 0.5)")

    # ✅ NEW: only keep tasks where all these presses exist (comma-separated)
    ap.add_argument(
        "--required_presses",
        type=str,
        default=None,
        help="Comma-separated press list. If set, keep only (family,task,cr) where ALL these presses have results."
    )

    args = ap.parse_args()

    required_presses = None
    if args.required_presses:
        required_presses = {x.strip() for x in args.required_presses.split(",") if x.strip()}
        if not required_presses:
            required_presses = None

    root = Path(args.root)
    metrics_files = sorted(root.rglob("metrics.json"))

    rows: List[Dict[str, Any]] = []
    for mp in metrics_files:
        try:
            data = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            txt = mp.read_text(encoding="utf-8", errors="ignore").strip()
            try:
                data = float(txt)
            except Exception:
                continue

        score = extract_scalar(data)
        if score is None:
            continue

        run_dir = mp.parent.name
        meta = parse_run_dir(run_dir)

        if not meta["task"]:
            meta["task"] = mp.parent.parent.name

        if not meta["model"]:
            meta["model"] = "unknown_model"
        if not meta["press"]:
            meta["press"] = "unknown_press"

        if args.fraction is not None and meta.get("fraction") is not None:
            if abs(meta["fraction"] - args.fraction) > 1e-9:
                continue

        if args.cr is not None and meta.get("cr") is not None:
            if abs(meta["cr"] - args.cr) > 1e-9:
                continue

        meta["family"] = canonical_family(meta["model"])
        meta["score"] = float(score)
        meta["metrics_path"] = str(mp)
        rows.append(meta)

    if not rows:
        raise SystemExit(f"[ERROR] No parsable metrics.json under: {root}")

    # 1) aggregate within (family, task, cr, press): average over model-variants / repeats
    agg: Dict[Tuple[str, str, float, str], List[float]] = {}
    models_in: Dict[Tuple[str, str, float, str], List[str]] = {}

    for r in rows:
        if r["cr"] is None:
            continue
        key = (r["family"], r["task"], float(r["cr"]), r["press"])
        agg.setdefault(key, []).append(r["score"])
        models_in.setdefault(key, []).append(r["model"])

    # ✅ NEW 2) compute which (family,task,cr) have all required presses
    keep_keys = None  # type: Optional[set]
    if required_presses is not None:
        available: Dict[Tuple[str, str, float], set] = {}
        for (family, task, cr, press) in agg.keys():
            k = (family, task, cr)
            available.setdefault(k, set()).add(press)

        keep_keys = {k for k, ps in available.items() if required_presses.issubset(ps)}

    # 3) for each (family, task, cr): rank presses by avg_score desc
    buckets: Dict[Tuple[str, str, float], List[Dict[str, Any]]] = {}
    for (family, task, cr, press), scores in agg.items():
        k2 = (family, task, cr)

        # ✅ NEW filter
        if keep_keys is not None and k2 not in keep_keys:
            continue

        entry = {
            "family": family,
            "task": task,
            "cr": cr,
            "press": press,
            "avg_score": mean(scores),
            "n_runs": len(scores),
            "models": "|".join(sorted(set(models_in[(family, task, cr, press)]))),
        }
        buckets.setdefault(k2, []).append(entry)

    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)

    header = ["family", "task", "cr", "rank", "press", "avg_score", "n_runs", "models"]
    lines = [",".join(header)]

    for (family, task, cr) in sorted(buckets.keys(), key=lambda x: (x[0], x[1], x[2])):
        items = sorted(buckets[(family, task, cr)], key=lambda d: d["avg_score"], reverse=True)
        for i, it in enumerate(items, 1):
            lines.append(",".join([
                it["family"],
                it["task"],
                f"{it['cr']:.2f}",
                str(i),
                it["press"],
                f"{it['avg_score']:.6f}",
                str(it["n_runs"]),
                it["models"],
            ]))

    outp.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote: {outp.resolve()}")

    print("\n=== Preview (first 40 lines) ===")
    for l in lines[:41]:
        print(l)

    if required_presses is not None:
        print(f"\n[INFO] required_presses={sorted(required_presses)}")
        print(f"[INFO] kept (family,task,cr) groups: {len(buckets)}")


if __name__ == "__main__":
    main()
