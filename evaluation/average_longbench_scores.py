#!/usr/bin/env python3
# average_longbench_scores.py

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Optional, Dict, List, Tuple, Set


def extract_scalar(obj: Any) -> Optional[float]:
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
        vals = []
        for v in obj.values():
            vv = extract_scalar(v)
            if vv is not None:
                vals.append(vv)
        if len(vals) == 1:
            return vals[0]
        if len(vals) >= 2:
            return float(mean(vals))
    if isinstance(obj, list):
        vals = []
        for v in obj:
            vv = extract_scalar(v)
            if vv is not None:
                vals.append(vv)
        if vals:
            return float(mean(vals))
    return None


def parse_from_path(metrics_path: Path) -> Dict[str, Any]:
    """
    run_dir pattern:
      longbench__<task>__<model>__<press>__cr0.10__fraction0.100
    """
    task_folder = metrics_path.parent.parent.name
    run_dir = metrics_path.parent.name

    parts = run_dir.split("__")
    if len(parts) >= 6 and parts[0].startswith("longbench"):
        idx_cr = next((i for i, p in enumerate(parts) if p.startswith("cr")), None)
        idx_frac = next((i for i, p in enumerate(parts) if p.startswith("fraction")), None)
        if idx_cr is not None and idx_frac is not None and idx_frac > idx_cr:
            task2 = parts[1]
            model_name = parts[2]
            press_name = "__".join(parts[3:idx_cr])

            cr_str = parts[idx_cr][len("cr"):]
            frac_str = parts[idx_frac][len("fraction"):]

            cr = float(cr_str) if cr_str else None
            frac = float(frac_str) if frac_str else None

            return {
                "task": task2 or task_folder,
                "model": model_name,
                "press": press_name,
                "cr": cr,
                "fraction": frac,
            }

    # fallback
    cr_m = re.search(r"__cr([0-9.]+)", run_dir)
    frac_m = re.search(r"__fraction([0-9.]+)", run_dir)
    return {
        "task": task_folder,
        "model": "unknown_model",
        "press": "unknown_press",
        "cr": float(cr_m.group(1)) if cr_m else None,
        "fraction": float(frac_m.group(1)) if frac_m else None,
    }


def canonical_model_family(model: str) -> str:
    """
    Heuristic: map different checkpoint names to the same backbone family.
    Adjust rules to your naming if needed.
    """
    s = (model or "").lower()

    # mistral
    if "mistral" in s:
        return "mistral-7b"
    # qwen
    if "qwen3-8b" in s or "qwen3" in s:
        return "qwen3-8b"
    # llama 8b
    if "llama" in s and "8b" in s:
        return "llama3-8b"

    return model  # fallback: keep original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default="summary_common_tasks.csv")
    ap.add_argument("--require_presses", nargs="+", default=None,
                    help="Only average over tasks where ALL these presses exist (within model family + cr).")
    ap.add_argument("--group_model_family", action="store_true",
                    help="Group different checkpoint names into the same backbone family before intersecting tasks.")
    ap.add_argument("--debug_missing", action="store_true",
                    help="Print which presses are missing for each (model_key, cr).")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
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

        meta = parse_from_path(mp)
        if meta.get("cr") is None:
            continue

        meta["score"] = float(score)
        meta["metrics_path"] = str(mp)
        meta["model_key"] = canonical_model_family(meta["model"]) if args.group_model_family else meta["model"]
        rows.append(meta)

    if not rows:
        print(f"[ERROR] No parsable metrics.json found under: {root.resolve()}")
        return

    required: Optional[Set[str]] = set(args.require_presses) if args.require_presses else None
    if required is None:
        print("[ERROR] Please pass --require_presses ... (the 6 methods you care about).")
        return

    # 1) availability: (model_key, cr, task) -> set(presses)
    avail: Dict[Tuple[str, float, str], Set[str]] = {}
    for r in rows:
        key = (r["model_key"], float(r["cr"]), r["task"])
        avail.setdefault(key, set()).add(r["press"])

    # 2) common tasks per (model_key, cr): tasks where required presses are all present
    common_tasks: Dict[Tuple[str, float], Set[str]] = {}
    missing_map: Dict[Tuple[str, float], Dict[str, Set[str]]] = {}

    for (mk, cr, task), presses in avail.items():
        if required.issubset(presses):
            common_tasks.setdefault((mk, cr), set()).add(task)
        else:
            miss = required - presses
            missing_map.setdefault((mk, cr), {}).setdefault(task, set()).update(miss)

    if args.debug_missing:
        # show missing summary
        for (mk, cr), task_miss in sorted(missing_map.items()):
            # count missing occurrences
            cnt = {}
            for _, miss_set in task_miss.items():
                for p in miss_set:
                    cnt[p] = cnt.get(p, 0) + 1
            top = ", ".join([f"{p}:{c}" for p, c in sorted(cnt.items(), key=lambda x: -x[1])])
            print(f"[DEBUG] ({mk}, cr={cr:.2f}) missing counts: {top}")

    # 3) compute avg per (model_key, press, cr) but only over common tasks
    groups: Dict[Tuple[str, str, float], List[float]] = {}
    tasks_used: Dict[Tuple[str, str, float], Set[str]] = {}

    for r in rows:
        mk = r["model_key"]
        cr = float(r["cr"])
        task = r["task"]
        press = r["press"]

        if (mk, cr) not in common_tasks or task not in common_tasks[(mk, cr)]:
            continue
        if press not in required:
            continue

        k = (mk, press, cr)
        groups.setdefault(k, []).append(r["score"])
        tasks_used.setdefault(k, set()).add(task)

    if not groups:
        print("[ERROR] After applying common-task filter, nothing left.")
        print("Most likely reasons:")
        print("  1) Your baseline vs indexer runs are under different model names -> add --group_model_family")
        print("  2) Some press name in --require_presses doesn't match the directory press tag exactly")
        print("  3) For some (model, cr), at least one of the required presses OOM/failed on every task")
        return

    # 4) write csv
    outp = Path(args.out_csv).expanduser()
    outp.parent.mkdir(parents=True, exist_ok=True)

    header = ["model", "press", "cr", "avg_score", "n_tasks", "tasks"]
    lines = [",".join(header)]

    for (mk, press, cr) in sorted(groups.keys(), key=lambda x: (x[0], x[2], x[1])):
        avg = mean(groups[(mk, press, cr)])
        tasks = sorted(tasks_used[(mk, press, cr)])
        lines.append(",".join([
            mk,
            press,
            f"{cr:.2f}",
            f"{avg:.6f}",
            str(len(tasks)),
            "|".join(tasks),
        ]))

    outp.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote: {outp.resolve()}")
    print("\n=== Summary (mean over common tasks) ===")
    for l in lines[:1] + lines[1:30]:
        print(l)


if __name__ == "__main__":
    main()
