#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from collections import defaultdict

BASE_PRESS = "expected_attention"
VARIANT_PRESSES_DEFAULT = [
    "expected_attention_layer_mean",
    "expected_attention_head_mean",
]

# When a task metric value is a dict, try these keys first.
PREFERRED_NUM_KEYS = [
    "macro_string_match",
    "string_match",
    "exact_match",
    "acc",
    "accuracy",
    "f1",
    "score",
    "value",
    "avg",
]

EPS = 1e-12  # treat abs(delta) <= EPS as tied


def to_float(x):
    """Convert a metric value to float; supports number/str/dict."""
    if isinstance(x, (int, float)):
        return float(x)

    if isinstance(x, str):
        try:
            return float(x)
        except Exception:
            return None

    if isinstance(x, dict):
        # 1) preferred keys
        for k in PREFERRED_NUM_KEYS:
            if k in x:
                v = to_float(x[k])
                if v is not None:
                    return v

        # 2) if exactly one numeric-like value exists, use it
        vals = []
        for v in x.values():
            fv = to_float(v)
            if fv is not None:
                vals.append(fv)
        if len(vals) == 1:
            return vals[0]

        return None

    return None


def parse_run_dir(dname: str):
    """
    Parse directory name like:
      ruler__4096__Llama-3.2-1B-Instruct__expected_attention_layer_mean__cr0.90
    Could have extra suffix after cr; we only care about first 4 tokens + cr.
    """
    parts = dname.split("__")
    if len(parts) < 5:
        return None
    dataset, data_dir, model, press = parts[:4]
    m = re.search(r"cr([0-9.]+)", dname)
    cr = float(m.group(1)) if m else None
    return dataset, data_dir, model, press, cr


def avg_of_metrics(metrics: dict):
    """
    Average over all numeric-like task values in this metrics dict.
    Returns: (avg, used_cnt, skipped_cnt)
    """
    used = 0
    skipped = 0
    s = 0.0
    for _, v_raw in metrics.items():
        v = to_float(v_raw)
        if v is None:
            skipped += 1
            continue
        s += v
        used += 1
    if used == 0:
        return None, 0, skipped
    return s / used, used, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="results_ea_means",
                    help="Results root dir that contains */metrics.json")
    ap.add_argument("--base_press", type=str, default=BASE_PRESS)
    ap.add_argument("--variant_presses", type=str, nargs="*", default=VARIANT_PRESSES_DEFAULT)

    ap.add_argument("--out_csv", type=str, default="set_deltas.csv")
    ap.add_argument("--summary_csv", type=str, default="set_delta_summary.csv")
    ap.add_argument("--avg_csv", type=str, default="avg_by_press.csv")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    if not outdir.exists():
        raise FileNotFoundError(f"outdir not found: {outdir}")

    # (dataset,data_dir,model,cr) -> press -> metrics dict
    groups = defaultdict(dict)

    for mp in outdir.rglob("metrics.json"):
        info = parse_run_dir(mp.parent.name)
        if not info:
            continue
        dataset, data_dir, model, press, cr = info
        if cr is None:
            continue
        try:
            metrics = json.loads(mp.read_text())
        except Exception:
            continue
        groups[(dataset, data_dir, model, cr)][press] = metrics

    delta_rows = []
    summary_rows = []
    avg_rows = []

    base_press = args.base_press
    variants = args.variant_presses

    for (dataset, data_dir, model, cr) in sorted(groups.keys()):
        pack = groups[(dataset, data_dir, model, cr)]

        # ---------- per-press avg (baseline + variants + any other press present) ----------
        for press, metrics in pack.items():
            avg, used_cnt, skipped_cnt = avg_of_metrics(metrics)
            avg_rows.append({
                "dataset": dataset,
                "data_dir": data_dir,
                "model": model,
                "cr": f"{cr:.2f}",
                "press": press,
                "avg": "" if avg is None else f"{avg:.6f}",
                "used_task_cnt": str(used_cnt),
                "skipped_non_numeric": str(skipped_cnt),
            })

        base_metrics = pack.get(base_press)
        if base_metrics is None:
            continue

        base_avg, _, _ = avg_of_metrics(base_metrics)

        # task keys (union) for detailed delta rows / missing rows
        all_task_keys = set(base_metrics.keys())
        for vpress in variants:
            if vpress in pack:
                all_task_keys |= set(pack[vpress].keys())
        all_task_keys = sorted(all_task_keys)

        for vpress in variants:
            vmetrics = pack.get(vpress)

            # variant missing: still output summary + per-task missing rows
            if vmetrics is None:
                summary_rows.append({
                    "dataset": dataset,
                    "data_dir": data_dir,
                    "model": model,
                    "cr": f"{cr:.2f}",
                    "variant": vpress,
                    "avg_base": "" if base_avg is None else f"{base_avg:.6f}",
                    "avg_variant": "",
                    "avg_delta": "",
                    "comparable_tasks": "0",
                    "improved_cnt": "0",
                    "worsened_cnt": "0",
                    "tied_cnt": "0",
                    "missing_cnt": str(len(all_task_keys)),
                    "skipped_non_numeric": "0",
                    "top_improved": "",
                    "top_worsened": "",
                    "note": "variant_missing",
                })
                for task in all_task_keys:
                    delta_rows.append({
                        "dataset": dataset,
                        "data_dir": data_dir,
                        "model": model,
                        "cr": f"{cr:.2f}",
                        "variant": vpress,
                        "task": task,
                        "base": "",
                        "variant_score": "",
                        "delta": "",
                        "direction": "MISSING",
                        "note": "variant_missing",
                    })
                continue

            v_avg, _, _ = avg_of_metrics(vmetrics)
            avg_delta = (v_avg - base_avg) if (v_avg is not None and base_avg is not None) else None

            improved = []
            worsened = []
            tied = []
            skipped = 0
            missing = 0

            for task in all_task_keys:
                b_raw = base_metrics.get(task, None)
                v_raw = vmetrics.get(task, None)

                if b_raw is None or v_raw is None:
                    missing += 1
                    delta_rows.append({
                        "dataset": dataset,
                        "data_dir": data_dir,
                        "model": model,
                        "cr": f"{cr:.2f}",
                        "variant": vpress,
                        "task": task,
                        "base": "" if b_raw is None else str(b_raw),
                        "variant_score": "" if v_raw is None else str(v_raw),
                        "delta": "",
                        "direction": "MISSING",
                        "note": "task_missing_in_base_or_variant",
                    })
                    continue

                b = to_float(b_raw)
                v = to_float(v_raw)
                if b is None or v is None:
                    skipped += 1
                    delta_rows.append({
                        "dataset": dataset,
                        "data_dir": data_dir,
                        "model": model,
                        "cr": f"{cr:.2f}",
                        "variant": vpress,
                        "task": task,
                        "base": "" if b is None else f"{b:.6f}",
                        "variant_score": "" if v is None else f"{v:.6f}",
                        "delta": "",
                        "direction": "SKIPPED",
                        "note": "non_numeric_or_ambiguous_metric_value",
                    })
                    continue

                d = v - b
                if d > EPS:
                    direction = "IMPROVED"
                    improved.append((task, d, b, v))
                elif d < -EPS:
                    direction = "WORSENED"
                    worsened.append((task, d, b, v))
                else:
                    direction = "TIED"
                    tied.append((task, d, b, v))

                delta_rows.append({
                    "dataset": dataset,
                    "data_dir": data_dir,
                    "model": model,
                    "cr": f"{cr:.2f}",
                    "variant": vpress,
                    "task": task,
                    "base": f"{b:.6f}",
                    "variant_score": f"{v:.6f}",
                    "delta": f"{d:.6f}",
                    "direction": direction,
                    "note": "",
                })

            improved.sort(key=lambda x: x[1], reverse=True)
            worsened.sort(key=lambda x: x[1])  # most negative first

            def top_list(items, k=5):
                return ";".join([f"{t}({d:+.3f})" for t, d, _, _ in items[:k]])

            comparable = len(improved) + len(worsened) + len(tied)

            summary_rows.append({
                "dataset": dataset,
                "data_dir": data_dir,
                "model": model,
                "cr": f"{cr:.2f}",
                "variant": vpress,
                "avg_base": "" if base_avg is None else f"{base_avg:.6f}",
                "avg_variant": "" if v_avg is None else f"{v_avg:.6f}",
                "avg_delta": "" if avg_delta is None else f"{avg_delta:.6f}",
                "comparable_tasks": str(comparable),
                "improved_cnt": str(len(improved)),
                "worsened_cnt": str(len(worsened)),
                "tied_cnt": str(len(tied)),
                "missing_cnt": str(missing),
                "skipped_non_numeric": str(skipped),
                "top_improved": top_list(improved, 5),
                "top_worsened": top_list(worsened, 5),
                "note": "",
            })

    # ---------- write CSVs ----------
    out_csv = Path(args.out_csv)
    summary_csv = Path(args.summary_csv)
    avg_csv = Path(args.avg_csv)

    with out_csv.open("w", newline="") as f:
        fieldnames = [
            "dataset","data_dir","model","cr","variant","task",
            "base","variant_score","delta","direction","note"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in delta_rows:
            w.writerow(r)

    with summary_csv.open("w", newline="") as f:
        fieldnames = [
            "dataset","data_dir","model","cr","variant",
            "avg_base","avg_variant","avg_delta",
            "comparable_tasks","improved_cnt","worsened_cnt","tied_cnt",
            "missing_cnt","skipped_non_numeric",
            "top_improved","top_worsened","note"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    with avg_csv.open("w", newline="") as f:
        fieldnames = [
            "dataset","data_dir","model","cr","press",
            "avg","used_task_cnt","skipped_non_numeric"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in avg_rows:
            w.writerow(r)

    print(f"[DONE] wrote {out_csv} ({len(delta_rows)} rows), {summary_csv} ({len(summary_rows)} rows), {avg_csv} ({len(avg_rows)} rows)")


if __name__ == "__main__":
    main()
