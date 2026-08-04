#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def read_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)


def json_hash(obj) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="root results dir")
    ap.add_argument("--out_prefix", type=str, default="quant_compare")
    args = ap.parse_args()

    root = Path(args.root)
    assert root.exists(), f"root not found: {root}"

    rows = []
    for mem_path in root.rglob("gpu_mem.json"):
        run_dir = mem_path.parent
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.yaml"

        if not metrics_path.exists():
            # 有些目录可能只有 mem 没有 metrics（或被跳过），直接略过
            continue

        mem = read_json(mem_path)
        metrics = read_json(metrics_path)

        cfg = {}
        if config_path.exists():
            # config.yaml 是你保存的 asdict(EvaluationConfig)，YAML 也能当 JSON 读一部分
            # 这里不用 yaml 库：简单处理（只读取我们关心字段），避免环境缺包
            try:
                import yaml  # type: ignore
                cfg = yaml.safe_load(config_path.read_text()) or {}
            except Exception:
                cfg = {}

        dataset = cfg.get("dataset", None)
        task = cfg.get("data_dir", None)
        press = cfg.get("press_name", None)
        cr = cfg.get("compression_ratio", None)
        frac = cfg.get("fraction", None)
        model = cfg.get("model", None)
        seed = cfg.get("seed", None)

        nbits = cfg.get("kv_cache_nbits", mem.get("kv_cache_nbits", None))
        backend = cfg.get("kv_cache_backend", mem.get("kv_cache_backend", None))
        kvq_mode = "none" if nbits in (None, "null") else str(nbits)

        # 有些 metrics.json 是 {task_name: {...}}；有些可能更复杂
        m_hash = json_hash(metrics)

        # 如果 metrics 里只有一个 task key，顺便取出那个 key（便于你核对）
        metrics_top_keys = list(metrics.keys()) if isinstance(metrics, dict) else []
        metrics_task_key = metrics_top_keys[0] if len(metrics_top_keys) == 1 else None

        rows.append(
            dict(
                run_dir=str(run_dir),
                dataset=dataset,
                task=task,
                metrics_task_key=metrics_task_key,
                press=press,
                cr=cr,
                fraction=frac,
                model=model,
                seed=seed,
                kvq_mode=kvq_mode,
                kvq_backend=backend,
                peak_alloc_mb=mem.get("peak_alloc_mb", None),
                peak_reserved_mb=mem.get("peak_reserved_mb", None),
                device_mem_mb_end=mem.get("device_mem_mb_end", None),
                metrics_hash=m_hash,
            )
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print("[ERROR] No runs found under root (no gpu_mem.json + metrics.json pairs).")
        return

    # 关键：对齐同一个 (dataset, task, press, cr, fraction, model, seed) 下的 kvq_mode
    keys = ["dataset", "task", "press", "cr", "fraction", "model", "seed"]
    df["cr"] = df["cr"].astype(float)

    # 1) 内存 pivot
    mem_pivot = df.pivot_table(
        index=keys,
        columns="kvq_mode",
        values=["peak_alloc_mb", "peak_reserved_mb", "device_mem_mb_end"],
        aggfunc="first",
    )

    # 2) metrics hash pivot（用 hash 看是否完全一致）
    met_pivot = df.pivot_table(
        index=keys,
        columns="kvq_mode",
        values="metrics_hash",
        aggfunc="first",
    )

    # 3) 生成“一致性”报告
    report_rows = []
    for idx, sub in df.groupby(keys, dropna=False):
        modes = sorted(sub["kvq_mode"].dropna().unique().tolist())
        # peak_alloc 是否一致（严格）
        pa = sub[["kvq_mode", "peak_alloc_mb"]].dropna()
        pa_same = (pa["peak_alloc_mb"].nunique() <= 1) if not pa.empty else False
        # metrics 是否一致（严格）
        mh = sub[["kvq_mode", "metrics_hash"]].dropna()
        mh_same = (mh["metrics_hash"].nunique() <= 1) if not mh.empty else False

        row = dict(zip(keys, idx if isinstance(idx, tuple) else (idx,)))
        row.update(
            dict(
                kvq_modes=",".join(modes),
                peak_alloc_same=bool(pa_same),
                metrics_same=bool(mh_same),
                peak_alloc_values={r.kvq_mode: r.peak_alloc_mb for r in pa.itertuples(index=False)},
                metrics_hash_values={r.kvq_mode: r.metrics_hash for r in mh.itertuples(index=False)},
            )
        )
        report_rows.append(row)

    report = pd.DataFrame(report_rows)

    # 输出文件
    out_prefix = Path(args.out_prefix)
    df.to_csv(out_prefix.with_suffix(".all_runs.csv"), index=False)
    mem_pivot.to_csv(out_prefix.with_suffix(".mem_pivot.csv"))
    met_pivot.to_csv(out_prefix.with_suffix(".metrics_hash_pivot.csv"))
    report.to_csv(out_prefix.with_suffix(".consistency_report.csv"), index=False)

    # 控制台摘要
    total = len(report)
    both_same = int(((report["peak_alloc_same"] == True) & (report["metrics_same"] == True)).sum())
    mem_diff = int((report["peak_alloc_same"] == False).sum())
    met_diff = int((report["metrics_same"] == False).sum())

    print(f"[OK] scanned runs = {len(df)}")
    print(f"[OK] groups (dataset/task/press/cr/frac/model/seed) = {total}")
    print(f"[SUMMARY] peak_alloc & metrics both same: {both_same}/{total}")
    print(f"[SUMMARY] peak_alloc differs in groups: {mem_diff}/{total}")
    print(f"[SUMMARY] metrics differs in groups: {met_diff}/{total}")
    print(f"[FILES] {out_prefix.with_suffix('.all_runs.csv')}")
    print(f"[FILES] {out_prefix.with_suffix('.mem_pivot.csv')}")
    print(f"[FILES] {out_prefix.with_suffix('.metrics_hash_pivot.csv')}")
    print(f"[FILES] {out_prefix.with_suffix('.consistency_report.csv')}")


if __name__ == "__main__":
    main()
