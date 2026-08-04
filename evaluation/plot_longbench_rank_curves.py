#!/usr/bin/env python3
# plot_longbench_rank_curves.py

import argparse
from pathlib import Path
import math

import pandas as pd
import matplotlib.pyplot as plt

# No seaborn; no explicit colors (use matplotlib defaults).

def _nice_grid(n: int, max_cols: int = 4):
    """Return (nrows, ncols) for n panels."""
    if n <= 0:
        return (1, 1)
    ncols = min(max_cols, max(1, int(math.ceil(math.sqrt(n)))))
    nrows = int(math.ceil(n / ncols))
    return nrows, ncols

def plot_by_task(df_fam: pd.DataFrame, family: str, out_path: Path, max_cols: int = 4, presses=None):
    # panels = tasks, lines = presses
    tasks = sorted(df_fam["task"].unique().tolist())
    # 如果显式给 presses，就用固定顺序；否则用数据里出现的
    if presses is None:
        presses = sorted(df_fam["press"].unique().tolist())
    nrows, ncols = _nice_grid(len(tasks), max_cols=max_cols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), sharex=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax_i, task in enumerate(tasks):
        ax = axes[ax_i]
        d_task = df_fam[df_fam["task"] == task]

        for press in presses:
            d_line = d_task[d_task["press"] == press].sort_values("cr")
            if d_line.empty:
                continue
            ax.plot(d_line["cr"], d_line["avg_score"], marker="o", linewidth=1.5, label=press)

        ax.set_title(task)
        ax.set_xlabel("cr")
        ax.set_ylabel("avg_score")
        ax.grid(True, linewidth=0.5, alpha=0.5)

    # hide extra axes
    for j in range(len(tasks), len(axes)):
        axes[j].axis("off")

    # global legend: 汇总所有子图的 handles，避免 axes[0] 缺线导致 legend 只有一条
    label2handle = {}
    for k in range(len(tasks)):
        hs, ls = axes[k].get_legend_handles_labels()
        for h, l in zip(hs, ls):
            label2handle.setdefault(l, h)
    if label2handle:
        labels = list(label2handle.keys())
        handles = [label2handle[l] for l in labels]
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=True)

    fig.suptitle(f"LongBench curves (by task) | family={family}", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_by_press(df_fam: pd.DataFrame, family: str, out_path: Path, max_cols: int = 4):
    # panels = presses, lines = tasks
    presses = sorted(df_fam["press"].unique().tolist())
    tasks = sorted(df_fam["task"].unique().tolist())
    nrows, ncols = _nice_grid(len(presses), max_cols=max_cols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), sharex=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax_i, press in enumerate(presses):
        ax = axes[ax_i]
        d_press = df_fam[df_fam["press"] == press]

        for task in tasks:
            d_line = d_press[d_press["task"] == task].sort_values("cr")
            if d_line.empty:
                continue
            ax.plot(d_line["cr"], d_line["avg_score"], marker="o", linewidth=1.5, label=task)

        ax.set_title(press)
        ax.set_xlabel("cr")
        ax.set_ylabel("avg_score")
        ax.grid(True, linewidth=0.5, alpha=0.5)

    for j in range(len(presses), len(axes)):
        axes[j].axis("off")

    # global legend (tasks)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        # put legend outside to avoid clutter
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6), frameon=True)

    fig.suptitle(f"LongBench curves (by press) | family={family}", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
def plot_llama_focus_4tasks(df: pd.DataFrame, out_path: Path):
    family = "llama3-8b"
    tasks_keep = ["hotpotqa", "multifieldqa_en", "trec", "triviaqa"]
    press_drop = "query_indexer_max_mode"

    d = df[(df["family"] == family) & (df["task"].isin(tasks_keep)) & (df["press"] != press_drop)].copy()
    if d.empty:
        print(f"[WARN] llama focus plot skipped: no data for family={family} tasks={tasks_keep} excluding {press_drop}")
        return

    # 固定顺序，保证每次画出来一致
    presses = sorted(d["press"].unique().tolist())
    tasks = tasks_keep  # 按你指定顺序

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes = axes.flatten()

    for i, task in enumerate(tasks):
        ax = axes[i]
        d_task = d[d["task"] == task]

        for press in presses:
            d_line = d_task[d_task["press"] == press].sort_values("cr")
            if d_line.empty:
                continue
            ax.plot(d_line["cr"], d_line["avg_score"], marker="o", linewidth=1.8, label=press)

        ax.set_title(task)
        ax.set_xlabel("cr")
        ax.set_ylabel("avg_score")
        ax.grid(True, linewidth=0.5, alpha=0.5)

    # 全局 legend（方法名）
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), frameon=True)

    fig.suptitle(f"LongBench | {family} | 4 tasks | exclude {press_drop}", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[OK] saved: {out_path}")
def plot_memory_compare_by_task(df: pd.DataFrame, out_dir: Path, family: str = None, max_cols: int = 4):
    # 只画 memory 系列 + memory_query_indexer_max
    memory_presses = ["memory_query_indexer_max", "memory_EA", "memory_snapkv", "memory_keydiff"]

    d = df[df["press"].isin(memory_presses)].copy()
    if family is not None:
        d = d[d["family"] == family].copy()

    if d.empty:
        print(f"[WARN] memory compare skipped: no rows for presses={memory_presses} family={family}")
        return

    fams = sorted(d["family"].unique().tolist())
    for fam in fams:
        d_fam = d[d["family"] == fam]
        # 🔍 debug：确保不止一条线
        print(f"[DEBUG] memory_compare family={fam} presses={sorted(d_fam['press'].unique().tolist())}")

        out_path = out_dir / f"longbench_{fam}_memory_compare_by_task.png"
        # 传入固定 presses 顺序，保证四条线都被尝试画出来
        plot_by_task(d_fam, fam, out_path, max_cols=max_cols, presses=memory_presses)
        
        print(f"[OK] saved: {out_path}")
def plot_pair_compare_by_task(df: pd.DataFrame, out_dir: Path, family: str = None, max_cols: int = 4):
    # 三组两两对比：每组单独出图
    pair_sets = [
        ("snapkv_vs_memory_snapkv", ["snapkv", "memory_snapkv"]),
        ("ea_vs_memory_ea", ["expected_attention", "memory_EA"]),
        ("indexer_vs_memory_indexer", ["query_indexer_max_mode", "memory_query_indexer_max"]),
    ]

    d = df.copy()
    if family is not None:
        d = d[d["family"] == family].copy()

    fams = sorted(d["family"].unique().tolist())
    for fam in fams:
        d_fam_all = d[d["family"] == fam].copy()
        if d_fam_all.empty:
            continue

        for tag, presses in pair_sets:
            d_fam = d_fam_all[d_fam_all["press"].isin(presses)].copy()
            if d_fam.empty:
                print(f"[WARN] pair_compare skipped: family={fam} tag={tag} presses={presses} (no rows)")
                continue

            print(f"[DEBUG] pair_compare family={fam} tag={tag} presses={sorted(d_fam['press'].unique().tolist())}")

            out_path = out_dir / f"longbench_{fam}_{tag}_by_task.png"
            # 每张图只有两条线
            plot_by_task(d_fam, fam, out_path, max_cols=max_cols, presses=presses)
            print(f"[OK] saved: {out_path}")
def plot_mistral_indexer_4tasks(df: pd.DataFrame, out_dir: Path, max_cols: int = 4):
    family = "mistral-7b"
    tasks_keep = ["multi_news", "passage_retrieval_zh", "qasper", "trec"]
    presses = ["query_indexer_max_mode", "memory_query_indexer_max"]

    d = df[(df["family"] == family) &
           (df["task"].isin(tasks_keep)) &
           (df["press"].isin(presses))].copy()

    if d.empty:
        print(f"[WARN] mistral indexer 4tasks skipped: no rows for family={family}, tasks={tasks_keep}, presses={presses}")
        return

    # 保证子图顺序按你指定的 tasks_keep
    d["task"] = pd.Categorical(d["task"], categories=tasks_keep, ordered=True)
    d = d.sort_values(["task", "press", "cr"])

    out_path = out_dir / f"longbench_{family}_indexer_vs_memory_4tasks.png"
    plot_by_task(d, family, out_path, max_cols=max_cols, presses=presses)
    print(f"[OK] saved: {out_path}")
def plot_llama_snapkv_4tasks(df: pd.DataFrame, out_dir: Path, max_cols: int = 4):
    family = "llama3-8b"
    tasks_keep = ["lcc", "musique", "passage_count", "triviaqa"]
    presses = ["snapkv", "memory_snapkv"]

    d = df[(df["family"] == family) &
           (df["task"].isin(tasks_keep)) &
           (df["press"].isin(presses))].copy()

    if d.empty:
        print(f"[WARN] llama snapkv 4tasks skipped: no rows for family={family}, tasks={tasks_keep}, presses={presses}")
        return

    # 保证子图顺序按你指定 tasks_keep
    d["task"] = pd.Categorical(d["task"], categories=tasks_keep, ordered=True)
    d = d.sort_values(["task", "press", "cr"])

    out_path = out_dir / f"longbench_{family}_snapkv_vs_memory_4tasks.png"
    plot_by_task(d, family, out_path, max_cols=max_cols, presses=presses)
    print(f"[OK] saved: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="/aifs4su/guhao/KVCache/kvpress/evaluation/results_longbench_frac0.1/rank_by_task_intersection.csv")
    ap.add_argument("--out_dir", type=str, required=True, help="output dir for pngs")
    ap.add_argument("--family", type=str, default=None,
                    help="optional: only plot one family, e.g. llama3-8b / mistral-7b / qwen3-8b")
    ap.add_argument("--max_cols", type=int, default=4, help="max subplot columns")
    ap.add_argument("--llama_focus", action="store_true",
                help="also plot llama3-8b focus figure for 4 tasks excluding query_indexer_max_mode")
    ap.add_argument("--memory_compare", action="store_true",
                help="plot by-task curves for memory_query_indexer_max + (memory_EA, memory_snapkv, memory_keydiff)")
    ap.add_argument("--pair_compare", action="store_true",
            help="plot by-task curves for pairs: (snapkv vs memory_snapkv), (EA vs memory_EA), (indexer vs memory_query_indexer_max)")
    ap.add_argument("--mistral_indexer_4tasks", action="store_true",
                help="plot mistral-7b only: query_indexer_max_mode vs memory_query_indexer_max on 4 tasks (multi_news, passage_retrieval_zh, qasper, trec)")
    ap.add_argument("--llama_snapkv_4tasks", action="store_true",
                help="plot llama3-8b only: snapkv vs memory_snapkv on 4 tasks (lcc, musique, passage_count, triviaqa)")

    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    need_cols = {"family", "task", "cr", "press", "avg_score"}
    missing = need_cols - set(df.columns)
    if missing:
        raise SystemExit(f"[ERROR] missing columns in csv: {sorted(missing)}")

    df["cr"] = df["cr"].astype(float)
    df["avg_score"] = df["avg_score"].astype(float)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    families = sorted(df["family"].unique().tolist())
    if args.family is not None:
        families = [args.family]
        df = df[df["family"] == args.family]
        if df.empty:
            raise SystemExit(f"[ERROR] no rows for family={args.family}")

    for fam in families:
        d = df[df["family"] == fam]
        if d.empty:
            continue

        out1 = out_dir / f"longbench_{fam}_by_task.png"
        out2 = out_dir / f"longbench_{fam}_by_press.png"

        plot_by_task(d, fam, out1, max_cols=args.max_cols)
        plot_by_press(d, fam, out2, max_cols=args.max_cols)

        print(f"[OK] saved: {out1}")
        print(f"[OK] saved: {out2}")
    if args.llama_focus:
        out_focus = out_dir / "longbench_llama3-8b_focus4tasks_no_query_indexer.png"
        plot_llama_focus_4tasks(df, out_focus)
    
    if args.memory_compare:
        plot_memory_compare_by_task(df, out_dir, family=args.family, max_cols=args.max_cols)
    if args.pair_compare:
        plot_pair_compare_by_task(df, out_dir, family=args.family, max_cols=args.max_cols)
    if args.mistral_indexer_4tasks:
        plot_mistral_indexer_4tasks(df, out_dir, max_cols=args.max_cols)
    if args.llama_snapkv_4tasks:
        plot_llama_snapkv_4tasks(df, out_dir, max_cols=args.max_cols)


if __name__ == "__main__":
    main()
