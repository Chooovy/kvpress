#!/usr/bin/env python3
# plot_summary_common_tasks.py

import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _safe_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="summary_common_tasks.csv path")
    ap.add_argument("--out_dir", type=str, required=True, help="output dir for pngs")
    ap.add_argument("--combined", action="store_true", help="also save a combined multi-model figure")
    ap.add_argument("--x_label", type=str, default="Compression Ratio (cr)")
    ap.add_argument("--y_label", type=str, default="Avg Score (mean over common tasks)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # expected columns: model, press, cr, avg_score, n_tasks, tasks
    needed = {"model", "press", "cr", "avg_score"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in csv: {sorted(missing)}")

    df["cr"] = pd.to_numeric(df["cr"], errors="coerce")
    df["avg_score"] = pd.to_numeric(df["avg_score"], errors="coerce")
    df = df.dropna(subset=["model", "press", "cr", "avg_score"])

    models = sorted(df["model"].unique().tolist())

    # ---- 1) one figure per model ----
    for model in models:
        sub = df[df["model"] == model].copy()
        sub = sub.sort_values(["press", "cr"])

        # Pivot: rows=cr, cols=press
        piv = sub.pivot_table(index="cr", columns="press", values="avg_score", aggfunc="mean")
        piv = piv.sort_index()

        plt.figure(figsize=(7.5, 4.8))

        for press in piv.columns:
            y = piv[press].dropna()
            if y.empty:
                continue
            x = y.index.values
            plt.plot(x, y.values, marker="o", linewidth=2, label=str(press))

        # Optional: show how many common tasks were used (best-effort)
        title = f"{model} (common-task mean)"
        if "n_tasks" in sub.columns:
            # n_tasks should be same across rows, but take max just in case
            try:
                nt = int(pd.to_numeric(sub["n_tasks"], errors="coerce").max())
                if nt > 0:
                    title += f" | n_tasks={nt}"
            except Exception:
                pass

        plt.title(title)
        plt.xlabel(args.x_label)
        plt.ylabel(args.y_label)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", frameon=True)
        plt.tight_layout()

        out_path = out_dir / f"longbench_common_tasks__{_safe_name(model)}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[OK] saved: {out_path}")

    # ---- 2) optional combined figure ----
    if args.combined and len(models) > 1:
        n = len(models)
        fig_w = max(10, 4.8 * n)
        plt.figure(figsize=(fig_w, 4.6))

        for i, model in enumerate(models, 1):
            ax = plt.subplot(1, n, i)
            sub = df[df["model"] == model].copy()
            sub = sub.sort_values(["press", "cr"])
            piv = sub.pivot_table(index="cr", columns="press", values="avg_score", aggfunc="mean").sort_index()

            for press in piv.columns:
                y = piv[press].dropna()
                if y.empty:
                    continue
                x = y.index.values
                ax.plot(x, y.values, marker="o", linewidth=2, label=str(press))

            ax.set_title(model)
            ax.set_xlabel(args.x_label)
            if i == 1:
                ax.set_ylabel(args.y_label)
            ax.grid(True, alpha=0.3)

        # Put a single legend on the right
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

        plt.tight_layout()
        out_path = out_dir / "longbench_common_tasks__ALL_MODELS.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()
