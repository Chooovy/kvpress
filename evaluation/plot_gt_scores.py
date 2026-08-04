#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import numpy as np
import torch
import matplotlib.pyplot as plt

def discover_layers(gt_root: Path):
    # layer_0, layer_1, ...
    layers = []
    for p in gt_root.iterdir():
        if p.is_dir() and re.match(r"layer_\d+$", p.name):
            layers.append(p)
    layers.sort(key=lambda x: int(x.name.split("_")[-1]))
    if not layers:
        raise FileNotFoundError(f"No layer_* directories under: {gt_root}")
    return layers

def load_score_vec(layer_dir: Path, sample: int, step: int):
    # supports both naming styles:
    #  - gt_sample{sample}_step{step}.pt
    #  - gt_score_sample{sample}_step{step}.pt
    cand = [
        layer_dir / f"gt_sample{sample}_step{step}.pt",
        layer_dir / f"gt_score_sample{sample}_step{step}.pt",
    ]
    for f in cand:
        if f.exists():
            obj = torch.load(f, map_location="cpu")
            v = obj.get("gt_score_mean", None)
            if v is None:
                raise KeyError(f"{f} has no key 'gt_score_mean'. keys={list(obj.keys())}")
            ctx_len = int(obj.get("ctx_len", v.numel()))
            v = v[:ctx_len].detach().float().cpu().numpy()
            return v, ctx_len
    raise FileNotFoundError(f"Not found for layer_dir={layer_dir}, sample={sample}, step={step}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_root", type=str, required=True, help="e.g. ./pilot.../gt_tokens_cr0p50")
    ap.add_argument("--sample", type=int, default=1)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--out", type=str, default=None, help="output png path")
    ap.add_argument("--log", action="store_true", help="plot log(score+eps) for contrast")
    ap.add_argument("--eps", type=float, default=1e-30)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    gt_root = Path(args.gt_root)
    layers = discover_layers(gt_root)

    # load layer 0 to get ctx_len, then load all
    v0, ctx_len = load_score_vec(layers[0], args.sample, args.step)
    mat = np.zeros((len(layers), ctx_len), dtype=np.float32)
    mat[0] = v0

    for li, layer_dir in enumerate(layers[1:], start=1):
        v, ctx_len_i = load_score_vec(layer_dir, args.sample, args.step)
        if ctx_len_i != ctx_len:
            # 保险起见：不同层ctx_len不一致就截到最小
            new_len = min(ctx_len, ctx_len_i)
            mat = mat[:, :new_len]
            ctx_len = new_len
            mat[li] = v[:new_len]
        else:
            mat[li] = v

    # stride=1: 不降采样，直接画
    show = mat
    title = f"GT score_mean heatmap | sample={args.sample} step={args.step} ctx_len={ctx_len} stride=1"
    if args.log:
        show = np.log(show + args.eps)
        title += " | scale=log"

    # 图会很宽：按 ctx_len 自适应画布宽度
    fig_w = max(14, ctx_len / 260)   # 260 token ~ 1 inch（你可改）
    fig_h = max(6, len(layers) / 6)
    plt.figure(figsize=(fig_w, fig_h))
    # 让对比更明显：剪掉极端 2% / 98%（你也可以改成 1/99）
    vmin = np.percentile(show, 2)
    vmax = np.percentile(show, 98)
    im = plt.imshow(show, aspect="auto", interpolation="nearest",
                    cmap="gray_r", vmin=vmin, vmax=vmax)  # low=white, high=black
    plt.title(title)
    plt.ylabel("layer")
    plt.xlabel("token position (stride=1)")
    plt.colorbar(im, fraction=0.025, pad=0.02)

    out = args.out
    if out is None:
        out = str(gt_root / f"gt_heatmap_sample{args.sample}_step{args.step}_stride1.png")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=args.dpi)
    print(f"[OK] saved: {out}")

if __name__ == "__main__":
    main()
