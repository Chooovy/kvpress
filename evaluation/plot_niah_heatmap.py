import os
import re
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= 配置 =========
ROOT = "./results_needle"     # 你的 output_dir
OUT_DIR = "./niah_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# 你实验扫的网格（按你现在跑 1k~12.5k）
CONTEXTS = [1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,10500,11000,11500,12000,12500]
DEPTHS   = [15,25,35,45,55,65,75,85,95]

# 目录名解析：needle_in_haystack__...__<press>__cr0.50__max_context12500__needle_depth75
# DIR_RE = re.compile(
#     r"needle_in_haystack__.*?__(?P<press>.+?)__cr(?P<cr>\d+\.\d+)__.*?max_context(?P<ctx>\d+)__.*?needle_depth(?P<depth>\d+)"
# )
DIR_RE = re.compile(
    r"needle_in_haystack__"
    r"(?P<model>.+?)__"
    r"(?P<press>.+?)__"
    r"cr(?P<cr>\d+\.\d+)__"
    r".*?max_context(?P<ctx>\d+)__"
    r".*?needle_depth(?P<depth>\d+)"
)

def find_metrics_json(run_dir: str) -> str | None:
    """
    递归找 metrics*.json。兼容：
      run_dir/metrics.json
      run_dir/1/metrics.json
      run_dir/**/metrics*.json
    """
    cand = glob.glob(os.path.join(run_dir, "**", "metrics*.json"), recursive=True)
    if not cand:
        return None
    # 选最短路径的一份（通常就是最外层或 /1/ 层）
    cand.sort(key=lambda p: (p.count(os.sep), len(p)))
    return cand[0]

def rougeL_f_from_metrics(metrics_path: str) -> float:
    """
    你的 metrics.json 结构示例是 list[dict]，其中 rouge-l: {r,p,f}
    返回 rouge-l 的 f (float)。取不到就返回 NaN。
    """
    obj = json.load(open(metrics_path, "r"))

    if isinstance(obj, list):
        obj = obj[0] if obj else {}

    rl = obj.get("rouge-l", None)
    if isinstance(rl, dict) and "f" in rl:
        return float(rl["f"])

    return float("nan")

# ========= 收集点 =========
points = []  # press, cr, ctx, depth, score

for d in glob.glob(os.path.join(ROOT, "needle_in_haystack__*")):
    base = os.path.basename(d)
    m = DIR_RE.search(base)
    if not m:
        continue

    press = m.group("press")
    cr = float(m.group("cr"))
    ctx = int(m.group("ctx"))
    depth = int(m.group("depth"))

    mp = find_metrics_json(d)
    if mp is None:
        # 没找到 metrics：记 NaN（也可以选择直接跳过）
        score = float("nan")
    else:
        score = rougeL_f_from_metrics(mp)

    points.append((press, cr, ctx, depth, score))

if not points:
    raise RuntimeError(f"No NIAH results found under: {ROOT}")

dfp = pd.DataFrame(points, columns=["press", "cr", "ctx", "depth", "score"])
print(f"[INFO] collected points: {len(dfp)}")
print("[INFO] score stats (non-nan):",
      dfp["score"].dropna().describe().to_string() if dfp["score"].notna().any() else "ALL NaN")

# ========= 画图 =========
idx_depth = {d: i for i, d in enumerate(DEPTHS)}
idx_ctx = {c: i for i, c in enumerate(CONTEXTS)}

for (press, cr), g in dfp.groupby(["press", "cr"], sort=True):
    mat = np.full((len(DEPTHS), len(CONTEXTS)), np.nan, dtype=float)

    for _, r in g.iterrows():
        if r["depth"] in idx_depth and r["ctx"] in idx_ctx:
            mat[idx_depth[r["depth"]], idx_ctx[r["ctx"]]] = r["score"]

    # debug：看看是不是常数/全 NaN（避免又遇到“一片颜色”但不知道原因）
    non_nan = mat[~np.isnan(mat)]
    uniq = np.unique(non_nan) if non_nan.size else []
    print(f"[DEBUG] press={press} cr={cr:.2f} filled={non_nan.size}/{mat.size} unique(non-nan)={uniq[:10]}{'...' if len(uniq)>10 else ''}")

    # colormap：NaN 灰色
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")  # NaN 显示灰色:contentReference[oaicite:3]{index=3}

    plt.figure(figsize=(12, 4))
    im = plt.imshow(
        mat,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0.0, vmax=1.0,   # 固定范围 0~1，避免 vmin≈vmax 导致色条怪:contentReference[oaicite:4]{index=4}
    )
    plt.colorbar(im, label="ROUGE-L F1 (0~1)")

    plt.xticks(range(len(CONTEXTS)), [str(c) for c in CONTEXTS], rotation=45, ha="right")
    plt.yticks(range(len(DEPTHS)), [str(d) for d in DEPTHS])
    plt.xlabel("Context Length (tokens)")
    plt.ylabel("Depth Percent")
    plt.title(f"NIAH Heatmap | press={press} | cr={cr:.2f} (ROUGE-L F1)")

    out = os.path.join(OUT_DIR, f"niah__{press}__cr{cr:.2f}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"[OK] saved: {out}")
