"""Token budget vs irreducible content, per RULER task.

The redundancy scan showed multikey_2/3 have ZERO duplicate lines while single_1/vt are ~99%
duplicates. That reframes the problem: a query-independent router wins on tasks whose context is
mostly repeated filler, and cannot win on tasks whose context is entirely distinct candidates.

This measures, per task, how the topk=2048 budget compares to the tokens occupied by
indistinguishable candidate lines -- i.e. whether the budget is even sufficient in principle.
"""
import re
import sys

import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, "/apdcephfs_tj5/share_300719894/user/guhao/kvpress_clean")
sys.path.insert(0, "/apdcephfs_tj5/share_300719894/user/guhao/kvpress_clean/evaluation")
from evaluate_sparse import DATASET_REGISTRY, load_cached_dataset  # noqa: E402

MODEL = "/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B"
TOPK, FORCE_LOCAL, FORCE_SINK = 2048, 64, 4
tok = AutoTokenizer.from_pretrained(MODEL)

NEEDLE = re.compile(r"One of the special magic", re.I)
TASKS = ["niah_single_1", "niah_single_3", "niah_multikey_1", "niah_multikey_2",
         "niah_multikey_3", "niah_multivalue", "niah_multiquery", "vt", "cwe", "qa_1", "qa_2"]

for DATA_DIR in ("8192", "16384"):
    df = load_cached_dataset(DATASET_REGISTRY["ruler"], DATA_DIR).to_pandas()
    df = df.sample(frac=0.1, random_state=42)
    rows = []
    for t in TASKS:
        sub = df[df["task"] == t]
        if not len(sub):
            continue
        r = sub.iloc[0]
        ctx = r["context"]
        n_ctx = len(tok(ctx).input_ids)
        lines = [ln.strip() for ln in ctx.split("\n") if ln.strip()]
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ctx) if len(s.strip()) > 15]
        units = lines if len(lines) > 10 else sents
        cand = [u for u in units if NEEDLE.search(u)]
        # tokens in indistinguishable candidates
        cand_tok = sum(len(tok(c).input_ids) for c in cand) if cand else 0
        uniq = set(units)
        uniq_tok = sum(len(tok(u).input_ids) for u in uniq)
        take = TOPK - FORCE_LOCAL - FORCE_SINK
        rows.append(dict(
            task=t, ctx_tok=n_ctx, keep_frac=TOPK / n_ctx,
            n_cand=len(cand), cand_tok=cand_tok,
            cand_over_budget=(cand_tok / take) if cand_tok else 0.0,
            uniq_tok=uniq_tok, uniq_over_budget=uniq_tok / take,
        ))
    out = pd.DataFrame(rows)
    print(f"\n===== RULER {DATA_DIR}  (topk={TOPK}, take={TOPK-FORCE_LOCAL-FORCE_SINK}) =====")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("  cand_over_budget > 1  => the indistinguishable candidates ALONE exceed the budget:")
    print("                          no query-independent score can fit them, by counting alone.")
