"""Measure sentence-level redundancy per RULER task, to see which tasks a query-independent
router can compress safely and which it cannot.

The hypothesis under test: niah_multikey_2/3 are hard because they add DISTRACTOR needles --
lines that look exactly like the target and are indistinguishable without knowing the query.
qa_1/2 are hard for a different reason (the answer span is not lexically marked at all).
And the reason a frozen score can still work at all is that some tasks contain heavily REPEATED
filler that is safe to drop regardless of the query.

For each task this reports:
  * n_lines / n_unique       -- exact-duplicate line rate (the compressible-for-free part)
  * top repeated line        -- what the filler actually is
  * needle-like line count   -- lines matching the task's needle pattern, i.e. how many
                               candidates a router must keep because it cannot tell them apart
  * answer-in-context        -- whether the gold answer is even lexically present
"""
import re
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, "/apdcephfs_tj5/share_300719894/user/guhao/kvpress_clean")
sys.path.insert(0, "/apdcephfs_tj5/share_300719894/user/guhao/kvpress_clean/evaluation")

from evaluate_sparse import DATASET_REGISTRY, load_cached_dataset  # noqa: E402

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "8192"
FOCUS = ["niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "qa_1", "qa_2",
         "niah_single_1", "niah_multivalue", "cwe", "vt"]

df = load_cached_dataset(DATASET_REGISTRY["ruler"], DATA_DIR).to_pandas()
df = df.sample(frac=0.1, random_state=42)
print(f"RULER {DATA_DIR}, fraction 0.1, seed 42 -> {len(df)} rows\n")

NEEDLE_PATS = {
    "niah": re.compile(r"One of the special magic", re.I),
    "uuid": re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}", re.I),
}

rows = []
for task in FOCUS:
    sub = df[df["task"] == task]
    if not len(sub):
        continue
    r = sub.iloc[0]
    ctx = r["context"]
    lines = [ln.strip() for ln in ctx.split("\n") if ln.strip()]
    # sentence-level too: RULER's haystack is one long paragraph for several tasks
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ctx) if len(s.strip()) > 15]
    uniq_l, uniq_s = len(set(lines)), len(set(sents))
    cnt = Counter(sents)
    top_s, top_n = cnt.most_common(1)[0] if cnt else ("", 0)

    needles = [ln for ln in lines if NEEDLE_PATS["niah"].search(ln)]
    if not needles:
        needles = [s for s in sents if NEEDLE_PATS["niah"].search(s)]
    ans = r["answer"]
    ans_list = list(ans) if isinstance(ans, (list, tuple)) else [ans]
    ans_str = str(ans_list[0])
    in_ctx = ans_str.lower() in ctx.lower()

    rows.append(dict(
        task=task, n_lines=len(lines), uniq_lines=uniq_l,
        dup_line_rate=1 - uniq_l / max(len(lines), 1),
        n_sents=len(sents), dup_sent_rate=1 - uniq_s / max(len(sents), 1),
        top_repeat=top_n, needle_like=len(needles), answer_in_ctx=in_ctx,
    ))

out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

print("\n--- most repeated sentence per task (the free-to-evict filler) ---")
for task in FOCUS:
    sub = df[df["task"] == task]
    if not len(sub):
        continue
    ctx = sub.iloc[0]["context"]
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ctx) if len(s.strip()) > 15]
    c = Counter(sents)
    if not c:
        continue
    s, n = c.most_common(1)[0]
    print(f"  {task:18s} x{n:4d}  {s[:100]!r}")

print("\n--- needle-like lines: what the router must disambiguate ---")
for task in ["niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "niah_multivalue"]:
    sub = df[df["task"] == task]
    if not len(sub):
        continue
    r = sub.iloc[0]
    ctx, q = r["context"], r["question"]
    lines = [ln.strip() for ln in ctx.split("\n") if ln.strip()]
    needles = [ln for ln in lines if NEEDLE_PATS["niah"].search(ln)]
    if not needles:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ctx) if s.strip()]
        needles = [s for s in sents if NEEDLE_PATS["niah"].search(s)]
    ans = r["answer"]
    ans_str = str(list(ans)[0] if isinstance(ans, (list, tuple)) else ans)
    print(f"\n  == {task}: {len(needles)} needle-like lines")
    print(f"     question: {str(q)[:110]!r}")
    print(f"     answer:   {ans_str[:60]!r}")
    for n in needles[:4]:
        hit = "<== GOLD" if ans_str.lower() in n.lower() else ""
        print(f"       {n[:105]!r} {hit}")
    if len(needles) > 4:
        print(f"       ... and {len(needles)-4} more")
