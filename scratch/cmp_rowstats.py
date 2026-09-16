"""Row counts + paired per-row deltas for the CMP arms.

Why this exists: the reported RULER MEAN is an *unweighted* mean over 13 tasks of unequal size (43 to
66 rows), so a small task moves it as much as a large one, and a delta of a few tenths can be a
handful of questions. Both arms answer the same questions in the same order (verified: the `question`
column matches on 100% of rows), so the comparison is **paired** -- re-score every row and count how
many actually changed.

**Parsing the `answer` column is the trap.** It is a *numpy* repr, not a Python literal:

    "['bush' 'swing' 'infinite']"      <- space separated, NO commas

`ast.literal_eval` on that does not fail, it silently concatenates into a single string
`'bushswinginfinite'`, which scores 0.00 on cwe. Reading it as a raw string instead makes `for r in
ref` iterate *characters*, and single characters are trivially contained in any prediction -- that
inflates cwe to 78.22. Both wrong, in opposite directions, neither raising. The scorer itself gets
this right because it receives the real list from the dataframe before any csv round-trip; this
report has to reproduce it with a regex. Verified against `metrics.json` per task before use.
"""

import glob
import json
import re

import pandas as pd

from evaluation.benchmarks.ruler.calculate_metrics import string_match_all, string_match_part

TASKS = [
    "cwe", "fwe", "qa_1", "qa_2",
    "niah_multiquery", "niah_multivalue", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_single_1", "niah_single_2", "niah_single_3", "vt",
]


def parse_answer(a):
    """numpy-style `['x' 'y']` -> ['x', 'y']. Falls back to a single-element list."""
    if not isinstance(a, str):
        return [str(a)]
    s = a.strip()
    if s.startswith("[") and s.endswith("]"):
        parts = re.findall(r"'([^']*)'|\"([^\"]*)\"", s)
        if parts:
            return [p[0] or p[1] for p in parts]
        inner = s[1:-1].strip()
        return inner.split() if inner else [""]
    return [s]


def score_one(task, pred, ref):
    fn = string_match_part if task.split("_")[0] == "qa" else string_match_all
    return fn([pred], [parse_answer(ref)]) / 100.0


def rows(pat):
    g = glob.glob(pat)
    assert len(g) == 1, (pat, g)
    df = pd.read_csv(g[0] + "/predictions.csv")
    df["predicted_answer"] = df["predicted_answer"].fillna("").astype(str)
    df["_s"] = [
        score_one(t, p, a)
        for t, p, a in zip(df["task"], df["predicted_answer"], df["answer"])
    ]
    out = {t: d["_s"].tolist() for t, d in df.groupby("task")}
    # Self-check: the per-row scores must average to what metrics.json reports, or the parse is
    # wrong again and every number below is meaningless.
    m = json.load(open(g[0] + "/metrics.json"))
    for t, sc in out.items():
        got, want = 100 * sum(sc) / len(sc), m[t]["string_match"]
        assert abs(got - want) < 0.05, f"{g[0]} {t}: rows give {got:.2f}, metrics.json says {want:.2f}"
    return out


def compare(a, x):
    n = min(len(a), len(x))
    w = sum(1 for i, j in zip(a[:n], x[:n]) if j < i - 1e-9)
    g = sum(1 for i, j in zip(a[:n], x[:n]) if j > i + 1e-9)
    return n, 100 * (sum(x[:n]) - sum(a[:n])) / n, w, g


BASE = "evaluation/results_sparse_scalar_longce_decay/ruler__%d__*topk2048*"
CMP = "evaluation/results_cmp/ruler__%d__%s"
CFG = [
    (8192, [("R64", "*cmp64-count__*"), ("R256", "*cmp256-count__*"), ("R64lrn", "*mass*")]),
    (16384, [("R64", "*cmp64-count__*"), ("R64lrn", "*mass*")]),
]

for L, arms in CFG:
    b = rows(BASE % L)
    n_total = sum(len(v) for v in b.values())
    cmps = [(nm, rows(CMP % (L, pt))) for nm, pt in arms]
    print("=" * 86)
    print("RULER %d -- %d scored rows.  per arm: delta | rows worse | rows better" % (L, n_total))
    print("=" * 86)
    print("%-18s%5s" % ("task", "n") + "".join("%10s%5s%5s" % (nm, "-", "+") for nm, _ in cmps))
    for t in [x for x in TASKS if x in b]:
        cells = [compare(b[t], c[t]) for _, c in cmps]
        print(
     "%-18s%5d" % (t, len(b[t]))
     + "".join("%+10.2f%5d%5d" % (d, w, g) for _, d, w, g in cells)
        )
    print("")
    for nm, c in cmps:
        tw = sum(compare(b[t], c[t])[2] for t in b)
        tg = sum(compare(b[t], c[t])[3] for t in b)
        print(
            "  %-8s %3d rows worse / %3d rows better / %3d of %d changed"
     % (nm, tw, tg, tw + tg, n_total)
 )
    print("")


