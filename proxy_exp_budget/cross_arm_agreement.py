# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Do the cross-replay arms agree with **each other** more than with the e2e arm?

The hypothesis (``cross_replay_e2e.md`` §18): the rectangle's degenerate attractor is not a flat gate
and not a position-only gate -- both are ruled out by §14.2 -- but a **query-agnostic content-salience**
score, ``s_i = g(h_i)`` picking "globally interesting" tokens. Salience is the unique content-based
function that ``N`` heterogeneous rectangle rows can all agree on without compromise, so it is the path
of least resistance for an objective that forces one score to serve every row.

It is invisible to every diagnostic currently in the repo, which is why it needs its own test:

* content-driven, so cross-document top-k Jaccard sits at the chance floor (§14.2 measured 0.149) ✓
* not positional, so ``corr(score, position) ~ 0`` ✓
* carries a real ranking, so ``shuffle_delta`` is large (+3.4..+6.0) ✓
* concentrated, so participation is low -- which *looks* like eviction working ✓

The discriminating prediction: if the cross-replay arms have converged to salience while the e2e arm
tracks query-relevance, then the cross-replay arms' selected sets should agree **with each other** much
more than with the e2e arm's:

    J(cross, cross) >> J(cross, e2e)

Every arm shares the same frozen backbone and is scored on identical hidden states, so nothing but the
learned score differs. The null is that all pairwise overlaps are equal, which kills the hypothesis.

Reported against two references: the chance floor ``topk / (2N - topk)`` for independent sets, and the
within-arm cross-document floor from §14.2 (~0.149 at ``topk=2048, N=8192``).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dissect_scores import build_indexers, capture_hidden_states  # noqa: E402

#: Arm label -> checkpoint, relative to ``--ckpt-root``. Extends :data:`dissect_scores.ARMS` with the
#: two later arms, so the cross-arm comparison covers every trained cross-replay configuration.
ALL_ARMS = {
    "A_cross_mid0": "Qwen-3-8B-gqa_indexer_cross_replay/stage1/final.pt",
    "B_cross_mid256_B2048": "Qwen-3-8B-gqa_indexer_cross_replay/stage1_256/final.pt",
    "D_cross_mid256_B1": "Qwen-3-8B-gqa_indexer_cross_replay/stage1_256_B1/final.pt",
    # The lookahead=0 arm: cross-replay OBJECTIVE but the e2e loss's causal supervision shape. §18.2
    # predicts this is the one intervention that removes the query-agnostic pressure, so its selected
    # set should move OFF the cross-replay cluster and toward the e2e arm's.
    "E_cross_la0": "Qwen-3-8B-gqa_indexer_cross_replay/stage1_256_B1_la0/final.pt",
    # Cross-DOCUMENT replay: same rectangle, but C' is an UNRELATED document, so the reconstruction
    # relation is removed while everything else is held fixed. §19's diagnostic. If this still lands
    # in the rectangle cluster, the selection does not depend on the replay text at all.
    "F_cross_xdoc": "Qwen-3-8B-gqa_indexer_cross_replay/stage1_256_B1_xdoc/final.pt",
    "C_e2e_mid256": "Qwen-3-8B-gqa_indexer_scalar/stage1/final.pt",
}
#: The unbounded-rectangle arms. `E_cross_la0` is deliberately EXCLUDED: it shares the objective but
#: not the rectangle, and whether it clusters with these is exactly the question.
CROSS_REPLAY = ("A_cross_mid0", "B_cross_mid256_B2048", "D_cross_mid256_B1")
RECTANGLE_TEST = "E_cross_la0"
XDOC_TEST = "F_cross_xdoc"
E2E = "C_e2e_mid256"


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / len(sa | sb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B")
    ap.add_argument("--ckpt-root", default="/apdcephfs_gy8/share_303843174/guhao/models/")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-tokens", type=int, default=7500)
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--n-docs", type=int, default=3)
    ap.add_argument("--data-dir", default="8192")
    ap.add_argument("--out", type=Path, default=Path("proxy_exp_budget/cross_arm_agreement.json"))
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    model = model.to(device).eval()
    n_layers = model.config.num_hidden_layers
    n_kv = model.config.num_key_value_heads
    hidden = model.config.hidden_size

    available = {k: v for k, v in ALL_ARMS.items() if Path(args.ckpt_root + v).exists()}
    missing = sorted(set(ALL_ARMS) - set(available))
    if missing:
        print(f"[skip] checkpoints not present: {missing}", flush=True)
    print(f"arms: {sorted(available)}", flush=True)

    df = load_dataset("simonjegou/ruler", data_dir=args.data_dir, split="test").to_pandas()
    docs, seen = [], set()
    for _, row in df.iterrows():
        if row["task"] in seen:
            continue
        ids = tok(row["context"], return_tensors="pt", add_special_tokens=False).input_ids
        if ids.shape[1] < args.n_tokens:
            continue
        seen.add(row["task"])
        docs.append((row["task"], ids[:, : args.n_tokens].to(device)))
        if len(docs) >= args.n_docs:
            break
    print(f"documents: {[t for t, _ in docs]}", flush=True)

    # Hidden states are arm-independent (a dense pass; the indexer never feeds back), so capture once
    # per document and score with every arm. This is what makes the comparison clean.
    hs_docs = [capture_hidden_states(model, ids, n_layers) for _, ids in docs]
    del model
    torch.cuda.empty_cache()

    # sets[arm][doc][layer] -> (n_kv, topk) selected indices
    sets: dict[str, list[list[torch.Tensor]]] = {}
    for arm, rel in available.items():
        indexers, _, _ = build_indexers(args.ckpt_root + rel, hidden, n_kv, device)
        per_doc = []
        for hs in hs_docs:
            per_layer = []
            with torch.inference_mode():
                for li in sorted(indexers):
                    scores = indexers[li].score_keys(hs[li].to(device))[0].float()
                    per_layer.append(scores.topk(args.topk, dim=-1).indices.cpu())
            per_doc.append(per_layer)
        sets[arm] = per_doc
        del indexers
        torch.cuda.empty_cache()
        print(f"  scored {arm}", flush=True)

    arms = sorted(sets)
    pair_j: dict[tuple[str, str], float] = {}
    for x, y in itertools.combinations(arms, 2):
        vals = [
            jaccard(sets[x][d][li][h], sets[y][d][li][h])
            for d in range(len(hs_docs))
            for li in range(n_layers)
            for h in range(n_kv)
        ]
        pair_j[(x, y)] = float(np.mean(vals))

    chance = args.topk / (2 * args.n_tokens - args.topk)
    # Restricted to the three unbounded-rectangle arms and their pairs with e2e. The lookahead=0 arm
    # is scored separately below: including it here would contaminate both means with the very thing
    # under test.
    rect = set(CROSS_REPLAY)
    cross_cross = [v for (x, y), v in pair_j.items() if x in rect and y in rect]
    cross_e2e = [
        v for (x, y), v in pair_j.items() if {x, y} <= rect | {E2E} and (x == E2E) != (y == E2E)
    ]
    # Where does the lookahead=0 arm sit -- with the rectangle cluster, or with e2e?
    la_vs_rect = [
        v for (x, y), v in pair_j.items()
        if RECTANGLE_TEST in (x, y) and (x in rect or y in rect)
    ]
    la_vs_e2e = [
        v for (x, y), v in pair_j.items() if {x, y} == {RECTANGLE_TEST, E2E}
    ]
    # Same treatment for the cross-document arm: is it with the rectangle cluster or with e2e?
    xd_vs_rect = [
        v for (x, y), v in pair_j.items()
        if XDOC_TEST in (x, y) and (x in rect or y in rect)
    ]
    xd_vs_e2e = [v for (x, y), v in pair_j.items() if {x, y} == {XDOC_TEST, E2E}]

    result = {
        "topk": args.topk,
        "n_tokens": args.n_tokens,
        "chance": chance,
        "documents": [t for t, _ in docs],
        "pairwise": {f"{x}|{y}": v for (x, y), v in pair_j.items()},
        "mean_cross_vs_cross": float(np.mean(cross_cross)) if cross_cross else float("nan"),
        "mean_cross_vs_e2e": float(np.mean(cross_e2e)) if cross_e2e else float("nan"),
        "mean_la0_vs_rectangle": float(np.mean(la_vs_rect)) if la_vs_rect else float("nan"),
        "mean_la0_vs_e2e": float(np.mean(la_vs_e2e)) if la_vs_e2e else float("nan"),
        "mean_xdoc_vs_rectangle": float(np.mean(xd_vs_rect)) if xd_vs_rect else float("nan"),
        "mean_xdoc_vs_e2e": float(np.mean(xd_vs_e2e)) if xd_vs_e2e else float("nan"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))

    print(f"\npairwise top-k Jaccard (same document, same layer, same head; {args.topk=}):")
    for (x, y), v in sorted(pair_j.items(), key=lambda kv: -kv[1]):
        tag = "cross-cross" if (x in CROSS_REPLAY and y in CROSS_REPLAY) else "cross-e2e"
        print(f"  {x:<24} {y:<24} {v:.4f}  [{tag}]")
    print(f"\nmean cross-replay vs cross-replay : {result['mean_cross_vs_cross']:.4f}")
    print(f"mean cross-replay vs e2e          : {result['mean_cross_vs_e2e']:.4f}")
    print(f"chance floor (independent sets)    : {chance:.4f}")
    print(
        "\nPrediction if the rectangle drives a shared salience solution: "
        "cross-cross >> cross-e2e.\nNull: the two are equal."
    )

    if not np.isnan(result["mean_xdoc_vs_rectangle"]):
        print(f"\n--- {XDOC_TEST}: does removing the RECONSTRUCTION relation move the selection? ---")
        print(f"  vs the rectangle arms : {result['mean_xdoc_vs_rectangle']:.4f}")
        print(f"  vs the e2e arm        : {result['mean_xdoc_vs_e2e']:.4f}")
        print(
            "  If it stays with the rectangle arms, the score does not depend on the replay text,\n"
            "  i.e. reconstruction was never what the router learned from (§19)."
        )

    if not np.isnan(result["mean_la0_vs_rectangle"]):
        rect_j = result["mean_la0_vs_rectangle"]
        e2e_j = result["mean_la0_vs_e2e"]
        print(f"\n--- {RECTANGLE_TEST}: does breaking the rectangle move the selection? ---")
        print(f"  vs the rectangle arms : {rect_j:.4f}")
        print(f"  vs the e2e arm        : {e2e_j:.4f}")
        print(f"  (rectangle cluster internal agreement: {result['mean_cross_vs_cross']:.4f})")
        # Above-chance ratios, since raw Jaccards all sit on top of the same floor.
        print(
            f"  above-chance: vs rectangle {rect_j - chance:+.4f}, vs e2e {e2e_j - chance:+.4f}"
        )
        print(
            "  §18.2 predicts lookahead=0 removes the query-agnostic pressure, so this arm should\n"
            "  sit BELOW the rectangle cluster's internal agreement and ABOVE the cross-e2e level.\n"
            "  If it still clusters with the rectangle arms, the supervision shape was not the cause."
        )


if __name__ == "__main__":
    main()
