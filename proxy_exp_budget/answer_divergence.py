# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Sparse-vs-dense divergence **at the answer position**, over many RULER rows.

This is the measurement that discriminates the arms, and the reason the earlier ones did not:

* :mod:`attn_recall` and :mod:`layer_divergence` measure averaged attention mass. All three arms
  look alike (recall ~0.72-0.79) because most of a long context's attention mass sits on sinks and
  recent tokens, which ``force_sink``/``force_local`` pin unconditionally. A needle is a tiny slice
  of mass, so losing it moves the average by ~1% and the RULER score by 50 points.
* :mod:`logit_divergence` samples mid-context positions. Those are ordinary next-token predictions
  that the local window already answers, so *every* arm agrees with dense there (B's ``vt``
  agreement is 1.000, KL 0.007). It measures language modelling, not retrieval.

The answer position is the only place the two differ: it is the one token whose correct prediction
*requires* a specific distant key to be in the support. So this script scores exactly there, over
``--rows-per-task`` rows for each task, and reports:

* ``gold_nll_sparse`` / ``gold_nll_dense`` -- NLL of the reference answer's first token. The gap is
  the retrieval damage, in nats.
* ``top1_agree`` -- does sparse still predict dense's argmax at the answer position.
* ``gold_rank_sparse`` -- rank of the gold token under sparse. A rank that falls out of the top few
  is a wrong answer regardless of how small the KL looks.

``force_sink``/``force_local``/``topk``/``block_k``/``precision`` all match ``evaluate_sparse.py``,
so a number here is comparable to the RULER score it is meant to explain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EVAL_DIR = REPO_ROOT / "evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from kvpress import SparseAttentionContext  # noqa: E402

from dissect_scores import ARMS  # noqa: E402
from sparse_vs_dense import attach_indexer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B")
    ap.add_argument("--ckpt-root", default="/apdcephfs_gy8/share_303843174/guhao/models/")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--force-sink", type=int, default=4)
    ap.add_argument("--force-local", type=int, default=64)
    ap.add_argument("--block-k", type=int, default=64)
    ap.add_argument("--precision", default="tf32")
    ap.add_argument(
        "--tasks",
        default="vt,niah_single_1,niah_single_2,niah_single_3,niah_multiquery,niah_multivalue,qa_1",
    )
    ap.add_argument("--data-dir", default="8192")
    ap.add_argument("--rows-per-task", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("proxy_exp_budget/answer_divergence.json"))
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

    df = load_dataset("simonjegou/ruler", data_dir=args.data_dir, split="test").to_pandas()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    prompts = []
    for task in tasks:
        for _, r in df[df["task"] == task].head(args.rows_per_task).iterrows():
            gold = r["answer"]
            gold = gold[0] if hasattr(gold, "__len__") and not isinstance(gold, str) else gold
            head = r["context"] + "\n" + r["question"] + str(r["answer_prefix"])
            head_ids = tok(head, return_tensors="pt").input_ids
            # First gold token only: it is the one the answer hinges on, and scoring it needs no
            # assumption about how the model formats the rest of its answer.
            gold_id = tok(str(gold), add_special_tokens=False).input_ids[0]
            prompts.append((task, head_ids, gold_id))
    print(f"{len(prompts)} rows over {len(tasks)} tasks", flush=True)

    sparse_kwargs = dict(
        topk=args.topk,
        force_sink=args.force_sink,
        force_local=args.force_local,
        block_k=args.block_k,
        precision=args.precision,
    )

    results = {}
    for name, rel in ARMS.items():
        press, cfg_rec = attach_indexer(model, args.ckpt_root + rel)
        rows = []
        for task, head_ids, gold_id in prompts:
            ids = head_ids.to(device)
            with torch.inference_mode():
                ld = model(input_ids=ids).logits[0, -1].float().log_softmax(-1)
                with SparseAttentionContext(model, press, **sparse_kwargs):
                    ls = model(input_ids=ids).logits[0, -1].float().log_softmax(-1)
            rows.append(
                {
                    "task": task,
                    "gold_nll_dense": float(-ld[gold_id]),
                    "gold_nll_sparse": float(-ls[gold_id]),
                    "gold_rank_dense": int((ld > ld[gold_id]).sum()),
                    "gold_rank_sparse": int((ls > ls[gold_id]).sum()),
                    "top1_agree": float(ls.argmax() == ld.argmax()),
                    "kl": float((ld.exp() * (ld - ls)).sum()),
                }
            )
            torch.cuda.empty_cache()

        def m(key):
            return float(np.mean([r[key] for r in rows]))

        by_task = {}
        for task in tasks:
            sel = [r for r in rows if r["task"] == task]
            if sel:
                by_task[task] = {
                    "gold_nll_sparse": float(np.mean([r["gold_nll_sparse"] for r in sel])),
                    "gold_nll_dense": float(np.mean([r["gold_nll_dense"] for r in sel])),
                    "top1_agree": float(np.mean([r["top1_agree"] for r in sel])),
                    "gold_top1_frac": float(np.mean([r["gold_rank_sparse"] == 0 for r in sel])),
                }
        results[name] = {
            "arm": name,
            "ckpt_config": cfg_rec,
            "gold_nll_sparse": m("gold_nll_sparse"),
            "gold_nll_dense": m("gold_nll_dense"),
            "nll_gap": m("gold_nll_sparse") - m("gold_nll_dense"),
            "top1_agree": m("top1_agree"),
            "kl": m("kl"),
            "gold_top1_frac_sparse": float(np.mean([r["gold_rank_sparse"] == 0 for r in rows])),
            "gold_top1_frac_dense": float(np.mean([r["gold_rank_dense"] == 0 for r in rows])),
            "by_task": by_task,
            "rows": rows,
        }
        r = results[name]
        print(
            f"arm {name}: gold_nll {r['gold_nll_sparse']:.3f} (dense {r['gold_nll_dense']:.3f}, "
            f"gap {r['nll_gap']:+.3f}) agree={r['top1_agree']:.3f} "
            f"gold_is_top1={r['gold_top1_frac_sparse']:.3f} kl={r['kl']:.3f}",
            flush=True,
        )

    results["_meta"] = {"tasks": tasks, "topk": args.topk, "n_rows": len(prompts)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))

    print(
        f"\nAt the answer position, {len(prompts)} rows:\n"
        f"{'arm':<5s}{'gold_nll_sp':>12s}{'gold_nll_de':>12s}{'gap':>8s}"
        f"{'agree':>8s}{'gold_top1':>11s}{'KL':>9s}"
    )
    print("-" * 65)
    for name in ARMS:
        r = results[name]
        print(
            f"{name:<5s}{r['gold_nll_sparse']:>12.3f}{r['gold_nll_dense']:>12.3f}"
            f"{r['nll_gap']:>+8.3f}{r['top1_agree']:>8.3f}"
            f"{r['gold_top1_frac_sparse']:>11.3f}{r['kl']:>9.3f}"
        )
    print(f"(dense gold_is_top1 = {results['A']['gold_top1_frac_dense']:.3f})")

    print(f"\ngold NLL under sparse, per task:\n{'task':<20s}" + "".join(f"{a:>10s}" for a in ARMS) + f"{'dense':>10s}")
    for task in tasks:
        cells = "".join(f"{results[a]['by_task'][task]['gold_nll_sparse']:>10.3f}" for a in ARMS)
        print(f"{task:<20s}{cells}{results['A']['by_task'][task]['gold_nll_dense']:>10.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
