# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Sparse-vs-dense logit divergence, measured over many positions and several RULER tasks.

:mod:`sparse_vs_dense` produced the first signal that separates the arms -- arm C's sparse forward
tracked dense almost exactly while arm B's did not -- but it scored only the 3 gold tokens of 2
rows, i.e. 6 positions. Six positions cannot carry a conclusion, so this script re-measures the
same quantity over ``--n-pos`` positions spread across the whole sequence (hundreds), for several
tasks, and reports the spread as well as the mean.

The measurement is deliberately *not* about the answer span: every position in a long context is a
next-token prediction the sparse forward should reproduce, and using all of them turns a 6-sample
anecdote into a stable estimate. Positions are sampled from the second half of the sequence, where
the support has the whole context to choose from and where a long-context answer is produced.

Columns:

* ``kl`` -- mean ``KL(dense || sparse)`` in nats over the sampled positions.
* ``kl_p50`` / ``kl_p90`` -- median and 90th percentile, so a mean driven by a few blown-up
  positions is distinguishable from a uniformly displaced distribution.
* ``top1_agree`` -- fraction of sampled positions whose argmax matches dense.
* ``nll_gap`` -- mean increase in the NLL that the *dense* model's own argmax token receives under
  sparse. Scale-free and directly about "would it still say the same thing".
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
    ap.add_argument("--tasks", default="vt,niah_single_2,niah_single_1,qa_1")
    ap.add_argument("--data-dir", default="8192")
    ap.add_argument("--rows-per-task", type=int, default=2)
    ap.add_argument("--n-pos", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path("proxy_exp_budget/logit_divergence.json"))
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
            text = r["context"] + "\n" + r["question"] + str(r["answer_prefix"])
            prompts.append((task, tok(text, return_tensors="pt").input_ids))
    print(f"{len(prompts)} prompts over {len(tasks)} tasks", flush=True)

    sparse_kwargs = dict(
        topk=args.topk,
        force_sink=args.force_sink,
        force_local=args.force_local,
        block_k=args.block_k,
        precision=args.precision,
    )

    # Dense logits are arm-independent (the indexer never feeds back into a dense pass), but the
    # indexer must be attached to run the sparse pass, so dense is recomputed per arm anyway --
    # cheap next to the sparse pass and it guarantees the two halves see the identical input.
    results = {}
    for name, rel in ARMS.items():
        press, cfg_rec = attach_indexer(model, args.ckpt_root + rel)
        per_prompt = []
        for task, ids in prompts:
            ids = ids.to(device)
            n = ids.shape[1]
            pos = torch.linspace(n // 2, n - 2, min(args.n_pos, n // 2 - 1)).long().to(device)
            with torch.inference_mode():
                ld = model(input_ids=ids).logits[0, pos].float().log_softmax(-1)
                with SparseAttentionContext(model, press, **sparse_kwargs):
                    ls = model(input_ids=ids).logits[0, pos].float().log_softmax(-1)
                kl = (ld.exp() * (ld - ls)).sum(-1)
                dense_arg = ld.argmax(-1, keepdim=True)
                agree = (ls.argmax(-1, keepdim=True) == dense_arg).float().mean()
                nll_gap = (-ls.gather(1, dense_arg) + ld.gather(1, dense_arg)).mean()
            per_prompt.append(
                {
                    "task": task,
                    "n_pos": int(pos.numel()),
                    "kl": float(kl.mean()),
                    "kl_p50": float(kl.median()),
                    "kl_p90": float(kl.quantile(0.9)),
                    "top1_agree": float(agree),
                    "nll_gap": float(nll_gap),
                }
            )
            del ld, ls
            torch.cuda.empty_cache()

        agg = {
            k: float(np.mean([p[k] for p in per_prompt]))
            for k in ("kl", "kl_p50", "kl_p90", "top1_agree", "nll_gap")
        }
        by_task = {}
        for task in tasks:
            sel = [p for p in per_prompt if p["task"] == task]
            if sel:
                by_task[task] = {
                    "kl": float(np.mean([p["kl"] for p in sel])),
                    "top1_agree": float(np.mean([p["top1_agree"] for p in sel])),
                }
        results[name] = {
            "arm": name,
            "ckpt_config": cfg_rec,
            **agg,
            "by_task": by_task,
            "per_prompt": per_prompt,
        }
        print(
            f"arm {name}: kl={agg['kl']:.4f} (p50 {agg['kl_p50']:.4f}, p90 {agg['kl_p90']:.4f}) "
            f"agree={agg['top1_agree']:.4f} nll_gap={agg['nll_gap']:.4f}",
            flush=True,
        )

    results["_meta"] = {
        "tasks": tasks,
        "topk": args.topk,
        "n_prompts": len(prompts),
        "total_positions": sum(p["n_pos"] for p in results["A"]["per_prompt"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))

    print(
        f"\n{'arm':<5s}{'KL':>9s}{'KL p50':>9s}{'KL p90':>9s}{'top1_agree':>12s}{'nll_gap':>10s}"
    )
    print("-" * 54)
    for name in ARMS:
        r = results[name]
        print(
            f"{name:<5s}{r['kl']:>9.4f}{r['kl_p50']:>9.4f}{r['kl_p90']:>9.4f}"
            f"{r['top1_agree']:>12.4f}{r['nll_gap']:>10.4f}"
        )
    print(f"\nper task (KL / top1_agree), {results['_meta']['total_positions']} positions per arm:")
    print(f"{'task':<18s}" + "".join(f"{a:>18s}" for a in ARMS))
    for task in tasks:
        cells = ""
        for a in ARMS:
            bt = results[a]["by_task"].get(task)
            cells += "".join(f"{'--':>18s}") if bt is None else f"{bt['kl']:>10.3f}/{bt['top1_agree']:<7.3f}"
        print(f"{task:<18s}{cells}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
