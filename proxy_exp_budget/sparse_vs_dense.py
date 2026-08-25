# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end divergence of the *real* sparse forward pass from the dense one, per arm.

Why this measurement is needed. :mod:`dissect_scores` shows arm B's score is content-driven
(cross-document top-k Jaccard 0.149 against a 0.143 chance floor -- the same as arms A and C), and
:mod:`attn_recall` shows its selected support carries essentially the same attention mass as the
other two arms (0.717 vs A 0.744 / C 0.740 on ``niah_multikey_2``; 0.719 vs A 0.758 / C 0.787 on
``vt``, where B scores **0.00** on RULER and A scores 49.76). A 4-point recall gap cannot produce a
50-point task gap, so either the failure is not in selection at all, or it is a compounding effect
that a per-layer average of one layer's recall cannot see.

This script settles that by running the **actual** :class:`SparseAttentionContext` -- the same class
``evaluate_sparse.py`` uses, with the same ``topk``/``force_sink``/``force_local``/``block_k``/
``precision`` -- over a real RULER row, and comparing its logits against the dense forward on the
identical input. If B's sparse forward tracks dense about as well as A's and C's do, the collapse
is not produced by anything this pipeline does to attention, and the checkpoint/eval plumbing is
where to look next. If B diverges sharply, selection error compounds across the 36 layers in a way
single-layer recall understates -- and the recall measurement, not the hypothesis, was the wrong
instrument.

Reported per arm, on the answer-region tokens only (where RULER's metric is decided):

* ``top1_agree`` -- fraction of positions where sparse and dense argmax agree. The most direct
  proxy for "does it generate the same text".
* ``kl`` -- mean KL(dense || sparse) in nats.
* ``gold_nll_sparse`` / ``gold_nll_dense`` -- NLL of the reference answer's tokens. A confabulating
  arm should show a large gap here even when its top-1 agreement looks tolerable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EVAL_DIR = REPO_ROOT / "evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from kvpress import GQAIndexerPress, SparseAttentionContext, load_indexer_state_dict  # noqa: E402
from kvpress.presses.gqa_indexer.train import detect_scorer, infer_scalar_mid_dim  # noqa: E402

from dissect_scores import ARMS  # noqa: E402


def attach_indexer(model, ckpt_path: str):
    """Build the press and load the checkpoint, mirroring ``evaluate_sparse.py::_setup_pipeline``.

    Deliberately the same code path rather than a re-implementation: the scorer, ``mid_dim`` and
    ``pos_slope`` all come from the checkpoint there, and ``pos_slope`` in particular is not a
    parameter -- a re-implementation that defaulted it would mis-score with every weight loading
    cleanly, which is the failure this whole investigation is trying not to repeat.

    ``force_reinit=True`` because this script attaches several arms in turn and they do not share a
    geometry (A is ``mid_dim=0``, B and C are 256). Without it ``post_init_from_model`` rightly
    refuses the second arm -- the guard exists so a mismatched geometry cannot score silently.
    Re-initialising discards the previous arm's weights, which is exactly what is wanted here since
    ``load_indexer_state_dict`` immediately overwrites them from this arm's checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("indexer", ckpt)
    cfg_rec = ckpt.get("config") or {}
    has_gate = any(str(k).endswith("gate_scale") for k in sd)
    scorer = detect_scorer(sd, cfg_rec)

    kwargs = {}
    if scorer == "scalar":
        kwargs["scalar_mid_dim"] = infer_scalar_mid_dim(sd, cfg_rec)
        if "scalar_pos_slope" in cfg_rec:
            kwargs["scalar_pos_slope"] = float(cfg_rec["scalar_pos_slope"])
    press = GQAIndexerPress(
        compression_ratio=0.0,
        gate_scale=has_gate,
        scorer_attr="indexer",
        scorer=scorer,
        **kwargs,
    )
    press.post_init_from_model(model, force_reinit=True)
    load_indexer_state_dict(model, sd, "indexer")
    return press, cfg_rec


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
    ap.add_argument("--task", default="vt")
    ap.add_argument("--data-dir", default="8192")
    ap.add_argument("--n-rows", type=int, default=2, help="RULER rows to average over")
    ap.add_argument("--out", type=Path, default=Path("proxy_exp_budget/sparse_vs_dense.json"))
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
    rows = df[df["task"] == args.task].head(args.n_rows)
    if rows.empty:
        raise SystemExit(f"no rows for task {args.task}")

    # Build each row's full prompt the way the pipeline does: context, then question and the task's
    # answer_prefix, then the reference answer. Scoring only the answer region is the point --
    # that is the span RULER's metric reads.
    prompts = []
    for _, r in rows.iterrows():
        gold = r["answer"]
        gold = gold[0] if hasattr(gold, "__len__") and not isinstance(gold, str) else gold
        head = r["context"] + "\n" + r["question"] + str(r["answer_prefix"])
        head_ids = tok(head, return_tensors="pt").input_ids
        gold_ids = tok(str(gold), add_special_tokens=False, return_tensors="pt").input_ids
        prompts.append((head_ids, gold_ids, str(gold)))
        print(f"row: ctx+q={head_ids.shape[1]} tok, gold={str(gold)[:40]!r} ({gold_ids.shape[1]} tok)")

    results = {}
    for name, rel in ARMS.items():
        press, cfg_rec = attach_indexer(model, args.ckpt_root + rel)
        agree, kls, nll_s, nll_d = [], [], [], []
        for head_ids, gold_ids, _ in prompts:
            ids = torch.cat([head_ids, gold_ids], dim=1).to(device)
            n_gold = gold_ids.shape[1]

            with torch.inference_mode():
                dense = model(input_ids=ids).logits[0, -n_gold - 1 : -1].float()
                with SparseAttentionContext(
                    model,
                    press,
                    topk=args.topk,
                    force_sink=args.force_sink,
                    force_local=args.force_local,
                    block_k=args.block_k,
                    precision=args.precision,
                ):
                    sparse = model(input_ids=ids).logits[0, -n_gold - 1 : -1].float()

            ld = dense.log_softmax(-1)
            ls = sparse.log_softmax(-1)
            agree.append((sparse.argmax(-1) == dense.argmax(-1)).float().mean().item())
            kls.append((ld.exp() * (ld - ls)).sum(-1).mean().item())
            tgt = gold_ids[0].to(device)
            nll_d.append(-ld.gather(1, tgt.view(-1, 1)).mean().item())
            nll_s.append(-ls.gather(1, tgt.view(-1, 1)).mean().item())

        results[name] = {
            "arm": name,
            "ckpt_config": cfg_rec,
            "top1_agree": sum(agree) / len(agree),
            "kl": sum(kls) / len(kls),
            "gold_nll_sparse": sum(nll_s) / len(nll_s),
            "gold_nll_dense": sum(nll_d) / len(nll_d),
        }
        r = results[name]
        print(
            f"arm {name}: top1_agree={r['top1_agree']:.4f} kl={r['kl']:.4f} "
            f"gold_nll sparse={r['gold_nll_sparse']:.4f} dense={r['gold_nll_dense']:.4f}",
            flush=True,
        )

    results["_meta"] = {"task": args.task, "topk": args.topk, "n_rows": int(len(prompts))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))

    print(f"\n{'arm':<6s}{'top1_agree':>12s}{'KL':>10s}{'gold_nll_sp':>13s}{'gold_nll_de':>13s}")
    print("-" * 54)
    for name in ARMS:
        r = results[name]
        print(
            f"{name:<6s}{r['top1_agree']:>12.4f}{r['kl']:>10.4f}"
            f"{r['gold_nll_sparse']:>13.4f}{r['gold_nll_dense']:>13.4f}"
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
