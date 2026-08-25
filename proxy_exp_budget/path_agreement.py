# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Do the gather path and the new flex path agree? A confound check, not a benchmark.

``sparse_inference.py`` grew a query-independent ``flex_attention`` selection path
(``qi_sparse_attention``) **while the topk sweep in §14.1 was running**: the file's mtime (20:45)
falls between the ``topk=2048`` run (19:35-19:45) and the ``topk=1024`` / ``topk=4096`` runs
(20:35-20:58, 20:59-21:17). ``_use_qi`` is chosen automatically from
``ScalarIndexer.is_query_independent``, so the later two runs took the flex path and the earlier one
took the gather path. That makes the sweep's three points **not necessarily one experiment**, and the
+43.7-point recovery attributed to ``topk`` could be partly a kernel change.

This script settles it by running both paths over the identical model, checkpoint and input and
comparing their *logits*. ``SparseAttentionContext`` accepts ``query_independent=`` to force the
choice, so the only thing that differs is the kernel.

If the two agree to bf16 rounding, the sweep is one experiment and §14.1 stands as written. If they
do not, every number in the sweep has to be re-taken on one path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EVAL_DIR = REPO_ROOT / "evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from kvpress import SparseAttentionContext  # noqa: E402

from sparse_vs_dense import attach_indexer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B")
    ap.add_argument(
        "--ckpt",
        default="/apdcephfs_gy8/share_303843174/guhao/models/"
        "Qwen-3-8B-gqa_indexer_cross_replay/stage1_256/final.pt",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--topks", default="1024,2048,4096")
    ap.add_argument("--force-sink", type=int, default=4)
    ap.add_argument("--force-local", type=int, default=64)
    ap.add_argument("--block-k", type=int, default=64)
    ap.add_argument("--precision", default="tf32")
    ap.add_argument("--task", default="niah_single_1")
    ap.add_argument("--data-dir", default="8192")
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
    press, _ = attach_indexer(model, args.ckpt)

    df = load_dataset("simonjegou/ruler", data_dir=args.data_dir, split="test").to_pandas()
    row = df[df["task"] == args.task].iloc[0]
    text = row["context"] + "\n" + row["question"] + str(row["answer_prefix"])
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    print(f"task={args.task} tokens={ids.shape[1]}", flush=True)

    print(f"\n{'topk':>6}{'max|dlogit|':>13}{'mean|dlogit|':>14}{'top1_agree':>12}{'KL':>10}")
    print("-" * 55)
    for topk in [int(t) for t in args.topks.split(",")]:
        outs = {}
        for label, qi in (("gather", False), ("flex", True)):
            with torch.inference_mode():
                with SparseAttentionContext(
                    model,
                    press,
                    topk=topk,
                    force_sink=args.force_sink,
                    force_local=args.force_local,
                    block_k=args.block_k,
                    precision=args.precision,
                    query_independent=qi,
                ):
                    outs[label] = model(input_ids=ids).logits[0, -256:].float()
            torch.cuda.empty_cache()
        g, f = outs["gather"], outs["flex"]
        lg, lf = g.log_softmax(-1), f.log_softmax(-1)
        kl = float((lg.exp() * (lg - lf)).sum(-1).mean())
        print(
            f"{topk:>6}{float((g - f).abs().max()):>13.5f}{float((g - f).abs().mean()):>14.6f}"
            f"{float((g.argmax(-1) == f.argmax(-1)).float().mean()):>12.4f}{kl:>10.6f}"
        )


if __name__ == "__main__":
    main()
