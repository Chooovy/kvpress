# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Fit the static per-head budget table, once, offline.

    python -m scripts.fit_head_budget_table --ckpt .../final.pt \\
        --topk 2048 --force-local 128 --force-sink 4 --docs 16 \\
        --out .../head_budget_2048.pt

Why a static table is worth fitting at all
------------------------------------------
The per-document allocation wins +6.55 RULER 8K, and a budget-shuffle control that permutes the
same budgets across heads scores *below* the uniform baseline -- so the gain is specifically the
head<->budget correspondence. The question this script answers is how much of that correspondence
is a property of the **model** rather than of the input.

Measured (``scratch/diag_skeleton.py``): the per-head demand ranking is as stable across
documents (Spearman 0.59) as it is across reference rows within a single document (0.59). If a
constant table recovers most of the gain, the method loses its only awkward parts -- the
prefill-time measurement and the question of what happens to a budget fitted at prefill over a
long decode.

Fitted on the TRAINING corpus, not on RULER: a table fitted on the evaluation distribution would
be tuned to the benchmark, and the claim "these are the model's streaming heads" would not be
testable. The corpus default therefore matches the router's own training data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kvpress.presses.gqa_indexer.head_budget import fit_static_table  # noqa: E402
from kvpress.presses.gqa_indexer.press import GQAIndexerPress  # noqa: E402
from kvpress.presses.gqa_indexer.train import (  # noqa: E402
    load_indexer_state_dict,
    press_kwargs_from_checkpoint,
)

MODELS = "/apdcephfs_gy8/share_303843174/guhao/models"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=f"{MODELS}/Qwen3-8B")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenized", default=f"{MODELS}/../datasets/longmino_tokenized_64k")
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--force-sink", type=int, default=4)
    p.add_argument("--force-local", type=int, default=128)
    p.add_argument("--floor", type=int, default=512,
                   help="minimum evictable budget per head; 512 was the RULER optimum")
    p.add_argument("--docs", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(args.device).eval()
    )
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck.get("indexer", ck)
    scorer, kwargs = press_kwargs_from_checkpoint(state, ck.get("config") or {})
    press = GQAIndexerPress(
        compression_ratio=0.0,
        gate_scale=any(str(x).endswith("gate_scale") for x in state),
        scorer_attr="indexer", scorer=scorer, **kwargs,
    )
    press.post_init_from_model(model, force_reinit=True)
    load_indexer_state_dict(model, state, "indexer")

    from kvpress.presses.gqa_indexer.data import TokenizedConfig, build_tokenized_dataloader

    loader = build_tokenized_dataloader(
        TokenizedConfig(root=args.tokenized, seq_len=args.seq_len, take_from="head"),
        batch_size=1, num_workers=0,
    )
    docs = []
    for batch in loader:
        docs.append(batch["input_ids"][:, : args.seq_len].to(args.device))
        if len(docs) >= args.docs:
            break
    print(f"fitting on {len(docs)} document(s) at L={args.seq_len}", flush=True)

    table = fit_static_table(
        model, press, docs,
        topk=args.topk, force_sink=args.force_sink, force_local=args.force_local,
        floor=args.floor,
    )

    total = args.topk * table.shape[1]
    bad = [(i, int(r.sum())) for i, r in enumerate(table) if int(r.sum()) != total]
    print(f"table {tuple(table.shape)}; rows not summing to {total}: {bad if bad else 'none'}")
    for layer_idx in range(0, table.shape[0], 4):
        row = table[layer_idx]
        print(f"  layer {layer_idx:2d}: {row.tolist()}  min {int(row.min())} max {int(row.max())}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "table": table,
            "topk": args.topk,
            "force_sink": args.force_sink,
            "force_local": args.force_local,
            "floor": args.floor,
            "seq_len": args.seq_len,
            "docs": len(docs),
            "ckpt": args.ckpt,
        },
        out,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
