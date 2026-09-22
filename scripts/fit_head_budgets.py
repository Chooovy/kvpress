# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from itertools import islice
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kvpress.indexmem.ablations.budget import fit_offline_head_budgets
from kvpress.indexmem.checkpoint import load_scorer_checkpoint
from kvpress.indexmem.training.data import document_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--scorer-checkpoint", required=True)
    parser.add_argument("--tokenized", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-budget", type=int, default=2048)
    parser.add_argument("--sink-size", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--min-head-budget", type=int, default=512)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--documents", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(args.device)
        .eval()
    )
    load_scorer_checkpoint(model, args.scorer_checkpoint)
    loader = document_loader(
        args.tokenized,
        args.sequence_length,
        ("2e15", "2e16", "synth_cwe", "synth_rex"),
        batch_size=1,
        workers=0,
        seed=args.seed,
    )
    documents = [sample["input_ids"].to(args.device) for sample in islice(loader, args.documents)]
    table = fit_offline_head_budgets(
        model,
        documents,
        cache_budget=args.cache_budget,
        sink_size=args.sink_size,
        window_size=args.window_size,
        min_head_budget=args.min_head_budget,
    )
    torch.save({"table": table, "config": vars(args)}, args.output)


if __name__ == "__main__":
    main()
