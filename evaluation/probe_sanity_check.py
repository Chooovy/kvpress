# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Sanity-check the router probe against the backbone's own dense attention.

**Why this exists.** ``probe_router_selection.py`` reported needle coverage of 0.048 for the
distilled checkpoint and 0.072 for the e2e one, against a random baseline of
``topk / k_len = 2048 / 21025 = 0.097``. Both are **at or below chance**, and the needle's median
rank was ~12000 of ~21000 -- the middle of the ordering. That cannot be right for the distilled
router, which scores **95.65** on this exact task at this exact ``topk``: a support set that
contains the needle only at chance rate cannot answer 46 of 46 questions.

So before any conclusion is drawn from that probe, this script establishes the ground truth the
probe should be consistent with:

1. **Dense attention** -- what the frozen backbone itself does with the needle. Distillation
   trains the router to reproduce this, so it bounds what any router can be expected to show. If
   the needle is *not* prominent here either, then the premise "the needle must rank highly" is
   wrong and the probe's numbers are fine; the fault is in the interpretation.
2. **Each router's ranking** of the needle, and the coverage of the real
   ``streaming_topk_support`` selection, computed on the same layer and the same rows.

The per-head maximum is reported alongside the mean, because retrieval only needs **one** head to
carry the needle -- a mean over 8 KV heads hides a single head that ranks it first, and the probe
only reported means. That alone could explain the discrepancy.

Usage::

    python -m evaluation.probe_sanity_check --model /path/Qwen3-8B \\
        --distill-ckpt ... --e2e-ckpt ... --layers 0 10 20 30
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.probe_router_selection import (  # noqa: E402
    attach_indexer,
    build_model,
    capture_hidden_states,
    find_needle_span,
)
from kvpress.presses.gqa_indexer.sparse_support import streaming_topk_support  # noqa: E402

logger = logging.getLogger("probe_sanity_check")


def language_model(model):
    inner = model.model
    return inner.language_model if hasattr(inner, "language_model") else inner


@torch.no_grad()
def dense_attention_stats(model, layer, hidden, rope, rows, span, n_keys):
    """
    The backbone's own attention on the needle: total weight, and rank per head.

    Recomputes q/k for one layer from the captured hidden states rather than hooking the attention
    output, because what is needed is the *weight distribution over keys*, which the attention
    implementation never materializes. This is the quantity distillation supervises against, so it
    is the ceiling for any router trained to imitate it.
    """
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    config = model.config
    n_heads, n_kv = config.num_attention_heads, config.num_key_value_heads
    head_dim = config.head_dim
    cos, sin = rope

    normed = layer.input_layernorm(hidden)
    query = layer.self_attn.q_proj(normed).view(1, n_keys, n_heads, head_dim).transpose(1, 2)
    key = layer.self_attn.k_proj(normed).view(1, n_keys, n_kv, head_dim).transpose(1, 2)
    query, key = layer.self_attn.q_norm(query), layer.self_attn.k_norm(key)
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    query = query[:, :, rows, :].float()
    key = key.repeat_interleave(n_heads // n_kv, dim=1).float()
    weights = ((query @ key.transpose(-1, -2)) / head_dim**0.5).softmax(-1)

    needle_weight = weights[..., span.start : span.stop].sum(-1)      # (1, H, rows)
    order = weights.argsort(-1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(-1, order, torch.arange(n_keys, device=weights.device).expand_as(order))
    needle_rank = rank[..., span.start : span.stop].float()

    return {
        "needle_weight_mean": float(needle_weight.mean()),
        "needle_weight_max_head": float(needle_weight.mean(-1).max()),
        "rank_median": float(needle_rank.median()),
        "rank_median_best_head": float(needle_rank.flatten(2).median(-1).values.min()),
        "uniform_weight": span.n_tokens / n_keys,
    }


@torch.no_grad()
def router_stats(press, layer, hidden, rope, rows, span, n_keys, topk, force_sink, force_local):
    """The router's ranking of the needle, and the coverage of the real top-k selection."""
    indexer = press.get_indexer(layer.self_attn)
    cos, sin = press.get_rope_tables(indexer, {"position_embeddings": rope})
    q_idx = indexer.project_q(hidden, cos, sin)
    k_idx = indexer.project_k(hidden, cos, sin)

    scores = torch.einsum("bhqd,bkd->bhqk", q_idx[:, :, rows, :].float(), k_idx.float())
    order = scores.argsort(-1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(-1, order, torch.arange(n_keys, device=scores.device).expand_as(order))
    needle_rank = rank[..., span.start : span.stop].float()

    support, _ = streaming_topk_support(
        q_idx, k_idx, topk, force_sink=force_sink, force_local=force_local
    )
    support = support[:, :, rows, :]
    n_kv = support.shape[1]
    selected = torch.zeros((1, n_kv, len(rows), n_keys + 1), dtype=torch.bool, device=scores.device)
    selected.scatter_(-1, support.clamp(min=0) + (support < 0).long() * n_keys, True)
    selected = selected[..., :n_keys]
    coverage = selected[..., span.start : span.stop].float().mean(-1)   # (1, h, rows)

    return {
        "rank_median": float(needle_rank.median()),
        "rank_median_best_head": float(needle_rank.flatten(2).median(-1).values.min()),
        "coverage_mean": float(coverage.mean()),
        "coverage_best_head": float(coverage.mean(-1).max()),
        "coverage_any_head_full": float((coverage.max(1).values >= 1.0).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--distill-ckpt", required=True)
    parser.add_argument("--e2e-ckpt", required=True)
    parser.add_argument("--data-dir", default="16384")
    parser.add_argument("--task", default="niah_multikey_3")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 10, 20, 30])
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--force-sink", type=int, default=4)
    parser.add_argument("--force-local", type=int, default=64)
    parser.add_argument("--query-rows", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")

    from datasets import load_dataset

    frame = load_dataset("simonjegou/ruler", data_dir=args.data_dir, split="test").to_pandas()
    row = frame[frame.task == args.task].iloc[0]
    answer = str(row["answer"][0])
    prompt = row["context"] + row["question"] + row["answer_prefix"]

    model, tokenizer = build_model(args.model, torch.bfloat16, args.device)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
    span = find_needle_span(input_ids[0], tokenizer, answer)
    if span is None:
        raise SystemExit("needle not located")
    n_keys = input_ids.shape[1]
    rows = list(range(n_keys - args.query_rows, n_keys))

    print(f"prompt {n_keys} tok; needle [{span.start},{span.stop}) = {span.n_tokens} tok; "
          f"answer {answer}")
    print(f"topk={args.topk} -> random coverage {args.topk / n_keys:.4f}, "
          f"random median rank {n_keys // 2}")

    hidden_states, rope = capture_hidden_states(model, input_ids)
    layers = language_model(model).layers

    # NOT pre-attached in a dict. `attach_indexer` writes the weights into the *model's* modules,
    # and a press only holds references to them -- so attaching both up front leaves the second
    # checkpoint's weights in place and both presses reading them. That produced bit-identical
    # numbers for the two checkpoints, which is what caught it.
    checkpoints = {"distill": args.distill_ckpt, "e2e": args.e2e_ckpt}

    print(f"\n{'layer':>6} {'source':<9} {'cover':>8} {'cover*':>8} {'rank med':>10} {'rank*':>10} {'weight':>9}")
    for index in args.layers:
        hidden = hidden_states[index].to(args.device)
        dense = dense_attention_stats(model, layers[index], hidden, rope, rows, span, n_keys)
        print(f"{index:>6} {'dense':<9} {'-':>8} {'-':>8} "
              f"{dense['rank_median']:10.0f} {dense['rank_median_best_head']:10.0f} "
              f"{dense['needle_weight_max_head']:9.5f}")
        for name, checkpoint in checkpoints.items():
            # Re-attached per use, so each checkpoint's weights are the ones in the model when it
            # is measured. Cheap next to the forward pass that produced the hidden states.
            press = attach_indexer(model, checkpoint)[0]
            stats = router_stats(
                press, layers[index], hidden, rope, rows, span, n_keys,
                args.topk, args.force_sink, args.force_local,
            )
            print(f"{index:>6} {name:<9} {stats['coverage_mean']:8.4f} "
                  f"{stats['coverage_best_head']:8.4f} {stats['rank_median']:10.0f} "
                  f"{stats['rank_median_best_head']:10.0f} {'-':>9}")
        del hidden
        torch.cuda.empty_cache()

    print("\ncover* / rank* are the BEST KV head, not the mean: retrieval needs one head to")
    print("carry the needle, so a mean over heads can hide a head that ranks it first.")
    print(f"dense 'weight' is that layer's best head's total attention on the needle;")
    print(f"uniform would be {dense['uniform_weight']:.6f}.")


if __name__ == "__main__":
    main()
