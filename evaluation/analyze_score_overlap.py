#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Analyze per-layer score dumps produced by ScorerPress.dump_layer_scores().

Input: a directory containing files named like:
  layer{layer_idx}_step{step}_pos{cache_pos}.pt

Each .pt is expected to contain a dict with:
  - layer_idx: int
  - cache_pos: Optional[int]
  - scores: Tensor[batch, num_kv_heads, seq_len]

Goal: quantify whether different layers focus on the same subset of tokens.
We do this by computing top-k token index overlaps across layers, per (step, pos).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch


FILE_RE = re.compile(r"layer(?P<layer>\d+)_step(?P<step>\d+)_pos(?P<pos>na|\d+)\.pt$")


@dataclass(frozen=True)
class Key:
    pos: Optional[int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dump_dir", required=True, help="Directory with layer*_step*_pos*.pt files")
    p.add_argument("--topk", type=int, default=128, help="Top-k tokens per head to compare (default: 128)")
    p.add_argument(
        "--topk_ratio",
        type=float,
        default=None,
        help="If set, use topk = int(seq_len * topk_ratio) instead of --topk",
    )
    p.add_argument(
        "--compare",
        choices=("prev", "layer0", "both"),
        default="both",
        help="Compare each layer to previous layer, layer0, or both (default: both)",
    )
    p.add_argument(
        "--max_groups",
        type=int,
        default=200,
        help="Max pos-groups to process (default: 200). Use to limit runtime.",
    )
    p.add_argument(
        "--pos",
        type=int,
        default=None,
        help="If set, analyze only this cache_pos (pos in filename).",
    )
    p.add_argument(
        "--batch_index",
        type=int,
        default=0,
        help="Which batch item to use from scores tensor (default: 0)",
    )
    p.add_argument(
        "--out_json",
        default=None,
        help="Optional path to write a JSON summary",
    )
    return p.parse_args()


def iter_files(dump_dir: str) -> Iterable[Tuple[str, int, int, Optional[int]]]:
    for name in os.listdir(dump_dir):
        m = FILE_RE.match(name)
        if not m:
            continue
        layer = int(m.group("layer"))
        step = int(m.group("step"))
        pos_s = m.group("pos")
        pos = None if pos_s == "na" else int(pos_s)
        yield os.path.join(dump_dir, name), layer, step, pos


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    # a,b: 1D integer tensors
    if a.numel() == 0 and b.numel() == 0:
        return 1.0
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    inter = torch.isin(a, b).sum().item()
    union = a.numel() + b.numel() - inter
    return float(inter) / float(max(1, union))


def topk_indices_per_head(scores_b: torch.Tensor, topk: int) -> torch.Tensor:
    """
    scores_b: Tensor[num_heads, seq_len]
    return: LongTensor[num_heads, topk]
    """
    k = max(1, min(int(topk), int(scores_b.shape[-1])))
    return torch.topk(scores_b, k=k, dim=-1, largest=True, sorted=False).indices


def main() -> None:
    args = parse_args()
    dump_dir = args.dump_dir
    assert os.path.isdir(dump_dir), f"Not a directory: {dump_dir}"
    assert args.topk > 0, "--topk must be > 0"
    if args.topk_ratio is not None:
        assert 0.0 < args.topk_ratio <= 1.0, "--topk_ratio must be in (0, 1]"

    # IMPORTANT: `step` in filenames is a global counter and is NOT aligned across layers.
    # To compare layers at the "same moment", we group by cache_pos and pick, for each layer,
    # the file with the largest step at that cache_pos.
    groups: Dict[Key, Dict[int, Tuple[int, str]]] = defaultdict(dict)  # Key(pos) -> layer -> (step, filepath)
    for path, layer, step, pos in iter_files(dump_dir):
        if args.pos is not None and pos != args.pos:
            continue
        k = Key(pos=pos)
        prev = groups[k].get(layer, None)
        if prev is None or step > prev[0]:
            groups[k][layer] = (step, path)

    # Sort groups by pos
    ordered_keys = sorted(groups.keys(), key=lambda k: -1 if k.pos is None else k.pos)
    if args.max_groups is not None and args.max_groups >= 0:
        ordered_keys = ordered_keys[: args.max_groups]

    # Accumulators
    per_layer_overlap_prev: Dict[int, List[float]] = defaultdict(list)
    per_layer_overlap_layer0: Dict[int, List[float]] = defaultdict(list)
    per_layer_token_freq: Dict[int, Counter] = defaultdict(Counter)  # layer -> token_idx -> count

    processed_groups = 0
    skipped_groups = 0

    for key in ordered_keys:
        layer_to_path = {layer: sp[1] for layer, sp in groups[key].items()}
        if not layer_to_path:
            continue

        layers = sorted(layer_to_path.keys())
        # Need at least 2 layers to talk about overlap
        if len(layers) < 2:
            skipped_groups += 1
            continue

        # Load all layers for this group (scores only)
        layer_to_topk: Dict[int, torch.Tensor] = {}
        layer_to_seq_len: Dict[int, int] = {}

        for layer in layers:
            obj = torch.load(layer_to_path[layer], map_location="cpu")
            scores = obj["scores"]  # (B, H, T)
            if args.batch_index >= scores.shape[0]:
                raise ValueError(f"batch_index={args.batch_index} out of range for {layer_to_path[layer]}")
            scores_b = scores[args.batch_index]  # (H, T)
            seq_len = int(scores_b.shape[-1])
            layer_to_seq_len[layer] = seq_len
            topk = int(seq_len * args.topk_ratio) if args.topk_ratio is not None else int(args.topk)
            idx = topk_indices_per_head(scores_b, topk=topk)  # (H, K)
            layer_to_topk[layer] = idx

            # Token frequency per layer (aggregate across heads)
            flat = idx.reshape(-1).tolist()
            per_layer_token_freq[layer].update(flat)

        # Compare overlaps
        layer0 = layers[0]
        topk0 = layer_to_topk[layer0]

        for i, layer in enumerate(layers):
            if i == 0:
                continue
            cur = layer_to_topk[layer]

            # Per-head overlap, averaged across heads
            H = min(cur.shape[0], topk0.shape[0])
            head_overlaps_layer0 = []
            for h in range(H):
                head_overlaps_layer0.append(jaccard(cur[h], topk0[h]))
            if args.compare in ("layer0", "both"):
                per_layer_overlap_layer0[layer].append(float(sum(head_overlaps_layer0) / max(1, len(head_overlaps_layer0))))

            if args.compare in ("prev", "both"):
                prev_layer = layers[i - 1]
                prev = layer_to_topk[prev_layer]
                H2 = min(cur.shape[0], prev.shape[0])
                head_overlaps_prev = []
                for h in range(H2):
                    head_overlaps_prev.append(jaccard(cur[h], prev[h]))
                per_layer_overlap_prev[layer].append(float(sum(head_overlaps_prev) / max(1, len(head_overlaps_prev))))

        processed_groups += 1

    def _mean(xs: List[float]) -> Optional[float]:
        if not xs:
            return None
        return float(sum(xs) / len(xs))

    all_layers = sorted(set(per_layer_overlap_prev.keys()) | set(per_layer_overlap_layer0.keys()) | set(per_layer_token_freq.keys()))
    summary = {
        "dump_dir": dump_dir,
        "topk": args.topk,
        "topk_ratio": args.topk_ratio,
        "compare": args.compare,
        "processed_groups": processed_groups,
        "skipped_groups": skipped_groups,
        "layers": {},
    }

    for layer in all_layers:
        layer_entry = {
            "overlap_prev_mean": _mean(per_layer_overlap_prev.get(layer, [])),
            "overlap_layer0_mean": _mean(per_layer_overlap_layer0.get(layer, [])),
        }
        # Show the most frequent tokens in this layer's topk across groups (aggregated over heads)
        most_common = per_layer_token_freq[layer].most_common(20)
        # layer_entry["top_tokens_most_common"] = [{"token": int(t), "count": int(c)} for (t, c) in most_common]
        summary["layers"][str(layer)] = layer_entry

    # Print a compact table
    print(json.dumps({k: summary[k] for k in ("dump_dir", "topk", "topk_ratio", "compare", "processed_groups")}, indent=2))
    print("\nPer-layer overlap means:")
    header = ["layer", "overlap_prev_mean", "overlap_layer0_mean"]
    print("\t".join(header))
    for layer in all_layers:
        e = summary["layers"][str(layer)]
        op = "NA" if e["overlap_prev_mean"] is None else f"{e['overlap_prev_mean']:.4f}"
        o0 = "NA" if e["overlap_layer0_mean"] is None else f"{e['overlap_layer0_mean']:.4f}"
        print(f"{layer}\t{op}\t{o0}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nWrote JSON summary to: {args.out_json}")


if __name__ == "__main__":
    main()

