# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Does chunk-wise selection actually keep the needle? Measured, per layer, on real RULER items.

The motivating puzzle. On RULER the answer lives in a handful of *specific* tokens (the needle), so
the naive expectation is that chunk-wise selection should be **better** at retrieval than
token-level: 31 chunks of 64 cover 31 whole neighbourhoods, and if the needle's chunk is picked the
needle is kept. Yet measured on the HSA-LSE checkpoint at a matched *token* budget
(``topk=2048, chunk_size=64``), chunk-wise scored 65.53 against token-level's 78.25, and the loss
was concentrated on exactly the retrieval tasks (``niah_multivalue`` -35.96, ``multikey_1`` -35.19,
``multiquery`` -34.21).

That combination is suspicious enough to check directly rather than explain. Two very different
causes predict the same RULER drop:

1. **Budget waste (the benign story).** The needle IS kept, but 63 of every 64 retained slots go to
   its neighbours, so with only 31 chunk slots a multi-needle item cannot hold all its needles.
   Prediction: needle recall stays high for *single*-needle items and falls for multi-needle ones.
2. **The chunk score does not rank the needle's chunk highly (the real bug).** Pooling the router's
   token scores to a chunk *loses* the needle. Prediction: needle recall is poor even for a single
   needle, and poor at the level of individual layers.

This script separates them: for each RULER item it locates the gold answer's token span, then for
both operators reports whether those tokens survive selection -- per layer, and averaged. It also
reports where the needle's chunk *ranks* under the pooled score, which distinguishes "just missed
the cut" from "not ranked at all".

    python -m scripts.probe_needle_in_support --model /path/Qwen3-8B \\
        --ckpt .../hsa/stage1/final.pt --task niah_multivalue --items 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "evaluation") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "evaluation"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="niah_multivalue")
    ap.add_argument("--items", type=int, default=8)
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--force-sink", type=int, default=4)
    ap.add_argument("--force-local", type=int, default=64)
    ap.add_argument("--layers", default="0,8,17,26,35")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from evaluate_sparse import load_cached_dataset
    from kvpress import GQAIndexerPress
    from kvpress.presses.gqa_indexer import load_indexer_state_dict
    from kvpress.presses.gqa_indexer.chunk_support import chunk_topk_support
    from kvpress.presses.gqa_indexer.sparse_support import streaming_topk_support
    from probe_router_selection import find_needle_span

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = payload.get("config") or {}
    sd = payload.get("indexer", payload)
    press = GQAIndexerPress(
        compression_ratio=0.0,
        gate_scale=any(str(k).endswith("gate_scale") for k in sd),
        scorer_attr="indexer",
    )
    press.post_init_from_model(model, force_reinit=True)
    load_indexer_state_dict(model, sd, "indexer")
    aggregate = str(cfg.get("chunk_aggregate") or "mean")
    scale = float(cfg.get("score_scale") or 1.0)
    print(
        f"ckpt objective={cfg.get('objective')} chunk_aggregate={aggregate!r} "
        f"score_scale={scale:.6g}; probing task={args.task}"
    )

    dataset = load_cached_dataset("simonjegou/ruler", "8192")
    rows = [r for r in dataset if r["task"] == args.task][: args.items]
    if not rows:
        raise SystemExit(f"no rows for task {args.task!r}")
    probe_layers = [int(x) for x in args.layers.split(",") if int(x) < model.config.num_hidden_layers]

    results: dict[str, list] = {"token": [], "chunk": [], "rank_frac": [], "n_needle_chunks": []}
    per_layer: dict[int, dict[str, list]] = {li: {"token": [], "chunk": []} for li in probe_layers}

    for row in rows:
        answers = row["answer"]
        answers = list(answers) if not isinstance(answers, str) else [answers]
        context = row["context"]
        ids = tokenizer(context, return_tensors="pt").input_ids
        spans = []
        for ans in answers:
            span = find_needle_span(ids[0], tokenizer, str(ans))
            if span is not None:
                spans.append(span)
        if not spans:
            continue
        ids = ids.to(device)
        k_len = ids.shape[1]
        needle_tokens = set()
        for s in spans:
            needle_tokens |= set(range(s.start, min(s.stop, k_len)))
        needle_chunks = {t // args.chunk_size for t in needle_tokens}

        # Capture hidden states for the probed layers only.
        hidden: dict[int, torch.Tensor] = {}
        kwargs_by_layer: dict[int, dict] = {}
        handles = []

        def mk(li):
            def hook(module, a, k):
                h = k.get("hidden_states")
                if h is None and a:
                    h = a[0]
                hidden[li] = h
                kwargs_by_layer[li] = k
                return None
            return hook

        layers = model.model.layers
        for li in probe_layers:
            handles.append(layers[li].self_attn.register_forward_pre_hook(mk(li), with_kwargs=True))
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        for h in handles:
            h.remove()

        for li in probe_layers:
            indexer = press.get_indexer(layers[li].self_attn)
            with torch.no_grad():
                cos, sin = press.get_rope_tables(indexer, kwargs_by_layer[li])
                q_idx = indexer.project_q(hidden[li], cos, sin)
                k_idx = indexer.project_k(hidden[li], cos, sin)
                # The LAST query row is the one that answers, so its support is what matters.
                q_last = q_idx[:, :, -1:].contiguous()
                tok, _ = streaming_topk_support(
                    q_last, k_idx, args.topk, mask=None,
                    force_sink=args.force_sink, force_local=args.force_local,
                    query_offset=k_len - 1,
                )
                chk, _ = chunk_topk_support(
                    q_last, k_idx, args.topk, chunk_size=args.chunk_size,
                    chunk_aggregate=aggregate, score_scale=scale,
                    force_sink=args.force_sink, force_local=args.force_local,
                    query_offset=k_len - 1,
                )
                # Recall of the needle TOKENS, over the union across KV heads (a token is kept if
                # any head keeps it -- attention is per head, and one head suffices to carry it).
                def recall(support):
                    kept = set(support[support >= 0].reshape(-1).tolist())
                    return len(needle_tokens & kept) / len(needle_tokens)

                r_tok, r_chk = recall(tok), recall(chk)
                per_layer[li]["token"].append(r_tok)
                per_layer[li]["chunk"].append(r_chk)
                results["token"].append(r_tok)
                results["chunk"].append(r_chk)

                # Where does the needle's chunk RANK under the pooled score? Distinguishes
                # "just missed the 31-chunk cut" from "not ranked at all".
                from kvpress.presses.gqa_indexer.indexer import MASK_NEG

                s = torch.einsum("bhd,bkd->bhk", q_idx[:, :, -1], k_idx).float().mean(1)[0]
                n_chunk = k_len // args.chunk_size
                blocks = (s[: n_chunk * args.chunk_size] * scale).reshape(n_chunk, args.chunk_size)
                pooled = (
                    torch.logsumexp(blocks, -1)
                    if aggregate in ("lse", "logsumexp")
                    else blocks.mean(-1)
                )
                order = pooled.argsort(descending=True).tolist()
                ranks = [order.index(c) for c in needle_chunks if c < n_chunk]
                if ranks:
                    results["rank_frac"].append(statistics.mean(ranks) / n_chunk)
                results["n_needle_chunks"].append(len(needle_chunks))

    if not results["token"]:
        raise SystemExit("no item yielded a locatable needle span")

    budget_chunks = (args.topk - args.force_sink - args.force_local) // args.chunk_size
    print(f"\ntask={args.task}  n_items={len(rows)}  chunk budget = {budget_chunks} chunks")
    print(f"mean needle chunks per item: {statistics.mean(results['n_needle_chunks']):.1f}")
    print(f"\n{'layer':>6s}{'needle recall token':>22s}{'needle recall chunk':>22s}")
    for li in probe_layers:
        t, c = per_layer[li]["token"], per_layer[li]["chunk"]
        if t:
            print(f"{li:6d}{statistics.mean(t):22.3f}{statistics.mean(c):22.3f}")
    print(f"{'ALL':>6s}{statistics.mean(results['token']):22.3f}{statistics.mean(results['chunk']):22.3f}")
    if results["rank_frac"]:
        rf = statistics.mean(results["rank_frac"])
        print(
            f"\nneedle chunk's mean rank = top {100 * rf:.1f}% of chunks "
            f"(budget keeps the top {100 * budget_chunks / (8192 // args.chunk_size):.1f}%)"
        )
        print(
            "  -> if the rank fraction is well inside the budget fraction but recall is low, the "
            "loss is NOT ranking; if it is outside, the pooled score genuinely fails to surface "
            "the needle's chunk."
        )
    if args.out:
        Path(args.out).write_text(json.dumps({"task": args.task, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
