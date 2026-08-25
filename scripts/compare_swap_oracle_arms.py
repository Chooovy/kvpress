# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Swap oracle for BOTH arms on one shared oracle: does each arm's score gradient predict swap utility?

This is the experiment that decides whether "change the gradient estimator" is worth pursuing. The
exact-K arm measures Spearman +0.31 against its own swap oracle, and loses 18.8 RULER points to the
gated arm. Two readings are possible and they imply opposite next steps:

* if gated's correlation is **much higher**, estimator quality does track downstream quality, and the
  continuous-relaxation family (LapSum / Sander / SparseMixer, doc §7 and §12-13) is worth building;
* if it is **comparable**, the estimator is not what separates the arms, and the candidate pool /
  budget-efficiency findings are where the work belongs.

Making it a controlled comparison
---------------------------------
The two arms produce different objects -- exact-K a chunk subset, gated an additive per-token gate --
so a naive comparison would confound the estimator with the operator. Two things are held fixed:

1. **One shared oracle.** ``ΔL = L(S − i + j) − L(S)`` is measured by forcing a chunk subset through
   :func:`~.exact_k_attention.exact_k_chunk_attention` with ``hard=True``, for both arms. The oracle
   is a property of the *model and the subset*, not of how the score was trained, so the same operator
   must define it on both sides.

2. **A probe variable, so neither forward is modified.** For each arm a zero bias ``b_c`` is added to
   the chunk score, and the gradient measured is ``dL/db_c`` under that arm's **own** training
   forward. At ``b = 0`` the forward is bit-identical to training, so this measures the arm as it
   actually trains rather than a re-parameterized variant. Same trick the exact-K DP uses to extract
   marginals, and the same one doc §31 uses for per-key utility.

The gated arm's caveat, stated plainly
--------------------------------------
Gated's native gate is per-``(query, key)`` token. To obtain a *chunk*-level gradient comparable to
exact-K's, its score is pooled to chunks and broadcast back, i.e. the gate is made piecewise-constant
within a chunk. That is a restriction of its objective, not the objective it trained under. It is the
closest controlled reading available -- the alternative, comparing a token-level gradient against a
chunk-level oracle, would not be a comparison at all.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kvpress import GQAIndexerPress  # noqa: E402
from kvpress.presses.gqa_indexer import (  # noqa: E402
    ExactKIndexerTrainer,
    exact_k_indexer_training_step,
    load_indexer_state_dict,
    swap_oracle_correlation,
)
from kvpress.presses.gqa_indexer.exact_k_attention import (  # noqa: E402
    build_candidates,
    chunk_visibility,
    exact_k_chunk_attention,
    gather_candidate_gradient,
    gather_candidate_scores,
)
from scripts.exact_k_swap_oracle import real_tokens  # noqa: E402

ARMS = {
    "exactK": "/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_exact_k/stage1/final.pt",
    "gated": "/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_e2e/stage1/final.pt",
}


def pooled_fisher_z(rs: list[float], n: int) -> tuple[float, float, float]:
    """Pooled correlation and 95% CI over layers. Per-layer values are rarely resolvable alone."""
    rs = [r for r in rs if r == r and abs(r) < 1.0]
    if not rs:
        return float("nan"), float("nan"), float("nan")
    zs = [0.5 * math.log((1 + r) / (1 - r)) for r in rs]
    zbar = sum(zs) / len(zs)
    se = 1.0 / math.sqrt(max(len(rs) * (n - 3), 1))
    return math.tanh(zbar), math.tanh(zbar - 1.96 * se), math.tanh(zbar + 1.96 * se)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", default="/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B")
    ap.add_argument("--truncate", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--pairs", type=int, default=64)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--query-block", type=int, default=256)
    ap.add_argument("--topk-chunk", type=int, default=8)
    ap.add_argument("--n-candidate", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    results: dict[str, dict] = {}

    for arm, ckpt_path in ARMS.items():
        print(f"\n{'=' * 78}\n{arm}: {ckpt_path}\n{'=' * 78}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        nl = args.truncate or model.config.num_hidden_layers
        model.model.layers = torch.nn.ModuleList(list(model.model.layers[:nl]))
        model.config.num_hidden_layers = nl
        model = model.to(device).eval()

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = payload.get("config") or {}
        sd = {
            k: v
            for k, v in payload["indexer"].items()
            if int([x for x, y in zip(k.split(".")[1:], k.split(".")) if y == "layers"][0]) < nl
        }
        press = GQAIndexerPress(compression_ratio=0.5, gate_scale=True, chunk_size=args.chunk_size)
        press.post_init_from_model(model)
        load_indexer_state_dict(model, sd, "indexer")
        print(f"objective={cfg.get('objective')} step={payload.get('step')}")

        # ONE trainer type for both arms: the shared oracle has to be the same operator, and the
        # probe measures dL/d(chunk score) under it. What differs between the arms is only the
        # *weights* -- which is exactly the variable under test.
        trainer = ExactKIndexerTrainer(
            press=press,
            chunk_size=args.chunk_size,
            query_block=args.query_block,
            topk_chunk=args.topk_chunk,
            n_candidate=args.n_candidate,
            explore_frac=0.0,  # deterministic pool, so the oracle's replays see the same candidates
        )
        trainer.freeze_backbone(model)
        ids = real_tokens(tokenizer, args.seq_len, device)

        probe_layers = [
            round(i * (nl - 1) / max(args.layers - 1, 1)) for i in range(args.layers)
        ]
        captured: dict[int, dict] = {}
        forced: dict[int, torch.Tensor] = {}
        current = {"idx": -1}

        import kvpress.presses.gqa_indexer.exact_k_trainer as tmod

        original = exact_k_chunk_attention

        def patched(q, k, v, candidate_scores, candidates, **kw):
            li = current["idx"]
            if li in forced:
                kw = {**kw, "hard": True}
                candidate_scores = candidate_scores + 1e4 * forced[li]
            out, stats = original(q, k, v, candidate_scores, candidates, **kw)
            if li not in captured:
                captured[li] = {"candidates": candidates, "selected": stats["selected"]}
            return out, stats

        tmod.exact_k_chunk_attention = patched
        real_routed = trainer.routed_forward

        def routed(module, *a, **kw):
            current["idx"] = int(module.layer_idx)
            return real_routed(module, *a, **kw)

        trainer.routed_forward = routed

        # The probe: a zero bias on each layer's chunk score, retained so dL/db is available. At b=0
        # the forward is bit-identical to training, so this measures the arm as it trains.
        scores_leaf: dict[int, torch.Tensor] = {}
        real_chunk_scores = trainer.chunk_scores

        def chunk_scores_probed(module, hs, kwargs, k_len):
            out = real_chunk_scores(module, hs, kwargs, k_len)
            li = int(module.layer_idx)
            if li in probe_layers:
                out = out + torch.zeros_like(out, requires_grad=True)
                out.retain_grad()
                scores_leaf[li] = out
            return out

        trainer.chunk_scores = chunk_scores_probed

        try:
            loss = exact_k_indexer_training_step(model, trainer, input_ids=ids)
            baseline = float(loss)
            loss.backward()
            trainer.chunk_scores = real_chunk_scores
            print(f"baseline LM loss {baseline:.6f} (unforced)")

            layer_results = {}
            for li in probe_layers:
                info = captured.get(li)
                leaf = scores_leaf.get(li)
                if info is None or leaf is None or leaf.grad is None:
                    print(f"  layer {li}: no gradient captured, skipping")
                    continue
                cand = info["candidates"]
                # NOT gather_candidate_scores: that fills pads with PAD_SCORE=-31, a score sentinel
                # that is meaningless as a gradient. See gather_candidate_gradient.
                score_grad = gather_candidate_gradient(leaf.grad, cand)
                if float(score_grad.abs().mean()) == 0.0:
                    print(f"  layer {li}: gradient identically zero, skipping")
                    continue

                def loss_fn(mask, _li=li):
                    forced.clear()
                    for other, other_info in captured.items():
                        forced[other] = other_info["selected"]
                    forced[_li] = mask
                    with torch.no_grad():
                        return float(
                            exact_k_indexer_training_step(model, trainer, input_ids=ids)
                        )

                # The forced replay must reproduce the unforced loss, or every ΔL is measured
                # against the wrong reference. This guard caught two real bugs before.
                forced_base = loss_fn(info["selected"])
                if abs(forced_base - baseline) > 0.05:
                    print(
                        f"  layer {li}: forcing moved the loss {baseline:.4f} -> {forced_base:.4f}; "
                        f"the replay does not reproduce the sampled subset. Skipping."
                    )
                    forced.clear()
                    continue

                res = swap_oracle_correlation(
                    loss_fn,
                    info["selected"],
                    score_grad,
                    max_pairs=args.pairs,
                    generator=torch.Generator().manual_seed(li),
                )
                print(f"  layer {li:2d}: {res}")
                layer_results[li] = dict(res.__dict__)
                forced.clear()
        finally:
            tmod.exact_k_chunk_attention = original
            trainer.routed_forward = real_routed

        results[arm] = layer_results
        del model
        torch.cuda.empty_cache()

    print(f"\n{'=' * 78}\nPOOLED COMPARISON (Fisher-z over layers)\n{'=' * 78}")
    print(f"{'arm':10} {'spearman':>20} {'pearson':>20} {'centered sign':>14}")
    summary = {}
    for arm, layers in results.items():
        if not layers:
            print(f"{arm:10} {'no data':>20}")
            continue
        n = next(iter(layers.values()))["n_pairs"]
        sp, lo, hi = pooled_fisher_z([v["spearman"] for v in layers.values()], n)
        pe, plo, phi = pooled_fisher_z([v["pearson"] for v in layers.values()], n)
        cs = [v["centered_sign_accuracy"] for v in layers.values()]
        cs = sum(c for c in cs if c == c) / max(len([c for c in cs if c == c]), 1)
        print(
            f"{arm:10} {f'{sp:+.3f} [{lo:+.3f},{hi:+.3f}]':>20} "
            f"{f'{pe:+.3f} [{plo:+.3f},{phi:+.3f}]':>20} {cs:14.3f}"
        )
        summary[arm] = {"spearman": sp, "ci": [lo, hi], "pearson": pe, "centered_sign": cs}

    if "exactK" in summary and "gated" in summary:
        a, b = summary["exactK"], summary["gated"]
        overlap = not (a["ci"][1] < b["ci"][0] or b["ci"][1] < a["ci"][0])
        print(f"\nexact-K {a['spearman']:+.3f} vs gated {b['spearman']:+.3f}")
        print(f"95% CIs {'OVERLAP' if overlap else 'are DISJOINT'}")
        print(
            "\n=> "
            + (
                "estimator quality is NOT what separates the arms; the gap lives elsewhere "
                "(candidate pool / budget efficiency). Changing the estimator is unlikely to help."
                if overlap
                else "the arms' estimators differ measurably. If gated is higher, estimator "
                "quality does track downstream quality and the relaxation family is worth building."
            )
        )

    if args.out:
        Path(args.out).write_text(json.dumps({"layers": results, "pooled": summary}, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
