# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Which selection operator should a checkpoint be evaluated with -- token top-k, or whole chunks?

This decides a real number. The exact-K arm was evaluated both ways on 8K RULER:

    token-level, topk 2048 : 68.29
    chunk-wise,  topk 2048 : 45.05
    token-level, topk 512  : 32.61
    chunk-wise,  topk 512  : 17.23   <- the FULLY train-consistent config, and the worst of the four

so guessing wrong is worth up to 23 points, and -- counterintuitively -- matching the training
granularity was the *wrong* choice there. Getting this right for a new checkpoint should therefore
be a measurement, not an inference from how the router was trained.

What predicts it
----------------
The fraction of the router's score variance that lies **within** a chunk rather than **between**
chunks. Measured previously on the two trained checkpoints:

======  =====================  ===================
layer   exact-K within/across  gated within/across
======  =====================  ===================
0       **0.17**               0.70
4       **0.16**               0.99
7       0.69                   0.74
======  =====================  ===================

exact-K learned an almost piecewise-constant score, because chunk-mean scores over query blocks were
the only thing its loss ever saw -- so within-chunk structure stayed near initialization and a
token-level top-k spends its decisions where the score carries no information. A **low** ratio is the
signature of a router whose token-level ordering is noise.

Why the prediction is *not* obvious for the HSA arm, and why this script exists
------------------------------------------------------------------------------
HSA also trains only chunk-level quantities, so the naive expectation is the exact-K pattern. But its
operator differs in a way that matters: the within-chunk distribution is the **frozen backbone's own**
``softmax(q k^T)``, never the router's. The router supplies only ``w_c``. So the token ordering inside
a kept chunk comes from the backbone -- which is exactly the quantity token-level top-k wants -- and
the router's own within-chunk structure may be irrelevant rather than merely uninformative.

Two readings, and they imply opposite eval configurations:

* if the *indexer's* score is piecewise-constant (low ratio) but selection should follow the
  *backbone's* q.k, then chunk-wise selection of chunks + backbone ordering within them is the
  faithful operator;
* if the indexer's score retains within-chunk structure (high ratio), token top-k is fine and is also
  what the gated arm uses, keeping the arms comparable.

This script measures the ratio and also reports the quantity that decides it more directly:
**oracle attention-mass recall** of each operator at a matched token budget, with random and recency
controls. The controls are not optional -- a previous recall measurement here looked publishable until
baselines showed random scoring 0.85, because at ``topk ~ Sq/2`` the oracle top-k *is* essentially the
visible set. Rows are therefore restricted to ``visible >= 8 * topk``.

    python -m scripts.probe_eval_operator --model /path/Qwen3-8B \\
        --ckpt .../hsa/stage1/final.pt --chunk-size 64 --topk 256
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


def real_tokens(tokenizer, seq_len: int, device: str) -> torch.Tensor:
    """Real text; random ids make every operator look equally irrelevant."""
    docs = [
        REPO_ROOT / "kvpress/presses/gqa_indexer/ROUTER_LEARNABILITY.md",
        REPO_ROOT / "differentiable_topk_for_sparse_attention.md",
        REPO_ROOT / "kvpress/presses/gqa_indexer/README.md",
        REPO_ROOT / "README.md",
    ]
    text = "\n\n".join(p.read_text() for p in docs if p.exists())
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :seq_len]
    if ids.shape[1] < seq_len:
        ids = ids.repeat(1, -(-seq_len // ids.shape[1]))[:, :seq_len]
    return ids.to(device)


def variance_ratio(scores: torch.Tensor, chunk_size: int) -> float:
    """
    Fraction of score variance lying *within* a chunk rather than between chunks.

    ``scores`` is ``(..., n_key)`` for one query row. Low means piecewise-constant: the router
    distinguishes chunks but not tokens inside them, so a token-level top-k is deciding on noise.
    """
    n = scores.shape[-1]
    n_chunk = n // chunk_size
    if n_chunk < 2:
        return float("nan")
    trimmed = scores[..., : n_chunk * chunk_size]
    blocks = trimmed.reshape(*trimmed.shape[:-1], n_chunk, chunk_size)
    within = blocks.var(dim=-1, unbiased=False).mean()
    across = blocks.mean(dim=-1).var(dim=-1, unbiased=False).mean()
    total = within + across
    return float(within / total.clamp_min(1e-12))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--layers", default="0,4,7")
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument(
        "--topk", type=int, default=256,
        help="TOKEN budget for the recall comparison. Keep it well below the visible count -- at "
        "topk ~ Sq/2 the oracle top-k is essentially the visible set and random scores 0.85.",
    )
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--truncate", type=int, default=0,
        help="keep only the first N layers. CAVEAT: a truncated probe already inverted one "
        "conclusion in this investigation (it said no router beats recency; the full eval said "
        "+49). Preliminary signal only.",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kvpress import GQAIndexerPress
    from kvpress.presses.gqa_indexer import load_indexer_state_dict

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    if args.truncate:
        model.model.layers = torch.nn.ModuleList(list(model.model.layers[: args.truncate]))
        model.config.num_hidden_layers = args.truncate
        print(f"--truncate {args.truncate}: preliminary only -- see --truncate in --help.")
    model = model.to(device).eval()

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = payload.get("config") or {}
    ckpt_chunk = int(cfg.get("chunk_size") or 0)
    if ckpt_chunk and ckpt_chunk != args.chunk_size:
        print(
            f"NOTE: checkpoint recorded chunk_size={ckpt_chunk} but --chunk-size={args.chunk_size}. "
            f"Using {args.chunk_size} for the ratio; the checkpoint's value is what eval should use."
        )
    print(f"checkpoint step {payload.get('step')} objective={cfg.get('objective')}")

    press = GQAIndexerPress(compression_ratio=0.5, gate_scale=True)
    press.post_init_from_model(model)
    sd = payload.get("indexer", payload)
    if args.truncate:
        keep = {}
        for name, tensor in sd.items():
            parts = name.split(".")
            layer = next((int(p) for p, q in zip(parts[1:], parts) if q == "layers"), None)
            if layer is None or layer < args.truncate:
                keep[name] = tensor
        sd = keep
    load_indexer_state_dict(model, sd, "indexer")

    probe = [int(x) for x in args.layers.split(",") if int(x) < model.config.num_hidden_layers]
    ids = real_tokens(tokenizer, args.seq_len, device)

    hidden: dict[int, torch.Tensor] = {}
    kw: dict[int, dict] = {}
    handles = []

    def make_hook(li):
        def hook(module, a, k):
            hs = k.get("hidden_states")
            if hs is None and a:
                hs = a[0]
            hidden[li] = hs
            kw[li] = k
            return None
        return hook

    for li in probe:
        handles.append(
            model.model.layers[li].self_attn.register_forward_pre_hook(make_hook(li), with_kwargs=True)
        )
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()

    results = {}
    cs = args.chunk_size
    for li in probe:
        attn = model.model.layers[li].self_attn
        indexer = press.get_indexer(attn)
        hs = hidden[li]
        b, q_len, _ = hs.shape
        n_heads = model.config.num_attention_heads
        n_kv = model.config.num_key_value_heads
        head_dim = getattr(model.config, "head_dim", None) or model.config.hidden_size // n_heads
        group = n_heads // n_kv

        with torch.no_grad():
            cos, sin = press.get_rope_tables(indexer, kw[li])
            q_idx = indexer.project_q(hs, cos, sin)          # (B, h, Sq, D)
            k_idx = indexer.project_k(hs, cos, sin)          # (B, Sq, D)

            # The ORACLE: the frozen backbone's own attention mass per key.
            q = attn.q_proj(hs).view(b, q_len, n_heads, head_dim).transpose(1, 2)
            k = attn.k_proj(hs).view(b, q_len, n_kv, head_dim).transpose(1, 2)
            if hasattr(attn, "q_norm"):
                q = attn.q_norm(q)
            if hasattr(attn, "k_norm"):
                k = attn.k_norm(k)
            pos = torch.arange(q_len, device=device).unsqueeze(0)
            rcos, rsin = model.model.rotary_emb(k, pos)
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

            q, k = apply_rotary_pos_emb(q, k, rcos, rsin)

            # Rows with plenty of visible keys, so recall is not degenerate.
            lo = 8 * args.topk
            if q_len <= lo:
                raise SystemExit(
                    f"--seq-len {q_len} is too short for --topk {args.topk}: recall needs rows with "
                    f"at least 8*topk = {lo} visible keys, or the oracle top-k IS the visible set "
                    f"(random then scores ~0.85 and the metric means nothing)."
                )
            rows = torch.linspace(lo, q_len - 1, args.rows, device=device).long().unique()

            ratios, rec = [], {"router_token": [], "router_chunk": [], "recency": [], "random": []}
            for ri in rows.tolist():
                vis = ri + 1
                # Router score for this query row, per KV head; average over heads to get the
                # per-key score the selector effectively ranks on at this granularity.
                s = torch.einsum("bhd,bkd->bhk", q_idx[:, :, ri], k_idx[:, :vis]).float().mean(1)[0]
                ratios.append(variance_ratio(s, cs))

                # Oracle: attention mass, averaged over the query heads of every group.
                a_ = torch.einsum(
                    "bhd,bhkd->bhk", q[:, :, ri], k[:, :, :vis].repeat_interleave(group, dim=1)
                ).float() * head_dim**-0.5
                mass = torch.softmax(a_, -1).mean(1)[0]
                oracle = mass.topk(args.topk).indices
                target = set(oracle.tolist())

                def recall(idx):
                    return len(target & set(idx.tolist())) / len(target)

                # token-level: top-k keys by router score
                rec["router_token"].append(recall(s.topk(args.topk).indices))
                # chunk-wise: top (topk // cs) chunks by mean score, expanded to tokens
                n_chunk = vis // cs
                if n_chunk >= 1:
                    cm = s[: n_chunk * cs].reshape(n_chunk, cs).mean(-1)
                    take = max(1, args.topk // cs)
                    picked = cm.topk(min(take, n_chunk)).indices
                    toks = (picked.unsqueeze(-1) * cs + torch.arange(cs, device=device)).flatten()
                    rec["router_chunk"].append(recall(toks))
                rec["recency"].append(recall(torch.arange(vis - args.topk, vis, device=device)))
                rec["random"].append(recall(torch.randperm(vis, device=device)[: args.topk]))

        row = {
            "within_across_ratio": statistics.mean([r for r in ratios if r == r]),
            "n_rows": len(rows),
            "recall": {k2: statistics.mean(v2) for k2, v2 in rec.items() if v2},
        }
        results[li] = row
        r = row["recall"]
        print(
            f"layer {li:2d}: within/across variance ratio {row['within_across_ratio']:.3f}\n"
            f"          oracle-mass recall@{args.topk}: token {r['router_token']:.3f}  "
            f"chunk {r.get('router_chunk', float('nan')):.3f}  "
            f"recency {r['recency']:.3f}  random {r['random']:.3f}"
        )

    if results:
        ratio = statistics.mean(v["within_across_ratio"] for v in results.values())
        tok = statistics.mean(v["recall"]["router_token"] for v in results.values())
        chk = statistics.mean(
            v["recall"]["router_chunk"] for v in results.values() if "router_chunk" in v["recall"]
        )
        rnd = statistics.mean(v["recall"]["random"] for v in results.values())
        rcy = statistics.mean(v["recall"]["recency"] for v in results.values())
        print("\n=== verdict ===")
        print(f"mean within/across ratio : {ratio:.3f}   (exact-K measured 0.16-0.17, gated 0.70-0.99)")
        print(f"mean recall  token-level : {tok:.3f}")
        print(f"mean recall  chunk-wise  : {chk:.3f}")
        print(f"controls     recency     : {rcy:.3f}      random : {rnd:.3f}")
        better = "chunk-wise" if chk > tok else "token-level"
        print(
            f"\n-> evaluate with {better} selection ({max(tok, chk):.3f} vs {min(tok, chk):.3f}).\n"
            f"   Sanity: if BOTH are at or below the controls, the recall metric is not resolving "
            f"anything here and the RULER number is the only thing to trust."
        )
    if args.out:
        Path(args.out).write_text(json.dumps({"ckpt": args.ckpt, "layers": results}, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
