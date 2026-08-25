# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Where does the needle rank, per checkpoint, at token level versus chunk level?

The question this settles. On RULER the answer lives in a few specific tokens, so a router is only
useful if those tokens survive selection. Two things were unclear:

1. Do the gated / distilled routers rank the needle well, or is the weak ranking measured on the HSA
   checkpoint (needle at the top 30-40% of tokens, and NOT its chunk's max token) a property of that
   objective or of the whole indexer family?
2. Is chunk-wise selection systematically worse, and by how much, when the chunk score is the
   ``logsumexp`` the HSA arm trains against?

**The backbone oracle is the control that makes this interpretable.** It scores keys with the
frozen model's own attention logit ``q k^T / sqrt(d)`` instead of the indexer. If the needle ranks
highly under the oracle but not under any indexer, the routers are underfitting a signal that exists.
If the needle ranks poorly even under the oracle, then the needle is simply not salient in ``q.k``
space and every arm of this investigation has been chasing a ceiling.

Two earlier mistakes this script is built to avoid:

* **Do not plant synthetic needles.** A previous version put the needle at the 99th token percentile,
  but the maximum of 64 iid N(0,1) draws already sits at the 98.4th, so the planted needle was not
  even above a typical chunk's natural maximum and "fell out of the chunk budget" for reasons that
  were pure artefact. Real needle spans, located in real RULER contexts, avoid that entirely.
* **Report ranks, not just hit rates.** "Inside the budget" collapses a continuous quantity onto a
  threshold and hides whether a miss was narrow or hopeless.

``needle_is_chunk_max`` is the diagnostic that explains any token/chunk gap: ``LSE >= max >= needle``
bounds *values*, so a chunk containing a high-scoring needle has a high LSE -- but only if the needle
is actually among the chunk's larger scores. When the needle is not its chunk's max, the chunk's LSE
is set by *other* tokens and the needle's own score barely influences whether the chunk is picked.

    python -m scripts.probe_needle_rank --model /path/Qwen3-8B --task niah_multivalue \\
        --ckpt gated=/path/e2e/final.pt --ckpt distill=/path/distill/step600.pt
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
for extra in (REPO_ROOT, REPO_ROOT / "evaluation"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def summarize(tag: str, acc: dict, k_len: int, take_tok: int, take_chk: int, n_chunk: int) -> dict:
    if not acc["tok_pct"]:
        print(f"{tag}: no locatable needle")
        return {}
    row = {
        "tok_rank_pct": statistics.mean(acc["tok_pct"]),
        "chk_rank_pct": statistics.mean(acc["chk_pct"]),
        "tok_hit": statistics.mean(acc["in_tok"]),
        "chk_hit": statistics.mean(acc["in_chk"]),
        "needle_is_chunk_max": statistics.mean(acc["is_max"]),
        "chunk_margin_sd": statistics.mean(acc["margin"]),
        "n": len(acc["tok_pct"]),
    }
    print(
        f"{tag:>22s} | tok rank top {row['tok_rank_pct']:5.1f}%  hit {100*row['tok_hit']:5.1f}% "
        f"| chk rank top {row['chk_rank_pct']:5.1f}%  hit {100*row['chk_hit']:5.1f}% "
        f"| needle=chunk max {100*row['needle_is_chunk_max']:4.0f}%  margin {row['chunk_margin_sd']:+.2f}sd"
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--ckpt", action="append", default=[],
        help="NAME=PATH, repeatable. The backbone oracle is always added as a control.",
    )
    ap.add_argument("--task", default="niah_multivalue")
    ap.add_argument("--items", type=int, default=6)
    ap.add_argument("--layers", default="0,4,7")
    ap.add_argument("--truncate", type=int, default=8)
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--force-sink", type=int, default=4)
    ap.add_argument("--force-local", type=int, default=64)
    ap.add_argument(
        "--score-scale", type=float, default=None,
        help="temperature inside the chunk logsumexp. Defaults to head_dim ** -0.5, which is what "
        "the HSA arm trained with and the backbone's own attention scale. Neither the gated nor the "
        "distilled checkpoint records one (they never aggregated to chunks), so it is imposed here "
        "and the choice is stated rather than hidden.",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    from evaluate_sparse import load_cached_dataset
    from kvpress import GQAIndexerPress
    from kvpress.presses.gqa_indexer import load_indexer_state_dict
    from probe_router_selection import find_needle_span

    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_cached_dataset("simonjegou/ruler", "8192")
    rows = [r for r in dataset if r["task"] == args.task][: args.items]
    if not rows:
        raise SystemExit(f"no rows for task {args.task!r}")
    probe_layers = [int(x) for x in args.layers.split(",")]
    take_tok = args.topk - args.force_sink - args.force_local
    take_chk = take_tok // args.chunk_size

    specs = [tuple(s.split("=", 1)) for s in args.ckpt]
    results: dict[str, dict] = {}

    for name, ckpt in specs + [("backbone-oracle", None)]:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        if args.truncate:
            model.model.layers = torch.nn.ModuleList(list(model.model.layers[: args.truncate]))
            model.config.num_hidden_layers = args.truncate
        model = model.to(device).eval()
        n_layer = model.config.num_hidden_layers
        layers_here = [li for li in probe_layers if li < n_layer]

        press = None
        if ckpt is not None:
            payload = torch.load(ckpt, map_location="cpu", weights_only=False)
            sd_all = payload.get("indexer", payload)
            sd = {}
            for key, tensor in sd_all.items():
                parts = key.split(".")
                layer = next((int(a) for a, b in zip(parts[1:], parts) if b == "layers"), None)
                if layer is None or layer < n_layer:
                    sd[key] = tensor
            press = GQAIndexerPress(
                compression_ratio=0.0,
                gate_scale=any(str(k).endswith("gate_scale") for k in sd_all),
                scorer_attr="indexer",
            )
            press.post_init_from_model(model, force_reinit=True)
            load_indexer_state_dict(model, sd, "indexer")

        nh = model.config.num_attention_heads
        nkv = model.config.num_key_value_heads
        hd = getattr(model.config, "head_dim", None) or model.config.hidden_size // nh
        grp = nh // nkv
        scale = args.score_scale if args.score_scale is not None else hd**-0.5

        per_layer: dict[int, dict] = {
            li: {"tok_pct": [], "chk_pct": [], "in_tok": [], "in_chk": [], "is_max": [], "margin": []}
            for li in layers_here
        }
        for row in rows:
            answers = row["answer"]
            answers = list(answers) if not isinstance(answers, str) else [answers]
            ids = tokenizer(row["context"], return_tensors="pt").input_ids
            spans = [find_needle_span(ids[0], tokenizer, str(a)) for a in answers]
            spans = [s for s in spans if s is not None]
            if not spans:
                continue
            ids = ids.to(device)
            k_len = ids.shape[1]
            n_chunk = k_len // args.chunk_size

            hidden: dict[int, torch.Tensor] = {}
            kwargs_by: dict[int, dict] = {}
            handles = []

            def mk(li):
                def hook(module, a, kw):
                    x = kw.get("hidden_states")
                    x = a[0] if x is None else x
                    hidden[li] = x
                    kwargs_by[li] = kw
                    return None
                return hook

            for li in layers_here:
                handles.append(
                    model.model.layers[li].self_attn.register_forward_pre_hook(mk(li), with_kwargs=True)
                )
            with torch.no_grad():
                model(input_ids=ids, use_cache=False)
            for h in handles:
                h.remove()

            for li in layers_here:
                attn = model.model.layers[li].self_attn
                h = hidden[li]
                with torch.no_grad():
                    if press is not None:
                        indexer = press.get_indexer(attn)
                        cos, sin = press.get_rope_tables(indexer, kwargs_by[li])
                        qi = indexer.project_q(h, cos, sin)
                        ki = indexer.project_k(h, cos, sin)
                        s = torch.einsum("bhd,bkd->bhk", qi[:, :, -1], ki).float().mean(1)[0]
                    else:
                        # Backbone oracle: the model's own attention logit, averaged over q-heads.
                        q = attn.q_proj(h).view(1, k_len, nh, hd).transpose(1, 2)
                        k = attn.k_proj(h).view(1, k_len, nkv, hd).transpose(1, 2)
                        if hasattr(attn, "q_norm"):
                            q = attn.q_norm(q)
                        if hasattr(attn, "k_norm"):
                            k = attn.k_norm(k)
                        pos = torch.arange(k_len, device=device).unsqueeze(0)
                        rc, rs = model.model.rotary_emb(k, pos)
                        q, k = apply_rotary_pos_emb(q, k, rc, rs)
                        kk = k.repeat_interleave(grp, 1).float()
                        s = (
                            torch.einsum("hd,hkd->hk", q[0, :, -1].float(), kk[0]) * hd**-0.5
                        ).mean(0)

                    blocks = (s[: n_chunk * args.chunk_size] * scale).reshape(n_chunk, args.chunk_size)
                    pooled = torch.logsumexp(blocks, -1)
                    sd_pool = float(pooled.std())
                    med = float(pooled.median())
                    for sp in spans:
                        toks = list(range(sp.start, min(sp.stop, n_chunk * args.chunk_size)))
                        if not toks:
                            continue
                        best = max(toks, key=lambda t: float(s[t]))
                        c = best // args.chunk_size
                        a = per_layer[li]
                        a["tok_pct"].append(100 * int((s > s[best]).sum()) / k_len)
                        a["chk_pct"].append(100 * int((pooled > pooled[c]).sum()) / n_chunk)
                        a["in_tok"].append(int((s > s[best]).sum()) < take_tok)
                        a["in_chk"].append(int((pooled > pooled[c]).sum()) < take_chk)
                        a["is_max"].append(bool(blocks[c].argmax().item() == best % args.chunk_size))
                        a["margin"].append((float(pooled[c]) - med) / max(sd_pool, 1e-9))

        merged = {k: [] for k in ("tok_pct", "chk_pct", "in_tok", "in_chk", "is_max", "margin")}
        print()
        for li in layers_here:
            summarize(f"{name} L{li}", per_layer[li], 8192, take_tok, take_chk, 128)
            for k in merged:
                merged[k] += per_layer[li][k]
        results[name] = summarize(f"{name} ALL", merged, 8192, take_tok, take_chk, 128)
        del model
        torch.cuda.empty_cache()

    print(
        f"\nbudget: token keeps top {100*take_tok/8192:.1f}% of tokens, "
        f"chunk keeps top {100*take_chk/128:.1f}% of chunks; score_scale={args.score_scale or 'head_dim**-0.5'}"
    )
    if args.out:
        Path(args.out).write_text(json.dumps({"task": args.task, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
