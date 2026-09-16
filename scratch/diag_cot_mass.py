"""Is CMP worth building for CoT? Measure the softmax mass eviction throws away on math500.

The question this answers
------------------------
CMP slots can only recover mass that eviction LOST. On documents that was measured at rho
0.21-0.23 (verify3_rho.py), which is what motivated the compensation work. But CoT is a different
regime and might not behave the same way:

* the evicted keys are the model's OWN generated reasoning, not a corpus;
* there is no context to speak of (math500's is a single space), so the cache is 100% CoT;
* attention over one's own recent reasoning may be far more local than attention over a document,
  in which case a 1024-key budget already covers it and no compensation is needed.

So this generates a real CoT with the real trained router, and at each of several points measures
`rho = 1 - exp(lse_S - lse_dense)` -- the fraction of each query row's mass sitting on keys the
router would have evicted at topk=1024.

How to read the result
----------------------
* rho near 0 -> the budget already holds everything that matters; CMP has nothing to recover and
  the whole streaming-CMP idea is not worth building.
* rho comparable to the document case (~0.2) -> there is real headroom, and it is worth building.

`o_E` variability is reported too, for the same reason verify3 reports it: if the evicted branch's
output direction barely moves across queries, a rank-0 summary captures it and R=64 slots are
wasted parameters.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kvpress.presses.gqa_indexer.press import GQAIndexerPress, get_language_model  # noqa: E402
from kvpress.presses.gqa_indexer.qi_flex_attention import deadlines  # noqa: E402
from kvpress.presses.gqa_indexer.train import (  # noqa: E402
    load_indexer_state_dict,
    press_kwargs_from_checkpoint,
)

MODELS = "/apdcephfs_gy8/share_303843174/guhao/models"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=f"{MODELS}/Qwen3-4B")
    p.add_argument(
        "--ckpt",
        default=f"{MODELS}/Qwen-3-4B-gqa_indexer_scalar/rvkl_8k_local128_b256_decay/final.pt",
    )
    p.add_argument("--topk", type=int, default=1024)
    p.add_argument("--n-sink", type=int, default=4)
    p.add_argument("--n-local", type=int, default=128)
    p.add_argument("--gen", type=int, default=3072, help="CoT tokens to generate")
    p.add_argument("--problems", type=int, default=3)
    p.add_argument("--layers", type=str, default="", help="comma list; default = 6 spread")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="scratch/diag_cot_mass.json")
    return p.parse_args()


@torch.no_grad()
def rho_for_layer(q, k, v, dl, *, n_sink, n_local, scale, q_rows, q_tile=64):
    """rho and the evicted-branch direction, for a chosen set of query rows.

    Mirrors verify3_rho.py's computation exactly (same deadline rule, same lse pair), restricted
    to `q_rows` -- the late rows are the interesting ones here, since early CoT rows have not
    accumulated enough history to evict anything.
    """
    group = q.shape[1] // k.shape[1]
    Sk = k.shape[2]
    dev = q.device
    kf = k[0].float()
    vf = v[0].float()
    key_idx = torch.arange(Sk, device=dev)
    dlg = dl.repeat_interleave(group, 0).to(torch.int64)

    rho_all, oE_all, live_all = [], [], []
    for start in range(0, q_rows.numel(), q_tile):
        rows = q_rows[start : start + q_tile]
        qt = q[0, :, rows].float()
        logits = torch.einsum("htd,hsd->hts", qt, kf.repeat_interleave(group, 0)) * scale
        causal = key_idx.view(1, 1, -1) <= rows.view(1, -1, 1)

        limit = rows.clamp(max=Sk - 1)
        horizon = limit - n_local
        sink = key_idx.view(1, 1, -1) < n_sink
        local = (key_idx.view(1, 1, -1) > limit.view(1, -1, 1) - n_local) & ~sink
        alive = horizon.view(1, -1, 1) <= dlg.unsqueeze(1)
        chosen = (~sink) & (key_idx.view(1, 1, -1) <= horizon.view(1, -1, 1)) & alive
        keep = causal & (sink | local | chosen)

        neg = torch.finfo(torch.float32).min
        lse_dense = torch.logsumexp(logits.masked_fill(~causal, neg), -1)
        lse_S = torch.logsumexp(logits.masked_fill(~keep, neg), -1)
        rho = (1.0 - torch.exp(lse_S - lse_dense)).clamp(0.0, 1.0)

        p_d = torch.softmax(logits.masked_fill(~causal, neg), -1)
        o_d = torch.einsum("hts,hsd->htd", p_d, vf.repeat_interleave(group, 0))
        p_s = torch.softmax(logits.masked_fill(~keep, neg), -1)
        o_s = torch.einsum("hts,hsd->htd", p_s, vf.repeat_interleave(group, 0))
        denom = rho.clamp(min=1e-4)
        oE = (o_d - (1.0 - rho).unsqueeze(-1) * o_s) / denom.unsqueeze(-1)
        live = rho > 1e-3

        rho_all.append(rho.reshape(-1))
        oE_all.append(torch.where(live.unsqueeze(-1), oE, torch.zeros_like(oE)))
        live_all.append(live)
        del logits, p_d, p_s, keep, causal

    rho = torch.cat(rho_all)
    oE = torch.cat(oE_all, 1)
    live = torch.cat(live_all, 1)
    out = {
        "rho_mean": float(rho.mean()),
        "rho_median": float(rho.median()),
        "rho_p90": float(rho.quantile(0.90)),
        "rows_with_eviction": float(live.float().mean()),
    }
    # How much of the evicted direction is a single constant per head (the rank-0 question).
    per_head = []
    for h in range(oE.shape[0]):
        sel = oE[h][live[h]]
        if sel.shape[0] < 8:
            continue
        mu = sel.mean(0, keepdim=True)
        resid = (sel - mu).norm(dim=-1).mean()
        per_head.append(float(resid / sel.norm(dim=-1).mean().clamp(min=1e-6)))
    out["oE_rank0_residual"] = sum(per_head) / max(len(per_head), 1)
    return out


def main():
    args = parse_args()
    dev = torch.device(args.device)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
    model = model.to(dev).eval()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt.get("indexer", ckpt)
    scorer, kw = press_kwargs_from_checkpoint(sd, ckpt.get("config") or {})
    has_gate = any(str(k).endswith("gate_scale") for k in sd)
    press = GQAIndexerPress(compression_ratio=0.0, gate_scale=has_gate, scorer=scorer, **kw)
    press.post_init_from_model(model)
    load_indexer_state_dict(model, sd, "indexer")

    layers = get_language_model(model).layers
    n_layers = len(layers)
    picked = (
        [int(x) for x in args.layers.split(",") if x != ""]
        if args.layers
        else sorted({0, n_layers // 5, 2 * n_layers // 5, 3 * n_layers // 5,
                     4 * n_layers // 5, n_layers - 1})
    )
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    scale = head_dim ** -0.5
    print(f"model {args.model} ({n_layers}L, {cfg.num_key_value_heads}KV, D={head_dim})")
    print(f"scorer={scorer} topk={args.topk} n_local={args.n_local} gen={args.gen}")

    # Real math500 problems, so the CoT is the model's genuine reasoning.
    import glob
    from pathlib import Path

    from datasets import Dataset

    hits = sorted(glob.glob(str(
        Path.home() / ".cache/huggingface/datasets/alessiodevoto___math500/*/*/*/*.arrow"
    )))
    ds = Dataset.from_file(hits[0])
    results = []
    for pi in range(args.problems):
        row = ds[pi]
        msg = [{"role": "user", "content": row["question"]}]
        text = tok.apply_chat_template(
            msg, add_generation_prompt=True, tokenize=False, enable_thinking=True
        )
        ids = tok.encode(text, return_tensors="pt", add_special_tokens=False).to(dev)
        gen = model.generate(
            ids, max_new_tokens=args.gen, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        full = gen  # prompt + CoT
        print(f"\nproblem {pi}: prompt {ids.shape[1]} + CoT {full.shape[1] - ids.shape[1]} "
              f"= {full.shape[1]} tokens")

        # One dense forward over the whole thing, capturing q/k/v and the hidden states.
        grabbed, hidden = {}, {}
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        import torch.nn.functional as F

        group = cfg.num_attention_heads // cfg.num_key_value_heads

        def impl(module, q, k, v, attention_mask, scaling=None, dropout=0.0, **_):
            li = int(module.layer_idx)
            if li in picked:
                grabbed[li] = (q.detach(), k.detach(), v.detach())
            o = F.scaled_dot_product_attention(
                q, k.repeat_interleave(group, 1), v.repeat_interleave(group, 1),
                is_causal=True, scale=scaling,
            )
            return o.transpose(1, 2).contiguous(), None

        def pre(module, a, kws):
            li = int(module.layer_idx)
            if li in picked:
                hs = kws.get("hidden_states")
                if hs is None and a:
                    hs = a[0]
                hidden[li] = hs.detach()
            return None

        name = "cot_mass_probe"
        gmap = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
        ALL_ATTENTION_FUNCTIONS.register(name, impl)
        handles = [
            lyr.self_attn.register_forward_pre_hook(pre, with_kwargs=True) for lyr in layers
        ]
        configs = [cfg] + ([cfg.text_config] if getattr(cfg, "text_config", None) else [])
        prev = [c._attn_implementation for c in configs]
        for c in configs:
            c._attn_implementation = name
        try:
            model(input_ids=full, use_cache=False)
        finally:
            for h in handles:
                h.remove()
            for c, p in zip(configs, prev):
                c._attn_implementation = p
            gmap.pop(name, None)

        Sk = full.shape[1]
        # Late rows only: they are the ones whose history actually exceeds the budget, i.e. the
        # rows a long CoT spends most of its time on.
        lo = max(args.topk + args.n_local, int(0.6 * Sk))
        if lo >= Sk - 1:
            print("  CoT never exceeded the budget; nothing is evicted. Skipping.")
            continue
        q_rows = torch.arange(lo, Sk, device=dev)
        for li in picked:
            q, k, v = grabbed[li]
            idx = press.get_indexer(layers[li].self_attn)
            hs = hidden[li]
            if getattr(idx, "decay", False):
                scores = idx.score_at(hs, float(Sk - 1))[0]
            else:
                scores = idx.score_keys(hs)[0]
            dl = deadlines(
                scores, args.topk, force_sink=args.n_sink, force_local=args.n_local
            )
            r = rho_for_layer(
                q, k, v, dl, n_sink=args.n_sink, n_local=args.n_local,
                scale=scale, q_rows=q_rows,
            )
            r.update(problem=pi, layer=li, seq_len=Sk, rows=int(q_rows.numel()))
            results.append(r)
            print(f"  layer {li:2d}: rho mean={r['rho_mean']:.4f} med={r['rho_median']:.4f} "
                  f"p90={r['rho_p90']:.4f}  rows_evicting={r['rows_with_eviction']:.2f} "
                  f"oE_rank0_resid={r['oE_rank0_residual']:.3f}")
        grabbed.clear()
        hidden.clear()
        torch.cuda.empty_cache()

    if results:
        import statistics

        rm = statistics.mean(r["rho_mean"] for r in results)
        print(f"\n=== OVERALL rho_mean over {len(results)} (problem, layer) pairs: {rm:.4f}")
        print("   document reference (verify3, 8B): rho 0.21-0.23")
        print("   rank-0 residual mean: "
              f"{statistics.mean(r['oE_rank0_residual'] for r in results):.3f}")
    with open(args.out, "w") as fh:
        json.dump({"config": vars(args), "rows": results}, fh, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
