# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Ablation: run the CMP k-means in pre-RoPE (content) space instead of post-RoPE (cache) space.

The concern this settles. ``k_cmp`` must be a mean of *post-RoPE* keys either way -- the slot's logit
is ``q . k_cmp`` and it has to approximate ``mean_j(q . R_j k_j) = q . mean_j(R_j k_j)``. So the only
thing the space changes is the **partition**. But a post-RoPE key's direction depends strongly on its
position, so k-means over it can silently degenerate into positional chunking, which is the variant
already measured to lose (cos 0.68-0.70 against 0.85-0.86).

Measured on the real model, that degeneracy is real: ``position_locality`` (member-position std
normalized so 1.0 = as spread as uniform) is **0.041-0.317** for post-RoPE against **0.868-0.969**
for pre-RoPE. So post-RoPE clustering is essentially *adaptive positional segmentation* -- variable
width spans placed where content changes -- rather than content clustering. That it still beat
*fixed* positional chunking says the adaptivity is worth something; it does not say it is the best
partition.

Pre-RoPE has a cost pulling the other way, which is why this needs measuring rather than arguing: its
clusters span arbitrary positions, so the post-RoPE mean they must still produce suffers more phase
cancellation. In the limit ``k_cmp -> 0`` the slot becomes content-blind and only ``b_r`` can address
it. The measured ``norm_ratio`` (0.70-0.99 pre against 0.75-1.00 post) says that cost is small, but
the read quality is the number that decides.

Reports the same two quantities as ``probe_compensation.py``, separately and for the same reason
(a healthy cos with a broken magnitude is a scale bug, not a training failure):

* ``cos(pred, oE*)`` on held-out query rows, threshold **0.8**;
* fused relative error against ``o_dense`` **given the oracle mass**, so the comparison is about the
  direction alone -- a correct mass on a mediocre direction was measured 3-5x WORSE than no slot.

Geometry is the C1/C2 split, so the evicted set is row-independent and one slot set genuinely serves
every reader -- which is what deployment does and what keeps this leak-free.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kvpress.presses.gqa_indexer.cmp_slots import (  # noqa: E402
    cluster_evicted,
    unrotate,
)
from kvpress.presses.gqa_indexer.press import (  # noqa: E402
    GQAIndexerPress,
    get_language_model,
)
from kvpress.presses.gqa_indexer.train import (  # noqa: E402
    load_indexer_state_dict,
    press_kwargs_from_checkpoint,
)

MODELS = "/apdcephfs_gy8/share_303843174/guhao/models"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=f"{MODELS}/Qwen3-8B")
    p.add_argument(
        "--router",
        default=f"{MODELS}/Qwen-3-8B-gqa_indexer_scalar/stage1_16k_mid256_longce/final.pt",
    )
    p.add_argument("--length", type=int, default=8192)
    p.add_argument("--split", type=int, default=6144)
    p.add_argument("--keep-ratio", type=float, default=0.25)
    p.add_argument("--slots", default="16,64,256")
    p.add_argument("--layers", default="0,9,18,27,32,35")
    p.add_argument("--n-sink", type=int, default=4)
    p.add_argument("--row-stride", type=int, default=4)
    p.add_argument("--docs", type=int, default=2)
    p.add_argument("--out", default="scratch/probe_cmp_prerope.json")
    return p.parse_args()


@torch.no_grad()
def run_layer(q, k, v, scores, cos, sin, *, args, scale, group, layer):
    L, L1 = args.length, args.split
    dev = q.device
    Hkv, D = k.shape[1], k.shape[-1]
    kf, vf = k[0].float(), v[0].float()
    pre_k = unrotate(kf, cos[0].float(), sin[0].float())
    budget = max(1, int(L1 * args.keep_ratio))

    rows = torch.arange(L1, L, args.row_stride, device=dev)
    out = []
    for R in [int(x) for x in args.slots.split(",")]:
        # budget-neutral: the slots are funded out of the exact budget
        n_exact = max(args.n_sink, budget - R)
        s_c1 = scores[:, :L1].clone()
        s_c1[:, : args.n_sink] = float("inf")
        keep = torch.zeros(Hkv, L1, dtype=torch.bool, device=dev)
        keep.scatter_(1, s_c1.topk(min(n_exact, L1), dim=-1).indices, True)
        ev_full = torch.zeros(Hkv, L, dtype=torch.bool, device=dev)
        ev_full[:, :L1] = ~keep

        arms = {}
        for name, ck in (("post_rope", None), ("pre_rope", pre_k)):
            g = torch.Generator(device=dev).manual_seed(0)
            k_cmp, v_cmp, b_cmp, diag = cluster_evicted(
                kf, vf, ev_full, R, cluster_keys=ck, generator=g, diagnostics=True
            )
            arms[name] = (k_cmp, v_cmp, b_cmp, diag)

        per_head = []
        for hkv in range(Hkv):
            hq0, hq1 = hkv * group, (hkv + 1) * group
            kh, vh = kf[hkv], vf[hkv]
            ev = ev_full[hkv]
            ei = ev.nonzero(as_tuple=True)[0]
            if ei.numel() < 16:
                continue
            acc = {kk: [] for kk in ("oE", "DE", "od", "oS", "lseS", "rho", "q")}
            for s0 in range(0, rows.numel(), 256):
                r = rows[s0 : s0 + 256]
                qt = q[0, hq0:hq1, r].float()
                lg = torch.einsum("htd,sd->hts", qt, kh) * scale
                neg = torch.finfo(torch.float32).min
                kmask = torch.cat(
                    [keep[hkv], torch.ones(L - L1, dtype=torch.bool, device=dev)]
                ).view(1, 1, -1)
                causal = torch.arange(L, device=dev).view(1, 1, -1) <= r.view(1, -1, 1)
                lse_d = torch.logsumexp(lg.masked_fill(~causal, neg), -1)
                lse_s = torch.logsumexp(lg.masked_fill(~(causal & kmask), neg), -1)
                acc["od"].append(
                    torch.einsum(
                        "hts,sd->htd", torch.softmax(lg.masked_fill(~causal, neg), -1), vh
                    )
                )
                acc["oS"].append(
                    torch.einsum(
                        "hts,sd->htd",
                        torch.softmax(lg.masked_fill(~(causal & kmask), neg), -1),
                        vh,
                    )
                )
                acc["lseS"].append(lse_s)
                acc["rho"].append((1.0 - torch.exp(lse_s - lse_d)).clamp(0, 1))
                lge = lg[:, :, ei]
                acc["oE"].append(torch.einsum("hts,sd->htd", torch.softmax(lge, -1), vh[ei]))
                acc["DE"].append(torch.exp(torch.logsumexp(lge, -1)))
                acc["q"].append(qt)
                del lg, lge, causal
            t = {kk: torch.cat(vv, 1) for kk, vv in acc.items()}
            live = t["rho"] > 1e-3
            DS = torch.exp(t["lseS"])
            wmass = t["DE"] / (DS + t["DE"]).clamp(min=1e-20)

            def score(pred):
                fused = (1 - wmass).unsqueeze(-1) * t["oS"] + wmass.unsqueeze(-1) * pred
                e = (fused - t["od"]).norm(dim=-1) / t["od"].norm(dim=-1).clamp(min=1e-9)
                c = torch.nn.functional.cosine_similarity(pred, t["oE"], dim=-1)
                return float(e[live].mean()), float(c[live].mean())

            row = {}
            e0 = (t["oS"] - t["od"]).norm(dim=-1) / t["od"].norm(dim=-1).clamp(min=1e-9)
            row["none"] = float(e0[live].mean())
            for name, (k_cmp, v_cmp, b_cmp, diag) in arms.items():
                l = (
                    torch.einsum("htd,rd->htr", t["q"], k_cmp[hkv]) * scale
                    + b_cmp[hkv].view(1, 1, -1)
                )
                w = l.exp()
                pred = torch.einsum("htr,rd->htd", w, v_cmp[hkv]) / w.sum(-1).clamp(
                    min=1e-30
                ).unsqueeze(-1)
                row[name], row["cos_" + name] = score(pred)
                lv = diag["live"][hkv]
                row["poslocal_" + name] = float(diag["position_locality"][hkv][lv].mean())
                row["normratio_" + name] = float(diag["norm_ratio"][hkv][lv].mean())
            row["rho"] = float(t["rho"].mean())
            per_head.append(row)
            del t
            torch.cuda.empty_cache()

        agg = {kk: float(sum(h[kk] for h in per_head) / len(per_head)) for kk in per_head[0]}
        agg.update(layer=layer, slots=R, n_exact=n_exact)
        out.append(agg)
        print(
            f"  L{layer:2d} R={R:>3}: none {agg['none']:.4f} | "
            f"post {agg['post_rope']:.4f} (cos {agg['cos_post_rope']:.3f}, "
            f"loc {agg['poslocal_post_rope']:.3f}) | "
            f"pre {agg['pre_rope']:.4f} (cos {agg['cos_pre_rope']:.3f}, "
            f"loc {agg['poslocal_pre_rope']:.3f})",
            flush=True,
        )
    return out


def main():
    args = parse_args()
    dev = "cuda"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model = model.to(dev).eval()
    model.requires_grad_(False)

    rck = torch.load(args.router, map_location="cpu", weights_only=False)
    rsd, rcfg = rck["indexer"], rck.get("config", {})
    _, kw = press_kwargs_from_checkpoint(rsd, rcfg)
    press = GQAIndexerPress(
        compression_ratio=1.0 - args.keep_ratio, scorer="scalar",
        gate_scale=any("gate_scale" in k for k in rsd), n_sink=args.n_sink, n_local=0, **kw,
    )
    press.post_init_from_model(model)
    load_indexer_state_dict(model, rsd)

    tcfg = getattr(model.config, "text_config", model.config)
    n_q, n_kv = tcfg.num_attention_heads, tcfg.num_key_value_heads
    head_dim = getattr(tcfg, "head_dim", tcfg.hidden_size // n_q)
    group = n_q // n_kv
    layers_all = get_language_model(model).layers
    WANT = [int(x) for x in args.layers.split(",")]

    grab, hid = {}, {}

    def impl(module, q, k, v, am, scaling=None, dropout=0.0, **kwargs):
        import torch.nn.functional as F

        i = int(module.layer_idx)
        if i in WANT:
            grab[i] = (q.detach(), k.detach(), v.detach(), scaling)
        g = q.shape[1] // k.shape[1]
        o = F.scaled_dot_product_attention(
            q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1), is_causal=True, scale=scaling
        )
        return o.transpose(1, 2).contiguous(), None

    def pre(module, a, kwargs):
        i = int(getattr(module, "layer_idx", -1))
        if i in WANT:
            hs = kwargs.get("hidden_states")
            hid[i] = (hs if hs is not None else a[0]).detach()
        return None

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register("probe_prerope", impl)
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is not None:
            cfg._attn_implementation = "probe_prerope"
    for layer in layers_all:
        layer.self_attn.register_forward_pre_hook(pre, with_kwargs=True)

    from kvpress.presses.gqa_indexer.data import LongminoConfig, build_dataloader

    loader = build_dataloader(
        LongminoConfig(
            root="/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_256k_filtered",
            seq_len=args.length, subsets=("2e16",), take_from="head",
        ),
        tok, batch_size=1, num_workers=0,
    )
    docs = []
    for b in loader:
        docs.append(b["input_ids"][:, : args.length].to(dev))
        if len(docs) >= args.docs:
            break

    rope = layers_all[0].self_attn if False else get_language_model(model).rotary_emb
    pos = torch.arange(args.length, device=dev).unsqueeze(0)
    results = []
    for di, ids in enumerate(docs):
        grab.clear()
        hid.clear()
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        print(f"\n=== doc {di} ===", flush=True)
        for idx in WANT:
            q, k, v, scaling = grab[idx]
            sc = press.get_indexer(layers_all[idx].self_attn).score_keys(hid[idx])[0].float()
            cos, sin = rope(k, pos)
            rows = run_layer(
                q, k, v, sc, cos, sin, args=args,
                scale=(head_dim**-0.5 if scaling is None else float(scaling)),
                group=group, layer=idx,
            )
            for r in rows:
                r["doc"] = di
            results.extend(rows)
            torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")
    summarize(results, args)


def summarize(results, args):
    layers = sorted({r["layer"] for r in results})
    for R in sorted({r["slots"] for r in results}):
        print("\n" + "=" * 96)
        print(f"R = {R}   (fused rel error given ORACLE MASS; ratio vs eviction)")
        print(
            f"{'layer':>5} {'none':>8} {'post':>8} {'ratio':>7} {'pre':>8} {'ratio':>7} "
            f"{'cos_post':>9} {'cos_pre':>8} {'loc_post':>9} {'loc_pre':>8} "
            f"{'nrm_post':>9} {'nrm_pre':>8}"
        )
        for L in layers:
            rr = [r for r in results if r["layer"] == L and r["slots"] == R]
            if not rr:
                continue
            m = {k: sum(r[k] for r in rr) / len(rr) for k in rr[0] if isinstance(rr[0][k], float)}
            print(
                f"{L:>5} {m['none']:>8.4f} {m['post_rope']:>8.4f} "
                f"{m['none']/max(m['post_rope'],1e-9):>6.2f}x {m['pre_rope']:>8.4f} "
                f"{m['none']/max(m['pre_rope'],1e-9):>6.2f}x {m['cos_post_rope']:>9.3f} "
                f"{m['cos_pre_rope']:>8.3f} {m['poslocal_post_rope']:>9.3f} "
                f"{m['poslocal_pre_rope']:>8.3f} {m['normratio_post_rope']:>9.3f} "
                f"{m['normratio_pre_rope']:>8.3f}"
            )
    print("\nloc = position_locality: 0 = clusters are contiguous position spans, 1 = position-blind.")
    print("nrm = ||k_cmp||/mean||k_j||: how much of the centroid survives RoPE phase cancellation.")


if __name__ == "__main__":
    main()
