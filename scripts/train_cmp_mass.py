# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Train :class:`CMPMassHead` -- 3 scalars per (layer, KV head) governing how loudly a slot speaks.

What is being learned, and why it is not just a constant offset
--------------------------------------------------------------
The training-free slot claims ``b_r = log n_r``: a cluster of ``n_r`` evicted keys asks for ``n_r``
times a singleton's softmax mass. That is right only if the members' logits are interchangeable, and
they are not -- a cluster's participation ratio sits well below its size, so the multiplicity a query
actually experiences grows like ``n_r^c`` with ``c < 1``. So the parameterization is

    ``b_r = count_coef * log n_r + var_coef * (1/2 Var_r) + bias``

with ``count_coef`` the one that matters: ``bias`` shifts every slot equally, while ``count_coef``
changes how mass *scales with cluster size*, which is the thing a flat count gets wrong.

Objective: fused output L2, not a regression on the mass
--------------------------------------------------------
The obvious target is the cluster's true ``logsumexp``, i.e. fit ``b_r`` to the mass it should carry.
Deliberately **not** that, for a measured reason: with the direction held fixed and only ``b_r``
improved, the fused error went from 1.02x (mass off by 2.0 nats) to **0.29x** -- 3.5x *worse* than
plain eviction. A slot whose ``v_r`` points the wrong way should claim *less* mass than its count
implies, not the correct amount. A mass regression cannot express that; it would confidently make
those layers worse.

So the loss is the ``o_proj``-space L2 of the **fused attention output** against dense, exactly as
Stage A does. ``o_proj`` weights the channels, so a channel-uniform MSE would over-serve channels the
model itself ignores, and the row weighting comes out right for free: ``o_fused - o_dense`` is
proportional to the evicted mass share, so rows eviction did not hurt contribute nothing.

The payoff is that **the per-layer gate falls out instead of being hand-set**. Layers whose direction
is bad (L27/L32 measured 0.25-0.42x) are ones where the objective's own optimum is a large negative
``bias``, i.e. silence. No ``cos >= 0.8`` threshold to pick.

Geometry: the C1/C2 split, so the evicted set is row-independent and one slot set genuinely serves
every reader -- which is what deployment does, and what keeps this leak-free. Trained on one
document, reported on a held-out one, because the question is whether 3 scalars per head transfer,
not whether they can fit 2048 rows of one document.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kvpress.presses.gqa_indexer.cmp_slots import (  # noqa: E402
    CMPMassHead,
    cluster_reduce,
    kmeans_assign,
    prefill_logit_var,
    slot_mass,
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
        default=f"{MODELS}/Qwen-3-8B-gqa_indexer_scalar/stage1_16k_mid256_longce_decay/final.pt",
    )
    p.add_argument("--length", type=int, default=8192)
    p.add_argument("--split", type=int, default=6144)
    p.add_argument("--keep-ratio", type=float, default=0.25)
    p.add_argument("--slots", type=int, default=64)
    p.add_argument("--space", default="post_rope", choices=["post_rope", "pre_rope"])
    p.add_argument("--layers", default="all")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--row-stride", type=int, default=4)
    p.add_argument("--docs", type=int, default=3, help="first N-1 train, last held out")
    p.add_argument("--out", default="")
    return p.parse_args()


@torch.no_grad()
def build_layer_cache(q, k, v, scores, cos, sin, *, args, scale, group, space):
    """
    Everything the loss needs for one (layer, document), with the CMP partition FROZEN.

    The k-means assignment does not depend on the mass head, so it is computed once and reused for
    every step -- which is both the cheap thing to do and the correct one: re-clustering per step
    would make the population counts move under the parameter being fitted to them.
    """
    L, L1 = args.length, args.split
    dev = q.device
    Hkv, D = k.shape[1], k.shape[-1]
    kf, vf = k[0].float(), v[0].float()
    budget = max(1, int(L1 * args.keep_ratio))
    R = args.slots
    n_exact = max(4, budget - R)

    s_c1 = scores[:, :L1].clone()
    s_c1[:, :4] = float("inf")
    keep = torch.zeros(Hkv, L1, dtype=torch.bool, device=dev)
    keep.scatter_(1, s_c1.topk(min(n_exact, L1), dim=-1).indices, True)
    ev = torch.zeros(Hkv, L, dtype=torch.bool, device=dev)
    ev[:, :L1] = ~keep
    w = ev.float()

    cluster_space = kf if space == "post_rope" else unrotate(kf, cos.float(), sin.float())
    g = torch.Generator(device=dev).manual_seed(0)
    _, assign = kmeans_assign(cluster_space, R, weights=w, generator=g)
    n_eff = R
    # POST-RoPE reduction regardless of the clustering space: the slot's logit is q.k_cmp, which must
    # approximate mean_j(q . R_j k_j) = q . mean_j(R_j k_j).
    k_cmp, pop = cluster_reduce(kf, assign, n_eff, w)
    v_cmp, _ = cluster_reduce(vf, assign, n_eff, w)
    q_kv = q[0].float().view(Hkv, group, L, D).mean(1)
    logit_var = prefill_logit_var(q_kv, kf, assign, n_eff, w, scaling=scale)

    rows = torch.arange(L1, L, args.row_stride, device=dev)
    acc = {kk: [] for kk in ("oS", "lseS", "od", "q")}
    for s0 in range(0, rows.numel(), 256):
        r = rows[s0 : s0 + 256]
        qt = q[0, :, r].float()  # (Hq, T, D)
        krep = kf.repeat_interleave(group, 0)
        vrep = vf.repeat_interleave(group, 0)
        lg = torch.einsum("htd,hsd->hts", qt, krep) * scale
        neg = torch.finfo(torch.float32).min
        kmask = torch.cat(
            [keep, torch.ones(Hkv, L - L1, dtype=torch.bool, device=dev)], dim=1
        ).repeat_interleave(group, 0).unsqueeze(1)
        causal = torch.arange(L, device=dev).view(1, 1, -1) <= r.view(1, -1, 1)
        acc["od"].append(
            torch.einsum("hts,hsd->htd", torch.softmax(lg.masked_fill(~causal, neg), -1), vrep)
        )
        ks = causal & kmask
        acc["oS"].append(
            torch.einsum("hts,hsd->htd", torch.softmax(lg.masked_fill(~ks, neg), -1), vrep)
        )
        acc["lseS"].append(torch.logsumexp(lg.masked_fill(~ks, neg), -1))
        acc["q"].append(qt)
        del lg, causal, ks
    out = {kk: torch.cat(vv, 1) for kk, vv in acc.items()}
    out.update(k_cmp=k_cmp, v_cmp=v_cmp, pop=pop, logit_var=logit_var, group=group, scale=scale)
    return out


def fused_loss(o_proj, cache, b_cmp, *, reduce=True):
    """``o_proj``-space L2 of the fused output against dense, plus the attention-space rel error."""
    from kvpress.presses.gqa_indexer.memory import fuse_memory

    group = cache["group"]
    scale = cache["scale"]
    q = cache["q"]  # (Hq, T, D)
    Hq, T, D = q.shape
    kc = cache["k_cmp"].repeat_interleave(group, 0)
    vc = cache["v_cmp"].repeat_interleave(group, 0)
    bc = b_cmp.repeat_interleave(group, 0)

    l_cmp = torch.einsum("htd,hrd->htr", q, kc) * scale + bc.unsqueeze(1)
    w = l_cmp.exp()
    n = torch.einsum("htr,hrd->htd", w, vc).unsqueeze(0)
    d = w.sum(-1).unsqueeze(0)
    fused = fuse_memory(
        cache["oS"].unsqueeze(0), cache["lseS"].unsqueeze(0), n, d, group=1
    )[0]

    od = cache["od"]
    fl = fused.permute(1, 0, 2).reshape(T, Hq * D)
    dl = od.permute(1, 0, 2).reshape(T, Hq * D)
    pf = o_proj(fl.to(o_proj.weight.dtype)).float()
    pd = o_proj(dl.to(o_proj.weight.dtype)).float()
    loss = (pf - pd).square().mean()
    with torch.no_grad():
        rel = float(
            ((fused - od).norm(dim=-1) / od.norm(dim=-1).clamp(min=1e-9)).mean()
        )
    return (loss, rel) if reduce else (loss, rel, fused)


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
        gate_scale=any("gate_scale" in k for k in rsd), n_sink=4, n_local=0, **kw,
    )
    press.post_init_from_model(model)
    load_indexer_state_dict(model, rsd)

    tcfg = getattr(model.config, "text_config", model.config)
    n_q, n_kv = tcfg.num_attention_heads, tcfg.num_key_value_heads
    head_dim = getattr(tcfg, "head_dim", tcfg.hidden_size // n_q)
    group = n_q // n_kv
    layers_all = get_language_model(model).layers
    n_layers = len(layers_all)
    WANT = (
        list(range(n_layers)) if args.layers == "all"
        else [int(x) for x in args.layers.split(",")]
    )

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

    ALL_ATTENTION_FUNCTIONS.register("cmp_mass_train", impl)
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is not None:
            cfg._attn_implementation = "cmp_mass_train"
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
    assert len(docs) >= 2, "need at least a train and a held-out document"

    rope = get_language_model(model).rotary_emb
    pos = torch.arange(args.length, device=dev).unsqueeze(0)
    scale = head_dim**-0.5

    # One forward per document, caching every layer's frozen partition + targets. Done up front so
    # the training loop touches no model weights at all -- it only re-evaluates b_r.
    # Capture the raw q/k/v/hidden per document ONCE (kept in bf16 on GPU: 36 layers x 8 kv heads is
    # ~100 MB/layer/doc for k/v and 400 MB for q, so this is the affordable half), then derive each
    # layer's fp32 targets on demand. Holding all (doc x layer) derived caches at once was ~3 GB on
    # top of the model and got the process OOM-killed at layer 13 with no traceback.
    raw: list[dict[int, tuple]] = []
    for di, ids in enumerate(docs):
        grab.clear()
        hid.clear()
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        raw.append(
            {
                idx: (
                    grab[idx][0].clone(), grab[idx][1].clone(), grab[idx][2].clone(),
                    grab[idx][3],
                    press.get_indexer(layers_all[idx].self_attn)
                    .score_keys(hid[idx])[0].float().clone(),
                )
                for idx in WANT
            }
        )
        print(f"captured doc {di} ({len(WANT)} layers)", flush=True)
        torch.cuda.empty_cache()

    heads = {idx: CMPMassHead(n_kv).to(dev) for idx in WANT}
    results = {}

    print(f"\ntraining {len(WANT)} layers x 3 scalars/head, {args.steps} steps, R={args.slots}")
    print(f"{'layer':>5} {'none':>8} {'free':>8} {'var':>8} {'learn':>8} {'ratio':>7} "
          f"{'count':>7} {'varc':>7} {'bias':>8}")
    for idx in WANT:
        head = heads[idx]
        o_proj = layers_all[idx].self_attn.o_proj
        # Derive this layer's caches for every document, use them, then drop them.
        per_doc = []
        for d in raw:
            q, k, v, scaling, sc = d[idx]
            cos, sin = rope(k, pos)
            per_doc.append(
                build_layer_cache(
                    q, k, v, sc, cos[0], sin[0], args=args,
                    scale=(scale if scaling is None else float(scaling)),
                    group=group, space=args.space,
                )
            )
        train_caches, val_cache = per_doc[:-1], per_doc[-1]
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
        for step in range(args.steps):
            tot = 0.0
            opt.zero_grad(set_to_none=True)
            for c in train_caches:
                b = head(c["pop"], c["logit_var"])
                loss, _ = fused_loss(o_proj, c, b)
                loss.backward()
                tot += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            if step == 0:
                first_loss = tot
        last_loss = tot

        # held out: the learned head against the two fixed modes and against no slot at all
        c = val_cache
        with torch.no_grad():
            b_learn = head(c["pop"], c["logit_var"])
            _, rel_learn = fused_loss(o_proj, c, b_learn)
            _, rel_count = fused_loss(o_proj, c, slot_mass(c["pop"]))
            _, rel_var = fused_loss(
                o_proj, c, slot_mass(c["pop"], logit_var=c["logit_var"])
            )
            silent = torch.full_like(c["pop"], -float("inf"))
            _, rel_none = fused_loss(o_proj, c, silent)
            l_learn, _ = fused_loss(o_proj, c, b_learn)
            l_var, _ = fused_loss(o_proj, c, slot_mass(c["pop"], logit_var=c["logit_var"]))
            l_count, _ = fused_loss(o_proj, c, slot_mass(c["pop"]))
            l_none, _ = fused_loss(o_proj, c, silent)
        results[idx] = dict(
            none=rel_none, count=rel_count, var=rel_var, learn=rel_learn,
            count_coef=float(head.count_coef.detach().mean()),
            var_coef=float(head.var_coef.detach().mean()),
            bias=float(head.bias.detach().mean()),
            train_first=first_loss, train_last=last_loss,
            val_l2_none=float(l_none), val_l2_count=float(l_count),
            val_l2_var=float(l_var), val_l2_learn=float(l_learn),
        )
        print(
            f"{idx:>5} {rel_none:>8.4f} {rel_count:>8.4f} {rel_var:>8.4f} {rel_learn:>8.4f} "
            f"{rel_none/max(rel_learn,1e-9):>6.2f}x {float(head.count_coef.detach().mean()):>7.3f} "
            f"{float(head.var_coef.detach().mean()):>7.3f} {float(head.bias.detach().mean()):>+8.3f}"
            f"  | trainL2 {first_loss:.4g}->{last_loss:.4g}"
            f"  valL2 none {float(l_none):.4g} var {float(l_var):.4g} learn {float(l_learn):.4g}",
            flush=True,
        )
        del per_doc, train_caches, val_cache, c
        torch.cuda.empty_cache()

    out = args.out or (
        f"{MODELS}/Qwen-3-8B-gqa_indexer_scalar/cmp_mass_R{args.slots}_{args.space}.pt"
    )
    torch.save(
        {
            "mass_heads": {idx: h.state_dict() for idx, h in heads.items()},
            "config": {
                "slots": args.slots, "space": args.space, "length": args.length,
                "split": args.split, "keep_ratio": args.keep_ratio, "steps": args.steps,
                "lr": args.lr, "router": args.router, "n_kv_heads": n_kv,
            },
            "held_out": results,
        },
        out,
    )
    print(f"\nsaved {out}")

    # Summary: how many layers each mode wins, and where the learned head chose silence.
    def wins(mode):
        return sum(1 for r in results.values() if r["none"] / max(r[mode], 1e-9) >= 1.0)

    print("\n==== held-out summary ====")
    for mode in ("count", "var", "learn"):
        rs = [r["none"] / max(r[mode], 1e-9) for r in results.values()]
        print(
            f"{mode:>6}: beats eviction on {wins(mode):>2}/{len(results)} layers, "
            f"mean ratio {sum(rs)/len(rs):.3f}, median {sorted(rs)[len(rs)//2]:.3f}"
        )
    quiet = [i for i, r in results.items() if r["bias"] < -2.0]
    print(f"\nlearned silence (bias < -2): {len(quiet)} layers {quiet}")
    print(f"mean count_coef {sum(r['count_coef'] for r in results.values())/len(results):.3f} "
          f"(1.0 = trust log n_r at face value)")
    with open(os.path.splitext(out)[0] + "_heldout.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)


if __name__ == "__main__":
    main()
