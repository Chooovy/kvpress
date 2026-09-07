# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Verification 3: how much softmax mass does eviction actually throw away, and is it low-rank?

The whole memory arm can only recover what eviction lost. Two numbers bound it:

* ``rho = 1 - exp(lse_S - lse_dense)`` -- the fraction of each row's softmax mass that lives on
  keys the router evicted. This is the ceiling on any compensation, memory or otherwise. Both
  lse's come out of the attention kernels for free.
* ``o_E* = (o_dense - rho_S * o_S) / (1 - rho_S)`` -- the exact output direction of the evicted
  part. If it barely varies across queries, then a rank-0 memory (one learned vector per head,
  scaled by mass) captures most of the benefit and R=16 is wasted parameters.

Run per layer, at several lengths. Uses the real trained scalar router, because rho under a
*trained* router is the number that matters -- an untrained one evicts differently.
"""

from __future__ import annotations

import argparse
import json
import math
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B")
    p.add_argument(
        "--ckpt",
        default="/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_scalar/"
        "stage1_16k_mid256_longce/step600.pt",
    )
    p.add_argument("--tokenized", default="/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_256k_tokenized")
    p.add_argument("--data-root", default="/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_256k_filtered")
    p.add_argument("--lengths", default="4096,8192,32768")
    p.add_argument("--keep-ratio", type=float, default=0.25)
    p.add_argument("--n-sink", type=int, default=4)
    p.add_argument("--n-local", type=int, default=0)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--out", default="scratch/verify3_rho.json")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    dev = "cuda"
    dtype = torch.bfloat16

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    except TypeError:  # `dtype` replaced `torch_dtype` mid-2025; accept either
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is not None:
            cfg._attn_implementation = "sdpa"
    model = model.to(dev).eval()
    model.requires_grad_(False)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd, cfg = ckpt["indexer"], ckpt.get("config", {})
    scorer, press_kw = press_kwargs_from_checkpoint(sd, cfg)
    assert scorer == "scalar", scorer
    press = GQAIndexerPress(
        compression_ratio=1.0 - args.keep_ratio,
        scorer="scalar",
        gate_scale=any("gate_scale" in k for k in sd),
        n_sink=args.n_sink,
        n_local=args.n_local,
        **press_kw,
    )
    press.post_init_from_model(model)
    load_indexer_state_dict(model, sd)

    tcfg = getattr(model.config, "text_config", model.config)
    n_q, n_kv = tcfg.num_attention_heads, tcfg.num_key_value_heads
    head_dim = getattr(tcfg, "head_dim", tcfg.hidden_size // n_q)
    group = n_q // n_kv
    scale = head_dim**-0.5
    n_layers = len(get_language_model(model).layers)
    print(f"model: {n_layers} layers, H={n_q}, Hkv={n_kv}, D={head_dim}", flush=True)

    # ---- capture per-layer (q, k, v, hidden) during one dense forward -------------------
    # Deliberately NOT storing all layers at once: 36 layers of (q,k,v) at 32K is ~50 GiB.
    # One layer at a time, driven by which layer we are currently measuring.
    want_layer = [0]
    grabbed: dict = {}

    def make_impl():
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        def impl(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
            import torch.nn.functional as F

            if int(module.layer_idx) == want_layer[0]:
                grabbed["q"] = query.detach()
                grabbed["k"] = key.detach()
                grabbed["v"] = value.detach()
            out = F.scaled_dot_product_attention(
                query,
                key.repeat_interleave(group, 1),
                value.repeat_interleave(group, 1),
                is_causal=True,
                scale=scaling,
            )
            return out.transpose(1, 2).contiguous(), None

        name = "verify3_capture"
        gm = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
        ALL_ATTENTION_FUNCTIONS.register(name, impl)
        return name, gm

    impl_name, gm = make_impl()

    hidden_cache: dict = {}

    def pre_hook(module, a, kwargs):
        idx = int(getattr(module, "layer_idx", -1))
        if idx == want_layer[0]:
            hs = kwargs.get("hidden_states")
            if hs is None and a:
                hs = a[0]
            hidden_cache["h"] = hs.detach()
        return None

    handles = [
        layer.self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
        for layer in get_language_model(model).layers
    ]
    configs = [model.config] + (
        [model.config.text_config] if getattr(model.config, "text_config", None) else []
    )
    prev = [c._attn_implementation for c in configs]
    for c in configs:
        c._attn_implementation = impl_name

    # ---- data ---------------------------------------------------------------------------
    from kvpress.presses.gqa_indexer.data import (
        LongminoConfig,
        TokenizedConfig,
        build_dataloader,
        build_tokenized_dataloader,
    )

    lengths = [int(x) for x in args.lengths.split(",")]
    results: list[dict] = []

    for seq_len in lengths:
        if os.path.isdir(args.tokenized):
            loader = build_tokenized_dataloader(
                TokenizedConfig(root=args.tokenized, seq_len=seq_len, take_from="head"),
                batch_size=1,
                num_workers=0,
            )
        else:
            loader = build_dataloader(
                LongminoConfig(root=args.data_root, seq_len=seq_len, subsets=("2e16",)),
                tokenizer,
                batch_size=1,
                num_workers=0,
            )
        batches = []
        for i, b in enumerate(loader):
            batches.append(b["input_ids"][:, :seq_len].to(dev))
            if len(batches) >= args.samples:
                break
        print(f"\n=== L={seq_len}: {len(batches)} sample(s) ===", flush=True)

        topk = max(1, int(seq_len * args.keep_ratio))
        # Layers sampled rather than all 36: each layer costs a full forward pass here (the
        # capture only keeps one layer's q/k/v), so 36 layers x 3 lengths x 2 samples would be
        # 216 forwards of a 32K context. Early/mid/late is what the diagnostic needs.
        layers = sorted({0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1})

        for layer_idx in layers:
            want_layer[0] = layer_idx
            per_sample = []
            for input_ids in batches:
                grabbed.clear()
                hidden_cache.clear()
                model(input_ids=input_ids, use_cache=False)
                q, k, v = grabbed["q"], grabbed["k"], grabbed["v"]
                h = hidden_cache["h"]
                L = k.shape[2]

                # router scores -> per-key deadline -> the exact same support the press evicts to
                indexer = press.get_indexer(get_language_model(model).layers[layer_idx].self_attn)
                s = indexer.score_keys(h)[0]  # (Hkv, L) fp32
                dl = deadlines(s, topk, force_sink=args.n_sink, force_local=args.n_local)

                stats = measure(q, k, v, dl, scale=scale, group=group,
                                n_sink=args.n_sink, n_local=args.n_local)
                per_sample.append(stats)
                del q, k, v, h
                torch.cuda.empty_cache()

            agg = {kk: float(sum(s[kk] for s in per_sample) / len(per_sample)) for kk in per_sample[0]}
            agg.update(seq_len=seq_len, layer=layer_idx, topk=topk)
            results.append(agg)
            print(
                f"  L{seq_len} layer {layer_idx:2d}: rho={agg['rho_mean']:.4f} "
                f"(med {agg['rho_median']:.4f}, p90 {agg['rho_p90']:.4f})  "
                f"oE_rel_std={agg['oE_rel_std']:.4f}  "
                f"rank0_err={agg['rank0_rel_err']:.4f}  "
                f"cos(oE,mean)={agg['oE_cos_mean']:.4f}",
                flush=True,
            )

    for handle in handles:
        handle.remove()
    for c, p in zip(configs, prev):
        c._attn_implementation = p
    gm.pop(impl_name, None)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")

    print("\n==== SUMMARY ====")
    for seq_len in lengths:
        rows = [r for r in results if r["seq_len"] == seq_len]
        if rows:
            print(
                f"L={seq_len:6d}  rho mean {sum(r['rho_mean'] for r in rows)/len(rows):.4f}   "
                f"oE_rel_std {sum(r['oE_rel_std'] for r in rows)/len(rows):.4f}   "
                f"rank0_err {sum(r['rank0_rel_err'] for r in rows)/len(rows):.4f}"
            )


def measure(q, k, v, dl, *, scale, group, n_sink, n_local, q_tile=1024):
    """
    Per-row evicted mass and the exact evicted output, tiled over queries.

    Computes both branches longhand in fp32 rather than through flex_attention: this is a
    measurement, so exactness matters more than speed, and the reference is what the memory
    module will later be checked against anyway. Tiled over the query axis because the full
    (Sq, Sk) logits at 32K would be 4 GiB per head.
    """
    B, H, Sq, D = q.shape
    Hkv, Sk = k.shape[1], k.shape[2]
    dev = q.device
    kf = k[0].float()
    vf = v[0].float()
    key_idx = torch.arange(Sk, device=dev)

    rho_all = []
    oE_all = []
    mass_all = []

    for start in range(0, Sq, q_tile):
        stop = min(start + q_tile, Sq)
        qt = q[0, :, start:stop].float()  # (H, T, D)
        rows = torch.arange(start, stop, device=dev)

        # (H, T, Sk) logits, per query head against its own KV head
        logits = torch.einsum("htd,hsd->hts", qt, kf.repeat_interleave(group, 0)) * scale
        causal = key_idx.view(1, 1, -1) <= rows.view(1, -1, 1)

        limit = rows.clamp(max=Sk - 1)
        horizon = limit - n_local
        dlg = dl.repeat_interleave(group, 0).to(torch.int64)  # (H, Sk)
        sink = key_idx.view(1, 1, -1) < n_sink
        local = (key_idx.view(1, 1, -1) > limit.view(1, -1, 1) - n_local) & ~sink
        alive = horizon.view(1, -1, 1) <= dlg.unsqueeze(1)
        chosen = (~sink) & (key_idx.view(1, 1, -1) <= horizon.view(1, -1, 1)) & alive
        keep = causal & (sink | local | chosen)

        neg = torch.finfo(torch.float32).min
        lse_dense = torch.logsumexp(logits.masked_fill(~causal, neg), dim=-1)
        lse_S = torch.logsumexp(logits.masked_fill(~keep, neg), dim=-1)
        # rho = evicted share of the row's total mass. exp of a non-positive difference, so it
        # cannot exceed 1 by construction; clamped only against fp error.
        rho = (1.0 - torch.exp(lse_S - lse_dense)).clamp(0.0, 1.0)

        p_dense = torch.softmax(logits.masked_fill(~causal, neg), dim=-1)
        o_dense = torch.einsum("hts,hsd->htd", p_dense, vf.repeat_interleave(group, 0))
        p_S = torch.softmax(logits.masked_fill(~keep, neg), dim=-1)
        o_S = torch.einsum("hts,hsd->htd", p_S, vf.repeat_interleave(group, 0))

        # The exact evicted-branch output. Undefined where nothing was evicted, so those rows
        # are dropped rather than divided by ~0 -- they carry no information about o_E anyway.
        denom = rho.clamp(min=1e-4)
        oE = (o_dense - (1.0 - rho).unsqueeze(-1) * o_S) / denom.unsqueeze(-1)
        live = rho > 1e-3

        rho_all.append(rho.reshape(-1))
        oE_all.append(torch.where(live.unsqueeze(-1), oE, torch.zeros_like(oE)))
        mass_all.append(live)
        del logits, p_dense, p_S, keep, causal

    rho = torch.cat(rho_all)
    oE = torch.cat(oE_all, dim=1)  # (H, Sq, D)
    live = torch.cat(mass_all, dim=1)  # (H, Sq)

    out = {
        "rho_mean": float(rho.mean()),
        "rho_median": float(rho.median()),
        "rho_p90": float(torch.quantile(rho.float(), 0.9)),
        "rho_max": float(rho.max()),
        "live_frac": float(live.float().mean()),
    }

    # Is o_E essentially a constant per head? If so, rank-0 (one learned vector) suffices and
    # the R=16 state is over-engineering. Measured as the relative std of o_E around its
    # per-head mean, and as the residual a rank-0 fit leaves.
    rel_stds, rank0_errs, coss = [], [], []
    for h in range(oE.shape[0]):
        sel = oE[h][live[h]]
        if sel.shape[0] < 8:
            continue
        mu = sel.mean(0)
        resid = sel - mu
        rel_stds.append(float(resid.norm(dim=-1).mean() / sel.norm(dim=-1).mean().clamp(min=1e-9)))
        rank0_errs.append(float(resid.norm() / sel.norm().clamp(min=1e-9)))
        coss.append(
            float(
                torch.nn.functional.cosine_similarity(sel, mu.unsqueeze(0).expand_as(sel), dim=-1).mean()
            )
        )
    out["oE_rel_std"] = sum(rel_stds) / max(len(rel_stds), 1)
    out["rank0_rel_err"] = sum(rank0_errs) / max(len(rank0_errs), 1)
    out["oE_cos_mean"] = sum(coss) / max(len(coss), 1)
    return out


if __name__ == "__main__":
    main()
