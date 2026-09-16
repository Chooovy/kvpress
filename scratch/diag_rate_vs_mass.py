# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Can a FoX-style per-head forget rate REPLACE the mass allocator as a head-budget mechanism?

The proposal
------------
``head_budget.py`` allocates per-head key counts by water-filling on retained softmax mass,
measured per document at prefill. That works (+6.55 RULER 8K / +12.89 16K over uniform) but it
needs a measurement pass and a bisection. A FoX-style scalar forget gate ``f^h_l`` per KV head is
far simpler: a head that forgets fast keeps few keys, a head that forgets slowly keeps many, and
the split falls out of one learned scalar per head with no measurement and no allocator.

The question is whether that split can be the RIGHT one, and there are two independent ways it can
fail. This script measures both.

Failure mode 1: gauge invariance
--------------------------------
The trained gate is ``score - lse + log B`` on history, which is bitwise invariant to a
per-(layer, head) additive constant (``head_budget.py``, verified). A FoX increment enters the
gate as ``-c^h_j``, and ``c^h_j`` is a *prefix sum* -- so a per-head change to the forget rate is
NOT a constant shift, it is a ramp, and it survives the normalizer. Good. But the budget is then
whatever top-k the ramp happens to produce, and nothing ties that to the head's actual demand.
This script asks: does the budget a forget rate induces correlate with the mass the head needs?

Failure mode 2: a rate is a MODEL constant, demand is a DOCUMENT property
-------------------------------------------------------------------------
A learned per-head rate is fixed after training. ``diag_head_window.py`` measured cross-document
Spearman 0.542 and relative spread 0.587 for per-head window widths, which is *below* the 0.59 that
already condemned the static budget table (81.93 vs uniform 82.18). So the honest hypothesis is
that any static per-head quantity -- rate, window, or budget -- reproduces the static-table failure.

The decisive comparison is therefore not "rate vs uniform" but **"best possible static rate" vs
"per-document mass allocation"**, with the static rate fitted with ORACLE knowledge of the
documents. That is a ceiling on the whole idea, so if it loses, no training procedure rescues it.

Reported
--------
* ``rho`` (retained mass) under four allocators at exactly matched total budget:
  ``uniform``, ``mass`` (today's per-document water-filling), ``rate_oracle`` (the single per-head
  rate vector fitted to maximise worst-head mass across ALL documents jointly), and
  ``rate_perdoc`` (a rate refitted per document -- an upper bound that a static rate cannot reach).
* ``min_h rho_h`` is the headline: allocation helps by lifting the floor, so the worst head is the
  number that matters, not the mean.
* ``shuffle`` control: the oracle rate vector applied to the wrong head permutation.

A forget rate cannot express a per-head COUNT directly -- it produces a ramp, and the count is
whatever survives top-k. So ``rate_*`` arms here convert a rate vector into counts by ranking
``score - lambda_h * (L - j)`` per head and taking the same total budget, i.e. exactly what the
mechanism would do at inference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kvpress.presses.gqa_indexer.press import get_language_model  # noqa: E402

MODELS = "/apdcephfs_gy8/share_303843174/guhao/models"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=f"{MODELS}/Qwen3-8B")
    p.add_argument("--tokenized", default=f"{MODELS}/../datasets/longmino_tokenized_64k")
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--n-sink", type=int, default=4)
    p.add_argument("--n-local", type=int, default=128)
    p.add_argument("--layers", type=int, nargs="+", default=[0, 4, 7, 14, 21, 28, 35])
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", default="scratch/diag_rate_vs_mass.json")
    return p.parse_args()


@torch.no_grad()
def capture(model, layers, ids, layer_ids, n_kv, group, head_dim):
    """Per-(layer, KV head) attention at the last row, plus the keys' positions. {li: (Hkv, Sk)}"""
    cap: dict[int, torch.Tensor] = {}
    scale = head_dim**-0.5

    def mk(li):
        def hook(mod, args_, kwargs_, out):
            hs = kwargs_.get("hidden_states", args_[0] if args_ else None)
            pe = kwargs_.get("position_embeddings")
            if hs is None or pe is None:
                return
            cos, sin = pe
            b, s, _ = hs.shape
            q = mod.q_proj(hs).view(b, s, -1, head_dim).transpose(1, 2)
            k = mod.k_proj(hs).view(b, s, -1, head_dim).transpose(1, 2)
            if hasattr(mod, "q_norm"):
                q, k = mod.q_norm(q), mod.k_norm(k)
            from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            qr = q[0, :, -1, :].float()
            kq = k[0].float().repeat_interleave(group, 0)
            p = torch.softmax(torch.einsum("hd,hsd->hs", qr, kq) * scale, dim=-1)
            cap[li] = p.view(n_kv, group, -1).mean(1).double().cpu()

        return hook

    handles = [layers[li].self_attn.register_forward_hook(mk(li), with_kwargs=True) for li in layer_ids]
    model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()
    return cap


def pool_and_pin(p, n_sink, n_local, L):
    """Split mass into (pinned, pool_mass_vector, pool_index)."""
    idx = torch.arange(L)
    sink = idx < n_sink
    local = idx > L - 1 - n_local
    pinned = sink | local
    return p[:, pinned].sum(-1), p[:, ~pinned], torch.nonzero(~pinned).squeeze(-1)


def rho_from_counts(pool_mass, order, counts, pinned_mass):
    """Retained mass per head given a per-head count and a per-head ranking of the pool."""
    H = pool_mass.shape[0]
    out = torch.zeros(H, dtype=torch.double)
    for h in range(H):
        c = int(counts[h])
        out[h] = pinned_mass[h] + (pool_mass[h, order[h, :c]].sum() if c > 0 else 0.0)
    return out


def mass_allocate(cum, total):
    """Water-fill a common mass target with the total conserved exactly (mirrors head_budget.py)."""
    H, N = cum.shape
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        need = torch.stack([torch.searchsorted(cum[h].contiguous(), torch.tensor(mid)) for h in range(H)])
        if int(need.clamp(max=N).sum()) > total:
            hi = mid
        else:
            lo = mid
    counts = torch.stack(
        [torch.searchsorted(cum[h].contiguous(), torch.tensor(lo)) for h in range(H)]
    ).clamp(max=N)
    # settle the residual by marginal value, conserving the total exactly
    while int(counts.sum()) < total:
        marg = torch.tensor(
            [cum[h, min(int(counts[h]), N - 1)] - cum[h, max(int(counts[h]) - 1, 0)] for h in range(H)]
        )
        marg[counts >= N] = -1.0
        counts[int(marg.argmax())] += 1
    while int(counts.sum()) > total:
        marg = torch.tensor(
            [cum[h, max(int(counts[h]) - 1, 0)] - cum[h, max(int(counts[h]) - 2, 0)] for h in range(H)]
        )
        marg[counts <= 0] = 1e9
        counts[int(marg.argmin())] -= 1
    return counts


def counts_from_rates(score, pos, lam, total):
    """
    Convert a per-head forget RATE into per-head counts, the way the mechanism would at inference.

    Every head ranks its pool by ``score - lam_h * age`` and the global top-``total`` over the
    union decides who gets slots. That union step is what makes a rate an allocator at all: a
    steeper head loses ties to a shallower one, so the counts come out uneven.
    """
    H, N = score.shape
    g = score - lam.view(-1, 1) * pos.view(1, -1)
    flat = g.reshape(-1)
    keep = torch.topk(flat, min(total, flat.numel())).indices
    counts = torch.bincount(keep // N, minlength=H)
    return counts


def main():
    args = parse_args()
    dev = torch.device(args.device)
    from transformers import AutoModelForCausalLM

    from kvpress.presses.gqa_indexer.data import TokenizedConfig, build_tokenized_dataloader

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to(dev)
        .eval()
    )
    cfg = model.config
    n_kv = cfg.num_key_value_heads
    group = cfg.num_attention_heads // n_kv
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    layers = get_language_model(model).layers
    L = args.seq_len

    loader = build_tokenized_dataloader(
        TokenizedConfig(root=args.tokenized, seq_len=L, take_from="head"),
        batch_size=1,
        num_workers=0,
    )
    docs = []
    for si, b in enumerate(loader):
        if si >= args.samples:
            break
        ids = b["input_ids"][:, :L].to(dev)
        docs.append(capture(model, layers, ids, args.layers, n_kv, group, head_dim))

    take = args.topk - args.n_sink - args.n_local
    total = take * n_kv
    # Rate grid, in units of "nats of decay across the whole 8K context". 0 = no decay.
    grid = torch.tensor([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=torch.double)

    res = {}
    print(f"\nmodel={os.path.basename(args.model)} L={L} topk={args.topk} "
          f"total_pool_budget={total} docs={len(docs)} layers={args.layers}")
    print("\nretained mass (min over heads / mean over heads), matched total budget")
    print(f"{'layer':>5} {'uniform':>15} {'mass(perdoc)':>15} {'rate_oracle':>15} "
          f"{'rate_perdoc':>15} {'rate_shuf':>15}")
    print("-" * 88)

    agg = {k: [] for k in ("uniform", "mass", "rate_oracle", "rate_perdoc", "shuffle")}
    for li in args.layers:
        # Per document: pool mass, the router-free ranking (by true mass, an oracle ordering that
        # is identical across arms so only the ALLOCATION differs), and the cumulative curve.
        per_doc = []
        for d in docs:
            p = d[li]
            pinned, pool, pidx = pool_and_pin(p, args.n_sink, args.n_local, L)
            order = torch.argsort(pool, dim=-1, descending=True)
            cum = pool.gather(-1, order).cumsum(-1) + pinned.unsqueeze(-1)
            # age of each pool key, normalised to [0,1] across the context
            age = (L - 1 - pidx).double() / L
            per_doc.append({"pinned": pinned, "pool": pool, "order": order, "cum": cum,
                            "score": pool.log().clamp_min(-40.0), "age": age})

        # --- uniform ---
        u = torch.full((n_kv,), take, dtype=torch.long)
        r_u = torch.stack([rho_from_counts(x["pool"], x["order"], u, x["pinned"]) for x in per_doc])

        # --- mass, per document (today's allocator) ---
        r_m = []
        for x in per_doc:
            c = mass_allocate(x["cum"], total)
            r_m.append(rho_from_counts(x["pool"], x["order"], c, x["pinned"]))
        r_m = torch.stack(r_m)

        # --- rate arms: search the grid for the rate vector maximising the WORST head's mass ---
        def eval_rates(lam, xs):
            out = []
            for x in xs:
                c = counts_from_rates(x["score"], x["age"], lam, total)
                out.append(rho_from_counts(x["pool"], x["order"], c, x["pinned"]))
            return torch.stack(out)

        # oracle STATIC rate: one vector for all documents, coordinate ascent on min-head mass
        lam = torch.zeros(n_kv, dtype=torch.double)
        best = eval_rates(lam, per_doc).min(-1).values.mean()
        for _ in range(3):
            for h in range(n_kv):
                for v in grid:
                    trial = lam.clone()
                    trial[h] = v
                    sc = eval_rates(trial, per_doc).min(-1).values.mean()
                    if sc > best:
                        best, lam = sc, trial
        r_ro = eval_rates(lam, per_doc)

        # per-document rate (an upper bound a static rate cannot reach)
        r_rp = []
        for x in per_doc:
            l2 = torch.zeros(n_kv, dtype=torch.double)
            b2 = eval_rates(l2, [x]).min(-1).values.mean()
            for _ in range(3):
                for h in range(n_kv):
                    for v in grid:
                        t2 = l2.clone()
                        t2[h] = v
                        s2 = eval_rates(t2, [x]).min(-1).values.mean()
                        if s2 > b2:
                            b2, l2 = s2, t2
            r_rp.append(eval_rates(l2, [x])[0])
        r_rp = torch.stack(r_rp)

        g = torch.Generator().manual_seed(0)
        r_sh = eval_rates(lam[torch.randperm(n_kv, generator=g)], per_doc)

        row = {}
        for name, r in (("uniform", r_u), ("mass", r_m), ("rate_oracle", r_ro),
                        ("rate_perdoc", r_rp), ("shuffle", r_sh)):
            row[name] = {"min": float(r.min(-1).values.mean()), "mean": float(r.mean())}
            agg[name].append(row[name]["min"])
        res[str(li)] = {"rates": lam.tolist(), **row}
        print(f"{li:>5} " + " ".join(
            f"{row[n]['min']:>7.3f}/{row[n]['mean']:>7.3f}"
            for n in ("uniform", "mass", "rate_oracle", "rate_perdoc", "shuffle")))

    print("-" * 88)
    print(f"{'MEAN':>5} " + " ".join(
        f"{sum(agg[n]) / len(agg[n]):>7.3f}{'':>8}"
        for n in ("uniform", "mass", "rate_oracle", "rate_perdoc", "shuffle")))
    print("\n(min over heads is the number that matters: allocation lifts the floor)")
    print(f"  mass       - uniform = {sum(agg['mass']) / len(agg['mass']) - sum(agg['uniform']) / len(agg['uniform']):+.4f}")
    print(f"  rate_oracle- uniform = {sum(agg['rate_oracle']) / len(agg['rate_oracle']) - sum(agg['uniform']) / len(agg['uniform']):+.4f}")
    print(f"  rate_perdoc- uniform = {sum(agg['rate_perdoc']) / len(agg['rate_perdoc']) - sum(agg['uniform']) / len(agg['uniform']):+.4f}")
    print(f"  rate_oracle- mass    = {sum(agg['rate_oracle']) / len(agg['rate_oracle']) - sum(agg['mass']) / len(agg['mass']):+.4f}")

    with open(args.out, "w") as fh:
        json.dump({"config": vars(args), "per_layer": res,
                   "agg_min": {k: sum(v) / len(v) for k, v in agg.items()}}, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
