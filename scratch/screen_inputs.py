# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Input screening: is ANY inference-time input predictive of the evicted branch?

The memory arm is blocked on addressability, not capacity: k-means at R=16 covers the ``oE*`` set
2-6x better than pure eviction, but ridge on ``q_t`` ties a per-head constant, so the query cannot
select within that cover. Before any further work on the arm, this asks whether some *other*
available input can.

Three corrections to the earlier probe, each of which could have produced a false negative:

1. **Nonlinear predictors.** ``phi`` is a two-layer MLP; ridge is linear. "Ridge ties the constant"
   therefore does not by itself rule out a learnable nonlinear map. kNN (nonparametric, arbitrary
   nonlinearity given data) and a small MLP are included, and kNN is the headline number.
2. **The mass is also a prediction.** ``D_E*`` needs ``lse_dense``, which inference does not have --
   it has to come out of ``d = gamma |E| <phi_hat, z_hat>``, i.e. from ``q`` again. So ``log D_E*``
   is screened as a target in its own right. It may well be the easier half: it is a scalar, and it
   plausibly correlates with position and with ``lse_S``.
3. **GQA group spread.** ``oE*`` is per *query* head (32), while the memory holds one state per *KV*
   head (8) shared by the group's 4. If ``oE*`` varies within a group, a per-KV-head memory cannot
   match it however well it is trained. Reported as ``group_mean``: the error of predicting each
   query head by its own group's mean, using the true values. That is a **structural ceiling** --
   nothing below it is reachable without one state per query head.

Every number is held-out (train documents disjoint from eval documents) and normalised so that
**1.0 = as bad as predicting zero** and the ``const`` column is the honest baseline to beat.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kvpress.presses.gqa_indexer.press import GQAIndexerPress
from kvpress.presses.gqa_indexer.qi_flex_attention import deadlines
from kvpress.presses.gqa_indexer.train import (
    load_indexer_state_dict,
    press_kwargs_from_checkpoint,
)

p = argparse.ArgumentParser()
p.add_argument("--model", default="/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B")
p.add_argument(
    "--router",
    default="/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_scalar/"
    "stage1_16k_mid256_longce/final.pt",
)
p.add_argument("--length", type=int, default=8192)
p.add_argument("--topk", type=int, default=2048)
p.add_argument("--layers", default="0,18,35")
p.add_argument("--train-docs", type=int, default=4)
p.add_argument("--val-docs", type=int, default=2)
p.add_argument("--stride", type=int, default=8)
p.add_argument("--mlp-heads", type=int, default=2, help="heads to also fit an MLP on (slow)")
args = p.parse_args()

SINK, LOCAL = 4, 64
L, TOPK = args.length, args.topk
DEV = "cuda"

from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(args.model)
try:
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
model = model.to(DEV).eval()
model.requires_grad_(False)

rck = torch.load(args.router, map_location="cpu", weights_only=False)
rsd, rcfg = rck["indexer"], rck.get("config", {})
_, kw = press_kwargs_from_checkpoint(rsd, rcfg)
press = GQAIndexerPress(
    compression_ratio=0.75, scorer="scalar", gate_scale=any("gate_scale" in k for k in rsd),
    n_sink=SINK, n_local=LOCAL, **kw,
)
press.post_init_from_model(model)
load_indexer_state_dict(model, rsd)

WANT = [int(x) for x in args.layers.split(",")]
from kvpress.presses.gqa_indexer.data import LongminoConfig, build_dataloader

loader = build_dataloader(
    LongminoConfig(
        root="/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_256k_filtered",
        seq_len=L, subsets=("2e16",), take_from="head",
    ),
    tok, batch_size=1, num_workers=0,
)
docs = []
for b in loader:
    docs.append(b["input_ids"][:, :L].to(DEV))
    if len(docs) >= args.train_docs + args.val_docs:
        break
print(f"{len(docs)} documents: {args.train_docs} train / {len(docs) - args.train_docs} val")

grab: dict = {}
hidden: dict = {}


def impl(module, q, k, v, am, scaling=None, dropout=0.0, **kwargs):
    import torch.nn.functional as F

    idx = int(module.layer_idx)
    if idx in WANT:
        grab[idx] = (q.detach(), k.detach(), v.detach(), scaling)
    g = q.shape[1] // k.shape[1]
    o = F.scaled_dot_product_attention(
        q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1), is_causal=True, scale=scaling
    )
    return o.transpose(1, 2).contiguous(), None


def pre(module, a, kwargs):
    idx = int(getattr(module, "layer_idx", -1))
    if idx in WANT:
        hs = kwargs.get("hidden_states")
        if hs is None and a:
            hs = a[0]
        hidden[idx] = hs.detach()
    return None


from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

ALL_ATTENTION_FUNCTIONS.register("probe_screen", impl)
model.config._attn_implementation = "probe_screen"


@torch.no_grad()
def collect(input_ids):
    """Per layer: the target ``oE*``/``log D_E*`` and every candidate input, for sampled rows."""
    grab.clear()
    hidden.clear()
    handles = [
        l.self_attn.register_forward_pre_hook(pre, with_kwargs=True) for l in model.model.layers
    ]
    model(input_ids=input_ids, use_cache=False)
    for h in handles:
        h.remove()

    out = {}
    for idx in WANT:
        q, k, v, scaling = grab[idx]
        h = hidden[idx]
        Hq, Hkv = q.shape[1], k.shape[1]
        group = Hq // Hkv
        scale = q.shape[-1] ** -0.5 if scaling is None else float(scaling)
        scores = press.get_indexer(model.model.layers[idx].self_attn).score_keys(h)[0].float()
        dl = deadlines(scores, TOPK, force_sink=SINK, force_local=LOCAL)
        kf, vf = k[0].float(), v[0].float()
        key_idx = torch.arange(L, device=DEV)
        dlg = dl.repeat_interleave(group, 0).to(torch.int64)
        rows = torch.arange(0, L, args.stride, device=DEV)

        acc = {kk: [] for kk in ("oE", "logDE", "live", "q", "oS", "lseS", "sinkattn", "h")}
        for s in range(0, rows.numel(), 512):
            r = rows[s : s + 512]
            qt = q[0, :, r].float()
            logits = torch.einsum("htd,hsd->hts", qt, kf.repeat_interleave(group, 0)) * scale
            causal = key_idx.view(1, 1, -1) <= r.view(1, -1, 1)
            limit = r.clamp(max=L - 1)
            horizon = limit - LOCAL
            sink = key_idx.view(1, 1, -1) < SINK
            local = (key_idx.view(1, 1, -1) > limit.view(1, -1, 1) - LOCAL) & ~sink
            alive = horizon.view(1, -1, 1) <= dlg.unsqueeze(1)
            chosen = (~sink) & (key_idx.view(1, 1, -1) <= horizon.view(1, -1, 1)) & alive
            keep = causal & (sink | local | chosen)
            neg = torch.finfo(torch.float32).min
            lse_d = torch.logsumexp(logits.masked_fill(~causal, neg), -1)
            lse_s = torch.logsumexp(logits.masked_fill(~keep, neg), -1)
            rho_S = torch.exp(lse_s - lse_d).clamp(0, 1)
            rho_ev = 1.0 - rho_S
            o_d = torch.einsum(
                "hts,hsd->htd",
                torch.softmax(logits.masked_fill(~causal, neg), -1),
                vf.repeat_interleave(group, 0),
            )
            o_s = torch.einsum(
                "hts,hsd->htd",
                torch.softmax(logits.masked_fill(~keep, neg), -1),
                vf.repeat_interleave(group, 0),
            )
            acc["oE"].append(
                (o_d - rho_S.unsqueeze(-1) * o_s) / rho_ev.clamp(min=1e-4).unsqueeze(-1)
            )
            # log of the mass eviction removed, which inference must also predict.
            acc["logDE"].append((torch.exp(lse_d) - torch.exp(lse_s)).clamp(min=1e-20).log())
            acc["live"].append(rho_ev > 1e-3)
            acc["q"].append(qt)
            acc["oS"].append(o_s)
            acc["lseS"].append(lse_s)
            # attention the row puts on the sink keys: a cheap summary of "what kind of row is this"
            acc["sinkattn"].append(
                torch.softmax(logits.masked_fill(~keep, neg), -1)[..., :SINK]
            )
            # The layer's own hidden state for these rows -- what the router scores from, and a
            # strictly richer input than q (q is a linear projection of it, post-RoPE and normed).
            acc["h"].append(h[0, r].float().unsqueeze(0).expand(Hq, -1, -1))
            del logits, causal, keep, o_d
        d = {kk: torch.cat(vv, 1) for kk, vv in acc.items()}
        # scalar, per-row position feature (same for every head)
        pos = (rows.float() / L).view(1, -1, 1).expand(Hq, -1, 1)
        d["pos"] = pos
        d["group"] = group
        out[idx] = d
    return out


train = [collect(d) for d in docs[: args.train_docs]]
val = [collect(d) for d in docs[args.train_docs :]]


def stack(sets, idx, key, head):
    """Concatenate one head's live rows across documents."""
    return torch.cat([s[idx][key][head][s[idx]["live"][head]] for s in sets], 0)


def ridge_fit(X, Y, lam):
    X1 = torch.cat([X, torch.ones_like(X[:, :1])], 1)
    A = X1.T @ X1 + lam * torch.eye(X1.shape[1], device=X.device, dtype=X.dtype)
    return torch.linalg.solve(A, X1.T @ Y)


def ridge_pred(W, X):
    return torch.cat([X, torch.ones_like(X[:, :1])], 1) @ W


def knn_pred(Xtr, Ytr, Xva, k=32):
    """Nonparametric: captures arbitrary nonlinearity in the input, given enough data."""
    # standardise so the metric is not dominated by whichever feature has the largest scale
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp(min=1e-6)
    a, b = (Xtr - mu) / sd, (Xva - mu) / sd
    out = torch.empty(Xva.shape[0], Ytr.shape[1], device=Xva.device)
    for s in range(0, b.shape[0], 256):
        dist = torch.cdist(b[s : s + 256], a)
        nn = dist.topk(min(k, a.shape[0]), largest=False).indices
        out[s : s + 256] = Ytr[nn].mean(1)
    return out


def mlp_pred(Xtr, Ytr, Xva, steps=600, width=256):
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp(min=1e-6)
    a, b = (Xtr - mu) / sd, (Xva - mu) / sd
    ym, ys = Ytr.mean(0, keepdim=True), Ytr.std(0, keepdim=True).clamp(min=1e-6)
    t = (Ytr - ym) / ys
    net = torch.nn.Sequential(
        torch.nn.Linear(a.shape[1], width), torch.nn.GELU(),
        torch.nn.Linear(width, width), torch.nn.GELU(),
        torch.nn.Linear(width, Ytr.shape[1]),
    ).to(a.device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(steps):
        sel = torch.randint(0, a.shape[0], (min(512, a.shape[0]),), device=a.device)
        loss = (net(a[sel]) - t[sel]).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return net(b) * ys + ym


CANDS = {
    "q": ("q",),
    "hidden": ("h",),
    "o_S": ("oS",),
    "lse_S": ("lseS",),
    "pos": ("pos",),
    "sink_attn": ("sinkattn",),
    "q+lse+pos": ("q", "lseS", "pos"),
    "all": ("q", "oS", "lseS", "pos", "sinkattn"),
}


def build(sets, idx, keys, head):
    cols = []
    for kk in keys:
        t = stack(sets, idx, kk, head)
        cols.append(t.unsqueeze(-1) if t.dim() == 1 else t)
    return torch.cat(cols, 1)


print()
print("=" * 96)
print("TARGET 1: oE*  (the evicted-branch DIRECTION)")
print("held-out relative error; 1.0 = as bad as predicting zero; beat 'const' to be useful")
print("=" * 96)
print(f"{'layer':>5} {'const':>7} {'group_mean':>11} | " + " ".join(f"{c:>10}" for c in CANDS))

group_ceiling = {}
for idx in WANT:
    Hq = train[0][idx]["oE"].shape[0]
    group = train[0][idx]["group"]
    res = {c: [] for c in CANDS}
    const_e, grp_e = [], []
    for hh in range(Hq):
        Ytr = stack(train, idx, "oE", hh)
        Yva = stack(val, idx, "oE", hh)
        if Ytr.shape[0] < 128 or Yva.shape[0] < 32:
            continue
        den = Yva.norm()
        const_e.append(float((Yva - Ytr.mean(0, keepdim=True)).norm() / den))
        for cname, keys in CANDS.items():
            Xtr = build(train, idx, keys, hh)
            Xva = build(val, idx, keys, hh)
            pred = knn_pred(Xtr, Ytr, Xva)
            res[cname].append(float((Yva - pred).norm() / den))

    # Structural ceiling: predict each query head by its GROUP's mean, using true values.
    for g0 in range(0, Hq, group):
        members = list(range(g0, min(g0 + group, Hq)))
        mats = [stack(val, idx, "oE", hh) for hh in members]
        n = min(m.shape[0] for m in mats)
        if n < 32:
            continue
        gm = torch.stack([m[:n] for m in mats], 0).mean(0)
        for m in mats:
            grp_e.append(float((m[:n] - gm).norm() / m[:n].norm()))

    group_ceiling[idx] = sum(grp_e) / max(len(grp_e), 1)
    row = f"{idx:>5} {sum(const_e)/len(const_e):>7.4f} {group_ceiling[idx]:>11.4f} | "
    row += " ".join(f"{sum(v)/max(len(v),1):>10.4f}" for v in res.values())
    print(row)

print()
print("=" * 96)
print("TARGET 2: log D_E*  (the evicted MASS -- inference has no lse_dense, so this is predicted too)")
print("=" * 96)
print(f"{'layer':>5} {'const':>7} | " + " ".join(f"{c:>10}" for c in CANDS))
for idx in WANT:
    Hq = train[0][idx]["oE"].shape[0]
    res = {c: [] for c in CANDS}
    const_e = []
    for hh in range(Hq):
        Ytr = stack(train, idx, "logDE", hh).unsqueeze(-1)
        Yva = stack(val, idx, "logDE", hh).unsqueeze(-1)
        if Ytr.shape[0] < 128 or Yva.shape[0] < 32:
            continue
        mu = Ytr.mean(0, keepdim=True)
        den = (Yva - mu).norm()  # relative to the constant predictor's own error
        const_e.append(1.0)
        for cname, keys in CANDS.items():
            Xtr = build(train, idx, keys, hh)
            Xva = build(val, idx, keys, hh)
            pred = knn_pred(Xtr, Ytr, Xva)
            res[cname].append(float((Yva - pred).norm() / den.clamp(min=1e-9)))
    row = f"{idx:>5} {1.0:>7.4f} | "
    row += " ".join(f"{sum(v)/max(len(v),1):>10.4f}" for v in res.values())
    print(row)

print()
print("=" * 96)
print(f"MLP confirmation on {args.mlp_heads} head(s) per layer, target oE*, input q")
print("(kNN and MLP disagreeing would mean the nonparametric fit was data-limited, not that q is uninformative)")
print("=" * 96)
for idx in WANT:
    for hh in range(min(args.mlp_heads, train[0][idx]["oE"].shape[0])):
        Ytr = stack(train, idx, "oE", hh)
        Yva = stack(val, idx, "oE", hh)
        Xtr = build(train, idx, ("q",), hh)
        Xva = build(val, idx, ("q",), hh)
        den = Yva.norm()
        c = float((Yva - Ytr.mean(0, keepdim=True)).norm() / den)
        r = float((Yva - ridge_pred(ridge_fit(Xtr, Ytr, 1e-2), Xva)).norm() / den)
        kn = float((Yva - knn_pred(Xtr, Ytr, Xva)).norm() / den)
        ml = float((Yva - mlp_pred(Xtr, Ytr, Xva)).norm() / den)
        print(f"  layer {idx:2d} head {hh}: n_train {Ytr.shape[0]:5d}  "
              f"const {c:.4f}  ridge {r:.4f}  kNN {kn:.4f}  MLP {ml:.4f}")

print()
print("READING THIS")
print("  Any column << const  -> that input predicts the target; the arm has a way forward.")
print("  All columns ~ const  -> no available input addresses the evicted branch.")
print("  group_mean is a STRUCTURAL floor: a per-KV-head state cannot beat it, however trained.")
print("  Target 2 is normalised so const = 1.0 by construction.")
