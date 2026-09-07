# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Is ``oE*`` predictable from ``q`` at all, or does the memory read the wrong input?

Stage A fixed the mass (``d/D_E*`` 0.000 -> 1.05) and the error still got worse, with
``cos(n/d, oE*)`` frozen at ~0.53 from step 0 to step 60. A quantity that does not move under a
direct regression on it is not a training-rate problem -- it is a signal that the target is not a
function of the available input.

The memory's direction is ``n/d = sum_r a_r (H_r/z_r)`` with ``a_r`` determined by ``phi(q_t)``. So it
can only vary with ``t`` through ``q_t``. Three questions, in order of increasing generosity, all
measured on real text against the exact ``oE*``:

1. **How much does ``oE*`` vary at all?** If it is nearly constant across rows, everything is
   explained by a per-head vector and rank buys nothing (the R=0 ablation).
2. **Is that variation a function of ``q``?** Fit ridge regression ``q_t -> oE*_t`` per head. This is
   far more expressive than ``phi(q)`` reading a rank-R state, so it is an upper bound on any
   architecture whose only query-side input is ``q_t``.
3. **Is it a function of the row's own retained output ``o_S``?** A cheap alternative input the design
   does not currently use.

If (2) barely beats (1), then no amount of training on ``q`` recovers ``oE*``, and the k-means bound
was measuring the wrong thing -- it bounds how well R centroids can *cover* the set of ``oE*`` values,
which says nothing about whether the query can *select* the right one.
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
    if len(docs) >= 2:
        break

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

ALL_ATTENTION_FUNCTIONS.register("probe_pred", impl)
model.config._attn_implementation = "probe_pred"


@torch.no_grad()
def collect(input_ids, stride):
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
        rows = torch.arange(0, L, stride, device=DEV)

        acc = {kk: [] for kk in ("oE", "oS", "live", "q")}
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
            acc["oS"].append(o_s)
            acc["live"].append(rho_ev > 1e-3)
            acc["q"].append(qt)
            del logits, causal, keep, o_d
        out[idx] = {kk: torch.cat(vv, 1) for kk, vv in acc.items()}
        out[idx]["group"] = group
    return out


tr = collect(docs[0], stride=4)
va = collect(docs[1], stride=8)


def ridge(X, Y, lam=1e-2):
    """Least squares with a small ridge, with an intercept column."""
    X1 = torch.cat([X, torch.ones_like(X[:, :1])], 1)
    A = X1.T @ X1 + lam * torch.eye(X1.shape[1], device=X.device, dtype=X.dtype)
    return torch.linalg.solve(A, X1.T @ Y), X1.shape[1]


def apply_ridge(W, X):
    X1 = torch.cat([X, torch.ones_like(X[:, :1])], 1)
    return X1 @ W


print(f"L={L} topk={TOPK}, real longmino text. Held-out relative error of predicted oE* vs true oE*.")
print("(1.0 = as bad as predicting zero; 'const' = per-head mean of the TRAIN document)")
print()
print(f"{'layer':>5} {'const':>8} {'ridge(q)':>10} {'ridge(oS)':>11} {'ridge(q,oS)':>12}  verdict")

for idx in WANT:
    t, w = tr[idx], va[idx]
    errs = {"const": [], "q": [], "oS": [], "both": []}
    for hh in range(t["oE"].shape[0]):
        Xtr_q = t["q"][hh][t["live"][hh]]
        Ytr = t["oE"][hh][t["live"][hh]]
        Xva_q = w["q"][hh][w["live"][hh]]
        Yva = w["oE"][hh][w["live"][hh]]
        if Ytr.shape[0] < 64 or Yva.shape[0] < 16:
            continue
        Xtr_s = t["oS"][hh][t["live"][hh]]
        Xva_s = w["oS"][hh][w["live"][hh]]

        den = Yva.norm()
        mu = Ytr.mean(0, keepdim=True)
        errs["const"].append(float((Yva - mu).norm() / den))
        for name, Xtr, Xva in (
            ("q", Xtr_q, Xva_q),
            ("oS", Xtr_s, Xva_s),
            ("both", torch.cat([Xtr_q, Xtr_s], 1), torch.cat([Xva_q, Xva_s], 1)),
        ):
            W_, _ = ridge(Xtr, Ytr)
            errs[name].append(float((Yva - apply_ridge(W_, Xva)).norm() / den))

    m = {kk: sum(vv) / max(len(vv), 1) for kk, vv in errs.items()}
    gain = m["const"] / max(m["q"], 1e-9)
    verdict = "q PREDICTS oE*" if gain > 1.15 else "q does NOT predict oE*"
    print(
        f"{idx:>5} {m['const']:>8.4f} {m['q']:>10.4f} {m['oS']:>11.4f} {m['both']:>12.4f}  {verdict}"
    )

print()
print("ridge(q) is far more expressive than phi(q) reading a rank-R state, so it UPPER-BOUNDS")
print("any design whose only query-side input is q_t. If it does not beat 'const', the memory's")
print("direction cannot be learned from q at all -- and the k-means bound was measuring coverage")
print("of the oE* set, not the query's ability to select within it.")
