# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Is the rank ceiling real, given that ``n/d`` can only be a CONVEX combination?

``n/d = (phi_hat . H)/(phi_hat . z)``, and ``phi, z >= 0`` by the ``abs`` in both kernels. Writing
``a_r = phi_r z_r / sum_r' phi_r' z_r'`` gives ``n/d = sum_r a_r (H_r / z_r)`` with ``a_r >= 0`` and
``sum a_r = 1``. So the expressible set is the **convex hull** of R stored vectors, not their affine
span -- which means the truncated-SVD oracle in ``diag_rank_ceiling.py`` is optimistic and cannot be
used on its own to claim the capacity exists.

The conservative counterpart: k-means with R centroids, each row assigned to its nearest centroid.
That is a *vertex* of the convex hull, so any convex combination can do at least as well -- k-means
error is therefore an upper bound on what the real parameterization can achieve. If k-means at R=16
still beats pure eviction by a wide margin, the capacity claim survives the constraint.
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

from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(args.model)
try:
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
model = model.to("cuda").eval()
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

from kvpress.presses.gqa_indexer.data import LongminoConfig, build_dataloader

loader = build_dataloader(
    LongminoConfig(
        root="/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_256k_filtered",
        seq_len=L, subsets=("2e16",), take_from="head",
    ),
    tok, batch_size=1, num_workers=0,
)
input_ids = next(iter(loader))["input_ids"][:, :L].to("cuda")

want = [int(x) for x in args.layers.split(",")]
grab: dict = {}
hidden: dict = {}


def impl(module, q, k, v, am, scaling=None, dropout=0.0, **kwargs):
    import torch.nn.functional as F

    idx = int(module.layer_idx)
    if idx in want:
        grab[idx] = (q.detach(), k.detach(), v.detach(), scaling)
    g = q.shape[1] // k.shape[1]
    o = F.scaled_dot_product_attention(
        q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1), is_causal=True, scale=scaling
    )
    return o.transpose(1, 2).contiguous(), None


def pre(module, a, kwargs):
    idx = int(getattr(module, "layer_idx", -1))
    if idx in want:
        hs = kwargs.get("hidden_states")
        if hs is None and a:
            hs = a[0]
        hidden[idx] = hs.detach()
    return None


from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

ALL_ATTENTION_FUNCTIONS.register("probe_cvx", impl)
handles = [l.self_attn.register_forward_pre_hook(pre, with_kwargs=True) for l in model.model.layers]
model.config._attn_implementation = "probe_cvx"
with torch.no_grad():
    model(input_ids=input_ids, use_cache=False)
for h in handles:
    h.remove()


def kmeans(x: torch.Tensor, r: int, iters: int = 30) -> torch.Tensor:
    """Nearest-centroid assignment for r centroids -- a convex-hull VERTEX, so conservative."""
    n = x.shape[0]
    g = torch.Generator(device=x.device).manual_seed(0)
    c = x[torch.randperm(n, generator=g, device=x.device)[:r]].clone()
    for _ in range(iters):
        assign = torch.cdist(x, c).argmin(1)
        for j in range(r):
            m = assign == j
            if bool(m.any()):
                c[j] = x[m].mean(0)
    return c[torch.cdist(x, c).argmin(1)]


STRIDE = 16
print(f"L={L} topk={TOPK}, real longmino text, every {STRIDE}th row")
print()
print(f"{'layer':>5} {'rho':>7} {'none':>8} {'svdR16':>9} {'kmeansR16':>10} {'kmeansR64':>10}")

for idx in want:
    q, k, v, scaling = grab[idx]
    h = hidden[idx]
    Hq, Hkv = q.shape[1], k.shape[1]
    group = Hq // Hkv
    scale = q.shape[-1] ** -0.5 if scaling is None else float(scaling)

    with torch.no_grad():
        scores = press.get_indexer(model.model.layers[idx].self_attn).score_keys(h)[0].float()
        dl = deadlines(scores, TOPK, force_sink=SINK, force_local=LOCAL)
        kf, vf = k[0].float(), v[0].float()
        key_idx = torch.arange(L, device=q.device)
        dlg = dl.repeat_interleave(group, 0).to(torch.int64)
        rows = torch.arange(0, L, STRIDE, device=q.device)

        oEs, oSs, lseSs, lseDs, rhos = [], [], [], [], []
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
            lse_s_ = torch.logsumexp(logits.masked_fill(~keep, neg), -1)
            rho_S = torch.exp(lse_s_ - lse_d).clamp(0, 1)
            rho_ev = 1.0 - rho_S
            o_d = torch.einsum(
                "hts,hsd->htd",
                torch.softmax(logits.masked_fill(~causal, neg), -1),
                vf.repeat_interleave(group, 0),
            )
            o_s_ = torch.einsum(
                "hts,hsd->htd",
                torch.softmax(logits.masked_fill(~keep, neg), -1),
                vf.repeat_interleave(group, 0),
            )
            oEs.append((o_d - rho_S.unsqueeze(-1) * o_s_) / rho_ev.clamp(min=1e-4).unsqueeze(-1))
            oSs.append(o_s_)
            lseSs.append(lse_s_)
            lseDs.append(lse_d)
            rhos.append(rho_ev)
            del logits, causal, keep, o_d

        oE = torch.cat(oEs, 1)
        oS = torch.cat(oSs, 1)
        lseS = torch.cat(lseSs, 1)
        lseD = torch.cat(lseDs, 1)
        rho_ev = torch.cat(rhos, 1)
        live = rho_ev > 1e-3
        D_E = torch.exp(lseD) * rho_ev
        inv = torch.exp(-lseS)
        o_dense = (oS + inv.unsqueeze(-1) * oE * D_E.unsqueeze(-1)) / (
            1.0 + (inv * D_E).unsqueeze(-1)
        )

        def err(approx):
            fused = (oS + inv.unsqueeze(-1) * approx * D_E.unsqueeze(-1)) / (
                1.0 + (inv * D_E).unsqueeze(-1)
            )
            e = (fused - o_dense).norm(dim=-1) / o_dense.norm(dim=-1).clamp(min=1e-9)
            return float(e[live].mean())

        def svd_approx(r_target):
            out = torch.empty_like(oE)
            for hh in range(oE.shape[0]):
                sel = oE[hh][live[hh]]
                mu = sel.mean(0, keepdim=True)
                _, _, Vh = torch.linalg.svd(sel - mu, full_matrices=False)
                kk = min(r_target, Vh.shape[0])
                out[hh] = (oE[hh] - mu) @ Vh[:kk].T @ Vh[:kk] + mu
            return out

        def km_approx(r_target):
            out = torch.empty_like(oE)
            for hh in range(oE.shape[0]):
                sel = oE[hh][live[hh]]
                if sel.shape[0] <= r_target:
                    out[hh] = oE[hh]
                    continue
                c = kmeans(sel, r_target)
                # map every row (not just live ones) to its nearest of the fitted centroids
                cent = torch.unique(c, dim=0)
                out[hh] = cent[torch.cdist(oE[hh], cent).argmin(1)]
            return out

        print(
            f"{idx:>5} {float(rho_ev[live].mean()):>7.4f} "
            f"{err(torch.zeros_like(oE)):>8.4f} {err(svd_approx(16)):>9.4f} "
            f"{err(km_approx(16)):>10.4f} {err(km_approx(64)):>10.4f}"
        )
        del grab[idx], hidden[idx], oE, oS
        torch.cuda.empty_cache()

print()
print("kmeans is a CONVEX-HULL VERTEX, so the real parameterization can do at least this well.")
print("If kmeansR16 << none, the capacity claim holds under the actual constraint.")
