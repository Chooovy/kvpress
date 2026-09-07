# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Bound what rank can buy, and compare the trained memory against that bound.

The memory's job is to predict ``oE*`` -- the exact evicted-branch output, one D-vector per
(row, head) -- from ``phi(q_t)`` and a rank-R state. So the best achievable rank-R prediction is the
rank-R truncated SVD of the ``oE*`` matrix itself. That is an *oracle*: it uses the true ``oE*`` to
build the approximation, so nothing trainable can beat it at that rank.

Four points, per layer, each measured as relative error of the fused output against ``o_dense``:

  none          o_S alone (pure eviction)                    -- the baseline to beat
  trained       the trained memory, as it actually runs
  oracle R=0    n/d = per-head mean of oE*, correct mass      -- the cheap ablation
  oracle R=16   n/d = rank-16 SVD of oE*, correct mass        -- the ceiling for this rank
  oracle exact  n/d = oE*, correct mass                       -- must be ~0

This separates the two hypotheses cleanly:

* If ``oracle R=16`` is much better than ``none`` but ``trained`` is not, the capacity is there and
  the training/objective failed to reach it. Fix the objective, not the rank.
* If ``oracle R=16`` is barely better than ``none``, a rank-16 summary of the evicted tail cannot
  help at this budget however it is trained. Raise R, or the arm does not work.
* If ``oracle R=16`` ~ ``oracle R=0``, rank buys nothing and R=16 is over-engineering.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kvpress.presses.gqa_indexer.memory_schedule import block_memory_states
from kvpress.presses.gqa_indexer.memory_trainer import MemoryTrainer
from kvpress.presses.gqa_indexer.press import GQAIndexerPress
from kvpress.presses.gqa_indexer.qi_flex_attention import FLEX_BLOCK, deadlines
from kvpress.presses.gqa_indexer.train import (
    load_indexer_state_dict,
    load_memory_state_dict,
    press_kwargs_from_checkpoint,
)

p = argparse.ArgumentParser()
p.add_argument("--model", default="/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B")
p.add_argument(
    "--router",
    default="/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_scalar/"
    "stage1_16k_mid256_longce/final.pt",
)
p.add_argument(
    "--memory",
    default="/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_memory/longce_8k_g7/final.pt",
)
p.add_argument("--length", type=int, default=8192)
p.add_argument("--topk", type=int, default=2048)
p.add_argument("--layers", default="0,9,18,27,35")
p.add_argument("--ranks", default="0,4,16,64")
args = p.parse_args()

SINK, LOCAL = 4, 64
L, TOPK = args.length, args.topk
RANKS = [int(x) for x in args.ranks.split(",")]

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
    n_sink=SINK, n_local=LOCAL, memory=True, memory_rank=16, **kw,
)
press.post_init_from_model(model)
load_indexer_state_dict(model, rsd)
load_memory_state_dict(
    model, torch.load(args.memory, map_location="cpu", weights_only=False)["memory"]
)

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

ALL_ATTENTION_FUNCTIONS.register("probe_svd", impl)
handles = [l.self_attn.register_forward_pre_hook(pre, with_kwargs=True) for l in model.model.layers]
model.config._attn_implementation = "probe_svd"
with torch.no_grad():
    model(input_ids=input_ids, use_cache=False)
for h in handles:
    h.remove()

# Rows are subsampled: the SVD is over (n_rows, D) per head and needs the rows in memory at once,
# and a stride keeps the estimate honest (a contiguous block would be one region of the document).
STRIDE = 8
print(f"L={L} topk={TOPK} sink={SINK} local={LOCAL}, real longmino text, every {STRIDE}th row")
print()
hdr = f"{'layer':>5} {'rho':>7} {'none':>8} {'trained':>9}"
for r in RANKS:
    hdr += f" {'oracleR%d' % r:>10}"
hdr += f" {'exact':>9}"
print(hdr)

for idx in want:
    q, k, v, scaling = grab[idx]
    h = hidden[idx]
    Hq, Hkv = q.shape[1], k.shape[1]
    group = Hq // Hkv
    scale = q.shape[-1] ** -0.5 if scaling is None else float(scaling)

    with torch.no_grad():
        scores = press.get_indexer(model.model.layers[idx].self_attn).score_keys(h)[0].float()
        dl = deadlines(scores, TOPK, force_sink=SINK, force_local=LOCAL)
        mem = press.get_memory(model.model.layers[idx].self_attn)
        H_s, z_s, W_s, cnts = block_memory_states(
            mem, k, v, dl, q_len=L, block=FLEX_BLOCK, n_local=LOCAL, scores=scores
        )
        q_kv = q.view(1, Hkv, group, L, q.shape[-1]).mean(2)
        n_mem, d_mem = MemoryTrainer._read_per_block(
            mem, q_kv, H_s, z_s, W_s, cnts, block=FLEX_BLOCK
        )

        kf, vf = k[0].float(), v[0].float()
        key_idx = torch.arange(L, device=q.device)
        dlg = dl.repeat_interleave(group, 0).to(torch.int64)
        rows = torch.arange(0, L, STRIDE, device=q.device)

        # Build oE*, o_S, lse and the memory's claim for the sampled rows, in query tiles.
        chunks = {kk: [] for kk in ("oE", "oS", "lseS", "lseD", "rho", "nmem", "dmem")}
        TILE = 512
        for s in range(0, rows.numel(), TILE):
            r = rows[s : s + TILE]
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
            chunks["oE"].append(
                (o_d - rho_S.unsqueeze(-1) * o_s_) / rho_ev.clamp(min=1e-4).unsqueeze(-1)
            )
            chunks["oS"].append(o_s_)
            chunks["lseS"].append(lse_s_)
            chunks["lseD"].append(lse_d)
            chunks["rho"].append(rho_ev)
            chunks["nmem"].append(n_mem[0, :, r].repeat_interleave(group, 0))
            chunks["dmem"].append(d_mem[0, :, r].repeat_interleave(group, 0))
            del logits, causal, keep, o_d

        oE = torch.cat(chunks["oE"], 1)  # (Hq, N, D)
        oS = torch.cat(chunks["oS"], 1)
        lseS = torch.cat(chunks["lseS"], 1)
        lseD = torch.cat(chunks["lseD"], 1)
        rho_ev = torch.cat(chunks["rho"], 1)
        nmem = torch.cat(chunks["nmem"], 1)
        dmem = torch.cat(chunks["dmem"], 1)
        del chunks

        live = rho_ev > 1e-3
        D_E_abs = torch.exp(lseD) * rho_ev  # the mass eviction removed
        inv = torch.exp(-lseS)

        def rel_err(n_term, d_term):
            fused = (oS + inv.unsqueeze(-1) * n_term) / (1.0 + (inv * d_term).unsqueeze(-1))
            o_d = oS + inv.unsqueeze(-1) * (oE * D_E_abs.unsqueeze(-1))
            o_d = o_d / (1.0 + (inv * D_E_abs).unsqueeze(-1))
            err = (fused - o_d).norm(dim=-1) / o_d.norm(dim=-1).clamp(min=1e-9)
            return float(err[live].mean())

        zero = torch.zeros_like(dmem)
        row = f"{idx:>5} {float(rho_ev[live].mean()):>7.4f} "
        row += f"{rel_err(torch.zeros_like(nmem), zero):>8.4f} "
        row += f"{rel_err(nmem, dmem):>9.4f} "

        # Oracle rank-R: truncated SVD of oE* per head, with the CORRECT mass.
        for r_target in RANKS:
            approx = torch.empty_like(oE)
            for hh in range(oE.shape[0]):
                sel = oE[hh][live[hh]]
                if sel.shape[0] < 4:
                    approx[hh] = oE[hh]
                    continue
                if r_target == 0:
                    approx[hh] = sel.mean(0, keepdim=True).expand_as(oE[hh])
                    continue
                mu = sel.mean(0, keepdim=True)
                U, S, Vh = torch.linalg.svd(sel - mu, full_matrices=False)
                k_use = min(r_target, S.numel())
                # Rank-R in the *state* sense: R basis directions plus the mean, which is what
                # phi(q).H/phi(q).z can express (a query-weighted combination of R stored rows).
                proj = (oE[hh] - mu) @ Vh[:k_use].T @ Vh[:k_use]
                approx[hh] = proj + mu
            row += f"{rel_err(approx * D_E_abs.unsqueeze(-1), D_E_abs):>10.4f} "
        row += f"{rel_err(oE * D_E_abs.unsqueeze(-1), D_E_abs):>9.2e}"
        print(row)

        del grab[idx], hidden[idx], oE, oS, nmem, dmem
        torch.cuda.empty_cache()

print()
print("none    = pure eviction, the number to beat")
print("trained = the trained memory as it runs")
print("oracleR = best possible rank-R prediction of oE* (uses the TRUE oE*, so unbeatable at that R)")
print("exact   = sanity check, must be ~0")
