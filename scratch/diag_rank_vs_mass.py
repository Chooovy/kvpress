# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Is the trained memory wrong in DIRECTION (rank too small) or in VOLUME (gamma overshot)?

Both have exact ground truth, so this does not need to be guessed at. From the dense and sparse
log-normalizers:

    rho_S = exp(lse_S - lse_dense)          retained share of the row's softmax mass
    D_E*  = exp(lse_dense) - exp(lse_S)     the mass eviction actually removed  <- what d SHOULD be
    oE*   = (o_dense - rho_S o_S)/(1-rho_S) the exact evicted-branch output     <- what n/d SHOULD be

The memory supplies ``d`` (its claimed mass) and ``n/d`` (its direction). So:

* ``d / D_E*``            -- 1.0 means correctly calibrated; >> 1 means it is claiming mass that
                             eviction never removed, which displaces retained attention.
* ``cos(n/d, oE*)`` and
  ``||n/d - oE*||/||oE*||`` -- how good a rank-R summary of the evicted tail actually is.

And the decomposition that decides the next step: rebuild the fused output three ways and measure
each against ``o_dense``.

  1. as trained                        (trained mass, trained direction)
  2. mass corrected to D_E*            (trained direction only)
  3. both corrected                    (must be ~0, i.e. the identity is right)

If (2) is much better than (1), the problem is volume and capping the mass share is the fix. If (2)
is no better, the problem is direction and rank/objective is the fix.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kvpress.presses.gqa_indexer.memory_schedule import block_memory_states
from kvpress.presses.gqa_indexer.memory_trainer import MemoryTrainer
from kvpress.presses.gqa_indexer.press import GQAIndexerPress
from kvpress.presses.gqa_indexer.qi_flex_attention import (
    FLEX_BLOCK,
    _flex,
    deadlines,
    qi_block_mask,
)
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
    n_sink=SINK, n_local=LOCAL, memory=True, memory_rank=16, **kw,
)
press.post_init_from_model(model)
load_indexer_state_dict(model, rsd)
load_memory_state_dict(model, torch.load(args.memory, map_location="cpu", weights_only=False)["memory"])
gam = torch.stack([l.self_attn.kv_memory.log_gamma for l in model.model.layers]).exp()
print(f"memory: {args.memory}")
print(f"  gamma mean {float(gam.mean()):.3e}  max {float(gam.max()):.3e}")
print()

# Real text, not random tokens: the evicted tail's structure is the whole subject here.
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

ALL_ATTENTION_FUNCTIONS.register("probe_rank", impl)
handles = [l.self_attn.register_forward_pre_hook(pre, with_kwargs=True) for l in model.model.layers]
model.config._attn_implementation = "probe_rank"
with torch.no_grad():
    model(input_ids=input_ids, use_cache=False)
for h in handles:
    h.remove()

trainer = MemoryTrainer(press=press, keep_ratio=0.25, force_sink=SINK, force_local=LOCAL)
Q_TILE = 512

print(f"L={L}, topk={TOPK}, sink={SINK}, local={LOCAL}, real longmino text")
print()
print(f"{'layer':>5} {'rho_true':>9} {'d/D_E*':>10} {'cos(n/d,oE*)':>13} {'dir_relerr':>11} "
      f"{'err:trained':>12} {'err:massfix':>12} {'err:both':>10}")

for idx in want:
    q, k, v, scaling = grab[idx]
    h = hidden[idx]
    Hq, Hkv = q.shape[1], k.shape[1]
    group = Hq // Hkv
    scale = q.shape[-1] ** -0.5 if scaling is None else float(scaling)

    with torch.no_grad():
        scores = press.get_indexer(model.model.layers[idx].self_attn).score_keys(h)[0].float()
        dl = deadlines(scores, TOPK, force_sink=SINK, force_local=LOCAL)

        # The memory's own (n, d), exactly as the fixed inference path builds them.
        H_s, z_s, W_s, cnts = block_memory_states(
            press.get_memory(model.model.layers[idx].self_attn), k, v, dl,
            q_len=L, block=FLEX_BLOCK, n_local=LOCAL, scores=scores,
        )
        q_kv = q.view(1, Hkv, group, L, q.shape[-1]).mean(2)
        n_mem, d_mem = MemoryTrainer._read_per_block(
            press.get_memory(model.model.layers[idx].self_attn),
            q_kv, H_s, z_s, W_s, cnts, block=FLEX_BLOCK,
        )

        acc = {kk: [] for kk in ("rho", "mass", "cos", "dir", "e_tr", "e_mf", "e_both")}
        kf, vf = k[0].float(), v[0].float()
        key_idx = torch.arange(L, device=q.device)
        dlg = dl.repeat_interleave(group, 0).to(torch.int64)

        for start in range(0, L, Q_TILE):
            stop = min(start + Q_TILE, L)
            qt = q[0, :, start:stop].float()
            rows = torch.arange(start, stop, device=q.device)
            logits = torch.einsum("htd,hsd->hts", qt, kf.repeat_interleave(group, 0)) * scale
            causal = key_idx.view(1, 1, -1) <= rows.view(1, -1, 1)
            limit = rows.clamp(max=L - 1)
            horizon = limit - LOCAL
            sink = key_idx.view(1, 1, -1) < SINK
            local = (key_idx.view(1, 1, -1) > limit.view(1, -1, 1) - LOCAL) & ~sink
            alive = horizon.view(1, -1, 1) <= dlg.unsqueeze(1)
            chosen = (~sink) & (key_idx.view(1, 1, -1) <= horizon.view(1, -1, 1)) & alive
            keep = causal & (sink | local | chosen)

            neg = torch.finfo(torch.float32).min
            lse_dense = torch.logsumexp(logits.masked_fill(~causal, neg), -1)
            lse_S = torch.logsumexp(logits.masked_fill(~keep, neg), -1)
            rho_S = torch.exp(lse_S - lse_dense).clamp(0, 1)  # retained share
            rho_evicted = 1.0 - rho_S

            o_dense = torch.einsum(
                "hts,hsd->htd",
                torch.softmax(logits.masked_fill(~causal, neg), -1),
                vf.repeat_interleave(group, 0),
            )
            o_S = torch.einsum(
                "hts,hsd->htd",
                torch.softmax(logits.masked_fill(~keep, neg), -1),
                vf.repeat_interleave(group, 0),
            )
            # Only rows where eviction actually removed something carry information about oE*.
            live = rho_evicted > 1e-3
            oE_star = (o_dense - rho_S.unsqueeze(-1) * o_S) / rho_evicted.clamp(min=1e-4).unsqueeze(-1)

            # The memory's claim for these rows, broadcast from KV heads to query heads.
            nt = n_mem[0, :, start:stop].repeat_interleave(group, 0)
            dt = d_mem[0, :, start:stop].repeat_interleave(group, 0)
            oE_mem = nt / dt.clamp(min=1e-20).unsqueeze(-1)

            # Mass: d against the mass eviction removed, both relative to D_total.
            d_rel = dt * torch.exp(-lse_dense)  # d / D_total
            mass_ratio = d_rel / rho_evicted.clamp(min=1e-6)  # (d/D_total)/(D_E*/D_total)

            # Direction quality.
            cos = torch.nn.functional.cosine_similarity(oE_mem, oE_star, dim=-1)
            dir_rel = (oE_mem - oE_star).norm(dim=-1) / oE_star.norm(dim=-1).clamp(min=1e-9)

            # Fused output three ways, each against the dense truth.
            inv = torch.exp(-lse_S)
            def fused(nn, dd):
                return (o_S + inv.unsqueeze(-1) * nn) / (1.0 + (inv * dd).unsqueeze(-1))

            # correct mass, trained direction
            D_E_abs = torch.exp(lse_dense) * rho_evicted
            n_massfix = oE_mem * D_E_abs.unsqueeze(-1)
            # correct mass AND direction -> must reproduce o_dense
            n_both = oE_star * D_E_abs.unsqueeze(-1)

            den = o_dense.norm(dim=-1).clamp(min=1e-9)
            e_tr = (fused(nt, dt) - o_dense).norm(dim=-1) / den
            e_mf = (fused(n_massfix, D_E_abs) - o_dense).norm(dim=-1) / den
            e_both = (fused(n_both, D_E_abs) - o_dense).norm(dim=-1) / den
            # and the do-nothing baseline: pure eviction, no memory at all
            e_none = (o_S - o_dense).norm(dim=-1) / den

            for name, t in (
                ("rho", rho_evicted), ("mass", mass_ratio), ("cos", cos), ("dir", dir_rel),
                ("e_tr", e_tr), ("e_mf", e_mf), ("e_both", e_both),
            ):
                acc[name].append(t[live])
            acc.setdefault("e_none", []).append(e_none[live])
            del logits, causal, keep

        m = {kk: torch.cat(vv).float() for kk, vv in acc.items()}
        print(
            f"{idx:>5} {float(m['rho'].mean()):>9.4f} {float(m['mass'].median()):>10.2f} "
            f"{float(m['cos'].mean()):>13.4f} {float(m['dir'].mean()):>11.4f} "
            f"{float(m['e_tr'].mean()):>12.4f} {float(m['e_mf'].mean()):>12.4f} "
            f"{float(m['e_both'].mean()):>10.2e}"
        )
        if idx == want[0]:
            print(f"      (no memory at all, for reference: err = {float(m['e_none'].mean()):.4f})")
        del grab[idx], hidden[idx]
        torch.cuda.empty_cache()

print()
print("READING THIS TABLE")
print("  d/D_E* >> 1        -> the memory claims mass eviction never removed: VOLUME problem")
print("  cos ~ 0, dir ~ 1   -> the rank-R summary is not the evicted direction: DIRECTION problem")
print("  err:massfix << err:trained -> capping the mass share is the fix")
print("  err:massfix ~ err:trained -> rank/objective is the fix, gamma is not the bottleneck")
print("  err:both ~ 0       -> the fusion identity itself is correct (sanity check)")
