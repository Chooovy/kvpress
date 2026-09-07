# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Stage A validation: does supervising ``oE*`` directly actually reach the capacity bound?

The LM-loss run failed for a reason the diagnostics pinned down: the objective never rewarded
reconstructing the evicted branch, so the memory learned a direction with ``cos(n/d, oE*)`` of only
0.49-0.65 and a mass that was near-zero on most rows. Meanwhile k-means at R=16 -- a *convex-hull
vertex*, hence a conservative bound on what ``n/d = sum_r a_r (H_r/z_r)`` can express -- sits 2-6x
below pure eviction. So the capacity is there and the objective was the problem.

This trains **one layer at a time** against the exact target, which is free from two log-normalizers::

    rho_S = exp(lse_S - lse_dense)
    oE*   = (o_dense - rho_S o_S) / (1 - rho_S)      the exact evicted-branch direction
    D_E*  = exp(lse_dense) - exp(lse_S)              the mass eviction actually removed

Loss is the ell2 of the **fused output after o_proj** against dense, as LESS does: ``o_proj`` weights
the channels, so a channel-uniform MSE would over-serve channels the model itself ignores. That form
also carries the right row weighting for free -- ``o_fused - o_dense`` is proportional to
``(1 - rho_S)(oE - oE*)``, so rows eviction did not hurt contribute nothing.

Deliberately cheap and deliberately falsifiable:

* 3 layers, not 36. If the objective cannot beat pure eviction on three layers it will not on 36.
* Trained on one document, reported on a **held-out** one. The question is whether the kernels learn
  a transferable summary, not whether 197K parameters can fit 8192 rows of one document.
* Reported in the same metric as the diagnostics, so the numbers line up against ``none`` and the
  k-means bound directly.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kvpress.presses.gqa_indexer.memory import (
    DEFAULT_SCALAR_EPS,
    MemoryConfig,
    MemoryKernel,
)
from kvpress.presses.gqa_indexer.memory_schedule import block_memory_states
from kvpress.presses.gqa_indexer.press import GQAIndexerPress
from kvpress.presses.gqa_indexer.qi_flex_attention import FLEX_BLOCK, deadlines
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
p.add_argument("--rank", type=int, default=16)
p.add_argument("--steps", type=int, default=400)
p.add_argument("--rows", type=int, default=1024, help="query rows sampled per step")
p.add_argument("--kernel-lr", type=float, default=1e-3)
p.add_argument("--scalar-lr", type=float, default=0.05)
p.add_argument(
    "--log-gamma-init", type=float, default=None,
    help="None calibrates gamma to the layer's own measured D_E*/D_S, which is what Stage A wants. "
    "The shipped DEFAULT_LOG_GAMMA is an OFF state, and it exists to protect an LM-loss run from "
    "starting worse than the eviction baseline -- Stage A regresses on a known target, so it has "
    "nothing to protect against and starting off just wastes ~10 log units of climbing.",
)
p.add_argument(
    "--mass-weight", type=float, default=1.0,
    help="weight on the explicit log(d) -> log(D_E*) term. Valid at lambda=1, a=0 (decay off), "
    "which is this configuration; it must be dropped once decay is enabled, since the memory then "
    "approximates a recency-biased summary rather than an unbiased estimate of the removed mass.",
)
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

# ----------------------------------------------------------------------
# Capture q/k/v/hidden for the layers of interest, for two DIFFERENT documents
# ----------------------------------------------------------------------
from kvpress.presses.gqa_indexer.data import LongminoConfig, build_dataloader

loader = build_dataloader(
    LongminoConfig(
        root="/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_256k_filtered",
        seq_len=L, subsets=("2e16",), take_from="head",
    ),
    tok, batch_size=1, num_workers=0,
)
docs = []
for i, b in enumerate(loader):
    docs.append(b["input_ids"][:, :L].to(DEV))
    if len(docs) >= 2:
        break
assert len(docs) == 2, "need two documents: one to train on, one held out"

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

ALL_ATTENTION_FUNCTIONS.register("probe_stageA", impl)
model.config._attn_implementation = "probe_stageA"


def capture(input_ids):
    grab.clear()
    hidden.clear()
    handles = [
        l.self_attn.register_forward_pre_hook(pre, with_kwargs=True) for l in model.model.layers
    ]
    with torch.no_grad():
        model(input_ids=input_ids, use_cache=False)
    for h in handles:
        h.remove()
    return {i: (grab[i], hidden[i]) for i in WANT}


@torch.no_grad()
def build_targets(idx, layer_data, stride):
    """
    The exact targets for a subsample of query rows: ``(o_S, lse_S, oE*, D_E*, o_dense, rows)``.

    Computed once per document and cached, because they do not depend on the memory's parameters --
    only the frozen backbone and the frozen router. So the training loop re-runs ``psi``/``phi``
    only, which is what makes a per-layer stage cheap.
    """
    (q, k, v, scaling), h = layer_data
    Hq, Hkv = q.shape[1], k.shape[1]
    group = Hq // Hkv
    scale = q.shape[-1] ** -0.5 if scaling is None else float(scaling)
    scores = press.get_indexer(model.model.layers[idx].self_attn).score_keys(h)[0].float()
    dl = deadlines(scores, TOPK, force_sink=SINK, force_local=LOCAL)
    kf, vf = k[0].float(), v[0].float()
    key_idx = torch.arange(L, device=DEV)
    dlg = dl.repeat_interleave(group, 0).to(torch.int64)
    rows = torch.arange(0, L, stride, device=DEV)

    acc = {kk: [] for kk in ("oS", "lseS", "oE", "DE", "od")}
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
        acc["oS"].append(o_s)
        acc["lseS"].append(lse_s)
        acc["oE"].append(
            (o_d - rho_S.unsqueeze(-1) * o_s) / rho_ev.clamp(min=1e-4).unsqueeze(-1)
        )
        acc["DE"].append(torch.exp(lse_d) - torch.exp(lse_s))
        acc["od"].append(o_d)
        del logits, causal, keep
    out = {kk: torch.cat(vv, 1) for kk, vv in acc.items()}
    out["rows"] = rows
    out["dl"] = dl
    out["scores"] = scores
    return out


def read_rows(kernel, q_kv_rows, H, z, counts, block_of_row):
    """
    ``(n, d)`` for a set of sampled rows, each reading its own query block's state.

    Same arithmetic as :func:`~.memory.memory_terms`, with the state gathered per row instead of
    broadcast per block -- the rows here are a stride-subsample, so they do not form whole blocks.
    """
    Hb = H[0, :, block_of_row]  # (Hkv, N, R, D)
    zb = z[0, :, block_of_row]  # (Hkv, N, R)
    cb = counts[:, block_of_row].float()  # (Hkv, N)
    phi = kernel.phi(q_kv_rows.unsqueeze(0))[0].float()  # (Hkv, N, R)
    phi_hat = phi / phi.sum(-1, keepdim=True).clamp(min=1e-20)
    z_sum = zb.sum(-1, keepdim=True).clamp(min=1e-20)
    z_hat = zb / z_sum
    gamma = kernel.gamma.view(-1, 1)
    d = gamma * cb * (phi_hat * z_hat).sum(-1)
    num = torch.einsum("hnr,hnrd->hnd", phi_hat, Hb)
    den = torch.einsum("hnr,hnr->hn", phi_hat, zb).clamp(min=1e-20)
    n = d.unsqueeze(-1) * num / den.unsqueeze(-1)
    return n, d


def fused_and_err(o_proj, tgt, n, d, group):
    """Fused output, its ``o_proj``-space loss against dense, and the attention-space rel error."""
    nt = n.repeat_interleave(group, 0).unsqueeze(0)
    dt = d.repeat_interleave(group, 0).unsqueeze(0)
    o_s = tgt["oS"].unsqueeze(0)
    lse = tgt["lseS"].unsqueeze(0)
    from kvpress.presses.gqa_indexer.memory import fuse_memory

    # fuse_memory wants n/d per KV head; pass the already-expanded ones with group=1.
    fused = fuse_memory(o_s, lse, nt, dt, group=1)[0]  # (Hq, N, D)
    o_d = tgt["od"]
    live = (tgt["DE"] / torch.exp(tgt["lseS"]).clamp(min=1e-20)) > 1e-3

    # o_proj space: (N, Hq*D) -> hidden. This is the trained objective.
    Hq, N, D = fused.shape
    fl = fused.permute(1, 0, 2).reshape(N, Hq * D)
    dl_ = o_d.permute(1, 0, 2).reshape(N, Hq * D)
    proj_f = o_proj(fl.to(o_proj.weight.dtype)).float()
    proj_d = o_proj(dl_.to(o_proj.weight.dtype)).float()
    loss = (proj_f - proj_d).square().mean()

    with torch.no_grad():
        rel = (fused - o_d).norm(dim=-1) / o_d.norm(dim=-1).clamp(min=1e-9)
        rel = float(rel[live].mean())
        cos = float(
            torch.nn.functional.cosine_similarity(
                (n.repeat_interleave(group, 0) / d.repeat_interleave(group, 0).clamp(min=1e-20).unsqueeze(-1)),
                tgt["oE"], dim=-1,
            )[live].mean()
        )
        mass = float(
            (d.repeat_interleave(group, 0) / tgt["DE"].clamp(min=1e-20))[live].median()
        )
    return loss, rel, cos, mass


print(f"Stage A validation: L={L} topk={TOPK} rank={args.rank} steps={args.steps}")
print("  train on document 0, report on HELD-OUT document 1")
print()

train_layers = capture(docs[0])
train_tgt = {i: build_targets(i, train_layers[i], stride=4) for i in WANT}
train_kv = {i: (train_layers[i][0][1].float(), train_layers[i][0][2].float()) for i in WANT}
train_q = {i: train_layers[i][0][0].float() for i in WANT}

val_layers = capture(docs[1])
val_tgt = {i: build_targets(i, val_layers[i], stride=8) for i in WANT}
val_kv = {i: (val_layers[i][0][1].float(), val_layers[i][0][2].float()) for i in WANT}
val_q = {i: val_layers[i][0][0].float() for i in WANT}

results = {}
for idx in WANT:
    torch.manual_seed(0)
    Hq = train_q[idx].shape[1]
    k_tr, v_tr = train_kv[idx]
    Hkv = k_tr.shape[1]
    group = Hq // Hkv
    D = k_tr.shape[-1]
    o_proj = model.model.layers[idx].self_attn.o_proj

    kernel = MemoryKernel(
        MemoryConfig(n_kv_heads=Hkv, head_dim=D, rank=args.rank, mid_dim=256)
    ).to(DEV)
    kernel.upcast_scalars()

    tgt0 = train_tgt[idx]
    if args.log_gamma_init is None:
        # d ~ gamma |E| <phi_hat, z_hat> ~ gamma |E| / R, and the target is d = D_E*. Solve for the
        # gamma that lands on the layer's OWN measured mass rather than on a global constant: rho
        # varies 0.11-0.21 across depth, so one number would start some layers an order out.
        with torch.no_grad():
            live0 = (tgt0["DE"] / torch.exp(tgt0["lseS"]).clamp(min=1e-20)) > 1e-3
            de_med = float(tgt0["DE"][live0].median())
            e_med = float(L - TOPK)
            g0 = de_med / max(e_med / args.rank, 1e-9)
            import math as _m
            kernel.log_gamma.data.fill_(_m.log(max(g0, 1e-12)))
        print(f"    calibrated log_gamma = {float(kernel.log_gamma.mean()):+.2f} "
              f"(median D_E* {de_med:.1f}, |E| ~ {e_med:.0f}, R {args.rank})")
    else:
        kernel.log_gamma.data.fill_(args.log_gamma_init)
    opt = torch.optim.AdamW(
        [
            {"params": kernel.kernel_parameters(), "lr": args.kernel_lr},
            {"params": kernel.scalar_parameters(), "lr": args.scalar_lr, "eps": DEFAULT_SCALAR_EPS},
        ],
        betas=(0.9, 0.95),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    tgt = train_tgt[idx]
    n_rows = tgt["rows"].numel()
    q_kv_all = train_q[idx].view(1, Hkv, group, L, D).mean(2)[0]
    block_of = (tgt["rows"] // FLEX_BLOCK).clamp(max=L // FLEX_BLOCK - 1)

    print(f"layer {idx}: {n_rows} train rows, {val_tgt[idx]['rows'].numel()} val rows")
    for step in range(args.steps):
        H, z, W, counts = block_memory_states(
            kernel, k_tr, v_tr, tgt["dl"], q_len=L, block=FLEX_BLOCK,
            n_local=LOCAL, scores=tgt["scores"],
        )
        sel = torch.randperm(n_rows, device=DEV)[: args.rows]
        n, d = read_rows(kernel, q_kv_all[:, tgt["rows"][sel]], H, z, counts, block_of[sel])
        sub = {kk: (tgt[kk][:, sel] if tgt[kk].dim() >= 2 else tgt[kk]) for kk in
               ("oS", "lseS", "oE", "DE", "od")}
        loss, rel, cos, mass = fused_and_err(o_proj, sub, n, d, group)
        if args.mass_weight:
            # An explicit log-space mass term. The o_proj ell2 alone is a weak signal for the mass:
            # it is dominated by the direction, and a memory that claims ~0 mass scores almost as
            # well as pure eviction (which is exactly what the LM-loss run converged to). In log
            # space the term is scale-free, so it does not fight the direction for gradient budget.
            dl_live = (sub["DE"] / torch.exp(sub["lseS"]).clamp(min=1e-20)) > 1e-3
            d_exp = d.repeat_interleave(group, 0)
            mass_loss = (
                (d_exp.clamp(min=1e-20).log() - sub["DE"].clamp(min=1e-20).log())[dl_live]
                .square().mean()
            )
            loss = loss + args.mass_weight * mass_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(kernel.parameters()), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == args.steps - 1:
            print(f"    step {step:4d}: loss {float(loss):.5f}  rel_err {rel:.4f}  "
                  f"cos {cos:.4f}  d/D_E* {mass:.3f}  gamma {float(kernel.gamma.mean()):.3e}")

    # Held-out document, every row of it.
    with torch.no_grad():
        vt = val_tgt[idx]
        kv_k, kv_v = val_kv[idx]
        Hv, zv, Wv, cv = block_memory_states(
            kernel, kv_k, kv_v, vt["dl"], q_len=L, block=FLEX_BLOCK,
            n_local=LOCAL, scores=vt["scores"],
        )
        q_kv_v = val_q[idx].view(1, Hkv, group, L, D).mean(2)[0]
        bo = (vt["rows"] // FLEX_BLOCK).clamp(max=L // FLEX_BLOCK - 1)
        nv, dv = read_rows(kernel, q_kv_v[:, vt["rows"]], Hv, zv, cv, bo)
        _, rel_v, cos_v, mass_v = fused_and_err(o_proj, vt, nv, dv, group)
        # pure eviction on the same rows, for the comparison that matters
        zero_n = torch.zeros_like(nv)
        zero_d = torch.zeros_like(dv)
        _, rel_none, _, _ = fused_and_err(o_proj, vt, zero_n, zero_d, group)
    results[idx] = (rel_none, rel_v, cos_v, mass_v)
    print(f"    HELD OUT: none {rel_none:.4f} -> stageA {rel_v:.4f}  "
          f"cos {cos_v:.4f}  d/D_E* {mass_v:.3f}")
    print()
    del kernel, opt
    torch.cuda.empty_cache()

print("=" * 76)
print("HELD-OUT SUMMARY (relative error of the fused attention output vs dense)")
print()
print(f"{'layer':>5} {'none':>9} {'stageA':>9} {'ratio':>8} {'cos':>8} {'d/D_E*':>9}  reference")
ref = {0: (0.1695, 0.0366), 18: (0.1837, 0.1104), 35: (0.1141, 0.0335)}
for idx, (rn, rv, cv, mv) in results.items():
    r = ref.get(idx)
    tag = f"LM-loss run / kmeans R16: -- / {r[1]:.4f}" if r else ""
    print(f"{idx:>5} {rn:>9.4f} {rv:>9.4f} {rn / max(rv, 1e-9):>8.2f}x {cv:>8.4f} {mv:>9.3f}  {tag}")
print()
print("ratio > 1 means Stage A BEATS pure eviction on held-out text.")
print("Compare stageA against the kmeans R=16 column: that is the conservative capacity bound.")
