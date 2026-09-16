# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Is there a SIMPLER, more stable allocation than bisecting a mass target?

The mass-matched allocator works (+6.55 RULER, and a budget-shuffle control drops to 80.49, below
uniform -- so the gain is the head<->budget correspondence, not raggedness). But the budget vector
it produces moves ~47% of its L1 mass when the reference row changes, which is uncomfortable for a
method and expensive to compute (a full cumulative-mass curve per layer).

That instability may be an artifact of measuring the wrong thing. L1 over budgets mixes a stable
SKELETON (which heads are streaming heads vs retrieval heads) with per-row noise on the exact
counts. This script separates them, and tests a candidate that is both simpler and, if the
skeleton hypothesis holds, more stable.

The candidate: **participation ratio**
--------------------------------------
``PR_h = 1 / sum_j p_j^2`` for head ``h``'s attention distribution is, by definition, the
*effective number of keys the head attends to*. So "how many keys does this head need" has a
closed-form answer that needs no target, no bisection, and no cumulative curve -- one scalar per
head:

    ``budget_h = total * PR_h / sum_h' PR_h'``

Three reasons this is the right shape for the paper, if it holds up:

1. **Gauge-invariant** for the same reason mass is: it is computed from the attention
   distribution, not from the router's score, so it is immune to the per-(layer, head) additive
   constant the trained gate cannot see.
2. It is **the same quantity training already logs.** ``E2EIndexerTrainer._gate_participation``
   computes ``exp(2*lse - lse2)``, a participation ratio, per (batch, head, row) -- and then
   averages the head axis away. The metric was there the whole time; only the reduction hid it.
3. One scalar per head, so it is O(Sk) per head instead of a sort plus a bisection over a
   cumulative curve.

What is measured here
---------------------
* Spearman rank correlation of head demand across reference rows and across documents -- the
  skeleton test. High rank stability with low L1 stability means the ordering is the real signal.
* Bottom-quartile set overlap: do the same heads get starved?
* How well PR-allocation agrees with the mass-matched allocation it would replace.
* PR's own stability, against the mass allocator's.
"""

from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kvpress.presses.gqa_indexer.head_budget import mass_head_budgets  # noqa: E402
from kvpress.presses.gqa_indexer.press import GQAIndexerPress, get_language_model  # noqa: E402
from kvpress.presses.gqa_indexer.train import (  # noqa: E402
    load_indexer_state_dict,
    press_kwargs_from_checkpoint,
)

MODELS = "/apdcephfs_gy8/share_303843174/guhao/models"
CKPT = f"{MODELS}/Qwen-3-8B-gqa_indexer_scalar/fwkl_ce01_8k_local128_b256_decay/final.pt"
dev = "cuda:0"
TOPK, FS, FL, SEQ = 2048, 4, 128, 8192
torch.set_grad_enabled(False)


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Rank correlation of two short vectors."""
    n = a.numel()
    if n < 3:
        return float("nan")
    ra = a.float().argsort().argsort().float()
    rb = b.float().argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm()).clamp(min=1e-9)
    return float((ra * rb).sum() / denom)


def participation(query, key, *, ref_row, scaling, group, n_kv):
    """
    ``PR_h = 1 / sum_j p_j^2`` per KV head -- the effective number of keys the head attends to.

    Computed over the FULL causal history, pins included: PR answers "how many keys does this
    head need", and the pins are part of that need. Query heads are averaged into their KV head,
    matching where the budget lives.
    """
    k_len = key.shape[2]
    qr = query[0, :, ref_row, :].float()
    kf = key[0].float().repeat_interleave(group, 0)
    logits = torch.einsum("hd,hsd->hs", qr, kf) * scaling
    key_idx = torch.arange(k_len, device=query.device)
    logits = logits.masked_fill(key_idx > ref_row, torch.finfo(torch.float32).min)
    p = torch.softmax(logits, dim=-1)
    pr = 1.0 / p.pow(2).sum(-1).clamp(min=1e-12)          # (Hq,)
    return pr.view(n_kv, group).mean(1)                    # (Hkv,)


def pr_budgets(pr, *, topk, force_sink, force_local, floor=0):
    """
    ``budget_h = total * PR_h / sum PR``, integerized with the total conserved exactly.

    Largest-remainder rounding, so the residual goes to the heads with the biggest fractional
    claim rather than being absorbed arbitrarily.
    """
    n = pr.numel()
    take = topk - force_sink - force_local
    total = take * n
    floor = max(0, min(int(floor), total // n))
    share = pr.double() / pr.double().sum().clamp(min=1e-12)
    ideal = share * (total - floor * n)
    base = ideal.floor().long()
    residual = int(total - floor * n - base.sum())
    if residual > 0:
        order = (ideal - base).argsort(descending=True)
        base[order[:residual]] += 1
    elif residual < 0:
        order = (ideal - base).argsort()
        base[order[: -residual]] -= 1
    return (base + floor + force_sink + force_local).to(torch.int64)


from transformers import AutoModelForCausalLM  # noqa: E402

model = (
    AutoModelForCausalLM.from_pretrained(
        f"{MODELS}/Qwen3-8B", torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(dev).eval()
)
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
state = ck.get("indexer", ck)
scorer, kw = press_kwargs_from_checkpoint(state, ck.get("config") or {})
press = GQAIndexerPress(
    compression_ratio=0.0,
    gate_scale=any(str(x).endswith("gate_scale") for x in state),
    scorer_attr="indexer", scorer=scorer, **kw,
)
press.post_init_from_model(model, force_reinit=True)
load_indexer_state_dict(model, state, "indexer")

layers = get_language_model(model).layers
cfg = model.config
n_q, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
group = n_q // n_kv
head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_q)
scale = head_dim ** -0.5

grabbed, hidden = {}, {}
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def impl(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwx):
    grabbed[int(module.layer_idx)] = (query.detach(), key.detach())
    out = F.scaled_dot_product_attention(
        query, key.repeat_interleave(group, 1), value.repeat_interleave(group, 1),
        is_causal=True, scale=scaling,
    )
    return out.transpose(1, 2).contiguous(), None


nm = "skel_capture"
gm = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
ALL_ATTENTION_FUNCTIONS.register(nm, impl)


def hook(module, a, kwx):
    hs = kwx.get("hidden_states")
    if hs is None and a:
        hs = a[0]
    hidden[int(module.layer_idx)] = hs.detach()
    return None


handles = [l.self_attn.register_forward_pre_hook(hook, with_kwargs=True) for l in layers]
for c in [cfg] + ([cfg.text_config] if getattr(cfg, "text_config", None) else []):
    c._attn_implementation = nm

from kvpress.presses.gqa_indexer.data import TokenizedConfig, build_tokenized_dataloader  # noqa: E402

loader = build_tokenized_dataloader(
    TokenizedConfig(root=f"{MODELS}/../datasets/longmino_tokenized_64k", seq_len=SEQ,
                    take_from="head"),
    batch_size=1, num_workers=0,
)
batches = []
for b in loader:
    batches.append(b["input_ids"][:, :SEQ].to(dev))
    if len(batches) >= 6:
        break

PROBE = [0, 2, 8, 17, 26, 35]
ROWS = [SEQ - 1, int(SEQ * 0.875), int(SEQ * 0.75), int(SEQ * 0.5)]


def mass_at(li, ref):
    q, k = grabbed[li]
    h = hidden[li]
    ind = press.get_indexer(layers[li].self_attn)
    sc = ind.score_at(h, float(ref))[0] if getattr(ind, "decay", False) else ind.score_keys(h)[0]
    return mass_head_budgets(
        q[:, :, : ref + 1], k[:, :, : ref + 1], sc[:, : ref + 1],
        topk=TOPK, force_sink=FS, force_local=FL, scaling=scale, ref_row=ref, floor=0,
    )


# Collect everything in one pass per document.
data = []   # (sample, layer, row) -> {mass, pr}
for si, ids in enumerate(batches):
    grabbed.clear(); hidden.clear()
    model(input_ids=ids, use_cache=False)
    for li in PROBE:
        q, k = grabbed[li]
        for r in ROWS:
            data.append({
                "s": si, "l": li, "r": r,
                "mass": mass_at(li, r),
                "pr": participation(q[:, :, : r + 1], k[:, :, : r + 1],
                                    ref_row=r, scaling=scale, group=group, n_kv=n_kv),
            })
    torch.cuda.empty_cache()
    print(f"  doc {si} captured", flush=True)


def get(si, li, r, key):
    for d in data:
        if d["s"] == si and d["l"] == li and d["r"] == r:
            return d[key]
    raise KeyError


print()
print("=" * 78)
print("1. SKELETON TEST: is the ORDERING stable where the counts are not?")
print("=" * 78)
print("Spearman rank corr of the budget vector across reference rows (mean over docs/layers),")
print("alongside the L1 fraction that looked alarming.\n")
print(f"{'row':>8}{'mass: rank rho':>17}{'mass: L1 frac':>16}{'PR: rank rho':>15}{'PR: L1 frac':>14}")
summary = {}
for r in ROWS[1:]:
    mr, ml, pr_, pl = [], [], [], []
    for si in range(len(batches)):
        for li in PROBE:
            m0, m1 = get(si, li, ROWS[0], "mass"), get(si, li, r, "mass")
            p0, p1 = get(si, li, ROWS[0], "pr"), get(si, li, r, "pr")
            mr.append(spearman(m0, m1))
            ml.append(float((m0 - m1).abs().sum()) / (TOPK * n_kv))
            pr_.append(spearman(p0, p1))
            b0 = pr_budgets(p0, topk=TOPK, force_sink=FS, force_local=FL)
            b1 = pr_budgets(p1, topk=TOPK, force_sink=FS, force_local=FL)
            pl.append(float((b0 - b1).abs().sum()) / (TOPK * n_kv))
    summary[r] = dict(mass_rho=sum(mr)/len(mr), mass_l1=sum(ml)/len(ml),
                      pr_rho=sum(pr_)/len(pr_), pr_l1=sum(pl)/len(pl))
    s = summary[r]
    print(f"{r:>8}{s['mass_rho']:>17.3f}{s['mass_l1']:>16.3f}"
          f"{s['pr_rho']:>15.3f}{s['pr_l1']:>14.3f}")

print()
print("=" * 78)
print("2. STARVATION SET: do the SAME heads get starved across rows?")
print("=" * 78)
print("Jaccard overlap of the bottom-2-of-8 head set (the heads the allocator squeezes).\n")
print(f"{'row':>8}{'mass':>10}{'PR':>10}")
for r in ROWS[1:]:
    mj, pj = [], []
    for si in range(len(batches)):
        for li in PROBE:
            for key, acc in (("mass", mj), ("pr", pj)):
                v0, v1 = get(si, li, ROWS[0], key), get(si, li, r, key)
                s0 = set(v0.argsort()[:2].tolist())
                s1 = set(v1.argsort()[:2].tolist())
                acc.append(len(s0 & s1) / len(s0 | s1))
    print(f"{r:>8}{sum(mj)/len(mj):>10.3f}{sum(pj)/len(pj):>10.3f}")

print()
print("=" * 78)
print("3. DOES PR AGREE WITH THE MASS ALLOCATOR IT WOULD REPLACE?")
print("=" * 78)
rhos, l1s = [], []
for si in range(len(batches)):
    for li in PROBE:
        m = get(si, li, ROWS[0], "mass")
        p = pr_budgets(get(si, li, ROWS[0], "pr"), topk=TOPK, force_sink=FS, force_local=FL)
        rhos.append(spearman(m, p))
        l1s.append(float((m - p).abs().sum()) / (TOPK * n_kv))
print(f"  Spearman(mass budgets, PR budgets) = {sum(rhos)/len(rhos):.3f}")
print(f"  L1 fraction between them           = {sum(l1s)/len(l1s):.3f}")
print("\n  Per-layer, doc 0 (mass vs PR budgets):")
for li in PROBE:
    m = get(0, li, ROWS[0], "mass")
    p = pr_budgets(get(0, li, ROWS[0], "pr"), topk=TOPK, force_sink=FS, force_local=FL)
    print(f"   layer {li:2d} mass {[int(x) for x in m]}")
    print(f"            PR   {[int(x) for x in p]}   rho={spearman(m,p):+.2f}")

print()
print("=" * 78)
print("4. CROSS-DOCUMENT: could the table be fitted ONCE and shipped?")
print("=" * 78)
print("If the ranking is stable across documents, allocation needs no prefill-time measurement")
print("at all -- which removes the decode question entirely.\n")
print(f"{'layer':>6}{'mass rank rho':>16}{'PR rank rho':>14}{'PR bottom-2 Jacc':>19}")
xd = {}
for li in PROBE:
    mr, pr_, pj = [], [], []
    for i in range(len(batches)):
        for j in range(i + 1, len(batches)):
            mi, mj_ = get(i, li, ROWS[0], "mass"), get(j, li, ROWS[0], "mass")
            pi, pjv = get(i, li, ROWS[0], "pr"), get(j, li, ROWS[0], "pr")
            mr.append(spearman(mi, mj_))
            pr_.append(spearman(pi, pjv))
            s0 = set(pi.argsort()[:2].tolist()); s1 = set(pjv.argsort()[:2].tolist())
            pj.append(len(s0 & s1) / len(s0 | s1))
    xd[li] = (sum(mr)/len(mr), sum(pr_)/len(pr_), sum(pj)/len(pj))
    print(f"{li:>6}{xd[li][0]:>16.3f}{xd[li][1]:>14.3f}{xd[li][2]:>19.3f}")
print(f"\n  mean: mass rho {sum(v[0] for v in xd.values())/len(xd):+.3f}, "
      f"PR rho {sum(v[1] for v in xd.values())/len(xd):+.3f}")

print()
print("=" * 78)
print("5. THE SHIPPABLE TABLE: PR budgets averaged over documents, all 36 layers")
print("=" * 78)
tbl = {}
for si, ids in enumerate(batches):
    grabbed.clear(); hidden.clear()
    model(input_ids=ids, use_cache=False)
    for li in range(len(layers)):
        q, k = grabbed[li]
        pr = participation(q, k, ref_row=SEQ - 1, scaling=scale, group=group, n_kv=n_kv)
        tbl.setdefault(li, []).append(pr)
    torch.cuda.empty_cache()
mean_pr = {li: torch.stack(v).mean(0) for li, v in tbl.items()}
print("  layer : PR-allocated budgets (mean PR over %d docs)" % len(batches))
for li in range(0, len(layers), 4):
    b = pr_budgets(mean_pr[li], topk=TOPK, force_sink=FS, force_local=FL)
    print(f"   {li:2d} : {[int(x) for x in b]}  sum={int(b.sum())}")

with open("scratch/diag_skeleton.json", "w") as f:
    json.dump({
        "row_stability": {str(k): v for k, v in summary.items()},
        "cross_doc": {str(k): list(v) for k, v in xd.items()},
        "pr_table": {str(li): [float(x) for x in v] for li, v in mean_pr.items()},
    }, f, indent=2)
for h in handles:
    h.remove()
gm.pop(nm, None)
print("\nwrote scratch/diag_skeleton.json")
