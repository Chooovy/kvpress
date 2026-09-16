# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Does a budget frozen at prefill degrade over a LONG generation?

The allocation is fitted on one reference row at the end of the prefill and then frozen -- it must
be, since ``deadlines``' monotonicity assumes a key never returns once evicted. RULER generates at
most ~50 tokens, so it cannot see any drift. But budgets refitted at L=3072 vs L=8192 disagree by
~0.53 of the total (``scratch/diag_budget_stability.py``), which says the *correct* allocation
moves as the sequence grows. If that matters, a reasoning-style 1-4K-token generation is where it
would show.

This measures it directly, without needing a benchmark: generate a long continuation under the
frozen budget, and at intervals compare the frozen allocation against the one that *would* be
fitted now. Two quantities:

* **budget drift** -- L1 between frozen and refitted, as a fraction of the total. The stability
  study's number, now measured along a real decode rather than by truncating a prefill.
* **retained mass under the frozen budget vs. under a refit** -- the quantity that actually
  matters. Drift is only harmful if it costs mass; a permutation among equally-good allocations
  would drift without hurting.

The second is the honest test, and it is the one to report: if frozen and refitted allocations
retain the same mass, freezing is free and the paper can say so.
"""

from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kvpress.presses.gqa_indexer.head_budget import head_mass_curve, mass_head_budgets  # noqa: E402
from kvpress.presses.gqa_indexer.press import GQAIndexerPress, get_language_model  # noqa: E402
from kvpress.presses.gqa_indexer.sparse_inference import SparseAttentionContext  # noqa: E402
from kvpress.presses.gqa_indexer.train import (  # noqa: E402
    load_indexer_state_dict,
    press_kwargs_from_checkpoint,
)

MODELS = "/apdcephfs_gy8/share_303843174/guhao/models"
CKPT = f"{MODELS}/Qwen-3-8B-gqa_indexer_scalar/fwkl_ce01_8k_local128_b256_decay/final.pt"
dev = "cuda:0"
TOPK, FS, FL = 2048, 4, 128
CTX = 6000
GEN = 3000
CHECKPOINTS = [0, 250, 500, 1000, 2000, 3000]
PROBE = [0, 2, 8, 17, 26, 35]
torch.set_grad_enabled(False)

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402

tok = AutoTokenizer.from_pretrained(f"{MODELS}/Qwen3-8B")
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

# ---- capture q/k/hidden on demand -------------------------------------------------------------
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


nm = "decode_drift"
gm = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
ALL_ATTENTION_FUNCTIONS.register(nm, impl)


def hook(module, a, kwx):
    hs = kwx.get("hidden_states")
    if hs is None and a:
        hs = a[0]
    hidden[int(module.layer_idx)] = hs.detach()
    return None


handles = [l.self_attn.register_forward_pre_hook(hook, with_kwargs=True) for l in layers]


def budgets_and_mass(li, ids, frozen=None):
    """Refit budgets on the current sequence, and the mass BOTH allocations retain."""
    q, k = grabbed[li]
    h = hidden[li]
    ind = press.get_indexer(layers[li].self_attn)
    ref = k.shape[2] - 1
    sc = ind.score_at(h, float(ref))[0] if getattr(ind, "decay", False) else ind.score_keys(h)[0]
    fresh = mass_head_budgets(
        q, k, sc, topk=TOPK, force_sink=FS, force_local=FL, scaling=scale, floor=512,
    )
    cum, n_pool = head_mass_curve(
        q, k, sc, ref_row=ref, scaling=scale, force_sink=FS, force_local=FL,
    )

    def mass_of(budget):
        idx = (budget - FS - FL - 1).clamp(0, max(n_pool - 1, 0))
        return cum.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

    out = {"fresh": fresh, "mass_fresh": mass_of(fresh)}
    if frozen is not None:
        out["mass_frozen"] = mass_of(frozen.clamp(max=n_pool + FS + FL))
    return out


# ---- run one long generation under the frozen budget -------------------------------------------
torch.manual_seed(0)
from kvpress.presses.gqa_indexer.data import TokenizedConfig, build_tokenized_dataloader  # noqa: E402

loader = build_tokenized_dataloader(
    TokenizedConfig(root=f"{MODELS}/../datasets/longmino_tokenized_64k", seq_len=CTX,
                    take_from="head"),
    batch_size=1, num_workers=0,
)
ctx_ids = next(iter(loader))["input_ids"][:, :CTX].to(dev)

for c in [cfg] + ([cfg.text_config] if getattr(cfg, "text_config", None) else []):
    c._attn_implementation = nm

rows = []
# Phase 1: prefill under the sparse context, capture the frozen budgets, then GENERATE.
# The measurement forwards are deliberately kept out of this `with` block: SparseAttentionContext
# maintains its own append-only indexer key-cache and asserts it stays in lockstep with the
# model's, so a throwaway dense forward inside the block trips that assertion (observed:
# "expected 6000 newly appended values but found 0"). Generate first, measure after.
with SparseAttentionContext(
    model, press, topk=TOPK, force_sink=FS, force_local=FL,
    head_budget="mass", head_budget_floor=512,
) as ctx:
    cache = DynamicCache()
    grabbed.clear(); hidden.clear()
    model.model(input_ids=ctx_ids, past_key_values=cache)
    frozen = {li: ctx._head_topk[li].clone() for li in PROBE}
    print(f"prefill L={CTX}; frozen budgets captured for {len(frozen)} probe layer(s)", flush=True)

    # Greedy decode under the frozen budget, recording the token sequence at each checkpoint so
    # the drift can be measured against exactly what the model produced.
    snapshots = {}
    cur = ctx_ids[:, -1:]
    produced = []
    pending = [c for c in CHECKPOINTS]
    if pending and pending[0] == 0:
        snapshots[pending.pop(0)] = []
    for step in range(1, GEN + 1):
        out = model(input_ids=cur, past_key_values=cache)
        nxt = int(out.logits[0, -1].argmax())
        produced.append(nxt)
        cur = torch.tensor([[nxt]], device=dev)
        if pending and step >= pending[0]:
            snapshots[pending.pop(0)] = list(produced)
            print(f"  generated {step} tokens", flush=True)
        if not pending:
            break

print(f"\ngenerated text (first 200 chars): {tok.decode(produced[:60])[:200]!r}\n", flush=True)

# Phase 2: measure. One dense forward per checkpoint, outside the sparse context.
for c in [cfg] + ([cfg.text_config] if getattr(cfg, "text_config", None) else []):
    c._attn_implementation = nm

for mark, toks in sorted(snapshots.items()):
    full = ctx_ids if not toks else torch.cat(
        [ctx_ids, torch.tensor([toks], device=dev)], dim=1
    )
    grabbed.clear(); hidden.clear()
    model(input_ids=full, use_cache=False)
    for li in PROBE:
        r = budgets_and_mass(li, full, frozen=frozen[li])
        rows.append({
            "step": mark, "layer": li, "L": int(full.shape[1]),
            "l1": float((frozen[li] - r["fresh"]).abs().sum()) / (TOPK * n_kv),
            "mass_frozen": float(r["mass_frozen"].mean()),
            "mass_fresh": float(r["mass_fresh"].mean()),
            "mass_frozen_min": float(r["mass_frozen"].min()),
            "mass_fresh_min": float(r["mass_fresh"].min()),
        })
    print(f"  step {mark:5d} (L={int(full.shape[1])}) measured", flush=True)
    torch.cuda.empty_cache()

for h in handles:
    h.remove()
gm.pop(nm, None)

print()
print("=" * 78)
print("BUDGET DRIFT AND ITS COST OVER A 3000-TOKEN GENERATION")
print("=" * 78)
print("l1        = |frozen - refit| / total budget  (how much the allocation moved)")
print("mass gap  = mean retained mass under refit MINUS under frozen (the actual cost)")
print("           <=0 means freezing costs nothing.\n")
print(f"{'step':>6}{'L':>7}{'l1':>8}{'mass frozen':>13}{'mass refit':>12}{'mass gap':>10}"
      f"{'min gap':>9}")
for st in CHECKPOINTS:
    rs = [r for r in rows if r["step"] == st]
    if not rs:
        continue
    l1 = sum(r["l1"] for r in rs) / len(rs)
    mf = sum(r["mass_frozen"] for r in rs) / len(rs)
    mn = sum(r["mass_fresh"] for r in rs) / len(rs)
    gmin = sum(r["mass_fresh_min"] - r["mass_frozen_min"] for r in rs) / len(rs)
    print(f"{st:>6}{rs[0]['L']:>7}{l1:>8.3f}{mf:>13.4f}{mn:>12.4f}{mn-mf:>+10.4f}{gmin:>+9.4f}")

print("\nper layer at the final step:")
for r in [x for x in rows if x["step"] == CHECKPOINTS[-1]]:
    print(f"  layer {r['layer']:2d}: l1={r['l1']:.3f}  mass frozen {r['mass_frozen']:.4f} "
          f"vs refit {r['mass_fresh']:.4f}  gap {r['mass_fresh']-r['mass_frozen']:+.4f}")

with open("scratch/diag_decode_drift.json", "w") as f:
    json.dump(rows, f, indent=2)
print("\nwrote scratch/diag_decode_drift.json")
