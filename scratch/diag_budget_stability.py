# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Two threats to the mass-allocation result, measured rather than argued.

**A. Is the allocation query-dependent?**  The budget is fitted on ONE reference row (the last
prefill position), before any question is seen. If a different question would want a different
split, the method is not "allocate per document" but "allocate per query" -- and at prefill the
query does not exist yet. RULER hides this: its question is appended after the context, so the
reference row is question-agnostic by construction, and every question for a context reuses the
same prefill. A multi-turn or multi-query deployment would not.

Measured as: fit budgets at the last context row, then refit using rows drawn from deeper inside
the context, and compare. Large disagreement = the allocation is a property of the row, not the
document, and the paper's claim has to be narrowed.

**B. Does the allocation drift as decoding proceeds?**  Budgets are frozen at prefill and reused
for every generated token. The concern is not the freeze itself (monotone deadlines require it)
but whether the correct allocation moves as the sequence grows. Measured by extending the context
and refitting.

**C. How much of the gain survives if the budget is quantized coarsely?**  A deployable
implementation may not want 8 distinct ragged lengths per layer. If rounding to a few buckets
keeps the gain, the engineering story is much easier.
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
TOPK, FS, FL = 2048, 4, 128
torch.set_grad_enabled(False)

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

grabbed: dict = {}
hidden: dict = {}
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def impl(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwx):
    grabbed[int(module.layer_idx)] = (query.detach(), key.detach())
    out = F.scaled_dot_product_attention(
        query, key.repeat_interleave(group, 1), value.repeat_interleave(group, 1),
        is_causal=True, scale=scaling,
    )
    return out.transpose(1, 2).contiguous(), None


nm = "drift_capture"
gm = type(ALL_ATTENTION_FUNCTIONS)._global_mapping
ALL_ATTENTION_FUNCTIONS.register(nm, impl)


def hook(module, a, kwx):
    hs = kwx.get("hidden_states")
    if hs is None and a:
        hs = a[0]
    hidden[int(module.layer_idx)] = hs.detach()
    return None


hs_handles = [l.self_attn.register_forward_pre_hook(hook, with_kwargs=True) for l in layers]
for c in [cfg] + ([cfg.text_config] if getattr(cfg, "text_config", None) else []):
    c._attn_implementation = nm

from kvpress.presses.gqa_indexer.data import TokenizedConfig, build_tokenized_dataloader  # noqa: E402

SEQ = 8192
loader = build_tokenized_dataloader(
    TokenizedConfig(root=f"{MODELS}/../datasets/longmino_tokenized_64k", seq_len=SEQ,
                    take_from="head"),
    batch_size=1, num_workers=0,
)
batches = []
for b in loader:
    batches.append(b["input_ids"][:, :SEQ].to(dev))
    if len(batches) >= 3:
        break


def budgets_at(layer_idx, ref_row, floor=0):
    q, k = grabbed[layer_idx]
    h = hidden[layer_idx]
    ind = press.get_indexer(layers[layer_idx].self_attn)
    sc = ind.score_at(h, float(ref_row))[0] if getattr(ind, "decay", False) \
        else ind.score_keys(h)[0]
    return mass_head_budgets(
        q[:, :, : ref_row + 1], k[:, :, : ref_row + 1], sc[:, : ref_row + 1],
        topk=TOPK, force_sink=FS, force_local=FL, scaling=scale,
        ref_row=ref_row, floor=floor,
    )


PROBE_LAYERS = [0, 2, 8, 17, 26, 35]
out: dict = {}

print("=" * 78)
print("A. QUERY DEPENDENCE: budgets fitted at different reference rows")
print("=" * 78)
print("L1 distance between budget vectors, as a fraction of the total budget (2048*8=16384).")
print("0 = identical allocation; 1.0 would mean completely disjoint.\n")

rows_to_try = [SEQ - 1, int(SEQ * 0.875), int(SEQ * 0.75), int(SEQ * 0.5)]
qd = []
for si, ids in enumerate(batches):
    grabbed.clear(); hidden.clear()
    model(input_ids=ids, use_cache=False)
    for li in PROBE_LAYERS:
        ref = budgets_at(li, rows_to_try[0])
        for r in rows_to_try[1:]:
            alt = budgets_at(li, r)
            l1 = float((ref - alt).abs().sum()) / (TOPK * n_kv)
            qd.append({"sample": si, "layer": li, "row": r, "l1": l1})
    torch.cuda.empty_cache()

print(f"{'layer':>6}" + "".join(f"{f'row={r}':>14}" for r in rows_to_try[1:]))
for li in PROBE_LAYERS:
    cells = []
    for r in rows_to_try[1:]:
        v = [x["l1"] for x in qd if x["layer"] == li and x["row"] == r]
        cells.append(sum(v) / len(v))
    print(f"{li:>6}" + "".join(f"{c:>14.3f}" for c in cells))
allv = [x["l1"] for x in qd]
print(f"\n  mean L1 fraction = {sum(allv)/len(allv):.3f}, max = {max(allv):.3f}")
out["query_dependence"] = qd

print()
print("=" * 78)
print("B. LENGTH DRIFT: refit as the context grows (proxy for long decode)")
print("=" * 78)
print("Budgets fitted on a prefix of length L, vs on the full 8192.\n")

drift = []
for si, ids in enumerate(batches[:2]):
    grabbed.clear(); hidden.clear()
    model(input_ids=ids, use_cache=False)
    for li in PROBE_LAYERS:
        full = budgets_at(li, SEQ - 1)
        for frac in [0.375, 0.5, 0.75]:
            r = int(SEQ * frac) - 1
            short = budgets_at(li, r)
            l1 = float((full - short).abs().sum()) / (TOPK * n_kv)
            drift.append({"sample": si, "layer": li, "frac": frac, "l1": l1})
    torch.cuda.empty_cache()

print(f"{'layer':>6}" + "".join(f"{f'L={int(SEQ*f)}':>12}" for f in [0.375, 0.5, 0.75]))
for li in PROBE_LAYERS:
    cells = []
    for f in [0.375, 0.5, 0.75]:
        v = [x["l1"] for x in drift if x["layer"] == li and x["frac"] == f]
        cells.append(sum(v) / len(v))
    print(f"{li:>6}" + "".join(f"{c:>12.3f}" for c in cells))
out["length_drift"] = drift

print()
print("=" * 78)
print("C. DOCUMENT DEPENDENCE: is the allocation a per-document or per-MODEL property?")
print("=" * 78)
print("If budgets barely differ across documents, they can be fitted ONCE offline and shipped")
print("as a constant table -- no prefill-time computation, no decode question at all.\n")

per_doc = {li: [] for li in PROBE_LAYERS}
for si, ids in enumerate(batches):
    grabbed.clear(); hidden.clear()
    model(input_ids=ids, use_cache=False)
    for li in PROBE_LAYERS:
        per_doc[li].append(budgets_at(li, SEQ - 1))
    torch.cuda.empty_cache()

print(f"{'layer':>6}{'cross-doc L1':>15}{'budgets (doc0)':>40}")
cross = []
for li in PROBE_LAYERS:
    bs = per_doc[li]
    ds = [float((bs[i] - bs[j]).abs().sum()) / (TOPK * n_kv)
          for i in range(len(bs)) for j in range(i + 1, len(bs))]
    m = sum(ds) / len(ds)
    cross.append(m)
    print(f"{li:>6}{m:>15.3f}   {bs[0].tolist()}")
print(f"\n  mean cross-document L1 fraction = {sum(cross)/len(cross):.3f}")
out["cross_document"] = {str(li): [b.tolist() for b in per_doc[li]] for li in PROBE_LAYERS}

for h in hs_handles:
    h.remove()
gm.pop(nm, None)
with open("scratch/diag_budget_stability.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nwrote scratch/diag_budget_stability.json")
