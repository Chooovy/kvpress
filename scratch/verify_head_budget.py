# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end checks for head_budget="mass" on the real trained router."""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kvpress.presses.gqa_indexer.press import GQAIndexerPress, get_language_model  # noqa: E402
from kvpress.presses.gqa_indexer.sparse_inference import SparseAttentionContext  # noqa: E402
from kvpress.presses.gqa_indexer.train import (  # noqa: E402
    load_indexer_state_dict,
    press_kwargs_from_checkpoint,
)

MODELS = "/apdcephfs_gy8/share_303843174/guhao/models"
CKPT = f"{MODELS}/Qwen-3-8B-gqa_indexer_scalar/fwkl_ce01_8k_local128_b256_decay/final.pt"
dev = "cuda:0"
torch.set_grad_enabled(False)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(f"{MODELS}/Qwen3-8B")
model = (
    AutoModelForCausalLM.from_pretrained(
        f"{MODELS}/Qwen3-8B", torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    .to(dev)
    .eval()
)
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
state = ck.get("indexer", ck)
scorer, kw = press_kwargs_from_checkpoint(state, ck.get("config") or {})
press = GQAIndexerPress(
    compression_ratio=0.0,
    gate_scale=any(str(x).endswith("gate_scale") for x in state),
    scorer_attr="indexer",
    scorer=scorer,
    **kw,
)
press.post_init_from_model(model, force_reinit=True)
load_indexer_state_dict(model, state, "indexer")

torch.manual_seed(0)
ids = torch.randint(1000, 100000, (1, 6000), device=dev)

from transformers import DynamicCache  # noqa: E402


def run(**kwargs):
    with SparseAttentionContext(
        model, press, topk=2048, force_sink=4, force_local=128, **kwargs
    ) as ctx:
        cache = DynamicCache()
        out = model.model(input_ids=ids, past_key_values=cache)
        return out.last_hidden_state.float(), dict(ctx._head_topk)


print("=== T1: head_budget='uniform' unchanged (default path) ===")
h_a, b_a = run()
h_b, b_b = run(head_budget="uniform")
print(f"  max|diff| = {(h_a - h_b).abs().max():.3e}   budgets cached: {len(b_a)}, {len(b_b)}")
assert torch.equal(h_a, h_b), "uniform path changed!"
assert not b_a and not b_b, "uniform should cache no budgets"
print("  BITWISE IDENTICAL, no allocation performed  OK")

print()
print("=== T2: head_budget='mass' runs, conserves budget on every layer ===")
h_m, b_m = run(head_budget="mass")
print(f"  layers allocated: {len(b_m)}")
bad = [(li, int(v.sum())) for li, v in b_m.items() if int(v.sum()) != 2048 * 8]
print(f"  layers whose budget != {2048*8}: {bad if bad else 'none'}")
assert not bad, bad
print(f"  output finite: {bool(torch.isfinite(h_m).all())}")
assert torch.isfinite(h_m).all()
print(f"  differs from uniform: max|diff| = {(h_m - h_a).abs().max():.3e}")
assert not torch.equal(h_m, h_a), "mass produced identical output -- allocation is a no-op!"

print()
print("=== T3: the allocation is actually non-uniform ===")
for li in [0, 2, 17, 35]:
    v = b_m[li]
    print(f"  layer {li:2d}: {v.tolist()}  min {int(v.min())} max {int(v.max())}")

print()
print("=== T4: floor raises the starved heads ===")
h_f, b_f = run(head_budget="mass", head_budget_floor=256)
bad = [(li, int(v.sum())) for li, v in b_f.items() if int(v.sum()) != 2048 * 8]
assert not bad, bad
mins_nofloor = min(int(v.min()) for v in b_m.values())
mins_floor = min(int(v.min()) for v in b_f.values())
print(f"  min budget over all layers: no-floor {mins_nofloor}, floor=256 {mins_floor}")
print(f"  all layers still conserve: {not bad}")
assert mins_floor >= 256 + 4 + 128, mins_floor
print("  OK")

print()
print("=== T5: decode agrees with the prefill's allocation ===")
with SparseAttentionContext(
    model, press, topk=2048, force_sink=4, force_local=128, head_budget="mass"
) as ctx:
    cache = DynamicCache()
    model.model(input_ids=ids, past_key_values=cache)
    committed = {li: v.clone() for li, v in ctx._head_topk.items()}
    nxt = torch.randint(1000, 100000, (1, 1), device=dev)
    o = model.model(input_ids=nxt, past_key_values=cache)
    same = all(torch.equal(committed[li], ctx._head_topk[li]) for li in committed)
    print(f"  decode output finite: {bool(torch.isfinite(o.last_hidden_state).all())}")
    print(f"  budgets unchanged by decode: {same}")
    assert same and torch.isfinite(o.last_hidden_state).all()
print("  OK")

print()
print("=== T6: reset() clears budgets between documents ===")
with SparseAttentionContext(
    model, press, topk=2048, force_sink=4, force_local=128, head_budget="mass"
) as ctx:
    model.model(input_ids=ids, past_key_values=DynamicCache())
    n_before = len(ctx._head_topk)
    ctx.reset()
    print(f"  cached before reset {n_before}, after {len(ctx._head_topk)}")
    assert n_before > 0 and len(ctx._head_topk) == 0
print("  OK")
print("\nALL PASS")
