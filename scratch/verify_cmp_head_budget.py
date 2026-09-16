# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
CMP slots x per-head budget: do the two compose without double-counting or losing keys?

Before this, the CMP branches computed their deadline from the scalar ``topk - cmp_slots`` and
never consulted ``_budget_for``, so enabling both silently discarded the allocation. Now both go
through ``_cmp_take``. Three things have to hold, and each failure is invisible in the output:

1. **CMP alone is unchanged.** ``head_budget="uniform"`` must reproduce the pre-change CMP path
   bitwise, or every CMP number already collected is invalidated.
2. **The budget is still conserved.** The exact branch reads ``budget_h - R`` and the slots add
   ``R``, so the per-head read count must be exactly ``budget_h`` and the layer total exactly
   ``topk * n_kv``. An unmatched budget would measure the budget rather than the idea.
3. **The two branches share ONE deadline.** The slots summarize what the exact branch dropped. If
   the build and the attend disagreed, a key would be double-counted in the softmax or dropped by
   both -- and the logits would still look reasonable.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kvpress.presses.gqa_indexer.press import GQAIndexerPress  # noqa: E402
from kvpress.presses.gqa_indexer.sparse_inference import SparseAttentionContext  # noqa: E402
from kvpress.presses.gqa_indexer.train import (  # noqa: E402
    load_indexer_state_dict,
    press_kwargs_from_checkpoint,
)

MODELS = "/apdcephfs_gy8/share_303843174/guhao/models"
CKPT = f"{MODELS}/Qwen-3-8B-gqa_indexer_scalar/rvkl_8k_local128_b256_decay/step300.pt"
dev = "cuda:6"
TOPK, FS, FL = 2048, 4, 128
torch.set_grad_enabled(False)

from transformers import AutoModelForCausalLM, DynamicCache  # noqa: E402

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

torch.manual_seed(0)
ids = torch.randint(1000, 100000, (1, 6000), device=dev)


def run(**kwargs):
    with SparseAttentionContext(
        model, press, topk=TOPK, force_sink=FS, force_local=FL, **kwargs
    ) as ctx:
        out = model.model(input_ids=ids, past_key_values=DynamicCache())
        return out.last_hidden_state.float(), dict(ctx._head_topk)


print("=== T1: CMP under uniform is unchanged (protects existing CMP numbers) ===")
a, ba = run(cmp_slots=64)
b, bb = run(cmp_slots=64, head_budget="uniform")
print(f"  max|diff| = {(a - b).abs().max():.3e}   budgets cached: {len(ba)}, {len(bb)}")
assert torch.equal(a, b), "CMP+uniform changed!"
assert not ba and not bb, "uniform must cache no budgets"
print("  BITWISE IDENTICAL  OK")

print()
print("=== T2: CMP + mass runs and conserves the budget per layer ===")
c, bc = run(cmp_slots=64, head_budget="mass", head_budget_floor=512)
print(f"  layers allocated: {len(bc)}")
bad = [(li, int(v.sum())) for li, v in bc.items() if int(v.sum()) != TOPK * 8]
print(f"  layers whose budget != {TOPK*8}: {bad if bad else 'none'}")
assert not bad, bad
assert torch.isfinite(c).all(), "non-finite output"
print(f"  output finite: True")
print(f"  differs from CMP+uniform: max|diff| = {(c - a).abs().max():.3e}")
assert not torch.equal(c, a), "mass had no effect -- CMP still overriding the allocation!"
print("  the allocation IS taking effect under CMP  OK")

print()
print("=== T3: per-head read count == budget_h exactly (slots funded out of it) ===")
from kvpress.presses.gqa_indexer.qi_flex_attention import deadlines  # noqa: E402

R = 64
for trial in range(4):
    # S must exceed the LARGEST budget, or a head legitimately reads its whole causal history and
    # the read count is capped by the sequence rather than by the budget -- not a bug, but it
    # tests nothing. (Hit exactly that with S=3000 and a 3736 budget.)
    H, S = 8, 5000
    budget = torch.tensor([700, 1200, 2600, 3400, 900, 1800, 2048, 3736], device=dev)
    assert int(budget.sum()) == TOPK * H, int(budget.sum())
    qv = torch.randn(1, H, 1, 32, device=dev)
    k = torch.randn(1, S, 32, device=dev)
    sc = torch.einsum("bhqd,bkd->bhk", qv, k)[0].contiguous()
    dl = deadlines(sc, budget - R, force_sink=FS, force_local=FL)
    row = S - 1
    limit, horizon = row, row - FL
    ok = True
    for h in range(H):
        n = sum(
            1 for j in range(S)
            if j <= limit and (j < FS or j > limit - FL
                               or (j <= horizon and horizon <= int(dl[h, j])))
        )
        want = int(budget[h]) - R
        if n != want:
            ok = False
            print(f"  trial{trial} h{h}: exact branch reads {n}, expected {want}")
    if trial == 0:
        print(f"  exact-branch reads (budget_h - R): {'all correct' if ok else 'MISMATCH'}")
        print(f"  + R={R} slots each  =>  total per head == budget_h, layer total "
              f"{int((budget).sum())} == topk*H {TOPK*H}")
    assert ok

print()
print("=== T4: R too large for the SMALLEST head is refused (not silently wrong) ===")
try:
    run(cmp_slots=1200, head_budget="mass", head_budget_floor=0)
    print("  NO ERROR RAISED -- the per-head floor guard did not fire")
    raise SystemExit(1)
except ValueError as exc:
    msg = str(exc)
    assert "smallest head" in msg, msg
    print(f"  refused: {msg[:140]}...")
    print("  OK")

print("\nALL PASS")
