# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU smoke: does the fused forward run, does it start at the eviction baseline, does it train?"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kvpress.presses.gqa_indexer.memory_trainer import MemoryTrainer, memory_lm_step  # noqa: E402
from kvpress.presses.gqa_indexer.press import GQAIndexerPress  # noqa: E402
from kvpress.presses.gqa_indexer.train import (  # noqa: E402
    load_indexer_state_dict,
    memory_state_dict,
    press_kwargs_from_checkpoint,
)

MODEL = "/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B"
CKPT = (
    "/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_scalar/"
    "stage1_16k_mid256_longce/final.pt"
)
SEQ = 4096
KEEP = 0.25

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL)
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
model = model.to("cuda").eval()
model.requires_grad_(False)

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
sd, cfg = ckpt["indexer"], ckpt.get("config", {})
scorer, kw = press_kwargs_from_checkpoint(sd, cfg)
print("scorer:", scorer, kw)

press = GQAIndexerPress(
    compression_ratio=1.0 - KEEP,
    scorer="scalar",
    gate_scale=any("gate_scale" in k for k in sd),
    n_sink=4,
    memory=True,
    memory_rank=16,
    **kw,
)
press.post_init_from_model(model)
load_indexer_state_dict(model, sd)

trainer = MemoryTrainer(press=press, keep_ratio=KEEP, force_sink=4, force_local=64, measure=True)
groups = trainer.parameter_groups(model, kernel_lr=1e-3, scalar_lr=0.05)
n_params = sum(p.numel() for g in groups for p in g["params"])
print(f"memory params: {n_params/1e6:.2f}M in {len(groups)} groups")
print("trainable check:", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, "M")

torch.manual_seed(0)
input_ids = torch.randint(1000, 40000, (1, SEQ), device="cuda")

# 1. Does the fused forward equal the plain eviction baseline at initialization?
from kvpress.presses.gqa_indexer.qi_flex_attention import qi_sparse_attention  # noqa: E402

with torch.no_grad():
    loss_mem, stats = memory_lm_step(model, trainer, input_ids=input_ids)
print(f"\nfused loss at init: {float(loss_mem):.6f}  stats={ {k: round(v, 8) for k, v in stats.items()} }")

# Same forward with the memory off entirely (gamma -> 0 exactly), as the reference.
for m in trainer.memory_modules(model):
    m.log_gamma.data.fill_(-100.0)
with torch.no_grad():
    loss_off, _ = memory_lm_step(model, trainer, input_ids=input_ids)
print(f"memory hard-off loss:  {float(loss_off):.6f}   delta = {abs(float(loss_mem)-float(loss_off)):.3e}")
for m in trainer.memory_modules(model):
    m.log_gamma.data.fill_(-18.0)

# 2. Does a backward pass fit and produce gradients?
torch.cuda.reset_peak_memory_stats()
opt = torch.optim.AdamW(groups)
losses = []
for step in range(12):
    loss, stats = memory_lm_step(model, trainer, input_ids=input_ids)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(
        [p for g in groups for p in g["params"]], 1.0
    )
    opt.step()
    losses.append(float(loss))
    print(
        f"  step {step}: loss {float(loss):.4f} gnorm {float(gnorm):.3e} "
        f"mass_share {stats.get('mass_share', 0):.3e} gamma {stats.get('gamma', 0):.3e}"
    )
print(f"peak memory: {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")

msd = memory_state_dict(model)
print(f"memory_state_dict: {len(msd)} tensors, "
      f"{sum(t.numel() for t in msd.values())/1e6:.2f}M params")
print("SMOKE OK")
