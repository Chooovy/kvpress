# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Locate the NaN: hook every memory tensor at the gamma where training dies."""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kvpress.presses.gqa_indexer import memory_lm_step  # noqa: E402
from kvpress.presses.gqa_indexer.memory_trainer import MemoryTrainer  # noqa: E402
from kvpress.presses.gqa_indexer.press import GQAIndexerPress  # noqa: E402
from kvpress.presses.gqa_indexer.train import (  # noqa: E402
    load_indexer_state_dict,
    load_memory_state_dict,
    press_kwargs_from_checkpoint,
)

MODEL = "/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B"
ROUTER = (
    "/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_scalar/"
    "stage1_16k_mid256_longce/final.pt"
)
MEM = "/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_memory/longce_8k_r16/step200.pt"
L = 8192

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MODEL)
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
model = model.to("cuda").eval()
model.requires_grad_(False)

rck = torch.load(ROUTER, map_location="cpu", weights_only=False)
rsd, rcfg = rck["indexer"], rck.get("config", {})
_, kw = press_kwargs_from_checkpoint(rsd, rcfg)
press = GQAIndexerPress(
    compression_ratio=0.75, scorer="scalar", gate_scale=any("gate_scale" in k for k in rsd),
    n_sink=4, n_local=64, memory=True, memory_rank=16, **kw,
)
press.post_init_from_model(model)
load_indexer_state_dict(model, rsd)
mck = torch.load(MEM, map_location="cpu", weights_only=False)
# step200 predates v_norm/L1 normalization, so its weights no longer load -- fresh init instead.
# load_memory_state_dict(model, mck["memory"])

trainer = MemoryTrainer(press=press, keep_ratio=0.25, force_sink=4, force_local=64, measure=True)

# Push gamma to just past where it died (~2.4e-5 -> log_gamma ~ -10.6).
for layer in model.model.layers:
    layer.self_attn.kv_memory.log_gamma.data.fill_(-10.0)

# Instrument: wrap the trainer's memory_terms and fusion to report the first non-finite tensor.
orig_terms = MemoryTrainer.memory_terms
reported = {"done": False}


def check(name, t, layer):
    if t is None or reported["done"]:
        return
    if not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).sum().item()
        print(f"  !! FIRST NON-FINITE: {name} at layer {layer}: {bad}/{t.numel()} entries, "
              f"absmax(finite)={t[torch.isfinite(t)].abs().max() if torch.isfinite(t).any() else 'none'}")
        reported["done"] = True


def patched_terms(self, memory, query, key, value, deadline, *, scores, q_len, group):
    from kvpress.presses.gqa_indexer.memory_schedule import block_memory_states

    layer = self.layers_fused
    H, z, W, counts = block_memory_states(
        memory, key, value, deadline, q_len=q_len, block=self.block,
        n_local=self.force_local, scores=scores,
    )
    check("psi(k)", memory.psi(key), layer)
    check("H", H, layer)
    check("z", z, layer)
    check("W", W, layer)
    q_kv = query.view(query.shape[0], -1, group, q_len, query.shape[-1]).mean(2)
    check("phi(q)", memory.phi(q_kv), layer)
    n, d = self._read_per_block(memory, q_kv, H, z, W, counts, block=self.block)
    check("n", n, layer)
    check("d", d, layer)
    if True:
        print(f"  layer {layer:2d}: W[min,max]=[{float(W.min()):.3e},{float(W.max()):.3e}] "
              f"z.min={float(z.min()):.3e} H.absmax={float(H.abs().max()):.3e} "
              f"d[min,max]=[{float(d.min()):.3e},{float(d.max()):.3e}] "
              f"n.absmax={float(n.abs().max()):.3e} counts.max={int(counts.max())}")
    return n, d


MemoryTrainer.memory_terms = patched_terms

torch.manual_seed(0)
input_ids = torch.randint(1000, 40000, (1, L), device="cuda")
print(f"forward at log_gamma=-10 (gamma={float(torch.exp(torch.tensor(-10.0))):.3e}), L={L}")
loss, stats = memory_lm_step(model, trainer, input_ids=input_ids)
print(f"\nloss = {float(loss):.6f}  stats = {stats}")
print(f"loss finite: {bool(torch.isfinite(loss))}")

if torch.isfinite(loss):
    loss.backward()
    grads = {
        n: p.grad for n, p in model.named_parameters()
        if p.grad is not None and "kv_memory" in n
    }
    bad = [n for n, g in grads.items() if not torch.isfinite(g).all()]
    print(f"\nbackward: {len(grads)} memory grads, {len(bad)} non-finite")
    for n in bad[:8]:
        print("   ", n)
