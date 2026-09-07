# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Why does the trained memory score 4.00 when training reported mass_share 1.7e-2?

Runs the TRAINING forward (streaming schedule) and the INFERENCE forward (oneshot, what eval uses)
over the same input and the same weights, and compares the memory's actual softmax mass share per
layer. A large gap means the two paths do not describe the same model, which is a train/eval
mismatch rather than a weak result.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kvpress.presses.gqa_indexer.memory import memory_mass_share
from kvpress.presses.gqa_indexer.memory_trainer import MemoryTrainer
from kvpress.presses.gqa_indexer.press import GQAIndexerPress
from kvpress.presses.gqa_indexer.qi_flex_attention import _flex, deadlines, qi_block_mask
from kvpress.presses.gqa_indexer.train import (
    load_indexer_state_dict,
    load_memory_state_dict,
    press_kwargs_from_checkpoint,
)

MODEL = "/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B"
ROUTER = (
    "/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_scalar/"
    "stage1_16k_mid256_longce/final.pt"
)
MEM = "/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_memory/longce_8k_g7/final.pt"
L, TOPK, SINK, LOCAL = 8192, 2048, 4, 64

from transformers import AutoModelForCausalLM

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
    n_sink=SINK, n_local=LOCAL, memory=True, memory_rank=16, **kw,
)
press.post_init_from_model(model)
load_indexer_state_dict(model, rsd)
load_memory_state_dict(model, torch.load(MEM, map_location="cpu", weights_only=False)["memory"])

gammas = torch.stack([l.self_attn.kv_memory.log_gamma for l in model.model.layers])
print(f"loaded log_gamma: mean {float(gammas.mean()):+.3f} "
      f"min {float(gammas.min()):+.3f} max {float(gammas.max()):+.3f}")
print(f"          gamma : mean {float(gammas.exp().mean()):.3e} max {float(gammas.exp().max()):.3e}")
print()

# Capture one layer's q/k/v + hidden, then run both memory paths over them.
grab: dict = {}
WANT = 18


def impl(module, q, k, v, am, scaling=None, dropout=0.0, **kwargs):
    import torch.nn.functional as F

    if int(module.layer_idx) == WANT:
        grab.update(q=q.detach(), k=k.detach(), v=v.detach(), scaling=scaling)
    g = q.shape[1] // k.shape[1]
    o = F.scaled_dot_product_attention(
        q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1), is_causal=True, scale=scaling
    )
    return o.transpose(1, 2).contiguous(), None


from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

ALL_ATTENTION_FUNCTIONS.register("probe_gap", impl)
hidden: dict = {}


def pre(module, args, kwargs):
    if int(getattr(module, "layer_idx", -1)) == WANT:
        hs = kwargs.get("hidden_states")
        if hs is None and args:
            hs = args[0]
        hidden["h"] = hs.detach()
    return None


handles = [l.self_attn.register_forward_pre_hook(pre, with_kwargs=True) for l in model.model.layers]
model.config._attn_implementation = "probe_gap"
torch.manual_seed(0)
with torch.no_grad():
    model(input_ids=torch.randint(1000, 40000, (1, L), device="cuda"), use_cache=False)
for h in handles:
    h.remove()

q, k, v, scaling = grab["q"], grab["k"], grab["v"], grab["scaling"]
memory = model.model.layers[WANT].self_attn.kv_memory
indexer = press.get_indexer(model.model.layers[WANT].self_attn)
H_q, Hkv = q.shape[1], k.shape[1]
group = H_q // Hkv

with torch.no_grad():
    scores = indexer.score_keys(hidden["h"])[0].float()
    dl = deadlines(scores, TOPK, force_sink=SINK, force_local=LOCAL)
    bm = qi_block_mask(dl, q_len=L, k_len=L, n_q_heads=H_q, force_sink=SINK,
                       force_local=LOCAL, device=q.device)
    o_s, lse_s = _flex()(q, k.repeat_interleave(group, 1), v.repeat_interleave(group, 1),
                         block_mask=bm, scale=scaling, return_lse=True)

    trainer = MemoryTrainer(press=press, keep_ratio=0.25, force_sink=SINK, force_local=LOCAL)

    print(f"layer {WANT}: comparing the two schedules on identical inputs")
    # training path
    trainer.schedule = "streaming"
    n_tr, d_tr = trainer.memory_terms(memory, q, k, v, dl, scores=scores, q_len=L, group=group)
    sh = memory_mass_share(lse_s, d_tr, group=group)
    print(f"  TRAIN (streaming): mass_share mean {float(sh.mean()):.4f} max {float(sh.max()):.4f}")

    # inference path, as SparseAttentionContext now runs it for a prefill
    from kvpress.presses.gqa_indexer.memory_schedule import block_memory_states
    from kvpress.presses.gqa_indexer.qi_flex_attention import FLEX_BLOCK
    Hs, zs, Ws, cnts = block_memory_states(
        memory, k, v, dl, q_len=L, block=FLEX_BLOCK, n_local=LOCAL, scores=scores
    )
    q_kv = q.view(q.shape[0], Hkv, group, L, q.shape[-1]).mean(2)
    n_ev, d_ev = MemoryTrainer._read_per_block(memory, q_kv, Hs, zs, Ws, cnts, block=FLEX_BLOCK)
    sh2 = memory_mass_share(lse_s, d_ev, group=group)
    print(f"  EVAL  (prefill)  : mass_share mean {float(sh2.mean()):.4f} max {float(sh2.max()):.4f}")
    print(f"  max |d_train - d_eval| = {float((d_tr - d_ev).abs().max()):.3e}  "
          f"(0 means the two paths are now the SAME model)")

    # The causality question for oneshot: does the state include keys a given row cannot see?
    enter = dl.to(torch.int64) + 1
    ingested = enter <= L - 1
    print()
    print(f"  oneshot ingests {int(ingested.sum(-1).float().mean())} keys per head (of {L})")
    print("  For query row t, keys > t are in its FUTURE. Rows that read a state containing them:")
    for t in (0, 128, 1024, 4096):
        future = (ingested & (torch.arange(L, device=q.device) > t)).sum(-1).float().mean()
        print(f"    row {t:5d}: {int(future)} of the ingested keys are at positions > t")
