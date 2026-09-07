# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Verification 2: what does a backward pass over the sparse branch actually cost?

``e2e_trainer``'s docstring records that ``stage="sparse"`` backward goes through the *gather
reference* -- the Triton sparse kernel has no ``autograd.Function`` -- retaining
``O(Hkv * Sq * topk * D)`` per layer, measured ~39 GiB/layer at Sq=16384, topk=512. Over 36
layers that is past 1 TiB, so the memory arm cannot be trained on that path no matter how cheap
the memory module itself is.

``flex_attention`` should not have this problem: it is a real block-sparse kernel with a real
fused backward, so its retained state should be ``O(Sq * D)`` per layer like dense flash, with
the *sparsity* cutting time rather than inflating memory.

Measures both, per layer, at matched (Sq, topk), and reports peak allocated bytes and the
implied 36-layer total.
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kvpress.presses.gqa_indexer.qi_flex_attention import (  # noqa: E402
    deadlines,
    qi_block_mask,
    _flex,
)

H, Hkv, D = 32, 8, 128
GROUP = H // Hkv
N_LAYERS = 36
dev = "cuda"


def flex_step(Sq, topk, *, backward, n_sink=4, n_local=0):
    """One layer's flex block-sparse forward (+backward), returning peak MiB and ms."""
    torch.manual_seed(0)
    q = torch.randn(1, H, Sq, D, device=dev, dtype=torch.bfloat16, requires_grad=backward)
    k = torch.randn(1, Hkv, Sq, D, device=dev, dtype=torch.bfloat16, requires_grad=backward)
    v = torch.randn(1, Hkv, Sq, D, device=dev, dtype=torch.bfloat16, requires_grad=backward)
    scores = torch.randn(Hkv, Sq, device=dev, dtype=torch.float32)
    dl = deadlines(scores, topk, force_sink=n_sink, force_local=n_local)
    bm = qi_block_mask(
        dl, q_len=Sq, k_len=Sq, n_q_heads=H, force_sink=n_sink, force_local=n_local, device=dev
    )

    def step():
        out, lse = _flex()(
            q,
            k.repeat_interleave(GROUP, 1),
            v.repeat_interleave(GROUP, 1),
            block_mask=bm,
            scale=D**-0.5,
            return_lse=True,
        )
        if backward:
            (out.float().square().sum() + lse.square().sum()).backward()
            q.grad = k.grad = v.grad = None
        return out

    step()  # warm the compile
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(3):
        step()
    ev1.record()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    ms = ev0.elapsed_time(ev1) / 3
    del q, k, v, bm, dl, scores
    torch.cuda.empty_cache()
    return peak, ms


def gather_step(Sq, topk, *, backward):
    """
    The same support through the gather reference the sparse stage's backward actually uses.

    Written out here rather than imported so the measurement is unambiguous about what is being
    retained: `k[..., idx, :]` materializes (Hkv, Sq, topk, D), and autograd keeps it.
    """
    torch.manual_seed(0)
    q = torch.randn(1, H, Sq, D, device=dev, dtype=torch.bfloat16, requires_grad=backward)
    k = torch.randn(1, Hkv, Sq, D, device=dev, dtype=torch.bfloat16, requires_grad=backward)
    v = torch.randn(1, Hkv, Sq, D, device=dev, dtype=torch.bfloat16, requires_grad=backward)
    idx = torch.randint(0, Sq, (1, Hkv, Sq, topk), device=dev)

    def step():
        kg = torch.gather(
            k.unsqueeze(2).expand(1, Hkv, Sq, Sq, D), 3, idx.unsqueeze(-1).expand(-1, -1, -1, -1, D)
        )
        vg = torch.gather(
            v.unsqueeze(2).expand(1, Hkv, Sq, Sq, D), 3, idx.unsqueeze(-1).expand(-1, -1, -1, -1, D)
        )
        qg = q.view(1, Hkv, GROUP, Sq, D)
        logits = torch.einsum("bhgqd,bhqkd->bhgqk", qg, kg) * D**-0.5
        p = torch.softmax(logits.float(), -1).to(q.dtype)
        out = torch.einsum("bhgqk,bhqkd->bhgqd", p, vg)
        if backward:
            out.float().square().sum().backward()
            q.grad = k.grad = v.grad = None
        return out

    try:
        step()
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float("nan"), float("nan")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    try:
        step()
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return float("nan"), float("nan")
    ev1.record()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    ms = ev0.elapsed_time(ev1)
    del q, k, v, idx
    torch.cuda.empty_cache()
    return peak, ms


print(f"Qwen3-8B geometry: H={H}, Hkv={Hkv}, D={D}, {N_LAYERS} layers")
print()
print("flex_attention block-sparse, per layer (return_lse=True):")
print(f"{'Sq':>7} {'topk':>6} {'fwd MiB':>9} {'fwd ms':>8} {'fwd+bwd MiB':>12} {'bwd ms':>8} {'x36 GiB':>9}")
for Sq, keep in [(4096, 0.25), (8192, 0.25), (16384, 0.25), (32768, 0.25)]:
    topk = int(Sq * keep)
    try:
        f_mem, f_ms = flex_step(Sq, topk, backward=False)
        b_mem, b_ms = flex_step(Sq, topk, backward=True)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"{Sq:>7} {topk:>6}  OOM")
        continue
    print(
        f"{Sq:>7} {topk:>6} {f_mem:>9.1f} {f_ms:>8.2f} {b_mem:>12.1f} {b_ms:>8.2f} "
        f"{b_mem * N_LAYERS / 1024:>9.1f}"
    )

print()
print("gather reference (what stage='sparse' backward runs today), per layer:")
print(f"{'Sq':>7} {'topk':>6} {'fwd+bwd MiB':>12} {'ms':>8} {'x36 GiB':>9}")
for Sq, topk in [(2048, 512), (4096, 512), (8192, 512), (16384, 512)]:
    mem, ms = gather_step(Sq, topk, backward=True)
    if math.isnan(mem):
        print(f"{Sq:>7} {topk:>6}   OOM (single layer)")
        continue
    print(f"{Sq:>7} {topk:>6} {mem:>12.1f} {ms:>8.2f} {mem * N_LAYERS / 1024:>9.1f}")

print()
print("VERIFICATION 2 DONE")
