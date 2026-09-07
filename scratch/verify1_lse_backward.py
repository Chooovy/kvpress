# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Verification 1: is ``flex_attention(..., return_lse=True)``'s lse differentiable, correct,
and affordable?

The memory fusion ``o = (N_S + n) / (D_S + d)`` needs ``D_S`` inside the graph: the gradient
``do/dd = -o/(D_S+d)`` only exists if the denominator carries grad. ``flex_attention`` returns
``lse = log D_S + m`` (natural log, *after* the scale), so ``D_S = exp(lse)`` -- but only if
``lse`` is a differentiable output of the same kernel, and only if backward through *both*
``out`` and ``lse`` at once is correct.

Compared against an fp64 masked reference (softmax + logsumexp written out longhand).
"""

from __future__ import annotations

import math
import time

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

torch.manual_seed(0)
dev = "cuda"

B, H, Hkv, Sq, Sk, D = 1, 8, 2, 512, 512, 64
group = H // Hkv
scale = 1.0 / math.sqrt(D)

# A block-sparse causal mask with holes, so we exercise the same code path the memory arm will:
# some keys are dropped from some rows (that is what eviction is).
dl = torch.randint(0, Sk, (Hkv, Sk), device=dev, dtype=torch.int32)


def mask_mod(b, h, q_i, k_j):
    return (k_j <= q_i) & (q_i <= dl[h // group, k_j])


block_mask = create_block_mask(mask_mod, B=None, H=H, Q_LEN=Sq, KV_LEN=Sk, device=dev)

flex = torch.compile(flex_attention, dynamic=None)


def make(dtype, req=True):
    q = torch.randn(B, H, Sq, D, device=dev, dtype=dtype, requires_grad=req)
    k = torch.randn(B, H, Sk, D, device=dev, dtype=dtype, requires_grad=req)
    v = torch.randn(B, H, Sk, D, device=dev, dtype=dtype, requires_grad=req)
    return q, k, v


def reference(q, k, v):
    """fp64 masked softmax + logsumexp, materialised. Ground truth for out and lse."""
    qd, kd, vd = q.double(), k.double(), v.double()
    logits = (qd @ kd.transpose(-1, -2)) * scale  # (B,H,Sq,Sk)
    q_i = torch.arange(Sq, device=dev).view(1, 1, Sq, 1)
    k_j = torch.arange(Sk, device=dev).view(1, 1, 1, Sk)
    keep = (k_j <= q_i) & (q_i <= dl.repeat_interleave(group, 0).view(1, H, 1, Sk))
    logits = logits.masked_fill(~keep, float("-inf"))
    lse = torch.logsumexp(logits, dim=-1)
    out = torch.softmax(logits, dim=-1) @ vd
    return out, lse


print("=" * 78)
print("1a. does lse require grad, and is its VALUE right? (fp32 inputs, fp64 reference)")
q, k, v = make(torch.float32)
out, lse = flex(q, k, v, block_mask=block_mask, scale=scale, return_lse=True)
print(f"   out {tuple(out.shape)} {out.dtype} requires_grad={out.requires_grad}")
print(f"   lse {tuple(lse.shape)} {lse.dtype} requires_grad={lse.requires_grad}")
ref_out, ref_lse = reference(q, k, v)
# rows with no unmasked key are -inf in the reference; flex fills them differently. Exclude.
live = ref_lse > -1e30
print(f"   live rows: {live.sum().item()}/{live.numel()}")
print(f"   |out - ref| max = {(out.double() - ref_out)[live].abs().max().item():.3e}")
print(f"   |lse - ref| max = {(lse.double() - ref_lse)[live].abs().max().item():.3e}")

print()
print("=" * 78)
print("1b. is d(lse)/d(q,k) CORRECT? (grad only through lse, fp64 reference)")
g = torch.randn_like(lse)
gq, gk, gv = torch.autograd.grad((lse * g).sum(), [q, k, v], retain_graph=True, allow_unused=True)
q2, k2, v2 = q.detach().double().requires_grad_(), k.detach().double().requires_grad_(), v.detach().double().requires_grad_()
_, rlse = reference(q2, k2, v2)
rq, rk, rv = torch.autograd.grad((rlse.masked_fill(~live, 0.0) * g.double()).sum(), [q2, k2, v2], allow_unused=True)


def cmp(name, a, b):
    if a is None:
        print(f"   d{name}: None (flex returned no grad); reference max |g| = "
              f"{0.0 if b is None else b.abs().max().item():.3e}")
        return
    if b is None:
        b = torch.zeros_like(a)
    num = (a.double() - b).abs().max().item()
    den = b.abs().max().item()
    print(f"   d{name}: max abs err {num:.3e}, ref scale {den:.3e}, rel {num / max(den, 1e-30):.3e}")


cmp("q", gq, rq)
cmp("k", gk, rk)
cmp("v", gv, rv)  # must be ~0: lse does not depend on v

print()
print("=" * 78)
print("1c. backward through out AND lse together (the configuration we actually need)")
q, k, v = make(torch.float32)
out, lse = flex(q, k, v, block_mask=block_mask, scale=scale, return_lse=True)
go, gl = torch.randn_like(out), torch.randn_like(lse)
gq, gk, gv = torch.autograd.grad((out * go).sum() + (lse * gl).sum(), [q, k, v])
q2, k2, v2 = (t.detach().double().requires_grad_() for t in (q, k, v))
ro, rl = reference(q2, k2, v2)
rq, rk, rv = torch.autograd.grad(
    (ro * go.double()).sum() + (rl.masked_fill(~live, 0.0) * gl.double()).sum(), [q2, k2, v2]
)
cmp("q", gq, rq)
cmp("k", gk, rk)
cmp("v", gv, rv)

print()
print("=" * 78)
print("1d. same, bf16 inputs (what training actually runs in)")
q, k, v = make(torch.bfloat16)
out, lse = flex(q, k, v, block_mask=block_mask, scale=scale, return_lse=True)
print(f"   out {out.dtype}, lse {lse.dtype} requires_grad={lse.requires_grad}")
go, gl = torch.randn_like(out), torch.randn_like(lse)
gq, gk, gv = torch.autograd.grad((out * go).sum() + (lse * gl).sum(), [q, k, v])
q2, k2, v2 = (t.detach().double().requires_grad_() for t in (q, k, v))
ro, rl = reference(q2, k2, v2)
rq, rk, rv = torch.autograd.grad(
    (ro * go.double()).sum() + (rl.masked_fill(~live, 0.0) * gl.double()).sum(), [q2, k2, v2]
)
cmp("q", gq, rq)
cmp("k", gk, rk)
cmp("v", gv, rv)

print()
print("=" * 78)
print("1e. cost of return_lse=True: time and peak memory, fwd and fwd+bwd")


def bench(return_lse, backward, iters=20):
    q, k, v = make(torch.bfloat16)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    def step():
        r = flex(q, k, v, block_mask=block_mask, scale=scale, return_lse=return_lse)
        if return_lse:
            o, l = r
            loss = o.float().square().sum() + l.float().square().sum()
        else:
            loss = r.float().square().sum()
        if backward:
            loss.backward()
            q.grad = k.grad = v.grad = None

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters * 1e3
    return dt, torch.cuda.max_memory_allocated() / 2**20


for backward in (False, True):
    for rl in (False, True):
        dt, mem = bench(rl, backward)
        tag = "fwd+bwd" if backward else "fwd    "
        print(f"   {tag}  return_lse={str(rl):5s}  {dt:7.3f} ms   peak {mem:8.1f} MiB")

print()
print("VERIFICATION 1 DONE")
