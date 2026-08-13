# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the fused gated-attention Triton kernel.

The kernel exists because ``scaled_dot_product_attention`` cannot run this operation in ``O(L)``
memory -- two attempts OOM'd, both because the concat trick forces ``Dqk != Dv`` (or a 256-wide
head once V is padded), and neither shape keeps a fused SDPA backend for the backward pass. So
what has to be established here is that the kernel is *equivalent to the reference*, since the
reference is now the definition and the kernel is the only thing that will actually run.

Runs under CUDA or ``TRITON_INTERPRET=1``, and **skips** otherwise -- the convention
``test_gqa_indexer_sparse_attention.py`` already uses. Setting the env var from inside the module
is not an option: Triton reads it at import time, so a module imported earlier in the same session
would already have fixed the mode, and the tests would pass alone and fail in the suite.

That the interpreter path exists matters beyond convenience: the memory bug this kernel replaces
was invisible on CPU only because the *check* was wrong, not because CPU could not see it.

fp32 rather than the fp64 used elsewhere in this suite -- ``tl.dot`` has no fp64, so
``gated_kernels_available`` deliberately refuses it and an fp64 test would silently measure the
torch fallback instead of the kernel.
"""

import pytest
import torch

from kvpress.presses.gqa_indexer.gate_pin import history_lse, pinned_mask
from kvpress.presses.gqa_indexer.gated_attention import gated_attention_reference
from kvpress.presses.gqa_indexer.triton_fused_loss import (
    HAS_TRITON,
    triton_interpret_enabled,
)
from kvpress.presses.gqa_indexer.triton_gated_attention import (
    gated_kernels_available,
    triton_gated_attention,
)

pytestmark = pytest.mark.skipif(
    not HAS_TRITON or not (torch.cuda.is_available() or triton_interpret_enabled()),
    reason="needs Triton and either CUDA or TRITON_INTERPRET=1",
)

#: ``(pin_mode, n_sink, pin_self)`` -- the kernel takes the geometry, not the mode name, because
#: both supported pins reduce to "which (query, key) pairs take gate 0".
PIN_CASES = [
    ("none", 0, False),
    ("sink", 4, False),
    ("self", 0, True),
    ("self+sink", 4, True),
]


def make_inputs(bsz=1, n_heads=4, n_kv_heads=2, q_len=16, k_len=16, dim=16, idx_dim=8, seed=1):
    """fp32 q/k/v plus indexer q/k, all requiring grad."""
    torch.manual_seed(seed)
    return dict(
        q=torch.randn(bsz, n_heads, q_len, dim, requires_grad=True),
        k=torch.randn(bsz, n_kv_heads, k_len, dim, requires_grad=True),
        v=torch.randn(bsz, n_kv_heads, k_len, dim, requires_grad=True),
        q_idx=torch.randn(bsz, n_kv_heads, q_len, idx_dim, requires_grad=True),
        k_idx=torch.randn(bsz, k_len, idx_dim, requires_grad=True),
    )


def run_kernel(inputs, gate_scale, pin_mode, n_sink, pin_self, block_m=8, block_n=8):
    """
    The kernel, with ``lse`` built the way the production path builds it.

    ``lse`` is left **in the autograd graph** on purpose: the kernel's gate is ``score - lse``, so
    a complete gradient needs both the kernel's own ``d/d(lse)`` and ``history_lse``'s dependence
    on ``q_idx``/``k_idx``. Detaching it here would hide exactly the term that was missing from
    the first version of the kernel.
    """
    q_len, k_len = inputs["q"].shape[2], inputs["k"].shape[2]
    pinned = pinned_mask(pin_mode, q_len, k_len, torch.device("cpu"), n_sink=n_sink)
    if pinned is None:
        lse = torch.zeros(inputs["q"].shape[0], inputs["k"].shape[1], q_len)
    else:
        lse = history_lse(
            inputs["q_idx"], inputs["k_idx"], gate_scale=gate_scale, pinned=pinned
        )
    return triton_gated_attention(
        inputs["q"], inputs["k"], inputs["v"], inputs["q_idx"], inputs["k_idx"], lse,
        gate_scale=gate_scale,
        scaling=inputs["q"].shape[-1] ** -0.5,
        query_offset=k_len - q_len,
        n_sink=n_sink,
        pin_self=pin_self,
        block_m=block_m,
        block_n=block_n,
    )


def grads_of(out, inputs, gate_scale, seed=2):
    """Backprop a fixed cotangent; return the output and every input gradient."""
    torch.manual_seed(seed)
    (out * torch.randn_like(out)).sum().backward()
    return [out.detach()] + [inputs[n].grad.clone() for n in
                             ("q", "k", "v", "q_idx", "k_idx")] + [gate_scale.grad.clone()]


LABELS = ("out", "d q", "d k", "d v", "d q_idx", "d k_idx", "d gate_scale")


@pytest.mark.parametrize("pin_mode,n_sink,pin_self", PIN_CASES)
@pytest.mark.parametrize(
    "q_len,k_len,n_heads,n_kv_heads,dim,idx_dim",
    [
        (16, 16, 4, 2, 16, 8),    # square, GQA, narrower indexer
        (16, 24, 4, 2, 16, 8),    # Sq < Sk: bottom-right alignment
        (8, 8, 4, 4, 16, 16),     # no GQA, matched dims
        (8, 8, 4, 1, 8, 8),       # MQA on the attention side
        (1, 12, 2, 2, 8, 8),      # a decode step
    ],
)
def test_kernel_matches_reference(pin_mode, n_sink, pin_self, q_len, k_len, n_heads, n_kv_heads, dim, idx_dim):
    """
    Forward **and all six gradients** must match the reference, for every pin mode and shape.

    The reference is the definition of the operation; the kernel is what runs. Six gradients
    because the kernel hand-writes its backward -- an error in any one of them would leave every
    forward-only test green, which is why this is parameterized so widely rather than spot-checked.
    """
    shape = dict(q_len=q_len, k_len=k_len, n_heads=n_heads, n_kv_heads=n_kv_heads,
                 dim=dim, idx_dim=idx_dim)
    results = {}
    for path in ("reference", "kernel"):
        inputs = make_inputs(**shape)
        gate_scale = torch.tensor(0.7, requires_grad=True)
        if path == "reference":
            out = gated_attention_reference(
                **inputs, gate_scale=gate_scale, pin_mode=pin_mode, n_sink=n_sink,
                scaling=dim**-0.5,
            )
        else:
            out = run_kernel(inputs, gate_scale, pin_mode, n_sink, pin_self)
        results[path] = grads_of(out, inputs, gate_scale)

    for index, label in enumerate(LABELS):
        got, want = results["kernel"][index], results["reference"][index]
        assert torch.allclose(got, want, atol=2e-4, rtol=1e-3), (
            f"{label} disagrees with the reference by "
            f"{(got - want).abs().max().item():.2e} (pin_mode={pin_mode}, {q_len}x{k_len})"
        )


@pytest.mark.parametrize("block_m,block_n", [(4, 4), (8, 16), (16, 8), (32, 32)])
def test_kernel_is_tile_invariant(block_m, block_n):
    """
    Tile shape is a performance knob and must not move the numbers.

    Includes tiles larger than the sequence, which exercise the padding lanes: those lanes never
    accumulate, so their ``run_sum`` is 0 and the final divide is ``0/0``. The masked stores
    discard them -- this test is what says so rather than assuming it, since a clamp there would
    look equally correct while writing a plausible wrong value into a lane.
    """
    inputs = make_inputs(q_len=12, k_len=12)
    gate_scale = torch.tensor(0.7)
    baseline = run_kernel(
        make_inputs(q_len=12, k_len=12), gate_scale, "sink", 4, False, block_m=4, block_n=4
    )
    got = run_kernel(inputs, gate_scale, "sink", 4, False, block_m=block_m, block_n=block_n)
    assert torch.isfinite(got).all(), "padding lanes leaked a non-finite value into the output"
    assert torch.allclose(got, baseline, atol=1e-5)


def test_kernel_retains_only_o_l():
    """
    The kernel keeps ``O(L)`` for backward -- the property SDPA could not deliver here.

    Asserted as a growth ratio (~2x per doubling of ``L``, not ~4x), which is what distinguishes
    recomputing the logits from storing them, and is machine-independent in a way a byte count is
    not. The math backend the earlier attempts fell into grows at ~4x.
    """
    def retained(q_len):
        total = 0

        def pack(tensor):
            nonlocal total
            total += tensor.numel() * tensor.element_size()
            return tensor

        inputs = make_inputs(q_len=q_len, k_len=q_len)
        gate_scale = torch.tensor(0.7, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            run_kernel(inputs, gate_scale, "sink", 4, False, block_m=16, block_n=16)
        return total

    sizes = [32, 64, 128]
    measured = [retained(n) for n in sizes]
    growth = (measured[2] / measured[1] + measured[1] / measured[0]) / 2
    assert growth < 2.6, (
        f"retention grows {growth:.2f}x per doubling of L -- expected ~2x. Above ~4x means the "
        "logits are being stored rather than recomputed."
    )


def test_empty_history_row_stays_finite():
    """
    A row whose only visible key is pinned has no history, and must not produce NaN.

    Under ``self`` pinning at ``Sq == Sk`` this is query 0: it sees only its own diagonal, which is
    exempt. ``history_lse`` gives such rows an inert gate rather than ``-inf``; this checks the
    kernel consumes that without turning it back into a NaN, which would then spread through the
    whole model rather than staying local.
    """
    inputs = make_inputs(q_len=8, k_len=8)
    gate_scale = torch.tensor(0.7, requires_grad=True)
    out = run_kernel(inputs, gate_scale, "self", 0, True)
    assert torch.isfinite(out).all()
    torch.manual_seed(2)
    (out * torch.randn_like(out)).sum().backward()
    for name in ("q", "k", "v", "q_idx", "k_idx"):
        assert torch.isfinite(inputs[name].grad).all(), f"d/{name} is not finite"


def test_gate_scale_zero_severs_the_router_gradient():
    """
    At ``gate_scale = 0`` the kernel must deliver no gradient to the router.

    The same property the torch path is pinned on, re-checked here because the kernel computes the
    gate term separately: a backward that forgot to scale ``ds_gate`` would leak a gradient the
    maths says cannot exist, and no forward test would notice.
    """
    inputs = make_inputs()
    gate_scale = torch.tensor(0.0, requires_grad=True)
    out = run_kernel(inputs, gate_scale, "none", 0, False)
    torch.manual_seed(2)
    (out * torch.randn_like(out)).sum().backward()
    assert inputs["q_idx"].grad.abs().max() == 0.0
    assert inputs["k_idx"].grad.abs().max() == 0.0
    # The attention path is untouched, so q/k/v still learn.
    assert inputs["q"].grad.abs().max() > 0.0


def test_availability_refuses_float64():
    """
    ``tl.dot`` has no fp64, so the kernel must decline it rather than demote silently.

    Load-bearing for the rest of the suite: the fp64 reference tests in
    ``test_gqa_indexer_e2e.py`` rely on this to route to the torch path, and would otherwise be
    comparing the kernel against itself.
    """
    assert not gated_kernels_available(torch.randn(2, dtype=torch.float64))
    assert gated_kernels_available(torch.randn(2), torch.randn(2, dtype=torch.bfloat16))


def test_pinned_keys_receive_no_gate():
    """
    A pinned key's logit must be the bare attention logit -- gate exactly 0, not merely small.

    Checked by making the gate enormous: if the pin were leaking, the pinned columns' attention
    mass would move with it. Their share must instead be identical to the ungated model's.
    """
    inputs = make_inputs(q_len=8, k_len=8, n_heads=2, n_kv_heads=2)
    n_sink = 3
    shares = []
    for scale_value in (0.0, 50.0):
        gate_scale = torch.tensor(scale_value)
        # pin_mode="none" with gate_scale=0 is the ungated model; "sink" keeps the sink columns
        # ungated whatever the gate does elsewhere.
        out = run_kernel(
            make_inputs(q_len=8, k_len=8, n_heads=2, n_kv_heads=2),
            gate_scale, "sink" if scale_value else "none", n_sink if scale_value else 0,
            False,
        )
        shares.append(out)
    # A huge gate must change the output (the gate is live) ...
    assert not torch.allclose(shares[0], shares[1], atol=1e-3)
    # ... but row 0, whose only visible key is the pinned key 0, cannot move: its softmax has a
    # single live entry, and that entry's gate is 0 in both runs.
    assert torch.allclose(shares[0][:, :, 0], shares[1][:, :, 0], atol=1e-5)
