# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the Triton stage-1 kernels.

The kernels are a transcription of :mod:`kvpress.presses.gqa_indexer.fused_loss`, whose own
tests establish that the objective and gradients are right. So these compare the two
implementations directly rather than re-deriving anything -- any divergence is a
transcription bug, which is the only failure mode left.

Kernel tests need CUDA (or ``TRITON_INTERPRET=1``, which runs the bodies on CPU). The mask
decomposition and dispatch-policy tests are pure PyTorch and always run; they are also where
the sharpest failure mode lives, since a mask that decomposes *wrongly* would train against a
mask the student never sees.
"""

import pytest
import torch

from kvpress.presses.gqa_indexer import (
    GQAIndexer,
    GQAIndexerConfig,
    HAS_TRITON,
    build_indexer_mask,
    decompose_mask,
    fused_indexer_ce_rows,
    fused_indexer_loss,
    kernels_available,
    make_recompute_teacher,
    teacher_lse_from_qk,
    triton_indexer_ce_rows,
    triton_indexer_loss,
    triton_interpret_enabled,
)
from kvpress.presses.gqa_indexer.indexer import MASK_NEG
from kvpress.presses.gqa_indexer.triton_fused_loss import block_pow2

requires_triton = pytest.mark.skipif(
    not HAS_TRITON or not (torch.cuda.is_available() or triton_interpret_enabled()),
    reason="needs Triton and either CUDA or TRITON_INTERPRET=1",
)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_case(bsz=2, h=2, group_size=2, q_len=16, k_len=16, dim=8, d_tea=8, seed=0):
    """fp32 fixture; the kernels have no fp64 path (tl.dot does not support it)."""
    torch.manual_seed(seed)
    dev = device()
    n_heads = h * group_size
    q_idx = torch.randn(bsz, h, q_len, dim, device=dev, requires_grad=True)
    k_idx = torch.randn(bsz, k_len, dim, device=dev, requires_grad=True)
    q_tea = torch.randn(bsz, n_heads, q_len, d_tea, device=dev)
    k_tea = torch.randn(bsz, h, k_len, d_tea, device=dev)
    scaling = d_tea**-0.5
    mask = build_indexer_mask(q_len, k_len, dev, dtype=torch.float32)
    lse = teacher_lse_from_qk(q_tea, k_tea, scaling, mask=mask, key_tile=8)
    return dict(
        q_idx=q_idx, k_idx=k_idx, q_tea=q_tea, k_tea=k_tea, scaling=scaling,
        mask=mask, lse=lse, group_size=group_size, h=h, q_len=q_len, k_len=k_len,
    )


def torch_rows(case, q=None, k=None, mask=None):
    """The PyTorch reference on the same inputs."""
    teacher = make_recompute_teacher(
        case["q_tea"], case["k_tea"], case["scaling"], case["group_size"]
    )
    return fused_indexer_ce_rows(
        case["q_idx"] if q is None else q,
        case["k_idx"] if k is None else k,
        teacher,
        case["lse"],
        group_size=case["group_size"],
        mask=case["mask"] if mask is None else mask,
        key_tile=8,
        query_tile=8,
    )


def live_rows(mask, bsz, h):
    """
    Rows with at least one visible key, ``(B, h, Sq)``.

    The two implementations deliberately differ on *dead* rows. The PyTorch path *adds* a
    finite ``MASK_NEG`` sentinel, so a fully-masked row's logits are about ``-1e4`` and its
    ``lse`` about ``-9997``; the kernels use a true ``-inf`` and clamp, landing near ``-23``.
    Both are finite (which is the property that matters -- a NaN would poison the whole
    batch), both are meaningless, and both are multiplied by zero: ``fused_indexer_loss``
    weights rows by ``row_valid``, so ``d(loss)/d(row)`` is exactly 0 there. So the
    comparison is restricted to live rows rather than papered over with a loose tolerance.
    """
    keep = (mask > MASK_NEG / 2).any(dim=-1)  # (B, 1, Sq)
    return keep.expand(bsz, h, keep.shape[-1])


# ----------------------------------------------------------------------
# Mask decomposition (no GPU needed -- and the sharpest failure mode)
# ----------------------------------------------------------------------
def rebuild_from_keep(keep, q_len, k_len, query_offset, bsz, dev):
    """What the kernel will actually apply, given (causal, keep)."""
    q_pos = torch.arange(q_len, device=dev).unsqueeze(-1) + query_offset
    k_pos = torch.arange(k_len, device=dev).unsqueeze(0)
    allowed = (k_pos <= q_pos).unsqueeze(0).expand(bsz, q_len, k_len).clone()
    if keep is not None:
        allowed &= keep.bool().unsqueeze(1)
    return allowed


@pytest.mark.parametrize("q_len,k_len", [(8, 8), (4, 9), (9, 4), (1, 6)])
def test_pure_causal_mask_needs_no_keep_vector(q_len, k_len):
    mask = build_indexer_mask(q_len, k_len, torch.device("cpu"))
    ok, keep = decompose_mask(mask, q_len, k_len, k_len - q_len)
    assert ok
    assert keep is None, "a purely causal mask must not allocate a keep vector"


@pytest.mark.parametrize("q_len,k_len", [(8, 8), (4, 9), (6, 6)])
def test_padding_is_recovered_exactly(q_len, k_len):
    """
    The recovered keep vector must rebuild the original mask bit for bit.

    This is the load-bearing assertion: a decomposition that merely looks plausible would
    have the kernel apply a *different* mask than the PyTorch path, and the loss would still
    be finite and the gradients still flow.
    """
    bsz = 2
    attention_mask = torch.ones(bsz, k_len, dtype=torch.long)
    attention_mask[1, : max(1, k_len // 3)] = 0
    mask = build_indexer_mask(q_len, k_len, torch.device("cpu"), attention_mask=attention_mask)

    ok, keep = decompose_mask(mask, q_len, k_len, k_len - q_len)
    assert ok and keep is not None
    torch.testing.assert_close(keep, attention_mask.to(torch.int8))

    rebuilt = rebuild_from_keep(keep, q_len, k_len, k_len - q_len, bsz, torch.device("cpu"))
    torch.testing.assert_close(rebuilt, (mask > MASK_NEG / 2).squeeze(1))


def test_sink_skip_decomposes_because_it_is_per_key():
    """Masking the first N keys is per-key, so the kernels can express it."""
    q_len = k_len = 8
    mask = build_indexer_mask(q_len, k_len, torch.device("cpu")).clone()
    mask[..., :3] = MASK_NEG

    ok, keep = decompose_mask(mask, q_len, k_len, 0)
    assert ok and keep is not None
    assert keep[0, :3].tolist() == [0, 0, 0]
    rebuilt = rebuild_from_keep(keep, q_len, k_len, 0, 1, torch.device("cpu"))
    torch.testing.assert_close(rebuilt, (mask > MASK_NEG / 2).squeeze(1))


def test_sliding_window_mask_is_rejected():
    """Per-row structure cannot factor into a per-key vector, so it must fall back."""
    q_len = k_len = 8
    mask = torch.full((1, 1, q_len, k_len), MASK_NEG)
    for t in range(q_len):
        mask[0, 0, t, max(0, t - 2) : t + 1] = 0.0
    ok, keep = decompose_mask(mask, q_len, k_len, 0)
    assert not ok and keep is None


def test_non_causal_mask_is_rejected():
    q_len = k_len = 8
    mask = build_indexer_mask(q_len, k_len, torch.device("cpu")).clone()
    mask[0, 0, 2, 5] = 0.0  # allows a future key
    ok, _ = decompose_mask(mask, q_len, k_len, 0)
    assert not ok


def test_random_per_pair_bias_is_rejected():
    torch.manual_seed(0)
    q_len = k_len = 8
    keep = torch.rand(1, 1, q_len, k_len) > 0.3
    causal = torch.tril(torch.ones(q_len, k_len, dtype=torch.bool))
    mask = torch.where(keep & causal, 0.0, MASK_NEG)
    ok, _ = decompose_mask(mask, q_len, k_len, 0)
    assert not ok


def test_mismatched_query_offset_is_rejected():
    """A mask built for a different alignment must not be silently reinterpreted."""
    q_len = k_len = 8
    mask = build_indexer_mask(q_len, k_len, torch.device("cpu"), query_offset=2)
    ok, _ = decompose_mask(mask, q_len, k_len, 0)
    assert not ok


def test_none_and_wrong_rank_masks():
    assert decompose_mask(None, 8, 8, 0) == (True, None)
    ok, keep = decompose_mask(torch.ones(2, 8), 8, 8, 0)
    assert not ok and keep is None


def test_decompose_mask_random_sweep():
    """
    Every accepted mask must rebuild exactly; anything else must be rejected.

    Covers causal, padded, random-per-pair and sliding-window shapes at assorted sizes,
    including Sq != Sk.
    """
    dev = torch.device("cpu")
    accepted = 0
    for trial in range(200):
        gen = torch.Generator().manual_seed(trial)
        q_len = int(torch.randint(1, 7, (1,), generator=gen))
        k_len = int(torch.randint(1, 7, (1,), generator=gen))
        bsz = int(torch.randint(1, 3, (1,), generator=gen))
        offset = k_len - q_len
        base = build_indexer_mask(q_len, k_len, dev).expand(bsz, 1, q_len, k_len).clone()
        allowed = base > MASK_NEG / 2

        style = trial % 4
        if style == 1:
            pad = torch.rand(bsz, k_len, generator=gen) > 0.4
            allowed = allowed & pad.view(bsz, 1, 1, k_len)
        elif style == 2:
            allowed = allowed & (torch.rand(bsz, 1, q_len, k_len, generator=gen) > 0.4)
        elif style == 3:
            window = torch.zeros(bsz, 1, q_len, k_len, dtype=torch.bool)
            width = int(torch.randint(1, k_len + 1, (1,), generator=gen))
            for t in range(q_len):
                hi = t + offset
                window[:, :, t, max(0, hi - width + 1) : hi + 1] = True
            allowed = allowed & window

        mask = torch.where(allowed, 0.0, MASK_NEG)
        ok, keep = decompose_mask(mask, q_len, k_len, offset)
        if ok:
            accepted += 1
            rebuilt = rebuild_from_keep(keep, q_len, k_len, offset, bsz, dev)
            torch.testing.assert_close(
                rebuilt, allowed.squeeze(1), msg=f"trial {trial} accepted but rebuilt wrong"
            )
    assert accepted > 0, "the sweep never exercised the accept path"


# ----------------------------------------------------------------------
# Helpers and availability (no GPU needed)
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "n,expected", [(1, 16), (8, 16), (16, 16), (17, 32), (32, 32), (64, 64), (100, 128), (128, 128)]
)
def test_block_pow2(n, expected):
    assert block_pow2(n) == expected


def test_kernels_decline_float64():
    """
    fp64 has no tl.dot, and silently demoting a caller who asked for it would be worse.

    The gradient tests in the sibling suites rely on fp64 to reach 1e-10; a quiet demotion
    to fp32 would make them fail confusingly rather than route to the right backend.
    """
    x = torch.randn(2, 2, dtype=torch.float64)
    assert not kernels_available(x)


def test_kernels_accept_low_precision_shapes():
    if not HAS_TRITON:
        pytest.skip("Triton not installed")
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        x = torch.randn(2, 2, dtype=dtype, device=device())
        expected = torch.cuda.is_available() or triton_interpret_enabled()
        assert kernels_available(x) is expected


# ----------------------------------------------------------------------
# Kernel vs PyTorch
# ----------------------------------------------------------------------
@requires_triton
@pytest.mark.parametrize("block_m,block_n", [(16, 16), (16, 32), (32, 16), (64, 64)])
def test_forward_matches_the_torch_path(block_m, block_n):
    case = make_case()
    got = triton_indexer_ce_rows(
        case["q_idx"], case["k_idx"], case["q_tea"], case["k_tea"], case["lse"],
        scaling=case["scaling"], block_m=block_m, block_n=block_n,
    )
    expected = torch_rows(case)
    torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-5)


@requires_triton
@pytest.mark.parametrize("q_len,k_len", [(16, 16), (8, 24), (24, 8), (17, 17), (1, 16)])
def test_forward_handles_ragged_shapes(q_len, k_len):
    """Shapes that are not multiples of the block size exercise the boundary masking."""
    case = make_case(q_len=q_len, k_len=k_len)
    got = triton_indexer_ce_rows(
        case["q_idx"], case["k_idx"], case["q_tea"], case["k_tea"], case["lse"],
        scaling=case["scaling"], block_m=16, block_n=16,
    )
    torch.testing.assert_close(got, torch_rows(case), rtol=2e-5, atol=2e-5)
    assert torch.isfinite(got).all()


@requires_triton
def test_forward_handles_padding():
    case = make_case(q_len=16, k_len=16)
    attention_mask = torch.ones(2, 16, dtype=torch.long, device=case["q_idx"].device)
    attention_mask[1, :5] = 0
    mask = build_indexer_mask(
        16, 16, case["q_idx"].device, attention_mask=attention_mask, dtype=torch.float32
    )
    case["lse"] = teacher_lse_from_qk(
        case["q_tea"], case["k_tea"], case["scaling"], mask=mask, key_tile=8
    )

    ok, keep = decompose_mask(mask, 16, 16, 0)
    assert ok and keep is not None

    got = triton_indexer_ce_rows(
        case["q_idx"], case["k_idx"], case["q_tea"], case["k_tea"], case["lse"],
        scaling=case["scaling"], keep=keep, block_m=16, block_n=16,
    )
    expected = torch_rows(case, mask=mask)
    live = live_rows(mask, 2, case["h"])
    assert not live.all(), "this fixture is meant to contain dead rows"
    torch.testing.assert_close(got[live], expected[live], rtol=2e-5, atol=2e-5)
    assert torch.isfinite(got).all(), "dead rows must still be finite"


@requires_triton
def test_fully_masked_rows_stay_finite():
    """
    A row with no visible key must not produce NaN.

    exp(-inf - -inf) is NaN, so the running max needs a finite stand-in. The row's value is
    meaningless either way -- the caller drops it via row validity -- but a NaN would
    propagate into every gradient in the batch.
    """
    case = make_case(q_len=16, k_len=16)
    mask = build_indexer_mask(16, 16, case["q_idx"].device, dtype=torch.float32).clone()
    mask[..., :4] = MASK_NEG  # rows 0..3 now see nothing
    case["lse"] = teacher_lse_from_qk(
        case["q_tea"], case["k_tea"], case["scaling"], mask=mask, key_tile=8
    )
    ok, keep = decompose_mask(mask, 16, 16, 0)
    assert ok

    got = triton_indexer_ce_rows(
        case["q_idx"], case["k_idx"], case["q_tea"], case["k_tea"], case["lse"],
        scaling=case["scaling"], keep=keep, block_m=16, block_n=16,
    )
    assert torch.isfinite(got).all(), "a fully-masked row produced a non-finite loss"
    got.sum().backward()
    assert torch.isfinite(case["q_idx"].grad).all()
    assert torch.isfinite(case["k_idx"].grad).all()

    # Live rows must still agree with the reference; the dead ones are excluded by design.
    live = live_rows(mask, case["q_idx"].shape[0], case["h"])
    torch.testing.assert_close(
        got[live], torch_rows(case, mask=mask)[live], rtol=2e-5, atol=2e-5
    )


@requires_triton
def test_dead_rows_are_excluded_by_the_row_weighting():
    """
    The two paths differ on dead rows, and that must be unobservable end to end.

    PyTorch *adds* a finite MASK_NEG so a dead row's lse is about -9997; the kernels use a
    true -inf and clamp, landing near -23. Both are finite and meaningless. What makes the
    difference harmless is that ``fused_indexer_loss`` weights rows by ``row_valid``, so the
    scalar losses -- and hence every gradient -- must agree even though the raw rows do not.
    """
    torch.manual_seed(0)
    dev = device()
    bsz, q_len, hidden, h, group_size, head_dim, d_tea = 2, 16, 32, 2, 2, 8, 8
    indexer = GQAIndexer(
        GQAIndexerConfig(hidden_size=hidden, n_heads=h, head_dim=head_dim, rope_dim=0)
    ).to(dev)
    hidden_states = torch.randn(bsz, q_len, hidden, device=dev)
    q_tea = torch.randn(bsz, h * group_size, q_len, d_tea, device=dev)
    k_tea = torch.randn(bsz, h, q_len, d_tea, device=dev)
    scaling = d_tea**-0.5

    attention_mask = torch.ones(bsz, q_len, dtype=torch.long, device=dev)
    attention_mask[1, :5] = 0
    mask = build_indexer_mask(
        q_len, q_len, dev, attention_mask=attention_mask, dtype=torch.float32
    )
    lse = teacher_lse_from_qk(q_tea, k_tea, scaling, mask=mask, key_tile=8)
    row_valid = (mask > MASK_NEG / 2).any(dim=-1)  # (B, 1, Sq), drops the dead rows
    ok, keep = decompose_mask(mask, q_len, q_len, 0)
    assert ok

    triton_loss = triton_indexer_loss(
        indexer, hidden_states, q_tea, k_tea, lse,
        scaling=scaling, keep=keep, row_valid=row_valid, block_m=16, block_n=16,
    )
    triton_loss.backward()
    triton_grads = {n: p.grad.clone() for n, p in indexer.named_parameters()}
    indexer.zero_grad(set_to_none=True)

    torch_loss = fused_indexer_loss(
        indexer, hidden_states, make_recompute_teacher(q_tea, k_tea, scaling, group_size), lse,
        group_size=group_size, mask=mask, row_valid=row_valid, key_tile=8, query_tile=8,
    )
    torch_loss.backward()

    torch.testing.assert_close(triton_loss, torch_loss, rtol=2e-5, atol=2e-5)
    for name, param in indexer.named_parameters():
        torch.testing.assert_close(
            triton_grads[name], param.grad, rtol=2e-4, atol=2e-5, msg=f"{name} gradient"
        )


@requires_triton
@pytest.mark.parametrize("block_m,block_n", [(16, 16), (32, 16), (64, 64)])
def test_gradients_match_the_torch_path(block_m, block_n):
    """
    dQ and dK must match the PyTorch implementation, whose own gradients are gradcheck'd.

    dK is the interesting half: the kernel parallelizes over key blocks (so each program's
    output range is disjoint and no atomics are needed) while the reduction runs over
    queries.
    """
    case = make_case()
    torch.manual_seed(1)
    upstream = torch.randn(
        case["q_idx"].shape[0], case["h"], case["q_len"], device=case["q_idx"].device
    )

    rows = triton_indexer_ce_rows(
        case["q_idx"], case["k_idx"], case["q_tea"], case["k_tea"], case["lse"],
        scaling=case["scaling"], block_m=block_m, block_n=block_n,
    )
    (rows * upstream).sum().backward()
    got_q, got_k = case["q_idx"].grad.clone(), case["k_idx"].grad.clone()

    q2 = case["q_idx"].detach().clone().requires_grad_(True)
    k2 = case["k_idx"].detach().clone().requires_grad_(True)
    (torch_rows(case, q=q2, k=k2) * upstream).sum().backward()

    torch.testing.assert_close(got_q, q2.grad, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(got_k, k2.grad, rtol=2e-4, atol=2e-5)


@requires_triton
def test_teacher_receives_no_gradient():
    case = make_case()
    case["q_tea"].requires_grad_(True)
    case["k_tea"].requires_grad_(True)
    triton_indexer_ce_rows(
        case["q_idx"], case["k_idx"], case["q_tea"], case["k_tea"], case["lse"],
        scaling=case["scaling"], block_m=16, block_n=16,
    ).sum().backward()
    assert case["q_tea"].grad is None
    assert case["k_tea"].grad is None


@requires_triton
def test_block_size_invariance():
    case = make_case(q_len=24, k_len=24)
    base = triton_indexer_ce_rows(
        case["q_idx"], case["k_idx"], case["q_tea"], case["k_tea"], case["lse"],
        scaling=case["scaling"], block_m=64, block_n=64,
    )
    for block_m, block_n in ((16, 16), (16, 64), (32, 32)):
        got = triton_indexer_ce_rows(
            case["q_idx"], case["k_idx"], case["q_tea"], case["k_tea"], case["lse"],
            scaling=case["scaling"], block_m=block_m, block_n=block_n,
        )
        torch.testing.assert_close(got, base, rtol=2e-5, atol=2e-5)


@requires_triton
def test_triton_indexer_loss_matches_fused_indexer_loss():
    """End-to-end through the indexer module, so the projections are shared."""
    torch.manual_seed(0)
    dev = device()
    bsz, q_len, hidden, h, group_size, head_dim, d_tea = 2, 16, 32, 2, 2, 8, 8
    indexer = GQAIndexer(
        GQAIndexerConfig(hidden_size=hidden, n_heads=h, head_dim=head_dim, rope_dim=0)
    ).to(dev)
    hidden_states = torch.randn(bsz, q_len, hidden, device=dev)
    q_tea = torch.randn(bsz, h * group_size, q_len, d_tea, device=dev)
    k_tea = torch.randn(bsz, h, q_len, d_tea, device=dev)
    scaling = d_tea**-0.5
    mask = build_indexer_mask(q_len, q_len, dev, dtype=torch.float32)
    lse = teacher_lse_from_qk(q_tea, k_tea, scaling, mask=mask, key_tile=8)
    row_valid = torch.ones(bsz, h, q_len, dtype=torch.bool, device=dev)

    triton_loss = triton_indexer_loss(
        indexer, hidden_states, q_tea, k_tea, lse,
        scaling=scaling, row_valid=row_valid, block_m=16, block_n=16, loss_coeff=2.0,
    )
    triton_loss.backward()
    triton_grads = {n: p.grad.clone() for n, p in indexer.named_parameters()}
    indexer.zero_grad(set_to_none=True)

    torch_loss = fused_indexer_loss(
        indexer, hidden_states,
        make_recompute_teacher(q_tea, k_tea, scaling, group_size), lse,
        group_size=group_size, mask=mask, row_valid=row_valid,
        key_tile=8, query_tile=8, loss_coeff=2.0,
    )
    torch_loss.backward()

    torch.testing.assert_close(triton_loss, torch_loss, rtol=2e-5, atol=2e-5)
    for name, param in indexer.named_parameters():
        torch.testing.assert_close(
            triton_grads[name], param.grad, rtol=2e-4, atol=2e-5, msg=f"{name} gradient"
        )


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def test_rows_validates_shapes():
    if not HAS_TRITON:
        pytest.skip("Triton not installed")
    case = make_case()
    kw = dict(scaling=case["scaling"])
    with pytest.raises(ValueError, match=r"q_idx must be"):
        triton_indexer_ce_rows(
            case["q_idx"][0], case["k_idx"], case["q_tea"], case["k_tea"], case["lse"], **kw
        )
    with pytest.raises(ValueError, match=r"k_idx must be"):
        triton_indexer_ce_rows(
            case["q_idx"], case["k_idx"][0], case["q_tea"], case["k_tea"], case["lse"], **kw
        )
    with pytest.raises(ValueError, match=r"teacher_lse must be"):
        triton_indexer_ce_rows(
            case["q_idx"], case["k_idx"], case["q_tea"], case["k_tea"], case["lse"][0], **kw
        )
    with pytest.raises(ValueError, match="powers of two"):
        triton_indexer_ce_rows(
            case["q_idx"], case["k_idx"], case["q_tea"], case["k_tea"], case["lse"],
            block_m=48, **kw
        )


def test_rejects_pre_expanded_teacher_keys():
    """
    key_states must carry h heads, not H.

    Passing a repeat_interleave'd tensor would have the kernel index the wrong head and
    silently distil against a different teacher, so this is checked rather than tolerated.
    """
    if not HAS_TRITON:
        pytest.skip("Triton not installed")
    case = make_case()
    expanded = case["k_tea"].repeat_interleave(case["group_size"], dim=1)
    with pytest.raises(ValueError, match="one head per indexer head"):
        triton_indexer_ce_rows(
            case["q_idx"], case["k_idx"], case["q_tea"], expanded, case["lse"],
            scaling=case["scaling"],
        )
