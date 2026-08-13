# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for indexer-driven sparse attention (GQA DSA inference).

Three layers, because each rules out a different class of bug:

1. **The definition.** Two independent torch references -- one gathering ``topk`` keys, one
   scattering a dense ``(Sq, Sk)`` mask -- must agree, and with ``topk == Sk`` both must
   reduce to ordinary dense causal attention. The gather's index arithmetic is the part
   easiest to get subtly wrong and is exactly what the two do not share.
2. **The kernel.** Compared against the gather reference across group sizes, head dims, dead
   rows and malformed indices. Runs under CUDA or ``TRITON_INTERPRET=1``.
3. **Varlen.** Each packed sequence must give *bit-identical* results to running the batched
   kernel on that sequence alone, which is the strongest available statement that packing is
   a no-op on the numbers rather than merely close.

The index tensor is :func:`~.sparse_support.sort_support`'s convention, so
:func:`~.sparse_support.streaming_topk_support` is used as the index source wherever a
realistic one is wanted -- that keeps the selector and the kernel wired together under test.
"""

import pytest
import torch

from kvpress.presses.gqa_indexer import (
    HAS_TRITON,
    pack_varlen,
    seq_ids_from_cu_seqlens,
    sparse_gqa_attention,
    sparse_gqa_attention_dense_reference,
    sparse_gqa_attention_reference,
    sparse_gqa_attention_varlen_reference,
    sparse_kernels_available,
    streaming_topk_support,
    triton_interpret_enabled,
    triton_sparse_gqa_attention,
    triton_sparse_gqa_attention_varlen,
    unpack_varlen,
)

from kvpress.presses.gqa_indexer.triton_fused_loss import block_pow2
from kvpress.presses.gqa_indexer.triton_sparse_attention import MIN_BLOCK_K, check_block_k

requires_triton = pytest.mark.skipif(
    not HAS_TRITON or not (torch.cuda.is_available() or triton_interpret_enabled()),
    reason="needs Triton and either CUDA or TRITON_INTERPRET=1",
)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_case(bsz=2, n_kv_heads=2, group_size=4, q_len=8, k_len=12, dim=16, dim_v=16, topk=5, seed=0):
    """fp32 inputs plus indices from the real selector; fp32 because the kernel accumulates there."""
    torch.manual_seed(seed)
    dev = device()
    n_heads = n_kv_heads * group_size
    q = torch.randn(bsz, n_heads, q_len, dim, device=dev)
    k = torch.randn(bsz, n_kv_heads, k_len, dim, device=dev)
    v = torch.randn(bsz, n_kv_heads, k_len, dim_v, device=dev)
    q_idx = torch.randn(bsz, n_kv_heads, q_len, dim, device=dev)
    k_idx = torch.randn(bsz, k_len, dim, device=dev)
    indices, _ = streaming_topk_support(q_idx, k_idx, topk=topk, key_tile=8, query_tile=8)
    return q, k, v, indices


def causal_indices(bsz, n_kv_heads, q_len, k_len, topk, seed=0):
    """
    Random causal, duplicate-free indices, ``-1``-padded.

    Duplicate-free so the dense reference (which collapses duplicates under ``scatter``) is
    comparable; ``streaming_topk_support`` never emits duplicates either.
    """
    g = torch.Generator().manual_seed(seed)
    out = torch.full((bsz, n_kv_heads, q_len, topk), -1, dtype=torch.int32)
    offset = k_len - q_len
    for b in range(bsz):
        for h in range(n_kv_heads):
            for t in range(q_len):
                limit = min(t + offset, k_len - 1)
                perm = torch.randperm(limit + 1, generator=g)[:topk]
                out[b, h, t, : perm.numel()] = perm.to(torch.int32)
    return out.to(device())


def compare(kernel, reference, *, atol=1e-5, rtol=1e-4):
    """Outputs close, and the two agree on *which* rows are dead (lse == -inf)."""
    out_k, lse_k = kernel
    out_r, lse_r = reference
    torch.testing.assert_close(out_k, out_r, atol=atol, rtol=rtol)
    assert torch.equal(torch.isfinite(lse_k), torch.isfinite(lse_r)), "dead rows disagree"
    live = torch.isfinite(lse_r)
    if live.any():
        # lse is fp32 in both paths regardless of the input dtype, so it keeps fp32 tolerance.
        torch.testing.assert_close(lse_k[live], lse_r[live], atol=1e-5, rtol=1e-4)


# ----------------------------------------------------------------------------------------
# 1. The definition: the two references against each other and against dense attention
# ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("group_size", [1, 2, 4, 8])
def test_references_agree(group_size):
    q, k, v, _ = make_case(group_size=group_size)
    indices = causal_indices(q.shape[0], k.shape[1], q.shape[2], k.shape[2], topk=5)
    compare(
        sparse_gqa_attention_reference(q, k, v, indices),
        sparse_gqa_attention_dense_reference(q, k, v, indices),
    )


def test_references_agree_on_selector_output():
    """The selector's own indices, not synthetic ones -- this is the wiring that ships."""
    q, k, v, indices = make_case()
    compare(
        sparse_gqa_attention_reference(q, k, v, indices),
        sparse_gqa_attention_dense_reference(q, k, v, indices),
    )


@pytest.mark.parametrize("q_len,k_len", [(16, 16), (8, 20), (1, 20)])
def test_full_topk_reduces_to_dense_attention(q_len, k_len):
    """
    With every key selected, sparse attention *is* dense causal attention.

    The sharpest check on the whole definition: it pins the scale, the softmax, and the
    bottom-right causal alignment simultaneously, against an implementation
    (``scaled_dot_product_attention``) that shares no code with any of this.
    """
    q, k, v, _ = make_case(q_len=q_len, k_len=k_len)
    bsz, n_kv_heads = k.shape[0], k.shape[1]
    group_size = q.shape[1] // n_kv_heads
    indices = (
        torch.arange(k_len, device=q.device)
        .view(1, 1, 1, k_len)
        .expand(bsz, n_kv_heads, q_len, k_len)
        .contiguous()
        .to(torch.int32)
    )

    out, lse = sparse_gqa_attention_reference(q, k, v, indices)

    q_pos = torch.arange(q_len, device=q.device).unsqueeze(-1) + (k_len - q_len)
    keep = torch.arange(k_len, device=q.device).unsqueeze(0) <= q_pos
    k_rep = k.repeat_interleave(group_size, dim=1).float()
    v_rep = v.repeat_interleave(group_size, dim=1).float()
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k_rep, v_rep, attn_mask=keep, scale=q.shape[-1] ** -0.5
    )
    torch.testing.assert_close(out.float(), expected, atol=1e-5, rtol=1e-4)

    logits = (q.float() @ k_rep.transpose(-1, -2)) * q.shape[-1] ** -0.5
    expected_lse = torch.logsumexp(logits.masked_fill(~keep, -float("inf")), dim=-1)
    torch.testing.assert_close(lse, expected_lse, atol=1e-5, rtol=1e-4)


def test_dead_row_is_zero_and_neg_inf():
    """An all-``-1`` row must be ``(0, -inf)`` and must not poison its neighbours with NaN."""
    q, k, v, indices = make_case()
    indices = indices.clone()
    indices[0, 0, 3, :] = -1
    for out, lse in (
        sparse_gqa_attention_reference(q, k, v, indices),
        sparse_gqa_attention_dense_reference(q, k, v, indices),
    ):
        group_size = q.shape[1] // k.shape[1]
        dead = slice(0, group_size)  # query heads reading KV head 0
        assert (out[0, dead, 3] == 0).all()
        assert torch.isneginf(lse[0, dead, 3]).all()
        assert torch.isfinite(out).all(), "a dead row leaked NaN/inf into the output"


def test_reference_rejects_bad_shapes():
    q, k, v, indices = make_case()
    with pytest.raises(ValueError, match="not a multiple"):
        sparse_gqa_attention_reference(q[:, :3], k, v, indices[:, :3])
    with pytest.raises(ValueError, match="indices must be"):
        sparse_gqa_attention_reference(q, k, v, indices[:, :1])
    with pytest.raises(ValueError, match="k has"):
        sparse_gqa_attention_reference(q, k, v[:, :, :-1], indices)


# ----------------------------------------------------------------------------------------
# 2. The kernel against the reference
# ----------------------------------------------------------------------------------------
@requires_triton
@pytest.mark.parametrize("group_size", [1, 2, 4, 8])
def test_kernel_matches_reference_over_group_sizes(group_size):
    """``group_size`` sets the GEMM's M dimension (padded to 16), so it is the risky axis."""
    q, k, v, indices = make_case(group_size=group_size)
    compare(
        triton_sparse_gqa_attention(q, k, v, indices, block_k=16),
        sparse_gqa_attention_reference(q, k, v, indices),
    )


@requires_triton
@pytest.mark.parametrize(
    "dim,dim_v",
    [(16, 16), (16, 8), (24, 24), (16, 12)],  # non-power-of-two dims exercise the masked loads
)
def test_kernel_matches_reference_over_head_dims(dim, dim_v):
    q, k, v, indices = make_case(dim=dim, dim_v=dim_v)
    compare(
        triton_sparse_gqa_attention(q, k, v, indices, block_k=16),
        sparse_gqa_attention_reference(q, k, v, indices),
    )


@requires_triton
@pytest.mark.parametrize("block_k", [16, 32, 64, 128])
def test_kernel_is_invariant_to_block_k(block_k):
    """``block_k`` only tiles the topk axis; a dependence on it means broken online softmax."""
    q, k, v, indices = make_case(topk=5)
    compare(
        triton_sparse_gqa_attention(q, k, v, indices, block_k=block_k),
        sparse_gqa_attention_reference(q, k, v, indices),
    )


@requires_triton
@pytest.mark.parametrize("q_len,k_len", [(1, 20), (8, 20), (16, 16), (16, 128)])
def test_kernel_prefill_and_decode(q_len, k_len):
    q, k, v, indices = make_case(q_len=q_len, k_len=k_len, topk=min(8, k_len))
    compare(
        triton_sparse_gqa_attention(q, k, v, indices, block_k=16),
        sparse_gqa_attention_reference(q, k, v, indices),
    )


@requires_triton
def test_kernel_handles_malformed_indices():
    """
    Out-of-range, wrong-negative and duplicated slots must be handled, not merely tolerated.

    The kernel clamps every gather address and masks by validity, so a malformed index can
    change the result but can never read out of bounds. That was a real bug during
    development: an out-of-range index crashed the reference's gather while the kernel
    happily returned a number.
    """
    q, k, v, indices = make_case()
    k_len = k.shape[2]

    cases = {
        "all dead": lambda i: i.fill_(-1),
        "row dead": lambda i: i[0, 0, 3, :].fill_(-1),
        "partial pad": lambda i: i[0, 0, 3, 2:].fill_(-1),
        "index == k_len": lambda i: i[0, 0, 2, 0].fill_(k_len),
        "index far past end": lambda i: i[0, 1, 4, 1].fill_(k_len + 100),
        "negative not -1": lambda i: i[0, 0, 1, 0].fill_(-7),
        "duplicates": lambda i: i[..., 1].copy_(i[..., 0]),
    }
    for name, mutate in cases.items():
        idx = indices.clone()
        mutate(idx)
        out, lse = triton_sparse_gqa_attention(q, k, v, idx, block_k=16)
        assert torch.isfinite(out).all(), f"{name}: non-finite output"
        compare((out, lse), sparse_gqa_attention_reference(q, k, v, idx))


@requires_triton
def test_kernel_causal_flag_masks_future_keys():
    """With every slot pointing past the diagonal, ``causal=True`` must empty every row."""
    q, k, v, _ = make_case(q_len=8, k_len=12)
    bsz, n_kv_heads, k_len = k.shape[0], k.shape[1], k.shape[2]
    indices = torch.full((bsz, n_kv_heads, q_len := q.shape[2], 5), k_len - 1, dtype=torch.int32, device=q.device)

    out, lse = triton_sparse_gqa_attention(q, k, v, indices, block_k=16, causal=True)
    # Only the last query row can legally see k_len - 1 under bottom-right alignment.
    assert torch.isneginf(lse[:, :, : q_len - 1]).all()
    assert (out[:, :, : q_len - 1] == 0).all()
    compare((out, lse), sparse_gqa_attention_reference(q, k, v, indices, causal=True))

    compare(
        triton_sparse_gqa_attention(q, k, v, indices, block_k=16, causal=False),
        sparse_gqa_attention_reference(q, k, v, indices, causal=False),
    )


@requires_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_kernel_dtypes(dtype):
    """
    Low precision affects only the stored output; accumulation stays fp32 in both paths.

    Tolerances are one ULP of the *output* dtype, since that is the only lossy step: both
    paths accumulate in fp32 and differ solely in rounding the final store. bf16 keeps 8
    mantissa bits, so a value near 2 quantizes in steps of 0.0156 -- an fp32-grade rtol here
    would be demanding precision the dtype cannot represent.
    """
    q, k, v, indices = make_case()
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    out, lse = triton_sparse_gqa_attention(q, k, v, indices, block_k=16)
    ref_out, ref_lse = sparse_gqa_attention_reference(q, k, v, indices)
    assert out.dtype == dtype and lse.dtype == torch.float32
    atol, rtol = {
        torch.float32: (1e-5, 1e-4),
        torch.float16: (1e-3, 1e-3),
        torch.bfloat16: (2e-2, 1e-2),
    }[dtype]
    compare((out, lse), (ref_out, ref_lse), atol=atol, rtol=rtol)


@requires_triton
def test_kernel_accepts_int64_and_non_contiguous_indices():
    q, k, v, indices = make_case()
    reference = sparse_gqa_attention_reference(q, k, v, indices)
    compare(triton_sparse_gqa_attention(q, k, v, indices.long(), block_k=16), reference)
    flipped = indices.flip(-1)
    compare(
        triton_sparse_gqa_attention(q, k, v, flipped, block_k=16),
        sparse_gqa_attention_reference(q, k, v, flipped),
    )


@requires_triton
@pytest.mark.parametrize("block_k,match", [(6, "power of two"), (8, "must be >= 16"), (0, "power of two")])
def test_kernel_rejects_bad_block_k(block_k, match):
    """
    Both failure modes, and the >= 16 one specifically.

    ``block_k=8`` is a power of two and runs correctly under ``TRITON_INTERPRET=1``, so
    without an eager check it would look fine here and fail to compile on a GPU -- the
    interpreter does not apply ``tl.dot``'s K >= 16 floor. This test is what keeps the
    interpreter-only test suite honest about hardware.
    """
    q, k, v, indices = make_case()
    with pytest.raises(ValueError, match=match):
        triton_sparse_gqa_attention(q, k, v, indices, block_k=block_k)
    cu = torch.tensor([0, q.shape[2]], dtype=torch.int32, device=q.device)
    with pytest.raises(ValueError, match=match):
        triton_sparse_gqa_attention_varlen(
            q.permute(0, 2, 1, 3)[0], k.permute(0, 2, 1, 3)[0], v.permute(0, 2, 1, 3)[0],
            indices.permute(0, 2, 1, 3)[0], cu, cu, block_k=block_k,
        )


# ----------------------------------------------------------------------------------------
# 3. Varlen
# ----------------------------------------------------------------------------------------
def varlen_case(q_lens, k_lens, n_kv_heads=2, group_size=4, dim=16, dim_v=16, topk=5, seed=0):
    """Pack per-sequence tensors with sequence-local causal indices."""
    torch.manual_seed(seed)
    dev = device()
    n_heads = n_kv_heads * group_size
    qs, ks, vs, ids = [], [], [], []
    for q_len, k_len in zip(q_lens, k_lens):
        qs.append(torch.randn(q_len, n_heads, dim, device=dev))
        ks.append(torch.randn(k_len, n_kv_heads, dim, device=dev))
        vs.append(torch.randn(k_len, n_kv_heads, dim_v, device=dev))
        idx = causal_indices(1, n_kv_heads, max(q_len, 1), k_len, topk, seed=seed)
        ids.append(idx[0].permute(1, 0, 2)[:q_len].contiguous())  # (q_len, Hkv, topk)
    q, cu_q = pack_varlen(qs)
    k, cu_k = pack_varlen(ks)
    v, _ = pack_varlen(vs)
    indices, _ = pack_varlen(ids)
    return q, k, v, indices, cu_q, cu_k, (qs, ks, vs, ids)


VARLEN_LAYOUTS = [
    ([6, 6, 6], [10, 10, 10]),      # equal lengths
    ([3, 7, 5], [8, 12, 9]),        # ragged on both axes
    ([9], [14]),                    # single sequence
    ([8, 1, 1, 1], [8, 12, 20, 5]), # one prefill alongside decodes
    ([5, 0, 4], [9, 7, 8]),         # an empty sequence in the middle
    ([0, 6, 0], [3, 11, 4]),        # empty at both ends
    ([7, 4], [7, 4]),               # q_len == k_len, zero offset
]


@pytest.mark.parametrize("q_lens,k_lens", VARLEN_LAYOUTS)
def test_varlen_reference_matches_per_sequence_batched(q_lens, k_lens):
    """The packed reference must equal the batched reference run on each sequence alone."""
    q, k, v, indices, cu_q, cu_k, (qs, ks, vs, ids) = varlen_case(q_lens, k_lens)
    out, lse = sparse_gqa_attention_varlen_reference(q, k, v, indices, cu_q, cu_k)

    for s, (q_len, _) in enumerate(zip(q_lens, k_lens)):
        if q_len == 0:
            continue
        start = int(cu_q[s])
        expected_out, expected_lse = sparse_gqa_attention_reference(
            qs[s].permute(1, 0, 2).unsqueeze(0),
            ks[s].permute(1, 0, 2).unsqueeze(0),
            vs[s].permute(1, 0, 2).unsqueeze(0),
            ids[s].permute(1, 0, 2).unsqueeze(0),
        )
        torch.testing.assert_close(
            out[start : start + q_len], expected_out[0].permute(1, 0, 2), atol=1e-5, rtol=1e-4
        )
        torch.testing.assert_close(
            lse[start : start + q_len], expected_lse[0].permute(1, 0), atol=1e-5, rtol=1e-4
        )


@requires_triton
@pytest.mark.parametrize("q_lens,k_lens", VARLEN_LAYOUTS)
def test_varlen_kernel_matches_reference(q_lens, k_lens):
    q, k, v, indices, cu_q, cu_k, _ = varlen_case(q_lens, k_lens)
    compare(
        triton_sparse_gqa_attention_varlen(q, k, v, indices, cu_q, cu_k, block_k=16),
        sparse_gqa_attention_varlen_reference(q, k, v, indices, cu_q, cu_k),
    )


@requires_triton
@pytest.mark.parametrize("q_lens,k_lens", VARLEN_LAYOUTS)
def test_varlen_is_bitwise_identical_to_batched_per_sequence(q_lens, k_lens):
    """
    Packing must not perturb the arithmetic at all.

    Exact equality rather than a tolerance: the same kernel does the same reductions in the
    same order either way, so anything but bit-identical means the packed path took a
    different route through memory -- which is how the token/head stride transposition bug was
    caught (it corrupted the heap rather than merely shifting a digit).
    """
    q, k, v, indices, cu_q, cu_k, (qs, ks, vs, ids) = varlen_case(q_lens, k_lens)
    out, lse = triton_sparse_gqa_attention_varlen(q, k, v, indices, cu_q, cu_k, block_k=16)

    for s, q_len in enumerate(q_lens):
        if q_len == 0:
            continue
        start = int(cu_q[s])
        out_b, lse_b = triton_sparse_gqa_attention(
            qs[s].permute(1, 0, 2).unsqueeze(0),
            ks[s].permute(1, 0, 2).unsqueeze(0),
            vs[s].permute(1, 0, 2).unsqueeze(0),
            ids[s].permute(1, 0, 2).unsqueeze(0),
            block_k=16,
        )
        assert torch.equal(out[start : start + q_len], out_b[0].permute(1, 0, 2))
        assert torch.equal(lse[start : start + q_len], lse_b[0].permute(1, 0))


@requires_triton
def test_varlen_precomputed_seq_ids_match():
    """Passing ``seq_ids`` is an optimization, so it must change nothing."""
    q, k, v, indices, cu_q, cu_k, _ = varlen_case([5, 3], [9, 7])
    seq_ids = seq_ids_from_cu_seqlens(cu_q, q.shape[0])
    assert seq_ids.tolist() == [0, 0, 0, 0, 0, 1, 1, 1]
    a = triton_sparse_gqa_attention_varlen(q, k, v, indices, cu_q, cu_k, block_k=16)
    b = triton_sparse_gqa_attention_varlen(q, k, v, indices, cu_q, cu_k, block_k=16, seq_ids=seq_ids)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_seq_ids_handles_empty_sequences():
    """``searchsorted`` (not ``repeat_interleave``) is what makes a zero-length run correct."""
    cu = torch.tensor([0, 2, 2, 5], dtype=torch.int32)
    assert seq_ids_from_cu_seqlens(cu, 5).tolist() == [0, 0, 2, 2, 2]


def test_seq_ids_rejects_inconsistent_total():
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    with pytest.raises(ValueError, match="packed axis"):
        seq_ids_from_cu_seqlens(cu, 6)


def test_pack_unpack_roundtrip():
    parts = [torch.randn(3, 2), torch.randn(0, 2), torch.randn(4, 2)]
    packed, cu = pack_varlen(parts)
    assert cu.tolist() == [0, 3, 3, 7]
    for original, restored in zip(parts, unpack_varlen(packed, cu)):
        assert torch.equal(original, restored)


@requires_triton
def test_varlen_rejects_batched_shapes():
    """A 4D tensor here means the caller confused the two entry points -- say so loudly."""
    q, k, v, indices = make_case()
    cu = torch.tensor([0, q.shape[2]], dtype=torch.int32, device=q.device)
    with pytest.raises(ValueError, match="packed 3D"):
        triton_sparse_gqa_attention_varlen(q, k, v, indices, cu, cu)


@requires_triton
@pytest.mark.parametrize("magnitude", [1e-6, 0.0, 1.0, 30.0, 300.0])
def test_kernel_is_stable_at_extreme_logit_magnitudes(magnitude):
    """
    The online softmax must stay finite when logits are huge, tiny, or all zero.

    Large magnitudes are where a rescaling bug shows up: ``exp(old_max - new_max)``
    underflows to 0 and, if the accumulator were not rescaled consistently with ``run_sum``,
    the row would drift or blow up. At ``magnitude=300`` the logits reach ~1e4, well past
    where a naive ``exp`` overflows -- so this also pins the max-subtraction.

    Tolerances scale with the magnitude because the *inputs* carry that much fp32 relative
    error before the kernel starts; the invariant under test is finiteness and agreement with
    the reference, not absolute error.
    """
    q, k, v, indices = make_case(topk=8, q_len=16, k_len=24)
    q, k = q * magnitude, k * magnitude
    out, lse = triton_sparse_gqa_attention(q, k, v, indices, block_k=16)
    assert torch.isfinite(out).all(), "non-finite output"
    ref_out, ref_lse = sparse_gqa_attention_reference(q, k, v, indices)
    assert not torch.isnan(ref_out).any()
    atol = max(1e-5, 1e-6 * max(magnitude, 1.0) ** 2)
    compare((out, lse), (ref_out, ref_lse), atol=atol, rtol=1e-3)


@requires_triton
def test_kernel_survives_many_rescaling_tiles():
    """Repeated online-softmax merges (topk=128 over block_k=16 is 8 tiles) must not drift."""
    q, k, v, indices = make_case(q_len=8, k_len=256, topk=128)
    q, k = q * 50.0, k * 50.0
    out, lse = triton_sparse_gqa_attention(q, k, v, indices, block_k=16)
    assert torch.isfinite(out).all()
    compare((out, lse), sparse_gqa_attention_reference(q, k, v, indices), atol=1e-2, rtol=1e-3)


# ----------------------------------------------------------------------------------------
# Hardware shape legality (checkable without a GPU)
# ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("group_size", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("dim,dim_v", [(8, 8), (16, 16), (16, 8), (24, 24), (16, 12), (64, 64), (128, 128)])
@pytest.mark.parametrize("block_k", [16, 32, 64, 128])
def test_dot_shapes_are_legal_on_hardware(group_size, dim, dim_v, block_k):
    """
    Every ``tl.dot`` this kernel emits must satisfy NVIDIA's ``min_dot_size`` of ``(1, 1, 16)``.

    This exists because the whole suite can run under ``TRITON_INTERPRET=1``, which does *not*
    apply the shape floors -- so an illegal tile would be green here and fail to compile on the
    first GPU. Rather than trust that, the rule is restated over the tile shapes the launcher
    actually computes.

    Both dots are covered, and the binding constraint differs between them: ``Q @ K^T``
    contracts over the padded head dim (safe because ``block_pow2`` floors it at 16, which this
    test is what pins down), while ``P @ V`` contracts over ``block_k`` (safe because
    :func:`~.triton_sparse_attention.check_block_k` rejects anything below 16).
    """
    block_g = block_pow2(group_size, minimum=1)
    block_d, block_dv = block_pow2(dim), block_pow2(dim_v)
    m_min, n_min, k_min = 1, 1, 16  # triton/backends/nvidia/compiler.py::min_dot_size, non-fp8

    # Q @ K^T : [BLOCK_G, BLOCK_D] @ [BLOCK_D, BLOCK_K]
    assert block_g >= m_min and block_k >= n_min and block_d >= k_min, (
        f"QK dot illegal: M={block_g}, N={block_k}, K={block_d}"
    )
    # P @ V : [BLOCK_G, BLOCK_K] @ [BLOCK_K, BLOCK_DV]
    assert block_g >= m_min and block_dv >= n_min and block_k >= k_min, (
        f"PV dot illegal: M={block_g}, N={block_dv}, K={block_k}"
    )


def test_min_block_k_matches_the_hardware_floor():
    """If Triton ever relaxes the K floor, this is the one place that needs revisiting."""
    assert MIN_BLOCK_K == 16
    for illegal in (1, 2, 4, 8):
        with pytest.raises(ValueError, match="must be >= 16"):
            check_block_k(illegal)
    for legal in (16, 32, 64, 128):
        check_block_k(legal)


# ----------------------------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------------------------
def test_dispatcher_falls_back_to_reference():
    q, k, v, indices = make_case()
    compare(
        sparse_gqa_attention(q, k, v, indices, force_reference=True),
        sparse_gqa_attention_reference(q, k, v, indices),
    )


@requires_triton
def test_dispatcher_uses_kernel_when_available():
    q, k, v, indices = make_case()
    assert sparse_kernels_available(q, k, v)
    compare(
        sparse_gqa_attention(q, k, v, indices),
        sparse_gqa_attention_reference(q, k, v, indices),
    )


def test_kernels_unavailable_for_float64():
    """fp64 has no ``tl.dot``; silently demoting a caller who asked for it would be wrong."""
    q, k, v, _ = make_case()
    assert not sparse_kernels_available(q.double(), k.double(), v.double())
