# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the query-independent flex_attention path.

The load-bearing claim is *exactness*: this path must select the same keys as
:func:`~kvpress.presses.gqa_indexer.sparse_support.streaming_topk_support`, because both arms of the
eval are compared against each other and a silently different support would move scores without
failing anything. So the tests compare against that function's real output rather than against a
reimplementation of the rule.
"""

import pytest
import torch

from kvpress.presses.gqa_indexer.qi_flex_attention import (
    HAS_FLEX,
    deadlines,
    qi_sparse_attention,
)
from kvpress.presses.gqa_indexer.sparse_support import streaming_topk_support
from kvpress.presses.gqa_indexer.triton_sparse_attention import sparse_gqa_attention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def query_independent_qk(n_heads, q_len, k_len, dim=8, device="cuda:0", seed=0):
    """A (q_idx, k_idx) pair whose score matrix has identical rows, like ScalarIndexer's."""
    torch.manual_seed(seed)
    q_const = torch.zeros(1, n_heads, 1, dim, device=device)
    for h in range(n_heads):
        q_const[0, h, 0, h % dim] = 1.0
    q = q_const.expand(1, n_heads, q_len, dim).contiguous()
    k = torch.randn(1, k_len, dim, device=device)
    scores = torch.einsum("bhqd,bkd->bhqk", q_const, k)[:, :, 0, :].float()
    return q, k, scores


def reference_mask(support, n_heads, q_len, k_len, device):
    """(h, Sq, Sk) bool selection matrix from a support tensor."""
    ref = torch.zeros(n_heads, q_len, k_len, dtype=torch.bool, device=device)
    for h in range(n_heads):
        for t in range(q_len):
            row = support[0, h, t]
            ref[h, t, row[row >= 0].long()] = True
    return ref


def deadline_mask(dl, n_heads, q_len, k_len, force_sink, force_local, device):
    """(h, Sq, Sk) bool selection matrix implied by the deadlines -- mask_mod's rule, in torch."""
    offset = k_len - q_len
    t_ax = torch.arange(q_len, device=device).view(q_len, 1)
    j_ax = torch.arange(k_len, device=device).view(1, k_len)
    limit = (t_ax + offset).clamp(max=k_len - 1)
    horizon = limit - force_local
    got = torch.zeros(n_heads, q_len, k_len, dtype=torch.bool, device=device)
    for h in range(n_heads):
        sink = j_ax < force_sink
        local = (j_ax > limit - force_local) & (j_ax >= force_sink)
        chosen = (j_ax >= force_sink) & (j_ax <= horizon) & (horizon <= dl[h].view(1, k_len))
        got[h] = (j_ax <= limit) & (sink | local | chosen)
    return got


@pytest.mark.parametrize(
    "n_heads,q_len,k_len,topk,force_sink,force_local",
    [
        (2, 32, 32, 10, 0, 0),
        (2, 32, 32, 10, 4, 6),
        (2, 64, 64, 20, 4, 8),
        (2, 48, 96, 24, 4, 8),  # bottom-right alignment, Sq < Sk
        (1, 128, 128, 32, 4, 16),
        (2, 40, 40, 40, 2, 4),  # topk == k_len, saturated
        (2, 24, 24, 6, 2, 1),  # tiny pool
    ],
)
def test_deadline_mask_matches_streaming_topk(n_heads, q_len, k_len, topk, force_sink, force_local):
    """
    The deadline rule must reproduce the gather path's selection **entry for entry**.

    This is the test that licenses using this path at all: the two arms of the sparse eval are read
    against each other, so a support that differs -- even by a tie-break -- moves scores without
    raising anything.
    """
    device = "cuda:0"
    q, k, scores = query_independent_qk(n_heads, q_len, k_len, device=device)
    support, _ = streaming_topk_support(
        q, k, topk, mask=None, force_sink=force_sink, force_local=force_local
    )
    dl = deadlines(scores[0], topk, force_sink=force_sink, force_local=force_local)

    ref = reference_mask(support, n_heads, q_len, k_len, device)
    got = deadline_mask(dl, n_heads, q_len, k_len, force_sink, force_local, device)
    n_diff = int((ref ^ got).sum())
    assert n_diff == 0, f"{n_diff} entries differ from streaming_topk_support's selection"


def test_every_key_is_selected_by_one_contiguous_interval():
    """
    The property the whole module rests on: a query-independent score gives each key a single
    interval of selecting rows, so one deadline scalar per key is a complete description.

    If this ever fails, the deadline representation is lossy and the path must be withdrawn rather
    than patched -- an interval is not an approximation of a gappy set.
    """
    device = "cuda:0"
    n_heads, q_len, k_len, topk, fs, fl = 2, 96, 96, 24, 4, 8
    q, k, _ = query_independent_qk(n_heads, q_len, k_len, device=device)
    support, _ = streaming_topk_support(q, k, topk, mask=None, force_sink=fs, force_local=fl)
    sel = reference_mask(support, n_heads, q_len, k_len, device)

    for h in range(n_heads):
        for j in range(k_len):
            rows = sel[h, :, j].nonzero().flatten()
            if rows.numel() > 1:
                span = int(rows[-1] - rows[0]) + 1
                assert span == rows.numel(), (
                    f"head {h} key {j} is selected by a NON-contiguous set of rows "
                    f"({rows.numel()} rows spanning {span})"
                )


def test_deadlines_are_tie_break_exact():
    """
    Equal scores must be resolved the same way the gather path resolves them.

    With few distinct score values, many keys tie; the deadline rule counts "keys that beat me", so
    counting ties as beating (or not) shifts how many keys a row keeps. Quantized scores make the
    ties dense enough that a wrong convention cannot pass.
    """
    device = "cuda:0"
    n_heads, q_len, k_len, topk, fs, fl = 2, 64, 64, 16, 4, 8
    torch.manual_seed(0)
    q_const = torch.zeros(1, n_heads, 1, 8, device=device)
    for h in range(n_heads):
        q_const[0, h, 0, h % 8] = 1.0
    q = q_const.expand(1, n_heads, q_len, 8).contiguous()
    # Quantize the indexer keys so the resulting scores collide heavily.
    k = (torch.randn(1, k_len, 8, device=device) * 2).round() / 2
    scores = torch.einsum("bhqd,bkd->bhqk", q_const, k)[:, :, 0, :].float()
    assert len(scores.unique()) < k_len, "scores did not actually collide; test is not exercising ties"

    support, _ = streaming_topk_support(q, k, topk, mask=None, force_sink=fs, force_local=fl)
    dl = deadlines(scores[0], topk, force_sink=fs, force_local=fl)
    ref = reference_mask(support, n_heads, q_len, k_len, device)
    got = deadline_mask(dl, n_heads, q_len, k_len, fs, fl, device)
    assert int((ref ^ got).sum()) == 0, "tie-break differs from the gather path's"


def test_deadlines_rejects_wrong_rank():
    with pytest.raises(ValueError, match="must be"):
        deadlines(torch.zeros(1, 2, 8, device="cuda:0"), 4)


def test_no_forced_slots_still_matches():
    """force_sink = force_local = 0: the mask is purely the top-k rule, with no unconditional slots."""
    device = "cuda:0"
    n_heads, q_len, k_len, topk = 2, 64, 64, 16
    q, k, scores = query_independent_qk(n_heads, q_len, k_len, device=device)
    support, _ = streaming_topk_support(q, k, topk, mask=None, force_sink=0, force_local=0)
    dl = deadlines(scores[0], topk, force_sink=0, force_local=0)
    ref = reference_mask(support, n_heads, q_len, k_len, device)
    got = deadline_mask(dl, n_heads, q_len, k_len, 0, 0, device)
    assert int((ref ^ got).sum()) == 0


@pytest.mark.skipif(not HAS_FLEX, reason="torch has no flex_attention")
def test_attention_output_matches_the_gather_kernel():
    """
    End to end: the same support must give the same attention output as the Triton gather kernel.

    Compared in fp32 with a bf16-scale tolerance -- the two kernels accumulate in a different order,
    so bit-equality is not the claim; agreeing to well inside bf16 resolution is.
    """
    device = "cuda:0"
    n_q, n_kv, dim, q_len, k_len = 8, 2, 64, 256, 256
    topk, fs, fl = 64, 4, 16
    torch.manual_seed(0)
    query = torch.randn(1, n_q, q_len, dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, n_kv, k_len, dim, device=device, dtype=torch.bfloat16)
    value = torch.randn(1, n_kv, k_len, dim, device=device, dtype=torch.bfloat16)

    q_idx, k_idx, scores = query_independent_qk(n_kv, q_len, k_len, device=device)
    support, _ = streaming_topk_support(q_idx, k_idx, topk, mask=None, force_sink=fs, force_local=fl)

    scale = dim**-0.5
    gather_out, _ = sparse_gqa_attention(
        query, key, value, support, scaling=scale, causal=True, block_k=16, precision="ieee"
    )
    flex_out = qi_sparse_attention(
        query, key, value, scores, topk, force_sink=fs, force_local=fl, scaling=scale
    )

    assert flex_out.shape == gather_out.shape
    diff = (flex_out.float() - gather_out.float()).abs().max().item()
    scale_of = gather_out.float().abs().max().item()
    assert diff / scale_of < 5e-2, f"outputs disagree: max abs {diff:.3e} (rel {diff / scale_of:.3e})"


@pytest.mark.skipif(not HAS_FLEX, reason="torch has no flex_attention")
def test_batch_greater_than_one_is_refused():
    """The block mask is built with B=None, so a per-sequence score would be silently dropped."""
    device = "cuda:0"
    query = torch.randn(2, 4, 32, 32, device=device, dtype=torch.bfloat16)
    key = torch.randn(2, 2, 32, 32, device=device, dtype=torch.bfloat16)
    value = torch.randn(2, 2, 32, 32, device=device, dtype=torch.bfloat16)
    scores = torch.randn(2, 2, 32, device=device)
    with pytest.raises(NotImplementedError, match="batch 1"):
        qi_sparse_attention(query, key, value, scores, 8, force_sink=1, force_local=2)


def test_scalar_indexer_declares_query_independence():
    """The fast path must be selected by a declared capability, not an isinstance check."""
    from kvpress.presses.gqa_indexer.indexer import GQAIndexer
    from kvpress.presses.gqa_indexer.scalar_indexer import ScalarIndexer

    assert ScalarIndexer.is_query_independent is True
    assert GQAIndexer.is_query_independent is False


def test_block_mask_is_compiled_dynamic():
    """``create_block_mask`` must not go back to automatic (``dynamic=None``) mode.

    Under ``dynamic=None`` a run that sees several key lengths dies with "CUDA error: an illegal
    memory access was encountered" inside the inductor reduction this call compiles to. Guarding the
    constant (rather than only the behaviour) makes the revert loud: the behavioural test below
    needs a GPU and enough distinct lengths to trip the transition, so on a CPU-only runner this
    assertion is the only thing standing between a one-word edit and a crash in the next sweep.
    """
    from kvpress.presses.gqa_indexer import qi_flex_attention as qf

    assert qf._BLOCK_MASK_DYNAMIC is True, (
        "create_block_mask must be compiled with dynamic=True; dynamic=None crashes with an "
        "illegal memory access once a second key length arrives"
    )


@pytest.mark.skipif(not HAS_FLEX, reason="torch has no flex_attention")
def test_many_lengths_through_one_compiled_callable():
    """
    Many key lengths through **one process and one compiled callable**, still exact.

    This is the coverage *shape* that was missing when the ``dynamic=None`` crash reached a full
    RULER sweep. Every other test here parametrises the shape, so pytest gives each length its own
    fresh compile and the specialise-then-generalise transition is never exercised -- yet that
    transition is the only thing that was broken, and it broke only after a second distinct length
    arrived. The four shards of the real run died at different contexts (45/163, 121/163, 117/162,
    109/162), the signature of a shape-sequence-dependent recompile.

    Lengths are deliberately not multiples of :data:`FLEX_BLOCK` and not monotonic, so the mask build
    sees partial trailing blocks and both grows and shrinks between calls. Selection is checked
    against ``streaming_topk_support`` at every length, so a compile mode that silently changed the
    support would fail here rather than merely not crashing.

    **Measured limitation, stated so this test is not trusted for more than it does.** Reverting
    ``_BLOCK_MASK_DYNAMIC`` to ``None`` does *not* make this test fail -- at these sub-1K lengths the
    faulty kernel is never generated. Reproducing the crash needed the real model at RULER 8K
    (~8-11K keys, 36 layers). So the actual regression guard for that revert is
    ``test_block_mask_is_compiled_dynamic``, and this test's job is narrower: it pins the *exactness*
    of the support across a recompile sequence, which is the property a future mode change could
    break silently rather than loudly.
    """
    device = "cuda:0"
    n_q, n_kv, dim = 8, 2, 64
    topk, fs, fl = 64, 4, 16
    # Grow, shrink, repeat a seen length, and straddle 128-boundaries.
    lengths = [257, 384, 300, 511, 257, 640, 333]

    for k_len in lengths:
        q_len = k_len
        torch.manual_seed(k_len)
        query = torch.randn(1, n_q, q_len, dim, device=device, dtype=torch.bfloat16)
        key = torch.randn(1, n_kv, k_len, dim, device=device, dtype=torch.bfloat16)
        value = torch.randn(1, n_kv, k_len, dim, device=device, dtype=torch.bfloat16)

        q_idx, k_idx, scores = query_independent_qk(n_kv, q_len, k_len, device=device, seed=k_len)
        out = qi_sparse_attention(
            query, key, value, scores, topk, force_sink=fs, force_local=fl, scaling=dim**-0.5
        )
        # Synchronize per length: the failure mode is an async illegal access, so without this the
        # error could surface inside a later length and misattribute the blame.
        torch.cuda.synchronize()
        assert out.shape == (1, n_q, q_len, dim)
        assert torch.isfinite(out.float()).all(), f"non-finite output at k_len={k_len}"

        # Selection must still be exact at every length, not just crash-free.
        support, _ = streaming_topk_support(
            q_idx, k_idx, topk, mask=None, force_sink=fs, force_local=fl
        )
        dl = deadlines(scores[0], topk, force_sink=fs, force_local=fl)
        ref = reference_mask(support, n_kv, q_len, k_len, device)
        got = deadline_mask(dl, n_kv, q_len, k_len, fs, fl, device)
        assert int((ref ^ got).sum()) == 0, f"support differs from the gather path at k_len={k_len}"
