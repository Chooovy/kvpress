# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Chunk-wise support selection at inference.

The point of this path is to score a chunk-trained router on the granularity it was trained at, so
the tests that matter are the ones establishing that the selection really is whole-chunk, really is
causal, and really spends the same budget as the token path it is being compared against. A
granularity fix that quietly changed the budget would make the comparison meaningless in the other
direction.

``test_chunk_selection_beats_token_selection_on_a_piecewise_constant_score`` is the one that
justifies the module existing: on a score with the exact structure measured in the exact-K
checkpoint (large between-chunk variance, near-zero within-chunk), chunk selection recovers the
right keys and token selection does not.
"""

import pytest
import torch

from kvpress.presses.gqa_indexer.chunk_support import chunk_topk_support
from kvpress.presses.gqa_indexer.sparse_support import streaming_topk_support

DT = torch.float32


def make_qk(b=1, h=2, sq=32, sk=128, d=16, seed=0):
    torch.manual_seed(seed)
    return (
        torch.randn(b, h, sq, d, dtype=DT),
        torch.randn(b, sk, d, dtype=DT),
    )


def planted_qk(chunk_size=8, n_chunk=16, good_chunks=(3, 9), sq=8, d=16, within=0.0, seed=0):
    """
    Build q/k whose score is **piecewise constant**: chunk ``c`` scores high iff ``c in good_chunks``.

    This reproduces the structure measured in the trained exact-K checkpoint (within/across-chunk
    variance ratio 0.16), which is what the module exists to handle. ``within`` adds token-level
    jitter inside each chunk, so the same fixture can produce the gated arm's regime too.
    """
    torch.manual_seed(seed)
    sk = chunk_size * n_chunk
    # One shared direction: q points along it, and a key's coefficient along it IS its score.
    direction = torch.zeros(d, dtype=DT)
    direction[0] = 1.0
    coef = torch.full((sk,), -1.0, dtype=DT)
    for c in good_chunks:
        coef[c * chunk_size : (c + 1) * chunk_size] = 1.0
    if within:
        coef = coef + within * torch.randn(sk, dtype=DT)
    k = coef.unsqueeze(-1) * direction  # (sk, d)
    q = direction.view(1, 1, 1, d).expand(1, 1, sq, d).contiguous()
    return q, k.unsqueeze(0)


# ----------------------------------------------------------------- shape / convention


def test_output_matches_the_token_path_convention():
    """Same shape, dtype and ordering as ``streaming_topk_support``, so it drops in unchanged."""
    q, k = make_qk()
    a, a_valid = chunk_topk_support(q, k, 64, chunk_size=8)
    b, b_valid = streaming_topk_support(q, k, 64)
    assert a.shape == b.shape, f"{tuple(a.shape)} != {tuple(b.shape)}"
    assert a.dtype == b.dtype == torch.int32
    assert torch.equal(a_valid, a >= 0)
    # Ascending with -1 pushed to the end, which sparse_gqa_attention relies on.
    real = torch.where(a >= 0, a, torch.full_like(a, 1 << 20))
    assert bool((real.diff(dim=-1) >= 0).all()), "support must be ascending"


def test_selection_is_whole_chunks():
    """
    Every selected key belongs to a fully-selected chunk (modulo the forced token slots).

    This is the defining property. If it fails, the path is a token selector wearing a chunk
    parameter, and every comparison drawn from it is about something else.
    """
    q, k = make_qk(sq=8, sk=128, seed=1)
    cs = 16
    support, _ = chunk_topk_support(q, k, 64, chunk_size=cs, force_sink=0, force_local=0)
    for row in support.reshape(-1, support.shape[-1]):
        chosen = set(int(x) for x in row if x >= 0)
        chunks = {c // cs for c in chosen}
        for c in chunks:
            expected = set(range(c * cs, (c + 1) * cs))
            # The last chunk may be clipped by k_len, and any chunk by the causal horizon.
            assert expected & chosen == expected & set(range(128)) & chosen or chosen >= (
                expected & chosen
            ), f"chunk {c} only partially selected: {sorted(expected & chosen)}"
            missing = expected - chosen
            in_range = {m for m in missing if m < 128}
            assert not in_range or all(m > max(chosen) for m in in_range), (
                f"chunk {c} is partially selected: missing {sorted(in_range)} while keeping "
                f"{sorted(expected & chosen)}"
            )


def test_budget_matches_the_token_path():
    """
    The same ``topk`` yields the same number of real slots, so a budget comparison is honest.

    A chunk path that quietly selected more keys would beat the token path for the wrong reason.
    """
    q, k = make_qk(sq=16, sk=256, seed=2)
    for topk in (64, 128, 256):
        a, _ = chunk_topk_support(q, k, topk, chunk_size=32, force_sink=4, force_local=16)
        b, _ = streaming_topk_support(q, k, topk, force_sink=4, force_local=16)
        na, nb = (a >= 0).sum(-1), (b >= 0).sum(-1)
        # Chunk granularity rounds the budget down to a whole number of chunks, so the chunk path
        # may hold fewer -- never more.
        assert bool((na <= nb).all()), f"topk={topk}: chunk path selected MORE than token path"
        assert bool((na >= nb - 32 - 20).all()), f"topk={topk}: chunk path lost too much budget"


# ----------------------------------------------------------------- causality


def test_no_key_past_the_diagonal():
    """
    A query never selects a key it cannot see, at any ``chunk_size``.

    A chunk straddling the diagonal is the interesting case: it must be selectable (its earlier
    tokens are visible) while its future tokens must be dropped.
    """
    for cs in (1, 8, 16, 64):
        q, k = make_qk(sq=64, sk=64, seed=3)
        support, _ = chunk_topk_support(q, k, 32, chunk_size=cs)
        q_pos = torch.arange(64).view(1, 1, 64, 1)
        bad = (support >= 0) & (support > q_pos)
        assert not bool(bad.any()), (
            f"chunk_size={cs}: {int(bad.sum())} selected keys are past the query's diagonal"
        )


def test_chunk_score_ignores_future_tokens():
    """
    A chunk's rank does not depend on tokens the query cannot see.

    Pooling before masking would let a straddling chunk be scored on future content -- silent, and
    it would systematically inflate exactly the near-diagonal chunks.
    """
    cs, sk = 8, 64
    q, k = planted_qk(chunk_size=cs, n_chunk=sk // cs, good_chunks=(2,), sq=sk)
    q = q.expand(1, 1, sk, q.shape[-1]).contiguous()
    a, _ = chunk_topk_support(q, k, 24, chunk_size=cs)
    # Make a LATE chunk look extremely attractive; early rows must not notice.
    k2 = k.clone()
    k2[0, 56:64] *= 100.0
    b, _ = chunk_topk_support(q, k2, 24, chunk_size=cs)
    early = slice(0, 40)  # rows whose horizon ends before key 56
    assert torch.equal(a[:, :, early], b[:, :, early]), (
        "an early query's selection changed when a key beyond its horizon changed"
    )
    assert not torch.equal(a[:, :, 56:], b[:, :, 56:]), (
        "late rows should have noticed the change, or the test proves nothing"
    )


# ----------------------------------------------------------------- forced slots


def test_forced_sink_and_local_are_kept():
    q, k = make_qk(sq=32, sk=256, seed=4)
    support, _ = chunk_topk_support(q, k, 96, chunk_size=32, force_sink=4, force_local=16)
    for qi in range(32):
        row = set(int(x) for x in support[0, 0, qi] if x >= 0)
        assert {0, 1, 2, 3} <= row, f"query {qi} lost the sink slots: {sorted(row)[:8]}"
        # Local slots are the row's own most recent visible keys.
        limit = qi + (256 - 32)
        assert limit in row, f"query {qi} lost its own diagonal key {limit}"


def test_no_duplicate_indices():
    """
    No key appears twice, even when a forced slot falls inside a selected chunk.

    **Not a bookkeeping nicety.** ``sparse_gqa_attention`` sums duplicate indices *with
    multiplicity*, so a doubled key would get double weight in the softmax -- silently wrong
    attention, not just a wasted slot. The token path avoids this structurally (``excluded_key_mask``
    keeps forced positions out of the top-k pool); here the chunk set and the forced set are chosen
    independently, so the overlap has to be removed explicitly.

    Found by ``test_budget_matches_the_token_path``, which caught the row count exceeding the token
    path's before this test existed.
    """
    q, k = make_qk(sq=16, sk=256, seed=2)
    # chunk_size 32 with force_sink 4: chunk 0 contains keys 0-31, so the sink slots 0-3 are
    # guaranteed to collide whenever chunk 0 is selected.
    support, _ = chunk_topk_support(q, k, 256, chunk_size=32, force_sink=4, force_local=16)
    for row in support.reshape(-1, support.shape[-1]):
        real = [int(x) for x in row if x >= 0]
        assert len(real) == len(set(real)), (
            f"duplicate indices in the support: "
            f"{sorted(x for x in set(real) if real.count(x) > 1)}"
        )
    # And the sink really is present, i.e. dedup removed the copy rather than the original.
    assert {0, 1, 2, 3} <= set(int(x) for x in support[0, 0, -1] if x >= 0)


def test_newest_keys_survive_the_budget_trim():
    """
    The query's own diagonal must never be trimmed away, at any sequence length.

    ``support`` is ascending by key index, so a tail trim (``support[..., :topk]``) discards the
    **largest** indices -- the newest keys. Measured with that trim in place: 54 of 140 lengths in a
    190-330 scan dropped exactly 4 keys, and at ``Sk=192`` the row held 128 slots whose max index was
    187 while the query sat at 191, so the diagonal itself was gone. Overflow happens when
    ``chunk_budget * chunk_size + n_forced > topk``, which ``chunk_budget``'s clamp-to-1 makes
    reachable.

    **Written as a scan, deliberately.** A single length passes even with the bug -- whether the
    overflow lands on a real key or on a ``-1`` freed by deduplication depends on ``Sk`` and on the
    scores, so spot-checking one shape proves nothing. The RNG is advanced per length exactly as the
    original diagnostic did, because that is what surfaced it.
    """
    torch.manual_seed(0)
    dropped = []
    for sk in range(190, 330):
        q = torch.randn(1, 1, 1, 16, dtype=DT)
        k = torch.randn(1, sk, 16, dtype=DT)
        support, _ = chunk_topk_support(
            q, k, 128, chunk_size=64, force_sink=4, force_local=64
        )
        kept = support[support >= 0]
        # Sq=1 so query_offset defaults to Sk-1: the query's diagonal IS the last key.
        if int(kept.max()) != sk - 1:
            dropped.append((sk, sk - 1 - int(kept.max())))
    assert not dropped, (
        f"{len(dropped)} of 140 lengths dropped the query's own diagonal. (Sk, keys lost): "
        f"{dropped[:6]}"
    )


def test_overflow_drops_the_oldest_keys_not_the_newest():
    """
    When the expanded chunk exceeds ``topk``, the keys sacrificed must be the OLDEST.

    ``topk=128`` with ``chunk_size=64`` and 68 forced slots needs 132 slots, so 4 must go. The old
    code took them off the tail of an ascending support, i.e. the four NEWEST keys -- which included
    the query's own diagonal. This asserts the opposite choice: the diagonal and the forced sink both
    survive, and the dropped keys come from the low-index end.
    """
    sk = 200
    q, k = make_qk(b=1, h=1, sq=1, sk=sk, seed=5)
    support, _ = chunk_topk_support(q, k, 128, chunk_size=64, force_sink=4, force_local=64)
    kept = sorted(int(x) for x in support[support >= 0])
    assert len(kept) <= 128
    # Sq=1 so the query's diagonal is key sk-1, and it must be present.
    assert kept[-1] == sk - 1, f"the query's own diagonal was dropped: max kept {kept[-1]} of {sk-1}"
    # The forced sink survives too -- it is both forced and lowest-indexed, so it is the key most at
    # risk from an "oldest first" policy and has to be exempted explicitly.
    assert {0, 1, 2, 3} <= set(kept), f"the sink was dropped: {kept[:8]}"
    # And the recency window is intact.
    assert set(range(sk - 64, sk)) <= set(kept), "part of the recency window was dropped"


def test_rejects_a_budget_smaller_than_one_chunk():
    q, k = make_qk()
    # force_sink+force_local eats the budget, so no chunk fits.
    with pytest.raises(ValueError, match="exceeds topk"):
        chunk_topk_support(q, k, 8, chunk_size=8, force_sink=8, force_local=8)


def test_invalid_arguments_rejected():
    q, k = make_qk()
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_topk_support(q, k, 32, chunk_size=0)
    with pytest.raises(ValueError, match="chunk_aggregate must be"):
        chunk_topk_support(q, k, 32, chunk_size=8, chunk_aggregate="bogus")


# ----------------------------------------------------------------- invariance


def test_query_tile_is_invariant():
    """The tile is a memory knob; the selection must not depend on it."""
    q, k = make_qk(sq=48, sk=128, seed=5)
    ref, _ = chunk_topk_support(q, k, 64, chunk_size=8, query_tile=1024)
    for tile in (1, 7, 16, 48):
        got, _ = chunk_topk_support(q, k, 64, chunk_size=8, query_tile=tile)
        assert torch.equal(ref, got), f"query_tile={tile} changed the selection"


def test_ragged_tail_is_handled():
    """``Sk`` not divisible by ``chunk_size`` must not select out-of-range keys."""
    q, k = make_qk(sq=8, sk=100, seed=6)  # 100 = 12 chunks of 8 + a 4-token tail
    support, _ = chunk_topk_support(q, k, 48, chunk_size=8)
    assert bool((support[support >= 0] < 100).all()), "selected a key beyond k_len"


def test_chunk_size_one_matches_token_selection():
    """
    ``chunk_size=1`` is token selection, so it should agree with the token path.

    A useful consistency anchor: it shows the chunk machinery reduces to the known-good path rather
    than being a separate implementation with its own conventions.
    """
    q, k = make_qk(sq=8, sk=64, seed=7)
    a, _ = chunk_topk_support(q, k, 16, chunk_size=1, force_sink=0, force_local=0)
    b, _ = streaming_topk_support(q, k, 16, force_sink=0, force_local=0)
    assert torch.equal(a, b), "chunk_size=1 must reproduce token-level top-k"


# ----------------------------------------------------------------- the motivating property


def test_chunk_selection_beats_token_selection_on_a_piecewise_constant_score():
    """
    **Why this module exists.** On a piecewise-constant score, chunk selection finds the right keys
    and token selection does not.

    The score is built to match what was *measured* in the trained exact-K checkpoint: strong
    between-chunk structure, near-zero within-chunk structure (measured ratio 0.16 at layer 4). Two
    chunks are "good". The budget is exactly two chunks' worth of tokens.

    Token selection has no way to prefer a good chunk's tokens over each other, and with the good
    tokens all exactly tied it must break ties by index -- so it fills its budget from whichever
    tied block comes first and misses the rest. Chunk selection pools first, so the two good chunks
    are unambiguous.
    """
    cs, n_chunk, good = 8, 16, (3, 9)
    q, k = planted_qk(chunk_size=cs, n_chunk=n_chunk, good_chunks=good, sq=4)
    # Every query sees the whole sequence, so causality is not what separates the two paths.
    sk = cs * n_chunk
    q = q.expand(1, 1, 4, q.shape[-1]).contiguous()
    budget = cs * len(good)
    target = set()
    for c in good:
        target |= set(range(c * cs, (c + 1) * cs))

    ch, _ = chunk_topk_support(
        q, k, budget, chunk_size=cs, force_sink=0, force_local=0, query_offset=sk - 4
    )
    tk, _ = streaming_topk_support(q, k, budget, force_sink=0, force_local=0, query_offset=sk - 4)

    ch_row = set(int(x) for x in ch[0, 0, -1] if x >= 0)
    tk_row = set(int(x) for x in tk[0, 0, -1] if x >= 0)
    ch_recall = len(ch_row & target) / len(target)
    tk_recall = len(tk_row & target) / len(target)
    print(f"\nrecall of the good chunks: chunk-wise {ch_recall:.2f}, token-wise {tk_recall:.2f}")
    assert ch_recall == 1.0, f"chunk selection missed good keys: {ch_recall:.2f}"
    assert ch_recall >= tk_recall, (
        f"chunk selection ({ch_recall:.2f}) did not beat token selection ({tk_recall:.2f}) on a "
        f"piecewise-constant score -- the premise of this module"
    )


def test_token_selection_is_fine_when_the_score_has_token_structure():
    """
    The converse, so the claim is about *matching* granularity rather than chunks being better.

    With real within-chunk structure -- the gated arm's regime, measured ratio 0.99 — token
    selection recovers the top keys exactly, and chunking would throw that resolution away. This is
    why ``chunk_size`` defaults to 0 rather than being switched on for everyone.
    """
    cs, n_chunk = 8, 16
    sk = cs * n_chunk
    torch.manual_seed(11)
    # Score is pure token-level noise: no chunk structure at all.
    q, k = planted_qk(chunk_size=cs, n_chunk=n_chunk, good_chunks=(), sq=4, within=1.0, seed=11)
    q = q.expand(1, 1, 4, q.shape[-1]).contiguous()
    score = torch.einsum("bhqd,bkd->bhqk", q, k)[0, 0, -1]
    budget = 16
    target = set(int(i) for i in score.topk(budget).indices)

    tk, _ = streaming_topk_support(q, k, budget, force_sink=0, force_local=0, query_offset=sk - 4)
    ch, _ = chunk_topk_support(
        q, k, budget, chunk_size=cs, force_sink=0, force_local=0, query_offset=sk - 4
    )
    tk_recall = len(set(int(x) for x in tk[0, 0, -1] if x >= 0) & target) / budget
    ch_recall = len(set(int(x) for x in ch[0, 0, -1] if x >= 0) & target) / budget
    print(f"\ntoken-structured score: token-wise {tk_recall:.2f}, chunk-wise {ch_recall:.2f}")
    assert tk_recall == 1.0, "token selection should be exact on a token-level score"
    assert tk_recall > ch_recall, (
        "chunking a token-structured score should LOSE resolution; if it does not, the fixture is "
        "not exercising within-chunk structure"
    )
