# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the LongCE weight cache and its objective.

The cache exists to make one specific failure impossible: **weights attached to the wrong tokens**.
That failure is silent -- the loss still falls, ``weight_participation`` still looks sane, and only
the benchmark reveals it days later. So most of what is tested here is the guard rails rather than
the happy path:

* **a token mismatch raises**, and it raises at the *stage width actually drawn* rather than only at
  the cache's full width. A cache verified only at 16K would go unchecked through the 8K stage,
  which is where the curriculum spends the first half of its steps.
* **truncating a cached vector to a shorter stage is exact.** This is what lets one cache serve the
  whole curriculum, and it holds only because the losses are causal -- so it is asserted against
  independently computed weights rather than assumed.
* **keys are ``doc_id``, not position.** The loader shuffles rows and partitions shards by
  ``(rank, worker)``, so a positional key would break silently whenever the world size or seed
  changed while still returning a correctly-shaped vector.
* **the objective reduces to the plain mean when every weight is 1.** LongCE's own design point is
  "most tokens near 1", so this is the limit the run must degrade to gracefully -- and it is what
  makes ``--peak-lr`` transferable from the plain arm.
* **the weights are detached**, so no gradient can flow into them. Otherwise the objective would be
  optimizable by making the loss *larger* where the weight is large.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from kvpress.presses.gqa_indexer.longce_weights import (
    CACHE_VERSION,
    LongCEWeightCache,
    WeightCacheMeta,
    longce_weighted_loss,
    longce_weights,
    shard_cache_path,
    token_checksum,
    write_shard_cache,
)

WIDTHS = (16, 32)


def _meta(seq_len=32, trunc_len=8, widths=WIDTHS):
    return WeightCacheMeta(
        seq_len=seq_len,
        trunc_len=trunc_len,
        window=8,
        gamma=5.0,
        model="toy",
        scored_from=trunc_len - 1,
        checksum_widths=tuple(widths),
    )


def _build(tmp_path, *, n_docs=3, seq_len=32, widths=WIDTHS, doc_prefix="doc"):
    """A small cache plus the token rows it was built from, as the loader would draw them."""
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 1000, size=(n_docs, seq_len), dtype=np.int64)
    weights = rng.uniform(0.5, 5.0, size=(n_docs, seq_len - 1)).astype(np.float16)
    doc_ids = [f"{doc_prefix}{i}" for i in range(n_docs)]
    checksums = np.array(
        [[token_checksum(row[:w]) for w in widths] for row in tokens], dtype="U16"
    )
    path = shard_cache_path(tmp_path, "2e16", "shard-0000")
    write_shard_cache(
        path,
        doc_ids=doc_ids,
        weights=weights,
        checksums=checksums,
        meta=_meta(seq_len=seq_len, widths=widths),
    )
    return tokens, weights, doc_ids


# ---------------------------------------------------------------------------------------------
# the weight itself
# ---------------------------------------------------------------------------------------------


def test_weight_is_exp_of_the_discrepancy_clamped_at_gamma():
    """``w = min(exp(L_short - L_long), gamma)``, the paper's ``Isoft`` (Eq. 7)."""
    long_loss = torch.tensor([1.0, 1.0, 1.0])
    short_loss = torch.tensor([1.0, 2.0, 9.0])
    scored = torch.ones(3, dtype=torch.bool)
    w = longce_weights(long_loss, short_loss, scored, gamma=5.0)
    assert w[0] == pytest.approx(1.0)  # no gain from long context -> neutral
    assert w[1] == pytest.approx(np.e, rel=1e-5)
    assert w[2] == pytest.approx(5.0)  # exp(8) clamped


def test_unscored_positions_get_exactly_one():
    """
    Unscored positions fall back to 1.0, the neutral value of this weighting.

    The reference leaves them at 1 too, but *implicitly*; here it is explicit and paired with a
    ``scored`` mask so the trainer can tell "measured 1.0" from "never measured".
    """
    long_loss = torch.tensor([1.0, 1.0])
    short_loss = torch.tensor([5.0, 5.0])  # a large gain, which must still be ignored
    scored = torch.tensor([False, True])
    w = longce_weights(long_loss, short_loss, scored, gamma=5.0)
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(5.0)


# ---------------------------------------------------------------------------------------------
# the alignment guard -- the whole reason the cache carries digests
# ---------------------------------------------------------------------------------------------


def test_lookup_returns_the_cached_weights_for_matching_tokens(tmp_path):
    tokens, weights, doc_ids = _build(tmp_path)
    cache = LongCEWeightCache(tmp_path, seq_len=32)
    got = cache.lookup(doc_ids[1], tokens[1])
    np.testing.assert_allclose(got, weights[1].astype(np.float32))


def test_lookup_raises_when_the_tokens_do_not_match(tmp_path):
    """
    The core guard. Wrong tokens must be fatal, not a warning.

    A mismatch means the cached weights describe different text: training continues happily,
    ``weight_participation`` looks normal, and the objective silently optimizes the wrong positions.
    Nothing downstream can detect that, so it has to stop here.
    """
    tokens, _, doc_ids = _build(tmp_path)
    cache = LongCEWeightCache(tmp_path, seq_len=32)
    wrong = tokens[1].copy()
    wrong[5] += 1  # a single token differs
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        cache.lookup(doc_ids[1], wrong)


def test_mismatch_is_caught_at_the_shorter_stage_width_too(tmp_path):
    """
    Verification must work at the width the stage actually draws, not just the cache's full width.

    The curriculum's first stage is 8K against a 16K cache, so a single full-width digest would
    leave that stage unverified -- half the run. This is why digests are stored per width.
    """
    tokens, _, doc_ids = _build(tmp_path)
    cache = LongCEWeightCache(tmp_path, seq_len=16)
    prefix = tokens[0][:16]
    cache.lookup(doc_ids[0], prefix)  # the true prefix verifies

    corrupted = prefix.copy()
    corrupted[3] += 1
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        cache.lookup(doc_ids[0], corrupted)


def test_a_stage_width_with_no_digest_is_refused(tmp_path):
    """
    Rather than skipping verification for an unlisted width, refuse to run.

    Silently proceeding unverified is the exact failure mode this cache exists to prevent, so an
    unverifiable configuration is an error and the message says how to fix it.
    """
    _build(tmp_path, widths=(32,))
    with pytest.raises(ValueError, match="cannot be verified"):
        LongCEWeightCache(tmp_path, seq_len=16)


def test_truncating_to_a_shorter_stage_matches_the_full_width_prefix(tmp_path):
    """
    One cache serves every stage: the 16-wide read is the 32-wide read's prefix, exactly.

    Sound because the losses are causal -- position ``i``'s long-context loss depends only on tokens
    ``<= i`` and its short-context loss only on its own window, which the prefix also contains.
    """
    tokens, weights, doc_ids = _build(tmp_path)
    long_stage = LongCEWeightCache(tmp_path, seq_len=32)
    short_stage = LongCEWeightCache(tmp_path, seq_len=16)
    full = long_stage.lookup(doc_ids[2], tokens[2])
    prefix = short_stage.lookup(doc_ids[2], tokens[2][:16])
    assert prefix.shape == (15,)
    np.testing.assert_allclose(prefix, full[:15])


def test_seq_len_beyond_the_cache_width_is_rejected(tmp_path):
    """Those positions were never scored, so there is nothing to truncate *to*."""
    _build(tmp_path, seq_len=32)
    with pytest.raises(ValueError, match="exceeds the cache's width"):
        LongCEWeightCache(tmp_path, seq_len=64)


def test_wrong_length_token_row_is_rejected(tmp_path):
    """A row of the wrong width cannot be the one this stage drew; do not hash it and hope."""
    tokens, _, doc_ids = _build(tmp_path)
    cache = LongCEWeightCache(tmp_path, seq_len=32)
    with pytest.raises(ValueError, match="token row to verify"):
        cache.lookup(doc_ids[0], tokens[0][:20])


def test_checksum_is_dtype_invariant():
    """
    The corpus stores ``uint32`` and the loader hands back ``int64``; both must hash alike.

    Otherwise every lookup in a correct run would fail, which reads as data corruption.
    """
    row = np.arange(16)
    assert token_checksum(row.astype(np.uint32)) == token_checksum(row.astype(np.int64))


def test_checksum_distinguishes_prefixes_of_different_length():
    """A digest covers exactly what it was given, so the 16- and 32-wide digests differ."""
    row = np.arange(32)
    assert token_checksum(row[:16]) != token_checksum(row)


# ---------------------------------------------------------------------------------------------
# keying and metadata
# ---------------------------------------------------------------------------------------------


def test_cache_is_keyed_by_doc_id_not_by_position(tmp_path):
    """
    An unknown ``doc_id`` is a ``KeyError``, and a known one resolves regardless of order.

    Positional keys would survive a re-shard or a seed change while returning the wrong document's
    weights -- correctly shaped, silently wrong.
    """
    tokens, weights, doc_ids = _build(tmp_path)
    cache = LongCEWeightCache(tmp_path, seq_len=32)
    assert cache.doc_ids() == sorted(doc_ids)
    assert "nope" not in cache
    with pytest.raises(KeyError):
        cache.lookup("nope", tokens[0])
    # Reverse order still resolves each id to its own row.
    for i in reversed(range(len(doc_ids))):
        np.testing.assert_allclose(
            cache.lookup(doc_ids[i], tokens[i]), weights[i].astype(np.float32)
        )


def test_a_stale_cache_version_is_refused(tmp_path):
    """A changed layout must be rebuilt, not reinterpreted."""
    payload = _meta().to_json()
    payload["version"] = CACHE_VERSION - 1
    with pytest.raises(ValueError, match="version"):
        WeightCacheMeta.from_json(payload)


def test_shards_built_with_different_settings_cannot_share_a_root(tmp_path):
    """
    Mixing settings would apply different weightings to different documents in one run.

    That is not a partially-built cache -- it is a silently inhomogeneous objective, so it is caught
    when the root is opened rather than tolerated.
    """
    _build(tmp_path)
    rng = np.random.default_rng(1)
    tokens = rng.integers(0, 1000, size=(2, 32), dtype=np.int64)
    write_shard_cache(
        shard_cache_path(tmp_path, "2e17", "shard-0000"),
        doc_ids=["other0", "other1"],
        weights=rng.uniform(0.5, 5.0, size=(2, 31)).astype(np.float16),
        checksums=np.array(
            [[token_checksum(r[:w]) for w in WIDTHS] for r in tokens], dtype="U16"
        ),
        meta=_meta(trunc_len=16),  # a different K
    )
    with pytest.raises(ValueError, match="built with"):
        LongCEWeightCache(tmp_path, seq_len=32)


def test_missing_cache_root_names_the_builder(tmp_path):
    with pytest.raises(FileNotFoundError, match="precompute_longce_weights"):
        LongCEWeightCache(tmp_path / "absent", seq_len=32)


def test_empty_cache_root_is_distinguished_from_a_missing_one(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no .npz shards"):
        LongCEWeightCache(tmp_path / "empty", seq_len=32)


def test_write_is_atomic_leaving_no_temporary_behind(tmp_path):
    """An interrupted precompute must leave complete shards or nothing -- never a partial file."""
    _build(tmp_path)
    assert not list(tmp_path.glob("**/*.tmp"))
    assert shard_cache_path(tmp_path, "2e16", "shard-0000").is_file()


def test_meta_survives_a_round_trip(tmp_path):
    _build(tmp_path)
    cache = LongCEWeightCache(tmp_path, seq_len=32)
    assert cache.meta == _meta()
    assert cache.summary()["documents"] == 3


def test_write_rejects_mismatched_column_count(tmp_path):
    """Digest columns must line up with ``meta.checksum_widths`` or lookups read the wrong column."""
    with pytest.raises(ValueError, match="columns"):
        write_shard_cache(
            shard_cache_path(tmp_path, "2e16", "bad"),
            doc_ids=["a"],
            weights=np.ones((1, 31), dtype=np.float16),
            checksums=np.array([["x" * 16]], dtype="U16"),  # 1 column, meta lists 2
            meta=_meta(),
        )


# ---------------------------------------------------------------------------------------------
# the objective
# ---------------------------------------------------------------------------------------------


def test_all_ones_weights_reduce_to_the_plain_mean():
    """
    LongCE's design point is "most tokens near 1", so the all-ones limit must be the plain mean.

    This is what makes ``--peak-lr`` transfer from the plain arm and the logged loss readable
    against it, and it means a cache miss (which falls back to 1.0) degrades gracefully.
    """
    losses = torch.tensor([1.0, 2.0, 3.0, 4.0])
    loss, stats = longce_weighted_loss(losses, torch.ones(4))
    assert float(loss) == pytest.approx(float(losses.mean()))
    assert stats["weight_participation"] == pytest.approx(1.0)


def test_participation_falls_when_the_weighting_concentrates():
    """
    ``(sum w)^2 / (n sum w^2)`` -- the same statistic the trainer logs for the delta arm.

    Comparable by construction: the failed delta run sat at 0.13-0.18 and LongCE measured 0.66-0.87
    offline, so this number is what separates the two runs rather than the loss.

    ``[1, 1, 1, 100]`` gives ``103^2 / (4 * 10003) = 0.265``: one position holds 97% of the mass, so
    the effective fraction collapses towards ``1/n = 0.25``.
    """
    spiked = torch.tensor([1.0, 1.0, 1.0, 100.0])
    _, stats = longce_weighted_loss(torch.ones(4), spiked)
    assert stats["weight_participation"] == pytest.approx(0.265, abs=0.01)
    # Uniform weights over the same positions would be 1.0; anything near 1/n means concentrated.
    assert stats["weight_participation"] < 0.3


def test_weighted_loss_favours_the_upweighted_positions():
    """The weighted mean must move toward the loss at the heavy position, not away from it."""
    losses = torch.tensor([0.0, 10.0])
    loss, _ = longce_weighted_loss(losses, torch.tensor([1.0, 4.0]))
    assert float(loss) == pytest.approx(8.0)


def test_mask_excludes_invalid_positions_entirely():
    """A masked position must not reach either the numerator or the denominator."""
    losses = torch.tensor([1.0, 99.0, 3.0])
    mask = torch.tensor([True, False, True])
    loss, _ = longce_weighted_loss(losses, torch.ones(3), mask=mask)
    assert float(loss) == pytest.approx(2.0)


def test_gradient_flows_to_the_loss_but_never_to_the_weights():
    """
    ``d loss / d L_t = w_t / sum w``, and the weights take no gradient.

    If gradient reached the weights the objective could be reduced by making the loss larger where
    the weight is large -- optimizable by getting worse.
    """
    losses = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    weights = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    loss, _ = longce_weighted_loss(losses, weights)
    loss.backward()
    np.testing.assert_allclose(
        losses.grad.numpy(), np.array([1.0, 2.0, 3.0]) / 6.0, rtol=1e-6
    )
    assert weights.grad is None


def test_zero_weight_sum_raises_rather_than_dividing_by_zero():
    """An empty objective is a bug worth naming, not a NaN to propagate."""
    with pytest.raises(RuntimeError, match="undefined"):
        longce_weighted_loss(torch.ones(3), torch.zeros(3))


def test_upweighted_frac_reports_the_paper_s_shape():
    """
    ``weight_upweighted_frac`` is this objective's analogue of ``delta_positive_frac``.

    Near 0 means the cache is effectively all ones and the run is the plain mean with extra
    machinery -- the thing to check before committing to a long run.
    """
    _, stats = longce_weighted_loss(
        torch.ones(4), torch.tensor([1.0, 1.0, 1.0, 3.0])
    )
    assert stats["weight_upweighted_frac"] == pytest.approx(0.25)
    assert stats["weight_median"] == pytest.approx(1.0)
    assert stats["weight_max"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------------------------
# End to end through the trainer
# ---------------------------------------------------------------------------------------------

UNIT_TEST_MODEL = "MaxJeblick/llama2-0b-unit-test"


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(UNIT_TEST_MODEL).eval()


def _trainer_and_press(model):
    from kvpress import GQAIndexerPress
    from kvpress.presses.gqa_indexer import E2EIndexerTrainer

    press = GQAIndexerPress(
        compression_ratio=0.5, scorer="scalar", scalar_mid_dim=16, gate_scale=True,
    )
    press.post_init_from_model(model)
    return E2EIndexerTrainer(press=press, stage="dense", pin_mode="sink", n_sink=4)


def test_step_with_all_ones_weights_equals_the_plain_objective(tiny_model):
    """
    The strongest correctness statement available: all-ones weights must reproduce
    ``e2e_indexer_training_step`` exactly.

    The two paths share nothing but the model -- this one computes per-token CE in chunks and takes a
    weighted mean, the other takes the model's own reduced loss. Agreement pins the next-token shift,
    the chunking, and the ``sum w L / sum w`` normalization simultaneously. It is also the limit a
    cache miss falls back to, so it is the behaviour an incomplete cache degrades into.
    """
    from kvpress.presses.gqa_indexer import (
        e2e_indexer_longce_step,
        e2e_indexer_training_step,
    )

    torch.manual_seed(0)
    trainer = _trainer_and_press(tiny_model)
    ids = torch.randint(0, 3000, (1, 96))

    with torch.no_grad():
        reference = float(e2e_indexer_training_step(tiny_model, trainer, input_ids=ids))
        weighted, stats = e2e_indexer_longce_step(
            tiny_model, trainer, input_ids=ids,
            weights=torch.ones(1, 95), logit_chunk=32,
        )
    assert float(weighted) == pytest.approx(reference, abs=1e-6)
    assert stats["weight_participation"] == pytest.approx(1.0, abs=1e-3)
    # The plain mean is reported alongside so the curve stays comparable to the plain arm's.
    assert stats["sparse_loss"] == pytest.approx(reference, abs=1e-6)


def test_step_produces_router_gradient(tiny_model):
    """The gated pass carries the graph, so the indexer parameters must receive gradient."""
    from kvpress.presses.gqa_indexer import e2e_indexer_longce_step

    torch.manual_seed(0)
    trainer = _trainer_and_press(tiny_model)
    tiny_model.zero_grad(set_to_none=True)
    torch.manual_seed(1)
    loss, _ = e2e_indexer_longce_step(
        tiny_model, trainer, input_ids=torch.randint(0, 3000, (1, 96)),
        weights=torch.rand(1, 95) * 4 + 0.5, logit_chunk=32,
    )
    loss.backward()
    # "indexer" is the attribute the press installs under (--scorer-attr's default), so this is the
    # parameter set the trainer actually optimizes.
    scorers = [
        p for name, p in tiny_model.named_parameters()
        if "indexer" in name and p.requires_grad
    ]
    assert scorers, "the press installed no trainable indexer parameters"
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in scorers)


def test_step_rejects_weights_of_the_wrong_length(tiny_model):
    """
    A length mismatch must raise rather than broadcast.

    This is the failure a cache built at the wrong ``seq_len`` produces, and elementwise
    broadcasting would quietly multiply the wrong positions while every logged number stayed
    plausible.
    """
    from kvpress.presses.gqa_indexer import e2e_indexer_longce_step

    torch.manual_seed(0)
    trainer = _trainer_and_press(tiny_model)
    with pytest.raises(ValueError, match="do not line up"):
        e2e_indexer_longce_step(
            tiny_model, trainer, input_ids=torch.randint(0, 3000, (1, 96)),
            weights=torch.ones(1, 64), logit_chunk=32,
        )


def test_step_runs_one_forward_pass_not_two(tiny_model):
    """
    LongCE must cost ONE forward pass -- the cached weights are why it is cheaper than delta.

    Counted by hooking the base model. Delta needs a second, ungated pass to form its weights; if
    this path ever grew one, the cache's entire cost argument would be void while the loss stayed
    identical.
    """
    from kvpress.presses.gqa_indexer import e2e_indexer_longce_step

    torch.manual_seed(0)
    trainer = _trainer_and_press(tiny_model)
    calls = []
    handle = tiny_model.model.register_forward_hook(lambda *_: calls.append(1))
    try:
        with torch.no_grad():
            e2e_indexer_longce_step(
                tiny_model, trainer, input_ids=torch.randint(0, 3000, (1, 96)),
                weights=torch.ones(1, 95), logit_chunk=32,
            )
    finally:
        handle.remove()
    assert len(calls) == 1


def test_labels_mask_excludes_ignored_positions_end_to_end(tiny_model):
    """
    ``IGNORE_INDEX`` labels must drop out of both the numerator and the denominator.

    Weight-1 positions with ignored labels would otherwise dilute the mean towards zero while
    looking like ordinary tokens.
    """
    from kvpress.presses.gqa_indexer import e2e_indexer_longce_step

    torch.manual_seed(0)
    trainer = _trainer_and_press(tiny_model)
    ids = torch.randint(0, 3000, (1, 96))
    labels = ids.clone()
    labels[:, :48] = -100
    with torch.no_grad():
        _, stats = e2e_indexer_longce_step(
            tiny_model, trainer, input_ids=ids, labels=labels,
            weights=torch.ones(1, 95), logit_chunk=32,
        )
    # 95 shifted positions, of which the first 47 predict an ignored label.
    assert stats["n_weighted"] == 48


# ---------------------------------------------------------------------------------------------
# the draw plan: does the cache cover the documents the run will actually ask for?
# ---------------------------------------------------------------------------------------------


def test_draw_plan_is_deterministic_and_covers_whole_shards():
    """
    ``loader_draw_plan`` must be reproducible: it is what decides which rows get scored.

    Determinism matters across *processes*, not just calls -- the precompute runs in 8 separate
    workers and each rebuilds the plan independently, so a plan that varied per process would leave
    holes. It holds because ``hash`` is only randomized for ``str``/``bytes``, not for tuples of ints.
    """
    from scripts.precompute_longce_weights import loader_draw_plan

    root = _fake_corpus_root()
    if root is None:
        pytest.skip("needs the pretokenized corpus")
    kwargs = dict(
        seq_len=8192, seed=0, world_size=1, num_workers=2, docs_per_worker=64
    )
    first = loader_draw_plan(root, ["2e16", "2e17"], **kwargs)
    second = loader_draw_plan(root, ["2e16", "2e17"], **kwargs)
    assert first == second
    assert first, "the plan must name at least one shard"
    # Each reader walks its shards sequentially, so a small draw concentrates in few shards rather
    # than spreading thinly -- which is the entire reason mirroring beats a per-shard slice.
    assert len(first) <= 4


def test_draw_plan_changes_with_the_stage_length():
    """
    Different stages draw different shards, because the loader seeds with ``seed + seq_len``.

    So every stage the run will reach has to be planned; planning only one would leave the other
    stage missing the cache entirely while reporting a healthy hit rate for the first.
    """
    from scripts.precompute_longce_weights import loader_draw_plan

    root = _fake_corpus_root()
    if root is None:
        pytest.skip("needs the pretokenized corpus")
    common = dict(seed=0, world_size=1, num_workers=2, docs_per_worker=64)
    at_8k = loader_draw_plan(root, ["2e16", "2e17"], seq_len=8192, **common)
    at_16k = loader_draw_plan(root, ["2e16", "2e17"], seq_len=16384, **common)
    assert set(at_8k) != set(at_16k)


def _fake_corpus_root():
    """The real corpus if present, else ``None`` -- the plan mirrors *its* shard names and sizes."""
    from pathlib import Path

    root = Path("/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_tokenized_64k")
    return root if (root / "index.json").is_file() else None


def test_rows_to_score_samples_rather_than_taking_a_prefix():
    """
    The fallback sampler must not return a stored-order prefix.

    The corpus is ordered by document, so a prefix is not a random sample; and the loader shuffles
    rows before reading them, so a prefix is also not what gets drawn.
    """
    from scripts.precompute_longce_weights import rows_to_score

    rows = rows_to_score(1000, max_docs=50, seed=0, shuffle=True)
    assert len(rows) == 50
    assert rows == sorted(rows)
    assert rows != list(range(50))
    assert max(rows) > 100  # reaches beyond the head of the shard


def test_rows_to_score_returns_everything_when_uncapped():
    from scripts.precompute_longce_weights import rows_to_score

    assert rows_to_score(10, max_docs=0, seed=0, shuffle=True) == list(range(10))
    assert rows_to_score(10, max_docs=99, seed=0, shuffle=True) == list(range(10))
