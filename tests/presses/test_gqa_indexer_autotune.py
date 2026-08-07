# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for :mod:`kvpress.presses.gqa_indexer.autotune`.

No GPU required. ``torch.cuda`` is faked where autotune only needs bookkeeping (availability,
device name, peak memory), and the ``step_fn`` under test is a stub -- what is being verified
is the search, ranking and caching logic, not the kernels, which have their own tests.

Two properties are pinned here because getting them wrong is silent rather than loud:

* ranking is per **token**, not per step (per-step would always prefer batch 1);
* a non-OOM exception is fatal, because ranking a broken configuration against working ones
  reports a confidently wrong answer.
"""

from __future__ import annotations

import json
from unittest import mock

import pytest
import torch

from kvpress.presses.gqa_indexer.autotune import (
    Candidate,
    Measurement,
    Profile,
    autotune,
    autotune_cached,
    batch_for_length,
    candidate_grid,
    default_token_budget,
    device_key,
    is_oom,
    load_cache,
    measure,
    pick_best,
    profile_key,
    save_cache,
)


class FakeTrainer:
    """Just the attributes autotune writes."""

    def __init__(self):
        self.key_tile = self.query_tile = 0
        self.block_m = self.block_n = 0
        self.backend_used = "torch"


@pytest.fixture
def fake_cuda():
    """Fake the parts of torch.cuda that autotune uses for bookkeeping only."""
    props = type("P", (), {"name": "FakeGPU", "total_memory": 80 * 1024**3})
    with mock.patch("torch.cuda.is_available", lambda: True), mock.patch(
        "torch.cuda.synchronize", lambda *a, **k: None
    ), mock.patch("torch.cuda.empty_cache", lambda: None), mock.patch(
        "torch.cuda.reset_peak_memory_stats", lambda *a, **k: None
    ), mock.patch(
        "torch.cuda.max_memory_allocated", lambda *a, **k: 10 * 1024**3
    ), mock.patch("torch.cuda.current_device", lambda: 0), mock.patch(
        "torch.cuda.get_device_properties", lambda *a, **k: props()
    ):
        yield


# ----------------------------------------------------------------------
# Batch sizing
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("seq_len", "expected"), [(8192, 4), (16384, 2), (32768, 1), (4096, 8)]
)
def test_batch_scales_inversely_with_length(seq_len, expected):
    """The token-budget rule: B * L is held constant, so B halves when L doubles."""
    assert batch_for_length(seq_len, 32768) == expected


def test_token_count_is_constant_across_the_curriculum():
    """
    The property the whole design rests on.

    Peak memory tracks ``B * L``, so an equal token count per stage means equal peak memory --
    and equal tokens/step, which is what keeps a single LR schedule valid across boundaries.
    """
    budget = 32768
    tokens = {L: batch_for_length(L, budget) * L for L in (8192, 16384, 32768)}
    assert set(tokens.values()) == {budget}


def test_a_constant_token_budget_does_not_make_step_cost_constant():
    """
    Guards a claim that is tempting and wrong.

    Stage 1 is ``O(B * L^2)``, so at constant ``B * L`` the per-step cost is ``O(tokens * L)``
    -- still linear in ``L``. A 32K step is ~4x an 8K one even at equal memory and equal
    tokens. Documentation promising "constant step cost" would set the wrong expectation for
    anyone reading the step timings, so the arithmetic is pinned here.
    """
    budget = 32768
    cost = {L: batch_for_length(L, budget) * L * L for L in (8192, 16384, 32768)}
    assert cost[16384] == 2 * cost[8192]
    assert cost[32768] == 4 * cost[8192]


def test_work_per_token_is_independent_of_batch_size():
    """
    Batching buys utilization, not FLOPs.

    ``O(B * L^2) / (B * L) = L`` for any ``B``, so anyone expecting a bigger batch to reduce
    total work will be disappointed -- and might misread a flat throughput curve as a broken
    autotuner rather than as the expected result.
    """
    L = 8192
    per_token = {B: (B * L * L) / (B * L) for B in (1, 2, 4, 8)}
    assert set(per_token.values()) == {L}


def test_batch_never_drops_below_one():
    """A budget smaller than one sequence is a statement about the budget, not a reason to
    refuse to train: the caller learns the truth from the measured peak instead."""
    assert batch_for_length(65536, 1024) == 1


def test_max_batch_caps_the_result():
    """For when the data path, not memory, is the limit."""
    assert batch_for_length(1024, 32768, max_batch=4) == 4


@pytest.mark.parametrize(("seq_len", "budget"), [(0, 1024), (-1, 1024), (1024, 0)])
def test_batch_rejects_nonsense(seq_len, budget):
    with pytest.raises(ValueError):
        batch_for_length(seq_len, budget)


def test_default_budget_leaves_headroom():
    """
    A quarter of the card is deliberately unspent.

    The analytic footprint omits fragmentation and per-tile scratch, and the capacity bench
    measured 1.13-1.30x above prediction, so budgeting the full card would OOM mid-run.
    """
    total, weights = 80 * 1024**3, 16 * 1024**3
    budget = default_token_budget(total, weights)
    naive = default_token_budget(total, weights, utilization=1.0)
    assert budget < naive
    assert budget % 1024 == 0


def test_default_budget_survives_weights_larger_than_the_card():
    """Clamped, not negative: a nonsensical input must not produce a nonsensical batch."""
    assert default_token_budget(8 * 1024**3, 80 * 1024**3) >= 1024


# ----------------------------------------------------------------------
# The candidate grid
# ----------------------------------------------------------------------
def test_grid_drops_tiles_wider_than_the_sequence():
    """A tile wider than the axis *is* the axis, so measuring both measures one thing twice."""
    grid = candidate_grid(1024, 4096, tiles=(1024, 2048, 4096), blocks=(64,))
    assert {c.key_tile for c in grid} == {1024}


def test_grid_offers_a_half_batch_fallback():
    """The token budget can be optimistic, so the next candidate down is always present."""
    grid = candidate_grid(8192, 32768, tiles=(2048,), blocks=(64,))
    assert {c.batch_size for c in grid} == {4, 2}


def test_grid_at_batch_one_does_not_offer_a_half():
    """``batch // 2 == 0`` would be an invalid configuration, so batch 1 stands alone."""
    grid = candidate_grid(32768, 32768, tiles=(2048,), blocks=(64,))
    assert {c.batch_size for c in grid} == {1}


def test_grid_rejects_non_power_of_two_blocks():
    """The Triton kernels index with power-of-two blocks; catch it here, not in the kernel."""
    with pytest.raises(ValueError, match="powers of two"):
        candidate_grid(8192, 32768, blocks=(100,))


# ----------------------------------------------------------------------
# Ranking
# ----------------------------------------------------------------------
def test_ranking_is_per_token_not_per_step():
    """
    The bug this prevents: a step at B=4 takes longer than a step at B=1 while doing four
    times the work, so per-step timing would always pick the smallest batch and silently
    undo the entire point of autotuning.
    """
    slow_big = Measurement(Candidate(4, 2048, 2048), 8192, True, seconds=2.0, peak_gib=40.0)
    fast_small = Measurement(Candidate(1, 2048, 2048), 8192, True, seconds=1.0, peak_gib=10.0)
    assert slow_big.seconds_per_token < fast_small.seconds_per_token
    assert pick_best([slow_big, fast_small]).candidate.batch_size == 4


def test_near_ties_prefer_lower_memory():
    """Within tolerance the two are the same speed, and headroom absorbs fragmentation."""
    hungry = Measurement(Candidate(4, 1024, 1024), 8192, True, seconds=1.00, peak_gib=40.0)
    lean = Measurement(Candidate(4, 2048, 2048), 8192, True, seconds=1.02, peak_gib=30.0)
    assert pick_best([hungry, lean], tolerance=0.03) is lean


def test_a_real_speed_win_beats_lower_memory():
    """Tolerance must not be so eager that it gives away actual throughput."""
    hungry = Measurement(Candidate(4, 1024, 1024), 8192, True, seconds=1.00, peak_gib=40.0)
    lean = Measurement(Candidate(4, 2048, 2048), 8192, True, seconds=1.50, peak_gib=30.0)
    assert pick_best([hungry, lean], tolerance=0.03) is hungry


def test_failed_candidates_are_never_chosen():
    ok = Measurement(Candidate(1, 2048, 2048), 8192, True, seconds=5.0, peak_gib=10.0)
    dead = Measurement(Candidate(8, 2048, 2048), 8192, False, error="OOM")
    assert pick_best([ok, dead]) is ok
    assert dead.seconds_per_token == float("inf")


def test_pick_best_returns_none_when_everything_failed():
    """So the caller can fall back deliberately instead of unpacking a None-shaped surprise."""
    assert pick_best([Measurement(Candidate(1, 1, 1), 8, False, error="OOM")]) is None


# ----------------------------------------------------------------------
# OOM classification
# ----------------------------------------------------------------------
def test_oom_is_recognized_from_the_message():
    assert is_oom(RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB"))


def test_an_illegal_access_is_not_an_oom():
    """
    The permissive direction is the dangerous one: misreading a bug as a capacity limit ends
    the search with a plausible number instead of a stack trace.
    """
    assert not is_oom(RuntimeError("CUDA error: an illegal memory access was encountered"))


def test_measure_records_oom_without_raising(fake_cuda):
    """OOM at the top of the range is how the ceiling is found, so it must be data."""

    def step(batch_size, seq_len):
        raise RuntimeError("CUDA out of memory")

    result = measure(step, FakeTrainer(), Candidate(8, 2048, 2048), 8192, warmup=0, iters=1)
    assert not result.ok and result.error == "OOM"


def test_measure_reports_a_non_oom_failure_as_an_error(fake_cuda):
    def step(batch_size, seq_len):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    result = measure(step, FakeTrainer(), Candidate(1, 2048, 2048), 8192, warmup=0, iters=1)
    assert not result.ok
    assert result.error is not None and result.error != "OOM"


def test_autotune_raises_on_a_non_oom_failure(fake_cuda):
    """A broken step must stop the sweep, not be ranked against working candidates."""

    def step(batch_size, seq_len):
        raise ValueError("the step itself is broken")

    with pytest.raises(RuntimeError, match="non-OOM"):
        autotune(
            step, FakeTrainer(), [8192], token_budget=8192, tiles=(2048,), blocks=(64,),
            warmup=0, iters=1,
        )


def test_measure_applies_the_candidate_to_the_trainer(fake_cuda):
    """Applied inside measure so a caller cannot forget and time the previous shape."""
    trainer = FakeTrainer()
    seen = {}

    def step(batch_size, seq_len):
        seen.update(
            key_tile=trainer.key_tile, query_tile=trainer.query_tile,
            block_m=trainer.block_m, block_n=trainer.block_n, batch=batch_size,
        )

    measure(step, trainer, Candidate(3, 1024, 512, 128, 256), 8192, warmup=0, iters=1)
    assert seen == dict(key_tile=1024, query_tile=512, block_m=128, block_n=256, batch=3)


def test_autotune_falls_back_when_every_candidate_ooms(fake_cuda):
    """Report a usable default and say so, rather than returning nothing."""

    def step(batch_size, seq_len):
        raise torch.OutOfMemoryError("CUDA out of memory")

    profiles = autotune(
        step, FakeTrainer(), [8192], token_budget=32768, tiles=(2048,), blocks=(64,),
        warmup=0, iters=1,
    )
    assert profiles[8192].batch_size == 1
    assert profiles[8192].measured is False


def test_autotune_skips_only_the_failing_candidate(fake_cuda):
    """One OOM must not condemn the length -- the next batch down is why the grid has one."""

    def step(batch_size, seq_len):
        if batch_size == 4:
            raise torch.OutOfMemoryError("CUDA out of memory")

    profiles = autotune(
        step, FakeTrainer(), [8192], token_budget=32768, tiles=(2048,), blocks=(64,),
        warmup=0, iters=1,
    )
    assert profiles[8192].batch_size == 2
    assert profiles[8192].measured is True


# ----------------------------------------------------------------------
# The cache
# ----------------------------------------------------------------------
def test_cache_round_trip(tmp_path, fake_cuda):
    path = tmp_path / "nested" / "autotune.json"  # parent must be created
    save_cache(path, {"k": {"8192": {"seq_len": 8192}}})
    assert load_cache(path) == {"k": {"8192": {"seq_len": 8192}}}


def test_missing_cache_is_empty_not_an_error(tmp_path):
    assert load_cache(tmp_path / "nope.json") == {}


def test_corrupt_cache_is_ignored(tmp_path):
    """
    Two ranks racing or a killed job can truncate the file. A profile is an optimization, so
    the right response is to re-measure -- never to stop training.
    """
    path = tmp_path / "autotune.json"
    path.write_text("{not json")
    assert load_cache(path) == {}


def test_save_leaves_no_partial_file_behind(tmp_path):
    path = tmp_path / "autotune.json"
    save_cache(path, {"a": 1})
    assert [p.name for p in tmp_path.iterdir()] == ["autotune.json"]


def test_key_covers_gpu_model_stage_backend_and_dtype(fake_cuda):
    """Anything absent from the key is something a stale entry could get wrong silently."""
    base = dict(
        model_name="Qwen3-8B", stage="dense", backend="auto", dtype="bfloat16", layers=36
    )
    key = profile_key(**base)
    for field, value in [
        ("model_name", "other"), ("stage", "sparse"), ("backend", "torch"),
        ("dtype", "float16"), ("layers", 12), ("topk", 512),
    ]:
        assert profile_key(**{**base, field: value}) != key, f"{field} must affect the key"


def test_key_changes_with_the_gpu(fake_cuda):
    """The cache lives on a shared filesystem, so a different card must miss."""
    args = dict(model_name="m", stage="dense", backend="auto", dtype="bfloat16", layers=4)
    key_a = profile_key(**args)
    other = type("P", (), {"name": "OtherGPU", "total_memory": 40 * 1024**3})
    with mock.patch("torch.cuda.get_device_properties", lambda *a, **k: other()):
        assert profile_key(**args) != key_a


def test_cached_run_does_not_measure_again(tmp_path, fake_cuda):
    calls = []

    def step(batch_size, seq_len):
        calls.append((batch_size, seq_len))

    kwargs = dict(
        cache_path=tmp_path / "c.json", key="k", token_budget=32768,
        tiles=(2048,), blocks=(64,), warmup=0, iters=1,
    )
    first = autotune_cached(step, FakeTrainer(), [8192, 16384], **kwargs)
    assert calls
    n = len(calls)
    second = autotune_cached(step, FakeTrainer(), [8192, 16384], **kwargs)
    assert len(calls) == n, "a cache hit must not re-measure"
    assert {L: p.batch_size for L, p in first.items()} == {
        L: p.batch_size for L, p in second.items()
    }


def test_extending_a_curriculum_measures_only_the_new_length(tmp_path, fake_cuda):
    """Adding a stage should not re-profile the ones already known."""
    calls = []

    def step(batch_size, seq_len):
        calls.append(seq_len)

    kwargs = dict(
        cache_path=tmp_path / "c.json", key="k", token_budget=32768,
        tiles=(2048,), blocks=(64,), warmup=0, iters=1,
    )
    autotune_cached(step, FakeTrainer(), [8192], **kwargs)
    calls.clear()
    profiles = autotune_cached(step, FakeTrainer(), [8192, 16384], **kwargs)
    assert set(calls) == {16384}
    assert set(profiles) == {8192, 16384}


def test_force_re_measures(tmp_path, fake_cuda):
    calls = []

    def step(batch_size, seq_len):
        calls.append(seq_len)

    kwargs = dict(
        cache_path=tmp_path / "c.json", key="k", token_budget=32768,
        tiles=(2048,), blocks=(64,), warmup=0, iters=1,
    )
    autotune_cached(step, FakeTrainer(), [8192], **kwargs)
    calls.clear()
    autotune_cached(step, FakeTrainer(), [8192], force=True, **kwargs)
    assert calls, "--autotune-force must ignore the cache"


def test_autotune_cached_returns_the_requested_order(tmp_path, fake_cuda):
    """The training loop indexes by seq_len, so a missing key would be a KeyError mid-run."""

    def step(batch_size, seq_len):
        pass

    profiles = autotune_cached(
        step, FakeTrainer(), [32768, 8192, 16384],
        cache_path=tmp_path / "c.json", key="k", token_budget=32768,
        tiles=(2048,), blocks=(64,), warmup=0, iters=1,
    )
    assert list(profiles) == [32768, 8192, 16384]


def test_profile_survives_a_json_round_trip(tmp_path, fake_cuda):
    """asdict/Profile(**...) is how the cache and the rank broadcast both move a profile."""
    from dataclasses import asdict

    original = Profile(
        seq_len=8192, batch_size=4, key_tile=2048, query_tile=2048,
        block_m=64, block_n=128, seconds=1.5, peak_gib=40.0, backend_used="triton",
    )
    restored = Profile(**json.loads(json.dumps(asdict(original))))
    assert restored == original


def test_fallback_is_labelled_unmeasured():
    """So a consumer can tell a guess from a measurement instead of trusting both equally."""
    profile = Profile.fallback(8192, 32768)
    assert profile.measured is False and profile.batch_size == 4
    assert "not measured" in profile.describe()


def test_device_key_works_without_cuda():
    """Importable and callable on a CPU box, so the cache logic stays testable anywhere."""
    with mock.patch("torch.cuda.is_available", lambda: False):
        assert device_key()["gpu"] == "cpu"
