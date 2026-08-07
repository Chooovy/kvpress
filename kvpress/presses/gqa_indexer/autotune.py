# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pick a batch size and tile shape per sequence length, by measuring instead of guessing.

Why this exists
---------------
A length curriculum (8K -> 16K -> 32K) has a utilization problem at the short end. Batch size
is fixed at 1 because 32K needs it to be, so the 8K stage runs a quarter of the memory and a
much smaller GEMM than the card can hold. :func:`batch_for_length` fixes that side
analytically; :func:`autotune` measures the rest.

The token-budget rule
---------------------
Every retained term in the stage-1 footprint is per-token-per-sequence (measured 654 KiB/token
on Qwen3-8B by ``tests/presses/bench_gqa_indexer_capacity.py``), so peak memory tracks ``B * L``
and not ``B`` or ``L`` alone. Holding ``B * L`` constant therefore holds memory constant:

======  ===  ==========  ============
``L``   ``B``  ``B * L``  retained
======  ===  ==========  ============
8K      4    32768       20.4 GiB
16K     2    32768       20.4 GiB
32K     1    32768       20.4 GiB
======  ===  ==========  ============

That is the whole reason to prefer ``B = budget // L`` over a per-length free-for-all: one
memory operating point across the curriculum, so peak memory and tokens/step both stay put
while only ``L`` moves.

Step *time* does not stay put, and it is worth being precise about why. Stage 1 is
``O(B * L^2)``, which under a constant ``B * L`` is ``O(tokens * L)`` -- still linear in ``L``.
So 32K costs about 4x an 8K step even at equal memory and equal tokens. Batching does not
change the FLOPs at all (work per token is ``L`` for any ``B``); what it changes is how well
those FLOPs occupy the device, via larger GEMMs and fewer kernel launches. That is exactly the
short-stage underutilization this module exists to fix, and it is why the win has to be
*measured* per length rather than predicted -- an arithmetic model would show no win.

It also protects the learning-rate schedule. The curriculum deliberately runs under a *single*
WSD schedule, which is only sound if the optimizer sees a stationary problem across the
boundaries. Tokens per step is the quantity a schedule is implicitly tuned against, so a
token-constant batch keeps the boundaries as pure length changes -- which is exactly the
property the boundary logging in ``scripts/train_gqa_indexer.py`` asserts.

Batching is exactly equivalent to averaging
-------------------------------------------
The loss is a plain mean over ``(B, h, Sq)`` rows (see ``fused_indexer_loss``), so a step at
``B=4`` equals the mean of four steps at ``B=1`` -- verified end to end to 1.2e-07 in bf16 and
to 8.9e-16 in fp64. Raising ``B`` at short lengths thus changes *throughput and effective batch*,
never the objective. What it does **not** do is reduce work: stage 1 is ``O(B * L^2)``, so
work per token is ``L`` regardless of ``B``. Expect better utilization, not fewer FLOPs.

What tuning tiles can and cannot buy
------------------------------------
Read this before trusting a tile sweep, because the honest answer is "less than you would
think", and the reason is structural.

**On the Triton path, ``key_tile``/``query_tile`` do nothing.** The kernels take their own
``block_m``/``block_n`` (both 64 by default) and never see the torch tile arguments. Since
``backend='auto'`` selects Triton for every decomposable mask -- which is every mask stage 1
builds -- the tile flags the launcher passes today (``--key-tile 2048 --query-tile 2048``)
reach exactly one component: ``teacher_lse_from_qk``, which runs in torch on both backends
because the kernel consumes ``lse`` as an input. So this module sweeps ``block_m``/``block_n``
*and* the torch tiles, and reports which backend actually ran, rather than tuning knobs that
are inert on the path being used.

**Total tile work is independent of tile size** on the torch path, because its key loop has no
causal skip: every query tile scans the full key axis, so the product ``pairs * tile_area`` is
``L^2`` for any tile. Tile size changes arithmetic intensity and allocator behaviour, not
operation count. Measured spread across a 3x3 sweep at L=2048 was only 1.36x, and the best
setting was *not* the largest -- which is precisely why this is measured per length rather than
set by a rule of thumb.

Choosing by measurement, and what "best" means
----------------------------------------------
Candidates are ranked on **seconds per token**, not seconds per step: a step at ``B=4`` does
four sequences of work, so per-step time would systematically prefer small batches. Ties inside
``tolerance`` go to the lower-memory candidate, since headroom is what absorbs the fragmentation
that a long run accumulates.

OOM is an expected outcome here, not an error -- it is how the top of the range is found. Any
*other* exception is fatal and stops the sweep, following the same rule as the capacity bench:
a search that treats a bug as a resource limit reports a confidently wrong number.

The cache
---------
Results are keyed on everything that changes the answer: GPU name, total VRAM, torch version,
model geometry, stage, backend, dtype and topk. Change the GPU or the model and the key misses,
so a stale entry cannot silently mis-tune a different machine -- which matters most when the
cache lives on a shared filesystem, as it does here.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

GIB = 1024**3

# Measured on Qwen3-8B by tests/presses/bench_gqa_indexer_capacity.py: 654 KiB of retained
# state per token per sequence for stage 1. Used only as the default token budget's basis; the
# sweep measures real memory and does not rely on this number being right.
STAGE1_KIB_PER_TOKEN = 654.0


def is_oom(exc: BaseException) -> bool:
    """
    Whether an exception is an allocation failure rather than a bug.

    A dedicated ``OutOfMemoryError`` counts unconditionally; a bare ``RuntimeError`` only if it
    says so. Only the "out of memory" phrase is accepted: "CUDA error: an illegal memory access"
    also mentions memory but is a bug, and classifying it as OOM would end a sweep with a
    plausible number instead of a stack trace.
    """
    named = getattr(torch, "OutOfMemoryError", None)
    if named is not None and isinstance(exc, named):
        return True
    return "out of memory" in str(exc).lower()


def batch_for_length(seq_len: int, token_budget: int, *, max_batch: int = 0) -> int:
    """
    Largest batch whose token count fits ``token_budget``: ``budget // seq_len``, floored at 1.

    Floored at 1 rather than raising, because a budget below one sequence is a statement about
    the budget, not a reason to refuse to train -- the caller finds out from the measured peak
    instead. ``max_batch`` caps the result when the data path, not memory, is the limit.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if token_budget <= 0:
        raise ValueError(f"token_budget must be positive, got {token_budget}")
    batch = max(1, token_budget // seq_len)
    if max_batch > 0:
        batch = min(batch, max_batch)
    return int(batch)


def default_token_budget(
    total_memory_bytes: float, weight_bytes: float, *, utilization: float = 0.75
) -> int:
    """
    A starting token budget from the memory actually free after weights.

    ``utilization`` deliberately leaves a quarter of the card unused: the analytic footprint
    omits fragmentation and per-tile scratch, and the capacity bench found measurement running
    1.13-1.30x above prediction. Rounded down to a multiple of 1024 so the derived batch sizes
    stay round across the curriculum.
    """
    free = max(0.0, total_memory_bytes - weight_bytes) * utilization
    tokens = int(free / (STAGE1_KIB_PER_TOKEN * 1024))
    return max(1024, tokens - tokens % 1024)


@dataclass(frozen=True)
class Candidate:
    """One (batch, tile, block) configuration to measure."""

    batch_size: int
    key_tile: int
    query_tile: int
    block_m: int = 64
    block_n: int = 64

    def label(self) -> str:
        return (
            f"B={self.batch_size} kt={self.key_tile} qt={self.query_tile} "
            f"bm={self.block_m} bn={self.block_n}"
        )


@dataclass
class Measurement:
    """What one candidate cost. ``ok=False`` means it OOMed, which is a valid result."""

    candidate: Candidate
    seq_len: int
    ok: bool
    seconds: float = float("nan")
    peak_gib: float = float("nan")
    backend_used: str | None = None
    error: str | None = None

    @property
    def seconds_per_token(self) -> float:
        """
        The ranking key: step time normalized by the tokens the step consumed.

        Per-step time would prefer small batches purely because they do less work; per-token
        time is what actually decides how long the run takes.
        """
        if not self.ok or not math.isfinite(self.seconds):
            return float("inf")
        return self.seconds / (self.candidate.batch_size * self.seq_len)


def candidate_grid(
    seq_len: int,
    token_budget: int,
    *,
    max_batch: int = 0,
    tiles: tuple[int, ...] = (1024, 2048, 4096),
    blocks: tuple[int, ...] = (64, 128),
    batches: tuple[int, ...] | None = None,
) -> list[Candidate]:
    """
    Configurations worth measuring at this length.

    Batch defaults to the token-budget value alone -- and optionally that value halved, which is
    the fallback when the budget turns out optimistic. Tiles above ``seq_len`` are dropped
    (a tile wider than the axis is just the axis, so measuring both is measuring the same thing
    twice), and blocks must stay powers of two for the kernels.
    """
    if batches is None:
        target = batch_for_length(seq_len, token_budget, max_batch=max_batch)
        batches = (target,) if target == 1 else (target, target // 2)

    usable_tiles = tuple(sorted({min(t, seq_len) for t in tiles}))
    grid: list[Candidate] = []
    for batch in dict.fromkeys(batches):  # dedupe, preserve order
        for tile in usable_tiles:
            for block in blocks:
                if block & (block - 1):
                    raise ValueError(f"block sizes must be powers of two, got {block}")
                grid.append(
                    Candidate(
                        batch_size=int(batch),
                        key_tile=int(tile),
                        query_tile=int(tile),
                        block_m=int(block),
                        block_n=int(block),
                    )
                )
    return grid


def reset_peak() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def peak_gib() -> float:
    if not torch.cuda.is_available():
        return float("nan")
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / GIB


def measure(
    step_fn,
    trainer,
    candidate: Candidate,
    seq_len: int,
    *,
    warmup: int = 1,
    iters: int = 2,
) -> Measurement:
    """
    Time one candidate over a real forward **and** backward.

    Backward is not optional: it is where the per-layer retained state is finally consumed, so a
    forward-only probe reports a configuration that then OOMs in training. ``warmup`` exists
    because the first call at a new shape pays Triton JIT compilation and allocator growth,
    which would otherwise be charged to whichever candidate happened to run first.

    ``step_fn(batch_size, seq_len) -> None`` performs one full training step. Applying the
    candidate to ``trainer`` happens here so the caller cannot forget to.
    """
    trainer.key_tile = candidate.key_tile
    trainer.query_tile = candidate.query_tile
    trainer.block_m = candidate.block_m
    trainer.block_n = candidate.block_n

    reset_peak()
    try:
        for _ in range(warmup):
            step_fn(candidate.batch_size, seq_len)
        reset_peak()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iters):
            step_fn(candidate.batch_size, seq_len)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) / max(1, iters)
    except Exception as exc:  # noqa: BLE001 -- classified immediately below
        # Catch broadly, then classify: only an allocation failure is a valid measurement.
        # Anything else is a bug in the step and is returned as an error for the caller to
        # raise on. Catching only the OOM types would let a TypeError or a shape mismatch escape
        # as a raw traceback from inside the sweep, which reads as "autotune is broken" rather
        # than "the step you asked me to time is broken".
        reset_peak()
        if is_oom(exc):
            return Measurement(candidate=candidate, seq_len=seq_len, ok=False, error="OOM")
        return Measurement(
            candidate=candidate,
            seq_len=seq_len,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            backend_used=getattr(trainer, "backend_used", None),
        )

    result = Measurement(
        candidate=candidate,
        seq_len=seq_len,
        ok=True,
        seconds=elapsed,
        peak_gib=peak_gib(),
        backend_used=getattr(trainer, "backend_used", None),
    )
    reset_peak()
    return result


def pick_best(results: list[Measurement], *, tolerance: float = 0.03) -> Measurement | None:
    """
    Fastest per token, breaking near-ties toward lower peak memory.

    ``tolerance`` is 3% because that is roughly the run-to-run spread of these measurements;
    treating anything inside it as equal-speed and then preferring headroom avoids chasing
    noise into a configuration that has none. Returns None when every candidate failed.
    """
    ok = [r for r in results if r.ok]
    if not ok:
        return None
    fastest = min(r.seconds_per_token for r in ok)
    contenders = [r for r in ok if r.seconds_per_token <= fastest * (1 + tolerance)]
    return min(contenders, key=lambda r: (r.peak_gib, r.candidate.batch_size))


# ----------------------------------------------------------------------
# The cache
# ----------------------------------------------------------------------
def device_key() -> dict:
    """Identify the hardware and software that make a measurement valid."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu, memory = props.name, int(props.total_memory)
    else:
        gpu, memory = "cpu", 0
    return {"gpu": gpu, "total_memory": memory, "torch": torch.__version__}


def profile_key(
    *,
    model_name: str,
    stage: str,
    backend: str,
    dtype: str,
    layers: int,
    topk: int | None = None,
) -> str:
    """
    A cache key covering everything that changes the answer.

    Model *and* layer count, because a truncated model has a different footprint; stage and
    topk, because stage 2's scratch is dominated by the support tensor; backend and dtype,
    because they change which code path is being tuned at all. Anything not in the key is
    something a stale entry could get wrong silently.
    """
    parts = device_key()
    parts.update(
        model=model_name, stage=stage, backend=backend, dtype=dtype, layers=layers, topk=topk
    )
    return json.dumps(parts, sort_keys=True)


def load_cache(path: str | Path) -> dict:
    """Read the cache, tolerating absence and corruption."""
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        with open(path) as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        # A truncated file (two ranks racing, a killed job) must not stop training: the
        # profile is an optimization, so the correct response is to re-measure.
        logger.warning("ignoring unreadable autotune cache %s: %s", path, exc)
        return {}


def save_cache(path: str | Path, cache: dict) -> None:
    """
    Write the cache atomically.

    Via a temp file in the same directory then ``os.replace``, so a crash or a second rank
    cannot leave a half-written JSON behind -- the failure mode that ``load_cache`` has to
    tolerate above, and which is cheaper to prevent here.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.partial")
    with open(temp, "w") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)
    os.replace(temp, path)


@dataclass
class Profile:
    """The chosen configuration for one sequence length, plus how it was chosen."""

    seq_len: int
    batch_size: int
    key_tile: int
    query_tile: int
    block_m: int
    block_n: int
    seconds: float = float("nan")
    peak_gib: float = float("nan")
    backend_used: str | None = None
    measured: bool = True

    @classmethod
    def from_measurement(cls, m: Measurement) -> Profile:
        return cls(
            seq_len=m.seq_len,
            batch_size=m.candidate.batch_size,
            key_tile=m.candidate.key_tile,
            query_tile=m.candidate.query_tile,
            block_m=m.candidate.block_m,
            block_n=m.candidate.block_n,
            seconds=m.seconds,
            peak_gib=m.peak_gib,
            backend_used=m.backend_used,
        )

    @classmethod
    def fallback(cls, seq_len: int, token_budget: int, *, max_batch: int = 0) -> Profile:
        """
        The un-measured default: token-budget batch, current tile defaults.

        Used when profiling is off or every candidate OOMed. ``measured=False`` so a consumer
        can tell a guess from a measurement instead of trusting both equally.
        """
        return cls(
            seq_len=seq_len,
            batch_size=batch_for_length(seq_len, token_budget, max_batch=max_batch),
            key_tile=min(2048, seq_len),
            query_tile=min(2048, seq_len),
            block_m=64,
            block_n=64,
            measured=False,
        )

    def describe(self) -> str:
        how = "measured" if self.measured else "default (not measured)"
        speed = (
            f", {self.seconds:.2f}s/step, peak {self.peak_gib:.1f} GiB"
            if self.measured and math.isfinite(self.seconds)
            else ""
        )
        backend = f", backend={self.backend_used}" if self.backend_used else ""
        return (
            f"L={self.seq_len}: batch={self.batch_size} key_tile={self.key_tile} "
            f"query_tile={self.query_tile} block=({self.block_m},{self.block_n}) "
            f"[{how}{speed}{backend}]"
        )


def autotune(
    step_fn,
    trainer,
    seq_lens: list[int],
    *,
    token_budget: int,
    max_batch: int = 0,
    tiles: tuple[int, ...] = (1024, 2048, 4096),
    blocks: tuple[int, ...] = (64, 128),
    warmup: int = 1,
    iters: int = 2,
    tolerance: float = 0.03,
    on_result=None,
) -> dict[int, Profile]:
    """
    Profile every length in the curriculum and return the winner for each.

    Raises on a non-OOM failure: that is a bug in the step, and continuing would rank the
    remaining candidates against a configuration that never ran. OOM is not a failure here --
    at the largest batch it is the expected way to find the ceiling.
    """
    profiles: dict[int, Profile] = {}
    for seq_len in seq_lens:
        grid = candidate_grid(
            seq_len, token_budget, max_batch=max_batch, tiles=tiles, blocks=blocks
        )
        logger.info("profiling L=%d over %d candidates", seq_len, len(grid))
        results: list[Measurement] = []
        for candidate in grid:
            result = measure(
                step_fn, trainer, candidate, seq_len, warmup=warmup, iters=iters
            )
            if result.error and result.error != "OOM":
                raise RuntimeError(
                    f"profiling {candidate.label()} at L={seq_len} failed with a non-OOM "
                    f"error, which means the step itself is broken: {result.error}"
                )
            results.append(result)
            if on_result is not None:
                on_result(result)
            status = (
                f"{result.seconds:7.2f}s peak {result.peak_gib:6.2f} GiB"
                if result.ok
                else "    OOM"
            )
            logger.info("  %-44s %s", candidate.label(), status)

        best = pick_best(results, tolerance=tolerance)
        if best is None:
            logger.warning(
                "every candidate OOMed at L=%d; falling back to batch=1 and default tiles. "
                "Lower --token-budget, or this length does not fit at all.",
                seq_len,
            )
            profiles[seq_len] = Profile.fallback(seq_len, seq_len, max_batch=1)
        else:
            profiles[seq_len] = Profile.from_measurement(best)
        logger.info("  -> %s", profiles[seq_len].describe())
    return profiles


def autotune_cached(
    step_fn,
    trainer,
    seq_lens: list[int],
    *,
    cache_path: str | Path,
    key: str,
    token_budget: int,
    force: bool = False,
    **kwargs,
) -> dict[int, Profile]:
    """
    :func:`autotune`, but reuse a cached result when the key and lengths match.

    Only the lengths actually missing are measured, so extending a curriculum re-profiles the
    new stage instead of the whole thing. The cache is written back under ``key``.
    """
    cache = {} if force else load_cache(cache_path)
    entry = cache.get(key, {}) if isinstance(cache.get(key), dict) else {}

    profiles: dict[int, Profile] = {}
    missing: list[int] = []
    for seq_len in seq_lens:
        stored = entry.get(str(seq_len))
        if stored is None:
            missing.append(seq_len)
        else:
            profiles[seq_len] = Profile(**stored)

    if missing:
        logger.info(
            "autotune cache %s: %d/%d lengths cached, measuring %s",
            cache_path,
            len(profiles),
            len(seq_lens),
            missing,
        )
        fresh = autotune(step_fn, trainer, missing, token_budget=token_budget, **kwargs)
        profiles.update(fresh)
        entry.update({str(k): asdict(v) for k, v in fresh.items()})
        cache[key] = entry
        save_cache(cache_path, cache)
    else:
        logger.info("autotune: all %d lengths served from %s", len(seq_lens), cache_path)

    return {seq_len: profiles[seq_len] for seq_len in seq_lens}
