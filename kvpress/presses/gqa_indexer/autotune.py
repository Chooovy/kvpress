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

A resource limit is an expected outcome here, not an error -- it is how the top of the range is
found. Two shapes occur and they call for opposite fixes, so they are reported separately:

* **OOM** -- not enough VRAM. Answered by a smaller batch, which the grid already offers.
* **SHMEM** -- Triton's ``OutOfResources``, raised when a block size needs more shared memory
  than the SM has. ``block=(128,128)`` wants ~256 KiB against the ~227 KiB an H20-class SM
  provides. This depends only on ``(BLOCK_M, BLOCK_N, head_dim)``, never on tile or batch, so
  the sweep records the pair once and skips it everywhere else.

Note that ``OutOfResources`` is neither a ``RuntimeError`` subclass nor does its message contain
"out of memory", so a classifier keyed on either treats it as a bug and aborts the whole run.
That happened. Both predicates now match on the message wording, which is what the Triton
``OutOfResources``, older bare-``RuntimeError`` and wrapped-``CompilationError`` forms share.

Any *other* exception is fatal and stops the sweep, tracked by an explicit ``fatal`` flag rather
than by comparing the error string: ``error != "OOM"`` is exactly what let the shared-memory case
through, and that trap returns every time a new limit gets its own wording.

Cost
----
Profiling runs real forward+backward steps -- a full grid is ~180 across an 8K/16K/32K
curriculum, which at 32K can exceed an hour. ``time_budget_s`` bounds it (15 minutes by default
from the training script) and the remaining lengths take the token-budget default, labelled
``measured=False`` and logged. Unmeasured fallbacks are deliberately **not** cached, so a later
run retries them instead of trusting a guess forever.

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


def is_resource_limit(exc: BaseException) -> bool:
    """
    Whether an exception is a resource limit rather than a bug.

    Three shapes count, and the third is the one that is easy to miss:

    1. ``torch.OutOfMemoryError`` -- unconditional.
    2. A message containing "out of memory", which is how pre-2.5 torch and some allocator
       paths report it.
    3. Triton's ``OutOfResources``, raised when a block size needs more **shared memory** than
       the SM has. This is neither a subclass of ``RuntimeError`` nor does it say "out of
       memory" -- it says "out of resource: shared memory, Required: N, Hardware limit: M".
       It is exactly the kind of thing the sweep exists to discover, since whether
       ``block=(128,128)`` fits is a property of the GPU: ~256 KB required against the 227 KB
       an H20/A100 SM provides. Missing it made the profiler abort the whole run with "the
       step itself is broken" on a candidate that simply does not fit this card.

    Everything else stays a bug. "CUDA error: an illegal memory access" also mentions memory,
    and classifying that as a resource limit would end a sweep with a plausible number instead
    of a stack trace -- so the match stays on specific phrases, not on the word "memory".
    """
    named = getattr(torch, "OutOfMemoryError", None)
    if named is not None and isinstance(exc, named):
        return True
    # Matched by name rather than by importing triton: this module must import on a box with no
    # triton installed, and the exception type is only reachable through it.
    if type(exc).__name__ == "OutOfResources":
        return True
    message = str(exc).lower()
    return "out of memory" in message or "out of resource" in message


def is_shared_memory_limit(exc: BaseException) -> bool:
    """
    Whether the limit hit was **shared memory** rather than VRAM.

    Worth separating because the fixes are opposite: an OOM is answered by a smaller batch
    (which the candidate grid already offers), while a shared-memory limit is a property of the
    block size and the GPU, so it will fail at every batch and every tile. Matched on the
    message rather than the exception type, since the same condition arrives as Triton's
    ``OutOfResources``, as a bare ``RuntimeError`` on older Triton, and wrapped in a
    ``CompilationError`` -- the wording is what they share.
    """
    return "out of resource" in str(exc).lower()


# Kept as an alias: `is_oom` reads naturally at the call site and the name is referenced from
# the capacity bench's docstring, but "resource limit" is what the predicate actually means now
# that shared-memory exhaustion counts too.
is_oom = is_resource_limit


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
    # Whether the failure was a resource limit (skip this candidate) or a bug (stop the sweep).
    # A flag rather than a string comparison on `error`: matching `error != "OOM"` is what let a
    # Triton shared-memory limit abort a whole profiling run as "the step is broken", and the
    # same trap reappears every time a new limit shape gets its own message.
    fatal: bool = False

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
    block_pairs: tuple[tuple[int, int], ...] | None = None,
    batches: tuple[int, ...] | None = None,
) -> list[Candidate]:
    """
    Configurations worth measuring at this length.

    Batch defaults to the token-budget value alone -- and optionally that value halved, which is
    the fallback when the budget turns out optimistic. Tiles above ``seq_len`` are dropped
    (a tile wider than the axis is just the axis, so measuring both is measuring the same thing
    twice), and blocks must stay powers of two for the kernels.

    ``blocks`` is expanded to the *asymmetric* pairs, not just the diagonal. The kernel's shared
    memory is roughly ``BLOCK_M * BLOCK_N + BLOCK_M * D``, which is not symmetric in the two, so
    ``(128, 64)`` can fit on a card where ``(128, 128)`` does not -- measured on an H20-class SM
    (227 KiB): ``(64,64)`` ~115, ``(64,128)`` ~154, ``(128,64)`` ~205, ``(128,128)`` ~256 KiB.
    Sweeping only the diagonal would make the one useful large block unreachable and leave the
    log reading as if 128 were simply impossible here. Pass ``block_pairs`` to override.
    """
    if batches is None:
        target = batch_for_length(seq_len, token_budget, max_batch=max_batch)
        batches = (target,) if target == 1 else (target, target // 2)

    for block in blocks:
        if block & (block - 1):
            raise ValueError(f"block sizes must be powers of two, got {block}")
    if block_pairs is None:
        block_pairs = tuple((m, n) for m in blocks for n in blocks)
    for block_m, block_n in block_pairs:
        if block_m & (block_m - 1) or block_n & (block_n - 1):
            raise ValueError(f"block sizes must be powers of two, got ({block_m}, {block_n})")

    usable_tiles = tuple(sorted({min(t, seq_len) for t in tiles}))
    grid: list[Candidate] = []
    for batch in dict.fromkeys(batches):  # dedupe, preserve order
        for tile in usable_tiles:
            for block_m, block_n in block_pairs:
                grid.append(
                    Candidate(
                        batch_size=int(batch),
                        key_tile=int(tile),
                        query_tile=int(tile),
                        block_m=int(block_m),
                        block_n=int(block_n),
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
        # Catch broadly, then classify: only a resource limit is a valid measurement. Anything
        # else is a bug in the step and is returned as an error for the caller to raise on.
        # Catching only the OOM types would let a TypeError or a shape mismatch escape as a raw
        # traceback from inside the sweep, which reads as "autotune is broken" rather than "the
        # step you asked me to time is broken".
        reset_peak()
        if is_resource_limit(exc):
            # Distinguish the two, because they call for opposite responses: too little VRAM
            # means lower the batch (the grid already offers a half), while too little shared
            # memory means this block size cannot run on this GPU at any batch size.
            #
            # Keyed on the message, not the class name: the same limit reaches us as Triton's
            # OutOfResources, as a RuntimeError from an older Triton, and wrapped inside a
            # CompilationError. All three say "out of resource", and only the message is common
            # to them -- classifying by type would silently mislabel two of the three as OOM
            # and send the reader to the wrong knob.
            if is_shared_memory_limit(exc):
                return Measurement(
                    candidate=candidate, seq_len=seq_len, ok=False, error=f"SHMEM ({exc})"
                )
            return Measurement(candidate=candidate, seq_len=seq_len, ok=False, error="OOM")
        return Measurement(
            candidate=candidate,
            seq_len=seq_len,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            fatal=True,
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
    time_budget_s: float = 0.0,
    on_result=None,
) -> dict[int, Profile]:
    """
    Profile every length in the curriculum and return the winner for each.

    Raises only on a genuine bug in the step: continuing would rank the remaining candidates
    against a configuration that never ran. Resource limits are *not* failures here -- at the
    largest batch an OOM is the expected way to find the ceiling, and a Triton shared-memory
    limit is how a block size that this GPU cannot run gets discovered.

    ``time_budget_s`` bounds the whole sweep. Profiling runs real forward+backward steps, so a
    full grid is 180 of them across an 8K/16K/32K curriculum -- at 32K that is potentially most
    of an hour before training starts. When the budget is exhausted the remaining lengths take
    the token-budget default and say so, which is the honest failure: a silent truncation would
    look identical to a completed sweep. 0 disables the bound.
    """
    profiles: dict[int, Profile] = {}
    # Block pairs this GPU cannot run at all. Shared-memory capacity depends only on
    # (BLOCK_M, BLOCK_N, head_dim) -- never on key_tile/query_tile or batch -- so once a pair
    # exceeds the SM limit it will exceed it for every other candidate too. Remembering that
    # skips the redundant probes (3 tiles x 2 batches = up to 6 per pair per length) instead of
    # re-learning the same fact, which matters when one probe at 32K is minutes long.
    dead_blocks: set[tuple[int, int]] = set()
    deadline = time.perf_counter() + time_budget_s if time_budget_s > 0 else None
    for seq_len in seq_lens:
        if deadline is not None and time.perf_counter() >= deadline:
            logger.warning(
                "autotune time budget (%.0f s) exhausted before L=%d; using the token-budget "
                "default for it. Raise --autotune-time-budget, or narrow the curriculum.",
                time_budget_s,
                seq_len,
            )
            profiles[seq_len] = Profile.fallback(seq_len, token_budget, max_batch=max_batch)
            continue
        grid = candidate_grid(
            seq_len, token_budget, max_batch=max_batch, tiles=tiles, blocks=blocks
        )
        logger.info("profiling L=%d over %d candidates", seq_len, len(grid))
        results: list[Measurement] = []
        for candidate in grid:
            if deadline is not None and time.perf_counter() >= deadline and results:
                # Stop mid-grid only if something already succeeded, so a partial sweep still
                # returns a measured answer rather than an unmeasured guess.
                logger.warning(
                    "  time budget exhausted at L=%d after %d/%d candidates; ranking those",
                    seq_len,
                    len(results),
                    len(grid),
                )
                break
            block = (candidate.block_m, candidate.block_n)
            if block in dead_blocks:
                logger.info(
                    "  %-44s     skipped: block %s exceeded shared memory earlier",
                    candidate.label(),
                    block,
                )
                continue
            result = measure(
                step_fn, trainer, candidate, seq_len, warmup=warmup, iters=iters
            )
            if result.fatal:
                raise RuntimeError(
                    f"profiling {candidate.label()} at L={seq_len} failed with an error that is "
                    f"not a resource limit, which means the step itself is broken: {result.error}"
                )
            if result.error and result.error.startswith("SHMEM"):
                dead_blocks.add(block)
            results.append(result)
            if on_result is not None:
                on_result(result)
            status = (
                f"{result.seconds:7.2f}s peak {result.peak_gib:6.2f} GiB"
                if result.ok
                # Report which limit was hit, not a generic "OOM": too little VRAM is fixed by
                # a smaller batch, too little shared memory by a smaller block, and a log that
                # conflates them sends you to the wrong knob.
                else f"    skipped: {result.error}"
            )
            logger.info("  %-44s %s", candidate.label(), status)

        best = pick_best(results, tolerance=tolerance)
        if best is None:
            logger.warning(
                "every candidate hit a resource limit at L=%d; falling back to batch=1 and "
                "default tiles. Reasons: %s. Lower --token-budget if these are OOM; if they are "
                "SHMEM, this GPU cannot run the swept block sizes and 64 is the safe one.",
                seq_len,
                "; ".join(sorted({str(r.error) for r in results})),
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
        # Cache only what was actually measured. An unmeasured fallback -- from a time budget
        # running out, or from every candidate hitting a resource limit -- is a guess, and
        # storing it would make the next run skip the length forever on the strength of a
        # measurement that never happened. Leaving it out means the next run retries.
        measured = {str(k): asdict(v) for k, v in fresh.items() if v.measured}
        skipped = [k for k, v in fresh.items() if not v.measured]
        if skipped:
            logger.info(
                "not caching unmeasured fallbacks for %s; a later run will retry them", skipped
            )
        if measured:
            entry.update(measured)
            cache[key] = entry
            save_cache(cache_path, cache)
    else:
        logger.info("autotune: all %d lengths served from %s", len(seq_lens), cache_path)

    return {seq_len: profiles[seq_len] for seq_len in seq_lens}
