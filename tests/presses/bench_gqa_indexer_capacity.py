# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
How long a sequence can the indexer actually train on?

Answers that empirically for a Qwen3-8B-shaped model, per stage, by binary-searching the
longest ``L`` that completes a full forward **and** ``backward()``. Backward is the point:
the forward pass alone understates peak memory badly, because both fused losses stash
per-layer state that only frees once gradients have flowed.

Run it as a module, not under pytest -- one probe can consume an entire GPU. ``-m`` puts the
repo root on ``sys.path``, which a bare path invocation would not:

    python -m tests.presses.bench_gqa_indexer_capacity --stage dense
    python -m tests.presses.bench_gqa_indexer_capacity --stage sparse --topk 512
    python -m tests.presses.bench_gqa_indexer_capacity --layers 4      # quick smoke

The file is named ``bench_*`` so pytest's default ``test_*.py`` collection skips it; the pure
arithmetic in :func:`predict_bytes_per_token` is unit-tested from
``test_gqa_indexer_fused_trainer.py`` instead.

By default it builds an **untrained** Qwen3-8B from config rather than downloading weights:
capacity is a function of geometry, not of parameter values, and this keeps the probe runnable
anywhere. ``--pretrained`` loads the real checkpoint if you want to confirm.

The predicted ceiling
---------------------
:func:`predict_bytes_per_token` models the retained state analytically, and the script prints
prediction next to measurement. That comparison is the actual product here -- a benchmark that
cannot predict its own answer cannot distinguish a genuine ceiling from a leak.

Measured on an H20 (95 GiB, 15.6 GiB of weights, B=1, bf16, 36 layers):

===============================  ==========  ============  ===========
run                              measured L  fitted slope  analytic
===============================  ==========  ============  ===========
stage 1 dense                    32K-64K     1406 KiB/tok  1086
stage 2 topk=2048                9216        7070 KiB/tok  6270
===============================  ==========  ============  ===========

The slope is the part worth trusting: 1.13-1.30x of analytic, the excess being allocator
fragmentation plus small terms deliberately omitted. Treat the predicted ``L`` as an upper
bound and expect roughly 0.7-0.8x of it in practice.

Timing, same machine. Stage 2 is *faster* at equal length, and scales better:

=======  ===========  =========  =============
``L``    stage 1      stage 2    stage 2 speed
=======  ===========  =========  =============
2048     16.1 s       18.8 s     0.9x
4096     37.8 s       36.4 s     1.0x
8192     105.6 s      75.7 s     **1.4x**
16384    384.0 s      --         --
32768    1506.0 s     --         --
=======  ===========  =========  =============

Stage 1's time grows 3.9x per doubling (clean ``O(L^2)``); stage 2's grows ~2.0x
(``O(L * topk)``, linear). Stage 2 stopping at a shorter ``L`` is a *memory* limit, not a speed
one -- see the support term below.

Three things dominate, and none is the tiled algorithm:

**The fp32 teacher is retained for every layer.** ``_FusedIndexerCE.forward`` stores
``ctx.teacher_alpha``, which is the closure from ``make_recompute_teacher`` holding fp32 copies
of the teacher's Q and K. One autograd node per layer means all 36 layers' teachers stay
resident until ``backward()`` -- 720 KiB/token of the 1086 KiB/token total, so **3.0x** the
footprint of freeing them (366 KiB/token). It also contradicts the ``fused_trainer`` docstring's
claim that "one layer's teacher tensors are alive at a time": true of the forward pass, false of
the autograd graph.

On an 80 GiB card that is the difference between L≈62K and L≈183K for stage 1.

**Stage 2's support tensor dominates, and is quadratic under ``keep_ratio``.** ``support`` is
``(B, h, Sq, topk)`` int64 plus a bool ``valid``, saved for backward. With ``topk = r * L`` that
is ``O(L^2)``: at ``L=32K, r=0.25`` it is 696 GB across 36 layers. A *fixed* ``topk`` keeps it
linear, so the sparse probe requires one and refuses a ratio.

Even fixed, it is the single largest term -- 83% of retained bytes at ``topk=2048``, which is
what capped the measured run at ``L=9216``:

========  =====================  ============  ==============
``topk``  support+valid/token    total/token   predicted max L
========  =====================  ============  ==============
256       648 KiB                1590 KiB      ~52K
512       1296 KiB               2238 KiB      ~37K
1024      2592 KiB               3534 KiB      ~24K
2048      5184 KiB               6126 KiB      ~13K
========  =====================  ============  ==============

``int32`` indices would halve it -- they only need to address ``Sk < 2^31`` -- and packing
``valid`` into ``support`` as a sentinel (which ``-1`` already is) would remove another 11%.
Note ``topk=2048`` at ``L=9216`` means the support covers 22% of the sequence, which is barely a
sparse regime; ``topk=512`` is both the cheaper and the more representative setting.

**Stage 2's per-tile gather is a multi-GB fixed cost.**
:func:`~kvpress.presses.gqa_indexer.sparse_support.gather_support_keys` materializes
``(B, h, query_tile, topk_tile, D)`` and the teacher gathers ``(B, H, ...)`` on top -- so the
scratch is ``O(query_tile * topk_tile * D)``, and the ``D`` factor makes it 5.0 GiB at
``512 x 512 x 128``. Stage 1 never gathers (its keys are contiguous) and its tile is 8 MB, which
is why only stage 2 carries an intercept worth modelling. ``--topk-tile 128`` recovers ~4 GiB.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import traceback

import torch

from kvpress import GQAIndexerPress
from kvpress.presses.gqa_indexer import (
    FusedIndexerTrainer,
    freeze_all_but_indexer,
    fused_indexer_training_step,
)

# Qwen3-8B, from its published config.
QWEN3_8B = dict(
    hidden_size=4096,
    intermediate_size=12288,
    num_hidden_layers=36,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    vocab_size=151936,
    max_position_embeddings=40960,
)

GIB = 1024**3

# torch.OutOfMemoryError only exists from torch 2.5; before that OOM arrives as a plain
# RuntimeError. Catching a tuple covers both without a version check, and the message is
# still inspected below because some allocator paths raise the base class either way.
OOM_ERRORS: tuple[type[BaseException], ...] = tuple(
    exc for exc in (getattr(torch, "OutOfMemoryError", None), RuntimeError) if exc is not None
)


# ----------------------------------------------------------------------
# The analytic model
# ----------------------------------------------------------------------
def predict_bytes_per_token(
    *,
    layers: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    indexer_dim: int,
    stage: str,
    topk: int = 0,
    retain_teacher: bool = True,
) -> dict[str, float]:
    """
    Bytes of *retained* state per token, itemized.

    Only tensors that survive until ``backward()`` are counted. Per-tile scratch is
    independent of ``L`` and so belongs to the intercept, not the slope -- see
    :func:`predict_tile_scratch`, which is a multi-GiB term for stage 2.

    ``retain_teacher=False`` reports the intended footprint -- what it would be if the teacher
    closure were dropped after each layer's loss -- so the caller can price the leak.
    """
    fp32, bf16 = 4, 2
    per_layer: dict[str, float] = {
        # ctx.save_for_backward
        "q_idx": n_kv_heads * indexer_dim * bf16,
        "dq_unit": n_kv_heads * indexer_dim * fp32,
        "lse_student": n_kv_heads * fp32,
        "teacher_lse": n_heads * fp32,
    }
    if retain_teacher:
        # ctx.teacher_alpha closes over fp32 copies of the teacher's Q and K.
        per_layer["teacher_q_fp32"] = n_heads * head_dim * fp32
        per_layer["teacher_k_fp32"] = n_kv_heads * head_dim * fp32
    if stage == "sparse":
        # support (int64) + valid (bool), both saved_for_backward.
        per_layer["support"] = n_kv_heads * topk * 8
        per_layer["valid"] = n_kv_heads * topk * 1

    itemized = {name: value * layers for name, value in per_layer.items()}
    # The model's own KV cache; use_cache=True is required to read the teacher's keys.
    itemized["kv_cache"] = layers * 2 * n_kv_heads * head_dim * bf16
    itemized["total"] = sum(itemized.values())
    return itemized


def predict_tile_scratch(
    *, n_heads: int, n_kv_heads: int, head_dim: int, stage: str, query_tile: int, topk_tile: int
) -> float:
    """
    Bytes of per-tile scratch: the length-independent term, i.e. the intercept.

    Zero for stage 1, whose tile intermediates are ``(B, h, query_tile, key_tile)`` -- 8 MB, not
    worth modelling. Stage 2 is different in kind, because it must **gather** its keys per query
    row: :func:`~kvpress.presses.gqa_indexer.sparse_support.gather_support_keys` materializes
    ``(B, h, query_tile, topk_tile, D)``, and the teacher gathers ``(B, H, ...)`` on top. That is
    ``O(query_tile * topk_tile * D)``, not ``O(query_tile * topk_tile)`` -- the ``D`` factor makes
    it 5.4 GB at ``query_tile=topk_tile=512, D=128`` on Qwen3-8B, which is why measurement came
    in at 0.69x of a prediction that ignored it.

    It is a fixed cost, so it does not change the slope; it just removes that much of the budget
    before the slope starts. Shrinking ``topk_tile`` is the direct lever.
    """
    if stage != "sparse":
        return 0.0
    fp32 = 4
    student = n_kv_heads * query_tile * topk_tile * head_dim * fp32
    teacher = n_heads * query_tile * topk_tile * head_dim * fp32
    return float(student + teacher)


def predict_max_length(
    budget_bytes: float, bytes_per_token: float, scratch_bytes: float = 0.0
) -> int:
    """Longest ``L`` that fits, after the fixed per-tile scratch is set aside."""
    return max(0, int((budget_bytes - scratch_bytes) // bytes_per_token))


# ----------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------
def build_model(layers: int | None, pretrained: bool, device: str, dtype: torch.dtype):
    """
    Qwen3-8B geometry. Untrained by default -- capacity depends on shape, not weights.

    Neither ``dtype`` nor ``attn_implementation`` is passed as a ``from_config`` kwarg:
    whether that function consumes them or forwards them to ``Model.__init__`` (where they are
    a ``TypeError``) varies by transformers version. They go on the config instead, which newer
    versions honour and older ones ignore, and the explicit cast below covers the latter.
    """
    from transformers import AutoModelForCausalLM

    if pretrained:
        # `dtype` replaced `torch_dtype` in the same release wave that changed _from_config,
        # so accept either rather than guessing which era the installed version is from.
        try:
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=dtype)
        if layers is not None:
            # Keep the PREFIX: FusedIndexerTrainer reads the KV cache by module.layer_idx, so
            # the surviving indices must stay 0..N-1 and contiguous. Slicing from the end
            # would leave layer_idx values that no longer match their cache slots.
            model.model.layers = model.model.layers[:layers]
            model.config.num_hidden_layers = layers
    else:
        from transformers import Qwen3Config

        # Qwen3Config directly rather than AutoConfig.for_model: same result, but it fails at
        # import with a clear name if the installed transformers predates Qwen3, instead of
        # raising "Unrecognized model identifier" from inside the registry.
        config = Qwen3Config(**QWEN3_8B)
        if layers is not None:
            config.num_hidden_layers = layers
        # Newer transformers reads `dtype` off the config (`kwargs.pop("dtype", config.dtype)`),
        # so this builds directly in bf16 and avoids a transient 33 GB fp32 model on CPU. Older
        # versions ignore the field, and the cast below then fixes it up.
        config.dtype = dtype
        config.torch_dtype = dtype
        model = AutoModelForCausalLM.from_config(config)

    model = model.to(device=device, dtype=dtype).eval()
    set_attn_implementation(model, "sdpa")
    return model


def set_attn_implementation(model, name: str) -> None:
    """
    Point the model at an attention kernel, on whichever config carries the flag.

    Assigning ``config._attn_implementation`` directly is what
    :func:`~kvpress.presses.gqa_indexer.teacher_lse.capture_teacher_lse` already does, and it
    sidesteps the ``from_config`` kwarg incompatibility entirely.
    """
    for config in (model.config, getattr(model.config, "text_config", None)):
        if config is not None:
            config._attn_implementation = name


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


def is_oom(exc: BaseException) -> bool:
    """
    Whether an exception is an allocation failure rather than a bug.

    A dedicated ``OutOfMemoryError`` counts unconditionally; a bare ``RuntimeError`` only if
    it says so. Getting this wrong in the permissive direction would let the search report a
    real bug as a capacity ceiling, which is the one outcome worse than crashing.
    """
    named = getattr(torch, "OutOfMemoryError", None)
    if named is not None and isinstance(exc, named):
        return True
    # Deliberately only this phrase. "CUDA error: an illegal memory access" also mentions
    # memory but is a bug, and classifying it as OOM would end the search with a plausible
    # number instead of a stack trace.
    return "out of memory" in str(exc).lower()


# ----------------------------------------------------------------------
# One probe
# ----------------------------------------------------------------------
def try_length(model, trainer, seq_len: int, *, verbose: bool = True) -> dict:
    """
    Attempt one forward + backward at ``seq_len``.

    ``backward()`` is not optional here: it is where the per-layer retained state is finally
    consumed, so a forward-only probe reports a length that then OOMs in real training.

    OOM is caught and recorded -- it is the expected outcome at the top of the search. Any
    *other* exception is recorded with ``fatal=True`` and stops the search, because a search
    that treats a bug as a capacity limit reports a confidently wrong number.
    """
    reset_peak()
    record: dict = {"seq_len": seq_len, "ok": False, "fatal": False, "peak_gib": float("nan")}
    input_ids = torch.randint(0, 1000, (1, seq_len), device=model.device)
    start = time.perf_counter()
    try:
        loss, _ = fused_indexer_training_step(model, trainer, input_ids=input_ids)
        loss.backward()
        record["ok"] = True
        record["loss"] = float(loss)
    except OOM_ERRORS as exc:
        if is_oom(exc):
            record["error"] = "OOM"
        else:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()
            record["fatal"] = True
    finally:
        record["peak_gib"] = peak_gib()
        record["seconds"] = time.perf_counter() - start
        del input_ids
        model.zero_grad(set_to_none=True)
        reset_peak()

    if verbose:
        if record["fatal"]:
            print(f"    L={seq_len:>7}  FAILED (not OOM): {record['error']}", flush=True)
        else:
            status = "ok  " if record["ok"] else "OOM "
            print(
                f"    L={seq_len:>7}  {status} peak {record['peak_gib']:6.2f} GiB  "
                f"{record['seconds']:6.2f}s",
                flush=True,
            )
    return record


def find_max_length(model, trainer, *, start: int, ceiling: int, verbose: bool = True) -> dict:
    """
    Double until failure, then bisect -- the standard capacity search.

    Doubling first keeps the cost dominated by the largest *successful* probe rather than by a
    long linear walk, and every probe is a full training step so the answer is directly usable.
    """
    history: list[dict] = []
    good, bad = 0, None
    length = start

    while length <= ceiling:
        record = try_length(model, trainer, length, verbose=verbose)
        history.append(record)
        if record["fatal"]:
            return {"max_length": good, "history": history, "hit_ceiling": False, "fatal": True}
        if not record["ok"]:
            bad = length
            break
        good = length
        length *= 2

    if bad is None:
        if verbose:
            print(f"    reached the ceiling {ceiling} without failing", flush=True)
        return {"max_length": good, "history": history, "hit_ceiling": True, "fatal": False}

    # Bisect the gap. Stop at 1024-token granularity: finer is noise, since allocator
    # fragmentation moves the boundary by more than that between runs.
    while bad - good > 1024:
        mid = (good + bad) // 2
        mid -= mid % 1024
        if mid <= good or mid >= bad:
            break
        record = try_length(model, trainer, mid, verbose=verbose)
        history.append(record)
        if record["fatal"]:
            break
        if record["ok"]:
            good = mid
        else:
            bad = mid

    return {"max_length": good, "history": history, "hit_ceiling": False, "fatal": False}


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def report(args, model, result: dict, prediction: dict, ideal: dict, scratch: float) -> None:
    layers = model.config.num_hidden_layers
    measured = result["max_length"]
    successes = [r for r in result["history"] if r["ok"]]

    print()
    print("=" * 74)
    print(f"stage={args.stage}  layers={layers}  topk={args.topk or '-'}  dtype={args.dtype}")
    print("=" * 74)

    if result.get("fatal"):
        broken = next(r for r in result["history"] if r.get("fatal"))
        print(f"\nSEARCH ABORTED at L={broken['seq_len']}: {broken['error']}")
        print("This is a bug, not a capacity limit -- the number below is a lower bound only.")
        print(broken.get("traceback", ""))

    print("\nretained bytes per token (analytic):")
    for name, value in sorted(prediction.items(), key=lambda kv: -kv[1]):
        if name == "total":
            continue
        share = 100 * value / prediction["total"]
        print(f"  {name:18s} {value / 1024:9.1f} KiB/tok  {share:5.1f}%")
    print(f"  {'TOTAL':18s} {prediction['total'] / 1024:9.1f} KiB/tok")

    if torch.cuda.is_available():
        total_gib = torch.cuda.get_device_properties(0).total_memory / GIB
        weights_gib = sum(p.numel() * p.element_size() for p in model.parameters()) / GIB
        budget = (total_gib - weights_gib) * GIB
        predicted = predict_max_length(budget, prediction["total"], scratch)
        predicted_ideal = predict_max_length(budget, ideal["total"], scratch)

        print(f"\ndevice {torch.cuda.get_device_name(0)}  {total_gib:.1f} GiB")
        print(f"  weights            {weights_gib:6.1f} GiB")
        print(f"  tile scratch       {scratch / GIB:6.1f} GiB   (fixed; O(query_tile*topk_tile*D))")
        print(f"  activation budget  {(budget - scratch) / GIB:6.1f} GiB")
        print(f"\n  predicted max L    {predicted:>8}")
        print(f"  MEASURED max L     {measured:>8}", end="")
        if predicted:
            print(f"   ({measured / predicted:.2f}x prediction)")
        else:
            print()
        print(f"  if teacher freed   {predicted_ideal:>8}   ({ideal['total'] / 1024:.1f} KiB/tok)")
        print(
            f"\n  => the retained fp32 teacher costs "
            f"{prediction['total'] / ideal['total']:.1f}x capacity"
        )

    if successes:
        best = successes[-1]
        print(
            f"\nlargest successful step: L={best['seq_len']} "
            f"peak {best['peak_gib']:.2f} GiB  {best['seconds']:.1f}s"
        )
        if len(successes) >= 2:
            a, b = successes[-2], successes[-1]
            span = b["seq_len"] - a["seq_len"]
            if span > 0 and math.isfinite(a["peak_gib"]) and math.isfinite(b["peak_gib"]):
                slope = (b["peak_gib"] - a["peak_gib"]) * GIB / span
                print(
                    f"observed slope between the last two: {slope / 1024:.1f} KiB/tok "
                    f"(analytic {prediction['total'] / 1024:.1f})"
                )

    if args.json:
        payload = {
            "stage": args.stage,
            "layers": layers,
            "topk": args.topk,
            "max_length": measured,
            "bytes_per_token": prediction,
            "bytes_per_token_if_teacher_freed": ideal,
            "tile_scratch_bytes": scratch,
            "history": result["history"],
        }
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.json}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--stage", choices=("dense", "sparse"), default="dense")
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="stage-2 support size. REQUIRED for sparse: a keep_ratio makes the retained "
        "support O(L^2) (696 GB at L=32K, r=0.25), so the search would measure that instead "
        "of the loss.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=None,
        help="truncate to this many layers for a fast run. Capacity scales linearly in layers, "
        "so a 4-layer probe times 9 approximates the full 36-layer answer.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="download the real 16 GB checkpoint. Capacity depends on geometry, not weight "
        "values, so this only confirms what the default already measures.",
    )
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--start", type=int, default=2048)
    parser.add_argument("--ceiling", type=int, default=1 << 20)
    parser.add_argument("--key-tile", type=int, default=512)
    parser.add_argument("--query-tile", type=int, default=512)
    parser.add_argument("--topk-tile", type=int, default=512)
    parser.add_argument("--backend", choices=("auto", "torch", "triton"), default="auto")
    parser.add_argument("--json", default=None, help="write the full record here")
    args = parser.parse_args()

    if args.stage == "sparse" and args.topk <= 0:
        parser.error(
            "--topk is required for --stage sparse. Deriving it from keep_ratio makes the "
            "retained support quadratic in L, so the probe would report the support's ceiling "
            "rather than the loss's."
        )
    if not torch.cuda.is_available():
        print("no CUDA device: this probe measures GPU capacity", file=sys.stderr)
        return 1

    dtype = getattr(torch, args.dtype)
    print(f"building Qwen3-8B ({'pretrained' if args.pretrained else 'untrained'})...", flush=True)
    model = build_model(args.layers, args.pretrained, "cuda", dtype)

    press = GQAIndexerPress(compression_ratio=0.5)
    press.post_init_from_model(model)
    freeze_all_but_indexer(model)

    trainer = FusedIndexerTrainer(
        press=press,
        stage=args.stage,
        key_tile=args.key_tile,
        query_tile=args.query_tile,
        topk_tile=args.topk_tile,
        topk=args.topk or None,
        backend=args.backend,
    )

    indexer = press.get_indexer(model.model.layers[0].self_attn)
    shared = dict(
        layers=model.config.num_hidden_layers,
        n_heads=model.config.num_attention_heads,
        n_kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
        indexer_dim=indexer.head_dim,
        stage=args.stage,
        topk=args.topk,
    )
    prediction = predict_bytes_per_token(**shared, retain_teacher=True)
    ideal = predict_bytes_per_token(**shared, retain_teacher=False)
    scratch = predict_tile_scratch(
        n_heads=model.config.num_attention_heads,
        n_kv_heads=model.config.num_key_value_heads,
        head_dim=indexer.head_dim,
        stage=args.stage,
        query_tile=args.query_tile,
        topk_tile=args.topk_tile,
    )

    print(f"searching from L={args.start} (doubling, then bisecting)...", flush=True)
    result = find_max_length(
        model, trainer, start=args.start, ceiling=args.ceiling, verbose=True
    )
    report(args, model, result, prediction, ideal, scratch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
