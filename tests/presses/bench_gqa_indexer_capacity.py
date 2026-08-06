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
cannot predict its own answer cannot distinguish a genuine ceiling from a leak, and the first
thing it turned up was a leak (see below).

Two things dominate, and neither is the tiled algorithm:

**The fp32 teacher is retained for every layer.** ``_FusedIndexerCE.forward`` stores
``ctx.teacher_alpha``, which is the closure from ``make_recompute_teacher`` holding fp32 copies
of the teacher's Q and K. One autograd node per layer means all 36 layers' teachers stay
resident until ``backward()`` -- 720 KiB/token of the 1086 KiB/token total, so **3.0x** the
footprint of freeing them (366 KiB/token). It also contradicts the ``fused_trainer`` docstring's
claim that "one layer's teacher tensors are alive at a time": true of the forward pass, false of
the autograd graph.

On an 80 GiB card that is the difference between L≈62K and L≈183K for stage 1.

**Stage 2's support tensor is quadratic under ``keep_ratio``.** ``support`` is
``(B, h, Sq, topk)`` int64 plus a bool ``valid``, saved for backward. With ``topk = r * L`` that
is ``O(L^2)``: at ``L=32K, r=0.25`` it is 696 GB across 36 layers. A *fixed* ``topk`` keeps it
linear, so the sparse probe requires one and refuses a ratio. At ``topk=512`` the support is
1296 KiB/token, which then dominates everything else -- ``int32`` indices would halve it, since
they only need to address ``Sk < 2^31``.
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

    Only tensors that survive until ``backward()`` are counted; tile scratch is
    ``O(query_tile * key_tile)`` and independent of ``L``, so it shifts the intercept rather
    than the slope.

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


def predict_max_length(budget_bytes: float, bytes_per_token: float) -> int:
    return max(0, int(budget_bytes // bytes_per_token))


# ----------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------
def build_model(layers: int | None, pretrained: bool, device: str, dtype: torch.dtype):
    """Qwen3-8B geometry. Untrained by default -- capacity depends on shape, not weights."""
    from transformers import AutoConfig, AutoModelForCausalLM

    if pretrained:
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-8B", dtype=dtype, attn_implementation="sdpa"
        )
    else:
        config = AutoConfig.for_model("qwen3", **QWEN3_8B)
        if layers is not None:
            config.num_hidden_layers = layers
        config.attn_implementation = "sdpa"
        # from_config skips the (slow, irrelevant) weight download.
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)

    if layers is not None and pretrained:
        # Keep the PREFIX: FusedIndexerTrainer reads the KV cache by module.layer_idx, so the
        # surviving indices must stay 0..N-1 and contiguous. Slicing from the end would leave
        # layer_idx values that no longer match their cache slots.
        language_model = model.model
        language_model.layers = language_model.layers[:layers]
        model.config.num_hidden_layers = layers

    return model.to(device).eval()


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
def report(args, model, result: dict, prediction: dict, ideal: dict) -> None:
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
        predicted = predict_max_length(budget, prediction["total"])
        predicted_ideal = predict_max_length(budget, ideal["total"])

        print(f"\ndevice {torch.cuda.get_device_name(0)}  {total_gib:.1f} GiB")
        print(f"  weights            {weights_gib:6.1f} GiB")
        print(f"  activation budget  {budget / GIB:6.1f} GiB")
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

    print(f"searching from L={args.start} (doubling, then bisecting)...", flush=True)
    result = find_max_length(
        model, trainer, start=args.start, ceiling=args.ceiling, verbose=True
    )
    report(args, model, result, prediction, ideal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
