# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Can this environment get the teacher's logsumexp from flash-attention for free?

    python -m scripts.check_flash_lse
    python -m scripts.check_flash_lse --model /path/to/Qwen3-8B   # also check the real model

The teacher distribution is recoverable exactly from ``(logits, lse)``, so distillation never
needs the ``(H, L, L)`` attention matrix. flash-attention already computes ``lse`` internally and
returns it via ``return_attn_probs=True``, which would make the teacher side of the objective
nearly free. The fallback,
:func:`~kvpress.presses.gqa_indexer.fused_loss.teacher_lse_from_qk`, is exact but *recomputes*
the whole ``Q @ K^T`` -- it runs on every layer of every step, on both the torch and Triton
paths, because the Triton kernel takes ``lse`` as an input.

This script answers whether that recompute can be avoided **today**, and reports the four
independent requirements separately, because "flash-attn is installed" is only the first of
them:

1. ``flash_attn`` imports, and exposes ``flash_attn_func``.
2. It actually runs on this GPU and returns an ``lse`` of the documented shape.
3. The model dtype is fp16/bf16 -- there is no fp32 kernel.
4. No padding is in play. flash-attention's ``lse`` covers the causal mask only, so with
   padding ``exp(alpha - lse)`` stops summing to one over the kept keys and the rows with the
   most padding get silently under-weighted.

It also reports the fifth requirement, which is the one that actually decides the answer for
this repo right now: whether the training path *calls* the capture at all.

Exit code is 0 when the capture is usable end to end, 1 otherwise, so this can gate a launch.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OK = "OK  "
NO = "NO  "
WARN = "WARN"


def check_import() -> tuple[bool, str]:
    """Is flash_attn importable, and does it expose the entry point we need?"""
    if importlib.util.find_spec("flash_attn") is None:
        return False, "flash_attn is not installed (pip install flash-attn --no-build-isolation)"
    try:
        import flash_attn
        from flash_attn import flash_attn_func  # noqa: F401
    except Exception as exc:  # noqa: BLE001 -- a broken build fails in many ways
        return False, f"flash_attn present but unusable: {type(exc).__name__}: {exc}"
    return True, f"flash_attn {getattr(flash_attn, '__version__', '?')} imports"


def check_kernel_returns_lse() -> tuple[bool, str]:
    """
    Run the kernel and verify the lse it returns is the documented shape and finite.

    A real call rather than a version check: a wheel built for another CUDA minor version
    imports fine and then fails at launch, which is precisely the case a version check misses.
    """
    if not torch.cuda.is_available():
        return False, "no CUDA device, so the kernel cannot be exercised here"
    try:
        from flash_attn import flash_attn_func
    except Exception as exc:  # noqa: BLE001
        return False, f"cannot import flash_attn_func: {exc}"

    b, s, h, d = 1, 64, 4, 64
    q, k, v = (
        torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    )
    try:
        out = flash_attn_func(q, k, v, causal=True, return_attn_probs=True)
    except Exception as exc:  # noqa: BLE001
        return False, f"kernel launch failed: {type(exc).__name__}: {exc}"

    if not isinstance(out, tuple) or len(out) < 2:
        return False, f"return_attn_probs did not yield a tuple with an lse (got {type(out)})"
    lse = out[1]
    # Documented layout is (batch, nheads, seqlen); newer builds may transpose, which
    # normalize_captured_lse handles -- so accept either and say which was seen.
    shapes = {(b, h, s): "(batch, heads, seq)", (b, s, h): "(batch, seq, heads)"}
    label = shapes.get(tuple(lse.shape))
    if label is None:
        return False, f"unexpected lse shape {tuple(lse.shape)}; expected {b, h, s} or {b, s, h}"
    if not torch.isfinite(lse).all():
        return False, f"lse contains non-finite values, shape {tuple(lse.shape)} {label}"
    return True, f"kernel returns a finite lse, shape {tuple(lse.shape)} {label}"


def check_dtype(dtype_name: str) -> tuple[bool, str]:
    """fp32 has no flash kernel, and casting behind the caller's back is refused."""
    from kvpress.presses.gqa_indexer.teacher_lse import (
        assert_flash_dtype_supported,
    )

    dtype = getattr(torch, dtype_name)
    try:
        assert_flash_dtype_supported(dtype)
    except TypeError as exc:
        return False, str(exc).split(".")[0]
    return True, f"{dtype_name} is a flash dtype"


def check_no_padding() -> tuple[bool, str]:
    """
    Padding invalidates a captured lse, and this corpus never pads.

    Every sample is a full ``seq_len`` window of a single document (the loader enforces
    ``min_tokens >= seq_len``), so batches are uniform-length and no attention mask is passed.
    That is what makes the capture applicable here at all.
    """
    from kvpress.presses.gqa_indexer.teacher_lse import assert_lse_mask_compatible

    try:
        assert_lse_mask_compatible(None, "check")
    except Exception as exc:  # noqa: BLE001
        return False, f"unpadded batches unexpectedly rejected: {exc}"
    padded = torch.tensor([[1, 1, 0, 0]])
    try:
        assert_lse_mask_compatible(padded, "check")
    except Exception:
        return True, "unpadded batches accepted, padded correctly refused (loader never pads)"
    return False, "a padded mask was accepted, which would mis-normalize the teacher"


def check_training_path_uses_capture() -> tuple[bool, str]:
    """
    Does the code that actually trains call the capture?

    This is the requirement that decides the answer today, and the only one the other checks
    cannot reveal: ``FusedIndexerTrainer`` calls ``teacher_lse_from_qk`` unconditionally, so the
    lse is recomputed even where flash-attention could have handed it over. Checked by
    inspecting the trainer's source rather than by trusting this docstring to stay true.
    """
    import inspect

    from kvpress.presses.gqa_indexer import fused_trainer

    source = inspect.getsource(fused_trainer)
    calls_capture = "capture_teacher_lse" in source
    calls_recompute = "teacher_lse_from_qk(" in source
    if calls_capture:
        return True, "FusedIndexerTrainer calls capture_teacher_lse"
    if calls_recompute:
        return False, (
            "FusedIndexerTrainer calls teacher_lse_from_qk (recompute) and never "
            "capture_teacher_lse -- so the lse is recomputed regardless of flash-attn"
        )
    return False, "could not find either lse path in fused_trainer"


def check_model_attn(model_path: str) -> tuple[bool, str]:
    """Whether the checkpoint's config would even select a flash kernel for the backbone."""
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path)
    except Exception as exc:  # noqa: BLE001
        return False, f"could not read config from {model_path}: {type(exc).__name__}: {exc}"
    impl = getattr(config, "_attn_implementation", None) or "(unset)"
    heads = getattr(config, "num_attention_heads", "?")
    kv = getattr(config, "num_key_value_heads", "?")
    return True, f"{Path(model_path).name}: attn={impl}, H={heads}, KV={kv}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--model", default=None, help="also report the checkpoint's attn config")
    parser.add_argument(
        "--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32")
    )
    args = parser.parse_args()

    print("Can the teacher logsumexp come from flash-attention instead of being recomputed?")
    print()
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  device: {name} ({total:.0f} GiB), torch {torch.__version__}")
    else:
        print(f"  device: no CUDA, torch {torch.__version__}")
    print()

    checks = [
        ("1. flash_attn importable", check_import()),
        ("2. kernel returns an lse", check_kernel_returns_lse()),
        (f"3. dtype {args.dtype}", check_dtype(args.dtype)),
        ("4. no padding in batches", check_no_padding()),
        ("5. training path uses it", check_training_path_uses_capture()),
    ]
    if args.model:
        checks.append(("6. model config", check_model_attn(args.model)))

    for label, (passed, detail) in checks:
        print(f"  [{OK if passed else NO}] {label:<26} {detail}")

    usable = all(passed for _, (passed, _) in checks)
    print()
    if usable:
        print("VERDICT: the captured lse is usable -- the teacher logsumexp is free.")
        return 0

    blocked = [label for label, (passed, _) in checks if not passed]
    print(f"VERDICT: NOT usable. Blocked by: {', '.join(blocked)}")
    print()
    print("The teacher lse is therefore recomputed by teacher_lse_from_qk on every layer of")
    print("every step. That is exact and correct -- it is a throughput cost, not a bug, and it")
    print("does not block training.")
    if any(label.startswith("5.") for label in blocked):
        print()
        print("Note that check 5 is the binding one: wiring capture_teacher_lse into")
        print("FusedIndexerTrainer is a code change, so installing flash-attn alone would not")
        print("make the lse free.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
