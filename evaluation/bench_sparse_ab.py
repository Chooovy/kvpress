# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Time the sparse-attention eval's inner loop under one variable at a time.

The question this answers: the RULER sparse eval went from ~1h to ~13h for 650 samples, and two
things changed together -- the indexer checkpoint (distill -> e2e) and ``BLOCK_G``, which the
Triton-3.3 ``min_dot_size`` fix padded from ``group_size`` up to 16. Wall-clock on the full eval
cannot separate them, so this measures the prefill directly with each variable pinned.

What it does *not* do is run the dataset: the eval's cost is dominated by the per-question sparse
prefill, so one synthetic prefill at the real shape is the signal, and it takes seconds instead of
hours. Generation length is measured separately (``--decode``) because it is the one channel
through which the checkpoint's *weights* -- not its shapes -- can change runtime, via the
EOS early-exit in ``kvpress/pipeline.py``.

Usage
-----
    python bench_sparse_ab.py --ckpt A.pt --ckpt-label distill \\
                              --ckpt B.pt --ckpt-label e2e \\
                              --block-g 4 --block-g 16 --length 8192 --topk 2048
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from contextlib import contextmanager

import torch

logger = logging.getLogger("bench")
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

@contextmanager
def pinned_block_g(value: int | None):
    """
    Force the kernel's ``M`` floor, or leave it as the backend reports.

    Patching :func:`~.triton_sparse_attention.min_dot_m` rather than editing the launcher is
    what makes the two arms differ in exactly one constexpr: same indices, same tiles, same
    everything else.
    """
    import kvpress.presses.gqa_indexer.triton_sparse_attention as tsa

    if value is None:
        yield tsa.min_dot_m()
        return
    original = tsa.min_dot_m
    tsa.min_dot_m = lambda: value
    try:
        yield value
    finally:
        tsa.min_dot_m = original


@contextmanager
def pinned_precision(value: str | None):
    """
    Force the kernel's ``tl.dot`` precision, which the eval path does not plumb.

    Worth measuring alongside ``BLOCK_G`` because it decides whether ``M`` costs anything at
    all: the kernel casts q/k/v to fp32 and defaults to ``"ieee"``, which does not use tensor
    cores, so ``M`` is a real loop bound. Under ``"tf32"`` the MMA is ``m16n8k16`` and M is
    padded to 16 by the hardware regardless -- if that is what is happening, padding is free
    and this arm shows it.
    """
    import kvpress.presses.gqa_indexer.sparse_inference as si

    if value is None:
        yield None
        return
    original = si.sparse_gqa_attention

    def with_precision(*args, **kwargs):
        kwargs.setdefault("precision", value)
        return original(*args, **kwargs)

    si.sparse_gqa_attention = with_precision
    try:
        yield value
    finally:
        si.sparse_gqa_attention = original


@contextmanager
def pinned_key_tile(value: int | None):
    """
    Force ``streaming_topk_support``'s key tile, which the inference path does not expose.

    The running-buffer top-k re-sorts ``take`` kept candidates alongside each new tile, so its
    work carries a factor of ``(1 + take/key_tile)`` -- at the default 512 with ``topk=2048``
    that is ~4.9x, almost all of it redundant. Patched here rather than threaded through
    ``SparseAttentionContext`` so the measurement can decide whether plumbing it is worth it.
    """
    import kvpress.presses.gqa_indexer.sparse_inference as si

    if value is None:
        yield None
        return
    original = si.streaming_topk_support

    def with_tile(*args, **kwargs):
        kwargs.setdefault("key_tile", value)
        return original(*args, **kwargs)

    si.streaming_topk_support = with_tile
    try:
        yield value
    finally:
        si.streaming_topk_support = original


def load_press_and_model(model_path: str, ckpt_path: str, device: str, dtype: torch.dtype):
    """Build the model + indexer exactly as evaluate_sparse.py does, so timings transfer."""
    from transformers import AutoModelForCausalLM

    from kvpress.presses.gqa_indexer import GQAIndexerPress, load_indexer_state_dict

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model = model.to(device).eval()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    indexer_sd = ckpt.get("indexer", ckpt)
    has_gate = any(str(k).endswith("gate_scale") for k in indexer_sd)
    press = GQAIndexerPress(
        compression_ratio=0.0, gate_scale=has_gate, scorer_attr="indexer"
    )
    press.post_init_from_model(model)
    load_indexer_state_dict(model, indexer_sd, "indexer")
    return model, press, ckpt.get("config"), has_gate


def time_prefill(model, press, *, length, topk, force_sink, force_local, block_k, reps, device):
    """
    Median wall-clock of one sparse prefill at the eval's real shape.

    Median over ``reps`` after a discarded warmup: the warmup pays Triton's JIT compile, which
    is per-``BLOCK_G`` and would otherwise land entirely on whichever arm runs first.

    Raises :class:`Unsupported` when the tile shape will not compile on this Triton, so one
    impossible arm reports itself instead of aborting the sweep -- which is the whole point on
    a 3.3 box, where ``BLOCK_G=4`` is exactly the configuration under investigation and also
    the one that cannot run.
    """
    from transformers import DynamicCache

    from kvpress.presses.gqa_indexer import SparseAttentionContext

    ids = torch.randint(0, 30000, (1, length), device=device)
    times = []
    for i in range(reps + 1):
        with SparseAttentionContext(
            model, press, topk=topk, force_sink=force_sink,
            force_local=force_local, block_k=block_k,
        ):
            cache = DynamicCache()
            torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                with torch.no_grad():
                    model.model(input_ids=ids, past_key_values=cache)
            except Exception as exc:  # noqa: BLE001 -- classified below, re-raised if unrelated
                if _is_dot_shape_failure(exc):
                    raise Unsupported(str(exc).strip().splitlines()[-1]) from exc
                raise
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        if i:  # drop the warmup / JIT compile
            times.append(elapsed)
        del cache
        torch.cuda.empty_cache()
    times.sort()
    return times[len(times) // 2], times


class Unsupported(RuntimeError):
    """This tile shape does not compile on the installed Triton."""


def _is_dot_shape_failure(exc: BaseException) -> bool:
    """
    Whether ``exc`` is Triton rejecting the dot's tile shape rather than a real error.

    Matched on the message across the whole cause chain, because Triton wraps the frontend
    ``AssertionError`` in a ``CompilationError`` -- the assert text is the only part that
    survives both layers and both 3.3/3.4 wordings.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if "Input shapes should have M >=" in str(exc):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def time_stages(model, press, *, length, topk, force_sink, force_local, block_k, reps, device):
    """
    Split the prefill into indexer-selection vs sparse-attention time.

    This is the measurement that decides where to spend effort at all: ``BLOCK_G`` only scales
    the attention kernel, so if selection dominates then padding ``M`` cannot be the story no
    matter what ratio the other arms show. Timed by wrapping the two functions
    ``SparseAttentionContext._attend`` calls and accumulating with a device sync around each,
    which is heavier than the real forward but attributes the cost honestly.
    """
    from transformers import DynamicCache

    from kvpress.presses.gqa_indexer import SparseAttentionContext
    import kvpress.presses.gqa_indexer.sparse_inference as si

    ids = torch.randint(0, 30000, (1, length), device=device)
    totals = {"select_s": 0.0, "attend_s": 0.0}
    orig_topk, orig_attn = si.streaming_topk_support, si.sparse_gqa_attention

    def timed(fn, key):
        def wrapper(*a, **k):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(*a, **k)
            torch.cuda.synchronize()
            totals[key] += time.perf_counter() - t0
            return out
        return wrapper

    si.streaming_topk_support = timed(orig_topk, "select_s")
    si.sparse_gqa_attention = timed(orig_attn, "attend_s")
    try:
        for i in range(reps + 1):
            if i == 1:  # reset after the warmup so JIT compile is excluded
                totals["select_s"] = totals["attend_s"] = 0.0
            with SparseAttentionContext(
                model, press, topk=topk, force_sink=force_sink,
                force_local=force_local, block_k=block_k,
            ):
                cache = DynamicCache()
                with torch.no_grad():
                    model.model(input_ids=ids, past_key_values=cache)
            del cache
            torch.cuda.empty_cache()
    finally:
        si.streaming_topk_support, si.sparse_gqa_attention = orig_topk, orig_attn
    return {k: round(v / max(reps, 1), 3) for k, v in totals.items()}


def count_generated(model, press, *, length, topk, force_sink, force_local,
                    block_k, max_new_tokens, device):
    """
    How many tokens the model actually emits before EOS.

    This is the checkpoint's only route to changing runtime without changing a single shape:
    ``kvpress/pipeline.py`` breaks the decode loop on EOS, so an indexer that degrades the
    model into never emitting one runs the full ``max_new_tokens`` on every question.
    """
    from transformers import DynamicCache

    from kvpress.presses.gqa_indexer import SparseAttentionContext

    ids = torch.randint(0, 30000, (1, length), device=device)
    stop = model.generation_config.eos_token_id
    stop = stop if isinstance(stop, list) else [stop]

    with SparseAttentionContext(
        model, press, topk=topk, force_sink=force_sink,
        force_local=force_local, block_k=block_k,
    ):
        cache = DynamicCache()
        with torch.no_grad():
            out = model(input_ids=ids, past_key_values=cache)
            nxt = out.logits[0, -1].argmax()
            n = 1
            pos = torch.tensor([[length]], device=device)
            for i in range(max_new_tokens - 1):
                out = model(
                    input_ids=nxt.view(1, 1), past_key_values=cache, position_ids=pos + i
                )
                nxt = out.logits[0, -1].argmax()
                n += 1
                if nxt.item() in stop:
                    break
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B")
    p.add_argument("--ckpt", action="append", required=True, help="repeat for each checkpoint")
    p.add_argument("--ckpt-label", action="append", default=[], help="label per --ckpt")
    p.add_argument("--block-g", action="append", type=int, default=[],
                   help="force this M floor; repeat. Omit for the backend's own answer.")
    p.add_argument("--precision", action="append", default=[],
                   help="tl.dot precision: ieee or tf32; repeat. Omit for the kernel default.")
    p.add_argument("--length", action="append", type=int, default=[],
                   help="context length; repeat to sweep and expose the scaling exponent.")
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--force-sink", type=int, default=4)
    p.add_argument("--force-local", type=int, default=64)
    p.add_argument("--block-k", type=int, default=64)
    p.add_argument("--key-tile", action="append", type=int, default=[],
                   help="streaming_topk_support key tile; repeat. Default 512 re-sorts the kept "
                        "buffer on every tile, so work carries a (1 + take/key_tile) factor.")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--stages", action="store_true",
                   help="split prefill into indexer-selection vs sparse-attention time")
    p.add_argument("--decode", action="store_true", help="also measure generated-token count")
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", default="")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = "cuda:0"  # CUDA_VISIBLE_DEVICES selects the physical GPU
    dtype = getattr(torch, args.dtype)

    import triton

    from kvpress.presses.gqa_indexer.triton_sparse_attention import min_dot_m

    logger.info("triton %s | torch %s | %s", triton.__version__, torch.__version__,
                torch.cuda.get_device_name(0))
    logger.info("backend reports min_dot_m() = %d  <-- 1 means BLOCK_G stays at group_size", min_dot_m())

    labels = args.ckpt_label + [f"ckpt{i}" for i in range(len(args.ckpt_label), len(args.ckpt))]
    block_gs = args.block_g or [None]
    precisions = args.precision or [None]
    lengths = args.length or [8192]
    key_tiles = args.key_tile or [None]
    results = []

    for ckpt_path, label in zip(args.ckpt, labels):
        logger.info("loading %s (%s)", label, ckpt_path)
        model, press, cfg, has_gate = load_press_and_model(args.model, ckpt_path, device, dtype)
        logger.info("  ckpt config=%s gate_scale=%s", cfg, has_gate)

        for length in lengths:
            for bg in block_gs:
                for prec in precisions:
                    for kt in key_tiles:
                        with (
                            pinned_block_g(bg) as effective,
                            pinned_precision(prec),
                            pinned_key_tile(kt),
                        ):
                            row = {
                                "ckpt": label, "length": length, "block_g_floor": effective,
                                "precision": prec or "default(ieee)",
                                "key_tile": kt or 512,
                            }
                            common = dict(
                                length=length, topk=args.topk, force_sink=args.force_sink,
                                force_local=args.force_local, block_k=args.block_k,
                                device=device,
                            )
                            try:
                                median, all_times = time_prefill(
                                    model, press, reps=args.reps, **common
                                )
                            except Unsupported as exc:
                                row["prefill_s"] = None
                                row["unsupported"] = str(exc)
                                logger.warning("  L=%d M=%s prec=%s: WILL NOT COMPILE -- %s",
                                               length, effective, row["precision"], exc)
                                results.append(row)
                                continue
                            row["prefill_s"] = round(median, 3)
                            row["all_s"] = [round(t, 3) for t in all_times]
                            if args.stages:
                                row.update(time_stages(
                                    model, press, reps=args.reps, **common
                                ))
                            if args.decode:
                                row["generated_tokens"] = count_generated(
                                    model, press, max_new_tokens=args.max_new_tokens, **common
                                )
                            logger.info("  %s", json.dumps(row))
                            results.append(row)

        del model, press
        torch.cuda.empty_cache()

    ok = [r for r in results if r.get("prefill_s") is not None]
    print(f"\n=== prefill seconds, topk={args.topk} (median of {args.reps}) ===")
    header = (f"{'ckpt':8s} {'L':>6s} {'M':>4s} {'precision':>14s} {'ktile':>6s} {'prefill_s':>10s}")
    if args.stages:
        header += f" {'select_s':>9s} {'attend_s':>9s}"
    print(header + f" {'gen_tok':>8s}")
    for r in results:
        cell = "N/A" if r.get("prefill_s") is None else f"{r['prefill_s']:.3f}"
        line = (f"{r['ckpt']:8s} {r['length']:>6} {r['block_g_floor']:>4} "
                f"{r['precision']:>14s} {r['key_tile']:>6} {cell:>10s}")
        if args.stages:
            line += f" {r.get('select_s', '-'):>9} {r.get('attend_s', '-'):>9}"
        print(line + f" {r.get('generated_tokens', '-'):>8}")
    for r in results:
        if r.get("unsupported"):
            print(f"  note: M={r['block_g_floor']} does not compile on this Triton "
                  f"({r['unsupported']})")

    # Each ratio moves exactly one variable, which is the whole point of the sweep.
    print("\n=== isolated ratios ===")

    def group(keyfn, varfn, name):
        seen = {}
        for r in ok:
            seen.setdefault(keyfn(r), []).append(r)
        for key, rows in seen.items():
            rows = sorted(rows, key=lambda r: str(varfn(r)))
            if len(rows) > 1:
                base = rows[0]
                for r in rows[1:]:
                    print(f"  [{key}] {name} {varfn(base)} -> {varfn(r)} = "
                          f"{r['prefill_s'] / base['prefill_s']:.2f}x")

    def tag(r, *skip):
        bits = [f"L={r['length']}", f"M={r['block_g_floor']}", r["precision"],
                f"ktile={r['key_tile']}"]
        return ", ".join(b for i, b in enumerate(bits) if i not in skip)

    group(lambda r: tag(r, 1), lambda r: r["block_g_floor"], "M floor")
    group(lambda r: tag(r, 2), lambda r: r["precision"], "precision")
    group(lambda r: tag(r, 3), lambda r: r["key_tile"], "key_tile")
    if len({r["ckpt"] for r in ok}) > 1:
        group(lambda r: tag(r), lambda r: r["ckpt"], "ckpt")

    # Scaling exponent: the attention kernel is O(L) at fixed topk, selection is O(L^2). An
    # exponent near 2 therefore says selection dominates, and near 1 says it does not -- which
    # is the question a single length cannot answer.
    if len(lengths) > 1:
        print("\n=== scaling in L (exponent: 1 = linear/kernel-bound, 2 = quadratic/select-bound) ===")
        import math

        seen = {}
        for r in ok:
            seen.setdefault((r["ckpt"], r["block_g_floor"], r["precision"], r["key_tile"]), []).append(r)
        for key, rows in seen.items():
            rows = sorted(rows, key=lambda r: r["length"])
            for a, b in zip(rows, rows[1:]):
                ratio = b["prefill_s"] / a["prefill_s"]
                expo = math.log(ratio) / math.log(b["length"] / a["length"])
                print(f"  M={key[1]}, {key[2]}, ktile={key[3]}: "
                      f"L {a['length']}->{b['length']} = {ratio:.2f}x  (exponent {expo:.2f})")

    if args.stages and ok:
        print("\n=== where the time goes (per prefill) ===")
        for r in ok:
            sel, att = r.get("select_s", 0.0), r.get("attend_s", 0.0)
            tot = sel + att
            if tot:
                print(f"  L={r['length']}, M={r['block_g_floor']}, {r['precision']}, "
                      f"ktile={r['key_tile']}: select {sel:.3f}s ({100*sel/tot:.0f}%), "
                      f"attend {att:.3f}s ({100*att/tot:.0f}%)")
        print("  (BLOCK_G/precision scale `attend` only -- if `select` dominates, look there)")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
