# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Precompute LongCE key-token weights for the pretokenized corpus, once, into a cache.

    # one GPU per shard-slice; 8 workers cover the corpus in parallel
    for i in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=$i python -m scripts.precompute_longce_weights \\
        --model /path/Qwen3-8B --tokenized /path/longmino_tokenized_64k \\
        --subsets 2e16 2e17 --seq-len 16384 --trunc-len 1024 \\
        --out /path/longce_weights_16k --shard-index $i --shard-count 8 &
    done; wait

Why precompute at all
---------------------
The backbone is **frozen**, so ``L^short`` and ``L^long`` are functions of the *data* only -- nothing
about them changes as the router trains. Recomputing them every step would pay
``(L - K) / d`` extra forward passes (15 of 2048 tokens at ``L=16384, K=d=1024``) for a constant, on
a step whose trainable part is just the indexer. It would also make each step's weights reflect that
batch's difficulty rather than anything intrinsic, which is one of the ways the earlier delta arm
went wrong. See :mod:`kvpress.presses.gqa_indexer.longce_weights` for the full argument.

What it writes
--------------
One ``.npz`` per source shard, mirroring the corpus's ``subset/shard`` layout:

* ``weights``   -- fp16 ``(n_docs, seq_len - 1)``, the ``min(exp(LSD), gamma)`` multiplier
* ``doc_ids``   -- the corpus's own ids, which is what the cache is keyed by
* ``checksums`` -- ``(n_docs, n_widths)`` ``blake2b`` digests of the token prefix at each stage width
* ``meta``      -- the settings, so a mismatched consumer is refused rather than tolerated

32 KB per 16K document, ~12 GB for the full 373K-document corpus at 16K -- and far less in practice,
because ``--max-docs-per-shard`` bounds it to what a run actually draws.

The alignment contract, which is the thing that can go silently wrong
--------------------------------------------------------------------
Weights attached to the wrong tokens still produce a falling loss and healthy diagnostics; only the
benchmark shows it. Three choices exist to make that impossible rather than unlikely:

1. **Keyed by ``doc_id``, not by position.** The training loader shuffles rows and partitions shards
   across ``(rank, worker)``, so any positional key would break whenever the world size, worker count
   or seed changed -- while still returning a correctly-*shaped* vector.
2. **Digests at every stage width.** ``--checksum-widths`` defaults to every distinct ``seq_len`` in
   the curriculum, so the 8K stage verifies against an 8K digest and the 16K stage against a 16K one.
   A single full-width digest could not be checked from a stage that only draws a prefix.
3. **``take_from="head"``, enforced.** The cached vector is truncated per stage, which is only valid
   because the losses are causal and a shorter stage reads a *prefix*. ``--take-from random`` would
   draw a window that is not a prefix, so this script hardcodes ``head`` and the trainer asserts it.

Resumable: a shard whose ``.npz`` already exists is skipped, and each file is written to a temporary
sibling and renamed, so an interrupted run leaves complete shards plus nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.probe_longce_key_tokens import (  # noqa: E402
    long_context_losses,
    short_context_losses,
)
from kvpress.presses.gqa_indexer.data import read_index  # noqa: E402
from kvpress.presses.gqa_indexer.longce_weights import (  # noqa: E402
    DEFAULT_GAMMA,
    DEFAULT_TRUNC_LEN,
    DEFAULT_WINDOW,
    WeightCacheMeta,
    longce_weights,
    shard_cache_path,
    token_checksum,
    write_shard_cache,
)
from scripts.train_gqa_indexer import build_model  # noqa: E402

logger = logging.getLogger("precompute_longce")


def shard_paths(root: Path, subsets: list[str]) -> list[tuple[str, Path]]:
    """``(subset, path)`` for every ``.npy`` shard, in a stable order across workers."""
    index = read_index(root)
    known = sorted(index["subsets"])
    chosen = subsets or known
    missing = [s for s in chosen if s not in index["subsets"]]
    if missing:
        raise SystemExit(f"subsets {missing} are not in the corpus (have {known})")
    found: list[tuple[str, Path]] = []
    for subset in sorted(chosen):
        paths = sorted((root / subset).glob("*.npy"))
        if not paths:
            raise SystemExit(f"no .npy shards under {root / subset}")
        found.extend((subset, path) for path in paths)
    return found


def weights_for_document(
    model,
    tokens: np.ndarray,
    *,
    trunc_len: int,
    window: int,
    gamma: float,
    logit_chunk: int,
    device: str,
    verify_alignment: bool,
) -> tuple[np.ndarray, dict]:
    """
    The ``(seq_len - 1,)`` fp16 weight vector for one document, plus its alignment evidence.

    Reuses the probe's ``long_context_losses`` / ``short_context_losses`` rather than reimplementing
    the sliding window: those are the functions whose indexing was verified against the reference
    (``start=0`` window reproducing the full-context loss to 0.0), and a second copy of that loop
    would be a second chance to get the off-by-one wrong.
    """
    input_ids = torch.from_numpy(tokens.astype(np.int64)).unsqueeze(0).to(device)
    long_loss = long_context_losses(model, input_ids, logit_chunk=logit_chunk)
    short_loss, scored, windows = short_context_losses(
        model,
        input_ids,
        long_loss,
        trunc_len=trunc_len,
        window=window,
        logit_chunk=logit_chunk,
    )
    weights = longce_weights(long_loss, short_loss, scored, gamma=gamma)

    evidence: dict = {}
    if verify_alignment:
        # The one structural check that catches an off-by-one: the start=0 window sees the true
        # prefix, so its short loss must equal the long loss. Run on the first document of each
        # shard rather than every document -- it is a property of the code path, not of the data,
        # and the per-document cost is a full extra comparison.
        first = windows[0]
        evidence = {
            "start0_max_abs_diff": first["max_abs_diff_vs_long"],
            "truncated_max_abs_diff": max(
                (w["max_abs_diff_vs_long"] for w in windows[1:]), default=0.0
            ),
        }
        if first["max_abs_diff_vs_long"] > 1e-3:
            raise AssertionError(
                f"the start=0 window disagrees with the long-context loss by "
                f"{first['max_abs_diff_vs_long']:.3e}, which is an off-by-one in the window "
                "slicing. Every weight in this cache would be attached to the wrong position."
            )

    stats = {
        "weight_mean": float(weights.mean()),
        "weight_upweighted_frac": float((weights > 1.01).float().mean()),
        "weight_at_ceiling_frac": float((weights >= gamma - 1e-6).float().mean()),
        "scored_frac": float(scored.float().mean()),
        "long_loss_mean": float(long_loss.mean()),
    }
    return weights.cpu().numpy().astype(np.float16), {**stats, **evidence}


def loader_draw_plan(
    root: Path,
    subsets: list[str],
    *,
    seq_len: int,
    seed: int,
    world_size: int,
    num_workers: int,
    docs_per_worker: int,
) -> dict[tuple[str, str], list[int]]:
    """
    The exact ``(shard, rows)`` the training loader will draw, mirroring :class:`TokenizedDataset`.

    Why this exists rather than "cache a slice of every shard": each ``(rank, worker)`` reader walks
    its assigned shards **sequentially** and shuffles the rows *within* a shard. So a 600-step run
    does not touch a thin slice of all 58 shards -- it exhausts the first one or two of each reader's
    list, at row positions determined by that reader's own RNG. Caching stored-order prefixes of
    every shard gives a 2.5% hit rate (measured: 128 cached of 5119 rows), which means ~97% of the
    run's documents silently fall back to weight 1 and the "LongCE" arm is the plain objective with a
    different directory name.

    Mirroring the loader instead makes the hit rate ~100% for a fraction of the compute -- ~4800
    documents for a 600-step run at ``--global-batch-size 8``, against 373K in the corpus.

    Three details have to match the loader exactly or the plan is worthless:

    * the **shard** shuffle is ``Random(config.seed)`` over the *whole* path list, before the
      ``(rank, worker)`` slice, and ``config.seed`` is ``args.seed + seq_len`` (see ``loader_for``) --
      so a different stage draws a different order, which is why ``seq_len`` is a parameter here;
    * the **row** shuffle is ``Random((seed, rank, worker_id).__hash__())``, which is deterministic
      across processes because ``hash`` is only randomized for ``str``/``bytes``, not for tuples of
      ints;
    * ``shuffle_buffer`` reorders *emission* but not *selection*, so it does not change which rows are
      read -- only when. ``docs_per_worker`` is padded by the caller to absorb the boundary.

    Returns ``{(subset, shard_stem): [rows]}``, so a worker that owns a shard scores exactly the rows
    some reader will ask for.
    """
    paths: list[Path] = []
    for subset in sorted(subsets):
        paths.extend(sorted((root / subset).glob("*.npy")))

    # loader_for(): TokenizedConfig(seed=args.seed + seq_len)
    config_seed = seed + seq_len
    shuffled = list(paths)
    random.Random(config_seed).shuffle(shuffled)

    plan: dict[tuple[str, str], list[int]] = {}
    for rank in range(world_size):
        for worker_id in range(max(num_workers, 1)):
            workers = max(num_workers, 1)
            assigned = shuffled[rank * workers + worker_id :: world_size * workers]
            rng = random.Random((config_seed, rank, worker_id).__hash__())
            remaining = docs_per_worker
            for path in assigned:
                if remaining <= 0:
                    break
                # np.load with mmap reads only the header for .shape, so this costs no I/O.
                n_rows = np.load(path, mmap_mode="r").shape[0]
                order = list(range(n_rows))
                rng.shuffle(order)
                take = order[:remaining]
                key = (path.parent.name, path.stem)
                plan.setdefault(key, [])
                plan[key].extend(take)
                remaining -= len(take)
    return {key: sorted(set(rows)) for key, rows in plan.items()}


def rows_to_score(n_rows: int, *, max_docs: int, seed: int, shuffle: bool) -> list[int]:
    """
    Which rows of a shard to score when not mirroring the loader.

    Sampled uniformly rather than taken in stored order: the corpus is ordered by document, so a
    stored prefix is not a random sample of anything, and the loader shuffles rows before reading
    them anyway. Uniform sampling at least makes a partial cache *unbiased* -- but see
    :func:`loader_draw_plan` for why ``--mirror-loader`` is the mode that actually gets used.
    """
    if max_docs <= 0 or max_docs >= n_rows:
        return list(range(n_rows))
    if not shuffle:
        return list(range(max_docs))
    rows = list(range(n_rows))
    random.Random(seed).shuffle(rows)
    return sorted(rows[:max_docs])


def process_shard(
    model,
    subset: str,
    path: Path,
    *,
    args,
    meta: WeightCacheMeta,
    rows: list[int] | None = None,
) -> dict | None:
    """Compute and write one shard's cache, or return ``None`` if it already exists."""
    # Suffixed with the worker index when rows are striped across workers: several workers hold
    # different rows of the SAME source shard, so a shared filename would have them overwrite each
    # other and the cache would end up with one worker's slice instead of the union. The reader globs
    # `*/*.npz` and unions by doc_id, so extra files per shard are transparent to it.
    stem = path.stem if args.shard_count == 1 else f"{path.stem}.w{args.shard_index}"
    out_path = shard_cache_path(args.out, subset, stem)
    if out_path.is_file() and not args.overwrite:
        logger.info("%s/%s: already cached, skipping", subset, path.stem)
        return None

    array = np.load(path, mmap_mode="r")
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        raise SystemExit(
            f"{path.name} has no .json sidecar, so its doc_ids are unknown. The cache is keyed by "
            "doc_id; without it there is no way to match a cached row to a loader sample."
        )
    with open(sidecar) as handle:
        doc_ids_all = json.load(handle)["doc_ids"]

    if rows is None:
        rows = rows_to_score(
            array.shape[0],
            max_docs=args.max_docs_per_shard,
            # Seeded per shard so parallel workers pick independent subsets and a re-run with
            # --overwrite reproduces the same choice.
            seed=hash((args.seed, path.stem)) & 0xFFFFFFFF,
            shuffle=not args.rows_in_order,
        )
    n_rows = len(rows)
    weights = np.zeros((n_rows, args.seq_len - 1), dtype=np.float16)
    checksums = np.empty((n_rows, len(args.checksum_widths)), dtype="U16")
    doc_ids: list[str] = []
    rollup: dict[str, float] = {}
    started = time.time()

    for position, row in enumerate(rows):
        tokens = np.array(array[row, : args.seq_len], dtype=np.int64)
        row_weights, stats = weights_for_document(
            model,
            tokens,
            trunc_len=args.trunc_len,
            window=args.window,
            gamma=args.gamma,
            logit_chunk=args.logit_chunk,
            device=args.device,
            verify_alignment=(position == 0),
        )
        weights[position] = row_weights
        for column, width in enumerate(args.checksum_widths):
            checksums[position, column] = token_checksum(tokens[:width])
        doc_ids.append(
            doc_ids_all[row] if row < len(doc_ids_all) else f"{path.stem}:{row}"
        )
        for key, value in stats.items():
            rollup[key] = rollup.get(key, 0.0) + value
        if position == 0:
            logger.info(
                "%s/%s: alignment start0=%.3e truncated=%.3e (checked on the first document)",
                subset,
                path.stem,
                stats["start0_max_abs_diff"],
                stats["truncated_max_abs_diff"],
            )
        if args.log_every > 0 and (position + 1) % args.log_every == 0:
            rate = (position + 1) / (time.time() - started)
            logger.info(
                "%s/%s: %d/%d docs (%.2f doc/s, w_mean=%.3f up=%.3f)",
                subset,
                path.stem,
                position + 1,
                n_rows,
                rate,
                rollup["weight_mean"] / (position + 1),
                rollup["weight_upweighted_frac"] / (position + 1),
            )

    write_shard_cache(
        out_path, doc_ids=doc_ids, weights=weights, checksums=checksums, meta=meta
    )
    elapsed = time.time() - started
    summary = {key: value / n_rows for key, value in rollup.items()}
    logger.info(
        "%s/%s: wrote %d/%d docs in %.1fs (%.2f doc/s) w_mean=%.3f up=%.3f ceil=%.3f scored=%.3f",
        subset,
        path.stem,
        n_rows,
        array.shape[0],
        elapsed,
        n_rows / elapsed,
        summary["weight_mean"],
        summary["weight_upweighted_frac"],
        summary["weight_at_ceiling_frac"],
        summary["scored_frac"],
    )
    return {
        "subset": subset,
        "shard": path.stem,
        "n_docs": n_rows,
        "shard_rows": int(array.shape[0]),
        **summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenized", required=True, help="pretokenized corpus root")
    parser.add_argument("--out", required=True, help="cache root to write")
    parser.add_argument("--subsets", nargs="+", default=["2e16", "2e17"])
    parser.add_argument(
        "--seq-len",
        type=int,
        default=16384,
        help="cache width. Must be >= the longest stage that will read it; shorter stages read a "
        "prefix of the same row, which is why one cache serves the whole curriculum",
    )
    parser.add_argument(
        "--checksum-widths",
        type=int,
        nargs="+",
        default=None,
        help="widths to record token digests at (default: 8192 and --seq-len, i.e. the curriculum's "
        "stages). A stage whose seq_len is missing here cannot be verified and will be refused",
    )
    parser.add_argument("--trunc-len", type=int, default=DEFAULT_TRUNC_LEN, help="LongCE's K")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="LongCE's d")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="LongCE's clamp ceiling")
    parser.add_argument("--logit-chunk", type=int, default=4096)
    parser.add_argument(
        "--max-docs-per-shard",
        type=int,
        default=0,
        help="cap documents per shard (0 = all). A 600-step run at global batch 8 draws ~4800 "
        "documents total, so the whole 373K-document corpus does not need scoring",
    )
    parser.add_argument(
        "--mirror-loader",
        action="store_true",
        help="score exactly the (shard, row) pairs the training loader will draw, reproducing its "
        "shard and row shuffles. STRONGLY preferred: each reader walks its shards sequentially and "
        "shuffles rows within one, so a stored-order slice of every shard has a ~2.5%% hit rate "
        "(measured) and the run would silently be the plain objective. Needs --mirror-* to match the "
        "training invocation.",
    )
    parser.add_argument(
        "--mirror-world-size",
        type=int,
        default=1,
        help="the training run's DATA-parallel world size. For stage1_16k this is 1: FFN_SP=8 on 8 "
        "GPUs is one data-parallel replica, so dp_world_size=1 rather than 8.",
    )
    parser.add_argument(
        "--mirror-num-workers", type=int, default=2, help="the training run's --num-workers"
    )
    parser.add_argument(
        "--mirror-docs",
        type=int,
        default=6000,
        help="documents to plan per reader. A 600-step run at --global-batch-size 8 consumes ~4800 "
        "in total; the default pads that so a longer run or a shuffle-buffer boundary still hits.",
    )
    parser.add_argument(
        "--mirror-seq-lens",
        type=int,
        nargs="+",
        default=None,
        help="stages to plan for (default: 8192 and --seq-len). The loader's shard shuffle is seeded "
        "with args.seed + seq_len, so each stage draws a DIFFERENT order and every stage the run "
        "will reach has to be planned or that stage misses the cache entirely.",
    )
    parser.add_argument(
        "--rows-in-order",
        action="store_true",
        help="take rows in stored order instead of sampling (ignored under --mirror-loader). Only "
        "useful for reproducing an older cache; the corpus is ordered by document, so a stored "
        "prefix is not a random sample.",
    )
    parser.add_argument("--seed", type=int, default=0, help="the training run's --seed")
    parser.add_argument("--shard-index", type=int, default=0, help="this worker's slice")
    parser.add_argument("--shard-count", type=int, default=1, help="number of parallel workers")
    parser.add_argument("--overwrite", action="store_true", help="recompute existing shards")
    parser.add_argument("--log-every", type=int, default=25, help="documents between log lines")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn", default="flash_attention_2")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not torch.cuda.is_available():
        raise SystemExit("this needs a GPU: the backbone must run to produce the losses")
    if args.trunc_len >= args.seq_len:
        raise SystemExit(
            f"--trunc-len {args.trunc_len} leaves nothing to score at --seq-len {args.seq_len}"
        )

    if args.checksum_widths is None:
        # The curriculum's stages. 8192 is the first stage of
        # `8192:300,16384:300,32768:900`, so both it and the cache width must be verifiable.
        args.checksum_widths = sorted({8192, args.seq_len})
    args.checksum_widths = sorted(set(args.checksum_widths))
    over = [w for w in args.checksum_widths if w > args.seq_len]
    if over:
        raise SystemExit(
            f"--checksum-widths {over} exceed --seq-len {args.seq_len}; those tokens are not in "
            "the cache so no digest can be taken over them"
        )

    root = Path(args.tokenized)
    every = shard_paths(root, args.subsets)

    # Under --mirror-loader the plan decides both WHICH shards matter and which rows within them, so
    # it is built before the work is split: a shard no reader reaches is not worth scoring at all.
    plan: dict[tuple[str, str], list[int]] | None = None
    if args.mirror_loader:
        seq_lens = args.mirror_seq_lens or sorted({8192, args.seq_len})
        seq_lens = [s for s in sorted(set(seq_lens)) if s <= args.seq_len]
        plan = {}
        for stage_len in seq_lens:
            for key, rows in loader_draw_plan(
                root,
                args.subsets,
                seq_len=stage_len,
                seed=args.seed,
                world_size=args.mirror_world_size,
                num_workers=args.mirror_num_workers,
                docs_per_worker=args.mirror_docs,
            ).items():
                plan.setdefault(key, [])
                plan[key].extend(rows)
        plan = {key: sorted(set(rows)) for key, rows in plan.items()}
        planned_docs = sum(len(rows) for rows in plan.values())
        logger.info(
            "mirror-loader: %d shard(s), %d document(s) across stages %s "
            "(dp_world_size=%d, num_workers=%d)",
            len(plan),
            planned_docs,
            seq_lens,
            args.mirror_world_size,
            args.mirror_num_workers,
        )
        # Split by ROW rather than by shard. The plan concentrates in ~8 shards of very unequal size
        # (measured: 5119 rows against 704), so a shard-per-worker split would leave most GPUs idle
        # waiting on the largest one. Every worker takes a strided slice of each shard's rows and
        # writes its own file, which the cache reader unions by doc_id.
        every = [(subset, path) for subset, path in every if (subset, path.stem) in plan]
        plan = {
            key: rows[args.shard_index :: args.shard_count] for key, rows in plan.items()
        }
        mine = [
            (subset, path)
            for subset, path in every
            if plan[(subset, path.stem)]  # a worker with no rows for a shard skips it
        ]
        logger.info(
            "worker %d/%d: %d document(s) over %d shard(s)",
            args.shard_index,
            args.shard_count,
            sum(len(plan[(s, p.stem)]) for s, p in mine),
            len(mine),
        )
    else:
        mine = every[args.shard_index :: args.shard_count]
    logger.info(
        "worker %d/%d: %d of %d shards; cache -> %s",
        args.shard_index,
        args.shard_count,
        len(mine),
        len(every),
        args.out,
    )
    logger.info(
        "LongCE: K=%d d=%d gamma=%.1f seq_len=%d digests at %s",
        args.trunc_len,
        args.window,
        args.gamma,
        args.seq_len,
        args.checksum_widths,
    )

    meta = WeightCacheMeta(
        seq_len=args.seq_len,
        trunc_len=args.trunc_len,
        window=args.window,
        gamma=args.gamma,
        model=args.model,
        scored_from=args.trunc_len - 1,
        checksum_widths=tuple(args.checksum_widths),
    )

    model, _ = build_model(args.model, getattr(torch, args.dtype), args.attn, args.device)

    written = []
    for subset, path in mine:
        rows = None if plan is None else plan[(subset, path.stem)]
        result = process_shard(model, subset, path, args=args, meta=meta, rows=rows)
        if result is not None:
            written.append(result)

    logger.info("worker %d: wrote %d shard(s)", args.shard_index, len(written))
    if written:
        n = sum(r["n_docs"] for r in written)
        logger.info(
            "totals: %d documents, w_mean=%.3f upweighted=%.3f",
            n,
            sum(r["weight_mean"] * r["n_docs"] for r in written) / n,
            sum(r["weight_upweighted_frac"] * r["n_docs"] for r in written) / n,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
