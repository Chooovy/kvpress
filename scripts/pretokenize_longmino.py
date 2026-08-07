# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pre-tokenize the longmino shards once so training never pays for tokenization.

    python -m scripts.pretokenize_longmino \\
        --data-root /path/to/longmino_256k_filtered \\
        --out /path/to/longmino_tokenized --seq-len 65536

Tokenizing during training is pure serial overhead on the data path: measured ~0.45 s per
64K-token sample, against a step that is itself seconds long -- so with too few workers it
throttles the GPU, and with enough workers it burns CPU that the dataloader could spend
prefetching. Doing it once is strictly better, and it also makes the corpus *reproducible*:
a run reads a fixed set of token arrays rather than re-deriving them from text.

Output layout
-------------
One directory per subset, one ``.npy`` per input shard plus a JSON index::

    out/
      index.json                  seq_len, tokenizer, per-shard doc counts
      2e16/
        longmino_2e16-0000.npy    uint32 (num_docs, seq_len)
        longmino_2e16-0000.json   doc ids + available_tokens, aligned by row
      ...

``uint32`` because Qwen3's vocabulary is 151936, which does not fit in ``uint16``. A flat
rectangular array is what makes the training-time loader trivial: ``np.load(mmap_mode="r")``
then index a row, with no parsing and no decompression.

**Store the longest length you will train on, then slice.** ``seq_len`` is baked into the
array width, so a 32K file cannot serve a 64K stage -- but a 64K file serves both, since
taking a prefix of a token array is exactly what a shorter ``seq_len`` means. That is why the
default is 65536 and why :func:`~kvpress.presses.gqa_indexer.data.TokenizedConfig` refuses a
``seq_len`` above what was stored instead of silently padding.

Only documents that reach ``seq_len`` tokens are written. A sample must be one document (see
:mod:`kvpress.presses.gqa_indexer.data`), so a short document has no use here, and dropping it
at pretokenize time means the training loader never has to check.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kvpress.presses.gqa_indexer.data import (  # noqa: E402
    CHARS_PER_TOKEN,
    SUBSETS,
    estimated_tokens,
    shard_paths,
)

logger = logging.getLogger("pretokenize")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# uint32 holds any Qwen3 id (vocab 151936); uint16 would silently wrap at 65535.
TOKEN_DTYPE = np.uint32


def tokenize_shard(
    shard: Path,
    out_dir: Path,
    model: str,
    seq_len: int,
    min_tokens: int,
    limit: int | None,
    overwrite: bool,
) -> dict:
    """
    Tokenize one shard into ``(num_docs, seq_len)`` uint32 plus a row-aligned JSON sidecar.

    Runs in a worker process: the tokenizer is loaded here rather than passed in, because a
    fast tokenizer does not pickle cheaply and each worker needs its own anyway.

    Writes to a ``.tmp`` path and renames on success, so an interrupted run leaves no
    half-written array that a later run would mistake for complete.
    """
    from transformers import AutoTokenizer

    target = out_dir / f"{shard.name.split('.')[0]}.npy"
    sidecar = target.with_suffix(".json")
    if target.exists() and sidecar.exists() and not overwrite:
        with open(sidecar) as handle:
            meta = json.load(handle)
        return {"shard": shard.name, "docs": meta["num_docs"], "skipped": True}

    tokenizer = AutoTokenizer.from_pretrained(model)
    budget = int(seq_len * CHARS_PER_TOKEN)

    rows: list[np.ndarray] = []
    doc_ids: list[str] = []
    available: list[int] = []
    stats = {"seen": 0, "too_short_meta": 0, "too_short_tokens": 0, "bad_lines": 0, "retries": 0}

    with gzip.open(shard, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stats["seen"] += 1
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                stats["bad_lines"] += 1
                continue
            text = row.get("text")
            if not isinstance(text, str) or not text:
                stats["bad_lines"] += 1
                continue

            # Metadata prefilter: skips the tokenizer for documents that cannot qualify.
            estimate = estimated_tokens(row)
            if estimate is not None and estimate < min_tokens:
                stats["too_short_meta"] += 1
                continue
            if estimate is None and len(text) < min_tokens * CHARS_PER_TOKEN:
                stats["too_short_meta"] += 1
                continue

            ids = tokenizer(text[:budget], add_special_tokens=False)["input_ids"]
            if len(ids) < seq_len:
                # CHARS_PER_TOKEN is an upper bound, so this is rare; retry on the full text
                # rather than dropping a document the metadata said was long enough.
                stats["retries"] += 1
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                if len(ids) < seq_len:
                    stats["too_short_tokens"] += 1
                    continue

            rows.append(np.asarray(ids[:seq_len], dtype=TOKEN_DTYPE))
            doc_ids.append(str(row.get("id", "")))
            available.append(len(ids))
            if limit is not None and len(rows) >= limit:
                break

    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        array = np.stack(rows)
    else:
        # An empty shard still gets a file, so a rerun does not retry it forever and the
        # index records the zero honestly instead of the shard looking unprocessed.
        array = np.zeros((0, seq_len), dtype=TOKEN_DTYPE)

    # np.save appends ".npy" unless the path already ends in it, so a ".npy.tmp" name would
    # be written as ".npy.tmp.npy" and the rename below would miss. Use a name that already
    # ends in .npy and carry the "temporary" marker in the stem instead.
    tmp = target.with_name(target.stem + ".partial.npy")
    np.save(tmp, array)
    tmp.replace(target)
    tmp_sidecar = sidecar.with_name(sidecar.stem + ".partial.json")
    with open(tmp_sidecar, "w") as handle:
        json.dump(
            {
                "shard": shard.name,
                "num_docs": int(array.shape[0]),
                "seq_len": seq_len,
                "doc_ids": doc_ids,
                "available_tokens": available,
                "stats": stats,
            },
            handle,
        )
    tmp_sidecar.replace(sidecar)

    return {"shard": shard.name, "docs": int(array.shape[0]), "stats": stats, "skipped": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--model",
        default="/apdcephfs_zw31/share_303843174/user/marcushaogu/models/Qwen3-8B",
        help="tokenizer to use; must match the model you will train against",
    )
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS), choices=list(SUBSETS))
    parser.add_argument(
        "--seq-len",
        type=int,
        default=65536,
        help="tokens stored per document. Store the LONGEST length you will train on: a "
        "shorter stage slices a prefix, but a longer one cannot be served at all.",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=None,
        help="metadata threshold for admitting a document; defaults to --seq-len",
    )
    parser.add_argument("--workers", type=int, default=8, help="shards tokenized in parallel")
    parser.add_argument("--limit-per-shard", type=int, default=None, help="cap docs, for testing")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    min_tokens = args.min_tokens or args.seq_len
    if min_tokens < args.seq_len:
        parser.error(f"--min-tokens {min_tokens} is below --seq-len {args.seq_len}")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    shards = shard_paths(args.data_root, tuple(args.subsets))
    logger.info(
        "%d shards -> %s at seq_len=%d (%.2f MB per document)",
        len(shards),
        out_root,
        args.seq_len,
        args.seq_len * 4 / 1e6,
    )

    started = time.time()
    per_subset: dict[str, int] = {}
    futures = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for shard in shards:
            subset = shard.parent.name
            futures[
                pool.submit(
                    tokenize_shard,
                    shard,
                    out_root / subset,
                    args.model,
                    args.seq_len,
                    min_tokens,
                    args.limit_per_shard,
                    args.overwrite,
                )
            ] = subset

        done = 0
        failed: list[str] = []
        for future in as_completed(futures):
            subset = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # a bad shard must not lose the other 99
                logger.error("subset %s shard failed: %s: %s", subset, type(exc).__name__, exc)
                failed.append(subset)
                continue
            done += 1
            per_subset[subset] = per_subset.get(subset, 0) + result["docs"]
            note = " (cached)" if result.get("skipped") else ""
            logger.info(
                "[%d/%d] %s: %d docs%s  %s",
                done,
                len(shards),
                result["shard"],
                result["docs"],
                note,
                result.get("stats", ""),
            )

    total = sum(per_subset.values())
    if failed:
        # Distinguish "shards crashed" from "nothing qualified": both end with a small or
        # zero document count, but only the first means the output is incomplete and a rerun
        # would change it.
        logger.error(
            "%d of %d shards failed (%s); the index below is INCOMPLETE -- fix the cause and "
            "rerun (finished shards are cached, so the rerun is cheap)",
            len(failed),
            len(shards),
            ", ".join(sorted(set(failed))),
        )
    index = {
        "seq_len": args.seq_len,
        "min_tokens": min_tokens,
        "model": args.model,
        "dtype": "uint32",
        "subsets": per_subset,
        "total_docs": total,
        "complete": not failed,
    }
    with open(out_root / "index.json", "w") as handle:
        json.dump(index, handle, indent=2)

    logger.info("wrote %d documents in %.1f min", total, (time.time() - started) / 60)
    for subset, count in sorted(per_subset.items()):
        logger.info("  %-10s %7d docs  %6.1f GB", subset, count,
                    count * args.seq_len * 4 / 1e9)
    if total == 0:
        logger.error(
            "no documents qualified at seq_len=%d; every subset's documents are shorter",
            args.seq_len,
        )
        return 1
    logger.info("index: %s", out_root / "index.json")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
