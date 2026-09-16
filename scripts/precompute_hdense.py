# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Precompute the dense final hidden states that :mod:`~kvpress.presses.gqa_indexer.fwkl` distills
against.

    # one GPU per shard-slice; 8 workers cover a 300-step run's draw in parallel
    for i in $(seq 0 7); do
      python -m scripts.precompute_hdense --model $MODEL --tokenized $TOK \\
        --out $OUT --seq-len 8192 --seed 1000 --shard-index $i --shard-count 8 \\
        --device cuda:$i &
    done; wait

Why cache this rather than run the teacher inline
-------------------------------------------------
For a *single* run it is a wall-clock wash -- one extra forward per step is ~+27% at 8K (~79 min
over 300 steps), and precomputing the same 2400 documents across 8 GPUs also costs ~79 min. It pays
from the second run onward: the teacher depends only on the frozen backbone and the token ids, so
every sweep over the KL weight, the gate budget or the router architecture reuses it.

Storage is the reason this is `h` and not logits: ``(L, 4096)`` fp16 is 64 MB per 8K document
against 2.3 GB of bf16 logits. A 300-step run's 2400 documents come to ~150 GB at 8K.

WHAT MUST MATCH THE TRAINING RUN
--------------------------------
``--seq-len``, ``--seed``, ``--subsets`` and the loader geometry (``--world-size``,
``--num-workers``, ``--global-batch-size``, ``--steps``). The plan mirrors
:class:`~kvpress.presses.gqa_indexer.data.TokenizedDataset`'s own draw order, and the reason is
measured: caching stored-order prefixes of every shard instead gives a **2.5% hit rate**, because
each ``(rank, worker)`` reader exhausts the first shards of its own shuffled list rather than
sampling all of them. See ``loader_draw_plan``, reused verbatim from
``scripts/precompute_longce_weights.py`` so the two caches cannot drift apart.

``--take-from head`` is assumed, matching the LongCE-trained router's regime. A random window would
not be a prefix of what was cached.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kvpress.presses.gqa_indexer.hdense_cache import (  # noqa: E402
    HDenseMeta,
    shard_cache_path,
    token_digest,
    write_shard_cache,
)

# Reused rather than reimplemented: this function is what takes the cache hit rate from 2.5% to
# ~100%, and a second copy of that logic would be a silent divergence the moment either changed.
from scripts.precompute_longce_weights import loader_draw_plan  # noqa: E402

logger = logging.getLogger("precompute_hdense")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenized", required=True, help="pretokenized corpus root")
    parser.add_argument("--out", required=True, help="cache root to write")
    parser.add_argument("--subsets", nargs="+", default=["2e16", "2e17"])
    parser.add_argument(
        "--seq-len", type=int, required=True,
        help="the training stage's sequence length. The cache is a prefix store, so a run at a "
        "SHORTER length can read this one but a longer one cannot.",
    )
    parser.add_argument(
        "--seed", type=int, required=True,
        help="the training run's --seed. loader_for derives its stream from seed + seq_len, so a "
        "different seed draws entirely different documents and this cache would miss.",
    )
    # The loader geometry the plan mirrors. Defaults match the joint/scalar scripts.
    parser.add_argument("--world-size", type=int, default=1,
                        help="data-parallel replicas the training run will have (FFN_SP=NGPU -> 1)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--pad-frac", type=float, default=0.25,
        help="extra documents per worker beyond the plan, absorbing shuffle_buffer's boundary and "
        "any restart. Cheap insurance: a miss falls back to an inline teacher forward.",
    )

    parser.add_argument("--shard-index", type=int, default=0, help="this worker's slice")
    parser.add_argument("--shard-count", type=int, default=1, help="number of parallel workers")
    parser.add_argument("--overwrite", action="store_true", help="recompute existing shards")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn", default="flash_attention_2")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    root = Path(args.tokenized)
    out = Path(args.out)

    # Documents each (rank, worker) reader will draw, padded.
    readers = max(args.world_size, 1) * max(args.num_workers, 1)
    total_docs = args.steps * args.global_batch_size
    per_worker = int(total_docs / readers * (1.0 + args.pad_frac)) + 1
    plan = loader_draw_plan(
        root=root,
        subsets=args.subsets,
        seq_len=args.seq_len,
        seed=args.seed,
        world_size=max(args.world_size, 1),
        num_workers=max(args.num_workers, 1),
        docs_per_worker=per_worker,
    )
    planned = sum(len(rows) for rows in plan.values())
    logger.info(
        "plan: %d documents across %d shard(s) (%d readers x %d each, incl. %.0f%% pad); "
        "the run itself draws %d",
        planned, len(plan), readers, per_worker, 100 * args.pad_frac, total_docs,
    )

    # This worker's slice of the planned shards, in a stable order.
    keys = sorted(plan)
    mine = keys[args.shard_index :: args.shard_count]
    if not mine:
        logger.info("shard-index %d has nothing to do", args.shard_index)
        return 0

    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, args.dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, attn_implementation=args.attn
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, attn_implementation=args.attn
        )
    model = model.to(args.device).eval()
    model.requires_grad_(False)
    hidden_size = int(model.config.hidden_size)
    meta = HDenseMeta(seq_len=args.seq_len, hidden_size=hidden_size, model=args.model)

    started = time.time()
    done_docs = 0
    for subset, stem in mine:
        path = shard_cache_path(out, subset, stem)
        if path.exists() and not args.overwrite:
            logger.info("skip %s/%s (exists)", subset, stem)
            continue
        rows = plan[(subset, stem)]
        source = root / subset / f"{stem}.npy"
        array = np.load(source, mmap_mode="r")
        sidecar = source.with_suffix(".json")
        doc_ids_all = []
        if sidecar.is_file():
            import json as _json

            with open(sidecar) as handle:
                doc_ids_all = _json.load(handle).get("doc_ids", [])

        hidden = np.empty((len(rows), args.seq_len, hidden_size), dtype=np.float16)
        ids, digests = [], []
        for index, row in enumerate(rows):
            tokens = np.array(array[row, : args.seq_len], dtype=np.int64)
            input_ids = torch.from_numpy(tokens).unsqueeze(0).to(args.device)
            with torch.no_grad():
                # output_hidden_states gives every layer; we want only the final one, which is
                # what lm_head consumes. Asking the base model directly avoids allocating the
                # (L, vocab) logits we are specifically trying not to store.
                out_h = model.model(input_ids=input_ids).last_hidden_state
            hidden[index] = out_h[0].float().cpu().numpy().astype(np.float16)
            ids.append(doc_ids_all[row] if row < len(doc_ids_all) else f"{stem}:{row}")
            digests.append(token_digest(tokens))
            done_docs += 1
            if done_docs % args.log_every == 0:
                rate = done_docs / max(time.time() - started, 1e-9)
                logger.info(
                    "%d docs at %.2f doc/s (%s/%s, %d/%d rows)",
                    done_docs, rate, subset, stem, index + 1, len(rows),
                )

        write_shard_cache(path, hidden=hidden, doc_ids=ids, digests=digests, meta=meta)
        logger.info(
            "wrote %s (%d docs, %.1f GB)", path, len(rows), hidden.nbytes / 1024**3
        )

    logger.info("done: %d documents in %.1f min", done_docs, (time.time() - started) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
