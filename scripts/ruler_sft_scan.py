# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Scan the RULER SFT corpus and report what a given ``--sft-max-len`` would keep.

Run this **before** a training run. The length filter is heavily task-skewed -- at a 16384 cap on
the ``16384`` config, ``cwe`` (median 22.4K prompt tokens) is lost outright while ``fwe`` (15.1K)
mostly survives -- so without this table a task simply goes missing from the training mix and the
only evidence is its absence.

    python -m scripts.ruler_sft_scan --ruler SNAPSHOT --config 16384 --max-len 24576
    python -m scripts.ruler_sft_scan --ruler SNAPSHOT --config 16384 --sweep

Costs one tokenizer pass over the rows examined (``--max-rows`` to cut it short). Nothing is
written; this only prints.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kvpress.presses.gqa_indexer.sft_data import (  # noqa: E402
    ALL_TASKS,
    TASK_GROUPS,
    RulerSFTConfig,
    RulerSFTDataset,
    build_prompt_ids,
    build_target_text,
    resolve_tasks,
)

logger = logging.getLogger("ruler_sft_scan")


def measure(dataset: RulerSFTDataset) -> dict[str, list[int]]:
    """
    Tokenize every row once and return ``task -> [prompt lengths]`` plus target lengths.

    Deliberately does **not** go through ``__iter__``: that applies the length filter, and the
    point here is to see the distribution the filter is cutting into, including the rows it drops.
    """
    per_task: dict[str, list[tuple[int, int]]] = {}
    for row in dataset.rows:
        task = str(row["task"])
        prompt_ids = build_prompt_ids(
            dataset.tokenizer,
            context=str(row["context"]),
            question=str(row["question"]),
            answer_prefix=str(row["answer_prefix"]),
        )
        target_text = build_target_text(task, row["answer"], str(row["answer_prefix"]))
        target_len = len(dataset.tokenizer.encode(target_text, add_special_tokens=False))
        per_task.setdefault(task, []).append((int(prompt_ids.numel()), target_len))
    return per_task


def percentile(values: list[int], fraction: float) -> int:
    """Nearest-rank percentile; no numpy dependency for a table this small."""
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--ruler", required=True, help="parquet file, snapshot dir, or repo id")
    parser.add_argument("--config", default=None, help="context-length config, e.g. 16384")
    parser.add_argument("--model", default="/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B")
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help=f"tasks or a group from {tuple(TASK_GROUPS)} (default: all)",
    )
    parser.add_argument("--max-len", type=int, default=24576, help="the cap to evaluate")
    parser.add_argument(
        "--sweep", action="store_true",
        help="report keep rates at several caps instead of one, to pick --sft-max-len",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="examine only this many rows")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = RulerSFTConfig(
        source=args.ruler,
        config=args.config,
        tasks=tuple(resolve_tasks(args.tasks)),
        max_len=args.max_len,
        max_rows=args.max_rows,
    )
    dataset = RulerSFTDataset(config, tokenizer)
    logger.info("examining %d rows from %s (config=%s)", len(dataset.rows), args.ruler, args.config)

    per_task = measure(dataset)
    caps = (
        [8192, 16384, 20480, 24576, 32768] if args.sweep else [args.max_len]
    )

    print(f"\nprompt+answer token lengths, tokenizer={Path(args.model).name}\n")
    header = f"{'task':<16} {'rows':>5} {'p50':>7} {'p95':>7} {'max':>7} {'tgt_p50':>7}"
    header += "".join(f"{'keep@' + str(c // 1024) + 'K':>11}" for c in caps)
    print(header)
    print("-" * len(header))

    totals = {cap: [0, 0] for cap in caps}
    for task in ALL_TASKS:
        if task not in per_task:
            continue
        pairs = per_task[task]
        prompts = [p for p, _ in pairs]
        targets = [t for _, t in pairs]
        totals_row = [p + t for p, t in pairs]
        line = (
            f"{task:<16} {len(pairs):>5} {percentile(prompts, 0.5):>7} "
            f"{percentile(prompts, 0.95):>7} {max(prompts):>7} {percentile(targets, 0.5):>7}"
        )
        for cap in caps:
            kept = sum(1 for total in totals_row if total <= cap)
            totals[cap][0] += kept
            totals[cap][1] += len(totals_row)
            line += f"{100 * kept / len(totals_row):>10.0f}%"
        print(line)

    print("-" * len(header))
    summary = f"{'TOTAL':<16} {sum(len(v) for v in per_task.values()):>5} {'':>7} {'':>7} {'':>7} {'':>7}"
    for cap in caps:
        kept, total = totals[cap]
        summary += f"{100 * kept / max(total, 1):>10.0f}%"
    print(summary)

    if not args.sweep:
        cap = args.max_len
        dropped = [
            task for task in per_task
            if not any(p + t <= cap for p, t in per_task[task])
        ]
        if dropped:
            print(
                f"\nLOST ENTIRELY at --sft-max-len {cap}: {sorted(dropped)}\n"
                "  Those tasks contribute no training rows. Raise the cap, or drop them from "
                "--sft-tasks so the intent is recorded in the run's config."
            )
        else:
            print(f"\nEvery selected task keeps at least one row at --sft-max-len {cap}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
