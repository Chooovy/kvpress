# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
RULER supervised fine-tuning data for the indexer: loss on the **answer only**.

The longmino loader (:mod:`kvpress.presses.gqa_indexer.data`) trains the router against a plain
next-token loss over a long document, so every position contributes. This one supervises a
*task*: the prompt (chat template + context + question + answer_prefix) is masked out with
``-100`` and only the gold answer's tokens carry loss. The router's gradient therefore says one
thing -- "make the answer predictable" -- and the only way it can comply is to route attention to
the keys the answer depends on.

What this arm can and cannot teach
----------------------------------
``E2EIndexerTrainer.freeze_backbone`` leaves **only the indexer trainable**, so this is not an SFT
in the usual sense: it cannot teach the model an answer format, and it cannot memorize an answer
into a weight. The one thing gradient descent can do here is move the router. That is a feature --
a frozen backbone means a gain on RULER cannot come from having learned the answers -- but it also
means the loss value itself is close to uninformative. **Judge this stage on the RULER metric, not
on the loss curve.**

Why the loss is so sparse, measured
-----------------------------------
Against the Qwen3 tokenizer on the ``16384`` config, a prompt is 13.6K-22.4K tokens while the gold
answer is 3-45 (median 8-35 by task). So **~0.1% of positions carry gradient**. Two consequences
that are easy to misread as bugs:

* The reported loss is not comparable to any longmino-trained number, and it is noisy -- a step's
  loss is a mean over ~8 x 20 tokens, not ~8 x 16384.
* Gradient accumulation averages ``loss / accum_steps``, which weights each *sample* equally
  rather than each *answer token*. A 3-token ``qa_2`` answer therefore pulls as hard as a 35-token
  ``niah_multivalue`` one. Left as is deliberately: the alternative (weighting by token count)
  makes the multi-answer tasks dominate, and neither choice is obviously right.

The prompt is not re-derived here
---------------------------------
:func:`build_prompt_ids` calls :meth:`KVPressTextGenerationPipeline.preprocess` -- the *same
unbound method* the eval runs -- rather than reassembling the chat template. That method touches
nothing but ``self.tokenizer``, so a shim carrying only a tokenizer satisfies it, and
``tests/presses/test_gqa_indexer_sft_data.py`` pins the two against each other token for token. A
reimplementation would be one ``enable_thinking`` default or one separator away from training the
router on a prompt shape it never sees at eval, and nothing downstream would flag it.

Context is dropped, never truncated
-----------------------------------
A row whose prompt+answer exceeds ``max_len`` is **skipped**, because truncating it would change
the ground truth rather than shorten the input: a needle lives at a fixed depth and may be in the
cut, and ``cwe``/``fwe`` answers are word *counts* over the whole list. The skip is heavily
task-skewed at any given cap -- at 16384 on the ``16384`` config, ``cwe`` (median 22.4K) is lost
outright while ``fwe`` (15.1K) mostly survives -- so :meth:`RulerSFTDataset.format_stats` reports
the per-task table and ``scripts/ruler_sft_scan.py`` prints it *before* a training run rather than
leaving it to be inferred from a task's absence.
"""

from __future__ import annotations

import glob
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

logger = logging.getLogger(__name__)

#: The 13 RULER tasks, grouped the way ``benchmarks/ruler/calculate_metrics.py`` groups them.
#: ``niah`` is the needle family; the rest are one task each.
NIAH_TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
)
OTHER_TASKS = ("vt", "cwe", "fwe", "qa_1", "qa_2")
ALL_TASKS = NIAH_TASKS + OTHER_TASKS

#: Task-group shorthands accepted by ``--sft-tasks``.
TASK_GROUPS = {"all": ALL_TASKS, "niah": NIAH_TASKS, "other": OTHER_TASKS}


def resolve_tasks(selectors: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    """
    Expand ``--sft-tasks`` selectors into an explicit task tuple.

    Accepts group names (``all``, ``niah``, ``other``) and individual task names, so
    ``--sft-tasks niah qa_1`` is meaningful. Order follows :data:`ALL_TASKS` and duplicates
    collapse, which keeps the recorded configuration stable regardless of how it was spelled.
    """
    if not selectors:
        return ALL_TASKS
    picked: set[str] = set()
    for selector in selectors:
        if selector in TASK_GROUPS:
            picked.update(TASK_GROUPS[selector])
        elif selector in ALL_TASKS:
            picked.add(selector)
        else:
            raise ValueError(
                f"unknown --sft-tasks entry {selector!r}; expected a task from {ALL_TASKS} "
                f"or a group from {tuple(TASK_GROUPS)}"
            )
    return tuple(task for task in ALL_TASKS if task in picked)


def load_ruler_frame(source: str | Path, config: str | None = None):
    """
    Read one RULER context-length config into a DataFrame.

    ``source`` may be a parquet file, a directory laid out like the HuggingFace snapshot
    (``<source>/<config>/test-*.parquet``), or a repo id to hand to ``datasets.load_dataset``.

    Parquet is read directly when the files are on disk, rather than going through
    ``load_dataset(repo, data_dir=...)``. That call hashes ``data_dir`` into a cache key computed
    from the *remote* builder, so with no network it raises ``Couldn't find cache`` and lists only
    opaque hashes -- the problem ``evaluate_sparse.load_cached_dataset`` works around by *measuring*
    context lengths. Reading the parquet path directly sidesteps the guessing entirely, and the
    config name is then known rather than inferred.
    """
    import pandas as pd

    path = Path(source)
    if path.is_file():
        return pd.read_parquet(path)
    if path.is_dir():
        pattern = str(path / config / "*.parquet") if config else str(path / "*.parquet")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"no parquet under {pattern}. Point --sft-ruler at the directory holding "
                f"<config>/test-*.parquet (the HuggingFace snapshot layout), or at one file."
            )
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    from datasets import load_dataset

    return load_dataset(str(source), data_dir=config, split="test").to_pandas()


class _PreprocessShim:
    """
    Carries a tokenizer so the eval's ``preprocess`` can be called without building a pipeline.

    :meth:`KVPressTextGenerationPipeline.preprocess` reads ``self.tokenizer`` and nothing else --
    no model, no device, no framework attribute -- so this is enough to reuse it verbatim instead
    of reimplementing the chat template. Pinned by
    ``test_prompt_matches_pipeline_preprocess``: if a future version of that method reaches for
    another attribute, the test fails rather than this silently diverging from what eval feeds the
    router.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


def build_prompt_ids(
    tokenizer,
    context: str,
    question: str,
    answer_prefix: str,
    *,
    enable_thinking: bool = False,
) -> torch.Tensor:
    """
    The eval's exact prompt for one row, as a 1-D ``int64`` tensor.

    Layout, from the reused ``preprocess``::

        <template prefix> context question <template suffix> answer_prefix

    ``max_context_length`` is deliberately passed as ``10**9``: truncation is this module's
    decision to make (a row is skipped whole, see the module docstring), and letting ``preprocess``
    silently clip the context would produce a sample whose gold answer no longer follows from its
    input.
    """
    from kvpress.pipeline import KVPressTextGenerationPipeline

    out = KVPressTextGenerationPipeline.preprocess(
        _PreprocessShim(tokenizer),
        context=context,
        questions=[question],
        answer_prefix=answer_prefix,
        max_context_length=10**9,
        enable_thinking=enable_thinking,
    )
    # context_ids is (1, Lc); questions_ids is a list of (1, Lq) -- one entry, since one question.
    return torch.cat([out["context_ids"][0], out["questions_ids"][0][0]]).to(torch.long)


def build_target_text(task: str, answers, answer_prefix: str) -> str:
    """
    The gold continuation of ``answer_prefix``, joined the way the task is *scored*.

    ``benchmarks/ruler/calculate_metrics.py`` splits on the task category: ``qa`` is graded by
    ``string_match_part`` (credit if any one reference appears), everything else by
    ``string_match_all`` (credit per reference found, averaged). So:

    * ``qa_*`` -> the first reference only. Its ``answer`` field repeats the same string once per
      supporting document -- ``['France', 'France', 'France', 'France']`` -- and joining that would
      train the router toward a stuttered output that the metric never asks for.
    * everything else -> all references, comma-joined, order preserved, duplicates dropped.

    A space is prepended only when ``answer_prefix`` does not already end in whitespace: ``vt``'s
    prefix ends ``"they are: "`` while ``cwe``'s ends ``"are:"``, so a blind ``" " + text`` would
    put a double space in one of them and tokenize to something the eval never produces.
    """
    references = [str(a) for a in list(answers)]
    if not references:
        raise ValueError(f"row for task {task!r} has no gold answer")
    if task.split("_")[0] == "qa":
        picked = references[:1]
    else:
        seen: set[str] = set()
        picked = [a for a in references if not (a in seen or seen.add(a))]
    text = ", ".join(picked)
    return text if answer_prefix.endswith((" ", "\t", "\n")) else " " + text


@dataclass
class RulerSFTConfig:
    """
    Configuration for :class:`RulerSFTDataset`.

    Attributes
    ----------
    source : str
        Parquet file, snapshot directory, or HF repo id -- see :func:`load_ruler_frame`.
    config : str, optional
        Context-length config to read (``"16384"``). Required for a directory or repo id.
    tasks : tuple of str
        Which RULER tasks to train on, already expanded by :func:`resolve_tasks`.
    max_len : int
        Skip any row whose prompt+answer exceeds this. Not a truncation bound -- see the module
        docstring for why a long row is dropped instead.
    append_eos : bool
        Append the tokenizer's EOS to the target. **Off by default.** The backbone is frozen, so
        its stopping behaviour is already fixed by pretraining and there is nothing here to teach
        it; what an EOS *can* do is give the router an incentive to make stopping likely, and
        ``generate_answer`` halts on EOS. On the ``string_match_all`` tasks -- where the gold answer
        is several comma-separated references -- an early stop drops references and costs score.
    seed, shuffle : int, bool
        Row order. Shuffled by default; the seed is combined with rank and worker id.
    max_rows : int, optional
        Cap on rows *examined* (before the length filter), for smoke runs.
    """

    source: str
    config: str | None = None
    tasks: tuple[str, ...] = ALL_TASKS
    max_len: int = 16384
    append_eos: bool = False
    seed: int = 0
    shuffle: bool = True
    max_rows: int | None = None

    def __post_init__(self):
        if self.max_len <= 0:
            raise ValueError(f"max_len must be positive, got {self.max_len}")
        self.tasks = resolve_tasks(self.tasks)


class RulerSFTDataset(IterableDataset):
    """
    Streams ``(input_ids, labels)`` pairs with the prompt masked to ``-100``.

    Iterable rather than map-style for one reason: whether a row fits under ``max_len`` is only
    known *after* tokenizing it, and a map-style dataset has no way to answer ``__getitem__`` with
    "skip this one". Filtering by a character estimate first would need an upper bound on
    chars-per-token and would still admit misses (the same problem
    :data:`~kvpress.presses.gqa_indexer.data.CHARS_PER_TOKEN` documents), so the length is verified
    against the real tokenizer and an over-long row is simply not yielded.

    Sharding is by **data-parallel** rank, matching the longmino loaders: under FFN sequence
    parallelism the ranks of one SP group cooperate on a single sequence and must read the *same*
    row, so they share a ``dp_rank`` and therefore a stream. Getting this wrong is silent -- each
    rank would tokenize a different document and the all-gather would stitch fragments together.
    """

    def __init__(
        self,
        config: RulerSFTConfig,
        tokenizer,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.rank = rank
        self.world_size = world_size

        frame = load_ruler_frame(config.source, config.config)
        required = {"context", "question", "answer_prefix", "answer", "task"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"RULER frame is missing {sorted(missing)}; got {sorted(frame.columns)}. "
                "Expected the simonjegou/ruler schema (see benchmarks/ruler/README.md)."
            )
        known = set(ALL_TASKS)
        unexpected = sorted(set(frame["task"].unique()) - known)
        if unexpected:
            logger.warning("ignoring unrecognized RULER task(s): %s", unexpected)
        frame = frame[frame["task"].isin(config.tasks)]
        if frame.empty:
            raise ValueError(
                f"no rows for tasks {config.tasks} in {config.source} "
                f"(config={config.config!r})"
            )
        self.rows = frame.to_dict("records")
        if config.max_rows is not None:
            self.rows = self.rows[: config.max_rows]

        #: task -> [kept, dropped]. Populated as rows are consumed, so it reflects one pass.
        self.stats: dict[str, list[int]] = {}

    def assigned_rows(self) -> list[dict]:
        """This ``(rank, worker)``'s rows, in a seeded order shared across an SP group."""
        rows = list(self.rows)
        if self.config.shuffle:
            random.Random(self.config.seed).shuffle(rows)
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0
        return rows[self.rank * num_workers + worker_id :: self.world_size * num_workers]

    def encode(self, row: dict) -> dict | None:
        """
        One row as ``(input_ids, labels)``, or ``None`` when it exceeds ``max_len``.

        Labels are ``-100`` over the prompt and the true token ids over the target, so the model's
        own shift-by-one puts the first supervised prediction on the token that *follows*
        ``answer_prefix`` -- exactly the position ``generate_answer`` samples first at eval.
        """
        task = str(row["task"])
        prompt_ids = build_prompt_ids(
            self.tokenizer,
            context=str(row["context"]),
            question=str(row["question"]),
            answer_prefix=str(row["answer_prefix"]),
        )
        target_text = build_target_text(task, row["answer"], str(row["answer_prefix"]))
        target_ids = self.tokenizer.encode(
            target_text, return_tensors="pt", add_special_tokens=False
        )[0].to(torch.long)
        if self.config.append_eos and self.tokenizer.eos_token_id is not None:
            target_ids = torch.cat(
                [target_ids, torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)]
            )

        total = int(prompt_ids.numel() + target_ids.numel())
        counts = self.stats.setdefault(task, [0, 0])
        if total > self.config.max_len:
            counts[1] += 1
            return None
        counts[0] += 1

        input_ids = torch.cat([prompt_ids, target_ids])
        labels = torch.full_like(input_ids, -100)
        labels[prompt_ids.numel() :] = target_ids
        return {
            "input_ids": input_ids,
            "labels": labels,
            "task": task,
            "n_target": int(target_ids.numel()),
            "n_prompt": int(prompt_ids.numel()),
        }

    def __iter__(self) -> Iterator[dict]:
        for row in self.assigned_rows():
            sample = self.encode(row)
            if sample is not None:
                yield sample

    def format_stats(self) -> str:
        """The per-task kept/dropped table, for logging after a pass."""
        lines = ["task              kept  dropped  keep%"]
        kept_total = dropped_total = 0
        for task in sorted(self.stats):
            kept, dropped = self.stats[task]
            kept_total += kept
            dropped_total += dropped
            total = kept + dropped
            lines.append(
                f"{task:<16} {kept:>5} {dropped:>8}  {100 * kept / max(total, 1):5.1f}"
            )
        total = kept_total + dropped_total
        lines.append(
            f"{'TOTAL':<16} {kept_total:>5} {dropped_total:>8}  "
            f"{100 * kept_total / max(total, 1):5.1f}"
        )
        return "\n".join(lines)


def sft_collate(batch: list[dict]) -> dict:
    """
    Stack a batch of equal-length samples; refuses a ragged one.

    RULER prompts vary in length by thousands of tokens, so a batch larger than 1 would need
    padding *and* an ``attention_mask`` threaded into the gate's selector and normalizer. Rather
    than pad silently -- which feeds the router masked positions and changes what its softmax
    normalizes over -- this asserts, and the trainer reaches its global batch through gradient
    accumulation at ``--batch-size 1``.
    """
    lengths = {int(item["input_ids"].shape[0]) for item in batch}
    if len(lengths) != 1:
        raise ValueError(
            f"ragged SFT batch: lengths {sorted(lengths)}. RULER prompts vary by thousands of "
            "tokens, so use --batch-size 1 and reach the global batch through --accum-steps "
            "(padding would need an attention_mask on the gate path)."
        )
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "tasks": [item["task"] for item in batch],
        "n_target": torch.tensor([item["n_target"] for item in batch], dtype=torch.long),
        "n_prompt": torch.tensor([item["n_prompt"] for item in batch], dtype=torch.long),
    }


def build_ruler_sft_dataloader(
    config: RulerSFTConfig,
    tokenizer,
    *,
    batch_size: int = 1,
    num_workers: int = 2,
    rank: int = 0,
    world_size: int = 1,
    prefetch_factor: int | None = 2,
) -> DataLoader:
    """
    A ``DataLoader`` over :class:`RulerSFTDataset`.

    ``persistent_workers`` is left **off**, unlike the longmino loaders. The stats a worker
    accumulates live in its own process, so keeping workers alive across epochs would hide the
    drop table behind a fork; more importantly the corpus is small enough that worker startup is
    not on the critical path.
    """
    dataset = RulerSFTDataset(config, tokenizer, rank=rank, world_size=world_size)
    kwargs: dict = {}
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=sft_collate,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        **kwargs,
    )
