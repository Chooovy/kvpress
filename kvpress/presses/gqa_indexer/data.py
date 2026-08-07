# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Streaming loader for the longmino gzipped-JSONL shards.

The indexer is trained on long documents, so the loader is built around one constraint that
shapes everything else: **a training sample must be a single document at least ``seq_len``
tokens long**. Concatenating short documents into a packed sequence would be wrong here in a
way it is not for LM pre-training -- the objective distils *attention*, and attention across
a document boundary is the model attending to unrelated text. That would teach the indexer to
score noise.

Consequences of that choice, and why the pipeline looks the way it does:

**Filter by metadata, not by tokenizing.** Every shard carries ``metadata.len_cl100k_base``
(verified present on 100% of sampled rows in all six subsets), and measured against the Qwen3
tokenizer the ratio is 1.02 (range 0.998-1.061 over four subsets). So a metadata threshold is
a reliable prefilter, and 92 GB never has to be tokenized to find out what is long enough.
The real length is still checked after tokenizing -- the ratio is a good estimate, not a
guarantee.

**Truncate by characters before tokenizing.** A 256K-token document costs ~1 s to tokenize
and is then thrown away down to ``seq_len``. Reading ``seq_len * CHARS_PER_TOKEN`` characters
first avoids that. The constant is an *upper* bound on chars-per-token (measured max 6.19 over
prefixes from all six subsets, so 6.5 with margin) because guaranteeing ``seq_len`` tokens
requires the most verbose case, not the most compact one. The count is verified after
tokenizing regardless, with one retry on the full text, so an unusual document costs an extra
pass rather than being dropped.

**Stream, never index.** ``IterableDataset`` over gzip handles, with shards partitioned across
workers and ranks. Nothing is materialized and no shard is opened twice in a step.

Subsets, from the dataset's own directory layout (median ``len_cl100k_base`` measured over the
first 300 rows of the first shard of each):

============  ======  =============  ==============================================
subset        shards  median tokens  use
============  ======  =============  ==============================================
``8k_32k``    9       10.5K          short; stage-1 warmup
``2e15``      9       42.7K          mid
``synth_cwe`` 11      43.0K          mid, synthetic
``synth_rex`` 13      44.7K          mid, synthetic
``2e16``      23      90.4K          long
``2e17``      35      168.5K         longest, up to 260K
============  ======  =============  ==============================================
"""

from __future__ import annotations

import glob
import gzip
import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

logger = logging.getLogger(__name__)

# Tokenizing happens inside worker processes, and the fast tokenizer's own thread pool warns
# (loudly, per worker) when it is forked after use. Workers tokenize one document at a time
# anyway, so its parallelism buys nothing here.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SUBSETS = ("2e15", "2e16", "2e17", "8k_32k", "synth_cwe", "synth_rex")

# Median tokens per subset, measured on the first 300 rows of each subset's first shard.
# Used only to order subsets and to warn about impossible seq_len/subset combinations.
SUBSET_MEDIAN_TOKENS = {
    "8k_32k": 10_550,
    "2e15": 42_716,
    "synth_cwe": 43_036,
    "synth_rex": 44_741,
    "2e16": 90_422,
    "2e17": 168_455,
}

# Characters to read per token wanted. This must be an UPPER bound on chars/token, not a
# lower one: to guarantee that a prefix yields >= N tokens you need the worst (most verbose)
# ratio, since that is the case where a given number of characters buys the fewest tokens.
# Measured max over prefixes across all six subsets is 6.19, so 6.5 leaves margin. The
# measured retry rate at 6.5 is 0/135 documents, versus 4/4 at the 3.2 this originally held --
# 3.2 came from the *minimum* ratio, which is the best case and therefore the wrong bound.
# The token count is verified after tokenizing regardless, so a miss costs a retry, not a bug.
CHARS_PER_TOKEN = 6.5

# metadata.len_cl100k_base -> Qwen3 tokens, measured mean 1.02 with a 0.998 floor. Kept at
# 1.0 so the prefilter is slightly conservative rather than slightly optimistic: admitting a
# document that turns out too short costs one wasted tokenization, whereas rejecting a good
# one is invisible.
CL100K_TO_QWEN = 1.0


def shard_paths(root: str | Path, subsets: tuple[str, ...] | list[str]) -> list[Path]:
    """
    Every shard in the requested subsets, sorted for reproducibility.

    Both ``.jsonl.gz`` and ``.json.gz`` are collected: the dataset README notes that a few
    shards carry the latter extension while still holding one JSON object per line, so
    skipping them would silently drop data.
    """
    root = Path(root)
    paths: list[Path] = []
    for subset in subsets:
        directory = root / subset
        if not directory.is_dir():
            raise FileNotFoundError(f"missing subset directory: {directory}")
        found: list[Path] = []
        for pattern in ("*.jsonl.gz", "*.json.gz"):
            found.extend(directory.glob(pattern))
        if not found:
            raise FileNotFoundError(f"no .jsonl.gz / .json.gz shards under {directory}")
        paths.extend(sorted({p.resolve() for p in found}, key=lambda p: p.name))
    return paths


def estimated_tokens(row: dict) -> int | None:
    """
    Qwen3 token estimate from ``metadata.len_cl100k_base``, or ``None`` if absent.

    Returning ``None`` rather than falling back to ``len(text) / 4`` keeps the two cases
    distinguishable: the caller decides whether an unlabelled row is worth tokenizing, and
    :class:`LongminoDataset` counts them so a shard that lost its metadata is visible in the
    logs instead of quietly changing the length distribution.
    """
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("len_cl100k_base")
    if not isinstance(value, (int, float)) or value <= 0:
        return None
    return int(value * CL100K_TO_QWEN)


@dataclass
class LongminoConfig:
    """
    What to read and how to slice it.

    Attributes
    ----------
    root : str
        Dataset root, the directory holding ``2e15``, ``8k_32k``, ... subdirectories.
    subsets : tuple of str
        Which subsets to draw from. Pick by length: see the table in the module docstring.
    seq_len : int
        Tokens per sample. Documents shorter than this are skipped, not padded -- a padded
        sample trains the indexer on positions the loss then masks out, so it is pure cost.
    min_tokens : int, optional
        Metadata threshold for admitting a document. Defaults to ``seq_len``; raise it to
        leave slack for the tokenizer estimate being off.
    take_from : str
        ``head`` takes the first ``seq_len`` tokens, ``random`` takes a random window. Random
        windows see more of each document but start mid-sentence; head windows are
        reproducible and always start at a real document beginning.
    shuffle_buffer : int
        Reservoir size for shuffling a stream. ``0`` disables it. Documents arrive in shard
        order, so without this a step's batch is highly correlated.
    seed : int
        Seeds shard order and window/shuffle choices.
    max_documents : int, optional
        Stop after this many yielded samples **per reader**, not globally: each
        ``(rank, worker)`` counts independently, because a shared counter would need
        cross-process coordination for what is only a smoke-test knob. With ``num_workers=2``
        and ``max_documents=3`` a loader yields 6 samples. Use ``num_workers=0`` when the
        exact count matters.
    """

    root: str
    subsets: tuple[str, ...] = SUBSETS
    seq_len: int = 32768
    min_tokens: int | None = None
    take_from: str = "head"
    shuffle_buffer: int = 64
    seed: int = 0
    max_documents: int | None = None

    def __post_init__(self):
        unknown = [s for s in self.subsets if s not in SUBSETS]
        if unknown:
            raise ValueError(f"unknown subsets {unknown}; choose from {SUBSETS}")
        if not self.subsets:
            raise ValueError("subsets must not be empty")
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.take_from not in ("head", "random"):
            raise ValueError(f"take_from must be 'head' or 'random', got {self.take_from!r}")
        if self.min_tokens is None:
            self.min_tokens = self.seq_len
        if self.min_tokens < self.seq_len:
            raise ValueError(
                f"min_tokens={self.min_tokens} is below seq_len={self.seq_len}, so admitted "
                "documents may be too short to fill a sample"
            )
        # A seq_len far above a subset's median means almost everything is discarded after
        # being read and parsed -- expensive silence, so say so up front.
        for subset in self.subsets:
            median = SUBSET_MEDIAN_TOKENS.get(subset)
            if median is not None and self.min_tokens > 2 * median:
                logger.warning(
                    "seq_len=%d (min_tokens=%d) is more than twice subset %r's median of "
                    "~%d tokens; most of that subset will be read and discarded",
                    self.seq_len,
                    self.min_tokens,
                    subset,
                    median,
                )


class LongminoDataset(IterableDataset):
    """
    Yields ``{"input_ids": LongTensor[seq_len], "num_tokens": int, "doc_id": str}``.

    One sample is one document window, never a concatenation of documents -- see the module
    docstring for why that matters for an attention-distillation objective.

    Shards are partitioned across ``(rank, worker)`` so no shard is read twice per epoch.
    Partitioning by *shard* rather than by row is what keeps this cheap: a worker opens its
    own gzip handles and never decompresses bytes destined for another worker.
    """

    def __init__(self, config: LongminoConfig, tokenizer, rank: int = 0, world_size: int = 1):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.rank = rank
        self.world_size = world_size
        self.paths = shard_paths(config.root, config.subsets)
        if len(self.paths) < world_size:
            raise ValueError(
                f"{len(self.paths)} shards cannot be split across world_size={world_size}; "
                "add subsets or lower world_size"
            )
        self.stats: dict[str, int] = {}

    def assigned_paths(self) -> list[Path]:
        """
        This ``(rank, worker)``'s shards, shuffled by seed.

        The shuffle happens before slicing so that every rank sees a mix of subsets rather
        than, say, rank 0 getting all of ``8k_32k``. That matters because the subsets have
        very different length distributions.
        """
        paths = list(self.paths)
        random.Random(self.config.seed).shuffle(paths)

        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0

        shard_index = self.rank * num_workers + worker_id
        total_shards = self.world_size * num_workers
        mine = paths[shard_index::total_shards]
        if not mine:
            logger.warning(
                "rank %d worker %d got no shards (%d shards, %d readers); it will idle",
                self.rank,
                worker_id,
                len(paths),
                total_shards,
            )
        return mine

    def read_documents(self, paths: list[Path]) -> Iterator[dict]:
        """
        Parse rows from the assigned shards, skipping ones that cannot fill a sample.

        A corrupt line is counted and skipped rather than raised: a 92 GB corpus with a bad
        line in shard 70 should not lose the run, but a *silently* dropped line would hide a
        decode problem, so the counts land in :attr:`stats` and are logged per shard.
        """
        config = self.config
        stats = self.stats
        for path in paths:
            kept = seen = 0
            try:
                handle = gzip.open(path, "rt", encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.error("cannot open shard %s: %s", path.name, exc)
                stats["shard_errors"] = stats.get("shard_errors", 0) + 1
                continue
            with handle:
                for line in handle:
                    seen += 1
                    try:
                        row = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        stats["bad_lines"] = stats.get("bad_lines", 0) + 1
                        continue
                    text = row.get("text")
                    if not isinstance(text, str) or not text:
                        stats["no_text"] = stats.get("no_text", 0) + 1
                        continue

                    tokens = estimated_tokens(row)
                    if tokens is None:
                        # No length metadata: fall back to the character bound rather than
                        # dropping the row, since a whole shard could be affected.
                        stats["no_length_meta"] = stats.get("no_length_meta", 0) + 1
                        if len(text) < config.min_tokens * CHARS_PER_TOKEN:
                            stats["too_short"] = stats.get("too_short", 0) + 1
                            continue
                    elif tokens < config.min_tokens:
                        stats["too_short"] = stats.get("too_short", 0) + 1
                        continue

                    kept += 1
                    yield row
            stats["shards_read"] = stats.get("shards_read", 0) + 1
            logger.debug("%s: kept %d of %d rows", path.name, kept, seen)

    def window(self, text: str, rng: random.Random) -> tuple[list[int], int] | None:
        """
        Tokenize a ``seq_len`` window, pre-truncating by characters first.

        Tokenizing a full 256K-token document to keep 32K of it wastes most of a second per
        sample, and the loader is otherwise cheap enough that this would dominate. The
        character bound is an *upper* bound on chars/token (:data:`CHARS_PER_TOKEN` = 6.5
        against a measured max of 6.19) and the token count is *verified* afterwards, with one
        retry on the untruncated text -- so an unusually verbose document costs an extra pass
        rather than being silently dropped. ``stats["char_estimate_short"]`` counts those
        retries; if it tracks the emitted count, the constant is too low and every sample is
        paying for two tokenizations.

        Returns ``None`` when even the full text is too short.
        """
        seq_len = self.config.seq_len
        budget = int(seq_len * CHARS_PER_TOKEN)

        if self.config.take_from == "random" and len(text) > budget:
            start = rng.randrange(0, len(text) - budget + 1)
        else:
            start = 0
        excerpt = text[start : start + budget]

        ids = self.tokenizer(excerpt, add_special_tokens=False)["input_ids"]
        if len(ids) < seq_len:
            # The character estimate came up short. Retry on the remainder of the document
            # (from the same start), which is the only way to be sure.
            self.stats["char_estimate_short"] = self.stats.get("char_estimate_short", 0) + 1
            ids = self.tokenizer(text[start:], add_special_tokens=False)["input_ids"]
            if len(ids) < seq_len:
                return None
        return ids[:seq_len], len(ids)

    def __iter__(self) -> Iterator[dict]:
        config = self.config
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        # Distinct streams per (rank, worker) so windows are not correlated across readers.
        rng = random.Random((config.seed, self.rank, worker_id).__hash__())

        buffer: list[dict] = []
        emitted = 0

        def make_sample(row: dict) -> dict | None:
            got = self.window(row["text"], rng)
            if got is None:
                self.stats["short_after_tokenize"] = (
                    self.stats.get("short_after_tokenize", 0) + 1
                )
                return None
            ids, available = got
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "num_tokens": len(ids),
                "available_tokens": available,
                "doc_id": str(row.get("id", "")),
            }

        for row in self.read_documents(self.assigned_paths()):
            sample = make_sample(row)
            if sample is None:
                continue

            if config.shuffle_buffer > 1:
                buffer.append(sample)
                if len(buffer) < config.shuffle_buffer:
                    continue
                index = rng.randrange(len(buffer))
                buffer[index], buffer[-1] = buffer[-1], buffer[index]
                sample = buffer.pop()

            yield sample
            emitted += 1
            if config.max_documents is not None and emitted >= config.max_documents:
                self.stats["emitted"] = emitted
                return

        # Drain what the reservoir still holds, or those documents are simply lost.
        rng.shuffle(buffer)
        for sample in buffer:
            yield sample
            emitted += 1
            if config.max_documents is not None and emitted >= config.max_documents:
                break
        self.stats["emitted"] = emitted


def collate(batch: list[dict]) -> dict:
    """
    Stack equal-length samples; no padding is possible or needed.

    Every sample is exactly ``seq_len`` tokens by construction, so this asserts that instead
    of padding. A ragged batch would mean the dataset changed contract, and padding it here
    would hide that -- while also feeding the indexer masked positions that contribute
    nothing but memory.
    """
    lengths = {int(item["input_ids"].shape[0]) for item in batch}
    if len(lengths) != 1:
        raise ValueError(f"ragged batch: sample lengths {sorted(lengths)} differ")
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "doc_ids": [item["doc_id"] for item in batch],
        "available_tokens": torch.tensor(
            [item["available_tokens"] for item in batch], dtype=torch.long
        ),
    }


def build_dataloader(
    config: LongminoConfig,
    tokenizer,
    *,
    batch_size: int = 1,
    num_workers: int = 2,
    rank: int = 0,
    world_size: int = 1,
    prefetch_factor: int | None = 2,
) -> DataLoader:
    """
    A ``DataLoader`` over :class:`LongminoDataset`.

    ``batch_size`` defaults to 1 because at the lengths this loader targets the sequence axis
    already saturates the GPU: at ``seq_len=32K`` the fused stage-1 loss holds ~654 KiB/token,
    so a batch of 2 halves the usable length. Raise it only for short-``seq_len`` runs.

    ``num_workers`` is capped by the shard count, since a worker with no shards just idles
    while still costing a process. ``prefetch_factor`` must be ``None`` when
    ``num_workers == 0``, which PyTorch enforces by raising -- handled here instead.
    """
    dataset = LongminoDataset(config, tokenizer, rank=rank, world_size=world_size)

    shards_per_rank = len(dataset.paths) // max(world_size, 1)
    if num_workers > shards_per_rank:
        logger.warning(
            "num_workers=%d exceeds %d shards per rank; lowering to avoid idle workers",
            num_workers,
            shards_per_rank,
        )
        num_workers = max(shards_per_rank, 0)

    kwargs: dict = {}
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = True

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        **kwargs,
    )


@dataclass
class LengthSchedule:
    """
    Sequence lengths to train at, in order.

    Indexer distillation benefits from starting short: stage 1 is ``O(L^2)`` in compute
    (measured 3.9x per doubling on an H20), so early steps at 8K cost ~16x less than at 32K
    while still teaching the indexer the basic shape of attention. MiniMax MSA and DSA both
    warm up before going long.

    ``stages`` is a list of ``(seq_len, steps)``. :meth:`lengths` expands it into one entry
    per step, which the training loop consumes to know when to rebuild the loader -- changing
    ``seq_len`` means new shard filtering, so the loader cannot be reused across stages.
    """

    stages: list[tuple[int, int]] = field(default_factory=lambda: [(8192, 200), (32768, 800)])

    def __post_init__(self):
        if not self.stages:
            raise ValueError("stages must not be empty")
        for seq_len, steps in self.stages:
            if seq_len <= 0 or steps <= 0:
                raise ValueError(f"stage ({seq_len}, {steps}) must have positive values")

    @property
    def total_steps(self) -> int:
        return sum(steps for _, steps in self.stages)

    def lengths(self) -> Iterator[tuple[int, int]]:
        """Yield ``(step, seq_len)`` for every step in the schedule."""
        step = 0
        for seq_len, steps in self.stages:
            for _ in range(steps):
                yield step, seq_len
                step += 1

    @classmethod
    def parse(cls, spec: str) -> LengthSchedule:
        """
        Parse ``"8192:200,32768:800"``.

        A string form keeps the schedule expressible on a command line, so a run's length
        curriculum is recorded in the launch command rather than in edited source.
        """
        stages = []
        for chunk in spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" not in chunk:
                raise ValueError(f"stage {chunk!r} must look like SEQ_LEN:STEPS")
            seq_text, steps_text = chunk.split(":", 1)
            stages.append((int(seq_text), int(steps_text)))
        return cls(stages=stages)


def describe_subsets(root: str | Path) -> str:
    """Shard counts and sizes per subset, for logging what a run actually read."""
    lines = []
    for subset in SUBSETS:
        directory = Path(root) / subset
        if not directory.is_dir():
            continue
        shards = [Path(p) for p in glob.glob(str(directory / "*.gz"))]
        size = sum(p.stat().st_size for p in shards) / 1e9
        median = SUBSET_MEDIAN_TOKENS.get(subset)
        lines.append(
            f"  {subset:10s} {len(shards):>3} shards  {size:>6.1f} GB  "
            f"median ~{median // 1000 if median else '?'}K tokens"
        )
    return "\n".join(lines)


def env_rank_and_world_size() -> tuple[int, int]:
    """
    Read ``RANK``/``WORLD_SIZE`` from the environment, defaulting to single-process.

    Read from the environment rather than from ``torch.distributed`` so the loader can be
    exercised without initializing a process group.
    """
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
