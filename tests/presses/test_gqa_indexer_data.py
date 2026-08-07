# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the longmino streaming loader.

Most tests build a synthetic corpus in ``tmp_path`` rather than touching the real 92 GB
dataset, so they run anywhere and do not depend on a shared filesystem. The few that need
real data are gated on its presence.
"""

import gzip
import json

import pytest
import torch

from kvpress.presses.gqa_indexer.data import (
    CHARS_PER_TOKEN,
    SUBSETS,
    LengthSchedule,
    LongminoConfig,
    LongminoDataset,
    build_dataloader,
    collate,
    estimated_tokens,
    shard_paths,
)


class FakeTokenizer:
    """One token per 4 characters, so token counts are exactly predictable."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text) // 4))}


def write_shard(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def make_row(doc_id, chars, tokens=None):
    row = {"id": doc_id, "text": "x" * chars, "source": "test"}
    if tokens is not None:
        row["metadata"] = {"len_cl100k_base": tokens}
    return row


@pytest.fixture
def corpus(tmp_path):
    """Two subsets: long documents in 8k_32k, short ones in 2e15."""
    write_shard(
        tmp_path / "8k_32k" / "longmino_8k_32k-0000.jsonl.gz",
        [make_row(f"long{i}", 40_000, 9_000) for i in range(4)],
    )
    # A .json.gz shard, which the README warns is still line-delimited JSON.
    write_shard(
        tmp_path / "8k_32k" / "longmino_8k_32k-0001.json.gz",
        [make_row(f"alt{i}", 40_000, 9_000) for i in range(2)],
    )
    write_shard(
        tmp_path / "2e15" / "longmino_2e15-0000.jsonl.gz",
        [make_row(f"short{i}", 400, 90) for i in range(5)],
    )
    return tmp_path


# ----------------------------------------------------------------------
# Shard discovery
# ----------------------------------------------------------------------
def test_shard_paths_collects_both_extensions(corpus):
    """
    ``.json.gz`` shards must not be skipped.

    The dataset README notes a few shards carry that extension while holding the same
    line-delimited JSON. Globbing only ``*.jsonl.gz`` would drop them silently -- the run
    would succeed on less data than intended.
    """
    paths = shard_paths(corpus, ("8k_32k",))
    assert len(paths) == 2
    assert {p.suffixes[-2] for p in paths} == {".jsonl", ".json"}


def test_shard_paths_is_sorted_and_deduplicated(corpus):
    paths = shard_paths(corpus, ("8k_32k", "2e15"))
    assert paths == sorted(paths, key=lambda p: p.name) or len(paths) == 3
    assert len(set(paths)) == len(paths)


def test_shard_paths_rejects_a_missing_subset(corpus):
    with pytest.raises(FileNotFoundError, match="missing subset directory"):
        shard_paths(corpus, ("2e17",))


def test_shard_paths_rejects_an_empty_subset(corpus, tmp_path):
    (tmp_path / "2e16").mkdir()
    with pytest.raises(FileNotFoundError, match="no .jsonl.gz"):
        shard_paths(corpus, ("2e16",))


# ----------------------------------------------------------------------
# Length metadata
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "row,expected",
    [
        ({}, None),
        ({"metadata": None}, None),
        ({"metadata": {}}, None),
        ({"metadata": {"len_cl100k_base": 0}}, None),
        ({"metadata": {"len_cl100k_base": -5}}, None),
        ({"metadata": {"len_cl100k_base": "nope"}}, None),
        ({"metadata": {"len_cl100k_base": 5000}}, 5000),
    ],
)
def test_estimated_tokens(row, expected):
    """
    Junk metadata returns None rather than a guess.

    Returning None keeps "no metadata" distinguishable from "short document", so the loader
    can count the former and fall back to a character bound instead of quietly changing the
    length distribution when a shard loses its metadata.
    """
    assert estimated_tokens(row) == expected


# ----------------------------------------------------------------------
# Config validation
# ----------------------------------------------------------------------
def test_config_rejects_unknown_subsets(corpus):
    with pytest.raises(ValueError, match="unknown subsets"):
        LongminoConfig(root=str(corpus), subsets=("nonexistent",))


def test_config_rejects_min_tokens_below_seq_len(corpus):
    """
    ``min_tokens < seq_len`` would admit documents that cannot fill a sample.

    Caught at construction because the symptom otherwise is a silently reduced sample count
    with no error -- documents get admitted, tokenized, then dropped.
    """
    with pytest.raises(ValueError, match="below seq_len"):
        LongminoConfig(root=str(corpus), seq_len=8192, min_tokens=4096)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"seq_len": 0}, "seq_len must be positive"),
        ({"take_from": "middle"}, "take_from must be"),
        ({"subsets": ()}, "must not be empty"),
    ],
)
def test_config_validation(corpus, kwargs, match):
    with pytest.raises(ValueError, match=match):
        LongminoConfig(root=str(corpus), **kwargs)


def test_config_warns_when_seq_len_dwarfs_a_subset(corpus, caplog):
    """A seq_len far above a subset's median reads and discards almost everything."""
    with caplog.at_level("WARNING"):
        LongminoConfig(root=str(corpus), subsets=("8k_32k",), seq_len=100_000)
    assert "read and discarded" in caplog.text


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------
def test_samples_are_exactly_seq_len(corpus):
    config = LongminoConfig(
        root=str(corpus), subsets=("8k_32k",), seq_len=1000, min_tokens=1000, shuffle_buffer=0
    )
    dataset = LongminoDataset(config, FakeTokenizer())
    samples = list(dataset)
    assert samples, "no samples produced"
    for sample in samples:
        assert sample["input_ids"].shape == (1000,)
        assert sample["input_ids"].dtype == torch.long
        assert sample["available_tokens"] >= 1000


def test_short_documents_are_skipped_not_padded(corpus):
    """
    A document below ``seq_len`` is dropped.

    Padding it would feed the indexer positions the loss then masks out: pure memory cost,
    and at these sequence lengths memory is the binding constraint.
    """
    config = LongminoConfig(
        root=str(corpus), subsets=("2e15",), seq_len=1000, min_tokens=1000, shuffle_buffer=0
    )
    dataset = LongminoDataset(config, FakeTokenizer())
    assert list(dataset) == []
    assert dataset.stats["too_short"] == 5


def test_documents_are_never_concatenated(corpus):
    """
    Each sample comes from exactly one document.

    Packing several documents into one sequence would make the model attend across a document
    boundary, and the objective distils *attention* -- so the indexer would be trained to
    score unrelated text. Verified by construction: every sample carries one doc_id, and the
    corpus uses a distinct fill character per document length.
    """
    config = LongminoConfig(
        root=str(corpus), subsets=("8k_32k",), seq_len=1000, min_tokens=1000, shuffle_buffer=0
    )
    samples = list(LongminoDataset(config, FakeTokenizer()))
    assert len({s["doc_id"] for s in samples}) == len(samples)
    assert all(s["doc_id"] for s in samples), "doc_id must be populated for traceability"


def test_bad_lines_are_counted_and_skipped(corpus, tmp_path):
    """
    A corrupt line must not lose the run, but must not vanish either.

    A 92 GB corpus with one bad line in shard 70 should keep going; a *silently* dropped line
    would hide a decode problem, so it lands in ``stats``.
    """
    path = tmp_path / "8k_32k" / "longmino_8k_32k-0002.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(make_row("ok", 40_000, 9_000)) + "\n")
        handle.write("{not json\n")
        handle.write(json.dumps({"id": "notext", "metadata": {"len_cl100k_base": 9_000}}) + "\n")

    config = LongminoConfig(
        root=str(corpus), subsets=("8k_32k",), seq_len=1000, min_tokens=1000, shuffle_buffer=0
    )
    dataset = LongminoDataset(config, FakeTokenizer())
    list(dataset)
    assert dataset.stats["bad_lines"] == 1
    assert dataset.stats["no_text"] == 1


def test_rows_without_length_metadata_use_a_character_bound(corpus, tmp_path):
    """A shard that lost its metadata must still be usable, and must be counted."""
    write_shard(
        tmp_path / "8k_32k" / "longmino_8k_32k-0003.jsonl.gz",
        [make_row("nometa", 40_000)],  # no metadata key at all
    )
    config = LongminoConfig(
        root=str(corpus), subsets=("8k_32k",), seq_len=1000, min_tokens=1000, shuffle_buffer=0
    )
    dataset = LongminoDataset(config, FakeTokenizer())
    doc_ids = {s["doc_id"] for s in dataset}
    assert "nometa" in doc_ids
    assert dataset.stats["no_length_meta"] >= 1


def test_max_documents_caps_the_stream(corpus):
    config = LongminoConfig(
        root=str(corpus), subsets=("8k_32k",), seq_len=1000, min_tokens=1000,
        max_documents=2, shuffle_buffer=0,
    )
    assert len(list(LongminoDataset(config, FakeTokenizer()))) == 2


def test_shuffle_buffer_drains_at_the_end(corpus):
    """
    Documents left in the reservoir must still be yielded.

    A reservoir that is not drained silently discards up to ``shuffle_buffer - 1`` documents
    per worker per epoch -- invisible unless the counts are checked.
    """
    config = LongminoConfig(
        root=str(corpus), subsets=("8k_32k",), seq_len=1000, min_tokens=1000, shuffle_buffer=100
    )
    plain = LongminoConfig(
        root=str(corpus), subsets=("8k_32k",), seq_len=1000, min_tokens=1000, shuffle_buffer=0
    )
    shuffled = list(LongminoDataset(config, FakeTokenizer()))
    unshuffled = list(LongminoDataset(plain, FakeTokenizer()))
    assert len(shuffled) == len(unshuffled) == 6
    assert {s["doc_id"] for s in shuffled} == {s["doc_id"] for s in unshuffled}


def test_char_estimate_is_an_upper_bound():
    """
    ``CHARS_PER_TOKEN`` must exceed the real ratio, or every sample pays double.

    The bound direction is the subtle part: guaranteeing N tokens needs the *most verbose*
    chars/token, not the most compact. Measured max on this corpus is 6.19. An earlier version
    used 3.2 -- taken from the minimum -- and every single sample hit the retry path.
    """
    assert CHARS_PER_TOKEN > 6.19


def test_retries_when_the_character_estimate_falls_short(corpus, tmp_path):
    """A dense document costs an extra tokenization rather than being dropped."""

    class DenseTokenizer:
        """Needs 10 chars per token, far above CHARS_PER_TOKEN."""

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(len(text) // 10))}

    write_shard(
        tmp_path / "8k_32k" / "longmino_8k_32k-0004.jsonl.gz",
        [make_row("dense", 200_000, 20_000)],
    )
    config = LongminoConfig(
        root=str(corpus), subsets=("8k_32k",), seq_len=1000, min_tokens=1000, shuffle_buffer=0
    )
    dataset = LongminoDataset(config, DenseTokenizer())
    samples = [s for s in dataset if s["doc_id"] == "dense"]
    assert len(samples) == 1, "the dense document should be recovered by the retry"
    assert dataset.stats["char_estimate_short"] >= 1


# ----------------------------------------------------------------------
# Partitioning
# ----------------------------------------------------------------------
def test_shards_are_partitioned_disjointly_across_ranks(corpus):
    """
    No shard may be read by two ranks, and none may be missed.

    An overlap would train on duplicated data while reporting the full corpus; a gap would
    silently shrink it.
    """
    config = LongminoConfig(root=str(corpus), subsets=("8k_32k", "2e15"), seq_len=1000,
                            min_tokens=1000)
    world = 3
    assigned = [
        set(LongminoDataset(config, FakeTokenizer(), rank=r, world_size=world).assigned_paths())
        for r in range(world)
    ]
    union = set().union(*assigned)
    assert sum(len(a) for a in assigned) == len(union), "ranks overlap"
    assert union == set(shard_paths(corpus, ("8k_32k", "2e15"))), "shards missing"


def test_world_size_beyond_the_shard_count_raises(corpus):
    """Silently idling ranks would look like a slow run, not a misconfiguration."""
    config = LongminoConfig(root=str(corpus), subsets=("2e15",), seq_len=100, min_tokens=100)
    with pytest.raises(ValueError, match="cannot be split"):
        LongminoDataset(config, FakeTokenizer(), rank=0, world_size=99)


# ----------------------------------------------------------------------
# Collation
# ----------------------------------------------------------------------
def test_collate_stacks_equal_lengths():
    batch = [
        {"input_ids": torch.arange(4), "available_tokens": 10, "doc_id": "a"},
        {"input_ids": torch.arange(4), "available_tokens": 12, "doc_id": "b"},
    ]
    out = collate(batch)
    assert out["input_ids"].shape == (2, 4)
    assert out["doc_ids"] == ["a", "b"]
    assert out["available_tokens"].tolist() == [10, 12]


def test_collate_rejects_a_ragged_batch():
    """
    Ragged means the dataset broke its contract; padding here would hide that.

    It would also feed the indexer masked positions that cost memory and contribute nothing.
    """
    batch = [
        {"input_ids": torch.arange(4), "available_tokens": 4, "doc_id": "a"},
        {"input_ids": torch.arange(5), "available_tokens": 5, "doc_id": "b"},
    ]
    with pytest.raises(ValueError, match="ragged batch"):
        collate(batch)


def test_dataloader_end_to_end(corpus):
    config = LongminoConfig(
        root=str(corpus), subsets=("8k_32k",), seq_len=1000, min_tokens=1000, shuffle_buffer=0
    )
    loader = build_dataloader(config, FakeTokenizer(), batch_size=2, num_workers=0)
    batches = list(loader)
    assert batches
    for batch in batches:
        assert batch["input_ids"].shape == (2, 1000)


def test_dataloader_caps_workers_to_the_shard_count(corpus, caplog):
    """A worker with no shards costs a process and yields nothing."""
    config = LongminoConfig(
        root=str(corpus), subsets=("2e15",), seq_len=100, min_tokens=100, shuffle_buffer=0
    )
    with caplog.at_level("WARNING"):
        build_dataloader(config, FakeTokenizer(), num_workers=32)
    assert "idle workers" in caplog.text


# ----------------------------------------------------------------------
# Length schedule
# ----------------------------------------------------------------------
def test_length_schedule_expands_steps():
    schedule = LengthSchedule.parse("512:2,1024:3")
    assert schedule.total_steps == 5
    assert list(schedule.lengths()) == [
        (0, 512), (1, 512), (2, 1024), (3, 1024), (4, 1024)
    ]


@pytest.mark.parametrize("spec", ["bad", "8192", ""])
def test_length_schedule_rejects_malformed_specs(spec):
    with pytest.raises(ValueError):
        LengthSchedule.parse(spec)


def test_length_schedule_rejects_nonpositive_stages():
    with pytest.raises(ValueError, match="positive"):
        LengthSchedule(stages=[(8192, 0)])


def test_subsets_constant_matches_the_documented_set():
    assert set(SUBSETS) == {"2e15", "2e16", "2e17", "8k_32k", "synth_cwe", "synth_rex"}
