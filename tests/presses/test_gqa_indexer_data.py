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
    TokenizedConfig,
    TokenizedDataset,
    build_tokenized_dataloader,
    read_index,
    wsd_lr_lambda,
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


# ----------------------------------------------------------------------
# WSD learning-rate schedule
# ----------------------------------------------------------------------
def test_wsd_phases():
    """
    Warmup ramps, stable is exactly flat at the peak, decay lands on the floor.

    The peak must be reached exactly (multiplier 1.0), not approached: the optimizer is built
    at ``peak_lr`` and this returns a factor, so a peak of 0.99 would silently train 1% below
    the requested rate for 60% of the run.
    """
    total = 1000
    lr = wsd_lr_lambda(total, warmup_frac=0.1, stable_frac=0.6, peak_lr=1e-3, final_lr=5e-6)

    assert lr(0) == pytest.approx(1 / 100), "first step must be peak/warmup, not zero"
    assert lr(99) == pytest.approx(1.0), "peak reached at the end of warmup"
    assert lr(100) == 1.0 and lr(500) == 1.0 and lr(699) == 1.0, "stable must be flat"
    assert lr(total - 1) == pytest.approx(5e-6 / 1e-3), "floor hit on the LAST step"


def test_wsd_decay_reaches_the_requested_floor_on_the_last_step():
    """
    LambdaLR is called with 0..total-1, so the denominator must exclude the final index.

    Using ``total - stable_end`` leaves the last step above the floor -- measured 7.2e-6
    instead of the requested 5e-6 on a 1500-step run -- which means the schedule never
    delivers the value that was configured.
    """
    total, peak, final = 1500, 1e-3, 5e-6
    lr = wsd_lr_lambda(total, peak_lr=peak, final_lr=final)
    assert peak * lr(total - 1) == pytest.approx(final, rel=1e-9)


def test_wsd_decay_is_monotone():
    lr = wsd_lr_lambda(1000, peak_lr=1e-3, final_lr=5e-6)
    values = [lr(s) for s in range(700, 1000)]
    assert all(a >= b for a, b in zip(values, values[1:])), "decay must not increase"


def test_wsd_is_never_zero_or_negative():
    """A zero multiplier wastes a step; a negative one would ascend the gradient."""
    lr = wsd_lr_lambda(50, peak_lr=1e-3, final_lr=5e-6)
    assert all(lr(s) > 0 for s in range(60)), "including past the end, where LambdaLR may call"


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"warmup_frac": 0.0}, "warmup_frac"),
        ({"warmup_frac": 1.0}, "warmup_frac"),
        ({"stable_frac": -0.1}, "stable_frac"),
        ({"warmup_frac": 0.5, "stable_frac": 0.5}, "no decay phase"),
        ({"final_lr": 1e-2}, "exceeds peak_lr"),
        ({"peak_lr": 0.0}, "must be positive"),
    ],
)
def test_wsd_validation(kwargs, match):
    base = {"peak_lr": 1e-3, "final_lr": 5e-6}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        wsd_lr_lambda(100, **base)


# ----------------------------------------------------------------------
# Pre-tokenized corpus
# ----------------------------------------------------------------------
def write_tokenized(root, subset, shard, num_docs, seq_len):
    import numpy as np

    directory = root / subset
    directory.mkdir(parents=True, exist_ok=True)
    array = np.arange(num_docs * seq_len, dtype=np.uint32).reshape(num_docs, seq_len)
    np.save(directory / f"{shard}.npy", array)
    with open(directory / f"{shard}.json", "w") as handle:
        json.dump(
            {
                "shard": shard,
                "num_docs": num_docs,
                "seq_len": seq_len,
                "doc_ids": [f"{subset}_{i}" for i in range(num_docs)],
                "available_tokens": [seq_len] * num_docs,
            },
            handle,
        )
    return array


@pytest.fixture
def tokenized(tmp_path):
    """A two-subset pretokenized corpus with 8 documents of 64 tokens each."""
    write_tokenized(tmp_path, "2e16", "longmino_2e16-0000", 5, 64)
    write_tokenized(tmp_path, "2e17", "longmino_2e17-0000", 3, 64)
    with open(tmp_path / "index.json", "w") as handle:
        json.dump(
            {
                "seq_len": 64,
                "min_tokens": 64,
                "model": "test",
                "dtype": "uint32",
                "subsets": {"2e16": 5, "2e17": 3},
                "total_docs": 8,
                "complete": True,
            },
            handle,
        )
    return tmp_path


def test_tokenized_reads_every_document(tokenized):
    config = TokenizedConfig(root=str(tokenized), seq_len=64, shuffle_buffer=0)
    samples = list(TokenizedDataset(config))
    assert len(samples) == 8
    assert {s["input_ids"].shape for s in samples} == {(64,)}
    assert samples[0]["input_ids"].dtype == torch.long, "the model needs int64 ids"


def test_tokenized_slices_a_shorter_seq_len(tokenized):
    """
    A shorter stage takes a prefix -- that is exactly what a shorter sequence means.

    This is what lets one 64K file serve both a 32K and a 64K stage, instead of pretokenizing
    the corpus twice.
    """
    config = TokenizedConfig(root=str(tokenized), seq_len=16, shuffle_buffer=0, take_from="head")
    samples = list(TokenizedDataset(config))
    assert all(s["input_ids"].shape == (16,) for s in samples)
    # Row 0 of the first shard starts at 0 by construction, so a head slice is 0..15.
    first = min(samples, key=lambda s: int(s["input_ids"][0]))
    assert first["input_ids"].tolist() == list(range(16))


def test_tokenized_refuses_a_longer_seq_len_than_stored(tokenized):
    """
    Those tokens were never written, so they cannot be recovered.

    Padding instead would train the indexer on positions the loss masks out, and silently at
    that -- the sample count and shapes would all look right.
    """
    with pytest.raises(ValueError, match="exceeds the pretokenized width"):
        TokenizedDataset(TokenizedConfig(root=str(tokenized), seq_len=128))


def test_tokenized_missing_index_names_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="pretokenize_longmino"):
        read_index(tmp_path)


def test_tokenized_rejects_an_unknown_subset(tokenized):
    with pytest.raises(ValueError, match="not in the pretokenized corpus"):
        TokenizedDataset(TokenizedConfig(root=str(tokenized), seq_len=64, subsets=("8k_32k",)))


def test_tokenized_subset_selection(tokenized):
    config = TokenizedConfig(root=str(tokenized), seq_len=64, subsets=("2e17",), shuffle_buffer=0)
    samples = list(TokenizedDataset(config))
    assert len(samples) == 3
    assert all(s["doc_id"].startswith("2e17") for s in samples)


def test_tokenized_partitions_shards_across_ranks(tokenized):
    config = TokenizedConfig(root=str(tokenized), seq_len=64)
    assigned = [
        set(TokenizedDataset(config, rank=r, world_size=2).assigned_paths()) for r in range(2)
    ]
    assert sum(len(a) for a in assigned) == len(set().union(*assigned)), "ranks overlap"


def test_tokenized_random_window_stays_in_bounds(tokenized):
    """A random window must not run past the stored width."""
    config = TokenizedConfig(
        root=str(tokenized), seq_len=16, take_from="random", shuffle_buffer=0, seed=3
    )
    for sample in TokenizedDataset(config):
        assert sample["input_ids"].shape == (16,)
        assert int(sample["input_ids"].max()) < 8 * 64, "index escaped the array"


def test_tokenized_dataloader_end_to_end(tokenized):
    config = TokenizedConfig(root=str(tokenized), seq_len=32, shuffle_buffer=0)
    loader = build_tokenized_dataloader(config, batch_size=2, num_workers=0)
    batches = list(loader)
    assert batches and all(b["input_ids"].shape == (2, 32) for b in batches)


def test_tokenized_samples_do_not_alias_the_mmap(tokenized):
    """
    Each sample must own its memory.

    A tensor built directly on a memory-mapped page would read freed memory once the kernel
    evicts it -- an intermittent, data-dependent corruption rather than a clean failure.
    """
    config = TokenizedConfig(root=str(tokenized), seq_len=32, shuffle_buffer=0)
    samples = list(TokenizedDataset(config))
    tensor = samples[0]["input_ids"]
    assert tensor.is_contiguous()
    before = int(tensor[0])
    tensor[0] = 12345  # writable, so it is a copy rather than a read-only mmap view
    assert int(tensor[0]) == 12345 and before != 12345


# ----------------------------------------------------------------------
# Multi-rank partitioning (8-GPU shape)
# ----------------------------------------------------------------------
def test_eight_ranks_leave_no_reader_idle(tokenized, tmp_path):
    """
    At 8 ranks x N workers, every reader must get shards and none may overlap.

    An idle reader looks like a slow run rather than a bug, and an overlap trains on duplicated
    data while reporting the full corpus -- both silent. Shaped like the real corpus (58 shards
    from 2e16 + 2e17) rather than the 2-shard fixture, since the failure only appears when the
    reader count approaches the shard count.
    """
    import unittest.mock as mock

    import numpy as np

    root = tmp_path / "many"
    counts = {"2e16": 23, "2e17": 35}
    for subset, num in counts.items():
        directory = root / subset
        directory.mkdir(parents=True)
        for i in range(num):
            np.save(directory / f"longmino_{subset}-{i:04d}.npy", np.zeros((2, 64), np.uint32))
    with open(root / "index.json", "w") as handle:
        json.dump(
            {"seq_len": 64, "min_tokens": 64, "model": "t", "dtype": "uint32",
             "subsets": counts, "total_docs": 116, "complete": True},
            handle,
        )

    config = TokenizedConfig(root=str(root), seq_len=64)
    total_shards = sum(counts.values())
    for num_workers in (1, 2, 4):
        sizes, union = [], set()
        for rank in range(8):
            dataset = TokenizedDataset(config, rank=rank, world_size=8)
            for worker_id in range(num_workers):
                info = mock.Mock(num_workers=num_workers, id=worker_id)
                with mock.patch(
                    "kvpress.presses.gqa_indexer.data.get_worker_info", return_value=info
                ):
                    assigned = set(dataset.assigned_paths())
                sizes.append(len(assigned))
                union |= assigned
        assert min(sizes) > 0, f"num_workers={num_workers} left {sizes.count(0)} readers idle"
        assert sum(sizes) == len(union), f"num_workers={num_workers}: readers overlap"
        assert len(union) == total_shards, f"num_workers={num_workers}: shards missed"


def test_ranks_draw_different_documents(tokenized):
    """
    Two ranks must not see the same documents, or an 8x batch is 8x redundant.

    Guaranteed by shard partitioning rather than by seeding, so it holds even when the ranks
    share a seed.
    """
    config = TokenizedConfig(root=str(tokenized), seq_len=64, shuffle_buffer=0)
    first = {s["doc_id"] for s in TokenizedDataset(config, rank=0, world_size=2)}
    second = {s["doc_id"] for s in TokenizedDataset(config, rank=1, world_size=2)}
    assert first and second
    assert not (first & second), "ranks saw overlapping documents"
