# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the RULER answer-only SFT data path.

Most tests run on a synthetic frame and a fake tokenizer, so they need neither the network nor
the 6500-row RULER cache. The two that check the *real* prompt shape are gated on the local
snapshot and on a real tokenizer, because that is the only place a chat-template or
``enable_thinking`` drift can show up.
"""

import pandas as pd
import pytest
import torch

from kvpress.presses.gqa_indexer.sft_data import (
    ALL_TASKS,
    NIAH_TASKS,
    OTHER_TASKS,
    RulerSFTConfig,
    RulerSFTDataset,
    build_prompt_ids,
    build_target_text,
    build_ruler_sft_dataloader,
    resolve_tasks,
    sft_collate,
)


class FakeTokenizer:
    """Whitespace tokenizer with no chat template, so prompt layout is exactly predictable."""

    chat_template = None
    bos_token = "<s>"
    eos_token_id = 99

    def encode(self, text, return_tensors=None, add_special_tokens=False):
        ids = [len(word) for word in text.split()]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def apply_chat_template(self, messages, **kwargs):  # pragma: no cover - template is None
        raise AssertionError("FakeTokenizer has no chat template")


def make_frame(tasks=ALL_TASKS, context="ctx " * 20):
    """One row per task, with the real schema's columns."""
    return pd.DataFrame(
        [
            {
                "context": context,
                "question": f"question about {task} ",
                "answer_prefix": "Answer:",
                "answer": ["alpha", "beta"],
                "task": task,
                "max_new_tokens": 32,
            }
            for task in tasks
        ]
    )


@pytest.fixture
def frame_path(tmp_path):
    path = tmp_path / "ruler.parquet"
    make_frame().to_parquet(path)
    return path


# ----------------------------------------------------------------------
# Task selection
# ----------------------------------------------------------------------
def test_resolve_tasks_expands_groups_and_dedupes():
    assert resolve_tasks(None) == ALL_TASKS
    assert resolve_tasks(["all"]) == ALL_TASKS
    assert resolve_tasks(["niah"]) == NIAH_TASKS
    assert resolve_tasks(["other"]) == OTHER_TASKS
    # A group plus one of its own members collapses, and the order follows ALL_TASKS.
    assert resolve_tasks(["qa_1", "niah"]) == NIAH_TASKS + ("qa_1",)
    assert resolve_tasks(["niah", "niah_single_1"]) == NIAH_TASKS


def test_resolve_tasks_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown --sft-tasks entry"):
        resolve_tasks(["niah_single_9"])


# ----------------------------------------------------------------------
# The target text, which the metric's own grouping decides
# ----------------------------------------------------------------------
def test_qa_target_takes_one_reference_not_the_repeated_list():
    """
    ``qa_*`` is graded by ``string_match_part`` and its gold field repeats one answer per
    supporting document. Joining that list would train toward a stutter the metric never asks
    for.
    """
    assert build_target_text("qa_1", ["France"] * 4, "Answer:") == " France"


def test_string_match_all_tasks_join_every_reference_in_order():
    assert build_target_text("vt", ["A", "B", "C"], "they are: ") == "A, B, C"
    assert build_target_text("niah_multivalue", ["1", "2"], "are") == " 1, 2"


def test_duplicate_references_collapse_but_order_survives():
    assert build_target_text("cwe", ["a", "b", "a", "c"], "are:") == " a, b, c"


def test_leading_space_respects_a_prefix_that_already_ends_in_whitespace():
    """
    ``vt``'s prefix ends "they are: " while ``cwe``'s ends "are:". A blind " " + text would put a
    double space in one of them and tokenize to something eval never produces.
    """
    assert build_target_text("vt", ["X"], "they are: ") == "X"
    assert build_target_text("cwe", ["X"], "are:") == " X"


def test_empty_gold_answer_raises():
    with pytest.raises(ValueError, match="no gold answer"):
        build_target_text("vt", [], "Answer:")


# ----------------------------------------------------------------------
# Masking: the property the whole arm rests on
# ----------------------------------------------------------------------
def test_only_the_answer_carries_loss(frame_path):
    dataset = RulerSFTDataset(RulerSFTConfig(source=frame_path, max_len=10_000), FakeTokenizer())
    for sample in dataset:
        labels, ids = sample["labels"], sample["input_ids"]
        assert labels.shape == ids.shape
        # Prompt is masked, target is not, and the boundary is exactly n_prompt.
        assert (labels[: sample["n_prompt"]] == -100).all()
        assert (labels[sample["n_prompt"] :] != -100).all()
        # The supervised labels are the true ids at those positions -- not shifted here; the
        # model shifts internally.
        assert torch.equal(labels[sample["n_prompt"] :], ids[sample["n_prompt"] :])
        assert int((labels != -100).sum()) == sample["n_target"]


def test_labels_are_the_tail_of_input_ids(frame_path):
    """The supervised span is a suffix, so the first predicted token follows answer_prefix."""
    dataset = RulerSFTDataset(RulerSFTConfig(source=frame_path, max_len=10_000), FakeTokenizer())
    sample = next(iter(dataset))
    assert sample["n_prompt"] + sample["n_target"] == sample["input_ids"].numel()


# ----------------------------------------------------------------------
# Length policy: skip, never truncate
# ----------------------------------------------------------------------
def test_an_over_long_row_is_skipped_not_truncated(tmp_path):
    """
    Truncating would change the ground truth rather than shorten the input -- a needle may be in
    the cut. So the row is dropped, and counted.
    """
    path = tmp_path / "long.parquet"
    make_frame(tasks=("niah_single_1",), context="word " * 500).to_parquet(path)
    dataset = RulerSFTDataset(RulerSFTConfig(source=path, max_len=50), FakeTokenizer())
    assert list(dataset) == []
    assert dataset.stats["niah_single_1"] == [0, 1]
    # Nothing was silently shortened: the drop is visible in the table.
    assert "niah_single_1" in dataset.format_stats()


def test_kept_rows_are_within_max_len(frame_path):
    dataset = RulerSFTDataset(RulerSFTConfig(source=frame_path, max_len=10_000), FakeTokenizer())
    for sample in dataset:
        assert sample["input_ids"].numel() <= 10_000


def test_stats_count_kept_and_dropped_per_task(tmp_path):
    path = tmp_path / "mixed.parquet"
    short = make_frame(tasks=("qa_1",), context="a ")
    long = make_frame(tasks=("cwe",), context="word " * 400)
    pd.concat([short, long], ignore_index=True).to_parquet(path)
    dataset = RulerSFTDataset(RulerSFTConfig(source=path, max_len=60), FakeTokenizer())
    list(dataset)
    assert dataset.stats["qa_1"][0] == 1
    assert dataset.stats["cwe"][1] == 1


# ----------------------------------------------------------------------
# Task filtering and schema
# ----------------------------------------------------------------------
def test_only_requested_tasks_are_emitted(frame_path):
    config = RulerSFTConfig(source=frame_path, tasks=("niah",), max_len=10_000)
    dataset = RulerSFTDataset(config, FakeTokenizer())
    assert {sample["task"] for sample in dataset} <= set(NIAH_TASKS)


def test_a_task_selection_matching_nothing_raises(tmp_path):
    path = tmp_path / "niah_only.parquet"
    make_frame(tasks=("niah_single_1",)).to_parquet(path)
    with pytest.raises(ValueError, match="no rows for tasks"):
        RulerSFTDataset(RulerSFTConfig(source=path, tasks=("qa_1",)), FakeTokenizer())


def test_a_missing_column_names_the_schema(tmp_path):
    path = tmp_path / "bad.parquet"
    make_frame().drop(columns=["answer_prefix"]).to_parquet(path)
    with pytest.raises(ValueError, match="missing \\['answer_prefix'\\]"):
        RulerSFTDataset(RulerSFTConfig(source=path), FakeTokenizer())


def test_append_eos_is_off_by_default_and_adds_one_token_when_on(frame_path):
    plain = RulerSFTDataset(RulerSFTConfig(source=frame_path, max_len=10_000), FakeTokenizer())
    with_eos = RulerSFTDataset(
        RulerSFTConfig(source=frame_path, max_len=10_000, append_eos=True), FakeTokenizer()
    )
    first_plain = next(iter(plain))
    first_eos = next(iter(with_eos))
    assert first_eos["n_target"] == first_plain["n_target"] + 1
    assert int(first_eos["input_ids"][-1]) == FakeTokenizer.eos_token_id


# ----------------------------------------------------------------------
# Collate: refuse to pad rather than mis-normalize the gate
# ----------------------------------------------------------------------
def test_collate_refuses_a_ragged_batch_and_names_the_fix():
    a = {"input_ids": torch.zeros(5, dtype=torch.long), "labels": torch.zeros(5, dtype=torch.long),
         "task": "vt", "n_target": 1, "n_prompt": 4}
    b = {"input_ids": torch.zeros(7, dtype=torch.long), "labels": torch.zeros(7, dtype=torch.long),
         "task": "vt", "n_target": 1, "n_prompt": 6}
    with pytest.raises(ValueError, match="ragged SFT batch"):
        sft_collate([a, b])


def test_collate_stacks_equal_length_samples():
    item = {"input_ids": torch.arange(5), "labels": torch.arange(5),
            "task": "vt", "n_target": 2, "n_prompt": 3}
    batch = sft_collate([item, item])
    assert batch["input_ids"].shape == (2, 5)
    assert batch["labels"].shape == (2, 5)
    assert batch["tasks"] == ["vt", "vt"]


def test_dataloader_shards_rows_across_ranks_without_overlap(frame_path):
    """
    Ranks must not see the same row, and together they must cover every row -- the loader's
    contract for data-parallel replicas.
    """
    seen = []
    for rank in range(2):
        loader = build_ruler_sft_dataloader(
            RulerSFTConfig(source=frame_path, max_len=10_000, shuffle=False),
            FakeTokenizer(),
            num_workers=0,
            rank=rank,
            world_size=2,
        )
        seen.append([batch["tasks"][0] for batch in loader])
    assert not set(seen[0]) & set(seen[1])
    assert len(seen[0]) + len(seen[1]) == len(ALL_TASKS)


# ----------------------------------------------------------------------
# The real prompt shape. This is what a reimplementation would get wrong.
# ----------------------------------------------------------------------
QWEN = "/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B"
SNAPSHOT = (
    "/data/home/marcushaogu/.cache/huggingface/hub/datasets--simonjegou--ruler/"
    "snapshots/24adceac8a0e6532936e8d721cd9e9084d2e4686"
)


@pytest.fixture(scope="module")
def real_tokenizer():
    transformers = pytest.importorskip("transformers")
    if not __import__("pathlib").Path(QWEN).is_dir():
        pytest.skip(f"no local model at {QWEN}")
    return transformers.AutoTokenizer.from_pretrained(QWEN)


def test_prompt_matches_pipeline_preprocess(real_tokenizer):
    """
    The SFT prompt must be **token-identical** to what the eval pipeline builds.

    This is the invariant the module exists to protect: the eval's ``preprocess`` is called
    directly, so any drift in the chat template, the ``enable_thinking`` default or the
    question/answer_prefix order would show up here rather than as a router trained against a
    prompt shape it never sees.
    """
    from kvpress.pipeline import KVPressTextGenerationPipeline

    context, question, prefix = "The magic number is 42. " * 10, "What is it? ", "It is"

    class Shim:
        tokenizer = real_tokenizer

    expected = KVPressTextGenerationPipeline.preprocess(
        Shim(), context=context, questions=[question],
        answer_prefix=prefix, max_context_length=10**9,
    )
    reference = torch.cat([expected["context_ids"][0], expected["questions_ids"][0][0]])
    assert torch.equal(build_prompt_ids(real_tokenizer, context, question, prefix), reference)


def test_prompt_ends_with_the_answer_prefix(real_tokenizer):
    """The first supervised position must be the token right after answer_prefix."""
    ids = build_prompt_ids(real_tokenizer, "ctx. ", "Q? ", "Answer:")
    assert real_tokenizer.decode(ids).endswith("Answer:")


@pytest.mark.skipif(
    not __import__("pathlib").Path(SNAPSHOT).is_dir(), reason="no local RULER snapshot"
)
def test_real_ruler_rows_mask_only_the_answer(real_tokenizer):
    """
    On the real 16384 config: every task that fits yields a supervised span that decodes to the
    gold answer, and the prompt is fully masked.
    """
    config = RulerSFTConfig(
        source=SNAPSHOT, config="16384", max_len=24576, max_rows=200, seed=0
    )
    dataset = RulerSFTDataset(config, real_tokenizer)
    samples = list(dataset)
    assert samples, "no RULER row fit under max_len=24576"
    for sample in samples:
        supervised = sample["labels"][sample["labels"] != -100]
        assert (sample["labels"][: sample["n_prompt"]] == -100).all()
        assert supervised.numel() == sample["n_target"] > 0
        # The supervised span is the gold answer, not a fragment of the prompt.
        assert real_tokenizer.decode(supervised).strip()


@pytest.mark.skipif(
    not __import__("pathlib").Path(SNAPSHOT).is_dir(), reason="no local RULER snapshot"
)
def test_cwe_is_the_task_a_16k_cap_drops(real_tokenizer):
    """
    Pins the measured length skew the module docstring claims: at ``max_len=16384`` on the
    ``16384`` config, ``cwe`` (median 22.4K prompt tokens) is dropped outright while ``fwe``
    (15.1K) largely survives. This is why the drop table is printed before a run -- otherwise a
    task simply goes missing from the training mix with no signal.
    """
    config = RulerSFTConfig(
        source=SNAPSHOT, config="16384", tasks=("cwe", "fwe"), max_len=16384, seed=0
    )
    dataset = RulerSFTDataset(config, real_tokenizer)
    list(dataset)
    assert dataset.stats["cwe"][0] == 0, "cwe should not fit under a 16384 cap"
    assert dataset.stats["fwe"][0] > 0, "fwe should mostly fit under a 16384 cap"
