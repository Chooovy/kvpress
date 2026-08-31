# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CLI-level tests for the RULER SFT wiring in ``scripts.train_gqa_indexer_e2e``.

The data path itself is tested in ``test_gqa_indexer_sft_data.py``. What is pinned here is the
argument validation, because every one of these combinations would otherwise fail *late* and
plausibly:

* ``--stage sparse`` would train with an identically-zero gradient on every unselected key and
  still produce a descending loss curve.
* ``--batch-size 2`` would only raise once two RULER rows of different length reached the collate.
* a bad ``--sft-tasks`` name would filter the frame to nothing.
* a missing ``--sft-config`` would glob a snapshot root and find no parquet.

Each test asserts the *message* names the fix, not just that it exits: these are the errors a
person hits at launch time, hours before a run would otherwise fall over.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def run_cli(*extra: str) -> subprocess.CompletedProcess:
    """
    Invoke the trainer's argument parsing in a subprocess.

    A subprocess rather than calling ``main()``: the parser's failures go through
    ``parser.error``, which raises ``SystemExit`` *after* writing to stderr, and the message is the
    thing under test. Validation runs before any CUDA check, so these do not need a GPU.
    """
    return subprocess.run(
        [sys.executable, "-m", "scripts.train_gqa_indexer_e2e", *extra],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


# A directory is enough for the path checks; nothing reads it before validation fails.
FAKE_RULER = "/tmp"


def test_sparse_stage_is_refused_with_a_reason():
    """
    The load-bearing guard. Under sparse scope an unselected key's gradient is exactly zero, so a
    router that misses the needle can never learn to select it -- and the run would look healthy.
    """
    result = run_cli(
        "--sft-ruler", FAKE_RULER, "--sft-config", "16384", "--stage", "sparse", "--topk", "512"
    )
    assert result.returncode != 0
    assert "needs --stage dense" in result.stderr
    # The message must explain *why*, since the config is otherwise the intuitive one.
    assert "gradient is exactly zero" in result.stderr


def test_batch_size_above_one_is_refused_with_the_alternative():
    result = run_cli("--sft-ruler", FAKE_RULER, "--sft-config", "16384", "--batch-size", "2")
    assert result.returncode != 0
    assert "needs --batch-size 1" in result.stderr
    assert "--global-batch-size" in result.stderr, "must name the way to get a larger batch"


def test_a_directory_source_requires_a_config():
    result = run_cli("--sft-ruler", FAKE_RULER)
    assert result.returncode != 0
    assert "--sft-config is needed" in result.stderr


def test_an_unknown_task_is_named_at_launch():
    result = run_cli(
        "--sft-ruler", FAKE_RULER, "--sft-config", "16384", "--sft-tasks", "niah_single_9"
    )
    assert result.returncode != 0
    assert "unknown --sft-tasks entry" in result.stderr


def test_task_groups_are_accepted():
    """
    ``niah`` / ``all`` / ``other`` must survive validation. Guarded by reaching a *later* error
    (the CUDA check) rather than success, since these tests run without a GPU.
    """
    for group in ("niah", "all", "other"):
        result = run_cli("--sft-ruler", FAKE_RULER, "--sft-config", "16384", "--sft-tasks", group)
        assert "unknown --sft-tasks entry" not in result.stderr, group
        assert "--sft-config is needed" not in result.stderr, group


def test_data_root_is_required_without_sft():
    """The longmino path keeps its old contract: no corpus, no run."""
    result = run_cli()
    assert result.returncode != 0
    assert "--data-root is required" in result.stderr


def test_data_root_is_not_required_with_sft():
    """
    ``--sft-ruler`` replaces the corpus entirely, so demanding ``--data-root`` alongside it would
    force a launch command to name a corpus the run never opens.
    """
    result = run_cli("--sft-ruler", FAKE_RULER, "--sft-config", "16384")
    assert "--data-root is required" not in result.stderr


def test_sft_flags_do_not_disturb_the_longmino_path():
    """
    A non-SFT invocation must not acquire any SFT requirement -- ``--batch-size 2`` is legal there
    (longmino samples are all exactly seq_len, so they stack without padding).

    Asserted against the *error* line rather than the whole stderr: argparse prints every flag name
    in its usage banner, so a substring search for "--sft-config" matches the banner even on a
    clean run.
    """
    result = run_cli("--data-root", FAKE_RULER, "--batch-size", "2")
    # Reached the CUDA check, i.e. passed validation -- this box has no GPU.
    assert "needs a GPU" in result.stderr
    assert "needs --batch-size 1" not in result.stderr
    assert "--sft-config is needed" not in result.stderr


def test_checkpoint_records_the_sft_provenance(tmp_path):
    """
    An SFT checkpoint must say it is one, and must record the EXPANDED task list.

    Both matter downstream. ``objective`` is how a reader tells this checkpoint from a stage-1 one
    (the weights are identical in shape and name). ``sft_tasks`` is the variable that decides
    whether a RULER score is a generalization claim or an in-distribution one, and it is not
    recoverable from the weights -- so storing the shorthand ``"niah"`` instead of the 8 task names
    would lose exactly the fact a later reader needs.
    """
    import argparse

    import torch
    from transformers import AutoModelForCausalLM

    from kvpress.presses.gqa_indexer import E2EIndexerTrainer, GQAIndexerPress
    from scripts.train_gqa_indexer_e2e import save

    model = AutoModelForCausalLM.from_pretrained("MaxJeblick/llama2-0b-unit-test").eval()
    press = GQAIndexerPress(
        compression_ratio=0.5, scorer="prefix", scalar_mid_dim=16,
        prefix_head_dim=8, prefix_value_dim=8, gate_scale=True,
    )
    press.post_init_from_model(model)
    E2EIndexerTrainer(press=press, stage="dense", pin_mode="sink", n_sink=2).freeze_backbone(model)

    args = argparse.Namespace(
        scorer_attr="indexer", model="Qwen3-8B", scorer="prefix", scalar_mid_dim=256,
        scalar_pos_slope=1e-6, prefix_head_dim=128, prefix_value_dim=128, prefix_zero_init=True,
        stage="dense", pin_mode="sink", n_sink=4, gate_budget=1.0, gate_budget_ratio=None,
        schedule="24576:400", subsets=[], topk=0, peak_lr=1e-4, final_lr=5e-6, seed=0,
        save_optimizer=False, sft_ruler="/snap", sft_config="16384", sft_tasks=["niah"],
        sft_max_len=24576, sft_append_eos=False,
    )
    path = tmp_path / "final.pt"
    save(path, model, args, 400)

    config = torch.load(path, map_location="cpu", weights_only=False)["config"]
    assert config["objective"] == "ruler_sft_answer_only"
    assert config["sft_config"] == "16384"
    assert config["sft_max_len"] == 24576
    # Expanded, not the shorthand.
    assert len(config["sft_tasks"]) == 8
    assert "niah_multivalue" in config["sft_tasks"]
    assert "cwe" not in config["sft_tasks"]


def test_a_non_sft_checkpoint_still_says_e2e(tmp_path):
    """The longmino objective's provenance must not change, or old checkpoints stop matching."""
    import argparse

    import torch
    from transformers import AutoModelForCausalLM

    from kvpress.presses.gqa_indexer import E2EIndexerTrainer, GQAIndexerPress
    from scripts.train_gqa_indexer_e2e import save

    model = AutoModelForCausalLM.from_pretrained("MaxJeblick/llama2-0b-unit-test").eval()
    press = GQAIndexerPress(compression_ratio=0.5, scorer="scalar", scalar_mid_dim=16,
                            gate_scale=True)
    press.post_init_from_model(model)
    E2EIndexerTrainer(press=press, stage="dense", pin_mode="sink", n_sink=2).freeze_backbone(model)
    args = argparse.Namespace(
        scorer_attr="indexer", model="Qwen3-8B", scorer="scalar", scalar_mid_dim=256,
        scalar_pos_slope=1e-6, prefix_head_dim=None, prefix_value_dim=None, prefix_zero_init=None,
        stage="dense", pin_mode="sink", n_sink=4, gate_budget=1.0, gate_budget_ratio=None,
        schedule="8192:300", subsets=["2e16"], topk=0, peak_lr=1e-3, final_lr=5e-6, seed=0,
        save_optimizer=False, sft_ruler=None, sft_config=None, sft_tasks=None,
        sft_max_len=None, sft_append_eos=False,
    )
    path = tmp_path / "final.pt"
    save(path, model, args, 300)
    config = torch.load(path, map_location="cpu", weights_only=False)["config"]
    assert config["objective"] == "e2e_lm_loss"
    assert config["sft_ruler"] is None and config["sft_tasks"] is None
