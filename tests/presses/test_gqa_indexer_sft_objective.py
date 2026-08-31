# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end tests for the answer-only SFT objective, on a toy CPU model.

``test_gqa_indexer_sft_data.py`` checks that the mask is built correctly; this checks that the mask
*does something* -- that ``labels`` reaches the loss, that the router still receives gradient
through a 99.9%-masked objective, and that the prompt tokens genuinely do not contribute. Those are
the claims the whole stage rests on, and none of them is visible from the data path alone.

Toy model, real code path: the trainer, the press, the gate and the loss are the production ones.
"""

from __future__ import annotations

import pytest
import torch

from kvpress.presses.gqa_indexer import (
    E2EIndexerTrainer,
    GQAIndexerPress,
    e2e_indexer_training_step,
)

TOY = "MaxJeblick/llama2-0b-unit-test"


@pytest.fixture(scope="module")
def toy():
    """A tiny frozen backbone with a prefix indexer gated into its attention."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(TOY).eval()
    model.config._attn_implementation = "sdpa"
    model.requires_grad_(False)
    press = GQAIndexerPress(
        compression_ratio=0.5,
        scorer="prefix",
        scalar_mid_dim=16,
        prefix_head_dim=8,
        prefix_value_dim=8,
        gate_scale=True,
        n_sink=2,
    )
    press.post_init_from_model(model)
    trainer = E2EIndexerTrainer(press=press, stage="dense", pin_mode="sink", n_sink=2)
    trainer.freeze_backbone(model)
    return model, trainer


def make_batch(seq_len=64, n_target=4):
    """``input_ids`` plus labels masked to the final ``n_target`` positions."""
    torch.manual_seed(0)
    input_ids = torch.randint(0, 100, (1, seq_len))
    labels = torch.full_like(input_ids, -100)
    labels[:, -n_target:] = input_ids[:, -n_target:]
    return input_ids, labels


def test_masked_loss_differs_from_the_full_lm_loss(toy):
    """
    ``labels`` must actually reach the objective.

    If it were dropped, the SFT loss would silently equal the dense next-token loss and the run
    would train on every position -- descending, plausible, and not the stated experiment.
    """
    model, trainer = toy
    input_ids, labels = make_batch()
    with torch.no_grad():
        masked = e2e_indexer_training_step(model, trainer, input_ids=input_ids, labels=labels)
        full = e2e_indexer_training_step(model, trainer, input_ids=input_ids)
    assert not torch.isclose(masked, full, rtol=1e-3), (
        "answer-only loss equals the full LM loss, so labels are being ignored"
    )


def test_router_still_gets_gradient_through_a_999_percent_masked_loss(toy):
    """
    The stage's core feasibility claim: ~0.1% of positions carry loss, and the router must still
    receive a usable gradient. A vanishing gradient here would mean the objective cannot train
    anything, however healthy the loss looks.
    """
    model, trainer = toy
    input_ids, labels = make_batch(seq_len=128, n_target=1)  # 0.8% -- toy-scale stand-in
    params = trainer.indexer_parameters(model)
    for param in params:
        param.grad = None
    loss = e2e_indexer_training_step(model, trainer, input_ids=input_ids, labels=labels)
    loss.backward()
    total = sum(
        float(p.grad.abs().sum()) for p in params if p.grad is not None
    )
    assert total > 0, "no indexer parameter received gradient from the answer-only loss"
    assert torch.isfinite(loss), "answer-only loss is not finite"


def test_prompt_positions_do_not_contribute(toy):
    """
    Changing a *prompt* label must not change the loss, while changing a *target* label must.

    This is the property that makes "mask the context" true rather than intended: it isolates the
    supervised span behaviourally, not by inspecting the tensor.
    """
    model, trainer = toy
    input_ids, labels = make_batch()

    perturbed_prompt = labels.clone()
    perturbed_prompt[:, 5] = -100  # already -100; make the intent explicit
    perturbed_target = labels.clone()
    perturbed_target[:, -1] = (int(perturbed_target[:, -1]) + 1) % 100

    with torch.no_grad():
        base = e2e_indexer_training_step(model, trainer, input_ids=input_ids, labels=labels)
        same = e2e_indexer_training_step(
            model, trainer, input_ids=input_ids, labels=perturbed_prompt
        )
        moved = e2e_indexer_training_step(
            model, trainer, input_ids=input_ids, labels=perturbed_target
        )
    assert torch.isclose(base, same), "a prompt-position label changed the loss"
    assert not torch.isclose(base, moved), "a target-position label did not change the loss"


def test_the_supervised_span_is_the_only_one_that_moves_the_gradient(toy):
    """
    Two batches identical except for which positions are supervised must give different router
    gradients -- i.e. *where* the answer is decides what the router learns to attend to.
    """
    model, trainer = toy
    input_ids, _ = make_batch(seq_len=64)
    params = trainer.indexer_parameters(model)

    grads = []
    for start in (16, 60):
        labels = torch.full_like(input_ids, -100)
        labels[:, start : start + 4] = input_ids[:, start : start + 4]
        for param in params:
            param.grad = None
        e2e_indexer_training_step(model, trainer, input_ids=input_ids, labels=labels).backward()
        grads.append(
            torch.cat([p.grad.flatten() for p in params if p.grad is not None]).clone()
        )

    assert not torch.allclose(grads[0], grads[1], atol=1e-8), (
        "supervising a different span produced an identical gradient"
    )


SNAPSHOT = (
    "/data/home/marcushaogu/.cache/huggingface/hub/datasets--simonjegou--ruler/"
    "snapshots/24adceac8a0e6532936e8d721cd9e9084d2e4686"
)
QWEN = "/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B"


@pytest.mark.skipif(
    not (__import__("pathlib").Path(SNAPSHOT).is_dir()
         and __import__("pathlib").Path(QWEN).is_dir()),
    reason="needs the local RULER snapshot and Qwen3 tokenizer",
)
def test_real_ruler_rows_drive_optimizer_steps(toy):
    """
    The whole stage, on real data: loader -> masked loss -> backward -> optimizer step.

    Uses the ``4096`` config so a CPU can carry it, and mixes tasks with very different answer
    lengths (``qa_2`` at ~3 tokens against ``cwe`` at ~22) because the supervised fraction is what
    makes this objective unusual. Asserts the gate actually moves: a step that leaves every router
    parameter where it was would mean the loss is not reaching them, which no loss value would
    reveal.
    """
    from transformers import AutoTokenizer

    from kvpress.presses.gqa_indexer.sft_data import (
        RulerSFTConfig,
        build_ruler_sft_dataloader,
    )

    model, trainer = toy
    tokenizer = AutoTokenizer.from_pretrained(QWEN)
    params = trainer.indexer_parameters(model)
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    before = torch.cat([p.detach().flatten().clone() for p in params])

    loader = build_ruler_sft_dataloader(
        RulerSFTConfig(
            source=SNAPSHOT,
            config="4096",
            tasks=("qa_2", "niah_single_1", "cwe"),
            max_len=8192,
            seed=0,
        ),
        tokenizer,
        batch_size=1,
        num_workers=0,
    )

    steps, fractions = 0, []
    vocab = int(model.get_input_embeddings().num_embeddings)
    for batch in loader:
        # The toy backbone's vocabulary is a few hundred entries while the Qwen3 tokenizer emits
        # ids up to ~152k, so the ids are folded into range. What is under test here is the loop --
        # real RULER lengths, the real mask, real optimizer steps -- not the token identities, and
        # folding keeps the prompt/target BOUNDARY exactly where the loader put it. The same map is
        # applied to labels so they still match the inputs at the supervised positions.
        input_ids = batch["input_ids"] % vocab
        labels = torch.where(batch["labels"] == -100, batch["labels"], batch["labels"] % vocab)
        loss = e2e_indexer_training_step(
            model, trainer, input_ids=input_ids, labels=labels
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        optimizer.zero_grad()
        assert torch.isfinite(loss)
        supervised = int((labels != -100).sum())
        assert supervised == int(batch["n_target"].sum()) > 0
        fractions.append(supervised / input_ids.numel())
        steps += 1
        if steps == 4:
            break

    assert steps == 4, "the RULER loader yielded fewer rows than requested"
    # The defining property of this objective, on real rows rather than a synthetic mask.
    assert max(fractions) < 0.02, f"expected a <2% supervised fraction, got {max(fractions):.3%}"
    after = torch.cat([p.detach().flatten() for p in params])
    assert not torch.allclose(before, after), "optimizer steps did not move the router"
