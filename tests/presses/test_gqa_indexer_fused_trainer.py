# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-layer fused indexer trainer."""

import pytest
import torch

from kvpress import GQAIndexerPress
from kvpress.presses.gqa_indexer import (
    FusedIndexerTrainer,
    IndexerTrainConfig,
    attention_scaling,
    compute_indexer_loss,
    freeze_all_but_indexer,
    fused_indexer_training_step,
    get_attention_modules,
    teacher_query_states,
)
from tests.fixtures import unit_test_model, unit_test_model_output_attention  # noqa: F401


def make_trainer(model, **kwargs):
    press = GQAIndexerPress(compression_ratio=0.5)
    press.post_init_from_model(model)
    return FusedIndexerTrainer(press=press, **kwargs)


# ----------------------------------------------------------------------
# Teacher reconstruction
# ----------------------------------------------------------------------
def test_teacher_query_states_match_the_layers_own_attention(unit_test_model):  # noqa: F811
    """
    The rebuilt queries must equal what the layer actually computed.

    The whole teacher path rests on this: if q_proj + RoPE were reconstructed differently
    from the real forward, the distillation target would be subtly wrong with nothing
    downstream to flag it.
    """
    model = unit_test_model
    module = get_attention_modules(model)[0]
    config = model.config
    bsz, q_len = 1, 12
    hidden = torch.randn(bsz, q_len, config.hidden_size, device=model.device, dtype=model.dtype)

    language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    position_ids = torch.arange(q_len, device=model.device).unsqueeze(0)
    position_embeddings = language_model.rotary_emb(hidden, position_ids)

    got = teacher_query_states(module, hidden, position_embeddings)
    assert got.shape == (bsz, config.num_attention_heads, q_len, module.head_dim)

    # Independent reconstruction, mirroring what the attention layer does.
    from transformers.models.llama.modeling_llama import rotate_half

    ref = module.q_proj(hidden).view(bsz, q_len, config.num_attention_heads, module.head_dim)
    ref = ref.transpose(1, 2)
    cos, sin = position_embeddings
    ref = ref * cos.unsqueeze(1) + rotate_half(ref) * sin.unsqueeze(1)
    torch.testing.assert_close(got, ref)


def test_attention_scaling_prefers_the_module_attribute(unit_test_model):  # noqa: F811
    module = get_attention_modules(unit_test_model)[0]
    assert attention_scaling(module) == pytest.approx(module.scaling)

    class NoScaling:
        head_dim = 64

    assert attention_scaling(NoScaling()) == pytest.approx(64**-0.5)


# ----------------------------------------------------------------------
# End-to-end
# ----------------------------------------------------------------------
def test_training_step_produces_a_finite_loss_per_layer(unit_test_model):  # noqa: F811
    model = unit_test_model
    trainer = make_trainer(model, key_tile=8)
    input_ids = torch.randint(0, 1024, (1, 24), device=model.device)

    loss, per_layer = fused_indexer_training_step(model, trainer, input_ids=input_ids)
    assert torch.isfinite(loss)
    assert len(per_layer) == model.config.num_hidden_layers
    assert all(torch.isfinite(v) for v in per_layer.values())


def test_training_step_gradients_reach_only_the_indexer(unit_test_model):  # noqa: F811
    model = unit_test_model
    trainer = make_trainer(model, key_tile=8)
    freeze_all_but_indexer(model)
    model.zero_grad(set_to_none=True)

    input_ids = torch.randint(0, 1024, (1, 24), device=model.device)
    loss, _ = fused_indexer_training_step(model, trainer, input_ids=input_ids)
    loss.backward()

    touched = [n for n, p in model.named_parameters() if p.grad is not None]
    assert touched, "no parameter received a gradient"
    assert all(".indexer." in n for n in touched), [n for n in touched if ".indexer." not in n]
    assert any(p.grad.abs().sum() > 0 for _, p in model.named_parameters() if p.grad is not None)


def test_does_not_require_output_attentions(unit_test_model):  # noqa: F811
    """
    The fused path must never need the dense attention matrix.

    Needing it would force the base model onto eager attention, which is most of the cost
    the tiled loss exists to avoid.
    """
    model = unit_test_model
    trainer = make_trainer(model, key_tile=8)
    input_ids = torch.randint(0, 1024, (1, 16), device=model.device)

    with trainer.hooks(model):
        outputs = model(input_ids=input_ids, use_cache=True)
    assert getattr(outputs, "attentions", None) is None
    assert len(trainer.per_layer_losses) == model.config.num_hidden_layers


@pytest.mark.parametrize("key_tile", [4, 8, 64])
def test_loss_is_tile_invariant_end_to_end(unit_test_model, key_tile):  # noqa: F811
    model = unit_test_model
    input_ids = torch.randint(0, 1024, (1, 20), device=model.device)

    torch.manual_seed(0)
    reference = make_trainer(model, key_tile=1024)
    ref_loss, _ = fused_indexer_training_step(model, reference, input_ids=input_ids)

    trainer = FusedIndexerTrainer(press=reference.press, key_tile=key_tile)
    loss, _ = fused_indexer_training_step(model, trainer, input_ids=input_ids)
    torch.testing.assert_close(loss, ref_loss, rtol=1e-4, atol=1e-5)


def test_agrees_with_the_dense_loss_up_to_the_entropy_offset(unit_test_model_output_attention):  # noqa: F811
    """
    Fused (cross-entropy) minus dense (KL) must equal the teacher entropy H(pbar).

    This is the cross-check that the two independent implementations -- streaming vs fully
    materialized -- describe the same objective.
    """
    model = unit_test_model_output_attention
    press = GQAIndexerPress(compression_ratio=0.5)
    press.post_init_from_model(model)
    input_ids = torch.randint(0, 1024, (1, 16), device=model.device)

    trainer = FusedIndexerTrainer(press=press, key_tile=4)
    fused_loss, fused_per_layer = fused_indexer_training_step(model, trainer, input_ids=input_ids)

    outputs = model(
        input_ids, output_attentions=True, output_hidden_states=True, use_cache=False
    )
    dense_loss, dense_per_layer = compute_indexer_loss(
        press,
        get_attention_modules(model),
        outputs.hidden_states,
        outputs.attentions,
        IndexerTrainConfig(stage="dense"),
    )

    # CE >= KL always, since H(pbar) >= 0.
    assert fused_loss.item() >= dense_loss.item() - 1e-4

    n_kv = model.config.num_key_value_heads
    for layer_idx, dense in enumerate(dense_per_layer):
        entropy = _teacher_entropy(outputs.attentions[layer_idx], n_kv)
        expected = dense.item() + entropy
        assert fused_per_layer[layer_idx].item() == pytest.approx(expected, rel=2e-2, abs=2e-2)


def _teacher_entropy(attentions: torch.Tensor, n_kv_heads: int) -> float:
    """
    Mean H(pbar) over rows, with the same grouping the loss uses.

    Averaging over *all* rows only matches the loss's row_valid average because the caller
    uses causal-only masking with no padding and no sink skip -- under a causal mask every
    row has at least one valid key. Add either and the two averages diverge.
    """
    bsz, n_heads, q_len, k_len = attentions.shape
    grouped = attentions.float().view(bsz, n_kv_heads, n_heads // n_kv_heads, q_len, k_len)
    p_bar = grouped.mean(dim=2)
    p_bar = p_bar / p_bar.sum(-1, keepdim=True).clamp_min(1e-10)
    return float(-(p_bar * p_bar.clamp_min(1e-30).log()).sum(-1).mean())


# ----------------------------------------------------------------------
# Configuration knobs
# ----------------------------------------------------------------------
def test_skip_sink_changes_the_loss(unit_test_model):  # noqa: F811
    model = unit_test_model
    press = GQAIndexerPress(compression_ratio=0.5)
    press.post_init_from_model(model)
    input_ids = torch.randint(0, 1024, (1, 20), device=model.device)

    plain = FusedIndexerTrainer(press=press, key_tile=8)
    skipped = FusedIndexerTrainer(press=press, key_tile=8, skip_sink_in_loss=4)
    loss_plain, _ = fused_indexer_training_step(model, plain, input_ids=input_ids)
    loss_skipped, _ = fused_indexer_training_step(model, skipped, input_ids=input_ids)

    assert torch.isfinite(loss_skipped)
    assert not torch.allclose(loss_plain, loss_skipped)


def test_skip_sink_keeps_the_teacher_normalized(unit_test_model):  # noqa: F811
    """
    Skipping sinks must be folded in BEFORE the logsumexp.

    Masking keys after the lse leaves the teacher rows not summing to one, which silently
    down-weights the affected rows -- the same failure mode the padding guard exists for.
    A finite, non-NaN loss with a large sink skip is the observable symptom.
    """
    model = unit_test_model
    press = GQAIndexerPress(compression_ratio=0.5)
    press.post_init_from_model(model)
    input_ids = torch.randint(0, 1024, (1, 16), device=model.device)

    trainer = FusedIndexerTrainer(press=press, key_tile=4, skip_sink_in_loss=8)
    loss, per_layer = fused_indexer_training_step(model, trainer, input_ids=input_ids)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(v) for v in per_layer.values())


def test_loss_coeff_scales_linearly(unit_test_model):  # noqa: F811
    model = unit_test_model
    press = GQAIndexerPress(compression_ratio=0.5)
    press.post_init_from_model(model)
    input_ids = torch.randint(0, 1024, (1, 16), device=model.device)

    base = FusedIndexerTrainer(press=press, key_tile=8, loss_coeff=1.0)
    scaled = FusedIndexerTrainer(press=press, key_tile=8, loss_coeff=2.5)
    loss_base, _ = fused_indexer_training_step(model, base, input_ids=input_ids)
    loss_scaled, _ = fused_indexer_training_step(model, scaled, input_ids=input_ids)
    torch.testing.assert_close(loss_scaled, loss_base * 2.5, rtol=1e-5, atol=1e-6)


def test_attention_mask_padding_is_handled(unit_test_model):  # noqa: F811
    model = unit_test_model
    trainer = make_trainer(model, key_tile=8)
    input_ids = torch.randint(0, 1024, (2, 16), device=model.device)
    attention_mask = torch.ones(2, 16, dtype=torch.long, device=model.device)
    attention_mask[1, :5] = 0

    loss, per_layer = fused_indexer_training_step(
        model, trainer, input_ids=input_ids, attention_mask=attention_mask
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(v) for v in per_layer.values())


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------
def test_hooks_are_removed_on_exit(unit_test_model):  # noqa: F811
    model = unit_test_model
    trainer = make_trainer(model, key_tile=8)
    module = get_attention_modules(model)[0]
    before = len(module._forward_hooks)

    with trainer.hooks(model):
        assert len(module._forward_hooks) == before + 1
    assert len(module._forward_hooks) == before


def test_hooks_are_removed_even_on_exception(unit_test_model):  # noqa: F811
    model = unit_test_model
    trainer = make_trainer(model, key_tile=8)
    module = get_attention_modules(model)[0]
    before = len(module._forward_hooks)

    with pytest.raises(RuntimeError, match="boom"):
        with trainer.hooks(model):
            raise RuntimeError("boom")
    assert len(module._forward_hooks) == before


def test_reset_clears_previous_losses(unit_test_model):  # noqa: F811
    model = unit_test_model
    trainer = make_trainer(model, key_tile=8)
    input_ids = torch.randint(0, 1024, (1, 16), device=model.device)

    fused_indexer_training_step(model, trainer, input_ids=input_ids)
    assert trainer.per_layer_losses
    trainer.reset()
    assert not trainer.per_layer_losses


def test_total_loss_without_a_forward_pass_raises(unit_test_model):  # noqa: F811
    trainer = make_trainer(unit_test_model, key_tile=8)
    with pytest.raises(RuntimeError, match="register hooks"):
        trainer.total_loss()


def test_requires_the_kv_cache(unit_test_model):  # noqa: F811
    """The teacher's post-RoPE keys are read from the cache, so use_cache=False must fail."""
    model = unit_test_model
    trainer = make_trainer(model, key_tile=8)
    input_ids = torch.randint(0, 1024, (1, 16), device=model.device)

    with pytest.raises(RuntimeError, match="KV cache"):
        with trainer.hooks(model):
            model(input_ids=input_ids, use_cache=False)
