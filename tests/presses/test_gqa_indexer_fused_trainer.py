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
    build_position_embeddings,
    get_input_layernorms,
    teacher_query_states,
)
from kvpress.presses.gqa_indexer import HAS_TRITON, build_indexer_mask, decompose_mask
from kvpress.presses.gqa_indexer.indexer import MASK_NEG
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
        model=model,
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


# ----------------------------------------------------------------------
# Stage 2: sparse
# ----------------------------------------------------------------------
def test_sparse_stage_produces_a_finite_non_negative_loss(unit_test_model):  # noqa: F811
    """Stage 2 reports full KL, so it must be non-negative -- a check stage 1's CE cannot make."""
    model = unit_test_model
    trainer = make_trainer(model, stage="sparse", key_tile=8, query_tile=8, topk_tile=4, keep_ratio=0.5)
    input_ids = torch.randint(0, 1024, (1, 24), device=model.device)

    loss, per_layer = fused_indexer_training_step(model, trainer, input_ids=input_ids)
    assert torch.isfinite(loss)
    assert len(per_layer) == model.config.num_hidden_layers
    assert all(v.item() >= -1e-5 for v in per_layer.values()), (
        f"negative KL: {min(v.item() for v in per_layer.values())}"
    )


def test_sparse_stage_gradients_reach_only_the_indexer(unit_test_model):  # noqa: F811
    model = unit_test_model
    trainer = make_trainer(model, stage="sparse", key_tile=8, topk_tile=4, keep_ratio=0.5)
    freeze_all_but_indexer(model)
    model.zero_grad(set_to_none=True)

    input_ids = torch.randint(0, 1024, (1, 24), device=model.device)
    loss, _ = fused_indexer_training_step(model, trainer, input_ids=input_ids)
    loss.backward()

    touched = [n for n, p in model.named_parameters() if p.grad is not None]
    assert touched, "no parameter received a gradient"
    assert all(".indexer." in n for n in touched), [n for n in touched if ".indexer." not in n]
    assert any(p.grad.abs().sum() > 0 for _, p in model.named_parameters() if p.grad is not None)


@pytest.mark.parametrize("query_tile,topk_tile", [(4, 2), (8, 4), (64, 64)])
def test_sparse_loss_is_tile_invariant_end_to_end(unit_test_model, query_tile, topk_tile):  # noqa: F811
    model = unit_test_model
    input_ids = torch.randint(0, 1024, (1, 20), device=model.device)

    reference = make_trainer(
        model, stage="sparse", key_tile=1024, query_tile=1024, topk_tile=1024, keep_ratio=0.5
    )
    ref_loss, _ = fused_indexer_training_step(model, reference, input_ids=input_ids)

    trainer = FusedIndexerTrainer(
        press=reference.press,
        stage="sparse",
        key_tile=1024,
        query_tile=query_tile,
        topk_tile=topk_tile,
        keep_ratio=0.5,
    )
    loss, _ = fused_indexer_training_step(model, trainer, input_ids=input_ids)
    torch.testing.assert_close(loss, ref_loss, rtol=1e-4, atol=1e-5)


def test_sparse_stage_with_full_keep_ratio_matches_the_dense_kl(unit_test_model):  # noqa: F811
    """
    keep_ratio=1.0 makes the support the whole key axis, so stage 2 must equal stage 1's KL.

    Stage 1 optimizes cross-entropy, which sits above the KL by H(pbar), so the two losses
    differ by a positive offset rather than matching -- checked as an inequality plus
    finiteness, since the entropy itself is not recoverable from the fused path.
    """
    model = unit_test_model
    input_ids = torch.randint(0, 1024, (1, 16), device=model.device)

    sparse = make_trainer(model, stage="sparse", key_tile=8, topk_tile=8, keep_ratio=1.0)
    sparse_loss, _ = fused_indexer_training_step(model, sparse, input_ids=input_ids)

    dense = FusedIndexerTrainer(press=sparse.press, stage="dense", key_tile=8)
    dense_loss, _ = fused_indexer_training_step(model, dense, input_ids=input_ids)

    assert torch.isfinite(sparse_loss) and sparse_loss.item() >= -1e-5
    assert dense_loss.item() > sparse_loss.item(), (
        "stage 1's cross-entropy must exceed stage 2's KL by the teacher entropy"
    )


def test_sparse_stage_reports_recall(unit_test_model):  # noqa: F811
    """
    Recall says whether topk is large enough; the loss alone never reveals that.

    A tiny support can show a healthy-looking KL while ignoring most of the teacher's mass,
    so this number is the one that has to be watched during stage 2.
    """
    model = unit_test_model
    input_ids = torch.randint(0, 1024, (1, 24), device=model.device)

    narrow = make_trainer(model, stage="sparse", key_tile=8, topk_tile=4, topk=2)
    fused_indexer_training_step(model, narrow, input_ids=input_ids)
    wide = FusedIndexerTrainer(
        press=narrow.press, stage="sparse", key_tile=8, topk_tile=8, keep_ratio=1.0
    )
    fused_indexer_training_step(model, wide, input_ids=input_ids)

    assert narrow.mean_recall() is not None
    assert 0.0 < narrow.mean_recall() <= 1.0 + 1e-6
    assert wide.mean_recall() == pytest.approx(1.0, abs=1e-4), (
        "a full support must capture all of the teacher's mass"
    )
    assert narrow.mean_recall() < wide.mean_recall()


def test_dense_stage_reports_no_recall(unit_test_model):  # noqa: F811
    model = unit_test_model
    trainer = make_trainer(model, key_tile=8)
    fused_indexer_training_step(model, trainer, input_ids=torch.randint(0, 1024, (1, 16), device=model.device))
    assert trainer.mean_recall() is None


def test_support_teacher_mode_needs_no_dense_lse(unit_test_model):  # noqa: F811
    """
    teacher_mode='support' normalizes over the support, making stage 2 O(L * topk) end to end.

    The two modes coincide when group_size == 1 -- and this fixture has
    num_attention_heads == num_key_value_heads, so it does. With one head per group the group
    mean is trivial, and renormalizing a full softmax onto the support IS the support softmax
    (verified to 1.7e-16). So agreement is what gets asserted here; the *divergence* at
    group_size > 1 is pinned in test_gqa_indexer_fused_sparse_loss.py, which controls the
    geometry directly instead of inheriting it from a fixture.
    """
    model = unit_test_model
    input_ids = torch.randint(0, 1024, (1, 20), device=model.device)
    group_size = model.config.num_attention_heads // model.config.num_key_value_heads

    glob = make_trainer(model, stage="sparse", key_tile=8, topk_tile=4, keep_ratio=0.4)
    global_loss, _ = fused_indexer_training_step(model, glob, input_ids=input_ids)

    sup = FusedIndexerTrainer(
        press=glob.press,
        stage="sparse",
        key_tile=8,
        topk_tile=4,
        keep_ratio=0.4,
        teacher_mode="support",
    )
    support_loss, _ = fused_indexer_training_step(model, sup, input_ids=input_ids)

    assert torch.isfinite(support_loss) and support_loss.item() >= -1e-5
    assert sup.mean_recall() == pytest.approx(1.0, abs=1e-5), "support mode normalizes to Z=1"
    if group_size == 1:
        torch.testing.assert_close(support_loss, global_loss, rtol=1e-4, atol=1e-5)
    else:
        assert not torch.allclose(global_loss, support_loss)


def test_sparse_force_local_changes_the_support(unit_test_model):  # noqa: F811
    model = unit_test_model
    input_ids = torch.randint(0, 1024, (1, 20), device=model.device)

    plain = make_trainer(model, stage="sparse", key_tile=8, topk_tile=4, topk=6)
    plain_loss, _ = fused_indexer_training_step(model, plain, input_ids=input_ids)

    forced = FusedIndexerTrainer(
        press=plain.press, stage="sparse", key_tile=8, topk_tile=4, topk=6,
        force_sink=1, force_local=2,
    )
    forced_loss, _ = fused_indexer_training_step(model, forced, input_ids=input_ids)

    assert torch.isfinite(forced_loss)
    assert not torch.allclose(plain_loss, forced_loss)


def test_sparse_stage_handles_padding(unit_test_model):  # noqa: F811
    model = unit_test_model
    input_ids = torch.randint(0, 1024, (2, 16), device=model.device)
    attention_mask = torch.ones(2, 16, dtype=torch.long, device=model.device)
    attention_mask[1, :5] = 0

    trainer = make_trainer(model, stage="sparse", key_tile=4, topk_tile=4, keep_ratio=0.5)
    loss, per_layer = fused_indexer_training_step(
        model, trainer, input_ids=input_ids, attention_mask=attention_mask
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(v) for v in per_layer.values())


def test_sparse_never_selects_a_padded_key(unit_test_model):  # noqa: F811
    """
    A padded key in the support would put teacher mass on a token that does not exist.

    Verified by intercepting the support the trainer actually built -- the loss would stay
    finite either way, so it cannot be inferred from the loss value. The interception wraps
    ``streaming_topk_support`` in the trainer's own module, so it observes the real call
    rather than re-deriving one that could drift out of sync.
    """
    import kvpress.presses.gqa_indexer.fused_trainer as trainer_mod

    model = unit_test_model
    q_len = 16
    input_ids = torch.randint(0, 1024, (2, q_len), device=model.device)
    attention_mask = torch.ones(2, q_len, dtype=torch.long, device=model.device)
    attention_mask[1, :6] = 0

    seen = []
    original = trainer_mod.streaming_topk_support

    def spy(*args, **kwargs):
        support, valid = original(*args, **kwargs)
        seen.append((support, valid, kwargs.get("mask")))
        return support, valid

    trainer = make_trainer(
        model, stage="sparse", key_tile=4, query_tile=8, topk_tile=4, keep_ratio=0.5
    )
    trainer_mod.streaming_topk_support = spy
    try:
        fused_indexer_training_step(
            model, trainer, input_ids=input_ids, attention_mask=attention_mask
        )
    finally:
        trainer_mod.streaming_topk_support = original

    assert len(seen) == model.config.num_hidden_layers, "the support was not built per layer"
    for support, valid, mask in seen:
        assert mask is not None, "the trainer must pass its mask into the selection"
        keep = mask > (MASK_NEG / 2)
        keep = keep.expand(support.shape[0], support.shape[1], *keep.shape[-2:])
        picked_allowed = keep.gather(-1, support.clamp_min(0))
        assert (picked_allowed | ~valid).all(), "a masked/padded key entered the support"
        # the padded row must also end up with fewer usable slots than the clean one
        assert valid[1].sum() < valid[0].sum()


def test_unknown_stage_raises(unit_test_model):  # noqa: F811
    press = GQAIndexerPress(compression_ratio=0.5)
    press.post_init_from_model(unit_test_model)
    with pytest.raises(ValueError, match="stage must be"):
        FusedIndexerTrainer(press=press, stage="mixture")


def test_invalid_keep_ratio_raises(unit_test_model):  # noqa: F811
    press = GQAIndexerPress(compression_ratio=0.5)
    press.post_init_from_model(unit_test_model)
    with pytest.raises(ValueError, match="keep_ratio"):
        FusedIndexerTrainer(press=press, keep_ratio=0.0)
    with pytest.raises(ValueError, match="keep_ratio"):
        FusedIndexerTrainer(press=press, keep_ratio=1.5)


def test_reset_clears_recall(unit_test_model):  # noqa: F811
    model = unit_test_model
    trainer = make_trainer(model, stage="sparse", key_tile=8, topk_tile=4, keep_ratio=0.5)
    fused_indexer_training_step(model, trainer, input_ids=torch.randint(0, 1024, (1, 16), device=model.device))
    assert trainer.per_layer_recall
    trainer.reset()
    assert not trainer.per_layer_recall
    assert trainer.mean_recall() is None


def test_invalid_teacher_mode_raises_at_construction(unit_test_model):  # noqa: F811
    """
    A typo must fail immediately, not once the run reaches stage 2.

    Validating only where the mode is consumed would let a dense warmup run to completion
    with a bad config, then crash hours later at the stage switch.
    """
    press = GQAIndexerPress(compression_ratio=0.5)
    press.post_init_from_model(unit_test_model)
    with pytest.raises(ValueError, match="teacher_mode"):
        FusedIndexerTrainer(press=press, teacher_mode="mixture")


def test_negative_forced_slots_raise(unit_test_model):  # noqa: F811
    press = GQAIndexerPress(compression_ratio=0.5)
    press.post_init_from_model(unit_test_model)
    with pytest.raises(ValueError, match="non-negative"):
        FusedIndexerTrainer(press=press, force_local=-1)


# ----------------------------------------------------------------------
# Backend dispatch
# ----------------------------------------------------------------------
def test_torch_backend_is_forced(unit_test_model):  # noqa: F811
    model = unit_test_model
    trainer = make_trainer(model, backend="torch", key_tile=8)
    loss, _ = fused_indexer_training_step(
        model, trainer, input_ids=torch.randint(0, 1024, (1, 16), device=model.device)
    )
    assert torch.isfinite(loss)
    assert trainer.backend_used == "torch"


def test_auto_backend_declines_without_cuda(unit_test_model):  # noqa: F811
    """
    On CPU 'auto' must land on torch -- and say so.

    Also the case under TRITON_INTERPRET=1: the interpreter is correct but far slower than
    the PyTorch path, so 'auto' choosing it would be a pessimization dressed as an
    optimization.
    """
    model = unit_test_model
    trainer = make_trainer(model, backend="auto", key_tile=8)
    loss, _ = fused_indexer_training_step(
        model, trainer, input_ids=torch.randint(0, 1024, (1, 16), device=model.device)
    )
    assert torch.isfinite(loss)
    expected = "triton" if (torch.cuda.is_available() and HAS_TRITON) else "torch"
    assert trainer.backend_used == expected


def test_auto_and_torch_backends_agree(unit_test_model):  # noqa: F811
    """Whichever path 'auto' takes, it must produce the same number as the reference."""
    model = unit_test_model
    input_ids = torch.randint(0, 1024, (1, 20), device=model.device)

    reference = make_trainer(model, backend="torch", key_tile=1024, query_tile=1024)
    ref_loss, _ = fused_indexer_training_step(model, reference, input_ids=input_ids)

    auto = FusedIndexerTrainer(press=reference.press, backend="auto", key_tile=1024)
    loss, _ = fused_indexer_training_step(model, auto, input_ids=input_ids)
    torch.testing.assert_close(loss, ref_loss, rtol=1e-4, atol=1e-5)


def test_triton_backend_raises_rather_than_falling_back(unit_test_model):  # noqa: F811
    """
    backend='triton' must fail loudly when it cannot run.

    Falling back silently would let a benchmark measure the PyTorch path while reporting a
    kernel number -- the failure mode that makes performance work untrustworthy.
    """
    model = unit_test_model
    if torch.cuda.is_available() and HAS_TRITON:
        pytest.skip("the kernels can actually run here")

    trainer = make_trainer(model, backend="triton", key_tile=8)
    with pytest.raises(RuntimeError, match="backend='triton'"):
        fused_indexer_training_step(
            model, trainer, input_ids=torch.randint(0, 1024, (1, 16), device=model.device)
        )


def test_triton_backend_rejects_stage_two(unit_test_model):  # noqa: F811
    press = GQAIndexerPress(compression_ratio=0.5)
    press.post_init_from_model(unit_test_model)
    with pytest.raises(NotImplementedError, match="stage 1 only"):
        FusedIndexerTrainer(press=press, backend="triton", stage="sparse")


def test_unknown_backend_raises(unit_test_model):  # noqa: F811
    press = GQAIndexerPress(compression_ratio=0.5)
    press.post_init_from_model(unit_test_model)
    with pytest.raises(ValueError, match="backend must be"):
        FusedIndexerTrainer(press=press, backend="cuda")


def test_non_power_of_two_block_raises(unit_test_model):  # noqa: F811
    press = GQAIndexerPress(compression_ratio=0.5)
    press.post_init_from_model(unit_test_model)
    with pytest.raises(ValueError, match="powers of two"):
        FusedIndexerTrainer(press=press, block_m=48)


def test_sink_skip_still_decomposes_for_the_kernels(unit_test_model):  # noqa: F811
    """
    A sink skip is per-key, so it must not knock the run off the kernel path.

    Checked via decompose_mask on the trainer's own mask rather than through backend_used,
    so the assertion holds on CPU too.
    """
    model = unit_test_model
    q_len = 16
    trainer = make_trainer(model, key_tile=8, skip_sink_in_loss=4)
    mask = trainer.apply_sink_skip(
        build_indexer_mask(q_len, q_len, model.device, dtype=torch.float32)
    )
    ok, keep = decompose_mask(mask, q_len, q_len, 0)
    assert ok, "a sink skip must remain kernel-representable"
    assert keep is not None and keep[0, :4].tolist() == [0, 0, 0, 0]


def test_backend_used_is_cleared_by_reset(unit_test_model):  # noqa: F811
    model = unit_test_model
    trainer = make_trainer(model, backend="torch", key_tile=8)
    fused_indexer_training_step(
        model, trainer, input_ids=torch.randint(0, 1024, (1, 16), device=model.device)
    )
    assert trainer.backend_used == "torch"
    trainer.reset()
    assert trainer.backend_used is None


def test_dense_loss_scores_the_post_layernorm_hidden_states(unit_test_model):  # noqa: F811
    """
    The dense path's student must match the student the press actually runs.

    The decoder block applies input_layernorm before calling self_attn, and kvpress hooks
    self_attn -- so at inference the indexer sees the POST-layernorm tensor, while
    output_hidden_states[i] is the PRE-layernorm one. Training on the wrong one does not fail
    loudly: the loss falls, gradients flow, and eval is just quietly worse.
    """
    model = unit_test_model
    press = GQAIndexerPress(compression_ratio=0.5)
    press.post_init_from_model(model)
    modules = get_attention_modules(model)
    norms = get_input_layernorms(model)
    assert len(norms) == len(modules)

    input_ids = torch.randint(0, 1024, (1, 12), device=model.device)

    seen = {}

    def hook(module, args, kwargs, output):
        seen[module] = kwargs["hidden_states"].detach()
        return output

    handles = [m.register_forward_hook(hook, with_kwargs=True) for m in modules]
    try:
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True, use_cache=True)
    finally:
        for handle in handles:
            handle.remove()

    for layer_idx, module in enumerate(modules):
        pre = outputs.hidden_states[layer_idx]
        post = norms[layer_idx](pre)
        actual = seen[module]
        torch.testing.assert_close(actual, post, rtol=1e-4, atol=1e-5)
        assert not torch.allclose(actual, pre, rtol=1e-3, atol=1e-4), (
            "pre- and post-layernorm are indistinguishable here, so this test proves nothing"
        )


def test_compute_indexer_loss_warns_without_layernorms(  # noqa: F811
    unit_test_model_output_attention, caplog
):
    """
    Omitting input_layernorms is a silent correctness bug, so it must at least warn.

    Uses the eager fixture: output_attentions=True returns None under SDPA, so the
    non-eager model would trip the attentions guard before reaching the warning.
    """
    model = unit_test_model_output_attention
    press = GQAIndexerPress(compression_ratio=0.5)
    press.post_init_from_model(model)
    input_ids = torch.randint(0, 1024, (1, 12), device=model.device)
    outputs = model(input_ids, output_attentions=True, output_hidden_states=True, use_cache=False)

    with caplog.at_level("WARNING"):
        without, _ = compute_indexer_loss(
            press, get_attention_modules(model), outputs.hidden_states, outputs.attentions,
            IndexerTrainConfig(stage="dense"),
            position_embeddings=build_position_embeddings(model, outputs.hidden_states[0]),
        )
    assert "post-layernorm" in caplog.text.lower()

    with_norms, _ = compute_indexer_loss(
        press, get_attention_modules(model), outputs.hidden_states, outputs.attentions,
        IndexerTrainConfig(stage="dense"),
        model=model,
    )
    assert not torch.allclose(without, with_norms), (
        "the two students must differ, or the layernorm argument would be pointless"
    )


# ----------------------------------------------------------------------
# The capacity model (see bench_gqa_indexer_capacity.py)
# ----------------------------------------------------------------------
def test_capacity_model_is_self_consistent():
    """
    The itemized prediction must add up, and move the right way with each knob.

    Pure arithmetic, no GPU: this pins the model the benchmark compares its measurements
    against. If the model is wrong the benchmark cannot tell a real ceiling from a leak, which
    is the only thing it is for.
    """
    from tests.presses.bench_gqa_indexer_capacity import (
        predict_bytes_per_token,
        predict_max_length,
    )

    geometry = dict(layers=36, n_heads=32, n_kv_heads=8, head_dim=128, indexer_dim=128)

    dense = predict_bytes_per_token(**geometry, stage="dense")
    assert sum(v for k, v in dense.items() if k != "total") == pytest.approx(dense["total"])

    # More support costs more; sparse costs more than dense.
    small = predict_bytes_per_token(**geometry, stage="sparse", topk=256)
    large = predict_bytes_per_token(**geometry, stage="sparse", topk=512)
    assert dense["total"] < small["total"] < large["total"]

    # topk doubling must double exactly the support terms and nothing else.
    assert large["support"] == 2 * small["support"]
    assert large["kv_cache"] == small["kv_cache"]
    # `valid` is not retained -- it is exactly `support >= 0`, recomputed per tile in backward.
    assert "valid" not in large

    # Halving the layers halves everything (all terms are per-layer except none).
    half = predict_bytes_per_token(**{**geometry, "layers": 18}, stage="dense")
    assert half["total"] == pytest.approx(dense["total"] / 2)

    # Length prediction is a plain floor division, and degrades gracefully.
    assert predict_max_length(dense["total"] * 1000, dense["total"]) == 1000
    assert predict_max_length(0, dense["total"]) == 0


def test_capacity_model_prices_the_retained_teacher():
    """
    The retained teacher must be the model's dtype, and the keys must not be counted twice.

    ctx.teacher_alpha holds the teacher query with one autograd node per layer, so all 36
    layers stay resident until backward(). Two things keep that cheap: the upcast to fp32
    happens per tile rather than up front, and the keys are held by reference to the KV cache,
    which is resident anyway. Both are pinned here rather than left as docstring claims.
    """
    from tests.presses.bench_gqa_indexer_capacity import predict_bytes_per_token

    geometry = dict(layers=36, n_heads=32, n_kv_heads=8, head_dim=128, indexer_dim=128)
    retained = predict_bytes_per_token(**geometry, stage="dense", retain_teacher=True)
    freed = predict_bytes_per_token(**geometry, stage="dense", retain_teacher=False)

    assert "teacher_q" in retained and "teacher_q" not in freed

    # Stored at the model's width, not fp32: 32 heads x 128 dim x 2 bytes x 36 layers.
    assert retained["teacher_q"] == 32 * 128 * 2 * 36

    # No teacher_k term at all -- the closure aliases the cache rather than copying it, so
    # counting it here would double-charge for kv_cache.
    assert not any(key.startswith("teacher_k") for key in retained), sorted(retained)

    # Recomputing the query in backward is the remaining opportunity, now well under 2x.
    assert 1.5 < retained["total"] / freed["total"] < 2.0


def test_capacity_model_agrees_with_a_measured_slope(unit_test_model):  # noqa: F811
    """
    The analytic slope must match what the allocator actually reports.

    Measured at two lengths on the small fixture, so it runs anywhere. A model that predicts
    the wrong slope would mis-attribute the ceiling; tolerance is loose because tile scratch
    and the LM head add a length-dependent term the model deliberately omits.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA to read peak allocation")

    from tests.presses.bench_gqa_indexer_capacity import predict_bytes_per_token, try_length

    model = unit_test_model
    config = model.config
    trainer = make_trainer(model, key_tile=128, query_tile=128)
    indexer = trainer.press.get_indexer(get_attention_modules(model)[0])

    predicted = predict_bytes_per_token(
        layers=config.num_hidden_layers,
        n_heads=config.num_attention_heads,
        n_kv_heads=config.num_key_value_heads,
        head_dim=indexer.head_dim,
        indexer_dim=indexer.head_dim,
        stage="dense",
    )["total"]

    short = try_length(model, trainer, 512, verbose=False)
    long = try_length(model, trainer, 1536, verbose=False)
    assert short["ok"] and long["ok"], "the fixture is far too small to OOM at these lengths"

    measured = (long["peak_gib"] - short["peak_gib"]) * (1024**3) / (1536 - 512)
    assert measured > 0, "peak memory must grow with sequence length"
    # Within 3x: the model omits O(query_tile * key_tile) scratch and the LM head's logits,
    # both of which inflate the measured slope on a model this small.
    assert 0.33 < measured / predicted < 3.0, (
        f"measured {measured / 1024:.1f} KiB/tok vs predicted {predicted / 1024:.1f}"
    )


def test_capacity_model_prices_the_stage2_tile_gather():
    """
    Stage 2's per-tile scratch is O(query_tile * topk_tile * D), not O(query_tile * topk_tile).

    gather_support_keys materializes (B, h, query_tile, topk_tile, D) and the teacher gathers
    (B, H, ...) on top, so the head_dim factor makes it 5 GiB at 512x512x128 -- large enough to
    move the predicted ceiling, which is why the first version of the model (which omitted it
    entirely) came in at 0.69x of the measured length.
    """
    from tests.presses.bench_gqa_indexer_capacity import predict_max_length, predict_tile_scratch

    shape = dict(n_heads=32, n_kv_heads=8, head_dim=128)

    # Stage 1's keys are contiguous -- it never gathers, so it has no scratch term.
    assert predict_tile_scratch(**shape, stage="dense", query_tile=512, topk_tile=512) == 0.0

    scratch = predict_tile_scratch(**shape, stage="sparse", query_tile=512, topk_tile=512)
    assert scratch / (1024**3) == pytest.approx(5.0, abs=0.1)

    # Linear in each tile axis and in head_dim: all three are the levers.
    half_topk = predict_tile_scratch(**shape, stage="sparse", query_tile=512, topk_tile=256)
    assert half_topk == pytest.approx(scratch / 2)
    half_query = predict_tile_scratch(**shape, stage="sparse", query_tile=256, topk_tile=512)
    assert half_query == pytest.approx(scratch / 2)
    narrow = predict_tile_scratch(
        n_heads=32, n_kv_heads=8, head_dim=64, stage="sparse", query_tile=512, topk_tile=512
    )
    assert narrow == pytest.approx(scratch / 2)

    # Scratch is subtracted from the budget, so it lowers the ceiling.
    budget, per_token = 80 * 1024**3, 6269.6 * 1024
    assert predict_max_length(budget, per_token, scratch) < predict_max_length(budget, per_token)
    # And a budget entirely consumed by scratch leaves no room at all.
    assert predict_max_length(scratch, per_token, scratch) == 0


def test_capacity_model_reproduces_the_measured_h20_run():
    """
    Pin the model against the one real measurement, so a refactor cannot silently drift.

    Measured on an H20 (95 GiB, 15.6 GiB weights) BEFORE the group-broadcast/int32 work:
    stage 2 at topk=2048 with 512x512 tiles reached L=9216 against a then-prediction of ~12.4K.
    The optimizations roughly doubled the predicted ceiling, so the check is that the ratio to
    that historical measurement improved rather than that it still holds -- a regression would
    show up as the ratio collapsing back toward 1.3x.
    """
    from tests.presses.bench_gqa_indexer_capacity import (
        predict_bytes_per_token,
        predict_max_length,
        predict_tile_scratch,
    )

    budget = (95.0 - 15.6) * 1024**3
    geometry = dict(layers=36, n_heads=32, n_kv_heads=8, head_dim=128, indexer_dim=128)
    itemized = predict_bytes_per_token(**geometry, stage="sparse", topk=2048)
    scratch = predict_tile_scratch(
        n_heads=32, n_kv_heads=8, head_dim=128, stage="sparse", query_tile=512, topk_tile=512
    )

    predicted = predict_max_length(budget, itemized["total"], scratch)
    historical = 9216  # observed before the group-broadcast and int32 changes
    assert predicted > historical * 1.8, (
        f"predicted {predicted} vs {historical} measured pre-optimization: the ~2x gain from "
        "group-broadcasting the teacher and storing support as int32 has regressed"
    )

    # The support is still the largest single term at this topk, just half what it was.
    assert itemized["support"] > itemized["teacher_q"]
    assert itemized["support"] / itemized["total"] > 0.6


# ----------------------------------------------------------------------
# Reusing flash-attention's logsumexp
# ----------------------------------------------------------------------
class _InjectedCapture:
    """Stand in for capture_teacher_lse, which needs CUDA to run flash-attn."""

    def __init__(self, lse_by_layer):
        self.lse_by_layer = lse_by_layer

    def __enter__(self):
        return self.lse_by_layer

    def __exit__(self, *exc):
        return False


def _flash_style_lse(model, input_ids):
    """
    The lse flash-attention would return: causal mask only, from the layer's real Q/K.

    Collected through the same reconstruction the trainer uses, which is the point -- if the
    rebuilt query did not match what attention used, the captured lse would be normalizing a
    different distribution than the loss scores.
    """
    from kvpress.presses.gqa_indexer.fused_loss import teacher_lse_from_qk
    from kvpress.presses.gqa_indexer.fused_trainer import (
        attention_scaling,
        teacher_query_states,
    )
    from kvpress.presses.gqa_indexer.indexer import build_indexer_mask
    from kvpress.utils import extract_keys_and_values

    out = {}

    def hook(module, args, kwargs, output):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        position = kwargs.get("position_embeddings")
        keys, _ = extract_keys_and_values(kwargs.get("past_key_values"), module.layer_idx)
        query = teacher_query_states(module, hidden, position).detach()
        causal = build_indexer_mask(
            query.shape[2], keys.shape[2], query.device, dtype=torch.float32
        )
        out[int(module.layer_idx)] = teacher_lse_from_qk(
            query, keys.detach(), attention_scaling(module), mask=causal,
            key_tile=16, query_tile=16,
        )
        return output

    handles = [
        layer.self_attn.register_forward_hook(hook, with_kwargs=True)
        for layer in model.model.layers
    ]
    with torch.no_grad():
        model(input_ids=input_ids, use_cache=True)
    for handle in handles:
        handle.remove()
    return out


def test_captured_lse_gives_the_same_loss_as_recomputing(unit_test_model, monkeypatch):  # noqa: F811
    """
    The property the whole optimization rests on.

    A captured lse is only a speedup if it is the *same* normalizer. If it is not, training is
    quietly wrong: the teacher rows stop summing to one and nothing downstream notices.
    """
    from kvpress.presses.gqa_indexer import fused_trainer as ft

    model = unit_test_model
    press = GQAIndexerPress(compression_ratio=0.5)
    press.post_init_from_model(model)
    freeze_all_but_indexer(model)
    input_ids = torch.randint(0, 1000, (1, 32), device=model.device)
    captured = _flash_style_lse(model, input_ids)

    baseline = FusedIndexerTrainer(press=press, key_tile=16, query_tile=16, backend="torch")
    loss_recomputed, _ = fused_indexer_training_step(model, baseline, input_ids=input_ids)
    assert baseline.lse_source == "recomputed"

    reuser = FusedIndexerTrainer(
        press=press, key_tile=16, query_tile=16, backend="torch", capture_lse="auto"
    )
    monkeypatch.setattr(ft, "capture_teacher_lse", lambda m: _InjectedCapture(captured))
    loss_captured, _ = fused_indexer_training_step(model, reuser, input_ids=input_ids)

    assert reuser.lse_source == "captured", "the capture path did not engage"
    torch.testing.assert_close(loss_captured, loss_recomputed, rtol=1e-6, atol=1e-6)


def test_skip_sink_forces_the_recompute(unit_test_model, monkeypatch):  # noqa: F811
    """
    ``skip_sink_in_loss`` folds extra per-key masking into the mask *before* the logsumexp, so
    the rows stay normalized. A causal-only lse would leave the skipped mass in the denominator,
    so the captured value must be refused rather than used.
    """
    from kvpress.presses.gqa_indexer import fused_trainer as ft

    model = unit_test_model
    press = GQAIndexerPress(compression_ratio=0.5)
    press.post_init_from_model(model)
    freeze_all_but_indexer(model)
    input_ids = torch.randint(0, 1000, (1, 32), device=model.device)
    captured = _flash_style_lse(model, input_ids)

    trainer = FusedIndexerTrainer(
        press=press, key_tile=16, query_tile=16, backend="torch",
        capture_lse="auto", skip_sink_in_loss=4,
    )
    monkeypatch.setattr(ft, "capture_teacher_lse", lambda m: _InjectedCapture(captured))
    fused_indexer_training_step(model, trainer, input_ids=input_ids)
    assert trainer.lse_source == "recomputed"


def test_capture_always_rejects_skip_sink_at_construction():
    """
    ``always`` is a promise the run could not keep with skip_sink set -- every layer would fall
    back -- so it fails loudly at construction instead of quietly measuring the slow path.
    """
    press = GQAIndexerPress(compression_ratio=0.5)
    with pytest.raises(ValueError, match="skip_sink_in_loss"):
        FusedIndexerTrainer(press=press, capture_lse="always", skip_sink_in_loss=4)


def test_capture_is_rejected_for_stage_two():
    """Stage 2's global teacher needs a dense full-axis logsumexp, not a causal one."""
    press = GQAIndexerPress(compression_ratio=0.5)
    with pytest.raises(NotImplementedError, match="stage 1 only"):
        FusedIndexerTrainer(press=press, stage="sparse", topk=8, capture_lse="auto")


def test_capture_lse_rejects_an_unknown_mode():
    press = GQAIndexerPress(compression_ratio=0.5)
    with pytest.raises(ValueError, match="capture_lse must be"):
        FusedIndexerTrainer(press=press, capture_lse="yes")


def test_capture_defaults_to_never():
    """Opt-in: it changes what the teacher is normalized against, so it is not silently on."""
    press = GQAIndexerPress(compression_ratio=0.5)
    assert FusedIndexerTrainer(press=press).capture_lse == "never"


def test_captured_dict_is_cleared_after_the_pass(unit_test_model, monkeypatch):  # noqa: F811
    """
    Holding the dict past the forward pass would keep every layer's lse alive between steps and
    risk a stale value being reused if a later pass captured nothing.
    """
    from kvpress.presses.gqa_indexer import fused_trainer as ft

    model = unit_test_model
    press = GQAIndexerPress(compression_ratio=0.5)
    press.post_init_from_model(model)
    freeze_all_but_indexer(model)
    input_ids = torch.randint(0, 1000, (1, 32), device=model.device)
    captured = _flash_style_lse(model, input_ids)

    trainer = FusedIndexerTrainer(
        press=press, key_tile=16, query_tile=16, backend="torch", capture_lse="auto"
    )
    monkeypatch.setattr(ft, "capture_teacher_lse", lambda m: _InjectedCapture(captured))
    fused_indexer_training_step(model, trainer, input_ids=input_ids)
    assert trainer.captured_lse is None
