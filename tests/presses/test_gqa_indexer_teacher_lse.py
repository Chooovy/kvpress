# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for capturing the teacher logsumexp from flash-attention.

Skipped unless flash-attn is importable and a CUDA device is present -- flash-attn has no
CPU kernel. The registration-cleanup tests do not touch flash-attn and always run.
"""

import pytest
import torch

from kvpress import GQAIndexerPress
from kvpress.presses.gqa_indexer import (
    assert_flash_dtype_supported,
    assert_lse_mask_compatible,
    build_indexer_mask,
    capture_teacher_lse,
    get_attention_modules,
    teacher_lse_from_qk,
)
from kvpress.presses.gqa_indexer.fused_trainer import attention_scaling, teacher_query_states
from tests.fixtures import unit_test_model  # noqa: F401

IMPL_NAME = "kvpress_teacher_lse_capture"


def flash_attn_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


requires_flash_attn = pytest.mark.skipif(
    not flash_attn_available(), reason="needs flash-attn and a CUDA device"
)


@pytest.fixture(scope="module")
def bf16_model(unit_test_model):  # noqa: F811
    """
    The unit fixture loads in fp32, which flash-attention cannot run.

    Deep-copied rather than cast in place: the fixture is session-scoped, so mutating its dtype
    would silently change every other test that shares it. `capture_teacher_lse` deliberately
    refuses to cast internally -- the kernel returns the attention output too, so a cast there
    would alter the model's own forward rather than just the lse.
    """
    import copy

    return copy.deepcopy(unit_test_model).to(torch.bfloat16).eval()


def global_mapping():
    """The class-level mapping that AttentionInterface.register writes into."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    return type(ALL_ATTENTION_FUNCTIONS)._global_mapping


# ----------------------------------------------------------------------
# Registration lifecycle (no flash-attn needed for the cleanup assertions)
# ----------------------------------------------------------------------
@requires_flash_attn
def test_registration_is_cleaned_up(unit_test_model):  # noqa: F811
    """
    The temporary implementation must not survive the context manager.

    AttentionInterface.register() writes to the CLASS-level _global_mapping while
    __delitem__ (and hence MutableMapping.pop) only touches the instance's _local_mapping,
    so the obvious pop() raises KeyError internally and leaks the entry forever.
    """
    model = unit_test_model
    assert IMPL_NAME not in global_mapping()

    with capture_teacher_lse(model):
        assert IMPL_NAME in global_mapping()

    assert IMPL_NAME not in global_mapping(), "temporary attention impl leaked"


@requires_flash_attn
def test_registration_is_cleaned_up_on_exception(unit_test_model):  # noqa: F811
    model = unit_test_model
    with pytest.raises(RuntimeError, match="boom"):
        with capture_teacher_lse(model):
            raise RuntimeError("boom")
    assert IMPL_NAME not in global_mapping()


@requires_flash_attn
def test_attn_implementation_is_restored(unit_test_model):  # noqa: F811
    model = unit_test_model
    before = model.config._attn_implementation
    with capture_teacher_lse(model):
        assert model.config._attn_implementation == IMPL_NAME
    assert model.config._attn_implementation == before


@requires_flash_attn
def test_attn_implementation_is_restored_on_exception(unit_test_model):  # noqa: F811
    model = unit_test_model
    before = model.config._attn_implementation
    with pytest.raises(RuntimeError):
        with capture_teacher_lse(model):
            raise RuntimeError("boom")
    assert model.config._attn_implementation == before


# ----------------------------------------------------------------------
# Capture correctness
# ----------------------------------------------------------------------
@requires_flash_attn
def test_captures_one_lse_per_layer(bf16_model):
    model = bf16_model
    input_ids = torch.randint(0, 1024, (1, 32), device=model.device)

    with capture_teacher_lse(model) as lse_by_layer:
        model(input_ids, use_cache=False)

    assert set(lse_by_layer) == set(range(model.config.num_hidden_layers))
    for lse in lse_by_layer.values():
        assert lse.shape == (1, model.config.num_attention_heads, 32)
        assert lse.dtype == torch.float32
        assert torch.isfinite(lse).all()


@requires_flash_attn
def test_captured_lse_matches_the_streaming_fallback(bf16_model):
    """
    The captured value must equal teacher_lse_from_qk on the same Q/K.

    This is the test that matters: it validates flash-attn's lse layout, its bottom-right
    causal alignment, and its GQA head mapping all agree with the reconstruction the fused
    loss performs. A mismatch in any of the three would otherwise train against a subtly
    wrong teacher with nothing to flag it.
    """
    model = bf16_model
    press = GQAIndexerPress(compression_ratio=0.5)
    press.post_init_from_model(model)
    modules = get_attention_modules(model)
    seq_len = 32
    input_ids = torch.randint(0, 1024, (1, seq_len), device=model.device)

    hidden_by_layer = {}

    def make_hook(layer_idx):
        def hook(module, args, kwargs, output):
            hidden_by_layer[layer_idx] = (
                kwargs["hidden_states"].detach(),
                kwargs["position_embeddings"],
                kwargs["past_key_values"],
            )
            return output

        return hook

    handles = [
        m.register_forward_hook(make_hook(i), with_kwargs=True) for i, m in enumerate(modules)
    ]
    try:
        with capture_teacher_lse(model) as lse_by_layer:
            model(input_ids, use_cache=True)
    finally:
        for handle in handles:
            handle.remove()

    from kvpress.utils import extract_keys_and_values

    for layer_idx in (0, len(modules) - 1):
        module = modules[layer_idx]
        hidden, position_embeddings, cache = hidden_by_layer[layer_idx]
        keys, _ = extract_keys_and_values(cache, layer_idx)

        query_states = teacher_query_states(module, hidden, position_embeddings)
        mask = build_indexer_mask(seq_len, keys.shape[2], hidden.device, dtype=torch.float32)
        expected = teacher_lse_from_qk(
            query_states, keys, attention_scaling(module), mask=mask, key_tile=16
        )
        # Both paths see the same bf16 Q/K and accumulate in fp32, so the only difference is
        # reduction order -- but the logits themselves come from a bf16 GEMM, so the tolerance
        # tracks bf16's ~4e-3 relative epsilon rather than fp32's.
        torch.testing.assert_close(
            lse_by_layer[layer_idx], expected.float(), rtol=1e-2, atol=1e-2
        )


@requires_flash_attn
def test_model_output_is_unchanged_by_capture(bf16_model):
    """
    Capturing must be observationally transparent to the model's own output.

    Compared by *prediction*, not by raw logits. Capture swaps sdpa for flash-attention, and two
    bf16 kernels with different reduction orders will not agree to any tight logit tolerance --
    a comparison loose enough to pass would no longer be evidence of anything. Argmax agreement
    plus a bounded relative difference is the property that actually matters: the capture must
    not change what the model *says*.
    """
    model = bf16_model
    input_ids = torch.randint(0, 1024, (1, 32), device=model.device)

    with torch.no_grad():
        reference = model(input_ids, use_cache=False).logits.float()
        with capture_teacher_lse(model):
            captured = model(input_ids, use_cache=False).logits.float()

    assert captured.shape == reference.shape
    assert torch.isfinite(captured).all()
    assert torch.equal(captured.argmax(-1), reference.argmax(-1)), (
        "capture changed the model's prediction, so it is not a drop-in replacement"
    )
    scale = reference.abs().amax().clamp_min(1e-6)
    assert ((captured - reference).abs().amax() / scale) < 0.05


@requires_flash_attn
def test_captured_lse_feeds_the_fused_loss(bf16_model):
    """End-to-end: a captured lse must drive the fused loss to a finite, differentiable value."""
    from kvpress.presses.gqa_indexer import fused_indexer_loss, make_recompute_teacher
    from kvpress.utils import extract_keys_and_values

    model = bf16_model
    press = GQAIndexerPress(compression_ratio=0.5)
    press.post_init_from_model(model)
    module = get_attention_modules(model)[0]
    seq_len = 32
    input_ids = torch.randint(0, 1024, (1, seq_len), device=model.device)

    captured_inputs = {}

    def hook(mod, args, kwargs, output):
        captured_inputs["hidden"] = kwargs["hidden_states"].detach()
        captured_inputs["pe"] = kwargs["position_embeddings"]
        captured_inputs["cache"] = kwargs["past_key_values"]
        return output

    handle = module.register_forward_hook(hook, with_kwargs=True)
    try:
        with capture_teacher_lse(model) as lse_by_layer:
            model(input_ids, use_cache=True)
    finally:
        handle.remove()

    hidden = captured_inputs["hidden"]
    keys, _ = extract_keys_and_values(captured_inputs["cache"], 0)
    query_states = teacher_query_states(module, hidden, captured_inputs["pe"])
    scaling = attention_scaling(module)
    group_size = query_states.shape[1] // keys.shape[1]

    indexer = press.get_indexer(module)
    cos, sin = press.get_rope_tables(indexer, {"position_embeddings": captured_inputs["pe"]})
    mask = build_indexer_mask(seq_len, keys.shape[2], hidden.device, dtype=torch.float32)

    loss = fused_indexer_loss(
        indexer,
        hidden,
        make_recompute_teacher(query_states, keys, scaling, group_size),
        lse_by_layer[0],
        group_size=group_size,
        cos=cos,
        sin=sin,
        mask=mask,
        key_tile=16,
        query_tile=16,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in indexer.parameters())


# ----------------------------------------------------------------------
# The padding guard (no flash-attn required)
# ----------------------------------------------------------------------
def test_padding_guard_rejects_a_padded_batch():
    """
    flash-attn's lse is causal-only, so padding would leave the teacher un-normalized.

    Callers combining capture_teacher_lse with a padded batch must be stopped, since the
    resulting mis-weighting is silent.
    """
    padded = torch.ones(2, 8, dtype=torch.long)
    padded[1, :3] = 0
    with pytest.raises(ValueError, match="un-normalized"):
        assert_lse_mask_compatible(padded, "flash-attn")


def test_padding_guard_accepts_causal_only():
    assert_lse_mask_compatible(None, "flash-attn")
    assert_lse_mask_compatible(torch.ones(2, 8, dtype=torch.long), "flash-attn")
    assert_lse_mask_compatible(build_indexer_mask(8, 8, torch.device("cpu")), "flash-attn")


# ----------------------------------------------------------------------
# The dtype guard (no flash-attn required)
# ----------------------------------------------------------------------
def test_flash_dtype_guard_accepts_half_precision():
    assert_flash_dtype_supported(torch.float16)
    assert_flash_dtype_supported(torch.bfloat16)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_flash_dtype_guard_rejects_wider_dtypes(dtype):
    """
    fp32 must be refused with the fallback named, not cast behind the caller's back.

    A cast would be wrong twice over: flash_attn_func also returns the attention *output*, which
    the model consumes, so casting would change the model's own forward; and a reduced-precision
    lse paired with a wider alpha leaves the teacher rows not summing to one.
    """
    with pytest.raises(TypeError, match="only fp16 and bf16"):
        assert_flash_dtype_supported(dtype)
    with pytest.raises(TypeError, match="teacher_lse_from_qk"):
        assert_flash_dtype_supported(dtype)


@requires_flash_attn
def test_capture_refuses_an_fp32_model(unit_test_model):  # noqa: F811
    """The guard must fire from inside the capture, naming the dtype rather than flash-attn's."""
    model = unit_test_model
    assert next(model.parameters()).dtype == torch.float32, "fixture is expected to be fp32"
    input_ids = torch.randint(0, 1024, (1, 8), device=model.device)
    with pytest.raises(TypeError, match="only fp16 and bf16"):
        with capture_teacher_lse(model):
            model(input_ids, use_cache=False)


@requires_flash_attn
def test_capture_cleans_up_after_a_dtype_failure(unit_test_model):  # noqa: F811
    """A mid-forward raise must still restore the attention implementation."""
    model = unit_test_model
    before = model.config._attn_implementation
    input_ids = torch.randint(0, 1024, (1, 8), device=model.device)
    with pytest.raises(TypeError):
        with capture_teacher_lse(model):
            model(input_ids, use_cache=False)
    assert model.config._attn_implementation == before
    assert IMPL_NAME not in global_mapping()
