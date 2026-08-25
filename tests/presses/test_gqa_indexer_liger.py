# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the liger-kernel patch applied by ``scripts/train_gqa_indexer_e2e.py``.

Two things are patched and each needs a different guard:

* **fused linear+CE** -- the loss head. Guarded at runtime by
  ``check_liger_loss_unchanged``, which toggles ``skip_logits`` on the *same* patched model.
* **SwiGLU** -- the MLP. That runtime check cannot see it, because SwiGLU is patched in both
  arms of the comparison, so a broken SwiGLU would be broken identically on both sides. The
  only way to catch it is against an **unpatched** model, which is what this module does.

What must NOT be patched is asserted too. ``rope`` would change the cos/sin convention the
press narrows for the indexer, training the router against a positional signal it never sees at
inference; ``rms_norm`` replaces the norm the indexer's own ``IndexerNorm`` sits beside. Both are
disabled deliberately, and a future liger version flipping a default is exactly the kind of
change that would otherwise pass unnoticed.

Liger's Triton kernels need a real CUDA driver, so the numerical comparisons skip without one.
The *structural* assertions (what got patched, what did not) run anywhere, since they only
inspect the module tree -- and they are the ones that catch a changed default.
"""

import pytest
import torch

pytest.importorskip("liger_kernel")

from scripts.train_gqa_indexer_e2e import apply_liger_fused_ce  # noqa: E402

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="liger's Triton kernels need a CUDA driver"
)


@pytest.fixture(autouse=True)
def restore_module_classes():
    """
    Undo liger's **global** class replacement after every test in this module.

    ``apply_liger_kernel_to_llama`` assigns ``modeling_llama.LlamaMLP = LigerSwiGLUMLP`` at module
    scope, so every model built afterwards -- anywhere in the session -- gets Liger's MLP. Without
    this fixture the patch leaked into the rest of the suite and 212 unrelated tests failed on
    Liger's Triton kernel for want of a CUDA driver. Restoring by name rather than by reloading the
    module keeps the identity of everything else intact.
    """
    from transformers.models.llama import modeling_llama

    saved = {
        name: getattr(modeling_llama, name)
        for name in ("LlamaMLP", "LlamaRMSNorm", "apply_rotary_pos_emb")
    }
    saved_forward = modeling_llama.LlamaForCausalLM.forward
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(modeling_llama, name, value)
        modeling_llama.LlamaForCausalLM.forward = saved_forward


def tiny_model(n_layers=2):
    """A small real Llama; liger patches per architecture, so the arch must be real."""
    transformers = pytest.importorskip("transformers")
    config = transformers.AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    config.num_attention_heads, config.num_key_value_heads = 8, 4
    config.hidden_size, config.intermediate_size = 64, 128
    config.num_hidden_layers, config.head_dim = n_layers, 8
    config._attn_implementation = "sdpa"
    model = transformers.AutoModelForCausalLM.from_config(config).to(torch.float32).eval()
    return model, config


def test_patch_replaces_swiglu_and_the_loss_head_only():
    """
    SwiGLU and the loss head are patched; RoPE and RMSNorm are not.

    Liger rebinds ``forward`` rather than swapping the class, so ``type(mlp)`` stays ``LlamaMLP``
    and only the bound method changes -- checking the class would silently pass whatever happened.
    That is asserted here because it is how the first attempt at this test fooled itself.
    """
    model, _ = tiny_model()
    mlp, norm = model.model.layers[0].mlp, model.model.layers[0].input_layernorm
    assert "Liger" not in mlp.forward.__func__.__qualname__

    apply_liger_fused_ce(model, "llama")

    mlp, norm = model.model.layers[0].mlp, model.model.layers[0].input_layernorm
    assert "Liger" in mlp.forward.__func__.__qualname__, (
        "SwiGLU was not patched -- the MLP is the second largest retained term (27% of it), so a "
        "silently-skipped patch costs ~6.8 GiB at L=8192"
    )
    assert "Liger" not in norm.forward.__func__.__qualname__, (
        "RMSNorm was patched, which this configuration deliberately avoids: the indexer carries "
        "its own IndexerNorm whose fp32-statistics behaviour is load-bearing"
    )


def test_rope_is_not_patched():
    """
    ``apply_rotary_pos_emb`` must be untouched -- the one patch that would corrupt the router.

    The press narrows the layer's ``position_embeddings`` to score with, so a different cos/sin
    convention would train the indexer against positions it never sees at inference. That is a
    silent train/inference mismatch: the loss would still fall and eval would just be worse.
    """
    from transformers.models.llama import modeling_llama

    before = modeling_llama.apply_rotary_pos_emb
    model, _ = tiny_model()
    apply_liger_fused_ce(model, "llama")
    assert modeling_llama.apply_rotary_pos_emb is before, (
        "liger replaced apply_rotary_pos_emb; rope=False is meant to prevent exactly this"
    )


def test_unknown_architecture_fails_loudly():
    """
    A model liger has no patch for must raise, not silently run unpatched.

    The whole point of the flag is a memory reduction; one that quietly does nothing is worse than
    an error, because the run OOMs later with no indication why.
    """
    class WeirdForCausalLM(torch.nn.Module):
        pass

    with pytest.raises(RuntimeError, match="no patch for"):
        apply_liger_fused_ce(WeirdForCausalLM(), "weird")


@requires_cuda
def test_swiglu_does_not_change_the_forward():
    """
    The patched MLP must be numerically equivalent to the unpatched one.

    Compared against an **unpatched** model, which is the comparison
    ``check_liger_loss_unchanged`` structurally cannot make. Liger recomputes ``silu`` in the
    backward instead of storing it, so the tolerance allows for a different accumulation order but
    nothing more.
    """
    torch.manual_seed(0)
    reference, config = tiny_model()
    reference = reference.cuda()
    patched, _ = tiny_model()
    patched.load_state_dict(reference.state_dict())
    patched = patched.cuda()
    apply_liger_fused_ce(patched, "llama")

    input_ids = torch.randint(0, config.vocab_size, (1, 64), device="cuda")
    with torch.no_grad():
        want = reference(input_ids=input_ids).logits
        got = patched(input_ids=input_ids).logits
    assert torch.allclose(got, want, atol=2e-3, rtol=1e-2), (
        f"patched MLP changed the forward by {(got - want).abs().max().item():.2e}"
    )


@requires_cuda
def test_swiglu_gradients_match_and_retain_less():
    """
    Equivalent gradients, strictly less retained memory -- the whole point of the swap.

    Standard SwiGLU keeps 3 tensors of ``(L, inter)``: ``silu`` saves its input and the elementwise
    ``mul`` saves *both* operands. Liger keeps only ``(gate, up)``. Measured 2.75x against 3.75x, a
    27% cut, and freezing the backbone does not avoid it -- a frozen MLP still retains its operands
    whenever the gradient passes through, which it must here since the router sits below every MLP
    above it.
    """
    def run(patch: bool):
        torch.manual_seed(0)
        model, config = tiny_model()
        model = model.cuda()
        if patch:
            apply_liger_fused_ce(model, "llama")
        hidden = torch.randn(1, 128, config.hidden_size, device="cuda", requires_grad=True)
        storages = {}

        def pack(tensor):
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            out = model.model.layers[0].mlp(hidden)
        torch.manual_seed(1)
        (out * torch.randn_like(out)).sum().backward()
        return hidden.grad.clone(), sum(storages.values())

    grad_ref, bytes_ref = run(patch=False)
    grad_liger, bytes_liger = run(patch=True)

    assert torch.allclose(grad_ref, grad_liger, atol=2e-3, rtol=1e-2), (
        f"gradients differ by {(grad_ref - grad_liger).abs().max().item():.2e}"
    )
    assert bytes_liger < bytes_ref, (
        f"liger retained {bytes_liger} bytes against {bytes_ref} -- no saving, so the patch is "
        "not doing what it was enabled for"
    )
