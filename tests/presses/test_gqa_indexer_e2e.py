# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for end-to-end (gated-attention) indexer training.

Five layers, each ruling out a different class of bug:

1. **The identity.** ``build_concat_qk`` folds the gate into the QK dot product; the folded
   form must match an explicit gate-then-softmax reference in the forward *and* in every
   gradient (``q_idx``, ``k_idx``, ``gate_scale``). This is the load-bearing claim of the whole
   approach -- if it fails, the router is being trained through the wrong function.
2. **The premises.** The two facts the design rests on, asserted rather than assumed: the
   ``log softmax`` normalizer is inert *when nothing is pinned* (so the un-pinned path needs no
   normalization pass), and ``gate_scale = 0`` severs the router gradient (so it must not be the
   init).
3. **Pinning.** The property the whole mechanism exists for: a pinned gate cannot be flattened
   into a no-op, and an un-pinned one can. Plus which pins fold into the concat and which need a
   second attention path -- the question that decides whether a kernel is required.
4. **The gradient structure.** Full scope must give unselected keys *content-dependent*
   gradients while sparse scope must not -- the SAS Figure 5 diagnostic, and the reason stage 1
   is worth its ``O(L^2)``.
5. **The wiring.** On a real model: the LM loss reaches every indexer parameter, the backbone
   stays frozen, the attention implementation is restored on exit, and the loss actually
   descends.

fp64 throughout for the reference comparisons -- the concat identity is exact, so the tolerance
should be measuring floating-point noise and nothing else.
"""

import pytest
import torch

from kvpress.presses.gqa_indexer import (
    E2EIndexerTrainer,
    GQAIndexer,
    GQAIndexerConfig,
    GQAIndexerPress,
    build_concat_qk,
    causal_mask_bottom_right,
    check_gate_shapes,
    e2e_indexer_training_step,
    gated_attention,
    gated_attention_full,
    gated_attention_reference,
    gated_attention_sparse,
)
from kvpress.presses.gqa_indexer.gate_pin import (
    PIN_MODES,
    history_lse,
    is_query_dependent,
    pinned_mask,
)
from kvpress.presses.gqa_indexer.gated_attention import (
    gated_attention_pinned_self,
    pad_value_to_width,
)

DT = torch.float64

#: ``(pin_mode, n_sink)`` pairs covering every mode with a meaningful sink count.
PIN_CASES = [("none", 0), ("sink", 4), ("self", 0), ("self+sink", 4)]


def make_inputs(bsz=2, n_heads=8, n_kv_heads=4, q_len=9, k_len=9, dim=16, idx_dim=16, seed=0):
    """Random q/k/v plus indexer q/k, all fp64 and all requiring grad."""
    torch.manual_seed(seed)
    return dict(
        q=torch.randn(bsz, n_heads, q_len, dim, dtype=DT, requires_grad=True),
        k=torch.randn(bsz, n_kv_heads, k_len, dim, dtype=DT, requires_grad=True),
        v=torch.randn(bsz, n_kv_heads, k_len, dim, dtype=DT, requires_grad=True),
        q_idx=torch.randn(bsz, n_kv_heads, q_len, idx_dim, dtype=DT, requires_grad=True),
        k_idx=torch.randn(bsz, k_len, idx_dim, dtype=DT, requires_grad=True),
    )


def backward_of(fn, inputs, gate_scale, seed=1, **kwargs):
    """Run ``fn``, backprop a fixed random cotangent, return (out, grads dict)."""
    lam = torch.tensor(gate_scale, dtype=DT, requires_grad=True)
    out = fn(**inputs, gate_scale=lam, **kwargs)
    torch.manual_seed(seed)
    (out * torch.randn_like(out)).sum().backward()
    grads = {name: tensor.grad.clone() for name, tensor in inputs.items()}
    grads["gate_scale"] = lam.grad.clone()
    return out.detach(), grads


def tiny_model(n_layers=3, n_heads=8, n_kv_heads=4, hidden=64):
    """A small real Llama, so the wiring tests exercise HF's actual attention plumbing."""
    transformers = pytest.importorskip("transformers")
    config = transformers.AutoConfig.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    )
    config.num_attention_heads = n_heads
    config.num_key_value_heads = n_kv_heads
    config.hidden_size = hidden
    config.intermediate_size = 2 * hidden
    config.num_hidden_layers = n_layers
    config.head_dim = hidden // n_heads
    config._attn_implementation = "sdpa"
    return transformers.AutoModelForCausalLM.from_config(config).to(torch.float32).eval(), config


# ----------------------------------------------------------------------
# 1. The concat identity
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "q_len,k_len,n_heads,n_kv_heads,dim,idx_dim",
    [
        (9, 9, 8, 4, 16, 16),    # square, GQA, matched dims (the common training shape)
        (9, 13, 8, 4, 16, 12),   # Sq < Sk: bottom-right alignment, narrower indexer
        (1, 7, 4, 4, 16, 16),    # a decode step, no GQA
        (7, 7, 4, 1, 16, 24),    # MQA on the attention side, wider indexer
        (5, 5, 2, 2, 8, 8),      # small everything
    ],
)
def test_concat_matches_explicit(q_len, k_len, n_heads, n_kv_heads, dim, idx_dim):
    """
    The folded QK form equals the explicit gate-then-softmax, forward and all gradients.

    ``scale * (q.k) + lam * (qi.ki) == scale * ([q, (lam/scale) qi] . [k, ki])`` is what makes
    token-level full-scope gating affordable at all: no gate tensor is ever materialized, so
    there is no ``O(Sq * Sk)`` table. Exact identity, hence the tight tolerance.
    """
    kwargs = dict(q_len=q_len, k_len=k_len, n_heads=n_heads, n_kv_heads=n_kv_heads,
                  dim=dim, idx_dim=idx_dim)
    out_ref, grad_ref = backward_of(gated_attention_reference, make_inputs(**kwargs), 0.7)
    out_cat, grad_cat = backward_of(gated_attention_full, make_inputs(**kwargs), 0.7)

    assert torch.allclose(out_ref, out_cat, atol=1e-11), "forward disagrees"
    for name in grad_ref:
        assert torch.allclose(grad_ref[name], grad_cat[name], atol=1e-10), f"d/{name} disagrees"


def test_sparse_over_all_keys_equals_full():
    """With every key in the support, the sparse scope reduces to the full scope."""
    inputs = make_inputs(q_len=9, k_len=13)
    k_len = inputs["k"].shape[2]
    indices = (
        torch.arange(k_len).view(1, 1, 1, k_len).expand(2, 4, 9, k_len).contiguous().int()
    )
    sparse = gated_attention_sparse(**inputs, indices=indices, gate_scale=0.7)
    full = gated_attention_full(**make_inputs(q_len=9, k_len=13), gate_scale=0.7)
    # sparse_gqa_attention_reference accumulates in fp32 by documented design, so this compares
    # at fp32 precision rather than fp64.
    assert torch.allclose(sparse, full, atol=1e-6)


def test_gate_scale_scales_the_gate_not_the_attention():
    """
    ``gate_scale`` must move only the gate term.

    A plausible wiring bug is to let the factor ride on the whole concatenated query, which
    would rescale the attention logits too -- silently changing the frozen model's attention
    temperature rather than gating it. Setting the indexer contribution to zero isolates this:
    the output must then be independent of ``gate_scale``.
    """
    inputs = make_inputs()
    with torch.no_grad():
        inputs["k_idx"].zero_()  # gate == 0 for every pair, whatever gate_scale is
    baseline = gated_attention_full(**inputs, gate_scale=1.0)
    for scale in (0.1, 5.0):
        assert torch.allclose(gated_attention_full(**inputs, gate_scale=scale), baseline, atol=1e-12)


def test_concat_puts_gate_factor_on_the_query_side_only():
    """``build_concat_qk`` leaves ``k_idx`` untouched, so a cached indexer key stays reusable."""
    inputs = make_inputs(idx_dim=12)
    dim, idx_dim = inputs["q"].shape[-1], 12
    query, key = build_concat_qk(
        inputs["q"], inputs["k"], inputs["q_idx"], inputs["k_idx"],
        scale=dim**-0.5, gate_scale=0.7, group_size=2,
    )
    assert query.shape[-1] == dim + idx_dim and key.shape[-1] == dim + idx_dim
    # The key's gate half is k_idx verbatim, broadcast over KV heads -- no scaling applied.
    assert torch.allclose(key[..., dim:], inputs["k_idx"].unsqueeze(1).expand_as(key[..., dim:]))
    # The attention halves pass through untouched on both sides.
    assert torch.allclose(query[..., :dim], inputs["q"])
    assert torch.allclose(key[..., :dim], inputs["k"])


# ----------------------------------------------------------------------
# 2. The premises the design rests on
# ----------------------------------------------------------------------
def test_gate_normalizer_is_inert():
    """
    ``softmax(qk + log softmax(s)) == softmax(qk + s)``, in forward and in ``d/ds``.

    SAS writes its gate as ``log softmax(s)``. The ``logsumexp(s)`` term is one constant per
    query row added to every key in that row, and such a constant cancels in the softmax --
    which is why this implementation skips the normalization pass entirely. The third assertion
    gives the mechanism: ``sum_k dS_k == 0``, so the normalizer's own gradient path vanishes.

    This is what makes it safe to *omit* the normalizer. It would stop holding if some keys were
    exempted from the gate (SAS pins its self-block gate to 1); this implementation gates
    uniformly and so stays in the regime where the equivalence holds.
    """
    torch.manual_seed(0)
    z = torch.randn(2, 3, 5, 7, dtype=DT)
    v = torch.randn(2, 3, 7, 4, dtype=DT)

    outs, grads = {}, {}
    for name in ("raw", "logsoftmax"):
        s = torch.randn(2, 3, 5, 7, dtype=DT, generator=torch.Generator().manual_seed(1))
        s.requires_grad_(True)
        gate = s if name == "raw" else s.log_softmax(-1)
        out = (z + gate).softmax(-1) @ v
        torch.manual_seed(2)
        (out * torch.randn_like(out)).sum().backward()
        outs[name], grads[name] = out.detach(), s.grad.clone()

    assert torch.allclose(outs["raw"], outs["logsoftmax"], atol=1e-14)
    assert torch.allclose(grads["raw"], grads["logsoftmax"], atol=1e-14)
    assert grads["raw"].sum(-1).abs().max() < 1e-13  # the reason: dS sums to zero per row


def test_pinning_a_key_revives_the_normalizer_and_breaks_the_fold():
    """
    Both of this design's shortcuts hinge on the gate being *uniform over keys*.

    SAS pins its self-block gate to 1 (``log 1 = 0``) so local causal attention always survives.
    That single exempted entry stops ``logsumexp(s)`` from being constant along the key axis, and
    two things follow at once:

    * the normalizer stops cancelling, so ``log softmax(s)`` is no longer interchangeable with
      raw ``s`` -- it becomes load-bearing;
    * the gate stops folding into the QK dot product, so it has to be materialized as a table.

    Same root cause, which is why SAS needs a kernel and this implementation does not. Pinned as a
    test because it is the assumption a future "let's also protect the sink keys" change would
    quietly violate -- and it would break the concat identity, not merely add a term.
    """
    torch.manual_seed(0)
    n_blocks, block = 4, 3
    k_len = n_blocks * block
    block_of = torch.arange(k_len) // block
    logits = torch.randn(2, 5, k_len, dtype=DT)
    values = torch.randn(2, k_len, 4, dtype=DT)

    def attend(gate_blocks):
        return (logits + gate_blocks[:, :, block_of]).softmax(-1) @ values

    score = torch.randn(2, 5, n_blocks, dtype=DT)
    pinned = score.log_softmax(-1).clone()
    pinned[:, :, -1] = 0.0  # the self block, exempted

    # Uniform: raw and log-softmax agree. Pinned: they do not.
    assert torch.allclose(attend(score), attend(score.log_softmax(-1)), atol=1e-13)
    assert not torch.allclose(attend(score), attend(pinned), atol=1e-3)


def test_block_level_bilinear_gate_also_folds():
    """
    Folding is not a property of *token* granularity -- a block gate folds too, if bilinear.

    Worth pinning because the tempting summary of this design is "token-wise avoids the kernel";
    the actual conditions are bilinearity and a uniform gate. Broadcasting one pooled indexer key
    per block back to that block's tokens on the key side leaves the identity intact, so a
    block-pooled variant of this indexer would need no kernel either.
    """
    torch.manual_seed(0)
    n_blocks, block, dim, idx_dim = 4, 3, 4, 3
    k_len = n_blocks * block
    block_of = torch.arange(k_len) // block
    scale, gate_scale = dim**-0.5, 0.7

    q = torch.randn(2, 5, dim, dtype=DT)
    k = torch.randn(2, k_len, dim, dtype=DT)
    q_idx = torch.randn(2, 5, idx_dim, dtype=DT)
    k_blocks = torch.randn(2, n_blocks, idx_dim, dtype=DT)  # one pooled key per block
    values = torch.randn(2, k_len, 4, dtype=DT)

    gate = torch.einsum("hqd,hcd->hqc", q_idx, k_blocks)[:, :, block_of]
    explicit = ((q @ k.transpose(-1, -2)) * scale + gate_scale * gate).softmax(-1) @ values

    query = torch.cat([q, (gate_scale / scale) * q_idx], dim=-1)
    key = torch.cat([k, k_blocks[:, block_of]], dim=-1)
    folded = ((query @ key.transpose(-1, -2)) * scale).softmax(-1) @ values

    assert torch.allclose(explicit, folded, atol=1e-13)


def test_zero_gate_scale_severs_the_router_gradient():
    """
    At ``gate_scale = 0`` the router receives no gradient at all.

    Worth pinning because zero is the *tempting* init -- it would start end-to-end training from
    exactly the frozen dense model, with no perturbation. It is also a fixed point: ``dL/dscore``
    is proportional to ``gate_scale``, so the run would never leave it. This is why
    ``GQAIndexer.GATE_SCALE_INIT`` is the natural scale instead.
    """
    inputs = make_inputs()
    _, grads = backward_of(gated_attention_full, inputs, 0.0)
    assert grads["q_idx"].abs().max() == 0.0
    assert grads["k_idx"].abs().max() == 0.0
    # The attention path is untouched, so q/k/v still learn -- it is only the router that is cut.
    assert grads["q"].abs().max() > 0.0


def test_gate_scale_init_matches_attention_logit_magnitude():
    """
    The default ``gate_scale`` brings the gate to the same magnitude as an attention logit.

    ``IndexerNorm`` leaves q/k at unit variance per channel, so the raw score has std
    ``~sqrt(head_dim)`` -- an order of magnitude louder than ``q @ k / sqrt(head_dim)``. Added
    unscaled it would swamp the attention it is meant to modulate rather than gate it.
    """
    head_dim = 64
    config = GQAIndexerConfig(hidden_size=256, n_heads=4, head_dim=head_dim, gate_scale=True)
    indexer = GQAIndexer(config)
    assert float(indexer.gate_scale.detach()) == pytest.approx(head_dim**-0.5)

    torch.manual_seed(0)
    hidden = torch.randn(2, 128, 256)
    score = indexer(hidden).detach()
    gated = score * indexer.gate_scale.detach()
    # Raw: std ~ sqrt(head_dim). Gated: std ~ 1, i.e. an attention logit's scale.
    assert score.std() > 3.0
    assert 0.3 < gated.std() < 3.0


def test_indexer_without_gate_scale_refuses_to_gate():
    """An indexer built for distillation raises rather than silently running a fixed scale."""
    indexer = GQAIndexer(GQAIndexerConfig(hidden_size=64, n_heads=2, head_dim=8))
    assert indexer.gate_scale is None
    with pytest.raises(RuntimeError, match="gate_scale"):
        indexer.require_gate_scale()


# ----------------------------------------------------------------------
# 2b. Pinning: what closes the no-op hole
# ----------------------------------------------------------------------
@pytest.mark.parametrize("pin_mode,n_sink", PIN_CASES)
@pytest.mark.parametrize("q_len,k_len", [(9, 9), (9, 13), (1, 7)])
def test_pinned_fast_path_matches_reference(pin_mode, n_sink, q_len, k_len):
    """
    Every pin mode's fast path reproduces the explicit reference, forward and all gradients.

    ``none`` and ``sink`` go through SDPA on the concatenated heads (``sink`` adding one extra
    dimension for the ``-LSE`` normalizer); ``self`` and ``self+sink`` cannot fold and take the
    two-branch path. This asserts all of them against one shared definition.
    """
    kwargs = dict(q_len=q_len, k_len=k_len)
    pin = dict(pin_mode=pin_mode, n_sink=n_sink)
    out_ref, grad_ref = backward_of(gated_attention_reference, make_inputs(**kwargs), 0.7, **pin)
    out_fast, grad_fast = backward_of(gated_attention_full, make_inputs(**kwargs), 0.7, **pin)

    assert torch.allclose(out_ref, out_fast, atol=1e-11), "forward disagrees"
    for name in grad_ref:
        assert torch.allclose(grad_ref[name], grad_fast[name], atol=1e-10), f"d/{name} disagrees"


@pytest.mark.parametrize("pin_mode,n_sink", PIN_CASES)
def test_pin_closes_the_no_op_hole(pin_mode, n_sink):
    """
    **The property pinning exists for.** A pinned gate cannot be turned off.

    Two routes reach the no-op, and a pin must block both:

    * ``q_idx = 0`` or any constant -- the gate is then flat along the key axis, and a per-key
      constant cancels in the softmax;
    * ``gate_scale -> 0`` -- the gate term vanishes outright.

    Either one recovers the frozen dense backbone, which is already strong, so the LM loss is
    satisfied with no ranking learned. Under a pin the normalizer leaves a fixed
    ``log(N_history)`` suppression of history against the pinned keys that no router output can
    cancel, so dense is unreachable.

    ``pin_mode="none"`` is asserted to be *exactly* reachable: it is the ablation baseline, and
    this test is what documents that it is broken by design rather than by accident.
    """
    inputs = make_inputs(q_len=16, k_len=16, seed=3)
    with torch.no_grad():
        for tensor in inputs.values():
            tensor.requires_grad_(False)
    # True dense: no gate at all.
    dense = gated_attention_full(**inputs, gate_scale=0.0, pin_mode="none")

    flat_q = dict(inputs)
    reachable = []
    for value in (0.0, 0.3):  # a zero and a nonzero constant q_idx both give a flat gate
        flat_q["q_idx"] = torch.full_like(inputs["q_idx"], value)
        out = gated_attention_full(**flat_q, gate_scale=0.7, pin_mode=pin_mode, n_sink=n_sink)
        reachable.append((out - dense).abs().max().item())
    muted = gated_attention_full(**inputs, gate_scale=0.0, pin_mode=pin_mode, n_sink=n_sink)
    reachable.append((muted - dense).abs().max().item())

    if pin_mode == "none":
        assert min(reachable) < 1e-12, "the un-pinned gate should be able to do nothing"
    else:
        assert min(reachable) > 1e-3, (
            f"pin_mode={pin_mode!r} still lets the gate reach a no-op (closest {min(reachable):.2e}), "
            "so the router can satisfy the loss without learning a ranking"
        )


def test_flat_gate_variants_all_reach_the_no_op():
    """
    SAS's Table 1 failure rows are one hole reached three ways -- assert that unification.

    ``raw s`` at ``s = 0``, ``log sigmoid(s)`` as ``s`` saturates, and ``log softmax(s)`` at
    uniform ``s`` all produce a gate that is constant along the key axis, and a per-key constant
    cancels in the softmax. Only normalizing *and* exempting some keys removes the flat solution.
    This is the claim :mod:`~kvpress.presses.gqa_indexer.gate_pin`'s table rests on.
    """
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 6, 12, dtype=DT)
    values = torch.randn(1, 4, 12, 3, dtype=DT)
    dense = logits.softmax(-1) @ values

    def gap(gate):
        return ((logits + gate).softmax(-1) @ values - dense).abs().max().item()

    zeros = torch.zeros_like(logits)
    assert gap(zeros) == 0.0                                    # raw s at s = 0
    assert gap(torch.nn.functional.logsigmoid(zeros + 30)) < 1e-12   # sigmoid saturated
    assert gap(zeros.log_softmax(-1)) < 1e-12                   # log_softmax, nothing pinned

    # With a pin, the same uniform score cannot flatten: pinned keys sit at log(1) = 0 while the
    # history multipliers are forced to sum to 1, so they land at log(1/N) instead.
    pinned = pinned_mask("sink", 6, 12, torch.device("cpu"), n_sink=3)
    lse = history_lse(
        torch.zeros(1, 4, 6, 2, dtype=DT), torch.zeros(1, 12, 2, dtype=DT),
        gate_scale=1.0, pinned=pinned,
    )
    gate = torch.where(pinned, torch.zeros(1, 4, 6, 12, dtype=DT), -lse.unsqueeze(-1))
    assert gap(gate) > 1e-3


def test_sink_pin_folds_into_concat_but_self_does_not():
    """
    The kernel question: a query-independent pin folds, a query-dependent one cannot.

    Folding needs the indexer key zeroed at pinned positions. ``sink`` pins the same keys for
    every query, so a static ``K`` expresses it and one extra concatenated dimension carries the
    rank-one ``-LSE`` term. ``self`` pins a *different* column per row, which no shared ``K`` can
    represent -- hence the separate two-branch path, and hence why a fused kernel is the thing
    that would make ``self`` cheap.
    """
    assert not is_query_dependent("sink")
    assert is_query_dependent("self") and is_query_dependent("self+sink")

    inputs = make_inputs(q_len=12, k_len=12, idx_dim=8)
    dim, idx_dim = inputs["q"].shape[-1], 8
    pinned = pinned_mask("sink", 12, 12, torch.device("cpu"), n_sink=3)
    lse = history_lse(inputs["q_idx"], inputs["k_idx"], gate_scale=0.7, pinned=pinned)
    query, key = build_concat_qk(
        inputs["q"], inputs["k"], inputs["q_idx"], inputs["k_idx"],
        scale=dim**-0.5, gate_scale=0.7, group_size=2, lse=lse, history=~pinned[0],
    )
    # One extra dimension over the un-pinned fold, for the rank-one normalizer.
    assert query.shape[-1] == dim + idx_dim + 1 == key.shape[-1]
    # The pinned keys' gate half is zeroed, so their gate term drops out entirely.
    assert key[..., :3, dim:].abs().max() == 0.0
    assert key[..., 3:, dim:-1].abs().max() > 0.0


@pytest.mark.parametrize("pin_mode,n_sink", [("self", 0), ("self+sink", 3)])
def test_self_pin_matches_two_branch_decomposition(pin_mode, n_sink):
    """
    Independent check on the ``self`` path: history branch + pinned branch, merged by log-sum-exp.

    Deliberately a *different* arithmetic route from the implementation (which builds the gate
    and does one softmax), so agreement is evidence about the operation rather than about one
    way of writing it. The merge is where the ``-LSE`` re-enters: inside the history branch it is
    a per-row constant and cancels, so it only sets how much mass history wins against the
    pinned keys -- which is exactly the budget the pin creates.
    """
    torch.manual_seed(1)
    bsz, n_heads, n_kv_heads, q_len, k_len, dim, idx_dim = 1, 4, 2, 20, 28, 16, 8
    group = n_heads // n_kv_heads
    q = torch.randn(bsz, n_heads, q_len, dim, dtype=DT)
    k = torch.randn(bsz, n_kv_heads, k_len, dim, dtype=DT)
    v = torch.randn(bsz, n_kv_heads, k_len, dim, dtype=DT)
    q_idx = torch.randn(bsz, n_kv_heads, q_len, idx_dim, dtype=DT)
    k_idx = torch.randn(bsz, k_len, idx_dim, dtype=DT)
    scale, gate_scale = dim**-0.5, 0.7

    def expand(x):
        return x.repeat_interleave(group, dim=1) if group > 1 else x

    q_pos = torch.arange(q_len).unsqueeze(-1) + (k_len - q_len)
    causal = torch.arange(k_len).unsqueeze(0) <= q_pos
    pinned = pinned_mask(pin_mode, q_len, k_len, torch.device("cpu"), n_sink=n_sink)
    history = causal & ~pinned

    score = gate_scale * torch.einsum("bhqd,bkd->bhqk", q_idx, k_idx)
    lse = score.masked_fill(~history, -float("inf")).logsumexp(-1, keepdim=True)
    lse = expand(torch.where(~history.any(-1).view(1, 1, q_len, 1), torch.zeros_like(lse), lse))
    base = torch.einsum("bhqd,bhkd->bhqk", q, expand(k)) * scale

    # Branch 1: history only, gated. The -LSE cancels within this softmax.
    hist_logits = (base + expand(score)).masked_fill(~history, -float("inf"))
    lse_hist = hist_logits.logsumexp(-1, keepdim=True)
    out_hist = torch.nan_to_num((hist_logits - lse_hist).exp(), 0.0) @ expand(v)
    # Branch 2: the pinned keys, ungated.
    pin_logits = base.masked_fill(~(pinned & causal), -float("inf"))
    lse_pin = pin_logits.logsumexp(-1, keepdim=True)
    out_pin = torch.nan_to_num((pin_logits - lse_pin).exp(), 0.0) @ expand(v)
    # Merge; the history branch's weight carries the -LSE budget.
    a = torch.nan_to_num(lse_hist - lse, neginf=-1e30)
    b = torch.nan_to_num(lse_pin, neginf=-1e30)
    top = torch.maximum(a, b)
    wa, wb = (a - top).exp(), (b - top).exp()
    merged = (wa * out_hist + wb * out_pin) / (wa + wb)

    got = gated_attention_pinned_self(
        q, k, v, q_idx, k_idx, scaling=scale, gate_scale=gate_scale,
        pin_mode=pin_mode, n_sink=n_sink,
    )
    assert torch.allclose(merged, got, atol=1e-11)


@pytest.mark.parametrize("key_tile", [1, 5, 24, 1024])
@pytest.mark.parametrize("pin_mode,n_sink", [("sink", 4), ("self", 0), ("self+sink", 4)])
def test_history_lse_is_tile_invariant_and_finite(key_tile, pin_mode, n_sink):
    """
    The streaming normalizer is exact at any tile size, and never emits a non-finite value.

    Tile invariance matters because the tile is a memory knob, not a modelling one: if the
    normalizer moved with it, the budget -- and so the trained gate -- would depend on how the
    run was configured to fit in memory.
    """
    torch.manual_seed(0)
    q_len, k_len = 16, 24
    q_idx = torch.randn(2, 3, q_len, 8, dtype=DT)
    k_idx = torch.randn(2, k_len, 8, dtype=DT)
    pinned = pinned_mask(pin_mode, q_len, k_len, torch.device("cpu"), n_sink=n_sink)

    score = 0.7 * torch.einsum("bhqd,bkd->bhqk", q_idx, k_idx)
    q_pos = torch.arange(q_len).unsqueeze(-1) + (k_len - q_len)
    history = (torch.arange(k_len).unsqueeze(0) <= q_pos) & ~pinned
    expected = score.masked_fill(~history, -float("inf")).logsumexp(-1)
    expected = torch.where(torch.isfinite(expected), expected, torch.zeros_like(expected))

    got = history_lse(q_idx, k_idx, gate_scale=0.7, pinned=pinned, key_tile=key_tile)
    assert torch.isfinite(got).all()
    assert torch.allclose(got, expected, atol=1e-12)


def retained_bytes(fn) -> int:
    """
    Bytes autograd keeps alive for the backward pass of ``fn``.

    Uses ``saved_tensors_hooks``, which sees exactly the tensors stashed for backward -- the
    quantity that decides whether a configuration fits in memory. Peak *forward* allocation is a
    different number and is not what OOMs a long-context run: every layer's saved tensors stay
    resident until backward, so a per-layer excess is multiplied by the layer count.
    """
    total = 0

    def pack(tensor):
        nonlocal total
        total += tensor.numel() * tensor.element_size()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        fn()
    return total


@pytest.mark.parametrize("pin_mode,n_sink", [("sink", 4), ("self", 0), ("self+sink", 4)])
def test_history_lse_retains_only_o_l(pin_mode, n_sink):
    """
    The normalizer must keep ``O(L)`` for backward, not ``O(Sq * Sk)``.

    This is a **regression test for a shipped bug**, and the test whose absence allowed it. The
    first implementation was a Python loop of differentiable tile ops: that bounds the *forward*
    peak, which is what its docstring claimed, but autograd retained every tile's intermediates
    until backward. Measured retention was 3.6x the full score matrix -- and it grew as the tile
    shrank, the opposite of what a streaming loop is for. Across 36 layers at ``L=8192`` that
    extrapolated to ~259 GiB and OOM'd on the first step of a real run.

    Asserted against the score matrix's own size, so the bound is about the shape rather than a
    machine-specific byte count: anything at or above ``1x`` means the tiles are being kept.
    """
    q_len = k_len = 256
    n_heads, idx_dim = 8, 32
    q_idx = torch.randn(1, n_heads, q_len, idx_dim, requires_grad=True)
    k_idx = torch.randn(1, k_len, idx_dim, requires_grad=True)
    gate_scale = torch.tensor(0.7, requires_grad=True)
    pinned = pinned_mask(pin_mode, q_len, k_len, torch.device("cpu"), n_sink=n_sink)

    score_bytes = n_heads * q_len * k_len * 4  # the (B, h, Sq, Sk) fp32 matrix
    retained = retained_bytes(
        lambda: history_lse(q_idx, k_idx, gate_scale=gate_scale, pinned=pinned, key_tile=64)
    )
    assert retained < score_bytes / 4, (
        f"history_lse retained {retained / 2**20:.2f} MiB against a "
        f"{score_bytes / 2**20:.2f} MiB score matrix -- the score tiles are being kept for "
        "backward, which is O(Sq*Sk) per layer and OOMs a long-context run"
    )


def test_history_lse_retention_does_not_grow_as_the_tile_shrinks():
    """
    A smaller tile must not cost *more* retained memory.

    The exact inversion the original bug produced: more tiles meant more saved intermediates, so
    the knob intended to reduce memory increased it. Worth its own test because the tile-size
    parameter is the first thing anyone reaches for when a run OOMs.
    """
    q_len = k_len = 256
    q_idx = torch.randn(1, 4, q_len, 32, requires_grad=True)
    k_idx = torch.randn(1, k_len, 32, requires_grad=True)
    pinned = pinned_mask("sink", q_len, k_len, torch.device("cpu"), n_sink=4)

    retained = {
        tile: retained_bytes(
            lambda t=tile: history_lse(
                q_idx, k_idx, gate_scale=0.7, pinned=pinned, key_tile=t
            )
        )
        for tile in (16, 64, 256)
    }
    assert retained[16] == retained[64] == retained[256], (
        f"retention varies with key_tile: {retained} -- the tiles are entering the autograd graph"
    )


def test_history_lse_gradients_match_a_dense_reference():
    """
    Recomputing in backward must give exactly the gradients the naive form would.

    The closed form used is ``d(lse)/d(s_k) = softmax(s)_k``, rebuilt per tile from the saved
    ``lse``. Checked against a single-shot differentiable ``logsumexp`` in fp64 -- if the
    hand-written backward were wrong, every earlier correctness test would still pass (they only
    compare forward values and go through the same function), so this is the one that pins it.
    """
    torch.manual_seed(0)
    q_len, k_len = 24, 32
    pinned = pinned_mask("self+sink", q_len, k_len, torch.device("cpu"), n_sink=3)
    q_pos = torch.arange(q_len).unsqueeze(-1) + (k_len - q_len)
    history = (torch.arange(k_len).unsqueeze(0) <= q_pos) & ~pinned

    results = {}
    for name in ("dense", "streamed"):
        q_idx = torch.randn(2, 3, q_len, 8, dtype=DT, requires_grad=True)
        k_idx = torch.randn(2, k_len, 8, dtype=DT, requires_grad=True)
        with torch.no_grad():  # identical inputs for both paths
            torch.manual_seed(1)
            q_idx.copy_(torch.randn_like(q_idx))
            k_idx.copy_(torch.randn_like(k_idx))
        gate_scale = torch.tensor(0.7, dtype=DT, requires_grad=True)

        if name == "dense":
            score = gate_scale * torch.einsum("bhqd,bkd->bhqk", q_idx, k_idx)
            out = score.masked_fill(~history, -float("inf")).logsumexp(-1)
            out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        else:
            out = history_lse(
                q_idx, k_idx, gate_scale=gate_scale, pinned=pinned, key_tile=7
            )
        torch.manual_seed(2)
        (out * torch.randn_like(out)).sum().backward()
        results[name] = (out.detach(), q_idx.grad.clone(), k_idx.grad.clone(), gate_scale.grad.clone())

    for index, label in enumerate(("lse", "d q_idx", "d k_idx", "d gate_scale")):
        assert torch.allclose(results["dense"][index], results["streamed"][index], atol=1e-11), (
            f"{label} disagrees with the dense reference"
        )


def test_value_is_padded_so_flash_stays_eligible():
    """
    **Regression test for a shipped OOM.** Flash requires ``Q.size(-1) == V.size(-1)``.

    The concat trick widens Q and K to ``D + Di`` while V stays at ``Dv``. That mismatch makes
    SDPA fall back to the **math** backend, which materializes the full ``(B, H, Sq, Sk)``
    attention weights *and retains them for backward*: 4.0 GiB per layer at ``L=8192, Hq=32`` in
    bf16, so 144 GiB across 36 layers. The first real run OOM'd inside SDPA for exactly this
    reason.

    The failure is invisible to a correctness test -- SDPA returns the right numbers on the math
    backend too. What shipped the bug was checking that SDPA *accepted* the shapes rather than
    which backend it *chose*. So this test asserts the shape property flash actually requires,
    which is the thing that has to hold.
    """
    dim, idx_dim = 16, 12
    inputs = make_inputs(dim=dim, idx_dim=idx_dim)
    query, _ = build_concat_qk(
        inputs["q"], inputs["k"], inputs["q_idx"], inputs["k_idx"],
        scale=dim**-0.5, gate_scale=0.7, group_size=2,
    )
    padded, original = pad_value_to_width(inputs["v"], query.shape[-1])
    assert original == dim
    assert padded.shape[-1] == query.shape[-1] == dim + idx_dim, (
        "V must be widened to the concatenated query width, or flash is ineligible and SDPA "
        "silently drops to the math backend"
    )
    # The pad is zero, so the sliced-off columns cannot leak into the kept output.
    assert padded[..., dim:].abs().max() == 0.0
    assert torch.equal(padded[..., :dim], inputs["v"])
    # A no-op when the widths already agree, so the fast path costs nothing.
    same, _ = pad_value_to_width(inputs["v"], dim)
    assert same is inputs["v"]


def test_padding_v_does_not_change_the_result():
    """
    Padding V and slicing the output back is exact, in the forward and every gradient.

    Zero rows contribute nothing to ``P @ V``, so the kept columns are untouched -- but this is
    the kind of "obviously fine" step that is worth pinning, because it sits between the gate and
    the model output where an off-by-one in the slice would be a subtle numerical bug rather than
    a crash.
    """
    dim, idx_dim = 16, 12
    results = {}
    for name in ("direct", "padded"):
        inputs = make_inputs(dim=dim, idx_dim=idx_dim)
        scale = dim**-0.5
        query, key = build_concat_qk(
            inputs["q"], inputs["k"], inputs["q_idx"], inputs["k_idx"],
            scale=scale, gate_scale=0.7, group_size=2,
        )
        if name == "direct":
            value, keep = inputs["v"], dim
        else:
            value, keep = pad_value_to_width(inputs["v"], query.shape[-1])
        out = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True, scale=scale, enable_gqa=True
        )[..., :keep]
        torch.manual_seed(1)
        (out * torch.randn_like(out)).sum().backward()
        results[name] = (
            out.detach(),
            inputs["q_idx"].grad.clone(),
            inputs["k_idx"].grad.clone(),
            inputs["v"].grad.clone(),
        )
    for index, label in enumerate(("out", "d q_idx", "d k_idx", "d v")):
        assert torch.allclose(results["direct"][index], results["padded"][index], atol=1e-11), (
            f"{label} changed when V was padded"
        )


def test_padding_v_makes_retention_linear_not_quadratic():
    """
    The pad is what turns SDPA's retention from ``O(L^2)`` into ``O(L)``. Measured, not asserted.

    Without it ``Dqk != Dv``, no fused backend is eligible, and the math fallback keeps the whole
    attention matrix for backward. Doubling ``L`` then roughly **quadruples** retention; with the
    pad it merely doubles. Measured here as a growth *ratio*, which is machine-independent in a way
    a byte count is not.

    This runs on CPU because torch ships a fused CPU SDPA kernel, so the backend choice this test
    depends on is reproduced without a GPU -- which is also how the bug could have been caught
    before it reached one.
    """
    dim = 16
    growth = {}
    for padded in (False, True):
        retained = []
        for q_len in (128, 256, 512):
            q = torch.randn(1, 8, q_len, dim, requires_grad=True)
            k = torch.randn(1, 4, q_len, dim, requires_grad=True)
            v = torch.randn(1, 4, q_len, dim, requires_grad=True)
            q_idx = torch.randn(1, 4, q_len, dim, requires_grad=True)
            k_idx = torch.randn(1, q_len, dim, requires_grad=True)
            query, key = build_concat_qk(
                q, k, q_idx, k_idx, scale=dim**-0.5, gate_scale=0.7, group_size=2
            )
            value, keep = (
                pad_value_to_width(v, query.shape[-1]) if padded else (v, dim)
            )
            retained.append(
                retained_bytes(
                    lambda: torch.nn.functional.scaled_dot_product_attention(
                        query, key, value, is_causal=True, scale=dim**-0.5, enable_gqa=True
                    )[..., :keep]
                )
            )
        # Growth factor per doubling of L: ~2 for O(L), ~4 for O(L^2).
        growth[padded] = (retained[2] / retained[1] + retained[1] / retained[0]) / 2

    assert growth[True] < 2.5, (
        f"padded retention grows {growth[True]:.2f}x per doubling -- expected ~2x (linear)"
    )
    assert growth[False] > 3.0, (
        f"unpadded retention grows only {growth[False]:.2f}x per doubling; this test can no "
        "longer tell the backends apart, so it is not protecting anything"
    )


def test_empty_history_row_stays_finite():
    """
    A row whose only visible key is pinned has no history, and must not produce NaN.

    Under ``self`` pinning with ``Sq == Sk`` this is query 0: it sees only its own diagonal, which
    is pinned. The ``logsumexp`` of an empty set is ``-inf``, so ``score - lse`` would be ``+inf``
    and the softmax NaN -- which would then spread through the whole model rather than staying
    local. Such rows get an inert gate instead; they have one key to attend to and nothing to
    rank.
    """
    torch.manual_seed(0)
    q_len = k_len = 8
    q_idx = torch.randn(1, 2, q_len, 4, dtype=DT)
    k_idx = torch.randn(1, k_len, 4, dtype=DT)
    pinned = pinned_mask("self", q_len, k_len, torch.device("cpu"))
    lse = history_lse(q_idx, k_idx, gate_scale=0.7, pinned=pinned)
    assert torch.isfinite(lse).all()
    assert lse[0, 0, 0] == 0.0  # query 0: empty history -> inert

    inputs = make_inputs(q_len=q_len, k_len=k_len, dim=8, idx_dim=4)
    for fn in (gated_attention_reference, gated_attention_full):
        out = fn(**inputs, gate_scale=0.7, pin_mode="self")
        assert torch.isfinite(out).all(), f"{fn.__name__} produced a non-finite output"


def test_pin_mode_is_rejected_under_the_sparse_scope():
    """
    Pinning the sparse scope would be inert, so it raises rather than pretending to apply.

    The sparse forward is already restricted to the router's own top-k, so a flat gate does not
    recover dense attention -- the same structural reason DMA and SparseK need no pin. Silently
    accepting the flag would let a run report having pinned when it had not.
    """
    inputs = make_inputs()
    indices = torch.zeros(2, 4, 9, 3, dtype=torch.int32)
    with pytest.raises(ValueError, match="meaningless under scope='sparse'"):
        gated_attention(**inputs, scope="sparse", indices=indices, pin_mode="self")


def test_pinned_mask_geometry():
    """``pinned_mask`` marks the right cells, including under bottom-right alignment."""
    assert pinned_mask("none", 4, 6, torch.device("cpu"), n_sink=2) is None

    sink = pinned_mask("sink", 3, 5, torch.device("cpu"), n_sink=2)
    assert sink[:, :2].all() and not sink[:, 2:].any()

    # Sq=3, Sk=5 -> query i sits at absolute position i + 2, so the diagonal is columns 2,3,4.
    self_pin = pinned_mask("self", 3, 5, torch.device("cpu"))
    assert self_pin.nonzero().tolist() == [[0, 2], [1, 3], [2, 4]]

    both = pinned_mask("self+sink", 3, 5, torch.device("cpu"), n_sink=2)
    assert both[:, :2].all() and both[0, 2] and both[1, 3] and both[2, 4]


# ----------------------------------------------------------------------
# 3. Gradient structure: why full scope
# ----------------------------------------------------------------------
def test_full_scope_gradients_are_independent():
    """
    Unselected keys get content-dependent gradients under full scope, but not under sparse.

    This is the SAS Figure 5 diagnostic and the reason stage 1 pays ``O(L^2)``. Under sparse
    scope an unselected key contributes nothing to the output, so it has no gate gradient of its
    own and moves only through the softmax normalizer -- which makes ``ds_j`` an exact affine
    function of the key's own gate value, i.e. the whole unselected set is dragged together
    rather than judged individually. SAS observes ``R^2 = 1.00`` on that scatter; here the
    stronger, discrete statement is available: the sparse-scope gradient at unselected slots is
    *identically zero*, while the full-scope one is not.
    """
    inputs = make_inputs(q_len=8, k_len=8, n_heads=4, n_kv_heads=2, seed=3)
    k_len, topk = 8, 3
    # A fixed support: the first `topk` keys. Rows are causal, so key 0..2 are legal everywhere.
    indices = (
        torch.arange(topk).view(1, 1, 1, topk).expand(2, 2, 8, topk).contiguous().int()
    )

    _, full_grads = backward_of(gated_attention_full, make_inputs(
        q_len=8, k_len=8, n_heads=4, n_kv_heads=2, seed=3), 1.0)

    lam = torch.tensor(1.0, dtype=DT, requires_grad=True)
    sparse_inputs = make_inputs(q_len=8, k_len=8, n_heads=4, n_kv_heads=2, seed=3)
    out = gated_attention_sparse(**sparse_inputs, indices=indices, gate_scale=lam)
    torch.manual_seed(1)
    (out * torch.randn_like(out)).sum().backward()
    sparse_grad_ki = sparse_inputs["k_idx"].grad.clone()

    # k_idx rows 3.. are never selected, so under sparse scope they receive nothing.
    assert sparse_grad_ki[:, topk:].abs().max() == 0.0
    # Under full scope those same keys are gated and do receive their own signal.
    assert full_grads["k_idx"][:, topk:].abs().max() > 1e-8


def test_sparse_scope_gradient_is_confined_to_the_support():
    """Sparse scope trains only the selected keys -- the complement of the test above."""
    inputs = make_inputs(q_len=6, k_len=6, n_heads=4, n_kv_heads=2, seed=5)
    keep = 2
    indices = torch.arange(keep).view(1, 1, 1, keep).expand(2, 2, 6, keep).contiguous().int()
    out = gated_attention_sparse(**inputs, indices=indices, gate_scale=1.0)
    torch.manual_seed(1)
    (out * torch.randn_like(out)).sum().backward()
    assert inputs["k_idx"].grad[:, :keep].abs().max() > 1e-8
    assert inputs["k_idx"].grad[:, keep:].abs().max() == 0.0


# ----------------------------------------------------------------------
# Shape and alignment contracts
# ----------------------------------------------------------------------
def test_causal_mask_is_bottom_right_aligned():
    """
    The mask must be bottom-right aligned, matching ``build_indexer_mask`` and flash-attention.

    SDPA's ``is_causal=True`` is **top-left** aligned and the two coincide only when
    ``Sq == Sk``. At ``Sq=3, Sk=5`` top-left lets query 0 see key 0 alone where bottom-right
    lets it see keys 0-2 -- both well-formed, so a mix-up is silent, and it would train the gate
    against a different key set than the indexer scored on any decode or chunked-prefill step.
    """
    mask = causal_mask_bottom_right(3, 5, torch.device("cpu"), DT)
    visible = (mask[0, 0] == 0).sum(-1).tolist()
    assert visible == [3, 4, 5]

    # And the op agrees with SDPA's own is_causal exactly when the shapes are square.
    square = make_inputs(q_len=6, k_len=6)
    explicit = gated_attention_reference(
        **square, mask=causal_mask_bottom_right(6, 6, torch.device("cpu"), DT), gate_scale=0.5
    )
    implicit = gated_attention_full(**make_inputs(q_len=6, k_len=6), gate_scale=0.5)
    assert torch.allclose(explicit, implicit, atol=1e-11)


def test_query_longer_than_keys_raises():
    """``Sq > Sk`` would leave leading queries with no visible key, hence NaN -- so it raises."""
    with pytest.raises(ValueError, match="exceeds k_len"):
        gated_attention_reference(**make_inputs(q_len=11, k_len=5))


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda d: d.update(q=d["q"][:, :, :, :-1]), "head_dim"),
        (lambda d: d.update(q_idx=d["q_idx"][:, :-1]), "q_idx must be"),
        (lambda d: d.update(k_idx=d["k_idx"][:, :-1]), "k_idx must be"),
        (lambda d: d.update(v=d["v"][:, :, :-1]), "v has"),
        (lambda d: d.update(k_idx=d["k_idx"].unsqueeze(1)), "k_idx must be"),
    ],
)
def test_shape_checks_are_eager(mutate, match):
    """Every one of these mismatches is otherwise silent -- wrong numbers, no crash."""
    inputs = make_inputs()
    mutate(inputs)
    with pytest.raises(ValueError, match=match):
        check_gate_shapes(inputs["q"], inputs["k"], inputs["v"], inputs["q_idx"], inputs["k_idx"])


def test_scope_dispatch_rejects_contradictory_arguments():
    """Passing indices to the full scope, or omitting them from the sparse one, is an error."""
    inputs = make_inputs()
    indices = torch.zeros(2, 4, 9, 3, dtype=torch.int32)
    with pytest.raises(ValueError, match="would be ignored"):
        gated_attention(**inputs, scope="full", indices=indices)
    with pytest.raises(ValueError, match="needs `indices`"):
        gated_attention(**inputs, scope="sparse")
    with pytest.raises(ValueError, match="takes its masking from"):
        gated_attention(**inputs, scope="sparse", indices=indices, mask=torch.zeros(2, 1, 9, 9))
    with pytest.raises(ValueError, match="scope must be"):
        gated_attention(**inputs, scope="nonsense")


# ----------------------------------------------------------------------
# 4. Wiring, on a real model
# ----------------------------------------------------------------------
@pytest.mark.parametrize("stage", ["dense", "sparse"])
def test_lm_loss_reaches_every_indexer_parameter(stage):
    """
    The whole point: the model's own loss produces a gradient on every indexer parameter.

    No auxiliary objective and no teacher -- if this passes, the router is on the forward path.
    """
    model, config = tiny_model()
    press = GQAIndexerPress(compression_ratio=0.5, gate_scale=True)
    trainer = E2EIndexerTrainer(press=press, stage=stage, keep_ratio=0.5)
    input_ids = torch.randint(0, config.vocab_size, (2, 24))

    loss = e2e_indexer_training_step(model, trainer, input_ids=input_ids)
    loss.backward()

    assert trainer.layers_gated == config.num_hidden_layers
    params = trainer.indexer_parameters(model)
    assert params, "no indexer parameters found"
    for param in params:
        assert param.grad is not None, "an indexer parameter received no gradient"
        assert param.grad.abs().max() > 0, "an indexer parameter received a zero gradient"


def test_backbone_is_frozen_and_indexers_are_not():
    """Only the indexers train, so the comparison against distillation is at matched capacity."""
    model, _ = tiny_model()
    press = GQAIndexerPress(compression_ratio=0.5, gate_scale=True)
    trainer = E2EIndexerTrainer(press=press)
    press.post_init_from_model(model)
    trainer.freeze_backbone(model)

    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable, "everything was frozen, including the indexers"
    assert all(f".{press.scorer_attr}." in name for name in trainable)
    # Every indexer parameter is in there, not just some.
    assert len(trainable) == len(trainer.indexer_parameters(model))


def test_attention_implementation_is_restored():
    """The registry entry and ``_attn_implementation`` must survive an exception."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    model, config = tiny_model()
    press = GQAIndexerPress(compression_ratio=0.5, gate_scale=True)
    trainer = E2EIndexerTrainer(press=press)
    before = model.config._attn_implementation
    registered_before = "kvpress_gqa_indexer_gated" in type(ALL_ATTENTION_FUNCTIONS)._global_mapping

    with pytest.raises(RuntimeError, match="boom"):
        with trainer.hooks(model):
            raise RuntimeError("boom")

    assert model.config._attn_implementation == before
    after = "kvpress_gqa_indexer_gated" in type(ALL_ATTENTION_FUNCTIONS)._global_mapping
    assert after == registered_before


def test_gating_changes_the_forward_and_leaves_no_residue():
    """
    The gate must actually alter the model's output, and only while the hooks are active.

    Together these rule out the two failure modes that a loss curve cannot distinguish: the
    swap never taking effect (loss looks fine, router untrained), and the swap leaking past the
    context (evaluation silently running gated attention).
    """
    model, config = tiny_model()
    input_ids = torch.randint(0, config.vocab_size, (2, 20))
    press = GQAIndexerPress(compression_ratio=0.5, gate_scale=True)
    trainer = E2EIndexerTrainer(press=press, freeze=False)

    with torch.no_grad():
        ungated = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
        with trainer.hooks(model):
            gated = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
        restored = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss

    assert not torch.allclose(ungated, gated), "the gate had no effect on the forward pass"
    assert torch.equal(ungated, restored), "gating leaked past the context manager"


def test_lm_loss_descends():
    """
    A short optimization run must reduce the LM loss.

    Weak as a quality claim, strong as a wiring claim: it can only happen if the gradient
    reaching the router is the one that actually lowers the model's own objective.
    """
    model, config = tiny_model()
    press = GQAIndexerPress(compression_ratio=0.5, gate_scale=True)
    trainer = E2EIndexerTrainer(press=press, stage="dense")
    press.post_init_from_model(model)
    input_ids = torch.randint(0, config.vocab_size, (2, 24))

    optimizer = torch.optim.Adam(trainer.indexer_parameters(model), lr=3e-2)
    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        loss = e2e_indexer_training_step(model, trainer, input_ids=input_ids)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"loss did not descend: {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert trainer.mean_gate_scale() is not None


def test_gate_scale_is_reported_per_layer():
    """
    ``gate_scales`` exposes each layer's learned strength.

    Worth having as a first-class readout: a layer whose ``gate_scale`` collapses towards zero is
    one whose router is not earning its place, and the loss curve alone would not say so.
    """
    model, config = tiny_model()
    press = GQAIndexerPress(compression_ratio=0.5, gate_scale=True)
    trainer = E2EIndexerTrainer(press=press)
    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    e2e_indexer_training_step(model, trainer, input_ids=input_ids)

    assert set(trainer.gate_scales) == set(range(config.num_hidden_layers))
    head_dim = config.head_dim
    assert all(v == pytest.approx(head_dim**-0.5) for v in trainer.gate_scales.values())


def test_press_without_gate_scale_is_rejected():
    """
    Forgetting ``gate_scale=True`` must fail loudly, not train a fixed-scale ablation.

    The failure is otherwise invisible: the run would produce a descending loss and a trained
    indexer, just not the configured one.
    """
    model, config = tiny_model()
    press = GQAIndexerPress(compression_ratio=0.5)  # gate_scale defaults to False
    trainer = E2EIndexerTrainer(press=press)
    input_ids = torch.randint(0, config.vocab_size, (1, 12))
    with pytest.raises(RuntimeError, match="gate_scale"):
        e2e_indexer_training_step(model, trainer, input_ids=input_ids)


def test_distillation_checkpoints_carry_no_gate_parameter():
    """
    ``gate_scale=False`` keeps the parameter out of the state dict entirely.

    So a distillation-trained checkpoint stays byte-compatible with what it was before this
    feature existed, and the two objectives can share checkpoints in the direction that matters
    (``load_indexer_state_dict`` uses ``strict=False``).
    """
    plain = GQAIndexer(GQAIndexerConfig(hidden_size=64, n_heads=2, head_dim=8))
    gated = GQAIndexer(GQAIndexerConfig(hidden_size=64, n_heads=2, head_dim=8, gate_scale=True))
    assert "gate_scale" not in plain.state_dict()
    assert "gate_scale" in gated.state_dict()
    # A distillation checkpoint loads into a gated indexer, leaving gate_scale at its init.
    missing, unexpected = gated.load_state_dict(plain.state_dict(), strict=False)
    assert list(missing) == ["gate_scale"] and not list(unexpected)


@pytest.mark.parametrize("stage", ["dense", "sparse"])
def test_trainer_state_resets_between_passes(stage):
    """Per-pass bookkeeping must not accumulate across steps."""
    model, config = tiny_model()
    press = GQAIndexerPress(compression_ratio=0.5, gate_scale=True)
    trainer = E2EIndexerTrainer(press=press, stage=stage, keep_ratio=0.5)
    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    for _ in range(3):
        e2e_indexer_training_step(model, trainer, input_ids=input_ids)
        assert trainer.layers_gated == config.num_hidden_layers


def test_invalid_configuration_is_rejected_at_construction():
    """Typos surface when the trainer is built, not when the run reaches stage 2."""
    press = GQAIndexerPress(compression_ratio=0.5, gate_scale=True)
    with pytest.raises(ValueError, match="stage must be"):
        E2EIndexerTrainer(press=press, stage="full")
    with pytest.raises(ValueError, match="keep_ratio"):
        E2EIndexerTrainer(press=press, keep_ratio=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        E2EIndexerTrainer(press=press, force_sink=-1)
    with pytest.raises(ValueError, match="pin_mode must be"):
        E2EIndexerTrainer(press=press, pin_mode="bogus")
    with pytest.raises(ValueError, match="does not apply to stage='sparse'"):
        E2EIndexerTrainer(press=press, stage="sparse", pin_mode="self")


def test_pin_mode_resolves_from_the_stage():
    """
    Left unset, ``pin_mode`` resolves to ``"self"`` for dense and ``"none"`` for sparse.

    The sparse scope makes pinning inert, so the stage already determines the answer -- and
    requiring every sparse-stage caller to restate it would make the natural spelling
    (``stage="sparse"`` alone) an error.
    """
    press = GQAIndexerPress(compression_ratio=0.5, gate_scale=True)
    assert E2EIndexerTrainer(press=press, stage="dense").gate_pin_mode == "self"
    assert E2EIndexerTrainer(press=press, stage="sparse").gate_pin_mode == "none"
    assert E2EIndexerTrainer(press=press, pin_mode="sink").gate_pin_mode == "sink"


@pytest.mark.parametrize("pin_mode,n_sink", PIN_CASES)
def test_every_pin_mode_trains_end_to_end(pin_mode, n_sink):
    """Each pin mode runs on a real model and delivers gradient to every indexer parameter."""
    model, config = tiny_model()
    press = GQAIndexerPress(compression_ratio=0.5, gate_scale=True, n_sink=n_sink)
    trainer = E2EIndexerTrainer(press=press, stage="dense", pin_mode=pin_mode)
    input_ids = torch.randint(0, config.vocab_size, (2, 24))

    loss = e2e_indexer_training_step(model, trainer, input_ids=input_ids)
    loss.backward()

    assert trainer.layers_gated == config.num_hidden_layers
    assert trainer.sink_count == n_sink
    for param in trainer.indexer_parameters(model):
        assert param.grad is not None and param.grad.abs().max() > 0


def test_trainer_defaults_to_the_press_sink_count():
    """
    ``n_sink=None`` inherits the press's own value.

    So the keys the press protects from eviction at inference are the keys the gate exempts
    during training. Letting them drift apart would train the router to spend budget on keys
    that are kept regardless.
    """
    press = GQAIndexerPress(compression_ratio=0.5, gate_scale=True, n_sink=7)
    assert E2EIndexerTrainer(press=press).sink_count == 7
    assert E2EIndexerTrainer(press=press, n_sink=2).sink_count == 2


def test_captured_hidden_states_are_freed_after_use():
    """
    Each layer's captured ``hidden_states`` must be released once the gate has consumed it.

    The pre-hook stashes ``hidden_states`` because the attention *interface* never receives it, and
    exactly one layer's copy is needed at a time -- the hook fires immediately before that layer's
    gated attention. Leaving the entry in the dict instead pinned every layer's ``(B, L, hidden)``
    tensor for the rest of the forward **and the whole backward**, since the context manager only
    clears on exit. At ``L=16384`` on Qwen3-8B that is 36 x 0.125 GiB held reachable, and it also
    prevents autograd from releasing each layer's activations as backward walks up the stack.

    Asserted at the moment the *last* layer runs, which is when the old behaviour peaked.
    """
    model, config = tiny_model()
    press = GQAIndexerPress(compression_ratio=0.5, gate_scale=True)
    trainer = E2EIndexerTrainer(press=press, pin_mode="sink")
    input_ids = torch.randint(0, config.vocab_size, (1, 32))

    peak = 0
    original = trainer.gated_forward

    def spy(*args, **kwargs):
        nonlocal peak
        out = original(*args, **kwargs)
        peak = max(peak, len(trainer._hidden_states))
        return out

    trainer.gated_forward = spy
    loss = e2e_indexer_training_step(model, trainer, input_ids=input_ids)
    loss.backward()

    assert trainer.layers_gated == config.num_hidden_layers
    assert peak == 0, (
        f"{peak} layers' hidden_states were still held after their gate ran; each one pins a "
        "(B, L, hidden) tensor across the whole forward and backward"
    )
    # And the gate still works -- popping must not break the thing it feeds.
    for param in trainer.indexer_parameters(model):
        assert param.grad is not None and param.grad.abs().max() > 0


def test_skip_logits_is_only_forwarded_when_asked():
    """
    ``skip_logits`` reaches the model only when passed, so an unpatched model still works.

    It exists for liger's fused linear+CE, which fuses ``lm_head`` into the loss and never
    materializes the ``(L, vocab)`` logits -- 7.0 GiB at ``L=8192`` on Qwen3-8B. It has to be
    explicit because liger's own default is ``self.training and labels is not None``, and this
    backbone is deliberately kept in ``eval()`` to hold dropout off, so the default would resolve to
    ``False`` and the patch would silently save nothing. A plain HF model does not accept the kwarg
    at all, which is why the default here is ``None`` (omit) rather than ``False`` (pass).
    """
    model, config = tiny_model()
    press = GQAIndexerPress(compression_ratio=0.5, gate_scale=True)
    trainer = E2EIndexerTrainer(press=press, pin_mode="sink")
    input_ids = torch.randint(0, config.vocab_size, (1, 16))

    # Omitted: works on an unpatched model.
    baseline = e2e_indexer_training_step(model, trainer, input_ids=input_ids)
    assert torch.isfinite(baseline)

    # Recorded so the assertion is about what the model received, not about liger being installed.
    seen = {}
    original = type(model).forward

    def spy(self, *args, **kwargs):
        seen["skip_logits"] = kwargs.get("skip_logits", "<absent>")
        kwargs.pop("skip_logits", None)
        return original(self, *args, **kwargs)

    type(model).forward = spy
    try:
        e2e_indexer_training_step(model, trainer, input_ids=input_ids)
        assert seen["skip_logits"] == "<absent>", "None must not be forwarded"
        e2e_indexer_training_step(model, trainer, input_ids=input_ids, skip_logits=True)
        assert seen["skip_logits"] is True
        e2e_indexer_training_step(model, trainer, input_ids=input_ids, skip_logits=False)
        assert seen["skip_logits"] is False
    finally:
        type(model).forward = original


def test_unpinned_mode_warns(caplog):
    """
    ``pin_mode="none"`` warns, because it is a valid ablation and an invalid default.

    It leaves the no-op reachable, so a run can report a healthy descending loss while the router
    has learned nothing -- the failure mode with no other symptom.
    """
    press = GQAIndexerPress(compression_ratio=0.5, gate_scale=True)
    with caplog.at_level("WARNING"):
        E2EIndexerTrainer(press=press, pin_mode="none")
    assert any("no-op" in record.message for record in caplog.records)
