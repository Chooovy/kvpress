# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the cross-replay LM loss (``cross_replay.py``).

Three of the properties below are **silent** failure modes: get them wrong and training runs
cleanly, the loss falls, and the router learns nothing. They are the reason this file exists rather
than a smoke test.

* ``pin_mode="self"`` pins zero keys under ``[C ; C']``, reopening the flat-gate no-op.
* An implicit mask yields a causal triangle, not the cross-context rectangle.
* ``use_cache=False`` does not stop the cache growing, which would admit ``C'`` to its own history.

Plus two exactness properties the memory story rests on: query-chunking must reproduce the unchunked
gradient, and the broadcast gate must equal an explicitly materialized one.
"""

from __future__ import annotations

import gc
import math

import pytest
import torch
import torch.nn.functional as F
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM

from kvpress.presses.gqa_indexer.cross_replay import (
    CrossReplayTrainer,
    ReadOnlyCache,
    cross_replay_training_step,
    rectangle_mask,
)
from kvpress.presses.gqa_indexer.gate_pin import pinned_mask
from kvpress.presses.gqa_indexer.press import GQAIndexerPress

N_LAYERS, HIDDEN, N_HEADS, N_KV_HEADS, HEAD_DIM = 2, 64, 4, 2, 16


def _retained_bytes() -> int:
    """Total bytes in live tensors -- a device-independent stand-in for peak memory.

    ``torch.cuda.max_memory_allocated`` is the real instrument but is unavailable on CPU, and this
    suite has to run without a GPU. Walking the GC is coarse but sufficient for the property under
    test: whether a chunk's graph is *released* or *retained*, which is an order-of-magnitude
    difference rather than a marginal one.
    """
    total, seen = 0, set()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and id(obj) not in seen:
                seen.add(id(obj))
                total += obj.numel() * obj.element_size()
        except Exception:  # noqa: BLE001 -- some objects raise on is_tensor during traversal
            pass
    return total


def _model(max_pos: int = 512):
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        max_position_embeddings=max_pos,
        attn_implementation="sdpa",
    )
    return Qwen3ForCausalLM(config).eval().to(torch.float64)


def _press(**kw):
    kw.setdefault("compression_ratio", 0.5)
    kw.setdefault("scorer", "scalar")
    kw.setdefault("gate_scale", True)
    kw.setdefault("n_sink", 2)
    return GQAIndexerPress(**kw)


def _trainer(press=None, **kw):
    return CrossReplayTrainer(press=press if press is not None else _press(), **kw)


# ----------------------------------------------------------------------
# Silent failure 1: the pin that pins nothing
# ----------------------------------------------------------------------
@pytest.mark.parametrize("pin_mode", ["self", "self+sink"])
def test_self_pin_is_rejected_because_it_pins_nothing_here(pin_mode):
    """``self`` pinning must be refused, not silently accepted.

    Query ``j``'s diagonal key lies in the ``C'`` block, which this objective masks out, so the
    pinned set *inside the visible key axis* is empty. With no pins the gate's normalizer is a
    per-row constant that cancels in the attention softmax, making it exactly interchangeable with a
    raw score -- the no-op the pin exists to close. The failure is invisible in the loss, hence an
    error at construction.
    """
    with pytest.raises(ValueError, match="pins nothing"):
        _trainer(pin_mode=pin_mode)


def test_self_pin_really_pins_zero_keys_in_this_geometry():
    """The geometric fact behind the rejection above, asserted directly.

    Without this, the rejection is a rule someone could 'fix' by relaxing it.
    """
    n = 16
    pinned = pinned_mask("self", n, 2 * n, torch.device("cpu"), n_sink=0)
    assert pinned[:, :n].sum() == 0, "a self pin should pin nothing inside C"
    assert pinned[:, n:].sum() == n, "every self pin lands in the masked-out C' block"

    # sink, by contrast, pins inside C for every query.
    sink = pinned_mask("sink", n, 2 * n, torch.device("cpu"), n_sink=2)
    assert sink[:, :2].all() and sink[:, 2:].sum() == 0


def test_zero_sinks_rejected():
    """Pinning zero keys makes the normalizer inert, which reopens the no-op."""
    with pytest.raises(ValueError, match="n_sink must be positive"):
        _trainer(n_sink=0)


def test_none_pin_rejected():
    with pytest.raises(ValueError, match="flatten into a no-op"):
        _trainer(pin_mode="none")


# ----------------------------------------------------------------------
# Silent failure 2: the mask must be explicit
# ----------------------------------------------------------------------
def test_gate_reaches_sdpa_with_the_query_axis_unmaterialized():
    """The gate must arrive at SDPA as ``(B, H, 1, N)``, never ``(B, H, Sq, N)``.

    This is the bug that caused a 73.1 GiB peak at 8K against a 23.9 GiB estimate. Adding the
    all-zero rectangle mask to the gate broadcasts the sum into a full ``(B, H, Sq, N)`` tensor --
    0.50 GiB per layer at Sq=1024, N=8192, i.e. 18 GiB over Qwen3-8B's 36 layers, and it also pushes
    SDPA onto a backend that retains the attention matrix. Adding zeros is never necessary.

    Asserted on the shape reaching SDPA rather than on memory, because this is the mechanism; the
    memory consequence is what made it visible.
    """
    model, trainer = _model(), _trainer(query_chunk=8)
    ids = torch.randint(0, 128, (1, 16))
    shapes = []
    original_sdpa = F.scaled_dot_product_attention

    def spy(query, key, value, attn_mask=None, **kwargs):
        shapes.append(None if attn_mask is None else tuple(attn_mask.shape))
        return original_sdpa(query, key, value, attn_mask=attn_mask, **kwargs)

    with trainer.hooks(model):
        F.scaled_dot_product_attention = spy
        try:
            cross_replay_training_step(model, trainer, input_ids=ids)
        finally:
            F.scaled_dot_product_attention = original_sdpa

    gated = [s for s in shapes if s is not None]
    assert gated, "no gated attention call reached SDPA"
    for shape in gated:
        assert shape[2] == 1, (
            f"SDPA received a mask of shape {shape}: the query axis was materialized, which costs "
            "O(Sq * Sk) per layer -- the gate is query-independent and must stay broadcast"
        )


def test_rectangle_mask_is_tagged_all_zero():
    """The tag is how the all-zero rectangle is recognized without a device sync."""
    mask = rectangle_mask(4, 8, torch.device("cpu"), torch.float32)
    assert getattr(mask, "_kvpress_all_zero", False)
    assert (mask == 0).all()


def test_a_real_mask_is_still_honoured():
    """Only the tagged all-zero rectangle is dropped; a mask that forbids a key must survive.

    Guards the obvious way to break the fix above -- ignoring ``attention_mask`` altogether, which
    would also 'save memory' while silently discarding masking.
    """
    model, trainer = _model(), _trainer()
    ids = torch.randint(0, 128, (1, 8))
    with trainer.hooks(model):
        cache, hidden = trainer.prefill(model, ids)
        trainer.score_context(model, hidden)
        trainer._context_len = 8

        blocking = torch.zeros(1, 1, 8, 8, dtype=model.dtype)
        blocking[..., 3] = -float("inf")  # nobody may attend to key 3
        assert not getattr(blocking, "_kvpress_all_zero", False)

        seen = {}
        original_sdpa = F.scaled_dot_product_attention

        def spy(query, key, value, attn_mask=None, **kwargs):
            seen["mask"] = attn_mask
            return original_sdpa(query, key, value, attn_mask=attn_mask, **kwargs)

        F.scaled_dot_product_attention = spy
        try:
            model.model(
                input_ids=ids,
                past_key_values=ReadOnlyCache(cache),
                position_ids=torch.arange(8, 16).unsqueeze(0),
                attention_mask=blocking,
                use_cache=True,
            )
        finally:
            F.scaled_dot_product_attention = original_sdpa

    mask = seen["mask"]
    assert mask is not None, "the blocking mask never reached SDPA"
    assert torch.isneginf(mask[..., 3]).all(), (
        "key 3 should be forbidden, but the mask was dropped along with the rectangle"
    )


def test_the_rectangle_is_not_the_causal_fast_path():
    """The rectangle is all-visible; SDPA's ``is_causal`` shortcut is not.

    With ``q_len == k_len`` (which a read-only cache guarantees) passing ``mask=None`` takes the
    causal fast path, which is the ordinary LM objective rather than the cross-context rectangle.
    """
    q_len = k_len = 8
    mask = rectangle_mask(q_len, k_len, torch.device("cpu"), torch.float32)
    assert mask.shape == (1, 1, q_len, k_len)
    assert (mask == 0).all(), "the rectangle adds nothing; it exists so masking is not skipped"

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, q_len, HEAD_DIM, dtype=torch.float64) for _ in range(3))
    rect = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert not torch.allclose(rect, causal), "rectangle and causal must differ, or the mask is inert"


def test_step_uses_a_full_rectangle_not_a_triangle():
    """End to end: every replay query must see every ``C`` key.

    Checked by capturing the mask the attention actually receives -- the property that matters is
    what reaches SDPA, not what the caller intended.
    """
    model, trainer = _model(), _trainer()
    ids = torch.randint(0, 128, (1, 12))
    seen = {}

    with trainer.hooks(model):
        original = trainer._attention

        def spy(module, query, key, value, attention_mask, **kw):
            seen[int(module.layer_idx)] = (query.shape[2], key.shape[2], attention_mask)
            return original(module, query, key, value, attention_mask, **kw)

        trainer._attention = spy
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS.register("kvpress_cross_replay_gated", spy)
        cross_replay_training_step(model, trainer, input_ids=ids)

    assert seen, "the gated attention never ran"
    for layer_idx, (q_len, k_len, mask) in seen.items():
        assert k_len == 12, f"layer {layer_idx} saw {k_len} keys, expected |C| = 12"
        # The gate is a (B, H, 1, N) bias; broadcast over queries means no -inf anywhere, i.e. no
        # query is denied any key. A causal mask would put -inf above the diagonal.
        assert torch.isfinite(mask).all(), f"layer {layer_idx} received a masked-out pair"


# ----------------------------------------------------------------------
# Silent failure 3: the cache must not grow
# ----------------------------------------------------------------------
def test_use_cache_false_does_not_prevent_growth():
    """The motivation for ``ReadOnlyCache``: ``use_cache=False`` still appends.

    If this ever changes upstream the wrapper becomes unnecessary -- but until then, relying on the
    flag would double the key axis and let ``C'`` attend to itself.
    """
    model = _model()
    ids = torch.randint(0, 128, (1, 8))
    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=ids, past_key_values=cache, use_cache=True)
    assert cache.layers[0].keys.shape[2] == 8

    with torch.no_grad():
        model.model(
            input_ids=ids,
            past_key_values=cache,
            position_ids=torch.arange(8, 16).unsqueeze(0),
            use_cache=False,
        )
    assert cache.layers[0].keys.shape[2] == 16, "use_cache=False unexpectedly stopped the append"


def test_read_only_cache_holds_the_key_axis():
    model = _model()
    ids = torch.randint(0, 128, (1, 8))
    cache = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=ids, past_key_values=cache, use_cache=True)

    read_only = ReadOnlyCache(cache)
    with torch.no_grad():
        model.model(
            input_ids=ids,
            past_key_values=read_only,
            position_ids=torch.arange(8, 16).unsqueeze(0),
            attention_mask=rectangle_mask(8, 8, ids.device, model.dtype),
            use_cache=True,
        )
    assert cache.layers[0].keys.shape[2] == 8, "the read-only cache grew"


def test_attention_rejects_a_grown_cache():
    """A guard, since a grown cache silently turns this into causal LM on a doubled axis."""
    model, trainer = _model(), _trainer()
    ids = torch.randint(0, 128, (1, 8))
    with trainer.hooks(model):
        cache, hidden = trainer.prefill(model, ids)
        trainer.score_context(model, hidden)
        trainer._context_len = 8
        with pytest.raises(RuntimeError, match="must see KV\\(C\\) only"):
            model.model(
                input_ids=ids,
                past_key_values=cache,  # the real cache: it WILL grow to 16
                position_ids=torch.arange(8, 16).unsqueeze(0),
                attention_mask=rectangle_mask(8, 16, ids.device, model.dtype),
                use_cache=True,
            )


# ----------------------------------------------------------------------
# Exactness: chunking, and the broadcast gate
# ----------------------------------------------------------------------
@pytest.mark.parametrize("chunk", [4, 8, 16])
def test_query_chunking_reproduces_the_unchunked_gradient(chunk):
    """Chunking replay queries is exact, which is what makes it a free memory win.

    Legitimate only because ``C'`` is masked from itself, so replay queries are independent. If the
    objective ever lets ``C'`` see ``C'``, this test fails and correctly so.

    **Tolerances are fp32, not fp64, even though the model here is fp64.** The gate path is
    deliberately fp32 at two points: ``ScalarIndexer.score_keys`` returns ``.float()`` (bf16 resolves
    too few distinct score values for a meaningful top-k) and ``gate_scale`` is upcast to fp32 (in
    bf16 a warmup learning rate cannot move it at all). So ``dL/ds`` accumulates in fp32 and the
    chunked/unchunked difference floors at fp32 epsilon, 1.2e-07 -- measured relative differences are
    6e-08 to 1.6e-07 across ``w_out``, ``in_norm.weight`` and ``gate_scale``, i.e. exactly that.
    A tighter ``rtol`` would be testing fp32 arithmetic, not the chunking.

    The ``atol`` floor matters separately: ``in_norm.bias`` receives a gradient of magnitude ~1e-11
    (numerically zero for this input), where a relative comparison is meaningless -- it reports
    ratios up to 9.0 while differing by 6e-11.
    """
    ids = torch.randint(0, 128, (1, 16))

    def run(query_chunk):
        model, trainer = _model(), _trainer(query_chunk=query_chunk)
        with trainer.hooks(model):
            loss = cross_replay_training_step(model, trainer, input_ids=ids)
        grads = {
            name: p.grad.clone()
            for name, p in model.named_parameters()
            if p.grad is not None and "indexer" in name
        }
        return loss.detach(), grads

    ref_loss, ref_grads = run(None)
    got_loss, got_grads = run(chunk)

    # The loss itself IS bit-exact: it is a sum of per-chunk sums divided by one total count.
    assert torch.equal(ref_loss, got_loss), (
        f"chunked loss {got_loss.item()!r} != unchunked {ref_loss.item()!r}"
    )
    assert set(ref_grads) == set(got_grads) and ref_grads
    for name, ref in ref_grads.items():
        assert torch.allclose(ref, got_grads[name], rtol=1e-5, atol=1e-9), (
            f"{name}: chunked gradient differs by {(ref - got_grads[name]).abs().max()} "
            f"against a magnitude of {ref.abs().max()}"
        )


@pytest.mark.parametrize("chunk", [4, 16])
def test_chunking_releases_each_chunk_graph(chunk):
    """The point of chunking: peak retained memory must actually fall.

    This is the test that was missing when the first implementation accumulated the chunks' losses
    and backwarded once at the end -- which holds every chunk's graph simultaneously and saves
    nothing (measured: retention *grew*, 1164 KiB to 1195 KiB at 8 chunks, from per-chunk overhead).
    Peak is sampled inside the loss, i.e. at the moment a chunk's graph is fully built.
    """
    ids = torch.randint(0, 128, (1, 32))
    original_ce = F.cross_entropy

    def run(query_chunk):
        model, trainer = _model(), _trainer(query_chunk=query_chunk)
        peak = [0]

        def sampling_ce(*args, **kwargs):
            peak[0] = max(peak[0], _retained_bytes())
            return original_ce(*args, **kwargs)

        with trainer.hooks(model):
            F.cross_entropy = sampling_ce
            try:
                cross_replay_training_step(model, trainer, input_ids=ids)
            finally:
                F.cross_entropy = original_ce
        return peak[0]

    unchunked = run(None)
    chunked = run(chunk)
    assert chunked < unchunked, (
        f"query_chunk={chunk} retained {chunked} bytes against {unchunked} unchunked -- the chunks' "
        "graphs are not being released, so chunking costs memory instead of saving it"
    )


def test_backward_false_is_rejected_when_it_would_hold_every_chunk():
    """Deferring backward with chunking on would silently undo the memory saving."""
    model, trainer = _model(), _trainer(query_chunk=4)
    ids = torch.randint(0, 128, (1, 16))
    with trainer.hooks(model):
        with pytest.raises(ValueError, match="chunks' graphs at once"):
            cross_replay_training_step(model, trainer, input_ids=ids, backward=False)


def test_backward_false_returns_a_graph_unchunked():
    """Unchunked, the caller may still own the backward."""
    model, trainer = _model(), _trainer()
    ids = torch.randint(0, 128, (1, 12))
    with trainer.hooks(model):
        loss = cross_replay_training_step(model, trainer, input_ids=ids, backward=False)
        assert loss.requires_grad
        loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for name, p in model.named_parameters()
        if "indexer" in name
    )


@pytest.mark.parametrize("logit_chunk", [1, 3, 8])
def test_logit_chunking_is_exact(logit_chunk):
    """Splitting ``lm_head`` over row blocks must not change the loss or the gradient.

    The logits are a first-order memory term ``query_chunk`` does not touch -- ``chunk * vocab``
    plus the fp32 copy the cross-entropy needs, 0.87 GiB per 1024 rows on Qwen3-8B -- so it gets its
    own knob, and its own exactness test.
    """
    ids = torch.randint(0, 128, (1, 16))

    def run(rows):
        model, trainer = _model(), _trainer()
        with trainer.hooks(model):
            loss = cross_replay_training_step(
                model, trainer, input_ids=ids, logit_chunk=rows
            )
        grads = {
            name: p.grad.clone()
            for name, p in model.named_parameters()
            if p.grad is not None and "indexer" in name
        }
        return loss, grads

    ref_loss, ref_grads = run(None)
    got_loss, got_grads = run(logit_chunk)
    assert torch.allclose(ref_loss, got_loss, rtol=1e-12, atol=1e-12)
    for name, ref in ref_grads.items():
        assert torch.allclose(ref, got_grads[name], rtol=1e-5, atol=1e-9), (
            f"{name}: differs by {(ref - got_grads[name]).abs().max()}"
        )


def test_logit_chunking_releases_each_block():
    """Peak retained memory must fall, not just stay correct."""
    ids = torch.randint(0, 128, (1, 24))

    def run(rows):
        model, trainer = _model(), _trainer()
        peak = [0]
        original_ce = F.cross_entropy

        def sampling_ce(*args, **kwargs):
            peak[0] = max(peak[0], _retained_bytes())
            return original_ce(*args, **kwargs)

        with trainer.hooks(model):
            F.cross_entropy = sampling_ce
            try:
                cross_replay_training_step(model, trainer, input_ids=ids, logit_chunk=rows)
            finally:
                F.cross_entropy = original_ce
        return peak[0]

    assert run(4) < run(None), "logit chunking retained no less than the unchunked run"


def test_broadcast_gate_equals_an_explicit_one():
    """The ``(B, H, 1, N)`` bias must equal a materialized ``(B, H, Sq, N)`` gate.

    This equality is why no kernel is needed: the whole content of query-independence is that every
    query row of the gate is the same.
    """
    torch.manual_seed(0)
    b, h, kv, q_len, k_len = 1, 4, 2, 6, 10
    group = h // kv
    q = torch.randn(b, h, q_len, HEAD_DIM, dtype=torch.float64)
    k = torch.randn(b, kv, k_len, HEAD_DIM, dtype=torch.float64)
    v = torch.randn(b, kv, k_len, HEAD_DIM, dtype=torch.float64)
    gate = torch.randn(b, kv, k_len, dtype=torch.float64, requires_grad=True)

    def broadcast():
        bias = gate.repeat_interleave(group, 1).unsqueeze(2)
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, scale=HEAD_DIM**-0.5, enable_gqa=True
        )

    def explicit():
        kx, vx = k.repeat_interleave(group, 1), v.repeat_interleave(group, 1)
        logits = HEAD_DIM**-0.5 * (q @ kx.transpose(-1, -2))
        logits = logits + gate.repeat_interleave(group, 1).unsqueeze(2)
        return torch.softmax(logits, -1) @ vx

    a, e = broadcast(), explicit()
    assert torch.allclose(a, e, atol=1e-12), f"forward differs by {(a - e).abs().max()}"

    gate.grad = None
    a.sum().backward()
    ga = gate.grad.clone()
    gate.grad = None
    explicit().sum().backward()
    assert torch.allclose(ga, gate.grad, atol=1e-12), (
        f"gradient differs by {(ga - gate.grad).abs().max()}"
    )


# ----------------------------------------------------------------------
# The gate itself
# ----------------------------------------------------------------------
def test_gate_pins_sinks_and_normalizes_the_rest():
    """Pinned keys take gate 0; gated keys sum to a total multiplier of 1."""
    trainer = _trainer(n_sink=2)
    torch.manual_seed(0)
    trainer._scores = {0: torch.randn(1, N_KV_HEADS, 12, dtype=torch.float64)}
    gate = trainer.gate(0, N_KV_HEADS)

    assert (gate[..., :2] == 0).all(), "sink keys must be pinned at log-space 0"
    total = gate[..., 2:].exp().sum(-1)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-12), (
        "gated keys must share a fixed total multiplier of 1"
    )


def test_gate_is_invariant_to_a_global_score_shift():
    """A shift ``s -> s + c`` must not change the gate: the normalizer absorbs it."""
    trainer = _trainer(n_sink=2)
    torch.manual_seed(0)
    scores = torch.randn(1, N_KV_HEADS, 12, dtype=torch.float64)
    trainer._scores = {0: scores}
    a = trainer.gate(0, N_KV_HEADS)
    trainer._scores = {0: scores + 3.7}
    b = trainer.gate(0, N_KV_HEADS)
    assert torch.allclose(a, b, atol=1e-12)


def test_budget_constant_would_not_change_the_ranking():
    """Why no ``+log K`` term exists: it is invisible to the ranking inference uses.

    Documents the retraction in ``cross_replay_e2e.md`` §2 as an executable fact, so the constant is
    not reintroduced on the belief that it constrains the cache.
    """
    torch.manual_seed(0)
    scores = torch.randn(1, N_KV_HEADS, 64, dtype=torch.float64)
    gated = torch.arange(64) >= 2
    lse = torch.logsumexp(scores.masked_fill(~gated, -float("inf")), -1, keepdim=True)
    base = torch.where(gated, scores - lse, torch.zeros_like(scores))

    for k in (4.0, 256.0, 2048.0):
        shifted = base + torch.where(gated, math.log(k), 0.0)
        # Ranking among gated keys is untouched...
        assert torch.equal(
            base[..., 2:].argsort(-1), shifted[..., 2:].argsort(-1)
        ), f"log K = {k} changed the gated ranking"
        # ...and so is the conditional distribution over them.
        pa = torch.softmax(base[..., 2:], -1)
        pb = torch.softmax(shifted[..., 2:], -1)
        assert torch.allclose(pa, pb, atol=1e-12)


# ----------------------------------------------------------------------
# Wiring
# ----------------------------------------------------------------------
def test_pass_one_is_ungated_and_builds_no_graph():
    """Pass 1 must produce dense ``KV(C)`` and no autograd graph.

    Gating it would train streaming prefill instead of eviction, and would close the loop
    ``gate -> h -> s -> gate`` because ``h`` depends on earlier layers' attention.
    """
    model, trainer = _model(), _trainer()
    ids = torch.randint(0, 128, (1, 10))

    dense = DynamicCache()
    with torch.no_grad():
        model.model(input_ids=ids, past_key_values=dense, use_cache=True)

    with trainer.hooks(model):
        cache, hidden = trainer.prefill(model, ids)

    assert not any(h.requires_grad for h in hidden.values()), "pass 1 retained a graph"
    for layer_idx in range(N_LAYERS):
        assert torch.allclose(
            cache.layers[layer_idx].keys, dense.layers[layer_idx].keys, atol=1e-12
        ), f"layer {layer_idx}: pass 1 was not the plain dense prefill"


def test_gradient_reaches_indexer_from_detached_hidden_states():
    """``dL/dw = sum_i (dL/ds_i) h_i`` needs ``h``'s values, not its graph."""
    model, trainer = _model(), _trainer()
    ids = torch.randint(0, 128, (1, 12))
    with trainer.hooks(model):
        loss = cross_replay_training_step(model, trainer, input_ids=ids)
    assert not loss.requires_grad, "backward=True should return a detached loss"

    trained = {
        name: p for name, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum()
    }
    assert trained, "no parameter received a gradient"
    assert all("indexer" in name for name in trained), (
        f"backbone parameters were trained: {[n for n in trained if 'indexer' not in n][:3]}"
    )
    assert any("w_out" in name for name in trained), "the score projection got no gradient"
    assert any("gate_scale" in name for name in trained), "gate_scale got no gradient"


def test_replay_ids_allow_a_cross_document_control():
    """Replaying unrelated text is the natural null and must be expressible."""
    model, trainer = _model(), _trainer()
    ids = torch.randint(0, 128, (1, 12))
    other = torch.randint(0, 128, (1, 12))
    with trainer.hooks(model):
        matched = cross_replay_training_step(model, trainer, input_ids=ids)
    with trainer.hooks(model):
        crossed = cross_replay_training_step(model, trainer, input_ids=ids, replay_ids=other)
    assert torch.isfinite(matched) and torch.isfinite(crossed)
    assert not torch.allclose(matched, crossed)


def test_hooks_restore_the_attention_implementation():
    model, trainer = _model(), _trainer()
    before = model.config._attn_implementation
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    with trainer.hooks(model):
        assert model.config._attn_implementation == "kvpress_cross_replay_gated"
    assert model.config._attn_implementation == before
    assert "kvpress_cross_replay_gated" not in type(ALL_ATTENTION_FUNCTIONS)._global_mapping


def test_long_context_warns_past_the_trained_position_range(caplog):
    """``d_max = 2N - 1`` past ``max_position_embeddings`` must be surfaced.

    Silent otherwise: RoPE evaluates fine at untrained positions and the loss curve looks normal.
    """
    model, trainer = _model(max_pos=16), _trainer()
    with caplog.at_level("WARNING"):
        trainer.check_positions(model, 12)  # d_max = 23 > 16
    assert "max_position_embeddings" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        trainer.check_positions(model, 8)  # d_max = 15 <= 16
    assert "max_position_embeddings" not in caplog.text


def test_pairwise_indexer_is_rejected():
    """A pairwise scorer would reintroduce the ``(Sq, Sk)`` gate this objective avoids."""
    model = _model()
    trainer = _trainer(press=_press(scorer="pairwise"))
    ids = torch.randint(0, 128, (1, 8))
    with trainer.hooks(model):
        _, hidden = trainer.prefill(model, ids)
        with pytest.raises(TypeError, match="query-independent"):
            trainer.score_context(model, hidden)


# ----------------------------------------------------------------------
# The memory fix: the gate must not reach CUDA SDPA as a mask
#
# These test the thing the optimization optimizes, per the lesson in cross_replay_e2e.md §9: three
# earlier bugs here were "correct but achieved nothing" and every exactness test passed through them.
# The exactness of the flex path is covered by test_flex_and_sdpa_paths_agree; what follows checks
# that the expensive route is actually *avoided*, which is the property that regressed silently.
# ----------------------------------------------------------------------
def test_a_masked_gqa_sdpa_call_has_no_fused_backend():
    """The measurement behind the whole fix, as an executable assertion.

    Three conditions have to hold at once for SDPA to lose every fused kernel, and this objective
    supplies all three: a non-``None`` ``attn_mask`` (excludes flash outright -- "Flash Attention does
    not support non-null attn_mask"), a GQA head mismatch (excludes mem-efficient *given* the mask --
    "both fused kernels require query, key and value to have the same num_heads"), and a mask that
    **requires grad** (excludes cuDNN). SDPA then runs MATH and retains the full ``(B, H, Sq, Sk)``
    score matrix: 1288 MiB per layer at Qwen3-8B's 8K geometry, 46.7 GiB over 36 layers, which is the
    gap between the measured 73.2 GiB peak and the 23.9 GiB estimate.

    The third condition is not incidental and was nearly missed. With ``requires_grad=False`` cuDNN
    *is* eligible and forcing it retains only 8.1 MiB -- so a probe built on a detached gate reports
    a fused backend and hides the bug. But ``dL/ds`` arriving through the mask is the entire
    objective, so the production gate always requires grad. The mask is built here exactly as
    :meth:`CrossReplayTrainer._attention` builds it, for that reason.

    Asserted on backend *eligibility* rather than on a memory number so it runs anywhere CUDA exists,
    at a shape small enough to be free. If a future torch teaches a fused backend to accept a
    grad-requiring broadcast mask under GQA, this test fails and the fallback stops being a
    catastrophe -- which is worth knowing, so it is a failure rather than a skip.
    """
    if not torch.cuda.is_available():
        pytest.skip("backend selection is a CUDA property; the CPU probe is not representative")
    from torch.backends.cuda import (
        SDPAParams,
        can_use_cudnn_attention,
        can_use_efficient_attention,
        can_use_flash_attention,
    )

    b, h_q, h_kv, s_q, s_k, dim = 1, 8, 2, 128, 256, 64
    q = torch.randn(b, h_q, s_q, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, h_kv, s_k, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(b, h_kv, s_k, dim, device="cuda", dtype=torch.bfloat16)
    # requires_grad, as in production: the gradient into the gate is the objective.
    gate = torch.randn(b, h_kv, s_k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bias = gate.repeat_interleave(h_q // h_kv, dim=1).unsqueeze(2)  # (B, H, 1, N), as the gate rides
    assert bias.requires_grad, "the premise of this test is a differentiable gate"

    def eligible(mask, enable_gqa):
        params = SDPAParams(q, k, v, mask, 0.0, False, enable_gqa)
        return {
            "flash": can_use_flash_attention(params, False),
            "mem_efficient": can_use_efficient_attention(params, False),
            "cudnn": can_use_cudnn_attention(params, False),
        }

    masked_gqa = eligible(bias, True)
    assert not any(masked_gqa.values()), (
        f"a masked GQA SDPA call now has a fused backend available ({masked_gqa}). The 46.7 GiB "
        "MATH fallback that flex_attention was introduced to avoid may no longer apply -- re-measure "
        "before simplifying _flex_attention away."
    )
    # Each condition alone is survivable; the *combination* is what costs. Dropping one at a time:
    assert eligible(None, True)["flash"], (
        "drop the mask and flash returns -- so the mask is one of the three conditions"
    )
    assert eligible(bias.detach(), True)["cudnn"], (
        "detach the gate and cuDNN returns -- so requires_grad is one of the three. If this fails, "
        "that is no longer what excludes cuDNN and this test's reasoning needs re-deriving"
    )
    replicated = SDPAParams(
        q,
        k.repeat_interleave(h_q // h_kv, dim=1),
        v.repeat_interleave(h_q // h_kv, dim=1),
        bias,
        0.0,
        False,
        False,
    )
    assert can_use_efficient_attention(replicated, False), (
        "replicate K/V and mem-efficient returns, grad-requiring mask and all -- so the GQA mismatch "
        "is the third condition. This is the row `_flex_attention` rejects as a stopgap: it costs "
        "group_size x the cache and still retains 3.7x what flex does"
    )


def test_the_cuda_path_never_calls_sdpa_with_a_gate_mask():
    """On CUDA the gate must go through ``score_mod``; reaching SDPA at all is the bug.

    The regression guard for the actual fix. A change that reverted ``_attention`` to the mask route
    would keep every exactness test green -- the two agree numerically -- and cost 46.7 GiB at 8K.
    Spying on SDPA is how that becomes visible without needing a 16 GiB model.
    """
    if not torch.cuda.is_available():
        pytest.skip("the flex path is CUDA-only; see flex_fallback_reason")
    # bf16 on CUDA: fp64 would legitimately fall back, which is a different code path.
    model = _model().to("cuda").to(torch.bfloat16)
    trainer = _trainer(query_chunk=8)
    ids = torch.randint(0, 128, (1, 16), device="cuda")

    calls = []
    original_sdpa = F.scaled_dot_product_attention

    def spy(query, key, value, attn_mask=None, **kwargs):
        calls.append(None if attn_mask is None else tuple(attn_mask.shape))
        return original_sdpa(query, key, value, attn_mask=attn_mask, **kwargs)

    with trainer.hooks(model):
        F.scaled_dot_product_attention = spy
        try:
            cross_replay_training_step(model, trainer, input_ids=ids)
        finally:
            F.scaled_dot_product_attention = original_sdpa

    assert trainer.layers_gated > 0, "no layer ran the gated attention, so nothing was measured"
    assert not [c for c in calls if c is not None], (
        f"the gated replay called SDPA with a mask {[c for c in calls if c is not None]} on CUDA. "
        "That is the MATH fallback: 1328 MiB retained per layer instead of 48 MiB."
    )


def test_flex_and_sdpa_paths_agree_on_forward_and_gate_gradient():
    """The two routes to the logits must be the same function, or the fix changed the objective.

    Tolerance is fp32, not fp64: ``ScalarIndexer.score_keys`` returns ``.float()`` and ``gate_scale``
    is upcast to fp32, so ``dL/ds`` accumulates in fp32 and this difference floors at fp32 epsilon
    (measured 4.8e-07 against the SDPA reference at fp64 inputs).
    """
    if not torch.cuda.is_available():
        pytest.skip("flex_attention is CUDA-only")
    from torch.nn.attention.flex_attention import flex_attention

    torch.manual_seed(0)
    b, h_q, h_kv, s_q, s_k, dim = 1, 8, 2, 128, 256, 64
    group = h_q // h_kv
    q = torch.randn(b, h_q, s_q, dim, device="cuda", dtype=torch.float64, requires_grad=True)
    k = torch.randn(b, h_kv, s_k, dim, device="cuda", dtype=torch.float64)
    v = torch.randn(b, h_kv, s_k, dim, device="cuda", dtype=torch.float64)
    gate = torch.randn(b, h_kv, s_k, device="cuda", dtype=torch.float32, requires_grad=True)
    cotangent = torch.randn(b, h_q, s_q, dim, device="cuda", dtype=torch.float64)

    bias = gate.to(q.dtype).repeat_interleave(group, dim=1).unsqueeze(2)
    reference = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=dim**-0.5, enable_gqa=True)
    ref_dq, ref_dgate = torch.autograd.grad(reference, (q, gate), cotangent)

    def score_mod(score, batch, head, q_idx, kv_idx):
        return score + gate[batch, head // group, kv_idx].to(score.dtype)

    out = flex_attention(q, k, v, score_mod=score_mod, scale=dim**-0.5, enable_gqa=True)
    dq, dgate = torch.autograd.grad(out, (q, gate), cotangent)

    assert (out - reference).abs().max() < 1e-12, "flex and SDPA disagree in the forward"
    assert (dq - ref_dq).abs().max() < 1e-12, "flex and SDPA disagree on dL/dq"
    assert (dgate - ref_dgate).abs().max() < 1e-5, "flex and SDPA disagree on dL/dgate"
    assert dgate.abs().max() > 0, "no gradient reached the gate at all, which is the whole objective"


@pytest.mark.parametrize("q_len", [1, 17, 64, 100, 127, 128, 129])
def test_flex_runs_for_every_ragged_chunk_length(q_len):
    """A ragged final replay chunk must not crash, and padding is what makes that true.

    Inductor's autotuner has **no valid triton config** for ``64 <= Sq < 128`` at a long key axis
    (every candidate exceeds the H20's 232448-byte shared-memory limit) and raises ``No valid triton
    configs`` rather than falling back. Any ``|C| % query_chunk`` in that band would hit it, so this
    is a real configuration rather than a hypothetical -- ``--context-len 8292 --query-chunk 1024``
    leaves 100.
    """
    if not torch.cuda.is_available():
        pytest.skip("flex_attention is CUDA-only")
    model = _model().to("cuda").to(torch.bfloat16)
    trainer = _trainer()
    ids = torch.randint(0, 128, (1, 256), device="cuda")
    with trainer.hooks(model):
        cache, hidden = trainer.prefill(model, ids)
        trainer.score_context(model, hidden)
        trainer._context_len = 256
        out = model.model(
            input_ids=ids[:, :q_len],
            past_key_values=ReadOnlyCache(cache),
            position_ids=torch.arange(256, 256 + q_len, device="cuda").unsqueeze(0),
            attention_mask=rectangle_mask(q_len, 256, ids.device, model.dtype),
            use_cache=True,
        )
    assert out.last_hidden_state.shape[1] == q_len, (
        "the padded rows must be sliced back off, or the replay's row count no longer matches its "
        "labels and the loss silently pairs the wrong targets"
    )


def test_query_padding_does_not_change_the_result():
    """Padding to ``_FLEX_Q_ALIGN`` is exact: padded rows contribute nothing to ``dL/ds``.

    Checked at a length that needs padding (100 -> 128) against the same length reached without it,
    because "pad and slice" is exactly the kind of change that could quietly shift rows by one.
    """
    if not torch.cuda.is_available():
        pytest.skip("flex_attention is CUDA-only")
    from kvpress.presses.gqa_indexer.cross_replay import _FLEX_Q_ALIGN

    assert _FLEX_Q_ALIGN % 128 == 0, (
        "the autotuner gap measured on an H20 is 64 <= Sq < 128, so the alignment must be a multiple "
        "of 128 for padding to escape it"
    )
    torch.manual_seed(0)
    model = _model().to("cuda").to(torch.float32)
    ids = torch.randint(0, 128, (1, 256), device="cuda")

    grads = {}
    for q_len in (100, 128):
        trainer = _trainer()
        with trainer.hooks(model):
            cache, hidden = trainer.prefill(model, ids)
            trainer.score_context(model, hidden)
            leaves = {i: s.detach().requires_grad_(True) for i, s in trainer._scores.items()}
            trainer._scores = leaves
            trainer._context_len = 256
            out = model.model(
                input_ids=ids[:, :100],  # the SAME 100 rows in both runs
                past_key_values=ReadOnlyCache(cache),
                position_ids=torch.arange(256, 356, device="cuda").unsqueeze(0),
                attention_mask=rectangle_mask(100, 256, ids.device, model.dtype),
                use_cache=True,
            )
            out.last_hidden_state.sum().backward()
            grads[q_len] = {i: leaf.grad.clone() for i, leaf in leaves.items()}

    for idx in grads[100]:
        assert torch.allclose(grads[100][idx], grads[128][idx], atol=1e-5), (
            f"layer {idx}: padding changed dL/ds, so the padded rows are contributing gradient"
        )


def test_flex_fallback_is_diagnosed_rather_than_silent():
    """Every reason flex cannot run must be reported, because the fallback costs 46.7 GiB on CUDA.

    A pure-function test so the branch that decides between a 48 MiB kernel and a 1328 MiB one is
    checked directly, on any machine. The failure this guards is the quiet one: a fallback that
    happens without a word looks exactly like the fix working.
    """
    from kvpress.presses.gqa_indexer.cross_replay import flex_fallback_reason

    cpu = torch.zeros(1, 4, 8, 64)
    assert "CUDA" in (flex_fallback_reason(cpu, None, 0.0) or ""), "a CPU tensor must be diagnosed"

    if not torch.cuda.is_available():
        pytest.skip("the remaining branches need a device to be reachable at all")
    good = torch.zeros(1, 4, 8, 64, device="cuda", dtype=torch.bfloat16)
    assert flex_fallback_reason(good, None, 0.0) is None, (
        "the production configuration must take the flex path, or the fix is inert"
    )
    assert flex_fallback_reason(good.double(), None, 0.0), "fp64 is NYI in the inductor lowering"
    assert flex_fallback_reason(good[..., :8], None, 0.0), "head_dim < 16 is NYI"
    assert flex_fallback_reason(good, torch.zeros(1, 1, 8, 8, device="cuda"), 0.0), (
        "a real mask cannot be dropped; it would have to enter score_mod"
    )
    assert flex_fallback_reason(good, None, 0.1), "dropout has no score_mod equivalent"


def test_recompile_pressure_is_warned_before_dynamo_gives_up(caplog):
    """Crossing dynamo's recompile limit reverts flex to eager, which is *worse* than the bug.

    Eager ``flex_attention`` materializes the score matrix: measured 18730 MiB for one layer at
    ``Sq=1024, N=8192`` against 40 MiB compiled, and against 1328 MiB for the MATH fallback this
    replaced. So a shape-churning caller does not merely lose the optimization, it inverts it -- and
    nothing in the loss would say so. Hence a warning at the limit rather than at the cliff.
    """
    trainer = _trainer()
    with caplog.at_level("WARNING"):
        for i in range(7):
            trainer._note_flex_shape(torch.zeros(1, 1, 128 + i, 1), torch.zeros(1, 1, 1, 256))
    assert "recompile" not in caplog.text, "warned too early; 7 distinct shapes still compile"

    with caplog.at_level("WARNING"):
        trainer._note_flex_shape(torch.zeros(1, 1, 999, 1), torch.zeros(1, 1, 1, 256))
    assert "EAGER" in caplog.text and "recompile limit" in caplog.text, (
        "the 8th shape is the last that compiles; the 9th silently reverts to the materializing "
        "eager path, so the warning has to land before it"
    )


# ----------------------------------------------------------------------
# Gradient accumulation: loss_scale must reach the GRADIENTS, not just the number
#
# The trap these guard: cross_replay_training_step differentiates internally and returns a DETACHED
# scalar, so the idiomatic `(loss / accum_steps).backward()` -- or any scaling of the return value --
# divides a tensor with no graph attached. The gradients stay accum_steps too large while the logged
# loss looks perfectly right. That is the exact shape of the four bugs in cross_replay_e2e.md §9, so
# the property is tested directly rather than inferred from a training curve.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("scale", [0.5, 0.25, 2.0])
def test_loss_scale_scales_the_gradient(scale):
    """``loss_scale=s`` must multiply every gradient by ``s``, exactly.

    Linearity is the whole contract: gradient accumulation is only correct if scaling the loss scales
    the gradient, and this objective's internal backward is what makes that non-obvious.
    """
    ids = torch.randint(0, 128, (1, 16))

    def run(loss_scale):
        model, trainer = _model(), _trainer(query_chunk=8)
        with trainer.hooks(model):
            loss = cross_replay_training_step(
                model, trainer, input_ids=ids, loss_scale=loss_scale
            )
        grads = {
            name: p.grad.clone()
            for name, p in model.named_parameters()
            if p.grad is not None and "indexer" in name
        }
        return float(loss), grads

    base_loss, base_grads = run(1.0)
    scaled_loss, scaled_grads = run(scale)

    assert base_grads, "no indexer gradients were produced, so nothing was measured"
    assert scaled_loss == pytest.approx(scale * base_loss, rel=1e-9), (
        "the returned loss must be scaled too, so a caller summing across an accumulation group "
        "gets the mean it expects"
    )
    for name, base in base_grads.items():
        assert torch.allclose(scaled_grads[name], scale * base, rtol=1e-6, atol=1e-12), (
            f"{name}: loss_scale={scale} did not scale the gradient. If this passes for the loss but "
            f"fails here, the scale is being applied AFTER the internal backward -- the silent bug "
            f"this test exists for. max diff {(scaled_grads[name] - scale * base).abs().max()}"
        )


def test_accumulating_micro_batches_equals_one_averaged_batch():
    """N micro-batches at ``loss_scale=1/N`` must equal the mean gradient over those N sequences.

    The property the training driver actually relies on: it never calls ``.backward()`` itself, so if
    this does not hold, every multi-GPU or accumulated run trains at the wrong effective learning
    rate while reporting a normal loss.

    Tolerance is the fp32 floor, not fp64, for the reason given at length in
    ``test_query_chunking_reproduces_the_unchunked_gradient``: the gate path is deliberately fp32
    (``score_keys`` returns ``.float()``, ``gate_scale`` is upcast), so ``dL/ds`` accumulates in fp32
    and any regrouping of the sum floors there. Measured difference here is 1.0e-10 against gradient
    magnitudes of ~1e-4, i.e. a relative 1e-6 -- exactly that floor. A tighter bound would be testing
    fp32 addition order, not accumulation.
    """
    seqs = [torch.randint(0, 128, (1, 16)) for _ in range(3)]

    # Accumulated: three steps at 1/3, gradients piling up in the same parameters.
    model, trainer = _model(), _trainer(query_chunk=8)
    with trainer.hooks(model):
        for ids in seqs:
            cross_replay_training_step(model, trainer, input_ids=ids, loss_scale=1.0 / len(seqs))
    accumulated = {
        name: p.grad.clone()
        for name, p in model.named_parameters()
        if p.grad is not None and "indexer" in name
    }

    # Reference: each sequence on its own at scale 1, averaged by hand.
    summed: dict[str, torch.Tensor] = {}
    for ids in seqs:
        model_i, trainer_i = _model(), _trainer(query_chunk=8)
        with trainer_i.hooks(model_i):
            cross_replay_training_step(model_i, trainer_i, input_ids=ids)
        for name, p in model_i.named_parameters():
            if p.grad is not None and "indexer" in name:
                summed[name] = summed.get(name, torch.zeros_like(p.grad)) + p.grad
    reference = {name: g / len(seqs) for name, g in summed.items()}

    assert set(accumulated) == set(reference) and accumulated
    for name, ref in reference.items():
        assert torch.allclose(accumulated[name], ref, rtol=1e-5, atol=1e-9), (
            f"{name}: accumulating 3 micro-batches at 1/3 does not equal their mean gradient "
            f"(max diff {(accumulated[name] - ref).abs().max()} against a magnitude of "
            f"{ref.abs().max()})"
        )
    # A scale error would be a FACTOR of 3, not a rounding difference -- assert the gap is nowhere
    # near that, so the loosened tolerance above cannot hide the bug this test exists for.
    worst = max(
        float((accumulated[name] - ref).abs().max() / ref.abs().max().clamp_min(1e-30))
        for name, ref in reference.items()
    )
    assert worst < 1e-4, f"relative disagreement {worst:.2e} is too large to be fp32 rounding"


def test_loss_scale_zero_is_rejected():
    """A zero scale contributes no gradient while still reporting a finite loss."""
    model, trainer = _model(), _trainer()
    ids = torch.randint(0, 128, (1, 8))
    with trainer.hooks(model):
        with pytest.raises(ValueError, match="loss_scale must be positive"):
            cross_replay_training_step(model, trainer, input_ids=ids, loss_scale=0.0)


# ----------------------------------------------------------------------
# The two diagnostics that separate "trained" from "trained-looking"
# ----------------------------------------------------------------------
def test_gate_participation_reads_flat_as_one_and_peaked_as_near_zero():
    """The number that decides "did the router learn" must point the right way.

    Not a tautology worth skipping: an inverted or mis-normalised participation would report a healthy
    falling curve for a gate that is going *flat*, which is precisely the failure it is watched for.
    Both ends are pinned, so a sign error cannot pass.
    """
    from kvpress.presses.gqa_indexer.cross_replay import gate_participation

    n_keys, n_sink = 64, 4
    n_gated = n_keys - n_sink

    # A flat gate: log(1/n_gated) on every gated key, so exp() sums to 1 and spreads evenly.
    flat = torch.full((1, 2, n_keys), -math.log(n_gated), dtype=torch.float64)
    assert gate_participation(flat, n_sink) == pytest.approx(1.0, rel=1e-9), (
        "a flat gate must read 1.0 -- that is the no-op the pin mechanism exists to close, and the "
        "value the training log is watched for"
    )

    # A one-hot gate: all mass on one key.
    peaked = torch.full((1, 2, n_keys), -60.0, dtype=torch.float64)
    peaked[..., 10] = 0.0
    assert gate_participation(peaked, n_sink) == pytest.approx(1.0 / n_gated, rel=1e-6), (
        "a one-hot gate must read 1/n_gated, the smallest reachable value"
    )

    # Half the keys sharing everything sits in between, at 0.5.
    half = torch.full((1, 2, n_keys), -60.0, dtype=torch.float64)
    half[..., n_sink : n_sink + n_gated // 2] = -math.log(n_gated // 2)
    assert gate_participation(half, n_sink) == pytest.approx(0.5, rel=1e-6)

    # The sinks are excluded, so whatever they hold cannot move the number.
    with_sinks = flat.clone()
    with_sinks[..., :n_sink] = 12.0
    assert gate_participation(with_sinks, n_sink) == pytest.approx(
        gate_participation(flat, n_sink), rel=1e-9
    ), "pinned keys are not gated and must not enter the participation ratio"


def test_shuffled_scores_permutes_then_restores_even_on_error():
    """The shuffle control must not leak its patch, or every later step trains on a permutation.

    A leaked ``score_context`` would be a total and silent corruption of the run: training would
    continue against scores permuted away from the hidden states that produced them, with a loss curve
    that still looks like a loss curve. So the restore is tested on the exception path, not just the
    happy one.
    """
    from kvpress.presses.gqa_indexer.cross_replay import shuffled_scores

    model, trainer = _model(), _trainer()
    ids = torch.randint(0, 128, (1, 8))
    with trainer.hooks(model):
        _, hidden = trainer.prefill(model, ids)
        trainer.score_context(model, hidden)
        learned = {idx: s.clone() for idx, s in trainer._scores.items()}
        perm = torch.randperm(8)

        with shuffled_scores(trainer, perm):
            # The step calls score_context internally; here we call it directly to the same effect.
            trainer.score_context(model, hidden)
            for idx, before in learned.items():
                assert torch.equal(trainer._scores[idx], before[..., perm]), (
                    f"layer {idx}: the gate was not fed the permuted scores, so the control would "
                    "compare a run against itself and always report a delta of ~0"
                )
        # Asserted on __dict__, not on method identity: `trainer.score_context` builds a fresh bound
        # method on every access, so `is` would fail even on a correct restore. What must be gone is
        # the instance attribute shadowing the class's method.
        assert "score_context" not in trainer.__dict__, "the score_context override was not removed"
        # And it must be the real thing again -- rescoring must overwrite the permutation.
        trainer.score_context(model, hidden)
        for idx, before in learned.items():
            assert torch.equal(trainer._scores[idx], before), (
                f"layer {idx}: after restore, score_context did not recompute the true scores"
            )

        # And on the exception path.
        try:
            with shuffled_scores(trainer, perm):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert "score_context" not in trainer.__dict__, (
            "score_context leaked after an exception: every later step would train against permuted "
            "scores, silently"
        )


# ----------------------------------------------------------------------
# The training driver's argument validation
#
# Every rejection below corresponds to a configuration that would RUN and produce a healthy-looking
# loss curve while not doing what the flag says. The precedent is concrete: Liger's `skip_logits` was
# accepted and threaded through this very objective for a full revision while being a no-op
# (cross_replay_e2e.md §6.1), and gradient checkpointing was silently gated on `module.training` in
# the sibling script. Tested here rather than trusted, and CPU-cheap because `validate` is pure.
# ----------------------------------------------------------------------
def _driver():
    """The driver module, skipped if importing it is not possible in this environment."""
    return pytest.importorskip("scripts.train_gqa_indexer_cross_replay")


@pytest.mark.parametrize(
    "extra, expected",
    [
        # Would fuse lm_head into *ForCausalLM.forward, which this objective never calls.
        (["--liger"], "liger"),
        # Slices the sequence per rank, which this objective's fixed |C| key axis forbids.
        (["--ffn-sp-size", "2"], "ffn-sp-size"),
        # Pins nothing visible under [C ; C'], reopening the flat-gate no-op.
        (["--pin-mode", "none"], "pin-mode"),
        (["--pin-mode", "self"], "pin-mode"),
        # An inert normalizer, same no-op by another route.
        (["--n-sink", "0"], "n-sink"),
        # cross_replay_training_step requires (1, N).
        (["--batch-size", "4"], "batch-size"),
        (["--init-from", "a.pt", "--resume-from", "b.pt"], "mutually exclusive"),
    ],
)
def test_driver_rejects_configurations_that_would_silently_lie(extra, expected):
    """Each of these must exit with a message naming the flag, not run with it ignored.

    The message is asserted, not just the exit: these all run *fine* if accepted, so the error text is
    the only thing that tells the next person why their flag was refused. A rejection that named the
    wrong flag would be worse than none.
    """
    import contextlib
    import io

    driver = _driver()
    parser = driver.build_parser()
    args = parser.parse_args(["--data-root", "/nonexistent", *extra])
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), pytest.raises(SystemExit):
        driver.validate(args, parser)
    assert expected in stderr.getvalue(), (
        f"rejected, but the message does not mention {expected!r}: {stderr.getvalue().strip()}"
    )


def test_driver_accepts_the_intended_configuration():
    """The complement: the recommended run must NOT be rejected.

    Without this, every check above could be satisfied by a `validate` that rejects everything.
    """
    driver = _driver()
    parser = driver.build_parser()
    args = parser.parse_args(
        [
            "--data-root", "/nonexistent",
            "--schedule", "8192:300,16384:300,32768:900",
            "--max-steps", "600",
            "--query-chunk", "1024",
            "--global-batch-size", "8",
            "--shuffle-control-every", "100",
        ]
    )
    assert args.pin_mode == "sink" and args.batch_size == 1 and not args.liger
    if not torch.cuda.is_available():
        # validate() ends with a CUDA check, so on CPU the only reachable assertion is that nothing
        # BEFORE it fired. `parser.error` writes the message to stderr and raises SystemExit(2), so
        # the reason has to be read from there rather than from the exception.
        import contextlib
        import io

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), pytest.raises(SystemExit):
            driver.validate(args, parser)
        assert "CUDA" in stderr.getvalue(), (
            f"the intended configuration was rejected for a reason other than the missing GPU: "
            f"{stderr.getvalue().strip()}"
        )
    else:
        driver.validate(args, parser)


def test_driver_scales_the_loss_by_accum_steps_not_after_the_fact():
    """The driver must pass ``loss_scale``, never divide the returned loss.

    A source-level assertion, which is unusual but justified: the wrong version is *numerically
    invisible* in any single-step test (with accum_steps=1 the two agree exactly) and only shows up as
    a silently wrong effective learning rate on multi-GPU or accumulated runs. Reading the call is the
    cheapest reliable check.
    """
    import inspect

    driver = _driver()
    source = inspect.getsource(driver.main)
    assert "loss_scale=1.0 / args.accum_steps" in source, (
        "the driver must hand the scale to cross_replay_training_step, which backwards internally"
    )
    assert "/ args.accum_steps).backward()" not in source, (
        "the returned loss is DETACHED -- scaling it after the fact leaves the gradients "
        "accum_steps times too large, with a perfectly normal-looking loss"
    )
