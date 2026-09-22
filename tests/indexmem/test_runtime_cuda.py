from __future__ import annotations

import pytest
import torch

from kvpress.indexmem.config import IndexMemConfig
from kvpress.indexmem.inference.cmp import CentroidMemory
from kvpress.indexmem.inference.context import IndexMemInferenceContext
from kvpress.indexmem.inference.paged_cache import PAGE_BLOCK, PagedKVPool

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("query_length", [1, 5])
def test_sparse_kernel_matches_explicit_attention(query_length):
    from kvpress.indexmem.kernels.sparse_attention import sparse_gqa_attention

    torch.manual_seed(23)
    query = torch.randn(2, 4, query_length, 16, device="cuda")
    key = torch.randn(2, 2, 19, 16, device="cuda")
    value = torch.randn(2, 2, 19, 16, device="cuda")
    support = (
        torch.tensor([0, 2, 5, 7, 11, 15, 18, -1], device="cuda", dtype=torch.int32)
        .view(1, 1, 1, -1)
        .expand(2, 2, query_length, -1)
        .clone()
    )
    actual, _ = sparse_gqa_attention(query, key, value, support, precision="ieee")
    expected = torch.empty_like(actual)
    for batch in range(2):
        for head in range(4):
            for row in range(query_length):
                indices = support[batch, head // 2, row].long()
                indices = indices[(indices >= 0) & (indices <= 19 - query_length + row)]
                logits = query[batch, head, row] @ key[batch, head // 2, indices].T / 4
                expected[batch, head, row] = logits.softmax(-1) @ value[batch, head // 2, indices]
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("query_length", [1, 3])
def test_paged_attention_with_empty_rows_and_cmp_matches_joint_softmax(query_length):
    torch.manual_seed(29)
    device = torch.device("cuda")
    pool = PagedKVPool(
        torch.tensor([[12, 16]]),
        batch_size=2,
        n_layers=1,
        n_kv_heads=2,
        n_sink=2,
        n_local=4,
        head_dim=16,
        device=device,
        dtype=torch.bfloat16,
    )
    for sequence, length in [(0, 0), (1, 20)]:
        key = torch.randn(2, length, 16, device=device, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        pool.commit(
            0,
            sequence,
            key=key,
            value=value,
            score_intercept=torch.randn(2, length, device=device),
            log_retention_rate=None,
            k_len=length,
        )
    memory = CentroidMemory(4, 3, 16, device=device)
    memory.centroid_keys[2:] = torch.randn(2, 3, 16, device=device)
    memory.centroid_values[2:] = torch.randn(2, 3, 16, device=device)
    memory.cluster_counts[2:] = torch.tensor([[1, 3, 0], [2, 4, 7]], device=device)
    query = torch.randn(2, 4, query_length, 16, device=device, dtype=torch.bfloat16)
    arriving_keys = torch.randn(2, 2, query_length, 16, device=device, dtype=torch.bfloat16)
    arriving_values = torch.randn_like(arriving_keys)
    extra = memory.read(torch.arange(4, device=device), query, group=2, scaling=0.25)
    actual = pool.attend(
        0, query, new_key=arriving_keys, new_value=arriving_values, scaling=0.25, extra=extra
    ).transpose(1, 2)
    expected = torch.empty_like(query, dtype=torch.float32)
    for batch in range(2):
        for head in range(4):
            row = batch * 2 + head // 2
            slots = torch.arange(int(pool.filled[row]), device=device)
            blocks = pool.block_table[row, slots // PAGE_BLOCK].long()
            cached_keys = pool.k_pool[blocks, slots % PAGE_BLOCK, 0].float()
            cached_values = pool.v_pool[blocks, slots % PAGE_BLOCK, 0].float()
            for token in range(query_length):
                keys = torch.cat(
                    [cached_keys, arriving_keys[batch, head // 2, : token + 1].float(), memory.centroid_keys[row]]
                )
                values = torch.cat(
                    [cached_values, arriving_values[batch, head // 2, : token + 1].float(), memory.centroid_values[row]]
                )
                logits = query[batch, head, token].float() @ keys.T / 4
                logits[-3:] += memory.cluster_counts[row].log()
                expected[batch, head, token] = logits.softmax(-1) @ values
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.02)


@torch.no_grad()
def test_model_forward_defers_cache_and_cmp_updates_until_finish_step():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from kvpress.indexmem.checkpoint import attach_scorers

    torch.manual_seed(31)
    model_config = Qwen3Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
    )
    model_config._attn_implementation = "flash_attention_2"
    model = Qwen3ForCausalLM(model_config).to(device="cuda", dtype=torch.bfloat16).eval()
    attach_scorers(model, dict(kind="mlp", hidden_size=64, n_heads=2, mid_dim=16, gate_scale=True, decay=True))
    config = IndexMemConfig(
        cache_budget=16, sink_size=2, window_size=4, cmp_slots=3, head_budget="uniform", min_head_budget=0
    )
    original = model.config._attn_implementation
    prompt = torch.randint(0, 97, (1, 40), device="cuda")
    with IndexMemInferenceContext(model, config) as context:
        context.prefill_and_commit(prompt)
        context.activate()
        before_seen = context.pool.seen.clone()
        before_counts = context.cmp.cluster_counts.clone()
        incoming = torch.randint(0, 97, (1, 3), device="cuda")
        output = model(
            input_ids=incoming,
            past_key_values=context.new_cache(),
            position_ids=torch.arange(40, 43, device="cuda")[None],
        )
        assert bool(torch.isfinite(output.logits).all())
        assert torch.equal(context.pool.seen, before_seen)
        assert torch.equal(context.cmp.cluster_counts, before_counts)
        context.finish_step()
        assert context.pool.seen.item() == 43
        assert bool((context.cmp.cluster_counts.sum(-1) > before_counts.sum(-1)).all())
        assert bool((context.pool.filled.long() == context.pool.row_budget).all())
    assert model.config._attn_implementation == original
