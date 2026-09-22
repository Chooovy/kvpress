from __future__ import annotations

import pytest
import torch

from kvpress.indexmem.inference.attention import _merge_lse
from kvpress.indexmem.inference.budget import allocate_by_mass
from kvpress.indexmem.inference.cmp import CentroidMemory, cluster_evicted
from kvpress.indexmem.inference.context import shift_router_key
from kvpress.indexmem.inference.paged_cache import PAGE_BLOCK, PagedKVPool, rank_key
from kvpress.indexmem.kernels.prefill_attention import deadlines


def retained_positions(scores, position, budget, sink, window):
    boundary = max(sink, position - window + 1)
    protected = set(range(min(sink, position + 1))) | set(range(boundary, position + 1))
    candidates = sorted(range(sink, boundary), key=lambda index: (-float(scores[index]), index))
    return protected | set(candidates[: max(0, budget - len(protected))])


@pytest.mark.parametrize("length", [7, 129, 1031])
def test_deadlines_match_independent_stable_topk(length):
    generator = torch.Generator().manual_seed(97)
    scores = torch.randn(3, length, generator=generator).bfloat16().float()
    scores[:, ::4] = 0
    budgets = torch.tensor([12, 21, 31])
    sink, window = (2, 4)
    expiry = deadlines(scores, budgets, sink_size=sink, window_size=window)
    for position in sorted({0, 2, min(7, length - 1), length // 2, length - 1}):
        for head, budget in enumerate(budgets.tolist()):
            horizon = position - window
            actual = {
                index
                for index in range(position + 1)
                if index < sink or index > horizon or horizon <= expiry[head, index]
            }
            assert actual == retained_positions(scores[head], position, budget, sink, window)


def test_rank_key_preserves_score_priority_and_oldest_ties():
    scores = torch.tensor([-2.0, -1.0, 0.0, 0.0, 1.0])
    positions = torch.tensor([0, 1, 2, 3, 1000000])
    packed = rank_key(scores, positions)
    assert packed.argsort().tolist() == [0, 1, 3, 2, 4]


def physical_positions(pool, layer, sequence, head):
    row = pool.rows_for(layer, sequence).start + head
    slots = torch.arange(int(pool.filled[row]))
    blocks = pool.block_table[row, slots // PAGE_BLOCK].long()
    return set(pool.k_pool[blocks, slots % PAGE_BLOCK, 0, 0].long().tolist())


@pytest.mark.parametrize("context_length", [4, 28])
def test_paged_cache_growth_eviction_and_replication(context_length):
    budgets = torch.tensor([[11, 17], [13, 15]])
    pool = PagedKVPool(
        budgets,
        batch_size=2,
        n_layers=2,
        n_kv_heads=2,
        n_sink=2,
        n_local=4,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(103)
    scores = torch.randn(2, 2, 2, 48, generator=generator).bfloat16().float()
    scores[:, 1, :, :context_length] = scores[:, 0, :, :context_length]
    keys = torch.arange(48).float().view(1, 1, 1, 48, 1).expand(2, 2, 2, 48, 8)
    for layer in range(2):
        pool.commit(
            layer,
            0,
            key=keys[layer, 0, :, :context_length],
            value=keys[layer, 0, :, :context_length],
            score_intercept=scores[layer, 0, :, :context_length],
            log_retention_rate=None,
            k_len=context_length,
        )
    pool.replicate_seq(0, 1)
    assert not torch.equal(pool.block_table[pool.rows_for(0, 0)], pool.block_table[pool.rows_for(0, 1)])
    for position in range(context_length, 48):
        pool.ingest(
            [0, 1],
            key=keys[:, :, :, position],
            value=keys[:, :, :, position],
            score_intercept=scores[:, :, :, position],
            log_retention_rate=None,
            positions=torch.tensor([position, position]),
        )
        pool.seen[:] = position + 1
        for layer in range(2):
            for sequence in range(2):
                for head in range(2):
                    expected = retained_positions(
                        scores[layer, sequence, head], position, int(budgets[layer, head]), 2, 4
                    )
                    assert physical_positions(pool, layer, sequence, head) == expected
    assert torch.equal(pool.filled.long(), pool.row_budget)


@pytest.mark.parametrize("total", [0, 2, 8, 27, 100])
def test_mass_allocation_conserves_total(total):
    probabilities = torch.tensor([[0.8, 0.15, 0.05], [0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])
    cumulative = probabilities.cumsum(-1)
    allocation = allocate_by_mass(cumulative, min(total, 9))
    assert allocation.sum().item() == min(total, 9)
    assert bool((allocation >= 0).all())
    assert bool((allocation <= 3).all())


def test_cmp_population_and_read_match_explicit_attention():
    memory = CentroidMemory(1, 1, 3, device=torch.device("cpu"))
    keys = torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    values = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])
    for key, value in zip(keys, values):
        memory.ingest(torch.tensor([0]), key[None], value[None])
    torch.testing.assert_close(memory.centroid_keys[0, 0], keys.mean(0))
    torch.testing.assert_close(memory.centroid_values[0, 0], values.mean(0))
    assert memory.cluster_counts.item() == 3
    query = torch.tensor([[[[0.25, 0.0, 0.0]]]])
    output, logsumexp = memory.read(torch.tensor([0]), query, group=1, scaling=1.0)
    torch.testing.assert_close(output.flatten(), values.mean(0))
    torch.testing.assert_close(logsumexp.flatten(), torch.tensor([0.75 + torch.log(torch.tensor(3.0))]))


def test_cmp_empty_rows_are_inert_and_masked_keys_do_not_contribute():
    memory = CentroidMemory(2, 3, 4, device=torch.device("cpu"))
    output, logsumexp = memory.read(torch.tensor([0, 1]), torch.randn(2, 2, 3, 4), group=2, scaling=0.5)
    assert torch.count_nonzero(output) == 0
    assert bool(torch.isneginf(logsumexp).all())
    keys = torch.tensor([[[1.0], [3.0], [1000.0]]])
    values = keys * 2
    centroids, representatives, log_mass = cluster_evicted(
        keys, values, torch.tensor([[True, True, False]]), 1, generator=torch.Generator().manual_seed(7)
    )
    torch.testing.assert_close(centroids, torch.tensor([[[2.0]]]))
    torch.testing.assert_close(representatives, torch.tensor([[[4.0]]]))
    torch.testing.assert_close(log_mass.exp(), torch.tensor([[2.0]]))


def test_three_branch_lse_merge_matches_one_softmax():
    generator = torch.Generator().manual_seed(5)
    logits = torch.randn(2, 3, 4, 11, generator=generator) * 20
    values = torch.randn(2, 3, 11, 6, generator=generator)
    branches = []
    for start, stop in [(0, 4), (4, 9), (9, 11)]:
        branch_logits = logits[..., start:stop]
        output = branch_logits.softmax(-1) @ values[:, :, start:stop]
        branches.append((output, branch_logits.logsumexp(-1)))
    torch.testing.assert_close(_merge_lse(branches), logits.softmax(-1) @ values, rtol=2e-06, atol=2e-06)
    empty = (torch.zeros_like(branches[0][0]), torch.full_like(branches[0][1], float("inf")))
    torch.testing.assert_close(_merge_lse([empty, branches[0]]), branches[0][0])
    assert torch.count_nonzero(_merge_lse([empty, empty])) == 0


def test_batched_absolute_position_correction():
    score_intercept = torch.tensor([[[2.0, 4.0]], [[3.0, 5.0]]])
    log_retention_rate = torch.tensor([[[-0.2, -0.3]], [[-0.4, -0.5]]])
    offsets = torch.tensor([[10, 11], [20, 21]])
    result = shift_router_key(score_intercept, log_retention_rate, offset=offsets, pos_slope=0.01, age_scale=100)
    expected = score_intercept + offsets[:, None] * 0.01 - log_retention_rate * offsets[:, None] / 100
    torch.testing.assert_close(result, expected)
