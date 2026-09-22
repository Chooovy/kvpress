import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from kvpress.indexmem.ablations.budget import fit_offline_head_budgets, permute_head_budgets
from kvpress.indexmem.checkpoint import attach_scorers


def test_budget_permutation_preserves_the_allocation_multiset():
    budgets = torch.tensor([32, 48, 52, 60, 64, 96, 128, 160])
    first = permute_head_budgets(budgets, 0)
    second = permute_head_budgets(budgets, 0)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first.sort().values, budgets.sort().values)
    assert not torch.equal(first, budgets)


@pytest.mark.parametrize("min_head_budget", [0, 1, 4])
def test_offline_budget_fitting_restores_attention_and_preserves_layer_totals(min_head_budget):
    torch.manual_seed(23)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    model = Qwen3ForCausalLM(config).eval()
    attach_scorers(model, {"kind": "mlp", "hidden_size": 32, "n_heads": 2, "mid_dim": 8, "decay": True})
    documents = [torch.randint(0, 32, (1, 10)) for _ in range(3)]
    previous = model.config._attn_implementation
    table = fit_offline_head_budgets(
        model,
        documents,
        cache_budget=4,
        sink_size=1,
        window_size=1,
        min_head_budget=min_head_budget,
    )
    assert table.shape == (2, 2)
    torch.testing.assert_close(table.sum(-1), torch.tensor([8, 8]))
    assert bool((table >= 2 + min(min_head_budget, 2)).all())
    assert model.config._attn_implementation == previous
    assert all(not layer.self_attn._forward_pre_hooks for layer in model.model.layers)
