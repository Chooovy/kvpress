import pytest
import torch

from kvpress.indexmem.ablations import (
    ConvRetentionScorer,
    ConvRetentionScorerConfig,
    KVzipRetentionScorer,
    KVzipRetentionScorerConfig,
    PrefixRetentionScorer,
    PrefixRetentionScorerConfig,
    RecurrentRetentionScorer,
    RecurrentRetentionScorerConfig,
)
from kvpress.indexmem.ablations.recurrent import gated_scan
from kvpress.indexmem.scorer import RetentionScorer, RetentionScorerConfig

HISTORY_SCORERS = [
    (ConvRetentionScorer, ConvRetentionScorerConfig, {"conv_kernel": 8, "conv_dim": 8}),
    (RecurrentRetentionScorer, RecurrentRetentionScorerConfig, {"state_dim": 8}),
    (PrefixRetentionScorer, PrefixRetentionScorerConfig, {"head_dim": 8, "value_dim": 8}),
]


@pytest.mark.parametrize("scorer_cls,config_cls,extra", HISTORY_SCORERS)
def test_zero_initialized_history_matches_mlp_outputs_and_input_gradients(scorer_cls, config_cls, extra):
    torch.manual_seed(4)
    base_config = {"hidden_size": 16, "n_heads": 2, "mid_dim": 8, "decay": True}
    base = RetentionScorer(RetentionScorerConfig(**base_config))
    history = scorer_cls(config_cls(**base_config, **extra))
    history.load_state_dict(base.state_dict(), strict=False)
    hidden = torch.randn(2, 13, 16, requires_grad=True)
    other = hidden.detach().clone().requires_grad_(True)
    expected = base.gate_key(hidden)
    actual = history.gate_key(other)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    weights = torch.randn_like(expected)
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    torch.testing.assert_close(other.grad, hidden.grad, rtol=0, atol=0)
    assert history.history_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("scorer_cls,config_cls,extra", HISTORY_SCORERS)
def test_history_scores_match_across_prefill_and_decode_chunks(scorer_cls, config_cls, extra):
    torch.manual_seed(5)
    scorer = scorer_cls(config_cls(hidden_size=16, n_heads=2, mid_dim=8, decay=True, **extra))
    torch.nn.init.normal_(scorer.history_proj.weight, std=0.1)
    hidden = torch.randn(2, 17, 16)
    expected = scorer.gate_key(hidden)
    scorer.enable_cache()
    actual = torch.cat(
        [
            scorer.gate_key(hidden[:, start:end], key_offset=start)
            for start, end in [(0, 3), (3, 11), (11, 16), (16, 17)]
        ],
        dim=1,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert scorer.cached_length == hidden.shape[1]
    scorer.disable_cache()
    torch.testing.assert_close(scorer.gate_key(hidden), expected)


@pytest.mark.parametrize("scorer_cls,config_cls,extra", HISTORY_SCORERS)
def test_history_scores_do_not_read_future_tokens(scorer_cls, config_cls, extra):
    torch.manual_seed(6)
    scorer = scorer_cls(config_cls(hidden_size=16, n_heads=2, mid_dim=8, **extra))
    torch.nn.init.normal_(scorer.history_proj.weight, std=0.1)
    original = torch.randn(2, 13, 16)
    changed = original.clone()
    changed[:, 7:] = torch.randn_like(changed[:, 7:]) * 3
    torch.testing.assert_close(scorer.score_keys(original)[..., :7], scorer.score_keys(changed)[..., :7])


def test_recurrent_scan_matches_stepwise_state_and_gradients():
    torch.manual_seed(7)
    retain = torch.sigmoid(torch.randn(2, 19, 5)).requires_grad_(True)
    updates = torch.randn(2, 19, 5, requires_grad=True)
    state = torch.zeros_like(updates[:, 0])
    states = []
    for step in range(updates.shape[1]):
        state = retain[:, step] * state + updates[:, step]
        states.append(state)
    expected = torch.stack(states, dim=1)
    actual = gated_scan(retain, updates)
    torch.testing.assert_close(actual, expected)
    weight = torch.randn_like(expected)
    expected_grads = torch.autograd.grad((expected * weight).sum(), (retain, updates), retain_graph=True)
    actual_grads = torch.autograd.grad((actual * weight).sum(), (retain, updates))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_kvzip_gate_uses_its_learned_retention_rate():
    torch.manual_seed(8)
    scorer = KVzipRetentionScorer(
        KVzipRetentionScorerConfig(
            hidden_size=16,
            n_heads=2,
            mid_dim=0,
            decay=True,
            kvzip_dim=4,
            kvzip_base=3,
            kvzip_ngroup=2,
            age_scale=32,
        )
    )
    hidden = torch.randn(1, 9, 16)
    magnitude = scorer.score_keys(hidden)
    at_end = scorer.score_at(hidden, 8)
    positions = torch.arange(9).view(1, 1, -1)
    torch.testing.assert_close(at_end, magnitude - (8 - positions) / 32)
    loss = at_end.square().mean()
    loss.backward()
    assert scorer.q_proj.weight.grad.abs().sum() > 0
    assert scorer.retention_proj.weight.grad.abs().sum() > 0
