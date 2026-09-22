import pytest
import torch

from kvpress.indexmem.scorer import RetentionScorer, RetentionScorerConfig


@pytest.mark.parametrize("decay", [False, True])
@pytest.mark.parametrize("mid_dim", [0, 16])
def test_gate_factorization_and_absolute_positions(decay, mid_dim):
    torch.manual_seed(17)
    scorer = RetentionScorer(RetentionScorerConfig(32, 4, mid_dim=mid_dim, decay=decay))
    if decay:
        with torch.no_grad():
            scorer.retention_proj.weight.normal_(0, 0.1)
    hidden = torch.randn(2, 9, 32)
    keys = scorer.gate_key(hidden, key_offset=16381)
    queries = scorer.gate_query(3, 2, 4, query_offset=16390, device=hidden.device, dtype=hidden.dtype)
    scores = torch.einsum("bhqd,bkd->bhqk", queries, keys)
    for row in range(3):
        expected = scorer.score_at(hidden, 16390 + row, key_offset=16381)
        torch.testing.assert_close(scores[:, :, row], expected, atol=5e-7, rtol=2e-6)


def test_retention_gradients_and_protected_padding():
    torch.manual_seed(29)
    scorer = RetentionScorer(RetentionScorerConfig(8, 2, mid_dim=8, decay=True, gate_scale=True))
    hidden = torch.randn(2, 6, 8, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    magnitude, retention = scorer._score_and_decay(hidden, mask=mask)
    assert torch.all(magnitude[0, :, -2:] == -10000)
    assert torch.all(retention[0, :, -2:] == 0)
    assert torch.all(retention <= 0)
    loss = scorer.score_at(hidden, 20).square().mean() * scorer.require_gate_scale().sum()
    loss.backward()
    for name in ("input_proj", "magnitude_proj", "retention_proj"):
        gradient = getattr(scorer, name).weight.grad
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    assert torch.isfinite(hidden.grad).all()
