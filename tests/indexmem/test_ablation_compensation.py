import torch

from kvpress.indexmem.ablations.compensation import compare_compensation, ttt_linear_read


def test_linear_memory_recovers_a_linear_key_value_map():
    torch.manual_seed(19)
    keys = torch.randn(2, 15, 4)
    matrix = torch.randn(2, 4, 6)
    queries = torch.randn(2, 7, 4)
    values = keys @ matrix
    torch.testing.assert_close(ttt_linear_read(queries, keys, values, 0), queries @ matrix)


def test_single_centroid_recovers_identical_keys():
    torch.manual_seed(20)
    keys = torch.randn(2, 1, 4).expand(2, 7, 4).contiguous()
    values = torch.randn(2, 7, 4)
    queries = torch.randn(2, 3, 4)
    results = compare_compensation(queries, keys, values, slots=(1,), mlp_hidden=4, ttt_steps=2)
    assert results["centroids_1"]["relative_l2"] < 1e-6
    assert abs(results["centroids_1"]["cosine"] - 1) < 1e-6
