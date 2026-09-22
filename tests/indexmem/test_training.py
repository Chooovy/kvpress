import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from kvpress.indexmem.ablations.joint import joint_parameters
from kvpress.indexmem.ablations.objectives import objective_loss
from kvpress.indexmem.checkpoint import attach_scorers
from kvpress.indexmem.kernels.gate_normalization import history_logsumexp
from kvpress.indexmem.training.optim import warmup_stable_decay


def explicit_normalizer(queries, keys, scale, sink_size, window_size, offset):
    q = torch.arange(queries.shape[2], device=queries.device)[:, None] + offset
    k = torch.arange(keys.shape[1], device=keys.device)[None, :]
    history = (k >= sink_size) & (k <= q - window_size)
    scores = torch.einsum("bhqd,bkd->bhqk", queries, keys) * scale
    scores = scores.masked_fill(~history, -torch.inf)
    nonempty = history.any(-1)
    scores = torch.where(nonempty[None, None, :, None], scores, 0.0)
    normalizer = torch.logsumexp(scores, -1)
    return torch.where(nonempty, normalizer, 0.0)


@pytest.mark.parametrize("sink_size,window_size", [(2, 0), (0, 3), (2, 3), (20, 30)])
@pytest.mark.parametrize("offset", [0, 4])
def test_training_normalizer_values_and_gradients(sink_size, window_size, offset):
    torch.manual_seed(4)
    values = [
        torch.randn(1, 2, 9, 4, dtype=torch.float64),
        torch.randn(1, 13, 4, dtype=torch.float64),
        torch.tensor([0.7], dtype=torch.float64),
    ]
    fast = [value.clone().requires_grad_() for value in values]
    reference = [value.clone().requires_grad_() for value in values]
    result = history_logsumexp(*fast, sink_size=sink_size, window_size=window_size, query_offset=offset, tile_size=5)
    expected = explicit_normalizer(*reference, sink_size, window_size, offset)
    cotangent = torch.randn_like(result)
    gradients = torch.autograd.grad(result, fast, cotangent)
    expected_gradients = torch.autograd.grad(expected, reference, cotangent)
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)
    for gradient, expected_gradient in zip(gradients, expected_gradients):
        torch.testing.assert_close(gradient, expected_gradient, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize(
    "objective,ce_weight", [("reverse_kl", 0.0), ("forward_kl", 0.0), ("reverse_kl", 0.1), ("ce", 0.0), ("longce", 0.0)]
)
def test_training_losses_match_full_logits(objective, ce_weight):
    torch.manual_seed(3)
    head = nn.Linear(8, 19, bias=False).requires_grad_(False)
    hidden = torch.randn(2, 7, 8, requires_grad=True)
    baseline = hidden.detach().clone().requires_grad_()
    teacher = torch.randn_like(hidden)
    tokens = torch.randint(19, (2, 7))
    weights = torch.rand(2, 6) + 0.1
    result = objective_loss(
        hidden,
        tokens,
        head,
        objective=objective,
        teacher_hidden=teacher,
        weights=weights,
        ce_weight=ce_weight,
        chunk_size=3,
    )
    student_logprobs = F.log_softmax(head(baseline).float(), -1)
    teacher_logprobs = F.log_softmax(head(teacher).float(), -1)
    ce_tokens = F.cross_entropy(
        head(baseline[:, :-1]).reshape(-1, 19).float(), tokens[:, 1:].reshape(-1), reduction="none"
    )
    if objective == "reverse_kl":
        expected = (student_logprobs.exp() * (student_logprobs - teacher_logprobs)).sum(-1).mean()
        expected = (1 - ce_weight) * expected + ce_weight * ce_tokens.mean()
    elif objective == "forward_kl":
        expected = (teacher_logprobs.exp() * (teacher_logprobs - student_logprobs)).sum(-1).mean()
    elif objective == "longce":
        expected = (ce_tokens * weights.reshape(-1)).sum() / weights.sum()
    else:
        expected = ce_tokens.mean()
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(torch.autograd.grad(result, hidden)[0], torch.autograd.grad(expected, baseline)[0])


def tiny_model(device="cpu", dtype=torch.float32):
    config = Qwen3Config(
        vocab_size=67,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(config).to(device=device, dtype=dtype).eval()


def test_training_joint_parameter_partition():
    model = tiny_model()
    attach_scorers(
        model, {"kind": "mlp", "hidden_size": 64, "n_heads": 2, "mid_dim": 16, "decay": True, "gate_scale": True}
    )
    scorer, backbone = joint_parameters(model)
    assert set(map(id, scorer)).isdisjoint(map(id, backbone))
    assert all(parameter.requires_grad for parameter in scorer + backbone)
    assert all(not parameter.requires_grad for layer in model.model.layers for parameter in layer.mlp.parameters())
    assert all(not parameter.requires_grad for parameter in model.model.embed_tokens.parameters())
    assert all(not layer.self_attn.retention_scorer.gate_scale.requires_grad for layer in model.model.layers)
    assert not model.lm_head.weight.requires_grad


def test_training_wsd_reaches_final_learning_rate():
    schedule = warmup_stable_decay(100, final_fraction=0.005)
    assert schedule(0) == pytest.approx(0.1)
    assert schedule(10) == 1.0
    assert schedule(69) == 1.0
    assert schedule(99) == pytest.approx(0.005)


def _sequence_parallel_worker(rank, init_file):
    import torch.distributed as dist

    from kvpress.indexmem.training.sequence_parallel import SequenceParallelFFN

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    torch.manual_seed(71)
    scorer = nn.Linear(8, 8)
    reference_scorer = copy.deepcopy(scorer)
    ffn = nn.Sequential(nn.Linear(8, 17), nn.GELU(), nn.Linear(17, 8)).requires_grad_(False)
    sharded_ffn = SequenceParallelFFN(ffn, dist.group.WORLD)
    inputs = torch.randn(2, 7, 8)
    hidden = scorer(inputs)
    result = hidden + sharded_ffn(hidden)
    reference_hidden = reference_scorer(inputs)
    expected = reference_hidden + ffn(reference_hidden)
    result.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(result, expected)
    for parameter, reference_parameter in zip(scorer.parameters(), reference_scorer.parameters()):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad)
    dist.destroy_process_group()


def test_training_sequence_parallel_preserves_full_gradient(tmp_path):
    torch.multiprocessing.spawn(_sequence_parallel_worker, args=(str(tmp_path / "process_group"),), nprocs=2, join=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA training kernel")
@pytest.mark.parametrize(
    "kind,extra",
    [
        ("mlp", {"mid_dim": 16}),
        ("linear", {"mid_dim": 0}),
        ("conv", {"mid_dim": 16, "conv_dim": 16, "conv_kernel": 8}),
        ("rnn", {"mid_dim": 16, "state_dim": 16}),
        ("prefix", {"mid_dim": 16, "head_dim": 16, "value_dim": 16}),
        ("kvzip", {"mid_dim": 0, "kvzip_dim": 16, "kvzip_base": 8, "kvzip_ngroup": 2}),
    ],
)
def test_training_scorer_forward_backward_cuda(kind, extra):
    from kvpress.indexmem.training.trainer import RetentionTrainer

    torch.manual_seed(19)
    model = tiny_model("cuda", torch.bfloat16)
    attach_scorers(model, {"kind": kind, "hidden_size": 64, "n_heads": 2, "decay": True, "gate_scale": True, **extra})
    trainer = RetentionTrainer(model, sink_size=2, window_size=4, gate_mass=8.0)
    parameters = trainer.freeze_backbone()
    loss = trainer.loss(torch.randint(67, (1, 23), device="cuda"), chunk_size=8)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters)
    assert any(parameter.grad.abs().sum() > 0 for parameter in parameters)
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if "retention_scorer" not in name)
    assert model.config._attn_implementation == "sdpa"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA training kernel")
@pytest.mark.parametrize(
    "objective,ce_weight,sink_size,window_size",
    [
        ("reverse_kl", 0.0, 2, 4),
        ("forward_kl", 0.0, 2, 4),
        ("ce", 0.0, 2, 4),
        ("longce", 0.0, 2, 4),
        ("reverse_kl", 0.1, 2, 4),
        ("ce", 0.0, 0, 0),
    ],
)
def test_training_objective_forward_backward_cuda(objective, ce_weight, sink_size, window_size):
    from kvpress.indexmem.training.trainer import RetentionTrainer

    torch.manual_seed(20)
    model = tiny_model("cuda", torch.bfloat16)
    attach_scorers(
        model, {"kind": "mlp", "hidden_size": 64, "n_heads": 2, "mid_dim": 16, "decay": True, "gate_scale": True}
    )
    trainer = RetentionTrainer(model, sink_size=sink_size, window_size=window_size, gate_mass=8.0)
    parameters = trainer.freeze_backbone()
    loss = trainer.loss(
        torch.randint(67, (1, 23), device="cuda"),
        objective=objective,
        ce_weight=ce_weight,
        weights=torch.linspace(0.1, 5.0, 22, device="cuda")[None],
        chunk_size=8,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters)
    assert any(parameter.grad.abs().sum() > 0 for parameter in parameters)
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if "retention_scorer" not in name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA training kernel")
@pytest.mark.parametrize("sink_size,window_size", [(0, 0), (2, 0), (0, 4), (2, 4)])
def test_training_fused_attention_values_and_gradients_cuda(sink_size, window_size):
    from kvpress.indexmem.kernels.gated_attention import gated_attention

    torch.manual_seed(21)
    shapes = [(1, 4, 19, 16), (1, 2, 23, 16), (1, 2, 23, 16), (1, 2, 19, 4), (1, 23, 4)]
    values = [torch.randn(shape, device="cuda") * 0.2 for shape in shapes]
    values.append(torch.tensor([0.7], device="cuda"))
    actual_inputs = [value.clone().requires_grad_() for value in values]
    expected_inputs = [value.clone().requires_grad_() for value in values]
    actual = gated_attention(*actual_inputs, scaling=0.25, gate_mass=8.0, sink_size=sink_size, window_size=window_size)
    query, key, value, gate_queries, gate_keys, scale = expected_inputs
    scores = torch.einsum("bhqd,bkd->bhqk", gate_queries, gate_keys) * scale
    query_position = torch.arange(19, device="cuda")[:, None] + 4
    key_position = torch.arange(23, device="cuda")[None, :]
    pinned = (key_position < sink_size) | (key_position > query_position - window_size)
    if sink_size or window_size:
        normalizer = explicit_normalizer(gate_queries, gate_keys, scale, sink_size, window_size, 4)
        gates = torch.where(pinned, 0.0, scores - normalizer[..., None] + torch.tensor(8.0).log().item())
    else:
        gates = scores
    logits = query @ key.repeat_interleave(2, dim=1).transpose(-1, -2) * 0.25 + gates.repeat_interleave(2, dim=1)
    logits = logits.masked_fill(key_position > query_position, -torch.inf)
    expected = logits.softmax(-1) @ value.repeat_interleave(2, dim=1)
    cotangent = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, actual_inputs, cotangent)
    expected_gradients = torch.autograd.grad(expected, expected_inputs, cotangent)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    for gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(gradient, expected_gradient, atol=3e-5, rtol=5e-4)
