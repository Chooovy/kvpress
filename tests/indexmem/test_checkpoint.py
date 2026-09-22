from dataclasses import fields

import pytest
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from kvpress.indexmem.checkpoint import attach_scorers, load_scorer_checkpoint, save_scorer_checkpoint, scorer_class
from scripts.convert_checkpoint import convert_checkpoint

KINDS = ("mlp", "linear", "conv", "rnn", "prefix", "kvzip")
LEGACY_NAMES = {
    "input_proj": "w_in",
    "magnitude_proj": "w_out",
    "retention_proj": "w_decay",
    "history_input_proj": "w_cin",
    "history_proj": "w_a",
    "history_norm": "a_norm",
    "state_input_proj": "w_u",
    "state_gate_proj": "w_g",
    "prefix_query_proj": "w_pq",
    "prefix_key_proj": "w_pk",
    "prefix_value_proj": "w_pv",
}


def model(layers=2, dtype=torch.float32):
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
    ).to(dtype)


def scorer_config(kind="mlp", **overrides):
    _, config_cls = scorer_class(kind)
    extra = {
        "mlp": {},
        "linear": {},
        "conv": {"conv_dim": 8, "conv_kernel": 3, "zero_init_conv": False},
        "rnn": {"state_dim": 8, "zero_init_state": False},
        "prefix": {"head_dim": 8, "value_dim": 8, "zero_init_prefix": False},
        "kvzip": {"kvzip_dim": 4, "kvzip_base": 3, "kvzip_ngroup": 2},
    }[kind]
    config = config_cls(
        hidden_size=32,
        n_heads=2,
        mid_dim=0 if kind in ("linear", "kvzip") else 16,
        gate_scale=True,
        decay=True,
        age_scale=7919.0,
        pos_slope=0.003,
        **(extra | overrides),
    )
    return {"kind": kind, **{field.name: getattr(config, field.name) for field in fields(config) if field.init}}


def legacy_state(net):
    state = {}
    for name, tensor in net.state_dict().items():
        if ".retention_scorer." in name:
            prefix, parameter = name.split(".retention_scorer.")
            parameter = ".".join(LEGACY_NAMES.get(part, part) for part in parameter.split("."))
            name = prefix + ".indexer." + parameter
            if parameter == "gate_scale":
                tensor = tensor.reshape(())
        state[name.replace(".mlp.", ".mlp.inner.")] = tensor.clone()
    return state


def legacy_checkpoint(net, joint=False):
    state = legacy_state(net)
    source = {
        "indexer": {key: value for key, value in state.items() if ".indexer." in key},
        "config": {"objective": "joint_lm_loss" if joint else "rvkl", "joint": joint},
        "optimizer": {"legacy_state": torch.ones(1)},
        "scheduler": {"step": 100},
        "step": 100,
    }
    if joint:
        source["model"] = state
    return source


def initialized_model(kind="mlp", dtype=torch.float32, **overrides):
    net = model(dtype=dtype)
    config = scorer_config(kind, **overrides)
    scorers = attach_scorers(net, config)
    with torch.no_grad():
        for layer, scorer in enumerate(scorers):
            scorer.gate_scale.fill_(1.234567 + layer)
            scorer.retention_proj.weight.normal_(std=0.02)
    return net, config


def assert_scorers_equal(original, restored, dtype):
    hidden = torch.randn(2, 5, 32, dtype=dtype)
    for first, second in zip(original.model.layers, restored.model.layers, strict=True):
        left = first.self_attn.retention_scorer
        right = second.self_attn.retention_scorer
        assert left.config == right.config
        assert right.gate_scale.dtype == torch.float32
        for name, value in left.state_dict().items():
            torch.testing.assert_close(right.state_dict()[name], value, rtol=0, atol=0)
        torch.testing.assert_close(
            left.gate_key(hidden, key_offset=113), right.gate_key(hidden, key_offset=113), rtol=0, atol=0
        )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_legacy_conversion_strict_roundtrip(tmp_path, kind, joint, dtype):
    torch.manual_seed(23)
    original, config = initialized_model(kind, dtype)
    source = legacy_checkpoint(original, joint)
    payload = convert_checkpoint(source, config)
    assert "optimizer" not in payload
    assert "scheduler" not in payload
    assert "schedule" not in payload
    assert payload["step"] == 0
    assert payload["training_config"] is None
    assert payload["source_config"] == source["config"]
    path = tmp_path / "converted.pt"
    torch.save(payload, path)
    restored = model(dtype=dtype)
    load_scorer_checkpoint(restored, path)
    assert_scorers_equal(original, restored, dtype)
    if joint:
        for name, value in original.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("joint", [False, True])
def test_native_checkpoint_roundtrip(tmp_path, kind, joint):
    original, config = initialized_model(kind, torch.bfloat16)
    path = tmp_path / "native.pt"
    save_scorer_checkpoint(path, original, config, backbone=joint, training_config={"objective": "ce"})
    restored = model(dtype=torch.bfloat16)
    payload = load_scorer_checkpoint(restored, path)
    assert payload["scorer_config"] == config
    assert payload["training_config"] == {"objective": "ce"}
    assert_scorers_equal(original, restored, torch.bfloat16)
    if joint:
        for name, value in original.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("kind", KINDS)
def test_missing_config_field_fails_conversion_and_load(tmp_path, kind):
    net, config = initialized_model(kind)
    source = legacy_checkpoint(net)
    payload = convert_checkpoint(source, config)
    path = tmp_path / "incomplete.pt"
    for missing in config:
        incomplete = {key: value for key, value in config.items() if key != missing}
        with pytest.raises((KeyError, ValueError)):
            convert_checkpoint(source, incomplete)
        torch.save({**payload, "scorer_config": incomplete}, path)
        with pytest.raises((KeyError, ValueError)):
            load_scorer_checkpoint(model(), path)


@pytest.mark.parametrize("damage", ["missing", "unexpected", "shape"])
def test_strict_parameter_validation(tmp_path, damage):
    net, config = initialized_model()
    source = legacy_checkpoint(net)
    payload = convert_checkpoint(source, config)
    old_key = "model.layers.1.self_attn.indexer.w_out.weight"
    new_key = "1.magnitude_proj.weight"
    for state, key in ((source["indexer"], old_key), (payload["scorers"], new_key)):
        if damage == "missing":
            state.pop(key)
        elif damage == "unexpected":
            state[key + ".unused"] = torch.zeros(1)
        else:
            state[key] = state[key][:-1]
    with pytest.raises(RuntimeError):
        convert_checkpoint(source, config)
    path = tmp_path / "invalid.pt"
    torch.save(payload, path)
    with pytest.raises(RuntimeError):
        load_scorer_checkpoint(model(), path)


@pytest.mark.parametrize("layers", [1, 3])
@pytest.mark.parametrize("joint", [False, True])
def test_target_layer_count_must_match(tmp_path, layers, joint):
    net, config = initialized_model()
    path = tmp_path / "two_layers.pt"
    torch.save(convert_checkpoint(legacy_checkpoint(net, joint), config), path)
    with pytest.raises(RuntimeError):
        load_scorer_checkpoint(model(layers=layers), path)


@pytest.mark.parametrize("layer_ids", [[], [1], [0, 2]])
def test_legacy_layers_must_be_contiguous(layer_ids):
    net, config = initialized_model()
    source = legacy_checkpoint(net)
    first = {key: value for key, value in source["indexer"].items() if ".layers.0." in key}
    source["indexer"] = {
        key.replace(".layers.0.", f".layers.{layer}."): value for layer in layer_ids for key, value in first.items()
    }
    with pytest.raises(ValueError, match="contiguous"):
        convert_checkpoint(source, config)


def test_legacy_prefix_is_not_guessed():
    net, config = initialized_model()
    source = legacy_checkpoint(net)
    source["indexer"] = {"module." + key: value for key, value in source["indexer"].items()}
    with pytest.raises(ValueError, match="Unsupported legacy scorer key"):
        convert_checkpoint(source, config)


def test_legacy_parameter_alias_collision_fails():
    net, config = initialized_model()
    source = legacy_checkpoint(net)
    source["indexer"]["model.layers.0.self_attn.indexer.input_proj.weight"] = source["indexer"][
        "model.layers.0.self_attn.indexer.w_in.weight"
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        convert_checkpoint(source, config)


def test_joint_checkpoint_requires_full_backbone(tmp_path):
    net, config = initialized_model()
    source = legacy_checkpoint(net, joint=True)
    with pytest.raises(ValueError, match="full model"):
        convert_checkpoint({key: value for key, value in source.items() if key != "model"}, config)
    source["model"].pop("model.embed_tokens.weight")
    path = tmp_path / "partial_backbone.pt"
    torch.save(convert_checkpoint(source, config), path)
    with pytest.raises(RuntimeError, match="model.embed_tokens.weight"):
        load_scorer_checkpoint(model(), path)


def test_gate_scale_upcast_keeps_old_bfloat16_value(tmp_path):
    net, config = initialized_model(dtype=torch.bfloat16)
    source = legacy_checkpoint(net, joint=True)
    for state in (source["indexer"], source["model"]):
        for key, value in state.items():
            if key.endswith(".gate_scale"):
                state[key] = value.bfloat16()
    payload = convert_checkpoint(source, config)
    path = tmp_path / "joint_bfloat16_gate.pt"
    torch.save(payload, path)
    restored = model(dtype=torch.bfloat16)
    load_scorer_checkpoint(restored, path)
    for layer, module in enumerate(restored.model.layers):
        gate = module.self_attn.retention_scorer.gate_scale
        assert gate.dtype == torch.float32
        assert gate.shape == (1,)
        torch.testing.assert_close(
            gate,
            source["indexer"][f"model.layers.{layer}.self_attn.indexer.gate_scale"].float().reshape(1),
            rtol=0,
            atol=0,
        )


def test_recurrent_fixed_gate_conversion(tmp_path):
    net, config = initialized_model("rnn", gate_mode="fixed")
    path = tmp_path / "fixed_recurrence.pt"
    torch.save(convert_checkpoint(legacy_checkpoint(net), config), path)
    restored = model()
    load_scorer_checkpoint(restored, path)
    assert_scorers_equal(net, restored, torch.float32)


def test_save_expands_actual_configuration(tmp_path):
    net = model()
    partial = {"kind": "mlp", "hidden_size": 32, "n_heads": 2, "mid_dim": 16}
    attach_scorers(net, partial)
    path = tmp_path / "explicit.pt"
    save_scorer_checkpoint(path, net, partial)
    payload = load_scorer_checkpoint(model(), path)
    assert payload["scorer_config"]["age_scale"] == net.model.layers[0].self_attn.retention_scorer.age_scale
    assert payload["scorer_config"]["decay"] is False
    assert "rope_dim" not in payload["scorer_config"]
    with pytest.raises(ValueError, match="differs"):
        save_scorer_checkpoint(path, net, partial | {"pos_slope": 0.5})


def test_linear_configuration_requires_linear_width():
    net, config = initialized_model()
    with pytest.raises(ValueError, match="mid_dim=0"):
        convert_checkpoint(legacy_checkpoint(net), config | {"kind": "linear"})


def test_new_checkpoint_preserves_training_state(tmp_path):
    net, config = initialized_model()
    parameters = list(net.model.layers[0].self_attn.retention_scorer.parameters())
    optimizer = torch.optim.AdamW(parameters)
    sum(parameter.sum() for parameter in parameters).backward()
    optimizer.step()
    path = tmp_path / "step.pt"
    save_scorer_checkpoint(path, net, config, step=7, optimizer=optimizer, schedule={"step": 7})
    payload = load_scorer_checkpoint(model(), path)
    assert payload["step"] == 7
    assert payload["schedule"]["step"] == 7
    for index, state in optimizer.state_dict()["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(payload["optimizer"]["state"][index][key], value, rtol=0, atol=0)


def test_native_joint_save_removes_ffn_wrapper(tmp_path):
    net, config = initialized_model()
    expected = net.state_dict()
    for layer in net.model.layers:
        wrapper = nn.Module()
        wrapper.inner = layer.mlp
        layer.mlp = wrapper
    path = tmp_path / "joint.pt"
    save_scorer_checkpoint(path, net, config, backbone=True)
    restored = model()
    load_scorer_checkpoint(restored, path)
    for name, value in expected.items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
