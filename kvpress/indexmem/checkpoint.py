from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import torch
from torch import nn

from .scorer import RetentionScorer, RetentionScorerConfig


def scorer_class(kind):
    if kind in ("mlp", "linear"):
        return RetentionScorer, RetentionScorerConfig
    from importlib import import_module

    module, name = {
        "conv": ("conv", "ConvRetentionScorer"),
        "rnn": ("recurrent", "RecurrentRetentionScorer"),
        "prefix": ("prefix", "PrefixRetentionScorer"),
        "kvzip": ("kvzip", "KVzipRetentionScorer"),
    }[kind]
    module = import_module(f"kvpress.indexmem.ablations.{module}")
    return getattr(module, name), getattr(module, name + "Config")


def make_scorer(scorer_config):
    kind = scorer_config["kind"]
    cls, config_cls = scorer_class(kind)
    fields = {k: v for k, v in scorer_config.items() if k != "kind"}
    return cls(config_cls(**fields))


def _validate_scorer_config(scorer_config):
    _, config_cls = scorer_class(scorer_config["kind"])
    expected = {field.name for field in fields(config_cls) if field.init} | {"kind"}
    missing = expected - scorer_config.keys()
    unexpected = scorer_config.keys() - expected
    if missing or unexpected:
        raise ValueError(f"Incomplete scorer configuration: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    if scorer_config["kind"] == "linear" and scorer_config["mid_dim"] != 0:
        raise ValueError("The linear scorer requires mid_dim=0")


def attach_scorers(model, scorer_config):
    scorers = []
    for layer in model.model.layers:
        attention = layer.self_attn
        scorer = make_scorer(scorer_config).to(
            device=attention.q_proj.weight.device,
            dtype=attention.q_proj.weight.dtype,
        )
        if scorer.gate_scale is not None:
            scorer.gate_scale.data = scorer.gate_scale.data.float()
        attention.retention_scorer = scorer
        scorers.append(scorer)
    return nn.ModuleList(scorers)


def scorer_modules(model):
    return nn.ModuleList([layer.self_attn.retention_scorer for layer in model.model.layers])


def scorer_state_dict(model):
    return {name: value.detach().cpu() for name, value in scorer_modules(model).state_dict().items()}


def load_scorer_checkpoint(model, path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _validate_scorer_config(payload["scorer_config"])
    scorers = attach_scorers(model, payload["scorer_config"])
    if "backbone" in payload:
        model.load_state_dict(payload["backbone"], strict=True)
    scorers.load_state_dict(payload["scorers"], strict=True)
    return payload


def save_scorer_checkpoint(
    path,
    model,
    scorer_config,
    *,
    step=0,
    optimizer=None,
    schedule=None,
    backbone=False,
    training_config=None,
):
    scorers = scorer_modules(model)
    config = {
        "kind": scorer_config["kind"],
        **{field.name: getattr(scorers[0].config, field.name) for field in fields(scorers[0].config) if field.init},
    }
    _validate_scorer_config(config)
    if any(key not in config or config[key] != value for key, value in scorer_config.items()):
        raise ValueError("The supplied scorer configuration differs from the model")
    payload = {
        "scorer_config": config,
        "scorers": scorer_state_dict(model),
        "step": step,
        "training_config": training_config,
    }
    if backbone:
        payload["backbone"] = {
            name.replace(".mlp.inner.", ".mlp."): value.detach().cpu() for name, value in model.state_dict().items()
        }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if schedule is not None:
        payload["schedule"] = schedule if isinstance(schedule, dict) else schedule.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
