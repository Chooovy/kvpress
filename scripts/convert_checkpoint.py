from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from kvpress.indexmem.checkpoint import _validate_scorer_config, make_scorer

PARAMETER_NAMES = {
    "w_in": "input_proj",
    "w_out": "magnitude_proj",
    "w_decay": "retention_proj",
    "w_cin": "history_input_proj",
    "w_a": "history_proj",
    "a_norm": "history_norm",
    "w_u": "state_input_proj",
    "w_g": "state_gate_proj",
    "w_pq": "prefix_query_proj",
    "w_pk": "prefix_key_proj",
    "w_pv": "prefix_value_proj",
}


def rename_parameter(name):
    return ".".join(PARAMETER_NAMES.get(part, part) for part in name.split("."))


def convert_checkpoint(source, scorer_config):
    _validate_scorer_config(scorer_config)
    source_config = source["config"]
    if (source_config.get("joint") or source_config.get("objective") == "joint_lm_loss") and "model" not in source:
        raise ValueError("Joint checkpoints require the full model state")
    layers = {}
    for name, value in source["indexer"].items():
        match = re.fullmatch(r"model\.layers\.(\d+)\.self_attn\.indexer\.(.+)", name)
        if match is None:
            raise ValueError(f"Unsupported legacy scorer key: {name}")
        layer, parameter = match.groups()
        parameter = rename_parameter(parameter)
        if parameter == "gate_scale":
            value = value.reshape(1).float()
        state = layers.setdefault(int(layer), {})
        if parameter in state:
            raise ValueError(f"Duplicate legacy scorer parameter: {name}")
        state[parameter] = value
    if not layers or sorted(layers) != list(range(len(layers))):
        raise ValueError("Legacy scorer layers must be contiguous and start at layer 0")
    scorers = {}
    for layer in range(len(layers)):
        make_scorer(scorer_config).load_state_dict(layers[layer], strict=True)
        scorers.update({f"{layer}.{name}": value for name, value in layers[layer].items()})
    payload = {
        "scorer_config": scorer_config,
        "scorers": scorers,
        "step": 0,
        "training_config": None,
        "source_config": source_config,
    }
    if "model" in source:
        backbone = {}
        for name, value in source["model"].items():
            name = name.replace(".mlp.inner.", ".mlp.")
            if ".indexer." in name:
                prefix, parameter = name.split(".indexer.")
                name = prefix + ".retention_scorer." + rename_parameter(parameter)
                if parameter == "gate_scale":
                    value = value.reshape(1).float()
            backbone[name] = value
        for name, value in scorers.items():
            layer, parameter = name.split(".", 1)
            backbone[f"model.layers.{layer}.self_attn.retention_scorer.{parameter}"] = value
        payload["backbone"] = backbone
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--scorer-config", type=Path, required=True)
    args = parser.parse_args()
    source = torch.load(args.source, map_location="cpu", weights_only=True)
    config = json.loads(args.scorer_config.read_text())
    payload = convert_checkpoint(source, config)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.destination)


if __name__ == "__main__":
    main()
