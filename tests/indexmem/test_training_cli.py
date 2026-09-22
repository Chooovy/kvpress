import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA training CLI")
def test_training_cli_resume_preserves_weights_and_data_position(tmp_path):
    torch.manual_seed(11)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=67,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
        )
    )
    model.save_pretrained(tmp_path / "model")
    data = tmp_path / "data" / "tiny"
    data.mkdir(parents=True)
    for shard in range(2):
        np.save(data / f"{shard}.npy", np.random.default_rng(shard).integers(0, 67, (4, 48), dtype=np.uint32))
        (data / f"{shard}.json").write_text(json.dumps({"doc_ids": [f"{shard}:{row}" for row in range(4)]}))
    output = tmp_path / "output"
    config = {
        "model": str(tmp_path / "model"),
        "tokenized": str(data.parent),
        "out": str(output),
        "subsets": ["tiny"],
        "scorer": {"kind": "mlp", "mid_dim": 16, "decay": True, "age_scale": 32.0},
        "schedule": "24:2",
        "gate_mass": 8.0,
        "sink_size": 2,
        "window_size": 4,
        "global_batch_size": 1,
        "workers": 0,
        "liger": False,
        "chunk_size": 8,
        "save_every": 1,
        "log_every": 1,
        "take_from": "random",
    }
    config_path = tmp_path / "train.json"
    config_path.write_text(json.dumps(config))
    environment = {key: value for key, value in os.environ.items() if key not in ("RANK", "WORLD_SIZE", "LOCAL_RANK")}
    command = [sys.executable, "-m", "scripts.train", "--config", str(config_path)]
    subprocess.run(command, env=environment, check=True, capture_output=True, text=True)
    expected = torch.load(output / "final.pt", map_location="cpu", weights_only=True)
    subprocess.run(
        command + ["--resume", str(output / "step1.pt")], env=environment, check=True, capture_output=True, text=True
    )
    resumed = torch.load(output / "final.pt", map_location="cpu", weights_only=True)
    assert resumed["step"] == 2
    assert resumed["schedule"] == expected["schedule"]
    for name, parameter in expected["scorers"].items():
        torch.testing.assert_close(resumed["scorers"][name], parameter, atol=1e-7, rtol=1e-6)
