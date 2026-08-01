# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
import yaml

EVALUATION_DIR = Path(__file__).parents[1] / "evaluation"
sys.path.insert(0, str(EVALUATION_DIR))

# Evaluation imports every benchmark scorer eagerly. Stub those scorer modules
# during collection so these configuration tests do not require optional metric
# packages such as jieba, rouge, or bert-score.
SCORER_MODULES = (
    "benchmarks.aime25.calculate_metrics",
    "benchmarks.infinite_bench.calculate_metrics",
    "benchmarks.longbench.calculate_metrics",
    "benchmarks.longbenchv2.calculate_metrics",
    "benchmarks.loogle.calculate_metrics",
    "benchmarks.math500.calculate_metrics",
    "benchmarks.needle_in_haystack.calculate_metrics",
    "benchmarks.ruler.calculate_metrics",
    "benchmarks.zero_scrolls.calculate_metrics",
)
ORIGINAL_SCORER_MODULES = {
    module_name: sys.modules.get(module_name) for module_name in SCORER_MODULES
}
for module_name in SCORER_MODULES:
    scorer_module = ModuleType(module_name)
    scorer_module.calculate_metrics = lambda df: {}  # type: ignore[attr-defined]
    scorer_module.calculate_metrics_e = lambda df: {}  # type: ignore[attr-defined]
    sys.modules[module_name] = scorer_module

from evaluate import EvaluationConfig, EvaluationRunner, _load_yaml_config  # noqa: E402
from evaluate_registry import PRESS_REGISTRY  # noqa: E402

from kvpress import BSAPress, KVzipPress, MeanPoolingPress  # noqa: E402

for module_name in SCORER_MODULES:
    original_module = ORIGINAL_SCORER_MODULES[module_name]
    if original_module is None:
        del sys.modules[module_name]
    else:
        sys.modules[module_name] = original_module


def test_kvzip_chunk_registry_and_config_use_chunk_without_protected_window(tmp_path):
    press = PRESS_REGISTRY["kvzip_chunk"]
    assert isinstance(press, KVzipPress)
    assert press.selection_granularity == "chunk"

    config = EvaluationConfig(
        press_name="kvzip_chunk",
        compression_ratio=0.5,
        chunk_size=32,
        protected_window_size=777,
    )
    results_dir = config.get_results_dir(tmp_path)
    assert "chunk32" in results_dir.name
    assert "window777" not in results_dir.name

    config_path = results_dir / "config.yaml"
    config.save_config(config_path)
    saved_config = yaml.safe_load(config_path.read_text())
    assert saved_config["chunk_size"] == 32
    assert "protected_window_size" not in saved_config


@pytest.mark.parametrize(
    "press_name, use_prerope_query, use_prerope_keys",
    (
        ("mean_pooling_pre_q_pre_k", True, True),
        ("mean_pooling_post_q_pre_k", False, True),
        ("mean_pooling_pre_q_post_k", True, False),
    ),
)
def test_mean_pooling_variant_registry_and_zero_window_config(
    tmp_path,
    press_name,
    use_prerope_query,
    use_prerope_keys,
):
    press = PRESS_REGISTRY[press_name]
    assert isinstance(press, MeanPoolingPress)
    assert press.use_prerope_query is use_prerope_query
    assert press.use_prerope_keys is use_prerope_keys

    config = EvaluationConfig(
        press_name=press_name,
        compression_ratio=0.5,
        chunk_size=32,
        protected_window_size=0,
    )
    results_dir = config.get_results_dir(tmp_path)
    assert "chunk32" in results_dir.name
    assert "window0" in results_dir.name

    runner = EvaluationRunner(config)
    runner._setup_press()

    assert runner.press is press
    assert press.compression_ratio == 0.5
    assert press.chunk_size == 32
    assert press.protected_window_size == 0


@pytest.mark.parametrize(
    "press_name, selection_granularity, expected_chunk_size",
    (
        ("kvzip", "token", 64),
        ("kvzip_chunk", "chunk", 96),
    ),
)
def test_setup_press_configures_kvzip_selection(
    monkeypatch,
    press_name,
    selection_granularity,
    expected_chunk_size,
):
    press = KVzipPress(
        selection_granularity=selection_granularity,
        selection_chunk_size=64,
    )
    monkeypatch.setitem(PRESS_REGISTRY, press_name, press)
    runner = EvaluationRunner(
        EvaluationConfig(
            press_name=press_name,
            compression_ratio=0.4,
            chunk_size=96,
        )
    )

    runner._setup_press()

    assert runner.press is press
    assert press.compression_ratio == 0.4
    assert press.selection_chunk_size == expected_chunk_size


def test_dataset_sampling_preserves_source_index_as_sample_id(monkeypatch):
    source_df = pd.DataFrame(
        {
            "context": [f"context-{index}" for index in range(10)],
            "question": [f"question-{index}" for index in range(10)],
        }
    )

    class FakeDataset:
        def to_pandas(self):
            return source_df.copy()

    monkeypatch.setattr("evaluate.load_dataset", lambda *args, **kwargs: FakeDataset())
    config = EvaluationConfig(
        press_name="no_press",
        fraction=0.4,
        seed=17,
    )
    runner = EvaluationRunner(config)

    runner._load_and_prepare_dataset()

    expected = source_df.sample(frac=0.4, random_state=17)
    assert runner.df.index.tolist() == expected.index.tolist()
    assert runner.df["sample_id"].tolist() == expected.index.tolist()
    assert runner.df["context"].tolist() == expected["context"].tolist()


def test_nested_model_kwargs_are_forwarded_and_saved_unchanged(monkeypatch, tmp_path):
    expected_model_kwargs = {
        "attn_implementation": "sdpa",
        "max_position_embeddings": 65536,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": 32768,
        },
    }
    model_kwargs = _load_yaml_config(
        EVALUATION_DIR / "configs" / "qwen3_8b_yarn2_64k.yaml"
    )["model_kwargs"]
    assert model_kwargs == expected_model_kwargs
    config = EvaluationConfig(
        model="Qwen/Qwen3-8B",
        device="cpu",
        model_kwargs=model_kwargs,
    )
    captured = {}

    class FakeModel:
        def eval(self):
            return self

    class FakePipeline:
        model = FakeModel()

    def fake_pipeline(task, **kwargs):
        captured["task"] = task
        captured["kwargs"] = kwargs
        return FakePipeline()

    monkeypatch.setitem(sys.modules, "flash_attn", None)
    monkeypatch.setattr("evaluate.pipeline", fake_pipeline)
    runner = EvaluationRunner(config)
    runner._setup_model_pipeline()

    assert captured["task"] == "kv-press-text-generation"
    assert captured["kwargs"]["model_kwargs"] == model_kwargs

    saved_path = tmp_path / "saved.yaml"
    config.save_config(saved_path)
    assert yaml.safe_load(saved_path.read_text())["model_kwargs"] == model_kwargs


@pytest.mark.parametrize(
    "selection_granularity, masked_chunks",
    (
        ("token", None),
        ("chunk", 1),
    ),
)
def test_kvzip_runtime_masking_stats_are_written_after_pipeline(
    selection_granularity,
    masked_chunks,
):
    press = KVzipPress(selection_granularity=selection_granularity)
    runner = EvaluationRunner(
        EvaluationConfig(
            press_name="kvzip_chunk" if selection_granularity == "chunk" else "kvzip",
            compression_ratio=0.5,
        )
    )
    runner.press = press
    runner.df = pd.DataFrame(
        {
            "context": ["context"],
            "question": [""],
            "answer_prefix": [""],
            "max_new_tokens": [1],
        },
        index=[7],
    )

    def fake_pipeline(*args, **kwargs):
        press.last_total_kv_slots = 128
        press.last_masked_kv_slots = 64
        press.last_actual_masked_slot_ratio = 0.5
        press.last_masked_chunks = masked_chunks
        press.last_masked_slots_per_layer = (16, 48)
        return {"answers": ["answer"]}

    runner.pipeline = fake_pipeline
    runner._run_inference()

    assert runner.df.loc[7, "predicted_answer"] == "answer"
    assert runner.df.loc[7, "total_kv_slots"] == 128
    assert runner.df.loc[7, "masked_kv_slots"] == 64
    assert runner.df.loc[7, "actual_masked_slot_ratio"] == pytest.approx(0.5)
    assert json.loads(runner.df.loc[7, "masked_slots_per_layer"]) == [16, 48]
    if masked_chunks is None:
        assert pd.isna(runner.df.loc[7, "masked_chunks"])
    else:
        assert runner.df.loc[7, "masked_chunks"] == masked_chunks
    assert "actual_compression_ratio" not in runner.df.columns


def test_chunk_scorer_runtime_compression_stats_are_unchanged():
    press = BSAPress(compression_ratio=0.5)
    runner = EvaluationRunner(
        EvaluationConfig(
            press_name="bsa",
            compression_ratio=0.5,
        )
    )
    runner.press = press
    runner.df = pd.DataFrame(
        {
            "context": ["context"],
            "question": [""],
            "answer_prefix": [""],
            "max_new_tokens": [1],
        }
    )

    def fake_pipeline(*args, **kwargs):
        press.last_input_tokens = 1024
        press.last_kept_tokens = 512
        press.last_actual_compression_ratio = 0.5
        press.last_protected_tokens = 512
        press.last_kept_remote_chunks = 0
        return {"answers": ["answer"]}

    runner.pipeline = fake_pipeline
    runner._run_inference()

    assert runner.df.loc[0, "actual_compression_ratio"] == pytest.approx(0.5)
    assert runner.df.loc[0, "input_tokens"] == 1024
    assert runner.df.loc[0, "kept_tokens"] == 512
    assert "actual_masked_slot_ratio" not in runner.df.columns
