import pandas as pd
import torch

from kvpress.indexmem.evaluation import (
    ROLLOUT_STRIDE,
    IndexMemEvaluationConfig,
    IndexMemEvaluationRunner,
    expand_rollouts,
    score_predictions,
    shard_dataset,
)


def test_context_shards_and_rollouts_preserve_source_rows():
    frame = pd.DataFrame({"context": ["a", "a", "b", "c", "c", "d"], "question": list("abcdef")})
    shards = [expand_rollouts(shard_dataset(frame, shard, 3, "context"), 2) for shard in range(3)]
    combined = pd.concat(shards, verify_integrity=True).sort_index()
    assert len(combined) == len(frame) * 2
    assert (combined.index % ROLLOUT_STRIDE).value_counts().eq(2).all()
    for context in frame["context"].unique():
        assert sum(context in shard["context"].values for shard in shards) == 1


def test_math_rows_split_even_when_they_share_one_context():
    frame = pd.DataFrame({"context": [" "] * 8, "question": [str(i) for i in range(8)]})
    shards = [shard_dataset(frame, shard, 3, "row") for shard in range(3)]
    assert [len(shard) for shard in shards] == [3, 3, 2]
    pd.testing.assert_frame_equal(pd.concat(shards, verify_integrity=True).sort_index(), frame)


def test_evaluation_passes_sampling_prompt_and_generation_limits(tmp_path):
    calls = []

    class Pipeline:
        def __call__(self, context, **kwargs):
            calls.append((context, kwargs, torch.initial_seed()))
            return {"answers": ["answer"] * len(kwargs["questions"])}

    config = IndexMemEvaluationConfig(
        scorer_checkpoint="test.pt",
        dataset="math500",
        inference_mode="mask",
        enable_thinking=True,
        rollouts=2,
        do_sample=True,
        max_new_tokens=32768,
        max_context_length=40960,
        seed=42,
    )
    runner = IndexMemEvaluationRunner(config)
    runner.pipeline = Pipeline()
    runner.df = pd.DataFrame(
        {
            "context": [" "],
            "question": ["solve"],
            "answer_prefix": [""],
            "max_new_tokens": [10],
        }
    )
    runner._run_inference(tmp_path)
    assert len(runner.df) == 2
    assert [call[2] for call in calls] == [42, 1042]
    assert all(call[1]["enable_thinking"] and call[1]["max_new_tokens"] == 32768 for call in calls)
    assert runner.pipeline.sampling == {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


def test_predictions_are_saved_before_scorer_normalization(tmp_path, monkeypatch):
    from kvpress.indexmem.evaluation import SCORER_REGISTRY

    frame = pd.DataFrame({"context": ["x"], "predicted_answer": ["raw\nanswer"]})

    def score(frame):
        frame["predicted_answer"] = "normalized"
        return {"score": 1}

    monkeypatch.setitem(SCORER_REGISTRY, "ruler", score)
    score_predictions(frame, "ruler", tmp_path)
    saved = pd.read_csv(tmp_path / "predictions.csv")
    assert saved["predicted_answer"].tolist() == ["raw\nanswer"]


def test_sharded_runner_launches_absolute_entries_and_combines_rows(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    import yaml

    import kvpress.indexmem.evaluation as evaluation

    calls = []

    def launch(command, stdout, stderr):
        entry = Path(command[1])
        assert entry.is_absolute() and entry.is_file()
        assert entry.name == "evaluate_indexmem.py"
        config = yaml.safe_load(Path(command[-1]).read_text())
        calls.append(config)
        shard = config["shard_index"]
        pd.DataFrame({"context": [" "], "predicted_answer": [f"answer{shard}"]}, index=[shard]).to_parquet(
            Path(config["results_dir"]) / f"predictions_shard{shard}.parquet"
        )
        return SimpleNamespace(args=command, wait=lambda: 0)

    monkeypatch.setattr(evaluation.subprocess, "Popen", launch)
    monkeypatch.setitem(evaluation.SCORER_REGISTRY, "math500", lambda frame: {"count": len(frame)})
    evaluation.sharded_main(devices=[2, 5], scorer_checkpoint="scorer.pt", dataset="math500", results_dir=str(tmp_path))
    assert [call["device"] for call in calls] == ["cuda:2", "cuda:5"]
    assert [call["shard_index"] for call in calls] == [0, 1]
    assert all(call["num_shards"] == 2 for call in calls)
    frame = pd.read_csv(tmp_path / "predictions.csv")
    assert frame["predicted_answer"].tolist() == ["answer0", "answer1"]


def test_package_math_metrics_without_script_import_paths(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    script = r"""
import json
import sys
from pathlib import Path
import pandas as pd
sys.path = [entry for entry in sys.path if Path(entry).resolve() != Path.cwd() / "evaluation"]
from kvpress.indexmem.evaluation import score_predictions
frame = pd.DataFrame({
    "context": [" "] * 3,
    "predicted_answer": [r"Final: \boxed{\dfrac{1}{4}}", r"\boxed{7}", "unfinished"],
    "answer": [r"\frac{1}{4}", "8", "3"],
})
for dataset in ("math500", "aime25"):
    directory = Path(sys.argv[1]) / dataset
    directory.mkdir()
    metrics = score_predictions(frame.copy(), dataset, directory)
    assert metrics == {"correct": 1, "answered": 2, "accuracy": 1 / 3, "total": 3}
    assert json.loads((directory / "metrics.json").read_text()) == metrics
    assert f"evaluation.benchmarks.{dataset}.calculate_metrics" in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
