[![Hugging Face Leaderboard](https://img.shields.io/badge/🤗%20HuggingFace-Leaderboard-orange)](https://huggingface.co/spaces/nvidia/kvpress-leaderboard)

# Evaluation

We support evaluation for all the presses implemented in the library, on a variety of popular benchmarks.

### Quick Start 🚀
> Evaluation requires some additional packages. You can install them with `uv sync --extra eval`

Running evaluation is straightforward! Make sure you are in the `evaluation` directory, then:

1. **Configure your evaluation** - Edit `evaluate_config.yaml` to specify your *method*, *press*, and *dataset*
2. **Run the evaluation** - Execute the script: ```python evaluate.py```

The script will read from `evaluate_config.yaml` and run inference accordingly. 
If you want, you can override the settings via command line, for instance:

```bash
python evaluate.py --dataset loogle --data_dir shortdep_qa --model meta-llama/Meta-Llama-3.1-8B-Instruct --press_name expected_attention --compression_ratio 0.5
```

or pass a custom configuration file:

```bash
python evaluate.py --config_file <your_config.yaml>
```

💡 Results (predictions & metrics) are automatically saved to the `output_dir` directory .


### Configuration 

Customize your evaluation by editing `evaluate_config.yaml`. This allows you to flexibly configure a variety of settings, like the `fraction` of dataset to use (for quick testing) and the model arguments (e.g. for scaling RoPE). For complete parameter details, see the `evaluation_config.yaml`

💡 Set `query_aware: true` to include the question in the context during compression. This enables query-aware compression as used in methods like SnapKV and FinchPress.


### Available Presses and Datasets 
We support evaluation with all the presses implemented in the library (and possible combinations). 

- All implemented presses are listed in the `PRESS_REGISTRY` variable in `evaluate_registry.py`.
- All implemented dataset are listed in `DATASET_REGISTRY` variable in `evaluate_registry.py`. 

At the moment, we support the following standard popular benchmarks:

- [Loogle](benchmarks/loogle/README.md) ([hf link](https://huggingface.co/datasets/simonjegou/loogle))
- [RULER](benchmarks/ruler/README.md) ([hf link](https://huggingface.co/datasets/simonjegou/ruler))
- [Zero Scrolls](benchmarks/zero_scrolls/README.md) ([hf link](https://huggingface.co/datasets/simonjegou/zero_scrolls))
- [Infinitebench](benchmarks/infinite_bench/README.md) ([hf link](https://huggingface.co/datasets/MaxJeblick/InfiniteBench))
- [longbench](benchmarks/longbench/README.md)([hf link](https://huggingface.co/datasets/Xnhyacinth/LongBench))
- [longbench-v2](benchmarks/longbenchv2/README.md)([hf link](https://huggingface.co/datasets/simonjegou/LongBench-v2))
- [Needle in a Haystack](benchmarks/needle_in_haystack/README.md)([hf link][Paul Graham's essays](https://huggingface.co/datasets/alessiodevoto/paul_graham_essays))

Each dataset directory is structured as follows:

```bash
$dataset
├── README.md
├── calculate_metrics.py
├── create_huggingface_dataset.py
```

Where:
- `create_huggingface_dataset.py` is a script that generates the Hugging Face dataset from the original dataset. Each dataset is associated with a set of parquet files with the following structure:
  - `context`: ... 
  - `question`: ...
  - `answer_prefix`: ...
  - `answer`:  ...
  - `max_new_tokens`:  ...
- `calculate_metrics.py` is a script that calculates the metrics based on the output of `evaluate.py`


### Multi GPU Evaluation
Use the provided `evaluate.sh` script to run multiple presses simultaneously across different GPUs with varying compression ratios.

Two axes of parallelism are available, and they are for different situations:

- **Over configurations** (`evaluate.sh`, `evaluate_sparse_scalar.sh`, `evaluate_dense_baseline.sh`): one (length, press, top-k) cell per GPU. Use it when the sweep has at least as many cells as GPUs.
- **Over rows** (`evaluate_sharded.py` for eviction presses, `evaluate_sparse_sharded.py` for the GQA indexer): one configuration split across every GPU, scored once over the union. Use it to get a single number quickly, or for a sweep with fewer cells than GPUs.

Row sharding defaults to splitting by **context**, because a context's questions share one prefill. Pass `--shard_by row` for datasets where every row shares a single context — `math500` and `aime25` put the whole problem in `question` and leave `context` a single space, so context sharding would put the entire dataset on shard 0 and idle the other GPUs.

### Reasoning and long-context multiple choice

`evaluate_reasoning_longbench.sh` runs `math500`, `aime25` and `longbench-v2` for both the dense
baseline and the sparse GQA indexer, then `summarize_reasoning_longbench.py` tables the results:

```bash
bash evaluate_reasoning_longbench.sh                # dense arm, then the indexer sweep
ARMS=sparse TASKS=math500 bash evaluate_reasoning_longbench.sh   # one arm, one task
python summarize_reasoning_longbench.py --output_dir ./results_reasoning_longbench
```

It exists as its own script because three things about these datasets differ from RULER in ways that
corrupt a number rather than raising:

- **`longbench-v2` needs `--max_context_length` pinned.** Its contexts reach 2.3M tokens (median 127K), the pipeline defaults the cap to `tokenizer.model_max_length` (131072 for Qwen3), and Qwen3-8B has 40960 RoPE positions — so the default silently feeds the model several times its positional range.
- **`math500`/`aime25` need `--shard_by row`**, per the note above.
- **They are generative** (`max_new_tokens` 4096 and 32000, against RULER's ~50), so decode dominates and the reported `answered` column matters: a missing `\boxed{}` means a truncated or looping trace, which is a different failure from a wrong answer.

`aime25` runs at `fraction 1.0` because it is 30 problems in total; at 0.1 it would be three. Its 95%
Wilson interval is roughly ±18 points, so read it as a smoke test rather than a ranking.

### Leaderboard 🥇
After evaluating your model, you can easily submit it to the [KVPress Leaderboard](https://huggingface.co/spaces/nvidia/kvpress-leaderboard) on Hugging Face! Just copy the output directory in the huggingface space, and your method will soon be displayed in the leaderboard.