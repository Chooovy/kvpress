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

For last-query whole-chunk eviction, select `bsa`, `mean_pooling`, or a registered `mean_pooling_*_q_*_k` scoring variant and configure the chunk and protected-tail sizes:

```bash
python evaluate.py --dataset ruler --data_dir 4096 --model meta-llama/Meta-Llama-3.1-8B-Instruct --press_name bsa --compression_ratio 0.5 --chunk_size 64 --protected_window_size 512
```

With `query_aware: true`, choose a protected window large enough to retain the question tokens appended to the compression context. A window of zero disables the explicit protected tail, while any final partial chunk remains retained.

or pass a custom configuration file:

```bash
python evaluate.py --config_file <your_config.yaml>
```

For matched Qwen3-8B 32K/64K runs with static 2x YaRN, reuse the provided
model-loading configuration and change only the RULER dataset length and
`max_context_length`:

```bash
python evaluate.py \
    --config_file configs/qwen3_8b_yarn2_64k.yaml \
    --model Qwen/Qwen3-8B \
    --dataset ruler \
    --data_dir 65536 \
    --max_context_length 65536
```

The same YaRN configuration should also be used for the matched 32K arm, with
both length arguments set to `32768`. Generate RULER inputs with the tested
model tokenizer and ensure the combined input and output budget fits the chosen
length; changing only `max_context_length` does not extend the model's RoPE.

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

### Leaderboard 🥇
After evaluating your model, you can easily submit it to the [KVPress Leaderboard](https://huggingface.co/spaces/nvidia/kvpress-leaderboard) on Hugging Face! Just copy the output directory in the huggingface space, and your method will soon be displayed in the leaderboard.
