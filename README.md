# IndexMem++

IndexMem++ learns a per-token retention score for each KV head. At inference, it allocates cache space by retained attention mass and represents evicted entries with centroid memory (CMP).

This branch contains the main method and the implemented paper ablations. It builds on [NVIDIA KVPress](https://github.com/NVIDIA/kvpress) and retains its general-purpose presses and text preprocessing. IndexMem++ supports Qwen3-0.6B, Qwen3-4B, and Qwen3-8B.

## Install

Use Python 3.10 or newer. Inference and training require an NVIDIA GPU. The reference stack is PyTorch 2.8.0 with CUDA 12.8, Transformers 4.57.3, Triton 3.4.0, and FlashAttention 2.8.3.

```bash
python -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'flash-attn==2.8.3' --no-build-isolation
python -m pip install -e '.[indexmem,eval]'
```

## Generate

Convert an existing scorer checkpoint first; see [checkpoint conversion](docs/checkpoints.md).

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvpress.indexmem import IndexMemConfig, IndexMemTextGenerationPipeline

model_id = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
).cuda().eval()
pipeline = IndexMemTextGenerationPipeline(
    model=model,
    tokenizer=AutoTokenizer.from_pretrained(model_id),
    scorer_checkpoint="checkpoints/indexmem.pt",
    config=IndexMemConfig(cache_budget=2048),
)
result = pipeline("Your document", question="Your question", max_new_tokens=256)
print(result)
```

The default pipeline runs sparse prefill followed by physical KV eviction. Each forward reads the existing cache and CMP; `finish_step()` then evicts and updates memory. The average per-head budget includes 64 CMP slots. Four sinks and a 128-token local window are protected; mass allocation uses an evictable-head floor of 512, with protected regions accounted for separately.

`IndexMemConfig(inference_mode="mask")` selects the historical quality-evaluation path, which retains dense KV storage. Masking and physical eviction have different ranking reference rows and cache occupancy. Their outputs are not interchangeable.

## Train and evaluate

Edit the model, tokenized-data, and output paths in `configs/train.json`, then run:

```bash
torchrun --standalone --nproc_per_node=8 scripts/train.py --config configs/train.json
python evaluation/evaluate_indexmem.py --config_file configs/evaluation/ruler.yaml --scorer_checkpoint checkpoints/indexmem.pt
```

The main recipe trains only the scorer with reverse KL, MLP width 256, learned retention rates, and gate mass 256. It uses 300 steps at 8K followed by 300 steps at 16K, global batch 8, and the LongMino `2e15`, `2e16`, `synth_cwe`, and `synth_rex` mixture. Data preparation and the optional training dependencies are described in [training](docs/training.md).

Evaluation presets cover RULER, LongBench, Math500, and AIME25. Read [paper protocols and gaps](docs/paper_protocols.md) before comparing results with the manuscript. The Math500/AIME25 presets require the appropriate offline head-budget table.

## Read the implementation

| Directory | Responsibility |
|---|---|
| `kvpress/indexmem/` | Retention scorer, explicit checkpoints, configuration, public pipeline |
| `training/` | Data, shared training loop, objectives, optimization, sequence parallelism |
| `inference/` | Prefill, head budgets, paged KV, centroid memory, generation |
| `kernels/` | Gated training attention, sparse prefill, sparse/paged reads |
| `ablations/` | Alternative scorers, offline allocation, compensation diagnostics |

| Code | Paper meaning |
|---|---|
| `magnitude_proj` | Content-dependent retention magnitude |
| `log_retention_rate` | Logarithm of the learned retention rate |
| `age_scale` | Fixed token-age scale |
| `score_intercept` | Magnitude minus the key-position retention term |
| `gate_mass` | Total training gate mass over evictable entries |
| `cache_budget` | Mean total inference capacity per KV head, including CMP |
| `cluster_counts` | Number of entries represented by each centroid |

Available ablations are listed in [training](docs/training.md). The old `gqa_indexer` APIs and checkpoint-loading heuristics are removed.

For a uniform-budget, CMP-disabled inference ablation:

```bash
python evaluation/evaluate_indexmem.py --config_file configs/evaluation/ruler.yaml --scorer_checkpoint checkpoints/indexmem.pt --head_budget uniform --cmp_slots 0
```

Fit an offline allocation table, or run the standalone compensation diagnostic:

```bash
python -m scripts.fit_head_budgets --model Qwen/Qwen3-8B --scorer-checkpoint checkpoints/indexmem.pt --tokenized data/longmino_tokenized --output head_budgets.pt
python -m kvpress.indexmem.ablations.compensation --tensors evicted_kv.pt --output compensation.json
```

The diagnostic input contains `queries`, `evicted_keys`, and `evicted_values` tensors shaped `[heads, tokens, head_dim]`; query and evicted-token counts may differ. It evaluates Linear/TTT and centroid reconstruction without training the language model.

## Validate and benchmark

```bash
python -m pytest tests/indexmem
CUDA_VISIBLE_DEVICES=0 python -m scripts.benchmark_decode --model /path/to/Qwen3-8B --scorer-checkpoint checkpoints/indexmem.pt --prompts prompts.json --length 8192 --batch 4 --method indexmem --output results/decode.jsonl
```

The benchmark accepts a JSON list of documents containing `token_ids`. It excludes prefill, initial compression, the first output token, and state restoration from timing. Each timed decode step includes scoring, eviction, CMP updates, and greedy token selection. Defaults are three documents, two warmups, five repeats, and 256 steps. Run with `--method full` for Full-KV. Throughput is total batch tokens divided by total measured seconds; errors and OOMs are recorded separately.

CPU test skips do not count as CUDA validation. See [validation](docs/validation.md) for the tested revision and evidence.

## License

Apache-2.0. Original KVPress copyright and attribution notices are preserved. See [LICENSE](LICENSE).
