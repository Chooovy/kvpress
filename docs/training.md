# Training

Training supports Qwen3 on Linux with CUDA. Install the pinned runtime and training dependencies from the repository root:

```bash
pip install -e '.[indexmem,training]'
```

## Prepare documents

The loader reads fixed-length rows from `data/longmino_tokenized/<subset>/*.npy`. Each array has a JSON sidecar containing the corresponding `doc_ids`. Raw LongMino shards contain JSON lines with `id`, `text`, and optional `metadata.len_cl100k_base` fields.

```bash
python -m scripts.pretokenize_longmino \
  --model Qwen/Qwen3-8B \
  --data-root data/longmino_raw \
  --out data/longmino_tokenized \
  --subsets 2e15 2e16 synth_cwe synth_rex \
  --sequence-length 65536 --minimum-tokens 65536 --workers 8
```

Set the model, dataset, and output paths in `configs/train.json`. The default run uses reverse KL, an MLP with width 256, learned retention rates, gate mass 256, four sink tokens, and a window of 128 tokens. It trains for 300 steps at 8K and 300 steps at 16K, with eight documents per optimizer step. `seed` controls document sampling; `model_seed` controls scorer initialization and is shared across ranks.

## Train the scorer

```bash
torchrun --standalone --nproc-per-node=8 -m scripts.train --config configs/train.json
```

`ffn_sp_size=8` makes those eight ranks cooperate on each document. For one GPU, set `ffn_sp_size=1`; the loop uses gradient accumulation to maintain `global_batch_size`. Sequence-parallel size must divide the process count, and global batch size must divide evenly across the resulting data replicas and per-rank batch size. Use `--max-steps 2` for a short run without changing the learning-rate schedule.

The output directory contains `metrics.jsonl`, periodic checkpoints, and `final.pt`. Checkpoints contain scorer weights, the explicit scorer configuration, optimizer state, and scheduler state. `--init-from` loads scorer weights; `--resume` restores a native checkpoint's optimizer, scheduler, and document-stream position. Keep the training configuration and process topology unchanged when resuming.

```bash
torchrun --standalone --nproc-per-node=8 -m scripts.train \
  --config configs/train.json --resume checkpoints/indexmem/step100.pt
```

Legacy checkpoints must first use the standalone converter described in [checkpoints.md](checkpoints.md).

## Optional teacher cache

With `teacher_cache=null`, the frozen model computes the dense teacher during training. To precompute it, run:

```bash
python -m scripts.precompute_hdense \
  --config configs/train.json --out data/teacher --data-world-size 1
```

Then set `teacher_cache` to `data/teacher`. `--data-world-size` is the number of data replicas: process count divided by `ffn_sp_size`. Preparation follows the configured loader and writes the documents that the run consumes. Use `--shard-index` and `--shard-count` to split cache preparation across processes. Caches require `take_from="head"`; uncached training also supports `"random"` windows. Existing teacher `.npy`/JSON and LongCE `.npz` caches use the same file layouts.

## Paper ablations

Alternate objective implementations live in `kvpress/indexmem/ablations/objectives.py`. Set `objective` to `ce`, `forward_kl`, or `longce`; reverse KL with `ce_weight=0.1` computes `0.9 * KL + 0.1 * CE`. For LongCE, prepare weights and set `longce_cache` to their directory:

```bash
python -m scripts.precompute_longce_weights \
  --config configs/train.json --out data/longce --data-world-size 1
```

The default LongCE truncation, window, and weight ceiling are 1024, 1024, and 5. Scorer variants use the same loop and select `scorer.kind` from `linear`, `conv`, `rnn`, `prefix`, and `kvzip`. Their configuration fields are in `kvpress/indexmem/ablations/`; the linear scorer uses `mid_dim=0`. The no-protected-region CE ablation uses `sink_size=0` and `window_size=0`.

Joint CE training lives in `kvpress/indexmem/ablations/joint.py`. It updates attention projections, layer norms, and the scorer, freezes the scorer's gate scale and the remaining backbone, and writes a complete adapted-model checkpoint. The example configuration preserves the original joint recipe: Qwen3-8B, the `2e16` and `2e17` subsets, 300 steps at 8K, eight documents per step, head windows, and data seed 1000. The attention learning rate peaks at `2e-5`; the scorer uses five times that rate, and both decay to one percent of their peak. It starts from the original LongCE scorer checkpoint after conversion. Set the tokenized path to those subsets, and use a checkpoint for the same Qwen3 model and gate geometry:

```bash
torchrun --standalone --nproc-per-node=8 -m scripts.train_joint \
  --config configs/train_joint.json --init-from checkpoints/longce/final.pt
```

This configuration preserves the source training recipe; it does not reproduce the manuscript's joint table by itself. Joint RKL is not implemented in this branch. See [paper_protocols.md](paper_protocols.md) for the recorded experiment scope and unresolved provenance differences.
