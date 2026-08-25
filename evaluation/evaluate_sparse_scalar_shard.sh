#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Evaluate a trained SCALAR (query-independent) GQA indexer as sparse attention, with each
# configuration DATA-PARALLEL across all GPUs. The 8-GPU counterpart of evaluate_sparse_scalar.sh.
#
#   bash evaluate_sparse_scalar_shard.sh
#   MODEL=/path CKPT=/path/final.pt LENGTHS=8192 TOPKS=2048 bash evaluate_sparse_scalar_shard.sh
#
# HOW THIS DIFFERS FROM evaluate_sparse_scalar.sh
# That script parallelizes over CONFIGURATIONS: one (length, topk) pair per GPU, because
# evaluate_sparse.py had no sharding option and running one configuration on 8 GPUs would have
# evaluated the same rows 8 times for 8 identical numbers. The cost is that a single configuration
# runs at one GPU's throughput -- and one configuration is the common case when you want one number.
#
# This script runs the configurations SEQUENTIALLY, each sharded over every GPU:
#   for each (length, topk):  8 shards x 1/8 of the contexts  ->  concatenate  ->  score once
# So a single (length, topk) finishes ~NGPU times faster, and the metric is identical to the
# unsharded run's -- the shards partition the same sampled rows and are scored as one union
# (evaluate_sparse_sharded.py). Rows are split by CONTEXT so a context's questions share one
# prefill instead of being re-prefilled in several shards.
#
# Use evaluate_sparse_scalar.sh when the sweep has at least as many (length, topk) pairs as GPUs and
# you want the whole grid; use this one to get a single configuration's number quickly, or for a
# sweep with fewer pairs than GPUs (where the other script leaves GPUs idle).
#
# Everything else -- the checkpoint-driven scorer/geometry detection, FRACTION/SEED comparability
# with evaluate_dense_baseline.sh, force_local/force_sink/block_k, tf32 precision -- matches
# evaluate_sparse_scalar.sh exactly, so numbers from the two are directly comparable.
set -euo pipefail

cd "$(dirname "$0")"

DATASET="${DATASET:-ruler}"
DATA_DIR="${DATA_DIR:-8192}"
MODEL="${MODEL:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B}"
CKPT="${CKPT:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_exact_k_M128/stage1/step400.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-./results_sparse_exact_k_M128}"

FORCE_LOCAL="${FORCE_LOCAL:-64}"
FORCE_SINK="${FORCE_SINK:-4}"
BLOCK_K="${BLOCK_K:-64}"
# tl.dot precision. tf32 because q/k/v are the model's own bf16, and every bf16 value is exactly
# representable in tf32 -- the kernel matches the fp32 reference to the same 7.5e-3 that storing
# the output in bf16 costs regardless. ieee forgoes tensor cores entirely: measured 67.0 s vs
# 9.4 s per 8K prefill on an H20 for identical error. Set PRECISION=ieee only to reproduce the
# fp32 reference bit for bit.
PRECISION="${PRECISION:-tf32}"
# Normally unset: the scorer and the scalar geometry are read from the checkpoint. Set SCORER to
# override detection for a checkpoint that records no config and whose weight names are ambiguous.
SCORER="${SCORER:-}"
# FRACTION and SEED must match evaluate_dense_baseline.sh, or the sparse run and the dense upper
# bound score different rows. Sampling is df.sample(frac, random_state=seed) and happens BEFORE
# sharding, so equal values give the identical subset for every shard count and every topk --
# which is what makes this script's numbers comparable to the single-GPU script's.
FRACTION="${FRACTION:-0.1}"
SEED="${SEED:-42}"

# Matches scripts/train_gqa_indexer_cross_replay_gy.sh: the shard subprocesses are spawned with
# this interpreter's sys.executable, so setting it here is enough to pin the whole run's env.
PYTHON="${PYTHON:-python}"

# This box reaches huggingface.co only through the proxy; without it load_dataset falls back to the
# local cache and fails for any length not already cached. Harmless when a direct route exists.
export https_proxy="${https_proxy:-http://star-proxy.oa.com:3128}"
export http_proxy="${http_proxy:-http://star-proxy.oa.com:3128}"

read -r -a TOPKS <<< "${TOPKS:-2048}"
read -r -a LENGTHS <<< "${LENGTHS:-$DATA_DIR}"

if [[ ! -f "$CKPT" ]]; then
  echo "indexer checkpoint not found at $CKPT (set CKPT=)" >&2
  exit 1
fi

num_gpus=$(nvidia-smi --list-gpus | wc -l)
NGPU="${NGPU:-$num_gpus}"
if [[ "$NGPU" -gt "$num_gpus" ]]; then
  echo "Error: NGPU=$NGPU exceeds the $num_gpus GPUs on this box" >&2
  exit 1
fi

echo "sharded over $NGPU GPU(s): lengths=${LENGTHS[*]} topks=${TOPKS[*]}"

# Sequential over configurations, data-parallel within each: the opposite of the sibling script.
for length in "${LENGTHS[@]}"; do
  for topk in "${TOPKS[@]}"; do
    EXTRA=()
    [[ -n "$length" ]] && EXTRA+=(--data_dir "$length")
    [[ -n "$SCORER" ]] && EXTRA+=(--scorer "$SCORER")
    echo "=== topk=$topk @ ${length:-default} across $NGPU GPU(s)"
    "$PYTHON" evaluate_sparse_sharded.py \
      --ngpu "$NGPU" \
      --dataset "$DATASET" --model "$MODEL" --indexer_ckpt "$CKPT" \
      --topk "$topk" --force_local "$FORCE_LOCAL" --force_sink "$FORCE_SINK" --block_k "$BLOCK_K" \
      --precision "$PRECISION" \
      --fraction "$FRACTION" --seed "$SEED" --output_dir "$OUTPUT_DIR" \
      "${EXTRA[@]}"
  done
done

echo "All sharded sparse evaluations completed."
