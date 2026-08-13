#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Evaluate a trained GQA indexer as sparse attention (no eviction: the full KV cache is kept and
# each query attends to its own top-k keys). Counterpart of evaluation/evaluate.sh, which sweeps
# eviction presses. Run from the evaluation/ directory, like evaluate.sh.
#
#   scripts:  bash evaluate_sparse.sh
#   override: MODEL=/path CKPT=/path/final.pt DATASET=ruler DATA_DIR=4096 bash evaluate_sparse.sh
#
# Sweeps TOPKS (one per GPU) so a topk-vs-quality curve comes out of one launch. force_local/
# force_sink and block_k match the sparse training stage. Compare the numbers against
# evaluate.sh's `no_press` row on the same dataset for the dense upper bound.
set -euo pipefail

DATASET="${DATASET:-ruler}"
DATA_DIR="${DATA_DIR:-8192}"
MODEL="${MODEL:-/apdcephfs_zw31/share_303843174/user/marcushaogu/models/Qwen3-8B}"
CKPT="${CKPT:-/apdcephfs_zw31/share_303843174/user/marcushaogu/models/Qwen-3-8B-gqa_indexer/stage1/step600.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-./results_sparse}"

FORCE_LOCAL="${FORCE_LOCAL:-64}"
FORCE_SINK="${FORCE_SINK:-4}"
BLOCK_K="${BLOCK_K:-64}"
# FRACTION and SEED must match evaluate_dense_baseline.sh, or the sparse run and the dense upper
# bound score different rows. Sampling is df.sample(frac, random_state=seed), so equal values give
# the identical subset on every run and for every topk -- which is what makes the sweep comparable.
FRACTION="${FRACTION:-0.1}"
SEED="${SEED:-42}"

# The sweep is (length x topk), one CONFIGURATION per GPU -- not a split of one dataset and not a
# bigger batch. evaluate_sparse.py has no sharding option, so running one identical configuration
# on 8 GPUs would evaluate the same rows 8 times for 8 identical numbers. For RULER the data_dir IS
# the context length, so LENGTHS sweeps that; DATA_DIR above is the fallback for datasets with no
# length split (set LENGTHS="" to use it).
#
# Every (length, topk) pair is one job, and the pairs are packed onto the GPUs round-robin: with
# more pairs than GPUs, a GPU runs its jobs sequentially rather than oversubscribing.
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

# Build the job list, then hand each GPU its own slice to run in sequence.
JOBS=()
for length in "${LENGTHS[@]}"; do
  for topk in "${TOPKS[@]}"; do
    JOBS+=("$length:$topk")
  done
done
echo "${#JOBS[@]} job(s) over $NGPU GPU(s): lengths=${LENGTHS[*]} topks=${TOPKS[*]}"

for ((g = 0; g < NGPU; g++)); do
  (
    for ((j = g; j < ${#JOBS[@]}; j += NGPU)); do
      length="${JOBS[$j]%%:*}"
      topk="${JOBS[$j]##*:}"
      EXTRA=()
      [[ -n "$length" ]] && EXTRA+=(--data_dir "$length")
      echo "  topk=$topk @ ${length:-default} on cuda:$g"
      python evaluate_sparse.py \
        --dataset "$DATASET" --model "$MODEL" --indexer_ckpt "$CKPT" \
        --topk "$topk" --force_local "$FORCE_LOCAL" --force_sink "$FORCE_SINK" --block_k "$BLOCK_K" \
        --fraction "$FRACTION" --seed "$SEED" --output_dir "$OUTPUT_DIR" \
        --device "cuda:$g" \
        "${EXTRA[@]}"
    done
  ) &
done

wait
echo "All sparse evaluations completed."
