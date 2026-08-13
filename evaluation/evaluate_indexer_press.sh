#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Evaluate the trained GQA indexer as an EVICTION PRESS: the indexer scores keys, the lowest-scoring
# ones are dropped from the cache, and every later query sees the same reduced cache.
#
#   bash evaluate_indexer_press.sh
#   MODEL=/path CKPT=/path/step600.pt LENGTHS="4096 8192" RATIOS="0.25 0.5 0.75" bash evaluate_indexer_press.sh
#
# The three evals and what each measures -- do not read one as the other:
#   evaluate_dense_baseline.sh   no compression, the upper bound.
#   evaluate_indexer_press.sh    THIS. Saves cache MEMORY; budget knob is compression_ratio.
#   evaluate_sparse.sh           per-query top-k, nothing dropped. Saves attention FLOPs; knob is topk.
#
# Eviction is the harder setting at matched budget: dropping a key is permanent, so a key one query
# did not need is unavailable to every later query, whereas sparse selection reconsiders every query.
#
# The sweep is (length x compression_ratio), one CONFIGURATION per GPU. evaluate_indexer_press.py has
# no sharding option, so running one identical configuration on N GPUs would evaluate the same rows N
# times for N identical numbers. For RULER the data_dir IS the context length, so LENGTHS sweeps that.
#
# FRACTION/SEED/ATTN are aligned with the other two scripts on purpose -- equal (fraction, seed) means
# all three score the IDENTICAL rows, which is what makes the comparison a comparison.
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="1"

DATASET="${DATASET:-ruler}"
MODEL="${MODEL:-/apdcephfs_zw31/share_303843174/user/marcushaogu/models/Qwen3-8B}"
CKPT="${CKPT:-/apdcephfs_zw31/share_303843174/user/marcushaogu/models/Qwen-3-8B-gqa_indexer/stage1/step600.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-./results_indexer_press_max_reduce}"

# Press protection, mirroring the sparse eval's reserved slots so the budgets protect the same tokens.
N_SINK="${N_SINK:-4}"
N_LOCAL="${N_LOCAL:-64}"
QUERY_REDUCE="${QUERY_REDUCE:-max}"

# Must match the other scripts, or the runs score different subsets.
FRACTION="${FRACTION:-0.1}"
SEED="${SEED:-42}"
# sdpa by default: an unverified flash-attn build silently returns wrong logits and every task scores
# ~0. See check_attention_backend.py.
ATTN="${ATTN:-sdpa}"

read -r -a RATIOS <<< "${RATIOS:-0.5}"
read -r -a LENGTHS <<< "${LENGTHS:-8192}"

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

JOBS=()
for length in "${LENGTHS[@]}"; do
  for ratio in "${RATIOS[@]}"; do
    JOBS+=("$length:$ratio")
  done
done
echo "${#JOBS[@]} job(s) over $NGPU GPU(s): lengths=${LENGTHS[*]} ratios=${RATIOS[*]}"

for ((g = 0; g < NGPU; g++)); do
  (
    for ((j = g; j < ${#JOBS[@]}; j += NGPU)); do
      length="${JOBS[$j]%%:*}"
      ratio="${JOBS[$j]##*:}"
      EXTRA=()
      [[ -n "$length" ]] && EXTRA+=(--data_dir "$length")
      echo "  compression_ratio=$ratio @ ${length:-default} on cuda:$g"
      python evaluate_indexer_press.py \
        --dataset "$DATASET" --model "$MODEL" --indexer_ckpt "$CKPT" \
        --compression_ratio "$ratio" \
        --n_sink "$N_SINK" --n_local "$N_LOCAL" --query_reduce "$QUERY_REDUCE" \
        --attn_implementation "$ATTN" \
        --fraction "$FRACTION" --seed "$SEED" --output_dir "$OUTPUT_DIR" \
        --device "cuda:$g" \
        "${EXTRA[@]}"
    done
  ) &
done

wait
echo "All indexer-press evaluations completed."
