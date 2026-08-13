#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Full (dense) attention baseline for the sparse-attention eval -- the upper bound that
# evaluation/evaluate_sparse.sh's numbers are read against.
#
#   bash evaluate_dense_baseline.sh
#   MODEL=/path DATASET=ruler LENGTHS="4096 8192 16384" FRACTION=0.1 bash evaluate_dense_baseline.sh
#
# This runs the EXISTING evaluate.py with --press_name no_press: no compression, no indexer, and
# none of the sparse code path. That is deliberate -- a baseline that shares the thing under test
# is not a baseline. (To instead self-check the sparse implementation, run evaluate_sparse.sh with
# a topk larger than the context: selection becomes a no-op and it should reproduce these numbers.)
#
# WHAT THE 8 GPUs DO
# One CONTEXT LENGTH per GPU, not a bigger batch and not a split of one dataset. evaluate.py has
# no sharding option, so running the same (dataset, data_dir, fraction, seed) on 8 GPUs would
# evaluate the identical rows 8 times and produce 8 identical numbers. For RULER the data_dir IS
# the context length, and length is the axis a sparse-attention method actually has to be measured
# on, so the sweep runs 4K/8K/16K/... concurrently instead.
#
# Batching stays at one context at a time, exactly as evaluate.py has always run.
#
# WHY EVERY FIELD IS PASSED EXPLICITLY
# evaluate.py layers ./evaluate_config.yaml on top of its dataclass defaults, and that file pins
# model=Meta-Llama-3.1-8B-Instruct and data_dir=4096. Anything not passed on the command line is
# silently taken from there, which would leave the baseline evaluating a different model than the
# sparse run. So dataset/data_dir/model/fraction/seed are all explicit.
#
# COMPARABILITY WITH THE SPARSE RUN
#   - FRACTION and SEED must match evaluate_sparse.sh, or the two score different rows. Sampling is
#     df.sample(frac, random_state=seed), so equal (fraction, seed) => identical rows every run,
#     for every topk. Defaults here match that script's defaults.
#   - both use greedy decoding and the same chat template / answer_prefix (shared pipeline code).
#   - the sparse script re-prefills per question while this shares one prefill across a context's
#     questions. Same answers -- the KV cache is identical either way -- only speed differs.
set -euo pipefail

cd "$(dirname "$0")"

DATASET="${DATASET:-ruler}"
MODEL="${MODEL:-/apdcephfs_zw31/share_303843174/user/marcushaogu/models/Qwen3-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-./results_dense}"
# Match evaluate_sparse.sh's defaults, or the baseline scores a different subset.
FRACTION="${FRACTION:-0.1}"
SEED="${SEED:-42}"

# Attention kernel for the backbone. Defaults to sdpa rather than letting evaluate.py autodetect
# flash_attention_2, because that autodetection only checks that `import flash_attn` succeeds --
# not that the build matches the installed torch. A mismatched build does not crash; it returns
# wrong logits, and the symptom is exactly "no_press scores 0.0 on every task" with the first
# generated token already garbage. Set ATTN=flash_attention_2 to opt back in once it is verified.
ATTN="${ATTN:-sdpa}"

# One context length per GPU. For RULER these are the dataset's data_dir subdirectories; keep the
# count <= the number of GPUs. For a dataset with no length split, set LENGTHS to a single value
# (or to "" to pass no data_dir at all).
read -r -a LENGTHS <<< "${LENGTHS:-4096 8192 16384 32768}"

num_gpus=$(nvidia-smi --list-gpus | wc -l)
if [[ "${#LENGTHS[@]}" -gt "$num_gpus" ]]; then
  echo "Error: ${#LENGTHS[@]} lengths exceed the $num_gpus GPUs on this box" >&2
  exit 1
fi

echo "dense baseline: $DATASET on $MODEL, lengths=${LENGTHS[*]}, fraction=$FRACTION seed=$SEED"

for i in "${!LENGTHS[@]}"; do
  length="${LENGTHS[$i]}"
  (
    EXTRA=()
    [[ -n "$length" ]] && EXTRA+=(--data_dir "$length")
    echo "  no_press @ ${length:-default} on cuda:$i"
    python evaluate.py \
      --dataset "$DATASET" --model "$MODEL" --press_name no_press \
      --attn_implementation "$ATTN" \
      --fraction "$FRACTION" --seed "$SEED" \
      --output_dir "$OUTPUT_DIR" --device "cuda:$i" \
      "${EXTRA[@]}"
  ) &
done

wait
echo "Dense baseline completed. Results under $OUTPUT_DIR/"
