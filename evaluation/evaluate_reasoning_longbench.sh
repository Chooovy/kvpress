#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Evaluate the scalar GQA indexer on the two benchmark families RULER does not cover -- short-context
# reasoning (math500, aime25) and long-context multiple choice (longbench-v2) -- against the dense
# no-press upper bound, everything data-parallel over the box's GPUs.
#
#   bash evaluate_reasoning_longbench.sh              # dense baseline, then the indexer sweep
#   ARMS=dense  bash evaluate_reasoning_longbench.sh  # just the baseline
#   ARMS=sparse bash evaluate_reasoning_longbench.sh  # just the indexer
#   TASKS="math500" TOPKS_MATH="2048" bash evaluate_reasoning_longbench.sh   # one cell
#
# The grid, which is the point of the script:
#
#            | fraction | topks       | why
#   math500  | 0.1      | 2048, 4096  | 500 problems; 0.1 -> 50, enough to separate arms
#   aime25    | 1.0      | 2048, 4096  | only 30 problems total, so 0.1 would be THREE of them
#   lbv2     | 0.1      | 1024, 2048  | 503 rows; the long-context arm, so a tighter budget bites
#
# aime25 deliberately runs at fraction 1.0. At 0.1 it is 3 problems, where one flipped answer moves
# the metric by 33 points -- the number would be noise wearing a decimal point. The other two keep
# fraction 0.1 to match evaluate_dense_baseline.sh and the RULER runs, and dense/sparse share
# FRACTION and SEED so both arms score the identical rows.
#
# WHY THIS SCRIPT EXISTS RATHER THAN A FLAG ON THE RULER ONES
# Three things about these datasets differ from RULER, and each one silently corrupts a number
# rather than failing:
#
# 1. MAX_CONTEXT_LENGTH is mandatory for longbench-v2. Its contexts run to 2.3M tokens (median 127K
#    at fraction 0.1), the pipeline defaults max_context_length to tokenizer.model_max_length =
#    131072, and Qwen3-8B has 40960 RoPE positions. The default therefore feeds the model 3.2x its
#    positional range on half the rows. Pinned to 32768 here -- inside the trained range, and the
#    largest length the indexer's schedule (8192/16384/32768) actually saw.
#
# 2. Sharding is by CONTEXT, and math500/aime25 have exactly ONE context (a single space -- the whole
#    problem lives in the `question` column). Round-robin over contexts puts every row on shard 0 and
#    leaves the other GPUs idle, so those two datasets shard by ROW instead; see --shard_by in
#    evaluate_sparse_sharded.py. longbench-v2 has 462 distinct contexts and keeps context sharding,
#    where it is the right call.
#
# 3. These are GENERATIVE benchmarks with long answers (max_new_tokens 4096 for math500, 32000 for
#    aime25) against RULER's ~50. Decode dominates, so a row is not cheap and the GPU count matters.
#
# Read the aime25 numbers as a 30-problem sample whatever the fraction: the 95% CI on 30 Bernoulli
# trials is roughly +/-18 points at p=0.5. It is a smoke test for "reasoning still works at all",
# not a ranking instrument. math500 (n=50) and longbench-v2 (n=50) are directionally usable.
set -euo pipefail

cd "$(dirname "$0")"

MODEL="${MODEL:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B}"
CKPT="${CKPT:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_scalar/stage1_16k_mid256_longce_decay/final.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-./results_reasoning_longbench}"

# Which arms to run, in order. Dense first so the upper bound is on disk before the sweep that is
# measured against it.
ARMS="${ARMS:-dense sparse}"
TASKS="${TASKS:-math500 aime25 longbench-v2}"

# Per-task top-k budgets. math500/aime25 probe whether a reasoning trace survives eviction at all;
# longbench-v2 goes lower because a 32K context under a 1024 budget is the regime the method is for.
TOPKS_MATH="${TOPKS_MATH:-2048 4096}"
TOPKS_AIME="${TOPKS_AIME:-2048 4096}"
TOPKS_LBV2="${TOPKS_LBV2:-2048 1024}"

# Per-task sampling. aime25 is 30 rows total; see the header for why it is not sampled.
FRACTION="${FRACTION:-0.1}"
FRACTION_AIME="${FRACTION_AIME:-1.0}"
SEED="${SEED:-42}"

# Mandatory for longbench-v2 -- see (1) in the header. Applies to every arm and both scripts, so the
# dense baseline and the indexer truncate identically and the comparison stays single-variable.
MAX_CONTEXT_LBV2="${MAX_CONTEXT_LBV2:-32768}"

# Caps the reasoning traces. aime25 ships max_new_tokens=32000 and math500 4096; at 32000 a single
# unlucky non-terminating trace costs hours of pure decode on its own. 8192 is past where Qwen3-8B
# closes a correct AIME trace, and the metric records `answered` (whether \boxed{} appeared at all),
# so truncation shows up as a number rather than hiding as a wrong answer. Set to 0 to use each
# dataset's own value.
# Qwen3 thinking mode. THINKING=1 leaves the assistant turn open so the model emits its own <think>
# block; the default emits a pre-closed empty one, which suppresses reasoning mode. This is the
# single biggest lever on the math benchmarks -- non-thinking Qwen3-8B measured 0.167 on aime25 here.
#
# It also changes the compute profile completely, which is why the cap moves with it: a thinking
# trace runs many thousands of tokens before the answer, so a cap tuned for non-thinking truncates
# every row *before* \boxed{} and reports ~0 accuracy with `answered` near zero. Because every
# decoded token is also a key the indexer must then rank, a longer trace means the sparse arm evicts
# far more aggressively at the same topk -- which makes this the more informative regime for the
# comparison, not merely the higher-scoring one.
THINKING="${THINKING:-0}"

# Cap on generated tokens, defaulted per mode (an explicit MAX_NEW_TOKENS= always wins). aime25
# ships 32000 and math500 4096; at 32000 one non-terminating non-thinking trace costs hours of pure
# decode on its own, so 8192 is the non-thinking default -- past where Qwen3-8B closes a correct
# trace. The metric records `answered` (whether \boxed{} appeared), so truncation surfaces as a
# number rather than hiding as a wrong answer. Set to 0 to use each dataset's own value.
if [[ "$THINKING" == "1" ]]; then
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
else
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
fi

FORCE_LOCAL="${FORCE_LOCAL:-64}"
FORCE_SINK="${FORCE_SINK:-4}"
BLOCK_K="${BLOCK_K:-64}"
PRECISION="${PRECISION:-tf32}"
ATTN="${ATTN:-sdpa}"

# The container interpreter. The shard subprocesses are spawned with this one's sys.executable, so
# setting it here pins the whole run.
PYTHON="${PYTHON:-/opt/conda/envs/torch-base/bin/python}"

# This box reaches huggingface.co only through the proxy; without it load_dataset falls back to the
# local cache. Harmless when a direct route exists.
export https_proxy="${https_proxy:-http://star-proxy.oa.com:3128}"
export http_proxy="${http_proxy:-http://star-proxy.oa.com:3128}"

if [[ ! -f "$CKPT" ]]; then
  echo "indexer checkpoint not found at $CKPT (set CKPT=)" >&2
  exit 1
fi

num_gpus=$(nvidia-smi --list-gpus | wc -l)
# DEVICES pins exact CUDA indices, e.g. DEVICES="0,1,2,3". Needed because NGPU=n always takes GPUs
# 0..n-1, which collides with anything already running there -- and an eval that lands on a training
# GPU OOMs the training run, not just itself.
DEVICES="${DEVICES:-}"
if [[ -n "$DEVICES" ]]; then
  IFS=',' read -r -a device_list <<< "$DEVICES"
  NGPU="${#device_list[@]}"
  SHARD_ARGS=(--devices "$DEVICES")
else
  NGPU="${NGPU:-$num_gpus}"
  if [[ "$NGPU" -gt "$num_gpus" ]]; then
    echo "Error: NGPU=$NGPU exceeds the $num_gpus GPUs on this box" >&2
    exit 1
  fi
  SHARD_ARGS=(--ngpu "$NGPU")
fi

echo "model:  $MODEL"
echo "ckpt:   $CKPT"
echo "arms:   $ARMS"
echo "tasks:  $TASKS"
echo "gpus:   $NGPU ${DEVICES:+[$DEVICES]}"
echo "output: $OUTPUT_DIR"

# Per-task knobs, resolved in one place so the dense and sparse arms cannot disagree about them.
task_fraction() { [[ "$1" == "aime25" ]] && echo "$FRACTION_AIME" || echo "$FRACTION"; }
task_topks() {
  case "$1" in
    math500)      echo "$TOPKS_MATH" ;;
    aime25)       echo "$TOPKS_AIME" ;;
    longbench-v2) echo "$TOPKS_LBV2" ;;
  esac
}
# Sharding axis -- see (2) in the header. One context means context-sharding degenerates to "all
# rows on shard 0".
task_shard_by() { [[ "$1" == "longbench-v2" ]] && echo "context" || echo "row"; }
task_extra() {
  # Only longbench-v2 needs the context cap; math500/aime25 prompts are a few hundred tokens.
  [[ "$1" == "longbench-v2" ]] && echo "--max_context_length $MAX_CONTEXT_LBV2"
}

for arm in $ARMS; do
  for task in $TASKS; do
    fraction="$(task_fraction "$task")"
    shard_by="$(task_shard_by "$task")"
    read -r -a extra <<< "$(task_extra "$task")"
    common=(
      --dataset "$task" --model "$MODEL"
      --fraction "$fraction" --seed "$SEED"
      --shard_by "$shard_by"
      "${SHARD_ARGS[@]}" "${extra[@]}"
    )
    [[ "$MAX_NEW_TOKENS" != "0" ]] && common+=(--max_new_tokens "$MAX_NEW_TOKENS")
    # Passed to BOTH arms, so thinking mode can never differ between the baseline and the indexer --
    # it changes the prompt, so a mismatch would make the two arms answer different questions.
    [[ "$THINKING" == "1" ]] && common+=(--enable_thinking True)

    case "$arm" in
      dense)
        # --config_file is evaluate_config_reasoning.yaml, NOT the default evaluate_config.yaml:
        # that one ships data_dir "4096" (these datasets have no length splits) and
        # model_kwargs.dtype, which transformers 4.56.0.dev0 rejects outright. See its header.
        echo "=== dense no_press: $task @ fraction $fraction across $NGPU GPU(s)"
        "$PYTHON" evaluate_sharded.py \
          --config_file ./evaluate_config_reasoning.yaml \
          "${common[@]}" \
          --press_name no_press --attn_implementation "$ATTN" \
          --output_dir "$OUTPUT_DIR/dense"
        ;;
      sparse)
        for topk in $(task_topks "$task"); do
          echo "=== sparse indexer topk=$topk: $task @ fraction $fraction across $NGPU GPU(s)"
          "$PYTHON" evaluate_sparse_sharded.py \
            "${common[@]}" \
            --indexer_ckpt "$CKPT" --topk "$topk" \
            --force_local "$FORCE_LOCAL" --force_sink "$FORCE_SINK" --block_k "$BLOCK_K" \
            --precision "$PRECISION" --attn_implementation "$ATTN" \
            --output_dir "$OUTPUT_DIR/sparse"
        done
        ;;
      *)
        echo "unknown arm '$arm' (expected 'dense' or 'sparse')" >&2
        exit 1
        ;;
    esac
  done
done

echo
echo "All evaluations completed. Summarize with:"
echo "  $PYTHON summarize_reasoning_longbench.py --output_dir $OUTPUT_DIR"
