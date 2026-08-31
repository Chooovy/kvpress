#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# SFT the PREFIX indexer on RULER, with the loss masked to the gold ANSWER only.
#
#   scripts/train_gqa_indexer_prefix_sft.sh scan       # RUN FIRST. keep-rate table, no training
#   scripts/train_gqa_indexer_prefix_sft.sh smoke      # 2 steps, verifies the path end to end
#   scripts/train_gqa_indexer_prefix_sft.sh niah       # THE HEADLINE RUN. train needles only
#   scripts/train_gqa_indexer_prefix_sft.sh all        # every task -- in-distribution, see below
#
# WHAT THIS STAGE IS
#
# Stage 1 trains the router against a plain next-token loss on long documents: every position
# contributes, and the router learns which keys matter on average. This stage asks a narrower
# question -- mask the prompt, keep loss only on RULER's gold answer, and the single thing gradient
# descent can do is route attention to the keys that answer depends on.
#
# THE BACKBONE IS FROZEN, SO THIS IS NOT AN SFT IN THE USUAL SENSE
#
# E2EIndexerTrainer.freeze_backbone leaves only the indexer trainable (~1.06M params/layer at
# MID_DIM=256 against 8B frozen). So this arm CANNOT learn an answer format and CANNOT memorize an
# answer into a weight -- which is the property that makes a RULER gain here meaningful rather than
# circular. It also means the loss value is close to uninformative: measured against the Qwen3
# tokenizer, a 16K prompt carries a 3-45 token answer, so ~0.1% of positions have gradient. The
# loss is a mean over ~20 tokens per sequence and is NOT comparable to any stage-1 number.
#
#   JUDGE THIS STAGE ON THE RULER METRIC, NOT ON THE LOSS CURVE.
#
# WHY --stage dense, AND WHY THE OBVIOUS ALTERNATIVE IS WRONG
#
# It is tempting to train under --stage sparse --topk 2048 to match evaluate_sparse_shard.sh's
# inference conditions. That is backwards here, for two measured reasons:
#
#   1. GRADIENT. Under sparse scope an unselected key's gradient is IDENTICALLY ZERO -- asserted in
#      tests/presses/test_gqa_indexer_e2e.py::test_full_scope_gradients_are_independent
#      ("k_idx rows 3.. are never selected, so under sparse scope they receive nothing"). A router
#      that currently MISSES the needle would therefore get exactly no signal to start selecting
#      it, which is the one thing this stage exists to fix. It would descend on the rows it already
#      gets right and look perfectly healthy.
#   2. MEMORY. The sparse backward runs through the gather REFERENCE, not the Triton kernel --
#      triton_sparse_attention.py has no autograd.Function, only the full-scope path does. Measured
#      retention (saved_tensors_hooks, deduped by storage, real Qwen3-8B geometry Hkv=8/D=136/Dv=128)
#      is independent of query_tile and linear in Sq*topk: 38.8 GiB per layer at Sq=16384/topk=512,
#      i.e. ~1.4 TiB across 36 layers, against the 72.4 GiB a dense 16K stage-1 step actually peaks
#      at. query_tile bounds the transient, not the graph. This is also why no stage2 checkpoint has
#      ever been produced by either arm.
#
# So the train/inference scope gap is measured by the EVAL, not closed by the training config.
#
# WHICH TASKS TO TRAIN ON -- THIS DECIDES WHAT THE EVAL NUMBER MEANS
#
# `niah` (the default) trains the 8 needle tasks and leaves vt/cwe/fwe/qa_1/qa_2 untouched, so
# their scores stay a generalization claim: did the router learn to RETRIEVE, or to fit a template?
# `all` makes every scored task in-distribution -- a useful upper bound, not a headline.
#
# LENGTH: TRAIN 16K, EVAL 8K
#
# Rows are DROPPED, never truncated, when prompt+answer exceeds --sft-max-len: a needle sits at a
# fixed depth and may be in the cut, and cwe/fwe answers are counts over the whole list.
#
# MAX_LEN=24576 is not a guess -- it is the smallest round cap that keeps every task. Measured by
# `scan` over all 6500 rows of the 16384 config against the Qwen3 tokenizer (prompt+answer tokens):
#
#   task              p50     max   keep@16K  keep@20K  keep@24K
#   niah_single_1/2/3  ~16.0K  16.0K    100%      100%      100%
#   niah_multikey_1     16.0K  16.0K    100%      100%      100%
#   niah_multikey_2     19.7K  19.8K      0%      100%      100%
#   niah_multikey_3     21.1K  21.2K      0%        0%      100%
#   niah_multivalue     16.0K  16.0K    100%      100%      100%
#   niah_multiquery     16.1K  16.1K    100%      100%      100%
#   vt                  16.3K  16.4K    100%      100%      100%
#   cwe                 22.3K  22.5K      0%        0%      100%
#   fwe                 15.0K  16.9K    100%      100%      100%
#   qa_1                13.5K  16.9K    100%      100%      100%
#   qa_2                16.5K  17.4K     47%      100%      100%
#   TOTAL                               73%       85%      100%
#
# So a 16384 cap would silently delete cwe, niah_multikey_2 and niah_multikey_3 from the mix and
# take half of qa_2 -- i.e. exactly the hardest tasks, which is the opposite of what you want and
# invisible except as those tasks' absence. Re-run `scan` before changing MAX_LEN.
#
# MEMORY: THE CAP IS NOT WHAT COSTS, THE LONGEST ROW IS
#
# Peak memory here is set by the longest row actually drawn, not by MAX_LEN -- a 24576 cap with a
# 22457-token longest row never builds a 24576-token graph. Prefix stage1_16k measured 43.9 GiB at
# L=8192 and 72.4 at L=16384 (FFN_SP=8, batch 1, liger on), which is linear in L:
#
#   peak ~= 15.4 + 0.00348 * L  GiB   ->   cwe's longest row (22457) lands at ~93.5 GiB
#
# That fits an H20's 96 GiB, but with only ~2.5 GiB of headroom, and the estimate is an
# extrapolation rather than a measurement. So:
#
#   * run `smoke` first -- it prints the real peak, and a `cwe` row will appear in the first steps.
#   * if it OOMs, MAX_LEN=20480 drops only cwe and niah_multikey_3 (~89 GiB), and MAX_LEN=16384
#     falls back to stage 1's measured 72.4 GiB at the cost of the four hardest task slices.
#   * GRAD_CHECKPOINT is deliberately NOT offered as the fix: the trainer's own --grad-checkpoint
#     is flagged EXPERIMENTAL with no passing test and raises CheckpointError on a toy model.
set -euo pipefail

MODE="${1:-scan}"

MODEL="${MODEL:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B}"
OUT="${OUT:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_prefix}"

# The RULER snapshot. A directory in the HuggingFace layout (<dir>/<config>/test-*.parquet), so
# --sft-config picks the context length. Read as parquet directly rather than through
# load_dataset(data_dir=...), which cannot map data_dir to a cache hash offline.
RULER="${RULER:-simonjegou/ruler}"
RULER_CONFIG="${RULER_CONFIG:-8192}"
MAX_LEN="${MAX_LEN:-24576}"

# THE checkpoint this decays from. Stage 1 stopped at step 600 still at peak LR (--max-steps
# truncated it before WSD's decay phase), so this stage is where the router finally anneals.
INIT="${INIT:-$OUT/stage1_16k_mid256_v128/final.pt}"

NNODES="${NNODES:-1}"
NGPU="${NGPU:-8}"
# 29511 distill, 29512 e2e, 29513 scalar, 29514 cross-replay, 29515 prefix, 29517 here.
MASTER_PORT="${MASTER_PORT:-29517}"

# LOW peak and a FULL decay, unlike stage 1. Two reasons: the router is already trained and this is
# an adaptation rather than a fresh fit, and --init-from resets AdamW's moments, so a 1e-3 peak
# against a 0.1%-dense loss would move a well-ordered ranking on very little evidence. WARMUP is
# short because there is nothing to warm up into, and STABLE is short because the whole point is to
# reach FINAL_LR by the end.
PEAK_LR="${PEAK_LR:-1e-4}"
FINAL_LR="${FINAL_LR:-5e-6}"
WARMUP_FRAC="${WARMUP_FRAC:-0.05}"
STABLE_FRAC="${STABLE_FRAC:-0.35}"

LIGER="${LIGER:-1}"
FFN_SP="${FFN_SP:-8}"

# MUST match the checkpoint being loaded, or --init-from's shape check rejects it. These are the
# stage1_16k_mid256_v128 geometry.
MID_DIM="${MID_DIM:-256}"
POS_SLOPE="${POS_SLOPE:-1e-6}"
PREFIX_HEAD_DIM="${PREFIX_HEAD_DIM:-128}"
PREFIX_VALUE_DIM="${PREFIX_VALUE_DIM:-128}"

# Same gate as stage 1. A gate flat along the key axis adds a per-row constant that cancels in the
# softmax, so without a pin the model reverts to the frozen dense backbone and the loss is
# satisfied with NO ranking learned -- which under a 0.1%-dense loss is an even cheaper escape.
PIN_MODE="${PIN_MODE:-sink}"
N_SINK="${N_SINK:-4}"
COMPRESSION_RATIO="${COMPRESSION_RATIO:-0.5}"

STEPS="${STEPS:-400}"
TASKS="${TASKS:-niah}"

cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

if [[ "$NGPU" -gt 1 ]]; then
  LAUNCH=(torchrun --nnodes "$NNODES" --nproc_per_node "$NGPU" --master_port "$MASTER_PORT")
else
  LAUNCH=(python)
fi

liger_arg() { [[ "$LIGER" != "0" ]] && echo "--liger"; }
gate_sparsity_arg() { [[ "${GATE_SPARSITY:-1}" != "0" ]] && echo "--gate-sparsity"; }

prefix_args() {
  echo "--scorer prefix --scalar-mid-dim $MID_DIM --scalar-pos-slope $POS_SLOPE \
--prefix-head-dim $PREFIX_HEAD_DIM --prefix-value-dim $PREFIX_VALUE_DIM"
}

# --data-root is NOT passed: --sft-ruler replaces the longmino corpus entirely, and the trainer
# only requires --data-root in its absence.
sft_args() {
  echo "--sft-ruler $RULER --sft-config $RULER_CONFIG --sft-max-len $MAX_LEN --sft-tasks $TASKS"
}

case "$MODE" in
  scan)
    # RUN THIS FIRST, and read the table rather than the exit code. Prints prompt-length
    # percentiles per task and the keep rate at several caps, so --sft-max-len is chosen from
    # measured lengths instead of guessed. Costs one tokenizer pass, no GPU.
    #
    # Scans ALL 13 tasks by default, not $TASKS: the cap is a property of the corpus, and seeing
    # what a cap does to the tasks you are NOT training on is the point -- those are the ones whose
    # eval scores carry the generalization claim. SCAN_TASKS overrides.
    exec python -m scripts.ruler_sft_scan \
      --ruler "$RULER" --config "$RULER_CONFIG" --model "$MODEL" \
      --tasks ${SCAN_TASKS:-all} --sweep
    ;;

  smoke)
    # 2 steps on 1 GPU, no checkpoint. Verifies the whole path: the prompt is built by the eval
    # pipeline's own preprocess, the mask leaves only the answer, and liger's fused CE agrees with
    # the unfused one ON MASKED LABELS -- the two reach the ignore-index by different routes, so
    # that check is worth more here than under a dense LM loss.
    exec python -m scripts.train_gqa_indexer_e2e \
      --model "$MODEL" $(prefix_args) $(sft_args) \
      --schedule "$MAX_LEN:50" --stage dense \
      --pin-mode "$PIN_MODE" --n-sink "$N_SINK" $(liger_arg) \
      --init-from "$INIT" \
      --num-workers 0 --log-every 1 --save-every 0 \
      --out "$OUT/sft_smoke" --dry-run
    ;;

  niah|all|other)
    TASKS="$MODE"
    if [[ ! -f "$INIT" ]]; then
      echo "SFT needs a stage-1 checkpoint at $INIT" >&2
      echo "  run scripts/train_gqa_indexer_prefix_gy.sh stage1_16k first, or set INIT=" >&2
      exit 1
    fi
    SUB="sft_${MODE}_${RULER_CONFIG}_len${MAX_LEN}"
    # FFN_SP=8 on 8 GPUs is ONE data-parallel replica, so GLOBAL_BATCH=8 accumulates to 8
    # sequences/step -- matching stage 1's, which keeps the two runs' step counts comparable.
    # --batch-size 1 is enforced by the trainer: RULER prompts differ by thousands of tokens, and
    # padding a batch would need an attention_mask threaded through the gate's selector.
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_e2e \
      --model "$MODEL" $(prefix_args) $(sft_args) \
      --schedule "$MAX_LEN:$STEPS" \
      --stage dense --pin-mode "$PIN_MODE" --n-sink "$N_SINK" $(liger_arg) \
      --ffn-sp-size "$FFN_SP" \
      --global-batch-size "${GLOBAL_BATCH:-8}" $(gate_sparsity_arg) \
      --compression-ratio "$COMPRESSION_RATIO" \
      --peak-lr "$PEAK_LR" --final-lr "$FINAL_LR" \
      --warmup-frac "$WARMUP_FRAC" --stable-frac "$STABLE_FRAC" \
      --batch-size 1 \
      --num-workers "${WORKERS:-2}" \
      --init-from "$INIT" \
      --out "$OUT/$SUB" --metrics-file "$OUT/$SUB/metrics.jsonl" \
      --save-every "${SAVE_EVERY:-100}" --log-every "${LOG_EVERY:-10}"
    ;;

  *)
    echo "usage: $0 {scan|smoke|niah|all|other}" >&2
    echo "  run 'scan' before any training run -- see the header on length skew" >&2
    echo "  evaluate with: DATA_DIR=8192 CKPT=$OUT/sft_niah_${RULER_CONFIG}_len${MAX_LEN}/final.pt \\" >&2
    echo "                 bash evaluation/evaluate_sparse_shard.sh" >&2
    exit 1
    ;;
esac
