#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The 0.6B ablation family, continued: scorer STRUCTURE (linear / conv / rnn) and two
# best-setting variants (objective rvkl+ce, and the two halves of the local+sink pin).
#
#   scripts/abl_06b.sh smoke   # 2 steps per arm, sequential, READ THE OUTPUT FIRST
#   scripts/abl_06b.sh run     # the real runs, one arm per GPU, in parallel
#   scripts/abl_06b.sh eval    # RULER 8K, mass + uniform, after `run` finishes
#
# WHY THESE SIX ARMS
# ------------------
# Three answer "does the scorer's STRUCTURE matter?", against `abl_anchor_rvkl` (mid_dim=256 MLP
# on h_j alone) as the shared baseline:
#
#   linear  mid_dim=0        -- LESS capacity. SparseK's plain `w . h`. 4k params vs 1M.
#   conv    K=8 causal conv  -- MORE information: reads h_{j-8..j-1} as well as h_j.
#   rnn     gated state      -- MORE information, UNBOUNDED: reads all of h_{<j}.
#
# Both history arms are supersets of the baseline BY CONSTRUCTION (zero-init `w_a`, so step 0 is
# bit-identical to the scalar arm at the same mid_dim), which is what makes each a single-variable
# A/B. Both read STRICTLY the past -- conv drops tap 0, rnn reads S_{j-1} -- so h_j reaches the
# score only through W_in, exactly as in the baseline.
#
# The prior is negative and that is the point: the strictly MORE expressive prefix arm already
# lost at matched objective (8B RULER 8K 73.45 vs 73.71 at 2.5x params; `abl_prefix` is its 0.6B
# twin), and a fixed-decay recurrent state died at probe level (t=0.79, sign test p=0.388).
# Bracketing the space from below (linear), from the local end (conv) and from the unbounded end
# (rnn) turns one architecture's negative result into a property of the problem.
#
# Three more vary the BEST SETTING one knob at a time, same baseline:
#
#   rvkl_ce01     objective (1-w)*rvkl + w*CE at w=0.1. The 8B `rvkl_ce01` arm scored 80.20,
#                 the best 8B number on record, but has no 0.6B twin -- so this closes that gap.
#   pin_local     pin_mode=local (n_local=128), i.e. the sink half REMOVED
#   pin_sink      pin_mode=sink  (n_sink=4),   i.e. the local half REMOVED
#
# The two pin arms decompose `local+sink`, which every existing 0.6B arm uses and none isolates.
# The 8B measurement says the two halves trade off sharply -- local+sink vs sink alone was
# RULER 72.53 vs 74.32 overall, but +2.48 on NIAH and -18.53 on cwe/fwe -- so judge these on the
# subtask split, not the 13-task mean.
#
# WHAT IS HELD FIXED (all six == abl_anchor_rvkl, from its final.pt['config'])
# ---------------------------------------------------------------------------
# model Qwen3-0.6B, objective rvkl (reverse KL on h_dense), kl_chunk 2048, seed 1000,
# schedule 8192:100, subsets 2e16+2e17, global batch 8 (= 1 x accum 8), WSD 1e-3 -> 5e-6
# (warmup 0.10 / stable 0.60), gate_budget 256, decay ON (ref 16384, init -1.0), liger, 1 GPU.
#
# THE HDENSE CACHE IS GEOMETRY-LOCKED. `hdense_06b_8k_seed1000` is keyed by the loader draw, so
# seed / schedule / subsets / batch geometry / world-size / --take-from must stay exactly as
# below or `fwkl_cache_miss_frac` leaves 0.0 and every step silently pays an inline teacher
# forward (~+27% wall clock) -- a slower run that still looks healthy.
#
# SANITY TARGETS (from the two existing arms)
#   step 0 loss ~0.27 -> final ~0.061 (decay on) / ~0.068 (decay off)
#   fwkl_cache_miss_frac 0.0 on EVERY step
#   trainable params 7.53M with --scalar-decay (7.47M without) on the scalar arms
#   ~22 s/step, ~36 min total, peak ~23.7 GiB of 95.6
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PY:-/opt/conda/envs/torch-base/bin/python}"
MODEL="${MODEL:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-0.6B}"
TOK="${TOK:-/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_tokenized_64k}"
HDENSE="${HDENSE:-/apdcephfs_gy8/share_303843174/guhao/datasets/hdense_06b_8k_seed1000}"
# Required by the trainer's argparse even when --tokenized supplies the actual corpus (it is the
# fallback path, and the check does not know about --tokenized). The pre-tokenized shards are
# what get read -- the log line to confirm that is "pre-tokenized corpus: 373038 docs".
DATA_ROOT="${DATA_ROOT:-/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_256k_filtered}"
OUT_ROOT="${OUT_ROOT:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-0.6B-gqa_indexer_scalar}"

# GPUs 5-7 are reserved for other work. Six arms over five GPUs, so one GPU takes two arms
# sequentially -- see the pairing in `run` below.
GPUS="${GPUS:-0 1 2 3 4}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

MAX_STEPS="${MAX_STEPS:-100}"
SCHEDULE="${SCHEDULE:-8192:100}"

# Everything the six arms share. Any change here changes all six together, which is the point.
common_args() {
  echo "--model $MODEL --data-root $DATA_ROOT --tokenized $TOK --subsets 2e16 2e17 \
--fwkl $HDENSE --fwkl-reverse --take-from head --kl-chunk 2048 \
--seed 1000 --schedule $SCHEDULE --max-steps $MAX_STEPS \
--stage dense --gate-budget 256 \
--scalar-decay --scalar-decay-ref 16384 --scalar-decay-init -1.0 \
--scalar-pos-slope 1e-6 \
--liger --global-batch-size 8 --batch-size 1 --gate-sparsity \
--peak-lr 1e-3 --final-lr 5e-6 --warmup-frac 0.10 --stable-frac 0.60 \
--shuffle-buffer 64 --num-workers 2 --log-every 10"
  return 0
}

# The one line that differs per arm. Baseline for all six: abl_anchor_rvkl.
arm_args() {
  case "$1" in
    # --- scorer structure ---
    linear)    echo "--scorer scalar --scalar-mid-dim 0   --pin-mode local+sink --n-sink 4 --n-local 128" ;;
    conv)      echo "--scorer conv   --scalar-mid-dim 256 --pin-mode local+sink --n-sink 4 --n-local 128 \
--conv-kernel 8 --conv-dim 256" ;;
    rnn)       echo "--scorer rnn    --scalar-mid-dim 256 --pin-mode local+sink --n-sink 4 --n-local 128 \
--state-dim 256 --rnn-gate-mode learned --rnn-gate-bias 2.0" ;;
    # --- best-setting variants ---
    rvkl_ce01) echo "--scorer scalar --scalar-mid-dim 256 --pin-mode local+sink --n-sink 4 --n-local 128 \
--fwkl-ce-weight 0.1" ;;
    pin_local) echo "--scorer scalar --scalar-mid-dim 256 --pin-mode local       --n-local 128" ;;
    pin_sink)  echo "--scorer scalar --scalar-mid-dim 256 --pin-mode sink        --n-sink 4" ;;
    *) echo "unknown arm: $1" >&2; return 1 ;;
  esac
  return 0
}

ARMS="${ARMS:-linear conv rnn rvkl_ce01 pin_local pin_sink}"

launch() {  # arm gpu extra_flags...
  local arm="$1" gpu="$2"; shift 2
  local out="$OUT_ROOT/abl_$arm"
  mkdir -p "$out"
  echo "[gpu$gpu] abl_$arm -> $out"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m scripts.train_gqa_indexer_e2e \
    $(common_args) $(arm_args "$arm") \
    --out "$out" --metrics-file "$out/metrics.jsonl" \
    --save-every "$MAX_STEPS" \
    "$@" > "$out/launch.log" 2>&1
}

case "${1:-}" in
  smoke)
    # 2 steps each, sequential, into a throwaway dir. Catches: flag typos, a scorer that will
    # not build, a shape mismatch, and -- the one that matters -- an hdense cache MISS, which
    # would otherwise only show up as a run that is quietly 27% slower.
    for arm in $ARMS; do
      out="$OUT_ROOT/_smoke_abl_$arm"; mkdir -p "$out"
      echo "=== smoke $arm"
      CUDA_VISIBLE_DEVICES=0 MAX_STEPS=2 SCHEDULE=8192:2 "$PY" -m scripts.train_gqa_indexer_e2e \
        $(MAX_STEPS=2 SCHEDULE=8192:2 common_args) $(arm_args "$arm") \
        --out "$out" --metrics-file "$out/metrics.jsonl" --save-every 2 \
        > "$out/launch.log" 2>&1 \
        && { echo "  OK"; grep -oE '"(loss|fwkl_cache_miss_frac)": [0-9.e-]+' "$out/metrics.jsonl" | tail -4; } \
        || { echo "  FAILED -- tail of $out/launch.log:"; tail -25 "$out/launch.log"; exit 1; }
      grep -E "trainable|cache" "$out/launch.log" | head -3 || true
    done
    echo "ALL SMOKE OK"
    ;;
  run)
    # One arm per GPU in parallel. Six arms, five GPUs: gpu4 takes pin_local then pin_sink
    # sequentially (they are the two cheapest -- plain scalar, no extra branch).
    set -- $GPUS
    launch linear    "$1" & p1=$!
    launch conv      "$2" & p2=$!
    launch rnn       "$3" & p3=$!
    launch rvkl_ce01 "$4" & p4=$!
    ( launch pin_local "$5" && launch pin_sink "$5" ) & p5=$!
    fail=0
    for p in $p1 $p2 $p3 $p4 $p5; do wait "$p" || fail=1; done
    for arm in $ARMS; do
      log="$OUT_ROOT/abl_$arm/launch.log"
      printf '%-12s %s\n' "$arm" "$(grep -oE 'done in [0-9.]+ min|Error|Traceback' "$log" 2>/dev/null | tail -1 || echo '(no log)')"
    done
    [ "$fail" = 0 ] || { echo "AT LEAST ONE ARM FAILED"; exit 1; }
    ;;
  eval)
    # RULER 8K, topk=2048, fraction 0.1 seed 42, both head-budget settings -- matching the
    # existing 0.6B evals exactly so the new arms drop into the same table.
    #
    # force_local/force_sink MUST match what the arm TRAINED with, or the eval scores a geometry
    # the router never learned. Hence the per-arm values for the two pin arms.
    #
    # EVAL_DEVICES pins the CUDA indices; default 0,1,2,3 deliberately leaves GPU 4 alone, which
    # matters while pin_sink is still training there -- an eval that lands on a training GPU OOMs
    # the training run, not just itself. The two head-budget settings run SEQUENTIALLY over the
    # same four GPUs (each shard loads its own model copy), which is where the existing
    # ~14 min/arm figure comes from.
    #
    # Skips any arm whose final.pt is missing, so this is safe to run while a late arm trains
    # and then re-run for the remainder.
    cd evaluation
    EVAL_DEVICES="${EVAL_DEVICES:-0,1,2,3}"
    for arm in $ARMS; do
      ckpt="$OUT_ROOT/abl_$arm/final.pt"
      if [ ! -f "$ckpt" ]; then
        echo "SKIP abl_$arm -- no final.pt yet"; continue
      fi
      case "$arm" in
        pin_local) fl=128; fs=0 ;;
        pin_sink)  fl=0;   fs=4 ;;
        *)         fl=128; fs=4 ;;
      esac
      # mass only. The uniform baseline is already established for this family
      # (abl_anchor_rvkl 58.86 / abl_decay_off 58.76) and the mass-vs-uniform gain is not what
      # these six arms vary -- so evaluating it again doubles the cost for no new information.
      # Set HEAD_BUDGETS="mass uniform" to restore both.
      for hb in ${HEAD_BUDGETS:-mass}; do
        floor=512; [ "$hb" = uniform ] && floor=0
        echo "=== eval abl_$arm hb=$hb (force_local=$fl force_sink=$fs)"
        MODEL="$MODEL" CKPT="$ckpt" PYTHON="$PY" \
        DATASET=ruler DATA_DIR=8192 LENGTHS=8192 TOPKS=2048 \
        HEAD_BUDGET="$hb" HEAD_BUDGET_FLOOR="$floor" \
        FORCE_LOCAL="$fl" FORCE_SINK="$fs" BLOCK_K=64 PRECISION=tf32 \
        FRACTION=0.1 SEED=42 DEVICES="$EVAL_DEVICES" \
        OUTPUT_DIR="./results_B_06b_abl_${arm}_${hb}" \
        bash evaluate_sparse_shard.sh
      done
    done
    ;;
  *)
    echo "usage: $0 {smoke|run|eval}" >&2; exit 2 ;;
esac
