#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the LongCE weight cache that `LONGCE=1 train_gqa_indexer_scalar_gy.sh` reads.
#
#   scripts/precompute_longce_weights.sh              # 16K cache, the stage1_16k default
#   scripts/precompute_longce_weights.sh 32768        # wider, for the full curriculum
#   MAX_DOCS=64 scripts/precompute_longce_weights.sh  # a small cache to smoke-test the path
#
# One GPU per slice, all in parallel. Resumable: a shard whose .npz already exists is skipped, so
# re-running after an interruption picks up where it stopped.
#
# WHY A CACHE (see kvpress/presses/gqa_indexer/longce_weights.py for the full argument)
# The backbone is FROZEN, so `L^short` and `L^long` do not depend on the router -- the weights are a
# property of the DATA. Recomputing them per step would pay (L-K)/d extra forward passes for a
# constant, on a step whose trainable part is only the indexer. It would also make each step's
# weights reflect that batch's difficulty, which is one of the ways the earlier delta arm went wrong.
#
# SIZING. --mirror-loader (MIRROR=1, the default) scores exactly the (shard, row) pairs the training
# loader will draw, reproducing its shard and row shuffles. This is not an optimization -- it is what
# makes the cache usable at all. Each reader walks its assigned shards SEQUENTIALLY and shuffles rows
# WITHIN a shard, so a 600-step run exhausts the first shard or two of each reader's list rather than
# sampling all 58. Caching stored-order prefixes of every shard measured a 2.5% hit rate, i.e. ~97% of
# the run's documents would fall back to weight 1 and the "LongCE" arm would be the plain objective
# with a different directory name. Mirroring measured 100%, verified against the real loader.
#
# MIRROR_WORLD_SIZE is the DATA-parallel size, which for stage1_16k is 1: FFN_SP=8 on 8 GPUs is ONE
# data-parallel replica. MIRROR_WORKERS must match the training run's --num-workers, and MIRROR_DOCS
# pads the ~4800 documents a 600-step run at --global-batch-size 8 consumes.
#
# Uncached documents are NOT an error at training time -- they fall back to weight 1, which is this
# weighting's neutral value -- but they are reported as `longce_cache_miss_frac` in the metrics, and a
# high value means the run is mostly the plain objective. Check it in longce_smoke.
set -euo pipefail

cd "$(dirname "$0")/.."

SEQ_LEN="${1:-16384}"

MODEL="${MODEL:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B}"
TOKENIZED="${TOKENIZED:-/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_tokenized_64k}"
OUT="${OUT:-/apdcephfs_gy8/share_303843174/guhao/datasets/longce_weights_$((SEQ_LEN / 1024))k}"
SUBSETS="${SUBSETS:-2e16 2e17}"

# K=1024, not the paper's 4096 default: their own Table 7 ablation has K=1k beating K=4k on RULER
# (55.9 vs 49.7 at 200 steps) at lower cost, and 1024 gave the largest discrepancy signal and the
# lowest spearman(w, L_long) in evaluation/probe_longce_key_tokens.py on this corpus.
TRUNC_LEN="${TRUNC_LEN:-1024}"
WINDOW="${WINDOW:-1024}"
GAMMA="${GAMMA:-5.0}"
LOGIT_CHUNK="${LOGIT_CHUNK:-4096}"
MAX_DOCS="${MAX_DOCS:-128}"

MIRROR="${MIRROR:-1}"
MIRROR_WORLD_SIZE="${MIRROR_WORLD_SIZE:-1}"
MIRROR_WORKERS="${MIRROR_WORKERS:-2}"
# MIRROR_DOCS is per reader per stage, and stage1_16k consumes EXACTLY 1200: 300 steps x
# --global-batch-size 8 = 2400 documents per stage, split over --num-workers 2. 1500 gives 25%
# headroom for the shuffle-buffer boundary without scoring documents no step will reach -- measured
# ~0.15 doc/s/GPU, so the plan (2 stages x 2 readers x 1500 = 6000 docs) is ~1.4 h on 8 GPUs against
# 5.5 h at 6000. Raise it for a longer run; MIRROR_DOCS=0 is not a thing, use MIRROR=0 for that.
MIRROR_DOCS="${MIRROR_DOCS:-1500}"
# The stages the training run will reach. Both must be planned: the loader seeds its shard shuffle
# with (seed + seq_len), so 8K and 16K draw DIFFERENT shards, and planning only one leaves the other
# stage missing the cache entirely.
MIRROR_SEQ_LENS="${MIRROR_SEQ_LENS:-8192 $SEQ_LEN}"

# `return 0` for the same reason as the training script's helpers: used in a command substitution
# under `set -e`, where a non-zero exit would kill the script silently.
mirror_args() {
  if [[ "$MIRROR" != "0" ]]; then
    echo "--mirror-loader --mirror-world-size $MIRROR_WORLD_SIZE" \
         "--mirror-num-workers $MIRROR_WORKERS --mirror-docs $MIRROR_DOCS" \
         "--mirror-seq-lens $MIRROR_SEQ_LENS"
  else
    echo "--max-docs-per-shard $MAX_DOCS"
  fi
  return 0
}

# Digest widths. Every stage that will read this cache must appear here, or the trainer refuses to
# run at that length rather than training on unverified weights. 8192 is the curriculum's first
# stage; SEQ_LEN is the cache's own width.
CHECKSUM_WIDTHS="${CHECKSUM_WIDTHS:-8192 $SEQ_LEN}"

NGPU="${NGPU:-$(nvidia-smi --list-gpus | wc -l)}"
PYTHON="${PYTHON:-python}"

echo "LongCE cache -> $OUT"
echo "  seq_len=$SEQ_LEN K=$TRUNC_LEN d=$WINDOW gamma=$GAMMA"
if [[ "$MIRROR" != "0" ]]; then
  echo "  mirror-loader: dp_world=$MIRROR_WORLD_SIZE workers=$MIRROR_WORKERS docs=$MIRROR_DOCS stages='$MIRROR_SEQ_LENS'"
else
  echo "  sampled: max_docs/shard=$MAX_DOCS  (NOT mirroring the loader -- expect a low hit rate)"
fi
echo "  digests at: $CHECKSUM_WIDTHS"
echo "  $NGPU GPU(s), subsets: $SUBSETS"

mkdir -p "$OUT"
pids=()
for i in $(seq 0 $((NGPU - 1))); do
  CUDA_VISIBLE_DEVICES="$i" "$PYTHON" -m scripts.precompute_longce_weights \
    --model "$MODEL" --tokenized "$TOKENIZED" --out "$OUT" \
    --subsets $SUBSETS \
    --seq-len "$SEQ_LEN" --trunc-len "$TRUNC_LEN" --window "$WINDOW" --gamma "$GAMMA" \
    --checksum-widths $CHECKSUM_WIDTHS \
    --logit-chunk "$LOGIT_CHUNK" $(mirror_args) --seed "${SEED:-0}" \
    --shard-index "$i" --shard-count "$NGPU" \
    > "$OUT/worker$i.log" 2>&1 &
  pids+=($!)
done

# Wait on each PID rather than a bare `wait`, so one worker's failure fails this script instead of
# leaving a partial cache that looks complete.
status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "worker $i FAILED; see $OUT/worker$i.log" >&2
    status=1
  fi
done
if [[ "$status" != "0" ]]; then
  echo "the cache is INCOMPLETE -- re-run to fill the missing shards (finished ones are skipped)" >&2
  exit "$status"
fi

echo "done: $(find "$OUT" -name '*.npz' | wc -l) shard(s)"
grep -h "^.*totals:" "$OUT"/worker*.log | tail -"$NGPU" || true
