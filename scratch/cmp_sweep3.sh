#!/usr/bin/env bash
# Phase 3: the experiment the per-row analysis identified as most informative.
#
# cwe collapsed -14.88 at 16K with R=64 (27 of 43 rows worse), having *gained* +3.49 at 8K with the
# same R. The mechanism should be slot capacity, not length: at 16K the keep budget is 12.5% instead
# of 25%, so each of the 64 slots summarizes roughly twice as many evicted keys and the centroid
# blurs -- on the one task that needs precision across the whole context.
#
# If that is right, raising R at 16K recovers cwe. R=128 is the count-matched analogue of R=64 at 8K;
# R=256 deliberately overshoots so the trend is visible rather than a single point. R comes out of
# topk either way, so 256 costs 12.5% of the read budget.
set -u
cd "$(dirname "$0")/.."

CKPT="${CKPT:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_scalar/stage1_16k_mid256_longce_decay/final.pt}"

run() {
  local tag="$1"; shift
  local log="/tmp/cmp_sweep_${tag}.log"
  echo "=== [$(date +%H:%M:%S)] $tag -> $log"
  env "$@" CKPT="$CKPT" TOPKS=2048 OUTPUT_DIR=./results_cmp \
    bash evaluation/evaluate_sparse_shard.sh > "$log" 2>&1
  echo "    [$(date +%H:%M:%S)] $tag done (exit $?)"
}

run r128_16k LENGTHS=16384 CMP_SLOTS=128 CMP_MASS=count
run r256_16k LENGTHS=16384 CMP_SLOTS=256 CMP_MASS=count
echo "PHASE3 DONE"
