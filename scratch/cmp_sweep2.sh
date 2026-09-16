#!/usr/bin/env bash
# Phase 2: the learned mass head, plus 16K for the best fixed config.
set -u
cd "$(dirname "$0")/.."
CKPT="${CKPT:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_scalar/stage1_16k_mid256_longce_decay/final.pt}"
MH=/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_scalar/cmp_mass_R64_post_rope.pt
run() { local tag="$1"; shift; local log="/tmp/cmp_sweep_${tag}.log"
  echo "=== [$(date +%H:%M:%S)] $tag -> $log"
  env "$@" CKPT="$CKPT" TOPKS=2048 OUTPUT_DIR=./results_cmp bash evaluation/evaluate_sparse_shard.sh > "$log" 2>&1
  echo "    [$(date +%H:%M:%S)] $tag done (exit $?)"; }

# learned mass at R=64, the config the head was trained for
run r64_learned LENGTHS=8192 CMP_SLOTS=64 CMP_MASS_CKPT="$MH"
# the analytic 1/2 Var correction, zero training -- the honest baseline the learned head must beat
run r64_var     LENGTHS=8192 CMP_SLOTS=64 CMP_MASS=count+var
# 16K, where the baseline is 59.57 and cwe collapses: more headroom than 8K
run r64_16k     LENGTHS=16384 CMP_SLOTS=64 CMP_MASS=count
run r64_16k_lrn LENGTHS=16384 CMP_SLOTS=64 CMP_MASS_CKPT="$MH"
echo "PHASE2 DONE"
