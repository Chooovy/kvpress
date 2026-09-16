#!/usr/bin/env bash
# Serial CMP sweep. One 8-GPU sharded RULER run at a time -- the shard script already saturates the
# box, so overlapping runs would just contend. Logs per config; each writes its own results dir
# because evaluate_sparse tags the directory with cmp<R>-<mass>[-<space>][-d<delta>].
set -u
cd "$(dirname "$0")/.."

CKPT="${CKPT:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_scalar/stage1_16k_mid256_longce_decay/final.pt}"
LEN="${LEN:-8192}"
OUT="${OUT:-./results_cmp}"

run() {  # run <tag> <env assignments...>
  local tag="$1"; shift
  local log="/tmp/cmp_sweep_${tag}.log"
  echo "=== [$(date +%H:%M:%S)] $tag -> $log"
  env "$@" CKPT="$CKPT" LENGTHS="$LEN" TOPKS=2048 OUTPUT_DIR="$OUT" \
    bash evaluation/evaluate_sparse_shard.sh > "$log" 2>&1
  echo "    [$(date +%H:%M:%S)] $tag done (exit $?)"
}

# --- R sweep, both spaces. R=64 post/pre already measured at 8K, so they are skipped here.
run r16_post  CMP_SLOTS=16  CMP_MASS=count CMP_SPACE=post_rope
run r256_post CMP_SLOTS=256 CMP_MASS=count CMP_SPACE=post_rope
run r16_pre   CMP_SLOTS=16  CMP_MASS=count CMP_SPACE=pre_rope
run r256_pre  CMP_SLOTS=256 CMP_MASS=count CMP_SPACE=pre_rope

# --- mass: the analytic 1/2 Var correction, and a hand sweep of a constant offset. These say
# whether a LEARNED bias is even needed -- if count+var captures most of it, it is not.
run r64_var   CMP_SLOTS=64  CMP_MASS=count+var CMP_SPACE=post_rope
run r64_d1    CMP_SLOTS=64  CMP_MASS=count CMP_DELTA=1.0 CMP_SPACE=post_rope
run r64_d2    CMP_SLOTS=64  CMP_MASS=count CMP_DELTA=2.0 CMP_SPACE=post_rope

echo "ALL DONE"
