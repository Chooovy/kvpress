#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Evaluate a JOINT checkpoint (router + adapted backbone) as sparse attention, data-parallel
# across all GPUs. The joint-training counterpart of evaluate_sparse_shard.sh.
#
#   bash evaluate_joint_shard.sh
#   CKPT=/path/final.pt LENGTHS="8192 16384" TOPKS=2048 bash evaluate_joint_shard.sh
#
# WHAT MAKES THIS DIFFERENT FROM evaluate_sparse_shard.sh
# ------------------------------------------------------
# One thing, and it is the whole reason this script exists: **the backbone is not the pretrained
# one.** Joint training (scripts/train_gqa_indexer_joint.py) moves 1.51B attention parameters, so
# the router was optimized against a model that --model does not contain. --backbone_ckpt injects
# the adapted weights before the press is built.
#
# Both come from the SAME file: a joint checkpoint carries the full model under "model" and a
# router-only view under "indexer", so CKPT is passed twice. evaluate_sparse.py REFUSES a
# checkpoint whose config records train_scope unless --backbone_ckpt is given, because that
# mistake produces a perfectly plausible number for a pairing that never existed -- the indexer
# loads cleanly against the pretrained backbone and nothing about the weights says it is wrong.
#
# THE COMPARISON THIS IS FOR
# --------------------------
# The frozen-backbone arm at the same geometry is the baseline:
#
#   # frozen router, pretrained backbone
#   CKPT=.../local128_b256/stage1_longce_decay_b256/final.pt \
#       LENGTHS=8192 TOPKS=2048 bash evaluate_sparse_shard.sh
#
#   # joint: same router lineage, adapted backbone
#   CKPT=.../joint_8k_local128_b256_bb2e-5_decay/final.pt \
#       LENGTHS=8192 TOPKS=2048 bash evaluate_joint_shard.sh
#
# FRACTION and SEED are identical to that script's defaults, so the two score the same sampled
# rows and the difference is the training regime alone.
#
# READ IT PER TASK. SP-KV's claim is that exposing the model to gating during training removes the
# train/test mismatch, so the gain should be largest where eviction hurts most -- niah_multikey and
# fwe in their Table 5. A uniform shift across all 13 tasks is more likely a scoring or geometry
# difference than an actual adaptation effect.
#
# GEOMETRY IS READ FROM THE CHECKPOINT, not set here: pin_mode / n_local / gate_budget /
# scalar_decay were recorded at save time and evaluate_sparse.py builds the press from them. Do not
# add flags for those -- a value passed here that disagreed with training would load every weight
# cleanly and silently score a different router.
set -euo pipefail

cd "$(dirname "$0")"

DATASET="${DATASET:-ruler}"
DATA_DIR="${DATA_DIR:-8192}"
MODEL="${MODEL:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B}"
CKPT="${CKPT:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_joint/joint_8k_local128_b256_bb2e-5_decay/final.pt}"

# The adapted backbone. Defaults to CKPT because a joint payload holds both; override only to pair
# a router with a backbone from a DIFFERENT step, which is a deliberate ablation rather than a
# normal run.
BACKBONE="${BACKBONE:-$CKPT}"

OUTPUT_DIR="${OUTPUT_DIR:-./results_joint_local128_b256_decay}"

# MUST match the n_local the router was TRAINED with (128 for the local+sink checkpoints), not
# the 64 that the older frozen-backbone scripts default to. The router never optimized against a
# 64-wide window: halving it takes away free neighbours it was counting on and it has to spend
# budget re-selecting them. This is not a small effect: the first run of this script defaulted to
# 64 and scored 68.81, against 77.62 for the FROZEN arm evaluated at its trained 128 -- a gap that
# would have read as "joint training costs 8.8 points" when most or all of it was the eval
# geometry. The checkpoint records n_local; keep the two in sync.
FORCE_LOCAL="${FORCE_LOCAL:-128}"
FORCE_SINK="${FORCE_SINK:-4}"
BLOCK_K="${BLOCK_K:-64}"
PRECISION="${PRECISION:-tf32}"
SCORER="${SCORER:-}"
# Must match evaluate_sparse_shard.sh / evaluate_dense_baseline.sh, or the joint arm and the
# frozen arm score different rows and the comparison is meaningless. Sampling happens BEFORE
# sharding, so these give the identical subset at any shard count.
FRACTION="${FRACTION:-0.1}"
SEED="${SEED:-42}"

PYTHON="${PYTHON:-/opt/conda/envs/torch-base/bin/python}"

export https_proxy="${https_proxy:-http://star-proxy.oa.com:3128}"
export http_proxy="${http_proxy:-http://star-proxy.oa.com:3128}"

read -r -a TOPKS <<< "${TOPKS:-2048}"
read -r -a LENGTHS <<< "${LENGTHS:-$DATA_DIR}"

if [[ ! -f "$CKPT" ]]; then
  echo "joint checkpoint not found at $CKPT (set CKPT=)" >&2
  exit 1
fi
if [[ ! -f "$BACKBONE" ]]; then
  echo "backbone checkpoint not found at $BACKBONE (set BACKBONE=)" >&2
  exit 1
fi

# A joint checkpoint written before the full-model fix stored ONLY the router (~73 MB against
# ~16 GB), and its adapted attention weights are gone for good. Caught here with the size, because
# the failure downstream is a missing 'model' key whose meaning is not obvious.
ckpt_bytes=$(stat -c %s "$BACKBONE" 2>/dev/null || echo 0)
if [[ "$ckpt_bytes" -lt 1000000000 ]]; then
  echo "ERROR: $BACKBONE is only $((ckpt_bytes / 1000000)) MB." >&2
  echo "  A joint checkpoint carrying the full model is ~16 GB. This one predates the" >&2
  echo "  full-model save fix and holds the router ONLY -- its adapted backbone is not" >&2
  echo "  recoverable, so that training run has to be redone." >&2
  exit 1
fi

num_gpus=$(nvidia-smi --list-gpus | wc -l)
DEVICES="${DEVICES:-}"
if [[ -n "$DEVICES" ]]; then
  IFS=',' read -r -a device_list <<< "$DEVICES"
  NGPU="${#device_list[@]}"
  SHARD_ARGS=(--devices "$DEVICES")
  echo "sharded over ${NGPU} pinned GPU(s) [$DEVICES]: lengths=${LENGTHS[*]} topks=${TOPKS[*]}"
else
  NGPU="${NGPU:-$num_gpus}"
  if [[ "$NGPU" -gt "$num_gpus" ]]; then
    echo "Error: NGPU=$NGPU exceeds the $num_gpus GPUs on this box" >&2
    exit 1
  fi
  SHARD_ARGS=(--ngpu "$NGPU")
  echo "sharded over $NGPU GPU(s): lengths=${LENGTHS[*]} topks=${TOPKS[*]}"
fi

echo "router   : $CKPT"
echo "backbone : $BACKBONE"

for length in "${LENGTHS[@]}"; do
  for topk in "${TOPKS[@]}"; do
    EXTRA=()
    [[ -n "$length" ]] && EXTRA+=(--data_dir "$length")
    [[ -n "$SCORER" ]] && EXTRA+=(--scorer "$SCORER")
    echo "=== topk=$topk @ ${length:-default} across $NGPU GPU(s)"
    "$PYTHON" evaluate_sparse_sharded.py \
      "${SHARD_ARGS[@]}" \
      --dataset "$DATASET" --model "$MODEL" \
      --indexer_ckpt "$CKPT" --backbone_ckpt "$BACKBONE" \
      --topk "$topk" --force_local "$FORCE_LOCAL" --force_sink "$FORCE_SINK" --block_k "$BLOCK_K" \
      --precision "$PRECISION" \
      --fraction "$FRACTION" --seed "$SEED" --output_dir "$OUTPUT_DIR" \
      "${EXTRA[@]}"
  done
done

echo "All sharded joint evaluations completed."
