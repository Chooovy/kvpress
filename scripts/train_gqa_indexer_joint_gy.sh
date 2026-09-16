#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Joint training: the router AND the backbone's attention, from the plain LM loss.
#
#   scripts/train_gqa_indexer_joint_gy.sh smoke     # 2 steps, verifies the whole path
#   scripts/train_gqa_indexer_joint_gy.sh joint_8k  # 8K, 300 steps, the real run
#
# Multi-node: NNODES is read by torchrun, and HYBRID_SHARD adapts automatically -- sharding
# within each node, replicating across them. The same command works at 1x8 / 2x8 / 4x8.
#
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=<host> scripts/train_gqa_indexer_joint_gy.sh joint_8k
#
# WHAT IS DIFFERENT FROM scripts/train_gqa_indexer_scalar_gy.sh
# -------------------------------------------------------------
# That script freezes the backbone and trains only the router. This one unfreezes attention,
# which is the one lever SP-KV has that a frozen run does not: their Appendix C.6 froze the LLM
# and trained only the predictor, and gate density stayed above 80% -- sparsity barely emerged.
#
# Three consequences, all of them load-bearing:
#
#   1. FSDP is required. Training 1.51B parameters needs grads + fp32 master + Adam moments,
#      which is ~21 GB replicated per rank under DDP on top of the weights. HYBRID_SHARD splits
#      that 8 ways inside the node.
#   2. The FFN stays FROZEN. This is correctness, not thrift. Without activation checkpointing
#      the run needs FFN-SP to fit, and _ScatterSequence's backward all-gathers so that every
#      rank holds the identical complete gradient -- which is what makes FSDP's uniform reduction
#      right. A trainable FFN would want a SUM (each rank sees 1/8 of the sequence) while every
#      attention-path gradient wants a MEAN. One reduction cannot serve both, and the failure is
#      silent: the FFN gradient would be 8x too small and the loss would still fall.
#   3. Two learning rates. SP-KV gives the predictor 5x the global LR (Table 4 baseline; their
#      ablation reads density 82.7% at multiplier 0.1, 37.8% at 1, 25.4% at 5). Reproduced here
#      as BACKBONE_LR x ROUTER_LR_MULT.
#
# THE GATE GEOMETRY IS INHERITED FROM THE CHECKPOINT AND MUST NOT DRIFT
# ---------------------------------------------------------------------
# INIT points at a router trained with local+sink / N_LOCAL=128 / GATE_BUDGET=256 / decay on.
# None of those change a tensor SHAPE, so a mismatch would load every weight cleanly and
# silently train a router whose score means something else. The trainer rejects a mismatch
# against the checkpoint's recorded config, which is why these defaults are not adjustable
# casually -- change them and you are starting a different experiment, not continuing this one.
#
# DATA: SEED=1000, NOT 0
# ----------------------
# The loaded router consumed 2400 documents under seed 0 (300 steps x 8 sequences). The corpus
# has 373,038, so that is 0.64%. loader_for derives its stream from `seed + seq_len`, so a fresh
# seed reshuffles both the shard order and the row order within each shard; expected overlap at
# this consumption rate is ~15 documents. The script REFUSES seed 0 for this reason.
#
# WHAT TO WATCH in --metrics-file
# -------------------------------
#   gate_sparsity_mean  -- THE readout. The frozen router reached 0.267. If joint training is
#                          doing what SP-KV claims, this should fall BELOW that. Rising toward
#                          1.0 means the backbone learned to route around the gate instead.
#   loss                -- should start near the frozen run's final 1.7997 and fall further. A
#                          JUMP at step 0 means the router did not load; check the init log line.
#   peak_gib            -- expected ~45 GiB. Above ~70 means FFN-SP is not taking effect.
#   lr_backbone/lr_router -- must stay in a 1:5 ratio for the whole run.
set -euo pipefail

MODE="${1:-smoke}"
DATA_ROOT="${DATA_ROOT:-/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_256k_filtered}"
TOKENIZED="${TOKENIZED:-/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_tokenized_64k}"
MODEL="${MODEL:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B}"
OUT="${OUT:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_joint}"

# The frozen-backbone router this continues from. Its recorded gate geometry is checked against
# the flags below and a mismatch is refused.
INIT="${INIT:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_scalar/local128_b256/stage1_longce_decay_b256/final.pt}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NGPU="${NGPU:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
# Distinct from the distillation (29511), pairwise-e2e (29512) and scalar (29513) scripts.
MASTER_PORT="${MASTER_PORT:-29514}"

# FFN sequence parallel across the node's 8 ranks. NOT optional without activation checkpointing:
# it is what takes the activation term from ~61 to ~40 GiB at 8K. It also composes with FSDP
# because it splits a DIFFERENT axis (sequence) than FSDP does (parameters).
FFN_SP="${FFN_SP:-$NGPU}"

# Two LRs. BACKBONE_LR is deliberately ~25x below SP-KV's 5.02e-4: theirs is a PRETRAINING LR
# spending 20 TPP on a 4.19M-token batch, this is a short adaptation of an already-trained model
# at a ~64x smaller batch. ROUTER_LR_MULT=5 is SP-KV's ratio.
BACKBONE_LR="${BACKBONE_LR:-2e-5}"
ROUTER_LR_MULT="${ROUTER_LR_MULT:-5.0}"
FINAL_LR_FRAC="${FINAL_LR_FRAC:-0.01}"
WARMUP_FRAC="${WARMUP_FRAC:-0.10}"
STABLE_FRAC="${STABLE_FRAC:-0.60}"

# Inherited from INIT -- see the header. Changing these starts a different experiment.
PIN_MODE="${PIN_MODE:-local+sink}"
N_SINK="${N_SINK:-4}"
N_LOCAL="${N_LOCAL:-128}"
GATE_BUDGET="${GATE_BUDGET:-256}"
MID_DIM="${MID_DIM:-256}"
POS_SLOPE="${POS_SLOPE:-1e-6}"
DECAY="${DECAY:-1}"
DECAY_REF="${DECAY_REF:-16384}"
DECAY_INIT="${DECAY_INIT:--1.0}"
COMPRESSION_RATIO="${COMPRESSION_RATIO:-0.5}"

# A fresh data stream. 0 is REFUSED by the trainer: it is what the loaded router already read.
SEED="${SEED:-1000}"
# `head` matches how the LongCE-trained router was fed, so a redrawn document yields the same
# window. `random` decorrelates the window too, at the cost of differing from that run's regime.
TAKE_FROM="${TAKE_FROM:-head}"

# Sequences per optimizer step, across all replicas. FFN_SP=8 makes the node ONE replica, so at
# 1x8 this is purely accumulation (8 = accum 8); at 4x8 there are 4 replicas and accum drops to 2.
GLOBAL_BATCH="${GLOBAL_BATCH:-8}"

# Which backbone parameters to train. `attention` (1.51B) is the only scope compatible with
# FFN-SP; `all` (8.19B) requires FFN_SP=1 and the trainer refuses the combination -- see
# --train-scope. The smoke_full mode below sets both together.
TRAIN_SCOPE="${TRAIN_SCOPE:-attention}"

# FREEZE_ROUTER=1 trains the backbone ONLY, against a fixed router. This is the answer to the
# first joint run's failure: router and backbone minimize the SAME loss, and the backbone (1.51B
# trainable vs the router's 38M) lowers it more cheaply by re-spreading attention through q/k
# than the router does by concentrating its budget. Measured on joint_8k_..._bb2e-5_decay:
#   gate_sparsity  0.282 -> 0.359   (the frozen-backbone arm reached 0.267 -- i.e. WORSE)
#   RULER 8K       77.62 -> 69.17   with every point of the loss in needle retrieval
#   topk=8192      93.71 vs dense 93.69  -- the backbone itself is UNHARMED
# So the backbone learned to route around the gate, not to live with it. Freezing the router
# removes that option.
FREEZE_ROUTER="${FREEZE_ROUTER:-0}"

# HARD_TOPK>0 trains on real hard eviction: keys outside each row's top-k are REMOVED (-inf), not
# down-weighted. This closes the soft/hard gap that the first joint run fell through -- the
# backbone learned to route around a constraint that is soft in training and absolute at
# inference. Verified exact against an explicit top-k dense reference to 6e-07.
#
# MATCH IT TO THE EVAL'S --topk (2048 for the current sweeps), and note it REQUIRES
# FREEZE_ROUTER=1: the mask is a per-row score threshold, which equals a top-k only while the
# scores are fixed. Costs no extra memory -- the threshold is O(Sq), the same shape as lse.
HARD_TOPK="${HARD_TOPK:-0}"

LIGER="${LIGER:-1}"

cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

if [[ ! -f "$INIT" ]]; then
  echo "INIT checkpoint not found: $INIT" >&2
  echo "  point INIT= at a frozen-backbone router, or train one with" >&2
  echo "  scripts/train_gqa_indexer_scalar_gy.sh stage1" >&2
  exit 1
fi

# The interpreter to launch under. torchrun must come from the SAME env as torch, or the
# rendezvous starts and then imports a different torch. /opt/conda/envs/torch-base is what these
# nodes have; override with PYBIN=/path/to/python.
PYBIN="${PYBIN:-/opt/conda/envs/torch-base/bin/python}"
if [[ ! -x "$PYBIN" ]]; then
  echo "PYBIN not executable: $PYBIN" >&2
  echo "  set PYBIN=/path/to/python (must be the env that has torch)" >&2
  exit 1
fi

LAUNCH=("$PYBIN" -m torch.distributed.run
  --nnodes "$NNODES" --node_rank "$NODE_RANK"
  --nproc_per_node "$NGPU"
  --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT")

liger_arg() { [[ "$LIGER" != "0" ]] && echo "--liger"; return 0; }
decay_args() {
  [[ "$DECAY" != "0" ]] && \
    echo "--scalar-decay --scalar-decay-ref $DECAY_REF --scalar-decay-init $DECAY_INIT"
  return 0
}
data_args() {
  if [[ -f "$TOKENIZED/index.json" ]]; then
    echo "--tokenized $TOKENIZED --subsets ${SUBSETS:-2e16 2e17}"
  else
    echo "--subsets ${SUBSETS:-2e16 2e17}"
  fi
  return 0
}

# Every flag that must not drift from INIT, in one place so the two modes cannot disagree.
common_args() {
  echo "--data-root $DATA_ROOT --model $MODEL"
  echo "--scorer scalar --scalar-mid-dim $MID_DIM --scalar-pos-slope $POS_SLOPE"
  echo "--pin-mode $PIN_MODE --n-sink $N_SINK --n-local $N_LOCAL --gate-budget $GATE_BUDGET"
  echo "--compression-ratio $COMPRESSION_RATIO"
  echo "--ffn-sp-size $FFN_SP --train-scope $TRAIN_SCOPE"
  [[ "$FREEZE_ROUTER" != "0" ]] && echo "--freeze-router"
  [[ "$HARD_TOPK" != "0" ]] && echo "--hard-topk $HARD_TOPK"
  echo "--backbone-lr $BACKBONE_LR --router-lr-mult $ROUTER_LR_MULT"
  echo "--final-lr-frac $FINAL_LR_FRAC --warmup-frac $WARMUP_FRAC --stable-frac $STABLE_FRAC"
  echo "--seed $SEED --take-from $TAKE_FROM"
  echo "--batch-size 1 --shuffle-buffer 64"
  echo "--init-from $INIT"
  return 0
}

case "$MODE" in
  smoke)
    # 2 steps. Checks, in order: the router loads (loss should start near 1.80, not 2.4+), FSDP
    # HYBRID_SHARD builds, FFN-SP composes with it, and peak_gib is in range.
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_joint \
      $(common_args) $(data_args) $(decay_args) $(liger_arg) \
      --schedule "${SCHEDULE:-8192:10}" \
      --global-batch-size "${GLOBAL_BATCH:-8}" --gate-sparsity \
      --num-workers 0 --log-every 1 --save-every 0 \
      --out "$OUT/smoke" --metrics-file "$OUT/smoke/metrics.jsonl" \
      --dry-run
    ;;

  smoke_full)
    # FULL-PARAMETER smoke: does 8.19B trainable fit at 8K without activation checkpointing?
    #
    # FFN_SP is forced to 1, not merely defaulted: a trainable FFN under FFN-SP gets a gradient
    # sp_size times too small with no error at all (see --train-scope). That removes the sharding
    # that was holding activations down, so this is the configuration most likely to OOM -- which
    # is the point of running it.
    #
    # Read peak_gib from the metrics. If it OOMs, the fallbacks in order are: --train-scope
    # attention (back to 1.51B and FFN_SP=8), or accept activation checkpointing (which needs the
    # _capture_hook pop-once fix first).
    TRAIN_SCOPE=all
    FFN_SP=1
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_joint \
      $(common_args) $(data_args) $(decay_args) $(liger_arg) \
      --schedule "${SCHEDULE:-8192:10}" \
      --global-batch-size "${GLOBAL_BATCH:-8}" --gate-sparsity \
      --num-workers 0 --log-every 1 --save-every "${SAVE_EVERY:-0}" \
      --out "$OUT/smoke_full" --metrics-file "$OUT/smoke_full/metrics.jsonl" \
      --dry-run
    ;;

  hard_backbone_only_8k)
    # Frozen router + HARD eviction: the backbone adapts to the exact geometry inference runs.
    #
    # This is the configuration the first two runs were missing. joint_8k let the router move and
    # it got worse (gate_sparsity 0.282 -> 0.359, RULER 77.62 -> 69.17); backbone_only_8k froze the
    # router but still trained under a SOFT gate, so the backbone could still spread attention
    # across keys that inference will delete. Here it cannot.
    #
    # WATCH: gate_sparsity is now a frozen router's property and should stay near 0.27 -- but the
    # number to actually judge this on is RULER at the SAME topk, because the training geometry and
    # the eval geometry finally agree.
    FREEZE_ROUTER=1
    # `${HARD_TOPK:-2048}` does NOT work here: the top-level default already set HARD_TOPK=0, so
    # the variable is non-empty and the fallback never fires -- which silently ran this mode as a
    # SOFT-gate job (no "HARD EVICTION" log line, hist_mass 0.16 rather than 0.35, peak 50.9 not
    # 56.3). Override only the default VALUE, so an explicit HARD_TOPK= from the caller still wins.
    [[ "$HARD_TOPK" == "0" ]] && HARD_TOPK=2048
    SUB="hard_bo_8k_local${N_LOCAL}_b${GATE_BUDGET}_k${HARD_TOPK}_bb${BACKBONE_LR}$([[ "$DECAY" != "0" ]] && echo _decay)"
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_joint \
      $(common_args) $(data_args) $(decay_args) $(liger_arg) \
      --schedule "${SCHEDULE:-8192:300}" \
      ${MAX_STEPS:+--max-steps $MAX_STEPS} \
      --global-batch-size "$GLOBAL_BATCH" --gate-sparsity \
      --num-workers "${WORKERS:-2}" \
      --out "$OUT/$SUB" --metrics-file "$OUT/$SUB/metrics.jsonl" \
      --save-every "${SAVE_EVERY:-100}" --log-every "${LOG_EVERY:-10}"
    ;;

  backbone_only_8k)
    # Priority-1 experiment: fixed router, backbone adapts TO the gating.
    #
    # WATCH gate_sparsity. It is now a property of a FROZEN router, so it can only move because
    # the BACKBONE changed what the scores are computed from -- the hidden states. Staying near
    # its 0.267 starting point means the backbone is adapting without dissolving the router's
    # selectivity, which is the whole hypothesis. Rising toward 0.36 as in the joint run would
    # mean the backbone re-spreads attention regardless of whether the router is training.
    FREEZE_ROUTER=1
    SUB="backbone_only_8k_local${N_LOCAL}_b${GATE_BUDGET}_bb${BACKBONE_LR}$([[ "$DECAY" != "0" ]] && echo _decay)"
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_joint \
      $(common_args) $(data_args) $(decay_args) $(liger_arg) \
      --schedule "${SCHEDULE:-8192:300}" \
      ${MAX_STEPS:+--max-steps $MAX_STEPS} \
      --global-batch-size "$GLOBAL_BATCH" --gate-sparsity \
      --num-workers "${WORKERS:-2}" \
      --out "$OUT/$SUB" --metrics-file "$OUT/$SUB/metrics.jsonl" \
      --save-every "${SAVE_EVERY:-100}" --log-every "${LOG_EVERY:-10}"
    ;;

  joint_8k)
    # 8K for 300 steps, matching the frozen run this continues so the comparison is one variable.
    SUB="joint_8k_local${N_LOCAL}_b${GATE_BUDGET}_bb${BACKBONE_LR}$([[ "$DECAY" != "0" ]] && echo _decay)"
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_joint \
      $(common_args) $(data_args) $(decay_args) $(liger_arg) \
      --schedule "${SCHEDULE:-8192:300}" \
      ${MAX_STEPS:+--max-steps $MAX_STEPS} \
      --global-batch-size "$GLOBAL_BATCH" --gate-sparsity \
      --num-workers "${WORKERS:-2}" \
      --out "$OUT/$SUB" --metrics-file "$OUT/$SUB/metrics.jsonl" \
      --save-every "${SAVE_EVERY:-100}" --log-every "${LOG_EVERY:-10}"
    ;;

  *)
    echo "usage: $0 {smoke|smoke_full|hard_backbone_only_8k|backbone_only_8k|joint_8k}" >&2
    echo "  hard_backbone_only_8k: frozen router + HARD eviction (topk 2048) -- the real thing" >&2
    echo "  backbone_only_8k: FROZEN router, backbone adapts to it (priority-1 experiment)" >&2
    echo "  smoke_full: FULL-parameter (8.19B) at 8K with FFN_SP=1 -- the OOM probe" >&2
    echo "  smoke first -- it verifies the router loaded (loss ~1.80, not 2.4+) and peak_gib" >&2
    echo "  NNODES/NODE_RANK/MASTER_ADDR for multi-node; HYBRID_SHARD adapts automatically" >&2
    exit 1
    ;;
esac
