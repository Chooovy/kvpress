#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Train the SCALAR (query-independent) indexer from the LM loss -- the SparseK/DMA arm.
#
#   scripts/train_gqa_indexer_scalar_gy.sh smoke       # 10 steps, verifies the path
#   scripts/train_gqa_indexer_scalar_gy.sh stage1_16k  # 16K, 600 steps, the A/B run
#   scripts/train_gqa_indexer_scalar_gy.sh stage1      # 8K -> 16K -> 32K curriculum
#   scripts/train_gqa_indexer_scalar_gy.sh matched     # stage1_16k at MID_DIM=1152 (param-matched)
#   scripts/train_gqa_indexer_scalar_gy.sh stage2      # sparse scope @ 64K, from stage 1
#
# To continue an interrupted stage 1, point RESUME at its checkpoint -- AdamW and the LR-schedule
# position come back and the completed steps are skipped:
#
#   RESUME=$OUT/stage1/step200.pt scripts/train_gqa_indexer_scalar_gy.sh stage1
#
# Every other flag must match the interrupted run, which --resume-from enforces for --schedule.
# Note that the data loader is NOT resumed: --take-from random reshuffles, so the resumed run sees
# different samples than the original would have from that point. The optimizer trajectory is
# continuous; the data order is not.
#
# The pairwise counterpart is scripts/train_gqa_indexer_e2e_gy.sh. Every setting except --scorer
# is matched to it so the difference is the router and nothing else; OUT differs so the two runs
# do not overwrite each other. Compare stage1_16k/step600.pt against that script's.
#
# The scalar router scores each key once from its own hidden state: O(1) per decode step and one
# score per token of cache instead of head_dim, at the cost of query-awareness.
#
# MID_DIM is the capacity knob. 256 (default) is ~1.06M params/layer against the pairwise arm's
# 4.72M; 1152 matches it exactly, which is the only setting that separates query-dependence from
# parameter count -- that is what `matched` runs.
set -euo pipefail

MODE="${1:-smoke}"
DATA_ROOT="${DATA_ROOT:-/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_256k_filtered}"
TOKENIZED="${TOKENIZED:-/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_tokenized_64k}"
MODEL="${MODEL:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B}"
OUT="${OUT:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_scalar}"

NNODES="${NNODES:-1}"
NGPU="${NGPU:-8}"
# Distinct from the distillation (29511) and pairwise-e2e (29512) scripts so all three can share a node.
MASTER_PORT="${MASTER_PORT:-29513}"

PEAK_LR="${PEAK_LR:-1e-3}"
FINAL_LR="${FINAL_LR:-5e-6}"
WARMUP_FRAC="${WARMUP_FRAC:-0.10}"
STABLE_FRAC="${STABLE_FRAC:-0.60}"

LIGER="${LIGER:-1}"
FFN_SP="${FFN_SP:-8}"

# Scalar router.
MID_DIM="${MID_DIM:-256}"
POS_SLOPE="${POS_SLOPE:-1e-6}"

# Gate. pin_mode is not optional: a gate that is flat along the key axis cancels in the softmax,
# so the model reverts to the frozen dense backbone and the LM loss is satisfied with no ranking
# learned. Verified for this router too -- no-op distance 5.6e-17 unpinned against 0.44 with sink.
PIN_MODE="${PIN_MODE:-sink}"
N_SINK="${N_SINK:-4}"
COMPRESSION_RATIO="${COMPRESSION_RATIO:-0.5}"

MAX_STEPS="${MAX_STEPS:-600}"

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
ffn_sp_arg() { [[ "$FFN_SP" != "1" ]] && echo "--ffn-sp-size $FFN_SP"; }
# gate_scale rising says nothing about sparsity -- a layer can lean hard on a gate that still
# spreads its mass over every key. ~1.0 means nothing learned; falling towards 0 is concentration.
gate_sparsity_arg() { [[ "${GATE_SPARSITY:-1}" != "0" ]] && echo "--gate-sparsity"; }

scalar_args() { echo "--scorer scalar --scalar-mid-dim $MID_DIM --scalar-pos-slope $POS_SLOPE"; }

data_args() {
  if [[ -f "$TOKENIZED/index.json" ]]; then
    echo "--tokenized $TOKENIZED --subsets ${SUBSETS:-2e16 2e17}"
  else
    echo "--subsets ${SUBSETS:-2e15 2e16 synth_cwe synth_rex}"
  fi
}

case "$MODE" in
  smoke)
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_e2e \
      --data-root "$DATA_ROOT" --model "$MODEL" $(scalar_args) \
      --subsets 8k_32k --schedule "${SCHEDULE:-16384:50}" \
      --stage dense --pin-mode "$PIN_MODE" --n-sink "$N_SINK" $(liger_arg) $(ffn_sp_arg) \
      --num-workers 0 --log-every 1 --save-every 0 \
      --out "$OUT/smoke" --dry-run
    ;;

  stage1_16k|matched)
    # 16K for 600 steps on distillation's exact 8192:300,16384:300,32768:900 curve, truncated by
    # MAX_STEPS -- so the LR over 0..600 is identical (warmup 150, then flat peak) and the O(L^2)
    # 32K stage is never built. FFN_SP=8 on 8 GPUs is ONE data-parallel replica, so GLOBAL_BATCH=8
    # accumulates to match the 8 sequences/step the pairwise run sees.
    [[ "$MODE" == "matched" ]] && MID_DIM="${MID_DIM_MATCHED:-1152}"
    SUB="${MODE}_mid${MID_DIM}"
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_e2e \
      --data-root "$DATA_ROOT" --model "$MODEL" $(data_args) $(scalar_args) \
      --schedule "${SCHEDULE:-8192:300,16384:300,32768:900}" \
      --max-steps "${MAX_STEPS:-600}" \
      --stage dense --pin-mode "$PIN_MODE" --n-sink "$N_SINK" $(liger_arg) \
      --ffn-sp-size "${FFN_SP_16K:-8}" \
      --global-batch-size "${GLOBAL_BATCH:-8}" $(gate_sparsity_arg) \
      --compression-ratio "$COMPRESSION_RATIO" \
      --peak-lr "$PEAK_LR" --final-lr "$FINAL_LR" \
      --warmup-frac "$WARMUP_FRAC" --stable-frac "$STABLE_FRAC" \
      --batch-size 1 --take-from random --shuffle-buffer 64 \
      --num-workers "${WORKERS:-2}" \
      --out "$OUT/$SUB" --metrics-file "$OUT/$SUB/metrics.jsonl" \
      --save-every "${SAVE_EVERY:-100}" --log-every "${LOG_EVERY:-10}"
    ;;

  stage1)
    # Full curriculum, full scope. Full scope is what gives every key's gate a content-dependent
    # gradient; under sparse scope an unselected key moves only through the normalizer.
    #
    # RESUME=<ckpt> continues an interrupted run in place: AdamW moments and the LR-schedule
    # position come back, and the steps already done are skipped. Deliberately a flag on THIS
    # case rather than a separate mode -- --resume-from checks the checkpoint's recorded
    # --schedule against the one passed, so a copy of this block that drifted by one flag would
    # be rejected here (or, worse, resume onto a different curve). One code path cannot drift.
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_e2e \
      --data-root "$DATA_ROOT" --model "$MODEL" $(data_args) $(scalar_args) \
      --schedule "${SCHEDULE:-8192:300,16384:300,32768:900}" \
      ${MAX_STEPS:+--max-steps $MAX_STEPS} \
      --stage dense --pin-mode "$PIN_MODE" --n-sink "$N_SINK" $(liger_arg) $(ffn_sp_arg) \
      --global-batch-size "${GLOBAL_BATCH:-8}" $(gate_sparsity_arg) \
      --compression-ratio "$COMPRESSION_RATIO" \
      --peak-lr "$PEAK_LR" --final-lr "$FINAL_LR" \
      --warmup-frac "$WARMUP_FRAC" --stable-frac "$STABLE_FRAC" \
      --batch-size "${BATCH_SIZE:-1}" --take-from random --shuffle-buffer 64 \
      --num-workers "${WORKERS:-2}" ${RESUME:+--resume-from "$RESUME"} \
      --out "$OUT/stage1" --metrics-file "$OUT/stage1/metrics.jsonl" \
      --save-every "${SAVE_EVERY:-100}" --log-every "${LOG_EVERY:-10}"
    ;;

  ablate)
    # Pinning OFF. Expected to look GOOD on the loss curve and BAD at eval -- the gate can flatten
    # into a no-op and recover the frozen dense model. Watch gate_scales, not the loss.
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_e2e \
      --data-root "$DATA_ROOT" --model "$MODEL" $(data_args) $(scalar_args) \
      --schedule "${SCHEDULE:-8192:300,16384:300,32768:300}" \
      ${MAX_STEPS:+--max-steps $MAX_STEPS} \
      --stage dense --pin-mode none $(liger_arg) $(ffn_sp_arg) \
      --global-batch-size "${GLOBAL_BATCH:-8}" $(gate_sparsity_arg) \
      --compression-ratio "$COMPRESSION_RATIO" \
      --peak-lr "$PEAK_LR" --final-lr "$FINAL_LR" \
      --warmup-frac "$WARMUP_FRAC" --stable-frac "$STABLE_FRAC" \
      --batch-size "${BATCH_SIZE:-1}" --take-from random --shuffle-buffer 64 \
      --num-workers "${WORKERS:-2}" \
      --out "$OUT/ablate_nopin" --metrics-file "$OUT/ablate_nopin/metrics.jsonl" \
      --save-every "${SAVE_EVERY:-100}" --log-every "${LOG_EVERY:-10}"
    ;;

  linear)
    # MID_DIM=0: SparseK's plain linear score, the papers' actual form. Isolates how much of the
    # arm's performance is the MLP rather than query-independence.
    MID_DIM=0
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_e2e \
      --data-root "$DATA_ROOT" --model "$MODEL" $(data_args) $(scalar_args) \
      --schedule "${SCHEDULE:-8192:300,16384:300,32768:900}" \
      --max-steps "${MAX_STEPS:-600}" \
      --stage dense --pin-mode "$PIN_MODE" --n-sink "$N_SINK" $(liger_arg) \
      --ffn-sp-size "${FFN_SP_16K:-8}" \
      --global-batch-size "${GLOBAL_BATCH:-8}" $(gate_sparsity_arg) \
      --compression-ratio "$COMPRESSION_RATIO" \
      --peak-lr "$PEAK_LR" --final-lr "$FINAL_LR" \
      --warmup-frac "$WARMUP_FRAC" --stable-frac "$STABLE_FRAC" \
      --batch-size 1 --take-from random --shuffle-buffer 64 \
      --num-workers "${WORKERS:-2}" \
      --out "$OUT/linear" --metrics-file "$OUT/linear/metrics.jsonl" \
      --save-every "${SAVE_EVERY:-100}" --log-every "${LOG_EVERY:-10}"
    ;;

  stage2)
    # Sparse scope at 64K. --pin-mode is not passed: the forward is already restricted to the
    # router's top-k, so there is no dense fallback to pin against and the trainer rejects it.
    # --init-from checks the checkpoint's recorded scorer, so a pairwise checkpoint is refused
    # here rather than silently loading nothing.
    INIT="${INIT:-$OUT/stage1_16k_mid${MID_DIM}/final.pt}"
    if [[ ! -f "$INIT" ]]; then
      echo "stage 2 needs stage 1's checkpoint at $INIT" >&2
      echo "  run stage1_16k first, or set INIT= (the matched/linear runs write elsewhere)" >&2
      exit 1
    fi
    EXTRA=()
    if [[ -f "$TOKENIZED/index.json" ]]; then
      EXTRA+=(--tokenized "$TOKENIZED")
    fi
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_e2e \
      --data-root "$DATA_ROOT" --model "$MODEL" "${EXTRA[@]}" $(scalar_args) \
      --subsets 2e16 2e17 \
      --schedule "65536:${STEPS:-600}" \
      --stage sparse --topk "${TOPK:-512}" \
      --force-local "${FORCE_LOCAL:-64}" --force-sink "${FORCE_SINK:-4}" $(liger_arg) $(ffn_sp_arg) \
      --global-batch-size "${GLOBAL_BATCH:-8}" $(gate_sparsity_arg) \
      --compression-ratio "$COMPRESSION_RATIO" \
      --peak-lr "$PEAK_LR" --final-lr "$FINAL_LR" \
      --warmup-frac "$WARMUP_FRAC" --stable-frac "$STABLE_FRAC" \
      --batch-size 1 --take-from random --shuffle-buffer 32 \
      --num-workers "${WORKERS:-2}" \
      --init-from "$INIT" \
      --out "$OUT/stage2" --metrics-file "$OUT/stage2/metrics.jsonl" \
      --save-every "${SAVE_EVERY:-100}" --log-every "${LOG_EVERY:-10}"
    ;;

  *)
    echo "usage: $0 {smoke|stage1_16k|matched|stage1|linear|ablate|stage2}" >&2
    echo "  (tokenize with scripts/train_gqa_indexer.sh tokenize -- the corpus is shared)" >&2
    exit 1
    ;;
esac
