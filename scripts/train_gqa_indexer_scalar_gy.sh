#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Train the SCALAR (query-independent) indexer from the LM loss -- the SparseK/DMA arm.
#
#   scripts/train_gqa_indexer_scalar_gy.sh smoke       # 10 steps, verifies the path
#   scripts/train_gqa_indexer_scalar_gy.sh delta_smoke # 20 steps at 8K, reads the delta diagnostics
#   scripts/train_gqa_indexer_scalar_gy.sh stage1_16k  # 16K, 600 steps, the A/B run
#   scripts/train_gqa_indexer_scalar_gy.sh stage1      # 8K -> 16K -> 32K curriculum
#   scripts/train_gqa_indexer_scalar_gy.sh matched     # stage1_16k at MID_DIM=1152 (param-matched)
#   scripts/train_gqa_indexer_scalar_gy.sh stage2      # sparse scope @ 64K, from stage 1
#
# DELTA=1 switches stage1_16k / stage1 / linear onto the delta-weighted loss (see the DELTA block
# below), writing to a `_delta<lambda>` sibling directory so it cannot overwrite the plain run it is
# compared against:
#
#   DELTA=1 scripts/train_gqa_indexer_scalar_gy.sh delta_smoke   # check the diagnostics FIRST
#   DELTA=1 scripts/train_gqa_indexer_scalar_gy.sh stage1_16k    # -> $OUT/stage1_16k_mid256_delta0.1
#   DELTA=1 DELTA_LAMBDA=1.0 scripts/train_gqa_indexer_scalar_gy.sh stage1_16k
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

# Delta-weighted loss (DELTA=1). Off by default, so existing invocations are unchanged.
#
# The plain objective is `mean_t L_t`, which spends the router's gradient equally on every position.
# A high L_t is either irreducible entropy ("he walked into the ___"), missing knowledge, or a
# retrieval failure -- only the last is the router's to fix. Weighting by L_t itself (a power mean,
# an LSE) promotes the FIRST hardest of all: irreducible-entropy positions sit permanently at the
# top of the loss distribution and their loss does not move with the support at all, so their
# gradient is noise with respect to the router. The gap against the same model run densely isolates
# the third case:
#
#   w_t  = clamp(L_t^dense - L_t^sparse, 0) + DELTA_LAMBDA      (detached)
#   loss = sum_t w_t L_t^sparse / sum_t w_t
#
# Why the gap cannot be the loss itself: the dense pass is frozen, so d/dtheta (dense - sparse) =
# d/dtheta (-sparse) and "train on the gap" is bit-identical to the present objective -- it only
# shifts the reported number towards zero while paying for a second forward pass.
#
# DELTA_LAMBDA is a floor, not a tolerance. At 0, a position the router already matches dense on
# gets weight 0 and stops receiving gradient, so nothing maintains it (measured: half the positions
# leave the objective). Large values recover the ordinary mean continuously -- verified equal to
# the plain path to 0.000e+00 at lambda=1e6 -- so it interpolates between the two objectives and
# the sweep has a known endpoint.
#
# WATCH `delta_positive_frac` IN --metrics-file, NOT THE LOSS. Near zero means dense and sparse
# agree everywhere, every weight falls back to lambda, and this is the ordinary mean with a wasted
# forward pass. `weight_participation` ~1.0 says the same from the weights' side.
#
# Cost: one extra forward per step, under no_grad with no gate, so it stores no activation graph --
# compute (order 50% of a step) rather than memory. Per-token loss needs the logits, which is what
# --liger's fused CE avoids materializing, so lm_head is applied in DELTA_LOGIT_CHUNK-row blocks.
# Sound because `sum w_t L_t` is LINEAR in L_t and decomposes across blocks; a power mean would
# need every L_t before its outer exponent.
#
# NOT MEASURED: peak memory at 16K. The chunking bounds the logits, but cross_entropy's backward
# saves its softmax and whether that is freed per block was never verified on a GPU. 16K already
# runs at ~89.6 GiB of 95 (HANDOFF_exact_k_subset.md), so there may be no headroom. Run
# `delta_smoke` (8K) first and read peak_gib from the metrics before taking this to stage1_16k.
# Whether DELTA was named explicitly, captured BEFORE the default is applied. Needed because the
# DELTA default below is currently 1, so a plain `LONGCE=1 ... stage1_16k` would otherwise trip the
# mutual-exclusion check on a value the caller never asked for.
DELTA_EXPLICIT="${DELTA+yes}"
DELTA="${DELTA:-1}"
DELTA_LAMBDA="${DELTA_LAMBDA:-0.1}"
DELTA_LOGIT_CHUNK="${DELTA_LOGIT_CHUNK:-8192}"

# LongCE-weighted loss (LONGCE=1). Off by default.
#
# Same objective SHAPE as DELTA -- `sum_t w_t L_t / sum_t w_t` -- but the weight is a different
# quantity, and that difference is the entire point:
#
#   w_t = min(exp(L_t^short - L_t^long), LONGCE_GAMMA)      (from a CACHE, detached)
#
# DELTA compared dense vs sparse. That gap turned out to be strongly rank-correlated with the loss
# it multiplies, so the objective degenerated into a power mean over the top ~15% of positions --
# exactly where irreducible entropy lives and routing cannot help. Result: RULER 66.24 -> ~35.2,
# with a DIRECTED collapse (niah_single_2 100 -> 19.7) because retrieval tokens are EASY under a
# dense run and so were never in that top 15%.
#
# LongCE compares SHORT context vs LONG context instead. Measured on this corpus before any
# training (evaluation/probe_longce_key_tokens.py, 8 docs x 8K/16K/32K):
#
#   spearman(w, L_long)   = -0.001 .. -0.029   <- decorrelated, which is what DELTA failed
#   weight_participation  =  0.66 .. 0.87      <- broad, against DELTA's 0.13-0.18
#   key_L - all_L         = -1.7               <- key tokens are EASIER than average under long ctx
#
# K=1024 rather than the paper's 4096 default: their own ablation (Table 7) has K=1k beating K=4k on
# RULER (55.9 vs 49.7 at 200 steps) at lower cost, and 1024 gave the largest signal here.
#
# COSTS NOTHING EXTRA PER STEP, unlike DELTA. The backbone is frozen, so L^short/L^long do not
# depend on the router -- the weights are a property of the DATA and are precomputed once:
#
#   scripts/precompute_longce_weights.sh 16384       # ~1 GPU-hour across 8 GPUs
#   LONGCE=1 scripts/train_gqa_indexer_scalar_gy.sh longce_smoke     # 20 steps, READ THE NUMBERS
#   LONGCE=1 scripts/train_gqa_indexer_scalar_gy.sh stage1_16k       # -> $OUT/stage1_16k_mid256_longce
#
# So peak_gib should MATCH the plain run, not exceed it. If it does not, the cache is not being used.
#
# WATCH `weight_participation` in --metrics-file, NOT the loss. If it lands at 0.13-0.18 again this
# is repeating DELTA's failure and should be stopped; ~1.0 means the weighting is inert (check
# longce_cache_miss_frac, which says how many drawn documents had no cached weights).
LONGCE="${LONGCE:-0}"
LONGCE_TRUNC="${LONGCE_TRUNC:-1024}"
LONGCE_GAMMA="${LONGCE_GAMMA:-5.0}"
LONGCE_CACHE="${LONGCE_CACHE:-/apdcephfs_gy8/share_303843174/guhao/datasets/longce_weights_16k}"

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

# Emits nothing at DELTA=0, so the flags never reach the trainer and the default path is byte for
# byte the one that produced the existing checkpoints.
#
# The trailing `|| true` is load-bearing, unlike in liger_arg above: these are used inside a
# variable ASSIGNMENT (`SUB="...$(delta_suffix)"`), and under `set -e` a command substitution that
# exits non-zero fails the whole assignment and kills the script -- silently, with no output. A
# bare `[[ cond ]] && echo` returns 1 when the condition is false, which is exactly that case.
delta_args() {
  [[ "$DELTA" != "0" ]] && \
    echo "--delta-weight --delta-lambda $DELTA_LAMBDA --delta-logit-chunk $DELTA_LOGIT_CHUNK"
  return 0
}

# OUT subdirectory suffix, so a delta run cannot overwrite the plain one it is compared against --
# and so the directory name records lambda, which is the variable of the sweep.
delta_suffix() {
  [[ "$DELTA" != "0" ]] && echo "_delta${DELTA_LAMBDA}"
  return 0
}

# The LongCE counterparts. `return 0` for the same load-bearing reason as delta_args/delta_suffix:
# both are used inside variable ASSIGNMENTS, and under `set -e` a command substitution that exits
# non-zero kills the whole script with no output at all.
#
# --take-from head is NOT optional and is passed here rather than left to the mode blocks: the cache
# stores one weight vector per document and each stage reads a PREFIX of it, which is only valid
# because the losses are causal. `random` would pair position i's weight with a different token, and
# the trainer rejects the combination for that reason.
longce_args() {
  [[ "$LONGCE" != "0" ]] && \
    echo "--longce-weights $LONGCE_CACHE --take-from head"
  return 0
}

longce_suffix() {
  [[ "$LONGCE" != "0" ]] && echo "_longce"
  return 0
}

# The two weightings are different objectives over the same loss; running both would silently produce
# a third that neither was validated as.
#
# LONGCE=1 turns DELTA off implicitly, because DELTA's default above is 1 rather than 0 -- so
# requiring `DELTA=0 LONGCE=1` would make the documented invocation fail on a value the caller never
# set. An EXPLICIT `DELTA=1 LONGCE=1` is still an error: that is a real contradiction rather than a
# default leaking through.
if [[ "$LONGCE" != "0" ]]; then
  if [[ -n "$DELTA_EXPLICIT" && "$DELTA" != "0" ]]; then
    echo "DELTA=1 and LONGCE=1 are mutually exclusive: they are two different weightings of the" >&2
    echo "same loss, and applying both would produce a third that neither was validated as." >&2
    exit 1
  fi
  DELTA=0
fi
if [[ "$LONGCE" != "0" && ! -d "$LONGCE_CACHE" ]]; then
  echo "LONGCE=1 needs a precomputed weight cache at $LONGCE_CACHE" >&2
  echo "  build it first:  scripts/precompute_longce_weights.sh 16384" >&2
  echo "  or point LONGCE_CACHE= at an existing one" >&2
  exit 1
fi

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

  delta_smoke)
    # RUN THIS FIRST when trying DELTA. 20 steps at 8K, metrics every step, and deliberately NOT
    # --dry-run: the point is to read real numbers off --metrics-file.
    #
    # Two things to check before committing to 600 steps:
    #   delta_positive_frac -- near zero means the weighting has nothing to work with and the extra
    #                          forward pass buys nothing. Stop here if so.
    #   peak_gib            -- the memory question the header flags as unmeasured. Compare against a
    #                          plain 8K run; a thin margin here means 16K will not fit.
    DELTA=1
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_e2e \
      --data-root "$DATA_ROOT" --model "$MODEL" $(data_args) $(scalar_args) $(delta_args) \
      --schedule "${SCHEDULE:-16384:20}" \
      --stage dense --pin-mode "$PIN_MODE" --n-sink "$N_SINK" $(liger_arg) $(ffn_sp_arg) \
      --global-batch-size "${GLOBAL_BATCH:-8}" $(gate_sparsity_arg) \
      --compression-ratio "$COMPRESSION_RATIO" \
      --peak-lr "$PEAK_LR" --final-lr "$FINAL_LR" \
      --warmup-frac "$WARMUP_FRAC" --stable-frac "$STABLE_FRAC" \
      --batch-size 1 --take-from random --shuffle-buffer 64 \
      --num-workers "${WORKERS:-2}" \
      --out "$OUT/delta_smoke" --metrics-file "$OUT/delta_smoke/metrics.jsonl" \
      --save-every 0 --log-every 1
    ;;

  longce_smoke)
    # RUN THIS FIRST when trying LONGCE. 20 steps at 8K, metrics every step, deliberately NOT
    # --dry-run: the point is to read real numbers off --metrics-file.
    #
    # Three things to check before committing to 600 steps:
    #   weight_participation      -- 0.66-0.87 is what the offline probe measured. If it lands at
    #                                DELTA's 0.13-0.18 this is repeating that failure: STOP.
    #                                ~1.0 means the weighting is inert.
    #   longce_cache_miss_frac    -- documents drawn with no cached weights, which fall back to
    #                                weight 1. High means the run is mostly the plain objective.
    #   peak_gib                  -- should MATCH a plain 8K run, since there is no second forward
    #                                pass. If it is higher, something is recomputing rather than
    #                                reading the cache.
    LONGCE=1
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_e2e \
      --data-root "$DATA_ROOT" --model "$MODEL" $(data_args) $(scalar_args) $(longce_args) \
      --schedule "${SCHEDULE:-8192:20}" \
      --stage dense --pin-mode "$PIN_MODE" --n-sink "$N_SINK" $(liger_arg) $(ffn_sp_arg) \
      --global-batch-size "${GLOBAL_BATCH:-8}" $(gate_sparsity_arg) \
      --compression-ratio "$COMPRESSION_RATIO" \
      --peak-lr "$PEAK_LR" --final-lr "$FINAL_LR" \
      --warmup-frac "$WARMUP_FRAC" --stable-frac "$STABLE_FRAC" \
      --batch-size 1 --shuffle-buffer 64 \
      --num-workers "${WORKERS:-2}" \
      --out "$OUT/longce_smoke" --metrics-file "$OUT/longce_smoke/metrics.jsonl" \
      --save-every 0 --log-every 1
    ;;

  stage1_16k|matched)
    # 16K for 600 steps on distillation's exact 8192:300,16384:300,32768:900 curve, truncated by
    # MAX_STEPS -- so the LR over 0..600 is identical (warmup 150, then flat peak) and the O(L^2)
    # 32K stage is never built. FFN_SP=8 on 8 GPUs is ONE data-parallel replica, so GLOBAL_BATCH=8
    # accumulates to match the 8 sequences/step the pairwise run sees.
    [[ "$MODE" == "matched" ]] && MID_DIM="${MID_DIM_MATCHED:-1152}"
    SUB="${MODE}_mid${MID_DIM}$(delta_suffix)$(longce_suffix)"
    # --take-from: `random` normally, but LONGCE requires `head` (cached weights are per-position
    # prefixes), so longce_args supplies it and this default drops out. Passing both would make
    # argparse take the LAST one, which is why it is a variable rather than a second flag.
    TAKE_FROM_ARG="--take-from random"
    [[ "$LONGCE" != "0" ]] && TAKE_FROM_ARG=""
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_e2e \
      --data-root "$DATA_ROOT" --model "$MODEL" $(data_args) $(scalar_args) $(delta_args) $(longce_args) \
      --schedule "${SCHEDULE:-8192:300,16384:300,32768:900}" \
      --max-steps "${MAX_STEPS:-600}" \
      --stage dense --pin-mode "$PIN_MODE" --n-sink "$N_SINK" $(liger_arg) \
      --ffn-sp-size "${FFN_SP_16K:-8}" \
      --global-batch-size "${GLOBAL_BATCH:-8}" $(gate_sparsity_arg) \
      --compression-ratio "$COMPRESSION_RATIO" \
      --peak-lr "$PEAK_LR" --final-lr "$FINAL_LR" \
      --warmup-frac "$WARMUP_FRAC" --stable-frac "$STABLE_FRAC" \
      --batch-size 1 $TAKE_FROM_ARG --shuffle-buffer 64 \
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
    STAGE1_SUB="stage1$(delta_suffix)"
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_e2e \
      --data-root "$DATA_ROOT" --model "$MODEL" $(data_args) $(scalar_args) $(delta_args) \
      --schedule "${SCHEDULE:-8192:300,16384:300,32768:900}" \
      ${MAX_STEPS:+--max-steps $MAX_STEPS} \
      --stage dense --pin-mode "$PIN_MODE" --n-sink "$N_SINK" $(liger_arg) $(ffn_sp_arg) \
      --global-batch-size "${GLOBAL_BATCH:-8}" $(gate_sparsity_arg) \
      --compression-ratio "$COMPRESSION_RATIO" \
      --peak-lr "$PEAK_LR" --final-lr "$FINAL_LR" \
      --warmup-frac "$WARMUP_FRAC" --stable-frac "$STABLE_FRAC" \
      --batch-size "${BATCH_SIZE:-1}" --take-from random --shuffle-buffer 64 \
      --num-workers "${WORKERS:-2}" ${RESUME:+--resume-from "$RESUME"} \
      --out "$OUT/$STAGE1_SUB" --metrics-file "$OUT/$STAGE1_SUB/metrics.jsonl" \
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
    LINEAR_SUB="linear$(delta_suffix)"
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_e2e \
      --data-root "$DATA_ROOT" --model "$MODEL" $(data_args) $(scalar_args) $(delta_args) \
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
      --out "$OUT/$LINEAR_SUB" --metrics-file "$OUT/$LINEAR_SUB/metrics.jsonl" \
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
    echo "usage: $0 {smoke|delta_smoke|longce_smoke|stage1_16k|matched|stage1|linear|ablate|stage2}" >&2
    echo "  DELTA=1 switches stage1_16k/stage1/linear onto the delta-weighted loss;" >&2
    echo "  run delta_smoke first and read delta_positive_frac + peak_gib from its metrics" >&2
    echo "  (tokenize with scripts/train_gqa_indexer.sh tokenize -- the corpus is shared)" >&2
    exit 1
    ;;
esac
