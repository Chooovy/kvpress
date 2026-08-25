#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Train the PREFIX indexer -- the "score each key from its whole prefix, stay query-independent" arm.
#
#   scripts/train_gqa_indexer_prefix_gy.sh smoke          # 10 steps, verifies the path
#   scripts/train_gqa_indexer_prefix_gy.sh identity       # THE FIRST RUN. 2 steps, asserts nesting
#   scripts/train_gqa_indexer_prefix_gy.sh variance       # prefill-only diagnostic, no training
#   scripts/train_gqa_indexer_prefix_gy.sh stage1_16k     # 16K, 600 steps, the A/B run
#   scripts/train_gqa_indexer_prefix_gy.sh stage1         # 8K -> 16K -> 32K curriculum
#   scripts/train_gqa_indexer_prefix_gy.sh cross_replay   # the query-independent objective
#   scripts/train_gqa_indexer_prefix_gy.sh random_init    # ablation: branch NOT zero-initialized
#   scripts/train_gqa_indexer_prefix_gy.sh linear         # MID_DIM=0, prefix branch on a linear score
#   scripts/train_gqa_indexer_prefix_gy.sh stage2         # sparse scope @ 64K, from stage 1
#
# WHAT THIS ARM IS
#
# The score reads `softmax(q_j K_{<=j}^T / sqrt(d)) V_{<=j}` where q/k/v are the indexer's own
# projections and the query comes from the KEY TOKEN's own hidden state. So it is NOT query-aware:
# s_j is fixed the moment j arrives, exactly as in the scalar arm, and the top-k stays irreversible
# -- which is what makes a dropped KV entry safe to free. What it buys is the feature set: the scalar
# arm sees only h_j, this one sees j's whole prefix.
#
# READ THIS BEFORE INTERPRETING ANY RESULT
#
# The bar is NOT "better than nothing". proxy_exp/HANDOFF.md §12.1 measured a per-KV-head router
# (which is what both arms already are -- §13) at rel_L2 0.0313 at keep 25%, i.e. 84% of the
# oracle_qi-to-recency band, against prefill_mean's 0.0341. So:
#
#   * the headroom inside this hypothesis class is the last 16%, and §11.4 measures the ACHIEVABLE
#     bound for anything frozen at prefill as ~1.9x looser than oracle_qi anyway (score on the first
#     64 future queries, pay damage on the last 64 -> 2.54x, 144/144 cells worse).
#   * two weakened forms of "let the per-key score read the prefix" have ALREADY LOST here: the
#     recurrent state z (§8.3, §10.8 -- `h+z` vs `h+z_shuffled` at matched width is +0.0132, t=0.79,
#     p=0.388, and the shuffled control WINS at every width at L14) and every redundancy score (§9.1
#     -- top-25 overlap 0.264 against a 0.25 chance floor).
#
# What is different here, and why it is worth one run: §9.5 refuted ONE set-aware signal (linear
# reconstructability), not set-awareness; §10.5 showed oracle_qi is beatable by >=5% with a better
# SET; and §12.5 found the prefix-neighbour signal does exist, with the OPPOSITE sign to the obvious
# one (`nn_novelty_neg` recovers 42% of the band -- "keep the keys whose earlier neighbours already
# point the same way, evict the novel ones").
#
# So run `identity`, then `variance`, then stage1_16k, and judge on the shuffle control -- not the
# loss curve. If the delta is <= 0, this arm is the third state-based feature to fail its own control
# and the answer is to stop, not to tune.
#
# THE A/B IS SINGLE-VARIABLE, AND THAT IS LOAD-BEARING
#
# PrefixIndexer subclasses ScalarIndexer and reuses its parameters under the same names, so with
# PREFIX_ZERO_INIT=1 (the default) the score starts BIT-IDENTICAL -- verified maxdiff 0.000e+00 at
# both MID_DIM 0 and 64. Every flag below is therefore matched to
# scripts/train_gqa_indexer_scalar_gy.sh, and MID_DIM in particular must NOT drift from it: that
# script's header records how a 256-vs-0 mismatch made an earlier A/B uninterpretable by confounding
# "which objective" with "26x the capacity". Compare stage1_16k/step600.pt against that script's.
#
# The zero init is an escapable saddle, not a dead start. Measured: at step 1 w_a's gradient norm is
# 2.07e2 while w_pq/w_pv are exactly 0; at step 2 they are 1.01e3 and 1.28e4. So the branch is live
# from the second step, and `dL/dscore` is never proportional to the parameter the way gate_scale's
# is. `random_init` runs the other side of it -- a different experiment, since the arm no longer
# nests inside the scalar one.
set -euo pipefail

MODE="${1:-smoke}"
# Same defaults as the sibling scripts, so all arms read one corpus.
DATA_ROOT="${DATA_ROOT:-/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_256k_filtered}"
TOKENIZED="${TOKENIZED:-/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_tokenized_64k}"
MODEL="${MODEL:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B}"
OUT="${OUT:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_prefix}"

NNODES="${NNODES:-1}"
NGPU="${NGPU:-8}"
# 29511 distill, 29512 e2e, 29513 scalar, 29514 cross-replay, 29515 here.
MASTER_PORT="${MASTER_PORT:-29515}"

# Matched to the scalar and e2e scripts.
PEAK_LR="${PEAK_LR:-1e-3}"
FINAL_LR="${FINAL_LR:-5e-6}"
WARMUP_FRAC="${WARMUP_FRAC:-0.10}"
STABLE_FRAC="${STABLE_FRAC:-0.60}"

LIGER="${LIGER:-1}"
FFN_SP="${FFN_SP:-8}"

# Readout capacity. MUST match scripts/train_gqa_indexer_scalar_gy.sh's default -- see the header.
MID_DIM="${MID_DIM:-256}"
POS_SLOPE="${POS_SLOPE:-1e-6}"

# The prefix branch. PREFIX_VALUE_DIM is the only number with a deployment cost attached: it is what
# a decode-time indexer cache would pay per token per layer (alongside PREFIX_HEAD_DIM for the keys),
# against n_heads scalars for the scalar arm. At 128/128 on Qwen3-8B that is 512 B/token/layer in
# bf16, which at keep 25% would be +50% on top of the compressed KV -- so a nominal 25% budget is
# really 37.5%. Training and prefill-time eviction do NOT pay it (the readout is consumed
# immediately), which is why this script exists before any decode work: measure whether the signal
# is there before paying for it. If it is, the next question is 32/32, not 128/128.
PREFIX_HEAD_DIM="${PREFIX_HEAD_DIM:-128}"
PREFIX_VALUE_DIM="${PREFIX_VALUE_DIM:-128}"
PREFIX_ZERO_INIT="${PREFIX_ZERO_INIT:-1}"

# Gate. Not optional, and for a reason that applies to this arm exactly as much as to the scalar one:
# a gate flat along the key axis adds a per-row constant, which cancels in the softmax, so the model
# reverts to the frozen dense backbone and the LM loss is satisfied with NO ranking learned
# (measured no-op distance 5.6e-17 unpinned against 0.44 with a sink pin).
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
gate_sparsity_arg() { [[ "${GATE_SPARSITY:-1}" != "0" ]] && echo "--gate-sparsity"; }

# --scorer prefix plus the branch geometry. Zero-init is the DEFAULT in the trainer, so the flag is
# only ever passed to turn it OFF -- which keeps the nesting property from depending on this script.
prefix_args() {
  local a="--scorer prefix --scalar-mid-dim $MID_DIM --scalar-pos-slope $POS_SLOPE"
  a="$a --prefix-head-dim $PREFIX_HEAD_DIM --prefix-value-dim $PREFIX_VALUE_DIM"
  [[ "$PREFIX_ZERO_INIT" == "0" ]] && a="$a --no-prefix-zero-init"
  echo "$a"
}

data_args() {
  if [[ -f "$TOKENIZED/index.json" ]]; then
    echo "--tokenized $TOKENIZED --subsets ${SUBSETS:-2e16 2e17}"
  else
    echo "--subsets ${SUBSETS:-2e15 2e16 synth_cwe synth_rex}"
  fi
}

case "$MODE" in
  identity)
    # RUN THIS FIRST, and read its output rather than its exit code alone.
    #
    # Asserts the property the whole comparison rests on: with the branch zeroed, this arm's score is
    # bit-identical to the scalar arm's, so "reads the prefix" is the only variable. Unit-tested at
    # toy scale (test_gqa_indexer_prefix.py::test_zero_init_is_bit_identical_to_scalar) -- this
    # re-checks it on the REAL model, in bf16, at a real sequence length, because that is where a
    # dtype or a norm could break the nesting without breaking the test.
    #
    # Also prints the parameter counts, so the capacity difference against the scalar arm is on the
    # record before any loss is compared.
    exec python -m scripts.prefix_indexer_identity_check \
      --model "$MODEL" --mid-dim "$MID_DIM" \
      --prefix-head-dim "$PREFIX_HEAD_DIM" --prefix-value-dim "$PREFIX_VALUE_DIM" \
      --seq-len "${SEQ_LEN:-4096}" --layers "${LAYERS:-0 17 35}"
    ;;

  variance)
    # THE diagnostic to run before training, not after.
    #
    # `softmax(...) V` is a convex combination of {v_i}, so a_j lies in their convex hull: as the
    # attention spreads, a_j -> mean(v) and the SPREAD OF SCORES ACROSS KEYS shrinks with position.
    # Top-k compares across positions, so that surfaces as a systematic position bias with pos_slope
    # dominating the tail -- and at the END of training the symptom is "the router just learned
    # recency", which is many confused steps away from the cause.
    #
    # Measured at toy scale on random input: the raw readout's across-j spread falls 4.7x from the
    # first 512 positions to the last (0.00403 -> 0.00086, ||a|| 0.584 -> 0.153), while the FINAL
    # score's per-bin variance stays flat (ratio 0.970) because the W_in norm(h_j) residual holds it
    # up. This mode checks that the residual still holds on REAL hidden states, whose ||v|| drifts
    # with depth in a way random input does not reproduce.
    #
    # A monotone decay across bins is the failure mode. If it appears, shrink PREFIX_VALUE_DIM or
    # reconsider the readout -- do not start a 600-step run.
    exec python -m scripts.prefix_indexer_identity_check --variance-only \
      --model "$MODEL" --mid-dim "$MID_DIM" \
      --prefix-head-dim "$PREFIX_HEAD_DIM" --prefix-value-dim "$PREFIX_VALUE_DIM" \
      --seq-len "${SEQ_LEN:-8192}" --layers "${LAYERS:-0 17 35}" \
      --data-root "$DATA_ROOT" ${TOKENIZED:+--tokenized "$TOKENIZED"}
    ;;

  smoke)
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_e2e \
      --data-root "$DATA_ROOT" --model "$MODEL" $(prefix_args) \
      --subsets 8k_32k --schedule "${SCHEDULE:-16384:50}" \
      --stage dense --pin-mode "$PIN_MODE" --n-sink "$N_SINK" $(liger_arg) $(ffn_sp_arg) \
      --num-workers 0 --log-every 1 --save-every 0 \
      --out "$OUT/smoke" --dry-run
    ;;

  stage1_16k|random_init|linear)
    # 16K for 600 steps on the same 8192:300,16384:300,32768:900 curve as the scalar and e2e arms,
    # truncated by MAX_STEPS -- so the LR over 0..600 is identical (warmup 150, then flat peak) and
    # the O(L^2) 32K stage is never built. FFN_SP=8 on 8 GPUs is ONE data-parallel replica, so
    # GLOBAL_BATCH=8 accumulates to the same 8 sequences/step the sibling runs see.
    #
    # random_init: PREFIX_ZERO_INIT=0. The arm no longer nests inside the scalar one, so this is a
    # different experiment -- run it to ask whether the zero start is holding the branch back, NOT as
    # the headline.
    # linear: MID_DIM=0, the prefix branch on top of SparseK's plain linear score. Isolates how much
    # of the arm is the MLP rather than the prefix.
    case "$MODE" in
      random_init) PREFIX_ZERO_INIT=0 ;;
      linear)      MID_DIM=0 ;;
    esac
    SUB="${MODE}_mid${MID_DIM}_v${PREFIX_VALUE_DIM}"
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_e2e \
      --data-root "$DATA_ROOT" --model "$MODEL" $(data_args) $(prefix_args) \
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
    # RESUME=<ckpt> continues in place -- AdamW moments and the LR-schedule position come back and
    # completed steps are skipped. A flag on THIS case rather than a separate mode, so --resume-from
    # can check the checkpoint's recorded --schedule against the one passed; a copied block that
    # drifted by a flag would be rejected here instead of resuming onto a different curve.
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_e2e \
      --data-root "$DATA_ROOT" --model "$MODEL" $(data_args) $(prefix_args) \
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

  cross_replay)
    # The objective built FOR query-independent scores: prefill C densely, then replay the same
    # tokens against KV(C) alone so every replay query sees every context key -- a rectangle, not the
    # triangle the e2e loss gives. One score is forced to serve many queries at once, which is the
    # query-agnostic reuse value eviction actually needs.
    #
    # --shuffle-control-every is NOT optional here and is set aggressively (50, against the flag's own
    # suggested 100). It is the single number that separates "trained" from "trained-looking": replay
    # loss with the scores permuted along the key axis, minus the unpermuted loss. <= 0 means there is
    # no ranking, whatever the curve does. Both prior prefix-flavoured features -- the recurrent state
    # z and the redundancy scores -- died on exactly this control, so pay the two extra passes.
    #
    # MID_DIM matches the cross-replay script's own default (256) rather than its argparse default
    # (0), for the reason recorded in its header: a 256-vs-0 mismatch confounded an earlier A/B.
    # BUDGET stays at 1 -- that script measured B=topk costing 27.8 RULER points.
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_cross_replay \
      --data-root "$DATA_ROOT" --model "$MODEL" $(data_args) $(prefix_args) \
      --schedule "${SCHEDULE:-8192:300,16384:300}" \
      ${MAX_STEPS:+--max-steps $MAX_STEPS} \
      --pin-mode sink --n-sink "$N_SINK" \
      --budget "${BUDGET:-1}" \
      --global-batch-size "${GLOBAL_BATCH:-8}" \
      --compression-ratio "$COMPRESSION_RATIO" \
      --peak-lr "$PEAK_LR" --final-lr "$FINAL_LR" \
      --warmup-frac "$WARMUP_FRAC" --stable-frac "$STABLE_FRAC" \
      --batch-size "${BATCH_SIZE:-1}" --take-from random --shuffle-buffer 64 \
      --num-workers "${WORKERS:-2}" \
      --shuffle-control-every "${SHUFFLE_EVERY:-50}" \
      ${QUERY_CHUNK:+--query-chunk $QUERY_CHUNK} ${LOGIT_CHUNK:+--logit-chunk $LOGIT_CHUNK} \
      --out "$OUT/cross_replay" --metrics-file "$OUT/cross_replay/metrics.jsonl" \
      --save-every "${SAVE_EVERY:-100}" --log-every "${LOG_EVERY:-10}"
    ;;

  stage2)
    # Sparse scope at 64K. --pin-mode is not passed: the forward is already restricted to the
    # router's top-k, so there is no dense fallback to pin against and the trainer rejects it.
    # --init-from checks the checkpoint's recorded scorer, so a scalar or pairwise checkpoint is
    # refused here rather than silently loading a subset of the parameters.
    INIT="${INIT:-$OUT/stage1_16k_mid${MID_DIM}_v${PREFIX_VALUE_DIM}/final.pt}"
    if [[ ! -f "$INIT" ]]; then
      echo "stage 2 needs stage 1's checkpoint at $INIT" >&2
      echo "  run stage1_16k first, or set INIT= (random_init/linear write elsewhere)" >&2
      exit 1
    fi
    EXTRA=()
    if [[ -f "$TOKENIZED/index.json" ]]; then
      EXTRA+=(--tokenized "$TOKENIZED")
    fi
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_e2e \
      --data-root "$DATA_ROOT" --model "$MODEL" "${EXTRA[@]}" $(prefix_args) \
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
    echo "usage: $0 {identity|variance|smoke|stage1_16k|stage1|cross_replay|random_init|linear|stage2}" >&2
    echo "  run 'identity' then 'variance' before any training run -- see the header" >&2
    echo "  (tokenize with scripts/train_gqa_indexer.sh tokenize -- the corpus is shared)" >&2
    exit 1
    ;;
esac
