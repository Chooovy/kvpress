#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch GQA indexer training by UTILITY SELF-DISTILLATION on longmino.
#
#   scripts/train_gqa_indexer_utility_gy.sh smoke      # 2 steps, verifies the whole path
#   scripts/train_gqa_indexer_utility_gy.sh stage1_8k  # 600 steps, LR-matched to the other arms
#   scripts/train_gqa_indexer_utility_gy.sh wide_band  # the sampler ablation
#
# Tokenize with the DISTILLATION script -- `scripts/train_gqa_indexer.sh tokenize` -- and share the
# corpus. All five objectives read the same shards.
#
# THE POINT OF THIS SCRIPT is a fair comparison against the other four _gy scripts. Every setting that
# is not the objective is matched: model, corpus, subsets, WSD schedule (warmup 10% to PEAK_LR, hold
# 60%, decay to FINAL_LR), peak/final LR, MAX_STEPS, GLOBAL_BATCH (so tokens/step is identical), seed,
# take-from, shuffle buffer, save/log cadence. Change one of those here and it must change there too,
# or the result stops being about the objective.
#
# WHAT THE OBJECTIVE IS
#
#   u_j = -dL/db_j = -alpha_j * <dL/do, v_j - o>          (b_j = an additive bias on key j's logit)
#   loss = mean over sampled pairs of  |u_i - u_j| * softplus(s_j - s_i)      where u_i >= u_j
#
# THE FORWARD PASS IS UNMODIFIED. No gate, no chunk mixture, no hard subset, no straight-through
# estimator -- attention runs exactly as the frozen backbone runs it, verified bit-identical. The
# router is supervised by a target read out of the backbone's own BACKWARD.
#
# So the router is NOT on the forward path, dL_LM/dtheta_router is None (absent, not small), and
# loss = loss_rank is the entire objective. THIS IS A DISTILLATION ARM, in the same class as
# scripts/train_gqa_indexer.sh -- not an end-to-end one. What it changes is the TEACHER:
#
#   against the true single-key drop effect, on real text:
#     alpha  (what attention-KL distillation teaches)   spearman +0.037
#     u      (this arm)                                 spearman +0.991
#
# alpha is nearly UNINFORMATIVE about which keys matter, because a key can hold a lot of attention
# while its value already sits at the row's output -- and u's (v_j - o) factor is exactly that
# correction. This is the mechanism behind SAS's 96.8% attention mass at 79.5% accuracy.
#
# WHAT TO WATCH, because rank_loss will not tell you
#
# score_corr = spearman(router score, u). THE number this run produces. rank_loss is weighted by
# |u_i - u_j|, which scales with ||dL/do||, so it falls when the BATCH GETS EASIER and cannot say
# whether the router is learning. score_corr is against a fixed quantity.
#
# AND IT HAS A KNOWN CEILING, which is the point of running this. u factors into alpha_j (a function
# of q.k -- REACHABLE by the router) times <dL/do, v_j - o> (a function of the VALUE and the loss
# direction -- a q.k scorer sees NEITHER). Measured on Qwen3-8B: spearman(u, alpha) = +0.03 to +0.32,
# spearman(u, value term) = +0.752, and the best construction that additionally uses v magnitude
# reaches only +0.24. So:
#
#   score_corr plateaus near +0.3   -> the router has hit the HYPOTHESIS CLASS limit. Tuning the loss
#                                      further is wasted; the next move is architectural (let the
#                                      indexer see values).
#   score_corr keeps climbing past  -> the probes were measured on too shallow a truncation. That has
#                                      ALREADY inverted one conclusion in this investigation (a
#                                      truncated recall probe said no router beats recency; the full
#                                      eval said +49), so this is a live outcome, not a hedge.
#
# recall (top-k overlap with the teacher) is closer to what the eval does, since inference takes a
# top-k. A router can have mediocre global rank correlation and still keep everything that matters.
#
# |u| is the teacher's scale. If it collapses, every |u_i - u_j| weight goes to 0 and rank_loss falls
# to 0 while the router learns NOTHING -- which reads exactly like convergence. Watch it beside the
# loss, not instead of it.
#
# ONE CAVEAT ON THE TARGET THAT LOOKS LIKE A RESULT
# u contains dL/do, computed from the LABEL. Selecting top-K by u beats DENSE ATTENTION ITSELF --
# measured 15.3 against 18.66 row loss at K=32 of 511 keys. That is target leakage, not a better
# operator. Legitimate for a teacher (a teacher may have privileged information), but it means u's
# absolute quality is NOT an achievable bound, and whatever part of its ranking exists only because it
# knows the answer is unlearnable in principle.
#
# COST
# One forward and one backward, both of the UNMODIFIED backbone -- so this is the cheapest arm here,
# close to a plain dense training step. The ranking loss is built and backwarded INSIDE the tensor hook
# that delivers dL/do, so nothing is stashed across layers (dL/do for all 36 layers would be 2.4 GiB
# at 8K) and alpha is recomputed for N_ROWS rows only, never materialized ((B, H, Sq, Sk) fp32 is 8 GiB
# per layer at 16K).
#
# NO CANDIDATE POOL, by construction: one backward assigns a utility to EVERY key of a sampled row.
# That is the exact-K arm's measured dead end removed -- 11-15% of oracle-best chunks never entered its
# M=32 pool, and a chunk outside the pool appears nowhere in the graph, so no estimator can reach it.
set -euo pipefail

MODE="${1:-smoke}"
# Same defaults as the other _gy scripts, so all arms read one corpus.
DATA_ROOT="${DATA_ROOT:-/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_256k_filtered}"
TOKENIZED="${TOKENIZED:-/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_tokenized_64k}"
MODEL="${MODEL:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B}"
OUT="${OUT:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_utility}"

NNODES="${NNODES:-1}"
NGPU="${NGPU:-8}"
# Distinct from distillation's 29511, the gated arm's 29512, exact-K's 29513 and HSA's 29514, so all
# five can run on one node concurrently.
MASTER_PORT="${MASTER_PORT:-29515}"

# Matched to scripts/train_gqa_indexer_e2e_gy.sh.
PEAK_LR="${PEAK_LR:-1e-3}"
FINAL_LR="${FINAL_LR:-5e-6}"
WARMUP_FRAC="${WARMUP_FRAC:-0.10}"
STABLE_FRAC="${STABLE_FRAC:-0.60}"
# Kept for parity with the other scripts. It saves LESS here: this arm's target is dL/do at each
# attention output, which the backward produces either way, and the (L, vocab) logits are freed as the
# backward passes them rather than RETAINED for a router sitting in the forward.
LIGER="${LIGER:-1}"

# FFN_SP=1, NOT 8 -- an 8x wall-clock difference, not a preference. Same argument as the exact-K and
# HSA scripts: FFN-SP spends its ranks on ONE sequence, so FFN_SP=8 on 8 GPUs leaves a single
# data-parallel replica and GLOBAL_BATCH=8 must be reached by 8 SEQUENTIAL micro-steps. FFN_SP=1 gives
# 8 replicas at accum=1, so the same 8 sequences run in PARALLEL. Measured on the exact-K arm at 8K:
# 116.6 s/step against 14.6 s/step, i.e. 19.4 h against 2.4 h for 600 steps.
FFN_SP="${FFN_SP:-1}"
COMPRESSION_RATIO="${COMPRESSION_RATIO:-0.5}"
MAX_STEPS="${MAX_STEPS:-600}"

# Supervision geometry. N_ROWS is NOT a candidate pool: the utility is exact for EVERY key of a
# sampled row, and only the set of QUERIES supervising the router is thinned. 16 rows x 36 layers x 8
# KV heads is ~4600 full rankings per step, all sharing the same parameters.
N_ROWS="${N_ROWS:-16}"
N_PAIRS="${N_PAIRS:-64}"
# BAND is small on purpose (section 23.3): top-k depends only on the order ACROSS the K-th boundary, so
# a pair at ranks 3 and 7000 is already ordered right by any usable router and its gradient is wasted.
# The band is drawn around the ROUTER's own current ranking, which makes the sampler self-correcting.
BAND="${BAND:-32}"

cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false
# Long sequences fragment the caching allocator; expandable_segments lets it grow a segment instead of
# failing on a large contiguous request.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

if [[ "$NGPU" -gt 1 ]]; then
  LAUNCH=(torchrun --nnodes "$NNODES" --nproc_per_node "$NGPU" --master_port "$MASTER_PORT")
else
  LAUNCH=(python)
fi

liger_arg() { [[ "$LIGER" != "0" ]] && echo "--liger"; }
ffn_sp_arg() { [[ "$FFN_SP" != "1" ]] && echo "--ffn-sp-size $FFN_SP"; }
target_args() { echo "--n-rows $N_ROWS --n-pairs $N_PAIRS --band $BAND"; }

# Prefer the pre-tokenized corpus, exactly as the other four scripts do.
data_args() {
  if [[ -f "$TOKENIZED/index.json" ]]; then
    echo "--tokenized $TOKENIZED --subsets ${SUBSETS:-2e16 2e17}"
  else
    echo "--subsets ${SUBSETS:-2e15 2e16 synth_cwe synth_rex}"
  fi
}

case "$MODE" in
  smoke)
    # Multi-process via torchrun (NGPU=8 by default), so the smoke test also exercises the FFN-SP
    # group when FFN_SP > 1: NCCL init, the all-gather per layer, the SP slicing. NGPU=1 falls back to
    # plain python, in which case FFN_SP must be 1.
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_utility \
      --data-root "$DATA_ROOT" --model "$MODEL" \
      --subsets 8k_32k --schedule "${SCHEDULE:-8192:10}" \
      $(target_args) $(liger_arg) $(ffn_sp_arg) \
      --num-workers 0 --log-every 1 --save-every 0 \
      --out "$OUT/smoke" --dry-run
    ;;

  stage1_8k)
    # 8K for 600 steps. Directly comparable to the exact-K and HSA arms' stage1_8k (same length, same
    # steps, same LR, same tokens/step) and LR-comparable to train_gqa_indexer_e2e_gy.sh stage1_16k.
    #
    # THE LR CURVE IS IDENTICAL, which is what makes a step-600 comparison meaningful. 8192:1500 has
    # the same 1500-step total as the gated arm's 8192:300,16384:300,32768:900, so WSD lands on the
    # same warmup (150), the same plateau and the same decay start (1050) -- over steps 0..600 the runs
    # see a bit-identical learning rate.
    #
    # 8K rather than the curriculum, for parity with those two arms. This arm is the LEAST memory-bound
    # of the five (unmodified forward, nothing retained across layers, alpha recomputed for N_ROWS rows
    # only), so 16K very likely fits at FFN_SP=1 -- try SCHEDULE=8192:300,16384:1200 and report the
    # peak rather than assuming.
    #
    # The honest comparison against the GATED arm is therefore at matched steps and matched LR but NOT
    # matched context length -- that arm sees 300 steps at 8K then 300 at 16K. Report it.
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_utility \
      --data-root "$DATA_ROOT" --model "$MODEL" $(data_args) \
      --schedule "${SCHEDULE:-8192:1500}" \
      --max-steps "${MAX_STEPS:-600}" \
      $(target_args) $(liger_arg) $(ffn_sp_arg) \
      --global-batch-size "${GLOBAL_BATCH:-8}" \
      --compression-ratio "$COMPRESSION_RATIO" \
      --peak-lr "$PEAK_LR" --final-lr "$FINAL_LR" \
      --warmup-frac "$WARMUP_FRAC" --stable-frac "$STABLE_FRAC" \
      --batch-size 1 --take-from random --shuffle-buffer 64 \
      --num-workers "${WORKERS:-2}" \
      --out "$OUT/stage1" --metrics-file "$OUT/stage1/metrics.jsonl" \
      --save-every "${SAVE_EVERY:-200}" --log-every "${LOG_EVERY:-10}"
    ;;

  wide_band)
    # The sampler ablation, and the one that tests this arm's main design choice rather than a
    # hyperparameter. BAND=4096 at 8K covers the whole row, so the boundary sampler degenerates to
    # UNIFORM pair sampling -- which is what section 23.3 argues wastes nearly all of its budget on
    # pairs whose order no top-k can be changed by.
    #
    # Worth running because the argument is theoretical and the cost of being wrong is asymmetric: if
    # score_corr comes out the same, the narrow band is buying nothing and the extra machinery should
    # go. If it comes out lower, the band is load-bearing and the number says by how much.
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_utility \
      --data-root "$DATA_ROOT" --model "$MODEL" $(data_args) \
      --schedule "${SCHEDULE:-8192:1500}" \
      --max-steps "${MAX_STEPS:-600}" \
      --n-rows "$N_ROWS" --n-pairs "$N_PAIRS" --band 4096 \
      $(liger_arg) $(ffn_sp_arg) \
      --global-batch-size "${GLOBAL_BATCH:-8}" \
      --compression-ratio "$COMPRESSION_RATIO" \
      --peak-lr "$PEAK_LR" --final-lr "$FINAL_LR" \
      --warmup-frac "$WARMUP_FRAC" --stable-frac "$STABLE_FRAC" \
      --batch-size 1 --take-from random --shuffle-buffer 64 \
      --num-workers "${WORKERS:-2}" \
      --out "$OUT/wide_band" --metrics-file "$OUT/wide_band/metrics.jsonl" \
      --save-every "${SAVE_EVERY:-200}" --log-every "${LOG_EVERY:-10}"
    ;;

  *)
    echo "usage: $0 {smoke|stage1_8k|wide_band}" >&2
    echo "  (tokenize with scripts/train_gqa_indexer.sh tokenize -- the corpus is shared)" >&2
    exit 1
    ;;
esac
