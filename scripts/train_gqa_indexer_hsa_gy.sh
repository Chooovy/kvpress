#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch GQA indexer training through TWO-LEVEL (HSA) CHUNK ATTENTION on longmino.
#
#   scripts/train_gqa_indexer_hsa_gy.sh smoke       # 2 steps, verifies the whole path
#   scripts/train_gqa_indexer_hsa_gy.sh stage1_8k   # 600 steps, LR-matched to the e2e run
#   scripts/train_gqa_indexer_hsa_gy.sh chunk128    # the chunk-size sweep's other point
#
# Tokenize with the DISTILLATION script -- `scripts/train_gqa_indexer.sh tokenize` -- and share the
# corpus. All four objectives read the same shards.
#
# THE POINT OF THIS SCRIPT is a fair comparison against scripts/train_gqa_indexer_e2e_gy.sh and
# scripts/train_gqa_indexer_exact_k_gy.sh. Every setting that is not the objective is matched: model,
# corpus, subsets, WSD schedule (warmup 10% to PEAK_LR, hold 60%, decay to FINAL_LR), peak/final LR,
# MAX_STEPS, GLOBAL_BATCH (so tokens/step is identical), seed, take-from, shuffle buffer, save/log
# cadence, liger. Change one of those here and it must change there too, or the result stops being
# about the objective.
#
# WHAT THE OBJECTIVE IS
#
#   out = sum_c  w_c * softmax_within-chunk-c(q k^T) @ v_c        w = softmax(s)
#
# Because the within-chunk softmax already sums to 1, w_c IS chunk c's share of the output -- verified
# to 1.1e-16 in fp64. Three things follow, and each removes machinery the other arms need:
#
#   1. NO PINNING. The gated arm's flat gate is a no-op (softmax is shift-invariant), so it reverts
#      to the frozen backbone with no ranking learned and needs --pin-mode to forbid that. Here a
#      flat router gives UNIFORM MIXING, measured 0.44 from dense. No zero-cost setting exists.
#   2. WHAT IS TRAINED IS WHAT INFERENCE RANKS ON. The gated arm's optimum is log(mass) - LSE_c,
#      which for a frozen backbone is a CONSTANT -- a correction carrying no ranking, while inference
#      top-k's on it anyway. This arm's score is the mass itself.
#   3. NO CANDIDATE POOL. The chunk softmax is over EVERY chunk, so every chunk gets a
#      content-dependent gradient every step. That is the exact-K arm's measured bottleneck removed
#      by construction: 11-15% of oracle-best chunks never entered its M=32 pool, and a chunk outside
#      the pool appears nowhere in the graph, so no backward estimator can reach it.
#
# See kvpress/presses/gqa_indexer/ROUTER_LEARNABILITY.md sections 5-6, which singles this out as the
# structurally sound option, and kvpress/presses/gqa_indexer/hsa_attention.py.
#
# WHAT TO WATCH, because the loss will not tell you
#
# entropy (normalized to [0, 1]) is this arm's equivalent of the gated arm's gate_sparsity and the
# exact-K arm's H(mu). 1.0 = the router is still mixing chunks uniformly and has learned no ranking;
# 0 = fully committed. Uniform mixing is a LEGITIMATE operator, so the LM loss can descend while the
# router never learns to choose -- entropy stuck near 1.0 with a falling loss is that failure, and it
# is the one thing this design does not rule out structurally.
#
# lse_corr has NO COUNTERPART IN ANY OTHER ARM and is the reason to prefer this objective for
# diagnosis. ROUTER_LEARNABILITY.md section 6 proves the optimal score for a frozen backbone IS the
# chunk's own log-sum-exp, up to a per-query constant. So the target is known in CLOSED FORM and the
# Spearman against it is measurable directly on real text -- no oracle, no swap experiment, no second
# forward pass. The exact-K arm needed a 75-forward swap-oracle script to answer the same question.
# Rising = the router is learning the right thing. Flat near 0 = it is not, whatever the loss does.
#
# top25% estimates what the eval's --topk truncation will retain: training learns a full distribution
# over chunks, and inference keeps only the top ones. A run with high entropy AND low top25% will
# evaluate badly no matter how good the loss looks, because there is no concentrated mass to keep.
#
# COST
# No DP and no candidate gather, unlike the exact-K arm -- the attention is one pass over the full
# key axis with a per-chunk softmax, so it is closer to dense attention than that arm's ~3x. The
# retained term is the same one every attention here fights: (B, Hq, tile, Sk) logits, checkpointed
# and sized by BYTES rather than by a query count (a count that fits at 8K does not at 16K).
set -euo pipefail

MODE="${1:-smoke}"
# Same defaults as the other _gy scripts, so all arms read one corpus.
DATA_ROOT="${DATA_ROOT:-/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_256k_filtered}"
TOKENIZED="${TOKENIZED:-/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_tokenized_64k}"
MODEL="${MODEL:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B}"
OUT="${OUT:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_hsa}"

NNODES="${NNODES:-1}"
NGPU="${NGPU:-8}"
# Distinct from distillation's 29511, the gated arm's 29512 and exact-K's 29513, so all four can run
# on one node concurrently.
MASTER_PORT="${MASTER_PORT:-29514}"

# Matched to scripts/train_gqa_indexer_e2e_gy.sh.
PEAK_LR="${PEAK_LR:-1e-3}"
FINAL_LR="${FINAL_LR:-5e-6}"
WARMUP_FRAC="${WARMUP_FRAC:-0.10}"
STABLE_FRAC="${STABLE_FRAC:-0.60}"
LIGER="${LIGER:-1}"

# FFN_SP=1, NOT 8 -- an 8x wall-clock difference, not a preference. Same argument as the exact-K
# script: FFN-SP spends its ranks on ONE sequence, so FFN_SP=8 on 8 GPUs leaves a single
# data-parallel replica and GLOBAL_BATCH=8 must be reached by 8 SEQUENTIAL micro-steps. FFN_SP=1
# gives 8 replicas at accum=1, so the same 8 sequences run in PARALLEL. Measured on the exact-K arm
# at 8K: 116.6 s/step against 14.6 s/step, i.e. 19.4 h against 2.4 h for 600 steps. Identical
# tokens/step, identical WSD curve, identical checkpoint semantics.
#
# Raise this only if the sequence length goes past what one GPU holds.
FFN_SP="${FFN_SP:-1}"
COMPRESSION_RATIO="${COMPRESSION_RATIO:-0.5}"
MAX_STEPS="${MAX_STEPS:-600}"

# Routing geometry. CHUNK_SIZE is also what the press ranks at during eval -- recorded in the
# checkpoint, since it is not a parameter shape and a mismatch would mis-score silently. There is no
# QUERY_BLOCK (selection is per query), no N_CANDIDATE and no EXPLORE_FRAC (no pool), and no
# TOPK_CHUNK: the budget is an INFERENCE parameter here, applied by evaluate_sparse.py --topk.
CHUNK_SIZE="${CHUNK_SIZE:-64}"
# lse, not mean. See --chunk-aggregate in the python script's --help for the measurement; briefly:
# the closed-form optimal chunk score is the chunk's own log-sum-exp (ROUTER_LEARNABILITY.md sec 6),
# and mean dilutes a lone needle ~64x -- needle recall at top-4 chunks is 1.000 for lse against 0.533
# for mean, which is where the first (mean-pooled) run lost 59.5 points on niah_multikey_2.
CHUNK_AGGREGATE="${CHUNK_AGGREGATE:-lse}"

cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false
# Long sequences fragment the caching allocator; expandable_segments lets it grow a segment instead
# of failing on a large contiguous request.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

if [[ "$NGPU" -gt 1 ]]; then
  LAUNCH=(torchrun --nnodes "$NNODES" --nproc_per_node "$NGPU" --master_port "$MASTER_PORT")
else
  LAUNCH=(python)
fi

liger_arg() { [[ "$LIGER" != "0" ]] && echo "--liger"; }
ffn_sp_arg() { [[ "$FFN_SP" != "1" ]] && echo "--ffn-sp-size $FFN_SP"; }
route_args() { echo "--chunk-size $CHUNK_SIZE --chunk-aggregate $CHUNK_AGGREGATE"; }

# Prefer the pre-tokenized corpus, exactly as the other three scripts do.
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
    # group when FFN_SP > 1: NCCL init, the all-gather per layer, the SP slicing. NGPU=1 falls back
    # to plain python, in which case FFN_SP must be 1.
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_hsa \
      --data-root "$DATA_ROOT" --model "$MODEL" \
      --subsets 8k_32k --schedule "${SCHEDULE:-8192:10}" \
      $(route_args) $(liger_arg) $(ffn_sp_arg) \
      --num-workers 0 --log-every 1 --save-every 0 \
      --out "$OUT/smoke" --dry-run
    ;;

  stage1_8k)
    # 8K for 600 steps. Directly comparable to scripts/train_gqa_indexer_exact_k_gy.sh stage1_8k
    # (same length, same steps, same LR, same tokens/step) and LR-comparable to
    # train_gqa_indexer_e2e_gy.sh stage1_16k.
    #
    # THE LR CURVE IS IDENTICAL, which is what makes a step-600 comparison meaningful. 8192:1500 has
    # the same 1500-step total as the gated arm's 8192:300,16384:300,32768:900, so WSD lands on the
    # same warmup (150), the same plateau and the same decay start (1050) -- over steps 0..600 the
    # runs see a bit-identical learning rate.
    #
    # 8K rather than the 8K->16K curriculum, for parity with the exact-K arm rather than because 16K
    # is known not to fit here. It may well fit: this objective has no candidate gather (the
    # exact-K arm's O(M * chunk_size) key gather is what pushed it over at 16K, with only 5.4 GiB
    # free beside the frozen backbone). Untested -- try SCHEDULE=8192:300,16384:1200 with FFN_SP=8 if
    # 16K is wanted, and report the peak rather than assuming either way.
    #
    # The honest comparison against the GATED arm is therefore at matched steps and matched LR but
    # NOT matched context length -- that arm sees 300 steps at 8K then 300 at 16K. Report it.
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_hsa \
      --data-root "$DATA_ROOT" --model "$MODEL" $(data_args) \
      --schedule "${SCHEDULE:-8192:1500}" \
      --max-steps "${MAX_STEPS:-600}" \
      $(route_args) $(liger_arg) $(ffn_sp_arg) \
      --global-batch-size "${GLOBAL_BATCH:-8}" \
      --compression-ratio "$COMPRESSION_RATIO" \
      --peak-lr "$PEAK_LR" --final-lr "$FINAL_LR" \
      --warmup-frac "$WARMUP_FRAC" --stable-frac "$STABLE_FRAC" \
      --batch-size 1 --take-from random --shuffle-buffer 64 \
      --num-workers "${WORKERS:-2}" \
      --out "$OUT/stage1" --metrics-file "$OUT/stage1/metrics.jsonl" \
      --save-every "${SAVE_EVERY:-200}" --log-every "${LOG_EVERY:-10}"
    ;;

  chunk128)
    # The other point of the chunk-size sweep, since chunk_size is the one geometry knob this arm
    # has. Worth running because it trades two things off in opposite directions and neither is
    # obviously dominant: a larger chunk means fewer, better-estimated chunk weights (the softmax is
    # over 64 rather than 128 items at 8K) but coarser selection, and NIAH needs ONE specific token
    # -- the eval breakdown that killed the exact-K arm's chunk-wise path showed retrieval collapsing
    # (multiquery 91 -> 35) precisely because a 64-token chunk spends 63 slots on neighbours.
    #
    # Also the cheapest test of whether that collapse was about chunk granularity at all or about
    # the exact-K objective: same granularity here, different objective.
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_hsa \
      --data-root "$DATA_ROOT" --model "$MODEL" $(data_args) \
      --schedule "${SCHEDULE:-8192:1500}" \
      --max-steps "${MAX_STEPS:-600}" \
      --chunk-size 128 --chunk-aggregate "$CHUNK_AGGREGATE" \
      $(liger_arg) $(ffn_sp_arg) \
      --global-batch-size "${GLOBAL_BATCH:-8}" \
      --compression-ratio "$COMPRESSION_RATIO" \
      --peak-lr "$PEAK_LR" --final-lr "$FINAL_LR" \
      --warmup-frac "$WARMUP_FRAC" --stable-frac "$STABLE_FRAC" \
      --batch-size 1 --take-from random --shuffle-buffer 64 \
      --num-workers "${WORKERS:-2}" \
      --out "$OUT/chunk128" --metrics-file "$OUT/chunk128/metrics.jsonl" \
      --save-every "${SAVE_EVERY:-200}" --log-every "${LOG_EVERY:-10}"
    ;;

  *)
    echo "usage: $0 {smoke|stage1_8k|chunk128}" >&2
    echo "  (tokenize with scripts/train_gqa_indexer.sh tokenize -- the corpus is shared)" >&2
    exit 1
    ;;
esac
