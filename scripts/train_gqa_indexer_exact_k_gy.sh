#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch GQA indexer training through an EXACT-K CHUNK SUBSET on longmino.
#
#   scripts/train_gqa_indexer_exact_k_gy.sh smoke       # 2 steps, verifies the whole path
#   scripts/train_gqa_indexer_exact_k_gy.sh stage1_8k   # 600 steps, LR-matched to the e2e run
#   scripts/train_gqa_indexer_exact_k_gy.sh ablate_noexplore  # candidate pool = plain top-M
#   scripts/train_gqa_indexer_exact_k_gy.sh ablate_hard       # deterministic top-K, no sampling
#
# Tokenize with the DISTILLATION script -- `scripts/train_gqa_indexer.sh tokenize` -- and share the
# corpus. All three objectives read the same shards.
#
# THE POINT OF THIS SCRIPT is a fair comparison against scripts/train_gqa_indexer_e2e_gy.sh. Every
# setting that is not the objective is matched to it: model, corpus, subsets, WSD schedule (warmup
# 10% to PEAK_LR, hold 60%, decay to FINAL_LR), peak/final LR, MAX_STEPS, GLOBAL_BATCH (so
# tokens/step is identical), seed, take-from, shuffle buffer, save/log cadence, liger. Change one of
# those here and it must change there too, or the result stops being about the objective.
#
# TWO settings deliberately DIFFER, both forced by memory, both stated so they are not mistaken for
# oversights:
#   - sequence length: 8K throughout, where the gated arm does 8K then 16K. 16K does not fit on this
#     arm; see stage1_8k for the measurement and for why the LR curve is still identical.
#   - FFN_SP: 1 here against 8 there. This does NOT change tokens/step or the LR curve -- only
#     whether the 8 sequences of a step run in parallel or sequentially. See the FFN_SP block below;
#     it is worth 8x wall clock and nothing else.
#
# WHAT THE OBJECTIVE IS
# The gated arm adds the router's score inside the attention softmax. A gate that is FLAT along the
# key axis cancels there, so the model reverts to the frozen dense backbone and the LM loss is
# satisfied with no ranking learned -- which is why that arm needs --pin-mode. This arm instead
# REPLACES attention with a sampled K-of-M chunk subset:
#
#   g   = (z - mu).detach() + mu        z ~ exactly-K subset,  mu = P(z_i = 1 | sum z = K)
#   out = (g * exp(a)) / sum(g * exp(a)) @ v
#
# The forward commits to exactly K chunks, so no configuration of the scores reproduces dense
# attention. There is nothing to pin -- the scarcity is structural rather than manufactured. See
# kvpress/presses/gqa_indexer/ROUTER_LEARNABILITY.md section 7 and HANDOFF_exact_k_subset.md.
#
# WHAT TO WATCH, because the loss will not tell you
# H(mu) -- the mean Bernoulli entropy of the marginals -- is this arm's equivalent of the gated
# arm's gate_sparsity. At init the marginals are uniform at K/M so it sits at its maximum; it falls
# as the router commits. A run whose loss descends while H(mu) stays flat has learned to USE
# whatever random subset it is handed rather than to CHOOSE one. That is the one failure mode
# exact-K does not rule out structurally, and no loss curve reveals it.
#
# eff_K reports the realized budget. Near the diagonal a query block cannot see M chunks yet (48% of
# blocks at 8K with these settings), and when it cannot see K either the budget is unreachable --
# correct behaviour, but the effective budget is then below K and only this number says so. Expect
# ~7.5 of a nominal K=8 at 8K; a number far below that means the geometry is wrong.
#
# jaccard is the selection stability across steps. torch.bernoulli makes the forward stochastic;
# ProbMoE does not report that as a problem, but their row count is not ours.
#
# COST, measured on an H20 rather than extrapolated
# HANDOFF_exact_k_subset.md section 4 concluded from CPU timings that this was "dead on arrival" at
# ~1690 s/layer/step. That was wrong by about four orders of magnitude, and wrong structurally: the
# marginals' DP is LAUNCH-bound on GPU, so its cost is independent of the row count (131072 rows
# cost the same as 1024). Measured per layer, fwd+bwd, at chunk 64 / query_block 256 / M=32 / K=8:
#
#   8K:  151 ms, 2.8 GiB   (dense SDPA at the same shape: 34.5 ms)
#   16K: 263 ms, 3.5 GiB   (dense SDPA: 87.6 ms)  <-- the OP fits; the full RUN does not, see below
#
# So ~3x dense per layer. N_CANDIDATE (M) is the knob that sets this -- it scales roughly linearly,
# because the DP is O(M) sequential launches and the training-time attention is over M*CHUNK_SIZE
# keys. TOPK_CHUNK (K) is nearly free by comparison.
set -euo pipefail

MODE="${1:-smoke}"
# Same defaults as scripts/train_gqa_indexer_e2e_gy.sh, so all arms read one corpus.
DATA_ROOT="${DATA_ROOT:-/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_256k_filtered}"
TOKENIZED="${TOKENIZED:-/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_tokenized_64k}"
MODEL="${MODEL:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B}"
OUT="${OUT:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_exact_k}"

NNODES="${NNODES:-1}"
NGPU="${NGPU:-8}"
# Distinct from the distillation script's 29511 and the gated one's 29512, so all three can run on
# one node concurrently.
MASTER_PORT="${MASTER_PORT:-29513}"

# Matched to scripts/train_gqa_indexer_e2e_gy.sh.
PEAK_LR="${PEAK_LR:-1e-3}"
FINAL_LR="${FINAL_LR:-5e-6}"
WARMUP_FRAC="${WARMUP_FRAC:-0.10}"
STABLE_FRAC="${STABLE_FRAC:-0.60}"
LIGER="${LIGER:-1}"

# FFN_SP=1, NOT 8 -- and this is an 8x wall-clock difference, not a preference.
#
# FFN-SP spends its ranks on ONE sequence, so with FFN_SP=8 on 8 GPUs there is a single
# data-parallel replica and GLOBAL_BATCH=8 has to be reached by 8 SEQUENTIAL micro-steps
# (accum=8). With FFN_SP=1 there are 8 replicas and accum=1, so the same 8 sequences run in
# PARALLEL. Measured at 8K: 116.6 s/step against 14.6 s/step, i.e. 19.4 h against 2.4 h for the
# 600-step run. Identical tokens/step (65536), identical WSD curve, identical checkpoint semantics.
#
# The gated arm needs FFN_SP=8 because it trains at 16K, where one sequence does not fit on one
# GPU. This arm is capped at 8K by memory (see stage1_8k), and at 8K a single sequence fits with
# room to spare -- 65.8 GiB of 95 measured. So the sharding buys nothing here and costs 8x.
#
# Raise this only if you also raise the sequence length past what one GPU holds.
FFN_SP="${FFN_SP:-1}"
COMPRESSION_RATIO="${COMPRESSION_RATIO:-0.5}"
MAX_STEPS="${MAX_STEPS:-600}"

# Routing geometry. CHUNK_SIZE is also what the press ranks at during eval -- it is recorded in the
# checkpoint, since it is not a parameter shape and a mismatch would mis-score silently.
CHUNK_SIZE="${CHUNK_SIZE:-64}"
QUERY_BLOCK="${QUERY_BLOCK:-256}"
TOPK_CHUNK="${TOPK_CHUNK:-8}"
N_CANDIDATE="${N_CANDIDATE:-32}"
# 10% of the pool drawn at random. Without it a chunk outside top-M appears nowhere in the graph and
# receives exactly zero gradient, so it can never be promoted -- the failure that kept the
# selected-gate proxy at 0.0% recall. `$0 ablate_noexplore` runs 0 on purpose.
EXPLORE_FRAC="${EXPLORE_FRAC:-0.10}"
N_SINK_CHUNK="${N_SINK_CHUNK:-1}"
N_LOCAL_CHUNK="${N_LOCAL_CHUNK:-1}"

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

route_args() {
  echo "--chunk-size $CHUNK_SIZE --query-block $QUERY_BLOCK --topk-chunk $TOPK_CHUNK" \
       "--n-candidate $N_CANDIDATE --explore-frac $EXPLORE_FRAC" \
       "--n-sink-chunk $N_SINK_CHUNK --n-local-chunk $N_LOCAL_CHUNK"
}

# Prefer the pre-tokenized corpus, exactly as the other two scripts do.
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
    # group: NCCL init, the all-gather per layer, and the SP slicing. NGPU=1 falls back to plain
    # python, in which case FFN_SP must be 1.
    #
    # 8K, not 16K: 16K provably does not fit on this arm -- see stage1_8k for the measurement. A
    # smoke test that OOMs by design tells you nothing about whether the path works.
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_exact_k \
      --data-root "$DATA_ROOT" --model "$MODEL" \
      --subsets 8k_32k --schedule "${SCHEDULE:-8192:10}" \
      $(route_args) $(liger_arg) $(ffn_sp_arg) \
      --num-workers 0 --log-every 1 --save-every 0 \
      --out "$OUT/smoke" --dry-run
    ;;

  stage1_8k)
    # 8K for 600 steps, matched to scripts/train_gqa_indexer_e2e_gy.sh stage1_16k's FIRST 300 steps
    # and then continuing at 8K rather than stepping up to 16K. Read the next paragraph before
    # comparing the two: this is the one setting that could not be matched.
    #
    # WHY 8K AND NOT THE 8K->16K CURRICULUM. MEASURED, not assumed: 16K does not fit. The frozen
    # Qwen3-8B backbone alone peaks at 89.6 GiB of 95 at 16K (read off the gated run's own
    # metrics.jsonl), leaving 5.4 GiB, and this objective needs ~3.6 GiB of that for the routing
    # attention plus the candidate/score tensors -- it OOM'd on a 256 MiB transient with everything
    # already checkpointed (the DP, the score tiles, and the attention tiles all recompute in the
    # backward, and q/k/v are no longer upcast up front). The gated arm fits at 16K because its
    # fused Triton kernel computes the gate inside the tile loop and materializes nothing extra;
    # this arm gathers M*CHUNK_SIZE keys per query block, which is a real O(M) tensor. Closing that
    # gap needs a fused kernel, which is not built.
    #
    # THE LR CURVE IS STILL IDENTICAL, which is what makes a step-600 comparison meaningful.
    # 8192:1500 has the same 1500-step total as 8192:300,16384:300,32768:900, so WSD lands on the
    # same warmup (150), the same plateau, and the same decay start (1050). Over steps 0..600 the
    # two runs see a bit-identical learning rate. What differs is the DATA: this run sees 600 steps
    # of 8K where the gated run sees 300 at 8K and 300 at 16K. So the honest comparison is at
    # matched steps and matched LR but NOT matched context length, and any long-context gap should
    # be read with that in mind -- report it rather than presenting the two as fully matched.
    #
    # FFN_SP=8 on 8 GPUs means ONE data-parallel replica: all 8 ranks cooperate on a single
    # sequence, so a step would see 1 sequence where an 8-way DDP run sees 8 -- 1/8 the tokens at the
    # same step number, which would make a step-600 comparison meaningless. GLOBAL_BATCH=8 restores
    # it by accumulating 8 micro-steps per optimizer step, matching the gated run exactly.
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_exact_k \
      --data-root "$DATA_ROOT" --model "$MODEL" $(data_args) \
      --schedule "${SCHEDULE:-8192:1500}" \
      --max-steps "${MAX_STEPS:-600}" \
      $(route_args) $(liger_arg) \
      --ffn-sp-size "${FFN_SP_16K:-1}" \
      --global-batch-size "${GLOBAL_BATCH:-8}" \
      --compression-ratio "$COMPRESSION_RATIO" \
      --peak-lr "$PEAK_LR" --final-lr "$FINAL_LR" \
      --warmup-frac "$WARMUP_FRAC" --stable-frac "$STABLE_FRAC" \
      --batch-size 1 --take-from random --shuffle-buffer 64 \
      --num-workers "${WORKERS:-2}" \
      --out "$OUT/stage1" --metrics-file "$OUT/stage1/metrics.jsonl" \
      --save-every "${SAVE_EVERY:-200}" --log-every "${LOG_EVERY:-10}"
    ;;

  ablate_noexplore)
    # The candidate pool becomes a plain top-M. Expected to look FINE on the loss curve and worse at
    # eval: a chunk outside top-M receives exactly zero gradient, so the router can only ever
    # re-rank chunks it already likes. This is the pool-level version of the disease that kept the
    # selected-gate proxy at 0.0% recall, and it is the honest test of whether exploration earns its
    # place. Watch H(mu) and eff_K, not the loss.
    # --explore-frac 0 is passed explicitly rather than by overriding EXPLORE_FRAC, so what runs is
    # visible in the command rather than depending on an env assignment.
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_exact_k \
      --data-root "$DATA_ROOT" --model "$MODEL" $(data_args) \
      --schedule "${SCHEDULE:-8192:1500}" \
      --max-steps "${MAX_STEPS:-600}" \
      --chunk-size "$CHUNK_SIZE" --query-block "$QUERY_BLOCK" --topk-chunk "$TOPK_CHUNK" \
      --n-candidate "$N_CANDIDATE" --explore-frac 0 \
      --n-sink-chunk "$N_SINK_CHUNK" --n-local-chunk "$N_LOCAL_CHUNK" \
      $(liger_arg) --ffn-sp-size "${FFN_SP_16K:-1}" \
      --global-batch-size "${GLOBAL_BATCH:-8}" \
      --compression-ratio "$COMPRESSION_RATIO" \
      --peak-lr "$PEAK_LR" --final-lr "$FINAL_LR" \
      --warmup-frac "$WARMUP_FRAC" --stable-frac "$STABLE_FRAC" \
      --batch-size 1 --take-from random --shuffle-buffer 64 \
      --num-workers "${WORKERS:-2}" \
      --out "$OUT/ablate_noexplore" --metrics-file "$OUT/ablate_noexplore/metrics.jsonl" \
      --save-every "${SAVE_EVERY:-200}" --log-every "${LOG_EVERY:-10}"
    ;;

  ablate_hard)
    # Deterministic top-K instead of sampling, i.e. plain STE. This is the ablation that isolates
    # what the stochastic forward buys: with a hard selection a chunk's score is only ever compared
    # against chunks that were already selected, so the exact marginals still supply gradient to
    # unselected candidates but the forward never *tries* them. SAS reports STE as worse
    # (61.30 -> 51.48 on AIME25); whether the exact-marginal backward rescues it is the question.
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_exact_k \
      --data-root "$DATA_ROOT" --model "$MODEL" $(data_args) \
      --schedule "${SCHEDULE:-8192:1500}" \
      --max-steps "${MAX_STEPS:-600}" \
      $(route_args) --hard $(liger_arg) --ffn-sp-size "${FFN_SP_16K:-1}" \
      --global-batch-size "${GLOBAL_BATCH:-8}" \
      --compression-ratio "$COMPRESSION_RATIO" \
      --peak-lr "$PEAK_LR" --final-lr "$FINAL_LR" \
      --warmup-frac "$WARMUP_FRAC" --stable-frac "$STABLE_FRAC" \
      --batch-size 1 --take-from random --shuffle-buffer 64 \
      --num-workers "${WORKERS:-2}" \
      --out "$OUT/ablate_hard" --metrics-file "$OUT/ablate_hard/metrics.jsonl" \
      --save-every "${SAVE_EVERY:-200}" --log-every "${LOG_EVERY:-10}"
    ;;

  *)
    echo "usage: $0 {smoke|stage1_8k|ablate_noexplore|ablate_hard}" >&2
    echo "  (tokenize with scripts/train_gqa_indexer.sh tokenize -- the corpus is shared)" >&2
    exit 1
    ;;
esac
