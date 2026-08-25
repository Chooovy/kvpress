#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch GQA indexer training from the LM loss (end-to-end gated attention) on longmino.
#
#   scripts/train_gqa_indexer_e2e.sh smoke      # 2 steps, verifies the whole path
#   scripts/train_gqa_indexer_e2e.sh stage1     # gated, 8K -> 16K -> 32K curriculum
#   scripts/train_gqa_indexer_e2e.sh stage2     # sparse scope @ 64K, from stage 1's checkpoint
#   scripts/train_gqa_indexer_e2e.sh ablate     # stage1 with pinning OFF, the baseline
#   scripts/train_gqa_indexer_e2e.sh stage1_16k  # 16K context, 600 steps, 8-way FFN-SP
#   SCORER=dma OUT=/path/to/dma scripts/train_gqa_indexer_e2e_gy.sh stage1_16k
#
# Tokenize with the DISTILLATION script -- `scripts/train_gqa_indexer.sh tokenize` -- and share
# the corpus. Both objectives read the same shards, and tokenizing twice would only invite the
# two runs to see different data.
#
# THE POINT OF THIS SCRIPT is a fair comparison against scripts/train_gqa_indexer.sh. Every
# setting that is not the objective is matched to it: model, corpus, subsets, WSD schedule
# (warmup 10% to PEAK_LR, hold 60%, decay to FINAL_LR), peak/final LR, batch size, seed, take-from,
# shuffle buffer, save/log cadence. Change one of those here and it must change there too, or the
# result stops being about distillation-vs-end-to-end.
#
# Deliberately NOT matched, because they have no counterpart:
#   - no --capture-lse / --backend / --key-tile for a teacher: there is no teacher.
#
# Attention runs on the fused kernel in kvpress/presses/gqa_indexer/triton_gated_attention.py, which
# computes the gate inside the tile loop. SDPA cannot host this operation in O(L) memory -- the
# concat form needs Dqk != Dv (flash ineligible) and padding V to match makes the head 256 wide,
# past what flash/mem-efficient support in the backward pass. Both variants OOM'd at 8K.
#
# The default schedule is 8192:300,16384:300,32768:300 -- 900 steps against distillation stage 1's
# 1500. Set SCHEDULE to match it exactly if you want equal step counts rather than equal
# per-length steps.
#
# WHY --pin-mode MATTERS (default: sink)
# Adding the same number to every key cancels in the softmax, so a FLAT gate is a no-op: the model
# reverts to the frozen dense backbone, which is already strong, and the LM loss is satisfied with
# no ranking learned. The router can reach that at zero cost. Pinning exempts some keys from the
# gate's normalizer, making a flat gate arithmetically impossible. In SAS's ablation this is the
# difference between 18.8 and 54.4.
#
#   sink       (default) exempt the first N_SINK keys, matching what the press protects at eval.
#   self       exempt each query's own token -- closest to SAS's always-retained current block.
#   self+sink  both.
#   none       the un-pinned ablation. Leaves the no-op reachable; `$0 ablate` runs it on purpose.
#
# All modes run on the fused Triton kernel at O(L) memory, so the choice is about WHICH keys
# deserve the exemption, not what it costs. (Without Triton, `self` falls back to an O(Sq*Sk) path.)
#
# KNOWN RISK, unmeasured: pinning lifts the pinned keys' attention share to a fixed ~1 multiplier,
# while the frozen model gives them a share that shrinks with context length. At 32K that is a
# ~2e4x promotion of the sink keys. It may over-bias the model toward the sequence start; long-
# context retrieval (needle-in-a-haystack) is where that would show, and the loss curve would not.
# See kvpress/presses/gqa_indexer/ROUTER_LEARNABILITY.md section 9.
set -euo pipefail

MODE="${1:-smoke}"
# Same defaults as scripts/train_gqa_indexer.sh, so both objectives read one corpus.
DATA_ROOT="${DATA_ROOT:-/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_256k_filtered}"
TOKENIZED="${TOKENIZED:-/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_tokenized_64k}"
MODEL="${MODEL:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B}"
OUT="${OUT:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_e2e}"

NNODES="${NNODES:-1}"
NGPU="${NGPU:-8}"
# Different from the distillation script's 29511 so both can run on one node concurrently.
MASTER_PORT="${MASTER_PORT:-29512}"

# Matched to scripts/train_gqa_indexer.sh.
PEAK_LR="${PEAK_LR:-1e-3}"
FINAL_LR="${FINAL_LR:-5e-6}"
WARMUP_FRAC="${WARMUP_FRAC:-0.10}"
STABLE_FRAC="${STABLE_FRAC:-0.60}"

# Fuse lm_head into the cross-entropy (liger-kernel) so the (L, vocab) logits are never
# materialized. ON by default: it is the largest single retained tensor on this path -- 7.0 GiB at
# 8K, 13.8 at 16K, 27.6 at 32K -- because the router's gradient comes from the LM loss, unlike
# distillation whose per-layer hook never touches the LM head. Only the fused CE is patched; liger's
# RoPE/RMSNorm/SwiGLU swaps stay off, since a different RoPE convention would train the router
# against a positional signal it never sees at inference. LIGER=0 to disable.
LIGER="${LIGER:-1}"

# Shard the FFN activations across FFN_SP ranks (sequence parallel). The MLP is position-wise, so
# each rank computes its own slice with NO communication inside the FFN -- one all-gather per layer
# afterwards. FFN is the largest activation term (49% of the measured total on Qwen3-8B), which is
# what makes 16K fit: ~93.7 GiB on one GPU vs ~61 with 8-way FFN-SP.
#
# Attention is deliberately NOT sharded: query i needs keys 0..i, so that needs the Ulysses
# all-to-all plus special handling for the indexer's MQA key. Consequence: this reaches 16K but not
# 32K (~105 GiB there).
#
# FFN_SP must divide NGPU. The NGPU/FFN_SP quotient is the number of data-parallel replicas, so
# NGPU=8 FFN_SP=8 means ONE replica -- effective batch 1, and tokens/step = the sequence length.
# FFN_SP=4 gives 2 replicas at 16K if that fits, which is the better trade when it does.
FFN_SP="${FFN_SP:-8}"

# Gate configuration.
SCORER="${SCORER:-pairwise}"
PIN_MODE="${PIN_MODE:-sink}"
N_SINK="${N_SINK:-4}"
COMPRESSION_RATIO="${COMPRESSION_RATIO:-0.5}"

# Stop after this many optimizer steps regardless of the schedule total (empty = run the whole
# schedule). Pair it with a matching SCHEDULE to truncate at a defined point WITHOUT reshaping the
# WSD curve. The fair-comparison run against distillation stage 1 at step 600:
#
#   SCHEDULE=8192:300,16384:300,32768:900 MAX_STEPS=600 scripts/train_gqa_indexer_e2e.sh stage1
#
# That reuses the distillation run's exact 1500-step schedule, so the LR over steps 0..600 is
# identical (warmup 150, then flat peak, no decay -- decay would not start until 1050), and stops
# at 600 before the O(L^2) 32K stage is ever built. Compare stage1/step600.pt to distillation's.
MAX_STEPS="${MAX_STEPS:-600}"

cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false
# Long sequences fragment the caching allocator; expandable_segments lets it grow a segment
# instead of failing on a large contiguous request.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

if [[ "$NGPU" -gt 1 ]]; then
  LAUNCH=(torchrun --nnodes "$NNODES" --nproc_per_node "$NGPU" --master_port "$MASTER_PORT")
else
  LAUNCH=(python)
fi

# --liger unless LIGER=0. A plain conditional rather than a parameter-expansion trick: this flag
# decides whether ~7-28 GiB is retained, so it should be obvious what it renders to.
liger_arg() { [[ "$LIGER" != "0" ]] && echo "--liger"; }
ffn_sp_arg() { [[ "$FFN_SP" != "1" ]] && echo "--ffn-sp-size $FFN_SP"; }
# Log whether the router became SELECTIVE, not just how hard each layer leans on it. gate_scale
# rising says nothing about sparsity: a layer can carry a large scale on a gate that still spreads
# its mass over every key, which is the flat-gate no-op this whole setup is built to avoid. ~1.0
# means no sparsity learned; falling towards 0 is the router concentrating. GATE_SPARSITY=0 to
# turn it off (it costs a second streaming pass over the keys, on logged steps only).
gate_sparsity_arg() { [[ "${GATE_SPARSITY:-1}" != "0" ]] && echo "--gate-sparsity"; }

# Prefer the pre-tokenized corpus, exactly as the distillation script does.
data_args() {
  if [[ -f "$TOKENIZED/index.json" ]]; then
    echo "--tokenized $TOKENIZED --subsets ${SUBSETS:-2e16 2e17}"
  else
    echo "--subsets ${SUBSETS:-2e15 2e16 synth_cwe synth_rex}"
  fi
}

case "$MODE" in
  smoke)
    # Multi-process via torchrun (NGPU=8 by default), so the smoke test also exercises the
    # FFN-SP group: NCCL init, the all-gather per layer, and the SP slicing. Tracebacks come
    # interleaved from all ranks -- that is the price of checking the distributed path too.
    # NGPU=1 still falls back to plain python, in which case FFN_SP must be 1.
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_e2e \
      --data-root "$DATA_ROOT" --model "$MODEL" \
      --scorer "$SCORER" \
      --subsets 8k_32k --schedule "${SCHEDULE:-16384:10}" \
      --stage dense --pin-mode "$PIN_MODE" --n-sink "$N_SINK" $(liger_arg) $(ffn_sp_arg) \
      --num-workers 0 --log-every 1 --save-every 0 \
      --out "$OUT/smoke" --dry-run
    ;;

  stage1)
    # Gated attention over the FULL key axis, on the same 8K -> 16K -> 32K curriculum and the
    # same single WSD schedule as distillation stage 1.
    #
    # Full scope is what makes every key's gate get a content-dependent gradient. Under sparse
    # scope an unselected key has no gradient of its own and moves only through the softmax
    # normalizer, so the whole unselected set is dragged together rather than judged individually
    # (SAS Figure 5; 47.4 -> 55.6 in their ablation). Cost is O(L^2) attention, same as dense.
    #
    # UNLIKE distillation, the reported loss is directly comparable across the curriculum: the
    # distillation loss grows like log(L) because its softmax gets wider, putting a ~0.69 step in
    # the curve at each boundary. An LM loss has no such term, so any jump here is real.
    #
    # Batch stays 1 at every length. Always correct (the loss is a mean over tokens), and it
    # leaves the 8K stage partly idle.
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_e2e \
      --data-root "$DATA_ROOT" --model "$MODEL" $(data_args) \
      --scorer "$SCORER" \
      --schedule "${SCHEDULE:-8192:300,16384:300,32768:900}" \
      ${MAX_STEPS:+--max-steps $MAX_STEPS} \
      --stage dense --pin-mode "$PIN_MODE" --n-sink "$N_SINK" $(liger_arg) $(ffn_sp_arg) \
      --global-batch-size "${GLOBAL_BATCH:-8}" $(gate_sparsity_arg) \
      --compression-ratio "$COMPRESSION_RATIO" \
      --peak-lr "$PEAK_LR" --final-lr "$FINAL_LR" \
      --warmup-frac "$WARMUP_FRAC" --stable-frac "$STABLE_FRAC" \
      --batch-size "${BATCH_SIZE:-1}" --take-from random --shuffle-buffer 64 \
      --num-workers "${WORKERS:-2}" \
      --out "$OUT/stage1" --metrics-file "$OUT/stage1/metrics.jsonl" \
      --save-every "${SAVE_EVERY:-200}" --log-every "${LOG_EVERY:-10}"
    ;;

  stage1_16k)
    # 16K context for 600 steps, matched to distillation stage 1's first 600.
    #
    # WHY THIS SCHEDULE. It reuses distillation's exact 8192:300,16384:300,32768:900 curve and
    # stops at 600 via MAX_STEPS, so the LR over steps 0..600 is IDENTICAL to that run (warmup
    # 150, then flat at PEAK_LR -- decay would not begin until step 1050). Steps 0..299 run at 8K
    # and 300..599 at 16K, exactly as distillation does. Comparing stage1_16k/step600.pt against
    # distillation's step600.pt is therefore a clean objective-vs-objective comparison.
    #
    # FFN_SP=8 on 8 GPUs means ONE data-parallel replica: all 8 ranks cooperate on a single
    # sequence. So a step would see 1 sequence where the distillation run's 8-way DDP sees 8 --
    # 1/8 the tokens at the same step number, which would make a step-600 comparison meaningless.
    # GLOBAL_BATCH=8 restores it by accumulating 8 micro-steps per optimizer step, matching
    # distillation's 8 sequences/step exactly. It costs ~8x the wall-clock per step, which is the
    # real price of the longer context; lower it only if you also stop comparing step-for-step.
    # FFN_SP=4 would give 2 replicas (accum 4) at ~2x the per-rank memory; use it if 16K fits.
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_e2e \
      --data-root "$DATA_ROOT" --model "$MODEL" $(data_args) \
      --scorer "$SCORER" \
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
      --out "$OUT/stage1" --metrics-file "$OUT/stage1/metrics.jsonl" \
      --save-every "${SAVE_EVERY:-200}" --log-every "${LOG_EVERY:-10}"
    ;;

  ablate)
    # Stage 1 with pinning OFF. This is the honest baseline for "does pinning matter", and it is
    # expected to look GOOD on the loss curve and BAD at eval: the gate can flatten into a no-op,
    # recovering the frozen dense model, so the LM loss falls without the router learning to rank.
    # Watch gate_scales in the metrics -- a collapse there is the tell the loss curve hides.
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_e2e \
      --data-root "$DATA_ROOT" --model "$MODEL" $(data_args) \
      --scorer "$SCORER" \
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
      --save-every "${SAVE_EVERY:-200}" --log-every "${LOG_EVERY:-10}"
    ;;

  stage2)
    # Sparse scope at 64K, matching distillation stage 2's length, topk and reserved slots.
    #
    # --pin-mode is not passed: the sparse forward is already restricted to the router's own
    # top-k, so a flat gate cannot recover dense attention and there is nothing to pin against.
    # The trainer resolves it to "none" for this stage and rejects anything else, rather than
    # accepting a flag that would silently do nothing.
    #
    # This stage is also the train/inference-consistent variant: the forward attends to exactly
    # the set inference will select. SAS finds the INCONSISTENT full-scope stage better (their STE
    # ablation loses 61.30 -> 51.48), so running both is the point rather than a hedge.
    INIT="${INIT:-$OUT/stage1/final.pt}"
    if [[ ! -f "$INIT" ]]; then
      echo "stage 2 needs stage 1's checkpoint at $INIT (run stage1 first, or set INIT=)" >&2
      exit 1
    fi
    EXTRA=()
    if [[ -f "$TOKENIZED/index.json" ]]; then
      EXTRA+=(--tokenized "$TOKENIZED")
    fi
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_e2e \
      --data-root "$DATA_ROOT" --model "$MODEL" "${EXTRA[@]}" \
      --scorer "$SCORER" \
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
      --save-every "${SAVE_EVERY:-200}" --log-every "${LOG_EVERY:-10}"
    ;;

  *)
    echo "usage: $0 {smoke|stage1|stage1_16k|ablate|stage2}" >&2
    echo "  (tokenize with scripts/train_gqa_indexer.sh tokenize -- the corpus is shared)" >&2
    exit 1
    ;;
esac
