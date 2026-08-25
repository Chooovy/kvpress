#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch cross-replay indexer training on longmino.
#
#   scripts/train_gqa_indexer_cross_replay_gy.sh smoke       # 2 steps @16K, verifies the whole path
#   scripts/train_gqa_indexer_cross_replay_gy.sh probe       # 100 steps @8K, ~15 min: does it LEARN?
#   scripts/train_gqa_indexer_cross_replay_gy.sh stage1_16k  # the real run: 8K->16K, 600 steps
#
# Tokenize with the DISTILLATION script -- `scripts/train_gqa_indexer.sh tokenize` -- and share the
# corpus. All three objectives read the same shards; tokenizing twice would only invite them to see
# different data.
#
# THE OBJECTIVE. Prefill C densely, then replay the same tokens as C' against KV(C) alone, so every
# replay query sees every context key -- a full rectangle rather than the causal triangle the ordinary
# LM loss gives. One per-key score is forced to serve many queries at once, which is the
# query-agnostic reuse value eviction needs. Design notes: cross_replay_e2e.md.
#
# READ THE DIAGNOSTICS, NOT THE LOSS. A gate that is flat along the key axis adds a per-row constant,
# which cancels in the softmax -- so the model reverts to the frozen dense backbone (already strong),
# the loss falls, and NO RANKING IS LEARNED. Two numbers catch that and are logged every step:
#
#   participation  ~1.0 = flat gate, nothing learned, whatever the loss says. Falling towards 0 is
#                  the concentration eviction needs -- but TOO FAR IS ALSO WRONG, and that is what
#                  the first real run did: it settled at 0.0070 while eval retains topk/N = 0.125,
#                  an 18x mismatch, and RULER @8K came out 44.75 against the LM-loss arm's 66.24.
#                  The target is the eval budget: participation ~ topk/N, so ~0.125 at 16K. BUDGET
#                  below is the knob that sets it.
#   shuffle_delta  replay loss with the learned scores PERMUTED along the key axis, minus unpermuted.
#                  <= 0 means the score carries no usable ranking. This is the one number that
#                  separates "trained" from "trained-looking". On random ids: +1.95 nats/token.
#
# WHAT THIS SCRIPT DELIBERATELY DOES NOT OFFER, and why (each would be a silent no-op; the driver
# rejects them at argparse rather than ignoring them):
#
#   LIGER   cross-replay calls the BASE model and computes the loss itself, so liger's fused CE --
#           which patches *ForCausalLM.forward and needs `labels` -- would never run. This exact flag
#           was threaded through this objective for a full revision while doing nothing
#           (cross_replay_e2e.md §6.1). LOGIT_CHUNK below bounds the same tensor and is exact.
#   FFN_SP  not needed: 16K peaks at a MEASURED 33.5 GiB with QUERY_CHUNK=1024, so it fits on one H20
#           (the e2e script needs 8-way FFN-SP only because its 16K is ~93.7 GiB). It is also unsound
#           here -- FFN-SP slices the sequence per rank, while this objective runs two passes over the
#           same tokens against a cache whose key axis must stay exactly |C|.
#   ablate  the e2e script's honest "does pinning matter" baseline (--pin-mode none) is STRUCTURALLY
#           IMPOSSIBLE here, not merely omitted: CrossReplayTrainer rejects any pin but `sink`,
#           because under [C ; C'] a `self` pin's diagonal key sits in the masked-out C' block and so
#           pins nothing visible (cross_replay_e2e.md §3). There is no mode to add.
#
# MEMORY, all measured on an H20 96 GiB at QUERY_CHUNK=1024 (cross_replay_e2e.md §6.4):
#   8K  27.8 GiB, 5.16 s/seq        16K  33.5 GiB, 13.65 s/seq
# QUERY_CHUNK is the knob: at 8K, 128/512/1024/2048 -> 21.9/24.4/27.8/34.6 GiB, and it is EXACT (every
# chunk still attends to the whole key axis, so the rectangle is preserved).
set -euo pipefail

MODE="${1:-smoke}"
# Same defaults as the other two scripts, so all three objectives read one corpus.
DATA_ROOT="${DATA_ROOT:-/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_256k_filtered}"
TOKENIZED="${TOKENIZED:-/apdcephfs_gy8/share_303843174/guhao/datasets/longmino_tokenized_64k}"
MODEL="${MODEL:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B}"
OUT="${OUT:-/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_cross_replay}"

NNODES="${NNODES:-1}"
# Every rank is its own DATA-PARALLEL replica here, because this objective needs no FFN-SP (see the
# header). That is the one structural difference from the sibling scripts' launch geometry, and it is
# worth being explicit about because it is what makes tokens/step comparable rather than what breaks
# it -- the arithmetic is (NGPU / FFN_SP) x batch_size x accum:
#
#   scalar/e2e stage1_16k   NGPU=8, FFN_SP=8 -> 1 replica  x accum 8 = 8 sequences/step
#   cross-replay stage1_16k NGPU=8, FFN_SP=1 -> 8 replicas x accum 1 = 8 sequences/step
#
# Same 8 sequences, so 65536 tokens/step at 8K and 131072 at 16K in both arms -- confirmed against
# both runs' metrics.jsonl, not just derived. GLOBAL_BATCH is what holds this invariant: it divides
# by the replica count, so tokens/step does not move with NGPU. If you change NGPU, do NOT also
# change GLOBAL_BATCH, or the step-for-step comparison against the other two arms is gone.
NGPU="${NGPU:-8}"
# 29511 distill, 29512 e2e, 29513 scalar, 29514 here -- so all four can run on one node.
MASTER_PORT="${MASTER_PORT:-29514}"

# Matched to the other two scripts.
PEAK_LR="${PEAK_LR:-1e-3}"
FINAL_LR="${FINAL_LR:-5e-6}"
WARMUP_FRAC="${WARMUP_FRAC:-0.10}"
STABLE_FRAC="${STABLE_FRAC:-0.60}"

# Gate configuration. PIN_MODE is not a variable: only `sink` is legal (see the header).
N_SINK="${N_SINK:-4}"
COMPRESSION_RATIO="${COMPRESSION_RATIO:-0.5}"
# 256 matches scripts/train_gqa_indexer_scalar_gy.sh's DEFAULT mid_dim, which is what the arm this
# objective is compared against was trained with. It was 0 here for one revision, which made the
# first A/B uninterpretable: cross-replay ran a 1.48M-parameter LINEAR score against the LM-loss
# arm's 38.14M-parameter MLP, so 66.24 vs 44.75 on RULER @8K confounded "which loss" with "26x the
# capacity". Only the checkpoints' own `config` revealed it. Do not change this without changing it
# there too, or the comparison stops being about the objective.
MID_DIM="${MID_DIM:-256}"
# B in the gate `s - LSE(s) + log B`, on the gated keys only. LEAVE AT 1 -- see the measurement below.
#
# `sum_{j gated} exp(g_j) = B`, so B is how many pinned-sink equivalents the whole history is worth,
# and it decides how CONCENTRATED the gate becomes at convergence -- which is a separate thing from
# the ranking it learns (within the gated softmax log B cancels exactly; against the sinks it does
# not).
#
# ⚠️ RETRACTED: "SET TO THE INFERENCE TOP-K." That was this block's recommendation, derived from the
# identity above -- a hard top-k gate holds `topk` keys at multiplier 1, whose sum exp(g) is exactly
# `topk`, so B=topk is the only value at which inference is "exactly representable" during training.
# The 2x2 grid was completed and **representability is not the property that matters**. Isolating B at
# fixed mid_dim=256, RULER 8K at topk=2048, fraction 0.100, step 600:
#
#     B=1     48.20        B=2048  20.43        i.e. B=topk costs 27.8 POINTS
#
# For scale, the objective gap this term was meant to help close (cross-replay vs e2e LM loss, both at
# B=1, mid_dim=256) is 18.0 points, and 26x the scorer capacity buys 3.4. The budget is the largest
# single effect measured on this objective and its sign is negative.
#
# Also retracted: the claim that B=1 CAUSED the first run's deficit. The e2e arm -- the best of the
# four at 66.24 -- also trains at B=1 (`E2EIndexerTrainer.gate_budget` defaults to 1.0), with 24x the
# participation of the cross-replay B=1 run at the same B. Concentration here is set by the loss
# geometry, not by B. See cross_replay_e2e.md §15.2 and §15.3 for both retractions and the grid.
#
#   B=n_gated  still degenerate: the flat-gate no-op, g_j = 0 everywhere, so the loss can be satisfied
#              with no ranking learned. That is the hole pinning exists to close. Unset resolves here,
#              and warns. Only the B=topk recommendation is withdrawn, not this end of the range.
BUDGET="${BUDGET:-1}"

# §16.5 items 1 and 2 (see cross_replay_e2e.md §17). Both default to the existing behaviour, so an
# unset run reproduces arm D exactly.
#
# DEMAND_REDUCE: how replay queries' demands on one key combine into dL/ds. "max" keeps each key's
# strongest demand per query CHUNK rather than averaging over all N queries -- what KVzip's
# max-attention label does for free because it never differentiates (§16.4). It needs >= 2 chunks,
# which QUERY_CHUNK=1024 already gives (8 at 8K, 16 at 16K), so this costs NO change to the chunk
# size and stays single-variable against arm D.
DEMAND_REDUCE="${DEMAND_REDUCE:-sum}"
# LOOKAHEAD: bound on how far past its own position a replay row may see. Unset = the unbounded
# rectangle. 0 reproduces the causal triangle the e2e loss trains on. See the flex-path caveat in
# CrossReplayTrainer.lookahead -- a bounded horizon is a real mask.
LOOKAHEAD="${LOOKAHEAD:-}"
# CROSS_DOC_REPLAY=1 replays an UNRELATED document against KV(C). The diagnostic for whether the
# reconstruction relation teaches the router anything at all -- see cross_replay_e2e.md §19.
CROSS_DOC_REPLAY="${CROSS_DOC_REPLAY:-}"

lookahead_arg() { [[ -n "$LOOKAHEAD" ]] && printf '%s %s' --lookahead "$LOOKAHEAD"; }
cross_doc_arg() { [[ -n "$CROSS_DOC_REPLAY" ]] && printf '%s' --cross-doc-replay; }

# The run directory has to name every axis that differs, or two ablations overwrite each other and the
# checkpoints become unattributable -- the §13 failure in its cheapest possible form.
RUN_TAG="stage1_${MID_DIM}_B${BUDGET}"
[[ "$DEMAND_REDUCE" != "sum" ]] && RUN_TAG="${RUN_TAG}_dr${DEMAND_REDUCE}"
[[ -n "$LOOKAHEAD" ]] && RUN_TAG="${RUN_TAG}_la${LOOKAHEAD}"
[[ -n "$CROSS_DOC_REPLAY" ]] && RUN_TAG="${RUN_TAG}_xdoc"
RUN_TAG="${RUN_TAG_OVERRIDE:-$RUN_TAG}"

# The memory knobs. QUERY_CHUNK 1024 is the measured sweet spot; lower it if something else is
# sharing the GPU. LOGIT_CHUNK is optional on this box (27.8 -> 26.0 GiB at 8K for 128) and is the
# replacement for LIGER.
QUERY_CHUNK="${QUERY_CHUNK:-1024}"
LOGIT_CHUNK="${LOGIT_CHUNK:-}"

# Steps between shuffle controls. Two extra replay passes each (~27 s at 16K), so one per 100 steps
# costs well under 1% of the run for the only readout that proves the router learned a ranking.
#
# Left EMPTY here on purpose, with the default applied per mode below -- `probe` wants a tighter
# interval than `stage1_16k`. Defaulting it to 100 at this point would make each mode's own
# `${SHUFFLE_EVERY:-20}` fallback dead code, which is how the first version of this script silently
# ran the 100-step probe with exactly one control (at step 0, before anything could be learned) and
# so reported nothing. Same shape as every other bug in this work: a knob that looks set and is not.
SHUFFLE_EVERY="${SHUFFLE_EVERY:-}"

MAX_STEPS="${MAX_STEPS:-600}"

cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false
# Long sequences fragment the caching allocator; expandable_segments lets it grow a segment instead
# of failing on a large contiguous request.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

# PYTHON/TORCHRUN are overridable because this repo's env is not always the activated one -- a
# non-interactive shell has neither on PATH, and the sibling scripts simply assume they are there.
# Defaults keep the same behaviour as those scripts when the env IS active.
PYTHON="${PYTHON:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"
if [[ "$NGPU" -gt 1 ]]; then
  LAUNCH=("$TORCHRUN" --nnodes "$NNODES" --nproc_per_node "$NGPU" --master_port "$MASTER_PORT")
else
  LAUNCH=("$PYTHON")
fi
if ! command -v "${LAUNCH[0]}" >/dev/null 2>&1; then
  echo "${LAUNCH[0]} not found on PATH. Activate the environment, or point at it explicitly:" >&2
  echo "  TORCHRUN=/opt/conda/envs/torch-base/bin/torchrun $0 $MODE" >&2
  exit 1
fi

logit_chunk_arg() { [[ -n "$LOGIT_CHUNK" ]] && echo "--logit-chunk $LOGIT_CHUNK"; }
resume_arg() { [[ -n "${RESUME:-}" ]] && echo "--resume-from $RESUME"; }

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
    # Two steps at 16K on all $NGPU ranks, so this also exercises NCCL init and the gradient
    # averaging, not just the objective. Watch for two things in the log beyond "it ran":
    #   * peak should be ~33.5 GiB. Much higher means the flex_attention path is NOT active.
    #   * NO "fell back to SDPA" and NO "recompile" warning. Either one means the memory fix is off,
    #     and the eager flex path is 14x WORSE than the bug it replaced (cross_replay_e2e.md §6.3).
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_cross_replay \
      --data-root "$DATA_ROOT" --model "$MODEL" $(data_args) \
      --schedule "${SCHEDULE:-16384:2}" \
      --n-sink "$N_SINK" --scalar-mid-dim "$MID_DIM" --budget "$BUDGET" \
      --query-chunk "$QUERY_CHUNK" $(logit_chunk_arg) \
      --num-workers 0 --log-every 1 --save-every 0 \
      --shuffle-control-every 1 \
      --out "$OUT/smoke" --dry-run
    ;;

  probe)
    # 100 steps at 8K, ~15 min on 4 idle GPUs. The point is NOT the loss -- it is to answer "does
    # this objective learn anything on real text", which random token ids cannot establish. Read
    # participation and shuffle_delta from the log or metrics.jsonl:
    #   participation falling from ~1.0, and shuffle_delta clearly > 0  -> proceed to stage1_16k
    #   participation stuck at ~1.0, or shuffle_delta <= 0              -> STOP and investigate
    # A cheap gate on a multi-hour run, and the only check that catches a trained-looking router.
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_cross_replay \
      --data-root "$DATA_ROOT" --model "$MODEL" $(data_args) \
      --schedule "${SCHEDULE:-8192:100}" \
      --n-sink "$N_SINK" --scalar-mid-dim "$MID_DIM" --budget "$BUDGET" \
      --query-chunk "$QUERY_CHUNK" $(logit_chunk_arg) \
      --global-batch-size "${GLOBAL_BATCH:-4}" \
      --compression-ratio "$COMPRESSION_RATIO" \
      --peak-lr "$PEAK_LR" --final-lr "$FINAL_LR" \
      --warmup-frac "$WARMUP_FRAC" --stable-frac "$STABLE_FRAC" \
      --batch-size 1 --take-from random --shuffle-buffer 64 \
      --num-workers "${WORKERS:-2}" \
      --shuffle-control-every "${SHUFFLE_EVERY:-20}" \
      --out "$OUT/probe" --metrics-file "$OUT/probe/metrics.jsonl" \
      --save-every "${SAVE_EVERY:-0}" --log-every "${LOG_EVERY:-5}"
    ;;

  stage1_16k)
    # THE RUN. 8K for 300 steps then 16K for 300, stopping at 600.
    #
    # WHY THIS SCHEDULE. It reuses the distillation and e2e runs' exact
    # 8192:300,16384:300,32768:900 curve and truncates with MAX_STEPS, so the LR at every step <= 600
    # is IDENTICAL to theirs (warmup 150, then flat at PEAK_LR -- decay would not begin until 1050),
    # and the 32K leg is never built. Steps 0..299 run at 8K and 300..599 at 16K, exactly as they do.
    # So step600.pt is comparable across all three objectives, which is the entire point of matching
    # everything that is not the objective.
    #
    # 32K is deliberately unreachable here: d_max = 2N-1 = 65535 exceeds Qwen3-8B's
    # max_position_embeddings of 40960, which would put replay queries at positions never trained
    # (cross_replay_e2e.md §7.3). The driver warns; MAX_STEPS=600 stops before it matters.
    #
    # GLOBAL_BATCH=8 on 4 ranks -> accum 2, i.e. 8 sequences per optimizer step, matching the other
    # runs' 8-way DDP. Expect ~0.9 h for the 8K leg and ~2.3 h for 16K on idle GPUs -- longer if the
    # box is shared, which it usually is.
    exec "${LAUNCH[@]}" -m scripts.train_gqa_indexer_cross_replay \
      --data-root "$DATA_ROOT" --model "$MODEL" $(data_args) \
      --schedule "${SCHEDULE:-8192:300,16384:300,32768:900}" \
      --max-steps "${MAX_STEPS:-600}" \
      --n-sink "$N_SINK" --scalar-mid-dim "$MID_DIM" --budget "$BUDGET" \
      --query-chunk "$QUERY_CHUNK" $(logit_chunk_arg) \
      --demand-reduce "$DEMAND_REDUCE" $(lookahead_arg) $(cross_doc_arg) \
      --global-batch-size "${GLOBAL_BATCH:-8}" \
      --compression-ratio "$COMPRESSION_RATIO" \
      --peak-lr "$PEAK_LR" --final-lr "$FINAL_LR" \
      --warmup-frac "$WARMUP_FRAC" --stable-frac "$STABLE_FRAC" \
      --batch-size 1 --take-from random --shuffle-buffer 64 \
      --num-workers "${WORKERS:-2}" \
      --shuffle-control-every "${SHUFFLE_EVERY:-100}" \
      $(resume_arg) \
      --out "$OUT/$RUN_TAG" --metrics-file "$OUT/$RUN_TAG/metrics.jsonl" \
      --save-every "${SAVE_EVERY:-200}" --log-every "${LOG_EVERY:-10}"
    ;;

  *)
    echo "usage: $0 {smoke|probe|stage1_16k}" >&2
    echo "  (tokenize with scripts/train_gqa_indexer.sh tokenize -- the corpus is shared)" >&2
    echo "  RESUME=\$OUT/stage1/step200.pt $0 stage1_16k   # continue an interrupted run" >&2
    echo "  no 'ablate' mode: --pin-mode none is structurally impossible here, see the header" >&2
    exit 1
    ;;
esac
