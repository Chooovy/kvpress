set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="1"

DATASET="${DATASET:-ruler}"
MODEL="${MODEL:-/apdcephfs_zw31/share_303843174/user/marcushaogu/models/Qwen3-8B}"
CKPT="${CKPT:-/apdcephfs_zw31/share_303843174/user/marcushaogu/models/Qwen-3-8B-gqa_indexer/stage1/step600.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-./results_indexer_press_max_reduce}"

N_SINK="${N_SINK:-4}"
N_LOCAL="${N_LOCAL:-64}"
QUERY_REDUCE="${QUERY_REDUCE:-max}"

FRACTION="${FRACTION:-0.1}"
SEED="${SEED:-42}"
ATTN="${ATTN:-sdpa}"

read -r -a RATIOS <<< "${RATIOS:-0.5}"
read -r -a LENGTHS <<< "${LENGTHS:-8192}"

if [[ ! -f "$CKPT" ]]; then
  echo "indexer checkpoint not found at $CKPT (set CKPT=)" >&2
  exit 1
fi

num_gpus=$(nvidia-smi --list-gpus | wc -l)
NGPU="${NGPU:-$num_gpus}"
if [[ "$NGPU" -gt "$num_gpus" ]]; then
  echo "Error: NGPU=$NGPU exceeds the $num_gpus GPUs on this box" >&2
  exit 1
fi

JOBS=()
for length in "${LENGTHS[@]}"; do
  for ratio in "${RATIOS[@]}"; do
    JOBS+=("$length:$ratio")
  done
done
echo "${#JOBS[@]} job(s) over $NGPU GPU(s): lengths=${LENGTHS[*]} ratios=${RATIOS[*]}"

for ((g = 0; g < NGPU; g++)); do
  (
    for ((j = g; j < ${#JOBS[@]}; j += NGPU)); do
      length="${JOBS[$j]%%:*}"
      ratio="${JOBS[$j]##*:}"
      EXTRA=()
      [[ -n "$length" ]] && EXTRA+=(--data_dir "$length")
      echo "  compression_ratio=$ratio @ ${length:-default} on cuda:$g"
      python evaluate_indexer_press.py \
        --dataset "$DATASET" --model "$MODEL" --indexer_ckpt "$CKPT" \
        --compression_ratio "$ratio" \
        --n_sink "$N_SINK" --n_local "$N_LOCAL" --query_reduce "$QUERY_REDUCE" \
        --attn_implementation "$ATTN" \
        --fraction "$FRACTION" --seed "$SEED" --output_dir "$OUTPUT_DIR" \
        --device "cuda:$g" \
        "${EXTRA[@]}"
    done
  ) &
done

wait
echo "All indexer-press evaluations completed."
