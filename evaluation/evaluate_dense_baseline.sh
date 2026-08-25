set -euo pipefail

cd "$(dirname "$0")"

DATASET="${DATASET:-ruler}"
MODEL="${MODEL:-/apdcephfs_zw31/share_303843174/user/marcushaogu/models/Qwen3-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-./results_dense}"
FRACTION="${FRACTION:-0.1}"
SEED="${SEED:-42}"

ATTN="${ATTN:-sdpa}"

read -r -a LENGTHS <<< "${LENGTHS:-4096 8192 16384 32768}"

num_gpus=$(nvidia-smi --list-gpus | wc -l)
if [[ "${#LENGTHS[@]}" -gt "$num_gpus" ]]; then
  echo "Error: ${#LENGTHS[@]} lengths exceed the $num_gpus GPUs on this box" >&2
  exit 1
fi

echo "dense baseline: $DATASET on $MODEL, lengths=${LENGTHS[*]}, fraction=$FRACTION seed=$SEED"

for i in "${!LENGTHS[@]}"; do
  length="${LENGTHS[$i]}"
  (
    EXTRA=()
    [[ -n "$length" ]] && EXTRA+=(--data_dir "$length")
    echo "  no_press @ ${length:-default} on cuda:$i"
    python evaluate.py \
      --dataset "$DATASET" --model "$MODEL" --press_name no_press \
      --attn_implementation "$ATTN" \
      --fraction "$FRACTION" --seed "$SEED" \
      --output_dir "$OUTPUT_DIR" --device "cuda:$i" \
      "${EXTRA[@]}"
  ) &
done

wait
echo "Dense baseline completed. Results under $OUTPUT_DIR/"
