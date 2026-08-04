#!/usr/bin/env bash
set -euo pipefail

CONFIG=./evaluate_config_longbench_kvzip.yaml
PRESS=kvzip
LOG_DIR=./logs
mkdir -p "$LOG_DIR"

MODELS=(
  "/aifs4su/guhao/Models/Mistral-7B-v0.3"
  "/aifs4su/guhao/Models/Llama-3.1-8B"
  "/aifs4su/guhao/Models/modelzoo/qwen3/qwen3-8b"
  "/aifs4su/guhao/Models/gemma-3-12b-it"
)

RATIOS=("0.0" "0.1" "0.25" "0.5" "0.75" "0.9")

model_short () { basename "$1"; }

cr_to_tag () {
  local cr="$1"
  python - <<PY
cr=float("$cr")
print(int(round(cr*100)))
PY
}

ctx_for_model () {
  local model="$1"
  if [[ "$model" == *"Llama-3.1"* ]] || [[ "$model" == *"gemma-3-12b-it"* ]]; then
    echo 131072
  else
    echo 32768
  fi
}

# Slurm 分配几张卡就用几张（一般是 8）
GPU_COUNT="${SLURM_GPUS_ON_NODE:-8}"
GPUS=()
for ((i=0; i<GPU_COUNT; i++)); do GPUS+=("$i"); done

echo "==== SLURM GPUs: ${GPU_COUNT} | Using: ${GPUS[*]} ===="

TASKS=()
for model in "${MODELS[@]}"; do
  ctx="$(ctx_for_model "$model")"
  for cr in "${RATIOS[@]}"; do
    TASKS+=( "${model}|||${ctx}|||${cr}" )
  done
done
total=${#TASKS[@]}
echo "==== Total tasks: ${total} (4 models x 6 ratios) ===="

FAIL_LIST="${LOG_DIR}/failed_${PRESS}_longbench.txt"
: > "$FAIL_LIST"
FAILED=0

declare -A pid2gpu=()
declare -A pid2task=()

run_one () {
  local gpu="$1" model="$2" ctx="$3" cr="$4"

  local mtag tag out_root out_dir log
  mtag="$(model_short "$model")"
  tag="$(cr_to_tag "$cr")"

  out_root="./results_longbench_${mtag}_${PRESS}"
  out_dir="${out_root}/Longbench_${tag}"
  mkdir -p "$out_dir"

  log="${LOG_DIR}/${PRESS}_${mtag}_longbench_ctx${ctx}_cr${tag}.log"

  echo "[LAUNCH] GPU=${gpu} ${PRESS} model=${mtag} ctx=${ctx} cr=${cr}"

  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
  TOKENIZERS_PARALLELISM=false \
  python -m evaluate \
    --config_file "$CONFIG" \
    --output_dir "$out_dir" \
    --dataset longbench \
    --model "$model" \
    --press_name "$PRESS" \
    --compression_ratio "$cr" \
    --max_context_length "$ctx" \
    --device 0 \
    >"$log" 2>&1
}

launch_task () {
  local gpu="$1" task="$2"
  local model ctx cr
  model="$(awk -F '\\|\\|\\|' '{print $1}' <<<"$task")"
  ctx="$(awk -F '\\|\\|\\|' '{print $2}' <<<"$task")"
  cr="$(awk -F '\\|\\|\\|' '{print $3}' <<<"$task")"

  run_one "$gpu" "$model" "$ctx" "$cr" &
  local pid=$!
  pid2gpu["$pid"]="$gpu"
  pid2task["$pid"]="$task"
}

task_idx=0

for gpu in "${GPUS[@]}"; do
  if [ "$task_idx" -ge "$total" ]; then break; fi
  launch_task "$gpu" "${TASKS[$task_idx]}"
  task_idx=$((task_idx + 1))
done

while [ "${#pid2gpu[@]}" -gt 0 ]; do
  donepid=""
  if wait -n -p donepid; then status=0; else status=$?; fi
  [ -z "${donepid:-}" ] && continue

  gpu="${pid2gpu[$donepid]}"
  task="${pid2task[$donepid]}"

  model="$(awk -F '\\|\\|\\|' '{print $1}' <<<"$task")"
  ctx="$(awk -F '\\|\\|\\|' '{print $2}' <<<"$task")"
  cr="$(awk -F '\\|\\|\\|' '{print $3}' <<<"$task")"
  mtag="$(model_short "$model")"
  tag="$(cr_to_tag "$cr")"

  if [ "$status" -ne 0 ]; then
    echo "[FAIL rc=${status}] GPU=${gpu} model=${mtag} ctx=${ctx} cr=${cr}" >&2
    printf "FAIL rc=%s | gpu=%s | model=%s | ctx=%s | cr=%s | log=%s\n" \
      "$status" "$gpu" "$mtag" "$ctx" "$cr" \
      "${LOG_DIR}/${PRESS}_${mtag}_longbench_ctx${ctx}_cr${tag}.log" >> "$FAIL_LIST"
    FAILED=1
  else
    echo "[OK] GPU=${gpu} model=${mtag} ctx=${ctx} cr=${cr}"
  fi

  unset pid2gpu["$donepid"]
  unset pid2task["$donepid"]

  if [ "$task_idx" -lt "$total" ]; then
    launch_task "$gpu" "${TASKS[$task_idx]}"
    task_idx=$((task_idx + 1))
  fi
done

echo "==== ALL ${total} TASKS FINISHED (${PRESS}) ===="
echo "Failure list: ${FAIL_LIST}"
exit "$FAILED"
