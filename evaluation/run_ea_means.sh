#!/usr/bin/env bash
set -euo pipefail

# =========================
# User-editable section
# =========================
DATASET="ruler"
DATA_DIR="4096"
FRACTION="0.10"                 # 你现在结果目录里是 fraction0.100
OUTPUT_DIR="./results_ea_means"  # 统一输出根目录（可改成 results_debug）
LOG_DIR="./logs_ea_means"

# 模型：按你的服务器实际路径填（你可以继续加）
MODELS=(
  "/aifs4su/guhao/Models/Llama-3.2-1B-Instruct"
  "/aifs4su/guhao/Models/Llama-3.2-3B-Instruct"
  "/aifs4su/guhao/Models/Qwen2.5-0.5B-Instruct"
  "/aifs4su/guhao/Models/DeepSeek-R1-Distill-Qwen-1.5B"
)

# 压缩率
CRS=(0.25 0.50 0.75 0.90)

# 三种要对比的方法
PRESSES=(
  "expected_attention"
  "expected_attention_layer_mean"
  "expected_attention_head_mean"
)

# 并行用哪些 GPU（按需改；单卡就填一个）
GPUS=(0 1 2 3)

# 可选：限制样本数（不想限制就留空）
SAMPLES=""   # 例如 "256"

# 可选：强制 max_context_length（不填则由数据/默认逻辑决定）
MAX_CTX=""   # 例如 "4096"

# =========================
# Do not edit below
# =========================
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# 检查 press 是否在 PRESS_REGISTRY 里（不在就跳过，不让脚本整体中断）
press_supported () {
  local press="$1"
  python - <<PY
from evaluate_registry import PRESS_REGISTRY
print("1" if "${press}" in PRESS_REGISTRY else "0")
PY
}

sanitize () {
  echo "$1" | sed 's#[/ ]#_#g' | sed 's#__*#__#g'
}

run_one () {
  local model="$1"
  local press="$2"
  local cr="$3"
  local gpu="$4"

  local model_name
  model_name="$(basename "$model")"
  local tag
  tag="$(sanitize "${DATASET}__${DATA_DIR}__${model_name}__${press}__cr${cr}")"
  local log="${LOG_DIR}/${tag}_gpu${gpu}.log"

  if [[ ! -d "$model" ]]; then
    echo "[SKIP] model path not found: $model"
    return 0
  fi

  if [[ "$(press_supported "$press")" != "1" ]]; then
    echo "[SKIP] press not in PRESS_REGISTRY: $press"
    return 0
  fi

  echo "[RUN] gpu=${gpu} model=${model_name} press=${press} cr=${cr} -> ${log}"

  local args=(
    python evaluate.py
    --dataset "$DATASET"
    --data_dir "$DATA_DIR"
    --model "$model"
    --press_name "$press"
    --compression_ratio "$cr"
    --fraction "$FRACTION"
    --output_dir "$OUTPUT_DIR"
    --device 0
    --log_level INFO
  )

  if [[ -n "$SAMPLES" ]]; then
    args+=(--samples "$SAMPLES")
  fi
  if [[ -n "$MAX_CTX" ]]; then
    args+=(--max_context_length "$MAX_CTX")
  fi

  CUDA_VISIBLE_DEVICES="$gpu" "${args[@]}" >"$log" 2>&1
  echo "[OK] ${tag}"
}

# 简易并行：按 GPUS 轮转，最多同时跑 len(GPUS) 个
FAIL=0
pids=()
idx=0

for model in "${MODELS[@]}"; do
  for cr in "${CRS[@]}"; do
    for press in "${PRESSES[@]}"; do
      gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
      idx=$((idx+1))

      # 起后台任务
      ( run_one "$model" "$press" "$cr" "$gpu" ) || FAIL=1 &
      pids+=($!)

      # 控制并发：达到 GPU 数就等一个结束
      if (( ${#pids[@]} >= ${#GPUS[@]} )); then
        wait -n || FAIL=1
        # 清理已结束 pid
        alive=()
        for pid in "${pids[@]}"; do
          if kill -0 "$pid" 2>/dev/null; then
            alive+=("$pid")
          fi
        done
        pids=("${alive[@]}")
      fi
    done
  done
done

# 等剩余的
for pid in "${pids[@]}"; do
  wait "$pid" || FAIL=1
done

echo "========================="
echo "[DONE] output_dir: $OUTPUT_DIR"
echo "[DONE] logs:       $LOG_DIR"
echo "========================="

if (( FAIL != 0 )); then
  echo "[WARN] Some runs failed. Check logs in: $LOG_DIR"
  exit 1
fi
