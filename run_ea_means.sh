#!/usr/bin/env bash
set -euo pipefail

# ======================
# Paths / basic settings
# ======================
ROOT_DIR="/aifs4su/guhao/KVCache/kvpress"
EVAL_DIR="${ROOT_DIR}/evaluation"

# 结果输出（可按需改绝对路径）
OUTDIR="${OUTDIR:-${EVAL_DIR}/results_ea_means}"
LOGDIR="${LOGDIR:-${ROOT_DIR}/logs/ea_means}"
mkdir -p "$OUTDIR" "$LOGDIR"

# 用你已有的 evaluate_config.yaml（如果你有别的，就 export CONFIG=... 覆盖它）
CONFIG="${CONFIG:-${EVAL_DIR}/evaluate_config.yaml}"

# 先 sanity 可以设小点（例如 0.1）；正式对比就 1.0
FRACTION="${FRACTION:-1.0}"

# ======================
# Experiment grid
# ======================
CRS=(0.25 0.50 0.75 0.90)

PRESSES=(
  "expected_attention"
  "expected_attention_layer_mean"
  "expected_attention_head_mean"
)

MODELS=(
  "/aifs4su/guhao/Models/Llama-3.2-1B-Instruct"
  "/aifs4su/guhao/Models/Llama-3.2-3B-Instruct"
  "/aifs4su/guhao/Models/Qwen2.5-0.5B-Instruct"
  "/aifs4su/guhao/Models/DeepSeek-R1-Distill-Qwen-1.5B"
)

# ======================
# GPU detection (from Slurm)
# ======================
# Slurm 通常会设置 CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
else
  GPUS=(0)
fi
NGPU="${#GPUS[@]}"

echo "[INFO] Using GPUs: ${GPUS[*]}"
echo "[INFO] CONFIG=$CONFIG"
echo "[INFO] OUTDIR=$OUTDIR"
echo "[INFO] FRACTION=$FRACTION"
echo

run_one () {
  local gpu="$1"
  local model="$2"
  local press="$3"
  local cr="$4"

  local mname
  mname="$(basename "$model")"
  local cr_tag
  cr_tag="$(printf "%.2f" "$cr")"

  local log="${LOGDIR}/ea_means__${mname}__${press}__cr${cr_tag}__gpu${gpu}.log"

  echo "===== RUN gpu=$gpu model=$mname press=$press cr=$cr_tag =====" | tee "$log"

  # 绑单卡：把物理 GPU 映射成进程内的 cuda:0，所以 --device 0
  CUDA_VISIBLE_DEVICES="$gpu" \
  python "${EVAL_DIR}/evaluate.py" \
    --config_file "$CONFIG" \
    --model "$model" \
    --press_name "$press" \
    --compression_ratio "$cr" \
    --fraction "$FRACTION" \
    --output_dir "$OUTDIR" \
    --device 0 \
    --log_level "INFO" \
    2>&1 | tee -a "$log"
}

# ======================
# Simple job pool: at most NGPU concurrent jobs
# ======================
cd "$ROOT_DIR"

for model in "${MODELS[@]}"; do
  for press in "${PRESSES[@]}"; do
    for cr in "${CRS[@]}"; do
      # 等到有空闲 slot
      while (( $(jobs -pr | wc -l) >= NGPU )); do
        sleep 2
      done

      # 轮询分配 GPU
      idx=$(( $(jobs -pr | wc -l) % NGPU ))
      gpu="${GPUS[$idx]}"

      run_one "$gpu" "$model" "$press" "$cr" &
    done
  done
done

wait
echo "[DONE] All runs finished. Results in: $OUTDIR"