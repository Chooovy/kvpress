#!/usr/bin/env bash
set -euo pipefail

# ========= 用户配置区（按你现在的设置来） =========
OUT_DIR="./results_needle"

dataset="needle_in_haystack"
data_dir=""   # 为空就行；为空时不会进入目录名（与你的 python 一致）:contentReference[oaicite:2]{index=2}

models=(
  "/aifs4su/guhao/checkpoints/llama3.1-8b-instruct-memory-max-mode"
)

press_names=(
  "memory_query_indexer_max"
)

compression_ratios=(0.10 0.25 0.50 0.75 0.90)
context_lengths=(1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000 10500 11000 11500 12000 12500)
needle_depths=(15 25 35 45 55 65 75 85 95)

device="cuda"
gpu_ids=(0 1 2 3 4 5 6 7)
max_jobs=${#gpu_ids[@]}
next_gpu_idx=0

# RUN=0 只列出缺的；RUN=1 自动补跑
RUN="${RUN:-1}"

# ========= 内部函数 =========
model_basename () {
  local m="$1"
  echo "${m##*/}"
}

# 复刻 evaluate_bak_xintong.py 的目录名拼法:contentReference[oaicite:3]{index=3}
build_dirname () {
  local model_path="$1"
  local press="$2"
  local cr="$3"
  local ctx="$4"
  local depth="$5"
  local mname
  mname="$(model_basename "$model_path")"
  local cr2
  cr2="$(printf "%.2f" "$cr")"
  echo "${dataset}__${mname}__${press}__cr${cr2}__max_context${ctx}__needle_depth${depth}"
}

# 判断该组合是否“已完成”
# 你的 python 会在目录存在时创建 /1 /2... 子目录:contentReference[oaicite:4]{index=4}
# 所以这里：base 以及 base 的一层子目录里，只要找到 predictions.csv + metrics.json 就算 done
is_done () {
  local base="$1"

  # 1) base 直接完成
  if [[ -f "${base}/predictions.csv" && -f "${base}/metrics.json" ]]; then
    return 0
  fi

  # 2) base 的一级子目录完成（/1 /2 ...）
  local hit
  hit="$(find "$base" -mindepth 2 -maxdepth 2 -type f -name metrics.json 2>/dev/null | head -n 1 || true)"
  if [[ -n "$hit" ]]; then
    local subdir
    subdir="$(dirname "$hit")"
    if [[ -f "${subdir}/predictions.csv" ]]; then
      return 0
    fi
  fi

  return 1
}

# ========= 1) 扫描缺口 =========
mkdir -p "$OUT_DIR"
missing_list="./missing_tasks.tsv"
: > "$missing_list"

total=0
done_cnt=0
miss_cnt=0

for press in "${press_names[@]}"; do
  for model in "${models[@]}"; do
    for cr in "${compression_ratios[@]}"; do
      for ctx in "${context_lengths[@]}"; do
        for depth in "${needle_depths[@]}"; do
          total=$((total+1))
          dname="$(build_dirname "$model" "$press" "$cr" "$ctx" "$depth")"
          base="${OUT_DIR}/${dname}"

          if [[ -d "$base" ]] && is_done "$base"; then
            done_cnt=$((done_cnt+1))
          else
            miss_cnt=$((miss_cnt+1))
            printf "%s\t%s\t%s\t%s\t%s\n" "$model" "$press" "$cr" "$ctx" "$depth" >> "$missing_list"
          fi
        done
      done
    done
  done
done

echo "[SCAN] total=${total} done=${done_cnt} missing=${miss_cnt}"
echo "[SCAN] missing list saved to: ${missing_list}"

if [[ "$RUN" == "0" ]]; then
  echo "[INFO] RUN=0, only scanned. Exit."
  exit 0
fi

if [[ "$miss_cnt" -eq 0 ]]; then
  echo "[INFO] Nothing to run."
  exit 0
fi

# ========= 2) 只补跑缺的 =========
while IFS=$'\t' read -r model press cr ctx depth; do
  # GPU slot
  while [ "$(jobs -rp | wc -l)" -ge "$max_jobs" ]; do
    wait -n
  done

  device_id=${gpu_ids[$next_gpu_idx]}
  next_gpu_idx=$(( (next_gpu_idx + 1) % max_jobs ))

  # 如果 base 目录存在但不完整，会触发 python 自动写到 /1 /2...
  # 为了避免目录越堆越多：只在“完全没完成”的情况下删掉空壳（没有 metrics/predictions 的）
  dname="$(build_dirname "$model" "$press" "$cr" "$ctx" "$depth")"
  base="${OUT_DIR}/${dname}"
  if [[ -d "$base" ]] && ! is_done "$base"; then
    rm -rf "$base"
  fi

  echo "RUN press=$press cr=$cr ctx=$ctx depth=$depth gpu=$device_id"
  CUDA_VISIBLE_DEVICES=$device_id \
  python evaluate_bak_xintong.py \
    --dataset "$dataset" \
    --model "$model" \
    --data_dir "$data_dir" \
    --press_name "$press" \
    --compression_ratio "$cr" \
    --max_context_length "$ctx" \
    --needle_depth "$depth" \
    --device "${device}:0" \
    --compress_questions False \
    --output_dir "$OUT_DIR" &

done < "$missing_list"

wait
echo "[DONE] All missing tasks completed."
