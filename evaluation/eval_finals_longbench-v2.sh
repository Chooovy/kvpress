#!/usr/bin/env bash
set -u  # 不用 -e，避免某个子任务失败把整个大循环直接炸掉

dataset="longbench-v2"

model=(
  "/aifs4su/guhao/KVCache/checkpoints/llama-3-8b-1m-query-indexer"
)

compression_ratios=(0.10 0.25 0.5 0.75 0.9)
press_names=("query_indexer_score_block")

device="cuda"
gpu_ids=(0 1 2 3 4 5 6 7)

# 关键：每个 job 独立 lockdir（SLURM_JOB_ID 是 Slurm 常用环境变量）
lockdir="./gpu_locks_${SLURM_JOB_ID:-$$}"
rm -rf "$lockdir"
mkdir -p "$lockdir"

# 让 python 输出更“实时”落到 slurm out/err（避免长时间看起来没动）
export PYTHONUNBUFFERED=1

cleanup_all_locks () {
  rm -rf "$lockdir"
}
trap cleanup_all_locks EXIT

# 原子抢锁：用 mkdir 当作 lock（mkdir 是原子操作，比 noclobber 文件更稳）
acquire_gpu_lock () {
  local gid="$1"
  local d="$lockdir/gpu${gid}.lockdir"
  local pidfile="$d/pid"

  # 1) 尝试直接拿锁
  if mkdir "$d" 2>/dev/null; then
    echo "$$" > "$pidfile"
    echo "$gid"
    return 0
  fi

  # 2) 拿不到：判断是不是陈旧锁（pid 不存在了就回收）
  if [[ -f "$pidfile" ]]; then
    local opid
    opid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -n "${opid:-}" ]] && ! kill -0 "$opid" 2>/dev/null; then
      rm -rf "$d"
      if mkdir "$d" 2>/dev/null; then
        echo "$$" > "$pidfile"
        echo "$gid"
        return 0
      fi
    fi
  fi

  return 1
}

pick_free_gpu () {
  while true; do
    for gid in "${gpu_ids[@]}"; do
      if out=$(acquire_gpu_lock "$gid"); then
        echo "$out"
        return 0
      fi
    done
    sleep 2
  done
}

release_gpu () {
  local gid="$1"
  rm -rf "$lockdir/gpu${gid}.lockdir"
}

for press_name in "${press_names[@]}"; do
  for model_path in "${model[@]}"; do
    for compression_ratio in "${compression_ratios[@]}"; do
      gid="$(pick_free_gpu)"

      echo "[LAUNCH] dataset=$dataset press=$press_name cr=$compression_ratio GPU=$gid model=$model_path mode=cloze-forward"

      (
        trap 'release_gpu "'"$gid"'"' EXIT

        CUDA_VISIBLE_DEVICES="$gid" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        USE_TF=0 \
        TRANSFORMERS_NO_TF=1 \
        USE_FLAX=0 \
        USE_TORCH=1 \
        python evaluate_bak.py \
          --dataset "$dataset" \
          --model "$model_path" \
          --press_name "$press_name" \
          --compression_ratio "$compression_ratio" \
          --fraction 1.0 \
          --seed 42 \
          --device "${device}:0" \
          --longbenchv2_eval cloze \
          --longbenchv2_cloze_impl forward \
          --max_input_tokens 65536 \
          --output_dir "./results_longbenchv2_cloze_forward_65536"
      ) &
    done
  done
done

wait
echo "[DONE] All evaluation tasks completed."
