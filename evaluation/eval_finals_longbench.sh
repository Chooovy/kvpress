#!/usr/bin/env bash
set -u  # 不用 -e，避免某个子任务失败把整个大循环直接炸掉

dataset="longbench"
tasks=(
  "2wikimqa" "dureader" "gov_report" "hotpotqa" "lcc" "lsht" "multi_news"
  "multifieldqa_en" "multifieldqa_zh" "musique" "narrativeqa" "passage_count"
  "passage_retrieval_en" "passage_retrieval_zh" "qasper" "qmsum" "repobench-p"
  "samsum" "trec" "triviaqa" "vcsum"
)

model=(
  "/aifs4su/guhao/KVCache/checkpoints/llama-3-8b-1m-query-indexer"
)

compression_ratios=(0.10 0.25 0.5 0.75 0.9)
press_names=("query_indexer_score_block")

device="cuda"
gpu_ids=(0 1 2 3 4 5 6 7)

# 关键：每个 job 独立 lockdir（SLURM_JOB_ID 是 Slurm 常用环境变量）:contentReference[oaicite:0]{index=0}
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

kvq_modes=("none" "8" "4")

for press_name in "${press_names[@]}"; do
  for model_path in "${model[@]}"; do
    for task in "${tasks[@]}"; do
      for compression_ratio in "${compression_ratios[@]}"; do
        for kvq in "${kvq_modes[@]}"; do

          gid="$(pick_free_gpu)"

          extra_args=()
          kvq_tag="kvq_${kvq}"

          if [[ "$kvq" == "8" ]]; then
            extra_args+=( --kv_cache_nbits 8 --kv_cache_backend hqq )
          elif [[ "$kvq" == "4" ]]; then
            extra_args+=( --kv_cache_nbits 4 --kv_cache_backend quanto )
          fi

          echo "[LAUNCH] task=$task press=$press_name cr=$compression_ratio kvq=$kvq GPU=$gid model=$model_path"

          (
            trap 'release_gpu "'"$gid"'"' EXIT

            CUDA_VISIBLE_DEVICES="$gid" \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            python evaluate_bak.py \
              --dataset "$dataset" \
              --data_dir "$task" \
              --model "$model_path" \
              --press_name "$press_name" \
              --compression_ratio "$compression_ratio" \
              --fraction 0.1 \
              --seed 42 \
              --device "${device}:0" \
              --compress_questions=False \
              --kvq_smoke_test True \
              --kvq_smoke_prompt_repeats 512 \
              --kvq_smoke_new_tokens 256 \
              --output_dir "./results_longbench_quant_1_frac0.1/${task}/${kvq_tag}" \
              "${extra_args[@]}"
          ) &

        done
      done
    done
  done
done

wait
echo "[DONE] All evaluation tasks completed."
