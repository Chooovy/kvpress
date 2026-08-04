dataset="zero_scrolls"
model="/aifs4su/guhao/Models/modelzoo/qwen3/qwen3-8b"
compression_ratios=(0.5)
press_names=("snapkv" "tova" "pyramidkv" "expected_attention")

gpu_ids=(0 1 2 3 4 5 6 7)
max_jobs=${#gpu_ids[@]}
next_gpu_idx=0

export HF_HOME=${HF_HOME:-~/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
# 如果你代码里已 trust_remote_code=True，这行可有可无；但并行跑更保险
export HF_DATASETS_TRUST_REMOTE_CODE=1

for press_name in "${press_names[@]}"; do
  for compression_ratio in "${compression_ratios[@]}"; do

    while [ "$(jobs -rp | wc -l)" -ge "$max_jobs" ]; do
      wait -n
    done

    device_id=${gpu_ids[$next_gpu_idx]}
    next_gpu_idx=$(( (next_gpu_idx + 1) % max_jobs ))

    echo "Running $press_name cr=$compression_ratio on GPU $device_id"
    CUDA_VISIBLE_DEVICES=$device_id python evaluate_xintong.py \
      --dataset "$dataset" \
      --model "$model" \
      --press_name "$press_name" \
      --compression_ratio "$compression_ratio" \
      --device "cuda:0" \
      --compress_questions False \
      --output_dir "./results_zeroscrolls" &
  done
done

wait
echo "All evaluation tasks completed."
