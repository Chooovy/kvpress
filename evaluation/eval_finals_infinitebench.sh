dataset="infinitebench"
data_dir=(
  "passkey"
  "number_string"
  "kv_retrieval"
  "math_calc"
  "math_find"
  "code_run"
  "code_debug"
  "longdialogue_qa_eng"
)
model=(
  "/aifs4su/guhao/Models/Llama-3.1-8B-Instruct"
)

compression_ratios=(0.10 0.25 0.5 0.75 0.9)
press_names=("expected_attention" "tova" "snapkv" "pyramidkv")

device="cuda"
gpu_ids=(0 1 2 3 4 5 6 7)
max_jobs=${#gpu_ids[@]}
next_gpu_idx=0


for press_name in "${press_names[@]}"; do
    for model in "${model[@]}"; do
        for data_dir in "${data_dir[@]}"; do
            for compression_ratio in "${compression_ratios[@]}"; do

        while [ "$(jobs -rp | wc -l)" -ge "$max_jobs" ]; do
          wait -n
        done

            device_id=${gpu_ids[$next_gpu_idx]}
            next_gpu_idx=$(( (next_gpu_idx + 1) % max_jobs ))

            echo "Task: $data_dir | press: $press_name | cr: $compression_ratio | GPU: $device_id"
            CUDA_VISIBLE_DEVICES=$device_id python evaluate_bak_xintong.py \
            --dataset $dataset \
            --data_dir "$data_dir" \
            --model "$model" \
            --press_name $press_name \
            --compression_ratio $compression_ratio \
            --device ${device}:0 \
            --compress_questions False \
            --output_dir "./results_infinitebench_le128k/${data_dir}/${press_name}/cr${compression_ratio}" &

      done
    done
  done
done

wait
echo "All evaluation tasks completed."

