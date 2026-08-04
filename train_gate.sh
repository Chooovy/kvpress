# accelerate launch --num_processes 4 train_score_gate.py \
#     --press_method gated \
#     --gated_objective mse_reg \
#     --model_name_or_path /aifs4su/guhao/Models/Llama-3.1-8B-Instruct \
#     --output_dir /aifs4su/guhao/checkpoints/llama31_8b_gated_ckpt \
#     --per_device_train_batch_size 1 \
#     --pt_context_len 8192 \
#     --max_train_samples 8192 \
#     --mixed_precision bf16 \
#     --gate_type elementwise \
#     --reg_type group_lasso \
#     --gate_init open \
#     --gate_init_open_p 0.99 \
#     --reg_lambda 0.06 \
#     --reg_warmup_steps 400 \
#     --num_train_epochs 1 \
#     --learning_rate 0.02 \
#     --logging_steps 10 \
#     --attn_implementation sdpa

# accelerate launch --num_processes 8 train_score_gate.py \
#   --press_method gated \
#   --model_name_or_path /aifs4su/guhao/Models/Llama-3.1-8B-Instruct \
#   --output_dir /aifs4su/guhao/checkpoints/llama31_8b_gated_distill_cr50 \
#   --per_device_train_batch_size 1 \
#   --pt_context_len 6144 \
#   --max_train_samples 8192 \
#   --mixed_precision bf16 \
#   --gate_type elementwise \
#   --gate_init open --gate_init_open_p 0.99 \
#   --num_train_epochs 2 \
#   --learning_rate 0.02 \
#   --logging_steps 10 \
#   --attn_implementation sdpa \
#   --gated_objective lm_distill \
#   --distill_temperature 2.0 --distill_lambda 1.0 \
#   --distill_bin_lambda 0.05 \
#   --key_channel_cr 0.5 \
#   --train_gate_mode ste_topk --ste_warmup_steps 200 \
#   --pairwise_prune --sync_kv_prune


# accelerate launch --num_processes 8 train_score_gate.py \
#   --press_method gated \
#   --model_name_or_path /aifs4su/guhao/Models/Llama-3.1-8B-Instruct \
#   --output_dir /aifs4su/guhao/checkpoints/llama31_8b_gated_distill_cr50 \
#   --per_device_train_batch_size 1 \
#   --pt_context_len 6144 \
#   --max_train_samples 8192 \
#   --mixed_precision bf16 \
#   --gate_type elementwise \
#   --gate_init open --gate_init_open_p 0.99 \
#   --num_train_epochs 2 \
#   --learning_rate 1e-3 \
#   --logging_steps 10 \
#   --attn_implementation sdpa \
#   --gated_objective lm_distill \
#   --distill_temperature 2.0 --distill_lambda 1.0 \
#   --distill_bin_lambda 0.05 \
#   --key_channel_cr 0.5 \
#   --train_gate_mode ste_topk --ste_warmup_steps 200 \
#   --pairwise_prune --sync_kv_prune
accelerate launch --num_processes 8 train_score_gate.py \
  --press_method gated \
  --model_name_or_path /aifs4su/guhao/Models/Llama-3.1-8B-Instruct \
  --output_dir /aifs4su/guhao/checkpoints/llama31_8b_gated_lm_distill_bin_0.5budget0.5_hard_ste \
  --per_device_train_batch_size 1 \
  --pt_context_len 6144 \
  --max_train_samples 8192 \
  --mixed_precision bf16 \
  --gate_type elementwise \
  --gate_init open --gate_init_open_p 0.99 \
  --num_train_epochs 1 \
  --learning_rate 1e-3 \
  --logging_steps 10 \
  --gated_objective lm_distill \
  --distill_temperature 2.0 --distill_lambda 1.0 \
  --distill_budget_lambda 5.0 --distill_budget_rho 0.5 \
  --distill_bin_lambda 0.05 \
  --key_channel_cr 0.5 \
  --train_gate_mode ste_topk --ste_warmup_steps 200 \
  --pairwise_prune --sync_kv_prune \
  --attn_implementation sdpa



# accelerate launch --num_processes 8 train_score_gate.py \
#     --press_method gated \
#     --model_name_or_path /aifs4su/guhao/Models/Llama-3.1-8B-Instruct \
#     --output_dir /aifs4su/guhao/checkpoints/llama31_8b_static_gated_lm_distill_bin_budget \
#     --per_device_train_batch_size 1 \
#     --pt_context_len 6144 \
#     --max_train_samples 8192 \
#     --mixed_precision bf16 \
#     --gate_type elementwise \
#     --gate_init open \
#     --gate_init_open_p 0.99 \
#     --num_train_epochs 1 \
#     --learning_rate 0.02 \
#     --logging_steps 10 \
#     --gated_objective lm_distill \
#     --distill_temperature 2.0 --distill_lambda 1.0 \
#     --distill_budget_lambda 0.1 --distill_budget_rho 0.5 \
#     --distill_bin_lambda 0.05 \
#     --gate_mode static \
#     --attn_implementation sdpa
