#!/usr/bin/env bash
# Run from any directory; logs: TRAIN_LOG_DIR (default /aifs4su/guhao/logs).
#
# FSDP rollback: If training fails with FSDP, remove --use_fsdp below and (optionally)
# change accelerate `--mixed_precision`. Revert Python per comments in
# train_score_query_long_fused_loss.py `main()` near the Accelerator construction.
#
# Alternate (Qwen) example — paste into the block below if needed:
#   accelerate launch --num_processes 8 train_score_query_long_fused_loss.py \
#     --model_name_or_path /aifs4su/guhao/Models/modelzoo/qwen3/qwen3-8b \
#     --output_dir /aifs4su/guhao/checkpoints/qwen3-8b-indexer-max \
#     ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_ROOT="${TRAIN_LOG_DIR:-/aifs4su/guhao/logs}"
mkdir -p "$LOG_ROOT"
LOG_FILE="${LOG_ROOT}/train-long-$(date +%Y%m%d-%H%M%S)-$(hostname -s).log"

export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"

{
  echo "=== $(date -Is) train_long.sh ==="
  echo "host=$(hostname) pwd=$PWD"
  echo "log=$LOG_FILE"
  echo "python=$(command -v python)"
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true
  echo "----------------------------------------"
  # FSDP+bf16: accelerate mixed_precision `no` avoids stacking autocast on FSDP.
  # Default --fsdp_auto_wrap is decoder_layer (70B on 80GB). For no_wrap use: --fsdp_auto_wrap no_wrap
  # Smoke test: --save_steps 10. Typical full run: 500 or 1000.
  export NCCL_TIMEOUT=1800          # 30分钟（默认10分钟太短）
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800  
  export NCCL_DEBUG=WARN            # 出问题时打印更多信息
  export TORCH_NCCL_TRACE_BUFFER_SIZE=1000  # 记录通信trace，方便排查
  accelerate launch --num_processes 8 --mixed_precision no train_score_query_long_fused_loss.py \
    --model_name_or_path /aifs4su/guhao/Models/Llama-3.3-70B-Instruct \
    --output_dir /aifs4su/guhao/KVCache/checkpoints/llama-3.3-70B-query-indexer-math-longalpaca \
    --use_fsdp \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --press_method query_indexer_score \
    --num_train_epochs 1 \
    --max_lr 3e-4 --min_lr 5e-6 \
    --warmup_ratio 0.03 \
    --save_steps 500 \
    --n_sink 4 \
    --pt_context_len 2048 \
    --max_train_samples 65536 \
    --attn_implementation eager \
    --chunk_size 256 \
    --aggregate_mode max
} 2>&1 | tee "$LOG_FILE"

exit "${PIPESTATUS[0]}"
