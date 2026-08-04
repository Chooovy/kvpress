#!/bin/bash
#SBATCH --job-name=eval_finals                 # Job name
#SBATCH --partition=llm                 # Select llm-debug partition for debugging
#SBATCH --qos=llm                       # Use the llm-debug QOS
#SBATCH --nodes=1                             # Number of nodes
#SBATCH --gres=gpu:8                          # Number of GPUs (1 GPU)
#SBATCH --ntasks-per-node=1                   # Number of tasks per node
#SBATCH --cpus-per-task=128                    # Number of CPU cores per task
#SBATCH --time=3-00:00:00                       # Time limit: 3 days
#SBATCH --output=/aifs4su/guhao/KVCache/kvpress/logs/slurm-%j.out   # Standard output log
#SBATCH --error=/aifs4su/guhao/KVCache/kvpress/logs/slurm-%j.err    # Standard error log


export PIP_USER=false
export PYTHONNOUSERSITE=0

source /home/guhao/miniconda3/etc/profile.d/conda.sh
conda activate /aifs4su/guhao/envs/kv_evict
module add cuda/12.8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ID: $SLURM_JOB_ID"

cd /aifs4su/guhao/KVCache/kvpress


# bash train_long.sh
# bash run_ea_means.sh
cd evaluation/
bash eval_finals_longbench.sh
# bash eval_finals_zeroscrolls.sh
# bash eval_finals_infinitebench.sh
# bash eval_finals.sh
# bash rerun_missing_niah.sh
# bash run_pilot_ruler100_gt_llama31_8b_instruct_h800.sh
# bash eval_others.sh
# bash eval_dim.sh
