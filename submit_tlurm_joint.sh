#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Submit the JOINT training run (router + backbone attention) to tlurm.
#
#   bash submit_tlurm_joint.sh                 # 1 node,  smoke first!
#   MODE=smoke bash submit_tlurm_joint.sh      # 2 steps, verifies the whole path
#   HOST_NUM=2 bash submit_tlurm_joint.sh      # 2 nodes
#   HOST_NUM=4 bash submit_tlurm_joint.sh      # 4 nodes
#
# RUN MODE=smoke FIRST on 1 node. It verifies four things that a 4-node run would only reveal
# after a long queue wait: the router loads (loss starts near 1.80, not 2.4+), FSDP HYBRID_SHARD
# builds, FFN-SP composes with it, and peak memory is in range.
#
# Scaling behaviour, stated because it is counter-intuitive
# --------------------------------------------------------
# HYBRID_SHARD shards parameters/grads/optimizer state WITHIN a node and replicates ACROSS nodes.
# So per-GPU memory does NOT fall as HOST_NUM grows -- extra nodes buy throughput only. The
# all-gather stays on NVLink and only the reduce-scatter crosses IB, which is the trade this
# strategy exists for.
#
# GLOBAL_BATCH is held at 8 sequences per optimizer step regardless of HOST_NUM, so a 4-node run
# is comparable to a 1-node run at the same step number. FFN-SP makes each node a single
# data-parallel replica, so accumulation absorbs the difference (1 node -> accum 8, 4 -> accum 2).
# Raise GLOBAL_BATCH deliberately if you want the larger batch rather than the shorter wall-clock.
set -euo pipefail

timestamp=$(date +%y%m%d-%H_%M_%S)
cluster="${CLUSTER:-ARCH_GZ_H20_gy}"

HOST_NUM="${HOST_NUM:-1}"
HOST_GPU_NUM="${HOST_GPU_NUM:-8}"
MODE="${MODE:-joint_8k}"

exp_name="joint_${MODE}_${HOST_NUM}x${HOST_GPU_NUM}"

# Env for the job, forwarded through start_cmd. HOST_NUM/GPUS_PER_NODE are what joint_job.sh
# turns into torchrun's --nnodes/--nproc_per_node; CHIEF_IP is the fallback master address for
# the case where setup_env.sh's NODE_IP_0 is unavailable.
JOB_ENV="MODE=${MODE} HOST_NUM=${HOST_NUM} GPUS_PER_NODE=${HOST_GPU_NUM}"
JOB_ENV+=" CHIEF_IP=\"\$NODE_IP_0\""
JOB_ENV+=" GLOBAL_BATCH=${GLOBAL_BATCH:-8}"
# Optional passthroughs: only exported when set, so the job script's own defaults stand otherwise.
[[ -n "${BACKBONE_LR:-}" ]]    && JOB_ENV+=" BACKBONE_LR=${BACKBONE_LR}"
[[ -n "${ROUTER_LR_MULT:-}" ]] && JOB_ENV+=" ROUTER_LR_MULT=${ROUTER_LR_MULT}"
[[ -n "${SCHEDULE:-}" ]]       && JOB_ENV+=" SCHEDULE=${SCHEDULE}"
[[ -n "${MAX_STEPS:-}" ]]      && JOB_ENV+=" MAX_STEPS=${MAX_STEPS}"
[[ -n "${SEED:-}" ]]           && JOB_ENV+=" SEED=${SEED}"
[[ -n "${INIT:-}" ]]           && JOB_ENV+=" INIT=${INIT}"
[[ -n "${FFN_SP:-}" ]]         && JOB_ENV+=" FFN_SP=${FFN_SP}"

START_CMD="source taiji/init_shim.sh"
START_CMD+=" && source taiji/setup_env.sh"
START_CMD+=" && export ${JOB_ENV}"
START_CMD+=" && bash taiji/joint_job.sh"

echo "cluster    : ${cluster}"
echo "nodes      : ${HOST_NUM} x ${HOST_GPU_NUM} GPU"
echo "mode       : ${MODE}"
echo "experiment : ${exp_name}_${timestamp}"
echo
echo "start_cmd  : ${START_CMD}"
echo

trun _tlurm/_${cluster}.yaml --auto-commit tlurm/guhao \
    name="${exp_name}_${timestamp}" \
    start_cmd="${START_CMD}" \
    host_gpu_num=${HOST_GPU_NUM} \
    host_num=${HOST_NUM}
