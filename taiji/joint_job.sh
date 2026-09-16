#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Taiji job entrypoint for JOINT training (router + backbone attention).
#
# Invoked once per node by submit_tlurm_joint.sh, after init_shim.sh and setup_env.sh have run.
# Its whole job is to turn taiji's environment into torchrun's arguments and launch.
#
# Rank discovery, and why it is done this way
# ------------------------------------------
# taiji exports INDEX (this node's ordinal) and NODE_IP_LIST; setup_env.sh turns the latter into
# NODE_IP_0..N. So MASTER_ADDR is NODE_IP_0 and NODE_RANK is INDEX -- the same mapping
# AngelPTM's train.sh uses on its non-BACKGROUND_MODE path. The hostfile route
# (/etc/taiji/hostfile) is a fallback for when INDEX is absent, matching that script's
# BACKGROUND_MODE branch, because a silently-wrong NODE_RANK is the worst failure here: two nodes
# claiming rank 0 makes torchrun hang at rendezvous with no error.
#
# HYBRID_SHARD needs no per-node configuration. The training script derives the device mesh from
# WORLD_SIZE and torch.cuda.device_count(), sharding within each node and replicating across
# them, so 1x8 / 2x8 / 4x8 all take this same path with only HOST_NUM changing.
set -euo pipefail

MODE="${MODE:-joint_8k}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
HOST_NUM="${HOST_NUM:-1}"
MASTER_PORT="${MASTER_PORT:-29514}"

# --- rank discovery -----------------------------------------------------------------
HOSTFILE="${HOSTFILE:-/etc/taiji/hostfile}"
if [[ -n "${INDEX:-}" && -n "${NODE_IP_0:-}" ]]; then
    NODE_RANK="${INDEX}"
    MASTER_ADDR="${NODE_IP_0}"
    echo "[joint] rank from INDEX: node_rank=${NODE_RANK} master=${MASTER_ADDR}"
elif [[ -f "${HOSTFILE}" ]]; then
    LOCAL_IP="${LOCAL_IP:-$(hostname -i | awk '{print $1}')}"
    IP_INDEX="$(awk -v ip="${LOCAL_IP}" '$1 == ip {print NR; exit}' "${HOSTFILE}")"
    if [[ -z "${IP_INDEX}" ]]; then
        echo "[joint] ERROR: local IP ${LOCAL_IP} not found in ${HOSTFILE}" >&2
        exit 1
    fi
    NODE_RANK=$((IP_INDEX - 1))
    MASTER_ADDR="$(awk 'NR == 1 {print $1}' "${HOSTFILE}")"
    echo "[joint] rank from hostfile: node_rank=${NODE_RANK} master=${MASTER_ADDR}"
else
    # Refuse rather than default to 0. On a single node that guess is harmless, but on four it
    # makes every node rank 0 and torchrun hangs at rendezvous with nothing in the log to say why.
    if [[ "${HOST_NUM}" -gt 1 ]]; then
        echo "[joint] ERROR: HOST_NUM=${HOST_NUM} but neither INDEX/NODE_IP_0 nor ${HOSTFILE}" >&2
        echo "        is available, so this node cannot know its rank. Refusing to guess 0." >&2
        exit 1
    fi
    NODE_RANK=0
    MASTER_ADDR="127.0.0.1"
fi

# --- NCCL / IB, copied from AngelPTM's train.sh ---------------------------------------
# These are the cluster's working values for H20 + IB. Kept identical rather than tuned, because
# multi-node FSDP is exactly the workload they were set for: HYBRID_SHARD's reduce-scatter is the
# only collective crossing the network, and it is on the critical path of every step.
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_IB_SL="${NCCL_IB_SL:-3}"
export NCCL_CHECK_DISABLE="${NCCL_CHECK_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_LL_THRESHOLD="${NCCL_LL_THRESHOLD:-16384}"
export NCCL_IB_CUDA_SUPPORT="${NCCL_IB_CUDA_SUPPORT:-1}"
export NCCL_COLLNET_ENABLE="${NCCL_COLLNET_ENABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-4}"
export NCCL_PXN_DISABLE="${NCCL_PXN_DISABLE:-0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_ALGO="${NCCL_ALGO:-^NVLS}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

# --- outputs on ceph ------------------------------------------------------------------
# OUTPUT_PATH is exported by init_shim.sh and already symlinked for long-term keeping.
export OUT="${OUT:-${OUTPUT_PATH:-$PWD/checkpoints}/gqa_indexer_joint}"
mkdir -p "$OUT"

echo "[joint] HOST_NUM=${HOST_NUM} GPUS_PER_NODE=${GPUS_PER_NODE} -> world $((HOST_NUM * GPUS_PER_NODE))"
echo "[joint] MODE=${MODE} OUT=${OUT}"

# NNODES/NODE_RANK/MASTER_ADDR are what train_gqa_indexer_joint_gy.sh forwards to torchrun.
export NNODES="${HOST_NUM}"
export NODE_RANK MASTER_ADDR MASTER_PORT
export NGPU="${GPUS_PER_NODE}"

# GLOBAL_BATCH is a per-STEP sequence count across replicas and should not change with node
# count, or a 4x8 run is not comparable to a 1x8 one at the same step number. FFN_SP makes each
# node one replica, so accum absorbs the difference: 1x8 -> accum 8, 4x8 -> accum 2.
export GLOBAL_BATCH="${GLOBAL_BATCH:-8}"

RUNNER=(bash)
if command -v pixi >/dev/null 2>&1 && [[ -f pyproject.toml ]]; then
    # The env under $ENV_PATH is prebuilt on ceph by init_shim.sh.
    RUNNER=(pixi run --)
fi

exec "${RUNNER[@]}" bash scripts/train_gqa_indexer_joint_gy.sh "${MODE}"
