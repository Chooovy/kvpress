# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.distributed as dist


def setup_distributed():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    return rank, world_size, torch.device("cuda", local_rank)


def sequence_parallel_group(rank, world_size, size):
    if size == 1:
        return None, rank, world_size
    groups = [dist.new_group(list(range(start, start + size))) for start in range(0, world_size, size)]
    return groups[rank // size], rank // size, world_size // size


def average_gradients(parameters, world_size):
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    flattened = torch._utils._flatten_dense_tensors(gradients)
    dist.all_reduce(flattened)
    flattened /= world_size
    for gradient, reduced in zip(gradients, torch._utils._unflatten_dense_tensors(flattened, gradients)):
        gradient.copy_(reduced)


def mean_loss(loss, device, world_size):
    if world_size == 1:
        return loss
    value = torch.tensor(loss, device=device)
    dist.all_reduce(value)
    return value.item() / world_size


def finish_distributed(world_size):
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
