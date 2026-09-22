# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from kvpress.indexmem.kernels.gate_normalization import history_logsumexp


def gated_attention(
    query, key, value, gate_queries, gate_keys, gate_scale, *, scaling, gate_mass=256.0, sink_size=4, window_size=128
):
    from kvpress.indexmem.kernels.triton_gated_attention import triton_gated_attention

    query_offset = key.shape[2] - query.shape[2]
    if sink_size or window_size:
        normalizer = history_logsumexp(
            gate_queries,
            gate_keys,
            gate_scale,
            sink_size=sink_size,
            window_size=window_size,
            query_offset=query_offset,
        ) - math.log(gate_mass)
    else:
        normalizer = torch.zeros(gate_queries.shape[:3], device=query.device, dtype=torch.float32)
    return triton_gated_attention(
        query,
        key,
        value,
        gate_queries,
        gate_keys,
        normalizer,
        gate_scale=gate_scale,
        scaling=scaling,
        query_offset=query_offset,
        sink_size=sink_size,
        window_size=window_size,
    )
