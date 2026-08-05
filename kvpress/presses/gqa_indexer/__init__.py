# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GQA lightning indexer: a small learned scorer that predicts, per KV head, which
cached tokens matter.

The design mirrors DeepSeek's lightning indexer (DSA) but is adapted to grouped-query
attention. Two structural differences from the MLA original:

1. MLA keeps a single shared latent KV, so DSA must collapse its indexer heads into
   one selection via a learned ``weights_proj`` pooling. GQA has ``num_key_value_heads``
   physically independent caches, so we keep one score per KV head and let each KV head
   evict independently.
2. MLA feeds the indexer the already-computed low-rank query (``q_lora``). GQA has no
   such tensor, so queries are projected straight from ``hidden_states`` -- the same
   choice MiniMax M3's GQA indexer makes.

Selection stays at token granularity inside the indexer; chunk/block aggregation is a
separate, optional post-processing step on the token scores (see
:mod:`kvpress.presses.gqa_indexer.aggregate`).
"""

from kvpress.presses.gqa_indexer.aggregate import (
    aggregate_chunk_scores,
    expand_chunk_indices,
    reduce_queries,
)
from kvpress.presses.gqa_indexer.indexer import (
    GQAIndexer,
    GQAIndexerConfig,
    build_indexer_mask,
    slice_rope_tables,
)
from kvpress.presses.gqa_indexer.loss import (
    build_dense_indexer_target,
    build_sparse_indexer_target,
    indexer_kl_per_row,
    indexer_loss_from_target,
    masked_log_softmax,
    masked_softmax,
    normalize_indexer_target,
)
from kvpress.presses.gqa_indexer.press import GQAIndexerPress
from kvpress.presses.gqa_indexer.train import (
    IndexerTrainConfig,
    compute_indexer_loss,
    freeze_all_but_indexer,
    get_attention_modules,
    indexer_layer_loss,
    indexer_state_dict,
    load_indexer_state_dict,
)

__all__ = [
    "GQAIndexer",
    "GQAIndexerConfig",
    "GQAIndexerPress",
    "build_indexer_mask",
    "slice_rope_tables",
    "aggregate_chunk_scores",
    "expand_chunk_indices",
    "reduce_queries",
    "build_dense_indexer_target",
    "build_sparse_indexer_target",
    "indexer_kl_per_row",
    "indexer_loss_from_target",
    "masked_log_softmax",
    "masked_softmax",
    "normalize_indexer_target",
    "IndexerTrainConfig",
    "compute_indexer_loss",
    "freeze_all_but_indexer",
    "get_attention_modules",
    "indexer_layer_loss",
    "indexer_state_dict",
    "load_indexer_state_dict",
]
