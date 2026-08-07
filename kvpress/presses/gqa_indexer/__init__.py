# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GQA lightning indexer: a small learned scorer that predicts, per KV head, which
cached tokens matter.

The design follows DeepSeek's lightning indexer (DSA) but is adapted to grouped-query
attention, which drops three of DSA's components rather than porting them:

1. MLA keeps a single shared latent KV cache, so DSA must collapse many indexer heads into
   ONE score -- hence its ReLU (to keep per-head contributions non-negative before they are
   summed) and its learned ``weights_proj`` pooling. GQA has ``num_key_value_heads``
   physically independent caches, so we emit one score per KV head and let each evict
   independently. With no cross-head sum, the activation and the pooling weights have
   nothing left to do: top-k is invariant to strictly increasing maps, and a per-head
   scalar is constant along the key axis so it cannot reorder a row. MiniMax M3, the one
   production GQA indexer, likewise has neither.
2. The key side is MQA (a single shared key head), so the indexer's own cache costs
   ``head_dim`` per token rather than ``n_heads * head_dim``. Heads differ on the query
   side only.
3. MLA feeds the indexer the already-computed low-rank query (``q_lora``). GQA has no such
   tensor, so queries are projected straight from ``hidden_states`` -- again matching M3.

Selection stays at token granularity inside the indexer; chunk/block aggregation is a
separate, optional post-processing step on the token scores (see
:mod:`kvpress.presses.gqa_indexer.aggregate`).
"""

from kvpress.presses.gqa_indexer.aggregate import (
    aggregate_chunk_scores,
    expand_chunk_indices,
    reduce_queries,
)
from kvpress.presses.gqa_indexer.data import (
    SUBSETS,
    LengthSchedule,
    LongminoConfig,
    LongminoDataset,
    build_dataloader,
    describe_subsets,
    shard_paths,
)
from kvpress.presses.gqa_indexer.fused_loss import (
    accumulation_dtype,
    group_view,
    fused_indexer_ce_rows,
    fused_indexer_loss,
    make_recompute_teacher,
    teacher_lse_from_qk,
    teacher_probs_from_lse,
)
from kvpress.presses.gqa_indexer.fused_sparse_loss import (
    expand_to_heads,
    fused_sparse_indexer_kl_rows,
    fused_sparse_indexer_loss,
    make_sparse_recompute_teacher,
    sparse_teacher_probs,
    support_teacher_lse,
)
from kvpress.presses.gqa_indexer.fused_trainer import (
    FusedIndexerTrainer,
    attention_scaling,
    fused_indexer_training_step,
    teacher_query_states,
)
from kvpress.presses.gqa_indexer.indexer import (
    GQAIndexer,
    GQAIndexerConfig,
    IndexerNorm,
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
from kvpress.presses.gqa_indexer.sparse_support import (
    causal_keep,
    excluded_key_mask,
    forced_support_positions,
    gather_support_keys,
    resolve_topk,
    sort_support,
    streaming_topk_support,
)
from kvpress.presses.gqa_indexer.teacher_lse import (
    assert_flash_dtype_supported,
    assert_lse_mask_compatible,
    capture_teacher_lse,
    normalize_captured_lse,
)
from kvpress.presses.gqa_indexer.train import (
    IndexerTrainConfig,
    build_position_embeddings,
    compute_indexer_loss,
    freeze_all_but_indexer,
    get_attention_modules,
    get_input_layernorms,
    indexer_layer_loss,
    indexer_state_dict,
    load_indexer_state_dict,
)
from kvpress.presses.gqa_indexer.triton_fused_loss import (
    HAS_TRITON,
    decompose_mask,
    kernels_available,
    triton_indexer_ce_rows,
    triton_indexer_loss,
    triton_interpret_enabled,
)

__all__ = [
    # Data loading
    "SUBSETS",
    "LengthSchedule",
    "LongminoConfig",
    "LongminoDataset",
    "build_dataloader",
    "describe_subsets",
    "shard_paths",
    "GQAIndexer",
    "GQAIndexerConfig",
    "IndexerNorm",
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
    "build_position_embeddings",
    "compute_indexer_loss",
    "freeze_all_but_indexer",
    "get_attention_modules",
    "get_input_layernorms",
    "indexer_layer_loss",
    "indexer_state_dict",
    "load_indexer_state_dict",
    "fused_indexer_ce_rows",
    "fused_indexer_loss",
    "accumulation_dtype",
    "group_view",
    "FusedIndexerTrainer",
    "attention_scaling",
    "fused_indexer_training_step",
    "teacher_query_states",
    "make_recompute_teacher",
    "teacher_lse_from_qk",
    "teacher_probs_from_lse",
    "assert_flash_dtype_supported",
    "assert_lse_mask_compatible",
    "capture_teacher_lse",
    "normalize_captured_lse",
    # Stage 2: sparse support selection + KL on the support
    "resolve_topk",
    "forced_support_positions",
    "excluded_key_mask",
    "causal_keep",
    "sort_support",
    "streaming_topk_support",
    "gather_support_keys",
    "expand_to_heads",
    "support_teacher_lse",
    "sparse_teacher_probs",
    "make_sparse_recompute_teacher",
    "fused_sparse_indexer_kl_rows",
    "fused_sparse_indexer_loss",
    # Triton kernels for stage 1
    "HAS_TRITON",
    "kernels_available",
    "triton_interpret_enabled",
    "decompose_mask",
    "triton_indexer_ce_rows",
    "triton_indexer_loss",
]
