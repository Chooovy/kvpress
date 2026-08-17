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

Two training objectives ship, and they are alternatives rather than stages of one recipe:

* **Distillation** (:mod:`~kvpress.presses.gqa_indexer.fused_trainer`) matches the frozen
  model's own attention weights. The score never enters the forward pass.
* **End-to-end** (:mod:`~kvpress.presses.gqa_indexer.e2e_trainer`) adds the score inside the
  attention softmax so the LM loss trains it directly, following SAS. Needs
  ``GQAIndexerPress(gate_scale=True)``.

Both expose a full-scope stage and a top-k-scope stage under the same ``stage`` names, so the
two objectives can be compared at matched budget.
"""

from kvpress.presses.gqa_indexer.aggregate import (
    aggregate_chunk_scores,
    expand_chunk_indices,
    reduce_queries,
)
from kvpress.presses.gqa_indexer.autotune import (
    Candidate,
    Measurement,
    Profile,
    autotune,
    autotune_cached,
    batch_for_length,
    candidate_grid,
    default_token_budget,
    is_resource_limit,
    is_shared_memory_limit,
    pick_best,
    profile_key,
)
from kvpress.presses.gqa_indexer.data import (
    SUBSETS,
    LengthSchedule,
    LongminoConfig,
    LongminoDataset,
    TokenizedConfig,
    TokenizedDataset,
    build_dataloader,
    build_tokenized_dataloader,
    describe_subsets,
    read_index,
    shard_paths,
    wsd_lr_lambda,
)
from kvpress.presses.gqa_indexer.e2e_trainer import (
    STAGES,
    E2EIndexerTrainer,
    e2e_indexer_training_step,
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
from kvpress.presses.gqa_indexer.gate_pin import (
    PIN_MODES,
    check_pin_mode,
    gate_from_score,
    history_lse,
    is_query_dependent,
    pinned_mask,
    pins_self,
    pins_sink,
)
from kvpress.presses.gqa_indexer.gated_attention import (
    SCOPES,
    build_concat_qk,
    pad_value_to_width,
    causal_mask_bottom_right,
    check_gate_shapes,
    gated_attention,
    gated_attention_full,
    gated_attention_pinned_self,
    gated_attention_reference,
    gated_attention_sparse,
)
from kvpress.presses.gqa_indexer.triton_gated_attention import (
    gated_kernels_available,
    triton_gated_attention,
)
from kvpress.presses.gqa_indexer.indexer import (
    GQAIndexer,
    GQAIndexerConfig,
    IndexerNorm,
    build_indexer_mask,
    slice_rope_tables,
)
from kvpress.presses.gqa_indexer.scalar_indexer import (
    DEFAULT_POS_SLOPE,
    ScalarIndexer,
    ScalarIndexerConfig,
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
from kvpress.presses.gqa_indexer.sparse_inference import (
    IMPL_NAME as SPARSE_ATTENTION_IMPL_NAME,
    SparseAttentionContext,
)
from kvpress.presses.gqa_indexer.sparse_attention import (
    check_sparse_shapes,
    pack_varlen,
    resolve_scaling,
    slot_validity,
    sparse_gqa_attention_dense_reference,
    sparse_gqa_attention_reference,
    sparse_gqa_attention_varlen_reference,
    unpack_varlen,
)
from kvpress.presses.gqa_indexer.sparse_support import (
    causal_keep,
    excluded_key_mask,
    forced_support_positions,
    gather_support_keys,
    resolve_topk,
    sort_support,
    streaming_topk_support,
    topk_tiles,
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
from kvpress.presses.gqa_indexer.train import (
    detect_scorer,
    detect_scorer_from_keys,
    infer_scalar_mid_dim,
)
from kvpress.presses.gqa_indexer.triton_fused_loss import (
    HAS_TRITON,
    decompose_mask,
    kernels_available,
    triton_indexer_ce_rows,
    triton_indexer_loss,
    triton_interpret_enabled,
)
from kvpress.presses.gqa_indexer.triton_sparse_attention import (
    seq_ids_from_cu_seqlens,
    sparse_gqa_attention,
    sparse_kernels_available,
    triton_sparse_gqa_attention,
    triton_sparse_gqa_attention_varlen,
)

__all__ = [
    # Data loading
    "SUBSETS",
    "LengthSchedule",
    "LongminoConfig",
    "LongminoDataset",
    "TokenizedConfig",
    "TokenizedDataset",
    "build_dataloader",
    "build_tokenized_dataloader",
    "describe_subsets",
    "read_index",
    "shard_paths",
    "wsd_lr_lambda",
    "GQAIndexer",
    "GQAIndexerConfig",
    "IndexerNorm",
    "ScalarIndexer",
    "ScalarIndexerConfig",
    "DEFAULT_POS_SLOPE",
    "GQAIndexerPress",
    "SparseAttentionContext",
    "SPARSE_ATTENTION_IMPL_NAME",
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
    "detect_scorer",
    "detect_scorer_from_keys",
    "infer_scalar_mid_dim",
    "fused_indexer_ce_rows",
    "fused_indexer_loss",
    "accumulation_dtype",
    "group_view",
    "FusedIndexerTrainer",
    "attention_scaling",
    "fused_indexer_training_step",
    "teacher_query_states",
    # End-to-end (gated-attention) training
    "E2EIndexerTrainer",
    "e2e_indexer_training_step",
    "STAGES",
    "SCOPES",
    "gated_attention",
    "gated_attention_full",
    "gated_attention_sparse",
    "gated_attention_pinned_self",
    "gated_attention_reference",
    "build_concat_qk",
    "causal_mask_bottom_right",
    "check_gate_shapes",
    "pad_value_to_width",
    "triton_gated_attention",
    "gated_kernels_available",
    # Gate pinning (what stops the gate flattening into a no-op)
    "PIN_MODES",
    "check_pin_mode",
    "gate_from_score",
    "history_lse",
    "is_query_dependent",
    "pinned_mask",
    "pins_self",
    "pins_sink",
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
    "topk_tiles",
    "gather_support_keys",
    "expand_to_heads",
    "support_teacher_lse",
    "sparse_teacher_probs",
    "make_sparse_recompute_teacher",
    "fused_sparse_indexer_kl_rows",
    "fused_sparse_indexer_loss",
    # Sparse attention at inference: per-query top-k, no eviction
    "check_sparse_shapes",
    "resolve_scaling",
    "slot_validity",
    "sparse_gqa_attention",
    "sparse_gqa_attention_reference",
    "sparse_gqa_attention_dense_reference",
    "sparse_gqa_attention_varlen_reference",
    "sparse_kernels_available",
    "triton_sparse_gqa_attention",
    "triton_sparse_gqa_attention_varlen",
    "seq_ids_from_cu_seqlens",
    "pack_varlen",
    "unpack_varlen",
    # Triton kernels for stage 1
    "HAS_TRITON",
    "kernels_available",
    "triton_interpret_enabled",
    "decompose_mask",
    "triton_indexer_ce_rows",
    "triton_indexer_loss",
    # Per-length batch/tile autotuning
    "Candidate",
    "Measurement",
    "Profile",
    "autotune",
    "autotune_cached",
    "batch_for_length",
    "candidate_grid",
    "default_token_budget",
    "is_resource_limit",
    "is_shared_memory_limit",
    "pick_best",
    "profile_key",
]
