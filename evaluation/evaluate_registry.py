# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from importlib import import_module

# Force transformers to avoid optional TF/Flax backends in this environment.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")

from kvpress import (
    AdaKVPress,
    BlockPress,
    ChunkKVPress,
    CompactorPress,
    ComposedPress,
    CriticalAdaKVPress,
    CriticalKVPress,
    DecodingPress,
    DuoAttentionPress,
    ExpectedAttentionPress,
    FinchPress,
    KeyDiffPress,
    KnormPress,
    KVzipPress,
    ObservedAttentionPress,
    PyramidKVPress,
    QFilterPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    ThinKPress,
    TOVAPress,
    # ---- Score Presses ----
    RkvPress,
    H2OPress,
    DMAScorePress,
    IndexerScorePress,
    CacheIndexerScorePress,
    QueryIndexerScorePress,
    QueryIndexer_KVzipScorePress,
    # ---- Decode Presses ----
    SelectiveDecodingPress,
    CacheIndexerDecodingPress,
    DecodePress,
    DecodeOnlyPress,
    QueryIndexerDecodingPress,
    # ---- Memory Presses ----
    MemoryScorerPress,
    # ---- Gated Presses ----
    GatedPress,
    # ---- Debug Presses ----
    FixedLayerScoreEvictPress,
    # ---- Ground Truth ----
    GTScorePress,
)

# These dictionaries define the available datasets, scorers, and KVPress methods for evaluation.
DATASET_REGISTRY = {
    "loogle": "simonjegou/loogle",
    "ruler": "simonjegou/ruler",
    # "zero_scrolls": "simonjegou/zero_scrolls",
    "zero_scrolls": "tau/zero_scrolls",
    "infinitebench": "MaxJeblick/InfiniteBench",
    "longbench": "Xnhyacinth/LongBench",
    "longbench-e": "Xnhyacinth/LongBench",
    "longbench-v2": "simonjegou/LongBench-v2",
    "needle_in_haystack": "alessiodevoto/paul_graham_essays",
    # Datasets used to be used for decoding compression
    "aime25": "alessiodevoto/aime25",
    "aime24": "HuggingFaceH4/aime_2024",
    "math500": "alessiodevoto/math500",
    "local_ruler": None,
}

SCORER_REGISTRY = {
    "loogle": lambda df: import_module("benchmarks.loogle.calculate_metrics").calculate_metrics(df),
    "ruler": lambda df: import_module("benchmarks.ruler.calculate_metrics").calculate_metrics(df),
    "local_ruler": lambda df: import_module("benchmarks.ruler.calculate_metrics").calculate_metrics(df),
    "zero_scrolls": lambda df: import_module("benchmarks.zero_scrolls.calculate_metrics").calculate_metrics(df),
    "infinitebench": lambda df: import_module("benchmarks.infinite_bench.calculate_metrics").calculate_metrics(df),
    "longbench": lambda df: import_module("benchmarks.longbench.calculate_metrics").calculate_metrics(df),
    "longbench-e": lambda df: import_module("benchmarks.longbench.calculate_metrics").calculate_metrics_e(df),
    "longbench-v2": lambda df: import_module("benchmarks.longbenchv2.calculate_metrics").calculate_metrics(df),
    "needle_in_haystack": lambda df: import_module("benchmarks.needle_in_haystack.calculate_metrics").calculate_metrics(df),
    "aime25": lambda df: import_module("benchmarks.aime25.calculate_metrics").calculate_metrics(df),
    "aime24": lambda df: import_module("benchmarks.aime24.calculate_metrics").calculate_metrics(df),
    "math500": lambda df: import_module("benchmarks.math500.calculate_metrics").calculate_metrics(df),
}


PRESS_REGISTRY = {
    "gt_score_max_mode": GTScorePress(query_reduce="max", head_reduce="kv_group_mean", smooth_kernel=0, use_vnorm=False),
    "gt_score_causal_mean_mode": GTScorePress(query_reduce="causal_mean", head_reduce="kv_group_mean", smooth_kernel=0, use_vnorm=False),

    "adakv_expected_attention": AdaKVPress(ExpectedAttentionPress()),
    "adakv_expected_attention_e2": AdaKVPress(ExpectedAttentionPress(epsilon=1e-2)),
    "adakv_snapkv": AdaKVPress(SnapKVPress()),
    "block_keydiff": BlockPress(press=KeyDiffPress(), block_size=128),
    "chunkkv": ChunkKVPress(press=SnapKVPress(), chunk_length=20),
    "critical_adakv_expected_attention": CriticalAdaKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "critical_adakv_snapkv": CriticalAdaKVPress(SnapKVPress()),
    "critical_expected_attention": CriticalKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "critical_snapkv": CriticalKVPress(SnapKVPress()),
    "duo_attention": DuoAttentionPress(),
    "duo_attention_on_the_fly": DuoAttentionPress(on_the_fly_scoring=True),
    "expected_attention": ExpectedAttentionPress(),
    "finch": FinchPress(),
    "keydiff": KeyDiffPress(),
    "kvzip": KVzipPress(),
    "knorm": KnormPress(),
    "observed_attention": ObservedAttentionPress(),
    "pyramidkv": PyramidKVPress(),
    "qfilter": QFilterPress(),
    "random": RandomPress(),
    "snap_think": ComposedPress([SnapKVPress(), ThinKPress()]),
    "snapkv": SnapKVPress(),
    "snapkv_128": SnapKVPress(window_size=128),
    "streaming_llm": StreamingLLMPress(),
    "think": ThinKPress(),
    "tova": TOVAPress(),
    "compactor": CompactorPress(),
    "adakv_compactor": AdaKVPress(CompactorPress()),
    "no_press": None,
    "decoding_knorm": DecodingPress(base_press=KnormPress()),
    "decoding_streaming_llm": DecodingPress(base_press=StreamingLLMPress()),
    "decoding_tova": DecodingPress(base_press=TOVAPress()),
    "decoding_qfilter": DecodingPress(base_press=QFilterPress()),
    "decoding_adakv_expected_attention_e2": DecodingPress(base_press=AdaKVPress(ExpectedAttentionPress(epsilon=1e-2))),
    "decoding_adakv_snapkv": DecodingPress(base_press=AdaKVPress(SnapKVPress())),
    "decoding_keydiff": DecodingPress(base_press=KeyDiffPress()),
    "decoding_snapkv": DecodingPress(base_press=SnapKVPress()),

    "dma_score": DMAScorePress(),
    # Indexer ------------------
    "indexer_score": IndexerScorePress(),
    "indexer_score_last128": IndexerScorePress(last_n_query=128),
    "indexer_score_top_p": IndexerScorePress(query_reduce="top_p"),
    # Query Indexer ------------------
    "query_indexer_score": QueryIndexerScorePress(),
    "query_indexer_mean_mode": QueryIndexerScorePress(query_reduce="auto"),
    "query_indexer_max_mode": QueryIndexerScorePress(query_reduce="max"),

    "query_indexer_ea_mode": QueryIndexerScorePress(query_reduce="ea"),
    
    "query_indexer_max_mode_last1": QueryIndexerScorePress(query_reduce="max", last_n_query=1),


    "query_indexer_max_mode_pooling3": QueryIndexerScorePress(query_reduce="max", use_pooling=True, pooling_kernel_size=3),
    "query_indexer_max_mode_pooling5": QueryIndexerScorePress(query_reduce="max", use_pooling=True, pooling_kernel_size=5),

    
    "query_indexer_max_mode_128": QueryIndexerScorePress(query_reduce="max", last_n_query=128),
    "query_indexer_max_mode_layer_mean": QueryIndexerScorePress(layer_running_mean=True, query_reduce="max"),

    
    "query_indexer_max_mode_layer_mean_ent_skip_high": QueryIndexerScorePress(layer_running_mean=True, entropy_gate=True, entropy_gate_level="head_layer", 
    entropy_skip="high", entropy_threshold_mode="mean", query_reduce="max"),
    "query_indexer_max_mode_layer_mean_ent_skip_high_negonly": QueryIndexerScorePress(layer_running_mean=True, entropy_gate=True, entropy_skip="high",
    entropy_shift_mode="neg_only", query_reduce="max"),

    "query_indexer_max_mode_layer_mean_ent_skip_low": QueryIndexerScorePress(layer_running_mean=True, entropy_gate=True, entropy_gate_level="head_layer", 
    entropy_skip="low", entropy_threshold_mode="mean", query_reduce="max"),
    "query_indexer_max_mode_layer_mean_ent_skip_low_negonly": QueryIndexerScorePress(layer_running_mean=True, entropy_gate=True, entropy_skip="low",
    entropy_shift_mode="neg_only", query_reduce="max"),
    "query_indexer_max_mode_layer_mean_ent_skip_low_softmax": QueryIndexerScorePress(layer_running_mean=True, entropy_gate=True, entropy_skip="low",
    entropy_prob_mode="softmax", entropy_softmax_tau=1.0, query_reduce="max"),

    "query_indexer_max_mode_layer_mean_peak_gate": QueryIndexerScorePress(layer_running_mean=True, meanmax_gate=True, layer_running_alpha=0.2, 
    layer_spiky_peak_thresh=0.0, query_reduce="max"),
    "query_indexer_max_mode_layer_mean_peak_gate_keephigh": QueryIndexerScorePress(layer_running_mean=True, meanmax_gate=True, meanmax_gate_keep_high=True, 
    layer_running_alpha=0.2, layer_spiky_peak_thresh=0.0, query_reduce="max"),


    "query_indexer_score_block": QueryIndexerScorePress(query_reduce="block"),
    "query_indexer_score_question": QueryIndexerScorePress(query_reduce="question"),
    "query_indexer_score_last128": QueryIndexerScorePress(last_n_query=128),
    "query_indexer_score_top_p": QueryIndexerScorePress(query_reduce="top_p"),
    "query_indexer_score_mean_last128": QueryIndexerScorePress(last_n_query=128, layer_running_mean=True, mean_head=True),
    "query_indexer_score_mean": QueryIndexerScorePress(layer_running_mean=True, mean_head=True),
    
    "query_indexer_kvzip_max": QueryIndexer_KVzipScorePress(score_reduce="amax_logit_softmax"),
    # Memory Press ------------------
    "memory_query_indexer_top_p": MemoryScorerPress(base_press=QueryIndexerScorePress(query_reduce="top_p")),
    "memory_query_indexer_last128": MemoryScorerPress(base_press=QueryIndexerScorePress(last_n_query=128)),
    "memory_query_indexer_max": MemoryScorerPress(base_press=QueryIndexerScorePress(query_reduce="max")),
    "memory_query_indexer_max_last1": MemoryScorerPress(base_press=QueryIndexerScorePress(query_reduce="max", last_n_query=1)),

    "memory_EA": MemoryScorerPress(base_press=ExpectedAttentionPress()),
    "memory_snapkv": MemoryScorerPress(base_press=SnapKVPress()),
    "memory_keydiff": MemoryScorerPress(base_press=KeyDiffPress()),
    "memory_kvzip": MemoryScorerPress(base_press=KVzipPress()),
    # running mean ------------------
    "expected_attention_layer_mean": ExpectedAttentionPress(layer_running_mean=True),
    "expected_attention_head_mean": ExpectedAttentionPress(mean_head=True),
    "expected_attention_layer_mean_from_2": ExpectedAttentionPress(layer_running_mean=True, running_mean_start_layer=2),
    "expected_attention_layer_mean_from_4": ExpectedAttentionPress(layer_running_mean=True, running_mean_start_layer=4),
    "fixed_EA": FixedLayerScoreEvictPress(press=ExpectedAttentionPress(), score_layer_idx=11),
    "expected_attention_layer_mean_ent_skip_low": ExpectedAttentionPress(
        layer_running_mean=True,
        entropy_gate=True,
        entropy_gate_level="head_layer",
        entropy_skip="low",                # 保护尖峰：低熵不mean
        entropy_threshold_mode="mean",
    ),

    "expected_attention_layer_mean_ent_skip_high": ExpectedAttentionPress(
        layer_running_mean=True,
        entropy_gate=True,
        entropy_gate_level="head_layer",
        entropy_skip="high",               # 保护分散：高熵不mean（反向对照）
        entropy_threshold_mode="mean",
    ),
    "expected_attention_layer_mean_peak_gate": ExpectedAttentionPress(
        layer_running_mean=True,
        meanmax_gate=True,
        layer_running_alpha=0.2,          # 尖峰 head 用 0.2（更偏向 max）
        layer_spiky_peak_thresh=0.0,      # 0 表示用 batch 内 mean(peak) 当阈值
    ),
    "expected_attention_layer_mean_peak_gate_keephigh": ExpectedAttentionPress(
        layer_running_mean=True,
        meanmax_gate=True,
        meanmax_gate_keep_high=True,      # ★ 反过来：peak大->更信mean；peak小->更信max
        layer_running_alpha=0.2,
        layer_spiky_peak_thresh=0.0,
    ),
    # Branch 1: only-if-negative shift
    "expected_attention_layer_mean_ent_skip_high_negonly": ExpectedAttentionPress(
        layer_running_mean=True,
        entropy_gate=True,
        entropy_skip="high",
        entropy_shift_mode="neg_only",
    ),

    # Branch 2: softmax probability (keep everything else same)
    "expected_attention_layer_mean_ent_skip_high_softmax": ExpectedAttentionPress(
        layer_running_mean=True,
        entropy_gate=True,
        entropy_skip="high",
        entropy_prob_mode="softmax",
        entropy_softmax_tau=1.0,
    ),
    # Branch 1: only-if-negative shift
    "expected_attention_layer_mean_ent_skip_low_negonly": ExpectedAttentionPress(
        layer_running_mean=True,
        entropy_gate=True,
        entropy_skip="low",
        entropy_shift_mode="neg_only",
    ),

    # Branch 2: softmax probability (keep everything else same)
    "expected_attention_layer_mean_ent_skip_low_softmax": ExpectedAttentionPress(
        layer_running_mean=True,
        entropy_gate=True,
        entropy_skip="low",
        entropy_prob_mode="softmax",
        entropy_softmax_tau=1.0,
    ),

    # Decoding Press ------------------
    "decoding_rkv": DecodingPress(base_press=RkvPress()),
    "decoding_h2o": H2OPress(),
    "select_decoding_dma_score": SelectiveDecodingPress(base_press=DMAScorePress()),
    "select_decoding_indexer_score": SelectiveDecodingPress(base_press=CacheIndexerScorePress()),
    "select_decoding_indexer_score_nocache": SelectiveDecodingPress(base_press=IndexerScorePress()),
    "decode_only_indexer_score": DecodeOnlyPress(base_press=IndexerScorePress()),
    "decoding_indexer_score_cache": CacheIndexerDecodingPress(base_press=CacheIndexerScorePress()),

    "decoding_query_indexer": QueryIndexerDecodingPress(base_press=QueryIndexerScorePress(query_reduce="max")),

    # Dimension Press ------------------
    "gated_all_hidden": GatedPress(window_size=-1, gate_type="elementwise", bias=True),
    "gated_local128": GatedPress(window_size=128, gate_type="elementwise", bias=True),
    "thinKV_pair": ThinKPress(pairwise_prune=True, sync_kv_prune=True),
    "gated_local128_pair_KV": GatedPress(window_size=128, gate_type="elementwise", bias=True, pairwise_prune=True, sync_kv_prune=True),
    # GatedRetrievalPress
    "gated_retrieval_all": GatedPress(gate_mode="dynamic", gate_type="elementwise", window_size=-1, apply_kv_gate=True, prune_in_compress=False, 
    cache_infer_mask=True, qk_scale_correction=True, pairwise_prune=True, sync_kv_prune=True, protect_retrieval_heads=True, retrieval_head_topk=2, preserve_global_key_channel_ratio=True,
    duo_num_samples=50, duo_q_len=500, duo_dataset="kmfoda/booksum", duo_split="train", duo_text_field="chapter", duo_seed=42),

    "gated_retrieval_local128": GatedPress(gate_mode="dynamic", gate_type="elementwise", window_size=128, apply_kv_gate=True, prune_in_compress=False, 
    cache_infer_mask=True, qk_scale_correction=True, pairwise_prune=True, sync_kv_prune=True, protect_retrieval_heads=True, retrieval_head_topk=2, preserve_global_key_channel_ratio=True,
    duo_num_samples=50, duo_q_len=500, duo_dataset="kmfoda/booksum", duo_split="train", duo_text_field="chapter", duo_seed=42),
}

# Debug: register a family of presses that force all layers to evict based on a fixed layer's scores
for _i in range(32):
    PRESS_REGISTRY[f"fixed_EA_layer{_i}"] = FixedLayerScoreEvictPress(press=ExpectedAttentionPress(), score_layer_idx=_i)
