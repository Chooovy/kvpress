# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from kvpress.attention_patch import patch_attention_functions
from kvpress.pipeline import KVPressTextGenerationPipeline
from kvpress.presses.adakv_press import AdaKVPress
from kvpress.presses.base_press import SUPPORTED_MODELS, BasePress
from kvpress.presses.block_press import BlockPress
from kvpress.presses.chunk_press import ChunkPress
from kvpress.presses.chunkkv_press import ChunkKVPress
from kvpress.presses.compactor_press import CompactorPress
from kvpress.presses.composed_press import ComposedPress
from kvpress.presses.criticalkv_press import CriticalAdaKVPress, CriticalKVPress
from kvpress.presses.cur_press import CURPress
from kvpress.presses.decoding_press import DecodingPress
from kvpress.presses.duo_attention_press import DuoAttentionPress
from kvpress.presses.expected_attention_press import ExpectedAttentionPress
from kvpress.presses.expected_attention_with_stats import ExpectedAttentionStatsPress
from kvpress.presses.finch_press import FinchPress
from kvpress.presses.key_rerotation_press import KeyRerotationPress
from kvpress.presses.keydiff_press import KeyDiffPress
from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.kvzip_press import KVzipPress
from kvpress.presses.lagkv_press import LagKVPress
from kvpress.presses.leverage_press import LeverageScorePress
from kvpress.presses.non_causal_attention_press import NonCausalAttnPress
from kvpress.presses.observed_attention_press import ObservedAttentionPress
from kvpress.presses.per_layer_compression_press import PerLayerCompressionPress
from kvpress.presses.prefill_decoding_press import PrefillDecodingPress
from kvpress.presses.pyramidkv_press import PyramidKVPress
from kvpress.presses.qfilter_press import QFilterPress
from kvpress.presses.random_press import RandomPress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.scorer_press_debug import FixedLayerScoreEvictPress
from kvpress.presses.simlayerkv_press import SimLayerKVPress
from kvpress.presses.snapkv_press import SnapKVPress
from kvpress.presses.streaming_llm_press import StreamingLLMPress
from kvpress.presses.think_press import ThinKPress
from kvpress.presses.tova_press import TOVAPress

# ---- Score Presses ----
from kvpress.presses.dma_score_press import DMAScorePress
from kvpress.presses.indexer_score_press import IndexerScorePress
from kvpress.presses.indexer_score_press_cache import CacheIndexerScorePress
from kvpress.presses.indexer_score_query_press import QueryIndexerScorePress, QueryIndexerDecodingPress
from kvpress.presses.random_press_w_sink import RandomPress_with_sink

from kvpress.presses.memory_scorer_press import MemoryScorerPress

from kvpress.presses.gated_press import GatedPress

from kvpress.presses.rkv_press import RkvPress
from kvpress.presses.h2o_press import H2OPress
# ---- Decode Presses ----
from kvpress.presses.selective_decoding_press import SelectiveDecodingPress
from kvpress.presses.decoding_cache_indexer import CacheIndexerDecodingPress
from kvpress.presses.decode_press import DecodePress
from kvpress.presses.decode_only_press import DecodeOnlyPress
from kvpress.presses.query_indexer_kvzip_press import QueryIndexer_KVzipScorePress
from kvpress.presses.gt_score_press import GTScorePress
# Patch the attention functions to support head-wise compression
patch_attention_functions()

__all__ = [
    "CriticalAdaKVPress",
    "CriticalKVPress",
    "CURPress",
    "AdaKVPress",
    "BasePress",
    "ComposedPress",
    "ScorerPress",
    "ExpectedAttentionPress",
    "KnormPress",
    "ObservedAttentionPress",
    "RandomPress",
    "SimLayerKVPress",
    "SnapKVPress",
    "StreamingLLMPress",
    "ThinKPress",
    "TOVAPress",
    "KVPressTextGenerationPipeline",
    "PerLayerCompressionPress",
    "KeyRerotationPress",
    "ChunkPress",
    "DuoAttentionPress",
    "ChunkKVPress",
    "QFilterPress",
    "PyramidKVPress",
    "FinchPress",
    "LagKVPress",
    "BlockPress",
    "KeyDiffPress",
    "KVzipPress",
    "DecodingPress",
    "PrefillDecodingPress",
    "ExpectedAttentionStatsPress",
    "DecodingPress",
    "PrefillDecodingPress",
    "CompactorPress",
    "LeverageScorePress",
    "NonCausalAttnPress",
    # ---- Score Presses ----
    "DMAScorePress",
    "IndexerScorePress",
    "CacheIndexerScorePress",
    "QueryIndexerScorePress",
    "MemoryScorerPress",
    "RandomPress_with_sink",
    "RkvPress",
    "H2OPress",
    "QueryIndexer_KVzipScorePress",
    "GatedPress",
    "FixedLayerScoreEvictPress",
    "GTScorePress",
    # ---- Decode Presses ----
    "SelectiveDecodingPress",
    "CacheIndexerDecodingPress",
    "DecodePress",
    "DecodeOnlyPress",
    "QueryIndexerDecodingPress",
]
