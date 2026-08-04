from transformers import pipeline
from kvpress import KnormPress
from kvpress import DecodingPress
from kvpress.presses.indexer_score_press_cache import CacheIndexerScorePress
from kvpress.presses.indexer_score_press import IndexerScorePress
from kvpress.presses.decoding_cache_indexer import CacheIndexerDecodingPress
from kvpress.presses.selective_decoding_press import SelectiveDecodingPress
from kvpress.presses.indexer_score_query_press import QueryIndexerDecodingPress, QueryIndexerScorePress
from dataclasses import dataclass, field
from collections import defaultdict
import torch


device = "cuda:0"
# model = "/aifs4su/guhao/checkpoints/llama3-1b-instruct-indexer_score"
model = "/aifs4su/guhao/Models/Llama-3.2-1B-Instruct"
model_kwargs = {"attn_implementation": "eager"}
pipe = pipeline("kv-press-text-generation", model=model, device=device, model_kwargs=model_kwargs)
# base_press = CacheIndexerScorePress()
# decoding_press = CacheIndexerDecodingPress(
#     base_press=base_press, 
#     compression_interval=8, 
#     target_size=10, 
#     hidden_states_buffer_size=10
# )
# base_press = IndexerScorePress()
# decoding_press = SelectiveDecodingPress(
#     base_press=base_press, 
#     compression_interval=8, 
#     target_size=10, 
#     hidden_states_buffer_size=10
# )

# base_press = QueryIndexerScorePress()
# decoding_press = QueryIndexerDecodingPress(
#     base_press=base_press, 
#     compression_interval=8, 
#     target_size=10, 
#     hidden_states_buffer_size=10
# )
base_press = QueryIndexerScorePress(compression_ratio=0.25)
context = "A very long text you want to compress during generation"
question = "Tell me a long story about this context"
response = pipe(context, question=question, press=base_press)["answer"]
print(response)