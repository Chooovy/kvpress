# 我想知道一个model的parameters
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from kvpress import SnapKVPress
from kvpress.presses.indexer_score_query_press import load_model_with_query_indexer_press, QueryIndexerScorePress
from kvpress.presses.memory_scorer_press import load_model_with_memory_params, MemoryScorerPress
import torch
import gc
import time


model_kwargs = {"attn_implementation": "sdpa", "dtype": "bfloat16", "device_map": "cuda:0"}
question = "What is the capital of France?"

path = "/aifs4su/guhao/Models/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(path)
# print(f"Number of backbone model parameters: {model.num_parameters() / 1e9:.5f}B")


pipe = pipeline("kv-press-text-generation", model=model, tokenizer=tokenizer)

def build_context_with_tokens(tokenizer, target_tokens: int) -> str:
    seed = (
        "This is a synthetic context for timing. It contains repeated sentences "
        "to reach a target token length for benchmarking. "
    )
    chunks = []
    total = 0
    while total < target_tokens:
        chunks.append(seed)
        total = len(tokenizer(" ".join(chunks), add_special_tokens=False)["input_ids"])
    context = " ".join(chunks)
    # Trim to exact length
    ids = tokenizer(context, add_special_tokens=False)["input_ids"][:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)

def clear_kv_cache():
    # Best-effort cleanup between runs; kv cache tensors should be freed when
    # generation outputs go out of scope.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def get_peak_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)

def get_allocated_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024 ** 3)

target_tokens = 1024 * 32
output_tokens = 1024 * 1
context = build_context_with_tokens(tokenizer, target_tokens)
base_model_mem_gb = get_allocated_gb()

reset_peak_memory()
start_time = time.time()
pipe(context, question=question, press=None, max_new_tokens=output_tokens, min_new_tokens=output_tokens)
end_time = time.time()
print(f"baseline: {end_time - start_time:.4f} seconds")
print(f"baseline_peak_mem_gb: {get_peak_memory_gb():.4f}")
print(f"baseline_peak_ex_model_gb: {get_peak_memory_gb() - base_model_mem_gb:.4f}")

# del model, tokenizer, pipe
# clear_kv_cache()

presses = [
    ("cr_0.25", SnapKVPress(window_size=32, compression_ratio=0.25)),
    ("cr_0.5", SnapKVPress(window_size=32, compression_ratio=0.5)),
    ("cr_0.75", SnapKVPress(window_size=32, compression_ratio=0.75)),
    ("cr_0.90", SnapKVPress(window_size=32, compression_ratio=0.90)),
]
for name, press in presses:
    reset_peak_memory()
    start_time = time.time()
    pipe(context, question=question, press=press, max_new_tokens=output_tokens, min_new_tokens=output_tokens)
    end_time = time.time()
    print(f"test_{name}: {end_time - start_time:.4f} seconds")
    print(f"test_{name}_peak_mem_gb: {get_peak_memory_gb():.4f}")
    print(f"test_{name}_peak_ex_model_gb: {get_peak_memory_gb() - base_model_mem_gb:.4f}")
    clear_kv_cache()



path = "/aifs4su/guhao/checkpoints/llama3-8b-query_indexer-max"
model = load_model_with_query_indexer_press(path, model_kwargs=model_kwargs)
if isinstance(model, tuple):
    model, tokenizer = model
pipe = pipeline("kv-press-text-generation", model=model, tokenizer=tokenizer)
context = build_context_with_tokens(tokenizer, target_tokens)
base_model_mem_gb = get_allocated_gb()
indexer_param_count = sum(
    p.numel() for n, p in model.named_parameters() if "indexer" in n
)
# print(f"Number of query indexer parameters: {indexer_param_count / 1e6:.2f}M ({indexer_param_count / 1e9:.5f}B)")

presses = [
    ("cr_0.25", QueryIndexerScorePress(compression_ratio=0.25, last_n_query=1)),
    ("cr_0.5", QueryIndexerScorePress(compression_ratio=0.5, last_n_query=1)),
    ("cr_0.75", QueryIndexerScorePress(compression_ratio=0.75, last_n_query=1)),
    ("cr_0.90", QueryIndexerScorePress(compression_ratio=0.90, last_n_query=1)),
]
for name, press in presses:
    reset_peak_memory()
    start_time = time.time()
    pipe(context, question=question, press=press, max_new_tokens=output_tokens, min_new_tokens=output_tokens)
    end_time = time.time()
    print(f"query_indexer_{name}: {end_time - start_time:.4f} seconds")
    print(f"query_indexer_{name}_peak_mem_gb: {get_peak_memory_gb():.4f}")
    print(f"query_indexer_{name}_peak_ex_model_gb: {get_peak_memory_gb() - base_model_mem_gb:.4f}")
    clear_kv_cache()

del model, tokenizer, pipe
torch.cuda.empty_cache()

# path = "/aifs4su/guhao/checkpoints/llama3-8b-memory-max-mode"
# model = load_model_with_memory_params(path, model_kwargs=model_kwargs)
# if isinstance(model, tuple):
#     model, tokenizer = model
# pipe = pipeline("kv-press-text-generation", model=model, tokenizer=tokenizer)
# context = build_context_with_tokens(tokenizer, target_tokens)
# memory_param_count = sum(
#     p.numel() for n, p in model.named_parameters() if "memory" in n
# )
# # print(f"Number of memory parameters: {memory_param_count / 1e6:.2f}M ({memory_param_count / 1e9:.5f}B)")

# presses = [
#     ("cr_0.25", MemoryScorerPress(base_press=QueryIndexerScorePress(compression_ratio=0.25, last_n_query=1))),
#     ("cr_0.5", MemoryScorerPress(base_press=QueryIndexerScorePress(compression_ratio=0.5, last_n_query=1))),
#     ("cr_0.75", MemoryScorerPress(base_press=QueryIndexerScorePress(compression_ratio=0.75, last_n_query=1))),
#     ("cr_0.90", MemoryScorerPress(base_press=QueryIndexerScorePress(compression_ratio=0.90, last_n_query=1))),
# ]
# for name, press in presses:
#     reset_peak_memory()
#     start_time = time.time()
#     pipe(context, question=question, press=press, max_new_tokens=output_tokens, min_new_tokens=output_tokens)
#     end_time = time.time()
#     print(f"memory_{name}: {end_time - start_time:.4f} seconds")
#     print(f"memory_{name}_peak_mem_gb: {get_peak_memory_gb():.4f}")
#     clear_kv_cache()