from transformers import pipeline
from kvpress import KnormPress
from kvpress import DecodingPress
from kvpress.presses.indexer_score_press_cache import CacheIndexerScorePress
from kvpress.presses.decoding_cache_indexer import CacheIndexerDecodingPress
from dataclasses import dataclass, field
from collections import defaultdict
import torch

# 创建一个带evict跟踪的CacheIndexerScorePress
@dataclass
class TrackedCacheIndexerScorePress(CacheIndexerScorePress):
    """带evict跟踪的CacheIndexerScorePress"""
    evict_count: int = 0
    total_tokens: int = 0
    evict_info: list = field(default_factory=list)
    
    def compress(
        self,
        module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """重写compress方法以跟踪evict信息"""
        if self.compression_ratio == 0:
            return keys, values

        # 记录压缩前的信息
        original_k_len = keys.shape[2]
        q_len = hidden_states.shape[1] if hidden_states is not None else 0
        
        # 判断是decode还是prefill阶段
        cache_position = kwargs.get("cache_position", None)
        is_decoding = kwargs.get("is_decoding", False)
        if cache_position is not None and len(cache_position) > 0:
            is_decode_phase = cache_position[-1] > q_len if q_len > 0 else True
        else:
            is_decode_phase = is_decoding or (q_len == 1)
        
        # 调用父类的compress逻辑
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        # 获取要保留的indices
        k_len = keys.shape[2]
        n_kept = int(k_len * (1 - self.compression_ratio))
        indices = scores.topk(n_kept, dim=-1).indices
        
        # 计算evict的数量
        n_evicted = k_len - n_kept
        self.evict_count += n_evicted
        self.total_tokens += k_len
        
        # 记录evict信息
        evict_info = {
            "layer": getattr(module, 'layer_idx', 'unknown'),
            "is_decode": is_decode_phase,
            "is_decoding_flag": is_decoding,
            "q_len": q_len,
            "cache_pos": cache_position[-1] if cache_position is not None and len(cache_position) > 0 else None,
            "original_k_len": original_k_len,
            "n_kept": n_kept,
            "n_evicted": n_evicted,
            "compression_ratio": self.compression_ratio,
        }
        self.evict_info.append(evict_info)
        
        # 打印evict信息
        phase = "DECODE" if is_decode_phase else "PREFILL"
        cache_pos_str = f", cache_pos={evict_info['cache_pos']}" if evict_info['cache_pos'] is not None else ""
        print(f"[{phase}] Layer {evict_info['layer']}: "
              f"q_len={q_len}{cache_pos_str}, "
              f"原始KV长度={original_k_len}, "
              f"保留={n_kept}, "
              f"Evict={n_evicted} tokens "
              f"(压缩率={self.compression_ratio:.2f})")
        
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        # Prune keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values


# 创建一个带evict跟踪的CacheIndexerDecodingPress
@dataclass
class TrackedCacheIndexerDecodingPress(CacheIndexerDecodingPress):
    """带evict跟踪的CacheIndexerDecodingPress"""
    evict_count: int = 0
    total_tokens: int = 0
    evict_info: list = field(default_factory=list)
    
    def __post_init__(self):
        # 重写__post_init__以接受TrackedCacheIndexerScorePress
        # 检查base_press是否是CacheIndexerScorePress或其子类
        from kvpress.presses.indexer_score_press_cache import CacheIndexerScorePress
        if not isinstance(self.base_press, CacheIndexerScorePress):
            raise ValueError("base_press must be CacheIndexerScorePress or its subclass")
        
        self.hidden_states_buffer = defaultdict(list)
        self.layer_step_counts = defaultdict(int)
    
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        """重写compress方法以跟踪evict信息"""
        layer_idx = module.layer_idx
        indexer = getattr(module, self.base_press.scorer_attr)
        k_len = keys.shape[2]
        original_k_len = k_len
        
        print(f"[DEBUG] Layer {layer_idx}: compress被调用!")
        print(f"  - k_len={k_len}, target_size={self.target_size}")
        print(f"  - hidden_states shape: {hidden_states.shape if hidden_states is not None else None}")
        
        # 1. 累积 buffered hidden states 到 indexer cache
        with torch.no_grad():
            _ = indexer(hidden_states, freqs_cis=None, mask=None, use_cache=True)
        
        # 2. 检查长度是否匹配
        indexer_len = indexer.k_cache.shape[1]
        print(f"[DEBUG] Layer {layer_idx}: indexer_len={indexer_len}, k_len={k_len}")
        if indexer_len != k_len:
            print(f"Layer {layer_idx}: indexer cache len ({indexer_len}) != KV cache len ({k_len}). 跳过压缩")
            return keys, values
        
        # 3. 计算压缩比例和目标大小
        if k_len <= self.target_size:
            # 不需要压缩
            print(f"[DEBUG] Layer {layer_idx}: k_len ({k_len}) <= target_size ({self.target_size}), 跳过压缩")
            return keys, values
        
        n_kept = self.target_size
        target_ratio = 1.0 - (n_kept / k_len)
        n_evicted = k_len - n_kept
        
        # 记录evict信息
        evict_info = {
            "layer": layer_idx,
            "is_decode": True,  # 这个press只在decode阶段工作
            "original_k_len": original_k_len,
            "n_kept": n_kept,
            "n_evicted": n_evicted,
            "compression_ratio": target_ratio,
            "target_size": self.target_size,
        }
        self.evict_info.append(evict_info)
        self.evict_count += n_evicted
        self.total_tokens += original_k_len
        
        # 打印evict信息
        print(f"[DECODE] Layer {layer_idx}: "
              f"原始KV长度={original_k_len}, "
              f"目标大小={self.target_size}, "
              f"保留={n_kept}, "
              f"Evict={n_evicted} tokens "
              f"(压缩率={target_ratio:.2%})")
        
        # 4. 计算 scores
        kwargs["is_decoding"] = True
        original_ratio = self.base_press.compression_ratio
        self.base_press.compression_ratio = target_ratio
        
        scores = self.base_press.score(
            module, hidden_states=None, keys=keys, values=values, 
            attentions=attentions, kwargs=kwargs
        )
        
        self.base_press.compression_ratio = original_ratio
        
        # 5. 选择 top-k tokens（保持原始顺序）
        indices = scores.topk(n_kept, dim=-1).indices
        sorted_indices = indices.sort(dim=-1).values
        gather_idx = sorted_indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        
        # 6. 压缩 KV cache
        compressed_keys = keys.gather(2, gather_idx).contiguous()
        compressed_values = values.gather(2, gather_idx).contiguous()
        
        # 7. 同步压缩 indexer cache
        bsz = sorted_indices.shape[0]
        token_indices = sorted_indices[:, 0, :]
        
        indexer.k_cache = indexer.k_cache.gather(
            1, token_indices.unsqueeze(-1).expand(-1, -1, indexer.k_cache.shape[-1])
        ).contiguous()
        
        indexer.q_cache = indexer.q_cache.gather(
            1, token_indices.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, indexer.q_cache.shape[2], indexer.q_cache.shape[3]
            )
        ).contiguous()
        
        indexer.weights_cache = indexer.weights_cache.gather(
            1, token_indices.unsqueeze(-1).expand(-1, -1, indexer.weights_cache.shape[-1])
        ).contiguous()
        
        return compressed_keys, compressed_values


# Initialize the pipeline
device = "cuda:0"
model = "/aifs4su/guhao/checkpoints/llama3-1b-instruct-indexer_score"
model_kwargs = {"attn_implementation": "eager"}
pipe = pipeline("kv-press-text-generation", model=model, device=device, model_kwargs=model_kwargs)

# 使用带跟踪功能的press
base_press = TrackedCacheIndexerScorePress()
decoding_press = TrackedCacheIndexerDecodingPress(
    base_press=base_press, 
    compression_interval=1, 
    target_size=10, 
    hidden_states_buffer_size=10
)

# 重置统计信息
decoding_press.evict_count = 0
decoding_press.total_tokens = 0
decoding_press.evict_info = []
base_press.evict_count = 0
base_press.total_tokens = 0
base_press.evict_info = []

print("=" * 80)
print("使用CacheIndexerDecodingPress进行decode阶段压缩")
print("=" * 80)

# Use with pipeline
context = "A very long text you want to compress during generation"
question = "Tell me a long story about this context"
response = pipe(context, question=question, press=decoding_press)["answer"]

print("\n" + "=" * 80)
print("Evict统计信息 (CacheIndexerDecodingPress):")
print("=" * 80)
print(f"总evict次数: {len(decoding_press.evict_info)}")
print(f"总evict tokens数: {decoding_press.evict_count}")
print(f"总tokens数: {decoding_press.total_tokens}")
if decoding_press.total_tokens > 0:
    print(f"平均evict率: {decoding_press.evict_count / decoding_press.total_tokens:.2%}")

print(f"\nBase Press (CacheIndexerScorePress) 统计:")
print(f"总evict次数: {len(base_press.evict_info)}")
print(f"总evict tokens数: {base_press.evict_count}")
print(f"总tokens数: {base_press.total_tokens}")
if base_press.total_tokens > 0:
    print(f"平均evict率: {base_press.evict_count / base_press.total_tokens:.2%}")

# 检查decode阶段的evict
decode_evicts = [info for info in decoding_press.evict_info if info['is_decode']]
print(f"\nDECODE阶段evict次数: {len(decode_evicts)}")

if decode_evicts:
    print("\nDECODE阶段的evict详情:")
    for info in decode_evicts:
        print(f"  Layer {info['layer']}: "
              f"原始长度={info['original_k_len']}, "
              f"保留={info['n_kept']}, "
              f"Evict={info['n_evicted']} tokens "
              f"(压缩率={info['compression_ratio']:.2%})")
else:
    print("\n注意: 没有发生decode阶段的evict，可能是因为:")
    print("  1. 序列长度没有超过target_size")
    print("  2. 还没有达到compression_interval的步数")

print(f"\n生成的回答长度: {len(response)} 字符")
