import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from kvpress.presses.scorer_press import ScorerPress
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


class DMAScorer(nn.Module):
    def __init__(self, config, head_dim):
        super().__init__()
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim
        fused_dim = self.num_kv_heads * head_dim * 2
        self.A = nn.Parameter(torch.full((self.num_kv_heads,), 0.1))
        self.kv_fuse_proj = nn.Linear(fused_dim, self.num_kv_heads * head_dim, bias=False)
        hidden_dim = (self.num_kv_heads * head_dim) // 2
        self.dt_proj = nn.Sequential(
            nn.Linear(self.num_kv_heads * head_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_kv_heads, bias=False),
        )

    def forward(self, key_states, value_states):
        # key_states/value_states: (batch, num_kv_heads, seq_len, head_dim)
        bsz, _, seq_len, _ = value_states.shape
        key_flat = key_states.transpose(1, 2).reshape(bsz, seq_len, -1)
        value_flat = value_states.transpose(1, 2).reshape(bsz, seq_len, -1)
        kv_concat = torch.cat([key_flat, value_flat], dim=-1)
        fused = self.kv_fuse_proj(kv_concat)
        dt_states = torch.clamp(self.dt_proj(fused), min=-10.0, max=10.0)
        scores = (self.A * F.softplus(dt_states)).transpose(1, 2)  # (batch, num_kv_heads, seq_len)
        return scores



@dataclass
class DMAScorePress(ScorerPress):
    scorer_attr: str = "dma_scorer"
    use_vnorm: bool = False
    n_sink: int = 4
    _initialized: bool = False

    def post_init_from_model(self, model, force_reinit=False):
        if self._initialized and not force_reinit:
            return
        
        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        
        first_attn = language_model.layers[0].self_attn
        already_has_scorer = hasattr(first_attn, self.scorer_attr)
        
        if already_has_scorer and not force_reinit:
            print(f"✓ Found existing {self.scorer_attr} modules")
            self._initialized = True
            return
        
        for layer in language_model.layers:
            attn = layer.self_attn
            if not hasattr(attn, self.scorer_attr) or force_reinit:
                scorer = DMAScorer(model.config, attn.head_dim).to(model.device, dtype=model.dtype)
                attn.register_module(self.scorer_attr, scorer)
        
        print(f"✓ Initialized new {self.scorer_attr} modules")
        self._initialized = True

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        scorer = getattr(module, self.scorer_attr)
        scores = scorer(keys[:, :, self.n_sink :], values[:, :, self.n_sink :])
        if self.use_vnorm:
            vnorm = values[:, :, self.n_sink :].norm(dim=-1)
            scores = (scores + 1e-6) * vnorm
        sink_fill = scores.max().detach()
        scores = F.pad(scores, (self.n_sink, 0), value=sink_fill)
        return scores

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.compression_ratio == 0:
            return keys, values

        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        k_len = keys.shape[2]
        n_kept = int(k_len * (1 - self.compression_ratio))
        n_kept = max(1, min(k_len, n_kept))

        topk = scores.topk(n_kept, dim=-1)
        sorted_indices = topk.indices.sort(dim=-1).values
        gather_idx = sorted_indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        keys = keys.gather(2, gather_idx).contiguous()
        values = values.gather(2, gather_idx).contiguous()

        return keys, values

def load_model_with_dma_press(model_path, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
    
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        has_trained_scorer = any("dma_scorer" in k for k in state_dict.keys())
        
        if has_trained_scorer:
            dummy_press = DMAScorePress(compression_ratio=0.0)
            dummy_press.post_init_from_model(model, force_reinit=False)
            
            model.load_state_dict(state_dict, strict=False)
            print("✓ Loaded trained DMAScorer weights from checkpoint")
        
        del state_dict
    
    return model, tokenizer