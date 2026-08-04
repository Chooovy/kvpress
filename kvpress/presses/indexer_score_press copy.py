from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from kvpress.presses.scorer_press import ScorerPress
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True) -> torch.Tensor:
    """Minimal RoPE helper that accepts complex-valued frequencies."""
    if freqs_cis is None:
        return x
    dtype = x.dtype
    shape = x.shape
    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    if not interleaved:
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
    return y.to(dtype)



class LayerNorm(nn.Module):
    """Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(x.float(), (self.dim,), self.weight, self.bias, self.eps).type_as(x)



class Indexer(nn.Module):
    def __init__(self, config, indexer_args):
        super().__init__()
        self.dim = indexer_args.dim
        self.n_heads = indexer_args.n_heads
        self.head_dim = indexer_args.head_dim

        self.w_q = nn.Linear(self.dim, self.n_heads * self.head_dim)
        self.w_k = nn.Linear(self.dim, self.head_dim)

        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.weights_proj = nn.Linear(self.dim, self.n_heads)
        self.softmax_scale = self.head_dim ** -0.5

    def forward(self, x, freqs_cis, mask):
        bsz, seqlen, _ = x.size()
        if x.dtype != self.w_q.weight.dtype:
            x = x.to(self.w_q.weight.dtype)

        q = self.w_q(x)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        q = self.q_norm(q)
        if freqs_cis is not None:
            q = apply_rotary_emb(q, freqs_cis, False)

        k = self.w_k(x)
        k = self.k_norm(k)
        if freqs_cis is not None:
            k = apply_rotary_emb(k.unsqueeze(2), freqs_cis, False).squeeze(2)
        
        weights = self.weights_proj(x) * self.n_heads ** -0.5 * self.softmax_scale

        full_k = k
        full_q = q
        full_weights = weights


        logits = torch.einsum("bshd,btd->bsht", full_q, full_k)
        logits = torch.relu(logits) * full_weights.unsqueeze(-1)
        index_score = logits.sum(dim=2)

        if mask is not None:
            index_score = index_score + mask
        return index_score


@dataclass
class IndexerScorePress(ScorerPress):
    scorer_attr: str = "indexer"
    use_vnorm: bool = False
    n_sink: int = 4
    mask_fill_value: float = -1e4
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
            if hasattr(attn, self.scorer_attr):
                continue
            args = SimpleNamespace(
                dim=getattr(attn, "hidden_size", model.config.hidden_size),
                n_heads=model.config.num_attention_heads // 2,
                head_dim=attn.head_dim // 4,
            )
            indexer = Indexer(model.config, args).to(model.device, dtype=model.dtype)
            attn.register_module(self.scorer_attr, indexer)

        print(f"✓ Initialized new {self.scorer_attr} modules")
        self._initialized = True



    def _prepare_mask(self, attention_mask: Optional[torch.Tensor], q_len: int, k_len: int, device: torch.device) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.dim() == 4:
            # (bsz, 1, q_len, k_len) -> (bsz, q_len, k_len)
            return attention_mask[:, 0, -q_len:, -k_len:]
        if attention_mask.dim() == 2:
            keep = attention_mask[:, -k_len:].unsqueeze(1).expand(-1, q_len, -1)
            return torch.where(keep > 0, torch.zeros(1, device=device), torch.full((), self.mask_fill_value, device=device))
        return None

    def indexer_logits(self, module, hidden_states, kwargs):
        mask = self._prepare_mask(kwargs.get("attention_mask"), hidden_states.size(1), hidden_states.size(1), hidden_states.device)
        freqs = self._prepare_freqs_cis(kwargs)

        return getattr(module, self.scorer_attr)(hidden_states, freqs, mask)

    # def _prepare_freqs_cis(self, kwargs) -> Optional[torch.Tensor]:
    #     freqs = kwargs.get("indexer_freqs_cis")
    #     if freqs is not None:
    #         return freqs
    #     position_embeddings = kwargs.get("position_embeddings")
    #     if position_embeddings is None:
    #         return None
    #     cos, sin = position_embeddings
    #     cos = cos.squeeze(1).float()
    #     sin = sin.squeeze(1).float()
    #     # Convert cos/sin tables (bsz=1) to complex-valued freqs.
    #     return torch.complex(cos[..., ::2], sin[..., ::2])
    def _prepare_freqs_cis(self, kwargs) -> Optional[torch.Tensor]:
        freqs = kwargs.get("indexer_freqs_cis")
        if freqs is not None:
            return freqs
        position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is None:
            return None
        cos, sin = position_embeddings
        cos = cos.squeeze(1).float()
        sin = sin.squeeze(1).float()
        head_dim_indexer = cos.size(-1) // 4
        cos = cos[..., :head_dim_indexer]
        sin = sin[..., :head_dim_indexer]
        return torch.complex(cos[..., ::2], sin[..., ::2])

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        indexer: Indexer = getattr(module, self.scorer_attr)
        q_len = hidden_states.size(1)
        k_len = keys.size(2)

        is_decoding_compression = kwargs.get("is_decoding_compression", False)
        buffer_start_pos = kwargs.get("buffer_start_pos", None)


        # mask = self._prepare_mask(kwargs.get("attention_mask"), q_len, k_len, hidden_states.device)
        mask = None
        # freqs = self._prepare_freqs_cis(kwargs)
        freqs = None

        indexer_scores = indexer(hidden_states, freqs, mask)
        if q_len == 1:
            token_scores = indexer_scores.squeeze(1)  # (bsz, k_len)
            # token_scores = indexer_scores[:, -1, :]  # (bsz, k_len)
        
        elif q_len <= 64:
            # weights = torch.linspace(0.5, 1.0, q_len, device=indexer_scores.device)
            # weights = weights / weights.sum()
            # token_scores = (indexer_scores * weights[None, :, None]).sum(dim=1)  # (bsz, k_len)
            token_scores = indexer_scores.mean(dim=1)  # (bsz, k_len)
        
        else:
            last_n = 32
            token_scores = indexer_scores[:, -last_n:, :].mean(dim=1)
            # weights = torch.linspace(0.1, 1.0, q_len, device=indexer_scores.device)
            # weights = weights / weights.sum()
            # token_scores = (indexer_scores * weights[None, :, None]).sum(dim=1)

        token_scores = token_scores[:, self.n_sink :]

        scores = token_scores.unsqueeze(1).expand(-1, keys.size(1), -1)
        if self.use_vnorm:
            vnorm = values[:, :, self.n_sink :].norm(dim=-1)
            scores = (scores + 1e-6) * vnorm

        sink_fill = scores.max().detach()
        scores = F.pad(scores, (self.n_sink, 0), value=sink_fill)
        return scores




def load_model_with_indexer_press(model_path, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
    
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        has_trained_scorer = any("indexer" in k for k in state_dict.keys())
        
        if has_trained_scorer:
            dummy_press = IndexerScorePress(compression_ratio=0.0)
            dummy_press.post_init_from_model(model, force_reinit=False)
            
            model.load_state_dict(state_dict, strict=False)
            print("✓ Loaded trained IndexerScorePress weights from checkpoint")
        
        del state_dict
    
    return model, tokenizer