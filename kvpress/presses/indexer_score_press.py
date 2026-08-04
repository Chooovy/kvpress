from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from kvpress.presses.scorer_press import ScorerPress
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

logger = logging.getLogger(__name__)


def _apply_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    """
    Activation used in the indexer *logits* path.

    Notes:
    - DeepSeek-style "lightning indexer" often uses ReLU to enforce non-negativity.
    - For KL(teacher||Softmax(student_logits)) training, `none` can be more expressive.
    """
    act = (activation or "relu").lower()
    if act in ("none", "identity", "linear"):
        return x
    if act == "relu":
        return F.relu(x)
    if act == "leaky_relu":
        return F.leaky_relu(x, negative_slope=0.01)
    if act == "softplus":
        return F.softplus(x)
    raise ValueError(f"Unknown activation: {activation!r}. Use one of: relu, softplus, leaky_relu, none.")


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
        self.activation = getattr(indexer_args, "activation", "relu")

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

        # chunk_size = getattr(self, "chunk_size", 128)
        # score_chunks = []
        # for k_start in range(0, seqlen, chunk_size):
        #     k_end = min(k_start + chunk_size, seqlen)
        #     k_slice = full_k[:, k_start:k_end, :]  # (b, t_chunk, d)
        #     logits_chunk = torch.einsum("bshd,btd->bsht", full_q, k_slice)  # (b, s, h, t_chunk)
        #     logits_chunk = torch.relu(logits_chunk) * full_weights.unsqueeze(-1)
        #     score_chunks.append(logits_chunk.sum(dim=2))  # (b, s, t_chunk)
        #     del k_slice, logits_chunk

        # index_score = torch.cat(score_chunks, dim=-1)
        # del score_chunks

        logits = torch.einsum("bshd,btd->bsht", full_q, full_k)
        logits = _apply_activation(logits, self.activation) * full_weights.unsqueeze(-1)
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
    chunk_size: int = 256
    activation: str = "relu"
    last_n_query: int | None = None
    # How to reduce (bsz, q_len, k_len) -> (bsz, k_len) before selecting KV tokens.
    # - "auto": keep current behavior: mean over queries, or mean over last_n_query if set and q_len>64.
    # - "recency_weighted": exponentially decayed weights favoring recent queries (optionally within last_n_query window).
    # - "top_p_gating": keep only "peaky" queries (top-p mass concentrated) then mean.
    query_reduce: str = "auto"
    # recency_weighted hyperparam: half-life in tokens (larger => flatter weights).
    recency_half_life: float = 32.0
    # top_p_gating hyperparams (approximate; uses per-row top-k logits only, avoids full softmax over k_len)
    top_p_mass: float = 0.9
    top_p_keep_ratio: float = 0.25
    top_p_min_queries: int = 8
    top_p_topk: int = 256

    def _select_query_window(self, scores_qk: torch.Tensor) -> torch.Tensor:
        """If last_n_query is set, restrict aggregation to the last N queries."""
        if self.last_n_query is None:
            return scores_qk
        q_len = scores_qk.size(1)
        n = min(int(self.last_n_query), q_len)
        return scores_qk[:, -n:, :]

    def _reduce_recency_weighted(self, scores_qk: torch.Tensor) -> torch.Tensor:
        """Exponential recency weighting over query positions."""
        if scores_qk.size(1) == 1:
            return scores_qk.squeeze(1)
        q_len = scores_qk.size(1)
        half_life = float(max(self.recency_half_life, 1e-3))
        decay = math.log(2.0) / half_life
        # newest query gets age=0 and highest weight
        ages = torch.arange(q_len - 1, -1, -1, device=scores_qk.device, dtype=torch.float32)
        w = torch.exp(-decay * ages)
        w = w / w.sum().clamp(min=1e-8)
        out = (scores_qk.float() * w.view(1, q_len, 1)).sum(dim=1)
        return out.to(scores_qk.dtype)

    def _top_p_mass_query_mask(self, scores_qk: torch.Tensor) -> torch.Tensor:
        """
        Compute a boolean mask of queries to keep based on top-p mass concentration.

        Approximation: take top_k logits per query row, softmax within those top_k,
        and count how many tokens are needed to reach cumulative prob >= top_p_mass.
        Smaller count => more "peaky"/informative query.
        Keep the fraction top_p_keep_ratio of queries with smallest counts (per sample).
        """
        bsz, q_len, k_len = scores_qk.shape
        if q_len == 1:
            return torch.ones((bsz, 1), device=scores_qk.device, dtype=torch.bool)

        top_p = float(min(max(self.top_p_mass, 0.0), 1.0))
        keep_ratio = float(min(max(self.top_p_keep_ratio, 0.0), 1.0))
        topk = int(max(1, min(int(self.top_p_topk), k_len)))

        vals = scores_qk.float().topk(topk, dim=-1).values  # (b, q, topk) sorted desc
        probs = F.softmax(vals, dim=-1)
        cdf = probs.cumsum(dim=-1)
        reached = cdf >= top_p
        any_reached = reached.any(dim=-1)  # (b, q)
        first_idx = reached.float().argmax(dim=-1)  # 0 if none reached
        mass_size = first_idx + 1
        sentinel = torch.full_like(mass_size, topk + 1)
        mass_size = torch.where(any_reached, mass_size, sentinel)

        n_keep = int(math.ceil(q_len * keep_ratio))
        n_keep = max(int(self.top_p_min_queries), n_keep)
        n_keep = min(n_keep, q_len)
        kept_idx = mass_size.topk(n_keep, dim=1, largest=False).indices  # (b, n_keep)
        mask = torch.zeros((bsz, q_len), device=scores_qk.device, dtype=torch.bool)
        mask.scatter_(1, kept_idx, True)
        return mask

    def _reduce_top_p_gating(self, scores_qk: torch.Tensor) -> torch.Tensor:
        mask = self._top_p_mass_query_mask(scores_qk)  # (b, q)
        denom = mask.sum(dim=1).clamp(min=1).view(-1, 1).to(scores_qk.dtype)
        out = (scores_qk * mask.unsqueeze(-1).to(scores_qk.dtype)).sum(dim=1) / denom
        return out

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
                # 如果已经存在，也更新 chunk_size
                existing_indexer = getattr(attn, self.scorer_attr)
                if hasattr(existing_indexer, "chunk_size"):
                    existing_indexer.chunk_size = self.chunk_size
                continue
            args = SimpleNamespace(
                dim=getattr(attn, "hidden_size", model.config.hidden_size),
                n_heads=model.config.num_attention_heads // 2,
                head_dim=attn.head_dim // 4,
                activation=self.activation,
            )
            indexer = Indexer(model.config, args).to(model.device, dtype=model.dtype)
            indexer.chunk_size = self.chunk_size
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

    def indexer_logits_chunks(self, module, hidden_states, kwargs, chunk_size=None):
        mask = self._prepare_mask(kwargs.get("attention_mask"), hidden_states.size(1), hidden_states.size(1), hidden_states.device)
        freqs = self._prepare_freqs_cis(kwargs)

        indexer: Indexer = getattr(module, self.scorer_attr)
        chunk_size = chunk_size or getattr(indexer, "chunk_size", self.chunk_size)
        act = getattr(indexer, "activation", self.activation)

        bsz, seqlen, _ = hidden_states.size()
        x = hidden_states
        if x.dtype != indexer.w_q.weight.dtype:
            x = x.to(indexer.w_q.weight.dtype)

        q = indexer.w_q(x)
        q = q.view(bsz, seqlen, indexer.n_heads, indexer.head_dim)
        q = indexer.q_norm(q)
        if freqs is not None:
            q = apply_rotary_emb(q, freqs, False)

        k = indexer.w_k(x)
        k = indexer.k_norm(k)
        if freqs is not None:
            k = apply_rotary_emb(k.unsqueeze(2), freqs, False).squeeze(2)

        weights = indexer.weights_proj(x) * indexer.n_heads ** -0.5 * indexer.softmax_scale

        for k_start in range(0, seqlen, chunk_size):
            k_end = min(k_start + chunk_size, seqlen)
            k_slice = k[:, k_start:k_end, :]  # (b, t_chunk, d)
            logits_chunk = torch.einsum("bshd,btd->bsht", q, k_slice)  # (b, s, h, t_chunk)
            logits_chunk = _apply_activation(logits_chunk, act)
            logits_chunk = logits_chunk * weights.unsqueeze(-1)
            chunk_score = logits_chunk.sum(dim=2)  # (b, s, t_chunk)
            if mask is not None:
                mask_chunk = mask[:, :, k_start:k_end]
                chunk_score = chunk_score + mask_chunk
            yield chunk_score
            del k_slice, logits_chunk, chunk_score

    def indexer_logits_chunks_with_ranges(self, module, hidden_states, kwargs, chunk_size=None):
        """
        Like `indexer_logits_chunks`, but also yields the key range so callers don't
        need to infer k_start/k_end from enumerate().

        Yields (k_start, k_end, chunk_score) where chunk_score is (bsz, seqlen, t_chunk).
        """
        mask = self._prepare_mask(kwargs.get("attention_mask"), hidden_states.size(1), hidden_states.size(1), hidden_states.device)
        freqs = self._prepare_freqs_cis(kwargs)

        indexer: Indexer = getattr(module, self.scorer_attr)
        chunk_size = chunk_size or getattr(indexer, "chunk_size", self.chunk_size)
        act = getattr(indexer, "activation", self.activation)

        bsz, seqlen, _ = hidden_states.size()
        x = hidden_states
        if x.dtype != indexer.w_q.weight.dtype:
            x = x.to(indexer.w_q.weight.dtype)

        q = indexer.w_q(x)
        q = q.view(bsz, seqlen, indexer.n_heads, indexer.head_dim)
        q = indexer.q_norm(q)
        if freqs is not None:
            q = apply_rotary_emb(q, freqs, False)

        k = indexer.w_k(x)
        k = indexer.k_norm(k)
        if freqs is not None:
            k = apply_rotary_emb(k.unsqueeze(2), freqs, False).squeeze(2)

        weights = indexer.weights_proj(x) * indexer.n_heads ** -0.5 * indexer.softmax_scale

        for k_start in range(0, seqlen, chunk_size):
            k_end = min(k_start + chunk_size, seqlen)
            k_slice = k[:, k_start:k_end, :]  # (b, t_chunk, d)
            logits_chunk = torch.einsum("bshd,btd->bsht", q, k_slice)  # (b, s, h, t_chunk)
            logits_chunk = _apply_activation(logits_chunk, act)
            logits_chunk = logits_chunk * weights.unsqueeze(-1)
            chunk_score = logits_chunk.sum(dim=2)  # (b, s, t_chunk)
            if mask is not None:
                mask_chunk = mask[:, :, k_start:k_end]
                chunk_score = chunk_score + mask_chunk
            yield k_start, k_end, chunk_score
            del k_slice, logits_chunk, chunk_score

    def indexer_logits_chunks_with_ranges_qchunk(self, module, hidden_states, kwargs, chunk_size=None, q_start: int = 0, q_end: int | None = None):
        """
        Q-chunked variant for memory efficiency.
        Yields (k_start, k_end, chunk_score_q) where chunk_score_q is (bsz, q_chunk, t_chunk).
        """
        seqlen = hidden_states.size(1)
        if q_end is None:
            q_end = seqlen
        assert 0 <= q_start < q_end <= seqlen

        mask = self._prepare_mask(kwargs.get("attention_mask"), q_end - q_start, seqlen, hidden_states.device)
        freqs = self._prepare_freqs_cis(kwargs)

        indexer: Indexer = getattr(module, self.scorer_attr)
        chunk_size = chunk_size or getattr(indexer, "chunk_size", self.chunk_size)
        act = getattr(indexer, "activation", self.activation)

        x = hidden_states
        if x.dtype != indexer.w_q.weight.dtype:
            x = x.to(indexer.w_q.weight.dtype)

        x_q = x[:, q_start:q_end, :]
        q = indexer.w_q(x_q)
        q = q.view(x_q.size(0), q_end - q_start, indexer.n_heads, indexer.head_dim)
        q = indexer.q_norm(q)
        if freqs is not None:
            q = apply_rotary_emb(q, freqs[:, q_start:q_end, ...], False)

        k = indexer.w_k(x)
        k = indexer.k_norm(k)
        if freqs is not None:
            k = apply_rotary_emb(k.unsqueeze(2), freqs, False).squeeze(2)

        weights = indexer.weights_proj(x_q) * indexer.n_heads ** -0.5 * indexer.softmax_scale  # (b, q_chunk, h)

        for k_start in range(0, seqlen, chunk_size):
            k_end = min(k_start + chunk_size, seqlen)
            k_slice = k[:, k_start:k_end, :]  # (b, t_chunk, d)
            logits_chunk = torch.einsum("bqhd,btd->bqht", q, k_slice)  # (b, q_chunk, h, t_chunk)
            logits_chunk = _apply_activation(logits_chunk, act)
            logits_chunk = logits_chunk * weights.unsqueeze(-1)
            chunk_score = logits_chunk.sum(dim=2)  # (b, q_chunk, t_chunk)
            if mask is not None:
                mask_chunk = mask[:, :, k_start:k_end]
                chunk_score = chunk_score + mask_chunk
            yield k_start, k_end, chunk_score
            del k_slice, logits_chunk, chunk_score

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

        # Reduce (bsz, q_len, k_len) -> (bsz, k_len)
        scores_qk = self._select_query_window(indexer_scores)
        mode = (self.query_reduce or "auto").lower()
        if mode == "auto":
            if q_len == 1:
                token_scores = indexer_scores.squeeze(1)
            elif q_len <= 64:
                token_scores = indexer_scores.mean(dim=1)
            else:
                token_scores = scores_qk.mean(dim=1) if scores_qk.size(1) > 1 else scores_qk.squeeze(1)
        elif mode in ("mean", "avg"):
            token_scores = scores_qk.mean(dim=1) if scores_qk.size(1) > 1 else scores_qk.squeeze(1)
        elif mode in ("recency", "recency_weighted", "recency-weighted"):
            token_scores = self._reduce_recency_weighted(scores_qk)
        elif mode in ("top_p", "top_p_gating", "top-p", "top-p-gating"):
            token_scores = self._reduce_top_p_gating(scores_qk)
        else:
            raise ValueError(f"Unknown query_reduce={self.query_reduce!r}.")



        token_scores = token_scores[:, self.n_sink :]

        scores = token_scores.unsqueeze(1).expand(-1, keys.size(1), -1)
        if self.use_vnorm:
            vnorm = values[:, :, self.n_sink :].norm(dim=-1)
            scores = (scores + 1e-6) * vnorm

        # Handle empty scores tensor (when decode_len < n_sink)
        if scores.numel() == 0:
            # If scores is empty, use a default value for sink_fill
            sink_fill = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        else:
            sink_fill = scores.max().detach()
        scores = F.pad(scores, (self.n_sink, 0), value=sink_fill)
        return scores




def load_model_with_indexer_press(model_path, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Support both single-file and sharded HF checkpoints.
    # Many HuggingFace checkpoints are saved as:
    # - pytorch_model.bin.index.json + pytorch_model-0000x-of-0000y.bin
    # rather than a single pytorch_model.bin.
    checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
    index_path = os.path.join(model_path, "pytorch_model.bin.index.json")

    def _ensure_indexer_modules():
        dummy_press = IndexerScorePress(compression_ratio=0.0)
        dummy_press.post_init_from_model(model, force_reinit=False)

    loaded = False

    # Case 1: sharded .bin checkpoint
    if os.path.exists(index_path):
        import json

        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        indexer_keys = [k for k in weight_map.keys() if "indexer" in k]

        if indexer_keys:
            _ensure_indexer_modules()

            shard_files = sorted({weight_map[k] for k in indexer_keys})
            indexer_state = {}
            for shard_file in shard_files:
                shard_path = os.path.join(model_path, shard_file)
                if not os.path.exists(shard_path):
                    logger.warning("Missing shard file referenced by index: %s", shard_path)
                    continue
                shard_sd = torch.load(shard_path, map_location="cpu")
                # Keep only the scorer weights to avoid re-loading the full model.
                for k, v in shard_sd.items():
                    if "indexer" in k:
                        indexer_state[k] = v
                del shard_sd

            if indexer_state:
                incompatible = model.load_state_dict(indexer_state, strict=False)
                logger.info(
                    "Loaded IndexerScorePress weights from sharded checkpoint: %s keys (missing=%s, unexpected=%s)",
                    len(indexer_state),
                    len(getattr(incompatible, "missing_keys", []) or []),
                    len(getattr(incompatible, "unexpected_keys", []) or []),
                )
                print("✓ Loaded trained IndexerScorePress weights from sharded checkpoint", flush=True)
                loaded = True
            else:
                logger.warning(
                    "Found %s indexer keys in index.json but loaded 0 indexer tensors from shards (%s).",
                    len(indexer_keys),
                    shard_files,
                )
        else:
            logger.warning("No indexer weights found in %s (weight_map size=%s).", index_path, len(weight_map))

    # Case 2: single-file .bin checkpoint (legacy)
    if (not loaded) and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        has_trained_scorer = any("indexer" in k for k in state_dict.keys())

        if has_trained_scorer:
            _ensure_indexer_modules()
            incompatible = model.load_state_dict(state_dict, strict=False)
            logger.info(
                "Loaded IndexerScorePress weights from single-file checkpoint: missing=%s, unexpected=%s",
                len(getattr(incompatible, "missing_keys", []) or []),
                len(getattr(incompatible, "unexpected_keys", []) or []),
            )
            print("✓ Loaded trained IndexerScorePress weights from checkpoint", flush=True)
            loaded = True

        del state_dict
    
    return model, tokenizer