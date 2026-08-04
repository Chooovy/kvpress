from __future__ import annotations

import json
import math
import logging
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import QuantizedCache

from kvpress.presses.decode_press import DecodePress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import extract_keys_and_values, get_prerope_query_states
import os


logger = logging.getLogger(__name__)


def _apply_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    """
    Activation used in the indexer *logits* path.
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
    # Accept common RoPE freq shapes:
    # - (seq_len, d) complex
    # - (bsz, seq_len, d) complex
    # - (bsz, seq_len, 1, d) complex (already broadcastable)
    if freqs_cis.dim() == 2:
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, s, 1, d)
    elif freqs_cis.dim() == 3:
        freqs_cis = freqs_cis.unsqueeze(2)  # (b, s, 1, d)
    elif freqs_cis.dim() != 4:
        raise ValueError(f"freqs_cis must be 2D/3D/4D complex, got shape={tuple(freqs_cis.shape)}")

    # Sanity check: ensure sequence len matches.
    if freqs_cis.size(1) != x.size(1):
        raise ValueError(
            f"RoPE freqs seq_len mismatch: freqs_cis.size(1)={freqs_cis.size(1)} vs x.size(1)={x.size(1)}. "
            "This usually means the provided position_embeddings were sliced incorrectly."
        )
    # Ensure last dim matches the complex rotary dimension.
    if freqs_cis.size(-1) != x.size(-1):
        raise ValueError(
            f"RoPE freqs dim mismatch: freqs_cis.size(-1)={freqs_cis.size(-1)} vs x.size(-1)={x.size(-1)}. "
            "Make sure `_prepare_freqs_cis(..., head_dim=...)` uses the indexer's head_dim."
        )
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



class QueryIndexer(nn.Module):
    def __init__(self, config, indexer_args):
        super().__init__()
        self.dim = indexer_args.dim
        self.n_heads = indexer_args.n_heads
        self.head_dim = indexer_args.head_dim
        self.enable_low_rank = indexer_args.enable_low_rank
        self.max_batch_size = indexer_args.max_batch_size
        self.max_seq_len = indexer_args.max_seq_len
        self.activation = getattr(indexer_args, "activation", "relu")

        if self.enable_low_rank:
            attention_q_dim = config.num_attention_heads * (config.hidden_size // config.num_attention_heads)
            self.w_q_proj = nn.Linear(attention_q_dim, self.n_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(self.dim, self.head_dim)

        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.weights_proj = nn.Linear(self.dim, self.n_heads)
        self.softmax_scale = self.head_dim ** -0.5

        self.register_buffer("k_cache", torch.empty(0, 0, self.head_dim), persistent=False)


    def forward(self, x, query_states, freqs_cis, mask, use_cache=False):
        bsz, seqlen, _ = x.size()
        if x.device != self.w_k.weight.device:
            self.to(x.device)
        target_dtype = self.w_k.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
        
        if self.enable_low_rank:
            # (bsz, num_heads, seqlen, head_dim) -> (bsz, seqlen, num_heads * head_dim)
            query_flat = query_states.transpose(1, 2).flatten(2).to(target_dtype)
            q = self.w_q_proj(query_flat)
            q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        else:
            # query_states: (bsz, num_heads, seqlen, head_dim) -> (bsz, seqlen, num_heads, head_dim)
            q = query_states.transpose(1, 2).to(target_dtype)
        q = self.q_norm(q)
        if freqs_cis is not None:
            q = apply_rotary_emb(q, freqs_cis, False)

        k = self.w_k(x)  # (bsz, seqlen, head_dim)
        k = self.k_norm(k)
        if freqs_cis is not None:
            k = apply_rotary_emb(k.unsqueeze(2), freqs_cis, False).squeeze(2)
        
        weights = self.weights_proj(x) * self.n_heads ** -0.5 * self.softmax_scale

        if use_cache:
            if self.k_cache.shape[1] == 0: self.k_cache = k
            else: self.k_cache = torch.cat([self.k_cache, k], dim=1)
            full_k = self.k_cache
        else:
            full_k = k

        logits = torch.einsum("bshd,btd->bsht", q, full_k)

        logits = _apply_activation(logits, self.activation) * weights.unsqueeze(-1) # logits: (bsz, seqlen, n_heads, seqlen)

        index_score = logits.sum(dim=2) # index_score: (bsz, seqlen, seqlen)，index_score[b,s,t] 表示：在第 b 个样本中，第 s 个 query 对第 t 个 key 的重要性评分。

        if mask is not None:
            index_score = index_score + mask

        return index_score


    def forward_cache(self, x):
        k = self.w_k(x)  # (bsz, seqlen, head_dim)
        k = self.k_norm(k)
        if self.k_cache.shape[1] == 0: self.k_cache = k
        else: self.k_cache = torch.cat([self.k_cache, k], dim=1)
        return self.k_cache


    def get_cache_score(self, x, query_states, freqs_cis):
        bsz, seqlen, _ = x.size()
        if x.device != self.w_k.weight.device:
            self.to(x.device)
        target_dtype = self.w_k.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
        if self.enable_low_rank:
            # (bsz, num_heads, seqlen, head_dim) -> (bsz, seqlen, num_heads * head_dim)
            query_flat = query_states.transpose(1, 2).flatten(2).to(target_dtype)
            q = self.w_q_proj(query_flat)
            q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        else:
            # query_states: (bsz, num_heads, seqlen, head_dim) -> (bsz, seqlen, num_heads, head_dim)
            q = query_states.transpose(1, 2).to(target_dtype)
        q = self.q_norm(q)
        if freqs_cis is not None:
            q = apply_rotary_emb(q, freqs_cis, False)
        weights = self.weights_proj(x) * self.n_heads ** -0.5 * self.softmax_scale
        logits = torch.einsum("bshd,btd->bsht", q, self.k_cache)
        logits = _apply_activation(logits, self.activation) * weights.unsqueeze(-1)
        indexer_scores = logits.sum(dim=2)
        return indexer_scores


    def compress_cache_by_indices(self, token_indices):
        bsz, n_kept = token_indices.shape
        
        self.k_cache = self.k_cache.gather(1, token_indices.unsqueeze(-1).expand(-1, -1, self.k_cache.shape[-1])).contiguous()

    def reset_cache(self):
        self.k_cache = torch.empty(0, 0, self.head_dim)


@dataclass
class QueryIndexerScorePress(ScorerPress):
    scorer_attr: str = "indexer"
    use_vnorm: bool = False
    n_sink: int = 4
    mask_fill_value: float = -1e4
    _initialized: bool = False
    enable_low_rank: bool = True
    max_batch_size: int = 8
    max_seq_len: int = 4096
    use_pooling: bool = False
    pooling_kernel_size: int = 3
    chunk_size: int = 128
    activation: str = "relu"
    last_n_query: int | None = None
    # How to reduce (bsz, q_len, k_len) -> (bsz, k_len) before selecting KV tokens.
    # - "auto": keep current behavior: mean over queries, or mean over last_n_query if set and q_len>64.
    # - "recency_weighted": exponentially decayed weights favoring recent queries (optionally within last_n_query window).
    # - "top_p_gating": keep only "peaky" queries (top-p mass concentrated) then mean.
    # - "block": select KV in fixed-size contiguous blocks; all tokens in a block share one score and
    #            compression keeps whole blocks (helps keep multi-token entities like numbers/UUIDs).
    query_reduce: str = "auto"
    # block mode hyperparam: block size along the key sequence (non-overlapping, applied after n_sink).
    block_size: int = 16
    # recency_weighted hyperparam: half-life in tokens (larger => flatter weights).
    recency_half_life: float = 32.0
    # top_p_gating hyperparams (approximate; uses per-row top-k logits only, avoids full softmax over k_len)
    top_p_mass: float = 0.9
    top_p_keep_ratio: float = 0.25
    top_p_min_queries: int = 8
    top_p_topk: int = 256
    question_len: int = 0
    # EA mode hyperparam: number of future positions to average RoPE over.
    n_future_positions: int = 512

    # --- Deferred finalize path only (unused when pipeline uses online chunk prefill; kept for experiments) ---
    # Per-layer hidden chunks for finalize_chunk_prefill; cat(dim=1) must match KV seq length.
    _prefill_hidden_states: dict[int, list[torch.Tensor]] | None = None

    def _reset_chunk_prefill_buffers(self):
        """Clear deferred hidden buffers; call before a new chunked prefill or when skipping finalize."""
        self._prefill_hidden_states = defaultdict(list)

    def _append_chunk_hidden_states(self, module: nn.Module, hidden_states: torch.Tensor):
        """Record one prefill segment's hidden states for this layer (detached; order = chunk order)."""
        if self._prefill_hidden_states is None:
            self._prefill_hidden_states = defaultdict(list)
        layer_idx = int(getattr(module, "layer_idx", 0))
        self._prefill_hidden_states[layer_idx].append(hidden_states.detach())

    def _blockify_key_scores(self, token_scores: torch.Tensor, k_len: int) -> torch.Tensor:
        """
        Convert per-token scores (bsz, k_len) into block scores repeated over tokens:
          - keys in [0, n_sink) are kept as-is (sink)
          - keys in [n_sink, k_len) are partitioned into non-overlapping blocks of size `block_size`
          - all tokens in the same block share one score (mean of token scores in that block)
        """
        if token_scores.numel() == 0 or k_len <= 0:
            return token_scores
        if token_scores.dim() != 2:
            raise ValueError(f"token_scores must be (bsz, k_len), got shape={tuple(token_scores.shape)}")
        if token_scores.size(1) != k_len:
            raise ValueError(f"token_scores.size(1)={token_scores.size(1)} != k_len={k_len}")

        bs = int(max(1, getattr(self, "block_size", 1)))
        if k_len <= self.n_sink or bs <= 1:
            return token_scores

        out = token_scores.clone()
        start = int(self.n_sink)
        end = int(k_len)
        # Blockify only the non-sink portion.
        for s in range(start, end, bs):
            e = min(s + bs, end)
            blk = token_scores[:, s:e]
            blk_score = blk.mean(dim=1, keepdim=True)
            out[:, s:e] = blk_score.expand(-1, e - s)
        return out

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
                continue
            layer_device = next(attn.parameters()).device
            layer_dtype = next(attn.parameters()).dtype
            if self.enable_low_rank:
                n_heads = model.config.num_attention_heads // 2
                head_dim = attn.head_dim // 16
            else:
                n_heads = model.config.num_attention_heads
                head_dim = attn.head_dim
            args = SimpleNamespace(
                dim=getattr(attn, "hidden_size", model.config.hidden_size),
                n_heads=n_heads,
                head_dim=head_dim,
                enable_low_rank=self.enable_low_rank,
                max_batch_size=self.max_batch_size,
                max_seq_len=self.max_seq_len,
                activation=self.activation,
            )
            indexer = QueryIndexer(model.config, args).to(device=layer_device, dtype=layer_dtype)
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
        indexer: QueryIndexer = getattr(module, self.scorer_attr)
        freqs = self._prepare_freqs_cis(kwargs, head_dim=getattr(indexer, "head_dim", None))
        query_states = kwargs.get("query_states")
        if query_states is None:
            query_states = get_prerope_query_states(module, hidden_states)
        return getattr(module, self.scorer_attr)(hidden_states, query_states, freqs, mask, use_cache=False)

    def indexer_logits_chunks(self, module, hidden_states, kwargs, chunk_size=None):
        """
        Chunked logits computation over key positions to reduce peak memory.
        Yields per-chunk scores shaped (bsz, seq_len, chunk).
        """
        mask = self._prepare_mask(kwargs.get("attention_mask"), hidden_states.size(1), hidden_states.size(1), hidden_states.device)
        indexer: QueryIndexer = getattr(module, self.scorer_attr)
        freqs = self._prepare_freqs_cis(kwargs, head_dim=getattr(indexer, "head_dim", None))
        chunk_size = chunk_size or getattr(indexer, "chunk_size", self.chunk_size)
        act = getattr(indexer, "activation", self.activation)

        bsz, seqlen, _ = hidden_states.size()
        x = hidden_states
        target_dtype = indexer.w_k.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        query_states = kwargs.get("query_states")
        if query_states is None:
            query_states = get_prerope_query_states(module, hidden_states)

        if indexer.enable_low_rank:
            query_flat = query_states.transpose(1, 2).flatten(2).to(indexer.w_q_proj.weight.dtype)
            q = indexer.w_q_proj(query_flat)
            q = q.view(bsz, seqlen, indexer.n_heads, indexer.head_dim)
        else:
            q = query_states.transpose(1, 2)
            if q.dtype != target_dtype:
                q = q.to(target_dtype)

        q = indexer.q_norm(q)
        if freqs is not None:
            q = apply_rotary_emb(q, freqs, False)

        weights = indexer.weights_proj(x) * indexer.n_heads ** -0.5 * indexer.softmax_scale
        weights = weights.unsqueeze(-1)  # (b, s, h, 1)

        for k_start in range(0, seqlen, chunk_size):
            k_end = min(k_start + chunk_size, seqlen)
            k_slice = indexer.w_k(x[:, k_start:k_end, :])  # (b, t_chunk, d)
            k_slice = indexer.k_norm(k_slice)
            if freqs is not None:
                freqs_slice = freqs[:, k_start:k_end, ...]
                k_slice = apply_rotary_emb(k_slice.unsqueeze(2), freqs_slice, False).squeeze(2)

            logits_chunk = torch.einsum("bshd,btd->bsht", q, k_slice)  # (b, s, h, t_chunk)
            logits_chunk = _apply_activation(logits_chunk, act)
            logits_chunk = logits_chunk * weights
            chunk_score = logits_chunk.sum(dim=2)  # (b, s, t_chunk)
            if mask is not None:
                mask_chunk = mask[:, :, k_start:k_end]
                chunk_score = chunk_score + mask_chunk

            yield chunk_score

            del k_slice, logits_chunk, chunk_score

    def indexer_logits_chunks_with_ranges(self, module, hidden_states, kwargs, chunk_size=None):
        """
        Like `indexer_logits_chunks`, but yields (k_start, k_end, chunk_score) so
        callers don't need to infer ranges via enumerate().

        chunk_score is shaped (bsz, seq_len, t_chunk).
        """
        mask = self._prepare_mask(kwargs.get("attention_mask"), hidden_states.size(1), hidden_states.size(1), hidden_states.device)
        # NOTE: define `indexer` before any `getattr(indexer, ...)` usage (fixes UnboundLocalError in older revisions).
        indexer: QueryIndexer = getattr(module, self.scorer_attr)
        head_dim = getattr(indexer, "head_dim", None)
        freqs = self._prepare_freqs_cis(kwargs, head_dim=head_dim)
        chunk_size = chunk_size or getattr(indexer, "chunk_size", self.chunk_size)
        act = getattr(indexer, "activation", self.activation)

        bsz, seqlen, _ = hidden_states.size()
        x = hidden_states
        target_dtype = indexer.w_k.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        query_states = kwargs.get("query_states")
        if query_states is None:
            query_states = get_prerope_query_states(module, hidden_states)

        if indexer.enable_low_rank:
            query_flat = query_states.transpose(1, 2).flatten(2).to(indexer.w_q_proj.weight.dtype)
            q = indexer.w_q_proj(query_flat)
            q = q.view(bsz, seqlen, indexer.n_heads, indexer.head_dim)
        else:
            q = query_states.transpose(1, 2)
            if q.dtype != target_dtype:
                q = q.to(target_dtype)

        q = indexer.q_norm(q)
        if freqs is not None:
            q = apply_rotary_emb(q, freqs, False)

        weights = indexer.weights_proj(x) * indexer.n_heads ** -0.5 * indexer.softmax_scale
        weights = weights.unsqueeze(-1)  # (b, s, h, 1)

        for k_start in range(0, seqlen, chunk_size):
            k_end = min(k_start + chunk_size, seqlen)
            k_slice = indexer.w_k(x[:, k_start:k_end, :])  # (b, t_chunk, d)
            k_slice = indexer.k_norm(k_slice)
            if freqs is not None:
                freqs_slice = freqs[:, k_start:k_end, ...]
                k_slice = apply_rotary_emb(k_slice.unsqueeze(2), freqs_slice, False).squeeze(2)

            logits_chunk = torch.einsum("bshd,btd->bsht", q, k_slice)  # (b, s, h, t_chunk)
            logits_chunk = _apply_activation(logits_chunk, act)
            logits_chunk = logits_chunk * weights
            chunk_score = logits_chunk.sum(dim=2)  # (b, s, t_chunk)
            if mask is not None:
                mask_chunk = mask[:, :, k_start:k_end]
                chunk_score = chunk_score + mask_chunk

            yield k_start, k_end, chunk_score

            del k_slice, logits_chunk, chunk_score

    def indexer_logits_chunks_with_ranges_qchunk(self, module, hidden_states, kwargs, chunk_size=None, q_start: int = 0, q_end: int | None = None):
        """
        Used by finalize_chunk_prefill via _score_block_chunked_finalize: indexer scores for a slice of
        query positions [q_start:q_end) against key positions in k-chunks, to avoid full (q×k) materialization.
        Yields (k_start, k_end, chunk_score_q) with chunk_score_q shaped (bsz, q_chunk, t_chunk).
        """
        seqlen = hidden_states.size(1)
        if q_end is None:
            q_end = seqlen
        assert 0 <= q_start < q_end <= seqlen

        mask = self._prepare_mask(kwargs.get("attention_mask"), q_end - q_start, seqlen, hidden_states.device)
        indexer: QueryIndexer = getattr(module, self.scorer_attr)
        freqs = self._prepare_freqs_cis(kwargs, head_dim=getattr(indexer, "head_dim", None))
        chunk_size = chunk_size or getattr(indexer, "chunk_size", self.chunk_size)
        act = getattr(indexer, "activation", self.activation)

        bsz, _, _ = hidden_states.size()
        x = hidden_states
        target_dtype = indexer.w_k.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)

        query_states = kwargs.get("query_states")
        if query_states is None:
            query_states = get_prerope_query_states(module, hidden_states)

        # Slice query positions to q_chunk.
        qs = query_states[:, :, q_start:q_end, :]  # (b, h, q_chunk, d)
        if indexer.enable_low_rank:
            query_flat = qs.transpose(1, 2).flatten(2).to(indexer.w_q_proj.weight.dtype)  # (b, q_chunk, h*d)
            q = indexer.w_q_proj(query_flat)
            q = q.view(bsz, q_end - q_start, indexer.n_heads, indexer.head_dim)
        else:
            q = qs.transpose(1, 2)
            if q.dtype != target_dtype:
                q = q.to(target_dtype)

        q = indexer.q_norm(q)
        if freqs is not None:
            q = apply_rotary_emb(q, freqs[:, q_start:q_end, ...], False)

        x_q = x[:, q_start:q_end, :]
        weights = indexer.weights_proj(x_q) * indexer.n_heads ** -0.5 * indexer.softmax_scale
        weights = weights.unsqueeze(-1)  # (b, q_chunk, h, 1)

        for k_start in range(0, seqlen, chunk_size):
            k_end = min(k_start + chunk_size, seqlen)
            k_slice = indexer.w_k(x[:, k_start:k_end, :])  # (b, t_chunk, d)
            k_slice = indexer.k_norm(k_slice)
            if freqs is not None:
                freqs_slice = freqs[:, k_start:k_end, ...]
                k_slice = apply_rotary_emb(k_slice.unsqueeze(2), freqs_slice, False).squeeze(2)

            logits_chunk = torch.einsum("bqhd,btd->bqht", q, k_slice)  # (b, q_chunk, h, t_chunk)
            logits_chunk = _apply_activation(logits_chunk, act)
            logits_chunk = logits_chunk * weights
            chunk_score = logits_chunk.sum(dim=2)  # (b, q_chunk, t_chunk)
            if mask is not None:
                mask_chunk = mask[:, :, k_start:k_end]
                chunk_score = chunk_score + mask_chunk

            yield k_start, k_end, chunk_score

            del k_slice, logits_chunk, chunk_score

    def _prepare_freqs_cis(self, kwargs, head_dim: int | None = None) -> Optional[torch.Tensor]:
        freqs = kwargs.get("indexer_freqs_cis")
        if freqs is not None:
            return freqs
        position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is None:
            return None
        cos, sin = position_embeddings
        cos = cos.squeeze(1).float()
        sin = sin.squeeze(1).float()
        # `cos/sin` are provided by the HF model and may use a rotary dim smaller than the
        # attention head_dim (partial RoPE). For the indexer we must match its `head_dim`,
        # otherwise `apply_rotary_emb` will crash due to shape mismatch.
        if head_dim is None:
            rot_dim = int(cos.size(-1))
        else:
            rot_dim = min(int(head_dim), int(cos.size(-1)))
        rot_dim = rot_dim - (rot_dim % 2)
        if rot_dim <= 0:
            return None
        cos = cos[..., :rot_dim]
        sin = sin[..., :rot_dim]
        return torch.complex(cos[..., 0::2], sin[..., 0::2])

    def _apply_avg_rope_to_mean(self, module: nn.Module, mu: torch.Tensor, q_len: int) -> torch.Tensor:
        """
        Apply average RoPE rotation to the mean query vector.
        """
        n_future = int(getattr(self, "n_future_positions", 0))
        if n_future <= 0:
            return mu
        position_ids = torch.arange(q_len, q_len + n_future, device=mu.device)
        head_dim = int(getattr(module, "head_dim", mu.shape[-1]))

        # Newer HF Llama computes RoPE in the model and passes `position_embeddings` to attention,
        # so `LlamaAttention` may not have `rotary_emb`. Fall back to standard RoPE formula.
        if hasattr(module, "rotary_emb"):
            pos = position_ids.unsqueeze(0)
            cos, sin = module.rotary_emb(mu, pos)
            cos, sin = cos[0], sin[0]
        else:
            theta = 10000.0
            half = head_dim // 2
            if head_dim % 2 != 0:
                raise ValueError(f"Expected even head_dim for RoPE, got head_dim={head_dim}")
            inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=mu.device, dtype=torch.float32) * (2.0 / head_dim)))
            freqs = torch.einsum("p,d->pd", position_ids.to(dtype=torch.float32), inv_freq)  # (P, half)
            emb = torch.cat([freqs, freqs], dim=-1)  # (P, head_dim)
            cos = emb.cos().to(dtype=mu.dtype)
            sin = emb.sin().to(dtype=mu.dtype)

        Id = torch.eye(head_dim, device=cos.device, dtype=cos.dtype)
        P = torch.zeros((head_dim, head_dim), device=cos.device, dtype=cos.dtype)
        P[head_dim // 2 :, : head_dim // 2], P[: head_dim // 2, head_dim // 2 :] = torch.eye(head_dim // 2), -torch.eye(
            head_dim // 2
        )
        R = cos.unsqueeze(1) * Id + sin.unsqueeze(1) * P
        R = R.mean(dim=0).to(mu.device)
        return torch.matmul(mu, R.T)

    def _build_ea_query_states(self, module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Build EA (expected-attention) query states by averaging pre-RoPE queries and
        applying average RoPE rotation over future positions.
        """
        if hidden_states is None:
            raise ValueError("hidden_states is required for EA query reduction")
        q_len = int(hidden_states.size(1))
        if q_len <= 0:
            raise ValueError("hidden_states must have non-zero sequence length for EA query reduction")

        h = hidden_states[:, self.n_sink :] if hidden_states.size(1) > self.n_sink else hidden_states
        if h.size(1) == 0:
            h = hidden_states
        query_states = get_prerope_query_states(module, h)
        mu = query_states.mean(dim=2)  # (bsz, n_heads, head_dim)
        mu = self._apply_avg_rope_to_mean(module, mu, q_len)
        return mu.unsqueeze(2).expand(-1, -1, q_len, -1).contiguous()

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        indexer: QueryIndexer = getattr(module, self.scorer_attr)
        k_len = keys.size(2)
        is_decoding = kwargs.get("is_decoding", False)
        query_states = kwargs.get("query_states", None)

        if query_states is None:
            query_states = get_prerope_query_states(module, hidden_states)

        if is_decoding:
            q_len = indexer.k_cache.shape[1] if indexer.k_cache.shape[1] > 0 else 0
        else:
            q_len = hidden_states.size(1) if hidden_states is not None else 0

        mode = (self.query_reduce or "auto").lower()

        # mask = self._prepare_mask(kwargs.get("attention_mask"), q_len=q_len, k_len=k_len, device=hidden_states.device)
        mask = None
        # freqs = self._prepare_freqs_cis(kwargs, head_dim=getattr(indexer, "head_dim", None))
        freqs = None

        if mode == "ea":
            ea_query_states = self._build_ea_query_states(module, hidden_states)
            if is_decoding:
                ea_scores = indexer.get_cache_score(hidden_states, ea_query_states, freqs)
            else:
                ea_scores = indexer.forward(hidden_states, ea_query_states, freqs, mask=None, use_cache=False)
            token_scores = ea_scores.mean(dim=1) if ea_scores.size(1) > 1 else ea_scores.squeeze(1)
        else:
            if is_decoding:
                indexer_scores = indexer.get_cache_score(hidden_states, query_states, freqs)
            else:
                indexer_scores = indexer.forward(hidden_states, query_states, freqs, mask, use_cache=False)

            # Reduce (bsz, q_len, k_len) -> (bsz, k_len)
            scores_qk = self._select_query_window(indexer_scores)
            if mode == "auto":
                if q_len == 1:
                    token_scores = indexer_scores.squeeze(1)
                elif q_len <= 256:
                    token_scores = indexer_scores.mean(dim=1)
                else:
                    token_scores = scores_qk.mean(dim=1) if scores_qk.size(1) > 1 else scores_qk.squeeze(1)
            elif mode == "question":
                question_len = int(getattr(self, "question_len", 0) or 0)
                if question_len <= 0:
                    raise ValueError("question_len is not set")
                else:
                    question_len = min(question_len, int(indexer_scores.size(1)))
                    token_scores = indexer_scores[:, -question_len:, :].mean(dim=1)
            elif mode == "max":
                token_scores = scores_qk.squeeze(1) if scores_qk.size(1) == 1 else scores_qk.max(dim=1).values
            elif mode in ("recency", "recency_weighted", "recency-weighted"):
                token_scores = self._reduce_recency_weighted(scores_qk)
            elif mode in ("top_p", "top-p"):
                token_scores = self._reduce_top_p_gating(scores_qk)
            elif mode == "block":
                # Base query aggregation: mimic "auto" behavior, then blockify on the key axis.
                if q_len == 1:
                    token_scores = indexer_scores.squeeze(1)
                elif q_len <= 64:
                    token_scores = indexer_scores.mean(dim=1)
                else:
                    token_scores = scores_qk.mean(dim=1) if scores_qk.size(1) > 1 else scores_qk.squeeze(1)
                token_scores = self._blockify_key_scores(token_scores, k_len=k_len)
            else:
                raise ValueError(f"Unknown query_reduce={self.query_reduce!r}.")


        token_scores = token_scores[:, self.n_sink :]

        scores = token_scores.unsqueeze(1).expand(-1, keys.size(1), -1)
        if self.use_pooling:
            kernel_size = int(getattr(self, "pooling_kernel_size", 5))
            scores = F.avg_pool1d(scores, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
        if self.use_vnorm:
            vnorm = values[:, :, self.n_sink :].norm(dim=-1)
            scores = (scores + 1e-6) * vnorm

        # Handle potentially-empty scores (e.g., very short sequences).
        if scores.numel() == 0:
            sink_fill = torch.tensor(0.0, device=values.device, dtype=values.dtype)
        else:
            sink_fill = scores.max().detach()
        scores = F.pad(scores, (self.n_sink, 0), value=sink_fill)
        return scores

    def _compress_block_from_scores(
        self,
        module: nn.Module,
        keys: torch.Tensor,
        values: torch.Tensor,
        scores: torch.Tensor,
        compression_ratio: float | None = None,
        force_exact_token_count: bool = False,
    ):
        token_scores = scores.mean(dim=1)  # (B, K)
        bsz, k_len = token_scores.shape
        bs = int(max(1, getattr(self, "block_size", 1)))
        effective_cr = self.compression_ratio if compression_ratio is None else compression_ratio

        if k_len <= self.n_sink or bs <= 1:
            return keys, values, None

        n_kept = max(1, int(k_len * (1 - effective_cr)))
        n_kept_nonsink = max(0, n_kept - int(self.n_sink))
        nonsink_len = k_len - int(self.n_sink)
        n_blocks = int(math.ceil(nonsink_len / float(bs)))
        n_blocks_keep = int(math.ceil(n_kept_nonsink / float(bs))) if n_kept_nonsink > 0 else 0
        n_blocks_keep = min(n_blocks_keep, n_blocks)
        target_total_len = int(self.n_sink) + int(n_kept_nonsink)

        if n_blocks_keep == 0:
            keep_positions = torch.arange(int(self.n_sink), device=keys.device).view(1, -1).expand(bsz, -1)
        else:
            block_scores = []
            for bi in range(n_blocks):
                s = int(self.n_sink) + bi * bs
                e = min(s + bs, k_len)
                block_scores.append(token_scores[:, s:e].mean(dim=1))
            block_scores = torch.stack(block_scores, dim=1)
            kept_blocks = block_scores.topk(n_blocks_keep, dim=1).indices

            keep_positions_list = []
            for b in range(bsz):
                pos = list(range(int(self.n_sink)))
                if force_exact_token_count:
                    # Select blocks by score rank until we accumulate >= required non-sink tokens,
                    # then truncate to exactly `target_total_len` while keeping positional order.
                    # This prevents different layers from ending up with different KV lengths due to
                    # the last partial block.
                    non_sink_needed = int(n_kept_nonsink)
                    tokens = 0
                    selected_blocks: list[int] = []
                    block_order = block_scores[b].argsort(dim=0, descending=True).tolist()
                    for bi in block_order:
                        s = int(self.n_sink) + int(bi) * bs
                        e = min(s + bs, k_len)
                        block_token_len = int(e - s)
                        tokens += block_token_len
                        selected_blocks.append(int(bi))
                        if tokens >= non_sink_needed:
                            break
                    for bi in sorted(selected_blocks):
                        s = int(self.n_sink) + int(bi) * bs
                        e = min(s + bs, k_len)
                        pos.extend(range(s, e))
                    # Truncate to exact total length (sink + non-sink).
                    pos = pos[:target_total_len]
                else:
                    for bi in sorted(kept_blocks[b].tolist()):
                        s = int(self.n_sink) + bi * bs
                        e = min(s + bs, k_len)
                        pos.extend(range(s, e))
                keep_positions_list.append(torch.tensor(pos, device=keys.device, dtype=torch.long))
            keep_positions = torch.stack(keep_positions_list, dim=0)

        gather_idx = keep_positions[:, None, :, None].expand(-1, keys.size(1), -1, keys.size(-1))
        new_keys = keys.gather(2, gather_idx).contiguous()
        new_values = values.gather(2, gather_idx).contiguous()
        return new_keys, new_values, keep_positions

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        return super().forward_hook(module, input, kwargs, output)

    def _chunk_prefill_compression_ratio(self, chunk_idx: int, num_chunks: int) -> float:
        progress = (chunk_idx + 1) / max(num_chunks, 1)

        if progress <= 0.33:
            return self.compression_ratio * 0.3
        elif progress <= 0.66:
            return self.compression_ratio * 0.6
        else:
            return self.compression_ratio

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        # Inert unless some caller still passes kvpress_defer_compression (pipeline no longer does).
        if kwargs.get("kvpress_defer_compression", False):
            return keys, values

        if self.compression_ratio == 0:
            return keys, values

        # Deferred route: when doing chunked prefill, only buffer hidden states for later finalize.
        # The chunk prefill stage is only for memory safety (avoid online compression / score here).
        if kwargs.get("kvpress_chunk_prefill", False):
            if hidden_states is not None:
                self._append_chunk_hidden_states(module, hidden_states)
            return keys, values

        mode = (self.query_reduce or "auto").lower()
        if mode != "block":
            return super().compress(module, hidden_states, keys, values, attentions, kwargs)

        is_chunk_prefill = bool(kwargs.get("kvpress_chunk_prefill", False))

        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        if self.layer_running_mean:
            if getattr(module, "layer_idx", 0) == 0:
                self._running_layer_scores = None
            layer_idx = int(getattr(module, "layer_idx", 0))
            contrib = scores / float(layer_idx + 1)
            if self._running_layer_scores is None or self._running_layer_scores.shape != scores.shape:
                self._running_layer_scores = contrib.detach()
            else:
                self._running_layer_scores = (self._running_layer_scores + contrib).detach()
            scores = self._running_layer_scores

        if not is_chunk_prefill:
            keys, values, _ = self._compress_block_from_scores(module, keys, values, scores)
            return keys, values

        chunk_idx = int(kwargs.get("kvpress_chunk_idx", 0))
        num_chunks = int(kwargs.get("kvpress_num_chunks", 1))
        effective_cr = self._chunk_prefill_compression_ratio(chunk_idx, num_chunks)

        keys, values, _ = self._compress_block_from_scores(
            module,
            keys,
            values,
            scores,
            compression_ratio=effective_cr,
        )
        return keys, values

    def _score_block_chunked_finalize(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        kwargs: dict,
    ):
        """
        After chunk prefill: full-sequence hidden_states (cat of segments) + current KV; compute per-key
        scores for block-mode compression without building one giant (q_len × k_len) tensor at once.

        Returns:
            scores: (bsz, n_kv_heads, k_len)
        """
        bsz, k_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = values.dtype

        # Accumulate sum over query positions; divide once at the end => mean over chosen q window.
        token_score_sum = torch.zeros((bsz, k_len), device=device, dtype=torch.float32)

        # Same q-window rule as score() block path: short seq => all queries; long seq => last_n_query tail.
        q_len = hidden_states.size(1)
        if q_len <= 64:
            q_start = 0
        else:
            if self.last_n_query is None:
                q_start = 0
            else:
                q_start = max(0, q_len - int(self.last_n_query))

        q_end = q_len
        q_effective_len = q_end - q_start
        if q_effective_len <= 0:
            raise ValueError(f"Invalid q window in finalize: q_start={q_start}, q_end={q_end}")

        # Outer: q slices; inner: k slices — keeps peak memory bounded vs full matmul.
        q_chunk_size = 256
        for qs in range(q_start, q_end, q_chunk_size):
            qe = min(qs + q_chunk_size, q_end)

            for k_start, k_end, chunk_score in self.indexer_logits_chunks_with_ranges_qchunk(
                module,
                hidden_states,
                kwargs,
                chunk_size=self.chunk_size,
                q_start=qs,
                q_end=qe,
            ):
                # Sum over queries in this q-slice; add into per-key totals for [k_start:k_end).
                token_score_sum[:, k_start:k_end] += chunk_score.float().sum(dim=1)

        token_scores = token_score_sum / float(q_effective_len)

        # Merge scores within each non-overlapping key block (block eviction).
        token_scores = self._blockify_key_scores(token_scores, k_len=k_len)

        # score() returns one row per KV head; expand from (bsz, k_len).
        scores = token_scores.unsqueeze(1).expand(-1, keys.size(1), -1)

        if self.use_pooling:
            kernel_size = int(getattr(self, "pooling_kernel_size", 5))
            scores = F.avg_pool1d(scores, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)

        if self.use_vnorm:
            vnorm = values.norm(dim=-1)
            scores = (scores + 1e-6) * vnorm

        return scores.to(dtype)

    def finalize_chunk_prefill(self, model: nn.Module, cache):
        """Run once after all prefill chunks: cat deferred hiddens, score, then block-compress KV (query_reduce=block only)."""
        if self.compression_ratio == 0:
            self._reset_chunk_prefill_buffers()
            return cache

        if self._prefill_hidden_states is None:
            return cache

        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        mode = (self.query_reduce or "auto").lower()

        for layer_idx, layer in enumerate(language_model.layers):
            attn = layer.self_attn
            cache_layer = cache.layers[layer_idx]
            keys = cache_layer.keys
            values = cache_layer.values

            if keys is None or values is None:
                continue
            if layer_idx not in self._prefill_hidden_states:
                continue

            # Reconstruct full prefill hidden sequence for this layer (order matches KV append order).
            hidden_states = torch.cat(self._prefill_hidden_states[layer_idx], dim=1)

            if hidden_states.size(1) != keys.size(2):
                raise ValueError(
                    f"[finalize_chunk_prefill] layer {layer_idx}: hidden_states len {hidden_states.size(1)} != KV len {keys.size(2)}"
                )

            # Non-block modes: no deferred finalize path here (would need equivalent one-shot score + compress).
            if mode != "block":
                continue

            scores = self._score_block_chunked_finalize(
                attn,
                hidden_states,
                keys,
                values,
                kwargs={},
            )

            if self.layer_running_mean:
                # Intentionally skip layer_running_mean at finalize (no partial per-chunk state to extend).
                pass

            # In chunk-prefill deferred finalize we must keep KV lengths identical across layers,
            # otherwise HF cache position semantics + our `_actual_cache_length()` consistency
            # checks will fail. Force exact token count avoids the "tail partial block" drift.
            new_keys, new_values, _ = self._compress_block_from_scores(
                attn, keys, values, scores, force_exact_token_count=True
            )
            cache_layer.keys = new_keys
            cache_layer.values = new_values
            # Keep HF cache metadata consistent with truncated tensors.
            if hasattr(cache_layer, "cumulative_length"):
                cache_layer.cumulative_length = new_keys.shape[2]
            if hasattr(cache_layer, "seen_tokens"):
                cache_layer.seen_tokens = new_keys.shape[2]
            if hasattr(cache_layer, "_seen_tokens"):
                cache_layer._seen_tokens = new_keys.shape[2]

        self._reset_chunk_prefill_buffers()
        return cache


def load_model_with_query_indexer_press(model_path, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}

    meta_path = os.path.join(model_path, "indexer_only_meta.json")
    weights_path = os.path.join(model_path, "indexer_weights.pt")
    indexer_only = os.path.isfile(meta_path) and os.path.isfile(weights_path)

    if not indexer_only:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        base = meta.get("base_model_name_or_path")
        if not base:
            raise ValueError(f"{meta_path} must contain 'base_model_name_or_path'")
        model = AutoModelForCausalLM.from_pretrained(base, **model_kwargs)
        tok_dir = model_path if os.path.isfile(os.path.join(model_path, "tokenizer_config.json")) else base
        tokenizer = AutoTokenizer.from_pretrained(tok_dir)

    # Support both single-file and sharded HF checkpoints.
    # Many HuggingFace checkpoints are saved as:
    # - pytorch_model.bin.index.json + pytorch_model-0000x-of-0000y.bin
    # rather than a single pytorch_model.bin.
    checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
    index_path = os.path.join(model_path, "pytorch_model.bin.index.json")

    def _infer_indexer_spec_from_state_dict(state_dict: dict) -> dict | None:
        """
        Infer (dim, n_heads, head_dim, enable_low_rank) from indexer tensors.
        - w_k.weight: (head_dim, dim)
        - weights_proj.weight: (n_heads, dim)
        - w_q_proj.weight exists => enable_low_rank
        """
        w_k = None
        w_proj = None
        enable_low_rank = False
        for k, v in state_dict.items():
            if "indexer.w_k.weight" in k and hasattr(v, "shape") and len(v.shape) == 2:
                w_k = v
            elif "indexer.weights_proj.weight" in k and hasattr(v, "shape") and len(v.shape) == 2:
                w_proj = v
            elif "indexer.w_q_proj.weight" in k:
                enable_low_rank = True
        if w_k is None or w_proj is None:
            return None
        head_dim = int(w_k.shape[0])
        dim = int(w_k.shape[1])
        n_heads = int(w_proj.shape[0])
        return {"dim": dim, "n_heads": n_heads, "head_dim": head_dim, "enable_low_rank": enable_low_rank}

    def _infer_indexer_spec_from_sharded_index(weight_map: dict) -> dict | None:
        """
        Infer indexer spec by loading only the shard(s) that contain w_k/weights_proj.
        """
        # Find representative keys and their shards.
        w_k_key = None
        w_k_shard = None
        w_proj_key = None
        w_proj_shard = None
        enable_low_rank = any("indexer.w_q_proj.weight" in k for k in weight_map.keys())
        for k, sf in weight_map.items():
            if w_k_key is None and "indexer.w_k.weight" in k:
                w_k_key, w_k_shard = k, sf
            if w_proj_key is None and "indexer.weights_proj.weight" in k:
                w_proj_key, w_proj_shard = k, sf
            if w_k_key is not None and w_proj_key is not None:
                break
        if w_k_key is None or w_proj_key is None:
            return None

        def _load_one(shard_file: str) -> dict:
            shard_path = os.path.join(model_path, shard_file)
            if not os.path.exists(shard_path):
                return {}
            return torch.load(shard_path, map_location="cpu")

        sd = {}
        if w_k_shard is not None:
            sd.update(_load_one(w_k_shard))
        if w_proj_shard is not None and w_proj_shard != w_k_shard:
            sd.update(_load_one(w_proj_shard))

        spec = _infer_indexer_spec_from_state_dict(sd)
        if spec is None:
            return None
        spec["enable_low_rank"] = bool(enable_low_rank)
        return spec

    def _ensure_indexer_modules(spec: dict | None):
        """
        Ensure each layer has a QueryIndexer module under scorer_attr.
        If `spec` is provided, initialize with the checkpoint-trained shapes.
        """
        dummy_press = QueryIndexerScorePress(compression_ratio=0.0)

        if spec is None:
            # Fallback to existing behavior (heuristic shapes).
            dummy_press.post_init_from_model(model, force_reinit=False)
            return

        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        scorer_attr = dummy_press.scorer_attr

        # device_map="auto" places each layer on a possibly different GPU; indexer must match
        # that layer's self_attn, not next(model.parameters()) (often cuda:0 only).
        # If an indexer already exists but with mismatched shape, re-init.
        first_attn = language_model.layers[0].self_attn
        existing = getattr(first_attn, scorer_attr, None)
        need_reinit = True
        if existing is not None and hasattr(existing, "w_k") and hasattr(existing.w_k, "weight"):
            try:
                need_reinit = tuple(existing.w_k.weight.shape) != (spec["head_dim"], spec["dim"])
            except Exception:
                need_reinit = True

        for layer in language_model.layers:
            attn = layer.self_attn
            layer_device = next(attn.parameters()).device
            layer_dtype = next(attn.parameters()).dtype
            if hasattr(attn, scorer_attr):
                if not need_reinit:
                    continue
                # Replace existing module safely.
                if scorer_attr in attn._modules:
                    del attn._modules[scorer_attr]
            args = SimpleNamespace(
                dim=int(spec["dim"]),
                n_heads=int(spec["n_heads"]),
                head_dim=int(spec["head_dim"]),
                enable_low_rank=bool(spec["enable_low_rank"]),
                max_batch_size=dummy_press.max_batch_size,
                max_seq_len=dummy_press.max_seq_len,
                activation=dummy_press.activation,
            )
            indexer = QueryIndexer(model.config, args).to(device=layer_device, dtype=layer_dtype)
            attn.register_module(scorer_attr, indexer)

    loaded = False

    # Case 0: indexer-only bundle (FSDP default training save)
    if indexer_only:
        try:
            pack = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            pack = torch.load(weights_path, map_location="cpu")
        if isinstance(pack, dict) and "state_dict" in pack:
            state_dict = pack["state_dict"]
        else:
            state_dict = pack
        spec = _infer_indexer_spec_from_state_dict(state_dict)
        _ensure_indexer_modules(spec)
        incompatible = model.load_state_dict(state_dict, strict=False)
        logger.info(
            "Loaded indexer-only checkpoint: %s keys (missing=%s, unexpected=%s)",
            len(state_dict),
            len(getattr(incompatible, "missing_keys", []) or []),
            len(getattr(incompatible, "unexpected_keys", []) or []),
        )
        print("✓ Loaded trained QueryIndexerScorePress weights from indexer-only bundle", flush=True)
        return model, tokenizer

    # Case 1: sharded .bin checkpoint
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        indexer_keys = [k for k in weight_map.keys() if "indexer" in k]

        if indexer_keys:
            spec = _infer_indexer_spec_from_sharded_index(weight_map)
            _ensure_indexer_modules(spec)

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
                    "Loaded QueryIndexerScorePress weights from sharded checkpoint: %s keys (missing=%s, unexpected=%s)",
                    len(indexer_state),
                    len(getattr(incompatible, "missing_keys", []) or []),
                    len(getattr(incompatible, "unexpected_keys", []) or []),
                )
                print("✓ Loaded trained QueryIndexerScorePress weights from sharded checkpoint", flush=True)
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
            spec = _infer_indexer_spec_from_state_dict(state_dict)
            _ensure_indexer_modules(spec)
            incompatible = model.load_state_dict(state_dict, strict=False)
            logger.info(
                "Loaded QueryIndexerScorePress weights from single-file checkpoint: missing=%s, unexpected=%s",
                len(getattr(incompatible, "missing_keys", []) or []),
                len(getattr(incompatible, "unexpected_keys", []) or []),
            )
            print("✓ Loaded trained QueryIndexerScorePress weights from checkpoint", flush=True)
            loaded = True

        del state_dict
    
    return model, tokenizer


@dataclass
class QueryIndexerDecodingPress(DecodePress):
    """
    Decode-time KV eviction driven by QueryIndexer scores computed from the
    current hidden state buffer.

    This press maintains the QueryIndexer caches (keys) and, at each
    compression interval, uses the buffered decode hidden states as queries
    (pre-RoPE via `get_prerope_query_states`) to score all cache tokens
    (prefill + decode). Tokens with the lowest scores are evicted globally.
    """

    base_press: QueryIndexerScorePress
    compression_interval: int = 50
    target_size: int = 2048
    hidden_states_buffer_size: int = 256

    def __post_init__(self):
        assert isinstance(self.base_press, QueryIndexerScorePress)
        self.hidden_states_buffer = defaultdict(list)
        self.layer_step_counts = defaultdict(int)
        self.prefill_cache_size = {}

    def post_init_from_model(self, model):
        self.base_press.post_init_from_model(model)

    def _maybe_get_indexer(self, module: nn.Module):
        scorer_attr = getattr(self.base_press, "scorer_attr", None)
        if not scorer_attr:
            return None
        return getattr(module, scorer_attr, None)

    def _update_indexer_cache(self, module: nn.Module, hidden_states: torch.Tensor):
        indexer = self._maybe_get_indexer(module)
        if indexer is None:
            return
        with torch.no_grad():
            _ = indexer.forward_cache(hidden_states)

    def _compress_core(self, scores, keys, values, n_kept, head_dim):
        indices = scores.topk(n_kept, dim=-1).indices  # (bsz, n_heads, n_kept)
        sorted_indices = indices.sort(dim=-1).values
        gather_idx = sorted_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        compressed_keys = keys.gather(2, gather_idx).contiguous()
        compressed_values = values.gather(2, gather_idx).contiguous()
        return compressed_keys, compressed_values, sorted_indices

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        layer_idx = module.layer_idx
        indexer = self._maybe_get_indexer(module)
        if indexer is None:
            logger.warning("QueryIndexerDecodingPress: layer %s has no indexer; skipping.", layer_idx)
            return keys, values

        # Always keep the indexer cache in sync with decode-time KV growth.
        # NOTE: `forward_hook` may call `compress()` periodically even when we don't evict
        # (e.g., `compression_interval` reached but `total_len <= target_size`). If we
        # return early without updating `indexer.k_cache`, it will lag behind the actual
        # KV cache length and later eviction will be skipped due to length mismatch.
        self._update_indexer_cache(module, hidden_states)

        total_len = keys.shape[2]
        if total_len <= self.target_size:
            return keys, values

        cache_len = indexer.k_cache.shape[1]
        if cache_len != total_len:
            logger.warning(
                "QueryIndexerDecodingPress: layer %s indexer cache len (%s) != KV len (%s); skipping compression.",
                layer_idx,
                cache_len,
                total_len,
            )
            return keys, values

        query_states = get_prerope_query_states(module, hidden_states)
        kwargs["query_states"] = query_states
        kwargs["is_decoding"] = True
        scores = self.base_press.score(module, hidden_states, keys, values, attentions, kwargs)
        if scores is None:
            return keys, values

        n_kept = min(self.target_size, total_len)
        compressed_keys, compressed_values, sorted_indices = self._compress_core(
            scores, keys, values, n_kept, module.head_dim
        )

        token_indices = sorted_indices[:, 0, :]  # use head 0 for indexer cache alignment
        indexer.compress_cache_by_indices(token_indices)

        return compressed_keys, compressed_values

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        q_len = hidden_states.shape[1]
        layer_idx = module.layer_idx

        # Prefill: record prefill size and populate indexer cache once.
        if kwargs["cache_position"][-1] <= q_len:
            if q_len > 1:
                self.prefill_cache_size[layer_idx] = cache.get_seq_length(layer_idx)
                self._update_indexer_cache(module, hidden_states)
            return output

        self.hidden_states_buffer[layer_idx].append(hidden_states.detach().clone())
        self.layer_step_counts[layer_idx] += 1

        current_seq_len = cache.get_seq_length(layer_idx)
        should_compress = (
            self.layer_step_counts[layer_idx] >= self.compression_interval
            or current_seq_len >= self.target_size
        )

        if should_compress:
            cache_layer = cache.layers[layer_idx]
            keys, values = extract_keys_and_values(cache, layer_idx)
            attentions = output[1] if len(output) > 1 and output[1] is not None else None

            if len(self.hidden_states_buffer[layer_idx]) > 1:
                buffered_hidden_states = torch.cat(self.hidden_states_buffer[layer_idx], dim=1)
            else:
                buffered_hidden_states = self.hidden_states_buffer[layer_idx][0]

            kwargs["is_decoding"] = True
            keys, values = self.compress(module, buffered_hidden_states, keys, values, attentions, kwargs)

            if isinstance(cache, QuantizedCache):
                cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
                cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
                cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)
                cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)
                cache_layer.cumulative_length = keys.shape[2]
            else:
                cache_layer.keys = keys
                cache_layer.values = values

            self.layer_step_counts[layer_idx] = 0
            self.hidden_states_buffer[layer_idx] = []

        if self.hidden_states_buffer_size > 0 and len(self.hidden_states_buffer[layer_idx]) > self.hidden_states_buffer_size:
            self.hidden_states_buffer[layer_idx] = self.hidden_states_buffer[layer_idx][-self.hidden_states_buffer_size :]

        return output

    def reset(self):
        self.hidden_states_buffer = defaultdict(list)
        self.layer_step_counts = defaultdict(int)
        self.prefill_cache_size = {}