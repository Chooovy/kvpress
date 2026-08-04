from __future__ import annotations

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

    def _select_query_window_mask(self, qk_mask: torch.Tensor) -> torch.Tensor:
        """Apply the same last_n_query slicing to a (bsz, q_len, k_len) boolean mask."""
        if self.last_n_query is None:
            return qk_mask
        q_len = qk_mask.size(1)
        n = min(int(self.last_n_query), q_len)
        return qk_mask[:, -n:, :]

    @staticmethod
    def _masked_mean_over_queries(scores_qk: torch.Tensor, qk_valid: torch.Tensor) -> torch.Tensor:
        """
        Compute mean over query dimension while excluding invalid (q,k) pairs.
        scores_qk: (bsz, q_len, k_len)
        qk_valid:  (bsz, q_len, k_len) bool
        returns:   (bsz, k_len)
        """
        s = scores_qk.float().masked_fill(~qk_valid, 0.0).sum(dim=1)
        denom = qk_valid.sum(dim=1)  # (bsz, k_len)
        out = s / denom.clamp(min=1).to(s.dtype)
        # If a key is never valid (e.g., fully padded), force it to -inf so it won't be selected.
        out = out.masked_fill(denom == 0, float("-inf"))
        return out.to(scores_qk.dtype)

    def _build_qk_valid_mask(self, attention_mask: Optional[torch.Tensor], q_len: int, k_len: int) -> Optional[torch.Tensor]:
        """
        Build a boolean (bsz, q_len, k_len) mask indicating which (query,key) pairs are valid.

        NOTE: This is used for query-reduction (mean/recency/gating). We should not add
        huge negative values into raw scores and then average, because that biases keys
        purely due to causality (future keys masked for earlier queries).
        """
        if attention_mask is None:
            return None
        if attention_mask.dim() == 4:
            m = attention_mask[:, 0, -q_len:, -k_len:]  # (bsz, q_len, k_len)
            valid_k = (m == 0).any(dim=1)  # (bsz, k_len)
            return valid_k.unsqueeze(1).expand(-1, q_len, -1)
        if attention_mask.dim() == 2:
            # Padding-only mask: allow all queries to attend non-pad keys.
            keep_k = attention_mask[:, -k_len:] > 0
            return keep_k.unsqueeze(1).expand(-1, q_len, -1)
        return None

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
            indexer = QueryIndexer(model.config, args).to(model.device, dtype=model.dtype)
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
        Q-chunked variant for memory efficiency.
        Yields (k_start, k_end, chunk_score_q) where chunk_score_q is (bsz, q_chunk, t_chunk).
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

        # mask = self._prepare_mask(kwargs.get("attention_mask"), q_len=q_len, k_len=k_len, device=hidden_states.device)
        freqs = self._prepare_freqs_cis(kwargs, head_dim=getattr(indexer, "head_dim", None))
        qk_valid = self._build_qk_valid_mask(kwargs.get("attention_mask"), q_len=q_len, k_len=k_len, device=hidden_states.device)

        if is_decoding:
            indexer_scores = indexer.get_cache_score(hidden_states, query_states, freqs)
        else:
            indexer_scores = indexer.forward(hidden_states, query_states, freqs, mask=None, use_cache=False)
        
        # Reduce (bsz, q_len, k_len) -> (bsz, k_len)
        scores_qk = self._select_query_window(indexer_scores)
        qk_valid_w = self._select_query_window_mask(qk_valid) if qk_valid is not None else None
        mode = (self.query_reduce or "auto").lower()
        if mode == "auto":
            if q_len == 1:
                token_scores = indexer_scores.squeeze(1)
            elif q_len <= 64:
                token_scores = indexer_scores.mean(dim=1) if qk_valid is None else self._masked_mean_over_queries(indexer_scores, qk_valid)
            else:
                if scores_qk.size(1) > 1:
                    token_scores = scores_qk.mean(dim=1) if qk_valid_w is None else self._masked_mean_over_queries(scores_qk, qk_valid_w)
                else:
                    token_scores = scores_qk.squeeze(1)
        elif mode == "question":
            question_len = int(getattr(self, "question_len", 0) or 0)
            if question_len <= 0:
                raise ValueError("question_len is not set")
            else:
                question_len = min(question_len, int(indexer_scores.size(1)))
                w = indexer_scores[:, -question_len:, :]
                if qk_valid is None:
                    token_scores = w.mean(dim=1)
                else:
                    qv = qk_valid[:, -question_len:, :]
                    token_scores = self._masked_mean_over_queries(w, qv)
        elif mode in ("recency", "recency_weighted", "recency-weighted"):
            if qk_valid_w is not None:
                scores_qk = scores_qk.masked_fill(~qk_valid_w, float("-inf"))
            token_scores = self._reduce_recency_weighted(scores_qk)
        elif mode in ("top_p", "top_p_gating", "top-p", "top-p-gating"):
            if qk_valid_w is not None:
                scores_qk = scores_qk.masked_fill(~qk_valid_w, float("-inf"))
            token_scores = self._reduce_top_p_gating(scores_qk)
        elif mode == "block":
            # Base query aggregation: mimic "auto" behavior, then blockify on the key axis.
            if q_len == 1:
                token_scores = indexer_scores.squeeze(1)
            elif q_len <= 64:
                token_scores = indexer_scores.mean(dim=1) if qk_valid is None else self._masked_mean_over_queries(indexer_scores, qk_valid)
            else:
                if scores_qk.size(1) > 1:
                    token_scores = scores_qk.mean(dim=1) if qk_valid_w is None else self._masked_mean_over_queries(scores_qk, qk_valid_w)
                else:
                    token_scores = scores_qk.squeeze(1)
            token_scores = self._blockify_key_scores(token_scores, k_len=k_len)
        else:
            raise ValueError(f"Unknown query_reduce={self.query_reduce!r}.")

        token_scores = token_scores[:, self.n_sink :]

        scores = token_scores.unsqueeze(1).expand(-1, keys.size(1), -1)
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

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Override ScorerPress.compress to support `query_reduce="block"`: select whole key blocks.
        """
        if self.compression_ratio == 0:
            return keys, values

        mode = (self.query_reduce or "auto").lower()
        if mode != "block":
            return super().compress(module, hidden_states, keys, values, attentions, kwargs)

        # Compute per-token scores (already blockified) shaped (B, H_kv, K).
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        # Keep running-mean behavior consistent with ScorerPress.
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

        # Block selection is shared across heads to keep the cache consistent.
        token_scores = scores.mean(dim=1)  # (B, K)
        bsz, k_len = token_scores.shape
        bs = int(max(1, getattr(self, "block_size", 1)))
        if k_len <= self.n_sink or bs <= 1:
            # Degenerate case: fall back to default token-wise selection.
            return super().compress(module, hidden_states, keys, values, attentions, kwargs)

        # Determine how many tokens we'd like to keep, then map to whole blocks.
        n_kept = max(1, int(k_len * (1 - self.compression_ratio)))
        n_kept_nonsink = max(0, n_kept - int(self.n_sink))
        nonsink_len = k_len - int(self.n_sink)
        n_blocks = int(math.ceil(nonsink_len / float(bs)))
        n_blocks_keep = int(math.ceil(n_kept_nonsink / float(bs))) if n_kept_nonsink > 0 else 0
        n_blocks_keep = min(n_blocks_keep, n_blocks)

        # If we only keep sink tokens, just truncate to sink (preserving order).
        if n_blocks_keep == 0:
            keep_positions = torch.arange(int(self.n_sink), device=keys.device).view(1, -1).expand(bsz, -1)
        else:
            # Compute block scores by averaging token scores within each block.
            # token_scores here are already constant within each block, but we keep the logic explicit.
            block_scores = []
            for bi in range(n_blocks):
                s = int(self.n_sink) + bi * bs
                e = min(s + bs, k_len)
                block_scores.append(token_scores[:, s:e].mean(dim=1))
            block_scores = torch.stack(block_scores, dim=1)  # (B, n_blocks)

            kept_blocks = block_scores.topk(n_blocks_keep, dim=1).indices  # (B, n_blocks_keep)

            # Build sorted (chronological) token indices per sample.
            keep_positions_list = []
            for b in range(bsz):
                pos = list(range(int(self.n_sink)))
                for bi in sorted(kept_blocks[b].tolist()):
                    s = int(self.n_sink) + bi * bs
                    e = min(s + bs, k_len)
                    pos.extend(range(s, e))
                keep_positions_list.append(torch.tensor(pos, device=keys.device, dtype=torch.long))
            # Stack to (B, K_keep)
            keep_positions = torch.stack(keep_positions_list, dim=0)

        # Expand indices to gather K/V: (B, H_kv, K_keep, D)
        gather_idx = keep_positions[:, None, :, None].expand(-1, keys.size(1), -1, module.head_dim)
        keys = keys.gather(2, gather_idx).contiguous()
        values = values.gather(2, gather_idx).contiguous()
        return keys, values




def load_model_with_query_indexer_press(model_path, model_kwargs=None):
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
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

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
            indexer = QueryIndexer(model.config, args).to(device=device, dtype=dtype)
            attn.register_module(scorer_attr, indexer)

    loaded = False

    # Case 1: sharded .bin checkpoint
    if os.path.exists(index_path):
        import json

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
        # `forward_hook` may invoke `compress()` periodically (e.g. by interval) even when
        # `total_len <= target_size`. If we return early without updating `indexer.k_cache`,
        # the cache will lag and future eviction attempts will be skipped due to length mismatch.
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