# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from torch import nn
from transformers.models.llama.modeling_llama import rotate_half

from kvpress.presses.base_press import BasePress
from kvpress.utils import get_prerope_key_states, get_prerope_query_states


@dataclass
class ChunkScorerPress(BasePress, ABC):
    """
    Base class for last-query, whole-chunk KV cache eviction.

    A scoring query ranks complete remote chunks. By default, the most recent
    post-RoPE query scores cached post-RoPE keys. A recent protected tail,
    including any final partial chunk, is always retained.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Requested fraction of key-value pairs to remove.
    chunk_size : int, default=64
        Number of consecutive tokens in each selectable remote chunk.
    protected_window_size : int, default=512
        Minimum number of recent tokens to retain without scoring. Set to 0 for
        no explicit protected window; a final partial chunk is still retained.
    """

    compression_ratio: float = 0.0
    chunk_size: int = 64
    protected_window_size: int = 512

    last_input_tokens: int | None = field(init=False, default=None, repr=False, compare=False)
    last_kept_tokens: int | None = field(init=False, default=None, repr=False, compare=False)
    last_actual_compression_ratio: float | None = field(init=False, default=None, repr=False, compare=False)
    last_protected_tokens: int | None = field(init=False, default=None, repr=False, compare=False)
    last_kept_remote_chunks: int | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self):
        self._validate_config()

    def _validate_config(self):
        if not 0.0 <= self.compression_ratio < 1.0:
            raise ValueError(f"compression_ratio must be in [0, 1), got {self.compression_ratio}")
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.protected_window_size < 0:
            raise ValueError(f"protected_window_size must be non-negative, got {self.protected_window_size}")

    def reset_runtime_stats(self):
        """Clear per-prefill statistics before starting a new context."""
        self.last_input_tokens = None
        self.last_kept_tokens = None
        self.last_actual_compression_ratio = None
        self.last_protected_tokens = None
        self.last_kept_remote_chunks = None

    def _record_runtime_stats(
        self,
        input_tokens: int,
        kept_tokens: int,
        protected_tokens: int,
        kept_remote_chunks: int,
    ):
        self.last_input_tokens = int(input_tokens)
        self.last_kept_tokens = int(kept_tokens)
        self.last_actual_compression_ratio = 1.0 - kept_tokens / input_tokens
        self.last_protected_tokens = int(protected_tokens)
        self.last_kept_remote_chunks = int(kept_remote_chunks)

    @staticmethod
    def _validate_unpadded_attention_mask(attention_mask: torch.Tensor | None, kv_len: int):
        """Fail closed when the last query cannot attend every cached position."""
        if attention_mask is None:
            return

        if attention_mask.shape[-1] < kv_len:
            raise ValueError(
                "Last-query chunk presses require an attention mask covering the full KV cache: "
                f"mask_length={attention_mask.shape[-1]}, input_tokens={kv_len}."
            )

        if attention_mask.ndim == 2:
            last_query_mask = attention_mask[..., :kv_len]
            invalid = ~last_query_mask.bool()
        elif attention_mask.ndim in (3, 4):
            last_query_mask = attention_mask[..., -1, :kv_len]
            if last_query_mask.dtype == torch.bool:
                invalid = ~last_query_mask
            else:
                invalid = last_query_mask != 0
        else:
            raise ValueError(
                "Last-query chunk presses only support 2D, 3D, or 4D attention masks, " f"got {attention_mask.ndim}D."
            )

        if torch.any(invalid):
            raise ValueError(
                "Last-query chunk presses currently support only unpadded, non-packed text prefill. "
                "The last query is masked from at least one cached token."
            )

    @staticmethod
    def _validate_attention_contract(module: nn.Module, kv_len: int):
        config = module.config
        attention_softcap = getattr(config, "attn_logit_softcapping", None)
        if attention_softcap not in (None, 0, False):
            raise ValueError(
                "Last-query chunk presses do not yet support attention logit softcapping: "
                f"attn_logit_softcapping={attention_softcap}."
            )

        sliding_window = getattr(config, "sliding_window", None)
        if hasattr(module, "is_sliding"):
            uses_sliding_window = bool(module.is_sliding)
        else:
            uses_sliding_window = bool(getattr(config, "use_sliding_window", sliding_window is not None))
        if uses_sliding_window and sliding_window is not None and kv_len > sliding_window:
            raise ValueError(
                "Last-query chunk presses do not score tokens masked by sliding-window attention: "
                f"input_tokens={kv_len}, sliding_window={sliding_window}."
            )

    @staticmethod
    def _group_query_heads(query_states: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        batch_size, num_query_heads, head_dim = query_states.shape
        key_batch_size, num_kv_heads, _, key_head_dim = keys.shape

        if batch_size != key_batch_size or head_dim != key_head_dim:
            raise ValueError(
                "Query and key shapes are incompatible: "
                f"query={tuple(query_states.shape)}, keys={tuple(keys.shape)}."
            )
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                "The number of query heads must be divisible by the number of KV heads: "
                f"num_query_heads={num_query_heads}, num_kv_heads={num_kv_heads}."
            )

        group_size = num_query_heads // num_kv_heads
        return query_states.reshape(batch_size, num_kv_heads, group_size, head_dim)

    @staticmethod
    def _normalize_and_reduce_gqa(
        remote_log_mass_proxy: torch.Tensor,
        local_log_mass: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalize each query head with both remote and protected mass, then
        protect a chunk when any query head sharing the KV head needs it.
        """
        remote_log_mass = torch.logsumexp(remote_log_mass_proxy, dim=-1)
        total_log_mass = torch.logaddexp(local_log_mass, remote_log_mass)
        normalized_scores = remote_log_mass_proxy - total_log_mass.unsqueeze(-1)
        return normalized_scores.max(dim=2).values

    @abstractmethod
    def _compute_remote_log_mass_proxy(
        self,
        query_states: torch.Tensor,
        remote_key_chunks: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """
        Return remote chunk scores shaped [batch, KV heads, query groups, chunks].

        Scores must be in log-mass units so they can be normalized jointly with
        the exact protected-tail log mass.
        """
        raise NotImplementedError

    def _get_postrope_last_query(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        query_states = get_prerope_query_states(module, hidden_states[:, -1:])
        cos, sin = position_embeddings
        cos_last = cos[:, -1:, :].unsqueeze(1)
        sin_last = sin[:, -1:, :].unsqueeze(1)
        query_states = query_states * cos_last + rotate_half(query_states) * sin_last
        return query_states.squeeze(2)

    def _get_scoring_query_and_keys(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the query and keys used only for chunk scoring."""
        return self._get_postrope_last_query(module, hidden_states, position_embeddings), keys

    def _expand_chunk_indices(
        self,
        chunk_indices: torch.Tensor,
        candidate_end: int,
        kv_len: int,
    ) -> torch.Tensor:
        batch_size, num_kv_heads, _ = chunk_indices.shape
        offsets = torch.arange(self.chunk_size, device=chunk_indices.device).view(1, 1, 1, -1)
        remote_token_indices = (chunk_indices.unsqueeze(-1) * self.chunk_size + offsets).flatten(-2)

        protected_indices = torch.arange(candidate_end, kv_len, device=chunk_indices.device).view(1, 1, -1)
        protected_indices = protected_indices.expand(batch_size, num_kv_heads, -1)
        return torch.cat((remote_token_indices, protected_indices), dim=-1)

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del attentions
        self.reset_runtime_stats()
        self._validate_config()

        if keys.ndim != 4 or values.shape != keys.shape:
            raise ValueError(
                "Expected matching key/value tensors shaped [batch, KV heads, tokens, head_dim], "
                f"got keys={tuple(keys.shape)}, values={tuple(values.shape)}."
            )

        batch_size, num_kv_heads, kv_len, head_dim = keys.shape
        if kv_len == 0:
            raise ValueError("Last-query chunk presses cannot compress an empty KV cache.")
        if hidden_states.shape[0] != batch_size or hidden_states.shape[1] != kv_len:
            raise ValueError(
                "Last-query chunk presses currently require a single unpadded prefill whose hidden-state "
                f"length matches the KV cache: hidden_states={tuple(hidden_states.shape)}, input_tokens={kv_len}."
            )

        candidate_end = max(0, kv_len - self.protected_window_size)
        candidate_end = candidate_end // self.chunk_size * self.chunk_size
        protected_tokens = kv_len - candidate_end
        num_remote_chunks = candidate_end // self.chunk_size

        if self.compression_ratio == 0.0:
            self._record_runtime_stats(kv_len, kv_len, protected_tokens, num_remote_chunks)
            return keys, values

        target_kept_tokens = int(kv_len * (1.0 - self.compression_ratio))
        maximum_feasible_compression_ratio = 1.0 - protected_tokens / kv_len

        if num_remote_chunks == 0 and target_kept_tokens < kv_len:
            raise ValueError(
                "No complete remote chunk can be evicted while preserving the protected region: "
                f"input_tokens={kv_len}, chunk_size={self.chunk_size}, "
                f"protected_window_size={self.protected_window_size}, "
                f"actual_protected_tokens={protected_tokens}, "
                f"requested_compression_ratio={self.compression_ratio:.6f}, "
                f"maximum_feasible_compression_ratio={maximum_feasible_compression_ratio:.6f}."
            )
        if target_kept_tokens < protected_tokens:
            raise ValueError(
                "Requested compression is infeasible while preserving the protected region: "
                f"input_tokens={kv_len}, chunk_size={self.chunk_size}, "
                f"protected_window_size={self.protected_window_size}, "
                f"actual_protected_tokens={protected_tokens}, "
                f"requested_compression_ratio={self.compression_ratio:.6f}, "
                f"maximum_feasible_compression_ratio={maximum_feasible_compression_ratio:.6f}."
            )

        self._validate_unpadded_attention_mask(kwargs.get("attention_mask"), kv_len)
        remote_token_budget = target_kept_tokens - protected_tokens
        num_kept_remote_chunks = min(num_remote_chunks, remote_token_budget // self.chunk_size)
        kept_tokens = protected_tokens + num_kept_remote_chunks * self.chunk_size

        if num_kept_remote_chunks == num_remote_chunks:
            self._record_runtime_stats(kv_len, kept_tokens, protected_tokens, num_kept_remote_chunks)
            return keys, values

        if num_kept_remote_chunks == 0:
            token_indices = torch.arange(candidate_end, kv_len, device=keys.device).view(1, 1, -1)
            token_indices = token_indices.expand(batch_size, num_kv_heads, -1)
        else:
            self._validate_attention_contract(module, kv_len)
            if "position_embeddings" not in kwargs:
                raise ValueError(
                    "Last-query chunk presses require position_embeddings to reconstruct the post-RoPE query."
                )

            query_states, scoring_keys = self._get_scoring_query_and_keys(
                module,
                hidden_states,
                keys,
                kwargs["position_embeddings"],
            )
            grouped_queries = self._group_query_heads(query_states, scoring_keys)
            scale = float(getattr(module, "scaling", head_dim**-0.5))

            remote_key_chunks = scoring_keys[:, :, :candidate_end, :].reshape(
                batch_size,
                num_kv_heads,
                num_remote_chunks,
                self.chunk_size,
                head_dim,
            )
            remote_log_mass_proxy = self._compute_remote_log_mass_proxy(
                grouped_queries,
                remote_key_chunks,
                scale,
            )

            local_keys = scoring_keys[:, :, candidate_end:, :]
            local_logits = (
                torch.einsum(
                    "bhgd,bhld->bhgl",
                    grouped_queries.float(),
                    local_keys.float(),
                )
                * scale
            )
            local_log_mass = torch.logsumexp(local_logits, dim=-1)
            chunk_scores = self._normalize_and_reduce_gqa(remote_log_mass_proxy, local_log_mass)

            expected_score_shape = (batch_size, num_kv_heads, num_remote_chunks)
            if chunk_scores.shape != expected_score_shape:
                raise ValueError(
                    f"Expected chunk scores shaped {expected_score_shape}, got {tuple(chunk_scores.shape)}."
                )

            kept_chunk_indices = chunk_scores.topk(num_kept_remote_chunks, dim=-1).indices
            kept_chunk_indices = kept_chunk_indices.sort(dim=-1).values
            token_indices = self._expand_chunk_indices(kept_chunk_indices, candidate_end, kv_len)

        gather_indices = token_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        compressed_keys = keys.gather(dim=2, index=gather_indices).contiguous()
        compressed_values = values.gather(dim=2, index=gather_indices).contiguous()

        self._record_runtime_stats(kv_len, kept_tokens, protected_tokens, num_kept_remote_chunks)
        return compressed_keys, compressed_values


@dataclass
class BSAPress(ChunkScorerPress):
    """
    Last-query BSA eviction using exact remote full-attention chunk mass.

    This is a KV eviction baseline, not the all-query NaiveBSA attention
    implementation from HiLS-Attention.
    """

    def _compute_remote_log_mass_proxy(
        self,
        query_states: torch.Tensor,
        remote_key_chunks: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        token_logits = (
            torch.einsum(
                "bhgd,bhnsd->bhgns",
                query_states.float(),
                remote_key_chunks.float(),
            )
            * scale
        )
        return torch.logsumexp(token_logits, dim=-1)


@dataclass
class MeanPoolingPress(ChunkScorerPress):
    """
    Last-query eviction using the dot product with each mean scoring key.

    Scoring uses the post-RoPE query and cached post-RoPE keys by default. The
    query and keys can independently use their pre-RoPE representations for
    controlled ablations. The mean logit is shifted by log(chunk_size) before
    local-inclusive GQA normalization.
    """

    use_prerope_query: bool = False
    use_prerope_keys: bool = False

    def _get_scoring_query_and_keys(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_prerope_query:
            query_states = get_prerope_query_states(module, hidden_states[:, -1:]).squeeze(2)
        else:
            query_states = self._get_postrope_last_query(module, hidden_states, position_embeddings)

        scoring_keys = get_prerope_key_states(module, hidden_states) if self.use_prerope_keys else keys
        return query_states, scoring_keys

    def _compute_remote_log_mass_proxy(
        self,
        query_states: torch.Tensor,
        remote_key_chunks: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        mean_keys = remote_key_chunks.mean(dim=-2, dtype=torch.float32)
        mean_logits = (
            torch.einsum(
                "bhgd,bhnd->bhgn",
                query_states.float(),
                mean_keys,
            )
            * scale
        )
        return mean_logits + math.log(self.chunk_size)
