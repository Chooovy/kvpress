# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass, field

import torch
from torch import nn
from transformers import QuantizedCache

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import extract_keys_and_values

logger = logging.getLogger(__name__)


@dataclass
class FixedLayerScoreEvictPress(BasePress):
    """
    Debug press: force ALL layers to evict/keep the same token positions, decided by scores from a single layer.

    This is useful to test the hypothesis "all layers should keep the same tokens".

    How it works
    ------------
    - During a prefill forward pass, when we reach `score_layer_idx`, we compute token indices to keep
      using `press.score(...)` on that layer.
    - We DO NOT compress any layer immediately.
    - At the LAST layer hook, we apply the same token indices to ALL cache layers.

    Notes
    -----
    - The token selection is shared across layers by token position only (same indices along seq_len).
    - If `press.mean_head=True`, selection is done with head-mean scores so all KV heads share indices.
    """

    press: ScorerPress
    score_layer_idx: int = 0

    _token_indices: torch.Tensor | None = field(default=None, init=False, repr=False)  # (B, Hkv, K)
    _num_hidden_layers: int | None = field(default=None, init=False, repr=False)
    _bad_layer_idx_warned: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "press must be a ScorerPress"
        assert self.score_layer_idx >= 0, "score_layer_idx must be >= 0"

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)
        # Prefer a stable "last layer" indicator from the model config.
        n = getattr(getattr(model, "config", None), "num_hidden_layers", None)
        if n is None and hasattr(getattr(model, "config", None), "text_config"):
            n = getattr(model.config.text_config, "num_hidden_layers", None)  # type: ignore[attr-defined]
        if n is None and hasattr(getattr(model, "config", None), "language_config"):
            n = getattr(model.config.language_config, "num_hidden_layers", None)  # type: ignore[attr-defined]
        if isinstance(n, int) and n > 0:
            self._num_hidden_layers = n
        else:
            self._num_hidden_layers = None

    @property  # type: ignore[misc]
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    def _compute_token_indices(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        """
        Compute token indices to KEEP based on `press.score(...)` at `score_layer_idx`.
        Returns indices with shape (B, num_kv_heads, n_kept).
        """
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)  # (B, Hkv, T)
        k_len = keys.shape[2]
        n_kept = max(1, int(k_len * (1 - float(self.press.compression_ratio))))

        selection_scores = scores
        if getattr(self.press, "mean_head", False):
            selection_scores = scores.mean(dim=1, keepdim=True)  # (B,1,T)
        indices = selection_scores.topk(n_kept, dim=-1).indices  # (B, Hkv or 1, K)
        if getattr(self.press, "mean_head", False):
            indices = indices.expand(-1, keys.shape[1], -1)  # (B, Hkv, K)
        # IMPORTANT: preserve the original token order (causal KV cache expects time order).
        # `topk` returns indices sorted by score; reorder indices by token position.
        indices = indices.sort(dim=-1).values
        return indices

    def _apply_token_indices_to_cache(self, cache, token_indices: torch.Tensor):
        """
        Apply token_indices (B,Hkv,K) to all layers in cache, in-place.
        """
        def _adapt_indices(idx_tok: torch.Tensor, k_len: int) -> torch.Tensor:
            """
            Ensure indices are within [0, k_len-1] for gather.
            If some indices are out of range (due to per-layer length mismatch), drop them and pad.
            """
            if k_len <= 0:
                return idx_tok[..., :0]
            # Fast path
            if int(idx_tok.max().item()) < k_len and int(idx_tok.min().item()) >= 0:
                return idx_tok

            # Drop invalid indices and pad to original K
            B, H, K = idx_tok.shape
            valid = (idx_tok >= 0) & (idx_tok < k_len)
            out = torch.empty_like(idx_tok)
            fallback = torch.zeros((B, H), device=idx_tok.device, dtype=idx_tok.dtype)
            for b in range(B):
                for h in range(H):
                    v = idx_tok[b, h][valid[b, h]]
                    if v.numel() == 0:
                        v = fallback[b, h].view(1)
                    if v.numel() >= K:
                        out[b, h] = v[:K]
                    else:
                        pad = v[-1].expand(K - v.numel())
                        out[b, h] = torch.cat([v, pad], dim=0)
            # Preserve causal order
            return out.sort(dim=-1).values

        num_layers = len(cache.layers)
        for layer_idx in range(num_layers):
            cache_layer = cache.layers[layer_idx]
            keys, values = extract_keys_and_values(cache, layer_idx)

            k_len = int(keys.shape[2])
            idx_tok = _adapt_indices(token_indices, k_len)

            # Expand to gather along head_dim
            head_dim = keys.shape[-1]
            idx = idx_tok.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            keys_new = keys.gather(2, idx).contiguous()
            values_new = values.gather(2, idx).contiguous()

            if isinstance(cache, QuantizedCache):
                cache_layer._quantized_keys = cache_layer._quantize(keys_new, axis=cache_layer.axis_key)
                cache_layer._quantized_values = cache_layer._quantize(values_new, axis=cache_layer.axis_value)
                cache_layer.keys = torch.zeros(0, dtype=keys_new.dtype, device=keys_new.device)  # type: ignore[index]
                cache_layer.values = torch.zeros(0, dtype=keys_new.dtype, device=keys_new.device)  # type: ignore[index]
                cache_layer.cumulative_length = keys_new.shape[2]
            else:
                cache_layer.keys = keys_new
                cache_layer.values = values_new
                # If the cache layer tracks cumulative length, keep it consistent.
                if hasattr(cache_layer, "cumulative_length"):
                    try:
                        cache_layer.cumulative_length = keys_new.shape[2]
                    except Exception:
                        pass

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        hidden_states = kwargs["hidden_states"]
        cache = kwargs.get("past_key_values", None)
        if cache is None:
            return output
        q_len = hidden_states.shape[1]

        # Don't compress after pre-filling (same guard as BasePress)
        cache_position = kwargs.get("cache_position", None)
        if cache_position is not None and cache_position[-1] > q_len:
            return output

        layer_idx = int(getattr(module, "layer_idx", 0))
        num_layers_so_far = len(cache.layers)
        # NOTE: during prefill forward, HF cache layers are created progressively.
        # So `len(cache.layers)` at layer L is typically `L+1`. We should NOT treat
        # `score_layer_idx >= len(cache.layers)` as "out of range" here; we just haven't reached it yet.

        # If we know the full number of layers and the user asked for an impossible layer, warn once.
        if self._num_hidden_layers is not None and self.score_layer_idx >= self._num_hidden_layers:
            if not self._bad_layer_idx_warned:
                logger.warning(
                    "FixedLayerScoreEvictPress: score_layer_idx=%s out of range (num_hidden_layers=%s). Disabling.",
                    self.score_layer_idx,
                    self._num_hidden_layers,
                )
                self._bad_layer_idx_warned = True
            return output

        # Reset per-forward state at start of stack
        if layer_idx == 0:
            self._token_indices = None

        # Compute token indices at the chosen layer
        if layer_idx == self.score_layer_idx:
            keys, values = extract_keys_and_values(cache, layer_idx)
            self._token_indices = self._compute_token_indices(module, hidden_states, keys, values, output[1], kwargs)
            logger.info(
                "FixedLayerScoreEvictPress: computed token indices at layer_idx=%s (k_len=%s, cr=%s).",
                layer_idx,
                int(keys.shape[2]),
                float(self.press.compression_ratio),
            )

        # Apply to ALL layers only at the *model* last layer hook (end of stack).
        # We cannot use `len(cache.layers) - 1` here because `len(cache.layers)` grows with `layer_idx`.
        last_layer_idx = None
        if self._num_hidden_layers is not None:
            last_layer_idx = self._num_hidden_layers - 1
        if last_layer_idx is not None and layer_idx == last_layer_idx:
            if self._token_indices is None:
                logger.warning(
                    "FixedLayerScoreEvictPress: reached last layer but token indices were not computed (score_layer_idx=%s).",
                    self.score_layer_idx,
                )
                return output

            # If compression_ratio=0, keep everything (no-op)
            if float(self.press.compression_ratio) == 0.0:
                return output

            # At this point, cache should have all layers, but be defensive.
            if len(cache.layers) != self._num_hidden_layers:
                logger.warning(
                    "FixedLayerScoreEvictPress: expected %s cache layers at last layer, got %s. Applying to available layers.",
                    self._num_hidden_layers,
                    len(cache.layers),
                )
            logger.info(
                "FixedLayerScoreEvictPress: applying token indices from layer_idx=%s to %s cache layers at last_layer_idx=%s.",
                self.score_layer_idx,
                len(cache.layers),
                last_layer_idx,
            )
            self._apply_token_indices_to_cache(cache, self._token_indices)

        return output

