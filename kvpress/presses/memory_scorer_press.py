# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MemoryScorerPress
-----------------

在 score-based KV eviction 的同时，把被 evict 的 (K, V) 写入一个小的 stateful memory。

实现对应 `KVCache/kvpress/memory.md` 里的 additive 写入版本：

- 对每层、每个 KV head，维护 per-sequence memory state:
  - A ∈ R^{d_phi × d_v} (这里 d_v = head_dim)
  - b ∈ R^{d_phi} (可选，用于分母稳定)

- eviction 时写入：
  A ← λ A + η Σ_t φ(k_t) ⊗ v_t
  b ← λ b + η Σ_t φ(k_t)

注意：这里只负责 “存”，不负责 “读出融合”(m(q)/gate)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress


def _infer_head_dim(module: nn.Module, keys: torch.Tensor) -> int:
    # Prefer module.head_dim when available, fallback to tensor.
    return int(getattr(module, "head_dim", keys.shape[-1]))


class KVPressMemoryLayer(nn.Module):
    """
    Per-layer trainable parameters for KVPress memory:
    - phi_proj:  φ(x) = x @ W_phi, shape (head_dim -> d_phi)
    - gate:      g in (0,1), scalar per layer (applied uniformly to all heads/tokens)
    - eta:       write strength > 0
    - decay:     memory decay in (0,1)
    """

    def __init__(
        self,
        head_dim: int,
        d_phi: int,
        # IMPORTANT: initialize conservatively to avoid exploding memory output at stage2 start.
        init_gate: float = 1e-3,
        init_eta: float = 1e-3,
        init_decay: float = 0.99,
    ):
        super().__init__()
        self.head_dim = int(head_dim)
        self.d_phi = int(d_phi)
        self.phi_proj = nn.Linear(self.head_dim, self.d_phi, bias=False)

        # Parameterizations for constraints.
        # gate = sigmoid(gate_logit)
        # eta  = softplus(eta_raw)
        # decay = sigmoid(decay_logit)
        self.gate_logit = nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(init_gate).clamp(1e-4, 1 - 1e-4)))))  # type: ignore[arg-type]
        # Choose eta_raw so softplus(eta_raw) ~= init_eta
        self.eta_raw = nn.Parameter(torch.tensor(float(torch.log(torch.expm1(torch.tensor(init_eta).clamp(min=1e-4))))))  # type: ignore[arg-type]
        self.decay_logit = nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(init_decay).clamp(1e-4, 1 - 1e-4)))))  # type: ignore[arg-type]

    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def eta(self) -> torch.Tensor:
        return F.softplus(self.eta_raw)

    def decay(self) -> torch.Tensor:
        return torch.sigmoid(self.decay_logit)

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        return self.phi_proj(x)


@dataclass
class MemoryScorerPress(ScorerPress):
    """
    Wrap an existing ScorerPress and, during compress, save evicted KV into a memory state.

    Parameters
    ----------
    base_press:
        The scorer press that provides the scoring function. We delegate `score()` to it.
    d_phi:
        Feature map dimension. If None, use identity φ(k)=k (d_phi=head_dim).
    decay:
        λ ∈ (0,1]. Memory decay per compress call (not per token).
    write_eta:
        η > 0. Global write strength.
    use_denominator:
        Whether to maintain b for stabilized readout later.
    attach_to_module:
        If True, also attach current memory tensors to attention module as attributes:
        - module._kvpress_memory_A
        - module._kvpress_memory_b
        so downstream code can read them easily.
    """

    # NOTE: dataclass inheritance restriction: base class has default fields, so subclass
    # cannot introduce non-default fields. Keep default=None and validate in __post_init__.
    base_press: ScorerPress | None = None

    d_phi: Optional[int] = None
    # Defaults used when per-layer learnable params are not attached.
    decay: float = 1.0
    write_eta: float = 1.0
    use_denominator: bool = True
    attach_to_module: bool = True

    # Per-layer memory state. Keys are layer_idx.
    _mem_A: Dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _mem_b: Dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _phi_W: Dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)  # (head_dim, d_phi)

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.base_press, BasePress), "base_press must be provided and must be a BasePress"
        assert 0.0 < self.decay <= 1.0, "decay must be in (0, 1]"
        assert self.write_eta >= 0.0, "write_eta must be >= 0"
        if self.d_phi is not None:
            assert self.d_phi > 0, "d_phi must be > 0"

    def post_init_from_model(self, model, force_reinit: bool = False):
        """
        Attach per-layer learnable memory params onto each attention module as `self_attn.kvpress_memory`.
        This makes phi_proj/gate/eta/decay trainable & saved in the model state_dict.
        """
        # Delegate any model-dependent init to base press (e.g., attach indexer modules).
        if hasattr(self.base_press, "post_init_from_model"):
            # Some presses accept force_reinit; pass it through if supported.
            try:
                self.base_press.post_init_from_model(model, force_reinit=force_reinit)
            except TypeError:
                self.base_press.post_init_from_model(model)

        # Attach per-layer memory params to each self_attn module.
        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        for layer in getattr(language_model, "layers", []):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            head_dim = int(getattr(attn, "head_dim", 0) or 0)
            if head_dim <= 0:
                continue
            d_phi = int(self.d_phi) if self.d_phi is not None else head_dim
            already = hasattr(attn, "kvpress_memory") and isinstance(getattr(attn, "kvpress_memory"), nn.Module)
            if already and not force_reinit:
                continue
            # IMPORTANT: when models are loaded with `device_map="auto"`, newly attached modules
            # will stay on CPU unless we explicitly place them. Use the attention module's own
            # parameter device/dtype to avoid CPU/GPU mismatch during `phi_proj`.
            try:
                p0 = next(attn.parameters())
                dev, dt = p0.device, p0.dtype
            except StopIteration:
                # Fallback: use model's first parameter.
                p0 = next(model.parameters())
                dev, dt = p0.device, p0.dtype

            mem_layer = KVPressMemoryLayer(
                head_dim=head_dim,
                d_phi=d_phi,
                init_gate=1e-3,
                init_eta=1e-3,
                init_decay=0.99,
            )
            mem_layer = mem_layer.to(device=dev, dtype=dt)
            attn.kvpress_memory = mem_layer

    def reset(self):
        """Reset per-sequence memory state."""
        self._mem_A.clear()
        self._mem_b.clear()
        self._phi_W.clear()
        # Also reset base press internal state if it has a reset.
        if hasattr(self.base_press, "reset") and callable(getattr(self.base_press, "reset")):
            self.base_press.reset()

    def _reset_cache(self):
        """Alias used by evaluation runner for non-DecodePress inference loops."""
        self.reset()

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        return self.base_press.score(module, hidden_states, keys, values, attentions, kwargs)

    def _get_phi_W(self, layer_idx: int, head_dim: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if self.d_phi is None or self.d_phi == head_dim:
            return None  # identity
        d_phi = int(self.d_phi)
        W = self._phi_W.get(layer_idx, None)
        if W is None or W.shape != (head_dim, d_phi) or W.device != device:
            # Fixed (non-trainable) random projection; scaled so φ(k) has reasonable magnitude.
            # Note: we keep this deterministic within a run by using the default RNG state.
            W = torch.randn(head_dim, d_phi, device=device, dtype=torch.float32) * (head_dim ** -0.5)
            W = W.to(dtype=dtype)
            self._phi_W[layer_idx] = W
        elif W.dtype != dtype:
            W = W.to(dtype=dtype)
            self._phi_W[layer_idx] = W
        return W

    def _ensure_memory_state(
        self,
        layer_idx: int,
        bsz: int,
        n_kv_heads: int,
        d_phi: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        A = self._mem_A.get(layer_idx, None)
        b = self._mem_b.get(layer_idx, None) if self.use_denominator else None

        target_A_shape = (bsz, n_kv_heads, d_phi, head_dim)
        target_b_shape = (bsz, n_kv_heads, d_phi)

        if A is None or A.shape != target_A_shape or A.device != device or A.dtype != dtype:
            A = torch.zeros(target_A_shape, device=device, dtype=dtype)
            self._mem_A[layer_idx] = A

        if self.use_denominator:
            if b is None or b.shape != target_b_shape or b.device != device or b.dtype != dtype:
                b = torch.zeros(target_b_shape, device=device, dtype=dtype)
                self._mem_b[layer_idx] = b

        return A, b

    def _get_layer_memory_module(self, module: nn.Module) -> Optional[KVPressMemoryLayer]:
        mem = getattr(module, "kvpress_memory", None)
        if isinstance(mem, KVPressMemoryLayer):
            return mem
        return None

    @torch.no_grad()
    def _write_evicted_to_memory(self, module: nn.Module, k_evict: torch.Tensor, v_evict: torch.Tensor):
        """
        k_evict, v_evict: (B, H_kv, T_evict, D)
        """
        if self.write_eta == 0.0:
            return
        if k_evict.numel() == 0:
            return

        layer_idx = int(getattr(module, "layer_idx", 0))
        bsz, n_kv_heads, t_evict, _ = k_evict.shape
        head_dim = _infer_head_dim(module, k_evict)
        d_phi = int(self.d_phi) if self.d_phi is not None else head_dim

        A, b = self._ensure_memory_state(
            layer_idx=layer_idx,
            bsz=bsz,
            n_kv_heads=n_kv_heads,
            d_phi=d_phi,
            head_dim=head_dim,
            device=k_evict.device,
            dtype=k_evict.dtype,
        )

        mem_mod = self._get_layer_memory_module(module)
        if mem_mod is not None:
            # Use learnable per-layer params.
            decay = float(mem_mod.decay().item())
            eta = float(mem_mod.eta().item())
            if decay != 1.0:
                A.mul_(decay)
                if b is not None:
                    b.mul_(decay)
            # φ(k): (B,H,T,d_phi)
            phi_k = mem_mod.phi(k_evict)
            alpha = eta
            W = None
        else:
            # Fallback to fixed/random projection.
            if self.decay != 1.0:
                A.mul_(self.decay)
                if b is not None:
                    b.mul_(self.decay)
            W = self._get_phi_W(layer_idx, head_dim=head_dim, device=k_evict.device, dtype=k_evict.dtype)
            if W is None:
                phi_k = k_evict  # (B,H,T,D)
            else:
                phi_k = torch.einsum("bhte,ed->bhtd", k_evict, W)
            alpha = float(self.write_eta)

        # A += η * Σ_t φ(k_t) ⊗ v_t
        # (B,H,T,d_phi) and (B,H,T,head_dim) -> (B,H,d_phi,head_dim)
        dA = torch.einsum("bhtd,bhte->bhde", phi_k, v_evict)
        A.add_(dA, alpha=float(alpha))

        if b is not None:
            # Use squared features for a positive, more stable denominator in readout.
            # This avoids cancellation in b when φ has mixed signs.
            db = (phi_k ** 2).sum(dim=2)  # (B,H,d_phi)
            b.add_(db, alpha=float(alpha))

        if self.attach_to_module:
            setattr(module, "_kvpress_memory_A", A)
            if b is not None:
                setattr(module, "_kvpress_memory_b", b)
            else:
                # Ensure stale b isn't used downstream.
                if hasattr(module, "_kvpress_memory_b"):
                    setattr(module, "_kvpress_memory_b", None)
            # Expose readout metadata for the attention wrapper.
            setattr(module, "_kvpress_memory_d_phi", d_phi)
            setattr(module, "_kvpress_memory_use_denominator", bool(self.use_denominator))
            setattr(module, "_kvpress_memory_eps", 1e-6)
            # Backward-compat: keep exposing a fixed projection if we used it.
            if W is not None:
                setattr(module, "_kvpress_memory_phi_W", W)
            else:
                if hasattr(module, "_kvpress_memory_phi_W"):
                    setattr(module, "_kvpress_memory_phi_W", None)

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
        Same top-k selection as `ScorerPress.compress`, plus:
        - gather evicted KV
        - write into memory state (A,b)
        """
        if self.compression_ratio == 0:
            return keys, values

        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        # Mirror ScorerPress behavior: optional running mean across layers and/or mean-head selection.
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

        selection_scores = scores
        if self.mean_head:
            selection_scores = scores.mean(dim=1, keepdim=True)

        k_len = int(keys.shape[2])
        n_kept = max(1, int(k_len * (1 - self.compression_ratio)))
        n_evict = max(0, k_len - n_kept)

        kept_idx = selection_scores.topk(n_kept, dim=-1).indices  # (B, H or 1, n_kept)
        if self.mean_head:
            kept_idx = kept_idx.expand(-1, keys.shape[1], -1)  # (B, num_kv_heads, n_kept)

        if n_evict > 0:
            evict_idx = selection_scores.topk(n_evict, dim=-1, largest=False).indices  # (B, H or 1, n_evict)
            if self.mean_head:
                evict_idx = evict_idx.expand(-1, keys.shape[1], -1)

            # Gather evicted KV for memory write.
            evict_gather = evict_idx.unsqueeze(-1).expand(-1, -1, -1, _infer_head_dim(module, keys))
            k_evict = keys.gather(2, evict_gather).contiguous()
            v_evict = values.gather(2, evict_gather).contiguous()
            self._write_evicted_to_memory(module, k_evict, v_evict)

        # Gather kept KV for the actual cache.
        kept_gather = kept_idx.unsqueeze(-1).expand(-1, -1, -1, _infer_head_dim(module, keys))
        keys = keys.gather(2, kept_gather).contiguous()
        values = values.gather(2, kept_gather).contiguous()
        return keys, values


def load_model_with_memory_params(model_path: str, model_kwargs=None):
    """
    Load a HF model + tokenizer, and (if present in checkpoint) load:
    - per-layer memory params under keys containing 'kvpress_memory'
    - indexer params under keys containing 'indexer' (compatible with QueryIndexerScorePress)

    This mirrors the logic in `load_model_with_query_indexer_press`, but extends it to memory params.
    """
    import os
    import json
    import logging
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger = logging.getLogger(__name__)

    if model_kwargs is None:
        model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
    index_path = os.path.join(model_path, "pytorch_model.bin.index.json")

    def _infer_d_phi_from_state_dict(state_dict: dict) -> Optional[int]:
        """
        Infer d_phi from any `*.kvpress_memory.phi_proj.weight` tensor.
        nn.Linear weight shape: (out_features=d_phi, in_features=head_dim).
        """
        for k, v in state_dict.items():
            if "kvpress_memory.phi_proj.weight" in k and hasattr(v, "shape") and len(v.shape) == 2:
                return int(v.shape[0])
        return None

    def _infer_d_phi_from_sharded_index(weight_map: dict) -> Optional[int]:
        """
        Infer d_phi by loading the single shard that contains `kvpress_memory.phi_proj.weight`.
        Avoids reading all shards just to learn the shape.
        """
        phi_key = None
        shard_file = None
        for k, sf in weight_map.items():
            if "kvpress_memory.phi_proj.weight" in k:
                phi_key = k
                shard_file = sf
                break
        if phi_key is None or shard_file is None:
            return None
        shard_path = os.path.join(model_path, shard_file)
        if not os.path.exists(shard_path):
            return None
        sd = torch.load(shard_path, map_location="cpu")
        w = sd.get(phi_key, None)
        if w is None:
            # Fallback: keys might differ slightly; scan.
            return _infer_d_phi_from_state_dict(sd)
        if hasattr(w, "shape") and len(w.shape) == 2:
            return int(w.shape[0])
        return None

    def _ensure_modules(inferred_d_phi: Optional[int] = None):
        # If checkpoint contains indexer weights, ensure QueryIndexer modules exist.
        try:
            from kvpress.presses.indexer_score_query_press import QueryIndexerScorePress
        except Exception:
            QueryIndexerScorePress = None  # type: ignore[assignment]

        base = QueryIndexerScorePress(compression_ratio=0.0) if QueryIndexerScorePress is not None else ScorerPress(compression_ratio=0.0)  # type: ignore[call-arg]
        dummy = MemoryScorerPress(base_press=base, compression_ratio=0.0, d_phi=inferred_d_phi)
        dummy.post_init_from_model(model, force_reinit=False)

    loaded = False

    def _load_filtered(sd: dict) -> dict:
        # Only load scorer-related weights, not the whole base model.
        return {k: v for k, v in sd.items() if ("indexer" in k) or ("kvpress_memory" in k)}

    # Case 1: sharded checkpoint
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        keys_of_interest = [k for k in weight_map.keys() if ("indexer" in k) or ("kvpress_memory" in k)]

        if keys_of_interest:
            inferred_d_phi = _infer_d_phi_from_sharded_index(weight_map)
            _ensure_modules(inferred_d_phi=inferred_d_phi)
            shard_files = sorted({weight_map[k] for k in keys_of_interest})
            partial_sd = {}
            for shard_file in shard_files:
                shard_path = os.path.join(model_path, shard_file)
                if not os.path.exists(shard_path):
                    logger.warning("Missing shard file referenced by index: %s", shard_path)
                    continue
                shard_sd = torch.load(shard_path, map_location="cpu")
                partial_sd.update(_load_filtered(shard_sd))
                del shard_sd

            if partial_sd:
                incompatible = model.load_state_dict(partial_sd, strict=False)
                logger.info(
                    "Loaded memory/indexer weights from sharded checkpoint: %s keys (missing=%s, unexpected=%s)",
                    len(partial_sd),
                    len(getattr(incompatible, "missing_keys", []) or []),
                    len(getattr(incompatible, "unexpected_keys", []) or []),
                )
                loaded = True
        else:
            logger.info("No memory/indexer weights found in %s (weight_map size=%s).", index_path, len(weight_map))

    # Case 2: single-file checkpoint
    if (not loaded) and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        has_partial = any(("indexer" in k) or ("kvpress_memory" in k) for k in state_dict.keys())
        if has_partial:
            inferred_d_phi = _infer_d_phi_from_state_dict(state_dict)
            _ensure_modules(inferred_d_phi=inferred_d_phi)
            partial_sd = _load_filtered(state_dict)
            incompatible = model.load_state_dict(partial_sd, strict=False)
            logger.info(
                "Loaded memory/indexer weights from single-file checkpoint: missing=%s, unexpected=%s",
                len(getattr(incompatible, "missing_keys", []) or []),
                len(getattr(incompatible, "unexpected_keys", []) or []),
            )
            loaded = True
        del state_dict

    if loaded:
        print("✓ Loaded trained kvpress memory/indexer weights from checkpoint", flush=True)

    return model, tokenizer

