# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch

# Force transformers to avoid optional TF/Flax backends in this environment.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def _maybe_add_kvpress_memory(module, query, attn_output):
    """
    If module has kvpress memory state (A,b), compute m(q) and add to attn_output.

    Conventions:
    - query: (B, num_heads, q_len, head_dim)
    - attn_output: (B, q_len, num_heads, head_dim)  (this is what HF attention fns return)
    - A: (B, num_kv_heads, d_phi, head_dim)
    - b: (B, num_kv_heads, d_phi) or None
    - phi_W: (head_dim, d_phi) or None (identity)
    """
    A = getattr(module, "_kvpress_memory_A", None)
    if A is None:
        return attn_output

    # Basic guards for odd cases.
    if query is None or attn_output is None or query.numel() == 0 or attn_output.numel() == 0:
        return attn_output

    b = getattr(module, "_kvpress_memory_b", None)
    phi_W = getattr(module, "_kvpress_memory_phi_W", None)
    eps = float(getattr(module, "_kvpress_memory_eps", 1e-6))

    # query: (B, H, T, D)
    B, H, T, D = query.shape
    # A: (B, H_kv, d_phi, D)
    H_kv = A.shape[1]
    if H_kv <= 0 or H % H_kv != 0:
        # Can't align GQA groups; skip.
        return attn_output
    G = H // H_kv

    # Reshape query into kv-head groups: (B, H_kv, G, T, D)
    qg = query.view(B, H_kv, G, T, D)

    # Prefer learnable per-layer params if present.
    mem_mod = getattr(module, "kvpress_memory", None)
    if mem_mod is not None and hasattr(mem_mod, "phi") and hasattr(mem_mod, "gate"):
        # (B,H_kv,G,T,d_phi)
        phi_q = mem_mod.phi(qg)
        gate = float(mem_mod.gate().item())
    else:
        # Fixed/random projection fallback via exposed phi_W.
        if phi_W is None:
            phi_q = qg
        else:
            phi_q = torch.einsum("bkgtd,df->bkgtf", qg, phi_W)
        gate = 1.0

    # m(q) = (φ(q)^T A) / ( (φ(q)^2)^T b + eps ) if b exists
    # where b is accumulated as Σ φ(k)^2 (see MemoryScorerPress).
    # numerator: (B, H_kv, G, T, D)
    m = torch.einsum("bkgtf,bkfd->bkgtd", phi_q, A)
    if b is not None:
        denom = torch.einsum("bkgtf,bkf->bkgt", phi_q ** 2, b).unsqueeze(-1)  # (B,H_kv,G,T,1)
        denom = denom.clamp(min=eps)
        m = m / denom

    m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)

    # Back to attention-head layout: (B, T, H, D)
    m = m.reshape(B, H, T, D).transpose(1, 2).contiguous()

    # Fusion: o = o_save + g * m(q)
    return attn_output + (m * gate)


def search_hyperplane(X, max_iter: int = 1000):
    """
    Given a tensor X of shape (bsz, seq_len, head_dim), search for a hyperplane Y (bsz, head_dim)
    such that for every i, <X[:, i], Y> <= 0. Returns - 1e5 * Y / ||Y|| ** 2 to ensure exp(<X, Y>) = 0
    Raises a ValueError if no such hyperplane is found

    Parameters
    ----------
    X : torch.Tensor
        Query tensor with shape (batch_size, seq_len, head_dim) representing
        the query vectors for which we want to find a nullifying hyperplane.
    max_iter : int, default=1000
        Maximum number of iterations to search for the hyperplane. If no valid
        hyperplane is found within this limit, a ValueError is raised.

    Returns
    -------
    torch.Tensor
        Hyperplane tensor with shape (batch_size, head_dim) scaled by -1e5 / ||Y||²
        to ensure that exp(<X, Y>) ≈ 0 for all queries in X.

    Raises
    ------
    ValueError
        If no valid hyperplane is found within max_iter iterations.
    """
    Y = X.mean(1)  # this initialization is enough for most cases
    for _ in range(max_iter):
        mask = torch.bmm(X, Y.unsqueeze(-1)) <= 0
        if not mask.any():
            return -1e5 * Y / Y.norm(dim=-1, keepdim=True) ** 2
        Y += (X * mask).sum(1) / mask.sum(1).clamp(min=1)
    raise ValueError("Could not find fake keys such that for every query q, exp(<q, k>) = 0")


def attention_patch(func):
    """
    Decorator to update the keys before the attention computation at the indices provided in module.masked_key_indices
    The keys are updated with a fake key k such that exp(<q, k>) = 0 to fake head-wise compression
    This solution is not optimal as it does not reduce peak memory and slightly increases runtime

    Parameters
    ----------
    func : callable
        The original attention function to be patched. Should accept parameters
        (module, query, key, value, attention_mask, dropout, **kwargs).

    Returns
    -------
    callable
        The wrapped attention function that supports head-wise key masking.
    """

    def wrapper(module, query, key, value, attention_mask, dropout, **kwargs):
        if query.shape[2] == key.shape[2]:
            # Prefilling
            module.masked_key_indices = None
        elif getattr(module, "masked_key_indices", None) is not None:
            # Decoding: build fake keys k s.t. exp(<q, k>) = 0
            bsz, num_heads, seq_len, head_dim = query.shape
            num_key_value_heads = key.shape[1]
            num_groups = num_heads // num_key_value_heads

            # Build a fake key k per key group such that for every query q, exp(<q, k>) = 0
            q = query.view(bsz, num_key_value_heads, num_groups, seq_len, head_dim)
            q = q.reshape(bsz * num_key_value_heads, num_groups * seq_len, head_dim)
            k = search_hyperplane(q)
            k = k.view(bsz, num_key_value_heads, head_dim)

            # At indices, update the keys to the fake keys
            batch_indices, head_indices, seq_indices = module.masked_key_indices
            key[batch_indices, head_indices, seq_indices] = k[batch_indices, head_indices]

        # see https://github.com/NVIDIA/kvpress/pull/115#issuecomment-3183785597
        # cu_seq_lens_k are only in kwargs if model.generate is used.
        if "cu_seq_lens_k" in kwargs:
            kwargs["cu_seq_lens_k"][-1] = key.shape[-2]
        out = func(module, query, key, value, attention_mask, dropout, **kwargs)

        # kvpress memory readout & fusion (if enabled by a press writing state onto module).
        if isinstance(out, tuple):
            attn_out = out[0]
            attn_out = _maybe_add_kvpress_memory(module, query, attn_out)
            return (attn_out,) + out[1:]
        else:
            return _maybe_add_kvpress_memory(module, query, out)

    return wrapper


def patch_attention_functions():
    """
    Apply attention patching to all transformer attention functions.

    This function automatically patches all attention functions registered in
    transformers' ALL_ATTENTION_FUNCTIONS to support head-wise key masking.
    It enables KVPress compression methods that require head-specific masking
    (like AdaKV) to work correctly during text generation.

    The patching is applied globally and affects all transformer models loaded
    after this function is called. It's automatically called when importing
    kvpress to ensure compatibility with head-wise compression methods.

    Notes
    -----
    This function modifies the global attention functions in the transformers
    library. The modifications do not affect models that don't use head-wise compression (i.e. don't have
    module.masked_key_indices).
    """
    for name, func in ALL_ATTENTION_FUNCTIONS.items():
        ALL_ATTENTION_FUNCTIONS[name] = attention_patch(func)
