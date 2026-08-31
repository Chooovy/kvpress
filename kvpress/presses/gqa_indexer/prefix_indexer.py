# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Query-independent scorer with a *prefix-attention* readout:
``s_j = w_out . phi(W_in norm(h_j) + W_a norm(softmax(q_j K_{<=j}^T / sqrt(D)) V_{<=j})) + j * eps``.

The third arm of the eviction question. :class:`~.indexer.GQAIndexer` scores every
``(query, key)`` pair, which is query-aware but cannot evict: key ``j``'s score moves with each
new query, so a freed entry may be needed again. :class:`~.scalar_indexer.ScalarIndexer` can
evict, because a key is scored once from its own hidden state and the ranking is irreversible --
but its whole view of key ``j`` is the single vector ``h_j``. This module keeps the
irreversibility and widens that view to ``j``'s entire prefix, through the indexer's own
one-head attention.

What this is and is not
-----------------------
It is **not** query-aware, and the distinction is worth stating precisely because the ``q . k``
form invites the opposite reading. The query in ``softmax(q_j K^T) V`` is projected from ``h_j``
-- the *key token's own* hidden state -- and the attention runs over ``j``'s prefix. So ``s_j``
is fixed the moment ``j`` arrives, exactly as in :class:`~.scalar_indexer.ScalarIndexer`. Compare
the information sets:

===================================  =========================================
scorer                               sees, when scoring key ``j``
===================================  =========================================
:class:`~.scalar_indexer.ScalarIndexer`   ``h_j``
**this module**                      ``prefix <= j``
:class:`~.indexer.GQAIndexer`        ``prefix <= j`` **and the query**
===================================  =========================================

Query-awareness is the ``{query}`` column, and this module does not add it. It therefore lives
inside the same hypothesis class as the scalar arm and inherits that class's measured ceiling
(``proxy_exp/HANDOFF.md`` §12.1: a per-head router already reaches 84% of the
``oracle_qi``-to-``recency`` band, and §11.4 measures the *achievable* bound for anything frozen
at prefill as ~1.9x looser than ``oracle_qi`` itself). What it buys is a strictly larger feature
set within that class, which two things in the audit say is still worth measuring:

* §10.5 -- ``oracle_qi`` is *not* a ceiling. Greedy set search against true damage beats it
  (0.0259 -> 0.0246, per-cell geo 0.952), so a scorer that can reason about a key *in the context
  of its neighbours* has real headroom that a token-local one does not.
* §12.5 -- the prefix-neighbour signal exists, with the **opposite** sign to the obvious one:
  ``nn_novelty_neg`` recovers 42% of the band and beats ``recency`` 21/24 cells. "Keep the keys
  whose earlier neighbours already point the same way; evict the novel ones."

And two things say to be pessimistic. Both weakened forms of "let the per-key score read the
prefix" have already been measured and lost to the token-local MLP: the recurrent state ``z``
(§8.3/§10.8) did not survive its own shuffle control (``h+z`` vs ``h+z_shuffled`` at matched
width: +0.0132, t=0.79, sign-test p=0.388; the shuffled control *wins* at every width at L14),
and every redundancy score sat at chance on future demand (§9.1, top-25 overlap 0.264 against a
0.25 floor). This module differs from both in using softmax attention rather than a linear
reconstruction -- §9.5 is explicit that only *one specific* set-aware signal was refuted -- but
the bar it has to clear is a measured **0.0313** ``rel_L2`` at keep 25%, not a plausibility
argument.

Superset by construction, so the A/B is single-variable
-------------------------------------------------------
:class:`PrefixIndexer` subclasses :class:`~.scalar_indexer.ScalarIndexer` and reuses its
parameters under the same names, adding only the prefix branch. With ``w_a`` zero-initialized
(the default) the score is **bit-identical** to the scalar arm's, verified in
``tests/presses/test_gqa_indexer_prefix.py``. So "read the prefix" is the only variable in the
comparison, and a checkpoint's ``w_a`` norm is a direct readout on whether the branch earned its
place.

Zero-init does *not* create the dead start that
:attr:`~.indexer.GQAIndexer.gate_scale` warns about, but it is a one-step saddle and that is
worth being precise about. At ``w_a == 0`` the branch's projections get no gradient
(``dL/da = W_a^T dL/dz = 0``), but ``w_a`` itself does (``dL/dW_a = dL/dz (x) norm(a)``, and
``norm(a)`` is not zero), so the first optimizer step moves ``w_a`` off zero and ``w_pq``/``w_pk``/
``w_pv`` receive gradient from the second step onward. Verified rather than argued:
``test_zero_init_prefix_branch_escapes_saddle``.

The variance-collapse failure mode to watch
-------------------------------------------
``softmax(...) V`` is a convex combination of ``{v_i}_{i<=j}``, so ``a_j`` lies in their convex
hull. As the attention spreads, ``a_j -> mean(v)`` -- which means **the spread of ``a_j`` across
``j`` shrinks as ``j`` grows**. Late keys' scores become mutually less distinguishable while
early keys' stay spread, and since top-k compares across positions this appears as a systematic
position bias with ``pos_slope`` dominating the tail outright.

``a_norm`` does not fix this: LayerNorm normalizes across *channels*, at fixed ``j``, so if two
late tokens both sit near ``mean(v)`` it maps them to nearly the same unit vector. It keeps the
score's magnitude from decaying, not the tokens apart. The ``W_in norm(h_j)`` residual is the
actual mitigation -- it carries position-independent spread -- which is a second reason the
readout is a sum rather than a pure prefix readout.

Measure it before training rather than after: :func:`score_variance_profile` reports
``Var_j(s_j)`` in position bins, and a monotone decay across bins is this failure mode.

Scope
-----
Training and **prefill-time** scoring by default. Decode-time scoring is opt-in through
:meth:`PrefixIndexer.enable_cache`, which keeps the indexer's own ``K``/``V`` across calls --
``O(L)`` state and ``O(t)`` per step, the cost :mod:`~.scalar_indexer` exists to avoid, so it is
paid only when asked for. :class:`~.sparse_inference.SparseAttentionContext` enables it for the
duration of the context, which is what lets an eval generate. Without it
:meth:`PrefixIndexer.score_keys` raises on a non-zero ``key_offset`` rather than silently scoring
a suffix against its own truncated prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.presses.gqa_indexer.indexer import MASK_NEG, IndexerNorm
from kvpress.presses.gqa_indexer.scalar_indexer import (
    DEFAULT_POS_SLOPE,
    ScalarIndexer,
    ScalarIndexerConfig,
)


@dataclass
class PrefixIndexerConfig(ScalarIndexerConfig):
    """
    Shape configuration for :class:`PrefixIndexer`.

    Extends :class:`~.scalar_indexer.ScalarIndexerConfig` with the prefix branch's geometry, and
    inherits ``hidden_size``, ``n_heads``, ``mid_dim``, ``norm_eps``, ``pos_slope`` and
    ``gate_scale`` unchanged -- the shared parameters are the same modules under the same names,
    which is what makes the scalar arm a reachable special case.

    Attributes
    ----------
    head_dim : int
        Width of the prefix attention's ``q``/``k``. Sets the ``1/sqrt(head_dim)`` softmax scale.
    value_dim : int
        Width of the prefix attention's ``v``, i.e. of the readout ``a_j``. This is the per-token
        indexer cache cost if decode-time scoring is ever wired up (``head_dim + value_dim``
        per token per layer, against ``n_heads`` scalars for the scalar arm).
    zero_init_prefix : bool
        Zero-initialize ``w_a``, so training starts *exactly* at the scalar arm and the prefix
        branch is the only variable. On by default; see the module docstring on why this is an
        escapable saddle rather than a dead start.

    Notes
    -----
    ``rope_dim`` is inherited as a fixed ``0``. The prefix attention is deliberately NoPE: ``h_j``
    already carries the backbone's own rotary signal from every layer below, so ``q``/``k`` are
    not position-blind, and the recency prior the ranking needs is carried explicitly by
    ``pos_slope``. Reporting ``0`` is also what lets the press's RoPE plumbing drive this scorer
    through the scalar arm's code path unchanged.
    """

    head_dim: int = 128
    value_dim: int = 128
    zero_init_prefix: bool = True

    #: Inherited from :class:`~.scalar_indexer.ScalarIndexerConfig` as ``init=False``; restated
    #: here only so this class's field order is well-defined.
    rope_dim: int = field(default=0, init=False)

    def __post_init__(self):
        super().__post_init__()
        for name in ("head_dim", "value_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")


class PrefixIndexer(ScalarIndexer):
    """
    One score per key, from that key's whole prefix.

    ``score_keys`` returns ``(B, n_heads, Sk)`` in fp32, the same contract as
    :class:`~.scalar_indexer.ScalarIndexer` -- and every other method (``forward``,
    ``expand_to_pairs``, ``gate_key``, ``gate_query``, ``project_q``, ``project_k``,
    ``require_gate_scale``) is inherited unchanged, because all of them are written against
    ``score_keys``. That is the whole reason this is a subclass: the press, the trainers, the
    gate and :mod:`~.qi_flex_attention`'s deadline path need no knowledge of it.

    The prefix attention is **one head**, shared by all ``n_heads`` outputs: the readout ``a_j``
    is computed once and the per-head split happens only in ``w_out``. Per-head prefix attention
    would multiply the (potential) indexer cache by ``n_heads`` for a readout the heads would
    largely agree on; the per-head *ranking* is preserved either way, since ``w_out`` still emits
    ``n_heads`` independent scores.

    Irreversibility is preserved, which is the property that makes eviction safe: ``s_j`` depends
    only on ``prefix <= j``, so it is fixed when ``j`` arrives and cannot be revised by a later
    query. The scalar arm's ``pos_slope`` tie-break carries over verbatim.
    """

    #: The prefix readout is a function of the key's prefix, not of any query -- see the module
    #: docstring's information-set table. Inherited from
    #: :class:`~.scalar_indexer.ScalarIndexer` and restated because it is the single attribute
    #: that routes callers onto the deadline path in :mod:`~.qi_flex_attention`, and a reader
    #: checking "is this really query-independent, given the ``q . k``?" should find the answer
    #: on the class itself.
    is_query_independent = True

    def __init__(self, config: PrefixIndexerConfig):
        super().__init__(config)
        self.head_dim = config.head_dim
        self.value_dim = config.value_dim

        self._cache_enabled = False
        self._cache_k: torch.Tensor | None = None
        self._cache_v: torch.Tensor | None = None

        self.w_pq = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.w_pk = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.w_pv = nn.Linear(config.hidden_size, config.value_dim, bias=False)

        # Normalizes a_j across channels before the readout. This keeps the branch's magnitude
        # comparable to the W_in path regardless of how v's scale drifts with depth -- it does
        # NOT address the across-j variance collapse (see the module docstring).
        self.a_norm = IndexerNorm(config.value_dim, eps=config.norm_eps)

        # Into whatever the readout consumes: the MLP's pre-activation when mid_dim > 0, or
        # hidden_size directly for the linear (SparseK) form. Either way w_a = 0 leaves the
        # scalar arm's arithmetic untouched.
        readout_width = config.mid_dim if config.mid_dim else config.hidden_size
        self.w_a = nn.Linear(config.value_dim, readout_width, bias=False)
        if config.zero_init_prefix:
            nn.init.zeros_(self.w_a.weight)

    def enable_cache(self) -> None:
        """
        Start caching the prefix attention's ``K``/``V`` so decode can score incrementally.

        Without this a decode step cannot be scored at all: the readout reads key ``j``'s whole
        prefix, and :meth:`score_keys` rejects a non-zero ``key_offset`` rather than silently
        scoring a suffix against a truncated one. With it, prefill fills the cache and each step
        appends its own ``k``/``v`` and attends against everything before it, so ``s_j`` is what a
        full-prefix rescore would give -- at ``O(t)`` per step and ``head_dim + value_dim`` of
        state per token per layer, instead of re-running the whole prefix.
        """
        self._cache_enabled = True
        self._cache_k = None
        self._cache_v = None

    def disable_cache(self) -> None:
        self._cache_enabled = False
        self._cache_k = None
        self._cache_v = None

    @property
    def cached_length(self) -> int:
        return 0 if self._cache_k is None else int(self._cache_k.shape[2])

    def prefix_readout(
        self, x: torch.Tensor, *, keep: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        The causal prefix attention, ``(B, Sk, value_dim)``.

        Parameters
        ----------
        x : torch.Tensor
            ``in_norm(h)``, ``(B, Sk, hidden_size)`` -- already normalized, and shared with the
            readout's own ``W_in`` path.
        keep : torch.Tensor, optional
            Boolean key-padding mask, ``(B, Sk)``, ``True`` = real token.

        Notes
        -----
        Row ``j`` attends to keys ``0..j`` **inclusive**, so no row is ever fully masked and the
        softmax cannot produce NaN -- a key always has at least itself.

        Without padding this takes SDPA's ``is_causal`` fast path, which is flash-attention and
        therefore ``O(Sk * head_dim)`` memory rather than ``O(Sk^2)``. Padding forces an explicit
        ``(B, 1, Sk, Sk)`` boolean mask, because SDPA accepts ``attn_mask`` or ``is_causal`` but
        not both, and a ``(B, 1, 1, Sk)`` key mask alone would drop causality. That mask is
        quadratic (67 MiB at ``Sk=8192``, ``B=1``), which is fine at training lengths and is why
        the training path packs full-length documents instead of padding.

        When the cache is enabled the new ``k``/``v`` are appended to it first and the attention
        runs against the whole cache. The first call has an empty cache, so it is an ordinary
        causal prefill and keeps the flash path. Later calls have ``Sq < Sk``, where ``is_causal``
        is wrong -- SDPA aligns it top-left, so it would hide the very prefix the cache exists to
        keep -- hence a bottom-right mask, built only when the call carries more than one query
        row. A single decode step sees every cached key and needs no mask at all.
        """
        bsz, k_len, _ = x.shape
        q = self.w_pq(x).view(bsz, k_len, 1, self.head_dim).transpose(1, 2)
        k = self.w_pk(x).view(bsz, k_len, 1, self.head_dim).transpose(1, 2)
        v = self.w_pv(x).view(bsz, k_len, 1, self.value_dim).transpose(1, 2)

        if self._cache_enabled:
            if keep is not None:
                raise ValueError(
                    "the prefix cache does not support a padding mask: the cache is a running "
                    "prefix, so a padded key would stay in every later step's attention. Pack "
                    "sequences instead, or disable the cache."
                )
            fresh = self._cache_k is None
            if not fresh:
                k = torch.cat([self._cache_k, k], dim=2)
                v = torch.cat([self._cache_v, v], dim=2)
            self._cache_k, self._cache_v = k, v
            if fresh:
                a = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                mask = None
                if k_len > 1:
                    total = k.shape[2]
                    offset = total - k_len
                    q_pos = torch.arange(k_len, device=x.device).unsqueeze(-1) + offset
                    k_pos = torch.arange(total, device=x.device).unsqueeze(0)
                    mask = (k_pos <= q_pos).view(1, 1, k_len, total)
                a = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        elif keep is None:
            a = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            causal = torch.ones((k_len, k_len), device=x.device, dtype=torch.bool).tril_()
            allowed = causal.view(1, 1, k_len, k_len) & keep.view(bsz, 1, 1, k_len)
            # A padded query row can end up with every key masked (its own position is padded
            # too). Re-allow the diagonal so the softmax stays finite; those rows are overwritten
            # with MASK_NEG by score_keys, so the value is never used.
            allowed = allowed | torch.eye(k_len, device=x.device, dtype=torch.bool).view(
                1, 1, k_len, k_len
            )
            a = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        return a.transpose(1, 2).reshape(bsz, k_len, self.value_dim)

    def score_keys(
        self,
        hidden_states: torch.Tensor,
        *,
        key_offset: int = 0,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Score each key once from its prefix -> ``(B, n_heads, Sk)`` in fp32.

        Overrides :meth:`~.scalar_indexer.ScalarIndexer.score_keys` and keeps its signature, so
        the press calls it unchanged.

        Parameters
        ----------
        hidden_states : torch.Tensor
            ``(B, Sk, hidden_size)``. Unlike the scalar arm this must be the sequence **from
            position 0**: the prefix attention reads it as the whole prefix.
        key_offset : int
            Absolute position of the first key, for the recency tilt. Must be ``0`` unless the
            prefix cache is enabled: without a cache a non-zero offset means ``hidden_states`` is
            a suffix, and the prefix attention would score each key against a *truncated* prefix
            -- silently, and differently depending on how the prefill was split. With the cache
            the earlier prefix is still there, so the offset is honoured and must equal the number
            of keys already cached.
        mask : torch.Tensor, optional
            Keep-mask over keys, broadcastable to ``(B, Sk)``. Unlike the scalar arm this also
            excludes padded keys from every *other* key's prefix attention, not just from the
            final ranking.

        Returns
        -------
        torch.Tensor
            ``(B, n_heads, Sk)`` fp32 scores.
        """
        if hidden_states.dim() != 3:
            raise ValueError(
                f"hidden_states must be (B, Sk, hidden_size), got {tuple(hidden_states.shape)}"
            )
        if key_offset != 0 and not self._cache_enabled:
            raise ValueError(
                f"PrefixIndexer.score_keys requires key_offset=0, got {key_offset}. The score "
                "reads the key's whole prefix, so a suffix would be scored against a truncated "
                "one -- a silent dependence on how the prefill was chunked. Call enable_cache() "
                "to carry the indexer's own K/V across calls, which makes an offset meaningful."
            )
        if self._cache_enabled and key_offset != self.cached_length:
            raise ValueError(
                f"key_offset={key_offset} but the prefix cache holds {self.cached_length} keys. "
                "The offset is the absolute position of the first new key, so a mismatch means "
                "the cache and the sequence have diverged -- keys would be scored against the "
                "wrong prefix and the recency tilt would be wrong too."
            )
        if hidden_states.dtype != self.weight_dtype:
            hidden_states = hidden_states.to(self.weight_dtype)

        keep = None
        if mask is not None:
            keep = (mask if mask.dtype == torch.bool else mask != 0).view(
                hidden_states.shape[0], -1
            )
            if bool(keep.all()):
                keep = None  # nothing masked: stay on the O(L) flash path

        x = self.in_norm(hidden_states)
        a = self.a_norm(self.prefix_readout(x, keep=keep))

        if self.w_in is not None:
            scores = self.w_out(nn.functional.gelu(self.mid_norm(self.w_in(x) + self.w_a(a))))
        else:
            scores = self.w_out(x + self.w_a(a))
        scores = scores.float().transpose(1, 2)  # (B, n_heads, Sk)

        if self.pos_slope:
            pos = torch.arange(
                key_offset,
                key_offset + hidden_states.shape[1],
                device=scores.device,
                dtype=scores.dtype,
            )
            scores = scores + self.pos_slope * pos

        if mask is not None:
            real = (mask if mask.dtype == torch.bool else mask != 0).view(
                hidden_states.shape[0], 1, -1
            )
            scores = scores.masked_fill(~real, MASK_NEG)
        return scores

    def project_k(self, hidden_states: torch.Tensor, cos=None, sin=None) -> torch.Tensor:
        self._reject_rope(cos, sin)
        return self.gate_key(
            hidden_states, key_offset=self.cached_length, dtype=hidden_states.dtype
        )

    def extra_repr(self) -> str:
        shape = f"hidden={self.config.hidden_size}, n_heads={self.n_heads}"
        shape += f", mid_dim={self.mid_dim}" if self.mid_dim else " (linear)"
        prefix = f"prefix(head_dim={self.head_dim}, value_dim={self.value_dim})"
        return f"{shape}, {prefix}, pos_slope={self.pos_slope:g}"


@torch.no_grad()
def score_variance_profile(
    scores: torch.Tensor, n_bins: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ``Var_j(s_j)`` in ``n_bins`` equal position bins -- the variance-collapse diagnostic.

    ``softmax(...) V`` lies in the convex hull of ``{v_i}_{i<=j}``, so as the attention spreads
    ``a_j -> mean(v)`` and the *spread of scores across keys* shrinks with position. Top-k
    compares across positions, so that shows up as a position bias rather than as a visibly
    degenerate score. A **monotone decay** across the returned bins is the failure mode; roughly
    flat is healthy.

    Run this on a real forward before training, not after: the symptom at the end of training is
    "the router just learned recency", which is many steps of confusion away from the cause.

    Parameters
    ----------
    scores : torch.Tensor
        ``(B, n_heads, Sk)``, as returned by :meth:`PrefixIndexer.score_keys`.
    n_bins : int
        Number of equal-width position bins.

    Returns
    -------
    centers : torch.Tensor
        ``(n_bins,)`` fractional position of each bin's centre, in ``[0, 1]``.
    variance : torch.Tensor
        ``(n_bins,)`` variance of the scores within each bin, pooled over batch and head. The
        recency tilt is removed first, so this measures the *content* spread rather than
        ``pos_slope``'s deterministic ramp.
    """
    if scores.dim() != 3:
        raise ValueError(f"scores must be (B, n_heads, Sk), got {tuple(scores.shape)}")
    k_len = scores.shape[-1]
    if n_bins <= 0 or n_bins > k_len:
        raise ValueError(f"n_bins must be in [1, {k_len}], got {n_bins}")

    valid = scores > (MASK_NEG / 2)
    edges = torch.linspace(0, k_len, n_bins + 1, dtype=torch.long)
    centers, variance = [], []
    for lo, hi in zip(edges[:-1].tolist(), edges[1:].tolist()):
        chunk, ok = scores[..., lo:hi], valid[..., lo:hi]
        centers.append((lo + hi) / (2 * k_len))
        n = ok.sum()
        if n < 2:
            variance.append(float("nan"))
            continue
        w = ok.to(scores.dtype)
        pos = torch.arange(hi - lo, device=scores.device, dtype=scores.dtype).expand_as(chunk)
        pos_c = pos - (w * pos).sum() / n
        chunk_c = chunk - (w * chunk).sum() / n
        denom = (w * pos_c * pos_c).sum()
        if denom > torch.finfo(scores.dtype).tiny:
            chunk_c = chunk_c - ((w * pos_c * chunk_c).sum() / denom) * pos_c
        residual = chunk_c[ok]
        variance.append((residual - residual.mean()).pow(2).sum().item() / (n.item() - 1))
    return (
        torch.tensor(centers, dtype=torch.float32),
        torch.tensor(variance, dtype=torch.float32),
    )
