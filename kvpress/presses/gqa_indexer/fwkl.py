# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Forward-KL against the frozen backbone's own dense output distribution.

The objective, per position ``t``::

    L = KL(p_dense_t || p_gated_t)      over the full vocabulary

where ``p_dense`` is the *same* model run without the gate. Self-distillation: no second model, no
teacher checkpoint, and the target moves only if the backbone does (it does not -- the backbone is
frozen).

Why this rather than the LM cross-entropy
-----------------------------------------
CE anchors on the gold token, so it only carries signal where the model would have been right. The
one thing it cannot express is the distinction that eviction turns on:

* **redundant key** -- evicting it leaves the prediction correct, just less confident. CE moves a
  little; the router should not spend budget here.
* **irreplaceable key** -- evicting it changes the prediction entirely, but if the gold token was
  never top-1 (a hard position), ``-log p(gold)`` barely moves. **CE is nearly blind to it.**

A needle is the second case. Measured on this corpus, needle tokens are *easier* than average under
long context (``key_L - all_L = -1.7``), which is also why LongCE's ``exp(L_short - L_long)`` weight
does not target them. KL sees the whole distribution, so "dense answered the needle and the gated
run did not" is a large, direct signal regardless of what the gold token's rank was.

This is what KVzip reaches at 94.23 RULER (against this router's 73.71) by way of a reconstruction
objective. KL asks the same question -- can this key's information be recovered from what is left --
but against the model's own output distribution rather than a proxy for it.

What KL does NOT fix
--------------------
**Position dilution.** KL is dense over the vocabulary but averages over positions exactly as CE
does, so a needle is still one position in 8191. That is a weighting/data problem
(LongCE, ``--sft-ruler``, needle-augmented mixes), and swapping the per-position loss does not touch
it.

The no-op hole, which is why the pin is mandatory
-------------------------------------------------
``g`` constant along the key axis leaves ``softmax(a + g) == softmax(a)``, so every layer's output
is unchanged, the logits are unchanged, and **KL is exactly 0** -- a global optimum reached with no
ranking learned. Identical to the hole ``gate_pin`` exists to close for the LM loss, and to the one
that sinks a gated-vs-dense *attention-weight* KL. :func:`fwkl_step` therefore rejects
``pin_mode="none"`` rather than warning.

Memory
------
The naive form needs two ``(L, V)`` logit tensors -- 4.64 GiB each in bf16 at 16K on Qwen3's
151936-wide vocabulary, plus fp32 log-softmax. Two things avoid that:

1. **The teacher is cached as hidden states, not logits.** ``lm_head`` is frozen, so
   ``logits = lm_head(h)`` is a deterministic function of ``h_dense``, which is ``(L, 4096)`` --
   **37x smaller** (0.125 GiB at 16K against 4.64). See :mod:`~.hdense_cache`.
2. **The vocabulary axis is never materialized whole.** Both sides' logits are formed in row chunks
   and freed per chunk, so the peak is ``chunk * V`` (0.58 GiB at ``chunk=2048``) rather than
   ``L * V``. Sound because KL decomposes over positions -- unlike a power mean, no outer
   nonlinearity needs every position first.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: Rows per ``lm_head`` call. 2048 puts the transient logits at 0.58 GiB on a 151936 vocabulary,
#: against 2.32 GiB for a whole 8K sequence. A pure memory/speed knob: KL is a per-position sum, so
#: the result is chunk-invariant.
DEFAULT_KL_CHUNK = 2048


def fwkl_chunked(
    hidden_gated: torch.Tensor,
    hidden_dense: torch.Tensor,
    lm_head: torch.nn.Module,
    *,
    chunk: int = DEFAULT_KL_CHUNK,
    temperature: float = 1.0,
    reverse: bool = False,
    recompute: bool = True,
) -> tuple[torch.Tensor, dict]:
    """
    ``mean_t KL(p_dense_t || p_gated_t)``, with the vocabulary axis never materialized whole.

    Parameters
    ----------
    hidden_gated : torch.Tensor
        ``(B, L, H)`` final hidden states from the **gated** forward. Carries gradient.
    hidden_dense : torch.Tensor
        ``(B, L, H)`` from the ungated forward, or from the cache. Detached here regardless --
        the teacher is a target, and letting gradient into it would train the router to move the
        target rather than to match it.
    lm_head : torch.nn.Module
        The frozen output projection. Applied to both sides inside the chunk loop, so the two
        logit blocks share one weight read.
    chunk : int
        Rows per ``lm_head`` call.
    temperature : float
        Softmax temperature applied to both sides. 1.0 is the plain objective.
    reverse : bool
        Swap to **reverse** KL, ``KL(p_gated || p_dense)``, TrimKV's ``rvkl``. The two differ in
        what they punish, and for eviction the difference is not cosmetic:

        * **Forward** ``KL(p_dense || p_gated)`` is *mass-covering*. It is large wherever the
          teacher puts mass and the student does not, so dropping the key that carried the
          needle is expensive. It tolerates the student spreading mass where the teacher has
          none.
        * **Reverse** ``KL(p_gated || p_dense)`` is *mode-seeking*. It punishes the student for
          putting mass where the teacher has none, and is **blind to a mode the student misses
          entirely** -- if the gated run drops the needle and confidently predicts something
          else, reverse KL only charges for that one wrong mode, whereas forward KL charges for
          the whole missing distribution.

        So forward is the one that matches "do not lose an irreplaceable key", which is why it is
        the default. Reverse is here as the ablation that tests whether that reasoning holds.

    Returns
    -------
    (loss, stats)
        ``loss`` is the mean over positions. ``stats`` carries diagnostics that are cheap here and
        impossible to recover later: ``kl_max`` (the worst position -- this is where an
        irreplaceable key was dropped), ``kl_p90``, and ``agree_top1`` (fraction of positions where
        both sides' argmax matches, i.e. how often the gate changed the model's answer at all).
    """
    if hidden_gated.shape != hidden_dense.shape:
        raise ValueError(
            f"gated {tuple(hidden_gated.shape)} and dense {tuple(hidden_dense.shape)} hidden "
            "states must have the same shape; a length mismatch usually means the cache was "
            "built at a different seq_len"
        )
    hidden_dense = hidden_dense.detach()
    flat_gated = hidden_gated.reshape(-1, hidden_gated.shape[-1])
    flat_dense = hidden_dense.reshape(-1, hidden_dense.shape[-1])
    n_rows = flat_gated.shape[0]

    total = flat_gated.new_zeros((), dtype=torch.float32)
    kl_max = flat_gated.new_zeros((), dtype=torch.float32)
    agree = flat_gated.new_zeros((), dtype=torch.float32)
    # Kept for a percentile, which needs the values rather than a running reduction. One fp32
    # scalar per position -- 32 KiB at 8K, against 2.3 GiB for the logits.
    per_row = []

    def _chunk_kl(h_gated_chunk, h_dense_chunk):
        """One chunk's per-row KL. Factored out so it can be gradient-checkpointed."""
        # fp32 for both log-softmaxes: the KL of two nearly-identical distributions is a small
        # difference of large numbers, and bf16's 8 mantissa bits lose it. This is the one place
        # the precision matters -- the logits themselves can stay in the model's dtype.
        logits_gated = lm_head(h_gated_chunk).float()
        with torch.no_grad():
            logits_dense = lm_head(h_dense_chunk).float()
        if temperature != 1.0:
            logits_gated = logits_gated / temperature
            logits_dense = logits_dense / temperature

        log_p_dense = F.log_softmax(logits_dense, dim=-1)
        log_p_gated = F.log_softmax(logits_gated, dim=-1)
        # Both forms are computed from the exp of a log-softmax rather than a second softmax, so
        # the weighting distribution and the log-ratio agree bit for bit on the normalizer.
        if reverse:
            # KL(p_gated || p_dense) = sum_v p_gated * (log p_gated - log p_dense).
            # p_gated CARRIES GRADIENT here, unlike the forward form where the weights are the
            # detached teacher -- the student's own probabilities weight the objective, which is
            # what makes reverse KL mode-seeking.
            p_gated = log_p_gated.exp()
            kl_rows = (p_gated * (log_p_gated - log_p_dense)).sum(-1)
        else:
            p_dense = log_p_dense.exp()
            kl_rows = (p_dense * (log_p_dense - log_p_gated)).sum(-1)
        agree_rows = (logits_dense.argmax(-1) == logits_gated.argmax(-1)).sum()
        return kl_rows, agree_rows

    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        if recompute and torch.is_grad_enabled() and flat_gated.requires_grad:
            # Recompute this chunk's logits in backward instead of retaining them.
            #
            # THE CHUNK SIZE DOES NOT BOUND THIS. `chunk` bounds the transient peak, but every
            # chunk's activations stay in the autograd graph until `total.backward()`, so the
            # RETAINED cost is `n_chunks * chunk * V = n_rows * V` regardless of `chunk`.
            # Reverse KL retains three fp32 (rows, V) tensors -- p_gated, log_p_gated,
            # log_p_dense -- because d/dlogits needs all three:
            #
            #     8K:  3 * 8192 * 151936 * 4 B = 13.9 GiB
            #     16K: 3 * 16384 * 151936 * 4 B = 27.8 GiB
            #
            # which is why 16K RVKL OOMs at 1024, 512 AND 256 identically (measured: LongCE's
            # 79.8 GiB at 16K + 27.8 = 107.6 against 95 available). Forward KL retains one such
            # tensor and is correspondingly cheaper. Checkpointing trades one extra lm_head
            # matmul per chunk in backward for dropping the (rows, V) retention entirely.
            kl_rows, agree_rows = torch.utils.checkpoint.checkpoint(
                _chunk_kl, flat_gated[start:stop], flat_dense[start:stop], use_reentrant=False
            )
        else:
            kl_rows, agree_rows = _chunk_kl(flat_gated[start:stop], flat_dense[start:stop])

        total = total + kl_rows.sum()
        kl_max = torch.maximum(kl_max, kl_rows.detach().max())
        with torch.no_grad():
            agree = agree + agree_rows
            per_row.append(kl_rows.detach())

    loss = total / n_rows
    with torch.no_grad():
        rows = torch.cat(per_row)
        stats = {
            "fwkl": float(loss),
            "kl_reverse": reverse,
            # THE readout for "did the gate destroy something irreplaceable somewhere". The mean
            # is dominated by the many positions the gate barely touches; the tail is the signal.
            "kl_max": float(kl_max),
            "kl_p90": float(rows.quantile(0.90)),
            "kl_p99": float(rows.quantile(0.99)),
            # 1.0 means the gate never changed the model's answer. Falling means it did -- which
            # is what the objective is trying to prevent, and a more interpretable number than
            # the KL's absolute scale.
            "agree_top1": float(agree / n_rows),
        }
    return loss, stats
