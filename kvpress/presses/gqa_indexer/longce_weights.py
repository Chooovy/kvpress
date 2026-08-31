# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LongCE key-token weights: computed once per document, cached, and read back by the trainer.

The weight
----------
LongCE (Fang et al., ICLR 2025) reweights each token by how much the *long* context helped it
against a *truncated short* context:

    ``w_t = min(exp(L_t^short - L_t^long), gamma)``

with ``gamma = 5`` (the paper's ``Isoft``, Eq. 7). The paper is explicit about the shape this
produces: *"too easy tokens (both short and long context give accurate prediction) and too hard
tokens (neither short or long context predicts correctly) will have a weight around 1, while those
long-context-dependent tokens will be upweighted above 1."* So this is a mostly-uniform weighting
with a small upweighted tail, not a sparse selection -- measured on this corpus: median weight
exactly 1.000, ``weight_participation`` 0.66-0.87, ~1% of positions above the key-token threshold.

Why this, and not the dense-vs-sparse gap
-----------------------------------------
The previous arm weighted by ``clamp(L^dense - L^sparse, 0) + lambda`` and took RULER from 66.24 to
~35.2. That gap is strongly rank-correlated with the loss it multiplies, so the objective
degenerated into a power mean over the top ~15% of positions -- which is where irreducible entropy
lives, and is exactly where routing cannot help. LongCE's contrast is short-vs-long rather than
dense-vs-sparse, and :mod:`evaluation.probe_longce_key_tokens` measured
``spearman(w, L_long) = -0.001 .. -0.029`` across 8K/16K/32K on this corpus. That decorrelation is
the whole reason this objective is worth running, and it was measured before training rather than
inferred.

Why a cache rather than recomputing per step
--------------------------------------------
1. **Cost.** The sliding-window loop is ``(L - K) / d`` extra forwards of ``K + d`` tokens each. At
   ``K=1024, d=1024, L=16384`` that is 15 forwards of 2048 tokens, ~1.9x the long forward's token
   count on top of the step itself. The paper measures +43% to +79% wall-clock for *full-parameter*
   finetuning; here only the indexer trains, so the main step is comparatively cheap and the same
   overhead lands harder.
2. **Correctness, which matters more.** The backbone is **frozen**, so ``L^short`` and ``L^long``
   do not depend on the router -- the weight is a property of the *data*. Recomputing it per step
   would pay full price for a constant. It would also reintroduce, from the other side, the drift
   the delta run showed: its ``dense_loss`` swung 1.19-2.42 between steps, so each step's weights
   were scaled by that batch's difficulty rather than anything intrinsic. A cache removes that by
   construction.

Alignment is the real risk
--------------------------
A cached weight vector attached to the wrong tokens is a **silent** failure: the loss still falls,
the diagnostics still look sane, and only the benchmark reveals it. Two mechanisms guard it:

* the cache is keyed by the corpus's own ``doc_id``, not by shard/row position, so re-sharding or
  re-ordering the corpus cannot quietly repoint an entry;
* every entry carries ``blake2b`` digests of its token prefix **at each stage width**, and the
  reader verifies the digest for the length actually drawn. One digest at the cache's full width
  would be unverifiable from an 8K stage, which draws only a prefix -- see :func:`token_checksum`.

Deliberate deviation from the reference
--------------------------------------
``LongPPL/finetune/finetune.py``'s ``loss_weight`` initializes ``loss_discrepancy = ones(...)``,
leaving the first ``trunc_len`` positions at weight 1. Those positions have no shorter context to
compare against, so 1 there is fabricated rather than measured. This module stores 1.0 for them
*and* records ``scored_from``, so the trainer can distinguish "measured 1.0" from "not measured" and
choose. At ``K=1024, L=16384`` that is 6% of the sequence; at ``K=4096, L=8192`` it would have been
half of it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

#: LongCE's ``gamma`` (``thre`` in the reference): the ceiling on ``exp(L_short - L_long)``.
DEFAULT_GAMMA = 5.0

#: LongCE's ``K`` (``trunc_len``): how much context the short pass keeps.
#:
#: 1024 rather than the reference default of 4096. Two reasons, both measured: the paper's own
#: ablation (Table 7) has ``K=1k, d=1k`` beating ``K=4k, d=1k`` on RULER (55.9 vs 49.7 at 200 steps)
#: while costing less (+43% vs +79% wall-clock); and on this corpus 1024 gave the largest
#: discrepancy signal and the lowest ``spearman(w, L_long)`` of the values swept.
DEFAULT_TRUNC_LEN = 1024

#: LongCE's ``d`` (``internal``/sliding window): positions scored per short forward pass.
DEFAULT_WINDOW = 1024

#: Cache layout version, so a reader refuses a layout it does not understand rather than
#: misinterpreting one.
CACHE_VERSION = 3


def token_checksum(tokens: np.ndarray) -> str:
    """
    A stable 16-hex-digit ``blake2b`` digest of a token row.

    Canonicalized to ``uint32`` little-endian first: the corpus stores ``uint32`` but the loader
    hands back ``int64``, and hashing whichever dtype happened to arrive would make a correct pair
    look like a mismatch.

    The digest covers **exactly the tokens passed in**, which is why the cache stores one per stage
    width rather than one per document. A single digest at the cache's full width could not be
    checked from an 8K stage -- that stage only ever holds a 8192-token prefix -- so verification
    would have to be skipped for the shorter stages, i.e. skipped where the curriculum spends half
    its steps.
    """
    canonical = np.ascontiguousarray(tokens, dtype="<u4")
    return hashlib.blake2b(canonical.tobytes(), digest_size=8).hexdigest()


def longce_weights(
    long_loss: torch.Tensor,
    short_loss: torch.Tensor,
    scored: torch.Tensor,
    *,
    gamma: float = DEFAULT_GAMMA,
) -> torch.Tensor:
    """
    ``w_t = min(exp(L^short_t - L^long_t), gamma)``, ``(N,)`` fp32, unscored positions at 1.0.

    1.0 is the neutral element of this weighting -- the paper's "weight around 1" for tokens the
    long context does not help -- so an unscored position falls back to plain-mean behaviour rather
    than being dropped or given an invented preference. The ``scored`` mask travels separately so
    the trainer can still exclude them.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    if long_loss.shape != short_loss.shape:
        raise ValueError(
            f"long {tuple(long_loss.shape)} and short {tuple(short_loss.shape)} disagree"
        )
    weights = torch.exp((short_loss - long_loss).float()).clamp(max=gamma)
    return torch.where(scored, weights, torch.ones_like(weights))


@dataclass(frozen=True)
class WeightCacheMeta:
    """What a cache was built with, so a mismatched consumer is rejected rather than tolerated."""

    seq_len: int
    trunc_len: int
    window: int
    gamma: float
    model: str
    #: First position (on the next-token index) with a short-context counterfactual. Equals
    #: ``trunc_len - 1``; stored explicitly so a reader never re-derives the off-by-one that
    #: ``per_token_ce``'s next-token indexing makes easy to get wrong.
    scored_from: int
    #: Widths at which token digests were recorded. A stage whose ``seq_len`` is absent here cannot
    #: be verified, and the reader refuses it rather than training on unverified weights.
    checksum_widths: tuple[int, ...]

    def to_json(self) -> dict:
        return {
            "version": CACHE_VERSION,
            "seq_len": self.seq_len,
            "trunc_len": self.trunc_len,
            "window": self.window,
            "gamma": self.gamma,
            "model": self.model,
            "scored_from": self.scored_from,
            "checksum_widths": list(self.checksum_widths),
        }

    @classmethod
    def from_json(cls, payload: dict) -> "WeightCacheMeta":
        version = payload.get("version")
        if version != CACHE_VERSION:
            raise ValueError(
                f"weight cache version {version} != {CACHE_VERSION} expected. The layout changed; "
                "re-run scripts/precompute_longce_weights.py rather than reading it as if it "
                "matched -- a stale layout would attach weights to the wrong positions."
            )
        return cls(
            seq_len=int(payload["seq_len"]),
            trunc_len=int(payload["trunc_len"]),
            window=int(payload["window"]),
            gamma=float(payload["gamma"]),
            model=str(payload["model"]),
            scored_from=int(payload["scored_from"]),
            checksum_widths=tuple(int(w) for w in payload["checksum_widths"]),
        )


def shard_cache_path(root: str | Path, subset: str, shard_stem: str) -> Path:
    """Where one source shard's weights live. Mirrors the corpus's ``subset/shard`` layout."""
    return Path(root) / subset / f"{shard_stem}.npz"


def write_shard_cache(
    path: Path,
    *,
    doc_ids: list[str],
    weights: np.ndarray,
    checksums: np.ndarray,
    meta: WeightCacheMeta,
) -> None:
    """
    Write one shard's weights, atomically.

    ``weights`` is fp16 ``(n_docs, seq_len - 1)``: 32 KB per 16K document, so storage does not
    arise as a constraint. fp16 suffices because the weight is a *relative* multiplier in
    ``[0, gamma]`` and the loss it scales stays fp32.

    ``checksums`` is ``(n_docs, len(meta.checksum_widths))`` of 16-char digests, aligned to
    ``meta.checksum_widths`` by column.

    Written to a temporary sibling and renamed, so an interrupted run leaves either a complete shard
    or nothing -- never a truncated file a later reader would treat as authoritative.
    """
    if weights.dtype != np.float16:
        raise ValueError(f"weights must be float16, got {weights.dtype}")
    n = len(doc_ids)
    if weights.shape[0] != n or checksums.shape[0] != n:
        raise ValueError(
            f"doc_ids {n}, weights {weights.shape[0]} and checksums {checksums.shape[0]} must "
            "describe the same number of documents"
        )
    if checksums.shape[1] != len(meta.checksum_widths):
        raise ValueError(
            f"checksums has {checksums.shape[1]} columns but meta lists "
            f"{len(meta.checksum_widths)} widths"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.tmp")
    # Written through an open handle rather than by passing `tmp` as a name: np.savez APPENDS ".npz"
    # to a str/Path target that lacks it, so it would create "....npz.tmp.npz" and the rename below
    # would fail on a file that does not exist. A handle is taken as-is.
    with open(tmp, "wb") as handle:
        np.savez(
            handle,
            doc_ids=np.array(doc_ids, dtype=object).astype("U64"),
            weights=weights,
            checksums=checksums.astype("U16"),
            meta=np.array(json.dumps(meta.to_json())),
        )
    tmp.replace(path)


class LongCEWeightCache:
    """
    Every cached shard under one root, keyed by ``doc_id``.

    Keyed by ``doc_id`` rather than ``(shard, row)`` because the loader shuffles rows and partitions
    shards by ``(rank, worker)``: a positional key would silently break whenever the world size,
    worker count or seed changed, and "silently" is the operative word -- the weights would still be
    the right *shape*.

    Shards are opened lazily and cached, so a training worker touches only the shards it draws from.
    ``.npz`` members decompress per-access anyway, which is why nothing is memory-mapped.
    """

    def __init__(self, root: str | Path, *, seq_len: int):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"no LongCE weight cache at {self.root}; build it with "
                "scripts/precompute_longce_weights.py"
            )
        self.seq_len = seq_len
        self._paths = sorted(self.root.glob("*/*.npz"))
        if not self._paths:
            raise FileNotFoundError(
                f"{self.root} holds no .npz shards; the precompute did not finish, or the root "
                "points at the wrong directory"
            )
        self._open: dict[Path, dict] = {}
        # doc_id -> (shard path, row within that shard's arrays)
        self._locate: dict[str, tuple[Path, int]] = {}
        self.meta: WeightCacheMeta | None = None
        for path in self._paths:
            with np.load(path, allow_pickle=False) as handle:
                meta = WeightCacheMeta.from_json(json.loads(str(handle["meta"])))
                ids = handle["doc_ids"]
            if self.meta is None:
                self.meta = meta
            elif meta != self.meta:
                raise ValueError(
                    f"{path.name} was built with {meta}, but earlier shards used {self.meta}. "
                    "Mixing settings in one cache root would apply different weightings to "
                    "different documents; rebuild into separate roots."
                )
            for row, doc_id in enumerate(ids):
                self._locate[str(doc_id)] = (path, row)

        assert self.meta is not None
        if seq_len > self.meta.seq_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds the cache's width {self.meta.seq_len}. Those positions "
                f"were never scored; rebuild with --seq-len {seq_len}."
            )
        if seq_len not in self.meta.checksum_widths:
            raise ValueError(
                f"the cache records token digests at widths {list(self.meta.checksum_widths)}, not "
                f"at seq_len={seq_len}, so a row drawn at this length cannot be verified. Rebuild "
                f"with --checksum-widths including {seq_len} -- training on unverified weights is "
                "exactly the silent misalignment this cache exists to prevent."
            )
        self._checksum_column = self.meta.checksum_widths.index(seq_len)

    def __len__(self) -> int:
        return len(self._locate)

    def doc_ids(self) -> list[str]:
        return sorted(self._locate)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._locate

    def _shard(self, path: Path) -> dict:
        if path not in self._open:
            with np.load(path, allow_pickle=False) as handle:
                self._open[path] = {
                    "weights": handle["weights"],
                    "checksums": handle["checksums"],
                }
        return self._open[path]

    def lookup(self, doc_id: str, tokens: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Weights for ``doc_id``, ``(seq_len - 1,)`` fp32, after verifying ``tokens``.

        ``tokens`` must be the ``seq_len`` row the loader actually drew. Verification is the point:
        a mismatch means the weights belong to different text, which would optimize the wrong
        positions while leaving every visible number intact, so it raises.

        Truncating the cached vector to this stage's length is sound because the losses are causal:
        position ``i``'s long-context loss depends only on tokens ``<= i``, and its short-context
        loss only on its own window, which is also fully contained in the prefix. So a weight
        computed at width 16384 is *identical* to one computed at 8192 for every ``i < 8191``. That
        is what lets one cache serve every stage of the curriculum -- and it is why the loader must
        use ``--take-from head``, since a random window would not be a prefix at all.
        """
        located = self._locate.get(doc_id)
        if located is None:
            raise KeyError(doc_id)
        path, row = located
        shard = self._shard(path)

        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().numpy()
        if tokens.shape[-1] != self.seq_len:
            raise ValueError(
                f"expected a {self.seq_len}-token row to verify against, got {tokens.shape[-1]}"
            )
        expected = str(shard["checksums"][row, self._checksum_column])
        actual = token_checksum(tokens)
        if actual != expected:
            raise RuntimeError(
                f"token checksum mismatch for doc {doc_id} at width {self.seq_len}: the cache "
                f"recorded {expected}, the loader drew {actual}. The cached weights belong to "
                "different tokens, so training on them would silently optimize the wrong "
                "positions. The corpus shard changed since the cache was built, or --take-from is "
                "not 'head' -- rebuild with scripts/precompute_longce_weights.py."
            )
        return shard["weights"][row, : self.seq_len - 1].astype(np.float32)

    def summary(self) -> dict:
        """Totals for the startup log line that says what the run will train on."""
        assert self.meta is not None
        return {
            "root": str(self.root),
            "shards": len(self._paths),
            "documents": len(self._locate),
            "seq_len": self.seq_len,
            "cache_seq_len": self.meta.seq_len,
            "trunc_len": self.meta.trunc_len,
            "window": self.meta.window,
            "gamma": self.meta.gamma,
            "scored_from": self.meta.scored_from,
        }


def longce_weighted_loss(
    sparse_loss: torch.Tensor,
    weights: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    scored: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    ``sum_t w_t L_t / sum_t w_t``, plus the diagnostics that say whether the weighting did anything.

    Deliberately the same normalization as :func:`~.delta_loss.delta_weighted_loss` -- by ``sum w``
    rather than ``N`` -- so the reported loss stays on the scale of the plain mean it replaces,
    ``--peak-lr`` transfers, and the number is readable against the plain run's curve. It also makes
    the objective invariant to a uniform rescale of ``w``, so only the *relative* weighting matters.

    Parameters
    ----------
    sparse_loss : torch.Tensor
        ``(N,)`` per-token loss from the gated pass, carrying the router's graph.
    weights : torch.Tensor
        ``(N,)`` cached LongCE weights, already on device. Detached here rather than trusting the
        caller: these come from a file, but a stray ``requires_grad`` would add a term that rewards
        making the loss large where the weight is large.
    mask : torch.Tensor, optional
        ``(N,)`` valid-label mask. Invalid positions get weight 0.
    scored : torch.Tensor, optional
        ``(N,)`` marking positions that had a short-context counterfactual. Supplied only for the
        diagnostics -- the weights already encode 1.0 for unscored positions.

    Returns
    -------
    (loss, stats)
        ``stats`` carries ``weight_participation`` -- ``(sum w)^2 / (n sum w^2)``, the effective
        *fraction* of positions carrying the objective, and the number to judge this run by. The
        failed delta arm sat at 0.13-0.18; this weighting measured 0.66-0.87 offline, and a value
        near 1.0 would mean the weighting is doing nothing at all whatever the loss says.
    """
    weights = weights.detach()
    if mask is not None:
        weights = weights * mask.to(weights.dtype)
    total = weights.sum()
    if not torch.isfinite(total) or total <= 0:
        raise RuntimeError(
            f"LongCE weights sum to {float(total)}, so the loss is undefined. Every weight is 0, "
            "which means the valid-label mask is empty rather than that the weighting is extreme."
        )
    loss = (weights * sparse_loss).sum() / total

    with torch.no_grad():
        active = weights > 0
        n_active = int(active.sum())
        sum_sq = float((weights * weights).sum())
        stats = {
            "weight_sum": float(total),
            "weight_participation": (
                float(total) ** 2 / (n_active * sum_sq) if n_active and sum_sq > 0 else 0.0
            ),
            "n_weighted": n_active,
        }
        live = weights[active]
        if n_active:
            stats["weight_mean"] = float(live.mean())
            stats["weight_median"] = float(live.median())
            # The paper's shape: most tokens near 1, a small tail upweighted. `upweighted_frac`
            # is this objective's analogue of delta_positive_frac -- near 0 means the cache is
            # effectively all ones and this is the plain mean with extra machinery.
            stats["weight_upweighted_frac"] = float((live > 1.01).float().mean())
            stats["weight_max"] = float(live.max())
        if scored is not None:
            valid = mask if mask is not None else torch.ones_like(scored, dtype=torch.bool)
            n_valid = int(valid.sum())
            stats["scored_frac"] = float((scored & valid).sum()) / n_valid if n_valid else 0.0
    return loss, stats
