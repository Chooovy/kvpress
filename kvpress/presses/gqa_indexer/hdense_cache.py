# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Cached dense final hidden states, the teacher for :mod:`~.fwkl`.

``lm_head`` is frozen, so the teacher distribution is a deterministic function of the dense
forward's final hidden state. Caching ``h`` rather than the logits is what makes this affordable:
``(L, 4096)`` against ``(L, 151936)`` is **37x smaller** -- 0.125 GiB per 16K document instead of
4.64 GiB.

Whether to cache at all is a wall-clock wash for a single run: the teacher costs one extra forward
per step (~+27% at 8K, ~79 min over 300 steps), and precomputing 2400 documents across 8 GPUs costs
the same ~79 min. It pays off from the **second** run onward -- sweeping the KL weight, the gate
budget, or the router architecture all reuse the same teacher, because the teacher depends on
nothing but the frozen backbone and the token ids.

Layout, mirroring :mod:`~.longce_weights`
-----------------------------------------
Two files per source shard, and the split is load-bearing:

* ``subset/shard.npy``  -- ``(n_docs, seq_len, hidden)`` float16, **memory-mapped**
* ``subset/shard.json`` -- ``doc_ids``, per-document token ``digest``, and ``meta``

**Not one ``.npz``, which is what LongCE uses.** An ``.npz`` member cannot be memory-mapped:
reading a single row forces the whole array into RAM. Measured here -- ``z["h"]`` on one shard
materialized **93.8 GiB in 83 s** to retrieve one 64 MB row, and eight ranks doing that would need
750 GiB. The layouts differ because the payloads differ by three orders of magnitude: LongCE stores
one 32 KB weight vector per document, so a whole shard is tens of MB and reading it once is free.
``h_dense`` is 64 MB *per document*.

``.npy`` + ``mmap_mode="r"`` reads only the header at open time and lets the page cache serve rows
on demand -- the same access pattern
:class:`~kvpress.presses.gqa_indexer.data.TokenizedDataset` uses on the corpus itself.

**float16, not bfloat16.** The teacher is consumed as an fp32 log-softmax, so what matters is
mantissa, not range: fp16 has 10 bits against bf16's 8. Hidden states run to ~1e2, far inside fp16's
6.5e4, so there is no overflow risk -- and ``numpy`` has no native bf16, which would force a
uint16 reinterpret and lose the dtype check that catches a corrupt shard.

Keyed by ``doc_id``, verified by digest
---------------------------------------
Positional keys break silently whenever world size, worker count or seed change -- the tensor would
still be the right *shape*. And unlike LongCE's per-position weights, a wrong ``h_dense`` does not
merely misweight the objective: it makes the router chase a target from a different document, which
would train it to *destroy* the current one. So the digest check is fatal rather than a warning.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

#: Bumped whenever the layout or the meaning of a field changes, so an old cache is refused
#: rather than silently reinterpreted.
CACHE_VERSION = 1


def token_digest(tokens: torch.Tensor | np.ndarray) -> int:
    """
    A cheap order-sensitive checksum of one document's token ids.

    Mirrors :mod:`~.longce_weights`'s digest so the two caches can be verified the same way. Not
    cryptographic -- it only has to catch "these are different tokens", which a sum alone would
    miss for a permutation.
    """
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.detach().cpu().numpy()
    arr = np.asarray(tokens, dtype=np.int64).reshape(-1)
    weights = np.arange(1, arr.size + 1, dtype=np.int64)
    return int((arr * weights).sum() % (2**61 - 1))


@dataclass(frozen=True)
class HDenseMeta:
    """What a cache was built with; a mismatched consumer is rejected rather than tolerated."""

    seq_len: int
    hidden_size: int
    model: str

    def to_json(self) -> dict:
        return {
            "version": CACHE_VERSION,
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "model": self.model,
        }

    @classmethod
    def from_json(cls, blob: dict) -> "HDenseMeta":
        version = blob.get("version")
        if version != CACHE_VERSION:
            raise ValueError(
                f"h_dense cache version {version} != {CACHE_VERSION}; rebuild it with "
                "scripts/precompute_hdense.py"
            )
        return cls(
            seq_len=int(blob["seq_len"]),
            hidden_size=int(blob["hidden_size"]),
            model=str(blob["model"]),
        )


def shard_cache_path(root: str | Path, subset: str, stem: str) -> Path:
    """The array file. Its metadata sidecar is the same path with ``.json``."""
    return Path(root) / subset / f"{stem}.npy"


def write_shard_cache(
    path: Path,
    *,
    hidden: np.ndarray,
    doc_ids: list[str],
    digests: list[int],
    meta: HDenseMeta,
) -> None:
    """
    Write one shard's array and its metadata sidecar, atomically.

    Temporary siblings then rename, so an interrupted precompute leaves complete shards and
    nothing half-written -- which is what makes the job resumable by skipping existing files. The
    ``.json`` is renamed **last**, so its presence is the completion marker: a reader that finds
    the sidecar knows the array beside it is whole.
    """
    if hidden.dtype != np.float16:
        raise ValueError(f"hidden must be float16, got {hidden.dtype}")
    if len(doc_ids) != hidden.shape[0] or len(digests) != hidden.shape[0]:
        raise ValueError(
            f"{hidden.shape[0]} rows against {len(doc_ids)} ids and {len(digests)} digests"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_npy = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp_npy, hidden)
    tmp_npy.rename(path)

    sidecar = path.with_suffix(".json")
    tmp_json = sidecar.with_name(sidecar.stem + ".tmp.json")
    with open(tmp_json, "w") as handle:
        json.dump(
            {"doc_ids": list(doc_ids), "digest": [int(d) for d in digests], "meta": meta.to_json()},
            handle,
        )
    tmp_json.rename(sidecar)


class HDenseCache:
    """
    Every cached shard under one root, keyed by ``doc_id``.

    Shards are memory-mapped, so a worker touches only the rows it draws and the page cache
    handles eviction. Opening costs a header read.
    """

    def __init__(self, root: str | Path, *, seq_len: int):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"no h_dense cache at {self.root}; build it with scripts/precompute_hdense.py"
            )
        self.seq_len = seq_len
        # Driven by the sidecars, not by the arrays: the .json is renamed last, so a shard whose
        # array is present but whose metadata is not was interrupted mid-write and is skipped.
        self._sidecars = sorted(self.root.glob("*/*.json"))
        if not self._sidecars:
            raise FileNotFoundError(
                f"{self.root} holds no completed shards; the precompute did not finish, or the "
                "root points at the wrong directory"
            )
        self._open: dict[Path, np.ndarray] = {}
        self._locate: dict[str, tuple[Path, int]] = {}
        self._digest: dict[str, int] = {}
        self.meta: HDenseMeta | None = None
        for sidecar in self._sidecars:
            with open(sidecar) as handle:
                blob = json.load(handle)
            meta = HDenseMeta.from_json(blob["meta"])
            if self.meta is None:
                self.meta = meta
            elif meta != self.meta:
                raise ValueError(
                    f"{sidecar.name} was built with {meta}, but earlier shards used {self.meta}. "
                    "Mixing settings in one root would hand different documents teachers from "
                    "different models."
                )
            array_path = sidecar.with_suffix(".npy")
            if not array_path.is_file():
                raise FileNotFoundError(
                    f"{sidecar} has no array beside it at {array_path.name}"
                )
            for row, doc_id in enumerate(blob["doc_ids"]):
                self._locate[str(doc_id)] = (array_path, row)
                self._digest[str(doc_id)] = int(blob["digest"][row])

        if self.meta is not None and seq_len > self.meta.seq_len:
            raise ValueError(
                f"cache was built at seq_len={self.meta.seq_len} but this stage reads {seq_len}. "
                "The teacher for the extra positions was never computed, so it cannot be "
                "recovered -- rebuild at the longer width."
            )

    def __contains__(self, doc_id: str) -> bool:
        return str(doc_id) in self._locate

    def summary(self) -> str:
        return (
            f"{len(self._locate)} documents in {len(self._sidecars)} shard(s) under {self.root}, "
            f"built at seq_len={self.meta.seq_len} hidden={self.meta.hidden_size} "
            f"model={self.meta.model}; reading a {self.seq_len}-token prefix"
            + ("" if self.seq_len == self.meta.seq_len else
               " (SHORTER than the cache: stored digests cover the full width, so the token "
               "check is skipped for this stage)")
        )

    def lookup(self, doc_id: str, tokens: torch.Tensor) -> np.ndarray:
        """
        ``(seq_len, hidden)`` float16 for this document, after verifying the tokens match.

        The array is memory-mapped, so this reads one row's pages rather than the whole shard.
        The returned view is copied by the caller before it becomes a tensor -- a mapped page can
        be evicted under memory pressure, and torch would then read freed memory (the same reason
        ``TokenizedDataset`` copies out of its mmap).

        The digest check is **fatal**, not a warning. A wrong teacher does not merely misweight the
        objective -- it makes the router match a distribution from a different document, i.e. it
        would actively train the router to destroy the one in front of it. Nothing downstream could
        detect that: the shape is right and the loss descends.
        """
        key = str(doc_id)
        if key not in self._locate:
            raise KeyError(f"{doc_id} is not in {self.root}")
        array_path, row = self._locate[key]
        if array_path not in self._open:
            self._open[array_path] = np.load(array_path, mmap_mode="r")
        array = self._open[array_path]

        # Stored digests cover the full cached width, so only an equal-width read compares
        # directly. A shorter stage reads a prefix, which the digest cannot verify -- flagged in
        # the summary rather than silently skipped.
        if self.seq_len == self.meta.seq_len:
            want = token_digest(tokens[: self.seq_len])
            got = self._digest[key]
            if want != got:
                raise ValueError(
                    f"token digest mismatch for {doc_id}: cache {got}, batch {want}. The cached "
                    "teacher belongs to different tokens, so matching it would train the router "
                    "against another document. Rebuild the cache for this corpus/seq_len."
                )
        return array[row, : self.seq_len]
