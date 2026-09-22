# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


class TokenizedDocuments(IterableDataset):
    def __init__(
        self, root, sequence_length, subsets, *, seed=1000, shuffle_buffer=64, rank=0, world_size=1, take_from="head"
    ):
        self.paths = [path for subset in subsets for path in sorted((Path(root) / subset).glob("*.npy"))]
        self.sequence_length = sequence_length
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.rank = rank
        self.world_size = world_size
        self.take_from = take_from

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        workers = worker.num_workers if worker else 1
        paths = list(self.paths)
        random.Random(self.seed).shuffle(paths)
        paths = paths[self.rank * workers + worker_id :: self.world_size * workers]
        rng = random.Random(hash((self.seed, self.rank, worker_id)))
        buffer = []
        for path in paths:
            array = np.load(path, mmap_mode="r")
            doc_ids = json.loads(path.with_suffix(".json").read_text())["doc_ids"]
            order = list(range(array.shape[0]))
            rng.shuffle(order)
            for row in order:
                start = (
                    rng.randrange(array.shape[1] - self.sequence_length + 1)
                    if self.take_from == "random" and array.shape[1] > self.sequence_length
                    else 0
                )
                sample = {
                    "input_ids": torch.from_numpy(
                        np.array(array[row, start : start + self.sequence_length], dtype=np.int64)
                    ),
                    "doc_id": doc_ids[row],
                }
                if self.shuffle_buffer > 1:
                    buffer.append(sample)
                    if len(buffer) < self.shuffle_buffer:
                        continue
                    pick = rng.randrange(len(buffer))
                    buffer[pick], buffer[-1] = buffer[-1], buffer[pick]
                    sample = buffer.pop()
                yield sample
        rng.shuffle(buffer)
        yield from buffer


def collate_documents(samples):
    return {
        "input_ids": torch.stack([sample["input_ids"] for sample in samples]),
        "doc_ids": [sample["doc_id"] for sample in samples],
    }


def document_loader(
    root,
    sequence_length,
    subsets,
    *,
    batch_size=1,
    workers=2,
    seed=1000,
    shuffle_buffer=64,
    rank=0,
    world_size=1,
    take_from="head",
):
    dataset = TokenizedDocuments(
        root,
        sequence_length,
        subsets,
        seed=seed + sequence_length,
        shuffle_buffer=shuffle_buffer,
        rank=rank,
        world_size=world_size,
        take_from=take_from,
    )
    options = {"prefetch_factor": 4, "persistent_workers": True} if workers else {}
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=collate_documents,
        pin_memory=True,
        drop_last=True,
        **options,
    )


class DocumentCache:
    def __init__(self, root, *, kind):
        self.kind = kind
        self.locations = {}
        self.arrays = {}
        pattern = "*/*.json" if kind == "teacher" else "*/*.npz"
        for path in sorted(Path(root).glob(pattern)):
            if kind == "teacher":
                doc_ids = json.loads(path.read_text())["doc_ids"]
                array_path = path.with_suffix(".npy")
            else:
                with np.load(path, allow_pickle=False) as archive:
                    doc_ids = archive["doc_ids"]
                array_path = path
            self.locations.update({str(doc_id): (array_path, row) for row, doc_id in enumerate(doc_ids)})

    def batch(self, doc_ids, sequence_length, device, dtype):
        rows = []
        for doc_id in doc_ids:
            path, row = self.locations[doc_id]
            if path not in self.arrays:
                if self.kind == "teacher":
                    self.arrays[path] = np.load(path, mmap_mode="r")
                else:
                    with np.load(path, allow_pickle=False) as archive:
                        self.arrays[path] = archive["weights"]
            length = sequence_length if self.kind == "teacher" else sequence_length - 1
            rows.append(torch.from_numpy(np.array(self.arrays[path][row, :length])))
        return torch.stack(rows).to(device=device, dtype=dtype, non_blocking=True)


def parse_schedule(spec):
    return [tuple(map(int, stage.split(":"))) for stage in spec.split(",")]


def next_batch(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator
