import json

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from kvpress.indexmem.training.data import DocumentCache, document_loader
from kvpress.indexmem.training.prepare import longce_for_document


def test_training_document_order_matches_original_reader(tmp_path):
    subset = tmp_path / "s"
    subset.mkdir()
    for shard in range(4):
        np.save(subset / f"{shard}.npy", np.arange(96).reshape(6, 16) + shard * 1000)
        (subset / f"{shard}.json").write_text(json.dumps({"doc_ids": [f"{shard}:{row}" for row in range(6)]}))
    batches = list(document_loader(tmp_path, 8, ["s"], seed=1000, shuffle_buffer=4, batch_size=2, workers=0))
    observed = [doc_id for batch in batches for doc_id in batch["doc_ids"]]
    assert observed == [
        "2:4",
        "2:1",
        "2:3",
        "0:3",
        "0:1",
        "0:0",
        "2:2",
        "2:5",
        "0:5",
        "2:0",
        "0:4",
        "1:0",
        "1:2",
        "1:4",
        "1:1",
        "1:3",
        "3:5",
        "3:2",
        "3:0",
        "3:4",
        "3:1",
        "1:5",
        "3:3",
        "0:2",
    ]
    for batch in batches:
        for doc_id, tokens in zip(batch["doc_ids"], batch["input_ids"]):
            shard, row = map(int, doc_id.split(":"))
            torch.testing.assert_close(tokens, torch.arange(row * 16, row * 16 + 8) + shard * 1000)


def test_training_reads_existing_teacher_and_longce_cache_schemas(tmp_path):
    teacher_root = tmp_path / "teacher" / "s"
    weights_root = tmp_path / "weights" / "s"
    teacher_root.mkdir(parents=True)
    weights_root.mkdir(parents=True)
    hidden = np.arange(2 * 12 * 8, dtype=np.float16).reshape(2, 12, 8)
    weights = np.arange(22, dtype=np.float16).reshape(2, 11)
    np.save(teacher_root / "0.npy", hidden)
    (teacher_root / "0.json").write_text(
        json.dumps(
            {
                "doc_ids": ["a", "b"],
                "meta": {"version": 1, "seq_len": 12, "hidden_size": 8, "model": "tiny"},
                "digest": [0, 1],
            }
        )
    )
    np.savez(
        weights_root / "0.npz",
        doc_ids=np.asarray(["a", "b"]),
        weights=weights,
        checksums=np.asarray([["0"], ["1"]]),
        meta=np.asarray(json.dumps({"version": 3, "seq_len": 12})),
    )
    teacher = DocumentCache(teacher_root.parent, kind="teacher").batch(["b", "a"], 8, "cpu", torch.float32)
    longce = DocumentCache(weights_root.parent, kind="longce").batch(["b", "a"], 8, "cpu", torch.float32)
    torch.testing.assert_close(teacher, torch.from_numpy(hidden[[1, 0], :8]).float())
    torch.testing.assert_close(longce, torch.from_numpy(weights[[1, 0], :7]).float())


def test_training_longce_windows_align_with_each_target():
    torch.manual_seed(6)
    config = Qwen3Config(
        vocab_size=31,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
    )
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    tokens = torch.randint(31, (1, 12))
    actual = longce_for_document(model, tokens, truncation=4, window=3, gamma=5.0, chunk_size=3)
    expected = []
    with torch.no_grad():
        for target in range(1, 12):
            if target < 4:
                expected.append(torch.tensor(1.0))
                continue
            start = ((target - 4) // 3) * 3
            full = model(tokens[:, :target], use_cache=False).logits[:, -1]
            short = model(tokens[:, start:target], use_cache=False).logits[:, -1]
            label = tokens[:, target]
            difference = F.cross_entropy(short.float(), label) - F.cross_entropy(full.float(), label)
            expected.append(difference.exp().clamp(max=5.0))
    torch.testing.assert_close(actual, torch.stack(expected), atol=1e-6, rtol=1e-5)
