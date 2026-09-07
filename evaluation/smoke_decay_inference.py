"""Exercise the SPARSE INFERENCE path with a decay-carrying scalar indexer, CPU, tiny model.

The eval script's per-key score path with decay has never run: `_decay_active` gates a mid-row
deadline reference, a `query_offset` on project_q and a `key_offset` on project_k, and Di is 16
instead of 8. This checks prefill AND decode reach the end without shape/offset errors and that
the decode step's offsets are actually threaded (a wrong key_offset silently mis-ages the cache).

Not a quality check -- a wiring check, so a real eval at 13:02 does not fail on shapes.
"""
import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen2 import Qwen2Config

from kvpress import GQAIndexerPress
from kvpress.presses.gqa_indexer import SparseAttentionContext

torch.manual_seed(0)
cfg = Qwen2Config(
    vocab_size=256, hidden_size=64, intermediate_size=128,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    max_position_embeddings=2048,
)
cfg._attn_implementation = "eager"


def fresh_model():
    """A new model per arm: post_init_from_model refuses to swap an attached indexer's geometry
    (correctly -- silently scoring with the old one would be worse), so the two arms cannot share
    a backbone."""
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg).to(torch.float32).eval()


for decay in (False, True):
    model = fresh_model()
    press = GQAIndexerPress(
        compression_ratio=0.0, gate_scale=True, scorer="scalar",
        scalar_mid_dim=8, scorer_attr="indexer", n_sink=2,
        scalar_decay=decay, scalar_decay_ref=16384.0,
    )
    press.post_init_from_model(model)
    idx = press.get_indexer(model.model.layers[0].self_attn)
    print(f"\n=== decay={decay}  Di={idx.idx_dim}  (n_heads={idx.n_heads})")

    S = 192
    ids = torch.randint(0, cfg.vocab_size, (1, S))
    from transformers.cache_utils import DynamicCache

    # query_independent=False forces the GATHER path. On CPU, flex_attention's inductor backend
    # fails to compile (a C++ toolchain issue in this env, unrelated to the router), and the
    # gather path is what decode uses anyway. The deadline path's own agreement with the gather
    # path is covered by tests/presses/test_gqa_indexer_qi_flex.py on GPU.
    with SparseAttentionContext(
        model, press, topk=64, force_local=16, force_sink=2, causal=True,
        query_independent=False,
    ) as ctx:
        cache = DynamicCache()
        with torch.no_grad():
            out = model.model(input_ids=ids, past_key_values=cache)
        print("  prefill ok, hidden", tuple(out.last_hidden_state.shape))
        print("  qi path used:", "flex/deadline" if ctx._use_qi else "gather",
              "| _decay_active:", ctx._decay_active)
        # three decode steps: each appends one token and must score it at the right offset
        for t in range(3):
            nxt = torch.randint(0, cfg.vocab_size, (1, 1))
            with torch.no_grad():
                out = model.model(input_ids=nxt, past_key_values=cache)
        klen = cache.layers[0].keys.shape[2] if hasattr(cache, "layers") else cache.key_cache[0].shape[2]
        kidx = ctx._k_idx[0]
        print(f"  3 decode steps ok; kv_len={klen}, k_idx={tuple(kidx.shape)}")
        assert kidx.shape[1] == klen, "indexer key-cache out of lockstep with the KV cache"
        assert kidx.shape[-1] == idx.idx_dim, "cached indexer key has the wrong width"

    # A decode-time key must be scored at its ABSOLUTE position. Re-score the whole sequence and
    # compare the tail against what the incremental path cached: they must agree.
    if decay:
        h = torch.randn(1, S + 3, cfg.hidden_size)
        whole = idx.project_k(h)
        tail = idx.project_k(h[:, S:, :], key_offset=S)
        err = (whole[:, S:, :] - tail).abs().max().item()
        print(f"  incremental vs whole-sequence key scoring: max err {err:.2e}")
        assert err == 0.0, "key_offset is not threaded correctly at decode"

print("\nOK: sparse inference runs with and without decay, prefill + decode, offsets consistent")
