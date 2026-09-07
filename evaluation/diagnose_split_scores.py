"""Why does the split-trained router score so badly at eval?

Hypothesis: the split objective never made C1 scores COMPARABLE to C2 scores.

During training, a C2 query saw:
  key in C1 -> gate = score - lse   (competes, gets gradient)
  key in C2 -> gate = 0             (pinned, NO gradient, out of the normalizer)

So the only pressure on the score function came from ranking C1 keys against each other. Nothing
ever required "an old key's score" and "a recent key's score" to live on one scale -- the recent
ones were removed from the competition entirely.

At eval there is no split: top-k compares every key's score directly, including the recent window.
If the two regions sit at different score levels, top-k is decided by position rather than content.

This measures that offset directly: score a real sequence and compare the score distribution in
the first half (trained-as-C1) against the second half (pinned-as-C2), for the split checkpoint vs
the decay checkpoint (which was trained with no split and scores 76.65).
"""
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvpress import GQAIndexerPress
from kvpress.presses.gqa_indexer.train import (
    load_indexer_state_dict,
    press_kwargs_from_checkpoint,
)

MODEL = "/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B"
O = "/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_scalar"
CKPTS = {
    "split0.5 (step300)": f"{O}/stage1_16k_mid256_longce_split0.5/step300.pt",
    "decay (final, 76.65)": f"{O}/stage1_16k_mid256_longce_decay/final.pt",
    "plain longce (73.71)": f"{O}/stage1_16k_mid256_longce/final.pt",
}
S = 8192
dev = "cuda:0"

tok = AutoTokenizer.from_pretrained(MODEL)
torch.manual_seed(0)
ids = torch.randint(1000, 20000, (1, S), device=dev)

for name, path in CKPTS.items():
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("indexer", ck)
    cfg = ck.get("config") or {}
    scorer, kw = press_kwargs_from_checkpoint(sd, cfg)
    has_gate = any(str(k).endswith("gate_scale") for k in sd)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16
    ).to(dev).eval()
    press = GQAIndexerPress(
        compression_ratio=0.0, gate_scale=has_gate, scorer_attr="indexer",
        scorer=scorer, **kw,
    )
    press.post_init_from_model(model)
    load_indexer_state_dict(model, sd, "indexer")

    # capture hidden states going into a mid-depth attention, then score them
    layer = model.model.layers[18]
    grabbed = {}

    def hook(mod, args, kwargs):
        grabbed["h"] = kwargs.get("hidden_states", args[0] if args else None)

    hnd = layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True)
    with torch.no_grad():
        model.model(input_ids=ids, use_cache=False)
    hnd.remove()

    idx = press.get_indexer(layer.self_attn)
    with torch.no_grad():
        sc = idx.score_keys(grabbed["h"])  # (1, Hkv, S) fp32, age-0 magnitude

    half = S // 2
    c1, c2 = sc[..., :half], sc[..., half:]
    print(f"\n=== {name}  (layer 18, S={S})")
    print(f"  first half  mean {c1.mean():+.4f}  std {c1.std():.4f}")
    print(f"  second half mean {c2.mean():+.4f}  std {c2.std():.4f}")
    print(f"  OFFSET (2nd - 1st) {c2.mean() - c1.mean():+.4f}"
          f"   in units of pooled std: {(c2.mean()-c1.mean())/sc.std():+.2f}")
    # what fraction of a global top-k (2048 of 8192) lands in each half?
    k = 2048
    top = sc[0].reshape(-1, S).topk(k, dim=-1).indices
    frac_late = (top >= half).float().mean().item()
    print(f"  top-{k} share falling in the SECOND half: {frac_late:.3f}"
          f"   (0.5 = position-neutral)")

    del model, press
    torch.cuda.empty_cache()
