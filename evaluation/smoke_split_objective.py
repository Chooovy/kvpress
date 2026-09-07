"""Smoke-check the C1/C2 objective on a real (tiny) causal LM, CPU, no Triton.

Exercises the pieces unit tests cannot: the trainer's hooks, the pin geometry reaching the
attention through them, per_token_ce over a real lm_head, and that the gradient actually lands on
the router's parameters. Not a correctness proof -- a wiring check.
"""
import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen2 import Qwen2Config

from kvpress import GQAIndexerPress
from kvpress.presses.gqa_indexer import E2EIndexerTrainer
from kvpress.presses.gqa_indexer.split_loss import (
    e2e_indexer_split_step,
    resolve_split,
    split_context_loss,
)

torch.manual_seed(0)
# Constructed locally rather than fetched: this box has no hub access, and a tiny GQA config is
# all the wiring check needs (2 layers, 4 q heads over 2 kv heads).
cfg = Qwen2Config(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=1024,
)
cfg._attn_implementation = "eager"
model = AutoModelForCausalLM.from_config(cfg).to(torch.float32).eval()
print("model:", cfg.hidden_size, "hidden,", cfg.num_key_value_heads, "kv heads")

press = GQAIndexerPress(
    compression_ratio=0.0, gate_scale=True, scorer="scalar",
    scalar_mid_dim=8, scorer_attr="indexer", n_sink=2,
)
press.post_init_from_model(model)

S = 256
split = resolve_split(S, 0.5, min_side=32)
trainer = E2EIndexerTrainer(
    press=press, stage="dense", pin_mode="sink", n_sink=2, split=split,
)
trainer.freeze_backbone(model)

ids = torch.randint(0, cfg.vocab_size, (1, S))
loss, stats = e2e_indexer_split_step(
    model, trainer, input_ids=ids, split=split, gap=16, logit_chunk=128
)
print(f"split={split}  loss={float(loss):.4f}  stats={ {k: v for k, v in stats.items()} }")
assert torch.isfinite(loss), "loss must be finite"

loss.backward()
router_grads = {
    n: float(p.grad.norm()) for n, p in model.named_parameters()
    if p.grad is not None and ".indexer." in n
}
backbone_grads = [
    n for n, p in model.named_parameters() if p.grad is not None and ".indexer." not in n
]
print("router params with grad:", len(router_grads))
for n, g in list(router_grads.items())[:6]:
    print(f"  {n.split('.indexer.')[-1]:24s} |g| {g:.4e}")
print("backbone params with grad:", len(backbone_grads), "(must be 0 -- backbone is frozen)")
assert router_grads, "no router parameter received gradient"
assert all(g > 0 for g in router_grads.values()), "a router parameter got a zero gradient"
assert not backbone_grads, f"backbone leaked gradient: {backbone_grads[:3]}"

# The split must matter: gating the whole sequence should give a different loss.
model.zero_grad(set_to_none=True)
trainer.split = None
unsplit, _ = split_context_loss(
    model.get_output_embeddings(),
    __import__(
        "kvpress.presses.gqa_indexer.e2e_trainer", fromlist=["_final_hidden_states"]
    )._final_hidden_states(model, input_ids=ids, attention_mask=None),
    ids, split, gap=16, logit_chunk=128,
)
print(f"same C2 loss but gate unsplit: {float(unsplit):.4f}  (split: {float(loss):.4f})")
assert abs(float(unsplit) - float(loss)) > 1e-6, "the pin geometry had no effect on the loss"
print("\nOK: split objective trains the router, backbone stays frozen, pin geometry is live")
