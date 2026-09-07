"""Where does the C1/C2 split objective lose the router?

The gate is `score - lse` on history and `0` on pinned keys, with `sum over history of
exp(gate) = gate_budget` (default 1). So the RELATIVE mass the gate grants is

    history : pinned  =  gate_budget : (number of pinned keys)

Without a split, "pinned" is just the n_sink=4 sink keys, so history competes 1 : 4.
With a split, every C2 key is pinned too, so at 8K/split 4096 history competes 1 : 4100 --
a log(4100) ~= 8.3 nat handicap applied to the whole C1 region, which is exactly the region
the objective is supposed to be teaching the router to rank.

This measures the actual attention probability landing on history under each configuration,
and checks whether `gate_budget_ratio` (budget proportional to the history length) restores it.
"""
import torch

from kvpress.presses.gqa_indexer.gated_attention import gated_attention_reference
from kvpress.presses.gqa_indexer.gate_pin import pinned_mask

torch.manual_seed(0)
B, Hq, Hkv, S, D, Di = 1, 4, 2, 512, 32, 4
SPLIT, N_SINK = S // 2, 4

q = torch.randn(B, Hq, S, D, dtype=torch.double)
k = torch.randn(B, Hkv, S, D, dtype=torch.double)
v = torch.randn(B, Hkv, S, D, dtype=torch.double)
qi = torch.randn(B, Hkv, S, Di, dtype=torch.double)
ki = torch.randn(B, S, Di, dtype=torch.double)
gs = torch.tensor(1.0, dtype=torch.double)
scale = D ** -0.5


def history_mass(pin_from, budget=1.0, ratio=None):
    """Mean attention probability on gated (non-pinned, visible) keys, over C2 query rows."""
    pinned = pinned_mask(
        "sink", S, S, q.device, n_sink=N_SINK, query_offset=0, pin_from=pin_from
    )
    # replicate the gate the reference builds, then read the softmax directly
    from kvpress.presses.gqa_indexer.gated_attention import _gate_lse, _visible
    from kvpress.presses.gqa_indexer.gate_pin import gate_from_score

    lse = _gate_lse(
        qi, ki, gs, budget, ratio, pinned,
        causal_keep=_visible(None, S, S, q.device, 0), key_tile=256,
    )
    score = torch.einsum("bhqd,bkd->bhqk", qi, ki) * gs
    gate = gate_from_score(score, lse, pinned)
    logits = torch.einsum("bhqd,bhkd->bhqk", q, k.repeat_interleave(Hq // Hkv, 1)) * scale
    logits = logits + gate.repeat_interleave(Hq // Hkv, 1)
    causal = torch.arange(S).unsqueeze(-1) >= torch.arange(S).unsqueeze(0)
    logits = logits.masked_fill(~causal, -float("inf"))
    p = logits.softmax(-1)

    gated_keys = causal & ~pinned  # what the router actually controls
    rows = slice(SPLIT, S)  # C2 queries: the ones whose loss is counted
    return (p[:, :, rows] * gated_keys[rows].unsqueeze(0).unsqueeze(0)).sum(-1).mean().item()


print("attention mass on ROUTER-CONTROLLED keys, averaged over C2 query rows")
print(f"  (S={S}, split={SPLIT}, n_sink={N_SINK})\n")
m_plain = history_mass(None)
m_split = history_mass(SPLIT)
print(f"  no split        (pinned = {N_SINK} sinks)          {m_plain:.4f}")
print(f"  split, budget=1 (pinned = {N_SINK}+{S-SPLIT} keys)      {m_split:.6f}"
      f"   <-- {m_plain/max(m_split,1e-12):.0f}x less")
for r in (0.25, 0.5, 1.0):
    m = history_mass(SPLIT, ratio=r)
    print(f"  split, ratio={r:<4}                          {m:.4f}")
print(f"\n  handicap from a fixed budget: log(1 + {N_SINK} + {S-SPLIT}) = "
      f"{torch.log(torch.tensor(1.0 + N_SINK + S - SPLIT)).item():.2f} nats against C1")
print(f"  at 8K/split 4096 that is log(4101) = "
      f"{torch.log(torch.tensor(4101.0)).item():.2f} nats")
