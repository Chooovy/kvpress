"""Are head_budget=mass and cmp_slots ACTUALLY active on math500/aime25?
import sys; sys.path.insert(0, "/apdcephfs_tj5/share_300719894/user/guhao/kvpress_clean")

math500/aime25 put the whole problem in `question`; `context` is a single space. So the
"prefill" is ~4 tokens and every token that could be evicted is GENERATED. Both features are
triggered off a multi-row prefill, which these datasets do not have. Measure, don't assume.
"""
import sys, torch
sys.path.insert(0, "evaluation")
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvpress.presses.gqa_indexer.press import GQAIndexerPress
from kvpress.presses.gqa_indexer.sparse_inference import SparseAttentionContext
from kvpress.presses.gqa_indexer.scalar_indexer import ScalarIndexer
from transformers import DynamicCache

M = "/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B"
C = "/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_scalar/rvkl_16k_local128_b256_decay/final.pt"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
ck = torch.load(C, map_location="cpu", weights_only=False)
sd = ck.get("indexer", ck); cfg = ck.get("config") or {}
press = GQAIndexerPress(scorer="scalar", scalar_mid_dim=256, scalar_decay=True,
                        scalar_decay_ref=cfg["scalar_decay_ref"], scalar_decay_init=cfg["scalar_decay_init"],
                        scalar_pos_slope=cfg["scalar_pos_slope"], gate_scale=True)
press.attach(model)
press.load_indexers(sd)

TOPK = 1024
ctx = SparseAttentionContext(model, press, topk=TOPK, force_sink=4, force_local=128,
                             head_budget="mass", head_budget_floor=512,
                             cmp_slots=64, cmp_mass="count", cmp_space="post_rope")
prompt = "Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b$.\nRemember to put your final answer within \\boxed{}."
msgs = [{"role": "user", "content": " " + prompt}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", enable_thinking=False).cuda()
print("prompt tokens:", ids.shape[1])

with torch.inference_mode(), ctx:
    ctx.set_context_length(1)
    cache = DynamicCache()
    out = model(input_ids=ids, past_key_values=cache)
    nid = out.logits[0, -1].argmax()
    gen = [nid]
    for i in range(1, 2600):
        out = model(input_ids=nid.view(1, 1), past_key_values=cache)
        nid = out.logits[0, -1].argmax()
        gen.append(nid)
        if i in (1, 500, 1000, 1500, 2000, 2500):
            klen = cache.get_seq_length()
            hb = ctx._head_topk.get(0)
            print(f"step {i:5d}  k_len={klen:6d}  head_topk[0]={'UNIFORM(scalar)' if hb is None else hb.tolist()[:4]}"
                  f"  cmp_layers={len(ctx._cmp)}  cmp_at[0]={ctx._cmp_at.get(0)}")
        if nid.item() in (tok.eos_token_id, 151645):
            break
print("generated:", len(gen), "final k_len:", cache.get_seq_length())
print("head_topk populated layers:", len(ctx._head_topk), " cmp populated layers:", len(ctx._cmp))
