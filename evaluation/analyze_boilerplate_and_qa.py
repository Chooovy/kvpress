"""Two questions, measured on real RULER data.

(1) The user's point: `One of the special magic numbers for ` IS boilerplate, repeated 426x.
    My earlier redundancy scan measured LINE-level duplication and found 0.00 -- but the router
    evicts at TOKEN granularity, so the right unit is the token. Two sub-questions:
      a. what fraction of the context is boilerplate, and does evicting all of it fit the budget?
      b. does the trained router actually score boilerplate below content? (if not, why not)

(2) What a qa_1 / qa_2 example actually looks like.
"""
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/apdcephfs_tj5/share_300719894/user/guhao/kvpress_clean")
sys.path.insert(0, "/apdcephfs_tj5/share_300719894/user/guhao/kvpress_clean/evaluation")
from evaluate_sparse import DATASET_REGISTRY, load_cached_dataset  # noqa: E402
from kvpress import GQAIndexerPress  # noqa: E402
from kvpress.presses.gqa_indexer.train import (  # noqa: E402
    load_indexer_state_dict,
    press_kwargs_from_checkpoint,
)

MODEL = "/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B"
CKPT = ("/apdcephfs_gy8/share_303843174/guhao/models/Qwen-3-8B-gqa_indexer_scalar/"
        "stage1_16k_mid256_longce_decay/final.pt")
TOPK, FL, FS = 2048, 64, 4
TAKE = TOPK - FL - FS
dev = "cuda:0"

tok = AutoTokenizer.from_pretrained(MODEL)
df = load_cached_dataset(DATASET_REGISTRY["ruler"], "8192").to_pandas()
df = df.sample(frac=0.1, random_state=42)

# ---------------------------------------------------------------- (1a) token accounting
row = df[df["task"] == "niah_multikey_2"].iloc[0]
ctx = row["context"]
BOILER = "One of the special magic numbers for "
n_lines = ctx.count(BOILER)
boiler_tok = len(tok(BOILER, add_special_tokens=False).input_ids)
total_tok = len(tok(ctx, add_special_tokens=False).input_ids)

# per line: boilerplate + slug + " is: " + number + "."
lines = [l.strip() for l in ctx.split("\n") if BOILER in l]
slug_tok = num_tok = 0
for l in lines:
    m = re.match(r"One of the special magic numbers for (.+?) is: (\d+)\.?", l)
    if m:
        slug_tok += len(tok(m.group(1), add_special_tokens=False).input_ids)
        num_tok += len(tok(m.group(2), add_special_tokens=False).input_ids)

print("=" * 74)
print("(1a) niah_multikey_2 TOKEN accounting -- is the boilerplate the way out?")
print("=" * 74)
print(f"  context tokens                 {total_tok}")
print(f"  needle lines                   {n_lines}")
print(f"  boilerplate '{BOILER.strip()}'")
print(f"    tokens each                  {boiler_tok}")
print(f"    total                        {n_lines*boiler_tok}  "
      f"({n_lines*boiler_tok/total_tok:.1%} of context)")
print(f"  slug tokens (total)            {slug_tok}")
print(f"  number tokens (total)          {num_tok}")
print(f"  CONTENT (slug+number)          {slug_tok+num_tok}")
print()
print(f"  budget (take)                  {TAKE}")
print(f"  content alone / budget         {(slug_tok+num_tok)/TAKE:.2f}x")
print(f"  -> evicting ALL boilerplate {'IS ENOUGH' if slug_tok+num_tok <= TAKE else 'is NOT enough'}")

# ---------------------------------------------------------------- (1b) does the router do it?
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck.get("indexer", ck)
cfg = ck.get("config") or {}
scorer, kw = press_kwargs_from_checkpoint(sd, cfg)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev).eval()
press = GQAIndexerPress(
    compression_ratio=0.0, gate_scale=True, scorer_attr="indexer", scorer=scorer, **kw
)
press.post_init_from_model(model)
load_indexer_state_dict(model, sd, "indexer")

ids = tok(ctx, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
toks = tok.convert_ids_to_tokens(ids[0])

# label each token position by role
role = ["other"] * len(toks)
pos = 0
text_so_far = ""
# simpler: re-tokenize line by line to get offsets
enc = tok(ctx, return_offsets_mapping=True, add_special_tokens=False)
offs = enc["offset_mapping"]
spans = []
for m in re.finditer(r"One of the special magic numbers for (.+?) is: (\d+)", ctx):
    spans.append(("boiler", m.start(), m.start(1)))
    spans.append(("slug", m.start(1), m.end(1)))
    spans.append(("num", m.start(2), m.end(2)))
for i, (a, b) in enumerate(offs):
    for kind, s, e in spans:
        if a >= s and b <= e:
            role[i] = kind
            break

grabbed = {}
LAYER = 18
def hook(mod, args, kwargs):
    grabbed["h"] = kwargs.get("hidden_states", args[0] if args else None)
hnd = model.model.layers[LAYER].self_attn.register_forward_pre_hook(hook, with_kwargs=True)
with torch.no_grad():
    model.model(input_ids=ids, use_cache=False)
hnd.remove()

idx = press.get_indexer(model.model.layers[LAYER].self_attn)
with torch.no_grad():
    sc = idx.score_keys(grabbed["h"])[0]  # (Hkv, S) fp32, age-0 magnitude
smean = sc.mean(0).float().cpu()

import collections
byrole = collections.defaultdict(list)
for i, r in enumerate(role[: len(smean)]):
    byrole[r].append(smean[i].item())

print()
print("=" * 74)
print(f"(1b) does the TRAINED router score boilerplate below content? (layer {LAYER})")
print("=" * 74)
order = ["boiler", "slug", "num", "other"]
for r in order:
    v = torch.tensor(byrole[r]) if byrole[r] else torch.tensor([0.0])
    print(f"  {r:8s} n={len(byrole[r]):5d}  mean {v.mean():+.4f}  std {v.std():.4f}  "
          f"median {v.median():+.4f}")

# what does a global top-take actually select?
k = min(TAKE, len(smean))
sel = smean.topk(k).indices.tolist()
selrole = collections.Counter(role[i] for i in sel)
print(f"\n  a global top-{k} selection contains:")
for r in order:
    tot = len(byrole[r])
    print(f"    {r:8s} {selrole.get(r,0):5d} of {tot:5d}  "
          f"({selrole.get(r,0)/max(tot,1):.1%} of that role kept)")
print(f"\n  numbers kept: {selrole.get('num',0)}/{len(byrole['num'])} -- the answer survives only")
print("  if the RIGHT number is among them, and the router cannot know which one.")

del model, press
torch.cuda.empty_cache()

# ---------------------------------------------------------------- (2) qa tasks
print()
print("=" * 74)
print("(2) what qa_1 / qa_2 look like")
print("=" * 74)
for t in ("qa_1", "qa_2"):
    r = df[df["task"] == t].iloc[0]
    ctx2, q, a = r["context"], r["question"], r["answer"]
    a_list = list(a) if isinstance(a, (list, tuple)) else [a]
    docs = re.split(r"Document \d+:", ctx2)
    print(f"\n--- {t}: {len(tok(ctx2, add_special_tokens=False).input_ids)} tokens, "
          f"{len(docs)-1} documents")
    print(f"    question: {str(q)[:120]!r}")
    print(f"    answer:   {[str(x)[:60] for x in a_list]}")
    head = ctx2[:400].replace("\n", " ")
    print(f"    head:     {head!r}")
    # where does the answer appear?
    ans0 = str(a_list[0])
    hit = ctx2.lower().find(ans0.lower())
    if hit >= 0:
        frac = hit / len(ctx2)
        print(f"    answer string found at {frac:.0%} through the context; surrounding text:")
        print(f"      ...{ctx2[max(0,hit-130):hit+90].replace(chr(10),' ')!r}...")
    else:
        print("    answer string NOT literally present (abstractive / reformulated)")
