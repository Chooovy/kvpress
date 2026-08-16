#!/usr/bin/env python
"""
Diagnose RNN-state design choices for the GQA indexer, on *real* hidden states.

Everything here is measurement, not training. The question is whether a fixed-size
recurrent state ``z`` can supply the one thing a per-key score cannot get from ``h_t``
alone -- "have I seen this before?" -- and, if so, which state shape is worth paying for.

Three things are measured, in the order that decides the design:

A. Effective rank of the hidden-state stream.
   The matrix-state (DeltaNet / GDN / KDN) family stores an associative map in a d x d
   state, and that state saturates once the seen keys span the space: S -> I, and the
   novelty signal ||h - Sh|| goes to exactly zero. On isotropic synthetic vectors the
   wall sits at t ~ d. Real hidden states are anisotropic, so the wall sits at the
   *effective* rank, which is what this measures. If effective rank << d, the wall
   arrives far earlier than d and matrix states are worse than they look on paper.

B. Redundancy-detection AUC, per state design.
   Labels come from the text itself, not from synthetic repeats: a token is "redundant"
   if an earlier token in the same sequence is highly similar to it (cosine above a
   percentile cut on the true full-history max-similarity). Full attention over the
   uncompressed history is the ceiling by construction, so each state design is scored
   as a fraction of a signal that is known to be there.

C. Saturation curve for the matrix state.
   ``||h_t - S h_t||`` against t, for several decay values, plus how many distinct values
   survive a cast to bf16. This is the direct test of "forgetting window and state size are
   two ends of one knob". Saturation turns out to damage the signal's *dynamic range* more
   than its ranking -- the ranking partly survives -- so the bf16 column is the one that
   matters: the indexer trains in bf16, and a compressed range loses the ordering outright.

Run (GPU):
    python scripts/diag_rnn_state_design.py --layers 8,16,24 --seq-len 32768
Run (CPU, slower but fine at 4K):
    python scripts/diag_rnn_state_design.py --layers 16 --seq-len 4096 --device cpu

``--model`` accepts any HF causal LM and defaults to a local Llama-3-8B. ``--fake`` runs the
whole harness on a synthetic stream in seconds, with no model, for checking changes.

Two controls are printed and both must be beaten: ``control: position`` (redundancy labels
correlate with position, and every decaying state drifts with position) and
``control: random``. The ``pos-matched`` column removes the position confound and is the
one to read -- on real hidden states the raw column credited a fully saturated matrix state
with a *perfect* score that the pos-matched column and part C both contradict.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import torch


# ----------------------------------------------------------------------
# State designs. Each takes (L, d) hidden states and returns an (L,) novelty
# score: LARGER means "more novel / less redundant".
# ----------------------------------------------------------------------
def novelty_full_attention(h: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """Ceiling: ``1 - max cosine`` against the whole uncompressed history.

    Tiled over queries so the ``L x L`` similarity matrix is never materialised: at 128K
    tokens that matrix alone is 64 GB in fp32. Each tile compares only against keys that
    are strictly in the past, which is what makes this an upper bound on any causal
    scorer's redundancy signal.
    """
    hn = torch.nn.functional.normalize(h, dim=-1)
    L = hn.shape[0]
    out = torch.empty(L, dtype=hn.dtype, device=hn.device)
    for start in range(0, L, chunk):
        stop = min(start + chunk, L)
        sim = hn[start:stop] @ hn[:stop].T  # (tile, stop)
        q = torch.arange(start, stop, device=hn.device).unsqueeze(-1)
        k = torch.arange(stop, device=hn.device).unsqueeze(0)
        sim.masked_fill_(k >= q, -1.0)  # strictly past only
        out[start:stop] = 1.0 - sim.amax(dim=-1).clamp_min(-1.0)
    out[0] = 1.0  # no history to repeat
    return out


def novelty_ema(h: torch.Tensor, half_life: float) -> torch.Tensor:
    """Vector state: ``||h_t - z_{t-1}||``, ``z`` an EMA. Stores the running MEAN, not identities.

    Computed in closed form rather than by stepping the recurrence. The recurrence
    ``z_t = lam*z_{t-1} + (1-lam)*h_t`` unrolls to

        z_t = lam^(t+1) * z_0 + (1-lam) * lam^t * sum_{i<=t} lam^(-i) h_i,

    so one cumulative sum gives every ``z_t`` at once. That matters on GPU, where an
    L-step Python loop is dominated by kernel-launch latency and runs *slower* than the
    same loop on CPU. It is also the concrete reason an input-independent decay is the
    cheap choice for the real design: input-dependent gating loses this closed form and
    needs a real associative-scan kernel.

    ``lam^(-i)`` overflows quickly, so the sum is taken in fixed-size chunks with the
    chunk's base power factored out, carrying the state across chunk boundaries. The chunk
    length is capped from the *actual dtype's* max exponent -- a cap tuned for fp64
    silently produces NaN in fp32 for short half-lives. Verified against the naive loop:
    exact to ~1e-16 relative in fp64 and ~2e-7 in fp32, for half-lives 1..1e9 and lengths
    up to 16K.
    """
    L, d = h.shape
    lam = 0.5 ** (1.0 / half_life)
    if lam >= 1.0:  # degenerate: no decay, z is the running mean
        csum = h.cumsum(0)
        z_prev = torch.zeros_like(h)
        if L > 1:
            denom = torch.arange(1, L, device=h.device, dtype=h.dtype).unsqueeze(-1)
            z_prev[1:] = csum[:-1] / denom
        return (h - z_prev).norm(dim=-1)

    # Cap the chunk so max(lam^-j) stays well inside the dtype's range. Budget half the
    # available exponent headroom, leaving the rest for the magnitude of h itself.
    max_exp = math.log10(torch.finfo(h.dtype).max)
    chunk = min(L, max(16, int(0.5 * max_exp / -math.log10(lam))))
    out = torch.empty(L, dtype=h.dtype, device=h.device)
    z = torch.zeros(d, dtype=h.dtype, device=h.device)
    for start in range(0, L, chunk):
        blk = h[start : start + chunk]
        n = blk.shape[0]
        j = torch.arange(n, device=h.device, dtype=h.dtype)
        inner = ((lam ** (-j)).unsqueeze(-1) * blk).cumsum(0)
        z_t = (lam ** (j + 1)).unsqueeze(-1) * z + (1 - lam) * (lam**j).unsqueeze(-1) * inner
        z_prev = torch.empty_like(blk)
        z_prev[0] = z
        if n > 1:
            z_prev[1:] = z_t[:-1]
        out[start : start + n] = (blk - z_prev).norm(dim=-1)
        z = z_t[-1]
    return out


def novelty_ema_scales(h: torch.Tensor, half_lives=(8, 64, 512, 4096)) -> torch.Tensor:
    """Per-scale deviations stacked: ``(n_scales, L)``.

    Kept unreduced because the real design learns how to combine scales. Collapsing them
    here with a fixed rule (max, mean) understates the design badly -- the scales disagree
    in sign and magnitude, so a hand-picked combiner cancels signal that a learned one
    would keep. :func:`lda_combine` measures the learnable ceiling instead.
    """
    return torch.stack([novelty_ema(h, hl) for hl in half_lives], dim=0)


def lda_combine(feats: torch.Tensor, keep: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
    """Best linear combination of ``feats`` (n_feat, L) for separating the two classes.

    Closed-form Fisher LDA, fit *in sample*, so this is an optimistic upper bound on what
    a learned combination could extract -- exactly the right quantity when the question is
    "is it worth building the multi-scale version", and dishonest for anything else. It is
    labelled as an upper bound wherever it is printed.
    """
    x = feats[:, keep].double()
    x = (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp_min(1e-9)
    a, b = x[:, lab], x[:, ~lab]
    mu = a.mean(1) - b.mean(1)
    cov = torch.cov(x) + 1e-3 * torch.eye(x.shape[0], dtype=x.dtype, device=x.device)
    w = torch.linalg.solve(cov, mu)
    return (w[:, None] * feats.double()).sum(0)


def novelty_matrix_states(
    h: torch.Tensor, configs: list[tuple[str, float, float]]
) -> dict[str, torch.Tensor]:
    """All matrix-state variants in one pass: ``{label: novelty}``.

    ``configs`` is a list of ``(label, beta, decay)``. ``beta=None`` selects the plain
    additive outer-product state (GLA / RetNet core, no delta correction), whose novelty
    proxy is ``-||S h_t||`` -- a large recall response means "already seen".

    The delta rule is genuinely sequential (``S_t`` depends on ``u_t`` which depends on
    ``S_{t-1}``), so the L-step loop cannot be removed. What it can be is *shared*: every
    config is advanced together as a batched ``(n_cfg, d, d)`` state, turning n separate
    L-step loops into one, which is what makes this tolerable on GPU where each step pays
    launch latency. All the einsums below are batched over the config axis.

    ``h`` is row-normalised internally. ``S += beta * u h^T`` is only the projector-building
    update it is meant to be when ``h'h == 1``; with ``||h|| ~ 5`` (typical for raw hidden
    states) each update overshoots by ``||h||^2`` and the state diverges to inf within ~25
    steps. Normalising here rather than relying on the caller keeps the diagnostic from
    silently reporting NaN for the whole matrix-state family.
    """
    h = torch.nn.functional.normalize(h, dim=-1)
    L, d = h.shape
    n = len(configs)
    dev, dt = h.device, h.dtype
    betas = torch.tensor([1.0 if b is None else b for _, b, _ in configs], device=dev, dtype=dt)
    decays = torch.tensor([dc for _, _, dc in configs], device=dev, dtype=dt)
    is_delta = torch.tensor([b is not None for _, b, _ in configs], device=dev)

    S = torch.zeros(n, d, d, device=dev, dtype=dt)
    out = torch.empty(n, L, device=dev, dtype=dt)
    b_ = betas.view(n, 1)
    dc_ = decays.view(n, 1, 1)
    delta_ = is_delta.view(n, 1)
    for t in range(L):
        x = h[t]
        Sx = torch.einsum("nij,j->ni", S, x)
        u = x.unsqueeze(0) - Sx  # delta-rule residual
        # Both variants in one branchless step so the batched loop stays uniform.
        out[:, t] = torch.where(delta_.squeeze(1), u.norm(dim=-1), -Sx.norm(dim=-1))
        write = torch.where(delta_, b_ * u, x.unsqueeze(0).expand(n, d))
        S = dc_ * S + torch.einsum("ni,j->nij", write, x)
    return {label: out[i] for i, (label, _, _) in enumerate(configs)}


def novelty_hidden_only(h: torch.Tensor) -> torch.Tensor:
    """Control: no state at all. ||h_t||, i.e. what a SparseK/DMA-style scorer can see.

    Any design that fails to beat this has not earned its state.
    """
    return h.norm(dim=-1)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def auc(novelty: torch.Tensor, is_redundant: torch.Tensor) -> float:
    """P(novelty[redundant] < novelty[novel]), ties at half credit. 0.5 == chance.

    Computed from rank sums (Mann-Whitney U) rather than an all-pairs comparison, which
    would allocate ``n_pos * n_neg`` elements -- fine at 4K tokens, tens of GB at 128K.
    """
    a = novelty[is_redundant].double()
    b = novelty[~is_redundant].double()
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    both = torch.cat([a, b])
    # Average ranks so exact ties get half credit, matching the pairwise definition.
    order = both.argsort()
    ranks = torch.empty_like(both)
    ranks[order] = torch.arange(1, both.numel() + 1, device=both.device, dtype=both.dtype)
    uniq, inv, counts = torch.unique(both, return_inverse=True, return_counts=True)
    tie_mean = torch.zeros_like(uniq).index_add_(0, inv, ranks) / counts
    ranks = tie_mean[inv]
    u = ranks[: a.numel()].sum() - a.numel() * (a.numel() + 1) / 2
    # U counts pairs where a > b; the definition above is P(a < b), hence 1 - U/(na*nb).
    return float(1.0 - u / (a.numel() * b.numel()))


def auc_position_matched(
    novelty: torch.Tensor, is_redundant: torch.Tensor, pos: torch.Tensor, n_bins: int = 8
) -> float:
    """AUC computed within position bins, then averaged over bins.

    Position is a confounder that has to be removed rather than reported around. A late
    token has more history available to repeat, so redundancy labels correlate with ``t``;
    any state whose novelty signal drifts with ``t`` -- which every decaying state does,
    since the state norm grows and then plateaus -- scores well by proxy. Measured on real
    Llama-3-8B hidden states, raw AUC credited a saturating delta-rule state with a
    *perfect* 0.500 info while its actual novelty signal had already collapsed to 0.075
    (part C), purely from this confound.

    Comparing only within narrow position bins removes the drift, so what is left is the
    part of redundancy that the state explains beyond "how far into the sequence am I".
    Bins are equal-count over the retained positions; bins containing a single class are
    skipped.
    """
    order = torch.argsort(pos)
    aucs = []
    for chunk in torch.chunk(order, n_bins):
        lab = is_redundant[chunk]
        if lab.all() or (~lab).all():
            continue
        aucs.append(auc(novelty[chunk], lab))
    return float(sum(aucs) / len(aucs)) if aucs else float("nan")


def effective_rank(h: torch.Tensor) -> dict:
    """Participation-ratio effective rank plus the 90/99% spectral-energy thresholds.

    Reported as "the t at which a d x d matrix state saturates", which is the quantity
    that matters for the matrix-state family -- not the algebraic rank.
    """
    x = (h - h.mean(0, keepdim=True)).double()
    sv = torch.linalg.svdvals(x)
    ev = sv**2
    p = ev / ev.sum()
    entropy_rank = float(torch.exp(-(p * p.clamp_min(1e-30).log()).sum()))
    csum = torch.cumsum(p, 0)
    return {
        "d": h.shape[-1],
        "participation_ratio": float((ev.sum() ** 2) / (ev**2).sum()),
        "entropy_rank": entropy_rank,
        "rank_90pct_energy": int((csum < 0.90).sum()) + 1,
        "rank_99pct_energy": int((csum < 0.99).sum()) + 1,
        "top1_energy_frac": float(p[0]),
    }


def redundancy_labels(h: torch.Tensor, frac: float = 0.25) -> tuple:
    """Label the ``frac`` most-repetitive tokens as redundant, by true max-similarity.

    Deliberately derived from the uncompressed history: it makes the ceiling exact (full
    attention scores 1.0 by construction) so every state design is measured as a
    recoverable fraction of a signal that provably exists in the stream. Tokens near the
    threshold are dropped so the two classes are separated and the AUC is not dominated
    by boundary noise.
    """
    nov = novelty_full_attention(h)
    warm = max(32, int(0.05 * h.shape[0]))  # early tokens have no history to repeat
    valid = torch.zeros_like(nov, dtype=torch.bool)
    valid[warm:] = True
    v = nov[valid]
    lo, hi = torch.quantile(v, frac), torch.quantile(v, 1.0 - frac)
    redundant = valid & (nov <= lo)
    novel = valid & (nov >= hi)
    return redundant, novel, nov


# ----------------------------------------------------------------------
def resolve_device(spec: str) -> torch.device:
    """``auto`` picks cuda when present. An explicit ``cuda`` request must not fall back."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            f"--device {spec} requested but torch reports no CUDA "
            f"(torch {torch.__version__}). Silently running on CPU would take hours "
            f"and look like a hang; install a CUDA build or pass --device cpu."
        )
    return torch.device(spec)


def get_hidden_states(args, device: torch.device) -> dict:
    if args.fake:
        g = torch.Generator().manual_seed(0)
        L, d = args.seq_len, 256
        # Anisotropic + repeating, so the harness is exercised on a stream that has
        # the structure we claim to measure.
        basis = torch.randn(16, d, generator=g)
        idx = torch.randint(16, (L,), generator=g)
        h = basis[idx] + 0.3 * torch.randn(L, d, generator=g)
        h[:, 0] += 8.0  # a dominant shared direction, as real hidden states have
        return {"fake": h.to(device)}

    from transformers import AutoModel, AutoTokenizer

    print(f"loading {args.model} on {device} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    # AutoModel, not AutoModelForCausalLM: only hidden states are wanted, and the LM head
    # is the single largest cost at long context. Qwen3's vocab is 152K, so at L=32768 the
    # logits alone are 9 GB in bf16 -- and HF upcasts them to fp32, for 28 GB of pure waste
    # on a tensor this script never reads. Dropping the head is what makes 32K fit.
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    kw = {"low_cpu_mem_usage": True}
    if args.attn_impl:
        kw["attn_implementation"] = args.attn_impl
    # The dtype kwarg was renamed: `torch_dtype` in transformers <=4.x, `dtype` in 5.x.
    # Passing the wrong one is not ignored -- it is forwarded to the model constructor and
    # raises TypeError -- and passing neither silently loads fp32, doubling weight memory.
    # Try the modern name, fall back on the signature complaint.
    try:
        model = AutoModel.from_pretrained(args.model, dtype=dtype, **kw)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        model = AutoModel.from_pretrained(args.model, torch_dtype=dtype, **kw)
    model = model.to(device)
    model.eval()
    got = next(model.parameters()).dtype
    if got != dtype:
        raise SystemExit(
            f"asked for {dtype} but the model loaded as {got}; memory estimates assume "
            f"{dtype}. Check the transformers version's dtype kwarg."
        )
    n_layers = model.config.num_hidden_layers
    print(f"  loaded in {time.time() - t0:.0f}s, {n_layers} layers, dtype={got}", flush=True)

    layers = [int(x) for x in args.layers.split(",")]
    bad = [i for i in layers if not 0 <= i <= n_layers]
    if bad:
        raise SystemExit(f"--layers {bad} out of range; hidden_states has 0..{n_layers}")

    # Capture only the requested layers via hooks. output_hidden_states=True would keep all
    # n_layers+1 of them resident (9 GB at 32K for a 36-layer model) to serve three.
    wanted = sorted(set(layers))
    grabbed: dict[int, torch.Tensor] = {}

    def make_hook(idx: int):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            # fp32 on CPU: the analysis needs the precision and the GPU needs the room.
            grabbed[idx] = t.detach()[0].float().cpu()

        return hook

    handles = []
    if 0 in wanted:  # layer 0 is the embedding output, before any decoder layer
        handles.append(model.embed_tokens.register_forward_hook(make_hook(0)))
    for i in wanted:
        if i > 0:
            handles.append(model.layers[i - 1].register_forward_hook(make_hook(i)))

    text = load_text(args)
    ids = tok(text, return_tensors="pt").input_ids[:, : args.seq_len].to(device)
    print(f"  forward on {ids.shape[1]} tokens (capturing layers {wanted}) ...", flush=True)
    t0 = time.time()
    try:
        with torch.no_grad():
            model(ids)
    except torch.OutOfMemoryError as exc:
        raise SystemExit(
            f"CUDA OOM during the forward pass at seq_len={args.seq_len}.\n"
            f"  {exc}\n"
            f"Attention itself is the remaining cost. Options, cheapest first:\n"
            f"  --seq-len {args.seq_len // 2}          (halves activation memory)\n"
            f"  --layers <one layer>       (fewer captured streams)\n"
            f"  --attn-impl flash_attention_2   (if flash-attn is installed)\n"
            f"  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  (reduces fragmentation)"
        ) from exc
    for hd in handles:
        hd.remove()
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"  forward done in {time.time() - t0:.1f}s, peak GPU {peak:.1f} GB", flush=True)
    else:
        print(f"  forward done in {time.time() - t0:.1f}s", flush=True)

    streams = {f"layer{i}": grabbed[i] for i in layers}
    # Free the model before the analysis, which wants the memory for its own tiles.
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return streams


def load_text(args) -> str:
    """Real long-context prose, long enough to fill ``--seq-len`` tokens.

    Genuine prose is the point: redundancy structure in real text is what the state designs
    are being asked to detect. RULER is deliberately *not* preferred even when cached --
    its haystack is synthetic filler ("The grass is green." repeated) whose redundancy is
    an artifact of the benchmark, which would flatter every design that keys on repetition.
    Sources are tried in order and each failure is reported, since silently substituting a
    different corpus would change what the numbers mean.

    Roughly 4 characters per token, so 6x headroom is requested and the result is checked
    against that floor rather than assumed.
    """
    # ~4 characters per token; 6x gives headroom. Callers that slice the text into several
    # documents (the state probe fits on held-out documents) need that many times more, so
    # honour n_docs here rather than letting the later documents come back empty.
    need_chars = args.seq_len * 6 * max(1, getattr(args, "n_docs", 1))

    if args.text_file:
        with open(args.text_file) as f:
            text = f.read()
        if len(text) < need_chars:
            print(f"  warning: --text-file has {len(text):,} chars, wanted >= {need_chars:,}; "
                  f"the effective sequence may be shorter than --seq-len", flush=True)
        return text

    # wikitext-103: real encyclopedic prose, small, no auth. Streamed and concatenated
    # until the character budget is met.
    try:
        from datasets import load_dataset

        print("  text: wikitext-103 (downloading if not cached) ...", flush=True)
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        buf, total = [], 0
        for row in ds:
            line = row["text"].strip()
            if len(line) < 32:  # skip blank lines and bare section headings
                continue
            buf.append(line)
            total += len(line) + 1
            if total >= need_chars:
                break
        if total >= need_chars:
            print(f"  text: {total:,} chars of wikitext prose", flush=True)
            return "\n".join(buf)
        print(f"    wikitext yielded only {total:,} chars", flush=True)
    except Exception as exc:  # noqa: BLE001 - report and try the next source
        print(f"    wikitext failed: {type(exc).__name__}: {exc}", flush=True)

    # Fall back to any cached parquet corpus, RULER included, with the caveat stated.
    roots = [
        os.environ.get("HF_HOME"),
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        os.path.expanduser("~/.cache/huggingface"),
    ]
    for root in filter(None, roots):
        hits = sorted(
            glob.glob(os.path.join(root, "**", "datasets--*", "**", "*.parquet"), recursive=True)
        )
        if not hits:
            continue
        import pandas as pd

        df = pd.read_parquet(hits[0])
        col = "context" if "context" in df.columns else df.columns[0]
        text = "\n\n".join(str(x) for x in df[col].head(args.n_docs))
        print(f"  text: cached parquet {hits[0]} ({len(text):,} chars)", flush=True)
        if "ruler" in hits[0].lower():
            print("    NOTE: RULER haystack is synthetic filler; its repetition is a "
                  "benchmark artifact, so treat redundancy numbers from it as optimistic.",
                  flush=True)
        if len(text) < need_chars:
            print(f"    warning: {len(text):,} chars < {need_chars:,} wanted; the effective "
                  f"sequence will be shorter than --seq-len", flush=True)
        return text

    raise SystemExit(
        "could not obtain text from any source. Pass --text-file with a long plain-text "
        f"file (at least ~{need_chars:,} characters of prose)."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="/apdcephfs_gy8/share_303843174/guhao/models/Qwen3-8B",
    )
    ap.add_argument("--layers", default="8,16,24", help="hidden_states indices to analyse")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument(
        "--state-dim", type=int, default=128, help="project h to this dim before scoring"
    )
    ap.add_argument("--device", default="auto", help="auto | cuda | cuda:0 | cpu")
    ap.add_argument(
        "--attn-impl",
        default=None,
        help="HF attn_implementation, e.g. flash_attention_2 or sdpa. Lowers forward memory "
        "at long context; requires the backend to be installed.",
    )
    ap.add_argument("--text-file", default=None, help="use this text instead of downloading")
    ap.add_argument(
        "--n-docs", type=int, default=8, help="documents to concatenate from a parquet source"
    )
    ap.add_argument("--fake", action="store_true", help="synthetic stream, no model")
    ap.add_argument("--out", default="rnn_state_diag.json")
    args = ap.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(0)
    streams = get_hidden_states(args, device)
    results = {}

    for name, h_full in streams.items():
        # Streams are captured to CPU so the forward pass stays lean; move each one to the
        # compute device just in time. One fp32 stream at 32K x 4096 is 0.5 GB, so this is
        # cheap once the model is gone.
        h_full = h_full.to(device)
        print(f"\n{'=' * 74}\n{name}: {tuple(h_full.shape)} on {h_full.device}\n{'=' * 74}",
              flush=True)

        # --- A. effective rank, on the raw stream ---
        er = effective_rank(h_full)
        print("A. effective rank of the hidden stream")
        print(f"   d = {er['d']}, participation ratio = {er['participation_ratio']:.1f}, "
              f"entropy rank = {er['entropy_rank']:.1f}")
        print(f"   rank at 90% energy = {er['rank_90pct_energy']}, "
              f"at 99% = {er['rank_99pct_energy']}, top-1 direction = "
              f"{er['top1_energy_frac'] * 100:.1f}% of variance")
        print(f"   -> a d x d matrix state saturates at t ~ {er['rank_90pct_energy']} "
              f"(90% energy), NOT t ~ {er['d']}")

        # The indexer projects to head_dim before scoring, so measure the state designs
        # in that space: a random projection stands in for an untrained w_k. A trained one
        # would preserve the directions that matter for scoring, so these numbers are a
        # lower bound on what the real design can reach, not a prediction of it.
        d_state = min(args.state_dim, h_full.shape[-1])
        if d_state < h_full.shape[-1]:
            g = torch.Generator(device="cpu").manual_seed(0)
            proj = (torch.randn(h_full.shape[-1], d_state, generator=g) / math.sqrt(d_state)).to(
                h_full.device
            )
            h = h_full @ proj
        else:
            h = h_full
        # Per-token normalisation: hidden-state norms grow with depth and would otherwise
        # dominate every distance. This is what IndexerNorm does before scoring.
        h = torch.nn.functional.normalize(h - h.mean(0, keepdim=True), dim=-1)

        # --- B. redundancy AUC ---
        redundant, novel, _ = redundancy_labels(h)
        keep = redundant | novel
        lab = redundant[keep]
        # Saturation is a LATE-position effect: a matrix state can score well overall on
        # signal earned before it fills up. Scoring the last quarter separately is what
        # distinguishes "works" from "worked until it saturated".
        L = h.shape[0]
        late = torch.zeros_like(keep)
        late[3 * L // 4 :] = True
        keep_late = keep & late
        lab_late = redundant[keep_late]
        print(f"\nB. redundancy detection (state dim {d_state}, "
              f"{int(redundant.sum())} redundant vs {int(novel.sum())} novel; "
              f"{int(keep_late.sum())} in the last quarter)", flush=True)

        t0 = time.time()
        designs = {
            "full attention (ceiling, O(L^2))": novelty_full_attention(h),
            "no state: ||h_t|| (SparseK-like)": novelty_hidden_only(h),
            "EMA vector hl=64": novelty_ema(h, 64),
            "EMA vector hl=512": novelty_ema(h, 512),
            "EMA vector hl=4096": novelty_ema(h, 4096),
        }
        # One shared L-step loop for every matrix-state variant.
        designs.update(
            novelty_matrix_states(
                h,
                [
                    ("delta rule beta=1 decay=1", 1.0, 1.0),
                    ("delta rule beta=1 decay=0.99", 1.0, 0.99),
                    ("delta rule beta=1 decay=0.95", 1.0, 0.95),
                    ("delta rule beta=0.5 decay=0.99", 0.5, 0.99),
                    ("linear attn outer decay=0.99", None, 0.99),
                ],
            )
        )
        print(f"   (state recurrences: {time.time() - t0:.1f}s)", flush=True)
        scales = novelty_ema_scales(h)
        designs["EMA multiscale, LDA (upper bnd)"] = lda_combine(scales, keep, lab)
        # Two controls. Position must be included: it is the confounder that
        # auc_position_matched exists to remove, and seeing it score high on the raw
        # column while collapsing on the matched column is what validates the correction.
        designs["control: position -t"] = -torch.arange(L, dtype=h.dtype, device=h.device)
        # Generator is CPU-only, so draw there and move: a cuda generator would need a
        # matching device argument and would not reproduce the CPU run's numbers.
        rand_ctl = torch.randn(L, generator=torch.Generator().manual_seed(0))
        designs["control: random"] = rand_ctl.to(device=h.device, dtype=h.dtype)

        pos_all = torch.arange(L, dtype=torch.float, device=h.device)[keep]
        pos_late = torch.arange(L, dtype=torch.float, device=h.device)[keep_late]
        scores = {}
        print(f"   {'design':<38} {'raw':>7} {'pos-matched':>12} {'late':>7}  state")
        for label, nov in designs.items():
            a = auc(nov[keep], lab)
            a_pm = auc_position_matched(nov[keep], lab, pos_all)
            a_late = auc_position_matched(nov[keep_late], lab_late, pos_late, n_bins=4)
            scores[label] = {"auc_raw": a, "auc_pos_matched": a_pm, "auc_late": a_late}
            state_sz = (
                "O(L*d) grow" if "full attention" in label
                else "0" if ("no state" in label or "control" in label)
                else f"{d_state * 4}" if "multiscale" in label
                else f"{d_state}" if "EMA" in label
                else f"{d_state**2}"
            )
            # |AUC - 0.5| is the quantity that matters, not AUC: a design scoring 0.2 is
            # as informative as one scoring 0.8 -- it ranks redundancy the other way up,
            # and a trained w_k absorbs the sign for free. Only 0.5 means "no signal".
            def info(x):
                return "  nan" if x != x else f"{abs(x - 0.5):.3f}"

            print(f"   {label:<38} {info(a):>7} {info(a_pm):>12} {info(a_late):>7}  {state_sz}")

        # --- C. saturation curve for the matrix state ---
        print("\nC. matrix-state saturation: mean ||h_t - S h_t|| by position")
        wins = [(0, L // 8), (L // 4, 3 * L // 8), (3 * L // 4, L)]
        print(f"   {'decay':>7} " + " ".join(f"t={a}-{b}".rjust(12) for a, b in wins)
              + "   alive?  bf16 distinct")
        sat = {}
        decays = (1.0, 0.999, 0.99, 0.95)
        curves = novelty_matrix_states(h, [(f"d{dc}", 1.0, dc) for dc in decays])
        for dc in decays:
            u = curves[f"d{dc}"]
            vals = [float(u[a:b].mean()) for a, b in wins]
            # Saturation collapses the signal's dynamic range while leaving its *ranking*
            # partly intact, so a magnitude-only check understates the damage in a way that
            # matters: at bf16 (the dtype the indexer trains in) a compressed range loses
            # the ranking outright. Count how many of the late values survive the cast.
            late_u = u[3 * L // 4 :]
            n_distinct = int(torch.unique(late_u.to(torch.bfloat16).float()).numel())
            frac = n_distinct / max(late_u.numel(), 1)
            sat[str(dc)] = {"windows": vals, "bf16_distinct_frac_late": frac}
            alive = "yes" if vals[-1] > 0.5 * vals[0] else "NO"
            print(f"   {dc:>7} " + " ".join(f"{v:12.4f}" for v in vals)
                  + f"   {alive:>5}  {n_distinct}/{late_u.numel()} ({frac * 100:.0f}%)")

        results[name] = {"effective_rank": er, "auc": scores, "saturation": sat,
                         "n_redundant": int(redundant.sum()), "n_novel": int(novel.sum())}

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")
    print("\nHow to read this:")
    print("  A: if rank_90pct_energy << d, matrix states saturate early -> prefer vector state.")
    print("  B: all three columns are info = |AUC-0.5|, not AUC. Direction is free (a trained")
    print("     w_k absorbs the sign), so 0.2 and 0.8 carry equal signal and 0.0 carries none.")
    print("     TRUST THE 'pos-matched' COLUMN. Redundancy labels correlate with position")
    print("     (late tokens have more history to repeat), and every decaying state drifts")
    print("     with position, so the 'raw' column credits that confound -- watch the")
    print("     'control: position' row score high on raw and collapse on pos-matched.")
    print("     A design must beat both controls AND the 'no state' row to earn its state.")
    print("  C: 'NO' in the decay=1 row is the saturation wall; it tells you whether")
    print("     aggressive forgetting is optional or mandatory for a matrix state.")


if __name__ == "__main__":
    main()
