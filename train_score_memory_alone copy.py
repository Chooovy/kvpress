from __future__ import annotations

import argparse
import math
import os
import importlib

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import concatenate_datasets
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, get_cosine_schedule_with_warmup, set_seed
import warnings

from datautils import load_datasets_for_training, SimplePaddingCollator
from data_load import load_math, load_longbench_bundle, load_c4, load_longalpaca
from kvpress.presses.memory_scorer_press import MemoryScorerPress
from transformers.models.llama.modeling_llama import repeat_kv
from kvpress import ExpectedAttentionPress, SnapKVPress, KeyDiffPress


# =========================
# Memory residual loss hook
# =========================
per_layer_memory_losses = []
current_attention_mask = None  # (bsz, seqlen) int/bool
current_labels = None  # (bsz, seqlen) or None
kvpress_collect_memory_losses = False
global_chunk_size = 128  # will be overwritten by args.chunk_size

# Memory training globals (configured in main)
current_memory_press = None
current_memory_compression_ratio = 0.0


def _kvpress_monkeypatch_attention_impl(model):
    """
    Ensure the model actually calls our eager attention forward.
    Many checkpoints (e.g. Mistral/Qwen/Gemma) don't use `transformers.models.llama.modeling_llama`.
    """
    global repeat_kv
    candidates = [
        "transformers.models.llama.modeling_llama",
        "transformers.models.mistral.modeling_mistral",
        "transformers.models.qwen3.modeling_qwen3",
        "transformers.models.gemma3.modeling_gemma3",
    ]
    patched = []
    for mod_path in candidates:
        try:
            mod = importlib.import_module(mod_path)
        except Exception:
            continue
        if hasattr(mod, "eager_attention_forward"):
            mod.eager_attention_forward = eager_attention_forward_mean
            patched.append(mod_path)
        if hasattr(mod, "repeat_kv"):
            repeat_kv = mod.repeat_kv
    # Force eager path in config when supported.
    try:
        model.config._attn_implementation = "eager"
    except Exception:
        pass
    try:
        model.config.attn_implementation = "eager"
    except Exception:
        pass
    return patched


def _make_attn_prehook(layer_idx: int):
    def prehook(module, inputs, kwargs=None):
        # HF models may call attention with keyword-only args (inputs can be empty).
        hidden_states = None
        if inputs is not None and len(inputs) > 0:
            hidden_states = inputs[0]
        elif kwargs is not None:
            hidden_states = kwargs.get("hidden_states", None)
            if hidden_states is None:
                hidden_states = kwargs.get("x", None)
        if hidden_states is not None:
            module._kvpress_hidden_states = hidden_states
        module._kvpress_layer_idx = layer_idx
        # Some Press implementations look for `module.layer_idx`.
        try:
            module.layer_idx = layer_idx
        except Exception:
            pass
        return None

    return prehook


def _mask_q_from_batch(attention_mask_2d: torch.Tensor | None, labels: torch.Tensor | None) -> torch.Tensor | None:
    if attention_mask_2d is None:
        return None
    if labels is None:
        return attention_mask_2d > 0
    return (labels != -100) & (attention_mask_2d > 0)


def eager_attention_forward_mean(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    """
    Teacher: full attention output (FlashAttention/SDPA when available).
    Student/save: o_save = softmax(q K_keep) V_keep (strict softmax restricted to kept keys).
    Memory: pred = g(q) ⊙ m(q) built from evicted KV (single-shot write), loss = MSE(pred, Δ).
    """
    chunk_size = int(kw.get("chunk_size", global_chunk_size))

    # Expand KV heads -> attention heads (GQA/MQA).
    key_states = repeat_kv(key, module.num_key_value_groups)  # (bsz, n_heads, k_len, d)
    value_states = repeat_kv(value, module.num_key_value_groups)

    bsz, n_heads, q_len, head_dim = query.size()
    k_len = key_states.size(-2)

    # ---- Teacher full attention output (prefer SDPA/FlashAttention kernels) ----
    # PyTorch SDPA always uses 1/sqrt(d) scaling internally; compensate to match `scaling`.
    # Desired logits: (q @ k^T) * scaling
    q_sdpa = query * (float(scaling) * math.sqrt(float(head_dim)))
    try:
        attn_out = F.scaled_dot_product_attention(
            q_sdpa,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=float(dropout) if module.training else 0.0,
            is_causal=False,
        )  # (bsz, n_heads, q_len, head_dim)
    except TypeError:
        # Older torch: no is_causal arg
        attn_out = F.scaled_dot_product_attention(
            q_sdpa,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=float(dropout) if module.training else 0.0,
        )

    # (bsz, q_len, n_heads, head_dim) to match HF eager_attention_forward output format
    attn_output = attn_out.transpose(1, 2).contiguous()

    # ---- Memory residual loss (train kvpress_memory only) ----
    if kvpress_collect_memory_losses and (current_memory_press is not None) and hasattr(module, "_kvpress_hidden_states"):
        mem_mod = getattr(module, "kvpress_memory", None)
        if mem_mod is not None:
            hidden_states = getattr(module, "_kvpress_hidden_states")

            # Score keys using base_press to decide keep vs evict (non-differentiable top-k selection).
            extra_kwargs = {}
            if kw.get("position_embeddings", None) is not None:
                extra_kwargs["position_embeddings"] = kw.get("position_embeddings")
            if kw.get("indexer_freqs_cis", None) is not None:
                extra_kwargs["indexer_freqs_cis"] = kw.get("indexer_freqs_cis")
            try:
                scores_kv = current_memory_press.base_press.score(  # type: ignore[union-attr]
                    module, hidden_states, key, value, attentions=None, kwargs=extra_kwargs
                )
            except Exception:
                scores_kv = None

            if scores_kv is not None and scores_kv.numel() > 0:
                n_kv_heads = key.size(1)
                n_groups = module.num_key_value_groups
                n_kept = max(1, int(key.size(2) * (1.0 - float(current_memory_compression_ratio))))
                kept_idx = scores_kv.topk(n_kept, dim=-1).indices  # (bsz, n_kv, n_kept)

                keep_mask_kv = torch.zeros((bsz, n_kv_heads, key.size(2)), device=key.device, dtype=torch.bool)
                keep_mask_kv.scatter_(2, kept_idx, True)
                keep_mask_heads = keep_mask_kv.repeat_interleave(n_groups, dim=1)  # (bsz, n_heads, k_len)

                # Compute o_keep (o_save): strict softmax attention output restricted to kept keys.
                #
                # If n_kept is small enough, we can gather K/V (and the matching attention_mask slice)
                # and rely on SDPA/FlashAttention to avoid materializing full logits.
                # If n_kept is large, keep the chunked exact softmax to stay memory safe.
                o_keep = None
                if n_kept <= chunk_size:
                    idx_exp_kv = kept_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    k_keep_kv = key.gather(2, idx_exp_kv).contiguous()  # (bsz, n_kv, n_kept, d)
                    v_keep_kv = value.gather(2, idx_exp_kv).contiguous()
                    k_keep = repeat_kv(k_keep_kv, n_groups)  # (bsz, n_heads, n_kept, d)
                    v_keep = repeat_kv(v_keep_kv, n_groups)

                    attn_mask_keep = None
                    if attention_mask is not None:
                        am = attention_mask
                        if am.dim() == 4 and am.size(1) == 1 and n_heads > 1:
                            am = am.expand(bsz, n_heads, q_len, k_len)
                        if am.dim() == 4 and am.size(-1) == k_len:
                            kept_idx_heads = kept_idx.repeat_interleave(n_groups, dim=1)  # (bsz, n_heads, n_kept)
                            gather_idx = kept_idx_heads.unsqueeze(2).expand(bsz, n_heads, q_len, n_kept)
                            attn_mask_keep = am.gather(-1, gather_idx).contiguous()
                        else:
                            # Unknown mask layout; fall back to no mask (common for pure causal SDPA path).
                            attn_mask_keep = None

                    try:
                        o_keep_h = F.scaled_dot_product_attention(
                            q_sdpa,
                            k_keep,
                            v_keep,
                            attn_mask=attn_mask_keep,
                            dropout_p=float(dropout) if module.training else 0.0,
                            is_causal=False,
                        )  # (bsz, n_heads, q_len, d)
                    except TypeError:
                        o_keep_h = F.scaled_dot_product_attention(
                            q_sdpa,
                            k_keep,
                            v_keep,
                            attn_mask=attn_mask_keep,
                            dropout_p=float(dropout) if module.training else 0.0,
                        )
                    o_keep = o_keep_h.transpose(1, 2).contiguous()  # (bsz, q_len, n_heads, d)
                else:
                    keep_max = None
                    keep_sum_exp = None
                    any_keep = False
                    for k_start in range(0, k_len, chunk_size):
                        k_end = min(k_start + chunk_size, k_len)
                        valid_k = keep_mask_heads[:, :, k_start:k_end]  # (bsz, h, t_chunk)
                        if not valid_k.any():
                            continue
                        any_keep = True
                        k_slice = key_states[:, :, k_start:k_end, :]
                        logits = torch.matmul(query, k_slice.transpose(2, 3)) * float(scaling)
                        if attention_mask is not None:
                            logits = logits + attention_mask[:, :, :, k_start:k_end]
                        logits_f = logits.float().masked_fill(~valid_k.unsqueeze(2), float("-inf"))
                        chunk_max = logits_f.max(dim=-1).values  # (bsz, h, q)
                        if keep_max is None:
                            keep_max = chunk_max
                            shifted = logits_f - keep_max.unsqueeze(-1)
                            shifted = shifted.masked_fill(torch.isnan(shifted), float("-inf"))
                            keep_sum_exp = torch.exp(shifted).sum(dim=-1)
                            del shifted
                        else:
                            new_max = torch.maximum(keep_max, chunk_max)
                            diff = keep_max - new_max
                            diff = diff.masked_fill(torch.isnan(diff), 0.0)
                            keep_sum_exp = keep_sum_exp * torch.exp(diff)
                            shifted = logits_f - new_max.unsqueeze(-1)
                            shifted = shifted.masked_fill(torch.isnan(shifted), float("-inf"))
                            keep_sum_exp = keep_sum_exp + torch.exp(shifted).sum(dim=-1)
                            keep_max = new_max
                            del new_max, diff, shifted
                        del k_slice, logits, logits_f, chunk_max

                    if any_keep and keep_max is not None and keep_sum_exp is not None:
                        keep_sum_exp = keep_sum_exp.clamp(min=1e-8)
                        keep_denom = keep_sum_exp.unsqueeze(-1)  # (bsz,h,q,1)
                        o_keep_h = query.new_zeros((bsz, n_heads, q_len, head_dim))
                        for k_start in range(0, k_len, chunk_size):
                            k_end = min(k_start + chunk_size, k_len)
                            valid_k = keep_mask_heads[:, :, k_start:k_end]
                            if not valid_k.any():
                                continue
                            k_slice = key_states[:, :, k_start:k_end, :]
                            v_slice = value_states[:, :, k_start:k_end, :]
                            logits = torch.matmul(query, k_slice.transpose(2, 3)) * float(scaling)
                            if attention_mask is not None:
                                logits = logits + attention_mask[:, :, :, k_start:k_end]
                            logits_f = logits.float().masked_fill(~valid_k.unsqueeze(2), float("-inf"))
                            shifted = logits_f - keep_max.unsqueeze(-1)
                            shifted = shifted.masked_fill(torch.isnan(shifted), float("-inf"))
                            exp_chunk = torch.exp(shifted)
                            probs = (exp_chunk / keep_denom).to(query.dtype)
                            probs = nn.functional.dropout(probs, p=float(dropout), training=module.training)
                            o_keep_h = o_keep_h + torch.matmul(probs, v_slice)
                            del k_slice, v_slice, logits, logits_f, shifted, exp_chunk, probs
                        o_keep = o_keep_h.transpose(1, 2).contiguous()  # (bsz, q_len, n_heads, head_dim)

                if o_keep is not None:
                    # Residual target: Δ = o_full - o_keep (detach target).
                    delta = (attn_output - o_keep).detach()

                    # Build memory state from evicted KV (single-shot write for this forward).
                    evict_mask_kv = ~keep_mask_kv  # (bsz, n_kv, k_len)
                    phi_k = mem_mod.phi(key)  # (bsz, n_kv, k_len, d_phi)
                    phi_ev = phi_k * evict_mask_kv.unsqueeze(-1)
                    v_ev = value * evict_mask_kv.unsqueeze(-1)
                    eta = mem_mod.eta()
                    A = torch.einsum("bhkd,bhke->bhde", phi_ev, v_ev) * eta  # (bsz, n_kv, d_phi, head_dim)
                    b_mem = (phi_ev ** 2).sum(dim=2) * eta  # (bsz, n_kv, d_phi)

                    # Readout: m(q)
                    qg = query.view(bsz, n_kv_heads, n_groups, q_len, head_dim)
                    phi_q = mem_mod.phi(qg)  # (bsz, n_kv, n_groups, q_len, d_phi)
                    m = torch.einsum("bhgqd,bhde->bhgqe", phi_q, A)
                    denom_m = torch.einsum("bhgqd,bhd->bhgq", phi_q ** 2, b_mem).unsqueeze(-1)
                    denom_m = denom_m.clamp(min=1e-6)
                    m = m / denom_m
                    m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
                    m = m.reshape(bsz, n_heads, q_len, head_dim).transpose(1, 2).contiguous()

                    gate = mem_mod.gate()
                    pred = m * gate

                    mask_q = _mask_q_from_batch(current_attention_mask, current_labels)
                    if mask_q is None:
                        mse = (pred.float() - delta.float()) ** 2
                        denom_mse = float(pred.numel())
                        mem_loss = mse.mean() if denom_mse > 0 else pred.sum() * 0.0
                    else:
                        mse = (pred.float() - delta.float()) ** 2
                        mse = mse * mask_q.unsqueeze(-1).unsqueeze(-1).float()
                        denom_mse = (mask_q.sum().float().clamp(min=1.0) * float(n_heads * head_dim))
                        mem_loss = mse.sum() / denom_mse
                    per_layer_memory_losses.append(mem_loss)

    return attn_output, None




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train learnable KV cache scorer.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=1000000)
    parser.add_argument("--save_steps", type=int, default=1000000)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=32)
    parser.add_argument("--press_method", type=str, default="EA")
    parser.add_argument("--aggregate_mode", type=str, default="mean")
    parser.add_argument("--n_sink", type=int, default=4)
    parser.add_argument("--pt_context_len", type=int, default=8192)
    parser.add_argument("--preprocessing_num_workers", type=int, default=32)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Enable gradient checkpointing to save memory")
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_wiki_ppl", action="store_true", help="Evaluate WikiText PPL after training")
    parser.add_argument("--wiki_compression_ratios", type=float, nargs="+", default=[1.0, 0.8, 0.6, 0.4, 0.2], help="Compression ratios for WikiText evaluation")
    parser.add_argument("--wiki_context_length", type=int, default=2048, help="Context length for WikiText evaluation")
    parser.add_argument("--wiki_decode_length", type=int, default=512, help="Decode length for WikiText evaluation")
    parser.add_argument("--wiki_num_samples", type=int, default=5, help="Number of samples for WikiText evaluation")
    parser.add_argument("--chunk_size", type=int, default=128, help="Chunk size for processing long sequences")

    # Memory module hyper-params
    parser.add_argument("--memory_d_phi", type=int, default=-1, help="Per-layer KVPressMemoryLayer d_phi. -1 => head_dim.")
    parser.add_argument("--memory_compression_ratio", type=float, default=0.5, help="Define kept vs evicted keys for memory residual loss.")
    parser.add_argument("--memory_loss_weight", type=float, default=1.0, help="Weight for memory residual MSE in stage2.")
    return parser.parse_args()




def get_language_model_layers(model: nn.Module):
    lm = model
    if hasattr(lm, "module"):  # Deepspeed / DDP wrapper
        lm = lm.module
    if hasattr(lm, "model"):
        lm = lm.model
    if hasattr(lm, "language_model"):
        lm = lm.language_model
    return lm.layers



def main():
    args = parse_args()
    global current_attention_mask, current_labels
    # DDP + gradient checkpointing can trigger "marked ready twice" with re-entrant backward.
    # `static_graph=True` is a common workaround when the forward graph is stable.
    try:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, static_graph=True)
    except TypeError:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.print(args)

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    max_token_length = tokenizer.model_max_length
    tokenizer.model_max_length = args.pt_context_len

    # 加载多个数据集并合并
    datasets_train = []
    datasets_eval = []
    
    # 加载 C4 数据集
    train_c4, eval_c4 = load_c4(tokenizer, args)
    datasets_train.append(train_c4)
    datasets_eval.append(eval_c4)
    
    # 加载 Math 数据集（取消注释以启用）
    train_math, eval_math = load_math(tokenizer, args)
    datasets_train.append(train_math)
    datasets_eval.append(eval_math)
    
    # 加载 LongAlpaca 数据集（取消注释以启用）
    train_longalpaca, eval_longalpaca = load_longalpaca(tokenizer, args)
    datasets_train.append(train_longalpaca)
    datasets_eval.append(eval_longalpaca)
    
    # 加载 LongBench bundle 数据集（取消注释以启用）
    # train_longbench, eval_longbench = load_longbench_bundle(tokenizer, args)
    # datasets_train.append(train_longbench)
    # datasets_eval.append(eval_longbench)
    
    # 合并所有数据集
    if len(datasets_train) > 1:
        train_dataset = concatenate_datasets(datasets_train)
        eval_dataset = concatenate_datasets(datasets_eval)
        accelerator.print(f"Merged datasets - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    else:
        train_dataset = datasets_train[0]
        eval_dataset = datasets_eval[0]

    warnings.filterwarnings("ignore", message=".*Creating a tensor from a list of numpy.ndarrays.*")


    pad_to_multiple = 8 if accelerator.mixed_precision in ("fp16", "bf16") else None
    # data_collator = DataCollatorForSeq2Seq(
    #     tokenizer=tokenizer,
    #     model=None,
    #     padding="longest",
    #     pad_to_multiple_of=pad_to_multiple,
    #     label_pad_token_id=-100,
    # )
    data_collator = SimplePaddingCollator(
        tokenizer=tokenizer,
        padding="longest",
        pad_to_multiple_of=pad_to_multiple,
        label_pad_token_id=-100,
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        shuffle=False,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else None,
    )
    model.config.output_attentions = True
    model.config.use_cache = True

    if hasattr(args, 'gradient_checkpointing') and args.gradient_checkpointing:
        # Prefer non-reentrant checkpointing when available to avoid DDP "marked ready twice".
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        model.config.use_cache = False

    _ = _kvpress_monkeypatch_attention_impl(model)



    if args.press_method == "EA":
        base_press = ExpectedAttentionPress()
    elif args.press_method == "snapkv":
        base_press = SnapKVPress()
    elif args.press_method == "keydiff":
        base_press = KeyDiffPress()


    d_phi = None if int(args.memory_d_phi) <= 0 else int(args.memory_d_phi)
    press = MemoryScorerPress(
        base_press=base_press,
        compression_ratio=float(args.memory_compression_ratio),
        d_phi=d_phi,
        use_denominator=True,
    )
    press.post_init_from_model(model)

    # Configure memory globals for stage2 loss (only meaningful for MemoryScorerPress).
    global current_memory_press, current_memory_compression_ratio
    if isinstance(press, MemoryScorerPress):
        current_memory_press = press
        current_memory_compression_ratio = float(args.memory_compression_ratio)
    
    # 设置全局 chunk_size 供 eager_attention_forward_mean 使用
    global global_chunk_size
    global_chunk_size = args.chunk_size

    # Memory-only training: freeze everything except per-layer `kvpress_memory` modules.
    memory_params = []
    for name, param in model.named_parameters():
        if "kvpress_memory" in name:
            param.requires_grad = True
            memory_params.append(param)
        else:
            param.requires_grad = False
    if len(memory_params) == 0:
        raise RuntimeError("No trainable params found: expected modules named '*kvpress_memory*'. Did you call press.post_init_from_model(model)?")
    optimizer = torch.optim.AdamW(memory_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.num_train_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    attn_modules = list(get_language_model_layers(model))
    attn_modules = [layer.self_attn for layer in attn_modules]
    attn_prehooks = []

    model, optimizer, train_loader, eval_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, lr_scheduler
    )

    attn_modules = list(get_language_model_layers(model))
    attn_modules = [layer.self_attn for layer in attn_modules]
    attn_prehooks = []
    for i, mod in enumerate(attn_modules):
        hook_fn = _make_attn_prehook(i)
        try:
            attn_prehooks.append(mod.register_forward_pre_hook(hook_fn, with_kwargs=True))
        except TypeError:
            # Older torch: no with_kwargs support, will only work if hidden_states is positional.
            attn_prehooks.append(mod.register_forward_pre_hook(hook_fn))

    global_step = 0
    running_loss = 0.0
    running_grad_norm = 0.0
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            with accelerator.accumulate(model):
                per_layer_memory_losses.clear()
                current_attention_mask = batch.get("attention_mask", None)
                current_labels = batch.get("labels", None)
                global kvpress_collect_memory_losses
                kvpress_collect_memory_losses = True
                # Avoid CE/logits allocation: drop labels.
                forward_batch = {k: v for k, v in batch.items() if k != "labels"}
                outputs = model(**forward_batch, output_attentions=False, output_hidden_states=False, use_cache=False, return_dict=True)
                del outputs
                kvpress_collect_memory_losses = False
                current_attention_mask = None
                current_labels = None

                loss_device = batch["input_ids"].device
                if len(per_layer_memory_losses) > 0:
                    loss = torch.stack(per_layer_memory_losses).mean() * float(args.memory_loss_weight)
                else:
                    # Keep a grad-requiring zero loss to avoid DDP/Accelerate backward crash.
                    try:
                        p0 = next(p for p in model.parameters() if p.requires_grad)
                        loss = p0.sum() * 0.0
                    except StopIteration:
                        loss = torch.tensor(0.0, device=loss_device)
                per_layer_memory_losses.clear()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(memory_params, max_norm=1.0)
                    running_grad_norm += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

                # IMPORTANT: Only step optimizer/scheduler on true optimizer steps
                # (i.e. when gradients are synchronized). Otherwise gradient_accumulation_steps
                # is effectively ignored and `total_steps/warmup_steps` become inconsistent.
                if accelerator.sync_gradients:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            running_loss += loss.detach().float()
            global_step += 1

            if global_step % args.logging_steps == 0:
                avg_loss = (running_loss / args.logging_steps).item()
                avg_grad_norm = running_grad_norm / args.logging_steps
                accelerator.print(f"Epoch {epoch} Step {global_step} / Total Steps {total_steps}: train_loss={avg_loss:.6f}, train_grad_norm={avg_grad_norm:.6f}")
                running_loss = 0.0
                running_grad_norm = 0.0

            # if global_step % args.eval_steps == 0:
            #     eval_loss = evaluate_epoch(
            #         accelerator, model, eval_loader, attn_modules, press.scorer_attr, args.n_sink, args.aggregate_mode
            #     )
            #     accelerator.print(f"[Eval] Step {global_step}: loss={eval_loss:.6f}")

            if global_step % args.save_steps == 0 and accelerator.is_main_process:
                accelerator.unwrap_model(model).save_pretrained(
                    os.path.join(args.output_dir, f"checkpoint-{global_step}"),
                    safe_serialization=False,
                )

    accelerator.wait_for_everyone()
    for h in attn_prehooks:
        h.remove()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(args.output_dir, safe_serialization=False)
        tokenizer.model_max_length = max_token_length
        tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()