from __future__ import annotations

import argparse
import math
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import concatenate_datasets
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, get_cosine_schedule_with_warmup, set_seed
import warnings

from datautils import load_datasets_for_training, SimplePaddingCollator
from data_load import load_math, load_longbench_bundle, load_c4, load_longalpaca
from kvpress.presses.indexer_score_press import IndexerScorePress
from trainer_utils import compute_press_loss, compute_indexer_warmup_loss, build_dense_warmup_targets_kl
from transformers.models.llama.modeling_llama import eager_attention_forward
from transformers.models.llama.modeling_llama import repeat_kv


def eager_attention_forward_mean(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    chunk_size = kw.get("chunk_size", global_chunk_size)

    key_states = repeat_kv(key, module.num_key_value_groups)  # (b, n_kv, q, d)
    value_states = repeat_kv(value, module.num_key_value_groups)

    bsz, n_heads, q_len, head_dim = query.size()
    k_len = key_states.size(-2)

    # 第一遍：online softmax 统计全局 max 与 sum_exp（严格全局 softmax 的归一化项）
    global_max = None
    sum_exp = None
    for k_start in range(0, k_len, chunk_size):
        k_end = min(k_start + chunk_size, k_len)
        k_slice = key_states[:, :, k_start:k_end, :]

        logits = torch.matmul(query, k_slice.transpose(2, 3)) * scaling  # (b, h, q, chunk)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, k_start:k_end]
            logits = logits + causal_mask

        logits_f = logits.float()
        chunk_max = logits_f.max(dim=-1).values  # (b, h, q)
        if global_max is None:
            global_max = chunk_max
            shifted = logits_f - global_max.unsqueeze(-1)
            shifted = shifted.masked_fill(torch.isnan(shifted), float("-inf"))
            exp_chunk = torch.exp(shifted)
            sum_exp = exp_chunk.sum(dim=-1)
            del shifted, exp_chunk
        else:
            new_max = torch.maximum(global_max, chunk_max)
            diff = global_max - new_max
            diff = diff.masked_fill(torch.isnan(diff), 0.0)
            sum_exp = sum_exp * torch.exp(diff)
            shifted = logits_f - new_max.unsqueeze(-1)
            shifted = shifted.masked_fill(torch.isnan(shifted), float("-inf"))
            sum_exp = sum_exp + torch.exp(shifted).sum(dim=-1)
            global_max = new_max
            del new_max, diff, shifted

        del k_slice, logits, logits_f, chunk_max

    # 第二遍：严格全局 softmax + 输出（仍按 chunk 计算以省显存）
    sum_exp = sum_exp.clamp(min=1e-8)
    denom = sum_exp.unsqueeze(-1)  # (b, h, q, 1)
    attn_output = query.new_zeros((bsz, n_heads, q_len, head_dim))
    attn_mean_chunks = []
    for k_start in range(0, k_len, chunk_size):
        k_end = min(k_start + chunk_size, k_len)
        k_slice = key_states[:, :, k_start:k_end, :]
        v_slice = value_states[:, :, k_start:k_end, :]

        logits = torch.matmul(query, k_slice.transpose(2, 3)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, k_start:k_end]
            logits = logits + causal_mask

        shifted = logits.float() - global_max.unsqueeze(-1)
        shifted = shifted.masked_fill(torch.isnan(shifted), float("-inf"))
        exp_chunk = torch.exp(shifted)
        probs = (exp_chunk / denom).to(query.dtype)
        probs = nn.functional.dropout(probs, p=dropout, training=module.training)

        attn_output = attn_output + torch.matmul(probs, v_slice)
        attn_mean_chunks.append(probs.mean(dim=1))  # (b, q, chunk)

        del k_slice, v_slice, logits, shifted, exp_chunk, probs

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_weights_mean = torch.cat(attn_mean_chunks, dim=-1) if len(attn_mean_chunks) > 1 else attn_mean_chunks[0]
    del attn_mean_chunks, global_max, sum_exp, denom, key_states, value_states
    return attn_output, attn_weights_mean

import transformers.models.llama.modeling_llama as m
m.eager_attention_forward = eager_attention_forward_mean

attn_by_layer = {}
per_layer_losses = []
current_attention_mask = None
current_labels = None
global_chunk_size = 128  # 默认值，会在 main 中被 args.chunk_size 覆盖

def _make_attn_hook(layer_idx: int):
    def hook(module, inputs, output):
        if isinstance(output, tuple) and len(output) >= 2:
            attn_output, attn_weights = output[0], output[1]
            if attn_weights is not None:
                attn_by_layer[layer_idx] = attn_weights
            return (attn_output, None)
        return output
    return hook

def _make_decoder_hook(layer_idx: int, press, n_sink):
    def hook(module, inputs, output):
        # LlamaDecoderLayer forward returns (hidden_states, self_attn_weights, present_key_value, ...)
        hidden_states = output[0] if isinstance(output, tuple) else output
        attn = attn_by_layer.pop(layer_idx, None)
        if attn is None:
            return output
        loss = compute_indexer_layer_loss(
            attn,
            hidden_states,
            current_attention_mask,
            current_labels,
            module.self_attn,
            press,
            n_sink,
        )
        per_layer_losses.append(loss)
        del attn
        del hidden_states
        return output
    return hook

def compute_indexer_layer_loss(attn, hidden_state, attention_mask, labels, module, press, n_sink):
    def _grad_safe_zero():
        # Return a scalar zero that still requires grad (anchors to indexer params).
        try:
            p0 = next(p for p in module.parameters() if p.requires_grad)
            return p0.sum() * 0.0
        except StopIteration:
            return hidden_state.sum() * 0.0

    # attn: (bsz, num_heads, q_len, k_len) 或 (bsz, q_len, k_len)
    if attn.dim() == 4:
        attn = attn.mean(dim=1)
    attn_f = attn.float()

    bsz, q_len, k_len = attn_f.shape
    if attention_mask is None:
        # 没有 mask 时只能退化成全 1
        attention_mask = attn_f.new_ones((bsz, k_len), dtype=torch.long)

    if labels is None:
        # Unsupervised (e.g. C4): treat all non-pad tokens as output; instruction region is [n_sink, end).
        output_q = attention_mask[:, :q_len] > 0
        instr_end = torch.full((bsz,), q_len, device=attention_mask.device, dtype=torch.long)
        dev = attention_mask.device
    else:
        # output query mask（SFT：labels!=-100 代表 output 部分）
        output_q = (labels != -100) & (attention_mask > 0)  # (bsz, q_len)
        # 每个样本的 instruction 结束位置：第一个 output token 的 index
        out_pos = labels != -100
        has_out = out_pos.any(dim=1)
        first_out = out_pos.int().argmax(dim=1)
        instr_end = torch.where(
            has_out,
            first_out,
            torch.full((bsz,), q_len, device=labels.device, dtype=torch.long),
        )  # (bsz,)
        dev = labels.device

    # instruction key mask（只保留全局 [n_sink, instr_end) 这段 key）
    k_idx = torch.arange(k_len, device=dev).view(1, -1)  # (1, k_len)
    attn_keep_k = attention_mask > 0
    instr_k = (k_idx >= n_sink) & (k_idx < instr_end.view(-1, 1)) & attn_keep_k  # (bsz, k_len)
    # Fallback for very short sequences: if instr_k is empty for a sample, use all valid keys.
    empty_rows = instr_k.sum(dim=1) == 0
    if empty_rows.any():
        instr_k = torch.where(empty_rows.view(-1, 1), attn_keep_k, instr_k)

    # target 的归一化分母：每个 query 对 instruction keys 的 attention mass
    instr_mass = (attn_f * instr_k.unsqueeze(1).float()).sum(dim=-1)  # (bsz, q_len)
    query_valid = output_q & (instr_mass > 0)
    instr_mass = instr_mass.clamp(min=1e-8)

    # loss 归一化项：按 (query,key) 元素数平均
    denom = (query_valid.sum(dim=1).float() * instr_k.sum(dim=1).float()).sum().clamp(min=1.0)

    kwargs = {"attention_mask": attention_mask[:, None, None, :]}
    chunk_size = getattr(press, "chunk_size", global_chunk_size)

    # 第一遍：全局 max（只覆盖 instruction keys）
    max_val = None
    for i, chunk in enumerate(press.indexer_logits_chunks(module, hidden_state, kwargs, chunk_size=chunk_size)):
        k_start = i * chunk_size
        k_end = k_start + chunk.size(-1)
        valid_k = instr_k[:, k_start:k_end]  # (bsz, t_chunk)
        if not valid_k.any():
            del chunk
            continue
        chunk_f = chunk.float().masked_fill(~valid_k.unsqueeze(1), float("-inf"))
        local_max = chunk_f.max(dim=-1).values  # (bsz, q_len)
        max_val = local_max if max_val is None else torch.maximum(max_val, local_max)
        del chunk, chunk_f, local_max

    if max_val is None:
        # 没有任何 instruction key（比如 n_sink>=instr_end）
        del attn_f, instr_k, instr_mass
        return _grad_safe_zero()

    # 第二遍：sum_exp（严格全局 softmax 分母，只覆盖 instruction keys）
    sum_exp = torch.zeros_like(max_val, dtype=torch.float32)
    for i, chunk in enumerate(press.indexer_logits_chunks(module, hidden_state, kwargs, chunk_size=chunk_size)):
        k_start = i * chunk_size
        k_end = k_start + chunk.size(-1)
        valid_k = instr_k[:, k_start:k_end]
        if not valid_k.any():
            del chunk
            continue
        chunk_f = chunk.float().masked_fill(~valid_k.unsqueeze(1), float("-inf"))
        shifted = chunk_f - max_val.unsqueeze(-1)
        shifted = shifted.masked_fill(torch.isnan(shifted), float("-inf"))
        sum_exp = sum_exp + torch.exp(shifted).sum(dim=-1)
        del chunk, chunk_f, shifted

    log_denom = max_val + torch.log(sum_exp.clamp(min=1e-8))
    del max_val, sum_exp

    # 第三遍：分块 KL（只对 output queries × instruction keys 计入 loss）
    loss_num = hidden_state.new_zeros((), dtype=torch.float32)
    for i, chunk in enumerate(press.indexer_logits_chunks(module, hidden_state, kwargs, chunk_size=chunk_size)):
        k_start = i * chunk_size
        k_end = k_start + chunk.size(-1)
        valid_k = instr_k[:, k_start:k_end]  # (bsz, t_chunk)
        if not valid_k.any():
            del chunk
            continue

        chunk_f = chunk.float().masked_fill(~valid_k.unsqueeze(1), float("-inf"))
        log_probs_chunk = chunk_f - log_denom.unsqueeze(-1)  # (bsz, q_len, t_chunk)

        attn_chunk = attn_f[:, :, k_start:k_end]  # (bsz, q_len, t_chunk)
        target_chunk = (attn_chunk * valid_k.unsqueeze(1).float()) / instr_mass.unsqueeze(-1)

        mask_chunk = query_valid.unsqueeze(-1) & valid_k.unsqueeze(1)
        # NOTE:
        # `log_probs_chunk` contains `-inf` for masked (invalid) keys. Even if we multiply by a 0/False mask later,
        # some ops (e.g. KL) can still produce NaNs at masked positions (NaN * 0 = NaN). So we zero-out masked
        # positions BEFORE computing KL to keep the loss numerically stable.
        log_probs_chunk = torch.where(mask_chunk, log_probs_chunk, torch.zeros_like(log_probs_chunk))
        target_chunk = torch.where(mask_chunk, target_chunk, torch.zeros_like(target_chunk))

        kl_chunk = F.kl_div(log_probs_chunk, target_chunk, reduction="none")
        loss_num = loss_num + kl_chunk.sum()

        del chunk, chunk_f, log_probs_chunk, attn_chunk, target_chunk, mask_chunk, kl_chunk

    loss = loss_num / denom
    del attn_f, instr_k, instr_mass, query_valid, log_denom, denom, loss_num
    return loss

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
    parser.add_argument("--press_method", type=str, default="dma_score", choices=["dma_score", "indexer_score"])
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
    global current_attention_mask
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
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

    # train_dataset, eval_dataset = load_longbench_bundle(tokenizer, args)
    train_dataset, eval_dataset = load_c4(tokenizer, args)
    # train_dataset, eval_dataset = load_math(tokenizer, args)
    # train_dataset, eval_dataset = load_longalpaca(tokenizer, args)

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
        model.gradient_checkpointing_enable()
        model.config.use_cache = False


    if args.press_method == "dma_score":
        # press = DMAScorePress(n_sink=args.n_sink)
        pass
    elif args.press_method == "indexer_score":
        press = IndexerScorePress(n_sink=args.n_sink, chunk_size=args.chunk_size)
    press.post_init_from_model(model)
    
    # 设置全局 chunk_size 供 eager_attention_forward_mean 使用
    global global_chunk_size
    global_chunk_size = args.chunk_size

    for name, param in model.named_parameters():
        if press.scorer_attr not in name:
            param.requires_grad = False

    scorer_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(scorer_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.num_train_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    attn_modules = list(get_language_model_layers(model))
    attn_modules = [layer.self_attn for layer in attn_modules]
    decoder_modules = list(get_language_model_layers(model))
    attn_hooks = []
    decoder_hooks = []

    model, optimizer, train_loader, eval_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, lr_scheduler
    )

    attn_modules = list(get_language_model_layers(model))
    decoder_modules = attn_modules  # same layers
    attn_modules = [layer.self_attn for layer in attn_modules]
    attn_hooks = [m.register_forward_hook(_make_attn_hook(i)) for i, m in enumerate(attn_modules)]
    decoder_hooks = [m.register_forward_hook(_make_decoder_hook(i, press, args.n_sink)) for i, m in enumerate(decoder_modules)]

    global_step = 0
    running_loss = 0.0
    running_grad_norm = 0.0
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            with accelerator.accumulate(model):
                if args.press_method == "dma_score":
                    # 不需要语言模型的 CE loss，去掉 labels 可显著降低峰值显存（长序列下尤其明显）
                    forward_batch = {k: v for k, v in batch.items() if k != "labels"}
                    outputs = model(**forward_batch, output_attentions=True, use_cache=True, return_dict=True)
                    loss = compute_press_loss(outputs.past_key_values, outputs.attentions, batch["attention_mask"], batch["labels"], attn_modules, press.scorer_attr, args.n_sink, args.aggregate_mode)
                    del outputs
                elif args.press_method == "indexer_score":
                    attn_by_layer.clear()
                    per_layer_losses.clear()
                    global current_attention_mask
                    current_attention_mask = batch["attention_mask"]
                    global current_labels
                    current_labels = batch.get("labels", None)
                    # 同上：去掉 labels，避免 forward 内部额外计算/缓存 cross entropy
                    forward_batch = {k: v for k, v in batch.items() if k != "labels"}
                    outputs = model(**forward_batch, output_attentions=False, output_hidden_states=False, use_cache=False, return_dict=True)
                    current_attention_mask = None
                    current_labels = None
                    loss_device = batch["input_ids"].device
                    del outputs  # 立即删除 outputs 以释放显存（避免保留巨大的 logits）
                    if len(per_layer_losses) > 0:
                        loss = torch.stack(per_layer_losses).mean()
                    else:
                        # Grad-safe zero to avoid backward crash when a batch yields no valid loss.
                        try:
                            p0 = next(p for p in model.parameters() if p.requires_grad)
                            loss = p0.sum() * 0.0
                        except StopIteration:
                            loss = torch.tensor(0.0, device=loss_device)
                    per_layer_losses.clear()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    # grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=float('inf'))
                    running_grad_norm += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

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
    for h in attn_hooks:
        h.remove()
    for h in decoder_hooks:
        h.remove()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(args.output_dir, safe_serialization=False)
        tokenizer.model_max_length = max_token_length
        tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()