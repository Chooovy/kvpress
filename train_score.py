from __future__ import annotations

import argparse
import math
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import concatenate_datasets
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, get_cosine_schedule_with_warmup, set_seed
import warnings

from datautils import load_datasets_for_training, SimplePaddingCollator
from data_load import load_math, load_longbench_bundle, load_c4, load_dump_data
from kvpress.presses.dma_score_press import DMAScorePress
from kvpress.presses.indexer_score_press import IndexerScorePress
from kvpress.presses.indexer_score_query_press import QueryIndexerScorePress
from trainer_utils import compute_press_loss, compute_indexer_warmup_loss, compute_query_indexer_warmup_loss, compute_query_indexer_sparse_loss


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
    parser.add_argument("--press_method", type=str, default="dma_score", choices=["dma_score", "indexer_score", "query_indexer_score"])
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

    warnings.filterwarnings("ignore", message=".*Creating a tensor from a list of numpy.ndarrays.*")
    # train_dataset, eval_dataset = load_longbench_bundle(tokenizer, args)
    # train_dataset, eval_dataset = load_c4(tokenizer, args)
    # train_dataset, eval_dataset = load_math(tokenizer, args)
    train_dataset, eval_dataset = load_dump_data(tokenizer, args)


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
        press = DMAScorePress(n_sink=args.n_sink)
    elif args.press_method == "indexer_score":
        press = IndexerScorePress(n_sink=args.n_sink)
    elif args.press_method == "query_indexer_score":
        press = QueryIndexerScorePress(n_sink=args.n_sink)
    press.post_init_from_model(model)

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

    model, optimizer, train_loader, eval_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, lr_scheduler
    )

    global_step = 0
    running_loss = 0.0
    running_grad_norm = 0.0
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            with accelerator.accumulate(model):
                if args.press_method == "dma_score":
                    outputs = model(**batch, output_attentions=True, use_cache=True, return_dict=True)
                    loss = compute_press_loss(outputs.past_key_values, outputs.attentions, batch["attention_mask"], batch["labels"], attn_modules, press.scorer_attr, args.n_sink, args.aggregate_mode)
                elif args.press_method == "indexer_score":
                    outputs = model(**batch, output_attentions=True, output_hidden_states=True, use_cache=False, return_dict=True)
                    loss = compute_indexer_warmup_loss(outputs.attentions, outputs.hidden_states, batch["attention_mask"], attn_modules, press, args.n_sink)
                elif args.press_method == "query_indexer_score":
                    outputs = model(**batch, output_attentions=True, output_hidden_states=True, use_cache=False, return_dict=True)
                    loss = compute_query_indexer_warmup_loss(outputs.attentions, outputs.hidden_states, batch["attention_mask"], attn_modules, press, args.n_sink)
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
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(args.output_dir, safe_serialization=False)
        tokenizer.model_max_length = max_token_length
        tokenizer.save_pretrained(args.output_dir)

        # if args.eval_wiki_ppl:
        #     accelerator.print("\n" + "="*80)
        #     accelerator.print("Starting WikiText PPL Evaluation")
        #     accelerator.print("="*80)
            
        #     from eval_utils import test_wikitext2_decode_ppl
            
        #     device = str(accelerator.device) if hasattr(accelerator, 'device') else "cuda:0"
        #     use_flash_attention = args.attn_implementation == "eager"
            
        #     results = test_wikitext2_decode_ppl(
        #         model_path=args.output_dir,
        #         compression_ratios=args.wiki_compression_ratios,
        #         press_type=args.press_method,
        #         context_length=args.wiki_context_length,
        #         decode_length=args.wiki_decode_length,
        #         num_samples=args.wiki_num_samples,
        #         device=device,
        #         use_flash_attention=use_flash_attention
        #     )
            
        #     accelerator.print("\nWikiText PPL Evaluation Completed!")


if __name__ == "__main__":
    main()