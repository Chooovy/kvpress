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
from data_load import load_math, load_longbench_bundle, load_c4, load_longalpaca, load_ruler
from kvpress.presses.indexer_score_press import IndexerScorePress
from trainer_utils import compute_press_loss, compute_indexer_warmup_loss
from kvpress.presses.gated_press import GatedPress
from trainer_utils import compute_gated_aux_losses, distill_kl_from_logits


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
    parser.add_argument("--press_method", type=str, default="indexer_score",
                    choices=["dma_score", "indexer_score", "gated"])
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
    parser.add_argument("--reg_lambda", type=float, default=1e-3)
    parser.add_argument("--reg_warmup_steps", type=int, default=0, help="Linearly warm up reg_lambda over this many (logging) steps. 0 disables.")
    parser.add_argument("--gate_sparsity_threshold", type=float, default=0.5, help="gate_sparsity = fraction of sigmoid(gate_score) < threshold (logged only).")
    parser.add_argument("--gate_type", type=str, default="elementwise", choices=["headwise", "elementwise"])
    parser.add_argument("--reg_type", type=str, default="group_lasso", choices=["l1", "group_lasso"])
    parser.add_argument("--gate_init", type=str, default="open", choices=["zeros", "open"])
    parser.add_argument("--gate_init_open_p", type=float, default=0.999)
    parser.add_argument("--gate_mode", type=str, default="dynamic", choices=["dynamic", "static"], help="Use dynamic (query-dependent) or static (per-layer per-head) gates.")
    parser.add_argument("--static_separate_kv", action="store_true", help="(static gate) learn separate gates for K and V. Default: False (share).")
    # ----- LM distillation + budget/bin regularizers (optional) -----
    parser.add_argument(
        "--gated_objective",
        type=str,
        default="mse_reg",
        choices=["mse_reg", "lm_distill"],
        help="Training objective for gated press: 'mse_reg' (SDPA-output MSE + reg) or 'lm_distill' (logits KL distillation only).",
    )
    parser.add_argument("--distill_lambda", type=float, default=1.0, help="Weight for logits distillation KL loss (only for gated_objective=lm_distill).")
    parser.add_argument("--distill_temperature", type=float, default=1.0, help="Temperature T for distillation (typically 1~2).")
    parser.add_argument("--distill_budget_lambda", type=float, default=0.0, help="(lm_distill only) Weight for budget loss on mean(gate). 0 disables.")
    parser.add_argument("--distill_budget_rho", type=float, default=None, help="(lm_distill only) Target keep ratio rho. If None, budget loss is disabled.")
    parser.add_argument("--distill_bin_lambda", type=float, default=0.0, help="(lm_distill only) Weight for binary regularizer mean(g(1-g)). 0 disables.")
    parser.add_argument("--budget_lambda", type=float, default=0.0, help="(mse_reg only) Weight for budget loss on mean(gate). 0 disables.")
    parser.add_argument("--budget_rho", type=float, default=None, help="(mse_reg only) Target keep ratio rho in budget loss. If None, budget loss is disabled.")
    parser.add_argument("--bin_lambda", type=float, default=0.0, help="(mse_reg only) Weight for binary regularizer sum g(1-g). 0 disables.")
    
    parser.add_argument("--key_channel_cr", type=float, default=0.5)
    parser.add_argument("--train_gate_mode", type=str, default="soft", choices=["soft","ste_topk"])
    parser.add_argument("--pairwise_prune", action="store_true")
    parser.add_argument("--sync_kv_prune", action="store_true")
    parser.add_argument("--ste_warmup_steps", type=int, default=0)

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

def collect_gate_params(model: nn.Module):#筛选需要更新的参数
    gate_params = []
    num_projs = 0
    for m in model.modules():
        proj = getattr(m, "_kvpress_gate_proj", None)
        if isinstance(proj, nn.Linear):
            num_projs += 1
            gate_params += list(proj.parameters())
        # static gate params live on attention modules as Parameters
        k_logits = getattr(m, "_kvpress_static_gate_logits_k", None)
        v_logits = getattr(m, "_kvpress_static_gate_logits_v", None)
        if isinstance(k_logits, nn.Parameter):
            num_projs += 1
            gate_params.append(k_logits)
        if isinstance(v_logits, nn.Parameter) and v_logits is not k_logits:
            num_projs += 1
            gate_params.append(v_logits)
    return gate_params, num_projs


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
    # train_dataset, eval_dataset = load_c4(tokenizer, args)
    train_dataset, eval_dataset = load_ruler(tokenizer, args)
    warnings.filterwarnings("ignore", message=".*Creating a tensor from a list of numpy.ndarrays.*")


    pad_to_multiple = 8 if accelerator.mixed_precision in ("fp16", "bf16") else None
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
    # gated 训练不需要 attentions（会爆显存）
    if args.press_method == "gated":
        model.config.output_attentions = False
    else:
        model.config.output_attentions = True
    model.config.use_cache = True

    if hasattr(args, 'gradient_checkpointing') and args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    press = None
    if args.press_method == "dma_score":
        pass

    elif args.press_method == "indexer_score":
        press = IndexerScorePress(n_sink=args.n_sink)
        press.post_init_from_model(model)
    if args.press_method in ("dma_score", "indexer_score"):
        for name, param in model.named_parameters():
            if press.scorer_attr not in name:
                param.requires_grad = False
    elif args.press_method == "gated":
        press = GatedPress(
            gate_mode=args.gate_mode,
            static_separate_kv=args.static_separate_kv,
            gate_type=args.gate_type,
            init=args.gate_init,
            init_open_p=args.gate_init_open_p,
            bias=True,
            reg_type=args.reg_type,
            record_mse=True,
            mse_detach_target=True,
            mse_reduction="token_mean",
            gate_sparsity_threshold=args.gate_sparsity_threshold,

            key_channel_compression_ratio=args.key_channel_cr,
            train_gate_mode=args.train_gate_mode,
            pairwise_prune=args.pairwise_prune,
            sync_kv_prune=args.sync_kv_prune,
        )

        # ====== 1) 明确拿到每层 self_attn ======
        attn_modules = [layer.self_attn for layer in get_language_model_layers(model)]

        # ====== 2) 直接“预创建”每层 gate 参数（不依赖 dummy forward 触发） ======
        # 避免了在前向传播时才动态创建层（那样会导致优化器找不到参数）。
        # 显式地在训练开始前，就给每一层挂好了 `_kvpress_gate_proj`。
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        for m in attn_modules:
            num_heads, num_kv_heads, head_dim, _ = press._infer_head_info(m)
            if args.gate_mode == "static":
                press._get_or_create_static_gates(m, num_kv_heads=num_kv_heads, head_dim=head_dim, device=device, dtype=dtype)
            else:
                # Gate is parameterized in KV head space (matches k_proj/v_proj output dims)
                out_dim = num_kv_heads if args.gate_type == "headwise" else (num_kv_heads * head_dim)
                # hidden_size 最稳：q_proj.in_features（Llama 一定有 q_proj）
                hidden_size = m.q_proj.in_features if hasattr(m, "q_proj") else m.o_proj.in_features
                press._get_or_create_gate_proj(
                    m, hidden_size=hidden_size, out_dim=out_dim,
                    device=device, dtype=dtype
                )

        # ====== 3) 手动注册 hooks（不要赌 BasePress 的自动遍历） ======
        # 注意：static gate 需要先创建参数，再注册 hooks（否则 hook 里读到的 gate_logits 会是 None）
        press._handles = []
        for m in attn_modules:
            press._handles += press.register_hooks(m)

        # ====== 4) 冻结全模型，只训练 gate_proj ======
        for p in model.parameters():
            p.requires_grad = False

        gate_params, num_projs = collect_gate_params(model)
        for p in gate_params:
            p.requires_grad = True

        if len(gate_params) == 0:
            raise RuntimeError("No gate parameters found even after manual creation.")

        accelerator.print(f"[gated] created gate_proj modules: {num_projs}, trainable params: {len(gate_params)}")
        optimizer = torch.optim.AdamW(gate_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.press_method in ("dma_score", "indexer_score"):
        scorer_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(scorer_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.num_train_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if args.press_method != "gated":
        attn_modules = [layer.self_attn for layer in get_language_model_layers(model)]

    model, optimizer, train_loader, eval_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, lr_scheduler
    )

    global_step = 0
    running_loss = 0.0
    running_grad_norm = 0.0
    running_mse = 0.0
    running_reg = 0.0
    running_gate_sp = 0.0
    running_gate_mean = 0.0
    running_gate_bin = 0.0
    running_distill = 0.0
    running_budget = 0.0
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            if args.train_gate_mode == "ste_topk":
                press.train_gate_mode = "soft" if global_step < args.ste_warmup_steps else "ste_topk"
            else:
                press.train_gate_mode = "soft"

            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            with accelerator.accumulate(model):
                if args.press_method == "dma_score":
                    pass
                elif args.press_method == "indexer_score":
                    outputs = model(**batch, output_attentions=True, output_hidden_states=True, use_cache=False, return_dict=True)
                    loss = compute_indexer_warmup_loss(outputs.attentions, outputs.hidden_states, batch["attention_mask"], attn_modules, press, args.n_sink)
                elif args.press_method == "gated":
                    step_num = global_step + 1
                    objective = getattr(args, "gated_objective", "mse_reg")

                    # If user selected lm_distill, force-disable aux losses to avoid mixing objectives.
                    if objective == "lm_distill":
                        press.record_mse = False
                        press.record_gate_stats = bool(
                            (getattr(args, "distill_budget_lambda", 0.0) and args.distill_budget_lambda > 0 and getattr(args, "distill_budget_rho", None) is not None)
                            or (getattr(args, "distill_bin_lambda", 0.0) and args.distill_bin_lambda > 0)
                        )
                    else:
                        press.record_mse = True
                        press.record_gate_stats = bool(
                            (getattr(args, "budget_lambda", 0.0) and args.budget_lambda > 0 and getattr(args, "budget_rho", None) is not None)
                            or (getattr(args, "bin_lambda", 0.0) and args.bin_lambda > 0)
                        )

                    # 1) Baseline forward (no gate): capture SDPA outputs as targets
                    press.apply_kv_gate = False
                    press.capture_mode = "baseline" if objective == "mse_reg" else "gated"
                    with torch.no_grad():
                        out_full = model(**batch, use_cache=False, return_dict=True, output_attentions=False)

                    # 2) Gated forward: apply KV gates and compute MSE vs stored baseline targets
                    press.apply_kv_gate = True
                    press.capture_mode = "gated"
                    out_gate = model(**batch, use_cache=False, return_dict=True, output_attentions=False)  # hooks write aux losses

                    # --- Objective A: LM logits distillation only ---
                    if objective == "lm_distill":
                        distill = distill_kl_from_logits(
                            out_full.logits,
                            out_gate.logits,
                            labels=batch.get("labels", None),
                            attention_mask=batch.get("attention_mask", None),
                            temperature=getattr(args, "distill_temperature", 1.0),
                        )
                        # Optional: budget/bin losses in lm_distill mode (based on gate stats)
                        _, _, gate_sp, gate_mean, gate_bin = compute_gated_aux_losses(attn_modules)
                        if gate_sp is None:
                            gate_sp = torch.zeros_like(distill)
                        if gate_mean is None:
                            gate_mean = torch.zeros_like(distill)
                        if gate_bin is None:
                            gate_bin = torch.zeros_like(distill)

                        if (getattr(args, "distill_budget_lambda", 0.0) and args.distill_budget_lambda > 0 and
                            getattr(args, "distill_budget_rho", None) is not None):
                            rho = float(args.distill_budget_rho)
                            budget = (gate_mean - rho).pow(2)
                        else:
                            budget = torch.zeros_like(distill)

                        if getattr(args, "distill_bin_lambda", 0.0) and args.distill_bin_lambda > 0:
                            bin_reg = gate_bin
                        else:
                            bin_reg = torch.zeros_like(distill)

                        loss = (
                            float(getattr(args, "distill_lambda", 1.0)) * distill
                            + float(getattr(args, "distill_budget_lambda", 0.0)) * budget
                            + float(getattr(args, "distill_bin_lambda", 0.0)) * bin_reg
                        )
                        # Dummy tensors for logging paths
                        mse = reg = torch.zeros_like(distill)

                    # --- Objective B: SDPA-output MSE + reg (optionally budget/bin) ---
                    else:
                        mse, reg, gate_sp, gate_mean, gate_bin = compute_gated_aux_losses(attn_modules)
                        if mse is None:
                            continue
                        if reg is None:
                            reg = torch.zeros_like(mse)
                        if gate_sp is None:
                            gate_sp = torch.zeros_like(mse)
                        if gate_mean is None:
                            gate_mean = torch.zeros_like(mse)
                        if gate_bin is None:
                            gate_bin = torch.zeros_like(mse)
                        distill = torch.zeros_like(mse)

                        # Reg warmup: avoid early collapse / over-regularization.
                        if getattr(args, "reg_warmup_steps", 0) and args.reg_warmup_steps > 0:
                            warm = min(1.0, float(step_num) / float(args.reg_warmup_steps))
                            reg_lambda_eff = args.reg_lambda * warm
                        else:
                            reg_lambda_eff = args.reg_lambda

                        # Optional: budget loss on mean gate (global mean across layers/heads/dims)
                        if (getattr(args, "budget_lambda", 0.0) and args.budget_lambda > 0 and
                            getattr(args, "budget_rho", None) is not None):
                            rho = float(args.budget_rho)
                            budget = (gate_mean - rho).pow(2)
                        else:
                            budget = torch.zeros_like(mse)

                        # Optional: binary regularizer g(1-g)
                        if getattr(args, "bin_lambda", 0.0) and args.bin_lambda > 0:
                            bin_reg = gate_bin
                        else:
                            bin_reg = torch.zeros_like(mse)

                        loss = mse + reg_lambda_eff * reg + float(getattr(args, "budget_lambda", 0.0)) * budget + float(getattr(args, "bin_lambda", 0.0)) * bin_reg

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    running_grad_norm += grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.detach().float()
            if args.press_method == "gated":
                running_mse += mse.detach().float()
                running_reg += reg.detach().float()
                running_gate_sp += gate_sp.detach().float()
                running_gate_mean += gate_mean.detach().float()
                running_gate_bin += gate_bin.detach().float()
                running_distill += distill.detach().float()
                running_budget += budget.detach().float()
            global_step += 1

            if global_step % args.logging_steps == 0:
                avg_loss = (running_loss / args.logging_steps).item()
                avg_grad_norm = running_grad_norm / args.logging_steps
                if args.press_method == "gated":
                    avg_mse = (running_mse / args.logging_steps).item()
                    avg_reg = (running_reg / args.logging_steps).item()
                    avg_gate_sp = (running_gate_sp / args.logging_steps).item()
                    avg_gate_mean = (running_gate_mean / args.logging_steps).item()
                    avg_gate_bin = (running_gate_bin / args.logging_steps).item()
                    avg_distill = (running_distill / args.logging_steps).item()
                    avg_budget = (running_budget / args.logging_steps).item()
                    objective = getattr(args, "gated_objective", "mse_reg")
                    # reg_lambda_eff is only defined in mse_reg mode; define a safe value for logging.
                    reg_lambda_eff_log = reg_lambda_eff if objective == "mse_reg" else 0.0
                    accelerator.print(
                        f"Epoch {epoch} Step {global_step}: "
                        f"train_loss={avg_loss:.6f} "
                        f"(objective={objective}, "
                        f"mse={avg_mse:.6f}, reg={avg_reg:.6f}, reg_lambda={reg_lambda_eff_log:.6g}, "
                        f"distill={avg_distill:.6f}, budget={avg_budget:.6f}, bin={avg_gate_bin:.6f}, "
                        f"gate_sp<{args.gate_sparsity_threshold:g}={avg_gate_sp:.4f}, gate_mean={avg_gate_mean:.4f}), "
                        f"train_grad_norm={avg_grad_norm:.6f}"
                    )
                    running_mse = 0.0
                    running_reg = 0.0
                    running_gate_sp = 0.0
                    running_gate_mean = 0.0
                    running_gate_bin = 0.0
                    running_distill = 0.0
                    running_budget = 0.0
                else:
                    accelerator.print(f"Epoch {epoch} Step {global_step}: train_loss={avg_loss:.6f}, train_grad_norm={avg_grad_norm:.6f}")
                running_loss = 0.0
                running_grad_norm = 0.0

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



if __name__ == "__main__":
    main()