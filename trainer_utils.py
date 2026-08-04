import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from typing import Sequence
from kvpress.utils import extract_keys_and_values, get_prerope_query_states
from dataclasses import dataclass


@dataclass
class BatchLoss:
    loss: torch.Tensor 
    tokens: torch.Tensor

def distill_kl_from_logits(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Logits distillation loss:
      L = T^2 * KL( softmax(z_teacher/T) || softmax(z_student/T) )

    Masking:
      - if labels is provided, only count tokens with labels != -100 (typical LM training)
      - else if attention_mask is provided, count attention_mask > 0
      - else count all tokens
    """
    if not isinstance(teacher_logits, torch.Tensor) or not isinstance(student_logits, torch.Tensor):
        raise TypeError("teacher_logits and student_logits must be torch.Tensor")
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(f"logits shape mismatch: teacher={teacher_logits.shape}, student={student_logits.shape}")

    T = float(temperature)
    if T <= 0:
        raise ValueError(f"temperature must be > 0, got {T}")

    # Use fp32 for numerical stability
    t = (teacher_logits.detach().float() / T)
    s = (student_logits.float() / T)
    teacher_probs = torch.softmax(t, dim=-1)
    student_log_probs = torch.log_softmax(s, dim=-1)

    # per-token KL (sum over vocab)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)  # [B, L]
    kl = kl * (T * T)

    if labels is not None:
        mask = (labels != -100).float()
    elif attention_mask is not None:
        mask = (attention_mask > 0).float()
    else:
        mask = torch.ones_like(kl)

    # Safe normalize
    denom = mask.sum().clamp(min=1.0)
    return (kl * mask).sum() / denom

def compute_gated_aux_losses(attn_modules):
    mses, regs = [], []
    gate_sps, gate_means, gate_bins = [], [], []
    for m in attn_modules:
        if not hasattr(m, "o_proj"):
            continue
        op = m.o_proj
        if hasattr(op, "_kvpress_aux_mse"):#收集 MSE Loss
            mses.append(op._kvpress_aux_mse)
            delattr(op, "_kvpress_aux_mse")
        if hasattr(op, "_kvpress_aux_reg"):#收集 Regularization Loss
            regs.append(op._kvpress_aux_reg)
            delattr(op, "_kvpress_aux_reg")
        if hasattr(op, "_kvpress_gate_sparsity"):
            gate_sps.append(op._kvpress_gate_sparsity)
            delattr(op, "_kvpress_gate_sparsity")
        if hasattr(op, "_kvpress_gate_mean"):
            gate_means.append(op._kvpress_gate_mean)
            delattr(op, "_kvpress_gate_mean")
        if hasattr(op, "_kvpress_gate_bin"):
            gate_bins.append(op._kvpress_gate_bin)
            delattr(op, "_kvpress_gate_bin")
    #对所有层取平均
    mse = torch.stack(mses).mean() if mses else None
    reg = torch.stack(regs).mean() if regs else None
    gate_sparsity = torch.stack(gate_sps).mean() if gate_sps else None
    gate_mean = torch.stack(gate_means).mean() if gate_means else None
    gate_bin = torch.stack(gate_bins).mean() if gate_bins else None
    return mse, reg, gate_sparsity, gate_mean, gate_bin



def build_targets(attn_tensor, attention_mask, labels, num_heads, num_kv_heads, n_sink, aggregate_mode="mean"):
    attn = attn_tensor
    bsz, _, seq_len, k_len = attn.shape
    assert k_len == attention_mask.size(1), "attention/k cache length mismatch"

    key_mask = attention_mask
    query_mask = (labels != -100).float()

    future_mask = torch.tril(torch.ones(seq_len, k_len, device=attn.device, dtype=attn.dtype), diagonal=-1)

    attn = attn * key_mask[:, None, None, :]
    attn = attn * future_mask[None, None, :, :]
    attn = attn * query_mask[:, None, :, None]

    future_counts = torch.matmul(query_mask, future_mask)  # (bsz, k_len)
    has_future = future_counts > 0
    future_counts = future_counts + (~has_future).float()

    if aggregate_mode == "mean":
        importance = attn.sum(dim=2) / future_counts[:, None, :]
        importance = importance * has_future[:, None, :]
    elif aggregate_mode == "max":
        importance = attn.max(dim=2)[0]  # (bsz, num_heads, k_len)
        importance = importance * has_future[:, None, :]
    elif aggregate_mode.startswith("top"):
        top_k = int(aggregate_mode[3:]) / 100.0
        k_keys = max(1, int(k_len * top_k))
        topk_attn, topk_indices = attn.topk(k_keys, dim=3)  # (bsz, num_heads, seq_len, k_keys)
        mask = torch.zeros_like(attn)
        mask.scatter_(3, topk_indices, 1.0)
        masked_attn = attn * mask
        top_k_future_counts = (masked_attn > 0).float().sum(dim=2)  # (bsz, num_heads, k_len)
        top_k_future_counts = top_k_future_counts.clamp(min=1.0)
        
        importance = masked_attn.sum(dim=2) / top_k_future_counts
        importance = importance * has_future[:, None, :]
    groups = num_heads // num_kv_heads
    importance = importance.view(bsz, num_kv_heads, groups, k_len).mean(dim=2)

    if n_sink >= k_len:
        raise ValueError(f"n_sink={n_sink} >= sequence length {k_len}")

    key_mask_slice = (key_mask[:, n_sink:] > 0).unsqueeze(1)
    future_mask_slice = has_future[:, n_sink:].unsqueeze(1)
    valid_mask = key_mask_slice & future_mask_slice

    importance = importance[:, :, n_sink:]
    importance = importance * valid_mask.float()

    denom = importance.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    target = importance / denom

    return target.detach(), valid_mask




def scorer_loss(
    scorer_logits: torch.Tensor,
    target_probs: torch.Tensor,
    mask: torch.Tensor,
) -> BatchLoss:
    log_probs = F.log_softmax(scorer_logits, dim=-1)
    loss = F.kl_div(log_probs, target_probs, reduction="none")
    loss = loss * mask
    denom = mask.sum().clamp(min=1.0)
    return BatchLoss(loss=loss.sum() / denom, tokens=denom)


def compute_press_loss(cache, attentions, attention_mask, labels, attn_modules, scorer_attr, n_sink, aggregate_mode="mean"):
    per_layer = []
    for idx, attn in enumerate(attentions):
        module = attn_modules[idx]
        keys, values = extract_keys_and_values(cache, idx)
        keys = keys[:, :, n_sink:]
        values = values[:, :, n_sink:]

        scorer: nn.Module = getattr(module, scorer_attr)
        logits = scorer(keys, values)

        target, mask = build_targets(
            attn,
            attention_mask,
            labels,
            module.config.num_attention_heads,
            module.config.num_key_value_heads,
            n_sink,
            aggregate_mode,
        )
        
        loss = scorer_loss(logits, target, mask).loss
        per_layer.append(loss)
        
        del attn, keys, values, logits, target, mask
        if idx % 4 == 0:
            torch.cuda.empty_cache()
    
    return torch.stack(per_layer).mean()




def build_dense_warmup_targets_kl(attn_tensor, attention_mask, n_sink):
    # attn_tensor: (bsz, num_heads, q_len, k_len)
    # 在 head 维度上求平均，得到 (bsz, q_len, k_len)
    if attn_tensor.dim() == 4:
        attn = attn_tensor.mean(dim=1)
    else:
        attn = attn_tensor
    # attn = attn_tensor.sum(dim=1)
    key_mask = attention_mask[:, None, :]  # (bsz, 1, k_len)
    attn = attn * key_mask
    
    causal = torch.tril(torch.ones(attn.size(-2), attn.size(-1), device=attn.device))
    attn = attn * causal
    
    target = attn[:, :, n_sink:]  # (bsz, q_len, k_len - n_sink)
    
    denom = target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    target = target / denom
    
    mask = (denom.squeeze(-1) > 0).unsqueeze(-1).expand_as(target)
    
    return target, mask


def compute_indexer_warmup_loss(attentions, hidden_states, attention_mask, attn_modules, press, n_sink):
    per_layer = []
    kwargs = {"attention_mask": attention_mask[:, None, None, :]}
    
    for idx, attn in enumerate(attentions):
        module = attn_modules[idx]
        
        targets, mask = build_dense_warmup_targets_kl(attn, attention_mask, n_sink)
        # targets: (bsz, q_len, k_len - n_sink)
        # mask: (bsz, q_len, k_len - n_sink)
        
        # indexer 输出: (bsz, q_len, k_len)
        logits = press.indexer_logits(module, hidden_states[idx + 1], kwargs)
        logits = logits[:, :, n_sink:]  # (bsz, q_len, k_len - n_sink)
        
        # 计算 KL divergence
        # log_probs: log(softmax(indexer_logits))
        # targets: softmax(attention)
        log_probs = F.log_softmax(logits, dim=-1)
        kl_loss = F.kl_div(log_probs, targets, reduction="none")
        
        kl_loss = kl_loss * mask
        loss = kl_loss.sum() / mask.sum().clamp(min=1.0)
        
        per_layer.append(loss)
    
    return torch.stack(per_layer).mean()

def compute_query_indexer_warmup_loss(attentions, hidden_states, attention_mask, attn_modules, press, n_sink):
    per_layer = []
    kwargs = {"attention_mask": attention_mask[:, None, None, :]}
    
    for idx, attn in enumerate(attentions):
        module = attn_modules[idx]
        query_states = get_prerope_query_states(module, hidden_states[idx + 1])
        kwargs["query_states"] = query_states
        
        targets, mask = build_dense_warmup_targets_kl(attn, attention_mask, n_sink)
        
        logits = press.indexer_logits(module, hidden_states[idx + 1], kwargs)
        logits = logits[:, :, n_sink:]
        log_probs = F.log_softmax(logits, dim=-1)
        kl_loss = F.kl_div(log_probs, targets, reduction="none")
        kl_loss = kl_loss * mask
        loss = kl_loss.sum() / mask.sum().clamp(min=1.0)
        per_layer.append(loss)
    return torch.stack(per_layer).mean()


def compute_indexer_sparse_loss(attentions, hidden_states, attention_mask, attn_modules, press, n_sink):
    per_layer = []
    kwargs = {"attention_mask": attention_mask[:, None, None, :]}

    # Use press.compression_ratio (same as inference) to decide how many tokens to keep
    compression_ratio = float(getattr(press, "compression_ratio", 0.0))

    for idx, attn in enumerate(attentions):
        module = attn_modules[idx]

        # Indexer logits (detach hidden states to avoid backprop into the main model)
        logits = press.indexer_logits(module, hidden_states[idx + 1].detach(), kwargs)
        logits = logits[:, :, n_sink:]  # (bsz, q_len, k_len_nosink)

        k_len = logits.size(-1)
        n_kept = max(1, int(k_len * (1 - compression_ratio)))
        n_kept = min(n_kept, k_len)

        topk_vals, topk_indices = logits.topk(n_kept, dim=-1)

        # Build dense attention targets then restrict to the selected tokens
        attn_mean = attn.mean(dim=1)  # (bsz, q_len, k_len)
        key_mask = attention_mask[:, None, :]  # (bsz, 1, k_len)
        attn_masked = attn_mean * key_mask

        causal = torch.tril(torch.ones(attn_masked.size(-2), attn_masked.size(-1), device=attn_masked.device))
        attn_masked = attn_masked * causal

        attn_masked = attn_masked[:, :, n_sink:]  # drop sinks to match logits
        selected_attn = torch.gather(attn_masked, -1, topk_indices)

        denom = selected_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        target = selected_attn / denom
        mask = (denom.squeeze(-1) > 0).unsqueeze(-1).expand_as(target)

        log_probs = F.log_softmax(topk_vals, dim=-1)
        kl_loss = F.kl_div(log_probs, target, reduction="none")
        kl_loss = kl_loss * mask
        loss = kl_loss.sum() / mask.sum().clamp(min=1.0)

        per_layer.append(loss)

    return torch.stack(per_layer).mean()


def compute_query_indexer_sparse_loss(attentions, hidden_states, attention_mask, attn_modules, press, n_sink):
    per_layer = []
    kwargs = {"attention_mask": attention_mask[:, None, None, :]}

    compression_ratio = float(getattr(press, "compression_ratio", 0.0))

    for idx, attn in enumerate(attentions):
        module = attn_modules[idx]
        query_states = get_prerope_query_states(module, hidden_states[idx + 1])
        kwargs["query_states"] = query_states
        
        logits = press.indexer_logits(module, hidden_states[idx + 1], kwargs)
        logits = logits[:, :, n_sink:]

        k_len = logits.size(-1)
        n_kept = max(1, int(k_len * (1 - compression_ratio)))
        n_kept = min(n_kept, k_len)

        topk_vals, topk_indices = logits.topk(n_kept, dim=-1)

        # Build dense attention targets then restrict to the selected tokens
        attn_mean = attn.mean(dim=1)  # (bsz, q_len, k_len)
        key_mask = attention_mask[:, None, :]  # (bsz, 1, k_len)
        attn_masked = attn_mean * key_mask

        causal = torch.tril(torch.ones(attn_masked.size(-2), attn_masked.size(-1), device=attn_masked.device))
        attn_masked = attn_masked * causal

        attn_masked = attn_masked[:, :, n_sink:]  # drop sinks to match logits
        selected_attn = torch.gather(attn_masked, -1, topk_indices)

        denom = selected_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        target = selected_attn / denom
        mask = (denom.squeeze(-1) > 0).unsqueeze(-1).expand_as(target)

        log_probs = F.log_softmax(topk_vals, dim=-1)
        kl_loss = F.kl_div(log_probs, target, reduction="none")
        kl_loss = kl_loss * mask
        loss = kl_loss.sum() / mask.sum().clamp(min=1.0)

        per_layer.append(loss)

    return torch.stack(per_layer).mean()

@torch.no_grad()
def evaluate_epoch_learned_score(
    accelerator: Accelerator,
    model: nn.Module,
    dataloader: DataLoader,
    attn_modules: Sequence[nn.Module],
    scorer_attr: str,
    n_sink: int,
    aggregate_mode: str = "mean",
) -> float:
    model.eval()
    losses = []
    for batch in dataloader:
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        outputs = model(
            **batch,
            output_attentions=True,
            use_cache=True,
            return_dict=True,
        )
        loss = compute_press_loss(
            outputs.past_key_values,
            outputs.attentions,
            batch["attention_mask"],
            batch["labels"],
            attn_modules,
            scorer_attr,
            n_sink,
            aggregate_mode,
        )
        losses.append(accelerator.gather(loss.detach().repeat(batch["input_ids"].size(0))))
    model.train()
    losses = torch.cat(losses)
    return losses.mean().item()

@torch.no_grad()
def evaluate_epoch_indexer(
    accelerator: Accelerator,
    model: nn.Module,
    dataloader: DataLoader,
    attn_modules: Sequence[nn.Module],
    press,
    n_sink: int,
) -> float:
    model.eval()
    losses = []
    for batch in dataloader:
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        outputs = model(
            **batch,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        loss = compute_indexer_warmup_loss(
            outputs.attentions,
            outputs.hidden_states,
            batch["attention_mask"],
            attn_modules,
            press,
            n_sink,
        )
        losses.append(accelerator.gather(loss.detach().repeat(batch["input_ids"].size(0))))
    model.train()
    losses = torch.cat(losses)
    return losses.mean().item()