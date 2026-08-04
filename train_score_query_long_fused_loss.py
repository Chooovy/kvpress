from __future__ import annotations

import argparse
import json
import math
import os
import importlib
from datetime import timedelta

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from datasets import concatenate_datasets
from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    FullyShardedDataParallelPlugin,
    InitProcessGroupKwargs,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, set_seed
import warnings

from datautils import load_datasets_for_training, SimplePaddingCollator
from data_load import load_math, load_longbench_bundle, load_c4, load_longalpaca, load_wikitext
from kvpress.presses.indexer_score_press import IndexerScorePress
from kvpress.presses.indexer_score_query_press import QueryIndexerScorePress
from trainer_utils import compute_press_loss, compute_indexer_warmup_loss, build_dense_warmup_targets_kl
from transformers.models.llama.modeling_llama import eager_attention_forward
from transformers.models.llama.modeling_llama import repeat_kv


def _build_instruction_masks(attention_mask_2d: torch.Tensor, labels: torch.Tensor | None, n_sink: int, k_len: int):
    """
    Returns:
      - output_q: (bsz, q_len) bool, output token positions (labels!=-100) and not padded
      - instr_k: (bsz, k_len) bool, instruction key positions in [n_sink, instr_end) and not padded
    """
    if attention_mask_2d is None:
        # Fall back to "all valid" if no mask is provided.
        if labels is None:
            raise ValueError("Either attention_mask_2d or labels must be provided.")
        attention_mask_2d = labels.new_ones(labels.shape, dtype=torch.long)

    bsz, q_len = attention_mask_2d.shape

    if labels is None:
        # Unsupervised data (e.g., C4): treat all non-pad tokens as output region,
        # and instruction region as all keys >= n_sink.
        output_q = attention_mask_2d > 0
        instr_end = torch.full((bsz,), q_len, device=attention_mask_2d.device, dtype=torch.long)
        dev = attention_mask_2d.device
    else:
        output_q = (labels != -100) & (attention_mask_2d > 0)  # (bsz, q_len)
        out_pos = labels != -100
        has_out = out_pos.any(dim=1)
        first_out = out_pos.int().argmax(dim=1)
        instr_end = torch.where(
            has_out,
            first_out,
            torch.full((bsz,), q_len, device=labels.device, dtype=torch.long),
        )  # (bsz,)
        dev = labels.device

    k_idx = torch.arange(k_len, device=dev).view(1, -1)  # (1, k_len)
    attn_keep_k = attention_mask_2d[:, :k_len] > 0
    instr_k = (k_idx >= n_sink) & (k_idx < instr_end.view(-1, 1)) & attn_keep_k  # (bsz, k_len)

    # Fallback: if a sample is too short (e.g. seq_len <= n_sink), instr_k becomes empty and
    # fused KL can't run. In that case, fall back to "all valid keys" for that sample.
    # This keeps training alive on short examples (common in unsupervised corpora).
    empty_rows = instr_k.sum(dim=1) == 0
    if empty_rows.any():
        instr_k = torch.where(empty_rows.view(-1, 1), attn_keep_k, instr_k)
    return output_q, instr_k


def _student_log_denom_over_instr_keys(
    press,
    module,
    hidden_states: torch.Tensor,
    attention_mask_2d: torch.Tensor,
    instr_k: torch.Tensor,
    chunk_size: int,
    extra_kwargs: dict,
):
    """
    Compute per-row log_denom for student distribution restricted to instruction keys.
    Returns log_denom: (bsz, seqlen) float32, or None if there is no valid instruction key.
    """
    # Pass 2D mask so Press._prepare_mask can apply proper mask_fill_value for padding.
    kwargs = {"attention_mask": attention_mask_2d}
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    gen = getattr(press, "indexer_logits_chunks_with_ranges", None)
    if gen is None:
        def _gen():
            for i, chunk in enumerate(press.indexer_logits_chunks(module, hidden_states, kwargs, chunk_size=chunk_size)):
                k_start = i * chunk_size
                k_end = k_start + chunk.size(-1)
                yield k_start, k_end, chunk
        it = _gen()
    else:
        it = gen(module, hidden_states, kwargs, chunk_size=chunk_size)

    max_val = None
    sum_exp = None
    any_valid = False
    for k_start, k_end, chunk in it:
        valid_k = instr_k[:, k_start:k_end]  # (bsz, t_chunk)
        if not valid_k.any():
            del chunk
            continue
        any_valid = True
        chunk_f = chunk.float().masked_fill(~valid_k.unsqueeze(1), float("-inf"))
        chunk_max = chunk_f.max(dim=-1).values  # (bsz, seqlen)
        if max_val is None:
            max_val = chunk_max
            shifted = chunk_f - max_val.unsqueeze(-1)
            shifted = shifted.masked_fill(torch.isnan(shifted), float("-inf"))
            sum_exp = torch.exp(shifted).sum(dim=-1)
            del shifted
        else:
            new_max = torch.maximum(max_val, chunk_max)
            diff = max_val - new_max
            diff = diff.masked_fill(torch.isnan(diff), 0.0)
            sum_exp = sum_exp * torch.exp(diff)
            shifted = chunk_f - new_max.unsqueeze(-1)
            shifted = shifted.masked_fill(torch.isnan(shifted), float("-inf"))
            sum_exp = sum_exp + torch.exp(shifted).sum(dim=-1)
            max_val = new_max
            del new_max, diff, shifted
        del chunk, chunk_f, chunk_max

    if not any_valid or max_val is None:
        return None
    log_denom = max_val + torch.log(sum_exp.clamp(min=1e-8))
    del max_val, sum_exp
    return log_denom


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

    # Optional: fuse row-wise KL (teacher = head-mean attention probs conditioned on instruction keys)
    do_fused_kl = True
    if not kvpress_collect_losses:
        do_fused_kl = False
    if current_press is None:
        do_fused_kl = False
    if current_attention_mask is None:
        do_fused_kl = False
    if not hasattr(module, "_kvpress_hidden_states"):
        do_fused_kl = False
    if do_fused_kl:
        hidden_states = getattr(module, "_kvpress_hidden_states")
        labels = current_labels  # can be None for unsupervised datasets (e.g., C4)
        attn_mask_2d = current_attention_mask

        extra_kwargs = {}
        if kw.get("position_embeddings", None) is not None:
            extra_kwargs["position_embeddings"] = kw.get("position_embeddings")
        if kw.get("indexer_freqs_cis", None) is not None:
            extra_kwargs["indexer_freqs_cis"] = kw.get("indexer_freqs_cis")

        output_q, instr_k = _build_instruction_masks(attn_mask_2d, labels, current_n_sink, k_len)

        agg_mode = (current_aggregate_mode or "mean").lower()
        if agg_mode in ("mean", "default"):
            log_denom_s = _student_log_denom_over_instr_keys(
                current_press,
                module,
                hidden_states,
                attn_mask_2d,
                instr_k,
                chunk_size=chunk_size,
                extra_kwargs=extra_kwargs,
            )

            if log_denom_s is None:
                do_fused_kl = False
            else:
                # streaming per-row stats: m, A=sum p log p, B=sum p log q
                m_mass = hidden_states.new_zeros((bsz, q_len), dtype=torch.float32)
                a_plogp = hidden_states.new_zeros((bsz, q_len), dtype=torch.float32)
                b_plogq = hidden_states.new_zeros((bsz, q_len), dtype=torch.float32)

                # Pass 2D mask for proper padding masking inside the Press.
                kwargs_s = {"attention_mask": attn_mask_2d}
                if extra_kwargs:
                    kwargs_s.update(extra_kwargs)

                gen2 = getattr(current_press, "indexer_logits_chunks_with_ranges", None)
                if gen2 is None:
                    def _gen2():
                        for i, chunk in enumerate(current_press.indexer_logits_chunks(module, hidden_states, kwargs_s, chunk_size=chunk_size)):
                            k_start = i * chunk_size
                            k_end = k_start + chunk.size(-1)
                            yield k_start, k_end, chunk
                    student_iter = _gen2()
                else:
                    student_iter = gen2(module, hidden_states, kwargs_s, chunk_size=chunk_size)
        elif agg_mode in ("max", "amax"):
            # Aggregate over query dimension by max-pooling (key importance view):
            # teacher: p(k) ∝ max_q p_attn(q,k) over output queries
            # student: q(k) = softmax_k( max_q logits(q,k) ) over output queries
            if not output_q.any():
                do_fused_kl = False
            else:
                p_key_chunks = []
                s_key_chunks = []

                kwargs_s = {"attention_mask": attn_mask_2d}
                if extra_kwargs:
                    kwargs_s.update(extra_kwargs)

                gen2 = getattr(current_press, "indexer_logits_chunks_with_ranges", None)
                if gen2 is None:
                    def _gen2():
                        for i, chunk in enumerate(current_press.indexer_logits_chunks(module, hidden_states, kwargs_s, chunk_size=chunk_size)):
                            k_start = i * chunk_size
                            k_end = k_start + chunk.size(-1)
                            yield k_start, k_end, chunk
                    student_iter = _gen2()
                else:
                    student_iter = gen2(module, hidden_states, kwargs_s, chunk_size=chunk_size)
        else:
            raise ValueError(f"Unknown aggregate_mode: {current_aggregate_mode}")
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

        if do_fused_kl and (current_aggregate_mode or "mean").lower() in ("mean", "default"):
            s_k_start, s_k_end, s_chunk = next(student_iter)
            if s_k_start != k_start or s_k_end != k_end:
                raise RuntimeError(f"student chunk range mismatch: got {(s_k_start, s_k_end)} vs teacher {(k_start, k_end)}")

            valid_k = instr_k[:, k_start:k_end]  # (bsz, t_chunk)
            if valid_k.any():
                # teacher p: head-mean probs over ALL keys (post-dropout), restricted to instr keys; detach to stop-grad.
                p_mean = probs.mean(dim=1).float().detach()  # (bsz, q_len, t_chunk)
                p = torch.where(valid_k.unsqueeze(1), p_mean, torch.zeros_like(p_mean))

                # log q: student softmax over instruction keys
                log_q = s_chunk.float() - log_denom_s.unsqueeze(-1)  # (bsz, q_len, t_chunk)
                log_q = torch.where(valid_k.unsqueeze(1), log_q, torch.zeros_like(log_q))

                m_mass = m_mass + p.sum(dim=-1)
                a_plogp = a_plogp + (p * torch.log(p.clamp(min=1e-8))).sum(dim=-1)
                # IMPORTANT: avoid NaNs from 0 * (-inf) when some instruction keys are causally masked
                # for a given query row (common when labels is None, e.g., C4).
                b_plogq = b_plogq + torch.where(p > 0, p * log_q, torch.zeros_like(p)).sum(dim=-1)

                del p_mean, p, log_q
            del s_chunk, valid_k
        elif do_fused_kl and (current_aggregate_mode or "mean").lower() in ("max", "amax"):
            s_k_start, s_k_end, s_chunk = next(student_iter)
            if s_k_start != k_start or s_k_end != k_end:
                raise RuntimeError(f"student chunk range mismatch: got {(s_k_start, s_k_end)} vs teacher {(k_start, k_end)}")

            valid_k = instr_k[:, k_start:k_end]  # (bsz, t_chunk)
            if valid_k.any():
                p_mean = probs.mean(dim=1).float().detach()  # (bsz, q_len, t_chunk)
                p_mean = torch.where(valid_k.unsqueeze(1), p_mean, torch.zeros_like(p_mean))
                p_mean = torch.where(output_q.unsqueeze(-1), p_mean, torch.zeros_like(p_mean))
                p_key = p_mean.max(dim=1).values  # (bsz, t_chunk)
                p_key_chunks.append(p_key)

                s_f = s_chunk.float().masked_fill(~valid_k.unsqueeze(1), float("-inf"))
                s_f = s_f.masked_fill(~output_q.unsqueeze(-1), float("-inf"))
                s_key = s_f.max(dim=1).values  # (bsz, t_chunk)
                s_key_chunks.append(s_key)
                del p_mean, p_key, s_f, s_key
            del s_chunk, valid_k

        del k_slice, v_slice, logits, shifted, exp_chunk, probs

    attn_output = attn_output.transpose(1, 2).contiguous()

    if do_fused_kl and (current_aggregate_mode or "mean").lower() in ("mean", "default"):
        m_clamped = m_mass.clamp(min=1e-8)
        kl_row = (a_plogp - b_plogq) / m_clamped - torch.log(m_clamped)  # (bsz, q_len)

        query_valid = output_q & (m_mass > 0)
        instr_cnt = instr_k.sum(dim=1).float()
        denom_loss = (query_valid.sum(dim=1).float() * instr_cnt).sum().clamp(min=1.0)
        loss = kl_row.masked_fill(~query_valid, 0.0).sum() / denom_loss
        per_layer_losses.append(loss)

        del hidden_states, labels, attn_mask_2d, extra_kwargs
        del output_q, instr_k, log_denom_s, kwargs_s, student_iter
        del m_mass, a_plogp, b_plogq, m_clamped, kl_row, query_valid, instr_cnt, denom_loss, loss
    elif do_fused_kl and (current_aggregate_mode or "mean").lower() in ("max", "amax"):
        if len(p_key_chunks) > 0 and len(s_key_chunks) > 0:
            p_key_all = torch.cat(p_key_chunks, dim=-1)  # (bsz, k_len)
            s_key_all = torch.cat(s_key_chunks, dim=-1)  # (bsz, k_len)
            p_sum = p_key_all.sum(dim=-1, keepdim=True)
            ok = p_sum.squeeze(-1) > 0
            if ok.any():
                p = p_key_all / p_sum.clamp(min=1e-8)
                log_p = torch.log(p.clamp(min=1e-8))
                log_q = s_key_all - torch.logsumexp(s_key_all, dim=-1, keepdim=True)
                # Avoid NaNs from 0 * (+inf) when log_q=-inf at masked keys.
                kl = torch.where(p > 0, p * (log_p - log_q), torch.zeros_like(p)).sum(dim=-1)  # (bsz,)
                instr_cnt = instr_k.sum(dim=1).float().clamp(min=1.0)
                loss = (kl / instr_cnt).masked_fill(~ok, 0.0).sum() / ok.float().sum().clamp(min=1.0)
                per_layer_losses.append(loss)
                del p, log_p, log_q, kl, instr_cnt, loss
            del p_key_all, s_key_all, p_sum, ok

        del hidden_states, labels, attn_mask_2d, extra_kwargs
        del output_q, instr_k, kwargs_s, student_iter, p_key_chunks, s_key_chunks

    del global_max, sum_exp, denom, key_states, value_states


    return attn_output, None

def _kvpress_monkeypatch_attention_impl(model):
    global repeat_kv
    # Try common HF backends that define eager_attention_forward + repeat_kv.
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

per_layer_losses = []
current_attention_mask = None
current_labels = None
current_press = None
current_n_sink = None
current_aggregate_mode = "mean"
kvpress_collect_losses = False
global_chunk_size = 128  # 默认值，会在 main 中被 args.chunk_size 覆盖

def _make_attn_prehook(layer_idx: int):
    def prehook(module, inputs, kwargs=None):
        # HF models may call attention with keyword-only args (inputs can be empty).
        hidden_states = None
        if inputs is not None and len(inputs) > 0:
            hidden_states = inputs[0]
        elif kwargs is not None:
            # Different backends may use different kwarg names.
            hidden_states = kwargs.get("hidden_states", None)
            if hidden_states is None:
                hidden_states = kwargs.get("x", None)
        if hidden_states is not None:
            module._kvpress_hidden_states = hidden_states
        module._kvpress_layer_idx = layer_idx
        return None
    return prehook

def compute_indexer_layer_loss(attn, hidden_state, attention_mask, labels, module, press, n_sink):
    if labels is None:
        return hidden_state.new_zeros(())

    # attn: (bsz, num_heads, q_len, k_len) 或 (bsz, q_len, k_len)
    if attn.dim() == 4:
        attn = attn.mean(dim=1)
    attn_f = attn.float()

    bsz, q_len, k_len = attn_f.shape
    if attention_mask is None:
        # 没有 mask 时只能退化成全 1
        attention_mask = attn_f.new_ones((bsz, k_len), dtype=torch.long)

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

    # instruction key mask（只保留全局 [n_sink, instr_end) 这段 key）
    k_idx = torch.arange(k_len, device=labels.device).view(1, -1)  # (1, k_len)
    instr_k = (k_idx >= n_sink) & (k_idx < instr_end.view(-1, 1)) & (attention_mask > 0)  # (bsz, k_len)

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
        return hidden_state.new_zeros(())

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
    parser.add_argument("--max_learning_rate", type=float, default=None, help="Cosine schedule max LR (overrides --learning_rate if set).")
    parser.add_argument("--min_learning_rate", type=float, default=0.0, help="Cosine schedule min LR after decay.")
    # Short aliases (preferred by some scripts/people).
    parser.add_argument("--max_lr", dest="max_learning_rate", type=float, default=None, help="Alias for --max_learning_rate.")
    parser.add_argument("--min_lr", dest="min_learning_rate", type=float, default=0.0, help="Alias for --min_learning_rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=1000000)
    parser.add_argument("--save_steps", type=int, default=1000000)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=32)
    parser.add_argument("--press_method", type=str, default="dma_score", choices=["dma_score", "indexer_score", "query_indexer_score", "gt_score"])
    parser.add_argument("--aggregate_mode", type=str, default="mean")
    parser.add_argument("--n_sink", type=int, default=4)
    parser.add_argument("--pt_context_len", type=int, default=8192)
    parser.add_argument("--preprocessing_num_workers", type=int, default=32)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Enable gradient checkpointing to save memory")
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_size", type=int, default=128, help="Chunk size for processing long sequences")
    parser.add_argument(
        "--use_fsdp",
        action="store_true",
        help="Shard the model with FSDP across GPUs (for models that do not fit with DDP full replica per GPU).",
    )
    parser.add_argument(
        "--no_fsdp_cpu_ram_efficient_loading",
        action="store_true",
        help="With --use_fsdp, disable rank-0-only checkpoint load (higher CPU RAM; not recommended for 70B).",
    )
    parser.add_argument(
        "--fsdp_auto_wrap",
        type=str,
        default="decoder_layer",
        choices=["no_wrap", "decoder_layer"],
        help=(
            "FSDP auto-wrap: "
            "'decoder_layer' (default) = wrap each LlamaDecoderLayer (fits 70B init on 80GB). "
            "'no_wrap' = single outer FSDP (can OOM at FSDP(...) on 70B)."
        ),
    )
    parser.add_argument(
        "--save_full_model",
        action="store_true",
        help=(
            "With --use_fsdp, save a full HF checkpoint (CPU-offloaded FULL_STATE_DICT on rank0; needs very large CPU RAM for 70B). "
            "Default without this flag: save only trainable indexer weights (small; base LM unchanged)."
        ),
    )
    return parser.parse_args()




def get_language_model_layers(model: nn.Module, accelerator: Accelerator | None = None):
    """
    Resolve `model.model...layers` for Llama-style causal LMs.

    After `accelerate.prepare`, pass `accelerator` so FSDP/DDP wrappers are peeled via
    `accelerator.unwrap_model` before walking subtrees.
    """
    if accelerator is not None:
        model = accelerator.unwrap_model(model)
    lm = model
    if hasattr(lm, "module"):  # Deepspeed / DDP inner (if unwrap did not apply)
        lm = lm.module
    if hasattr(lm, "model"):
        lm = lm.model
    if hasattr(lm, "language_model"):
        lm = lm.language_model
    return lm.layers


def _fsdp_broadcast_indexer_tensors(press, model: nn.Module) -> None:
    """Align indexer weights/buffers on all ranks with rank 0 (needed with cpu_ram_efficient_loading)."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    if torch.distributed.get_world_size() <= 1:
        return
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    for layer in lm.layers:
        attn = layer.self_attn
        if not hasattr(attn, press.scorer_attr):
            continue
        indexer = getattr(attn, press.scorer_attr)
        for t in list(indexer.parameters()) + list(indexer.buffers()):
            if t.numel() == 0:
                continue
            torch.distributed.broadcast(t.data, src=0)


def _collect_press_scorer_modules(model: nn.Module, press) -> list[nn.Module]:
    """
    Per-layer indexer/scorer modules attached under self_attn.

    Register these as FSDP `ignored_modules` so their parameters are not flattened/sharded.
    That avoids backward errors (e.g. "data is not allocated yet") when training only the
    indexer inside cpu_ram_efficient + decoder_layer FSDP, while the 70B stack stays sharded.
    """
    lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
    out: list[nn.Module] = []
    for layer in lm.layers:
        attn = layer.self_attn
        if hasattr(attn, press.scorer_attr):
            out.append(getattr(attn, press.scorer_attr))
    return out


def _extract_indexer_state_dict(unwrapped: nn.Module, scorer_attr: str) -> dict[str, torch.Tensor]:
    """
    HF-prefixed keys for `self_attn.{scorer_attr}` only.

    Do **not** call `unwrapped.state_dict()` under FSDP: that walks the full LM and can trigger
    a massive unshard/_ALLGATHER on rank0 while other ranks wait (NCCL timeout after ~600s).
    Indexer modules are FSDP-ignored: read each small `indexer.state_dict()` locally.
    """
    lm = unwrapped.model.language_model if hasattr(unwrapped.model, "language_model") else unwrapped.model
    out: dict[str, torch.Tensor] = {}
    for li, layer in enumerate(lm.layers):
        attn = layer.self_attn
        if not hasattr(attn, scorer_attr):
            continue
        indexer = getattr(attn, scorer_attr)
        prefix = f"model.layers.{li}.self_attn.{scorer_attr}."
        for k, v in indexer.state_dict().items():
            out[prefix + k] = v.detach().cpu().clone()
    return out


def _save_model_checkpoint(
    accelerator: Accelerator,
    model: nn.Module,
    tokenizer,
    save_directory: str,
    args: argparse.Namespace,
    press,
    *,
    save_tokenizer: bool = False,
    max_token_length: int | None = None,
) -> None:
    """
    Checkpoints:
    - FSDP + default: indexer-only (`indexer_weights.pt` + `indexer_only_meta.json`). No full-model gather; tiny disk.
    - FSDP + `--save_full_model`: FULL_STATE_DICT offload_to_cpu on rank0, then HF `save_pretrained` (needs huge CPU RAM).
    - Non-FSDP: full `save_pretrained` (typical 8B DDP).
    """
    if accelerator.is_main_process:
        os.makedirs(save_directory, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.use_fsdp and not getattr(args, "save_full_model", False):
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            scorer_attr = getattr(press, "scorer_attr", "indexer")
            idx_sd = _extract_indexer_state_dict(unwrapped, scorer_attr)
            if not idx_sd:
                accelerator.print(f"Warning: no indexer tensors found for scorer_attr={scorer_attr!r}; skipping indexer save.")
            else:
                torch.save({"state_dict": idx_sd}, os.path.join(save_directory, "indexer_weights.pt"))
                meta = {
                    "format": "indexer_only_v1",
                    "base_model_name_or_path": args.model_name_or_path,
                    "scorer_attr": scorer_attr,
                }
                with open(os.path.join(save_directory, "indexer_only_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                accelerator.print(
                    f"Saved indexer-only checkpoint ({len(idx_sd)} tensors) to {save_directory} "
                    f"(use load_model_with_query_indexer_press with this dir; base={args.model_name_or_path})."
                )
        accelerator.wait_for_everyone()
    elif args.use_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = model.state_dict()
        if accelerator.is_main_process:
            accelerator.unwrap_model(model).save_pretrained(
                save_directory,
                state_dict=state_dict,
                safe_serialization=False,
            )
            del state_dict
        accelerator.wait_for_everyone()
    else:
        if accelerator.is_main_process:
            accelerator.unwrap_model(model).save_pretrained(save_directory, safe_serialization=False)
        accelerator.wait_for_everyone()

    if save_tokenizer and accelerator.is_main_process:
        if max_token_length is not None:
            tokenizer.model_max_length = max_token_length
        tokenizer.save_pretrained(save_directory)

    accelerator.wait_for_everyone()


def main():
    args = parse_args()
    global current_attention_mask
    # ---------------------------------------------------------------------------
    # FSDP rollback: If `--use_fsdp` causes failures, remove the `if args.use_fsdp`
    # branch below and restore the original single `Accelerator(...)` block (kept
    # in comments). Then drop `--use_fsdp` from shell scripts and pass
    # `accelerator=None` to `get_language_model_layers` after `prepare` (or omit
    # the second argument — default None). Prefer smaller models / DDP if FSDP
    # and custom attention hooks conflict.
    #
    # Original DDP-only `Accelerator` construction (was unconditional):
    # try:
    #     ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, static_graph=True)
    # except TypeError:
    #     ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # accelerator = Accelerator(
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
    #     kwargs_handlers=[ddp_kwargs],
    # )
    # ---------------------------------------------------------------------------
    # PyTorch ProcessGroupNCCL watchdog uses init_process_group(timeout=...), not NCCL_TIMEOUT.
    # Accelerate defaults NCCL backend to 600s; honor NCCL_TIMEOUT or DIST_INIT_TIMEOUT_SEC (seconds).
    _pg_timeout_sec = int(
        os.environ.get("DIST_INIT_TIMEOUT_SEC", os.environ.get("NCCL_TIMEOUT", "600"))
    )
    init_pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=_pg_timeout_sec))
    if args.use_fsdp:
        # Avoid stacking Accelerate autocast + FSDP MixedPrecision on weights that are
        # already loaded in bf16: that path triggered "Upcasted low precision parameters"
        # and then RuntimeError: setStorage ... storage of size 0 on backward.
        fsdp_mp_policy = None
        accel_mp = None if args.mixed_precision == "no" else args.mixed_precision
        if args.mixed_precision == "bf16":
            fsdp_mp_policy = None
            accel_mp = "no"
        elif args.mixed_precision == "fp16":
            fsdp_mp_policy = "fp16"
            accel_mp = "fp16"
        if args.fsdp_auto_wrap == "decoder_layer":
            fsdp_wrap = "transformer_based_wrap"
            fsdp_layer_names = ["LlamaDecoderLayer"]
        else:
            fsdp_wrap = "no_wrap"
            fsdp_layer_names = None
        fsdp_plugin = FullyShardedDataParallelPlugin(
            auto_wrap_policy=fsdp_wrap,
            transformer_cls_names_to_wrap=fsdp_layer_names,
            use_orig_params=True,
            mixed_precision_policy=fsdp_mp_policy,
            cpu_ram_efficient_loading=not args.no_fsdp_cpu_ram_efficient_loading,
            activation_checkpointing=args.gradient_checkpointing,
        )
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=accel_mp,
            fsdp_plugin=fsdp_plugin,
            kwargs_handlers=[init_pg_kwargs],
        )
    else:
        # DDP + gradient checkpointing can trigger "marked ready twice" with re-entrant backward.
        # `static_graph=True` is a common workaround when the forward graph is stable.
        try:
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, static_graph=True)
        except TypeError:
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
            kwargs_handlers=[init_pg_kwargs, ddp_kwargs],
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
    
    # # 加载 C4 数据集
    # train_c4, eval_c4 = load_c4(tokenizer, args)
    # datasets_train.append(train_c4)
    # datasets_eval.append(eval_c4)
    
    # # 加载 wikitext
    # train_wikitext, eval_wikitext = load_wikitext(tokenizer, args)
    # datasets_train.append(train_wikitext)
    # datasets_eval.append(eval_wikitext)

    # Math SFT (parquet) + LongAlpaca-12k（经 datautils 缓存）
    train_math, eval_math = load_math(tokenizer, args)
    datasets_train.append(train_math)
    datasets_eval.append(eval_math)
    train_longalpaca, eval_longalpaca = load_longalpaca(tokenizer, args)
    datasets_train.append(train_longalpaca)
    datasets_eval.append(eval_longalpaca)


    # 加载 LongBench bundle
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


    # Use args (not accelerator.mixed_precision): FSDP+bf16 may set Accelerator to "no".
    pad_to_multiple = 8 if args.mixed_precision in ("fp16", "bf16") else None
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
        low_cpu_mem_usage=args.use_fsdp and not args.no_fsdp_cpu_ram_efficient_loading,
    )
    # model.config.output_attentions = True
    # model.config.use_cache = True

    if hasattr(args, "gradient_checkpointing") and args.gradient_checkpointing:
        model.config.use_cache = False
        if args.use_fsdp:
            # FSDP applies activation checkpointing via plugin after wrap (saves VRAM on long ctx).
            pass
        else:
            # Prefer non-reentrant checkpointing when available to avoid DDP "marked ready twice".
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()

    _ = _kvpress_monkeypatch_attention_impl(model)


    if args.press_method == "dma_score":
        # press = DMAScorePress(n_sink=args.n_sink)
        pass
    elif args.press_method == "indexer_score":
        press = IndexerScorePress(n_sink=args.n_sink, chunk_size=args.chunk_size)
    elif args.press_method == "query_indexer_score":
        press = QueryIndexerScorePress(n_sink=args.n_sink, chunk_size=args.chunk_size)
    press.post_init_from_model(model)

    # With FSDP + cpu_ram_efficient_loading, the base LM can still be on meta / not fully
    # materialized when indexer is created; `QueryIndexer(...).to(model.device)` then leaves
    # parameters without storage → backward "data is not allocated yet". Force real CUDA
    # tensors on each rank before FSDP wraps the model.
    if args.use_fsdp and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        idx_device = torch.device("cuda", local_rank)
        if args.mixed_precision == "bf16":
            idx_dtype = torch.bfloat16
        elif args.mixed_precision == "fp16":
            idx_dtype = torch.float16
        else:
            idx_dtype = torch.float32
        lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
        for layer in lm.layers:
            attn = layer.self_attn
            if hasattr(attn, press.scorer_attr):
                getattr(attn, press.scorer_attr).to(device=idx_device, dtype=idx_dtype)

    # 设置全局 chunk_size 供 eager_attention_forward_mean 使用
    global global_chunk_size
    global_chunk_size = args.chunk_size

    for name, param in model.named_parameters():
        if press.scorer_attr not in name:
            param.requires_grad = False

    if args.use_fsdp:
        _fsdp_broadcast_indexer_tensors(press, model)

    # Do not shard indexer params inside FSDP flat buffers (fixes backward + keeps 70B sharded).
    if args.use_fsdp and getattr(accelerator.state, "fsdp_plugin", None) is not None:
        _ignored = _collect_press_scorer_modules(model, press)
        accelerator.state.fsdp_plugin.ignored_modules = _ignored if _ignored else None

    max_lr = args.max_learning_rate if args.max_learning_rate is not None else args.learning_rate
    min_lr = args.min_learning_rate

    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.num_train_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got total_steps={total_steps}")
    if max_lr <= 0:
        raise ValueError(f"max_learning_rate must be > 0, got max_lr={max_lr}")
    if min_lr < 0:
        raise ValueError(f"min_learning_rate must be >= 0, got min_lr={min_lr}")

    def _lr_lambda(step: int) -> float:
        """
        step: scheduler's internal step count (0-indexed; first scheduler.step() => step=0).
        We use (step + 1) so that the final scheduler step (step=total_steps-1) reaches min_lr.
        """
        # Warmup: linearly ramp from 0 -> max_lr.
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        # Cosine decay: max_lr -> min_lr.
        decay_range = total_steps - warmup_steps
        if decay_range <= 0:
            return min_lr / max_lr

        progress = (step + 1 - warmup_steps) / float(decay_range)  # 0.0 .. 1.0
        progress = max(0.0, min(1.0, progress))
        target_lr = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
        return target_lr / max_lr

    # (Previously: unused pre-prepare attn_modules; hooks are registered only after prepare.)
    # attn_modules = list(get_language_model_layers(model, None))
    # attn_modules = [layer.self_attn for layer in attn_modules]
    # attn_prehooks = []

    scorer_params = [p for p in model.parameters() if p.requires_grad]
    if not scorer_params:
        raise RuntimeError("No trainable parameters; check indexer init and freeze logic.")
    optimizer = torch.optim.AdamW(scorer_params, lr=max_lr, weight_decay=args.weight_decay)
    lr_scheduler = LambdaLR(optimizer, lr_lambda=_lr_lambda)
    model, optimizer, train_loader, eval_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, lr_scheduler
    )

    # FSDP: do not use next(model.parameters()); pick a real trainable tensor from the optimizer.
    zero_loss_anchor = None
    if optimizer.param_groups and optimizer.param_groups[0].get("params"):
        zero_loss_anchor = optimizer.param_groups[0]["params"][0]

    attn_modules = list(get_language_model_layers(model, accelerator))
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
                if args.press_method == "dma_score":
                    # 不需要语言模型的 CE loss，去掉 labels 可显著降低峰值显存（长序列下尤其明显）
                    forward_batch = {k: v for k, v in batch.items() if k != "labels"}
                    outputs = model(**forward_batch, output_attentions=True, use_cache=True, return_dict=True)
                    loss = compute_press_loss(outputs.past_key_values, outputs.attentions, batch["attention_mask"], batch["labels"], attn_modules, press.scorer_attr, args.n_sink, args.aggregate_mode)
                    del outputs
                elif args.press_method in ["indexer_score", "query_indexer_score"]:
                    per_layer_losses.clear()
                    global current_attention_mask
                    current_attention_mask = batch["attention_mask"]
                    global current_labels
                    current_labels = batch.get("labels", None)
                    global current_press, current_n_sink, kvpress_collect_losses
                    current_press = press
                    current_n_sink = args.n_sink
                    global current_aggregate_mode
                    current_aggregate_mode = args.aggregate_mode
                    kvpress_collect_losses = True
                    # 同上：去掉 labels，避免 forward 内部额外计算/缓存 cross entropy
                    forward_batch = {k: v for k, v in batch.items() if k != "labels"}
                    outputs = model(**forward_batch, output_attentions=False, output_hidden_states=False, use_cache=False, return_dict=True)
                    kvpress_collect_losses = False
                    current_attention_mask = None
                    current_labels = None
                    current_press = None
                    current_n_sink = None
                    loss_device = batch["input_ids"].device
                    del outputs  # 立即删除 outputs 以释放显存（避免保留巨大的 logits）
                    if len(per_layer_losses) > 0:
                        loss = torch.stack(per_layer_losses).mean()
                    else:
                        # Keep a grad-requiring zero loss to avoid DDP/Accelerate backward crash.
                        # This can happen if a batch has no valid (query,key) positions after masking.
                        if zero_loss_anchor is not None:
                            loss = zero_loss_anchor.sum() * 0.0
                        else:
                            loss = torch.tensor(0.0, device=loss_device)
                    per_layer_losses.clear()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    trainable_only = [p for p in model.parameters() if p.requires_grad]
                    grad_norm = (
                        accelerator.clip_grad_norm_(trainable_only, max_norm=1.0)
                        if trainable_only
                        else 0.0
                    )
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

            if args.save_steps and global_step > 0 and global_step % args.save_steps == 0:
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.print(f"Saving checkpoint to {ckpt_dir} ...")
                _save_model_checkpoint(accelerator, model, tokenizer, ckpt_dir, args, press)

    accelerator.wait_for_everyone()
    for h in attn_prehooks:
        h.remove()
    accelerator.print(f"Saving final model to {args.output_dir} ...")
    _save_model_checkpoint(
        accelerator,
        model,
        tokenizer,
        args.output_dir,
        args,
        press,
        save_tokenizer=True,
        max_token_length=max_token_length,
    )

if __name__ == "__main__":
    main()