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
    k_chunk = kw.get("chunk_size", global_chunk_size)
    q_chunk = kw.get("q_chunk_size", global_q_chunk_size)

    # IMPORTANT: avoid `repeat_kv` to reduce peak memory; compute by KV-head groups.
    # query: (b, n_heads, q_len, d)
    # key/value: (b, n_kv_heads, k_len, d)
    bsz, n_heads, q_len, head_dim = query.size()
    n_kv_heads = key.size(1)
    k_len = key.size(-2)
    group = module.num_key_value_groups
    assert n_heads == n_kv_heads * group, "inconsistent GQA dims"

    # Optional: fuse row-wise KL (teacher = head-mean attention probs conditioned on instruction keys)
    do_fused_kl = (
        kvpress_collect_losses
        and (current_press is not None)
        and (current_attention_mask is not None)
        and hasattr(module, "_kvpress_hidden_states")
    )
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
        log_denom_s = _student_log_denom_over_instr_keys(
            current_press,
            module,
            hidden_states,
            attn_mask_2d,
            instr_k,
            chunk_size=k_chunk,
            extra_kwargs=extra_kwargs,
        )
        do_fused_kl = log_denom_s is not None

        if do_fused_kl:
            kwargs_s = {"attention_mask": attn_mask_2d}
            if extra_kwargs:
                kwargs_s.update(extra_kwargs)
            # We'll create a q-chunked iterator inside the q-loop to keep student/teacher chunk counts aligned.
            student_kwargs_s = kwargs_s

    # Output tensor (same as original attention): (b, q, h, d)
    attn_output = query.new_zeros((bsz, q_len, n_heads, head_dim))

    # Accumulate fused-KL numerator/denom across q-chunks to avoid O(L) buffers.
    if do_fused_kl:
        loss_num_total = hidden_states.new_zeros((), dtype=torch.float32)
        denom_total = hidden_states.new_zeros((), dtype=torch.float32)
        instr_cnt = instr_k.sum(dim=1).float()  # (bsz,)

    for q_start in range(0, q_len, q_chunk):
        q_end = min(q_start + q_chunk, q_len)
        qc = q_end - q_start

        # Pass1: per-head rowmax/logsumexp over K chunks for this Q chunk
        global_max = query.new_full((bsz, n_heads, qc), float("-inf"), dtype=torch.float32)
        sum_exp = query.new_zeros((bsz, n_heads, qc), dtype=torch.float32)

        for k_start in range(0, k_len, k_chunk):
            k_end = min(k_start + k_chunk, k_len)
            # causal/pad mask slice for this tile
            mask_tile = None
            if attention_mask is not None:
                mask_tile = attention_mask[:, :, q_start:q_end, k_start:k_end]  # (b,1,qc,kc)

            for kv in range(n_kv_heads):
                h0 = kv * group
                h1 = h0 + group
                q_group = query[:, h0:h1, q_start:q_end, :]  # (b,g,qc,d)
                k_slice = key[:, kv, k_start:k_end, :]  # (b,kc,d)
                logits = torch.matmul(q_group, k_slice.transpose(1, 2)) * scaling  # (b,g,qc,kc)
                if mask_tile is not None:
                    logits = logits + mask_tile
                logits_f = logits.float()
                chunk_max = logits_f.max(dim=-1).values  # (b,g,qc)

                old_max = global_max[:, h0:h1, :]
                old_sum = sum_exp[:, h0:h1, :]
                new_max = torch.maximum(old_max, chunk_max)
                # rescale old sum
                old_sum = old_sum * torch.exp((old_max - new_max).masked_fill(torch.isnan(old_max - new_max), 0.0))
                shifted = logits_f - new_max.unsqueeze(-1)
                shifted = shifted.masked_fill(torch.isnan(shifted), float("-inf"))
                new_sum = old_sum + torch.exp(shifted).sum(dim=-1)
                global_max[:, h0:h1, :] = new_max
                sum_exp[:, h0:h1, :] = new_sum
                del q_group, k_slice, logits, logits_f, chunk_max, old_max, old_sum, new_max, shifted, new_sum

        denom = sum_exp.clamp(min=1e-8).unsqueeze(-1)  # (b,h,qc,1)

        # fused KL per q-chunk stats
        if do_fused_kl:
            m_mass = hidden_states.new_zeros((bsz, qc), dtype=torch.float32)
            a_plogp = hidden_states.new_zeros((bsz, qc), dtype=torch.float32)
            b_plogq = hidden_states.new_zeros((bsz, qc), dtype=torch.float32)
            output_q_chunk = output_q[:, q_start:q_end]
            # Create student iterator ONCE per q-chunk so it yields successive k-ranges.
            if hasattr(current_press, "indexer_logits_chunks_with_ranges_qchunk"):
                student_iter = current_press.indexer_logits_chunks_with_ranges_qchunk(
                    module, hidden_states, student_kwargs_s, chunk_size=k_chunk, q_start=q_start, q_end=q_end
                )
            else:
                # Fallback: recompute full-q student chunks and slice q-range.
                gen_full = getattr(current_press, "indexer_logits_chunks_with_ranges", None)
                if gen_full is None:
                    def _gen_full():
                        for i, chunk in enumerate(current_press.indexer_logits_chunks(module, hidden_states, student_kwargs_s, chunk_size=k_chunk)):
                            ks = i * k_chunk
                            ke = ks + chunk.size(-1)
                            yield ks, ke, chunk
                    full_iter = _gen_full()
                else:
                    full_iter = gen_full(module, hidden_states, student_kwargs_s, chunk_size=k_chunk)
                def _slice_iter():
                    for ks, ke, ch in full_iter:
                        yield ks, ke, ch[:, q_start:q_end, :]
                student_iter = _slice_iter()

        # Pass2: output + optional KL
        for k_start in range(0, k_len, k_chunk):
            k_end = min(k_start + k_chunk, k_len)
            kc = k_end - k_start
            mask_tile = None
            if attention_mask is not None:
                mask_tile = attention_mask[:, :, q_start:q_end, k_start:k_end]

            # head-mean probs for KL (accumulate across head groups)
            if do_fused_kl:
                p_sum_heads = hidden_states.new_zeros((bsz, qc, kc), dtype=torch.float32)
                s_k_start, s_k_end, s_chunk = next(student_iter)
                if s_k_start != k_start or s_k_end != k_end:
                    raise RuntimeError(f"student chunk range mismatch: got {(s_k_start, s_k_end)} vs teacher {(k_start, k_end)}")
                valid_k = instr_k[:, k_start:k_end]  # (bsz,kc)

            for kv in range(n_kv_heads):
                h0 = kv * group
                h1 = h0 + group
                q_group = query[:, h0:h1, q_start:q_end, :]  # (b,g,qc,d)
                k_slice = key[:, kv, k_start:k_end, :]  # (b,kc,d)
                v_slice = value[:, kv, k_start:k_end, :]  # (b,kc,d)
                logits = torch.matmul(q_group, k_slice.transpose(1, 2)) * scaling
                if mask_tile is not None:
                    logits = logits + mask_tile
                shifted = logits.float() - global_max[:, h0:h1, :].unsqueeze(-1)
                shifted = shifted.masked_fill(torch.isnan(shifted), float("-inf"))
                exp_chunk = torch.exp(shifted)
                probs = (exp_chunk / denom[:, h0:h1, :, :]).to(query.dtype)  # (b,g,qc,kc)
                probs = nn.functional.dropout(probs, p=dropout, training=module.training)

                # output slice (b,qc,g,d)
                out = torch.matmul(probs, v_slice)  # (b,g,qc,d)
                attn_output[:, q_start:q_end, h0:h1, :] = attn_output[:, q_start:q_end, h0:h1, :] + out.transpose(1, 2)

                if do_fused_kl:
                    p_sum_heads = p_sum_heads + probs.sum(dim=1).float().detach()  # sum over group heads

                del q_group, k_slice, v_slice, logits, shifted, exp_chunk, probs, out

            if do_fused_kl and valid_k.any():
                p_mean = p_sum_heads / float(n_heads)  # (b,qc,kc)
                p = torch.where(valid_k.unsqueeze(1), p_mean, torch.zeros_like(p_mean))
                # s_chunk is already q-chunked: (bsz, q_chunk, k_chunk)
                log_q = s_chunk.float() - log_denom_s[:, q_start:q_end].unsqueeze(-1)  # (b,qc,kc)
                log_q = torch.where(valid_k.unsqueeze(1), log_q, torch.zeros_like(log_q))
                m_mass = m_mass + p.sum(dim=-1)
                a_plogp = a_plogp + (p * torch.log(p.clamp(min=1e-8))).sum(dim=-1)
                b_plogq = b_plogq + (p * log_q).sum(dim=-1)
                del p_mean, p, log_q

            if do_fused_kl:
                del p_sum_heads, s_chunk, valid_k

        if do_fused_kl:
            m_clamped = m_mass.clamp(min=1e-8)
            kl_row = (a_plogp - b_plogq) / m_clamped - torch.log(m_clamped)  # (b,qc)
            query_valid = output_q_chunk & (m_mass > 0)
            loss_num_total = loss_num_total + kl_row.masked_fill(~query_valid, 0.0).sum()
            denom_total = denom_total + (query_valid.sum(dim=1).float() * instr_cnt).sum()
            del m_mass, a_plogp, b_plogq, m_clamped, kl_row, query_valid, output_q_chunk

        del global_max, sum_exp, denom

    if do_fused_kl:
        loss = loss_num_total / denom_total.clamp(min=1.0)
        per_layer_losses.append(loss)
        del hidden_states, labels, attn_mask_2d, extra_kwargs
        del output_q, instr_k, log_denom_s, kwargs_s, student_iter, instr_cnt
        del loss_num_total, denom_total, loss

    return attn_output, None

def _kvpress_monkeypatch_attention_impl(model):
    """
    Ensure the model actually calls our eager attention forward.
    Many checkpoints (e.g. Mistral) don't use `transformers.models.llama.modeling_llama`.
    """
    global repeat_kv
    # Try common HF backends that define eager_attention_forward + repeat_kv.
    candidates = [
        "transformers.models.llama.modeling_llama",
        "transformers.models.mistral.modeling_mistral",
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
kvpress_collect_losses = False
global_chunk_size = 128  # 默认值，会在 main 中被 args.chunk_size 覆盖
global_q_chunk_size = 1024  # 默认值，会在 main 中被 args.q_chunk_size 覆盖

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
    parser.add_argument("--chunk_size", type=int, default=128, help="Chunk size for processing long sequences")
    parser.add_argument("--q_chunk_size", type=int, default=1024, help="Query chunk size for attention to reduce peak memory")
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
        # Prefer non-reentrant checkpointing when available to avoid DDP "marked ready twice".
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        model.config.use_cache = False

    _ = _kvpress_monkeypatch_attention_impl(model)


    if args.press_method == "dma_score":
        # press = DMAScorePress(n_sink=args.n_sink)
        pass
    elif args.press_method == "indexer_score":
        press = IndexerScorePress(n_sink=args.n_sink, chunk_size=args.chunk_size)
    elif args.press_method == "query_indexer_score":
        press = QueryIndexerScorePress(n_sink=args.n_sink, chunk_size=args.chunk_size)
    press.post_init_from_model(model)
    
    # 设置全局 chunk_size 供 eager_attention_forward_mean 使用
    global global_chunk_size
    global_chunk_size = args.chunk_size
    global global_q_chunk_size
    global_q_chunk_size = args.q_chunk_size

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
    for h in attn_prehooks:
        h.remove()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(args.output_dir, safe_serialization=False)
        tokenizer.model_max_length = max_token_length
        tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()