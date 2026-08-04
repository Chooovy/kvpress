from __future__ import annotations

import argparse
import math
import os
import importlib
import json

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
from kvpress.presses.memory_scorer_press import MemoryScorerPress
from trainer_utils import compute_press_loss, compute_indexer_warmup_loss, build_dense_warmup_targets_kl
from transformers.models.llama.modeling_llama import eager_attention_forward
from transformers.models.llama.modeling_llama import repeat_kv


def _load_indexer_state_dict(ckpt_path: str, scorer_attr: str = "indexer") -> dict[str, torch.Tensor]:
    """
    Load ONLY indexer weights from a HF checkpoint directory or a .bin/.pt file.
    This is meant for "init from ckpt" where the checkpoint may not contain memory modules.
    Supports:
      - pytorch_model.bin
      - pytorch_model.bin.index.json + shards
    """
    key_pat = f".{scorer_attr}."
    if os.path.isdir(ckpt_path):
        index_json = os.path.join(ckpt_path, "pytorch_model.bin.index.json")
        bin_path = os.path.join(ckpt_path, "pytorch_model.bin")
        st: dict[str, torch.Tensor] = {}

        if os.path.exists(index_json):
            with open(index_json, "r") as f:
                meta = json.load(f)
            weight_map = meta.get("weight_map", {})
            shard_files = sorted({sf for k, sf in weight_map.items() if key_pat in k})
            for sf in shard_files:
                shard_path = os.path.join(ckpt_path, sf)
                if not os.path.exists(shard_path):
                    continue
                sd = torch.load(shard_path, map_location="cpu")
                for k, v in sd.items():
                    if key_pat in k:
                        st[k] = v
            return st

        if os.path.exists(bin_path):
            sd = torch.load(bin_path, map_location="cpu")
            return {k: v for k, v in sd.items() if key_pat in k}

        raise FileNotFoundError(f"No pytorch_model.bin or pytorch_model.bin.index.json found under: {ckpt_path}")

    # single file
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        raise ValueError(f"Unsupported checkpoint format at: {ckpt_path}")
    return {k: v for k, v in sd.items() if key_pat in k}


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
        elif agg_mode in ("max", "amax"):
            if not output_q.any():
                do_fused_kl = False
            else:
                p_key_chunks = []
                s_key_chunks = []
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
                # Avoid NaNs from 0 * (-inf) when some keys are causally masked for a given query row.
                b_plogq = b_plogq + torch.where(p > 0, p * log_q, torch.zeros_like(p)).sum(dim=-1)

                del p_mean, p, log_q
            del s_chunk, valid_k
        elif do_fused_kl and (current_aggregate_mode or "mean").lower() in ("max", "amax"):
            s_k_start, s_k_end, s_chunk = next(student_iter)
            if s_k_start != k_start or s_k_end != k_end:
                raise RuntimeError(f"student chunk range mismatch: got {(s_k_start, s_k_end)} vs teacher {(k_start, k_end)}")

            valid_k = instr_k[:, k_start:k_end]  # (bsz, t_chunk)
            if valid_k.any():
                # teacher importance: max_q p_attn(q,k) over output queries
                p_mean = probs.mean(dim=1).float().detach()  # (bsz, q_len, t_chunk)
                p_mean = torch.where(valid_k.unsqueeze(1), p_mean, torch.zeros_like(p_mean))
                p_mean = torch.where(output_q.unsqueeze(-1), p_mean, torch.zeros_like(p_mean))
                p_key = p_mean.max(dim=1).values  # (bsz, t_chunk)
                p_key_chunks.append(p_key)

                # student logits: max_q logits(q,k) over output queries
                s_f = s_chunk.float().masked_fill(~valid_k.unsqueeze(1), float("-inf"))
                s_f = s_f.masked_fill(~output_q.unsqueeze(-1), float("-inf"))
                s_key = s_f.max(dim=1).values  # (bsz, t_chunk)
                s_key_chunks.append(s_key)
                del p_mean, p_key, s_f, s_key
            del s_chunk, valid_k

        del k_slice, v_slice, logits, shifted, exp_chunk, probs

    attn_output = attn_output.transpose(1, 2).contiguous()

    if do_fused_kl:
        # ===== Stage2: Memory residual loss (train kvpress_memory) =====
        if kvpress_collect_memory_losses and (current_memory_press is not None):
            mem_mod = getattr(module, "kvpress_memory", None)
            if mem_mod is not None:
                # Score keys using QueryIndexer (kv-head space).
                # key/value are (bsz, n_kv_heads, k_len, head_dim) before repeat_kv.
                try:
                    scores_kv = current_memory_press.base_press.score(  # type: ignore[union-attr]
                        module, hidden_states, key, value, attentions=None, kwargs={}
                    )
                except Exception:
                    scores_kv = None

                if scores_kv is not None and scores_kv.numel() > 0:
                    n_kv_heads = key.size(1)
                    # Decide kept vs evicted by top-k over key positions (non-differentiable selection).
                    n_kept = max(1, int(key.size(2) * (1.0 - float(current_memory_compression_ratio))))
                    kept_idx = scores_kv.topk(n_kept, dim=-1).indices  # (bsz, n_kv_heads, n_kept)
                    keep_mask_kv = torch.zeros(
                        (bsz, n_kv_heads, key.size(2)), device=key.device, dtype=torch.bool
                    )
                    keep_mask_kv.scatter_(2, kept_idx, True)
                    keep_mask_heads = keep_mask_kv.repeat_interleave(module.num_key_value_groups, dim=1)  # (bsz, n_heads, k_len)

                    # Compute o_keep: strict softmax attention output restricted to kept keys.
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
                        logits = torch.matmul(query, k_slice.transpose(2, 3)) * scaling
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
                        o_keep = query.new_zeros((bsz, n_heads, q_len, head_dim))
                        for k_start in range(0, k_len, chunk_size):
                            k_end = min(k_start + chunk_size, k_len)
                            valid_k = keep_mask_heads[:, :, k_start:k_end]
                            if not valid_k.any():
                                continue
                            k_slice = key_states[:, :, k_start:k_end, :]
                            v_slice = value_states[:, :, k_start:k_end, :]
                            logits = torch.matmul(query, k_slice.transpose(2, 3)) * scaling
                            if attention_mask is not None:
                                logits = logits + attention_mask[:, :, :, k_start:k_end]
                            logits_f = logits.float().masked_fill(~valid_k.unsqueeze(2), float("-inf"))
                            shifted = logits_f - keep_max.unsqueeze(-1)
                            shifted = shifted.masked_fill(torch.isnan(shifted), float("-inf"))
                            exp_chunk = torch.exp(shifted)
                            probs = (exp_chunk / keep_denom).to(query.dtype)
                            probs = nn.functional.dropout(probs, p=dropout, training=module.training)
                            o_keep = o_keep + torch.matmul(probs, v_slice)
                            del k_slice, v_slice, logits, logits_f, shifted, exp_chunk, probs
                        o_keep = o_keep.transpose(1, 2).contiguous()  # (bsz, q_len, n_heads, head_dim)

                        # Residual target: Δ = o_full - o_keep (detach target to keep graph smaller; base model is frozen anyway).
                        delta = (attn_output - o_keep).detach()

                        # Build memory state from evicted KV (single-shot write for this forward).
                        evict_mask_kv = ~keep_mask_kv  # (bsz, n_kv, k_len)
                        phi_k = mem_mod.phi(key)  # (bsz, n_kv, k_len, d_phi)
                        phi_ev = phi_k * evict_mask_kv.unsqueeze(-1)
                        v_ev = value * evict_mask_kv.unsqueeze(-1)
                        eta = mem_mod.eta()
                        A = torch.einsum("bhkd,bhke->bhde", phi_ev, v_ev) * eta  # (bsz, n_kv, d_phi, head_dim)
                        # Use squared features for a positive, stable denominator.
                        b_mem = (phi_ev ** 2).sum(dim=2) * eta  # (bsz, n_kv, d_phi)

                        # Readout: m(q) for each attention head via kv-groups
                        n_groups = module.num_key_value_groups
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

                        # Mask to output tokens only (same mask as fused KL).
                        mask_q = output_q  # (bsz, q_len) bool
                        mse = (pred.float() - delta.float()) ** 2
                        mse = mse * mask_q.unsqueeze(-1).unsqueeze(-1).float()
                        denom_mse = (mask_q.sum().float().clamp(min=1.0) * float(n_heads * head_dim))
                        mem_loss = mse.sum() / denom_mse
                        per_layer_memory_losses.append(mem_loss)

                        del o_keep, delta, phi_k, phi_ev, v_ev, A, b_mem, phi_q, m, pred, mse, denom_mse, mem_loss
                    del keep_mask_kv, keep_mask_heads, kept_idx

        mode = (current_aggregate_mode or "mean").lower()
        if mode in ("mean", "default"):
            m_clamped = m_mass.clamp(min=1e-8)
            kl_row = (a_plogp - b_plogq) / m_clamped - torch.log(m_clamped)  # (bsz, q_len)

            query_valid = output_q & (m_mass > 0)
            instr_cnt = instr_k.sum(dim=1).float()
            denom_loss = (query_valid.sum(dim=1).float() * instr_cnt).sum().clamp(min=1.0)
            loss = kl_row.masked_fill(~query_valid, 0.0).sum() / denom_loss
            per_layer_losses.append(loss)

            del m_mass, a_plogp, b_plogq, m_clamped, kl_row, query_valid, instr_cnt, denom_loss, loss
            del log_denom_s
        elif mode in ("max", "amax"):
            if len(p_key_chunks) > 0 and len(s_key_chunks) > 0:
                p_key_all = torch.cat(p_key_chunks, dim=-1)  # (bsz, k_len)
                s_key_all = torch.cat(s_key_chunks, dim=-1)  # (bsz, k_len)
                p_sum = p_key_all.sum(dim=-1, keepdim=True)
                ok = p_sum.squeeze(-1) > 0
                if ok.any():
                    p = p_key_all / p_sum.clamp(min=1e-8)
                    log_p = torch.log(p.clamp(min=1e-8))
                    log_q = s_key_all - torch.logsumexp(s_key_all, dim=-1, keepdim=True)
                    kl = torch.where(p > 0, p * (log_p - log_q), torch.zeros_like(p)).sum(dim=-1)  # (bsz,)
                    instr_cnt = instr_k.sum(dim=1).float().clamp(min=1.0)
                    loss = (kl / instr_cnt).masked_fill(~ok, 0.0).sum() / ok.float().sum().clamp(min=1.0)
                    per_layer_losses.append(loss)
                    del p, log_p, log_q, kl, instr_cnt, loss
                del p_key_all, s_key_all, p_sum, ok
            del p_key_chunks, s_key_chunks
        else:
            raise ValueError(f"Unknown aggregate_mode: {current_aggregate_mode}")

        del hidden_states, labels, attn_mask_2d, extra_kwargs
        del output_q, instr_k, kwargs_s, student_iter

    del global_max, sum_exp, denom, key_states, value_states
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
per_layer_memory_losses = []
current_attention_mask = None
current_labels = None
current_press = None
current_n_sink = None
current_aggregate_mode = "mean"
kvpress_collect_losses = False
kvpress_collect_memory_losses = False
global_chunk_size = 128  # 默认值，会在 main 中被 args.chunk_size 覆盖

# Memory training globals (configured in main)
current_memory_press = None
current_memory_compression_ratio = 0.0

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
    parser.add_argument(
        "--init_indexer_from",
        type=str,
        default=None,
        help="Initialize QueryIndexer weights from a checkpoint dir/file (no optimizer resume). Only loads *.indexer.* keys.",
    )
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

    # Two-stage training + memory module training (only used when press_method=query_indexer_score)
    parser.add_argument("--stage1_steps", type=int, default=-1, help="Stage1 steps (train QueryIndexer only). -1 => half total steps.")
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


    base_press = QueryIndexerScorePress(n_sink=args.n_sink, chunk_size=args.chunk_size, query_reduce="auto", last_n_query=128)
    d_phi = None if int(args.memory_d_phi) <= 0 else int(args.memory_d_phi)
    press = MemoryScorerPress(base_press=base_press, compression_ratio=float(args.memory_compression_ratio), d_phi=d_phi, use_denominator=True)
    press.post_init_from_model(model)

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



    # Optionally initialize indexer weights from a checkpoint that may NOT include memory modules.
    if args.init_indexer_from:
        scorer_attr = getattr(base_press, "scorer_attr", "indexer")
        idx_sd = _load_indexer_state_dict(args.init_indexer_from, scorer_attr=scorer_attr)
        if len(idx_sd) == 0:
            accelerator.print(f"[WARN] No indexer keys found in init checkpoint: {args.init_indexer_from}")
        else:
            incompatible = model.load_state_dict(idx_sd, strict=False)
            key_pat = f".{scorer_attr}."
            missing_indexer = [k for k in incompatible.missing_keys if key_pat in k]
            unexpected_indexer = [k for k in incompatible.unexpected_keys if key_pat in k]
            missing_non_indexer = [k for k in incompatible.missing_keys if key_pat not in k]
            unexpected_non_indexer = [k for k in incompatible.unexpected_keys if key_pat not in k]
            accelerator.print(
                f"Initialized indexer from {args.init_indexer_from}: "
                f"loaded={len(idx_sd)} missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)} "
                f"(indexer-only missing={len(missing_indexer)} unexpected={len(unexpected_indexer)}; "
                f"non-indexer missing={len(missing_non_indexer)} unexpected={len(unexpected_non_indexer)})"
            )
            if len(missing_indexer) > 0:
                head_n = 50
                accelerator.print(f"[init_indexer_from] First {min(head_n, len(missing_indexer))} missing indexer keys:")
                for k in missing_indexer[:head_n]:
                    accelerator.print(f"  - {k}")

    # Configure memory globals for stage2 loss (only meaningful for MemoryScorerPress).
    global current_memory_press, current_memory_compression_ratio
    if isinstance(press, MemoryScorerPress):
        current_memory_press = press
        current_memory_compression_ratio = float(args.memory_compression_ratio)
    
    # 设置全局 chunk_size 供 eager_attention_forward_mean 使用
    global global_chunk_size
    global_chunk_size = args.chunk_size

    # Stage-wise trainable params:
    # - stage1: train QueryIndexer only
    # - stage2: train QueryIndexer + KVPressMemoryLayer (kvpress_memory)
    if isinstance(press, MemoryScorerPress):
        indexer_params = []
        memory_params = []
        for name, param in model.named_parameters():
            if "indexer" in name:
                indexer_params.append(param)
            elif "kvpress_memory" in name:
                memory_params.append(param)
            else:
                param.requires_grad = False

        def _set_stage(stage: int):
            for p in indexer_params:
                p.requires_grad = True
            for p in memory_params:
                p.requires_grad = (stage >= 2)

        _set_stage(stage=1)
        optimizer = torch.optim.AdamW(
            [
                {"params": indexer_params, "lr": args.learning_rate, "weight_decay": args.weight_decay},
                {"params": memory_params, "lr": args.learning_rate, "weight_decay": args.weight_decay},
            ]
        )
    else:
        for name, param in model.named_parameters():
            if press.scorer_attr not in name:
                param.requires_grad = False
        scorer_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(scorer_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.num_train_epochs
    stage1_steps = int(args.stage1_steps)
    if stage1_steps < 0:
        stage1_steps = max(1, total_steps // 2)
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
    current_stage = 1
    optimizer_step = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            with accelerator.accumulate(model):
                # Switch to stage2 when reaching stage1_steps (only for MemoryScorerPress).
                # IMPORTANT: stage1_steps is measured in *optimizer steps* (not micro-batches).
                if isinstance(press, MemoryScorerPress) and current_stage == 1 and optimizer_step >= stage1_steps:
                    current_stage = 2
                    _set_stage(stage=2)
                    accelerator.print(f"==> Switched to stage2 at optimizer_step={optimizer_step}: training indexer + kvpress_memory")
                if args.press_method == "dma_score":
                    # 不需要语言模型的 CE loss，去掉 labels 可显著降低峰值显存（长序列下尤其明显）
                    forward_batch = {k: v for k, v in batch.items() if k != "labels"}
                    outputs = model(**forward_batch, output_attentions=True, use_cache=True, return_dict=True)
                    loss = compute_press_loss(outputs.past_key_values, outputs.attentions, batch["attention_mask"], batch["labels"], attn_modules, press.scorer_attr, args.n_sink, args.aggregate_mode)
                    del outputs
                elif args.press_method in ["indexer_score", "query_indexer_score"]:
                    per_layer_losses.clear()
                    per_layer_memory_losses.clear()
                    global current_attention_mask
                    current_attention_mask = batch["attention_mask"]
                    global current_labels
                    current_labels = batch.get("labels", None)
                    global current_press, current_n_sink, kvpress_collect_losses, kvpress_collect_memory_losses
                    # Fused KL uses the base press with indexer_logits_chunks.
                    current_press = press.base_press if isinstance(press, MemoryScorerPress) else press
                    current_n_sink = args.n_sink
                    global current_aggregate_mode
                    current_aggregate_mode = args.aggregate_mode
                    kvpress_collect_losses = True
                    kvpress_collect_memory_losses = isinstance(press, MemoryScorerPress) and (current_stage >= 2)
                    # 同上：去掉 labels，避免 forward 内部额外计算/缓存 cross entropy
                    forward_batch = {k: v for k, v in batch.items() if k != "labels"}
                    outputs = model(**forward_batch, output_attentions=False, output_hidden_states=False, use_cache=False, return_dict=True)
                    kvpress_collect_losses = False
                    kvpress_collect_memory_losses = False
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
                    if isinstance(press, MemoryScorerPress) and (current_stage >= 2) and len(per_layer_memory_losses) > 0:
                        loss = loss + torch.stack(per_layer_memory_losses).mean() * float(args.memory_loss_weight)
                    per_layer_memory_losses.clear()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    # grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=float('inf'))
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
            if accelerator.sync_gradients:
                optimizer_step += 1

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