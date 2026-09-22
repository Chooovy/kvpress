# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def distribution_kl(student_hidden, teacher_hidden, lm_head, *, direction="reverse", chunk_size=2048):
    student = student_hidden.reshape(-1, student_hidden.shape[-1])
    teacher = teacher_hidden.detach().reshape_as(student)

    def chunk_loss(student_chunk, teacher_chunk):
        student_logprobs = F.log_softmax(lm_head(student_chunk).float(), dim=-1)
        with torch.no_grad():
            teacher_logprobs = F.log_softmax(lm_head(teacher_chunk).float(), dim=-1)
        if direction == "reverse":
            return (student_logprobs.exp() * (student_logprobs - teacher_logprobs)).sum(-1).sum()
        return (teacher_logprobs.exp() * (teacher_logprobs - student_logprobs)).sum(-1).sum()

    total = student.new_zeros((), dtype=torch.float32)
    for start in range(0, student.shape[0], chunk_size):
        total = total + checkpoint(
            chunk_loss,
            student[start : start + chunk_size],
            teacher[start : start + chunk_size],
            use_reentrant=False,
        )
    return total / student.shape[0]


def token_cross_entropy(hidden, labels, lm_head, *, chunk_size=1024):
    states = hidden[:, :-1].reshape(-1, hidden.shape[-1])
    targets = labels[:, 1:].reshape(-1)

    def chunk_loss(state_chunk, target_chunk):
        return F.cross_entropy(lm_head(state_chunk).float(), target_chunk, reduction="none")

    return torch.cat(
        [
            checkpoint(
                chunk_loss, states[start : start + chunk_size], targets[start : start + chunk_size], use_reentrant=False
            )
            for start in range(0, states.shape[0], chunk_size)
        ]
    )


def reverse_kl_loss(student_hidden, teacher_hidden, lm_head, *, chunk_size=2048):
    return distribution_kl(student_hidden, teacher_hidden, lm_head, direction="reverse", chunk_size=chunk_size)
