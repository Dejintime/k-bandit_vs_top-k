"""Paper Eq. (5): corrected top-k reverse KL without subset renormalization."""

import torch


def topk_reverse_kl(student_logits, teacher_ids, teacher_logprobs, mask=None):
    """Return the sequence-mean corrected top-k reverse-KL loss.

    Shapes are ``[batch, time, vocab]`` for ``student_logits`` and
    ``[batch, time, k]`` for the teacher tensors. Both policies are normalized
    over the full vocabulary. The top-k slice must never be re-softmaxed.
    """
    if student_logits.ndim != 3 or teacher_ids.ndim != 3:
        raise ValueError("Expected [batch,time,vocab] and [batch,time,k] tensors")
    if teacher_ids.shape != teacher_logprobs.shape:
        raise ValueError("Teacher token IDs and log-probabilities must align")
    if student_logits.shape[:2] != teacher_ids.shape[:2]:
        raise ValueError("Student and teacher batch/time dimensions must align")
    if teacher_ids.dtype != torch.long or teacher_ids.shape[-1] < 1:
        raise ValueError("Teacher IDs must be non-empty int64 top-k indices")
    if mask is None:
        mask = torch.ones(student_logits.shape[:2], dtype=torch.bool,
                          device=student_logits.device)
    if mask.dtype != torch.bool or mask.shape != student_logits.shape[:2]:
        raise ValueError("mask must be boolean [batch,time]")

    losses = []
    for batch_index in range(student_logits.shape[0]):
        valid = mask[batch_index]
        if not valid.any():
            raise ValueError("Every sequence needs at least one response token")
        ids = teacher_ids[batch_index, valid]
        teacher_lp = teacher_logprobs[batch_index, valid].detach().float()
        if (ids < 0).any() or (ids >= student_logits.shape[-1]).any():
            raise ValueError("Teacher token ID is outside the student vocabulary")
        ordered = ids.sort(dim=-1).values
        if ordered.shape[-1] > 1 and (ordered[:, 1:] == ordered[:, :-1]).any():
            raise ValueError("Duplicate token IDs in a teacher top-k row")
        if (not torch.isfinite(teacher_lp).all() or (teacher_lp > 1e-5).any()
                or (teacher_lp.exp().sum(-1) > 1.0001).any()):
            raise ValueError("Teacher values must be raw full-vocabulary log-probabilities")
        student_lp = student_logits[batch_index, valid].float().log_softmax(-1)
        student_lp = student_lp.gather(-1, ids)
        p, q = student_lp.exp(), teacher_lp.exp()
        # p log(p/q) - p + q is the truncation correction from paper Eq. (5).
        losses.append((p * (student_lp - teacher_lp) - p + q).sum(-1).mean())
    return torch.stack(losses).mean()


def topk_reverse_kl_from_logprobs(student_logprobs, teacher_ids,
                                  teacher_logprobs):
    """Memory-friendly Eq. (5) for an already-normalized student block."""
    if student_logprobs.ndim != 2 or teacher_ids.ndim != 2:
        raise ValueError("Expected [time,vocab] and [time,k] tensors")
    if teacher_ids.shape != teacher_logprobs.shape:
        raise ValueError("Teacher tensors must align")
    if student_logprobs.shape[0] != teacher_ids.shape[0]:
        raise ValueError("Student and teacher time dimensions must align")
    teacher_ids = teacher_ids.to(student_logprobs.device)
    teacher_lp = teacher_logprobs.to(student_logprobs.device).detach().float()
    student_lp = student_logprobs.gather(-1, teacher_ids)
    p, q = student_lp.exp(), teacher_lp.exp()
    return (p * (student_lp - teacher_lp) - p + q).sum()


def adaptive_topk_reverse_kl_from_logprobs(student_logprobs, teacher_ids,
                                           teacher_logprobs, token_k):
    """Eq. (5) with a possibly different nested teacher support per token."""
    if token_k.ndim != 1 or token_k.shape[0] != student_logprobs.shape[0]:
        raise ValueError("token_k must be a one-dimensional value per time step")
    if (token_k < 1).any() or (token_k > teacher_ids.shape[1]).any():
        raise ValueError("token_k is outside the available teacher support")
    teacher_ids = teacher_ids.to(student_logprobs.device)
    teacher_logprobs = teacher_logprobs.to(student_logprobs.device)
    total = student_logprobs.new_zeros((), dtype=torch.float32)
    for k in token_k.unique(sorted=True).tolist():
        selected = token_k == k
        total = total + topk_reverse_kl_from_logprobs(
            student_logprobs[selected], teacher_ids[selected, :k],
            teacher_logprobs[selected, :k],
        )
    return total
