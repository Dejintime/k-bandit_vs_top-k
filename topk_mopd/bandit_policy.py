"""Frozen token-level LCB controller for adaptive top-k MOPD."""

import json
from pathlib import Path

import numpy as np
import torch


EXPECTED_FEATURES = (
    "position_fraction", "student_entropy", "sampled_token_logprob",
    "teacher_top1_log_margin", "teacher_mass_8", "teacher_mass_16",
    "teacher_mass_32", "teacher_mass_64", "teacher_mass_128",
    "teacher_top128_conditional_entropy",
)


def load_bandit_policy(path):
    policy = json.loads(Path(path).read_text())
    if policy.get("type") != "token_lcb_ridge":
        raise ValueError("Unsupported Bandit-K policy type")
    if tuple(policy.get("feature_names", ())) != EXPECTED_FEATURES:
        raise ValueError("Bandit-K policy feature schema does not match trainer")
    arms = policy.get("arms")
    if not arms or arms != sorted(set(arms)) or any(type(k) is not int or k < 1 for k in arms):
        raise ValueError("Bandit-K policy arms must be sorted positive integers")
    width = 1 + len(policy["domains"]) + len(EXPECTED_FEATURES)
    if any(len(policy["parameters"].get(str(k), ())) != width for k in arms):
        raise ValueError("Bandit-K policy parameter width is invalid")
    if len(policy["continuous_mean"]) != len(EXPECTED_FEATURES):
        raise ValueError("Bandit-K normalization mean has invalid width")
    if len(policy["continuous_scale"]) != len(EXPECTED_FEATURES):
        raise ValueError("Bandit-K normalization scale has invalid width")
    return policy


@torch.no_grad()
def token_features(student_logprobs, teacher_logprobs, response_ids,
                   domain, start, total_tokens, policy):
    maximum = max(policy["arms"])
    teacher_lp = teacher_logprobs[:, :maximum].float()
    teacher_probabilities = teacher_lp.exp()
    mass = teacher_probabilities.cumsum(-1)
    reference_mass = mass[:, maximum - 1]
    conditional = teacher_probabilities / reference_mass.unsqueeze(-1).clamp_min(1e-20)
    conditional_entropy = -(conditional * conditional.clamp_min(1e-20).log()).sum(-1)
    student_probabilities = student_logprobs.detach().exp()
    entropy = -(student_probabilities * student_logprobs.detach()).sum(-1)
    response = torch.as_tensor(response_ids, dtype=torch.long,
                               device=student_logprobs.device).view(-1, 1)
    sampled = student_logprobs.detach().gather(1, response).flatten()
    continuous = torch.stack([
        torch.arange(start + 1, start + 1 + len(response_ids),
                     device=student_logprobs.device, dtype=torch.float32) / total_tokens,
        entropy,
        sampled,
        teacher_lp[:, 0].to(student_logprobs.device) - teacher_lp[:, 1].to(student_logprobs.device),
        mass[:, 7].to(student_logprobs.device),
        mass[:, 15].to(student_logprobs.device),
        mass[:, 31].to(student_logprobs.device),
        mass[:, 63].to(student_logprobs.device),
        reference_mass.to(student_logprobs.device),
        conditional_entropy.to(student_logprobs.device),
    ], dim=1).cpu().numpy().astype(np.float64)
    mean = np.asarray(policy["continuous_mean"], dtype=np.float64)
    scale = np.asarray(policy["continuous_scale"], dtype=np.float64)
    continuous = (continuous - mean) / scale
    domains = np.asarray([[float(domain == value) for value in policy["domains"]]],
                         dtype=np.float64)
    domain_matrix = np.repeat(domains, len(response_ids), axis=0)
    return np.concatenate([
        np.ones((len(response_ids), 1), dtype=np.float64),
        domain_matrix,
        continuous,
    ], axis=1)


def select_token_k(student_logprobs, teacher_logprobs, response_ids,
                   domain, start, total_tokens, policy):
    x = token_features(student_logprobs, teacher_logprobs, response_ids,
                       domain, start, total_tokens, policy)
    predictions = np.stack([
        x @ np.asarray(policy["parameters"][str(k)], dtype=np.float64)
        + float(policy["corrections"][str(k)])
        for k in policy["arms"]
    ], axis=1)
    choices = []
    for row in predictions:
        feasible = [k for k, prediction in zip(policy["arms"], row)
                    if prediction >= policy["threshold"]]
        choices.append(min(feasible) if feasible else max(policy["arms"]))
    return torch.tensor(choices, dtype=torch.long, device=student_logprobs.device)
