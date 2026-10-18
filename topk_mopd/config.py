"""Configuration and read-only preflight validation."""

import json
from pathlib import Path

import torch


REQUIRED_DOMAINS = {"if", "math", "code"}


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    required = {
        "student", "teachers", "train_data", "output", "steps", "batch_size",
        "top_k", "learning_rate", "max_prompt_tokens", "max_new_tokens",
        "student_device", "teacher_device", "domain_weights", "seed",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")
    if set(config["teachers"]) != REQUIRED_DOMAINS:
        raise ValueError("teachers must contain exactly: if, math, code")
    if set(config["domain_weights"]) != REQUIRED_DOMAINS:
        raise ValueError("domain_weights must contain exactly: if, math, code")
    if any(type(config[key]) is not int or config[key] < 1
           for key in ("steps", "batch_size", "top_k", "max_prompt_tokens", "max_new_tokens")):
        raise ValueError("steps, batch_size, top_k and token limits must be positive integers")
    if config["learning_rate"] <= 0 or config.get("grad_clip", 1.0) <= 0:
        raise ValueError("learning_rate and grad_clip must be positive")
    if any(value <= 0 for value in config["domain_weights"].values()):
        raise ValueError("All domain weights must be positive")
    student_device = torch.device(config["student_device"])
    teacher_device = torch.device(config["teacher_device"])
    if student_device.type != "cuda":
        raise ValueError("The student must run on CUDA")
    if (student_device.type == teacher_device.type == "cuda"
            and student_device == teacher_device):
        raise ValueError("Student and teacher cannot share one CUDA device")
    if config.get("temperature", 1.0) != 1.0:
        raise ValueError("The paper configuration uses temperature=1.0")
    if config.get("top_p", 1.0) != 1.0 or config.get("sampling_top_k", 0) != 0:
        raise ValueError("On-policy sampling must not use top-p/top-k truncation")
    for key in ("student", "train_data", "output"):
        value = Path(config[key])
        config[key] = str(value if value.is_absolute() else (path.parent / value).resolve())
    config["teachers"] = {
        domain: str(Path(value) if Path(value).is_absolute()
                    else (path.parent / value).resolve())
        for domain, value in config["teachers"].items()
    }
    for value in (config["student"], config["train_data"], *config["teachers"].values()):
        if not Path(value).exists():
            raise ValueError(f"Input path does not exist: {value}")
    if config.get("bandit_policy"):
        value = Path(config["bandit_policy"])
        config["bandit_policy"] = str(
            value if value.is_absolute() else (path.parent / value).resolve()
        )
        if not Path(config["bandit_policy"]).is_file():
            raise ValueError(f"Bandit-K policy does not exist: {config['bandit_policy']}")
    online = config.get("online_bandit")
    if online and config.get("bandit_policy"):
        raise ValueError("Use either online_bandit or frozen bandit_policy, not both")
    if online:
        arms = online.get("arms")
        valid_arms = (isinstance(arms, list) and arms == sorted(set(arms))
                      and all(type(arm) is int and arm > 0 for arm in arms))
        if not valid_arms:
            raise ValueError("online_bandit arms must be sorted unique positive integers")
        if max(arms) > config["top_k"]:
            raise ValueError("top_k must cover every online Bandit-K arm")
        reference_k = online.get("reference_k", config["top_k"])
        if type(reference_k) is not int or reference_k < max(arms) or reference_k > config["top_k"]:
            raise ValueError("online_bandit reference_k must cover all arms and fit top_k")
        forced = online.get("forced_exploration_steps", len(arms))
        if type(forced) is not int or forced < 0:
            raise ValueError("forced_exploration_steps must be a non-negative integer")
        for key in ("alpha", "ridge_lambda", "reward_clip", "reward_ema_beta",
                    "reward_epsilon"):
            if key in online and online[key] <= 0:
                raise ValueError(f"online_bandit {key} must be positive")
        if not 0 < online.get("reward_ema_beta", 0.9) < 1:
            raise ValueError("online_bandit reward_ema_beta must be between zero and one")
        if online.get("cost_lambda", 0.1) < 0:
            raise ValueError("online_bandit cost_lambda must be non-negative")
    schedule = config.get("domain_schedule")
    if schedule is not None:
        if not isinstance(schedule, list) or not schedule:
            raise ValueError("domain_schedule must be a non-empty list")
        if any(domain not in REQUIRED_DOMAINS for domain in schedule):
            raise ValueError("domain_schedule contains an unknown domain")
    if config.get("teacher_cpu_threads", 8) < 1:
        raise ValueError("teacher_cpu_threads must be positive")
    checkpoint_interval = config.get("checkpoint_interval")
    if (checkpoint_interval is not None
            and (type(checkpoint_interval) is not int or checkpoint_interval < 1)):
        raise ValueError("checkpoint_interval must be a positive integer")
    if ("keep_only_latest_checkpoint" in config
            and type(config["keep_only_latest_checkpoint"]) is not bool):
        raise ValueError("keep_only_latest_checkpoint must be boolean")
    if Path(config["output"]).exists():
        raise ValueError(f"Refusing to reuse output directory: {config['output']}")
    return config


def tokenizer_preflight(config):
    from transformers import AutoTokenizer

    student = AutoTokenizer.from_pretrained(config["student"], local_files_only=True)
    vocabulary = student.get_vocab()
    special = student.all_special_ids
    for domain, path in config["teachers"].items():
        teacher = AutoTokenizer.from_pretrained(path, local_files_only=True)
        if teacher.get_vocab() != vocabulary or teacher.all_special_ids != special:
            raise ValueError(f"{domain} teacher does not share the student's token IDs")
    if config["top_k"] > len(vocabulary):
        raise ValueError("top_k exceeds tokenizer vocabulary size")
    return {"vocab_size": len(vocabulary), "tokenizers_aligned": True}
