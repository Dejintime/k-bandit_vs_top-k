"""Bounded-memory trainer for fixed, frozen, or online Bandit-K MOPD."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import time

import torch

from .bandit_policy import load_bandit_policy
from .config import load_config, tokenizer_preflight
from .data import DomainSampler, PromptDataset
from .modeling import (TeacherPool, backward_trajectory, generate_rollouts,
                       evaluate_reference_loss, render_prompt,
                       score_trajectories)
from .online_bandit import OnlineLinUCB, relative_reference_gain
from .optimizer import CPUAdamW


def emit(event, **values):
    print(json.dumps({"event": event, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                      **values}, ensure_ascii=False), flush=True)


def preflight(config):
    dataset = PromptDataset(config["train_data"], config["teachers"])
    report = {
        "student": config["student"],
        "teachers": config["teachers"],
        "samples": dict(dataset.counts()),
        "top_k": config["top_k"],
        "formula": "paper Eq. (5): p*log(p/q)-p+q",
        "routing": "one matching domain teacher per prompt",
        "teacher_device": config["teacher_device"],
        "online_reward": bool(config.get("online_bandit")),
        "packages": {},
    }
    for package in ("torch", "transformers", "safetensors"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    report.update(tokenizer_preflight(config))
    report["cuda_available"] = torch.cuda.is_available()
    report["cuda_devices"] = torch.cuda.device_count()
    try:
        fields = Path("/proc/meminfo").read_text().splitlines()
        available_kib = int(next(line for line in fields if line.startswith("MemAvailable:"))
                            .split()[1])
        report["available_ram_gib"] = round(available_kib / 2 ** 20, 2)
    except (OSError, StopIteration, ValueError):
        report["available_ram_gib"] = None
    if torch.cuda.is_available():
        report["gpu"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "free_gib": round(torch.cuda.mem_get_info(index)[0] / 2 ** 30, 2),
                "total_gib": round(torch.cuda.mem_get_info(index)[1] / 2 ** 30, 2),
            }
            for index in range(torch.cuda.device_count())
        ]
    return report, dataset


def save_student(student, tokenizer, destination, metadata):
    destination.mkdir(parents=True, exist_ok=False)
    # Moving state one tensor at a time avoids a second full GPU model copy.
    state = {name: tensor.detach().cpu() for name, tensor in student.state_dict().items()}
    student.save_pretrained(destination, state_dict=state, safe_serialization=True,
                            max_shard_size="2GB")
    tokenizer.save_pretrained(destination)
    (destination / "mopd_metadata.json").write_text(json.dumps(metadata, indent=2))
    (destination / "COMPLETE").write_text(str(metadata["completed_steps"]))


def checkpoint_metadata(config, completed):
    return {
        "completed_steps": completed,
        "student_initialization": config["student"],
        "teachers": config["teachers"],
        "top_k": config["top_k"],
        "online_bandit": config.get("online_bandit"),
        "loss": "corrected top-k reverse KL (paper Eq. 5)",
    }


def checkpoint_due(completed, config):
    interval = config.get("checkpoint_interval")
    return interval is not None and completed % interval == 0


CHECKPOINT_PATTERN = re.compile(r"checkpoint-[0-9]{6}")


def prune_old_checkpoints(output, keep):
    """Remove only completed, directly nested MOPD checkpoints except ``keep``."""
    output = Path(output).resolve()
    keep = Path(keep).resolve()
    if keep.parent != output or not (keep / "COMPLETE").is_file():
        raise ValueError("Refusing to prune before the new checkpoint is complete")
    removed = []
    for candidate in output.iterdir():
        if candidate.resolve() == keep:
            continue
        if (candidate.is_dir() and CHECKPOINT_PATTERN.fullmatch(candidate.name)
                and (candidate / "COMPLETE").is_file()):
            shutil.rmtree(candidate)
            removed.append(str(candidate))
    return removed


def train(config, dataset):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    required_free = config.get("min_free_gpu_gib", 20)
    for name in ("student_device", "teacher_device"):
        device = torch.device(config[name])
        if device.type != "cuda":
            continue
        free_gib = torch.cuda.mem_get_info(device)[0] / 2 ** 30
        if free_gib < required_free:
            raise RuntimeError(
                f"{config[name]} has only {free_gib:.1f} GiB free; "
                f"require at least {required_free} GiB before loading models"
            )
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    metrics_stream = (output / "metrics.jsonl").open("w", buffering=1)
    torch.manual_seed(config["seed"])
    if torch.device(config["teacher_device"]).type == "cpu":
        torch.set_num_threads(config.get("teacher_cpu_threads", 8))
    dtype = getattr(torch, config.get("dtype", "bfloat16"))
    tokenizer = AutoTokenizer.from_pretrained(config["student"], local_files_only=True)
    student = AutoModelForCausalLM.from_pretrained(
        config["student"],
        dtype=dtype,
        local_files_only=True,
        attn_implementation="sdpa",
        device_map=config["student_device"],
        low_cpu_mem_usage=True,
    )
    student.requires_grad_(True)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.config.use_cache = False
    if any(isinstance(module, torch.nn.Dropout) and module.p for module in student.modules()):
        raise ValueError("Non-zero dropout would change the on-policy distribution")
    optimizer = CPUAdamW(student.parameters(), lr=config["learning_rate"],
                         weight_decay=config.get("weight_decay", 0.0))
    optimizer.initialize_state()
    teachers = TeacherPool(config)
    bandit_policy = (load_bandit_policy(config["bandit_policy"])
                     if config.get("bandit_policy") else None)
    online_bandit = (OnlineLinUCB(config["online_bandit"])
                     if config.get("online_bandit") else None)
    if bandit_policy is not None and config["top_k"] < max(bandit_policy["arms"]):
        raise ValueError("top_k must cover every arm in the Bandit-K policy")
    sampler = DomainSampler(dataset, config["domain_weights"], config["seed"])
    schedule = config.get("domain_schedule")
    completed = 0
    try:
        for step in range(config["steps"]):
            started = time.monotonic()
            def eligible(row):
                return len(render_prompt(tokenizer, row["messages"], config)) <= config["max_prompt_tokens"]

            scheduled_domain = schedule[step % len(schedule)] if schedule else None
            domain, rows = sampler.next_batch(
                config["batch_size"], eligible, domain=scheduled_domain,
            )
            emit("step_started", step=step + 1, domain=domain,
                 sample_ids=[row["id"] for row in rows])
            rollout_started = time.monotonic()
            trajectories = generate_rollouts(student, tokenizer, rows, config, step)
            rollout_seconds = time.monotonic() - rollout_started
            emit("rollout_completed", step=step + 1,
                 response_tokens=[len(item["response_ids"]) for item in trajectories])
            teacher_started = time.monotonic()
            teacher = teachers.activate(domain)
            trajectories = score_trajectories(teacher, trajectories, config)
            teacher_seconds = time.monotonic() - teacher_started
            emit("teacher_prefill_completed", step=step + 1, domain=domain,
                 top_k=config["top_k"])
            selected_step_k = None
            decision = None
            context = None
            reference_k = None
            if online_bandit is not None:
                context = online_bandit.context(
                    domain, trajectories, config["max_new_tokens"],
                )
                selected_step_k, decision = online_bandit.select(context, step)
                reference_k = config["online_bandit"].get(
                    "reference_k", config["top_k"],
                )
                emit("bandit_action_selected", step=step + 1, domain=domain,
                     selected_k=selected_step_k, forced=decision["forced"])
            optimizer.zero_grad()
            student.train()
            backward_started = time.monotonic()
            losses = []
            references_before = []
            for trajectory in trajectories:
                result = backward_trajectory(
                    student, trajectory, config,
                    scale=1 / len(trajectories),
                    bandit_policy=bandit_policy,
                    selected_k=selected_step_k,
                    reference_k=reference_k,
                )
                if online_bandit is None:
                    losses.append(result)
                else:
                    losses.append(result["train_loss"])
                    references_before.append(result["reference_loss"])
            grad_norm = torch.nn.utils.clip_grad_norm_(
                student.parameters(), config.get("grad_clip", 1.0),
                error_if_nonfinite=True,
            )
            optimizer.step()
            optimizer.zero_grad()
            student.eval()
            backward_seconds = time.monotonic() - backward_started
            completed = step + 1
            metric = {
                "step": completed,
                "domain": domain,
                "loss": sum(losses) / len(losses),
                "grad_norm": float(grad_norm),
                "samples": len(trajectories),
                "response_tokens": [len(item["response_ids"]) for item in trajectories],
                "rollout_seconds": rollout_seconds,
                "teacher_seconds": teacher_seconds,
                "backward_optimizer_seconds": backward_seconds,
            }
            if online_bandit is not None:
                reward_started = time.monotonic()
                references_after = [
                    evaluate_reference_loss(student, trajectory, config, reference_k)
                    for trajectory in trajectories
                ]
                reference_before = sum(references_before) / len(references_before)
                reference_after = sum(references_after) / len(references_after)
                raw_gain = relative_reference_gain(
                    reference_before, reference_after,
                    config["online_bandit"].get("reward_epsilon", 1e-8),
                )
                reward = online_bandit.observe(
                    selected_step_k, context, domain, raw_gain,
                )
                online_bandit.save(output / "bandit_state.json")
                metric.update({
                    "selected_k": selected_step_k,
                    "bandit_forced": decision["forced"],
                    "bandit_context": context.tolist(),
                    "bandit_scores": decision["scores"],
                    "reference_k": reference_k,
                    "reference_loss_before": reference_before,
                    "reference_loss_after": reference_after,
                    **reward,
                    "reward_eval_seconds": time.monotonic() - reward_started,
                })
            selected_k = [k for trajectory in trajectories
                          for k in trajectory.get("selected_k", [])]
            if selected_k:
                metric["mean_k"] = sum(selected_k) / len(selected_k)
                metric["k_counts"] = {
                    str(k): selected_k.count(k) for k in bandit_policy["arms"]
                }
            metric["seconds"] = time.monotonic() - started
            metrics_stream.write(json.dumps(metric) + "\n")
            (output / "latest.json").write_text(json.dumps(metric, indent=2))
            emit("step_completed", **metric)
            if checkpoint_due(completed, config):
                destination = output / f"checkpoint-{completed:06d}"
                save_student(
                    student, tokenizer, destination,
                    checkpoint_metadata(config, completed),
                )
                emit("checkpoint_completed", path=str(destination), step=completed)
                if config.get("keep_only_latest_checkpoint", False):
                    removed = prune_old_checkpoints(output, destination)
                    if removed:
                        emit("checkpoints_pruned", kept=str(destination), removed=removed)
            del trajectories
            torch.cuda.empty_cache()
        if config.get("save_final", True):
            destination = output / f"checkpoint-{completed:06d}"
            if not destination.exists():
                save_student(
                    student, tokenizer, destination,
                    checkpoint_metadata(config, completed),
                )
                emit("checkpoint_completed", path=str(destination), step=completed)
                if config.get("keep_only_latest_checkpoint", False):
                    removed = prune_old_checkpoints(output, destination)
                    if removed:
                        emit("checkpoints_pruned", kept=str(destination), removed=removed)
    finally:
        metrics_stream.close()
        teachers.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    report, dataset = preflight(config)
    print(json.dumps(report, indent=2), flush=True)
    if args.check_only:
        return
    if not report["cuda_available"] or report["cuda_devices"] < 1:
        raise SystemExit("Training requires at least one visible CUDA GPU")
    train(config, dataset)


if __name__ == "__main__":
    main()
