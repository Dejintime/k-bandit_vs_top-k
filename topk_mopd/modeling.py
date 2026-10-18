"""Rollout, teacher prefill, and bounded-memory student backward."""

import gc
import torch

from .bandit_policy import select_token_k
from .loss import (adaptive_topk_reverse_kl_from_logprobs,
                   topk_reverse_kl_from_logprobs)


def render_prompt(tokenizer, messages, config):
    messages = [dict(message) for message in messages]
    instruction = config.get("rollout_system_prompt", "")
    if instruction:
        if messages[0]["role"] == "system":
            messages[0]["content"] = messages[0]["content"].rstrip() + "\n\n" + instruction
        else:
            messages.insert(0, {"role": "system", "content": instruction})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=config.get("enable_thinking", False),
        return_dict=False,
    )


@torch.no_grad()
def generate_rollouts(student, tokenizer, rows, config, step):
    student.eval()
    device = next(student.parameters()).device
    trajectories = []
    for index, row in enumerate(rows):
        prompt = render_prompt(tokenizer, row["messages"], config)
        if not prompt or len(prompt) > config["max_prompt_tokens"]:
            raise ValueError(f"Prompt {row['id']} is empty or exceeds max_prompt_tokens")
        seed = config["seed"] + step * config["batch_size"] + index
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
        generated = student.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=config["max_new_tokens"],
            pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                          else tokenizer.eos_token_id),
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        response = generated[0, len(prompt):].tolist()
        if not response:
            raise ValueError(f"Student produced an empty response for {row['id']}")
        trajectories.append({
            "id": row["id"],
            "domain": row["domain"],
            "prompt_ids": prompt,
            "response_ids": response,
            "policy_step": step,
        })
        del input_ids, generated
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    return trajectories


@torch.no_grad()
def response_hidden(model, input_ids, prompt_length, chunk_size):
    """Prefill with a full-context KV cache and retain response predictor states."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    cache = None
    chunks = []
    stop = input_ids.shape[1] - 1
    for start in range(0, stop, chunk_size):
        end = min(start + chunk_size, stop)
        output = model.model(
            input_ids=input_ids[:, start:end],
            attention_mask=torch.ones((1, end), dtype=torch.long, device=device),
            past_key_values=cache,
            use_cache=True,
        )
        cache = output.past_key_values
        offset = max(prompt_length - 1 - start, 0)
        if offset < end - start:
            chunks.append(output.last_hidden_state[0, offset:].detach().cpu())
    if not chunks:
        raise ValueError("No response positions found during teacher prefill")
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def score_trajectories(teacher, trajectories, config):
    teacher.eval()
    top_k = config["top_k"]
    prefill_chunk = config.get("prefill_chunk_tokens", 512)
    logit_chunk = config.get("teacher_logit_chunk_tokens", 128)
    result = []
    head = teacher.get_output_embeddings()
    for trajectory in trajectories:
        tokens = trajectory["prompt_ids"] + trajectory["response_ids"]
        input_ids = torch.tensor([tokens], dtype=torch.long)
        hidden = response_hidden(teacher, input_ids, len(trajectory["prompt_ids"]), prefill_chunk)
        all_ids, all_values = [], []
        for start in range(0, hidden.shape[0], logit_chunk):
            block = hidden[start:start + logit_chunk].to(head.weight.device)
            logprobs = head(block).float().log_softmax(-1)
            values, ids = logprobs.topk(top_k, dim=-1)
            all_ids.append(ids.cpu())
            all_values.append(values.cpu())
            del block, logprobs, values, ids
        teacher_ids = torch.cat(all_ids)
        teacher_logprobs = torch.cat(all_values)
        if teacher_ids.shape != (len(trajectory["response_ids"]), top_k):
            raise ValueError("Teacher top-k response alignment failed")
        result.append({
            **trajectory,
            "teacher_ids": teacher_ids,
            "teacher_logprobs": teacher_logprobs,
        })
        del hidden, teacher_ids, teacher_logprobs
    teacher_device = next(teacher.parameters()).device
    if teacher_device.type == "cuda":
        with torch.cuda.device(teacher_device):
            torch.cuda.empty_cache()
    return result


def backward_trajectory(student, trajectory, config, scale, bandit_policy=None,
                        selected_k=None, reference_k=None):
    """Exact full-parameter gradient with a chunked output projection."""
    if bandit_policy is not None and selected_k is not None:
        raise ValueError("Frozen token policy and online step-level k are mutually exclusive")
    device = next(student.parameters()).device
    prompt = trajectory["prompt_ids"]
    response = trajectory["response_ids"]
    tokens = torch.tensor([prompt + response], dtype=torch.long, device=device)
    output = student.model(
        input_ids=tokens,
        attention_mask=torch.ones_like(tokens),
        use_cache=False,
    ).last_hidden_state
    hidden = output[0, len(prompt) - 1:-1]
    leaf = hidden.detach().requires_grad_(True)
    teacher_ids = trajectory["teacher_ids"]
    teacher_logprobs = trajectory["teacher_logprobs"]
    token_count = len(response)
    chunk_size = config.get("student_logit_chunk_tokens", 64)
    total = 0.0
    reference_total = 0.0
    for start in range(0, token_count, chunk_size):
        end = min(start + chunk_size, token_count)
        logits = student.get_output_embeddings()(leaf[start:end])
        student_logprobs = logits.float().log_softmax(-1)
        if selected_k is not None:
            if selected_k < 1 or selected_k > teacher_ids.shape[1]:
                raise ValueError("selected_k exceeds cached teacher support")
            loss = topk_reverse_kl_from_logprobs(
                student_logprobs,
                teacher_ids[start:end, :selected_k],
                teacher_logprobs[start:end, :selected_k],
            ) / token_count
            ref_k = reference_k if reference_k is not None else teacher_ids.shape[1]
            if ref_k < selected_k or ref_k > teacher_ids.shape[1]:
                raise ValueError("reference_k must cover selected_k and cached support")
            reference = topk_reverse_kl_from_logprobs(
                student_logprobs.detach(),
                teacher_ids[start:end, :ref_k],
                teacher_logprobs[start:end, :ref_k],
            ) / token_count
            reference_total += reference.item()
        elif bandit_policy is None:
            loss = topk_reverse_kl_from_logprobs(
                student_logprobs,
                teacher_ids[start:end],
                teacher_logprobs[start:end],
            ) / token_count
        else:
            token_k = select_token_k(
                student_logprobs, teacher_logprobs[start:end],
                response[start:end], trajectory["domain"], start, token_count,
                bandit_policy,
            )
            trajectory.setdefault("selected_k", []).extend(token_k.cpu().tolist())
            loss = adaptive_topk_reverse_kl_from_logprobs(
                student_logprobs, teacher_ids[start:end],
                teacher_logprobs[start:end], token_k,
            ) / token_count
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite top-k MOPD loss")
        total += loss.detach().item()
        (loss * scale).backward()
        del logits, student_logprobs, loss
    hidden.backward(leaf.grad)
    del tokens, output, hidden, leaf
    if selected_k is not None:
        return {"train_loss": total, "reference_loss": reference_total}
    return total


@torch.no_grad()
def evaluate_reference_loss(student, trajectory, config, reference_k):
    """Evaluate the common top-k reference objective without a teacher rerun."""
    device = next(student.parameters()).device
    prompt = trajectory["prompt_ids"]
    response = trajectory["response_ids"]
    teacher_ids = trajectory["teacher_ids"]
    teacher_logprobs = trajectory["teacher_logprobs"]
    if reference_k < 1 or reference_k > teacher_ids.shape[1]:
        raise ValueError("reference_k exceeds cached teacher support")
    tokens = torch.tensor([prompt + response], dtype=torch.long, device=device)
    output = student.model(
        input_ids=tokens,
        attention_mask=torch.ones_like(tokens),
        use_cache=False,
    ).last_hidden_state
    hidden = output[0, len(prompt) - 1:-1]
    token_count = len(response)
    chunk_size = config.get("student_logit_chunk_tokens", 64)
    total = 0.0
    for start in range(0, token_count, chunk_size):
        end = min(start + chunk_size, token_count)
        logits = student.get_output_embeddings()(hidden[start:end])
        student_logprobs = logits.float().log_softmax(-1)
        loss = topk_reverse_kl_from_logprobs(
            student_logprobs,
            teacher_ids[start:end, :reference_k],
            teacher_logprobs[start:end, :reference_k],
        ) / token_count
        total += loss.item()
        del logits, student_logprobs, loss
    del tokens, output, hidden
    return total


class TeacherPool:
    """Keep frozen teachers in RAM; optionally move only the routed one to GPU."""

    def __init__(self, config):
        from transformers import AutoModelForCausalLM

        dtype = getattr(torch, config.get("dtype", "bfloat16"))
        self.device = torch.device(config["teacher_device"])
        self.models = {}
        self.active = None
        for domain, path in config["teachers"].items():
            self.models[domain] = AutoModelForCausalLM.from_pretrained(
                path,
                dtype=dtype,
                local_files_only=True,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
            ).requires_grad_(False).eval()

    def activate(self, domain):
        if self.device.type == "cpu":
            self.active = domain
            return self.models[domain]
        if self.active == domain:
            return self.models[domain]
        if self.active is not None:
            self.models[self.active].to("cpu")
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        self.models[domain].to(self.device)
        self.active = domain
        return self.models[domain]

    def close(self):
        if self.active is not None and self.device.type == "cuda":
            self.models[self.active].to("cpu")
        self.models.clear()
        self.active = None
        gc.collect()
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
