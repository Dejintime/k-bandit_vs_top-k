"""Full-parameter AdamW with FP32 master weights and moments on CPU."""

import math
import torch


class CPUAdamW:
    """Keep only BF16 parameters/gradients on GPU and Adam state on CPU."""

    def __init__(self, parameters, lr=2e-6, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, chunk_size=1_000_000):
        self.params = [parameter for parameter in parameters if parameter.requires_grad]
        self.lr = float(lr)
        self.betas = tuple(betas)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.chunk_size = int(chunk_size)
        self.state = {}

    def zero_grad(self):
        for parameter in self.params:
            parameter.grad = None

    @torch.no_grad()
    def initialize_state(self):
        for index, parameter in enumerate(self.params):
            if index in self.state:
                continue
            master = parameter.detach().to("cpu", torch.float32, copy=True).contiguous().view(-1)
            self.state[index] = {
                "master": master,
                "m": torch.zeros_like(master),
                "v": torch.zeros_like(master),
                "step": 0,
            }

    @torch.no_grad()
    def step(self):
        beta1, beta2 = self.betas
        for index, parameter in enumerate(self.params):
            if parameter.grad is None:
                continue
            if index not in self.state:
                master = parameter.detach().to("cpu", torch.float32, copy=True).contiguous().view(-1)
                self.state[index] = {
                    "master": master,
                    "m": torch.zeros_like(master),
                    "v": torch.zeros_like(master),
                    "step": 0,
                }
            state = self.state[index]
            state["step"] += 1
            step = state["step"]
            flat_parameter = parameter.view(-1)
            flat_gradient = parameter.grad.view(-1)
            for start in range(0, parameter.numel(), self.chunk_size):
                section = slice(start, min(start + self.chunk_size, parameter.numel()))
                gradient = flat_gradient[section].float().cpu()
                master = state["master"][section]
                first = state["m"][section]
                second = state["v"][section]
                first.mul_(beta1).add_(gradient, alpha=1 - beta1)
                second.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                master.mul_(1 - self.lr * self.weight_decay)
                denominator = second.sqrt().div_(math.sqrt(1 - beta2 ** step)).add_(self.eps)
                master.addcdiv_(first, denominator, value=-self.lr / (1 - beta1 ** step))
                flat_parameter[section].copy_(master)
