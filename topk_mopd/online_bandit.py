"""Online contextual LinUCB for choosing one MOPD support size per update."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "bias",
    "domain_if",
    "domain_math",
    "domain_code",
    "response_length_fraction",
    "teacher_top8_mass",
    "recent_domain_gain_ema",
)
DOMAINS = ("if", "math", "code")


class OnlineLinUCB:
    """Disjoint LinUCB with an observed update-gain reward.

    The controller chooses one ``k`` for the entire optimizer update.  Its
    reward is supplied only after the student has been updated.
    """

    def __init__(self, config):
        self.arms = tuple(config["arms"])
        self.alpha = float(config.get("alpha", 0.5))
        self.ridge = float(config.get("ridge_lambda", 1.0))
        self.cost_lambda = float(config.get("cost_lambda", 0.1))
        self.reward_clip = float(config.get("reward_clip", 5.0))
        self.reward_ema_beta = float(config.get("reward_ema_beta", 0.9))
        self.epsilon = float(config.get("reward_epsilon", 1e-8))
        self.forced_steps = int(config.get("forced_exploration_steps", len(self.arms)))
        self.dimension = len(FEATURE_NAMES)
        self.a = {
            arm: np.eye(self.dimension, dtype=np.float64) * self.ridge
            for arm in self.arms
        }
        self.b = {arm: np.zeros(self.dimension, dtype=np.float64) for arm in self.arms}
        self.pulls = {arm: 0 for arm in self.arms}
        self.ema_abs_gain = 0.0
        self.domain_gain_ema = {domain: 0.0 for domain in DOMAINS}
        self.updates = 0

    def context(self, domain, trajectories, max_new_tokens):
        if domain not in DOMAINS:
            raise ValueError(f"Unsupported Bandit-K domain: {domain}")
        if not trajectories:
            raise ValueError("Bandit-K context requires at least one trajectory")
        response_fraction = np.mean([
            len(item["response_ids"]) / max_new_tokens for item in trajectories
        ])
        masses = []
        for item in trajectories:
            values = item["teacher_logprobs"]
            if values.shape[1] < 8:
                raise ValueError("Online Bandit-K requires at least teacher top-8")
            masses.append(float(values[:, :8].float().exp().sum(-1).mean()))
        return np.asarray([
            1.0,
            float(domain == "if"),
            float(domain == "math"),
            float(domain == "code"),
            float(response_fraction),
            float(np.mean(masses)),
            float(self.domain_gain_ema[domain]),
        ], dtype=np.float64)

    def select(self, context, step):
        context = np.asarray(context, dtype=np.float64)
        if context.shape != (self.dimension,) or not np.isfinite(context).all():
            raise ValueError("Invalid online Bandit-K context")
        details = {}
        for arm in self.arms:
            inverse_context = np.linalg.solve(self.a[arm], context)
            theta = np.linalg.solve(self.a[arm], self.b[arm])
            prediction = float(theta @ context)
            uncertainty = float(np.sqrt(max(context @ inverse_context, 0.0)))
            details[str(arm)] = {
                "prediction": prediction,
                "uncertainty": uncertainty,
                "score": prediction + self.alpha * uncertainty,
            }
        forced = step < self.forced_steps
        if forced:
            arm = self.arms[step % len(self.arms)]
        else:
            arm = max(self.arms, key=lambda value: (details[str(value)]["score"], -value))
        return arm, {"forced": forced, "scores": details}

    def observe(self, arm, context, domain, raw_gain):
        if arm not in self.a or domain not in self.domain_gain_ema:
            raise ValueError("Unknown arm or domain in online Bandit-K update")
        if not np.isfinite(raw_gain):
            raise ValueError("Online Bandit-K reward gain must be finite")
        absolute = abs(float(raw_gain))
        if self.updates == 0:
            self.ema_abs_gain = absolute
        else:
            beta = self.reward_ema_beta
            self.ema_abs_gain = beta * self.ema_abs_gain + (1 - beta) * absolute
        normalized = float(raw_gain) / max(self.ema_abs_gain, self.epsilon)
        normalized = float(np.clip(normalized, -self.reward_clip, self.reward_clip))
        width = max(self.arms) - min(self.arms)
        cost = 0.0 if width == 0 else self.cost_lambda * (arm - min(self.arms)) / width
        reward = normalized - cost
        context = np.asarray(context, dtype=np.float64)
        self.a[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        self.pulls[arm] += 1
        beta = self.reward_ema_beta
        self.domain_gain_ema[domain] = (
            beta * self.domain_gain_ema[domain] + (1 - beta) * float(raw_gain)
        )
        self.updates += 1
        return {
            "raw_gain": float(raw_gain),
            "normalized_gain": normalized,
            "support_cost": float(cost),
            "reward": float(reward),
            "ema_abs_gain": float(self.ema_abs_gain),
        }

    def state_dict(self):
        return {
            "type": "online_disjoint_linucb",
            "feature_names": list(FEATURE_NAMES),
            "arms": list(self.arms),
            "alpha": self.alpha,
            "ridge_lambda": self.ridge,
            "cost_lambda": self.cost_lambda,
            "forced_exploration_steps": self.forced_steps,
            "updates": self.updates,
            "pulls": {str(key): value for key, value in self.pulls.items()},
            "ema_abs_gain": self.ema_abs_gain,
            "domain_gain_ema": dict(self.domain_gain_ema),
            "a": {str(key): value.tolist() for key, value in self.a.items()},
            "b": {str(key): value.tolist() for key, value in self.b.items()},
        }

    def save(self, path):
        Path(path).write_text(json.dumps(self.state_dict(), indent=2))


def relative_reference_gain(loss_before, loss_after, epsilon=1e-8):
    if not np.isfinite(loss_before) or not np.isfinite(loss_after):
        raise ValueError("Reference losses must be finite")
    return float((loss_before - loss_after) / (abs(loss_before) + epsilon))
