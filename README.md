# Bandit-k vs Top-K

比较固定 Top-K MOPD 与 Bandit-K MOPD。两者均由学生模型生成 on-policy 回答，再按样本领域固定选择 IF、Math 或 Code 教师；Bandit 只选择蒸馏支持集大小 `k`，不选择教师。

## 固定 Top-K

Naive MOPD基线在所有训练步骤固定使用 `k=64`。用于判断 Bandit-K 是否能以更小的平均 k 保持相同效果。

## Bandit-K

Bandit-K 是 step-level Disjoint LinUCB，每个训练步骤从以下候选中选择一个 k，并应用于该 batch 的所有 token：

```text
k ∈ {8, 16, 32, 64, 128}
```

### Context

每一步使用7维上下文：

```text
[bias, domain_if, domain_math, domain_code,
 response_length_fraction, teacher_top8_mass,
 recent_domain_gain_ema]
```

- `response_length_fraction`：回答长度占最大生成长度的比例。
- `teacher_top8_mass`：教师 top-8 概率质量，用于表示分布集中程度。
- `recent_domain_gain_ema`：该领域近期参考 loss 改善的 EMA。

### LinUCB

每个 k 独立维护 `A_k` 和 `b_k`：

```text
theta_k = inverse(A_k) * b_k
score_k = theta_k · x + alpha * sqrt(x · inverse(A_k) · x)
```

选择 score 最大的 k。前15步按顺序强制探索所有候选，之后由 UCB 平衡预测收益与不确定性。

### Reward

使用统一的 `reference_k=128` 比较学生更新前后的参考 loss：

```text
raw_gain = (loss_before - loss_after) / (abs(loss_before) + epsilon)
reward = normalized_gain - support_cost(k)
```

`raw_gain` 通过绝对增益 EMA 归一化并裁剪到 `[-5, 5]`。`support_cost(k)` 随 k 线性增加，使策略倾向于选择“能够带来足够学习收益的最小 k”。得到 reward 后，只更新本次所选 k 的 LinUCB 参数。

## 训练参数

Naive MOPD 与 Bandit-K MOPD 除 k 策略外使用相同训练设置。

| 参数 | 设置 |
| --- | --- |
| Student 初始化 | M2RL-SFT |
| Teachers | M2RL-IF / M2RL-Math / M2RL-Coding（按 domain 固定路由） |
| 训练数据 | `data/processed/nemotron-clean/train.jsonl` |
| Training steps | 200 |
| Mini-batch size | 2 trajectories |
| Gradient accumulation steps | 1 |
| Effective batch size | 2 trajectories / optimizer update |
| Optimizer | CPUAdamW |
| Learning rate | `2e-6` |
| Weight decay / gradient clipping | `0` / `1.0` |
| Max prompt / response length | 1,024 / 8,192 tokens |
| Sampling | `temperature=1.0`, `top_p=1.0`, `top_k=0` |
| Precision | BF16 |
| Device | Student 使用单张 GPU；三个 Teacher 在 CPU 上推理 |
| Memory settings | Gradient checkpointing；student/teacher logit chunk 均为 8 tokens |
| Checkpoint | 每 50 steps 保存，仅保留最新 checkpoint，并保存最终模型 |

代码在一个 mini-batch 内依次反向传播 2 条 trajectory 的平均 loss，然后执行一次 optimizer step；没有跨训练 step 累积梯度，因此 `gradient_accumulation_steps=1`。

| 方法 | k 相关参数 |
| --- | --- |
| Naive MOPD | 固定 `k=64` |
| Bandit-K MOPD | `arms=[8,16,32,64,128]`, `reference_k=128`, `alpha=0.5`, `ridge_lambda=1.0`, `cost_lambda=0.1`, `forced_exploration_steps=15` |

## 实验结果

所有数值均为百分比。所有模型使用相同的评测 prompt 和生成参数；Naive 与 Bandit-K 均训练200步。

<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="2">Math</th>
      <th colspan="2">Instruction Following</th>
      <th>Code</th>
    </tr>
    <tr>
      <th>AIME 2024<br><sub>avg@8</sub></th>
      <th>AIME 2025<br><sub>avg@8</sub></th>
      <th>IFBench<br><sub>Prompt Strict</sub></th>
      <th>IFEval<br><sub>Prompt Strict</sub></th>
      <th>LiveCodeBench v5<br><sub>pass@1</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Student (SFT-only)</td>
      <td>47.92</td>
      <td>38.33</td>
      <td>48.00</td>
      <td>79.48</td>
      <td>64.66</td>
    </tr>
    <tr>
      <td>RL Teachers (Math / IF / Code)</td>
      <td><strong>67.50</strong></td>
      <td><strong>64.58</strong></td>
      <td><strong>69.00</strong></td>
      <td><strong>92.24</strong></td>
      <td>71.93</td>
    </tr>
    <tr>
      <td>Naive MOPD (fixed k=64, checkpoint-200)</td>
      <td>66.25</td>
      <td>57.92</td>
      <td>59.67</td>
      <td>85.03</td>
      <td>74.32</td>
    </tr>
    <tr>
      <td>Bandit-K MOPD (Disjoint LinUCB, checkpoint-200)</td>
      <td>65.42</td>
      <td>58.75</td>
      <td>60.67</td>
      <td>84.47</td>
      <td><strong>74.77</strong></td>
    </tr>
  </tbody>
</table>

Bandit-K 的平均选择为 `k=34.96`，五项简单平均为 `68.82`，固定 `k=64` 为 `68.64`，二者最终效果基本持平。
