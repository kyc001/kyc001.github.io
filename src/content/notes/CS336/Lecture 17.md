---
title: '从零构建语言模型 - 对齐与强化学习进阶 (Alignment - RL 2)'
category: 'CS336'
order: 17
description: ""
tags: []
---
# 从零构建语言模型 - 对齐与强化学习进阶 (Alignment - RL 2)

## 1. 课程概览 (Overview)

- **回顾 (Recap):** 上节课介绍了从 Verified Rewards 进行 RL 以及 Policy Gradient 算法（如 PPO 和 GRPO）。
- **本课目标 (Goals):**
    - 深入探讨 Policy Gradient 的机制 (Mechanics)。
    - 结合代码 (Code) 和数学 (Math) 进行详细解析。
    - 通过一个简化的排序任务 (Sorting Task) 演示 GRPO 的实现细节。

---

## 2. 语言模型中的强化学习框架 (RL Framework in LMs)

- **状态 (State, $s$):** Prompt + 当前生成的 Response。
    - 在 LM 中，状态空间是生成的 Token 序列，具有极高的自由度 (Degrees of Freedom)，模型可以构建自己的 "Scratch pad"。
- **动作 (Action, $a$):** 生成下一个 Token。
- **奖励 (Reward, $r$):** 衡量 Response 质量的函数。
    - **Outcome Rewards:** 基于完整 Response 的确定性计算函数（如数学题答案验证）。
- **转换动力学 (Transition Dynamics):** 确定性的 Append 操作 (tokens)。
    - 这意味着我们可以理解环境动力学，理论上允许进行 Planning (Test-time compute)。

---

## 3. 策略梯度基础 (Policy Gradient Basics)

- **目标函数 (Objective Function):** 最大化 Expected Reward。
    - $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$。
- **梯度计算 (Gradient Computation):**
    - 利用 "Log-derivative trick" (Policy Gradient Theorem)。
    - $\nabla J(\theta) = \mathbb{E}[R \cdot \nabla \log \pi(a|s)]$。
- **直观理解 (Intuition):**
    - Naive Policy Gradient 类似于 SFT (Supervised Fine-Tuning)，但样本由模型自己生成，并根据 Reward 进行加权。
    - 如果是 Binary Reward (0/1)，这就像是只在正确答案上做 SFT。

---

## 4. 稀疏奖励与方差问题 (Sparse Rewards & Variance)

- **稀疏奖励 (Sparse Rewards):**
    - 例如数学题，大多数生成的 Response 可能是错误的 ($r=0$)。
    - 如果模型很差，很难获得非零梯度，导致无法更新 (Gradient is zero)。
- **高方差 (High Variance):**
    - RL 的噪声远高于监督学习。
    - **案例分析 (Toy Example):**
        - State $S_1$ (简单题): Actions $A_1 \to 11, A_2 \to 9$。
        - State $S_2$ (难题): Actions $A_1 \to 0, A_2 \to 2$。
        - Naive PG 可能过度强化 $S_1$ 的次优动作 ($A_2, r=9$)，而忽略 $S_2$ 的最优动作 ($A_2, r=2$)，因为 $9 > 2$。

---

## 5. 基线 (Baselines) 引入

- **核心思想 (Key Idea):**
    - $\nabla J(\theta) = \mathbb{E}[\nabla \log \pi(a|s) \cdot (R - b(s))]$。
    - 只要 Baseline $b(s)$ 不依赖于 Action $a$，梯度的 Expectation 保持不变，但 Variance 可以显著降低。
- **最优基线 (Optimal Baseline):**
    - 理论上存在最小化方差的 Closed-form solution，但计算复杂。
- **启发式基线 (Heuristic):**
    - $b(s) \approx \mathbb{E}[R|s]$ (Expected Reward given State)。
- **优势函数 (Advantage Function):**
    - $A(s, a) = Q(s, a) - V(s)$。
    - 使用上述 Baseline 本质上是在优化 Advantage：即当前 Action 比平均水平好多少。

---

## 6. GRPO (Group Relative Policy Optimization)

- **背景 (Context):**
    - 针对语言模型优化的算法，去除了 PPO 中的 Critic (Value Function) 模型。
- **机制 (Mechanism):**
    - 利用 LM 的特性：同一个 Prompt 可以生成一组 (Group) Response。
    - **Baseline:** 使用该组 Response 的平均 Reward 作为 Baseline。
    - 这种 Group Structure 提供了自然的比较基准，降低方差。

---

## 7. 案例研究：排序任务 (Case Study: Sorting Task)

- **任务定义 (Task):** 输入 $n$ 个数字，输出排序后的数字。
- **奖励设计 (Reward Design):**
    - **Exact Match:** 奖励为 1 或 0。问题：过于稀疏。
    - **Partial Credit (位置匹配):** 计算匹配正确位置的数量。
    - **Partial Credit (包含 + 顺序):** Token 包含在 Prompt 中得 1 分，相邻有序对得 1 分。
        - _注意:_ Reward Hacking 风险，模型可能为了 Partial Credit 而生成虽长但错误的答案。
- **模型架构 (Model):**
    - 为简化演示，使用非自回归 (Non-autoregressive) 的简单模型，独立解码每个位置。

---

## 8. 算法实现流程 (Implementation Pipeline)

1. **Inference (生成):**
    - 对于每个 Prompt，生成 $G$ 个 Responses (Batch, Trial, Position)。
2. **Compute Rewards (计算奖励):**
    - 对每个 Response 计算数值奖励。
3. **Compute Deltas (计算优势/信号):**
    - **Standard:** 直接使用 Rewards。
    - **Centered:** 减去该 Group 的 Mean ($r - \mu$)。这使得低于平均的表现受到惩罚 (Negative Update)。
    - **Normalized:** $(r - \mu) / (\sigma + \epsilon)$ (GRPO 标准做法)。
4. **Compute Log Probs:**
    - 计算当前模型对生成 Response 的 Log Probabilities。

---

## 9. 损失函数计算 (Loss Calculation)

- **Naive PG Loss:** $\text{LogProbs} \times \text{Deltas}$。
- **GRPO Loss (With Clipping):**
    - 计算 Ratio: $r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}$。
        - _注意:_ $\pi_{\text{old}}$ 需要 `no_grad` 处理，视为 Constant。
    - Objective: $\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)$。
    - **Clipping** 限制了单次更新的幅度，防止策略偏离过远。
- **KL Penalty (Regularization):**
    - 防止模型偏离 Reference Model ($\pi_{\text{ref}}$) 太远。
    - 估计量: $\frac{\pi}{\pi_{\text{ref}}} - \log(\frac{\pi}{\pi_{\text{ref}}}) - 1$ (低方差无偏估计)。

---

## 10. 训练循环与系统复杂性 (Training Loop & Complexity)

- **双重循环结构:**
    - **Outer Loop (Epochs):** 采样 Prompts，生成 Responses (Rollout)。
    - **Inner Loop (Steps):** 在同一批 Responses 上进行多次梯度更新。
- **多模型管理 (Multiple Models):**
    - $\pi_\theta$ (Current Policy): 正在训练的模型。
    - $\pi_{\text{old}}$ (Old Policy): 用于计算 Importance Sampling Ratio (通常在 Inner Loop 保持不变或缓存 Log Probs)。
    - $\pi_{\text{ref}}$ (Reference Model): 用于计算 KL Penalty，更新频率较低。
- **实验观察 (Observations):**
    - Loss Curve 在 RL 中具有误导性，因为 Dataset (生成的 Responses) 随时间变化。应关注 Average Reward。
    - Centered Rewards 对于从全部失败 (Reward=0) 的 Batch 中提取信号没有帮助，但在有差异的 Batch 中至关重要。

---

## 11. 总结与展望 (Conclusion)

- **RL 的价值:** 能够超越监督数据 (Human Labels)，因为 Label 只是模仿，而 RL 直接优化 Reward。
- **挑战 (Challenges):**
    - **Reward Engineering:** 设计难以被 Hack 且可泛化的奖励函数。
    - **System Engineering:** 推理成本 (Inference)、多模型显存管理、分布式训练。
- **核心回顾:** Policy Gradient = Expected Reward + Baseline + Gradient Estimator。