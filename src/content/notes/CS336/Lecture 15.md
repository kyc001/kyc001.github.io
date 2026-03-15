---
title: 'Stanford CS336 Lecture 15: Alignment - Supervised Fine-Tuning (SFT) & RLHF'
category: 'CS336'
order: 15
description: ""
tags: []
---
# Stanford CS336 Lecture 15: Alignment - Supervised Fine-Tuning (SFT) & RLHF

## 1. 课程概览 (Overview)

- **从 Pre-training 到 Post-training 的转变**
    - Pre-training (GPT-3): 拥有强大的能力，但不一定有用或安全,。
    - Post-training (ChatGPT): 将预训练模型转化为能够遵循指令、有用且安全的产品。
- **本课重点**
    - Part 1: Supervised Fine-Tuning (SFT) - 数据与方法。
    - Part 2: RLHF (Reinforcement Learning from Human Feedback) - 数据收集与算法。

---

## 2. Supervised Fine-Tuning (SFT)

### 2.1 SFT 数据集的三种范式 (Data Paradigms)

1. **Aggregation of NLP Tasks (e.g., FLAN)**
    - 方法：聚合现有的 NLP 任务（如 QA、摘要）并转化为指令形式。
    - 特点：数据量大，但风格偏向 Benchmark，不太像自然对话。
2. **Human-Written Instructions (e.g., Open Assistant)**
    - 方法：由爱好者或众包编写真实的指令和回复。
    - 特点：质量高，包含复杂的查询和详细回复，但收集成本高。
3. **Model-Generated / AI Feedback (e.g., Stanford Alpaca)**
    - 方法：使用强模型（如 Text-Davinci-003）生成指令和回复,。
    - 特点：模拟人类与 Chatbot 的交互风格，成本较低。

### 2.2 SFT 的挑战 (Challenges in SFT)

- **幻觉 (Hallucination) 与知识边界**
    - **现象：** 训练模型在回答复杂问题时添加引用（Citation），可能导致模型学会编造事实（Hallucinate）以匹配训练数据的格式。
    - **John Schulman 的观点：** 如果 SFT 数据包含模型 Pre-training 中未见过的知识，强制模型回答会教导它去编造（"Type check" the response），而不是承认不知道,。
    - **Behaviors：** 模型倾向于学习输出的风格（Style）和类型签名（Type signature），而非事实本身。
- **安全性 (Safety)**
    - 需要在“拒绝回答有害问题”和“过度拒绝（Over-refusal）”之间通过少量数据进行权衡。

### 2.3 SFT 的扩展方法：Mid-Training

- **方法：** 不仅仅是最后的小规模 Fine-tuning，而是在 Pre-training 的最后阶段（Decay stage）混合 SFT 数据。
- **案例 (MiniCPM)：**
    - Stage 1: Pure Pre-training.
    - Stage 2 (Decay): 混合高质量 Pre-training 数据与大量 SFT 数据（Code, UltraChat 等）。
- **优势：** 避免 Catastrophic Forgetting，使 Base Model 具备指令跟随能力,。

---

## 3. Reinforcement Learning from Human Feedback (RLHF)

### 3.1 核心范式转变 (Paradigm Shift)

- **Generative Modeling (SFT):** 试图模仿某个参考分布 $P^*$。
- **Reward Maximization (RLHF):** 寻找一个策略 (Policy) $P(Y|X)$ 以最大化奖励函数 $R(Y, X)$。

### 3.2 为什么要在大规模 SFT 后做 RLHF？

1. **数据成本：** SFT 需要专家撰写长篇回复（Generation），昂贵且困难。
2. **验证与生成的差异 (Generator-Validator Gap)：** 人类评估（Verification）比创作（Generation）更容易。且研究表明，人类有时更偏好 AI 生成的摘要而非人类自己写的。

### 3.3 RLHF 数据收集 (Data Collection)

- **形式：** Pairwise Feedback (A vs. B)。
    - 模型生成两个输出，标注员选择更好的一个。
- **标注指南 (Guidelines) - 以 InstructGPT 为例:**
    - **Helpful:** 清晰，回应用户意图。
    - **Truthful:** 不包含幻觉。
    - **Harmless:** 无毒，不包含冒犯性内容。
- **数据收集的困难与偏差,,:**
    - 标注员在时间压力下（如1分钟/题）难以验证事实准确性。
    - **Length Bias:** 标注员和 AI 裁判都倾向于更长的回复，即使包含幻觉,。
    - **AI Feedback:** 越来越多地使用强模型（如 GPT-4）代替人类进行打分（Constitutional AI, Tulu 3），因为成本更低且一致性高,。

---

## 4. RLHF 算法 (Algorithms)

### 4.1 优化目标 (Objective)

目标是最大化 Reward，同时限制策略不偏离 Reference Model 太远： $$ \max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)} [R(x, y)] - \beta \text{KL}(\pi_\theta(y|x) || \pi_{\text{ref}}(y|x)) $$

- $R(x, y)$: Reward Model。
- $\text{KL}$: Kullback-Leibler Divergence，用于防止模型 Mode Collapse 或遗忘。

### 4.2 奖励模型 (Reward Modeling) - Bradley-Terry Model

假设存在一个潜在的标量奖励 $R$，人类偏好遵循 Logistic 分布,: $$ P(y_w \succ y_l | x) = \sigma(R(x, y_w) - R(x, y_l)) $$

### 4.3 PPO (Proximal Policy Optimization)

- **概述：** InstructGPT 使用的标准算法。
- **组件：** 需要训练一个独立的 Reward Model，计算 Advantage，并使用 Importance Sampling 和 Clipping 更新 Policy,。
- **缺点：** 复杂，不稳定，计算资源消耗大。

### 4.4 DPO (Direct Preference Optimization)

- **核心思想：** 移除独立的 Reward Model，直接在偏好数据上优化 Policy。
- **推导过程 (Derivation),:**
    1. **Optimal Policy Form:** 在带有 KL 约束的 Reward Maximization 问题中，最优策略  $\pi^*$ 的解析解为： $$
\pi^*(y \mid x)
= \frac{1}{Z(x)} \, \pi_{\text{ref}}(y \mid x)
\exp\!\left(\frac{R(x, y)}{\beta}\right)
$$
    
    2. **Inverse Map:** 将上述公式反转，用 Policy 表示 Reward： $$ R(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x) $$
    3. **Substitution:** 将 $R(x, y)$ 代入 Bradley-Terry 模型公式，消去配分函数 $Z(x)$： $$ P(y_w \succ y_l) = \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) $$
- **结果：** 将 RL 问题转化为类似于 Supervised Learning 的 Maximum Likelihood 问题，更加稳定和简单。

---

## 5. 总结 (Conclusion)

- **SFT:** 即使少量高质量数据也能产生巨大影响，但需注意幻觉和 Style over Substance 的问题。现代 Scaling 倾向于在 Pre-training 后期混合 SFT 数据。
- **RLHF:** 从概率分布匹配转向奖励最大化。虽然 PPO 是经典方法，但 DPO 因其简洁性和去除了显式 Reward Model 而变得流行。
- **Data is Key:** 无论是 SFT 还是 RLHF，数据的质量、多样性和标注者的偏差（Bias）都是决定模型性能的关键因素。