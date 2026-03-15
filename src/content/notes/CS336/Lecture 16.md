---
title: 'Alignment - RL from Verifiable Rewards (基于可验证奖励的强化学习)'
category: 'CS336'
order: 16
description: ""
tags: []
---
# Alignment - RL from Verifiable Rewards (基于可验证奖励的强化学习)

## Slide 1: 引言与背景回顾

- **本课重点**：从 RLHF (Reinforcement Learning from Human Feedback) 转向基于 _Verifiable Rewards_ 的强化学习（如推理模型）。
- **RLHF 回顾**：
    - 目标：最大化基于人类偏好数据的 _Reward_。
    - **DPO (Direct Preference Optimization)**：
        - 将 _Reward_ 重写为 _Policy_ 的比率。
        - 本质上是在偏好数据上进行 _Supervised Learning_，增加好样本的 _Likelihood_，降低坏样本的 _Likelihood_。
- **RLHF 的局限性**：
    - **Overoptimization (Goodhart’s Law)**：随着 _Proxy Reward_ 的增加，真实的 _Human Preference_ 最终会下降。
    - **Calibration** 问题：RL 模型通常会出现过度自信 (_Overconfident_)，不再是校准良好的 _Probabilistic Model_。

## Slide 2: 动机：从人类反馈到可验证奖励

- **Human Feedback 的难点**：难以规模化 (Hard to scale)，容易被 Hack，且人类偏好本身具有噪声。
- **新的范式 (Paradigm Shift)**：
    - 借鉴 AlphaGo/AlphaFold 的成功经验。
    - 在拥有 _True Reward_ 的领域（如数学、代码）进行 RL。
    - 目标：利用 _Verifiable Rewards_ 进行大规模 Post-training。

## Slide 3: PPO (Proximal Policy Optimization) 及其痛点

- **基础：Policy Gradient**
    - 公式：$\nabla J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [\sum \nabla \log \pi_\theta(a|s) R(\tau)]$。
    - 问题：需要频繁采样 (_Rollouts_ 昂贵)，属于 _On-policy_。
- **PPO 的改进**：
    - 利用 _Importance Sampling_ 进行 _Off-policy_ 更新。
    - **Clipping**：限制 _Policy_ 更新幅度，防止参数偏离过远。
- **PPO 的缺点**：
    - 实现极其复杂 (Implementation details matter)。
    - **显存占用**：需要维护一个与 _Policy_ 模型一样大的 **Value Model (Critic)**，显存成本加倍。
    - 需要计算 **Generalized Advantage Estimation (GAE)**。

## Slide 4: GRPO (Group Relative Policy Optimization) 核心概念

- **动机**：去除 **Value Model**，降低系统资源消耗，保留 PPO 的稳定性。
- **算法机制**：
    - 对于每个输入问题 $q$，采样一组输出 (_Group of outputs_) ${o_1, o_2, ..., o_G}$。
    - **Baseline**：利用组内输出的平均 _Reward_ 作为基线。
- **Advantage 计算**：
    - 使用组内的 _Z-score_： $$A_i = \frac{R_i - \text{mean}({R_1, ... R_G})}{\text{std}({R_1, ... R_G})} + \epsilon$$
    - 直觉：同一问题内的输出互为参照，消除了问题难度的影响。

## Slide 5: GRPO 的实现细节

- **KL Divergence Estimation**：
    - 使用一种低方差的估计器： $$D_{KL}(\pi || \pi_{ref}) \approx \frac{\pi_{ref}(x)}{\pi_{\theta}(x)} - \log \frac{\pi_{ref}(x)}{\pi_{\theta}(x)} - 1$$
- **Loss Function**：
    - 在 _Online_ 设定下（单步更新），简化为加权 _Policy Gradient_。
    - 通常包含 _Clip_ 机制以保证训练稳定。
- **Reward Shaping**：
    - _Accuracy Reward_：在序列末尾计算。
    - _KL Penalty_：通常按 _Token_ 级别计算。

## Slide 6: GRPO 的数学争议与修正 (Dr. GRPO)

- **理论问题**：
    - **Standard Deviation Division**：除以标准差违反了 _Policy Gradient Theorem_（Baseline 必须独立于 Action）。这会导致算法偏向于极易或极难的问题（即 _Variance_ 小的情况）。
    - **Length Normalization**：如果 _Reward_ 为负，模型倾向于生成极长的回复（BSing）以稀释惩罚。
- **修正建议**：
    - 移除 _Standard Deviation_ 项。
    - 修正长度归一化问题，可以避免 CoT 长度无限制增长。

## Slide 7: Case Study 1: DeepSeek R1

- **R1-Zero (Pure RL)**：
    - 直接在 Base Model 上运行 RL。
    - Rewards: **Accuracy** (Correctness) + **Format** (Thinking tags)。
    - 现象：模型自发学会了长思维链 (Long CoT)、自我修正 (Self-correction/Aha moment)。
- **R1 Pipeline (Full Recipe)**：
    1. **Cold Start SFT**：使用少量长思维链数据微调，提高可读性和收敛起点。
    2. **Reasoning RL**：使用 GRPO，增加 **Language Consistency Reward** 防止语言混合。
    3. **General Post-training**：加入非推理数据 SFT 和 RLHF，恢复通用能力。
- **重要发现**：
    - **Distillation**：将 R1 的思维链数据用于微调小模型 (如 Qwen)，效果显著提升。
    - **Negative Results**：**PRM (Process Reward Models)** 和 **MCTS (Search)** 在 R1 中并未带来明显收益，**Outcome Reward** 依然是主流。

## Slide 8: Case Study 2: Kimi k1.5

- **概述**：性能匹敌 OpenAI o1，采用类似的 Long CoT + RL 路线。
- **数据策略**：
    - **Filtering**：使用 **Best-of-N** 策略。如果是简单问题（任何一次采样都正确）则剔除，保留有挑战性的样本。
- **RL 算法变体**：
    - 类似 DPO 的推导，使用 **Squared Loss** 对齐 _Policy Ratio_ 和 _Reward Difference_。
    - 本质上仍是带 _Baseline_ 的 _Policy Gradient_。
- **长度控制 (Length Control)**：
    - 不鼓励无限长的 CoT。
    - **Length Reward**：只有当答案正确 (_Correct_) 时，才奖励较短的回复；答案错误时，推向平均长度。
    - **Curriculum**：从易到难，按 $1 - \text{Success Rate}$ 比例采样。

## Slide 9: Kimi k1.5 的系统架构

- **RL 基础设施挑战**：
    - _Rollouts_ 昂贵，且长 CoT 导致 _Batch_ 极度不均匀。
- **架构设计**：
    - 解耦 **Training Workers** 和 **Inference Workers**。
    - 使用消息传递同步权重和数据。
    - 技巧：使用带有 Dummy weights 的 vLLM，甚至定期重启 Worker 以清理显存。

## Slide 10: Case Study 3: Qwen 3 (Reasoning)

- **流程**：Long CoT SFT -> Reasoning RL -> **Thinking Mode Fusion** -> RLHF。
- **数据效率**：仅使用 ~4000 个样本进行 RL 即可获得显著提升。
- **Thinking Mode Fusion (思维模式融合)**：
    - 目的：在一个模型中同时支持 _Thinking_ 和 _Non-thinking_ 模式。
    - 方法：微调模型识别 `<think>` 和 `<no-think>` 标签。
    - **Test-time Scaling**：支持强制提前结束思考（截断 CoT），在性能和推理成本之间进行权衡 (Trade-off)。
- **Trade-off 观察**：
    - Reasoning RL 提升了数学/STEM 能力，但可能会损害通用指令遵循能力 (General Instruction Following)，需要后期的 RLHF 修复。

## Slide 11: 总结与展望

- **RL is Powerful**：从 RLHF 到基于 _Verifiable Rewards_ 的 RL 是当前的趋势。
- **算法趋势**：GRPO 及其变体证明了去除 **Value Model** 的可行性，简化了训练。
- **关键要素**：
    - **Outcome Based Rewards** 目前优于 Process Rewards。
    - **Data Curation** (难度分级、去重) 至关重要。
    - **Inference Efficiency**：控制 CoT 长度和推理成本是产品化的关键。