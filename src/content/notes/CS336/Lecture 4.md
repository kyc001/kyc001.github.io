---
title: 'Mixture of Experts (MoE) From Scratch to DeepSeek V3'
category: 'CS336'
order: 4
description: ""
tags: []
---
# Mixture of Experts (MoE) From Scratch to DeepSeek V3

## Slide 1: 课程概览 (Introduction)

- **MoE 的现状 (2025)**：
    - MoE 已成为构建高性能大模型的标准架构。
    - 代表模型：**DeepSeek V3**, **Llama 4**, **Grok**, **Mixtral**。
    - **NVIDIA Leak** 暗示 GPT-4 实际上可能是 MoE 架构。
- **核心优势**：
    - 在相同的 FLOPs (浮点运算次数) 下，MoE 的性能优于 Dense 模型。
    - "More parameters without affecting FLOPs"（增加参数量但不增加推理计算量）。

## Slide 2: 什么是 Mixture of Experts? (What is MoE?)

- **基本定义**：
    - MoE 是一种**Sparsely Activated**（稀疏激活）的架构。
    - 它并不是针对特定领域（如“编码专家”、“英语专家”）的语义专家，而是一种架构组件。
- **架构变换**：
    - 主要针对 **MLP (Multilayer Perceptron)** 层（即 Transformer 中的 **FFN**）进行改造。
    - 将原本的一个大 **Dense FFN** 替换为：
        1. 一个 **Router / Selector Layer**。
        2. 多个较小的 **Experts** (FFNs)。
- **稀疏性 (Sparsity)**：
    - 对于每个 Token，Router 只选择 **Top-k** 个 Experts 进行计算。
    - **Inference Cost**：如果激活的 Experts 大小总和等于原 Dense FFN，则推理 FLOPs 保持不变，但总参数量显著增加。

## Slide 3: 扩展定律与性能 (Scaling Laws & Performance)

- **训练收益**：
    - 在相同的 **Training FLOPs** 下，增加 Experts 数量可以持续降低 **Training Loss**。
    - **Switch Transformer** 论文 (Fedus et al., 2022) 展示了随着 Expert 数量增加，性能显著提升。
- **验证**：
    - **OLMoE** (AI2) 的消融实验证实：相比 Dense 模型，MoE 的 Loss 下降速度快得多。
    - **DeepSeek V2** 展示了极高的 **Active Parameters** 效率：极少的激活参数实现了很高的 **MMLU** 性能。

## Slide 4: 路由机制 (Routing Mechanisms)

- **核心问题**：如何将 Input Token $x$ 分配给 Experts？
- **路由策略分类**：
    1. **Token Choice**: 每个 Token 选择 Top-k Experts（主流方案，如 DeepSeek, Mixtral）。
    2. **Expert Choice**: 每个 Expert 选择 Top-k Tokens（保证负载均衡，但在 Token 处理上可能不均匀）。
- **Token Choice Top-k Routing 公式**：
    - Input: $x$ (Residual Stream state)
    - Router Weights: $W_g$ (Learnable parameters)
    - Score: $h(x) = x \cdot W_g$
    - Gating: $p = \text{Softmax}(h(x))$
    - Selection: $Gate(x) = \text{TopK}(p)$
    - Output: $y = \sum_{i \in \text{TopK}} Gate(x)_i \cdot E_i(x) + x$。
- **超参数 $k$**：
    - 通常 $k=2$ 以保证 **Exploration** 并提供梯度信号。

## Slide 5: 专家架构设计 (Expert Architecture Design)

- **Standard MoE**: 简单复制 FFNs。
- **DeepSeek Innovation: Shared + Fine-grained Experts**：
    1. **Fine-grained Experts**: 将大 Expert 切分为更小的 Experts（如 hidden dim 缩小4倍，数量增加4倍）。
        - 优势：增加专家数量而不增加计算成本，提升灵活性。
    2. **Shared Experts**: 始终激活的 Experts。
        - 目的：捕获通用的、共享的知识 (Common Knowledge)，减少路由冗余。
- **Ablation Results**:
    - Fine-grained + Shared Experts 组合带来的性能提升显著优于 Vanilla MoE (如 GShard)。

## Slide 6: 训练挑战：负载均衡 (Training Challenges: Load Balancing)

- **Collapsed Mode (崩塌模式)**：
    - 如果不加约束，Router 倾向于将所有 Tokens 发送给同一个 "Super Expert"，导致其他 Experts 死亡 (Dead Experts)。
- **解决方案：Auxiliary Loss (辅助损失)**：
    - 目标：平衡 Experts 的负载。
    - Loss Term: $L_{aux} = N \sum_{i=1}^{N} f_i \cdot P_i$
        - $f_i$: Fraction of tokens actually dispatched to expert $i$.
        - $P_i$: Average router probability for expert $i$.
- **DeepSeek V3 的创新：Auxiliary-Loss-Free Balancing**：
    - 移除传统的 Aux Loss（避免干扰主 Loss）。
    - 引入 Bias term $b_i$：$Score_i = x \cdot e_i + b_i$。
    - 动态更新 $b_i$：如果 Expert $i$ 过载，减少 $b_i$；如果空闲，增加 $b_i$。
    - _注_：V3 最终仍保留了 **Sequence-wise Balance Loss** 以处理极端情况。

## Slide 7: 系统与并行 (Systems & Parallelism)

- **Expert Parallelism**：
    - 将不同的 Experts 放置在不同的 **Devices (GPUs)** 上。
    - **All-to-All Communication**:
        1. **Dispatch**: 将 Tokens 路由到对应 GPU。
        2. **Computation**: 本地 Expert 计算。
        3. **Combine**: 将结果传回原 GPU。
- **DeepSeek V2/V3 通信优化**：
    - 问题：Fine-grained experts 可能导致通信过于碎片化。
    - 解决方案：**Top-m Device Selection**。
        - 先选最多 $m$ 个目标 Devices。
        - 再在这些 Devices 中选 **Top-k** Experts。
- **Token Dropping**：如果某 Expert 的 Buffer 满了，多余的 Tokens 会被丢弃（未处理），导致随机性和性能损失。

## Slide 8: 训练稳定性与 Upcycling (Stability & Upcycling)

- **稳定性技巧**：
    - **Router Precision**: 强制使用 `float32` 进行 Softmax 计算。
    - **Router z-loss**: $L_z = \log^2(\sum e^{logit})$，防止 Logits 变得过大。
- **Sparse Upcycling (升级回收)**：
    - 方法：从已训练好的 **Dense Checkpoint** 初始化 MoE。
    - 步骤：复制 MLP 权重初始化 Experts，随机初始化 Router。
    - 收益：极具成本效益，比从头训练 MoE 更快收敛（如 Qwen, MiniCPM 的实践）。

## Slide 9: 案例研究：DeepSeek V3 架构详解 (DeepSeek V3 Deep Dive)

- **基本配置**：
    - Parameters: 671B Total / 37B Active。
    - Architecture: Shared Experts + Fine-grained Experts (继承自 V1/V2)。
- **V3 关键改进**：
    1. **Sigmoid Routing**: 替代 Softmax，对每个 Expert 独立打分。
    2. **No Aux Loss (mostly)**: 使用 Bias $b_i$ 策略，仅保留 Sequence-wise loss。
    3. **MTP (Multi-Token Prediction)**: 预测未来多个 Tokens（如 Main Model 预测 $t+1$, Lightweight Head 预测 $t+2$），提升训练效率。
    4. **MLA (Multi-Head Latent Attention)**:
        - 通过低维压缩向量 $c$ (Compressed Latent Vector) 替代完整的 KV Cache。
        - **Matrix Absorption**: 将 Up-projection 矩阵吸收到 Query 投影中，不增加推理 FLOPs。

## Slide 10: 总结 (Conclusion)

- **MoE is the Standard**: 在 Compute-constrained 环境下，MoE 是最优解。
- **Key Design Choices**:
    - **Top-k Routing** (Token Choice).
    - **Fine-grained Experts** (for efficiency).
    - **Load Balancing** (via Aux Loss or Bias terms).
- **Future**: 系统优化（通信、内存）与算法设计（Router 学习）的深度结合。