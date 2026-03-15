---
title: '从零构建语言模型 (Language Modeling from Scratch) - 第10讲：推理 (Inference)'
category: 'CS336'
order: 10
description: ""
tags: []
---
# 从零构建语言模型 (Language Modeling from Scratch) - 第10讲：推理 (Inference)

## 第一部分：推理概述与重要性 (Overview & Importance)

### 1. 推理的定义与应用场景

- **定义**：给定一个训练好的固定模型，根据提示词 (prompts) 生成响应。
- **应用场景**：
    - 聊天机器人 (Chatbots)、代码补全 (Code completion)。
    - 批量数据处理 (Batch data processing)。
    - 模型评估 (Evaluation)：例如指令遵循测试。
    - 强化学习训练 (Training with RL)：需要采样响应并评估奖励。
    - Test-time compute：“思考”过程本质上也是生成 tokens。
- **规模**：OpenAI 每天生成约 100 billion words；Cursor 每天生成约 1 billion lines of code。

### 2. 推理的评价指标 (Inference Metrics)

- **Time to First Token (TTFT)**：用户在看到任何输出前需要等待的时间。对交互式应用至关重要。
- **Latency (延迟)**：首个 token 之后，后续 tokens 到达的速度。影响用户体验。
- **Throughput (吞吐量)**：单位时间内生成的 tokens 总量 (tokens/sec)。
    - 主要用于批量处理 (batch processing)。
    - 高 Throughput 并不一定意味着低 Latency。

---

## 第二部分：推理的算力与内存分析 (Arithmetic Intensity & Memory)

### 3. 推理的核心挑战

- **训练 (Training)**：可以并行处理整个序列 (Parallelize over the sequence)。
- **推理 (Inference)**：生成是顺序的 (Sequential)，因为当前 token 依赖于所有过去的 tokens。无法有效并行，导致难以利用全部算力。

### 4. 算术强度 (Arithmetic Intensity)

- **定义**：$\text{FLOPs} / \text{Bytes transferred}$ (计算量与内存传输量的比值)。
- **Compute Limited (计算受限)**：$\text{Arithmetic Intensity} > \text{Accelerator Intensity}$ (例如 H100 上大于 295)。
- **Memory Limited (内存受限)**：$\text{Arithmetic Intensity}$ 较低，GPU 大部分时间在等待内存读写。
    - **矩阵乘法案例**：对于 Matrix $X (B \times D)$ 和 $W (D \times F)$，Intensity $\approx B$ (当 $D, F$ 很大时)。
    - 如果 Batch size $B$ 很大，则是 Compute Limited；如果 $B=1$ (Matrix-Vector product)，Intensity $\approx 1$，这是极差的，属于 Memory Limited。

### 5. Transformer 推理的两个阶段

- **Prefill (预填充)**：
    - 处理 Prompt，计算并行化。
    - $T = S$ (Sequence length)。
    - $\text{Intensity} \approx S$ (对于 Attention)。通常是 **Compute Limited**。
- **Generation (生成/解码)**：
    - 逐个生成 token。
    - $T = 1$。
    - $\text{Intensity} \approx 1$ (对于 Attention)。本质上是 **Memory Limited**。

---

## 第三部分：KV Cache 与性能权衡 (KV Cache & Trade-offs)

### 6. KV Cache (键值缓存)

- **动机**：避免在生成每个 token 时重新计算整个前缀的 Key 和 Value。
- **机制**：将过去 tokens 的 KV 向量存储在 HBM (High Bandwidth Memory) 中。
- **代价**：显存占用巨大。
    - Cache Size per sequence = $L \times N_{kv} \times D_{head} \times S \times 2 \text{ (bytes for BF16)}$。
    - 总显存占用 = $B \times \text{Cache per sequence} + \text{Model Parameters}$。

### 7. Latency 与 Throughput 的权衡

- 假设：通信与计算完美重叠，且处于 Memory Limited 状态。
- **Latency**：$\text{Latency} \approx \frac{\text{Memory Transferred}}{\text{Memory Bandwidth}}$。随着 Batch size $B$ 增加而增加。
- **Throughput**：$\text{Throughput} \approx \frac{B}{\text{Latency}}$。
    - $B=1$：低 Latency，极低 Throughput (如 124 tokens/s on H100 for Llama-2-13B)。
    - $B$ 增大：Throughput 增加，但在达到显存上限或计算瓶颈后收益递减。
- **简单并行**：运行 $M$ 个模型副本，Throughput 增加 $M$ 倍，Latency 不变。

---

## 第四部分：架构优化 (Architectural Optimizations)

### 8. 缩减 KV Cache (Reducing KV Cache)

- 核心目标：减少显存占用以提高 Throughput (允许更大的 Batch size) 并减少内存传输。
- **Group Query Attention (GQA)**：
    - 减少 KV heads 的数量 (例如 Llama 3)。
    - 多个 Query heads 共享一组 KV heads。
    - 效果：显著减少显存，提高 Throughput，精度损失可忽略。
- **Multi-Head Latent Attention (MLA)** (DeepSeek)：
    - 将 KV 投影到低维潜在空间 (Low-dimensional latent space)。
    - 大幅压缩 KV Cache (如从 16384 维压缩到 512 维)。
- **Cross Layer Attention (CLA)**：
    - 在不同层之间共享 KV projections。
- **Local / Sliding Window Attention**：
    - 只关注最近的 $K$ 个 tokens。
    - KV Cache 大小变为常数 $O(K)$，不再随序列长度 $S$ 增长。
    - 缺点：无法捕捉长距离依赖。解决方案：混合全局注意力 (Hybrid with global attention)。

### 9. 新型模型架构 (Novel Architectures)

- **State Space Models (SSMs)** (如 Mamba, S4)：
    - 基于线性动态系统 (Linear dynamical systems)。
    - 推理状态大小固定 (Constant state size)，无 $O(S^2)$ 复杂度。
    - 挑战：在“联想回忆” (Associative Recall) 任务上表现较弱。
- **Linear Attention**：
    - 使用核技巧 (Kernel trick) 或泰勒展开将 Attention 线性化。
    - 表现类似 RNN，推理效率高 (如 MiniMax 模型)。
- **Diffusion Models**：
    - 并行生成所有 tokens，然后迭代优化 (Refinement)。
    - 打破自回归 (Autoregressive) 瓶颈，利用并行算力。

---

## 第五部分：近似与系统优化 (Approximation & Systems)

### 10. 量化与剪枝 (Quantization & Pruning)

- **Quantization (量化)**：
    - 降低数值精度：FP16/BF16 $\rightarrow$ INT8 $\rightarrow$ INT4。
    - **Post-training quantization**：处理 Outliers (离群值) 是关键。
        - LLM.int8()：对 Outliers 使用 FP16，其余使用 INT8。
        - Activation-aware Quantization (AWQ)：基于激活值重要性进行量化。
- **Model Pruning (剪枝)**：
    - 移除不重要的层、头或维度。
    - 剪枝后进行蒸馏 (Distillation) 以恢复精度。

### 11. 投机解码 (Speculative Decoding)

- **原理**：利用“验证比生成快”的特性 (Verification is parallel/fast; Generation is serial/slow)。
- **流程**：
    1. 使用小模型 (Draft Model) 快速生成 $K$ 个 tokens。
    2. 使用大模型 (Target Model) 并行评估这些 tokens。
    3. 基于拒绝采样 (Rejection Sampling) 决定接受或拒绝。
- **优势**：在保证分布完全一致 (Exact sample from target distribution) 的前提下实现加速 (约 2x)。

### 12. 系统级优化 (System Optimizations)

- **Continuous Batching (连续批处理)**：
    - 处理动态到达的请求，一旦某请求结束，立即插入新请求，不等待整个 Batch 完成。
- **PagedAttention (vLLM)**：
    - 解决显存碎片化 (Fragmentation) 问题。
    - 借鉴操作系统虚拟内存概念，将 KV Cache 分块 (Blocks/Pages) 存储在非连续内存中。
    - 支持复杂的内存共享机制，如 Copy-on-Write。

## 第六部分：总结 (Summary)

- 推理是 **Memory Limited** 且 **Dynamic** 的。
- **优化方向**：
    - **架构层面**：GQA, MLA, SSMs, Linear Attention (减少 KV Cache，改变复杂度)。
    - **近似算法**：Quantization, Pruning。
    - **算法层面**：Speculative Decoding (利用并行验证)。
    - **系统层面**：PagedAttention, Continuous Batching (提高显存利用率和吞吐)。