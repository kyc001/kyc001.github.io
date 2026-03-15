---
title: 'Parallelism 1 (Multi-Machine Optimization)'
category: 'CS336'
order: 7
description: ""
tags: []
---
# Parallelism 1 (Multi-Machine Optimization)

## Slide 1: 课程目标与背景

- **从单机到多机 (Single GPU to Multi-Machine):**
    - 随着模型规模增大，单张 GPU 无法容纳完整模型，必须跨机器分割。
    - 目标：优化 Compute 和 Memory，同时处理跨机器 Communication。
- **核心挑战:**
    - **Compute Scaling:** 单 GPU 算力增长虽快，但训练 SOTA LLM 需要依赖超级计算机规模的 Exaflops。
    - **Memory Constraints:** 参数量（Parameters）达数十亿，单卡显存无法装下。
- **硬件层级 (Hardware Hierarchy):**
    - **Intra-node:** 单机内（如 8x GPU）通过 NVLink/NVSwitch 连接，带宽极高。
    - **Inter-node:** 跨机器通过 Infiniband/Ethernet 连接，带宽较低（这也是设计并行策略的关键约束）。
    - **Threshold:** 通常 256 GPUs 是一道坎，超过后通信速度会显著下降。

---

## Slide 2: 集合通信原语 (Collective Communication Primitives)

- **基础操作:**
    - **Broadcast:** 将数据从一个 rank 发送到所有其他 rank。
    - **Reduce:** 汇总数据并发送给单一 rank。
    - **All-Reduce:** 所有 rank 进行 Reduce 操作，结果分发给所有 rank（Cost $\approx 2 \times N$）。
- **关键等价性 (Key Identity):**
    - **All-Gather:** 每个 rank 广播其部分数据给所有人。
    - **Reduce-Scatter:** 汇总数据，但结果分散存储在不同 rank 上。
    - **重要结论:** $\text{All-Reduce} = \text{Reduce-Scatter} + \text{All-Gather}$。
    - 在 Bandwidth limited 场景下，这两种方式成本相同，是理解 ZeRO/FSDP 的基础。

---

## Slide 3: 数据并行 (Data Parallelism, DP)

- **基本原理:**
    - 复制 Parameters 到所有 GPU，分割 Batch。
    - **Naive DP:** 每个 GPU 计算 $\frac{B}{M}$ 个样本的 Gradient，通过 All-Reduce 同步 Gradients，然后更新参数。
- **Scaling 特性:**
    - **Compute:** 线性扩展，只要 Batch Size 足够大。
    - **Communication:** 每步需传输 $2 \times$ Parameters 数量的数据。
    - **Memory:** 极差。每张卡都需存储完整的 Parameters、Gradients 和 Optimizer States。

---

## Slide 4: 内存瓶颈与优化器状态 (Memory Bottleneck & Optimizer States)

- **内存消耗分析:**
    - 对于 Mixed Precision 训练，存储消耗远超参数本身。
    - 假设模型参数量为 $\Phi$，**Adam Optimizer** 状态占用极大：
        - Parameters (FP16/BF16): $2$ bytes
        - Gradients (FP16/BF16): $2$ bytes
        - Optimizer States (FP32 Master weights + Momentum + Variance): $4+4+4 = 12$ bytes
    - 总计约 $16 \times \Phi$ bytes。
- **解决方案:** 既然我们要同步 Gradient，为何不在不同 GPU 上分片存储 Optimizer States？这就是 **ZeRO (Zero Redundancy Optimizer)** 的核心思想。

---

## Slide 5: ZeRO (Zero Redundancy Optimizer) 阶段详解

- **ZeRO Stage 1 (Optimizer State Sharding):**
    - 将 Optimizer States 分片存储。每张卡只负责更新一部分参数。
    - **流程:** 计算 Gradient -> Reduce-Scatter (sum gradients) -> 各卡更新自己负责的 Parameters -> All-Gather (broadcast updated parameters)。
    - **代价:** 通信量不变（利用了 All-Reduce = Reduce-Scatter + All-Gather 的等价性），显存显著降低。
- **ZeRO Stage 2 (Gradient Sharding):**
    - 不再存储完整 Gradient 向量。
    - **流程:** 反向传播中，算完一层 Gradient 立刻 Reduce-Scatter 到对应 GPU 并释放显存。
    - 增加少量通信开销，进一步节省内存。
- **ZeRO Stage 3 / FSDP (Fully Sharded Data Parallel):**
    - **Parameter Sharding:** 模型参数也不再全量存储，随用随取。
    - **流程:** Forward/Backward 时，按需 All-Gather 当前层参数 -> 计算 -> 立即 Free 参数。
    - **通信:** 通信量增加到 $3 \times$ Parameters，但内存占用除以 $N_{GPU}$。
    - **Prefetching:** 利用计算与通信重叠 (Overlap) 掩盖延迟。

---

## Slide 6: 模型并行 (Model Parallelism) 之 Pipeline Parallelism (PP)

- **动机:** 即使 ZeRO-3 解决了参数内存，Activations 依然占用大量内存，且 Batch Size 有限。
- **基本原理:** 将模型按 Layers 切分，不同 GPU 负责不同层。
- **Bubble 问题:**
    - Naive Pipeline 会导致严重的空闲时间 (Idle time)，即 "Bubble"。
    - GPU 利用率低，通常只有 $1/N$。
- **优化策略 (1F1B):**
    - 将 Batch 切分为 Micro-batches。
    - 交错执行 Forward 和 Backward，减少 Bubble 大小。
    - **限制:** 依然需要较大的 Batch Size 来填充 Pipeline。
- **Zero Bubble / DualPipe (DeepSeek):**
    - 将反向传播拆分为 "Backward for Activations" 和 "Backward for Weights ($W$)"。
    - 将对 $W$ 的梯度计算调度到原本是 Bubble 的时间段执行。

---

## Slide 7: 模型并行 之 Tensor Parallelism (TP)

- **基本原理:**
    - 切分矩阵乘法 (Matrix Multiplication) 的 Width 维度。
    - **Column Parallel:** 切分权重矩阵 $A$，输入 $X$ 广播 -> 输出拼接。
    - **Row Parallel:** 切分权重矩阵 $B$，输入 $X$ 切分 -> 输出 All-Reduce 汇总。
- **通信特性:**
    - 每层都需要 All-Reduce 同步 Activations。
    - 通信极度频繁，必须依赖高带宽连接。
- **Rule of Thumb:**
    - 仅在单机内部（Intra-node，如 NVLink 连接的 8 卡）使用 TP。
    - 跨机使用 TP 会导致性能暴跌（如从 8 卡到 16 卡吞吐量下降 42%）。

---

## Slide 8: 激活值显存与序列并行 (Activation Memory & Sequence Parallelism)

- **Activation Memory 增长:**
    - 参数和优化器内存可以通过 Sharding 线性减少，但 Activation Memory 随 Model Size 和 Sequence Length 增长。
    - 显存公式近似: $SBH \times (\text{layers}) + \text{Attention overhead}$。
- **Sequence Parallelism (SP):**
    - TP 无法切分 LayerNorm 和 Dropout 等 Point-wise 操作的 Activation。
    - **策略:** 在 Sequence Dimension 上切分这些操作。
    - LayerNorm 前进行 Reduce-Scatter，计算后 All-Gather，消除冗余存储。

---

## Slide 9: 3D 并行策略 (Putting it all together)

- **资源权衡 (Resources):** Memory, Bandwidth, Batch Size。
- **决策流程 (Rule of Thumb):**
    1. **Fit in Memory:** 首先确保模型能装入显存。
        - 优先使用 **Tensor Parallel (TP)** (直到单机上限，如 8 卡)。
        - 其次使用 **Pipeline Parallel (PP)** 或 **ZeRO-3 (FSDP)** 处理跨机内存需求。
    2. **Scale Compute:** 显存足够后，使用 **Data Parallel (DP/ZeRO-1)** 扩展算力。
- **Batch Size 资源:**
    - DP 和 PP 都需要消耗 Batch Size。如果 Batch Size 过小，效率会下降。
    - TP 不消耗 Batch Size。

---

## Slide 10: 业界案例 (Case Studies)

- **常见配置:**
    - **Llama 3:** TP=8 (Intra-node), PP (Inter-node), DP (Outer loop), Context Parallel (For long context).
    - **Megatron-LM:** 经典的 3D Parallelism 范例，TP=8 是最优点。
    - **DeepSeek V3:** 16-way PP + 64-way Expert Parallel (MoE variant) + ZeRO-1.
    - **Google PaLM (TPU):** 倾向于使用 ZeRO-3 (FSDP) + MP，因为 TPU 网络拓扑（Toroidal Mesh）更适合 Collective Communication，减少了对 PP 的依赖。
- **现实挑战:**
    - 大规模训练中硬件故障是常态（如 Llama 3 遇到 148 次 GPU 故障），需具备 Fault Tolerance。

---