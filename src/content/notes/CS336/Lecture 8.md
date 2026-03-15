---
title: '大规模语言模型训练——并行计算基础 (Parallelism 2)'
category: 'CS336'
order: 8
description: ""
tags: []
---
# 大规模语言模型训练——并行计算基础 (Parallelism 2)

## 第一部分：背景与动机 (Background & Motivation)

### 1. 从单机到多机 (Single GPU to Multi-Machine)

- **现状：** 单个 GPU 的 _Throughput_（吞吐量）优化是基础，但大模型无法放入单卡内存。
- **核心挑战：**
    - **Memory Scaling：** 模型参数（_Parameters_）、梯度（_Gradients_）和优化器状态（_Optimizer States_）超出单卡显存限制。
    - **Compute Scaling：** 训练最大规模模型需要 _Exaflops_ 级别的算力，单卡无法满足。
- **硬件层级与通信 (Hardware Hierarchy)：**
    - **Intra-node (机内)：** NVLink/NVSwitch 提供极高带宽（如 8 GPUs 互联）。
    - **Inter-node (机间)：** Infiniband/Ethernet，带宽显著低于机内互联。
    - **结论：** 这种硬件层级的差异决定了并行策略的选择。

### 2. 集合通信原语 (Collective Communication Primitives)

并行算法的基础构建模块：

- **All-Reduce：** 所有节点求和并同步结果（Sum inputs, copy to all）。成本约为 $2 \times N$。
- **Broadcast：** 从一个节点广播数据到所有节点。
- **Reduce：** 汇总数据到一个节点。
- **All-Gather：** 收集所有节点的数据并分发给所有人。
- **Reduce-Scatter：** 汇总数据，但结果分散在不同节点上（Partial sum distributed）。
- **关键等式 (Key Identity)：**
    - $\text{All-Reduce} = \text{Reduce-Scatter} + \text{All-Gather}$
    - 在带宽受限的情况下，这两种实现方式成本相同，但拆分后对优化器状态分片（ZeRO）至关重要。

---

## 第二部分：数据并行 (Data Parallelism, DP)

### 1. 朴素数据并行 (Naive Data Parallelism)

- **原理：** 复制模型参数到所有设备，将 _Batch Size_ 切分到不同设备。
- **计算公式：**
    - $g = \sum_{i=1}^{B} \nabla L(x_i, y_i; \theta)$
- **缺点：** 内存效率极低。训练一个模型所需的显存约为参数量的 16 倍（_Weights_ + _Gradients_ + _Optimizer States_）。
    - 例如：Adam 优化器状态占据了绝大部分内存。

### 2. ZeRO / FSDP (Fully Sharded Data Parallel)

核心思想：既然 _All-Reduce_ 可以拆分为 _Reduce-Scatter_ 和 _All-Gather_，那么就没有必要在所有设备上保留完整的数据副本。

- **ZeRO Stage 1 (Optimizer State Sharding)：**
    
    - 仅切分 _Optimizer States_。
    - 每张卡只更新自己负责的那部分参数，最后同步。
    - **收益：** 内存显著减少，且在带宽受限时几乎无额外通信开销。
- **ZeRO Stage 2 (Gradient Sharding)：**
    
    - 切分 _Gradients_。
    - 在反向传播（_Backward Pass_）计算完某层梯度后，立即执行 _Reduce-Scatter_ 并释放梯度内存。
- **ZeRO Stage 3 / FSDP (Parameter Sharding)：**
    
    - 切分 _Parameters_。
    - **流程：**
        1. **Forward Pass：** 需要某层参数时执行 _All-Gather_，计算完立即释放（Free weights）。
        2. **Backward Pass：** 同样按需获取参数，计算梯度后执行 _Reduce-Scatter_。
    - **优化：** 通信与计算重叠（_Communication computation overlap_ / Prefetching），利用 GPU 在计算时的空隙提前加载下一层参数。
- **资源限制：** _Batch Size_ 是一种资源。数据并行无法超过 _Batch Size_ 的大小，且过大的 _Batch Size_ 会导致收益递减。
    

---

## 第三部分：模型并行 (Model Parallelism, MP)

当 _Data Parallelism_ 无法满足内存需求或 _Batch Size_ 受限时使用。

### 1. 流水线并行 (Pipeline Parallelism, PP)

- **原理：** 将模型按层（_Layers_）切分，不同设备负责不同深度的层。
- **问题：** 存在 "Bubble"（气泡/空闲时间），设备利用率低。
- **优化：**
    - **Micro-batches：** 将一个 _Batch_ 切分成多个微批次，让不同阶段的 GPU 尽快开始工作，减小 Bubble。
    - **Zero Bubble / DualPipe：** 通过重新调度反向传播中对权重的梯度计算（_Gradient w.r.t weights_），填充 Bubble 时间。
- **适用场景：** 跨节点或跨机架连接（低带宽），因为 PP 仅需点对点传输 _Activations_。

### 2. 张量并行 (Tensor Parallelism, TP)

- **原理：** 将矩阵乘法（_Matrix Multiplication_）切分到多个 GPU 上并行计算。
    - 切分维度：沿宽度（Width）切分。
- **实现 (MLP 示例)：**
    - 对于 $Y = \text{GeLU}(X A)$，将 $A$ 按列切分（Column Parallel），得到 $Y_1, Y_2$。
    - 对于 $Z = Y B$，将 $B$ 按行切分（Row Parallel）。
    - 最后执行 _All-Reduce_ 汇总结果。
- **通信特点：**
    - 每层都需要 _All-Reduce_，通信量极大。
    - **适用场景：** 必须在 NVLink 等高带宽互联的单机内部使用（通常不超过 8 GPUs）。跨机使用会导致性能骤降。

---

## 第四部分：激活值内存与序列并行 (Activation Memory & Sequence Parallelism)

### 1. 激活值瓶颈 (Activation Memory)

- 即便使用了 TP 和 PP，激活值内存（_Activations_）仍随模型规模和序列长度增长。
- **公式：** $SBH \times 34 + 5 AS/H$。
    - TP 可以减少 _MatMul_ 部分的激活值，但 _LayerNorm_、_Dropout_ 等点对点操作（Point-wise ops）的激活值无法被 TP 减少。

### 2. 序列并行 (Sequence Parallelism, SP)

- **原理：** 针对 TP 无法处理的 _LayerNorm_ 和 _Dropout_，沿着序列长度（_Sequence Length_）维度进行切分。
- **实现：** 在 TP 的 _All-Reduce_ 处，拆分为 _Reduce-Scatter_（切分序列）和 _All-Gather_（恢复序列），在两者之间进行分片的点对点计算。
- **效果：** 使得激活值内存也能随 GPU 数量线性减少。

---

## 第五部分：总结与最佳实践 (Summary & Best Practices)

### 1. 3D 并行策略 (3D Parallelism)

如何组合 DP、TP、PP？

1. **首要目标：** 将模型和激活值放入内存。
    - 优先使用 **Tensor Parallelism (TP)**（机内，通常 <= 8）。
    - 如果还不够，使用 **Pipeline Parallelism (PP)** 或 **ZeRO-3/FSDP**（跨机）。
2. **次要目标：** 线性扩展计算能力（Scale Compute）。
    - 一旦内存满足，剩余的 GPU 全部用于 **Data Parallelism (DP)**。
3. **Batch Size 权衡：**
    - TP 不消耗 _Batch Size_。
    - DP 和 PP 都需要消耗 _Batch Size_（DP 需要切分数据，PP 需要 Micro-batches 掩盖 Bubble）。

### 2. 业界案例 (Case Studies)

- **Megatron-LM：** 随着模型变大，先开 TP，再开 PP，最后减少 DP。
- **Llama 3：**
    - TP = 8 (机内)。
    - PP + DP (跨机)。
    - 使用了 Context Parallel (CP) 处理长文本。
- **Google TPU：** 由于 TPU 网络拓扑（Toroidal Mesh）优势，通常较少依赖 PP，更多使用 FSDP 和 TP。

### 3. 故障处理 (Fault Tolerance)

- 大规模训练（如 Llama 3）面临频繁的 GPU 故障和静默数据损坏（Silent Data Corruption），需要健壮的检查点和容错机制。

---

**End of Lecture Notes**