---
title: 'GPU 架构与大模型高效计算 (GPU Architecture & Efficient Computing for LLMs)'
category: 'CS336'
order: 5
description: ""
tags: []
---
# GPU 架构与大模型高效计算 (GPU Architecture & Efficient Computing for LLMs)

---

## 1. 课程目标 (Learning Goals)

- **去神秘化 (Demystify GPUs)**: 理解 GPU 的工作原理，消除其作为“黑盒”的神秘感。
- **性能分析 (Performance Analysis)**: 理解为什么 GPU 在某些特定维度下速度快，而在其他情况下变慢（例如 Wave Quantization）。
- **算法加速 (Algorithm Acceleration)**: 学习如何设计快速算法（如 Flash Attention），掌握优化 primitives（原语）,。

---

## 2. 为什么要学习硬件？(Why Hardware Matters)

- **Scaling Laws**: 更多的 Compute（计算）意味着更好的模型性能。深度学习的进步很大程度上由更快的硬件和更好的并行化驱动。
- **Dennard Scaling 的终结**: 单线程性能 (Single-thread performance) 在 2000 年代初期已趋于平缓。现在的扩展主要依赖于 **Parallel Scaling**,。
- **CPU vs. GPU 设计理念差异**,:
    - **CPU**: 优化 **Latency** (延迟)。拥有巨大的 Control Unit (控制单元) 和 Branch Prediction (分支预测)，适合复杂的串行任务。
    - **GPU**: 优化 **Throughput** (吞吐量)。拥有大量的 Compute Units (ALUs) 和较少的 Control logic。通过海量线程并行来隐藏延迟。

---

## 3. GPU 架构与执行模型 (GPU Architecture & Execution Model)

### 3.1 物理架构 (Physical Anatomy)

- **SM (Streaming Multiprocessor)**: GPU 的核心计算单元，包含控制逻辑。一个 GPU（如 A100）包含多个 SM（例如 108 个）,,。
- **SP (Streaming Processor)**: SM 内部的单个处理单元。
- **Tensor Cores**: 专用于 **Matrix Multiplication** (矩阵乘法) 的硬件单元,。

### 3.2 逻辑执行模型 (Logical Execution Model),

- **Grid / Blocks**: 任务被分解为 Blocks，每个 Block 被分配给一个 SM 执行。
- **Warps**: Block 进一步细分为 Warps。Warp 是 GPU 执行的最小调度单位，通常包含 32 个连续线程。
- **Threads**: 每个线程执行相同的指令（SIMT - Single Instruction, Multi-Thread），但处理不同的数据。

---

## 4. 内存层次结构 (Memory Hierarchy)

### 4.1 速度与距离 (Speed & Proximity),,

- **Registers / L1 / Shared Memory**: 位于 SM 内部，速度极快（~20 clock cycles）。优化关键在于尽可能多地利用 Shared Memory。
- **L2 Cache**: 位于芯片上但 SM 之外，速度中等。
- **HBM (High Bandwidth Memory) / Global Memory**: 位于芯片外部 (DRAM)，通过 HBM connectors 连接。速度最慢（~200-300 clock cycles）。
- **Scaling Gap**: Compute (计算能力) 的增长速度远快于 Memory Bandwidth (内存带宽),。这导致现代 LLM 训练往往是 **Memory Bound** (受内存限制) 而非 Compute Bound。

---

## 5. GPU 性能优化技巧 (Performance Optimization Tricks)

为了达到理论峰值性能（接近 Roofline），需要运用以下技巧：

### 5.1 避免条件分支 (Avoid Conditionals),

- GPU 采用 SIMT 模式。如果在同一个 Warp 内出现 `if-else` 分支，硬件必须**序列化 (Serialize)** 执行：先执行 `if` 分支的线程（暂停其他线程），再执行 `else` 分支。这会显著降低利用率。

### 5.2 低精度计算 (Lower Precision),,

- **FP32 $\to$ FP16 / BF16**:
    - 减少了数据传输量（从 8 bytes/flop 降至 4 bytes/flop），相当于免费加倍了有效带宽。
    - **Mixed Precision**: 在 Tensor Core 上使用 FP16 进行乘法，但在 **Accumulator** 中使用 FP32 以保持数值稳定性。

### 5.3 算子融合 (Operator Fusion),,

- **Kernel Fusion**: 避免将中间结果写回 Global Memory。
- **例子**: 计算 `sin^2(x) + cos^2(x)`。
    - _Naive_: Read $x$ $\to$ Calc $\sin$ $\to$ Write $\to$ Read $\to$ Calc $\cos$ ... (多次往返 Memory)。
    - _Fused_: Read $x$ $\to$ Keep in Register $\to$ Calc all steps $\to$ Write Result (仅一次往返)。
- 工具建议: 使用 `torch.compile` 自动进行 fusion。

### 5.4 重计算 (Recomputation),,

- **Gradient Checkpointing**: 在 Backward pass 中，不存储所有中间的 Activations (显存消耗大)，而是从 Input 重新计算它们。
- **Trade-off**: 用多余的 Compute (通常过剩) 换取宝贵的 Memory Bandwidth 和 Capacity。

### 5.5 内存合并 (Memory Coalescing),,

- **Burst Mode**: DRAM 访问不是读取单个字节，而是返回一个 **Burst Section**（例如一块连续内存）。
- **Coalescing**: 确保 Warp 中的线程访问连续的内存地址。这样，硬件可以将多个线程的读取请求合并为一次 DRAM Transaction。
- **反例**: 如果按列访问行存储的矩阵 (Column-major access on Row-major data)，会导致每个线程都需要单独的 Memory Transaction，吞吐量可能下降 4 倍以上。

### 5.6 分块 (Tiling),,

- **原理**: 将大矩阵切分为小块 (Tiles)，加载到 **Shared Memory** 中复用。
- **收益**: 对于 $N \times N$ 的 **Matrix Multiplication**，如果不分块，每个元素需从 Global Memory 读取 $N$ 次。分块后（Tile size $T$），只需读取 $N/T$ 次。显著降低 Global Memory 压力,。

---

## 6. 矩阵乘法性能之谜 (The Mystery of Matrix Multiplication Performance)

### 6.1 Roofline Model

- 性能受限于两个瓶颈之一：
    1. **Memory Bound** (左侧): 算术强度低，受带宽限制。
    2. **Compute Bound** (右侧): 算术强度高，受计算单元限制。

### 6.2 波浪状性能图 (Wavelike Patterns),,

- **Wave Quantization**: 为什么矩阵维度增加一点点，性能会骤降？
    - 假设 Tile 数量从 98 增加到 120，而 GPU 有 108 个 SM。
    - 前 108 个 Tile 并行执行（第一波 Wave，利用率 100%）。
    - 剩下的 12 个 Tile 需要单独占用整个 GPU 执行（第二波 Wave，利用率极低）。
    - **结论**: 矩阵大小或 Tile 数量最好是 SM 数量的整数倍。

### 6.3 对齐与填充 (Alignment & Padding),,

- 如果矩阵维度不是 **Burst Section** 或 **Tile Size** 的倍数（例如 2 的幂次，如 64, 128），会导致边界处的 Memory Access 无法 Coalesce，甚至需要额外的读取操作。
- **案例**: NanoGPT 将词表大小从 50257 增加到 50304 (64 的倍数)，获得了显着的性能提升。

---

## 7. 案例研究：Flash Attention -

Flash Attention 是将上述技巧综合运用的典范。

### 7.1 问题 (The Problem)

- Standard Attention 需要计算 $S = QK^T$ (大小 $N \times N$)，然后计算 $P = \text{Softmax}(S)$，再计算 $O = PV$。
- $N \times N$ 的矩阵对于长序列来说太大，无法放入 SRAM，且读写 HBM 开销巨大 ($O(N^2)$ memory access)。

### 7.2 解决方案 (The Solution)

1. **Tiling**: 将 $Q, K, V$ 切块，在 SRAM 中进行分块矩阵乘法。
2. **Online Softmax**:
    - Softmax 通常需要整行的由数据来计算 Normalization term。
    - 使用 **Online Softmax** 技巧，可以在处理流式数据时动态更新 max 值和 sum 值，无需一次性物化整个 $N \times N$ 矩阵。
3. **Recomputation**:
    - 在 Backward pass 中，不存储巨大的 Attention Matrix，而是利用保存在 SRAM 中的统计量重新计算，从而实现 **Sub-quadratic HBM accesses**,。

### 7.3 总结 (Summary)

Flash Attention 通过 **Tiling** 和 **Recomputation** 避免了 $O(N^2)$ 的 HBM 访问，将瓶颈从 Memory Bandwidth 转移回 Compute，从而大幅提升速度。

---

## 8. 总结 (Conclusion),

- 硬件是现代语言模型扩展的基石。
- **Memory Movement** 是最大的瓶颈。
- 高性能编程的核心在于：利用 **Memory Hierarchy**（如 Shared Memory），通过 **Tiling**、**Coalescing** 和 **Fusion** 减少数据移动。