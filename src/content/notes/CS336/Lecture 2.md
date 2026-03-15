---
title: 'CS336: PyTorch Primitives & Resource Accounting (Lecture 2)'
category: 'CS336'
order: 2
description: ""
tags: []
---
# CS336: PyTorch Primitives & Resource Accounting (Lecture 2) 

**(PyTorch 原语与资源核算)**
## 1. Introduction & Motivation (引言与动机)

- **课程目标**：不仅仅是实现模型，更要关注**Efficiency**（效率）和**Resource Accounting**（资源核算）。
- **Napkin Math (餐巾纸数学)**：在开始训练前，通过简单的数学估算所需的计算资源和时间。
    - **Example Question**：在使用 1024 张 H100 GPU 上训练一个 70B 参数、15T Tokens 的模型需要多久？
    - **Key Formula**：总计算量 (Flops) $\approx 6 \times \text{Number of Parameters} \times \text{Number of Tokens}$。
- **Hardware Constraints**：
    - H100 GPU 拥有 80GB HBM (High Bandwidth Memory)。
    - 如果不进行优化，单张卡能容纳的最大参数量约为 40B（仅考虑参数、梯度和优化器状态，不含 Activations）。

---

## 2. Memory Accounting: Tensors (内存核算：张量)

### Tensor Data Types (数据类型)

- **Tensors** 是深度学习中存储 Parameters, Gradients, Optimizer States, Activations 的基本单元。
- **Float32 (FP32 / Single Precision)**：
    - 默认数据类型，占用 **4 bytes** (32 bits)。
    - 结构：1 bit 符号位，8 bits Exponent，23 bits Fraction。
    - 通常用于 Parameters 和 Optimizer States 以保证数值稳定性。
- **Float16 (FP16 / Half Precision)**：
    - 占用 **2 bytes** (16 bits)。
    - **缺点**：Dynamic range（动态范围）较小，容易导致 Underflow（下溢）。
- **BFloat16 (BF16 / Brain Float)**：
    - 占用 **2 bytes**，但在深度学习中通常优于 FP16。
    - **优势**：拥有与 Float32 相同的 Exponent 位数（动态范围大），牺牲了 Fraction 精度，但这对 Deep Learning 影响较小。
    - 通常用于 Matrix Multiplications 等计算过程。
- **FP8**：H100 支持的更低精度格式，用于进一步提升速度。

### Memory Calculation Example (内存计算示例)

- 一个 $4 \times 8$ 的 Tensor (Float32) 占用内存： $$ \text{Memory} = \text{Num Elements} \times \text{Size of Element} = 32 \times 4 \text{ bytes} = 128 \text{ bytes} $$。

---

## 3. PyTorch Internals & Tensor Views (PyTorch 内部机制与视图)

- **Storage vs. View**：
    - Tensor 本质上是指向内存中连续数组的 Metadata（元数据）指针。
    - Metadata 包含 Size 和 **Stride**（步长）。
    - **Stride** 决定了在某一维度上移动索引时，在内存中需要跳过多少个元素。
- **Zero-Copy Operations**：
    - 操作如 `transpose`, `view`, 切片通常不复制数据，而是创建新的 View（共享底层 Storage）。
    - **Mutation Risk**：修改一个 View 会影响原始 Tensor。
- **Contiguous**：
    - 某些操作（如 `transpose`）会导致 Tensor 在内存中不再连续 (Non-contiguous)。
    - 调用 `.contiguous()` 会强制复制数据并在内存中重新排列，这会消耗额外的 Memory。

---

## 4. Compute Accounting: Flops (计算核算：浮点运算)

### Definitions (定义)

- **Flops (Floating Point Operations)**：计算操作的总数（小写 's'）。
- **Flops/s (Floating Point Operations per Second)**：硬件的计算速度（通常用 '/s' 表示）。

### Cost of Operations (运算成本)

- **Matrix Multiplication (MatMul)** 是深度学习中最主要的计算消耗。
- **MatMul Flops Formula**：
    - 对于矩阵乘法 $(M \times K) \times (K \times N) \rightarrow (M \times N)$：
    - $\text{Flops} \approx 2 \times M \times N \times K$。
    - 系数 **2** 来源于每个输出元素包含一次乘法和一次加法。
- **Linear Model Forward Pass**：
    - $\text{Flops} \approx 2 \times \text{Number of Tokens} (B) \times \text{Number of Parameters} (P)$。

### Model Flops Utilization (MFU)

- **Definition**： $$ \text{MFU} = \frac{\text{Actual Flops} / \text{Training Time}}{\text{Promised Peak Flops/s}} $$。
- **Benchmark**：
    - MFU 反映了硬件利用率，通常  $\approx 4 \times \text{Number of Tokens} \times \text{Number of Parameters}$。
    - 系数 **4** 的来源：需要计算对 Weights 的梯度以及对 Inputs (Activations) 的梯度（以便传给上一层），每部分约为 Forward Pass 的 2 倍。
- **Total Training Flops**：
    - 总计算量 = Forward ($2P$) + Backward ($4P$)。
    - Rule of Thumb: **$6 \times \text{Number of Tokens} \times \text{Number of Parameters}$**。

---

## 6. Optimization & Total Memory Footprint (优化与总内存占用)

### Tensor Operations with `einsum`

- 使用 `torch.einsum` 或 `einops` 库代替复杂的索引操作（如 `x.transpose(-1, -2)`），提高代码可读性和安全性。
- 示例：`einsum('b s h, b s h -> b s', x, y)` 直观表达了维度操作。

### Optimizer Implementation

- **Optimizers** (如 SGD, Adam, Adagrad) 负责更新参数。
- 实现自定义 Optimizer 时，需要继承 `Optimizer` 类并管理 `state`。

### Total Memory Components (总内存组成)

训练一个模型所需的总显存包括以下部分：

1. **Parameters** (Weights)：$P$ 个元素。
2. **Gradients**：与 Parameters 形状相同，$P$ 个元素。
3. **Optimizer States**：
    - 取决于优化器。例如 Adagrad 需要存储梯度平方和，额外需要 $P$ 个元素。
    - Adam 通常需要存储一阶和二阶动量，需要 $2P$ 个元素。
4. **Activations**：
    - 用于反向传播计算梯度，大小为 $B \times T \times D \times \text{Layers}$。
    - 可通过 **Activation Checkpointing** (重计算) 来减少显存占用，以时间换空间。

### Mixed Precision Training (混合精度训练)

- **Strategy**：
    - **Parameters & Optimizer States**：保持 **Float32** 以确保累积更新的精度。
    - **Forward/Backward Computation** (MatMul)：使用 **BF16** 或 **FP8** 提高速度并减少内存。
- PyTorch 提供自动混合精度工具 (AMP) 来管理这种转换。

---

## 7. Best Practices (最佳实践)

- **Initialization**：
    - 使用 Xavier/Kaiming Initialization (例如除以  $\sqrt{\text{fan}_{\text{in}}}$) 防止 Activation 值在深层网络中爆炸或消失。
- **Randomness**：
    - 固定 Random Seed 以确保 Debug 时的可复现性。
- **Checkpointing**：
    - 定期保存 Model, Optimizer State 和 Iteration number，防止训练崩溃导致进度丢失。

---

### Reference Formula Summary (参考公式汇总)

- **Training Flops**: $C \approx 6 P D$ ($P$=Params, $D$=Dataset size in tokens).
- **Forward Flops**: $2 P D$.
- **Backward Flops**: $4 P D$.
- **Matrix Multiply Flops**: $2 M N K$.
- **Memory (Bytes)**: $\text{Num Elements} \times \text{Bytes/Type}$ (FP32=4, BF16=2).