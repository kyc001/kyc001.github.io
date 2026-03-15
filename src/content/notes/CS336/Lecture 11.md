---
title: 'Scaling Laws 2 (扩展定律进阶与实战)'
category: 'CS336'
order: 11
description: ""
tags: []
---
# Scaling Laws 2 (扩展定律进阶与实战)

## 1. 课程概览 (Overview)

**核心目标**：探讨扩展大型语言模型（LLM）的最佳实践。

- **回顾**：Chinchilla Scaling Laws 及其局限性。
- **今日重点**：
    1. **案例研究 (Case Studies)**：现代模型（Cerebras-GPT, MiniCPM, DeepSeek）如何应用扩展定律进行模型设计和超参数选择。
    2. **$\mu P$ (Maximal Update Parameterization)**：一种使超参数在模型扩展过程中保持稳定的参数化方法，包含数学推导与实证分析。

---

## 2. 案例研究：实战中的扩展策略 (Scaling in the Wild)

### 2.1 Cerebras-GPT

- **策略**：严格遵循 Chinchilla recipe（计算量与参数量的最优权衡）。
- **核心发现**：
    - 使用 **$\mu P$** 使得扩展过程更加稳定。
    - **超参数传递**：在小规模模型（如 40M 参数）上进行广泛的 **Hyperparameter search**，然后通过 $\mu P$ 直接扩展到大模型，避免在大模型上调参。
- **结果**：相比标准参数化（Standard Parameterization, SP），$\mu P$ 的 Scaling 曲线更平滑，且性能优于 Pythia 或 GPT-J。

### 2.2 MiniCPM

- **目标**：利用高算力训练极强的小型模型。
- **创新点：WSD Learning Rate Schedule**
    - **问题**：传统的 Cosine Decay 使得一次训练只能对应一个固定的数据量目标，无法高效进行 Chinchilla 分析（即无法通过 checkpoints 模拟不同数据量的训练结束状态）。
    - **解决方案 (WSD)**：**Warmup - Stable - Decay**。
        - **Warmup**：预热。
        - **Stable**：Constant learning rate 阶段，可长期训练。
        - **Decay**：快速退火（Cool down）至零或最小值。
    - **优势**：只需一次训练（在 Stable 阶段），通过“回溯（Rewind）”并在不同步数进行 Decay，即可模拟不同数据量的训练结果。大大降低了拟合 Scaling Laws 的成本。
- **数据配比**：MiniCPM 发现了极高的 Token-to-Parameter ratio (192:1)，远超 Chinchilla 的 20:1。

### 2.3 DeepSeek LLM (V1)

- **策略**：不依赖 $\mu P$，而是采用“暴力”但严谨的 Scaling 分析。
- **方法**：
    - 在小规模模型上对 **Batch Size** 和 **Learning Rate** 进行网格搜索（Grid Search）。
    - 拟合这些超参数随计算量（Flops）变化的 Scaling Law，外推至大模型。
    - **Isoflops Analysis**：重新验证 Chinchilla Scaling，计算最优 Token/Parameter 比例。
- **结论**：即使不使用 $\mu P$，通过拟合 Optimal Batch Size 和 Learning Rate 的 Scaling Law 也能准确预测大模型性能。

### 2.4 其他近期模型 (Modern Trends)

- **Llama 3**：数据量极大，Token/Parameter ratio 约为 39:1，验证了高质量数据和训练效率允许超越 Chinchilla 的 20倍限制。尝试建立 **Compute** $\to$ **NLL** $\to$ **Downstream Accuracy** 的映射。
- **Hunuan**：报告了 96:1 的数据配比。
- **Minimax 01**：利用 Scaling Laws 验证线性注意力（Linear Attention）与 Softmax Attention 在扩展性上的一致性。

---

## 3. $\mu P$ (Maximal Update Parameterization) 深度解析

### 3.1 动机 (Motivation)

- **问题**：在标准参数化（Standard Parameterization, SP）下，随着模型 **Width** ($n$) 增加，Optimal **Learning Rate** ($\eta$) 会发生漂移（通常变小）。这意味着每次扩展都需要重新调参。
- **目标**：寻找一种参数化方法，使得 Optimal Hyperparameters 在模型扩展时保持不变（Scale Invariant）。

### 3.2 核心数学假设 (Spectral Conditions)

为了推导 $\mu P$，我们基于深度线性网络（Deep Linear Networks）引入两个稳定性条件：

1. **Condition A1 (Activation Stability)**: 随着 **Width** 增加，每一层的 **Activations** ($h_l$) 在初始化时应保持 $O(1)$，即不爆炸也不消失。
    
    - Vector norm: $|h_l|_2 \sim \Theta(\sqrt{n_l})$.
2. **Condition A2 (Update Stability)**: 经过一次梯度更新（One gradient step）后，**Update** 的大小 ($\Delta h_l$) 也应保持 $O(1)$。
    
    - 即：$\Delta W_l$ 导致的 Activation 变化量应与初始 Activation 同数量级。

### 3.3 数学推导 (Derivation Summary)

#### 初始化 (Initialization)

考虑矩阵乘法 $h_l = W_l h_{l-1}$。若 $W_l$ 为高斯随机矩阵，其算子范数（Operator Norm）集中于 $\sigma \sqrt{n_l + n_{l-1}}$。 为了满足 **Condition A1** ($h_l \sim O(1)$)，我们需要： $$
\text{Init Variance} \propto \frac{1}{\text{fan}_{\text{in}}}
$$
 即标准差设置为 $\frac{1}{\sqrt{n_{l-1}}}$。这与标准的 He/Xavier 初始化一致。

#### 学习率 (Learning Rate)

考虑 SGD 更新：$\Delta W_l = -\eta \nabla_L W_l = -\eta \delta_l h_{l-1}^T$。 为了满足 **Condition A2**，我们需要推导 $\Delta W_l$ 的量级。 假设 Loss 的变化 $\Delta \mathcal{L} \sim \Theta(1)$。 经过推导可得，对于 SGD，最优学习率应满足： 
$$
\eta_{\text{SGD}} \propto \frac{\text{fan}_{\text{out}}}{\text{fan}_{\text{in}}}
$$

**关键区别**：对于 **Adam** 优化器（Transformer 常用），推导结果显示最优学习率应按以下方式缩放： $$
\eta_{\text{Adam}} \propto \frac{1}{\text{fan}_{\text{in}}}
$$
 通常简化为按模型宽度 $1/\text{width}$ 进行缩放。

### 3.4 $\mu P$ 实施总结 (Implementation Recipe)

对比标准参数化 (SP) 与 $\mu P$ (针对 Transformer + Adam)：

| 参数/层 (Parameter/Layer) | 标准参数化 (SP)                              | $\mu P$ (Maximal Update)                           |
| :--------------------- | :-------------------------------------- | :------------------------------------------------- |
| **Embedding**          | Init: $\sim 1$                          | Init: $\sim 1$ (不缩放)                               |
| **Hidden Weights**     | Init: $1/\sqrt{\text{fan}_{\text{in}}}$ | Init: $1/\sqrt{\text{fan}_{\text{in}}}$            |
| **Learning Rate**      | Constant (Global)                       | Scaled by $1/\text{width}$                         |
| **Output Layer**       | Init: $1/\sqrt{\text{fan}_{\text{in}}}$ | Init: $1/\text{fan}_{\text{in}}$ (通常更小以防止logits爆炸) |
|                        |                                         |                                                    |

_注意：实际操作中，通常将 Embedding 和 Readout 层的 multipliers 单独处理，核心中间层的 Learning Rate 随 $1/\text{width}$ 缩放。_

---

## 4. 实证分析：$\mu$-Transfer (Empirical Results)

基于论文 "A Large Scale Exploration of $\mu$-Transfer"。

### 4.1 验证 $\mu P$ 的有效性

- **实验设计**：仅扩展 **Width**，保持 Depth 固定。
- **结果**：在 $\mu P$ 下，从小模型（width=128）到大模型（width=2048），**Optimal Learning Rate** 保持一致。而在 SP 下，LR 必须随规模减小。

### 4.2 鲁棒性测试 (Ablation Studies)

$\mu P$ 在以下情况**有效**（Robust）：

- 改变非线性激活函数 (e.g., ReLU $\to$ SwiGLU)。
- 改变 **Batch Size**。
- 改变初始化策略（如 Zero Query Init）。

$\mu P$ 在以下情况**失效**（Not Robust）：

- 引入可学习的增益（Learnable Gains/Biases），需移除 Biases。
- 使用非标准优化器（如 Lion，因为 $\mu P$ 是基于 Adam/SGD 推导的）。
- 大幅改变 **Weight Decay** 策略。

---

## 5. 总结 (Summary)

- **扩展的一致性**：尽管各家实验室（DeepSeek, MiniCPM, Llama）的具体做法不同，但核心都在于寻找可预测的 Scaling 规律。
- **工具箱**：
    1. **Chinchilla (Isoflops)**：确定数据与模型大小的配比。
    2. **WSD Schedule**：低成本进行多数据量模拟。
    3. **$\mu P$**：稳定超参数，实现从 Proxy Model 到 Target Model 的 Zero-shot Hyperparameter Transfer。
- **结论**：通过正确的参数化和 Scaling Laws，可以在小规模实验上验证设计选择，从而大幅降低大模型训练的风险和成本。

---
