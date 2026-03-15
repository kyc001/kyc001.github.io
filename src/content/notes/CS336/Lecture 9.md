---
title: 'Scaling Laws (Part 1) - 基础与数据/模型扩展'
category: 'CS336'
order: 9
description: ""
tags: []
---
# Scaling Laws (Part 1) - 基础与数据/模型扩展

**来源：Stanford CS336 Lecture 9**

---

## 1. 引言与动机 (Introduction & Motivation)

### 为什么需要 Scaling Laws？

- **场景假设**：如果你有 100,000 张 H100 显卡和一个月的限制，如何构建最佳的 _Open Source LM_？
- **传统方法的局限**：
    - 简单模仿现有模型（如 Llama）无法推动前沿创新。
    - 直接在大模型上进行超参数调整（Hyperparameter tuning）成本过高，不可行。
- **Scaling Laws 的核心理念**：
    - 通过训练一系列 _Small Models_ 来预测 _Big Models_ 的行为。
    - 建立简单的预测定律（Predictive laws），通过从小规模实验中学习，通过外推法（Extrapolation）在极大规模上一次性成功。

---

## 2. 历史背景与理论基础 (History & Context)

### 统计机器学习的视角 (Statistical Machine Learning)

- Scaling Laws 实际上是 _Statistical Machine Learning_ 理论的经验延伸。
- **理论界限 (Theoretical Bounds)**：
    - _Generalization bound_：误差随样本量 $n$ 的衰减通常为 $O(1/\sqrt{n})$。
    - _Nonparametric rates_：对于灵活的非参数类，误差衰减可能为 $n^{-\frac{\beta}{2\beta+1}}$。
- **Scaling Laws 的本质**：从理论上的 _Upper Bounds_ 跨越到对实际 _Loss Values_ 的经验拟合。

### 早期关键研究

- **Bell Labs (1993)**：早期提出在训练完整模型前预测性能的方法，形式类似于现代 Scaling Laws ($Error \approx C_0 + C_1 n^{-\alpha}$)。,
- **Banko & Brill (NLP)**：展示了数据规模扩大带来的性能提升遵循 _Log-linear_ 关系。,
- **Hestness et al. (2017)**：
    - 展示了从机器翻译到语音识别等任务的 _Power Law_ 误差衰减。
    - **三个区域**：1. Random Guessing（随机猜测区）；2. Power Law Region（幂律区）；3. Irreducible Error（不可约误差区）。,

---

## 3. 数据扩展定律 (Data Scaling Laws)

### 经验观察 (Empirical Observation)

- **定义**：将 _Dataset Size ($n$)_ 映射到 _Excess Error_。
- **现象**：在 _Log-Log Plot_ 上，模型性能（Test Loss）与数据量呈现线性关系，即 _Power Law_。
    - 公式形式：$L(n) \propto n^{-\alpha}$。

### 为什么是多项式衰减？(Why Polynomial Decay?)

通过两个数学示例解释其自然性：

1. **均值估计 (Mean Estimation)**：
    - 任务：估计高斯分布的均值。
    - 误差：$Error \propto \sigma^2/n$。
    - 在 _Log-Log_ 图上，斜率为 -1 ($-\log n$)。,
2. **非参数回归 (Nonparametric Regression)**：
    - 任务：拟合任意平滑函数 $f$。
    - 误差：$Error \propto n^{-1/d}$，其中 $d$ 是维度。
    - 这意味着对于高维数据或灵活函数类，学习率受 _Intrinsic Dimensionality_ 限制。

### 实际观察到的斜率 (Exponents)

- 理论期望可能是 $1/n$ 或 $1/\sqrt{n}$，但在实际中：
    - Machine Translation: $\alpha \approx 0.13$
    - Language Modeling: $\alpha \approx 0.095$
- 这表明任务的 _Intrinsic Dimensionality_ 很高。

### 数据扩展的应用

- **数据混合 (Data Mixture)**：数据质量通常只影响 _Offset_（截距），不影响 _Slope_（斜率）。可以在小模型上进行数据选择实验。
- **多轮训练 (Multi-epoch)**：存在收益递减。约 4 个 epoch 后收益迅速下降，可视为 _Effective Sample Size_ 的减少。

---

## 4. 模型扩展与超参数 (Model Scaling & Hyperparameters)

### 架构选择 (Architecture)

- **Transformer vs. LSTM**：在 _Scaling Law_ 图上表现为常数因子的差距（Constant factor gap）。LSTM 在任何规模下计算效率都低于 Transformer。
- **其他架构**：大多数架构无法超越 Transformer，唯有 _Mixture of Experts (MoE)_ 和 _Gated Linear Units (GLU)_ 显示出优势。
- **参数计算**：分析时应排除 _Embedding Parameters_，仅计算 _Non-embedding parameters_，否则曲线会弯曲。

### 优化器与超参数 (Optimizer & Hyperparameters)

- **Adam vs. SGD**：Adam 表现出常数级优势。
- **宽深比 (Aspect Ratio)**：存在一个很宽的最佳区域（Wide basin of optimality），在此范围内 _Width/Depth_ 的变化对 Loss 影响很小。,

### 批量大小与学习率 (Batch Size & Learning Rate)

1. **Critical Batch Size**：
    - 定义：从完美扩展（Perfect scaling）过渡到收益递减（Diminishing returns）的临界点。,
    - 规律：目标 _Loss_ 越低（模型越好），_Critical Batch Size_ 越大。
    - 这意味着随着训练进行，应增大 _Batch Size_。
2. **Learning Rate Scaling**：
    - 传统做法：$LR \propto 1/Width$。
    - 现代做法：**$\mu P$ (Maximal Update Parametrization)**。通过重新参数化，使最佳 _Learning Rate_ 在不同模型规模间保持稳定，无需重新调参。,

### 警告 (Caution)

- **Downstream Tasks**：Scaling Laws 对 _Log Perplexity_ (Cross Entropy) 预测极其准确，但对具体的下游任务（如 Accuracy）预测较差，可能出现非线性或突变。,

---

## 5. 联合扩展：计算、数据与模型 (Joint Scaling: Compute, Data, Model)

### 问题定义

- 在固定的 _Compute Budget (FLOPs)_ 下，应该分配多少给 _Model Size ($N$)_，多少给 _Dataset Size ($D$)_？,

### 早期理论 (Kaplan et al. / Rosenfeld)

- 提出联合误差公式：$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_{irr}$。
- Kaplan 的结论倾向于更大的模型和较少的数据，但这后来被证明是不准确的。,

### Chinchilla 分析 (Hoffmann et al.)

- **核心发现**：Kaplan 的分析因 _Learning Rate Schedule_（未能正确衰减）等原因产生偏差。
- **Chinchilla Scaling Laws**：
    - 最优比例：**$\approx 20$ Tokens per Parameter**。
    - 模型大小和数据大小的 Scaling Coefficients 均约为 0.5，即两者应同比例增加。

#### Chinchilla 的三种估算方法-

1. **Envelope Method (Min over curves)**：拟合不同模型训练曲线的下包络线（Lower envelope）。
2. **Isoflop Analysis**（最标准方法）：固定 _FLOPs_，训练不同大小的模型，找到每个 FLOPs 等级下的 Loss 最低点，拟合抛物线。
3. **Parametric Fitting**：直接拟合联合误差公式。（注：Epoch AI 复现发现原作者拟合有误，修正后结果与其他方法一致）。

---

## 6. 推理成本与现代趋势 (Inference Costs & Modern Trends)

### 训练最优 vs. 推理最优 (Training-Optimal vs. Inference-Optimal)

- **Chinchilla** 关注的是 _Training Compute Optimal_。
- **实际产品需求**：推理成本（Inference Cost）通常远高于训练成本。
- **Over-training**：为了降低推理成本，现代模型（如 Llama 3）倾向于使用远超 Chinchilla 比例的数据量（如 30T tokens），以此换取更小的模型尺寸。,

### 普适性

- Scaling Laws 不仅适用于 LLM，也适用于 _Diffusion Models_ 等其他生成式模型。,

---

## 7. 总结 (Conclusion)

- **Log-Linearity** 是深度学习规模化的核心特征。
- **工程价值**：
    - 通过小规模实验预测大规模行为。
    - 在训练前通过 _Isoflop Analysis_ 确定最佳的数据/模型比例。
    - 指导 _Batch Size_、_Learning Rate_ 和架构决策。

---