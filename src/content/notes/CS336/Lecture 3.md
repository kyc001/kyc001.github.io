---
title: 'LLM 架构与超参数 (Architectures and Hyperparameters)'
category: 'CS336'
order: 3
description: ""
tags: []
---
# LLM 架构与超参数 (Architectures and Hyperparameters)

**来源：Stanford CS336 Lecture 3**

---

### Slide 1: 课程概览

- **本课主题**：深入探讨大语言模型（LLM）架构的演变、现代共识以及训练超参数的选择。
- **核心内容**：
    1. **Architecture Variations**：从原始 Transformer 到现代变体（如 Llama 架构）。
    2. **Hyperparameters**：如何选择维度、词表大小等参数。
    3. **Training Stability**：解决训练不稳定的技巧（如 Z-loss, QK Norm）。
    4. **Attention Variants**：针对推理效率的改进（MQA, GQA）。

---

### Slide 2: Transformer 架构演变：Normalization 位置

- **Original Transformer**：使用 **Post-norm**（在残差连接后进行归一化）。
    - 缺点：训练不稳定，需要 warmup。
- **Modern Consensus**：普遍使用 **Pre-norm**（在残差分支开始前进行归一化）。
    - 优点：训练更稳定，梯度传播更顺畅，允许移除 warmup,。
- **最新变体 (Double Norm)**：
    - 部分模型（如 Grok, Gemma 2）在残差块前后都添加 Layer Norm。
    - 目的：进一步提高大型模型的训练稳定性。

---

### Slide 3: Normalization 类型：Layer Norm vs. RMS Norm

- **Layer Norm**：
    - 公式：需要计算 **mean** 和 **variance**，并包含可学习参数 $\beta$ (bias) 和 $\gamma$ (scale)。
- **RMS Norm (Root Mean Square Normalization)**：
    - 公式：移除 **mean** 的减法计算，仅保留缩放，通常移除 $\beta$。
    - 优势：
        1. **Computational Efficiency**：减少算力消耗（虽然仅占 FLOPs 的极小部分）。
        2. **Memory Efficiency**：减少 **Memory Movement**，这在 Transformer 运行时间中占比很大（约 25%）。
    - 现状：Llama, PaLM, Gopher 等现代模型几乎全部采用 RMS Norm。

---

### Slide 4: 线性层与 Bias Terms

- **趋势**：现代 Transformer 架构大多移除了线性层（Linear Layers）和归一化层中的 **Bias Terms**。
- **原因**：
    1. **Optimization Stability**：经验表明移除 Bias 有助于训练稳定性。
    2. 性能无损：仅使用 **Matrix Multiplies** 即可达到同等效果。
- **例外**：Command R+ 等少数模型仍保留 Layer Norm。

---

### Slide 5: 激活函数 (Activations) 的演进

- **ReLU**：$\max(0, x)$。早期模型使用。
- **GeLU (Gaussian Error Linear Unit)**：$x \Phi(x)$。GPT-2/3 使用，比 ReLU 更平滑,。
- **Swish**：$x \cdot \text{sigmoid}(\beta x)$。
- **现代共识：Gated Linear Units (GLU)**
    - **SwiGLU**：目前最流行（Llama, PaLM 等使用）。
    - 机制：包含两个线性投影，一个用于门控（gate）。
    - 公式结构：$(x W_1) \odot \text{Act}(x V) W_2$,。
    - 证据：GLU 变体在困惑度（Perplexity）和下游任务上表现持续优于非门控版本。

---

### Slide 6: 模块结构：Serial vs. Parallel Layers

- **Serial (标准)**：
    - 流程：Input $\rightarrow$ Attention $\rightarrow$ Add $\rightarrow$ MLP $\rightarrow$ Add。
    - 大多数模型采用此方式。
- **Parallel (并行)**：
    - 流程：Input 同时进入 Attention 和 MLP，然后求和：$x + \text{Attn}(x) + \text{MLP}(x)$。
    - 代表模型：GPT-J, PaLM。
    - 优点：便于并行计算，系统效率可能更高,。
    - 缺点：可能牺牲一定的表达能力（Expressiveness）。

---

### Slide 7: 位置编码 (Position Embeddings)

- **演变路径**：Sinusoidal $\rightarrow$ Absolute Learned $\rightarrow$ Relative $\rightarrow$ **RoPE**。
- **Rotary Position Embeddings (RoPE)**：
    - 核心思想：通过旋转向量来编码相对位置。
    - 数学性质：寻找变换 $f$，使得 inner product $\langle f(x, i), f(y, j) \rangle = g(x, y, j-i)$ 仅依赖于相对距离。
    - 实现：将向量分组为 2D 对，每一对根据位置 $m$ 旋转角度 $m\theta$。
    - 应用位置：在每一层的 Attention 计算 Key 和 Query 时应用，而非在底层叠加,。
    - 现状：几乎所有 2023 年后的模型（Llama, Mistral 等）都使用 RoPE。

---

### Slide 8: 超参数：MLP 维度 ($d_{ff}$)

- **标准设置 (Non-gated)**：
    - $d_{ff} = 4 \times d_{model}$。
- **GLU 设置 (SwiGLU 等)**：
    - 由于 GLU 引入了额外的门控参数矩阵，为了保持参数总量不变，需缩小维度。
    - 经验公式：$d_{ff} \approx \frac{2}{3} \times 4 \times d_{model} = \frac{8}{3} d_{model}$。
    - 常见比例：约 2.6x 到 2.7x（如 Llama 遵循此比例）。
- **例外**：T5 使用了极大的 $d_{ff}$ (64x)，但后续版本回归了标准比例。

---

### Slide 9: 超参数：Attention Heads 与宽高比

- **Head Dimension**：
    - 共识：$d_{head} \times n_{heads} = d_{model}$。
    - 即随着 Head 数量增加，每个 Head 的维度相应减小，总维度保持不变。
- **Aspect Ratio (Depth vs. Width)**：
    - $N_{layers}$ vs $d_{model}$。
    - 经验法则：存在一个较宽的最优区间。Kaplan 的 Scaling Law 显示宽高比在大范围内对 Loss 影响不大。
- **Vocab Size (词表大小)**：
    - 趋势：从早期的 30k-50k 增加到 100k-250k（如 Llama 3, GPT-4）。
    - 原因：多语言支持，提高压缩率，降低推理成本,。

---

### Slide 10: 正则化 (Regularization)

- **Dropout**：
    - 现状：在预训练中已很少使用（通常只训练 1 个 Epoch，不过拟合）。
- **Weight Decay**：
    - 现状：仍然广泛使用。
    - **反直觉现象**：Weight Decay 在预训练中并非为了防止过拟合（Validation Loss 与 Training Loss 差距未变），而是为了改善 Training Dynamics。
    - 作用机制：Weight Decay 与 Learning Rate Schedule（如 Cosine Decay）相互作用，在训练末期能够加速优化，获得更低的 Training Loss,。

---

### Slide 11: 训练稳定性技巧 (Stability Tricks)

- **问题**：大模型训练中常见的 Loss Spikes 和 Gradient Norm 爆炸。
- **解决方案 1: Z-Loss**
    - 针对：Output Softmax 的 Logits 漂移。
    - 方法：惩罚 $\log(Z)$（Partition function），使其接近 0。
    - 公式：$L_{aux} = \alpha \log^2(Z)$。
    - 效果：使得 Softmax 更加数值稳定,。
- **解决方案 2: QK Norm**
    - 针对：Attention 内部的 Softmax。
    - 方法：在计算 Attention Score 之前，对 Query 和 Key 进行 Layer Norm。
    - 应用：ViT, Gemma 2 等模型使用，有效控制 Logits 幅度。
- **解决方案 3: Logit Soft-capping**
    - 方法：使用 $\tanh$ 限制 Logits 的最大值。

---

### Slide 12: 推理效率与 Attention 变体

- **瓶颈**：推理时的 **Memory Bandwidth**（内存带宽）。
    - **KV Cache**：自回归生成时需要存储 Key 和 Value。
    - **Arithmetic Intensity**：标准 Attention 在推理时算术强度极低（$O(1)$ ratio of flops to memory bytes）,。
- **Multi-Query Attention (MQA)**：
    - 方法：所有 Query Heads 共享同一组 Key 和 Value Head。
    - 效果：大幅减少 KV Cache 的显存占用和搬运量，提升推理吞吐量。
- **Grouped-Query Attention (GQA)**：
    - 方法：MQA 与 MHA 的折中，将 Query 分组，每组共享 KV。
    - 现状：Llama 2/3 等模型采用，平衡了性能与效果。

---

### Slide 13: 现代架构前沿：超长上下文 (Long Context)

- **混合 Attention 模式**：
    - 例如 Llama 4 或 Command A 的设计。
    - **Full Self-Attention**：每隔几层使用一次，不加位置编码（No RoPE）。
    - **Sliding Window Attention**：其余层使用，配合 RoPE。
- **优势**：结合了局部高精度（带位置信息）和全局信息流（无位置偏差），利于外推到超长序列（如 10M tokens）。