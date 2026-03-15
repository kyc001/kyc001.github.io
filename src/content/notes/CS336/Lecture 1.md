---
title: 'CS336: Language Models from Scratch (Lecture 1)'
category: 'CS336'
order: 1
description: ""
tags: []
---
# CS336: Language Models from Scratch (Lecture 1)

**课程概览与分词 (Overview and Tokenization)**

## 第一部分：课程介绍与核心理念 (Introduction & Philosophy)

### 1. 课程目标与动机

- **核心理念**：要真正理解它，你必须亲手构建它（"To understand it, you have to build it"）。
- **研究现状的危机**：
    - 研究人员正逐渐与底层技术脱节。几年前研究者还会自己训练模型，现在很多人仅停留在对私有模型进行 prompting。
    - 抽象层虽然方便，但也是有泄漏的（leaky abstractions），阻碍了需要拆解整个技术栈（stack）的基础研究。
- **“工业化”带来的挑战**：
    - 前沿模型（Frontier models）参数量巨大（如 GPT-4 传闻为 1.8 trillion parameters），训练成本极高（> $100 million），且细节不公开。
    - **小模型不能完全代表大模型**：
        - **Flops 分布差异**：小模型中 Attention 和 MLP 的计算量相当，但在大模型（如 175B）中，MLP 占据主导地位。
        - **涌现行为 (Emergent Behavior)**：某些能力（如 in-context learning）只有在 scale 达到一定程度后才会突然出现。

### 2. 课程将教授的三类知识

1. **机制 (Mechanics)**：事物如何运作（如 Transformer 的实现、GPU 并行）。
2. **思维模式 (Mindset)**：这是本课的重点。即如何尽可能压榨硬件性能，并严肃对待 Scaling。
3. **直觉 (Intuitions)**：哪些数据和架构决策能带来好模型。这部分很难完全教授，因为小规模上的直觉在 scale 变大后可能失效（例如某些 trick 被称为“divine benevolence”——无法解释的神恩）。

### 3. 关于“苦涩的教训” (The Bitter Lesson) 与效率

- **误解**：认为“苦涩的教训”意味着算法不重要，只有 Scale 重要。
- **正解**：在 Scale 下的算法（Algorithms at scale）才是关键。
    - 模型精度是效率（Efficiency）与资源投入的乘积。
    - 在大规模下，效率至关重要，因为无法承受浪费数亿美元的代价。
- **历史数据**：2012-2019年间，ImageNet 训练的算法效率提升了 44倍，超过了摩尔定律。
- **核心思维**：在给定的 compute 和 data budget 下，如何构建最好的模型？即最大化 efficiency。

### 4. 语言模型简史

- **早期**：Shannon (Entropy estimation), N-gram models (2007年 Google 已训练 2 trillion tokens 的 5-gram 模型)。
- **2010s 深度学习革命**：Neural Language Model (Bengio, 2003), Seq2Seq, Adam Optimizer, Attention Mechanism。
- **Transformer (2017)**：关键转折点，随后出现了 scaling laws 的探索。
- **现状**：从封闭模型（Closed models, e.g., GPT-4）到开放权重（Open weights）再到完全开源（Open source, 权重与数据均公开）。

---

## 第二部分：课程大纲与五大支柱 (The Five Pillars)

本课程围绕**效率 (Efficiency)** 展开，包含五个核心单元：

### Unit 1: 基础 (Basics)

- **目标**：构建一个完整的 pipeline。
- **内容**：
    - **Tokenizer**：实现 BPE (Byte Pair Encoding)。
    - **Architecture**：实现 Transformer。虽然骨架未变，但现代实现包含许多改进（如 SwiGLU, Rotary Positional Embeddings, RMSNorm, GQA/MLA）。
    - **Training**：实现 AdamW 优化器和训练循环。
- **作业**：从零实现上述内容，并在 OpenWebText 数据集上优化 perplexity。

### Unit 2: 系统 (Systems)

- **目标**：从硬件中获取最大性能。
- **内容**：
    - **Kernels**：学习 GPU 架构（HBM vs SRAM），使用 Triton 编写 kernel，通过 fusion 和 tiling 减少数据移动（Data movement）。
    - **Parallelism**：当单卡无法容纳时，使用 Data Parallelism 和 Model Parallelism (如 FSDP, Tensor Parallelism)。
    - **Inference**：包含 Prefill（并行，Compute-bound）和 Decode（自回归，Memory-bound）。技术包括 Speculative Decoding。

### Unit 3: 扩展定律 (Scaling Laws)

- **核心问题**：给定 Flops budget，最佳的模型大小（Model size）是多少？
- **Chinchilla Optimal**：通过小规模实验预测大规模下的最优超参数。
- **经验法则**：对于 $N$ 参数的模型，应训练约 $20N$ 的 tokens。
- **作业**：在有限的 Flops budget 下，设计实验并预测更大规模模型的 Loss。

### Unit 4: 数据 (Data)

- **核心观点**：数据决定模型做什么，数据不是天上掉下来的，必须主动获取和处理。
- **流程**：
    - **Curation**：Common Crawl 的原始数据包含大量垃圾（HTML, 广告），需要处理。
    - **Processing**：HTML 转 Text，Filtering（去重、去毒），De-duplication。
    - **Evaluation**：Perplexity, MMLU 等。

### Unit 5: 对齐 (Alignment)

- **目标**：将 Base Model（只会预测下一个 token）转化为有用的助手（遵循指令、风格、安全）。
- **技术**：
    - **SFT (Supervised Fine-Tuning)**：利用高质量的 Prompt-Response 对进行微调。
    - **Preference Learning**：利用人类偏好数据（A > B）或验证器（Verifiers）。算法包括 PPO, DPO, GRPO (DeepSeek)。

---

## 第三部分：技术深究 —— 分词 (Tokenization)

### 1. 定义与基本概念

- **Tokenization**：将原始文本（Strings）转换为整数序列（Sequences of Integers）的过程。
- **Vocabulary Size ($V$)**：Token 取值的范围。
- **Compression Ratio**：$\frac{\text{Number of Bytes}}{\text{Number of Tokens}}$。比率越高，表示每个 Token 包含的信息越多，效率越高。

### 2. 朴素方法的局限性

- **Character-based (基于字符)**：
    - 将每个 Unicode 字符映射为一个整数。
    - **缺点**：Compression ratio 接近 1（对于多字节字符甚至更低），导致序列过长，Attention 计算开销大。
    - **缺点**：词表利用率低（某些字符极罕见）。
- **Byte-based (基于字节)**：
    - 基于 UTF-8 编码的字节，Vocab size 为 256。
    - **缺点**：Compression ratio 为 1。对于 Transformer 这种 $O(L^2)$ 复杂度的模型，序列长度 $L$ 过长是灾难性的效率问题。
- **Word-based (基于单词)**：
    - 按空格或标点分割。
    - **缺点**：词表大小无限（Unbounded），对于新词必须使用 `UNK` (Unknown token)，处理稀有词非常低效。

### 3. Byte Pair Encoding (BPE)

- **背景**：源自 1994 年的数据压缩算法，后被用于机器翻译，GPT-2 开始用于大模型。
- **核心思想**：基于语料库统计数据，自适应地将最常见的**相邻字节对 (adjacent pairs of bytes)** 合并为一个新 Token。
- **算法流程**：
    1. 将字符串转换为字节序列。
    2. 统计所有相邻 Token 对的频率。
    3. 找到出现频率最高的一对（例如 `(116, 104)` 对应 `('t', 'h')`）。
    4. 在词表（Vocab）中新增一个 Token（如 `256`）代表该对。
    5. 将序列中所有的该对替换为新 Token。
    6. 重复上述步骤，直到达到预设的 Vocab size。
- **优势**：
    - 自适应：常见词（如 "the"）变为单个 Token，稀有词保留为字节序列。
    - 压缩率：GPT-2 Tokenizer 的 Compression ratio 约为 1.6 bytes/token。
    - 可逆：不需要 `UNK` token。

### 4. 实践中的注意事项

- **Pre-tokenization**：为了效率，通常先用正则（如 GPT-2 的正则）将文本切分为片段，再对每个片段运行 BPE，以阻断跨类别（如标点和字母）的合并。
- **空格处理**：通常空格被视为 Token 的一部分（例如前置空格），不同于传统 NLP。

---

## 课程后勤 (Logistics)

- **作业量**：非常重，第一个作业的工作量相当于普通课程（CS224N）五个作业的总和。
- **资源**：提供 H100 集群供作业跑分。
- **代码**：不提供框架代码（Scaffolding），给你一个空文件，从零开始写。
- **评分**：基于单元测试（Correctness）和 Leaderboard（Perplexity/Efficiency）。