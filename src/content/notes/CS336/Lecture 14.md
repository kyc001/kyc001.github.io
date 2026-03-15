---
title: 'CS336 Language Modeling from Scratch'
category: 'CS336'
order: 14
description: ""
tags: []
---
# CS336 Language Modeling from Scratch

## Lecture 14: Data 2 - Quality Filtering & Deduplication (数据筛选与去重)

---

### 1. 引言与背景 (Introduction)

- **回顾 (Recap):** 上节课讨论了从 BERT 到 Llama 等模型的训练数据集来源。
- **核心观点:** 数据并非凭空产生，需要经过复杂的处理流程：抓取 (Crawling) $\rightarrow$ 提取文本 $\rightarrow$ 质量筛选 (Quality Filtering) $\rightarrow$ 去重 (Deduplication)。
- **本课重点:** 深入探讨**Quality Filtering**（质量筛选）和**Deduplication**（去重）的机制与算法。

---

### 2. 过滤算法 (Filtering Algorithms)

#### 2.1 问题定义 (Problem Formulation)

- **输入:**
    - 目标数据集 (Target Data, $T$): 高质量但数量少。
    - 原始数据集 (Raw Data, $R$): 数量巨大但质量参差不齐（如 Common Crawl）。
- **目标:** 从 $R$ 中找到一个子集 $T'$，使其与 $T$ 相似。
- **要求:** 必须具备极高的计算效率 (Extremely fast)，因为需要在海量 Web 数据上运行。

#### 2.2 方法一：N-gram 语言模型 ($N$-gram Language Models)

- **原理:**
    - 基于统计语言处理时代的经典方法。
    - 使用 Maximum Likelihood Estimation (MLE) 统计 $n$-gram 计数并进行归一化。
    - 使用 Smoothing 技术（如 Kneser-Ney Smoothing）处理稀疏性问题（Unseen grams）。
- **实现:**
    - 常用工具: **KenLM** (高效、开源)。
    - **Metric:** 计算文档的 Perplexity。
    - **Perplexity** 越低，表示该文档与高质量数据的分布越接近。
- **案例:** **CCNet** (Facebook) 对段落按 Perplexity 排序，保留 Top 1/3 用于训练 Llama 早期版本。
- **局限性:** 仅关注局部上下文 (Local context)，容易被无意义但语法通顺的文本（如重复短语）欺骗。

#### 2.3 方法二：线性分类器 (Linear Classifiers)

- **工具:** **FastText** (Facebook, 2016)。
- **动机:** 传统的 Bag-of-words 分类器在词汇量 $V$ 和类别 $K$ 很大时参数过多。
- **机制:**
    - 将词汇空间映射到较小的 Hidden Dimension (降维)。
    - 使用 Hashing 处理 $n$-grams 以解决 unbounded vocabulary 问题。
    - 结构：Linear classifier，无非线性激活层，计算极其高效。
- **权衡:** 虽然可以使用 BERT 或 Llama 进行筛选，但在海量数据上计算成本过高 (Compute Cost)，简单的线性模型是更好的折衷。

#### 2.4 方法三：重要性重采样 (Importance Resampling)

- **原理:** Data Selection for Language Models (DSIR)。
- **数学基础:**
    - 目标分布 $P$ (Target)，提议分布 $Q$ (Proposal/Raw)。
    - 计算 Importance Weights: $w = P(x) / Q(x)$。
- **实现:**
    - 对 $T$ 和 $R$ 分别训练简单的生成模型（如 Hashed $n$-gram models）。
    - 计算两个分布的 Likelihood Ratio 作为评分。
    - 根据 Importance Weights 进行重采样。
- **优势:** 相比二分类器，更注重匹配目标分布 (Matching the distribution)，理论上能提供更好的多样性。

---

### 3. 筛选的应用 (Applications of Filtering)

#### 3.1 语言识别 (Language Identification)

- **挑战:** 即使是大模型，如果训练数据中目标语言占比过低（如 Bloom 仅 30% English），性能也会受限。
- **方法:** 使用预训练的 **FastText** 分类器。
- **Dolma 数据集策略:** 保留 $P(\text{English}) > 0.5$ 的页面。
- **难点:** 短文本、低资源语言 (Low-resource languages) 和代码切换 (Code-switching) 容易识别错误。

#### 3.2 质量筛选 (Quality Filtering)

- **GPT-3:** 训练线性分类器。正样本 = 高质量源，负样本 = Common Crawl。
- **Llama 1:** 正样本 = 被 Wikipedia 引用的页面。
- **Phi-1 (Textbooks Are All You Need):**
    - 目标：创建类似教科书的高质量代码数据。
    - 方法：使用 **GPT-4** 标注 100k 样本的 "Educational Value"。
    - 利用这些标注训练一个 Random Forest Classifier，然后对大规模数据集 $R$ 进行筛选。
    - **结果:** 仅用极少数据量即在 HumanEval 上取得优异成绩。

#### 3.3 毒性筛选 (Toxicity Filtering)

- **数据集:** Jigsaw Toxic Comments (源自 Wikipedia talk pages)。
- **方法:** 训练 FastText 分类器识别 "Hate" 和 "NSFW" 内容。
- **Dolma 实践:** 使用分类器过滤有害内容。

---

### 4. 去重 (Deduplication)

#### 4.1 概述

- **类型:**
    - **Exact Duplicates:** 完全一致（源于网站镜像 Mirroring）。
    - **Near Duplicates:** 近似一致（源于模板、许可证、细微修改）。
- **为什么要去重?**
    - 提高训练效率 (Efficiency)。
    - 减少 **Memorization** (死记硬背)，降低版权和隐私风险。
- **设计空间:** Unit (句子/文档), Match (精确/近似), Action (移除所有/保留一个)。

#### 4.2 精确去重 (Exact Deduplication)

- **基础工具:** Hash Functions (e.g., MurmurHash)。
- **方法 A: MapReduce / Set:**
    - 计算所有文档的 Hash，保留唯一的 Hash 对应的文档。
    - **C4 数据集:** 对三个句子的 span 进行去重。
- **方法 B: Bloom Filters (布隆过滤器)**
    - **特点:** 极其节省内存 (Memory Efficient)，不允许删除，存在 False Positives (假阳性)。
    - **算法:**
        - 定义一个长度为 $m$ 的 Bit Array。
        - 使用 $k$ 个 Hash 函数。对于每个元素，将 $k$ 个位置置为 1。
        - 查询时：如果 $k$ 个位置全为 1，则认为元素存在。
    - **False Positive Rate ($f$):**
        - 当插入 $n$ 个元素后，特定位为 0 的概率是 $(1 - 1/m)^{kn} \approx e^{-kn/m}$。
        - $f \approx (1 - e^{-kn/m})^k$。
    - **Optimal $k$:** 当 $k = \ln(2) \cdot (m/n)$ 时，$f$ 最小。

#### 4.3 近似去重 (Near Deduplication)

- **相似度度量:** **Jaccard Similarity**
    - $J(A, B) = \frac{|A \cap B|}{|A \cup B|}$。
    - 目标：找到 $J(A, B) > \text{threshold}$ 的文档对。
- **算法核心:** **MinHash**
    - 将集合 $A$ 中的元素 Hash 后取最小值 $h_{min}(A)$。
    - **性质:** $P(h_{min}(A) = h_{min}(B)) = J(A, B)$。
    - 通过比较 MinHash 值来估计 Jaccard Similarity。
- **加速查找: Locality Sensitive Hashing (LSH)**
    - **Banding Technique:**
        - 计算 $N$ 个 MinHash 值（签名）。
        - 将签名分为 $b$ 个 Band，每个 Band 包含 $r$ 个 Row ($N = b \cdot r$)。
    - **碰撞规则:** 只要有一个 Band 中的 $r$ 个 Hash 值完全匹配，则认为 $A$ 和 $B$ 是 Candidate Pair。
    - **概率分析:**
        - 在一个 Band 中所有 $r$ 个值匹配的概率: $s^r$ (其中 $s$ 是 Jaccard Similarity)。
        - 在一个 Band 中不匹配的概率: $1 - s^r$。
        - 所有 $b$ 个 Band 都不匹配的概率: $(1 - s^r)^b$。
        - **最终碰撞概率 (Probability of Collision):** $1 - (1 - s^r)^b$。
    - **效果:** 形成一个 S 形曲线 (Sigmoid-like curve)，使得高于阈值的相似度碰撞概率极高，低于阈值的极低。

---

### 5. 总结 (Summary)

- **Filtering:** 使用 $n$-gram models, Linear classifiers, Importance Resampling 将原始数据清洗为目标分布的数据。
- **Deduplication:** 利用 Hashing 技术（Bloom Filter, MinHash + LSH）将成对比较 (Pair-wise) 问题转化为线性时间复杂度 (Linear time) 问题，有效提升数据质量和模型训练效率。