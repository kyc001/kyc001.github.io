---
title: 'CS336 Language Modeling from Scratch | Lecture 13: Data 1'
category: 'CS336'
order: 13
description: ""
tags: []
---
# CS336 Language Modeling from Scratch | Lecture 13: Data 1

## 1. 引言：数据的重要性 (Introduction)

- **核心观点 (Hot Take):** 在构建 _Language Models_ 时，数据是比 _Architecture_ 更关键的因素。
- **行业现状:**
    - 尽管 _Scaling Laws_ 很重要，但在公开的论文（如 Llama 3, DeepSeek）中，公司倾向于公开 _Architecture_，但对数据细节严格保密。
    - 保密原因包括：**竞争动态 (Competitive dynamics)** 和 **法律风险 (Legal liability)**。
- **数据的特性:**
    - 数据处理是高度可并行化的 (_highly parallelizable_)，相比于 _Architecture_ 设计，更容易通过扩大团队规模来扩展。

## 2. 训练阶段概览 (Training Stages)

现代 LLM 开发通常分为三个阶段，但界限日益模糊：

1. **Pre-training (预训练):**
    - 使用海量的 **Raw Data**（通常来自 Web）。
    - 产出：**Base Model**。
2. **Mid-training (中期训练):**
    - 使用较小规模、高质量的数据集。
    - 目标：增强特定能力（如 _Math_, _Code_, _Long Context_）。
3. **Post-training (后训练):**
    - 包括 _Fine-tuning_ on instruction/chat data 和 _Reinforcement Learning_ (RLHF)。
    - 目标：指令遵循、对话能力、安全性。
    - 产出：**Instruct Model**。

## 3. 预训练数据源演变 (Evolution of Pre-training Data)

### 3.1 早期模型 (2018-2019)

- **BERT (2018):**
    - 数据源：**BooksCorpus** (7k books) + **Wikipedia**。
    - 意义：从基于 _Sentences_ 的训练转向基于 _Documents_ 的训练。
- **GPT-2 (2019):**
    - **WebText:** 基于 Reddit 的出站链接，筛选标准为 _Karma_ > 3。
    - 目标：从低质量的 Web 中快速获取多样化的高质量子集。

### 3.2 Common Crawl与清洗策略

- **Common Crawl (CC):**
    - 互联网的学术近似版本，自2007年起每月抓取。
    - 格式：**WARC** (Raw HTTP response/HTML) vs. **WET** (Text only)。HTML 转 Text 是一个 _lossy process_，工具选择（如 _trafilatura_ vs _justtext_）会显著影响数据质量。
- **清洗策略 (Filtering Strategies):**
    1. **CCNet (Meta):**
        - 使用 **Model-based filtering**。
        - 在 Wikipedia 上训练一个 _5-gram model_ 作为质量分类器 (_Quality Classifier_) 来筛选网页。
    2. **C4 (Google/T5):**
        - 使用 **Heuristics** (启发式规则)。
        - 规则包括：移除包含脏话的页面、移除代码（"{"）、保留以标点结尾的行。

### 3.3 GPT-3 与 The Pile (2020)

- **GPT-3 数据混合:**
    - Common Crawl, WebText2, Books1, Books2, Wikipedia。
    - 使用 _Quality Classifier_（逻辑回归）区分高质量数据（WebText/Wiki/Books）与低质量 CC 数据。
- **The Pile (EleutherAI):**
    - 为了复现 GPT-3 而创建的开源数据集，包含22个高质量域。
    - 包括：ArXiv, PubMed, StackExchange, Enron Emails (唯一的大型公开邮件数据集)。

## 4. 特定领域数据 (Domain Specific Data)

### 4.1 书籍 (Books)

- **重要性:** 提供长上下文 (_Long Context_) 和连贯叙事。
- **来源:**
    - **Project Gutenberg:** 公共领域书籍（版权过期）。
    - **Shadow Libraries (如 LibGen, Books3):** 包含版权书籍，法律风险极高，已被多次下架。

### 4.2 代码 (Code)

- **GitHub:** 主要来源。
- **处理难点:**
    - 包含非代码文件、提交历史等。
    - 需要处理 **Licenses**（通常只保留 permissively licensed 代码）和 **De-duplication**（去重）。
    - **The Stack:** Hugging Face 发布的经过清洗的 GitHub 数据集。

### 4.3 问答与对话 (Q&A)

- **Stack Exchange:** 包含问题、答案、评论、投票数。结构天然类似于 _Instruction Following_ 数据。

## 5. 现代数据处理流水线 (Modern Data Pipelines)

### 5.1 从 LLaMA 到 DCLM

- **LLaMA:**
    - 使用 Wikipedia 引用作为启发式信号来筛选 Common Crawl。
- **RefinedWeb (Falcon) / FineWeb:**
    - 假设：如果过滤得当，仅仅 Web 数据就足够了。
    - 强调大规模去重 (_Deduplication_) 和基于规则的清洗，最初避免使用 _Model-based filtering_ 以减少偏见。
- **DataComp (DCLM):**
    - 将数据清洗视为一个优化问题。
    - **DCLM-Baseline:** 使用 **Model-based filtering** 回归主流。
    - 训练分类器（FastText），正样本来自 OpenHermes (Instruction data) 和 ELI5 (Reddit Q&A)，仅保留前 1-2% 的数据。

### 5.2 前沿技术：Nemotron-CC (Nvidia)

- **Scaling Up:** 过于激进的过滤会导致 _Token_ 数量不足。
- **新方法:**
    - **Quality Filtering:** 使用 _Large Language Model_ 对文档的“教育价值 (_Educational Value_)”进行打分。
    - **Synthetic Data (合成数据):**
        - 对低质量数据：使用 LLM 重写 (_Rewrite_)。
        - 对高质量数据：使用 LLM 生成任务 (_Generate Tasks/QA pairs_)。

## 6. 法律与版权 (Copyright & Legal)

- **基本概念:** 版权法保护“固定在有形媒介中的原创表达 (_Original works of authorship fixed in a tangible medium_)”，不保护 _Ideas_ 或 _Algorithms_。
- **Fair Use (合理使用):** 训练受版权保护的数据是否合法主要依赖 _Fair Use_ 抗辩，需考量四个因素：
    1. 使用的目的（商业 vs 教育，是否具有转换性 _Transformative_）。
    2. 作品的性质（事实 vs 虚构）。
    3. 使用的数量。
    4. 对潜在市场的影响 (_Effect on the market_)。
- **现状:** 即使通过 _Fair Use_ 或拥有 _License_，许多平台（如 YouTube）的 _Terms of Service_ 也会限制数据抓取。

## 7. Mid-training 与 Post-training 数据

### 7.1 长上下文扩展 (Long Context Extension)

- 由于 _Transformers_ 的计算复杂度是序列长度的平方 ($O(N^2)$)，通常在 Mid-training 阶段才引入长文本训练。
- 数据源：书籍、数学推导。

### 7.2 指令微调 (Instruction Tuning)

- **SuperNaturalInstructions / FLAN:** 将传统 NLP 任务转化为 Prompt/Response 格式。
- **Synthetic Instruction Data:**
    - **Self-Instruct (Alpaca):** 提示 LLM (如 GPT-4) 生成指令-输出对。
    - **Chat Logs (WildChat/ShareGPT):** 用户与 Chatbot 的真实对话记录。
    - **Evol-Instruct:** 将简单指令复杂化以提升难度。

## 8. 总结 (Summary)

- 数据来源从简单的 Web 抓取演变为复杂的流水线：**Raw Dump -> De-duplication -> Heuristic/Model-based Filtering -> Synthetic Augmentation**。
- 数据工程目前仍充满启发式规则 (_Heuristics_)，缺乏统一的理论原则，但它是模型性能差异化的核心。