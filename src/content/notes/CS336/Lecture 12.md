---
title: '斯坦福 CS336：从零构建语言模型 (Language Modeling from Scratch)'
category: 'CS336'
order: 12
description: ""
tags: []
---
# 斯坦福 CS336：从零构建语言模型 (Language Modeling from Scratch)

## 第12讲：评估 (Evaluation)

---

### 第一部分：评估概论 (Introduction to Evaluation)

**1. 评估的现状与挑战**

- **表面简单，实则复杂**：机械过程看似只是“给定模型，计算指标”，但在实际操作中是一个深刻的话题。
- **评估危机 (Evaluation Crisis)**：Andrej Karpathy 指出当前面临评估危机。基准测试（Benchmarks）层出不穷，但存在饱和（Saturated）或被利用（Gamed）的问题。
- **发展风向标**：评估指标直接影响模型的开发方向，是未来发展的先行指标。

**2. 为什么要评估？(The Purpose of Evaluation)** 没有“唯一正确”的评估，取决于你想回答的问题：

- **用户/企业 (User/Company)**：购买决策（例如：选择 Claude, Gemini 还是 03？）。
- **研究人员 (Researcher)**：科学进展，衡量模型的原始能力 (Raw capabilities)。
- **政策制定者 (Policy Makers)**：了解模型的社会效益与危害 (Benefits and harms)。
- **模型开发者 (Developer)**：开发周期中的反馈，用于改进模型。

**3. 评估框架 (Evaluation Framework)** 一个简单的思考框架：

1. **Inputs (Prompts)**：提示词的来源？是否覆盖长尾情况 (Tails)？是否适应特定模型？。
2. **Model Call**：如何调用模型？(Zero-shot, Few-shot, Chain-of-Thought, Tool use)。
3. **Outputs**：如何评估输出？(Reference outputs, Pass@k, Cost, Error types)。
4. **Interpretation**：如何解读结果？(Train-test overlap, System vs. Method)。

---

### 第二部分：困惑度 (Perplexity) —— 经典指标

**1. 定义与作用**

- **Perplexity** 衡量模型对数据集的概率分配。
- **Pre-training** 目标是最小化训练集的 **Perplexity**。
- **优点**：
    - 比下游任务准确率 (Downstream task accuracy) 更平滑 (Smoother)，包含细粒度的 **Logits/Probabilities** 信息。
    - 具有“通用性 (Universal)”，关注每一个 **Token**。
- **局限**：作为排行榜指标时，需要信任模型提供者正确计算且概率和为1 (Sum to one)，容易出现 Bug 或被操纵。

**2. 历史演变**

- **2010s 标准数据集**：Penn Treebank, WikiText, 1 Billion Word Benchmark。
- **研究范式**：在特定分割 (Split) 上训练并在同分布测试集上测试。
- **转折点 (GPT-2)**：零样本 (Zero-shot) 评估。在 **WebText** 上训练，直接在上述标准集上测试，证明了 **Out-of-distribution (OOD)** 的泛化能力。

---

### 第三部分：知识与能力基准测试 (Knowledge & Capabilities Benchmarks)

**1. 标准化考试类 (Standardized Tests)**

- **MMLU (Massive Multitask Language Understanding)**：
    - 包含57个学科的 **Multiple choice questions**。
    - 2020年发布时 GPT-3 仅有 ~45% 准确率，现已达到 ~90%（饱和）。
    - 主要测试知识 (Knowledge) 而非纯语言理解。
- **MMLU-Pro**：
    - 针对 MMLU 饱和问题，删除了简单的题目，选项从4个增加到10个，增加了难度。
- **GPQA (Google-Proof Q&A)**：
    - PhD 级别的领域问题。
    - 被称为 "Google-proof"，非专家即便使用 Google 也很难在30分钟内回答。
- **Humanity's Last Exam (HLE)**：
    - 多模态 (Multimodal)，极高难度，由专家众包并在 Frontier models 上筛选以确保难度。
    - 目前最佳模型得分仅约 20%。

**2. 指令遵循与开放式生成 (Instruction Following & Open-Ended Generation)**

- **挑战**：没有 **Ground truth**，难以评估。
- **Chatbot Arena**：
    - 基于 **Elo rating** 系统。
    - 人类对两个匿名模型的输出进行成对偏好排序 (Pairwise preference)。
    - 动态更新，反映真实分布，但面临 "Leaderboard Illusion" 和被刷榜的风险。
- **IFEval (Instruction Following Eval)**：
    - 基于硬性约束 (Constraints) 评估（例如：“不使用逗号”、“至少10个单词”）。
    - 易于验证，但仅评估形式而非内容质量。
- **AlpacaEval**：
    - 使用 **LM-as-a-judge** (如 GPT-4) 计算针对基准模型的胜率 (Win rate)。
    - **Length bias** 问题：模型倾向于偏好更长的回复。

---

### 第四部分：高级能力评估 (Advanced Capabilities)

**1. 智能体 (Agents)**

- **定义**：涉及工具使用 (Tool use)、多步迭代、长期规划。
- **主要基准**：
    - **SWE-bench**：软件工程任务。给定 GitHub Issue，模型需生成代码补丁 (PR)并通过单元测试。
    - **Cybench**：网络安全任务 (Capture the Flag)，涉及渗透测试操作。
    - **MLE-bench**：Kaggle 竞赛代理。自动进行代码编写、模型训练和提交。

**2. 纯推理 (Pure Reasoning)**

- **ARC-AGI**：
    - 旨在剥离语言和知识，仅测试核心智力/推理能力。
    - 通过少量示例推断抽象图形模式 (Abstract patterns)。
    - 传统 LLM 表现极差，近期 03 模型通过大量 **Compute** 提升了表现。

---

### 第五部分：安全性评估 (Safety Evaluation)

**1. 安全性定义**

- **Contextual**：依赖于法律、文化规范和使用场景。
- **Refusal vs. Capability**：安全性不仅仅是拒绝回答 (Refusal)。在某些场景（如医疗），减少幻觉 (Hallucinations) 既提升了能力也提升了安全性。

**2. 基准测试与攻防**

- **HarmBench & AirBench**：测试模型对有害指令的拒绝能力及对法规政策的遵循。
- **Jailbreaking**：通过优化提示词 (Optimizing prompts) 绕过安全护栏。
- **Pre-deployment Testing**：安全机构（如 US/UK Safety Institutes）在模型发布前进行自愿测试。

---

### 第六部分：有效性与反思 (Validity & Reflection)

**1. 训练集污染 (Train-Test Overlap/Contamination)**

- **问题**：在网络规模数据上训练，很难保证测试集未被见过。
- **对策**：
    - **Decontamination**：移除 n-gram 重叠的数据。
    - **Inference**：通过模型对特定序列顺序的偏好来推断是否见过数据。

**2. 真实性 (Realism)**

- **Quizzing vs. Asking**：标准化考试往往是“测验”（用户知道答案），而真实使用是“询问”（用户不知道答案）。
- **Dataset Quality**：许多基准测试（如 Math, GSM8K）存在大量标签噪声 (Label noise)。

**3. 总结：我们在评估什么？**

- **System vs. Method**：过去我们评估“方法”（控制变量法，固定架构/算法）；现在我们评估“系统”（System），包含数据、训练策略、推理技巧等整体。
- 没有完美的评估，关键在于明确评估的目的和游戏规则。