---
title: K-means 聚类算法
published: 2025-7-28 23:50:22
slug: k-means-clustering-algorithm
tags: ['深度学习', '机器学习', '聚类算法', 'K-means']
category: '机器学习'
draft: false
image: ./bg.jpg
---
## K-means 聚类算法

## 前言

今天开始学习无监督学习，K-means 是最基础也是最重要的一个算法。和之前学的分类算法不同，聚类是在没有标签的情况下，让算法自己去发现数据中的结构，把相似的数据点“物以类聚”。感觉就像是在整理一堆混在一起的乐高积木，要把相同颜色和形状的放在一起。这篇笔记记录了我对 K-means 的学习和实践。

## 核心思想：物以类聚，人以群分

K-means 的目标非常明确：把一堆数据点分成 $K$ 个簇 (Cluster)，并且让**簇内的数据点尽可能相似（距离近），簇间的数据点尽可能不同（距离远）**。

为了实现这个目标，算法会迭代地做两件事：

1. **分配**：把每个数据点分给离它最近的那个簇的中心。
2. **更新**：把每个簇的中心移动到这个簇里所有数据点的平均位置。

不断重复这两步，直到簇的中心不再变化，就找到了最终的聚类结果。

## 深入数学原理

### 目标函数：我们要优化什么？

K-means 优化的目标函数是所有数据点到其所属簇中心的**距离平方和 (Sum of Squared Errors, SSE)**，也称为**簇内平方和 (Within-Cluster Sum of Squares, WCSS)** 或**惯性 (Inertia)**。我们的目标是最小化这个值。

$$
J = \sum_{i=1}^n \sum_{k=1}^K r_{ik} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2
$$

其中：

* $n$ 是样本数量，$K$ 是要分的簇的数量。
* $\mathbf{x}_i$ 是第 $i$ 个数据点。
* $\boldsymbol{\mu}_k$ 是第 $k$ 个簇的中心点（质心）。
* $r_{ik}$ 是一个指示变量：如果 $\mathbf{x}_i$ 属于簇 $k$，则 $r_{ik}=1$，否则为0。

### 算法步骤：E-M思想的体现

K-means 的迭代过程完美地体现了 **EM (Expectation-Maximization) 算法**的思想。

1. **初始化**: 随机选择 $K$ 个数据点作为初始的簇中心 $\boldsymbol{\mu}_1, \boldsymbol{\mu}_2, \dots, \boldsymbol{\mu}_K$。

2. **重复以下步骤直到收敛**:
    * **E-Step (Expectation/分配步)**: 对于每个数据点 $\mathbf{x}_i$，计算它到所有 $K$ 个簇中心的距离，并把它分配给最近的那个簇。这一步是在固定簇中心 $\boldsymbol{\mu}_k$ 的情况下，优化分配 $r_{ik}$ 来最小化目标函数 $J$。
        $$
        r_{ik} =
        \begin{cases}
        1 & \text{if } k = \arg\min_j \|\mathbf{x}_i - \boldsymbol{\mu}_j\|^2 \\
        0 & \text{otherwise}
        \end{cases}
        $$
    * **M-Step (Maximization/更新步)**: 对于每个簇 $k$，重新计算它的中心点 $\boldsymbol{\mu}_k$，使其成为该簇内所有数据点的均值。这一步是在固定分配 $r_{ik}$ 的情况下，优化簇中心 $\boldsymbol{\mu}_k$ 来最小化目标函数 $J$。
        $$
        \boldsymbol{\mu}_k = \frac{\sum_{i=1}^n r_{ik} \mathbf{x}_i}{\sum_{i=1}^n r_{ik}}
        $$

**收敛性证明**:

* 在E-step，我们把点分给最近的中心，这必然会使总的距离平方和 $J$ 减小或不变。
* 在M-step，把中心更新为均值，根据均值的性质，这也会使该簇内的距离平方和达到最小，从而使总的 $J$ 减小或不变。
* 因为 $J$ 是一个非负值，它有下界。一个单调递减且有下界的序列必然会收敛。
* **注意**：K-means 只能保证收敛到**局部最优解**，而不是全局最优解。最终结果对初始点的选择很敏感。

### 如何选择合适的 K 值？

这是 K-means 的一个核心问题。$K$ 值选得太大或太小，都得不到有意义的结果。有几种常用的方法来帮助我们选择。

1. **肘部法则 (Elbow Method)**:
    * **方法**：计算不同 $K$ 值对应的目标函数 $J$ (即 WCSS)。
    * **原理**：随着 $K$ 的增加，WCSS 会不断减小。我们寻找曲线下降速率由快变缓的那个点，就像人的“手肘”一样。这个拐点通常被认为是比较合适的 $K$ 值。
    * **缺点**：“手肘”有时不明显，主观性较强。

2. **轮廓系数 (Silhouette Coefficient)**:
    * **方法**：对每个样本 $i$，计算其轮廓系数 $s(i)$。
        $$
        s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}
        $$
        * $a(i)$: 样本 $i$ 到**同簇**其他所有点的平均距离（衡量簇内凝聚度）。
        * $b(i)$: 样本 $i$ 到**最近的异簇**所有点的平均距离（衡量簇间分离度）。
    * **原理**：$s(i)$ 的取值范围是 $[-1, 1]$。
        * 值越接近1，说明聚类效果越好。
        * 值越接近-1，说明可能分到了错误的簇。
        * 值在0附近，说明样本在两个簇的边界上。
    * **选择**：计算所有样本的平均轮廓系数，选择使平均轮廓系数最大的那个 $K$ 值。

3. **Gap 统计量 (Gap Statistic)**:
    * **方法**：将真实数据的 WCSS 与在数据边界内随机生成的“无结构”数据的 WCSS 进行比较。
    * **原理**：寻找一个 $K$ 值，使得真实数据的 WCSS 下降程度，远大于随机数据的 WCSS 下降程度。这个“Gap”最大的点就是最优的 $K$。
    * **优点**：比肘部法则更自动化，但计算量更大。

---

## 实战演练：客户分群分析

理论学完了，我们来模拟一个真实的商业场景：对客户进行分群，以便实施精准营销。

### 0. 准备工作

首先，导入需要的库并生成模拟数据。在真实项目中，这一步是加载和清洗你的客户数据。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn 工具集
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```

### 1. 客户数据预处理和特征工程

我们生成一些模拟数据，代表客户的两个特征：**年收入**和**消费分数**。

```python
# 生成模拟客户数据
X, y_true = make_blobs(n_samples=300, centers=5, cluster_std=0.8, random_state=42)

# 为了更贴近真实场景，我们对数据进行一些变换
rng = np.random.RandomState(42)
X[:, 0] = X[:, 0] * 20000 + 80000  # 年收入
X[:, 1] = X[:, 1] * 30 + 50        # 消费分数 (1-100)

# 将数据转换为DataFrame，便于分析
customer_df = pd.DataFrame(X, columns=['Annual Income (k$)', 'Spending Score (1-100)'])

# --- 数据标准化 ---
# 年收入和消费分数的量纲差异很大，必须进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 2. 最优K值选择

我们同时使用**肘部法则**和**轮廓系数**来确定最佳的 $K$ 值。

```python
wcss = []
silhouette_scores = []
K_range = range(2, 11) # K值至少为2

for k in K_range:
    # 训练K-means模型
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    
    # 记录WCSS (惯性)
    wcss.append(kmeans.inertia_)
    
    # 记录轮廓系数
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# 绘制K值选择图
plt.figure(figsize=(12, 5))

# 肘部法则
plt.subplot(1, 2, 1)
plt.plot(K_range, wcss, 'bo-')
plt.xlabel('簇数量 K')
plt.ylabel('簇内平方和 (WCSS)')
plt.title('肘部法则')
plt.grid(True)

# 轮廓系数
plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'ro-')
plt.xlabel('簇数量 K')
plt.ylabel('平均轮廓系数')
plt.title('轮廓系数法')
plt.grid(True)

plt.tight_layout()
plt.show()

# 根据图像，我们选择 K=5
optimal_k = 5
```

从图中我们可以清晰地看到，肘部法则在 K=5 处有一个明显的拐点，而轮廓系数也在 K=5 时达到最大值。因此，我们确定最佳的簇数量是5。

### 3. 客户分群和特征分析

现在，我们使用找到的最佳 $K$ 值来训练最终模型，并对分群结果进行分析。

```python
# 使用最佳K值训练最终模型
final_kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
customer_df['Cluster'] = final_kmeans.fit_predict(X_scaled)

# 可视化分群结果
plt.figure(figsize=(10, 8))
sns.scatterplot(data=customer_df, x='Annual Income (k$)', y='Spending Score (1-100)', 
                hue='Cluster', palette='viridis', s=100, alpha=0.8, legend='full')
centroids = scaler.inverse_transform(final_kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title('客户分群结果')
plt.xlabel('年收入 (k$)')
plt.ylabel('消费分数 (1-100)')
plt.legend()
plt.grid(True)
plt.show()

# 分析每个客户群的特征
cluster_analysis = customer_df.groupby('Cluster').agg({
    'Annual Income (k$)': ['mean', 'median'],
    'Spending Score (1-100)': ['mean', 'median'],
    'Cluster': 'count'
}).rename(columns={'count': 'Num Customers'})

print("\n--- 各客户群特征分析 ---")
print(cluster_analysis)
```

### 4. 业务洞察和建议

最后，也是最重要的一步，是根据分群结果给出有价值的业务建议。

* **簇 0: 谨慎型**
  * **特征**: 收入低，消费分数低。
  * **洞察**: 可能是学生或刚入职的年轻人，对价格敏感。
  * **建议**: 推送折扣、优惠券等促销信息，进行低价商品营销。

* **簇 1: 挥霍型**
  * **特征**: 收入低，但消费分数高。
  * **洞察**: "月光族"，追求时尚和潮流，但购买力有限。
  * **建议**: 推送新品、潮流单品，提供分期付款等服务。

* **簇 2: 普通型**
  * **特征**: 收入和消费分数都处于中等水平。
  * **洞察**: 这是最主流的客户群体，构成了消费的基本盘。
  * **建议**: 维持客户关系，通过会员制度、积分兑换等方式提高客户忠诚度。

* **簇 3: 目标型 (高价值)**
  * **特征**: 收入高，消费分数高。
  * **洞察**: **核心高价值客户**，有钱又愿意花钱。
  * **建议**: 提供VIP服务、专属定制、高端新品发布会等，重点维护，提升品牌忠诚度。

* **簇 4: 理性型**
  * **特征**: 收入高，但消费分数低。
  * **洞察**: 有钱但消费谨慎，可能对品质和实用性要求高。
  * **建议**: 强调产品的质量、耐用性和性价比，避免过度营销，提供详细的产品信息。

---

## 学习总结

### 容易踩的坑

* **忘记数据标准化**：K-means 基于距离计算，如果特征量纲不一致，量纲大的特征会主导聚类结果。这是最致命的错误！
* **K 值选择不当**：业务理解和多种技术方法结合来确定 $K$ 值非常重要。
* **初始点敏感性**：传统的随机初始化可能导致很差的局部最优解。使用 `k-means++` 初始化（`sklearn` 的默认方法）能极大地改善这个问题。
* **对非球形簇和异常值敏感**：K-means 假设簇是凸的、球形的，对不规则形状的簇和异常值处理得不好。

### 学习感悟

K-means 是一个非常直观且高效的聚类算法，完美地展示了无监督学习的魅力——在没有答案的情况下发现数据中隐藏的模式。从理论推导的 E-M 步骤，到如何选择 $K$ 值的各种权衡，再到最终如何将聚类结果转化为有价值的商业洞察，整个过程构成了一个完整的、从数据到决策的闭环。虽然它有局限性，但作为探索性数据分析的第一步，K-means 无疑是强大且不可或缺的工具。
