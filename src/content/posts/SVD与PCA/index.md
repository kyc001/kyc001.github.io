---
title: SVD与PCA
published: 2025-7-26 15:53:33
slug: SVD-and-PCA
tags: ['深度学习', '机器学习', '线性代数']
category: '学习笔记'
draft: false
image: ./bg.jpg
---


## SVD与PCA

## 目录

1. [数学基础](#数学基础)
2. [奇异值分解(SVD)](#奇异值分解svd)
3. [主成分分析(PCA)](#主成分分析pca)
4. [SVD与PCA的关系](#svd与pca的关系)
5. [实际应用](#实际应用)
6. [代码实现](#代码实现)
7. [进阶话题](#进阶话题)

---

## 数学基础

### 线性代数核心概念

#### 1. 矩阵的秩(Rank)

- **定义**: 矩阵中线性无关的行(或列)的最大数目
- **几何意义**: 矩阵变换后空间的维度
- **性质**:
  - $\text{rank}(A) \leq \min(m,n)$ 对于 $m \times n$ 矩阵
  - $\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$

#### 2. 特征值与特征向量

对于方阵 $A$，如果存在标量 $\lambda$ 和非零向量 $\mathbf{v}$ 使得：

$$
A\mathbf{v} = \lambda\mathbf{v}
$$

则 $\lambda$ 是特征值，$\mathbf{v}$ 是对应的特征向量。

**几何意义**: 特征向量在变换下方向不变，只是长度被特征值缩放。

#### 3. 正交矩阵

- **定义**: $Q^T Q = I$（转置等于逆矩阵）
- **性质**: 保持向量长度和角度不变
- **几何意义**: 表示旋转或反射变换

---

## 奇异值分解(SVD)

### 核心定理

对于任意 $m \times n$ 实矩阵 $A$，都存在分解：

$$
A = U\Sigma V^T
$$

其中：

- **U**: $m \times m$ 正交矩阵（左奇异向量）
- **Σ**: $m \times n$ 对角矩阵（奇异值）
- **V**: $n \times n$ 正交矩阵（右奇异向量）

### 数学推导

#### Step 1: 构造对称矩阵

考虑 $A^T A$（$n \times n$ 对称正定矩阵）：

$$
A^T A = (U\Sigma V^T)^T (U\Sigma V^T) = V\Sigma^T U^T U\Sigma V^T = V\Sigma^T \Sigma V^T
$$

#### Step 2: 特征值分解

$A^T A$ 的特征值分解：

$$
A^T A = V\Lambda V^T
$$

其中 $\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$，$\lambda_i \geq 0$

#### Step 3: 奇异值定义

奇异值 $\sigma_i = \sqrt{\lambda_i}$，按降序排列：

$$
\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0 = \sigma_{r+1} = \cdots = \sigma_n
$$

#### Step 4: 左奇异向量构造

对于 $i = 1, 2, \ldots, r$：

$$
\mathbf{u}_i = \frac{1}{\sigma_i} A\mathbf{v}_i
$$

### 几何解释

SVD将任意线性变换分解为三个步骤：

1. **$V^T$**: 在输入空间中旋转
2. **$\Sigma$**: 沿坐标轴缩放
3. **$U$**: 在输出空间中旋转

### 重要性质

#### 1. 最优低秩近似

对于秩为 $k$ 的最优近似：

$$
A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T
$$

这是Frobenius范数意义下的最优近似。

#### 2. Eckart-Young定理

$$
\|A - A_k\|_F = \sqrt{\sigma_{k+1}^2 + \sigma_{k+2}^2 + \cdots + \sigma_r^2}
$$

#### 3. 能量保存

$$
\|A\|_F^2 = \sum_{i=1}^{r} \sigma_i^2
$$

---

## 主成分分析(PCA)

### 问题设定

给定数据矩阵 $X \in \mathbb{R}^{n \times d}$（$n$个样本，$d$个特征），寻找$k$维子空间使得投影后的方差最大。

### 数学推导

#### Step 1: 数据中心化

$$
\tilde{X} = X - \boldsymbol{\mu}
$$

其中 $\boldsymbol{\mu}$ 是样本均值向量。

#### Step 2: 协方差矩阵

$$
C = \frac{1}{n-1} \tilde{X}^T \tilde{X}
$$

#### Step 3: 特征值分解

$$
C = V\Lambda V^T
$$

其中特征值按降序排列：$\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d$

#### Step 4: 主成分选择

前$k$个特征向量构成主成分：

$$
W = [\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k]
$$

#### Step 5: 降维变换

$$
Y = \tilde{X}W
$$

### 优化视角

PCA等价于求解优化问题：

$$
\max \text{tr}(W^T C W) \quad \text{s.t.} \quad W^T W = I
$$

使用拉格朗日乘数法可得解为协方差矩阵的前$k$个特征向量。

### 方差解释

第$i$个主成分解释的方差比例：

$$
\text{explained\_variance\_ratio}_i = \frac{\lambda_i}{\sum_{j=1}^{d} \lambda_j}
$$

累积方差解释比例：

$$
\text{cumulative\_variance\_ratio}_k = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{j=1}^{d} \lambda_j}
$$

---

## SVD与PCA的关系

### 核心联系

PCA实际上是通过SVD实现的！

对于中心化数据矩阵 $\tilde{X}$：

$$
\tilde{X} = U\Sigma V^T
$$

则协方差矩阵：

$$
C = \frac{1}{n-1} \tilde{X}^T \tilde{X} = \frac{1}{n-1} V\Sigma^2 V^T
$$

**关键发现**：

- PCA的主成分 = SVD的右奇异向量 $V$
- PCA的特征值 = SVD奇异值的平方除以$(n-1)$
- 投影后的坐标 = SVD的左奇异向量乘以奇异值

### 计算优势

使用SVD计算PCA的优势：

1. **数值稳定性**: 避免计算 X^T X（可能病态）
2. **计算效率**: 当 n << d 时更高效
3. **内存友好**: 不需要存储完整的协方差矩阵

---

## 实际应用

### 1. 图像压缩

```python
# 你的代码分析
def compress_img(img, percent):
    u, s, vt = svd(img)  # SVD分解
    # 选择保留的奇异值数量
    count = int(sum(s) * percent)
    k = 0
    cursum = 0
    while cursum <= count:
        cursum += s[k]
        k += 1
    # 重构图像
    D = u[:,:k] @ np.diag(s[:k]) @ vt[:k, :]
    return np.clip(D, 0, 255).astype(np.uint8)
```

**改进建议**：

- 使用能量保存比例而非奇异值和
- 添加压缩率评估
- 支持灰度图像处理

### 2. 推荐系统

用户-物品评分矩阵的SVD分解：

$$
R \approx U_k\Sigma_k V_k^T
$$

- $U$: 用户潜在因子
- $V$: 物品潜在因子
- 预测评分: $\hat{R}_{ij} = \mathbf{u}_i^T \mathbf{v}_j$

### 3. 自然语言处理

- **LSA (Latent Semantic Analysis)**: 文档-词汇矩阵的SVD
- **词嵌入**: 通过SVD降维获得词向量

### 4. 计算机视觉

- **人脸识别**: Eigenfaces方法
- **图像去噪**: 保留主要成分，去除噪声
- **特征提取**: 降维后的特征用于分类

---

## 代码实现

### 完整的SVD图像压缩实现

```python
import numpy as np
from scipy.linalg import svd
from PIL import Image
import matplotlib.pyplot as plt

class SVDImageCompressor:
    def __init__(self):
        self.original_shape = None
        self.compression_ratio = None
    
    def compress_by_energy(self, img, energy_ratio=0.95):
        """基于能量保存比例的压缩"""
        u, s, vt = svd(img, full_matrices=False)
        
        # 计算累积能量
        energy = np.cumsum(s**2) / np.sum(s**2)
        k = np.argmax(energy >= energy_ratio) + 1
        
        # 重构
        compressed = u[:, :k] @ np.diag(s[:k]) @ vt[:k, :]
        
        # 计算压缩比
        original_size = img.size
        compressed_size = u[:, :k].size + s[:k].size + vt[:k, :].size
        self.compression_ratio = compressed_size / original_size
        
        return np.clip(compressed, 0, 255).astype(np.uint8), k
    
    def compress_rgb_image(self, filename, energy_ratio=0.95):
        """RGB图像压缩"""
        img = Image.open(filename)
        img_array = np.array(img)
        
        if len(img_array.shape) == 3:  # RGB
            compressed_channels = []
            ranks = []
            
            for channel in range(3):
                compressed_channel, k = self.compress_by_energy(
                    img_array[:, :, channel], energy_ratio
                )
                compressed_channels.append(compressed_channel)
                ranks.append(k)
            
            compressed_img = np.stack(compressed_channels, axis=2)
        else:  # 灰度图
            compressed_img, k = self.compress_by_energy(img_array, energy_ratio)
            ranks = [k]
        
        return compressed_img, ranks

# PCA实现
class PCAAnalyzer:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
    
    def fit(self, X):
        """拟合PCA模型"""
        # 中心化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # SVD分解
        U, s, Vt = svd(X_centered, full_matrices=False)
        
        # 计算解释方差
        self.explained_variance_ = (s**2) / (X.shape[0] - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(self.explained_variance_)
        
        # 选择成分数量
        if self.n_components is None:
            self.n_components = min(X.shape)
        
        self.components_ = Vt[:self.n_components]
        
        return self
    
    def transform(self, X):
        """变换数据"""
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def fit_transform(self, X):
        """拟合并变换"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_transformed):
        """逆变换"""
        return X_transformed @ self.components_ + self.mean_
```

### 性能分析工具

```python
def analyze_compression_performance(image_path, energy_ratios):
    """分析不同压缩比的性能"""
    compressor = SVDImageCompressor()
    original_img = np.array(Image.open(image_path))
    
    results = []
    for ratio in energy_ratios:
        compressed_img, ranks = compressor.compress_rgb_image(image_path, ratio)
        
        # 计算PSNR
        mse = np.mean((original_img - compressed_img)**2)
        psnr = 20 * np.log10(255 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        results.append({
            'energy_ratio': ratio,
            'ranks': ranks,
            'compression_ratio': compressor.compression_ratio,
            'psnr': psnr
        })
    
    return results
```

---

## 进阶话题

### 1. 截断SVD (Truncated SVD)

当矩阵很大时，完整SVD计算代价昂贵。截断SVD只计算前k个奇异值：

```python
from sklearn.decomposition import TruncatedSVD

def truncated_svd_example():
    # 大型稀疏矩阵的降维
    from scipy.sparse import random
    X = random(1000, 10000, density=0.01)

    # 只计算前50个成分
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_reduced = svd.fit_transform(X)

    print(f"原始维度: {X.shape}")
    print(f"降维后: {X_reduced.shape}")
    print(f"解释方差比: {svd.explained_variance_ratio_.sum():.3f}")
```

### 2. 增量PCA (Incremental PCA)

处理无法一次性载入内存的大数据集：

```python
from sklearn.decomposition import IncrementalPCA

def incremental_pca_example():
    # 模拟大数据集的批处理
    n_samples, n_features = 10000, 100
    n_components = 20
    batch_size = 1000

    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    # 分批训练
    for i in range(0, n_samples, batch_size):
        X_batch = np.random.randn(batch_size, n_features)
        ipca.partial_fit(X_batch)

    # 变换新数据
    X_new = np.random.randn(100, n_features)
    X_transformed = ipca.transform(X_new)

    return X_transformed
```

### 3. 核PCA (Kernel PCA)

处理非线性数据的降维：

```python
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles

def kernel_pca_example():
    # 生成非线性数据
    X, y = make_circles(n_samples=1000, factor=0.3, noise=0.1)

    # 线性PCA（效果不好）
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 核PCA（RBF核）
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
    X_kpca = kpca.fit_transform(X)

    return X_pca, X_kpca
```

### 4. 稀疏PCA

当需要稀疏主成分时：

```python
from sklearn.decomposition import SparsePCA

def sparse_pca_example():
    X = np.random.randn(100, 50)

    # 标准PCA
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X)

    # 稀疏PCA
    spca = SparsePCA(n_components=10, alpha=0.1)
    X_spca = spca.fit_transform(X)

    # 比较成分的稀疏性
    pca_sparsity = np.mean(np.abs(pca.components_) < 1e-10)
    spca_sparsity = np.mean(np.abs(spca.components_) < 1e-10)

    print(f"PCA稀疏性: {pca_sparsity:.3f}")
    print(f"稀疏PCA稀疏性: {spca_sparsity:.3f}")
```

### 5. 概率PCA (Probabilistic PCA)

基于概率模型的PCA：

```python
def probabilistic_pca(X, n_components, max_iter=100, tol=1e-6):
    """概率PCA的EM算法实现"""
    n, d = X.shape

    # 初始化
    W = np.random.randn(d, n_components)
    sigma2 = 1.0
    mu = np.mean(X, axis=0)
    X_centered = X - mu

    for iteration in range(max_iter):
        # E步：计算后验期望
        M = W.T @ W + sigma2 * np.eye(n_components)
        M_inv = np.linalg.inv(M)

        # 计算期望的潜在变量
        Z = X_centered @ W @ M_inv

        # M步：更新参数
        W_new = X_centered.T @ Z @ np.linalg.inv(Z.T @ Z + sigma2 * M_inv)

        # 更新噪声方差
        reconstruction = Z @ W_new.T
        sigma2_new = np.mean((X_centered - reconstruction)**2)

        # 检查收敛
        if np.linalg.norm(W_new - W) < tol:
            break

        W, sigma2 = W_new, sigma2_new

    return W, sigma2, mu
```

---

## 实战案例深度分析

### 案例1: 人脸识别中的Eigenfaces

```python
def eigenfaces_analysis():
    """Eigenfaces方法的完整实现"""
    from sklearn.datasets import fetch_olivetti_faces
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # 加载人脸数据
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X, y = faces.data, faces.target

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # PCA降维
    n_components = 150
    pca = PCA(n_components=n_components, whiten=True)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # 可视化主成分（Eigenfaces）
    eigenfaces = pca.components_.reshape((n_components, 64, 64))

    # 使用SVM分类
    from sklearn.svm import SVC
    clf = SVC(kernel='rbf', C=1000, gamma=0.0001)
    clf.fit(X_train_pca, y_train)

    # 预测和评估
    y_pred = clf.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"识别准确率: {accuracy:.3f}")
    print(f"解释方差比: {pca.explained_variance_ratio_.sum():.3f}")

    return eigenfaces, accuracy
```

### 案例2: 文本挖掘中的LSA

```python
def latent_semantic_analysis():
    """潜在语义分析实现"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.datasets import fetch_20newsgroups

    # 加载文本数据
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    newsgroups = fetch_20newsgroups(subset='train', categories=categories)

    # TF-IDF向量化
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X = vectorizer.fit_transform(newsgroups.data)

    # SVD分解（LSA）
    n_topics = 100
    svd = TruncatedSVD(n_components=n_topics, random_state=42)
    X_lsa = svd.fit_transform(X)

    # 分析主题
    feature_names = vectorizer.get_feature_names_out()

    def print_top_words(model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            top_words_idx = topic.argsort()[-n_top_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            print(f"主题 {topic_idx}: {' '.join(top_words)}")

    print("LSA发现的主题:")
    print_top_words(svd, feature_names, 10)

    return X_lsa, svd
```

### 案例3: 推荐系统中的矩阵分解

```python
class MatrixFactorizationRecommender:
    """基于矩阵分解的推荐系统"""

    def __init__(self, n_factors=50, learning_rate=0.01, regularization=0.01, n_epochs=100):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs

    def fit(self, ratings_matrix):
        """训练矩阵分解模型"""
        self.n_users, self.n_items = ratings_matrix.shape

        # 初始化用户和物品因子矩阵
        self.user_factors = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (self.n_items, self.n_factors))

        # 获取非零评分的位置
        self.user_indices, self.item_indices = np.nonzero(ratings_matrix)
        self.ratings = ratings_matrix[self.user_indices, self.item_indices]

        # 训练
        for epoch in range(self.n_epochs):
            self._sgd_step(ratings_matrix)

            if epoch % 10 == 0:
                rmse = self._compute_rmse(ratings_matrix)
                print(f"Epoch {epoch}, RMSE: {rmse:.4f}")

    def _sgd_step(self, ratings_matrix):
        """随机梯度下降步骤"""
        for i in range(len(self.ratings)):
            user_id = self.user_indices[i]
            item_id = self.item_indices[i]
            rating = self.ratings[i]

            # 预测评分
            prediction = np.dot(self.user_factors[user_id], self.item_factors[item_id])
            error = rating - prediction

            # 更新因子
            user_factor = self.user_factors[user_id].copy()

            self.user_factors[user_id] += self.learning_rate * (
                error * self.item_factors[item_id] -
                self.regularization * self.user_factors[user_id]
            )

            self.item_factors[item_id] += self.learning_rate * (
                error * user_factor -
                self.regularization * self.item_factors[item_id]
            )

    def _compute_rmse(self, ratings_matrix):
        """计算RMSE"""
        predictions = self.user_factors @ self.item_factors.T
        mask = ratings_matrix != 0
        return np.sqrt(np.mean((ratings_matrix[mask] - predictions[mask])**2))

    def predict(self, user_id, item_id):
        """预测用户对物品的评分"""
        return np.dot(self.user_factors[user_id], self.item_factors[item_id])

    def recommend(self, user_id, n_recommendations=10):
        """为用户推荐物品"""
        user_ratings = self.user_factors[user_id] @ self.item_factors.T
        # 排除已评分的物品
        # 返回评分最高的n个物品
        return np.argsort(user_ratings)[-n_recommendations:][::-1]
```

---

## 数值稳定性与计算优化

### 1. 数值稳定性问题

#### 病态矩阵处理

```python
def stable_pca(X, regularization=1e-10):
    """数值稳定的PCA实现"""
    # 中心化
    X_centered = X - np.mean(X, axis=0)

    # 使用SVD而非协方差矩阵特征分解
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # 正则化小奇异值
    s_reg = np.maximum(s, regularization)

    # 计算主成分
    components = Vt
    explained_variance = s_reg**2 / (X.shape[0] - 1)

    return components, explained_variance
```

#### 大数据处理策略

```python
def memory_efficient_pca(X, n_components, batch_size=1000):
    """内存高效的PCA实现"""
    n_samples, n_features = X.shape

    # 计算均值
    mean = np.zeros(n_features)
    for i in range(0, n_samples, batch_size):
        batch = X[i:i+batch_size]
        mean += np.sum(batch, axis=0)
    mean /= n_samples

    # 计算协方差矩阵（分批）
    cov = np.zeros((n_features, n_features))
    for i in range(0, n_samples, batch_size):
        batch = X[i:i+batch_size] - mean
        cov += batch.T @ batch
    cov /= (n_samples - 1)

    # 特征分解
    eigenvals, eigenvecs = np.linalg.eigh(cov)

    # 排序（降序）
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]

    return eigenvecs[:, :n_components], eigenvals[:n_components]
```

### 2. 并行计算优化

```python
import multiprocessing as mp
from joblib import Parallel, delayed

def parallel_svd_compression(image_channels, energy_ratio=0.95, n_jobs=-1):
    """并行SVD图像压缩"""

    def compress_channel(channel):
        u, s, vt = svd(channel, full_matrices=False)
        energy = np.cumsum(s**2) / np.sum(s**2)
        k = np.argmax(energy >= energy_ratio) + 1
        return u[:, :k] @ np.diag(s[:k]) @ vt[:k, :]

    # 并行处理各通道
    compressed_channels = Parallel(n_jobs=n_jobs)(
        delayed(compress_channel)(channel) for channel in image_channels
    )

    return np.stack(compressed_channels, axis=2)
```

---

## 理论深度扩展

### 1. SVD的唯一性

SVD分解在以下意义下是唯一的：

- 奇异值 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r$ 是唯一的（按降序排列）
- 当奇异值互不相等时，奇异向量在符号上是唯一的
- 当存在重复奇异值时，对应的奇异向量张成的子空间是唯一的

### 2. PCA的概率解释

PCA可以看作是寻找数据的最优线性子空间，使得：

1. **最大方差**: 投影后方差最大化 $\max \text{Var}(XW)$
2. **最小重构误差**: 重构误差最小化 $\min \|X - \hat{X}\|_F^2$
3. **最大似然**: 在高斯噪声假设下的最大似然估计

### 3. 信息论视角

从信息论角度，PCA实现了：

- **信息压缩**: 用较少维度表示大部分信息
- **去相关**: 主成分之间线性无关，$\text{Cov}(Y_i, Y_j) = 0$ for $i \neq j$
- **熵最大化**: 在给定约束下最大化信息熵

---

## 常见陷阱与解决方案

### 1. 数据预处理陷阱

```python
# ❌ 错误：忘记标准化
pca_wrong = PCA(n_components=2)
X_wrong = pca_wrong.fit_transform(X_raw)

# ✅ 正确：先标准化再PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
pca_correct = PCA(n_components=2)
X_correct = pca_correct.fit_transform(X_scaled)
```

### 2. 维度选择陷阱

```python
def optimal_components_selection(X, variance_threshold=0.95):
    """自动选择最优成分数量"""
    pca = PCA()
    pca.fit(X)

    # 累积方差解释比
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum_var >= variance_threshold) + 1

    # 使用肘部法则验证
    def compute_reconstruction_error(n_comp):
        pca_temp = PCA(n_components=n_comp)
        X_transformed = pca_temp.fit_transform(X)
        X_reconstructed = pca_temp.inverse_transform(X_transformed)
        return np.mean((X - X_reconstructed)**2)

    errors = [compute_reconstruction_error(i) for i in range(1, min(50, X.shape[1]))]

    return n_components, errors
```

### 3. 过拟合问题

```python
def cross_validated_pca(X, y, n_components_range, cv=5):
    """交叉验证选择PCA成分数量"""
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    scores = []
    for n_comp in n_components_range:
        pipeline = Pipeline([
            ('pca', PCA(n_components=n_comp)),
            ('classifier', LogisticRegression())
        ])

        cv_scores = cross_val_score(pipeline, X, y, cv=cv)
        scores.append(cv_scores.mean())

    optimal_n_comp = n_components_range[np.argmax(scores)]
    return optimal_n_comp, scores
```

---

## 学习总结

### 容易踩的坑

#### 数据预处理的坑

- 忘记标准化导致某个特征主导整个PCA结果
- 没处理异常值，第一主成分全是噪声
- 数据泄露：在分割训练/测试集之前就做了PCA

#### 维度选择的坑

- 盲目选择解释90%方差，结果维度还是太高
- 没考虑下游任务，降维后分类效果反而变差
- 过度降维丢失了关键信息

#### 实现上的坑

- 直接算协方差矩阵特征分解，数值不稳定
- 大矩阵直接SVD，内存爆炸
- 没有正确处理奇异值为0的情况

### 学到的小技巧

#### 快速判断PCA效果

```python
# 看累积方差解释比例的"肘部"
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('成分数量')
plt.ylabel('累积方差解释比例')
```

#### 处理大数据集

```python
# 分批处理，避免内存问题
from sklearn.decomposition import IncrementalPCA
ipca = IncrementalPCA(n_components=50, batch_size=1000)
```

#### 检查线性假设

```python
# 如果前几个主成分解释方差很少，考虑非线性方法
if pca.explained_variance_ratio_[:5].sum() < 0.5:
    print("数据可能不适合线性降维")
```

### 使用场景总结

**用SVD的场景**

- 图像/信号压缩
- 推荐系统的矩阵分解
- 文本挖掘的LSA
- 需要精确控制秩的时候

**用PCA的场景**

- 特征降维预处理
- 数据可视化（降到2D/3D）
- 去除特征间相关性
- 噪声过滤

**别用PCA的场景**

- 数据本身就是非线性结构（比如螺旋形）
- 特征已经很少了（<10个）
- 需要保持特征的可解释性
- 分类边界是非线性的

### 参数选择心得

**成分数量选择**

- 先看解释方差比例，一般80-95%
- 再用交叉验证在下游任务上验证
- 可视化任务通常2-3个成分就够

**预处理选择**

- 特征量纲差异大：一定要标准化
- 特征都是同类型：可以只中心化
- 有异常值：考虑robust scaling

**性能优化**

- n >> d：用协方差矩阵的特征分解
- n << d：用SVD
- 超大数据：用随机化SVD或增量PCA

---
