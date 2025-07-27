---
title: 支持向量机(SVM)
published: 2025-7-27 00:36:22
slug: support-vector-machine-svm
tags: ['深度学习', '机器学习', '分类算法', 'SVM', '支持向量机']
category: '机器学习'
draft: false
image: ./bg.jpg
---

## 支持向量机 (SVM)

### 前言

说到分类算法，SVM (Support Vector Machine) 是一个绕不开的经典模型。它名字听起来很酷，背后的数学思想也非常优美和强大。它在线性分类和非线性分类问题上都有出色的表现，尤其是在中小型数据集上，效果常常能超过更复杂的模型。这篇笔记整理了我从理论推导到动手实现的全过程。

## 基础知识回顾

### 什么是超平面？

在线性回归里我们拟合一条直线，在SVM里我们寻找一个**超平面 (Hyperplane)** 来分割数据。

* 在二维空间，它是一条直线。
* 在三维空间，它是一个平面。
* 在更高维的空间，就叫超平面了。

它的数学表达式和线性函数一样：
$$
\mathbf{w}^T\mathbf{x} + b = 0
$$
其中 $\mathbf{w}$ 是法向量，决定了超平面的方向；$b$ 是偏置，决定了超平面与原点的距离。

### 函数间隔与几何间隔

我们希望超平面能很好地分开始两类数据。那么如何衡量“分得多好”呢？

1. **函数间隔**：$\hat{\gamma} = y(\mathbf{w}^T\mathbf{x} + b)$。它能表示分类的正确性，但如果我们同时缩放 $\mathbf{w}$ 和 $b$，函数间隔也会跟着变，这不合理。
2. **几何间隔**：$\gamma = \frac{y(\mathbf{w}^T\mathbf{x} + b)}{\|\mathbf{w}\|} = \frac{\hat{\gamma}}{\|\mathbf{w}\|}$。它代表了数据点到超平面的真实距离，不受 $\mathbf{w}$ 缩放的影响。这是我们真正关心的。

---

## 核心思想：间隔最大化

SVM 的核心思想非常直观：找到一个超平面，不仅能正确地将两类数据分开，而且要让这个超平面到两边最近的数据点的**间隔（Margin）最大化**。

想象在两类数据点之间画一条“街道”，SVM的目标就是让这条“街道”最宽。

* 街道的**中心线**就是我们的决策超平面。
* 街道的**两条边界**穿过的那些数据点，就是**支持向量 (Support Vectors)**。

**目标**：找到合适的 $\mathbf{w}$ 和 $b$，最大化几何间隔。

$$
\max_{\mathbf{w}, b} \left( \min_{i} \frac{y_i(\mathbf{w}^T\mathbf{x}_i + b)}{\|\mathbf{w}\|} \right)
$$

<br/>

## 深入数学原理

### 从原问题到对偶问题 (硬间隔)

直接优化上面的式子很复杂。我们可以做一个聪明的简化：

1. 注意到几何间隔不受 $\mathbf{w}$ 和 $b$ 的缩放影响，我们可以固定函数间隔 $\hat{\gamma} = 1$。
2. 这样，最大化 $\frac{1}{\|\mathbf{w}\|}$ 就等价于最小化 $\|\mathbf{w}\|$，为了方便求导，我们通常最小化 $\frac{1}{2}\|\mathbf{w}\|^2$。

**原问题 (Primal Problem):**
$$
\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 \\
\text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1, \quad i=1, \dots, n
$$

这是一个带约束的凸优化问题。通过**拉格朗日乘子法**，我们可以得到其**对偶问题 (Dual Problem)**。

**1. 构造拉格朗日函数**：引入拉格朗日乘子 $\alpha_i \geq 0$
$$
L(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_{i=1}^n \alpha_i [y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1]
$$

**2. 求解KKT条件**：对 $\mathbf{w}$ 和 $b$ 求偏导并令其为0，得到：
$$
\nabla_{\mathbf{w}} L = \mathbf{w} - \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i = 0 \quad \Rightarrow \quad \mathbf{w} = \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i
$$
$$
\nabla_{b} L = -\sum_{i=1}^n \alpha_i y_i = 0
$$

**3. 得到对偶问题**：将上述结果代回拉格朗日函数，得到只关于 $\boldsymbol{\alpha}$ 的最大化问题：
$$
\max_{\boldsymbol{\alpha}} \quad \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j \\
\text{s.t.} \quad \sum_{i=1}^n \alpha_i y_i = 0, \quad \alpha_i \geq 0
$$

**关键洞察**:

* 最终模型的权重 $\mathbf{w}$ 只是输入数据 $\mathbf{x}$ 的线性组合。
* 在KKT条件中，有一个互补松弛条件 $\alpha_i [y_i(\mathbf{w}^T\mathbf{x}_i + b) - 1] = 0$。它意味着，只有当数据点在间隔边界上时（即 $y_i(\mathbf{w}^T\mathbf{x}_i + b) = 1$），对应的 $\alpha_i$ 才可能大于0。这些点就是**支持向量**！
* 求解SVM最终变成了求解稀疏的拉格朗日乘子 $\boldsymbol{\alpha}$。

### 软间隔SVM

硬间隔要求所有点都必须被正确分类，对噪声和异常值非常敏感。**软间隔 (Soft Margin)** 允许一些点越过间隔边界甚至被错误分类。

我们引入**松弛变量** $\xi_i \geq 0$ 和**惩罚参数** $C > 0$。

**新的原问题**:
$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i \\
\text{s.t.} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
$$

**关键洞察**:

* $C$ 是一个超参数，用于权衡**间隔大小**和**分类错误**。
  * **$C$很大**：对误分类的惩罚很重，模型趋向于硬间隔，容易过拟合。
  * **$C$很小**：对误分类容忍度高，间隔更大，模型可能欠拟合。

其对偶问题与硬间隔非常相似，只是 $\alpha_i$ 多了一个上界：
$$
0 \leq \alpha_i \leq C
$$

### 核心武器：核技巧 (Kernel Trick)

当数据线性不可分时，SVM的真正威力才显现出来。

**思想**：将数据从原始特征空间映射到一个更高维的特征空间，让它在这个高维空间里变得线性可分。

**问题**：直接计算高维映射 $\phi(\mathbf{x})$ 可能非常复杂，甚至是无限维的，计算成本极高。

**核技巧的魔力**：观察SVM的对偶问题，我们发现数据点总是以内积的形式出现 ($\mathbf{x}_i^T \mathbf{x}_j$) 。核技巧允许我们**只定义一个核函数 K**，它的计算结果等于数据点在高维空间中的内积：
$$
K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)
$$
我们**不需要知道具体的映射函数 $\phi$ 是什么**，就可以完成高维空间中的计算！

**对偶问题（使用核函数）**:
$$
\max_{\boldsymbol{\alpha}} \quad \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j) \\
\text{s.t.} \quad \sum_{i=1}^n \alpha_i y_i = 0, \quad 0 \leq \alpha_i \leq C
$$

**常用核函数**:

1. **线性核**：$K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j$ (就是原始的SVM)
2. **多项式核**：$K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T \mathbf{x}_j + r)^d$
3. **高斯核 (RBF)**：$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$ (最常用，能映射到无限维空间，对各种数据都有很好的表现)
4. **Sigmoid 核**：$K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\gamma \mathbf{x}_i^T \mathbf{x}_j + r)$

---

## 实战演练：MNIST手写数字分类

理论学完了，不动手等于白学！现在我们就用学到的SVM知识，来挑战经典的MNIST手写数字识别任务。这个项目将带我们走完一个完整的机器学习流程。

### 0. 准备工作：导入必要的库

首先，把我们需要的工具都准备好。这里我们使用 `scikit-learn` 这个强大的机器学习库，它内置了高效的SVM实现（基于SMO算法）和各种评估工具。

```python
import time
import numpy as np
import matplotlib.pyplot as plt

# Scikit-learn 工具集
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
```

### 1. 加载并准备数据

我们会从 `openml` 上加载完整的 MNIST 数据集。

**重要提示**：完整的MNIST数据集有70000张图片，直接在上面训练SVM（尤其是带核函数的）会非常非常慢。为了能快速地看到结果，我们只取一小部分数据（例如，6000张用于训练，1000张用于测试）来进行实验。这在实际项目中也是常见的做法，先用小数据集快速迭代，找到方向后再用全部数据进行最终训练。

```python
# 加载 MNIST 数据集
print("开始加载 MNIST 数据集...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
print("数据集加载完成！")

# X 是像素数据 (784个特征)，y 是数字标签 (0-9)
X = mnist.data
y = mnist.target.astype(int) # 转换为整数

# --- 使用数据子集以加快训练速度 ---
# 从70000个样本中随机抽取7000个
X_subset, _, y_subset, _ = train_test_split(X, y, train_size=7000, stratify=y, random_state=42)

# 将子集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset, test_size=1000, stratify=y_subset, random_state=42
)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# --- 数据标准化 ---
# 为什么要标准化？因为SVM对特征的尺度非常敏感。
# 像素值范围是0-255，如果不处理，数值大的特征会主导模型。
# 标准化后，所有特征的均值为0，方差为1。
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2. 不同核函数的SVM比较

接下来，我们用线性核、RBF核和多项式核分别训练一个SVM，看看它们的性能和速度如何。

```python
# 定义要比较的核函数
kernels = ['linear', 'rbf', 'poly']
results = {}

for kernel in kernels:
    print(f"\n--- 正在训练核函数: {kernel} ---")
    start_time = time.time()
    
    # 创建并训练SVM分类器
    # C=1 是一个常用的默认惩罚参数
    # gamma='scale' 是一个不错的默认值，它会根据特征数量自动调整
    # degree=3 是多项式核常用的次数
    model = SVC(kernel=kernel, C=1, gamma='scale', degree=3, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    end_time = time.time()
    
    # 在测试集上评估模型
    accuracy = model.score(X_test_scaled, y_test)
    
    # 记录结果
    results[kernel] = {
        'accuracy': accuracy,
        'training_time': end_time - start_time
    }
    
    print(f"训练耗时: {results[kernel]['training_time']:.2f} 秒")
    print(f"测试集准确率: {results[kernel]['accuracy']:.4f}")

# 打印最终比较结果
print("\n--- 不同核函数性能比较 ---")
for kernel, result in results.items():
    print(f"核: {kernel:<8} | 准确率: {result['accuracy']:.4f} | 耗时: {result['training_time']:.2f}s")
```

**初步结论**：
通常你会发现：

* **线性核 (`linear`)**：训练速度最快，但准确率可能不是最高的。
* **RBF核 (`rbf`)**：通常准确率最高，但训练时间比线性核长。
* **多项式核 (`poly`)**：训练时间最长，效果不一定比RBF好。

基于这个结果，我们选择最有潜力的 **RBF 核**进行下一步的超参数调优。

### 3. 超参数调优 (Grid Search)

SVM的性能很大程度上取决于超参数 $C$ 和 $\gamma$。我们将使用网格搜索（Grid Search）来系统地寻找最佳组合。

```python
print("\n--- 开始对 RBF 核进行超参数调优 (网格搜索) ---")
# 定义要搜索的参数网格
# C: 惩罚系数。值越大，模型越容易过拟合。
# gamma: RBF核的系数。值越大，决策边界越复杂，也越容易过拟合。
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1]
}

# 创建一个RBF核的SVM模型
svc_rbf = SVC(kernel='rbf', random_state=42)

# 创建GridSearchCV对象
# cv=3 表示进行3折交叉验证
# n_jobs=-1 表示使用所有可用的CPU核心来并行计算，大大加快速度
grid_search = GridSearchCV(svc_rbf, param_grid, cv=3, n_jobs=-1, verbose=2)

# 在训练集上进行搜索
start_time = time.time()
grid_search.fit(X_train_scaled, y_train)
end_time = time.time()

print(f"网格搜索耗时: {end_time - start_time:.2f} 秒")

# 打印最佳参数和对应的分数
print(f"找到的最佳参数: {grid_search.best_params_}")
print(f"交叉验证最佳准确率: {grid_search.best_score_:.4f}")

# 获取我们最终的最佳模型
best_svm = grid_search.best_estimator_
```

### 4. 最终模型评估

现在我们有了调优后的最佳模型，让我们在从未见过的测试集上对它进行一次全面的评估。

```python
print("\n--- 使用最佳模型在测试集上进行最终评估 ---")

# 使用最佳模型进行预测
y_pred = best_svm.predict(X_test_scaled)

# 打印详细的分类报告
# precision (精确率): 预测为正的样本中，有多少是真的正样本。
# recall (召回率): 真实为正的样本中，有多少被预测为正。
# f1-score: 精确率和召回率的调和平均数。
print("分类报告:")
print(classification_report(y_test, y_pred))

# 可视化混淆矩阵
# 对角线上的数字表示预测正确的样本数，颜色越深越好。
# 非对角线上的数字表示预测错误的样本数，是我们关心的重点。
print("正在绘制混淆矩阵...")
cm = confusion_matrix(y_test, y_pred, labels=best_svm.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_svm.classes_)

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap='Blues')
plt.title("最终模型的混淆矩阵")
plt.show()
```

### 5. 可视化错误分类样本

光看数字还不够直观，让我们把一些模型搞错的样本画出来，看看它到底错在了哪里。这有助于我们理解模型的弱点。

```python
# 找到预测错误的样本的索引
misclassified_indices = np.where(y_pred != y_test)[0]

# 随机选择一些错误样本进行可视化
num_samples_to_show = 9
if len(misclassified_indices) < num_samples_to_show:
    num_samples_to_show = len(misclassified_indices)

# 确保有错误样本可供显示
if num_samples_to_show > 0:
    random_indices = np.random.choice(misclassified_indices, num_samples_to_show, replace=False)

    plt.figure(figsize=(10, 10))
    plt.suptitle("错误分类样本示例", fontsize=16)

    for i, index in enumerate(random_indices):
        plt.subplot(3, 3, i + 1)
        # 将一维像素数据变回 28x28 的图像
        image = X_test[index].reshape(28, 28)
        plt.imshow(image, cmap='gray_r')
        
        predicted_label = y_pred[index]
        true_label = y_test[index]
        
        plt.title(f"预测: {predicted_label}, 真实: {true_label}")
        plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
else:
    print("模型在测试集上没有分错的样本！太棒了！")
```

---

## 学习总结

### 容易踩的坑

* **忘记数据标准化**：SVM对特征的尺度非常敏感，尤其是RBF核。使用前必须进行标准化或归一化。
* **超参数选择**：$C$ 和 $\gamma$ 的选择对模型性能至关重要，需要通过交叉验证来仔细调优。
* **核函数选择**：没有通用的最佳核函数。通常可以从RBF核开始尝试，但线性核在特征维度很高时可能更快更好。
* **计算成本**：对于超大规模的数据集（几十万样本以上），SVM的训练会非常慢，因为核矩阵的计算是 $O(n^2)$。

### 学习感悟

SVM 是一个非常优雅的算法，它将几何直觉（最大化间隔）与严谨的凸优化理论完美结合。核技巧更是神来之笔，为解决非线性问题提供了一个高效而强大的框架。理解SVM的过程，也是对拉格朗日对偶性、KKT条件、凸优化等数学知识的一次绝佳复习。虽然现在深度学习很流行，但SVM在很多场景下依然是一个强大且值得信赖的基线模型。完成这个从理论到实战的项目，让我对机器学习的整个流程有了更深刻的理解。
