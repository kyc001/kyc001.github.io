---
title: 线性回归与逻辑回归
published: 2025-7-26 16:30:22
slug: Linear-Regression-and-Logistic-Regression
tags: ['深度学习', '机器学习', '统计学']
category: '学习笔记'
draft: false
image: ./bg.jpg
---

## 线性回归与逻辑回归

## 前言

最近在学机器学习，发现线性回归和逻辑回归是最基础但也很重要的两个算法。虽然名字很像，但其实用途完全不同。这里整理一下学习过程中的理解和心得。

## 基础知识回顾

### 什么是线性函数？

就是我们高中学过的那种直线方程，只不过现在扩展到多维：

$$
y = w_1x_1 + w_2x_2 + ... + w_dx_d + b
$$

用向量表示就是：$y = \mathbf{w}^T\mathbf{x} + b$

这里w是权重，b是偏置。权重决定了每个特征的重要性，偏置相当于y轴截距。

### 损失函数的概念

简单说就是衡量我们预测得有多准。预测值和真实值差得越远，损失就越大。

### 梯度下降

这个概念一开始有点难理解。想象你在一个山坡上，想找到最低点，但是眼睛被蒙住了。你能做的就是感受脚下的坡度，然后往最陡的下坡方向走一小步，重复这个过程直到走到最低点。

数学表达：

$$
w_{new} = w_{old} - \alpha \times \text{梯度}
$$

其中α是学习率，控制每次走多大的步子。

---

## 线性回归

### 核心思想

线性回归就是用一条直线（或者高维空间中的超平面）来拟合数据点。比如预测房价，我们可能用房子面积、房间数等特征，假设房价和这些特征之间存在线性关系。

目标很简单：找到一条最合适的直线，让所有数据点到这条直线的距离之和最小。

### 深入数学原理

#### 损失函数的选择与推导

均方误差(MSE)不是随便选的，它有深刻的统计学基础：

$$
L(\mathbf{w}) = \frac{1}{2n} \sum_{i=1}^n (y_i - \mathbf{w}^T\mathbf{x}_i)^2
$$

**为什么是MSE？**

1. **最大似然估计**：假设噪声服从高斯分布 $\epsilon \sim N(0, \sigma^2)$，那么：

   $$
   y = \mathbf{w}^T\mathbf{x} + \epsilon
   $$

   似然函数：

   $$
   L(\mathbf{w}) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \mathbf{w}^T\mathbf{x}_i)^2}{2\sigma^2}\right)
   $$

   取对数并最大化，等价于最小化MSE！

2. **几何解释**：MSE对应欧几里得距离的平方，在几何上就是找到最小二乘解

#### 正规方程的完整推导

设计矩阵 $X \in \mathbb{R}^{n \times (d+1)}$，目标向量 $\mathbf{y} \in \mathbb{R}^n$：

$$
L(\mathbf{w}) = \frac{1}{2n} \|\mathbf{y} - X\mathbf{w}\|^2
$$

求导：

$$
\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{n} X^T(X\mathbf{w} - \mathbf{y}) = 0
$$

解得：

$$
\mathbf{w}^* = (X^TX)^{-1}X^T\mathbf{y}
$$

**关键洞察**：

- $X^TX$ 是Gram矩阵，衡量特征间的相关性
- $(X^TX)^{-1}X^T$ 是Moore-Penrose伪逆
- 当 $X^TX$ 不可逆时（特征共线性），需要正则化

#### 梯度下降的收敛性分析

MSE是凸函数，梯度下降保证收敛到全局最优。收敛速度取决于条件数：

$$
\kappa(X^TX) = \frac{\lambda_{\max}}{\lambda_{\min}}
$$

- 条件数大：收敛慢，需要小学习率
- 条件数小：收敛快，可以用大学习率

**自适应学习率**：

$$
\alpha_t = \frac{\alpha_0}{1 + \beta t}
$$

#### 统计性质深入分析

**无偏性**：

$$
\mathbb{E}[\hat{\mathbf{w}}] = \mathbb{E}[(X^TX)^{-1}X^T\mathbf{y}] = (X^TX)^{-1}X^T\mathbb{E}[\mathbf{y}] = \mathbf{w}
$$

**方差**：

$$
\text{Var}(\hat{\mathbf{w}}) = \sigma^2(X^TX)^{-1}
$$

这告诉我们：

- 特征越相关，方差越大（多重共线性问题）
- 样本越多，方差越小
- 噪声越大，方差越大

**置信区间**：

$$
\hat{w}_j \pm t_{\alpha/2, n-p-1} \cdot \hat{\sigma} \sqrt{(X^TX)^{-1}_{jj}}
$$

#### 几何解释：投影视角

线性回归实际上是在做正交投影：

$$
\hat{\mathbf{y}} = X\mathbf{w} = X(X^TX)^{-1}X^T\mathbf{y} = P\mathbf{y}
$$

其中 $P = X(X^TX)^{-1}X^T$ 是投影矩阵，将 $\mathbf{y}$ 投影到 $X$ 的列空间上。

**投影矩阵性质**：

- $P^2 = P$（幂等性）
- $P^T = P$（对称性）
- $\text{rank}(P) = \text{rank}(X)$

残差向量 $\mathbf{r} = \mathbf{y} - \hat{\mathbf{y}} = (I - P)\mathbf{y}$ 与列空间正交。

### 线性回归的特点

1. **简单直观**：结果容易解释，每个特征的权重就代表了它的重要性
2. **计算快速**：有闭式解，或者梯度下降收敛很快
3. **假设强**：假设特征和目标之间是线性关系
4. **对异常值敏感**：因为用的是平方误差，异常值会被放大

---

## 逻辑回归

### 从线性到概率：逻辑回归的数学基础

#### 问题的本质

分类问题的核心是建模条件概率 $P(y=1|\mathbf{x})$。直接用线性回归会有问题：

1. 输出可能超出[0,1]范围
2. 概率的边际效应应该是非线性的（接近0或1时变化缓慢）

#### Logit变换的深层原理

考虑几率(odds)：$\text{odds} = \frac{P(y=1|\mathbf{x})}{P(y=0|\mathbf{x})} = \frac{p}{1-p}$

对数几率(log-odds)：$\text{logit}(p) = \log\frac{p}{1-p}$

**关键洞察**：logit函数将概率空间[0,1]映射到实数空间(-∞,+∞)，这样我们就可以用线性模型：

$$
\log\frac{P(y=1|\mathbf{x})}{P(y=0|\mathbf{x})} = \mathbf{w}^T\mathbf{x} + b
$$

反解得到：

$$
P(y=1|\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}} = \sigma(\mathbf{w}^T\mathbf{x} + b)
$$

#### Sigmoid函数的深入分析

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**重要性质**：

1. **导数性质**：$\sigma'(z) = \sigma(z)(1-\sigma(z))$
2. **对称性**：$\sigma(-z) = 1 - \sigma(z)$
3. **单调性**：严格单调递增
4. **渐近性**：$\lim_{z \to -\infty} \sigma(z) = 0$，$\lim_{z \to +\infty} \sigma(z) = 1$

**为什么选择Sigmoid？**

1. **自然的概率解释**：来自logistic分布的CDF
2. **良好的数值性质**：处处可导，梯度计算简单
3. **生物学意义**：神经元激活的合理模型

#### 最大似然估计推导

假设样本独立，伯努利分布：

$$
P(y_i|\mathbf{x}_i) = p_i^{y_i}(1-p_i)^{1-y_i}
$$

其中 $p_i = \sigma(\mathbf{w}^T\mathbf{x}_i + b)$

似然函数：

$$
L(\mathbf{w}) = \prod_{i=1}^n p_i^{y_i}(1-p_i)^{1-y_i}
$$

对数似然：

$$
\ell(\mathbf{w}) = \sum_{i=1}^n [y_i \log p_i + (1-y_i) \log(1-p_i)]
$$

**交叉熵损失**就是负对数似然：

$$
J(\mathbf{w}) = -\frac{1}{n} \sum_{i=1}^n [y_i \log p_i + (1-y_i) \log(1-p_i)]
$$

#### 梯度计算的优雅性

对权重求偏导：

$$
\frac{\partial J}{\partial w_j} = \frac{1}{n} \sum_{i=1}^n (p_i - y_i) x_{ij}
$$

**惊人的简洁性**！梯度形式与线性回归完全一样：误差×特征。

这不是巧合，而是指数族分布的一般性质。

#### 决策边界的几何解释

决策边界由 $P(y=1|\mathbf{x}) = 0.5$ 确定，即：

$$
\mathbf{w}^T\mathbf{x} + b = 0
$$

这是一个超平面，将特征空间分为两个区域：

- $\mathbf{w}^T\mathbf{x} + b > 0$：预测为类别1
- $\mathbf{w}^T\mathbf{x} + b < 0$：预测为类别0

**权重的几何意义**：

- $\mathbf{w}$ 是决策边界的法向量
- $||\mathbf{w}||$ 控制边界附近概率变化的陡峭程度
- $b$ 控制边界的位置

#### 多分类扩展：Softmax回归

对于K分类问题，使用softmax函数：

$$
P(y=k|\mathbf{x}) = \frac{e^{\mathbf{w}_k^T\mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^T\mathbf{x}}}
$$

这是logistic回归的自然推广，保持了概率的归一化性质。

---

## 两者的联系和区别

### 相同点

1. **都是线性模型**：核心都是 $w_1x_1 + w_2x_2 + ... + b$
2. **都用梯度下降**：优化方法基本一样
3. **都有明确的数学解释**：不是黑盒模型
4. **计算都比较快**：相比深度学习算法

### 不同点

最大的区别就是用途：

- **线性回归**：预测连续值（房价、温度、销售额...）
- **逻辑回归**：预测概率/分类（是否、会不会、属于哪类...）

技术上的区别：

- **激活函数**：线性回归直接输出，逻辑回归要过Sigmoid
- **损失函数**：线性回归用均方误差，逻辑回归用交叉熵
- **输出解释**：线性回归的输出就是预测值，逻辑回归的输出是概率

### 一个有趣的观察

如果你把逻辑回归的Sigmoid函数去掉，它就变成了线性回归！这说明逻辑回归本质上就是在线性回归的基础上加了一个"概率转换"。

---

## 实际应用举例

### 线性回归的典型场景

**房价预测**

- 输入：面积、房间数、地段、楼层...
- 输出：具体的房价数字
- 为什么用线性回归：房价是连续值，而且很多特征确实和房价呈线性关系

**股票价格预测**

- 输入：历史价格、交易量、技术指标...
- 输出：明天的股价
- 注意：实际效果通常不好，因为股价的影响因素太复杂

**销售额预测**

- 输入：广告投入、季节、促销活动...
- 输出：下个月的销售额

### 逻辑回归的典型场景

**垃圾邮件检测**

- 输入：邮件中的关键词、发件人信息、链接数量...
- 输出：是垃圾邮件的概率
- 为什么用逻辑回归：问题本质是二分类，而且特征和结果的关系相对简单

**医疗诊断辅助**

- 输入：症状、检查结果、病史...
- 输出：患某种疾病的概率
- 优点：医生可以理解每个特征的影响

**信用卡欺诈检测**

- 输入：交易金额、时间、地点、频率...
- 输出：是欺诈交易的概率

### 选择建议

- **预测数值**：用线性回归
- **预测类别**：用逻辑回归
- **需要解释性**：两者都不错，比深度学习好理解
- **数据量不大**：两者都合适
- **特征和目标关系复杂**：考虑更复杂的模型

---

## 代码实现

### 线性回归从零实现

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, max_iters=1000):
        self.lr = learning_rate
        self.max_iters = max_iters
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        # 初始化参数
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for i in range(self.max_iters):
            # 前向传播
            y_pred = self.predict(X)
            
            # 计算损失
            cost = self._compute_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # 计算梯度
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def _compute_cost(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def score(self, X, y):
        """计算R²分数"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X.ravel() + np.random.randn(100)
    
    # 训练模型
    model = LinearRegression(learning_rate=0.1, max_iters=1000)
    model.fit(X, y)
    
    # 预测
    y_pred = model.predict(X)
    
    # 可视化
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.6)
    plt.plot(X, y_pred, color='red', linewidth=2)
    plt.title('线性回归拟合结果')
    plt.xlabel('X')
    plt.ylabel('y')
    
    plt.subplot(1, 2, 2)
    plt.plot(model.cost_history)
    plt.title('损失函数收敛过程')
    plt.xlabel('迭代次数')
    plt.ylabel('MSE')
    
    plt.tight_layout()
    plt.show()
    
    print(f"R² Score: {model.score(X, y):.4f}")
    print(f"权重: {model.weights[0]:.4f}")
    print(f"偏置: {model.bias:.4f}")
```

### 逻辑回归从零实现

```python
class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iters=1000):
        self.lr = learning_rate
        self.max_iters = max_iters
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def _sigmoid(self, z):
        # 防止溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        # 初始化参数
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for i in range(self.max_iters):
            # 前向传播
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(linear_pred)
            
            # 计算损失
            cost = self._compute_cost(y, predictions)
            self.cost_history.append(cost)
            
            # 计算梯度
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            # 更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_pred)
        return [0 if y <= 0.5 else 1 for y in y_pred]
    
    def predict_proba(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_pred)
    
    def _compute_cost(self, y_true, y_pred):
        # 交叉熵损失
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)  # 防止log(0)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def score(self, X, y):
        """计算准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

# 使用示例
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # 生成示例数据
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                             n_informative=2, n_clusters_per_class=1, random_state=42)
    
    # 训练模型
    model = LogisticRegression(learning_rate=0.1, max_iters=1000)
    model.fit(X, y)
    
    # 预测
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # 可视化
    plt.figure(figsize=(15, 5))
    
    # 数据分布
    plt.subplot(1, 3, 1)
    plt.scatter(X[y==0, 0], X[y==0, 1], c='red', marker='o', alpha=0.6, label='类别 0')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', marker='s', alpha=0.6, label='类别 1')
    plt.title('原始数据分布')
    plt.legend()
    
    # 决策边界
    plt.subplot(1, 3, 2)
    plt.scatter(X[y==0, 0], X[y==0, 1], c='red', marker='o', alpha=0.6)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', marker='s', alpha=0.6)
    
    # 绘制决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(mesh_points)
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
    plt.title('决策边界')
    
    # 损失函数
    plt.subplot(1, 3, 3)
    plt.plot(model.cost_history)
    plt.title('损失函数收敛过程')
    plt.xlabel('迭代次数')
    plt.ylabel('交叉熵损失')
    
    plt.tight_layout()
    plt.show()
    
    print(f"准确率: {model.score(X, y):.4f}")
    print(f"权重: {model.weights}")
    print(f"偏置: {model.bias:.4f}")
```

---

## 学习总结

### 容易踩的坑

#### 数据预处理

- **忘记标准化**：特征量纲差异很大时，权重会被带偏
- **没处理异常值**：线性回归对异常值很敏感
- **特征选择不当**：包含了无关特征或者遗漏了重要特征

#### 模型选择

- **线性回归用于分类**：新手容易犯的错误，输出没有概率意义
- **逻辑回归用于回归**：虽然名字叫"回归"，但它是分类算法
- **假设线性关系**：现实中很多关系是非线性的

#### 评估方法

- **只看训练误差**：没有用验证集，容易过拟合
- **阈值选择随意**：逻辑回归的0.5阈值不一定最优
- **忽略业务含义**：模型指标好不代表业务效果好

### 学到的技巧

#### 特征工程

```python
# 多项式特征：把线性模型变成非线性
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
```

#### 正则化

```python
# 防止过拟合
from sklearn.linear_model import Ridge, Lasso
ridge = Ridge(alpha=1.0)  # L2正则化
lasso = Lasso(alpha=1.0)  # L1正则化
```

#### 模型诊断

```python
# 检查残差图
plt.scatter(y_pred, y_true - y_pred)
plt.xlabel('预测值')
plt.ylabel('残差')
# 好的模型残差应该随机分布
```

### 使用场景总结

**什么时候用线性回归？**

- 目标是连续数值
- 特征和目标大致呈线性关系
- 需要快速得到结果
- 需要解释模型

**什么时候用逻辑回归？**

- 二分类问题
- 需要概率输出
- 特征相对简单
- 需要解释性

**什么时候不用这两个？**

- 数据量特别大，关系特别复杂
- 明显的非线性关系
- 图像、文本等复杂数据
- 需要很高的预测精度

### 进一步学习方向

1. **正则化方法**：Ridge、Lasso、Elastic Net
2. **广义线性模型**：泊松回归、伽马回归等
3. **非线性扩展**：多项式回归、样条回归
4. **集成方法**：随机森林、梯度提升
5. **深度学习**：神经网络

---

## 高级话题深入

### 正则化理论

#### 为什么需要正则化？

1. **过拟合问题**：当特征数量接近或超过样本数量时
2. **多重共线性**：特征间高度相关，导致参数不稳定
3. **数值稳定性**：$X^TX$ 接近奇异时，逆矩阵计算不稳定

#### Ridge回归（L2正则化）

$$
J(\mathbf{w}) = \frac{1}{2n} \|\mathbf{y} - X\mathbf{w}\|^2 + \lambda \|\mathbf{w}\|^2
$$

**解析解**：

$$
\mathbf{w}_{ridge} = (X^TX + \lambda I)^{-1}X^T\mathbf{y}
$$

**几何解释**：在权重空间中添加球形约束，解在椭圆等高线与圆的切点。

**统计解释**：等价于对权重施加零均值高斯先验 $\mathbf{w} \sim N(0, \frac{1}{\lambda}I)$

#### Lasso回归（L1正则化）

$$
J(\mathbf{w}) = \frac{1}{2n} \|\mathbf{y} - X\mathbf{w}\|^2 + \lambda \|\mathbf{w}\|_1
$$

**特点**：

- 产生稀疏解（自动特征选择）
- 无解析解，需要迭代算法（如坐标下降）
- 几何上对应菱形约束

#### Elastic Net

结合L1和L2：

$$
J(\mathbf{w}) = \frac{1}{2n} \|\mathbf{y} - X\mathbf{w}\|^2 + \lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \|\mathbf{w}\|^2
$$

平衡了Ridge的稳定性和Lasso的稀疏性。

### 广义线性模型(GLM)理论

#### 指数族分布

形式：$f(y|\theta) = \exp\left(\frac{y\theta - b(\theta)}{a(\phi)} + c(y, \phi)\right)$

其中：

- $\theta$：自然参数
- $b(\theta)$：对数配分函数
- $a(\phi)$：散度参数
- $c(y, \phi)$：归一化常数

#### GLM的三个组成部分

1. **随机成分**：响应变量的分布（指数族）
2. **系统成分**：线性预测器 $\eta = X\mathbf{w}$
3. **链接函数**：$g(\mu) = \eta$，其中 $\mu = E[Y]$

#### 常见GLM实例

| 分布   | 链接函数                               | 应用       |
| ------ | -------------------------------------- | ---------- |
| 高斯   | 恒等 $g(\mu) = \mu$                    | 线性回归   |
| 伯努利 | Logit $g(\mu) = \log\frac{\mu}{1-\mu}$ | 逻辑回归   |
| 泊松   | 对数 $g(\mu) = \log\mu$                | 计数数据   |
| 伽马   | 倒数 $g(\mu) = \frac{1}{\mu}$          | 正偏态数据 |

### 优化算法深入

#### 牛顿法

利用二阶信息加速收敛：

$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - H^{-1} \nabla J
$$

其中 $H$ 是Hessian矩阵。

**优点**：二次收敛
**缺点**：计算Hessian代价高，可能不正定

#### 拟牛顿法（BFGS）

用近似Hessian避免直接计算：

$$
H_{k+1} = H_k + \frac{\mathbf{y}_k\mathbf{y}_k^T}{\mathbf{y}_k^T\mathbf{s}_k} - \frac{H_k\mathbf{s}_k\mathbf{s}_k^T H_k}{\mathbf{s}_k^T H_k \mathbf{s}_k}
$$

#### 坐标下降

逐个优化每个坐标：

$$
w_j^{(t+1)} = \arg\min_{w_j} J(w_1^{(t+1)}, ..., w_{j-1}^{(t+1)}, w_j, w_{j+1}^{(t)}, ..., w_d^{(t)})
$$

特别适合Lasso等非光滑问题。

### 模型诊断与假设检验

#### 线性回归假设检验

1. **线性性**：残差vs拟合值图应随机分布
2. **独立性**：Durbin-Watson检验
3. **同方差性**：Breusch-Pagan检验
4. **正态性**：Shapiro-Wilk检验

#### 逻辑回归诊断

1. **Hosmer-Lemeshow检验**：拟合优度
2. **ROC曲线**：分类性能
3. **Cook距离**：影响点检测
4. **VIF**：多重共线性检验

#### 信息准则

- **AIC**：$-2\log L + 2k$
- **BIC**：$-2\log L + k\log n$
- **交叉验证**：更稳健的模型选择

### 贝叶斯视角

#### 贝叶斯线性回归

先验：$\mathbf{w} \sim N(\mathbf{0}, \alpha^{-1}I)$，$\tau \sim \text{Gamma}(a, b)$

后验：$p(\mathbf{w}|\mathbf{y}) \propto p(\mathbf{y}|\mathbf{w}) p(\mathbf{w})$

**预测分布**：

$$
p(y^*|\mathbf{x}^*, \mathbf{y}) = \int p(y^*|\mathbf{x}^*, \mathbf{w}) p(\mathbf{w}|\mathbf{y}) d\mathbf{w}
$$

提供了预测的不确定性量化。

#### 贝叶斯逻辑回归

由于非共轭性，需要近似推断：

1. **拉普拉斯近似**：用高斯近似后验
2. **变分推断**：优化KL散度
3. **MCMC**：采样方法

### 现代发展

#### 在线学习

**随机梯度下降**：

$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \alpha_t \nabla J_i(\mathbf{w}^{(t)})
$$

**自适应学习率**：

- AdaGrad：$\alpha_t = \frac{\alpha}{\sqrt{\sum_{i=1}^t g_i^2}}$
- Adam：结合动量和自适应学习率

#### 深度学习连接

- **多层感知机**：多个逻辑回归的组合
- **激活函数**：Sigmoid → ReLU → 更复杂函数
- **正则化**：Dropout、Batch Normalization

学习这两个算法最大的收获是理解了机器学习的基本思路：定义问题→选择模型→设计损失函数→优化求解→评估效果。虽然简单，但这个框架适用于几乎所有机器学习算法。更重要的是，它们展示了统计学、优化理论、线性代数如何优雅地结合解决实际问题。
