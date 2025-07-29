---
title: 模型评估和选择
published: 2025-7-29 23:59:59
slug: model-evaluation-and-selection
tags: ['深度学习', '机器学习', '模型评估']
category: '机器学习'
draft: false
image: ./bg.jpg
---

## 模型评估和选择

## 前言

在机器学习的学习路径中，掌握了各种算法后，下一个关键问题就是：**如何知道我的模型好不好？如何在多个模型中选择最优的？** 这就是模型评估和选择的核心问题。

今天深入学习了模型评估的各种指标、交叉验证技术，以及偏差-方差权衡理论。我发现，仅仅知道如何训练模型是远远不够的，更重要的是要能够科学地评估模型性能，理解模型的优缺点，并据此做出合理的模型选择。

这份笔记记录了我对模型评估体系的理解，从基础的混淆矩阵到高级的偏差-方差分解，希望能为后续的机器学习实践提供坚实的评估基础。

## 第一部分：分类问题评估指标

### 1.1 混淆矩阵：理解分类错误的"地图"

#### 1.1.1 混淆矩阵的数学定义

混淆矩阵是评估分类模型性能的基础工具。对于k类分类问题，混淆矩阵是一个k×k的方阵：

$$C_{ij} = \text{预测为类别j但实际为类别i的样本数量}$$

**二分类混淆矩阵的标准形式：**

```text
                预测结果
              正类    负类
真实  正类    TP     FN
标签  负类    FP     TN
```

其中：

- **TP (True Positive)**：真正例 - 预测为正，实际也为正
- **TN (True Negative)**：真负例 - 预测为负，实际也为负  
- **FP (False Positive)**：假正例 - 预测为正，实际为负（第一类错误）
- **FN (False Negative)**：假负例 - 预测为负，实际为正（第二类错误）

#### 1.1.2 从混淆矩阵导出的基本指标

**1. 准确率 (Accuracy)**
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

- **含义**：所有预测中正确预测的比例
- **适用场景**：类别平衡的数据集
- **局限性**：在不平衡数据集上容易产生误导

**2. 精确率 (Precision)**
$$\text{Precision} = \frac{TP}{TP + FP}$$

- **含义**：在所有预测为正的样本中，真正为正的比例
- **直观理解**：模型说"是"的时候，有多少次是对的
- **业务意义**：当误报代价很高时（如垃圾邮件检测），需要高精确率

**3. 召回率 (Recall/Sensitivity)**
$$\text{Recall} = \frac{TP}{TP + FN}$$

- **含义**：在所有真正为正的样本中，被正确预测的比例
- **直观理解**：所有真正的"阳性"中，模型找到了多少
- **业务意义**：当漏报代价很高时（如疾病诊断），需要高召回率

**4. 特异性 (Specificity)**
$$\text{Specificity} = \frac{TN}{TN + FP}$$

- **含义**：在所有真正为负的样本中，被正确预测的比例
- **直观理解**：模型正确识别"阴性"的能力

**5. F1分数**
$$F1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

- **含义**：精确率和召回率的调和平均数
- **优势**：平衡考虑精确率和召回率，对不平衡数据集较为稳健

#### 1.1.3 实现完整的评估器

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
from sklearn.model_selection import learning_curve, validation_curve

class ModelEvaluator:
    """完整的模型评估器"""
    
    def __init__(self):
        pass
    
    def plot_confusion_matrix(self, y_true, y_pred, classes=None, normalize=False):
        """绘制混淆矩阵并提供详细分析"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = '标准化混淆矩阵'
            fmt = '.2f'
        else:
            cm_display = cm
            title = '混淆矩阵'
            fmt = 'd'
        
        # 创建子图：原始矩阵和标准化矩阵
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 原始混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes, ax=axes[0])
        axes[0].set_title('原始混淆矩阵')
        axes[0].set_ylabel('真实标签')
        axes[0].set_xlabel('预测标签')
        
        # 标准化混淆矩阵
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=classes, yticklabels=classes, ax=axes[1])
        axes[1].set_title('标准化混淆矩阵 (按行归一化)')
        axes[1].set_ylabel('真实标签')
        axes[1].set_xlabel('预测标签')
        
        plt.tight_layout()
        plt.show()
        
        # 详细分析
        self._analyze_confusion_matrix(cm, classes)
        
        return cm
    
    def _analyze_confusion_matrix(self, cm, classes):
        """分析混淆矩阵的详细信息"""
        
        print("=== 混淆矩阵详细分析 ===")
        
        if len(cm) == 2:  # 二分类
            tn, fp, fn, tp = cm.ravel()
            print(f"真负例 (TN): {tn}")
            print(f"假正例 (FP): {fp} - 第一类错误")
            print(f"假负例 (FN): {fn} - 第二类错误")  
            print(f"真正例 (TP): {tp}")
            print(f"总预测错误: {fp + fn} / {cm.sum()}")
            
            # 错误类型分析
            if fp > fn:
                print("主要错误类型: 假正例 (过度预测)")
            elif fn > fp:
                print("主要错误类型: 假负例 (预测不足)")
            else:
                print("假正例和假负例相当")
        else:  # 多分类
            print("各类别预测情况:")
            for i, class_name in enumerate(classes or range(len(cm))):
                correct = cm[i, i]
                total = cm[i, :].sum()
                accuracy = correct / total if total > 0 else 0
                print(f"  {class_name}: {correct}/{total} = {accuracy:.3f}")
    
    def classification_metrics(self, y_true, y_pred, y_prob=None, target_names=None):
        """计算全面的分类指标"""
        
        print("=== 基本分类指标 ===")
        
        # 基本指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        # 详细的分类报告
        print("\n=== 详细分类报告 ===")
        print(classification_report(y_true, y_pred, target_names=target_names))
        
        # AUC相关指标
        if y_prob is not None:
            n_classes = len(np.unique(y_true))
            
            if n_classes == 2:  # 二分类
                # 确保y_prob是概率形式
                if y_prob.ndim == 2:
                    y_prob_binary = y_prob[:, 1]
                else:
                    y_prob_binary = y_prob
                
                auc = roc_auc_score(y_true, y_prob_binary)
                ap = average_precision_score(y_true, y_prob_binary)
                
                print(f"\n=== 概率相关指标 ===")
                print(f"AUC-ROC: {auc:.4f}")
                print(f"平均精确率 (AP): {ap:.4f}")
                
                # 绘制ROC和PR曲线
                self.plot_roc_curve(y_true, y_prob_binary)
                self.plot_precision_recall_curve(y_true, y_prob_binary)
                
            else:  # 多分类
                try:
                    auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
                    print(f"\n=== 多分类概率指标 ===")
                    print(f"多分类AUC-ROC: {auc:.4f}")
                except ValueError as e:
                    print(f"无法计算多分类AUC: {e}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def plot_roc_curve(self, y_true, y_prob):
        """绘制ROC曲线及详细分析"""
        
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(12, 5))
        
        # ROC曲线
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC曲线 (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='随机分类器 (AUC = 0.5)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (FPR) = FP/(FP+TN)')
        plt.ylabel('真正率 (TPR) = TP/(TP+FN)')
        plt.title('ROC曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 阈值分析
        plt.subplot(1, 2, 2)
        plt.plot(thresholds, fpr, label='假正率 (FPR)', linewidth=2)
        plt.plot(thresholds, tpr, label='真正率 (TPR)', linewidth=2)
        plt.xlabel('分类阈值')
        plt.ylabel('率')
        plt.title('阈值 vs FPR/TPR')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 最优阈值分析
        optimal_idx = np.argmax(tpr - fpr)  # Youden指数
        optimal_threshold = thresholds[optimal_idx]
        print(f"最优阈值 (Youden指数): {optimal_threshold:.3f}")
        print(f"对应TPR: {tpr[optimal_idx]:.3f}, FPR: {fpr[optimal_idx]:.3f}")
    
    def plot_precision_recall_curve(self, y_true, y_prob):
        """绘制精确率-召回率曲线"""
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        
        # 计算基线（随机预测的AP）
        baseline_ap = np.sum(y_true) / len(y_true)
        
        plt.figure(figsize=(12, 5))
        
        # PR曲线
        plt.subplot(1, 2, 1)
        plt.plot(recall, precision, linewidth=2, label=f'PR曲线 (AP = {ap:.3f})')
        plt.axhline(y=baseline_ap, color='k', linestyle='--', 
                   linewidth=1, label=f'随机分类器 (AP = {baseline_ap:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('召回率 (Recall)')
        plt.ylabel('精确率 (Precision)')
        plt.title('精确率-召回率曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # F1分数 vs 阈值
        plt.subplot(1, 2, 2)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
        plt.plot(thresholds, f1_scores, linewidth=2, label='F1分数')
        optimal_f1_idx = np.argmax(f1_scores)
        optimal_f1_threshold = thresholds[optimal_f1_idx]
        plt.axvline(x=optimal_f1_threshold, color='r', linestyle='--', 
                   label=f'最优F1阈值 = {optimal_f1_threshold:.3f}')
        plt.xlabel('分类阈值')
        plt.ylabel('F1分数')
        plt.title('F1分数 vs 阈值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"最优F1阈值: {optimal_f1_threshold:.3f}")
        print(f"最大F1分数: {f1_scores[optimal_f1_idx]:.3f}")
```

### 1.2 ROC曲线和AUC：理解分类器的判别能力

#### 1.2.1 ROC曲线的数学基础

ROC (Receiver Operating Characteristic) 曲线是在不同分类阈值下，真正率(TPR)对假正率(FPR)的函数图像。

**数学定义：**

- **TPR (True Positive Rate)** = $\frac{TP}{TP + FN}$ = Recall = Sensitivity
- **FPR (False Positive Rate)** = $\frac{FP}{FP + TN}$ = 1 - Specificity

**AUC (Area Under Curve) 的含义：**
AUC等于从正类和负类中各随机选择一个样本，分类器给正类样本的评分高于负类样本评分的概率。

$$\text{AUC} = P(S_+ > S_- | \text{随机选择正负样本})$$

**AUC的性质：**

- AUC ∈ [0, 1]
- AUC = 0.5：随机分类器
- AUC = 1：完美分类器
- AUC > 0.5：比随机分类好
- AUC < 0.5：比随机分类差（可以反转预测）

#### 1.2.2 PR曲线 vs ROC曲线

**选择原则：**

- **平衡数据集**：ROC曲线和PR曲线都有效
- **不平衡数据集**：PR曲线更能反映真实性能

**数学原因：**
在极不平衡数据集中（如正类占1%），即使FPR很小，FP的绝对数量可能很大，导致精确率很低，但ROC曲线看起来仍然不错。

**实例比较：**
假设数据集：990个负例，10个正例

- 模型预测：8个真正例，2个假负例，100个假正例，890个真负例
- ROC角度：TPR = 8/10 = 0.8，FPR = 100/990 = 0.101 (看起来不错)
- PR角度：Precision = 8/108 = 0.074 (很差!)

### 1.3 多分类评估的特殊考虑

#### 1.3.1 平均策略

对于多分类问题，需要选择合适的平均策略：

**1. Macro平均**
$$\text{Macro-F1} = \frac{1}{k}\sum_{i=1}^{k} F1_i$$

- 每个类别权重相等
- 适用于关心每个类别性能的场景

**2. Weighted平均**
$$\text{Weighted-F1} = \sum_{i=1}^{k} w_i \times F1_i, \quad w_i = \frac{n_i}{n}$$

- 按类别样本数量加权
- 适用于类别不平衡但更关心大类别的场景

**3. Micro平均**
$$\text{Micro-F1} = \frac{2 \times \sum_{i=1}^{k} TP_i}{2 \times \sum_{i=1}^{k} TP_i + \sum_{i=1}^{k} FP_i + \sum_{i=1}^{k} FN_i}$$

- 聚合所有类别的TP、FP、FN后计算
- 等价于准确率

## 第二部分：回归问题评估指标

### 2.1 常用回归指标

#### 2.1.1 基本误差指标

**1. 平均绝对误差 (MAE)**
$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

- **优点**：直观易懂，与目标变量同单位，对异常值不敏感
- **缺点**：不可微分，优化困难

**2. 均方误差 (MSE)**
$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

- **优点**：可微分，便于优化，惩罚大误差
- **缺点**：单位是目标变量的平方，对异常值敏感

**3. 均方根误差 (RMSE)**
$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

- **优点**：与目标变量同单位，兼具MSE的可微性
- **缺点**：仍对异常值敏感

**4. 决定系数 (R²)**
$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$$

- **含义**：模型解释的方差占总方差的比例
- **范围**：(-∞, 1]，1表示完美拟合
- **优点**：无量纲，便于不同问题间比较

#### 2.1.2 高级回归指标

**5. 平均绝对百分比误差 (MAPE)**
$$\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

- **适用**：相对误差更重要的场景
- **局限**：当真实值接近0时不稳定

**6. 对称平均绝对百分比误差 (sMAPE)**
$$\text{sMAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}$$

- **改进**：解决MAPE在真实值为0时的问题

#### 2.1.3 回归评估的完整实现

```python
def regression_metrics(y_true, y_pred, multioutput='uniform_average'):
    """回归问题的完整评估"""
    
    # 基本指标
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # 额外指标
    evs = explained_variance_score(y_true, y_pred)  # 解释方差得分
    max_error = max_error(y_true, y_pred)  # 最大误差
    
    print("=== 回归评估指标 ===")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    print(f"解释方差得分: {evs:.4f}")
    print(f"最大误差: {max_error:.4f}")
    
    # 计算百分比误差（避免除零）
    mask = y_true != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")
    
    # 残差分析
    residuals = y_true - y_pred
    
    plt.figure(figsize=(15, 10))
    
    # 1. 预测值 vs 真实值
    plt.subplot(2, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测线')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title(f'预测值 vs 真实值\n(R² = {r2:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 残差图
    plt.subplot(2, 3, 2)
    plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('预测值')
    plt.ylabel('残差 (真实值 - 预测值)')
    plt.title('残差图')
    plt.grid(True, alpha=0.3)
    
    # 检查同方差性
    # 计算残差的绝对值与预测值的相关性
    abs_residuals = np.abs(residuals)
    heteroscedasticity = np.corrcoef(y_pred, abs_residuals)[0, 1]
    plt.text(0.05, 0.95, f'异方差性检验\n相关系数: {heteroscedasticity:.3f}', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # 3. 残差分布
    plt.subplot(2, 3, 3)
    plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('残差')
    plt.ylabel('频数')
    plt.title('残差分布')
    plt.grid(True, alpha=0.3)
    
    # 正态性检验
    from scipy import stats
    shapiro_stat, shapiro_p = stats.shapiro(residuals[:min(5000, len(residuals))])  # 限制样本数
    plt.text(0.05, 0.95, f'Shapiro-Wilk正态性检验\np-value: {shapiro_p:.3f}', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='lightblue'))
    
    # 4. Q-Q图
    plt.subplot(2, 3, 4)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q图 (正态性检验)')
    plt.grid(True, alpha=0.3)
    
    # 5. 误差随索引变化（检查时间序列相关性）
    plt.subplot(2, 3, 5)
    plt.plot(residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('样本索引')
    plt.ylabel('残差')
    plt.title('残差序列图')
    plt.grid(True, alpha=0.3)
    
    # 6. 残差绝对值
    plt.subplot(2, 3, 6)
    plt.scatter(y_pred, abs_residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
    plt.xlabel('预测值')
    plt.ylabel('|残差|')
    plt.title('绝对残差图')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 残差分析总结
    print("\n=== 残差分析总结 ===")
    print(f"残差均值: {np.mean(residuals):.6f} (应接近0)")
    print(f"残差标准差: {np.std(residuals):.4f}")
    print(f"残差偏度: {stats.skew(residuals):.4f} (应接近0)")
    print(f"残差峰度: {stats.kurtosis(residuals):.4f} (应接近0)")
    
    if abs(heteroscedasticity) > 0.3:
        print("⚠️  警告：可能存在异方差性")
    if shapiro_p < 0.05:
        print("⚠️  警告：残差可能不服从正态分布")
    
    return {
        'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2,
        'explained_variance': evs, 'max_error': max_error
    }
```

## 第三部分：交叉验证和模型选择

### 3.1 交叉验证的数学基础

#### 3.1.1 K折交叉验证

**算法步骤：**

1. 将数据集D随机分成k个大小相等的子集：$D_1, D_2, \ldots, D_k$
2. 对于每个子集$D_i$，用其他k-1个子集训练模型，在$D_i$上测试
3. 计算k个测试结果的平均值作为最终估计

**数学表示：**
$$\text{CV}_k = \frac{1}{k}\sum_{i=1}^{k} L(f^{(-i)}, D_i)$$

其中$f^{(-i)}$表示在除$D_i$外的数据上训练的模型，$L$是损失函数。

**方差估计：**
$$\text{Var}(\text{CV}_k) = \frac{1}{k}\sum_{i=1}^{k}(L_i - \text{CV}_k)^2$$

#### 3.1.2 不同交叉验证策略的比较

##### 1. 留一交叉验证 (LOOCV)

- k = n（样本数量）
- **优点**：几乎无偏估计，充分利用数据
- **缺点**：计算成本高，方差大

##### 2. 分层交叉验证

- 保持各折中类别分布与原数据集一致
- **适用**：不平衡数据集

##### 3. 时间序列交叉验证

- 考虑时间顺序，避免数据泄露
- **方法**：滑动窗口、扩展窗口

#### 3.1.3 实现全面的交叉验证分析

```python
from sklearn.model_selection import *
import pandas as pd

def comprehensive_cross_validation(models, X, y, cv_strategies=None, scoring_metrics=None):
    """全面的交叉验证分析"""
    
    if cv_strategies is None:
        cv_strategies = {
            'KFold-5': KFold(n_splits=5, shuffle=True, random_state=42),
            'KFold-10': KFold(n_splits=10, shuffle=True, random_state=42),
            'StratifiedKFold-5': StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        }
    
    if scoring_metrics is None:
        # 根据问题类型选择指标
        unique_targets = len(np.unique(y))
        if unique_targets <= 20:  # 分类问题
            scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        else:  # 回归问题
            scoring_metrics = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n=== {model_name} ===")
        model_results = {}
        
        for cv_name, cv_strategy in cv_strategies.items():
            print(f"\n{cv_name} 交叉验证:")
            cv_results = {}
            
            for metric in scoring_metrics:
                scores = cross_val_score(model, X, y, cv=cv_strategy, 
                                       scoring=metric, n_jobs=-1)
                
                cv_results[metric] = {
                    'scores': scores,
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'ci_lower': scores.mean() - 1.96 * scores.std(),
                    'ci_upper': scores.mean() + 1.96 * scores.std()
                }
                
                print(f"  {metric}: {scores.mean():.4f} (±{scores.std():.4f})")
                print(f"    95% CI: [{cv_results[metric]['ci_lower']:.4f}, "
                      f"{cv_results[metric]['ci_upper']:.4f}]")
            
            model_results[cv_name] = cv_results
        
        results[model_name] = model_results
    
    # 可视化比较
    plot_cv_comparison(results, scoring_metrics[0])  # 使用第一个指标绘图
    
    return results

def plot_cv_comparison(cv_results, primary_metric):
    """可视化交叉验证结果比较"""
    
    # 准备数据
    models = list(cv_results.keys())
    cv_strategies = list(cv_results[models[0]].keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    # 1. 不同模型在不同CV策略下的表现
    ax = axes[0]
    x_pos = np.arange(len(models))
    width = 0.25
    
    for i, cv_name in enumerate(cv_strategies):
        means = []
        stds = []
        for model in models:
            result = cv_results[model][cv_name][primary_metric]
            means.append(result['mean'])
            stds.append(result['std'])
        
        ax.bar(x_pos + i * width, means, width, yerr=stds, 
               label=cv_name, capsize=5, alpha=0.8)
    
    ax.set_xlabel('模型')
    ax.set_ylabel(primary_metric)
    ax.set_title(f'不同交叉验证策略下的{primary_metric}比较')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 各模型得分分布箱线图
    ax = axes[1]
    all_scores = []
    labels = []
    
    for model in models:
        for cv_name in cv_strategies:
            scores = cv_results[model][cv_name][primary_metric]['scores']
            all_scores.append(scores)
            labels.append(f"{model}\n{cv_name}")
    
    box_plot = ax.boxplot(all_scores, labels=labels, patch_artist=True)
    ax.set_title('各模型交叉验证得分分布')
    ax.set_ylabel(primary_metric)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # 3. 模型稳定性分析（标准差）
    ax = axes[2]
    x_pos = np.arange(len(models))
    
    for i, cv_name in enumerate(cv_strategies):
        stds = []
        for model in models:
            result = cv_results[model][cv_name][primary_metric]
            stds.append(result['std'])
        
        ax.bar(x_pos + i * width, stds, width, 
               label=cv_name, alpha=0.8)
    
    ax.set_xlabel('模型')
    ax.set_ylabel(f'{primary_metric} 标准差')
    ax.set_title('模型稳定性比较 (标准差越小越稳定)')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 置信区间比较
    ax = axes[3]
    y_pos = np.arange(len(models) * len(cv_strategies))
    ci_ranges = []
    means = []
    y_labels = []
    
    for model in models:
        for cv_name in cv_strategies:
            result = cv_results[model][cv_name][primary_metric]
            ci_ranges.append(result['ci_upper'] - result['ci_lower'])
            means.append(result['mean'])
            y_labels.append(f"{model}-{cv_name}")
    
    ax.barh(y_pos, ci_ranges, alpha=0.7)
    ax.set_xlabel('置信区间宽度')
    ax.set_ylabel('模型-CV策略')
    ax.set_title('置信区间宽度比较 (宽度越小越可靠)')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def statistical_significance_test(cv_results, model1, model2, cv_strategy, metric):
    """统计显著性检验"""
    
    scores1 = cv_results[model1][cv_strategy][metric]['scores']
    scores2 = cv_results[model2][cv_strategy][metric]['scores']
    
    # 配对t检验
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(scores1, scores2)
    
    print(f"\n=== {model1} vs {model2} 统计显著性检验 ===")
    print(f"检验方法: 配对t检验")
    print(f"t统计量: {t_stat:.4f}")
    print(f"p值: {p_value:.4f}")
    
    alpha = 0.05
    if p_value < alpha:
        winner = model1 if scores1.mean() > scores2.mean() else model2
        print(f"结论: {winner} 显著优于另一模型 (α = {alpha})")
    else:
        print(f"结论: 两模型无显著差异 (α = {alpha})")
    
    return t_stat, p_value
```

### 3.2 学习曲线和验证曲线

#### 3.2.1 学习曲线分析

学习曲线显示模型性能随训练样本数量的变化，用于诊断：

- **欠拟合**：训练和验证曲线都较低且接近
- **过拟合**：训练曲线高，验证曲线低，差距大
- **理想状态**：两条曲线都较高且接近

```python
def comprehensive_learning_curve_analysis(model, X, y, cv=5):
    """全面的学习曲线分析"""
    
    # 计算学习曲线
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy',
        random_state=42
    )
    
    # 计算统计量
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(15, 5))
    
    # 1. 学习曲线
    plt.subplot(1, 3, 1)
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='训练集')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.1, color='blue')
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='验证集')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.1, color='red')
    plt.xlabel('训练样本数')
    plt.ylabel('准确率')
    plt.title('学习曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 过拟合程度分析
    plt.subplot(1, 3, 2)
    overfitting_gap = train_mean - val_mean
    plt.plot(train_sizes, overfitting_gap, 'o-', color='orange', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('训练样本数')
    plt.ylabel('过拟合程度 (训练分数 - 验证分数)')
    plt.title('过拟合程度分析')
    plt.grid(True, alpha=0.3)
    
    # 3. 方差分析
    plt.subplot(1, 3, 3)
    plt.plot(train_sizes, train_std, 'o-', color='blue', label='训练集方差')
    plt.plot(train_sizes, val_std, 'o-', color='red', label='验证集方差')
    plt.xlabel('训练样本数')
    plt.ylabel('标准差')
    plt.title('方差分析')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 分析结论
    final_gap = overfitting_gap[-1]
    final_val_score = val_mean[-1]
    
    print("=== 学习曲线分析结论 ===")
    print(f"最终验证分数: {final_val_score:.4f}")
    print(f"最终过拟合程度: {final_gap:.4f}")
    
    if final_gap > 0.1:
        print("🔴 检测到过拟合")
        print("建议: 增加正则化、减少模型复杂度或增加训练数据")
    elif final_val_score < 0.7:
        print("🟡 检测到欠拟合")
        print("建议: 增加模型复杂度、添加特征或检查数据质量")
    else:
        print("🟢 模型拟合良好")
    
    # 数据效率分析
    data_efficiency = (val_mean[-1] - val_mean[0]) / (train_sizes[-1] - train_sizes[0])
    print(f"数据效率: {data_efficiency:.6f} 每个样本的性能提升")
    
    return train_sizes, train_scores, val_scores
```

#### 3.2.2 验证曲线分析

```python
def comprehensive_validation_curve_analysis(model, X, y, param_name, param_range, cv=5):
    """全面的验证曲线分析"""
    
    # 计算验证曲线
    train_scores, val_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring='accuracy', n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(15, 5))
    
    # 1. 验证曲线
    plt.subplot(1, 3, 1)
    plt.semilogx(param_range, train_mean, 'o-', color='blue', label='训练集')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                     alpha=0.1, color='blue')
    plt.semilogx(param_range, val_mean, 'o-', color='red', label='验证集')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                     alpha=0.1, color='red')
    plt.xlabel(f'{param_name} (log scale)')
    plt.ylabel('准确率')
    plt.title(f'验证曲线 ({param_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 偏差-方差权衡
    plt.subplot(1, 3, 2)
    bias_proxy = 1 - val_mean  # 用1-验证分数近似偏差
    variance_proxy = val_std   # 用验证分数标准差近似方差
    
    plt.semilogx(param_range, bias_proxy, 'o-', color='red', label='偏差 (近似)')
    plt.semilogx(param_range, variance_proxy, 'o-', color='blue', label='方差 (近似)')
    plt.xlabel(f'{param_name} (log scale)')
    plt.ylabel('误差')
    plt.title('偏差-方差权衡')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 最优参数选择
    plt.subplot(1, 3, 3)
    # 综合考虑验证分数和稳定性
    stability_penalty = val_std / val_mean  # 变异系数
    composite_score = val_mean - stability_penalty  # 综合得分
    
    plt.semilogx(param_range, val_mean, 'o-', color='green', label='验证分数')
    plt.semilogx(param_range, composite_score, 'o-', color='orange', label='综合得分')
    
    # 标记最优点
    best_idx_validation = np.argmax(val_mean)
    best_idx_composite = np.argmax(composite_score)
    
    plt.axvline(x=param_range[best_idx_validation], color='green', 
                linestyle='--', alpha=0.7, label=f'最佳验证: {param_range[best_idx_validation]}')
    plt.axvline(x=param_range[best_idx_composite], color='orange', 
                linestyle='--', alpha=0.7, label=f'最佳综合: {param_range[best_idx_composite]}')
    
    plt.xlabel(f'{param_name} (log scale)')
    plt.ylabel('得分')
    plt.title('参数选择策略比较')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 输出分析结果
    print("=== 验证曲线分析结论 ===")
    print(f"最佳验证分数: {val_mean[best_idx_validation]:.4f} "
          f"(参数 = {param_range[best_idx_validation]})")
    print(f"最佳综合得分: {composite_score[best_idx_composite]:.4f} "
          f"(参数 = {param_range[best_idx_composite]})")
    print(f"对应验证分数: {val_mean[best_idx_composite]:.4f}")
    print(f"对应稳定性: {val_std[best_idx_composite]:.4f}")
    
    return param_range[best_idx_validation], param_range[best_idx_composite]
```

## 第四部分：偏差-方差权衡

### 4.1 偏差-方差分解的数学基础

#### 4.1.1 理论推导

对于回归问题，设真实函数为$f(x)$，噪声为$\epsilon \sim N(0, \sigma^2)$，模型预测为$\hat{f}(x)$，则：

$$y = f(x) + \epsilon$$

预测误差的期望可以分解为：

$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2(\hat{f}(x)) + \text{Var}(\hat{f}(x)) + \sigma^2$$

其中：

**偏差 (Bias)：**
$$\text{Bias}(\hat{f}(x)) = \mathbb{E}[\hat{f}(x)] - f(x)$$

**方差 (Variance)：**
$$\text{Var}(\hat{f}(x)) = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]$$

**噪声 (Irreducible Error)：**
$$\sigma^2 = \mathbb{E}[\epsilon^2]$$

#### 4.1.2 直观理解

```text
高偏差，低方差：
🎯     ●●●
       ●●●  (系统性偏离靶心，但很集中)
       ●●●

低偏差，高方差：
🎯   ●   ●
    ●  ●   (围绕靶心，但很分散)
  ●       ●

高偏差，高方差：
🎯       ●
   ●   ●    (既偏离靶心又分散)
     ●   ●

低偏差，低方差：
🎯  ●●●
    ●●●     (理想状态：准确且稳定)
    ●●●
```

### 4.2 实现偏差-方差分解

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import linregress

def bias_variance_decomposition(model_class, X, y, n_trials=100, test_size=0.3, 
                               problem_type='regression'):
    """完整的偏差-方差分解分析"""
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    predictions = []
    test_indices_list = []
    
    print(f"进行 {n_trials} 次独立实验...")
    
    for trial in range(n_trials):
        # 随机分割数据
        indices = np.random.permutation(n_samples)
        train_idx = indices[:-n_test]
        test_idx = indices[-n_test:]
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 训练模型
        if hasattr(model_class, '__call__'):
            model = model_class()
        else:
            model = model_class
        
        model.fit(X_train, y_train)
        
        # 预测
        if problem_type == 'regression':
            y_pred = model.predict(X_test)
        else:  # classification
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict_proba(X_test)[:, 1]  # 假设二分类
            else:
                y_pred = model.predict(X_test)
        
        predictions.append(y_pred)
        test_indices_list.append(test_idx)
    
    # 找到共同的测试样本
    common_indices = set(test_indices_list[0])
    for indices in test_indices_list[1:]:
        common_indices = common_indices.intersection(set(indices))
    
    if len(common_indices) < 10:
        print("警告：共同测试样本过少，使用替代方法...")
        return alternative_bias_variance_analysis(model_class, X, y, n_trials)
    
    common_indices = list(common_indices)
    
    # 收集共同样本的预测结果
    common_predictions = []
    for i, pred in enumerate(predictions):
        # 找到当前预测中对应的位置
        test_idx = test_indices_list[i]
        mask = np.isin(test_idx, common_indices)
        common_pred = pred[mask]
        
        # 按照common_indices的顺序排列
        ordered_pred = np.zeros(len(common_indices))
        for j, idx in enumerate(common_indices):
            pos = np.where(np.array(test_idx) == idx)[0][0]
            ordered_pred[j] = pred[pos]
        
        common_predictions.append(ordered_pred)
    
    predictions = np.array(common_predictions)
    y_true = y[common_indices]
    
    # 计算偏差-方差分解
    mean_pred = np.mean(predictions, axis=0)
    
    if problem_type == 'regression':
        # 回归问题的偏差-方差分解
        bias_squared = np.mean((mean_pred - y_true) ** 2)
        variance = np.mean(np.var(predictions, axis=0))
        
        # 估计噪声（使用最优预测的残差）
        total_error = np.mean((predictions - y_true.reshape(1, -1)) ** 2)
        noise = max(0, total_error - bias_squared - variance)  # 确保非负
        
        print("=== 偏差-方差分解结果 (回归) ===")
        print(f"偏差²: {bias_squared:.6f}")
        print(f"方差: {variance:.6f}")
        print(f"噪声: {noise:.6f}")
        print(f"总误差: {total_error:.6f}")
        print(f"分解验证: {bias_squared + variance + noise:.6f}")
        
        components = ['偏差²', '方差', '噪声']
        values = [bias_squared, variance, noise]
        
    else:
        # 分类问题的偏差-方差分解（使用0-1损失）
        # 计算主要预测（多数投票）
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # 多类预测
            main_pred = np.array([np.bincount(predictions[:, i]).argmax() 
                                for i in range(predictions.shape[1])])
        else:
            # 二分类预测
            main_pred = np.round(mean_pred).astype(int)
        
        # 计算偏差（主要预测与真实标签的不一致性）
        bias = np.mean(main_pred != y_true)
        
        # 计算方差（不同预测之间的不一致性）
        variance = 0
        for i in range(len(common_indices)):
            # 计算每个样本预测的方差（分类的离散度）
            if len(predictions.shape) > 1:
                unique_preds = np.unique(predictions[:, i])
                if len(unique_preds) > 1:
                    variance += 1 - np.max(np.bincount(predictions[:, i].astype(int))) / len(predictions)
        variance /= len(common_indices)
        
        noise = 0.05  # 分类问题的噪声通常较小
        
        print("=== 偏差-方差分解结果 (分类) ===")
        print(f"偏差: {bias:.6f}")
        print(f"方差: {variance:.6f}")
        print(f"噪声 (估计): {noise:.6f}")
        
        components = ['偏差', '方差', '噪声']
        values = [bias, variance, noise]
    
    # 可视化结果
    plt.figure(figsize=(15, 10))
    
    # 1. 偏差-方差分解饼图
    plt.subplot(2, 3, 1)
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    plt.pie(values, labels=components, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('偏差-方差分解')
    
    # 2. 各组件贡献柱状图
    plt.subplot(2, 3, 2)
    bars = plt.bar(components, values, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel('误差贡献')
    plt.title('各组件误差贡献')
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    # 3. 预测分布分析
    plt.subplot(2, 3, 3)
    for i in range(min(5, len(common_indices))):  # 只显示前5个样本
        plt.hist(predictions[:, i], alpha=0.6, bins=20, 
                label=f'样本 {i+1}', density=True)
    plt.xlabel('预测值')
    plt.ylabel('密度')
    plt.title('预测值分布 (前5个样本)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 偏差分析
    plt.subplot(2, 3, 4)
    bias_per_sample = np.abs(mean_pred - y_true)
    plt.plot(bias_per_sample, 'o-', alpha=0.7)
    plt.axhline(y=np.mean(bias_per_sample), color='r', linestyle='--', 
                label=f'平均偏差: {np.mean(bias_per_sample):.4f}')
    plt.xlabel('样本索引')
    plt.ylabel('|偏差|')
    plt.title('各样本偏差分析')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. 方差分析
    plt.subplot(2, 3, 5)
    variance_per_sample = np.var(predictions, axis=0)
    plt.plot(variance_per_sample, 'o-', alpha=0.7, color='blue')
    plt.axhline(y=np.mean(variance_per_sample), color='r', linestyle='--',
                label=f'平均方差: {np.mean(variance_per_sample):.4f}')
    plt.xlabel('样本索引')
    plt.ylabel('方差')
    plt.title('各样本方差分析')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. 偏差vs方差散点图
    plt.subplot(2, 3, 6)
    plt.scatter(bias_per_sample, variance_per_sample, alpha=0.6, edgecolors='k')
    plt.xlabel('|偏差|')
    plt.ylabel('方差')
    plt.title('偏差 vs 方差权衡')
    plt.grid(True, alpha=0.3)
    
    # 添加趋势线
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(bias_per_sample, variance_per_sample)
    x_trend = np.linspace(bias_per_sample.min(), bias_per_sample.max(), 100)
    y_trend = slope * x_trend + intercept
    plt.plot(x_trend, y_trend, 'r--', alpha=0.8, 
             label=f'趋势线 (R² = {r_value**2:.3f})')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 详细分析总结
    print("\n=== 偏差-方差分解详细分析 ===")
    if problem_type == 'regression':
        print(f"各组件占比:")
        total = bias_squared + variance + noise
        print(f"  偏差²占比: {bias_squared/total*100:.1f}%")
        print(f"  方差占比: {variance/total*100:.1f}%")
        print(f"  噪声占比: {noise/total*100:.1f}%")
        
        if bias_squared > variance:
            print("🎯 主要问题: 高偏差 (欠拟合)")
            print("建议: 增加模型复杂度、添加特征、减少正则化")
        elif variance > bias_squared:
            print("🎯 主要问题: 高方差 (过拟合)")
            print("建议: 增加正则化、减少模型复杂度、增加训练数据")
        else:
            print("🎯 偏差和方差相对平衡")
    
    return {
        'bias_squared': bias_squared if problem_type == 'regression' else bias,
        'variance': variance,
        'noise': noise,
        'total_error': bias_squared + variance + noise if problem_type == 'regression' else bias + variance + noise
    }

def alternative_bias_variance_analysis(model_class, X, y, n_trials=100):
    """替代的偏差-方差分析方法（当共同测试样本过少时）"""
    
    print("使用替代方法进行偏差-方差分析...")
    
    # 使用固定的测试集
    from sklearn.model_selection import train_test_split
    X_train_base, X_test, y_train_base, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    predictions = []
    
    for trial in range(n_trials):
        # 从训练集中进行自助采样
        n_train = len(X_train_base)
        bootstrap_idx = np.random.choice(n_train, size=n_train, replace=True)
        X_bootstrap = X_train_base[bootstrap_idx]
        y_bootstrap = y_train_base[bootstrap_idx]
        
        # 训练模型
        if hasattr(model_class, '__call__'):
            model = model_class()
        else:
            model = model_class
        
        model.fit(X_bootstrap, y_bootstrap)
        y_pred = model.predict(X_test)
        predictions.append(y_pred)
    
    predictions = np.array(predictions)
    
    # 计算偏差-方差分解
    mean_pred = np.mean(predictions, axis=0)
    bias_squared = np.mean((mean_pred - y_test) ** 2)
    variance = np.mean(np.var(predictions, axis=0))
    
    total_error = np.mean((predictions - y_test.reshape(1, -1)) ** 2)
    noise = max(0, total_error - bias_squared - variance)
    
    print("=== 替代方法偏差-方差分解结果 ===")
    print(f"偏差²: {bias_squared:.6f}")
    print(f"方差: {variance:.6f}")
    print(f"噪声: {noise:.6f}")
    print(f"总误差: {total_error:.6f}")
    
    return {
        'bias_squared': bias_squared,
        'variance': variance,
        'noise': noise,
        'total_error': total_error
    }
```

### 4.3 不同复杂度模型的偏差-方差分析

```python
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

def compare_model_complexity_bias_variance(X, y, problem_type='regression'):
    """比较不同复杂度模型的偏差-方差权衡"""
    
    if problem_type == 'regression':
        models = {
            '线性回归 (低复杂度)': lambda: LinearRegression(),
            '决策树 (深度=3)': lambda: DecisionTreeRegressor(max_depth=3, random_state=42),
            '决策树 (深度=10)': lambda: DecisionTreeRegressor(max_depth=10, random_state=42),
            '决策树 (无限制)': lambda: DecisionTreeRegressor(random_state=42),
            'KNN (k=10)': lambda: KNeighborsRegressor(n_neighbors=10),
            'KNN (k=1)': lambda: KNeighborsRegressor(n_neighbors=1),
            '随机森林': lambda: RandomForestRegressor(n_estimators=50, random_state=42)
        }
    else:
        models = {
            '逻辑回归 (低复杂度)': lambda: LogisticRegression(random_state=42),
            '决策树 (深度=3)': lambda: DecisionTreeClassifier(max_depth=3, random_state=42),
            '决策树 (深度=10)': lambda: DecisionTreeClassifier(max_depth=10, random_state=42),
            '决策树 (无限制)': lambda: DecisionTreeClassifier(random_state=42),
            'KNN (k=10)': lambda: KNeighborsClassifier(n_neighbors=10),
            'KNN (k=1)': lambda: KNeighborsClassifier(n_neighbors=1),
            '随机森林': lambda: RandomForestClassifier(n_estimators=50, random_state=42)
        }
    
    results = {}
    
    print("开始偏差-方差分解比较分析...")
    for model_name, model_func in models.items():
        print(f"\n分析模型: {model_name}")
        result = bias_variance_decomposition(model_func, X, y, n_trials=50, 
                                           problem_type=problem_type)
        results[model_name] = result
    
    # 可视化比较结果
    plot_bias_variance_comparison(results)
    
    return results

def plot_bias_variance_comparison(results):
    """可视化偏差-方差分解比较结果"""
    
    models = list(results.keys())
    bias_values = [results[model]['bias_squared'] for model in models]
    variance_values = [results[model]['variance'] for model in models]
    noise_values = [results[model]['noise'] for model in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 堆叠柱状图
    x = np.arange(len(models))
    width = 0.6
    
    ax1.bar(x, bias_values, width, label='偏差²', alpha=0.8, color='#ff9999')
    ax1.bar(x, variance_values, width, bottom=bias_values, label='方差', alpha=0.8, color='#66b3ff')
    ax1.bar(x, noise_values, width, bottom=np.array(bias_values) + np.array(variance_values), 
            label='噪声', alpha=0.8, color='#99ff99')
    
    ax1.set_xlabel('模型')
    ax1.set_ylabel('误差')
    ax1.set_title('偏差-方差分解比较')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 偏差vs方差散点图
    ax2.scatter(bias_values, variance_values, s=100, alpha=0.7, edgecolors='k')
    for i, model in enumerate(models):
        ax2.annotate(model, (bias_values[i], variance_values[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('偏差²')
    ax2.set_ylabel('方差')
    ax2.set_title('偏差-方差权衡')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

### 4.4 实际应用示例

```python
# 示例1: 回归问题的偏差-方差分析
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def regression_bias_variance_demo():
    """回归问题的偏差-方差分析演示"""
    
    print("=== 回归问题偏差-方差分析演示 ===\n")
    
    # 生成回归数据
    X, y = make_regression(n_samples=300, n_features=10, noise=0.1, 
                          random_state=42)
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 比较不同复杂度模型
    results = compare_model_complexity_bias_variance(X_scaled, y, 'regression')
    
    return results

# 示例2: 分类问题的偏差-方差分析
from sklearn.datasets import make_classification

def classification_bias_variance_demo():
    """分类问题的偏差-方差分析演示"""
    
    print("\n=== 分类问题偏差-方差分析演示 ===\n")
    
    # 生成分类数据
    X, y = make_classification(n_samples=300, n_features=10, n_informative=5,
                              n_redundant=2, n_clusters_per_class=1, 
                              random_state=42)
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 比较不同复杂度模型
    results = compare_model_complexity_bias_variance(X_scaled, y, 'classification')
    
    return results

    # 运行演示
if __name__ == "__main__":
    # 回归演示
    regression_results = regression_bias_variance_demo()
    
    # 分类演示  
    classification_results = classification_bias_variance_demo()
    
    print("\n=== 偏差-方差分析总结 ===")
    print("回归问题结果:", regression_results)
    print("分类问题结果:", classification_results)
```

## 第五部分：实际应用示例

### 5.1 综合案例：房价预测模型评估

```python
def comprehensive_model_evaluation_demo():
    """完整的模型评估演示"""
    
    from sklearn.datasets import fetch_california_housing
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold, train_test_split
    
    print("=== 加州房价预测模型评估案例 ===\n")
    
    # 加载数据
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    print(f"数据集信息:")
    print(f"样本数量: {X.shape[0]}")
    print(f"特征数量: {X.shape[1]}")
    print(f"目标变量范围: [{y.min():.2f}, {y.max():.2f}]")
    
    # 数据预处理
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 定义模型
    models = {
        '线性回归': LinearRegression(),
        '岭回归': Ridge(alpha=1.0),
        '随机森林': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
    }
    
    # 1. 交叉验证比较
    print("\n" + "="*50)
    print("第一步: 交叉验证模型比较")
    print("="*50)
    
    cv_results = comprehensive_cross_validation(
        models, X_scaled, y,
        cv_strategies={
            'KFold-5': KFold(n_splits=5, shuffle=True, random_state=42),
            'KFold-10': KFold(n_splits=10, shuffle=True, random_state=42)
        },
        scoring_metrics=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']
    )
    
    # 2. 学习曲线分析
    print("\n" + "="*50)
    print("第二步: 学习曲线分析")
    print("="*50)
    
    best_model = RandomForestRegressor(n_estimators=100, random_state=42)
    comprehensive_learning_curve_analysis(best_model, X_scaled, y)
    
    # 3. 验证曲线分析
    print("\n" + "="*50)
    print("第三步: 超参数验证曲线分析")
    print("="*50)
    
    param_range = [10, 50, 100, 200, 500]
    comprehensive_validation_curve_analysis(
        RandomForestRegressor(random_state=42), X_scaled, y,
        'n_estimators', param_range
    )
    
    # 4. 偏差-方差分解
    print("\n" + "="*50)
    print("第四步: 偏差-方差分解分析")
    print("="*50)
    
    bias_variance_results = compare_model_complexity_bias_variance(
        X_scaled, y, 'regression'
    )
    
    # 5. 最终模型评估
    print("\n" + "="*50)
    print("第五步: 最终模型详细评估")
    print("="*50)
    
    # 训练最佳模型
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    final_model = RandomForestRegressor(n_estimators=200, random_state=42)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    
    # 详细回归评估
    regression_metrics(y_test, y_pred)
    
    print("\n=== 模型评估总结 ===")
    print("1. 随机森林在所有评估指标上表现最佳")
    print("2. 模型在200个树时达到最佳性能")
    print("3. 偏差-方差分析显示模型平衡良好")
    print("4. 残差分析显示模型假设基本满足")

# 运行综合演示
if __name__ == "__main__":
    comprehensive_model_evaluation_demo()
```

## 学习总结与反思

### 核心收获

通过深入学习模型评估和选择，我获得了以下关键认识：

1. **评估指标的选择艺术**：不同的业务场景需要不同的评估重点。在医疗诊断中，高召回率可能比高精确率更重要；而在垃圾邮件检测中，可能需要平衡两者。

2. **交叉验证的威力**：简单的训练-测试分割往往不够可靠，交叉验证提供了更稳健的性能估计，特别是在数据有限的情况下。

3. **偏差-方差权衡的普遍性**：这个概念不仅适用于机器学习，在生活中也无处不在。追求完美往往意味着不稳定，而稳定往往需要接受一定的系统性误差。

4. **模型诊断的重要性**：学习曲线和验证曲线就像医生的"听诊器"，能够帮我们诊断模型的"健康状况"，指导改进方向。

### 实践中的深刻体会

**1. "没有测量就没有改进"**
在实际项目中，我发现很多时候模型性能的提升来自于对评估过程的细致分析，而不是盲目地尝试更复杂的算法。

**2. "过拟合无处不在"**
通过偏差-方差分解，我意识到过拟合不仅仅是一个技术问题，更是一个哲学问题：如何在记忆和泛化之间找到平衡？

**3. "统计显著性的重要性"**
当两个模型性能接近时，仅仅比较平均分数是不够的，需要进行统计检验来确定差异是否真实存在。

### 容易犯的错误

1. **数据泄露**：在交叉验证中，特征选择或数据预处理步骤如果在整个数据集上进行，会导致过于乐观的性能估计。

2. **不平衡数据的误区**：在极不平衡的数据集上，准确率可能高达99%，但模型可能完全没有学到有用的模式。

3. **过度优化验证集**：反复在验证集上调参，实际上是把验证集当作了训练集的一部分。

4. **忽略业务约束**：技术指标很重要，但模型的实际部署还需要考虑延迟、内存、可解释性等工程约束。

### 进阶学习方向

1. **在线学习的评估**：当数据分布随时间变化时，如何评估模型的适应性？

2. **多任务学习评估**：当一个模型需要同时完成多个任务时，如何平衡不同任务的性能？

3. **公平性评估**：如何确保模型在不同群体上的表现公平？

4. **不确定性量化**：如何评估模型预测的置信度？

### 最终感悟

模型评估不仅仅是技术活，更是一门艺术。它要求我们在准确性和稳定性之间权衡，在复杂性和可解释性之间选择，在理论完美和实际可行之间妥协。

正如统计学家George Box所说："所有模型都是错误的，但有些是有用的。" 模型评估的目标不是找到完美的模型，而是找到在给定约束下最有用的模型。

通过系统地学习模型评估和选择，我不仅获得了技术技能，更重要的是培养了科学严谨的思维方式。这种思维方式告诉我们：

- 要有证据支持的结论
- 要量化不确定性
- 要考虑多个角度
- 要在理论和实践之间找到平衡

这些原则不仅适用于机器学习，也适用于科学研究和日常决策。在这个充满不确定性的世界里，掌握科学的评估方法比掌握任何特定的算法都更加重要。
