---
title: IGGT： 融合几何与语义的3D场景理解
published: 2025-11-21 19:27:50
slug: iggt
tags: ['计算机视觉', '3D重建', '场景理解']
category: 计算机视觉
draft: false
---

# IGGT
- Transformer 架构
- 几何语义同出 相互促进 一致性
- 构建了 InsScene-15k
    - 数据生成 SAM2
![导入](./img/Pasted%20image%2020251121215940.png)

## 旧有方法
- 解耦 3D 几何重建和高级语义理解为独立任务
- 先利用几何方法预测 3D 结构，再通过 VLMs / 2D seg 模型进行语义分割
- ->对齐
    - 1.对齐高级文本概念可能会使表示过度平滑
    - 2.依赖基础模型能力（没有集成）
    - 3.缺乏 3D 能力 （数据问题）
- 几何头 + 实例头 解码为几何场和实例场
- 滑动窗口移位注意力的跨模态融合模块
- 一致性对比学习
![数据生成](./img/Pasted%20image%2020251121220011.png)

## 数据集
- 合成数据集 Aria, Infinigen
- 视频捕获 初始帧 prompt 时间传递
- RGBD 扫描 ScanNet++
![数据集来源](./img/Pasted%20image%2020251121220029.png)

## 方法
### 架构
- $N$ 个输入图像
- 预测相机 $t$ ，深度 $D$ ，点图 $P$ ，实例特征 $S$
- 1.统一 Transformer，从多幅图像捕获统一 Token 表示
- 2.跨模态融合 互相增强
- 3.3D 一致性监督
![架构流程图](./img/Pasted%20image%2020251121220054.png)

## 统一 Transformer 1B
- 多视图图像编码成一组 Token
    - $M$ （Token数量） $\times D$ （Token维度） $\times N$
- 预训练的 DINOv2 提取图像 Token
    - 连接一个可学习的相机 Token
- self/cross attention

## 下游头
### 几何头
- 相机预测器
    - 从相机 Token 中回归相机参数
- 深度预测器 点预测器
    - 类似 DPT 架构 从统一 Token 重建特征

### 实例头
- 滑动窗口 cross attention
- 实例特征 通过 $3 \times 3$ 映射为 8 维特征

## 3D 一致性对比监督
- 多视图监督
- loss

## 基于实例的场景理解
- 实例空间跟踪 聚类
![实例空间跟踪与聚类](./img/Pasted%20image%2020251121220123.png)
- 开放词汇语义分割
    - 我理解的是对一整个区域分配语义？
![开放词汇语义分割示意图](./img/Pasted%20image%2020251121220150.png)
- QA 场景定位
    - 大型多模态模型交互 是/否
    - LMMs(大型多模态模型)
    - 做了一个监督？
![QA 场景定位流程图 1](./img/Pasted%20image%2020251121220213.png)
![QA 场景定位流程图 2](./img/Pasted%20image%2020251121220224.png)
![QA 场景定位图 3](./img/Pasted%20image%2020251121220240.png)

## 实验
### 评估细节
- ScanNet ScanNet++
- 10 场景 8-10 张图片
- 跟踪 时间 mIOU（衡量分割进度），时间成功率
- 分割 mIOU，mACC
- 实例空间跟踪评估
![评估细节表格](./img/Pasted%20image%2020251121220302.png)

## 相关工作
- 空间基础模型
- 3D 场景理解

## 训练细节
- $8 \times A800$ GPUs 2 天