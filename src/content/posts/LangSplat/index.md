---
title: LangSplat： Language-Guided 3D Scene Splatting with Implicit Language Fields
published: 2025-11-14 15:13:20
slug: langsplat
tags: ['计算机视觉', '3D重建', '场景理解']
category: 计算机视觉
draft: false
---
# LangSplat
- 利用一组三维高斯，每个高斯编码从CLIP提取的语言特征，来表示语言场
- 避免NeRF内在的高成本渲染过程
- 并不是直接学习CLIP嵌入，而是首先训练场景级的语言自编码器，在特定的潜在空间中学习语言特征
- 缓解显示建模带来的巨大内存需求
- 使用SAM学习分层语义

## 介绍
- LERF速度和精度
- CLIP嵌入是与图像对齐的，而不是与像素对齐的
- 同一个3D位置可能与不同尺度的语义概念相关
- 首先学习一个场景级语言自编码器，将场景中的CLIP嵌入映射到低维潜在空间。
- 对于每张二维图像，我们使用SAM获得三个不同语义层级的良好分割图。

## 相关工作
- 3DGS
- SAM 有没有连续尺度的工作？ 处理多层语义嵌套？ 目前没有
- 3D Language Fields 嵌入NeRF

## 方法
- 重新审视语义场的挑战
- 使用SAM学习层次语义
- 语言场的3D高斯点云
- 开放词汇查询

## 训练场景级编码器
- 1440x1080 25min 3090 4GB
