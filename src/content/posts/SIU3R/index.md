---
title: SIU3R： Learning Scene Understanding with Implicit Unified 3D Representation
published: 2025-11-14 16:26:23
slug: siu3r
tags: ['计算机视觉', '3D重建', '场景理解']
category: 计算机视觉
draft: false
---
# SIU3R
- 提取2D特征，对齐到NeRF/3DGS表示
- 问题：
    - 3D理解能力有限
    - 语义压缩损失
- 提出SIU3R，无需特征对齐的框架
    - 像素对齐的3D表示
    - 多个理解任务统一
    - 双向互助机制

## 引言
- 现有方法：CLIP+LSeg -> rasterize到3D表示 -> per-scene优化对齐
- 局限：
    - 多视图，逐场景，不可扩展
    - 2D能力
    - 特征必须压缩降维
- 直接学习像素对齐的3D表示，无需2D feature alignment

## 相关工作
- 3D重建：每场景优化
- 场景理解
- 2D：缺乏跨视角一致性
    - 3D需预先扫描点云，不适用于重建联合任务
- 同步理解+3D重建
    - LERF+LangSplat 2D -> 3D

## 方法
### Pipeline
### 问题定义
- 给一组无位姿图像，3xHxW，同时执行场景理解和三维重建

### 无对齐方法
- 图像编码器：ViT
- 文本编码器
- 高斯解码器：DPT头
- 互惠机制：
    - 从重建中促进理解：多视图掩码聚合模块
    - 通过理解促进重建：掩码引导几何细化模块

### 统一查询解码器
- 使用一组可学习的统一查询Q以联合解码实例分割和语义分割任务中的跨视角一致掩码
- 每个查询 `qn` 显式地表示一个潜在的物体实例或语义区域

### 互惠机制
- 多视图掩码聚合
- 掩码引导的几何优化
