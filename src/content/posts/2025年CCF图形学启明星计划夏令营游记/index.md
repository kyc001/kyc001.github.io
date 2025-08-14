---
title: 2025年CCF图形学启明星计划夏令营游记
published: 2025-8-4 20:00:20
slug: 2025-ccf-graphics-summer-camp
tags: ['计算机图形学','夏令营','CCF','游记']
category: 游记
draft: false
image: ./bg.jpg
---

## 2025年CCF图形学启明星计划夏令营游记

## 目录

### Day 1 - 基础理论与工业应用

- **上午场次**
  - 黄慧老师：近十年SIGGRAPH成果回顾
  - 熊卫丹老师：三维模型处理技术
- **下午场次**
  - IEG校招宣讲
  - 腾讯工程师：游戏图形学技术分享
  - 吕辰雷老师：三角网格重建基础理论
  - 方青老师：表征,编辑,去噪,法相提取

### Day 2 - 深度学习与3D理解

- **上午场次**
  - 赵恒爽老师：面向三维点云/网格的深度学习基础 + 分割检测下游任务
  - 李元琪老师：CAD课程
- **下午场次**
  - 业界前沿分享
  - 上机实训：各向同性网格优化算法

### Day 3 - 神经渲染与场景交互

- **上午场次**
  - 韩晓光老师：MVS多目视图立体匹配方法
  - 高林老师：神经渲染和几何重建
- **下午场次**
  - 胡瑞珍老师：几何引导的三维交互建模生成

---

## Day 1

### Day 1 上午

#### 黄慧老师 - 近十年SIGGRAPH成果回顾

#### 熊卫丹老师 - 三维模型处理技术

- **三维模型数据压缩**
- **通过设计师手稿仿真出力学结构**

### Day 1 下午

#### IEG校招宣讲

#### 腾讯工程师 - 游戏图形学技术分享

**游戏内渲染学：**

- **用神经网络压缩全局光照**
  - 直接光照 + 间接光照 + 材质 probe irradiance
  - 其他管线可以吗？
    - 核心是构建一个可微的正向渲染管线

**其他技术方向：**

- 美术资产生成
- 自动蒙皮与动作绑定
- 游戏外图形学
- 星瞳（基于UE5自研）
- 全景动力飞行模拟机视景系统
- 腾讯脑力锻炼

#### 吕辰雷老师 - 三角网格重建基础理论

**核心概念：**

- **3D Triangle Mesh**
- **水密问题**：水从缝隙通过

**技术要点：**

- 等边三角形解决重边，更均匀方便
- 把由物体直接得到的不规则的点集编辑成等边三角形，更规整
- 保持几何关系不变，对每个点有拉回操作，由其他几个点进行赋权
- collapse操作可能导致几何关系变化

**方法对比：**

- 与Mesh Lab方法对比，VCG在边缘处出现钝角三角形和锐角三角形

**挑战与解决方案：**

- **Problems**：复杂几何部分数据爆炸，做一个权衡，自适应等边三角形边长
- 有实际含义的几何信息会被破坏（语义分析）

#### 方青老师 - 表征,编辑,去噪,法相提取

**几何表征方法：**

- 点云、样条函数、符号距离场、体素、面网格、体网格
- 隐式表征

**技术应用：**

- 编辑和交互方式
- 欧拉庞加莱公式
- 三角网格曲率计算：平均曲率、高斯曲率
- 编辑法向应用：去噪

## Day 2

### Day 2 上午

#### 赵恒爽老师 - 面向三维点云/网格的深度学习基础 + 分割检测下游任务

**基础模型概述：**

- **LLM** (Large Language Models)
- **MLLM / LVLM** (Multimodal Large Language Models / Large Vision-Language Models)
- **Foundation Models**
  - Language models
  - Vision models
  - Multimodal models

**剩余挑战：**

- Scene Understanding
- Foundation Models with Spatial Intelligence

#### 3D理解技术深入

**3D数据应用领域：**

- 自动驾驶 (Auto Driving)
- 机器人技术 (Robotics)
- 增强现实 (Augmented Reality)
- 医学图像分析 (Medical Image Analysis)

**3D表示方法：**

- Point Cloud, Mesh, 等

**3D点云处理方法：**

**核心特点：**

- 无序且在3D空间中分散
- 传统ConvNets不适用

##### 1. 基于投影的方法 (Projection-based Method)

- 不同角度的投影做CNN
- 挑战：
  - Geometric Collapsed
  - Projection Planes
  - Network Efficiency
- 代表方法：MVCNN

##### 2. 基于体素的方法 (Voxel-based Method)

- 不同分辨率像素点云的不同视图(三视图?)做CNN
- Problems：...
- 相关论文：
  - OctNet: Learning Deep 3D Representations at High Resolutions. CVPR 2017
  - O-CNN: Octree-based Convolutional Neural Networks for 3D Shape Analysis

##### 3. 基于点的方法 (Point-based Method)

- 核心特性：
  - Permutation Invariant (排列不变性)
  - Representation Ability (表示能力)

发展历程：

- PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
- PointNet++:
  - Sampling: FPS最远点采样
  - Grouping: KNN等
- DGCNN
- PointWeb
- Point Transformer
  - Transformer and Self-attention
  - Point Transformer Layer
- Point Transformer V2

##### 3D Representation

- Image Representation DINO v2
- **Representation Learning**
  - 2D: ImageNet, etc.
  - 3D: ModelNet, etc.
- **PointContrast**
  - Frame Matching
    - Is matching RGB-D frames a good choice?
  - → Masked Scene Contrast
  - Scene Augmentation
  - View Mixing
  - Masked Scene and Color Reconstruction
  - Contrastive and Reconstructive
  - Multi-dataset Synergistic Training
  - PointPrompt Training
    - Domain Prompt Adapter
    - Categorical Alignment
- Geometric Shortcut: Unique Challenge Point Cloud
- → 3D Representation To Limit: SONATA
  - SONATA: Self-Supervised Learning ...
- → 3D Representation To Super
  - ...在投

##### 3D Reasoning

- Multimodal 3D Foundation gpt4o
- ...
- Spatial CLIP
- 3D Situation ...
- ...

#### 李元琪老师 CAD课程

##### CAD 三维重建

###### 1. 为什么需要计算机辅助设计

- 设计制造,最早是通过画图的方式

###### 2. 三维表示 CAD表示

- B-Rep 用面边角表示几何图形，通常是水密的
- B-spline B-样条曲线/曲面 维数=点数-1
- CSG(Constructive Solid Geometry) 用基本三维图形 通过交并差等布尔运算构成新图形
- 方式对比,优缺点...

###### 3. CAD逆向建模

- 激光雷达原理
- 非结构化重建
  - 滚球法重建
  - 泊松表面重建
  - 基于多视角的非结构化重建 SFM
- 结构化重建
  - 基元拟合

### Day 2 下午

#### 业界前沿分享

##### 元象科技 - LBVR技术速通攻略

1. 简介
2. 定位技术
3. 渲染技术
4. 交互技术
5. 总结

##### 中建八局

- 工业应用

##### 中望软件

- 工业软件设计

#### 上机实训:各向同性网格优化算法

## Day 3

### Day 3 上午

#### 韩晓光老师 MVS多目视图立体匹配方法

##### 基于图像的三维重建

- 为什么要三维重建
- 相机模型
  - Pinhole camera
    1. coordinate systems
       - off set
    2. converting to pixels
       - ...
    - world reference system
- Structure-from-Motion
  - minimize f(R, T, P)
  - 特征点匹配 - detect features
- Multi-view Streo - basic idea
  - Results
- 用深度学习进行优化
  - → MVSNet
  - → VGGT (CVPR 2025 Best Paper)

#### 高林老师 神经渲染和几何重建

##### 神经渲染和几何重建

- 3D LivePhoto技术发展
- 3D LivePhoto的关键要素
- 混合表达与生成式重建
- NeRF(Neural Radiance Field)
- 高斯泼溅
- 混合表征在推动集合表征的发展

###### 几何表征方法

- 点云
- 体素(三维空间分割中的最小单位)
- 网格 性质好,可以形变,可无限细分为连续曲面
- 网格与纹理贴图 映射
- 隐式场 三维空间点坐标到SDF值的映射 SDF(x) 可以用神经网络来拟合一个模型

###### 经典几何表征的局限性

- → 通过神经网络近似反射函数

###### 神经网络基础 - 多层感知机

###### 神经辐射场的基本原理

- 使用全连接网络建模5D全光函数,使用体渲染得到最终的任意视角图像
- 基于ray marching: 从相机出发,向空间中投射光线,并在光线上采样.
- 每个采样点送入全连接网络,预测颜色和密度值,整个过程是可微的.
- 体渲染

###### 建立带符号的距离场SDF到密度场的映射 NeuS

- 建立无偏的权重函数

###### 几何场感知的神经辐射场编辑方法

- 研究挑战 如何对神经图形进行几何变形？
- →编辑后的三角网格驱动四面体网格变形从而扭曲空间中的光线达到空间变形的效果

###### 3D-GS(3D Gaussian Splatting)

- Part1 3D Gaussians
- Part2 Splatting 3D对象映射到投影平面,每个splat计算是并行的

### Day 3 下午

#### 胡瑞珍老师 几何引导的三维交互建模生成

##### 三维场景交互

- Modeling and Generating Interactions in 3D World
- Feedforward Generative Model (on Interactions?)
- Optimization-based Interaction Generation

###### Setting 1. Specific Interaction Policy Learning

- RL →
- Challenge: Generalization to Unseen Objects?

###### 1. Example-based Interaction Transfer

- Demo → Similarity
- Challenge: Precise Similarity Measure

###### 2. Text-driven Interaction Generation

- A person is surfing ← Image / Videos
- Challenge: Cross-modality Alignment

###### 1. State representation

- Object-centric
  - ! Significant difference in geometry
- Interaction-centric state representation
- Much similar geometry

###### 2. Prior: Bounding box

- ! 局部细节不到位
- Key idea: define an interaction template to measure the similarity
- 找到一个特征场,在新场景匹配
- Space Coverage Feature

###### 3. Text to HOI

- Text-to-Image Diffusion Model(ECCV 2024)
- SDS Loss 对齐
- HOI Pose codebook

###### Motion Planning in Virtual World

- Decompose a given activity task ... 生成上层工作规划(代码)
- Key Challenges:
  - Raionality,Executability

---
