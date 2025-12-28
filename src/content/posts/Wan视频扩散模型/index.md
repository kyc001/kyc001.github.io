---
title: Wan 学习笔记
published: 2025-10-18 00:13:20
slug: wan-video-diffusion
tags: ['视频生成', '扩散模型', '计算机视觉']
category: 计算机视觉
draft: false
image: ./bg.jpg
---
## Wan 视频扩散模型学习笔记

Wan 是最近比较火的视频生成扩散模型，可以理解为"Stable Diffusion 的视频版"。它的核心挑战在于同时建模空间结构（图像内容）和时间一致性（运动轨迹），还要在有限显存下搞定高分辨率视频生成。

整个系统由三个核心模块组成：Video VAE 负责压缩视频到时空 latent 表征，Diffusion Transformer 在 latent 空间中做扩散去噪，Text Encoder 把文本 prompt 转换为语义向量来引导生成。

---

## Wan 模型总览与核心创新

作为阿里自研的视频生成模型，Wan 的创新性主要体现在以下几个方面：

### 多阶段层级生成架构

Wan 采用了 **分层生成（Hierarchical Generation）** 策略来解决高分辨率视频的显存瓶颈：

1. **Base Model**：生成低分辨率、低帧率的基础 latent（如 480p@16fps）
2. **Temporal Super-Resolution**：提升时间分辨率（如 16fps → 48fps）
3. **Spatial Super-Resolution**：提升空间分辨率（如 480p → 1080p）

这种架构类似于 Sora、HunyuanVideo 等顶级模型，是当前解决视频生成显存问题的主流方案。

### 大规模 Transformer 主干

Wan 使用 **Video DiT（Diffusion Transformer）** 作为主干网络：
- 模型规模约百亿级参数（官方未公开具体数值）
- 采用改进的 T5 文本编码器（`WanT5EncoderModel`）用于文本理解
- 时空注意力分解设计，支持长视频生成

### 多模态训练语料

训练数据涵盖：
- 中英文双语视频-文本对（上亿级规模）
- 影视素材、广告、短视频等多种类型
- 音视频脚本、镜头描述等结构化标注

这使得 Wan 在多语言理解、复杂场景建模上具有优势。

### 显存优化策略

针对高分辨率、长视频生成的显存挑战，Wan 采用了多种优化技术：
- **分块注意力（Chunked Attention）**：将长序列分块处理
- **FlashAttention 集成**：加速注意力计算，降低显存占用
- **混合精度训练（FP16/BF16）**：减少一半显存
- **动态显存调度**：类似 ZeRO-offload，在 CPU 和 GPU 间动态交换

### Pipeline 整体架构

```
用户 Prompt（中/英文）
       ↓
┌──────────────────────────┐
│  Text Encoder (T5)        │
│  - 编码文本语义            │
│  - 输出条件向量            │
└──────────┬───────────────┘
           ↓
┌──────────────────────────┐
│  Video VAE Encoder        │
│  - 3D 卷积压缩            │
│  - 时间4倍，空间8倍下采样 │
└──────────┬───────────────┘
           ↓
     Latent Space (z)
           ↓
┌──────────────────────────┐
│ Diffusion Transformer      │
│  - 时空注意力分解          │
│  - 文本交叉注意力          │
│  - 逐步去噪 (50 步)        │
└──────────┬───────────────┘
           ↓
  去噪后的 Latent
           ↓
┌──────────────────────────┐
│  Video VAE Decoder        │
│  - 3D 转置卷积            │
│  - 恢复视频帧             │
└──────────┬───────────────┘
           ↓
     生成的视频帧
```

### 与其他模型的技术对比

| 模型 | 主干架构 | 文本编码器 | 层级生成 | 核心特征 | 特点 |
|------|---------|-----------|---------|---------|------|
| Stable Video Diffusion | 3D UNet | CLIP | ✗ | 基于 SD 扩展 | 开源，但功能有限 |
| VideoCrafter | DiT | T5-XXL | ✗ | 可控生成 | 支持多种控制 |
| HunyuanVideo | Video DiT | Qwen | ✓ | 多阶段 | 官方中文支持 |
| **Wan（阿里）** | **Video DiT** | **T5 编码器** | **✓** | **高分辨率+语义一致** | **开源实现（VideoX-Fun）** |

VideoX-Fun 实现的 Wan 模型特点：
1. **完整的开源实现**：包含 VAE、Transformer、Pipeline 等完整组件
2. **T5 文本编码器**：支持灵活的文本长度处理和相对位置编码
3. **时空分解注意力**：有效降低显存占用
4. **Flow Matching 采样**：采用现代的 Flow Matching 范式替代 DDPM

---

## 从输入到输出的完整流程

整个生成过程其实很直观：

```
文本（prompt）
    ↓
Text Encoder → 文本特征
    ↓
随机噪声 latent（初始视频块）
    ↓
Diffusion Transformer（逐步去噪）
    ↓
预测干净 latent（表示生成视频）
    ↓
Video VAE Decoder（解码）
    ↓
生成连续视频帧
```

简单来说就是："文字" → "潜在时空表征" → "连续视频"。

---

## Video VAE：3D视频自编码器

在像素级别生成视频成本实在太高了，所以需要先用 Video VAE 压缩视频到时空 latent：

$$
x \in \mathbb{R}^{T\times H\times W\times 3}
\quad\Rightarrow\quad
z \in \mathbb{R}^{C\times T'\times H'\times W'}
$$

压缩比例一般是：
- 时间维度：1/4
- 空间维度：1/8

### 编码器结构

核心算子是 **因果 3D 卷积（CausalConv3d）** 和标准 **3D 卷积（Conv3D）**。

#### CausalConv3d - 序列生成的因果约束

在 VideoX-Fun 实现（`videox_fun/models/wan_vae.py:21-40`）中，使用因果卷积实现时间维度的因果掩膜：

```python
import torch.nn as nn
import torch.nn.functional as F

class CausalConv3d(nn.Conv3d):
    """
    因果 3D 卷积 - 时间维度上具有因果性
    前向帧只能看到历史帧和当前帧，不能看到未来帧
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 手动设置 padding：(后, 前, 下, 上, 右, 左)
        self._padding = (
            self.padding[2], self.padding[2],      # 空间维 y
            self.padding[1], self.padding[1],      # 空间维 x
            2 * self.padding[0], 0                 # 时间维（因果：前padding，后无padding）
        )
        # 不使用默认 padding，全手动处理
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        """
        Args:
            x: 输入张量 [B, C, T, H, W]
            cache_x: 可选的缓存前一帧，用于块级生成
        """
        padding = list(self._padding)

        # 关键：使用缓存的前一帧作为历史上下文
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)    # 沿时间维拼接
            # 减少前向 padding，因为缓存已提供历史帧
            padding[4] -= cache_x.shape[2]

        # 应用手动 padding
        x = F.pad(x, padding)
        return super().forward(x)


class Conv3DBlock(nn.Module):
    """标准 Conv3D 块，无因果约束"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = CausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 4, 4),
            stride=(1, 4, 4),  # 时间步长=1，空间步长=4
            padding=(1, 1, 1)
        )

    def forward(self, x, cache_x=None):
        return self.conv(x, cache_x)
```

**关键特性**：
- **时间方向步长 = 1**：保留所有时间信息（不下采样）
- **空间方向步长 = 4**：降低空间分辨率 4 倍
- **因果掩膜**：通过 `_padding = (2, 2, 1, 1, 2, 0)` 实现
  - `(2, 2)`：空间 y 维度两侧 padding
  - `(1, 1)`：空间 x 维度两侧 padding
  - `(2, 0)`：时间维度**前 padding（历史帧），后不 padding**（实现因果性）
- **缓存机制**：支持 `cache_x` 参数，允许块级生成时复用前一帧信息

然后用残差结构保持梯度稳定，防止信息在深层网络中衰减：

$$
y = x + \text{Conv3D}(\text{ReLU}(\text{Conv3D}(x)))
$$

来看个具体例子：

```
输入视频:  (B, 3, 16, 512, 512)   [批次, 通道, 时间帧数, 高, 宽]
    ↓
Conv3D + ResBlock ×3
    ↓
latent:    (B, 8, 16, 64, 64)    [空间压缩8倍, 时间维度保持16帧]
```

压缩率相当可观，这也是为什么能在消费级显卡上跑视频生成的关键。

### 解码器结构

解码器就是编码器的反向镜像，使用 **ConvTranspose3D**（转置卷积）逐步恢复空间和时间分辨率：

```python
class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.deconv_blocks = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, 64,
                             kernel_size=(3,4,4),
                             stride=(1,4,4)),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32,
                             kernel_size=(3,4,4),
                             stride=(1,4,4)),
            nn.ReLU(),
            nn.Conv3d(32, 3, kernel_size=(3,3,3), padding=(1,1,1))
        )

    def forward(self, z):
        return self.deconv_blocks(z)
```

训练目标是重建损失加 KL 散度：

$$
\mathcal{L}_{\text{VAE}} = \|x - \hat{x}\|_2^2 + \beta \cdot \text{KL}(q(z|x)\|p(z))
$$

第一项保证重建质量，第二项保证 latent 分布的规整性。$\beta$ 通常设置为 0.00001 ~ 0.001，是个玄学参数，需要根据数据集调整。

#### VAE 数学原理深入理解

VAE 的核心思想是学习一个概率分布，而不是直接映射。具体来说：

**编码器学习后验分布**：给定输入视频 $x$，编码器输出均值 $\mu(x)$ 和方差 $\sigma^2(x)$：

$$
q_\phi(z|x) = \mathcal{N}(z; \mu(x), \sigma^2(x)I)
$$

**重参数化技巧**：为了让采样过程可导，使用重参数化：

$$
z = \mu(x) + \sigma(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

这样梯度可以通过 $\mu$ 和 $\sigma$ 反向传播。

**KL 散度的闭式解**：对于高斯分布，KL 散度有解析解：

$$
\text{KL}(q(z|x)\|p(z)) = \frac{1}{2}\sum_{i=1}^{d}\left(\mu_i^2 + \sigma_i^2 - \log\sigma_i^2 - 1\right)
$$

这个公式的直观理解：
- $\mu_i^2$：惩罚均值偏离原点
- $\sigma_i^2 - 1$：惩罚方差偏离 1
- $-\log\sigma_i^2$：防止方差坍缩到 0

**为什么需要 KL 散度项？**

没有 KL 约束的话，编码器会把每个视频映射到完全不同的 latent 空间区域，导致：
1. Latent 空间不连续，无法插值
2. 解码器很难泛化到训练时没见过的 latent
3. 生成时从随机噪声出发会失败

KL 散度强制所有视频的 latent 都集中在标准正态分布附近，保证了 latent 空间的平滑性。

---

## Diffusion Transformer：视频扩散主体

### 扩散原理

扩散模型的训练思想很简单：

1. **前向加噪**：向 latent 逐步加噪声
   $$
   x_t = \sqrt{\alpha_t}x_0 + \sqrt{1-\alpha_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0,I)
   $$
   其中 $\alpha_t = \prod_{i=1}^{t}(1-\beta_i)$，$\beta_t$ 为噪声调度

2. **反向去噪**：训练模型预测噪声
   $$
   \hat{\epsilon} = \epsilon_\theta(x_t, t, c) \quad \text{其中 } c \text{ 为文本条件}
   $$

模型学会"去噪"，就能从噪声恢复 latent。这个过程就像是教模型如何从一团迷雾中逐渐看清图像。

#### 扩散过程的数学推导

**前向扩散的马尔可夫链**（DDPM 标准形式）：

扩散过程可以看作一个马尔可夫链，每一步都向数据中添加一点噪声：

$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)
$$

这里 $\beta_t$ 是噪声调度，通常从 $\beta_1=10^{-4}$ 线性增长到 $\beta_T=0.02$。

**注意**：虽然本章节以 DDPM 为例讲解原理，但实际的 VideoX-Fun 实现（`pipeline_wan.py`）使用 **Flow Matching** 范式，采用向量场预测而非噪声预测。Flow Matching 在稳定性和推理速度上都更优。

**任意时刻的分布（重参数化性质）**：

通过递推，可以直接从 $x_0$ 采样 $x_t$，而不需要逐步加噪：

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon
$$

其中 $\bar{\alpha}_t = \prod_{i=1}^{t}\alpha_i = \prod_{i=1}^{t}(1-\beta_i)$。

这个公式非常关键，它允许训练时随机选择任意时间步 $t$，而不需要从 $t=0$ 一步步加噪到 $t$。

**为什么用这个形式？**

这个形式保证了：
- 当 $t \to 0$ 时，$\bar{\alpha}_t \to 1$，$x_t \approx x_0$（几乎没噪声）
- 当 $t \to T$ 时，$\bar{\alpha}_t \to 0$，$x_t \approx \epsilon$（纯噪声）
- 过程是连续的、可微的

**反向过程的推导**：

反向过程想要从 $x_t$ 恢复 $x_{t-1}$。根据贝叶斯定理：

$$
q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)
$$

其中：

$$
\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t
$$

问题是我们不知道 $x_0$，所以用神经网络 $\epsilon_\theta$ 预测噪声 $\epsilon$，然后反推：

$$
x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}
$$

代入上面的公式，就得到了采样公式。

**训练目标的推导**：

完整的变分下界（ELBO）推导会得到：

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t,x_0,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
$$

这个简化版本在实践中效果最好。原因是它给每个时间步相同的权重，而完整的 ELBO 会给不同时间步不同的权重。

**噪声调度的选择**：

噪声调度 $\beta_t$ 的设计非常重要：

**Linear Schedule**（线性调度）：
$$
\beta_t = \beta_{\text{start}} + \frac{t}{T}(\beta_{\text{end}} - \beta_{\text{start}})
$$
简单但在视频生成中效果一般。

**Cosine Schedule**（余弦调度）：
$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2
$$
在视频生成中效果更好，因为它在前期加噪更慢，保留更多信息。

直观理解：
- Linear：噪声均匀增加，可能太激进
- Cosine：前期慢后期快，更符合人类感知

### Video DiT 模型主体

输入的时空 latent：

$$
z_t \in \mathbb{R}^{C\times T'\times H'\times W'}
$$

会被展平成 token 序列：

$$
\{z_1, z_2, \ldots, z_N\}, \quad N = C \times T' \times H' \times W'
$$

然后进入 Wan 的 Transformer 结构（实现参考：`wan_transformer3d.py`）：

```python
class WanRMSNorm(nn.Module):
    """根均方层归一化，比标准 LayerNorm 更稳定"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x: [B, L, C]
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class WanSelfAttention(nn.Module):
    """
    Wan 的自注意力模块，支持 QK 归一化
    """
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm  # 是否启用 QK 归一化
        self.eps = eps

        # 投影层
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        # QK 归一化（提高数值稳定性）
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16, t=0):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # QK 归一化
        q = self.norm_q(self.q(x.to(dtype))).view(b, s, n, d)
        k = self.norm_k(self.k(x.to(dtype))).view(b, s, n, d)
        v = self.v(x.to(dtype)).view(b, s, n, d)

        # 应用 3D RoPE 位置编码
        q, k = rope_apply_qk(q, k, grid_sizes, freqs)

        # 注意力计算（支持窗口约束）
        x = attention(q.to(dtype), k.to(dtype), v=v.to(dtype),
                     k_lens=seq_lens, window_size=self.window_size)

        # 输出投影
        x = x.flatten(2)
        x = self.o(x)
        return x


class DiTBlock(nn.Module):
    """Diffusion Transformer 块"""
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        # 使用 RMS 归一化而非标准 LayerNorm
        self.norm1 = WanRMSNorm(hidden_dim)
        self.attn = WanSelfAttention(hidden_dim, num_heads, qk_norm=True)
        self.norm2 = WanRMSNorm(hidden_dim)

        # 交叉注意力（用于文本条件）
        self.cross_attn = WanT2VCrossAttention(hidden_dim, num_heads)

        # FFN
        self.norm3 = WanRMSNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x, text_cond, seq_lens, grid_sizes, freqs):
        # Self-attention
        x = x + self.attn(self.norm1(x), seq_lens, grid_sizes, freqs)

        # Cross-attention with text
        x = x + self.cross_attn(self.norm2(x), text_cond)

        # FFN
        x = x + self.mlp(self.norm3(x))
        return x
```

**关键设计点**：
1. **WanRMSNorm**：根均方归一化，比标准 LayerNorm 更稳定（实现位置：`wan_transformer3d.py:173-189`）
2. **QK 归一化**：对 Query 和 Key 进行 RMS 归一化，提高注意力计算的数值稳定性（代码行 227-228）
3. **3D RoPE**：支持三维（时、高、宽）的旋转位置编码
4. **窗口注意力**：可选的局部注意力窗口，进一步降低计算复杂度

#### 时空注意力分解的详细机制

**为什么需要分解？**

假设视频 latent 是 $(T=16, H=64, W=64)$，如果直接做 3D attention：
- Token 数量：$N = 16 \times 64 \times 64 = 65536$
- Attention 矩阵大小：$65536 \times 65536 \approx 4.3$ 亿
- 存储需求（float32）：$4.3 \times 10^9 \times 4 \text{ bytes} = 17.2 \text{ GB}$

这还只是一层、一个 head 的情况！显然不可行。

**时空分解的数学表示**：

**方法1：串行分解（Sequential）**

先做时间 attention，再做空间 attention：

$$
\text{Output} = \text{SpatialAttn}(\text{TemporalAttn}(X))
$$

具体来说：
1. **时间 Attention**：将 $(T, H, W, C)$ 重塑为 $(HW, T, C)$，在时间维度做 attention
   $$
   Z_{\text{temp}} = \text{Attention}(Q_t, K_t, V_t), \quad Q_t, K_t, V_t \in \mathbb{R}^{HW \times T \times d}
   $$
   复杂度：$O(HW \cdot T^2 \cdot d)$

2. **空间 Attention**：将结果重塑为 $(T, HW, C)$，在空间维度做 attention
   $$
   Z_{\text{out}} = \text{Attention}(Q_s, K_s, V_s), \quad Q_s, K_s, V_s \in \mathbb{R}^{T \times HW \times d}
   $$
   复杂度：$O(T \cdot (HW)^2 \cdot d)$

总复杂度：$O(HWT^2d + TH^2W^2d) = O(T(HW)^2 + (HW)T^2)$

对于 $(T=16, HW=4096)$：
- 全 3D：$N^2 = 65536^2 \approx 4.3 \times 10^9$
- 分解后：$T(HW)^2 + (HW)T^2 = 16 \times 4096^2 + 4096 \times 16^2 \approx 2.7 \times 10^8$

降低了约 **16 倍**！

**方法2：并行分解（Parallel）**

同时做时间和空间 attention，然后融合：

$$
\text{Output} = \alpha \cdot \text{TemporalAttn}(X) + \beta \cdot \text{SpatialAttn}(X)
$$

其中 $\alpha, \beta$ 是可学习的权重。这种方法：
- 优点：两个分支可以并行计算，更快
- 缺点：缺少时空信息的交互

**实际实现细节**：

```python
class FactorizedAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        # 时间注意力
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads)
        # 空间注意力
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads)

    def forward(self, x):
        B, C, T, H, W = x.shape

        # 时间注意力：(B, C, T, H, W) -> (B*H*W, T, C)
        x_temp = x.permute(0, 3, 4, 2, 1).reshape(B*H*W, T, C)
        x_temp, _ = self.temporal_attn(x_temp, x_temp, x_temp)
        x_temp = x_temp.reshape(B, H, W, T, C).permute(0, 4, 3, 1, 2)

        # 空间注意力：(B, C, T, H, W) -> (B*T, H*W, C)
        x_spatial = x_temp.permute(0, 2, 1, 3, 4).reshape(B*T, C, H*W)
        x_spatial = x_spatial.permute(0, 2, 1)  # (B*T, H*W, C)
        x_spatial, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)
        x_spatial = x_spatial.permute(0, 2, 1).reshape(B, T, C, H, W)
        x_spatial = x_spatial.permute(0, 2, 1, 3, 4)

        return x_spatial
```

**窗口注意力进一步优化**：

即使分解后，$(HW)^2$ 对于高分辨率仍然很大。可以用窗口注意力：

只在局部窗口内计算 attention（典型窗口大小 $8 \times 8$ 或 $16 \times 16$）：
- 复杂度从 $O((HW)^2)$ 降到 $O(HW \cdot w^2)$，其中 $w$ 是窗口大小
- 代价是失去了全局信息，但可以通过多层或 shifted window 缓解

### 文本引导机制

文本经过编码器转换为特征向量。VideoX-Fun 使用 T5 编码器：

```python
from videox_fun.models import WanT5EncoderModel, AutoTokenizer

# 初始化编码器和分词器
tokenizer = AutoTokenizer.from_pretrained("model_name")
text_encoder = WanT5EncoderModel.from_pretrained("encoder_path")

prompt = "一个人在沙滩上奔跑，阳光明媚"
tokens = tokenizer(prompt, return_tensors="pt", max_length=512, padding="max_length")
text_features = text_encoder(
    input_ids=tokens.input_ids,
    attention_mask=tokens.attention_mask
)[0]  # [1, seq_len, hidden_dim]
```

通过 **cross-attention** 融合进 Transformer：

$$
\text{Attention}(Q=z, K=t, V=t) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

其中 $Q$ 来自视频 latent，$K,V$ 来自文本特征 $t$。模型据此在去噪过程中引入语义信息，确保生成的视频符合文本描述。

**Wan 的文本编码器特点**：

Wan 采用 **T5 文本编码器**（具体实现见 `wan_text_encoder.py` 的 `WanT5EncoderModel`），特点如下：

1. **基于 T5 架构**：
   - 使用改进的 T5 编码器（来自官方 Wan 代码）
   - 支持相对位置编码（T5RelativeEmbedding）
   - 包含自注意力和 FFN 层的堆叠

2. **文本特征处理**：
   - Token embedding 将输入 token 转换为密集向量
   - 通过多层 T5SelfAttention 块逐步精化特征
   - 最后通过 T5LayerNorm 进行归一化

3. **编码输入**：
   ```python
   from transformers import AutoTokenizer
   from videox_fun.models import WanT5EncoderModel

   # 使用对应的分词器
   tokenizer = AutoTokenizer.from_pretrained("model_name")
   text_encoder = WanT5EncoderModel.from_pretrained("encoder_path")

   prompt = "一个人在沙滩上奔跑，阳光明媚"
   tokens = tokenizer(prompt, return_tensors="pt")
   text_features = text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask)[0]
   ```

相比 CLIP 的优势：
- 支持更灵活的文本长度处理
- 相对位置编码对长序列更友好
- T5 架构在大规模文本数据上的预训练效果优秀

#### Cross-Attention 的深入理解

**Self-Attention vs Cross-Attention**：

**Self-Attention**（自注意力）：
$$
\text{Self-Attn}(X) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V, \quad Q=XW_Q, K=XW_K, V=XW_V
$$
$Q, K, V$ 都来自同一个输入 $X$，用于捕捉输入内部的依赖关系。

**Cross-Attention**（交叉注意力）：
$$
\text{Cross-Attn}(X, Y) = \text{softmax}\left(\frac{Q_XK_Y^T}{\sqrt{d}}\right)V_Y
$$
其中：
- $Q = XW_Q$：来自视频 latent（Query："我要找什么？"）
- $K = YW_K, V = YW_V$：来自文本特征（Key/Value："我能提供什么？"）

**直观理解**：

可以把 Cross-Attention 看作一个"查询-检索"过程：
1. 视频 latent 的每个位置问："我应该包含什么内容？"
2. 在文本特征中查找相关信息（通过 $QK^T$ 计算相似度）
3. 根据相似度加权聚合文本信息（通过 softmax 和 $V$）

**注意力权重的可视化**：

假设 prompt 是："一个人在沙滩上奔跑"，注意力权重可能是：

```
视频位置          最关注的文本 token
--------------------------------------------
前景中心区域  →   "人" (0.6), "奔跑" (0.3)
下半部分      →   "沙滩" (0.7), "上" (0.2)
背景          →   "沙滩" (0.5), "一个" (0.3)
运动轨迹      →   "奔跑" (0.8)
```

**Classifier-Free Guidance（CFG）**：

实际使用中，通常结合有条件和无条件生成来增强控制：

$$
\hat{\epsilon}_{\text{guided}} = \epsilon_\theta(z_t, t, \emptyset) + s \cdot (\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset))
$$

其中：
- $\epsilon_\theta(z_t, t, c)$：有文本条件的预测
- $\epsilon_\theta(z_t, t, \emptyset)$：无条件预测（文本为空）
- $s$：引导强度（通常 7.5-15）

这个技巧的作用：
- $s=0$：完全忽略文本，随机生成
- $s=1$：正常的条件生成
- $s>1$：过度强调文本，生成更符合描述但可能失真

**多模态融合的其他方法**：

除了 Cross-Attention，还有：

1. **AdaLN（Adaptive Layer Normalization）**：
   $$
   \text{AdaLN}(x, c) = c_s \cdot \text{LayerNorm}(x) + c_b
   $$
   文本条件 $c$ 通过 MLP 预测缩放和偏移参数 $c_s, c_b$。

2. **Concatenation**：
   直接把文本特征拼接到视频 latent：
   $$
   z_{\text{combined}} = [z; \text{Repeat}(t)]
   $$

3. **FiLM（Feature-wise Linear Modulation）**：
   $$
   \text{FiLM}(x, c) = \gamma(c) \odot x + \beta(c)
   $$

Cross-Attention 在视频生成中效果最好，因为它允许精细的、位置相关的文本控制。

### 训练损失

训练过程就是标准的去噪回归：

$$
\mathcal{L}_{\text{diff}} = \mathbb{E}_{z_0, t, \epsilon}[\|\epsilon - \hat{\epsilon}_\theta(z_t, t, c)\|_2^2]
$$

完整训练流程：

```python
def train_step(video_batch, prompt_batch):
    # 1. 编码视频到 latent
    z0 = vae_encoder(video_batch)  # (B, C, T', H', W')

    # 2. 随机采样时间步 t
    t = torch.randint(0, num_timesteps, (batch_size,))

    # 3. 采样高斯噪声
    epsilon = torch.randn_like(z0)

    # 4. 前向扩散：加噪
    zt = sqrt_alpha[t] * z0 + sqrt_one_minus_alpha[t] * epsilon

    # 5. 编码文本提示
    text_cond = text_encoder(prompt_batch)

    # 6. 模型预测噪声
    epsilon_pred = diffusion_transformer(zt, t, text_cond)

    # 7. 计算损失
    loss = F.mse_loss(epsilon_pred, epsilon)

    # 8. 反向传播
    loss.backward()
    optimizer.step()

    return loss
```

简单粗暴，但就是有效。

---

## 推理流程

生成视频时的反向过程：

```python
# VideoX-Fun 实际实现（pipeline_wan.py:386+）
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

@torch.no_grad()
def generate_video(
    self,
    prompt: str,
    num_frames: int = 49,
    height: int = 480,
    width: int = 640,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[str] = None
):
    # 1. 编码文本（正向和负向）
    prompt_embeds = self.encode_prompt(prompt)           # [B, L_text, 768]
    neg_prompt_embeds = self.encode_prompt(negative_prompt or "")

    # 2. 获取时间步序列（支持 Flow Matching 调度器）
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,  # FlowMatchEulerDiscreteScheduler 或其他 Flow Matching 调度器
        num_inference_steps,
        device=self.device
    )

    # 3. 初始化随机噪声 latent
    latents = torch.randn(
        batch_size=1,
        num_channels=4,
        num_frames=num_frames // 4,     # 时间下采样
        height=height // 8,              # 空间下采样
        width=width // 8,
        device=self.device,
        dtype=self.dtype
    )  # [B, C, T', H', W']

    # 4. 去噪循环（Flow Matching）
    for t_idx, t in enumerate(timesteps):
        # 分类器自由引导（CFG）：复制 latent 以计算有条件和无条件预测
        latent_model_input = torch.cat([latents, latents], dim=0) if do_classifier_free_guidance else latents
        if hasattr(self.scheduler, "scale_model_input"):
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latent_model_input.shape[0])

        # Transformer 前向（预测向量场，不是噪声）
        model_output = self.transformer(
            x=latent_model_input,
            context=in_prompt_embeds,
            t=timestep,
            seq_len=seq_len,
        )  # [2B, C, T', H', W']

        # CFG 加权
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = model_output.chunk(2)
            model_output = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # 调度器步骤（Flow Matching）
        latents = self.scheduler.step(model_output, t, latents).prev_sample

    # 5. VAE 解码
    video = self.vae.decode(latents).sample  # [B, T, H, W, 3]

    return video
```

### 支持的调度器（Flow Matching）

VideoX-Fun 支持以下调度器（`pipeline_wan.py:8-19`）：

| 调度器 | 类型 | 推荐步数 | 特点 |
|--------|------|---------|------|
| FlowMatchEulerDiscreteScheduler | 一阶欧拉 | 30-50 | 简单快速 |
| FlowDPMSolverMultistepScheduler | ODE求解 | 15-25 | **最快最精准** |
| FlowUniPCMultistepScheduler | UniPC多步 | 20-30 | 平衡精度和速度 |

### TeaCache KV 缓存优化

**实现位置**：`models/cache_utils.py`

TokenEarlyExit Attention (TeaCache) 基于观察：在序列生成过程中，后续帧的 attention 特征与前面帧的相似度往往很高。可以基于相似度阈值复用缓存的 K/V：

**性能收益**：
- KV 计算时间减少 **30-50%**
- 显存占用减少 **20-30%**
- 对生成质量影响小（< 2%）

**参数调整**：
- 阈值 0.05-0.10：激进复用，加速明显
- 阈值 0.10-0.20：**平衡模式（推荐）**
- 阈值 0.20-0.30：保守模式，质量更好

#### Flow Matching 采样方法（VideoX-Fun 的实现）

**重要说明**：虽然扩散模型的理论基于 DDPM、DDIM 等，但 VideoX-Fun 的实际实现使用的是更现代的 **Flow Matching** 范式。以下详细说明。

**Flow Matching vs DDPM**：

|方面|DDPM|Flow Matching|
|----|----|------------|
|预测目标|噪声 $\epsilon$|速度/向量场 $v(x_t, t)$|
|调度器|噪声调度 $\beta_t$|sigma 调度 $\sigma_t$|
|稳定性|需要谨慎调参|更稳定，收敛更快|
|推理速度|需要较多步数|较少步数达到同等质量|
|实现方式|噪声预测→去噪|直接学习路径速度|

**Flow Matching 采样过程**（VideoX-Fun 实现）：

```python
# 实际使用（来自 pipeline_wan.py）
from diffusers import FlowMatchEulerDiscreteScheduler

# 方法1：一阶欧拉（简单快速）
scheduler = FlowMatchEulerDiscreteScheduler()
latents = scheduler.step(model_output, t, latents).prev_sample

# 方法2：ODE 求解（精确高效）
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
scheduler = FlowDPMSolverMultistepScheduler()
```

**采样步数建议（基于 Flow Matching）**：

- 10-15 步：快速预览，质量较低
- 20-30 步：**生产环境推荐**，质量-速度平衡
- 30-50 步：高质量输出
- 50+ 步：研究/演示用途

**历史背景（仅供参考）**：

注：以下 DDPM、DDIM、DPM-Solver 的讨论仅供理论参考。VideoX-Fun 实现不使用这些采样器。

**DDPM（Denoising Diffusion Probabilistic Models）**：

DDPM 是最原始的采样方法，严格按照训练时的反向过程：

$$
z_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(z_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\hat{\epsilon}_\theta(z_t, t, c)\right) + \sigma_t \epsilon
$$

特点：
- 需要完整的 $T$ 步（通常 1000 步）
- 速度最慢，但质量最好

**DDIM（Denoising Diffusion Implicit Models）**：

DDIM 改进了 DDPM，允许跳步采样，通常 20-50 步就能得到不错结果。

**DPM-Solver（Diffusion Probabilistic Model Solver）**：

DPM-Solver 使用 ODE 求解方法，10-20 步可达到 DDIM 50 步的质量。

---

## 多阶段层级生成架构

前面介绍的是单阶段生成流程，但实际上 **Wan 采用的是多阶段层级生成**，这也是 Sora、HunyuanVideo 等顶级模型的共同特征。

### 为什么需要多阶段生成？

直接生成高分辨率、高帧率视频面临三个核心问题：

1. **显存限制**：
   - 高分辨率视频的 latent 空间维度很大
   - Attention 矩阵随序列长度平方增长
   - 单卡难以容纳完整计算图

2. **训练难度**：
   - 高分辨率数据稀缺且昂贵
   - 直接在高分辨率上训练容易过拟合
   - 梯度传播困难

3. **语义与细节的解耦**：
   - 低分辨率阶段专注语义和运动结构
   - 高分辨率阶段专注纹理细节
   - 分开建模更高效

### Wan 的三阶段架构

**阶段 1：Base Model（基础生成）**

生成低分辨率、低帧率的 latent（如 480p@16fps）：

```
输入：文本 prompt
输出：基础视频 latent (如 480p@16fps)
```

这一阶段的重点是：
- **语义一致性**：确保内容符合文本描述
- **运动结构**：建立物体的运动轨迹
- **时空布局**：确定场景的基本构成

**阶段 2：Temporal Super-Resolution（时间上采样）**

提升时间分辨率，让运动更流畅（如 16fps → 48fps）：

```
输入：低帧率 latent (16fps)
输出：高帧率 latent (48fps)
```

核心技术：

1. **光流一致性损失**：
   $$
   \mathcal{L}_{\text{flow}} = \sum_{t=1}^{T-1}\|I_{t+1} - \text{Warp}(I_t, F_{t\to t+1})\|^2
   $$
   确保插值帧与光流预测一致。

2. **双向插值**：
   同时参考前后帧：
   $$
   I_{\text{mid}} = \alpha \cdot \text{Warp}(I_{\text{prev}}, F_{\text{fwd}}) + (1-\alpha) \cdot \text{Warp}(I_{\text{next}}, F_{\text{bwd}})
   $$

3. **时间注意力加权**：
   关键帧（原始 16 帧）权重更高，插值帧参考关键帧。

**阶段 3：Spatial Super-Resolution（空间上采样）**

提升空间分辨率，增加细节（如 480p → 1080p）：

```
输入：低分辨率、高帧率 latent (480p@48fps)
输出：高分辨率、高帧率 latent (1080p@48fps)
```

核心技术：

1. **Frame-wise VAE Upsampler**：
   对每一帧独立做上采样，然后用时间一致性约束：
   $$
   \mathcal{L}_{\text{temp}} = \sum_{t=1}^{T-1}\|\nabla I_t - \nabla I_{t-1}\|^2
   $$

2. **高频细节注入**：
   使用 wavelet transform 分离低频和高频：
   ```python
   low_freq, high_freq = wavelet_decompose(frame)
   # Base model 生成低频
   # SR model 专注生成高频细节
   upsampled = combine(upsample(low_freq), generate(high_freq))
   ```

3. **GAN 损失**（可选）：
   使用判别器增强纹理真实感：
   $$
   \mathcal{L}_{\text{GAN}} = \mathbb{E}[\log D(x_{\text{real}})] + \mathbb{E}[\log(1-D(G(z)))]
   $$

### 层级生成的优势

相比单阶段生成：

1. **显存效率**：
   - 单阶段需要一次性加载整个高分辨率计算图
   - 多阶段可分别运行，每阶段显存需求更低

2. **训练效率**：
   - Base model 用大规模低分辨率数据快速收敛
   - SR model 用较少高质量数据精细调优

3. **质量提升**：
   - 分阶段建模让每个模型专注特定任务
   - 时空一致性显著改善

4. **灵活性**：
   - 可以只跑 Base model（快速预览）
   - 根据需求选择是否上采样

### 实际推理流程

**说明**：在 VideoX-Fun 当前版本中，主要实现是单阶段生成（Base Model）。多阶段架构（TSR、SSR）在官方 Wan 中存在，但在开源的 VideoX-Fun 中还未完整实现。以下是概念性的多阶段流程示例：

```python
@torch.no_grad()
def generate_video_hierarchical(prompt, target_resolution='720p', target_fps=24):
    # 当前 VideoX-Fun 支持的：单阶段生成
    # 基础生成（480p，49 帧）
    video = pipeline(
        prompt=prompt,
        height=480,
        width=640,
        num_frames=49,
        num_inference_steps=30,
        guidance_scale=7.5
    )

    # 注：以下多阶段功能在官方 Wan 中存在，但 VideoX-Fun 开源版本还未实现
    #
    # # 阶段 2：时间上采样（如果需要更高帧率）
    # tsr_video = temporal_sr_model(video)
    #
    # # 阶段 3：空间上采样（如果需要更高分辨率）
    # ssr_video = spatial_sr_model(tsr_video)

    return video
```

**多阶段架构说明**：

虽然官方 Wan 模型采用多阶段架构（Base + TSR + SSR），但 VideoX-Fun 开源实现目前主要提供单阶段生成。这种简化的设计：
- 更容易理解和使用
- 对显存要求更低
- 已经可以生成 480p-720p 的高质量视频

多阶段架构是 Wan、Sora、HunyuanVideo 等官方实现的标准范式，用于支持超高分辨率视频生成，但不是必需的。

---

## 训练细节与显存优化

### 数据集与预处理

**数据规模**：
- 训练数据：大规模视频-文本对
- 数据来源：影视素材、广告、短视频、游戏录屏
- 多语言支持：中英文为主，兼顾其他语言

**预处理流程**：

1. **质量过滤**：
   ```python
   def filter_video(video, text):
       # 1. 美学评分
       aesthetic_score = aesthetic_model(video)
       if aesthetic_score < threshold_aesthetic:
           return False

       # 2. CLIP 相似度（文本-视频对齐）
       clip_score = clip_similarity(text, video)
       if clip_score < threshold_clip:
           return False

       # 3. 技术质量（分辨率、帧率、编码）
       if not meets_technical_requirements(video):
           return False

       return True
   ```

2. **时长采样**：
   - Base model：短片段训练
   - TSR model：相同片段，更高帧率标注
   - SSR model：关键帧 + 高分辨率版本

3. **数据增强**：
   ```python
   augmentations = [
       RandomCrop(),              # 随机裁剪
       RandomHorizontalFlip(),    # 水平翻转
       ColorJitter(0.1),          # 颜色抖动
       TemporalCrop(),            # 时间裁剪
       FrameRateSampling(),       # 帧率采样
   ]
   ```

### 训练策略

**分辨率渐进训练（Progressive Training）**：

不同阶段使用不同分辨率，逐步提升：

```
第 1 阶段：
  - 低分辨率、低帧率
  - 较大 batch size
  - 较高学习率
  - 目标：快速收敛基本结构

第 2 阶段：
  - 中等分辨率、中等帧率
  - 中等 batch size
  - 降低学习率
  - 目标：提升细节和时间一致性

第 3 阶段：
  - 高分辨率、高帧率
  - 较小 batch size
  - 低学习率
  - 目标：精细调优
```

**混合精度训练（Mixed Precision）**：

使用 FP16/BF16 降低显存和加速训练：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for video, text in dataloader:
    optimizer.zero_grad()

    # 前向传播用 FP16
    with autocast():
        loss = model(video, text)

    # 反向传播处理精度
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

效果：
- 显存占用大幅减少
- 训练速度显著提升
- 质量几乎无损

**梯度检查点（Gradient Checkpointing）**：

不保存所有中间激活，需要时重新计算：

```python
from torch.utils.checkpoint import checkpoint

class DiTBlock(nn.Module):
    def forward(self, x, text):
        # 使用 checkpoint 包裹
        return checkpoint(self._forward, x, text)

    def _forward(self, x, text):
        # 实际计算
        x = self.attention(x)
        x = self.cross_attention(x, text)
        x = self.mlp(x)
        return x
```

效果：
- 显存明显减少
- 训练时间略有增加
- 适合显存不足的情况

**XFuser 多卡并行策略**（实现位置：`dist/fuser.py`）：

VideoX-Fun 使用 XFuser 而非 DeepSpeed ZeRO，支持多种并行方式：

```python
from videox_fun.dist import set_multi_gpus_devices

# 配置多卡参数
set_multi_gpus_devices(
    ulysses_degree=4,           # 序列并行度（沿序列长度）
    ring_degree=2,              # 环形并行度（优化通信）
    classifier_free_guidance_degree=1  # CFG 维度并行
)
# 总 GPU 数 = 4 × 2 × 1 = 8 张卡
```

**并行维度解析**：
1. **Ulysses 序列并行**：沿序列长度分割注意力计算
   - 计算复杂度从 $O(N^2)$ 降至 $O((N/K)^2)$，K 为并行度
   - 额外通信：All-to-All

2. **Ring Attention**：环形通信优化
   - 改进 KV 缓存共享策略
   - 降低通信开销约 30%
   - 支持长视频生成（256+ 帧）

3. **CFG 并行**：有条件和无条件推理在不同 GPU 执行

**效果**：
- 8 卡可有效推理 1080p@49 帧视频
- 显存消耗更均衡
- 支持更长的视频序列

### 关键损失函数

除了基本的去噪损失，Wan 还使用了多个辅助损失：

**1. 时空一致性损失**：

$$
\mathcal{L}_{\text{consistency}} = \underbrace{\|\nabla_t z_t - \nabla_t z_{t-1}\|^2}_{\text{时间平滑}} + \underbrace{\|\nabla_x z_t\|_{\text{TV}}}_{\text{空间平滑}}
$$

确保相邻帧之间变化平滑，避免抖动。

**2. 感知损失（Perceptual Loss）**：

$$
\mathcal{L}_{\text{perceptual}} = \sum_{l}\|\phi_l(x) - \phi_l(\hat{x})\|^2
$$

其中 $\phi_l$ 是预训练 VGG 的第 $l$ 层特征。比 L2 损失更符合人类感知。

**3. 光流一致性损失**（用于 TSR）：

$$
\mathcal{L}_{\text{flow}} = \sum_{t}\|I_{t+1} - \text{Warp}(I_t, F_{t\to t+1})\|^2
$$

确保插值帧符合运动轨迹。

**总损失**：

$$
\mathcal{L}_{\text{total}} = \lambda_1\mathcal{L}_{\text{diff}} + \lambda_2\mathcal{L}_{\text{consistency}} + \lambda_3\mathcal{L}_{\text{perceptual}} + \lambda_4\mathcal{L}_{\text{flow}}
$$

这些权重需要根据具体任务调整。

---

## 关键设计汇总

| 模块 | 技术核心 | 功能 | 计算复杂度 | 关键优势 |
|------|--------|------|---------|---------|
| Video VAE | 3D卷积 (CausalConv3d) + 残差结构 | 压缩/解压视频 | O(THW) | 降低计算量 8-16x |
| Diffusion Transformer | 分解式3D注意力 + QK归一化 + Cross-Attention | 建模时空 latent | O(TH²W² + T²HW) | 长程依赖建模 |
| 文本编码器 | T5 编码器 (WanT5EncoderModel) | 文本→语义向量 | O(seq_len²) | 相对位置编码，灵活长度处理 |
| 单阶段生成 | Base Model (480p@49帧) | 文本条件生成 | 分解注意力降显存 | 高质量视频 |
| 训练优化 | FP16 + XFuser + Checkpointing | 高效分布式训练 | 显存减少 50% | 多卡并行 (Ulysses + Ring) |
| 推理加速 | Flow Matching 调度器 + TeaCache | 快速采样 | 20-50 步 | 推理加速 30-50% |

---

## 实际案例

假设我们输入这样的 prompt：

> "一个人在沙滩上奔跑，阳光明媚，海浪拍打着海岸。"

生成过程大概是这样的：

1. **文本编码**：提取"人"、"沙滩"、"奔跑"、"阳光"、"海浪"等语义特征
2. **初始状态**：随机 latent，看起来就是一团噪声
3. **去噪步骤1-10**：模糊的色块开始出现，能隐约看出天空和沙滩的区域
4. **去噪步骤11-30**：人物轮廓逐渐清晰，能看出在运动
5. **去噪步骤31-50**：细节完善，动作连贯，海浪的动态效果出现
6. **解码输出**：最终的高清视频，每帧都很清晰，动作流畅自然

可视化过程：

```
第 1 步：随机噪声
    [█████████████████████]  (全是随机块)

第 10 步：粗轮廓显现
    [░░█████░░░░███░░░░░░]  (看到大轮廓)

第 30 步：细节逐渐填充
    [░░▒▒███▒▒▒▒███▒▒░░░░]  (人形、海滩、海浪)

第 50 步：最终高质量视频
    [人物 + 沙滩 + 海浪 + 光影效果]
```

整个过程就像在迷雾中逐渐看清画面，先是大的轮廓，然后是细节，最后是精致的纹理。

---

## 一些技术细节和坑

### 时空一致性问题

生成视频最大的挑战不是单帧质量，而是**时空一致性**。你可能会遇到：
- 人物突然变形
- 场景突然跳变
- 动作不连贯

Wan 通过时间维 attention 来解决这个问题，但还是会有玄学成分。有时候换个 prompt 写法效果就完全不同。

#### 时空一致性的深层原因

**为什么会不一致？**

1. **独立的去噪过程**：
   即使有时间 attention，每一帧的去噪仍然有一定独立性。在高噪声阶段（$t$ 大），帧间的关联较弱，可能产生不同的高频细节。

2. **Attention 的局部性**：
   Attention 虽然能建模长程依赖，但在实践中，由于 softmax 的特性，往往更关注近邻帧。对于 16 帧的视频，第 1 帧和第 16 帧的关联可能很弱。

3. **噪声的累积效应**：
   每步采样都可能引入微小的误差，在视频中这些误差会跨帧累积，导致"漂移"现象。

**改进方法**：

**1. 分层生成（Hierarchical Generation）**：

先生成关键帧，再插值中间帧：

```python
# 第一阶段：生成关键帧 (如 0, 4, 8, 12, 16)
keyframes = generate_video(prompt, frames=[0, 4, 8, 12, 16])

# 第二阶段：以关键帧为条件，生成中间帧
for i in [2, 6, 10, 14]:
    frames[i] = generate_frame(
        prompt,
        prev_frame=keyframes[i-2],
        next_frame=keyframes[i+2]
    )
```

**2. 光流约束（Optical Flow Constraint）**：

在训练时添加光流一致性损失：

$$
\mathcal{L}_{\text{flow}} = \|I_{t+1} - \text{Warp}(I_t, F_{t\to t+1})\|_2^2
$$

其中 $F_{t\to t+1}$ 是从第 $t$ 帧到第 $t+1$ 帧的光流。

**3. 循环一致性（Cycle Consistency）**：

确保前向生成和后向生成一致：

$$
\mathcal{L}_{\text{cycle}} = \|I_t - G_{\text{backward}}(G_{\text{forward}}(I_t))\|_2^2
$$

**4. 更长的上下文窗口**：

使用 sliding window attention，让每一帧能看到更多前后帧：

```python
# 普通：每帧只看前后 2 帧
attention_window = 5

# 改进：每帧看前后 8 帧
attention_window = 17
```

代价是计算量增加。

### 显存和速度

即使有 VAE 压缩，生成长视频仍然很吃显存。通常的做法是：
- 先生成短片段
- 然后用滑动窗口的方式拼接
- 或者直接分段生成再后期融合

速度受显卡性能影响较大，生成时间随分辨率和帧数增长。

#### 显存瓶颈的数学分析

**显存占用的主要来源**：

1. **模型参数**：
   - Attention 层的权重矩阵
   - FFN 层的权重
   - 参数量随模型规模增长

2. **Attention 矩阵**：
   - 时间 attention 矩阵
   - 空间 attention 矩阵（通常最大）
   - 每层都需要存储

3. **中间激活**：
   - 每层的输出特征
   - 多层累积
   - 受 batch size 和 latent 尺寸影响

4. **梯度（训练时）**：
   - 训练时需要存储梯度
   - 显存需求约为推理的2倍

**长视频生成策略**：

**策略1：Sliding Window（滑动窗口）**

```python
def generate_long_video(prompt, total_frames=64, window_size=16, overlap=4):
    frames = []

    for start in range(0, total_frames, window_size - overlap):
        end = min(start + window_size, total_frames)

        # 生成窗口
        if start == 0:
            window = generate_video(prompt, num_frames=window_size)
        else:
            # 以前面的帧为条件
            window = generate_video(
                prompt,
                num_frames=window_size,
                init_frames=frames[-overlap:]  # 重叠部分作为条件
            )

        # 添加非重叠部分
        frames.extend(window[overlap:] if start > 0 else window)

    return frames
```

这种方法的问题：拼接处可能有不连续。

**策略2：AutoRegressive（自回归）**

```python
def generate_autoregressive(prompt, total_frames=64, chunk_size=16):
    frames = []

    # 生成第一个 chunk
    chunk = generate_video(prompt, num_frames=chunk_size)
    frames.extend(chunk)

    # 逐个生成后续 chunk
    while len(frames) < total_frames:
        # 以最后几帧为条件
        chunk = generate_next_chunk(
            prompt,
            condition_frames=frames[-4:],  # 最后 4 帧
            num_frames=chunk_size
        )
        frames.extend(chunk)

    return frames[:total_frames]
```

**策略3：Latent Space Interpolation（潜空间插值）**

生成首尾两个片段，在 latent 空间插值：

```python
# 生成开始和结束
start_latent = generate_latent(prompt_start)  # t=0 的 latent
end_latent = generate_latent(prompt_end)      # t=T 的 latent

# 线性插值
latents = []
for alpha in np.linspace(0, 1, num_frames):
    latent = (1-alpha) * start_latent + alpha * end_latent
    latents.append(latent)

# 解码
video = vae_decoder(torch.stack(latents))
```

这种方法快但控制力弱。

**显存优化技巧**：

1. **梯度检查点（Gradient Checkpointing）**：
   不存储所有中间激活，需要时重新计算。以时间换空间。

2. **混合精度（Mixed Precision）**：
   使用 float16 而非 float32，显存减半，速度显著提升。

3. **CPU Offloading**：
   将不活跃的层参数放到 CPU，需要时再搬到 GPU。

### 采样器的选择

VideoX-Fun 采用 **Flow Matching** 范式，支持多种高效的采样器：

| 采样器 | 推荐步数 | 推理时间 | 质量 | 特点 |
|--------|---------|---------|------|------|
| FlowMatchEulerDiscreteScheduler | 30-50 | 标准 | 优 | 一阶欧拉，简单稳定 |
| FlowDPMSolverMultistepScheduler | 15-25 | **最快** | 优 | ODE求解，精度高 |
| FlowUniPCMultistepScheduler | 20-30 | 快 | 优 | 平衡精度和速度 |

**推荐配置**：生产环境使用 **FlowDPMSolverMultistepScheduler**（20步）或 **FlowMatchEulerDiscreteScheduler**（30步），可实现 20-30 秒内生成 480p@49帧视频。

结合 **TeaCache** 优化，推理时间可进一步减少 30-50%。

### 工程优化建议

**显存优化：**
- 梯度累积：模拟更大 batch size
- Attention 分解：时空分离显著降低显存
- VAE 缓存：预计算所有编码，节省训练显存

**质量优化：**
- 采样器选择：FlowDPMSolverMultistepScheduler 平衡速度与质量
- 引导强度：guidance_scale 设置为 6-8（过大会导致失真）
- 采样步数：根据质量需求选择 20-30 步（标准）或 30-50 步（高质量）

**加速策略：**
- 模型剪枝：去除低重要性头部
- 量化：8-bit 推理显著加速
- 批处理：同时生成多个视频摊销开销

---

## 总结

Wan 的核心思想可以概括为：**Video VAE + Diffusion Transformer + 文本交叉注意力**。

| 阶段 | 输入 | 输出 | 说明 |
|------|------|------|------|
| 编码 | 视频帧 (B,3,T,H,W) | 时空 latent (B,C,T',H',W') | 时间4倍，空间8倍压缩 |
| 加噪 | latent + 时间步 t | 噪声版 latent | 前向扩散 |
| 去噪 | 噪声 latent + 文本 + t | 预测噪声 | 反复调用（主要耗时）|
| 解码 | 干净 latent | 视频帧 (B,3,T,H,W) | 升维恢复 |

视频生成的核心在于"时空一致性"的建模。VAE 提供了高效的 latent 表征，Transformer 提供了长程依赖建模能力，文本条件通过跨模态注意力实现语义约束。

当然，工程实现上有很多折衷：压缩率怎么选、attention 怎么分解、采样器如何优化效率，这些都需要大量实验调参。不过整体架构还是相当优雅的。

如果你也在做视频生成相关的工作，Wan 的这套**时空 latent + 分解注意力**的思路很值得借鉴。特别是分解式注意力的设计，可以推广到其他需要处理高维数据的场景。

最后一个小建议：真要用这个模型的话，多准备些显存，多试试不同的 prompt 写法。视频生成还是个很玄学的领域，有时候一个词的差异就能让效果天差地别。

---

## Wan 2.x 的改进方向

**说明**：官方 Wan 2.x（包括 2.2 及更新版本）在以下方向有改进，但 VideoX-Fun 开源版本对这些特性的支持可能不完整：

- **长视频生成**：Ring Attention 机制。VideoX-Fun 通过 XFuser 的 Ulysses + Ring 并行已支持 256+ 帧。
- **时空一致性**：多尺度时间建模和物理约束。
- **MoE 架构**：Mixture of Experts 用于提升模型容量。
- **推理加速**：渐进式蒸馏实现 2-3 倍加速。
- **多模态控制**：支持更多类型的控制信号和相机轨迹。

这些改进的详细实现可参考官方 Wan 论文和代码仓库。相比 Wan 1.x，这些特性显著提升了生成质量、推理速度和控制灵活性，但具体实现细节超出 VideoX-Fun 开源版本的范围。

**在 VideoX-Fun 中的实际支持**：
- **XFuser 多卡并行**：已实现 Ulysses + Ring 序列并行策略
- **TeaCache 优化**：已支持 KV 缓存相似度复用
- **Flow Matching 采样**：已支持高效的 Flow Matching 调度器（Euler、DPMSolver、UniPC）
- **TSR/SSR 多阶段**：官方特性，开源版本主要实现单阶段生成
- **完整的 T5 编码器**：采用改进的 T5 编码器（WanT5EncoderModel）

总的来说，VideoX-Fun 是官方 Wan 模型的开源实现，虽然不包含所有最新的优化，但已经提供了接近原始性能的生成能力，非常适合研究、学习和实际应用。
