---
title: "Scalable Diffusion Models with Transformers"
authors:
  - William Peebles
  - Saining Xie
year: 2023
venue: "Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)"
doi: "10.1109/ICCV51070.2023.00387"
source: "/mnt/data/DiT.pdf"
tags:
  - diffusion
  - transformer
  - DiT
  - latent-diffusion
  - conditional-generation
---

> [!abstract] 原文摘要翻译
> 本文探索了一类基于 Transformer 架构的新型扩散模型。作者训练图像的潜空间扩散模型，用一个作用于潜变量 patch 的 Transformer 替换常用的 U-Net 主干网络，并以一次前向传播的计算复杂度（Gflops）为尺度分析 Diffusion Transformer（DiT）的可扩展性。实验发现，无论通过增加 Transformer 的深度/宽度，还是通过增加输入 token 数，只要 DiT 的 Gflops 提升，其 Fréchet Inception Distance（FID，越低越好）都稳定下降。除具有良好的规模扩展性质外，最大的 DiT-XL/2 在 ImageNet 512×512 与 256×256 类别条件生成基准上超过此前的扩散模型，其中在 256×256 上达到 FID 2.27。

> [!note] Obsidian 公式兼容性
> 本笔记的行内公式统一使用单美元符号包围，独立公式统一使用双美元符号包围，以确保 Obsidian MathJax 正常识别。

# 核心结论

这篇论文最重要的结论不是“Transformer 也能做扩散”，而是：

1. **扩散模型并不依赖 U-Net 的卷积归纳偏置。** 在 Latent Diffusion Model（LDM，潜空间扩散模型）框架中，标准 Vision Transformer（ViT，视觉 Transformer）式主干可以直接承担噪声预测网络的角色。
2. **DiT 的生成质量与前向计算量 Gflops 强相关。** 论文在 12 个模型上得到 Gflops 与 FID-50K 的相关系数约为 **-0.93**。
3. **条件信息如何注入 Transformer 极其关键。** 作者比较 in-context token、cross-attention、adaptive Layer Normalization（adaLN，自适应层归一化）和 adaLN-Zero，最终发现 **adaLN-Zero 在质量与计算效率上均最好**。
4. **增加模型本身的计算能力，比单纯增加采样步数更有效。** 小模型即便用更多采样计算，也难以补偿主干模型容量不足。
5. 在类别条件 ImageNet 上，DiT-XL/2 配合 Classifier-Free Guidance（CFG，无分类器引导）达到 **256×256 FID 2.27**、**512×512 FID 3.04**。

# 1. 问题与动机

扩散模型在本文之前已经成为图像生成的重要范式，但其主干网络几乎统一采用卷积 U-Net。Ho 等人的 Denoising Diffusion Probabilistic Model（DDPM，去噪扩散概率模型）把 U-Net 引入扩散模型后，后续工作虽然修改了残差块、注意力、归一化和条件注入方式，但整体的多尺度 U-Net 结构长期没有发生根本变化。

作者提出两个问题：

- **U-Net 的卷积、多尺度结构究竟是扩散模型性能的必要条件，还是历史惯性？**
- 如果把扩散模型主干换成已经在语言和视觉领域表现出优秀 scaling 行为的 Transformer，模型是否也能随着计算量增加而稳定提升？

因此本文不是主要在发明新的扩散目标函数，而是在研究 **扩散模型的主干架构设计空间与 scaling law**。

# 2. 方法总览

DiT 采用一个混合式生成管线：

```text
RGB 图像 x
  ↓ 预训练 VAE Encoder E（冻结）
latent z
  ↓ 前向扩散加噪
z_t
  ↓ Patchify + 位置编码
token sequence
  ↓ N 个 DiT Block（条件：timestep + class label）
预测噪声 ε̂ 与协方差 Σ̂
  ↓ 反向扩散采样
z_0
  ↓ 预训练 VAE Decoder D
生成图像 x̂
```

也就是说：

> [!important] Patchify 的对象
> DiT Patchify 的对象是 **VAE 编码后的 latent feature map，而不是原始 RGB 像素**。例如 256×256×3 图像先经 VAE Encoder 变成 32×32×4 latent，再把这个 latent 切成 patch tokens。

- **VAE 仍是卷积网络**，负责图像空间和低分辨率潜空间之间的压缩/解码；
- **扩散去噪主干完全由 Transformer 替代 U-Net**；
- Transformer 处理的不是原始像素，而是 VAE 潜变量切成的 patch token。


# 3. 扩散模型基础

## 3.1 前向加噪

Gaussian diffusion 的前向过程逐渐把真实样本 $x_0$ 变成高斯噪声：

$$
q(x_t|x_0)
=
\mathcal N
\left(
x_t;\sqrt{\bar{\alpha}_t}x_0,
(1-\bar{\alpha}_t)I
\right)
$$

利用重参数化，可以直接从 $x_0$ 得到任意时刻的 $x_t$：

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,I)
$$

## 3.2 反向过程

模型学习：

$$
p_\theta(x_{t-1}|x_t)
=
\mathcal N
\left(
\mu_\theta(x_t),
\Sigma_\theta(x_t)
\right)
$$

通常把均值预测改写为噪声预测网络 $\epsilon_\theta$，用简单均方误差训练：

$$
L_{\mathrm{simple}}
=
\left\|
\epsilon_\theta(x_t)-\epsilon_t
\right\|_2^2
$$

本文沿用 Improved DDPM / ADM 的做法：

- $\epsilon_\theta$ 用 $L_{\mathrm{simple}}$ 训练；
- 反向过程的协方差 $\Sigma_\theta$ 仍使用完整变分目标中的 KL 项训练。

因此 DiT 最终同时输出：

- 噪声预测 $\hat\epsilon$
- 对角协方差预测 $\hat\Sigma$

# 4. 为什么在潜空间中做扩散

直接在高分辨率像素空间上运行 Transformer 会产生极长序列和很大的计算量。本文沿用 LDM 的两阶段思路：

$$
z=E(x),\qquad x=D(z)
$$

其中：

- $E$：预训练 VAE 编码器；
- $D$：预训练 VAE 解码器；
- 训练 DiT 时 VAE **冻结**；
- DiT 学的是 $z$ 的扩散分布，而不是 $x$ 的像素分布。

论文直接使用 Stable Diffusion 的预训练 VAE，其下采样倍率为 8：

$$
256\times256\times3
\rightarrow
32\times32\times4
$$

因此 256×256 图像对应的 DiT 空间输入为 $32\times32\times4$。

> [!important]
> DiT 并不是“完全取消卷积”。论文取消的是 **扩散模型去噪主干中的 U-Net**；图像压缩/解码仍使用现成的卷积 VAE。

# 5. Patchify：把潜变量变成 Transformer token

设潜变量空间尺寸为：

$$
I\times I\times C
$$

选择 patch 大小 $p\times p$，每个 patch 经线性映射得到一个 $d$ 维 token，则 token 数：

$$
T=\left(\frac{I}{p}\right)^2
$$

随后添加标准 ViT 的正弦-余弦位置编码。


对于 256×256 图像，VAE latent 的 $I=32$：

| patch size $p$ | token 数 $T=(32/p)^2$ |
|---:|---:|
| 8 | 16 |
| 4 | 64 |
| 2 | 256 |

当 $p$ 减半时，token 数增加到 4 倍，因此 Transformer 计算量显著增加；但 patch size 的变化对参数量几乎没有影响。

这给论文提供了一个非常干净的实验变量：

> **在模型参数量近似不变时，仅通过增加 token 数增加前向计算量，观察生成质量是否改善。**

# 6. 条件信息：timestep 与 class label

DiT 是类别条件扩散模型。每次前向传播除了输入带噪 latent $z_t$，还输入：

- diffusion timestep $t$
- ImageNet class label $c$

二者先分别映射到与 hidden dimension 相同的向量，然后组合为条件表示。

论文比较了四种条件注入方式。

## 6.1 In-context conditioning

In-context conditioning 是四种条件注入方式中最接近“把条件也当作普通 token”的方法。

设 Patchify 后得到 $T$ 个 latent tokens：

$$
X=[x_1,x_2,\ldots,x_T],
\qquad
x_i\in\mathbb{R}^{d}
$$

timestep $t$ 和 class label $c$ 分别被嵌入为：

$$
e_t,e_c\in\mathbb{R}^{d}
$$

然后直接拼到 token 序列：

$$
Z=
[x_1,x_2,\ldots,x_T,e_t,e_c]
\in
\mathbb{R}^{(T+2)\times d}
$$

接下来不需要额外的条件模块，直接使用标准 Transformer Self-Attention：

$$
Q=ZW_Q,\qquad
K=ZW_K,\qquad
V=ZW_V
$$

$$
\operatorname{SelfAttn}(Z)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

因此 image token、timestep token、class token 都处于**同一个 self-attention 图**中。理论上同时存在：

| Query 来源 | Key/Value 来源 | 含义 |
|---|---|---|
| image token | image token | 图像 token 之间交互 |
| image token | condition token | 图像 token 读取条件 |
| condition token | image token | 条件 token 读取图像 |
| condition token | condition token | 条件之间交互 |

也就是说，条件 token 与 latent token 在结构上没有角色区分。论文对此的描述是：把 $t$、$c$ 当作两个额外 token，并“treat them no differently from the image tokens”。

最后一个 DiT block 后，把两个条件 token 删除，只保留原来的 $T$ 个 latent tokens 用于输出噪声和协方差预测。

### 6.1.1 直观理解

可以把 in-context conditioning 理解为：

```text
latent token 1 ─┐
latent token 2 ─┤
...             ├─ 拼成同一个序列 ─→ Self-Attention ─→ ...
latent token T ─┤
timestep token ─┤
class token ────┘
```

其优点是几乎不改变标准 ViT block，只增加两个 token，因此新增计算量很小。

缺点是：**模型没有显式区分“主体 token”和“条件 token”的角色**，条件信息需要通过同一套 self-attention 权重间接传播到 latent tokens。论文实验中该方法表现最差。

---

## 6.2 Cross-attention conditioning

Cross-attention conditioning 与 in-context conditioning 的本质区别是：

> **条件仍然被表示成 token，但条件 token 不再与 latent tokens 拼成同一个序列，而是作为独立的 Key/Value memory；latent tokens 作为 Query 主动读取条件。**

设 latent token 序列为：

$$
X\in\mathbb{R}^{T\times d}
$$

条件序列为：

$$
C=[e_t,e_c]
\in
\mathbb{R}^{2\times d}
$$

其中 $e_t$ 是 timestep embedding，$e_c$ 是 class embedding。

DiT 的 cross-attention block 先对 latent tokens 做普通 self-attention：

$$
X_1
=
X+
\operatorname{SelfAttn}
\left(
\operatorname{LN}(X)
\right)
$$

这一层只负责 latent token 之间的信息交换。

随后插入额外的 Multi-Head Cross-Attention（多头交叉注意力）：

$$
Q=X_1W_Q
$$

$$
K=CW_K,\qquad
V=CW_V
$$

然后：

$$
\operatorname{CrossAttn}(X_1,C)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

并通过残差连接更新 latent tokens：

$$
X_2
=
X_1+
\operatorname{CrossAttn}(X_1,C)
$$

随后再进入 MLP。

### 6.2.1 Cross-attention 中究竟“谁看谁”

这里最重要的是：

$$
Q\leftarrow \text{latent tokens}
$$

$$
K,V\leftarrow \text{condition tokens}
$$

所以是：

```text
latent token ── Query ─────┐
                           ├─ Cross-Attention → conditioned latent token
[timestep, class] ─ K,V ───┘
```

每个 latent token 都会计算自己对 timestep token 和 class token 的注意力权重。

如果某个 latent token $x_i$ 的 Query 为 $q_i$，两个条件 token 的 Key 分别为 $k_t,k_c$，那么该 token 对条件的注意力权重为：

$$
[a_{i,t},a_{i,c}]
=
\operatorname{softmax}
\left(
\frac{
[q_i^\top k_t,\ q_i^\top k_c]
}{
\sqrt{d_k}
}
\right)
$$

最终读取的条件信息是：

$$
r_i
=
a_{i,t}v_t
+
a_{i,c}v_c
$$

因此不同 latent token 可以学习不同的条件读取比例：

```text
token 1：更关注 timestep
token 2：更关注 class
token 3：两者接近
...
```

这里的交互是**非对称的**：

- latent token 可以读取 condition；
- condition token 在这个 cross-attention 层中不会反过来读取 latent tokens。

### 6.2.2 与 in-context conditioning 的本质区别

两者都把 timestep/class 表示成 $d$ 维 token，但 attention 拓扑不同。

| 维度 | In-context | Cross-attention |
|---|---|---|
| 条件是否表示为 token | 是 | 是 |
| 是否和 latent token 拼成一个序列 | 是 | 否 |
| latent-latent 交互 | 同一次 self-attention | 独立 self-attention |
| latent-condition 交互 | 同一次 self-attention | 独立 cross-attention |
| condition-latent 交互 | 有 | cross-attention 中没有 |
| Query | 联合序列所有 token | latent tokens |
| Key/Value | 联合序列所有 token | condition tokens |
| 条件角色 | 序列中的普通成员 | 外部 memory / information source |
| 结构修改 | 几乎不需要 | 需要额外 cross-attention layer |
| 论文新增计算量 | 很小 | 约 +15% Gflops |

因此可以把二者分别理解为：

```text
In-context：
“把条件请进同一个会议室，大家互相交流。”

Cross-attention：
“latent tokens 先内部交流，再主动去查阅一份独立条件资料。”
```

论文中条件只有两个 token：timestep 和 class label。因此 cross-attention 的表达能力很强，但相对这个简单条件也比较“重”，论文测得约增加 15% Gflops，而且性能仍不如 adaLN-Zero。

---

## 6.3 adaLN：Adaptive Layer Normalization

Adaptive Layer Normalization（adaLN，自适应层归一化）的思路与前两种完全不同：

> **它不让 latent token 去 attend 条件，而是让条件直接生成一组动态的 scale/shift 参数，改变每一层 token feature 的归一化后表示。**

可以把它理解为：

```text
条件
  ↓
可学习条件映射网络
  ↓
生成 scale / shift
  ↓
直接调制所有 token 的 LayerNorm 输出
```

### 6.3.1 什么是 hidden dimension / hidden feature

设 Transformer 中共有 $T$ 个 token，每个 token 的 embedding dimension 为 $d$：

$$
X\in\mathbb{R}^{T\times d}
$$

例如取一个很小的演示：

$$
T=3,\qquad d=4
$$

那么：

$$
X=
\begin{bmatrix}
x_{11} & x_{12} & x_{13} & x_{14}\\
x_{21} & x_{22} & x_{23} & x_{24}\\
x_{31} & x_{32} & x_{33} & x_{34}
\end{bmatrix}
$$

每一行对应一个 token：

$$
x_i\in\mathbb{R}^{4}
$$

每一列对应 token embedding 的一个 hidden feature dimension：

```text
              hidden dim 1   dim 2   dim 3   dim 4
token 1             x11       x12     x13     x14
token 2             x21       x22     x23     x24
token 3             x31       x32     x33     x34
```

这里的 hidden dimension 不是 RGB channel，也不是 VAE latent 的 4 个输入 channel；它是 Patchify 后线性投影得到的 Transformer 内部特征维度。

例如 DiT-XL 的 hidden size 为：

$$
d=1152
$$

所以每一个 latent patch token 实际是一个 1152 维向量。每一维都是网络学习出来的 latent feature，并没有预先规定“第 1 维代表边缘、第 2 维代表颜色”之类的明确语义。

### 6.3.2 LayerNorm 先对每一个 token 独立归一化

Transformer 的 LayerNorm 是对**每个 token 自己的 $d$ 个 hidden features**做归一化。

对于第 $i$ 个 token：

$$
x_i=[x_{i1},x_{i2},\ldots,x_{id}]
$$

计算：

$$
\mu_i
=
\frac{1}{d}
\sum_{j=1}^{d}x_{ij}
$$

$$
\sigma_i^2
=
\frac{1}{d}
\sum_{j=1}^{d}
(x_{ij}-\mu_i)^2
$$

得到：

$$
\tilde x_{ij}
=
\frac{x_{ij}-\mu_i}
{\sqrt{\sigma_i^2+\epsilon}}
$$

因此：

$$
\tilde X=\operatorname{LN}(X)
\in\mathbb{R}^{T\times d}
$$

不同 token 的均值和方差分别计算，不会把整个 token 序列混在一起做归一化。

### 6.3.3 条件如何产生 scale 和 shift

DiT 将 timestep embedding 与 class embedding 相加：

$$
c=t_{\mathrm{emb}}+y_{\mathrm{emb}}
$$

然后通过可学习条件映射网络：

$$
[\beta(c),\gamma(c)]
=
f_{\mathrm{cond}}(c)
$$

得到：

$$
\beta(c),\gamma(c)
\in\mathbb{R}^{d}
$$

这里：

- $\beta$：shift 向量；
- $\gamma$：scale 向量。

注意：它们不是两个标量，而是**两个长度均为 $d$ 的向量**。

例如 $d=4$ 时：

$$
\gamma=
[0.5,-0.2,1.0,0.1]
$$

$$
\beta=
[1.0,0.0,-0.5,2.0]
$$

adaLN 的调制形式为：

$$
\operatorname{adaLN}(X;c)
=
\operatorname{LN}(X)
\odot
(1+\gamma)
+
\beta
$$

### 6.3.4 为什么“所有 token 共享一组 $\beta,\gamma$”，同时“每个 hidden feature 又有自己的 scale/shift”

这两个说法不矛盾，因为它们描述的是两个不同维度。

设：

$$
\gamma=
[\gamma_1,\gamma_2,\gamma_3,\gamma_4]
$$

$$
\beta=
[\beta_1,\beta_2,\beta_3,\beta_4]
$$

它们沿 token 维进行 broadcast：

$$
1+\gamma
\rightarrow
\begin{bmatrix}
1+\gamma_1 & 1+\gamma_2 & 1+\gamma_3 & 1+\gamma_4\\
1+\gamma_1 & 1+\gamma_2 & 1+\gamma_3 & 1+\gamma_4\\
1+\gamma_1 & 1+\gamma_2 & 1+\gamma_3 & 1+\gamma_4
\end{bmatrix}
$$

$$
\beta
\rightarrow
\begin{bmatrix}
\beta_1 & \beta_2 & \beta_3 & \beta_4\\
\beta_1 & \beta_2 & \beta_3 & \beta_4\\
\beta_1 & \beta_2 & \beta_3 & \beta_4
\end{bmatrix}
$$

因此对任意 token $i$、hidden feature $j$：

$$
x'_{ij}
=
(1+\gamma_j)\tilde x_{ij}
+
\beta_j
$$

可以总结为：

$$
\boxed{
\text{token 维共享，hidden-feature 维独立}
}
$$

也就是说：

- token 1、token 2、token 3 都用同一个 $\gamma_1$ 调节各自的第 1 hidden feature；
- 都用同一个 $\gamma_2$ 调节各自的第 2 hidden feature；
- 以此类推。

但因为不同 token 原本的 $\tilde x_{ij}$ 不同，所以接受同一套 affine modulation 后，它们仍然会得到不同的表示。

### 6.3.5 三个 token 的完整数值演示

假设 LayerNorm 后有 3 个 token：

$$
\tilde X=
\begin{bmatrix}
0.2 & -1.0 & 0.5 & 1.5\\
-0.4 & 0.8 & -1.2 & 0.3\\
1.1 & -0.2 & 0.0 & -0.9
\end{bmatrix}
$$

条件网络根据当前 timestep 和 class 生成：

$$
\gamma=
[1.0,-0.5,0.0,2.0]
$$

$$
\beta=
[0.1,1.0,-0.2,0.0]
$$

于是：

$$
1+\gamma
=
[2.0,0.5,1.0,3.0]
$$

对于 token 1：

$$
\tilde x_1
=
[0.2,-1.0,0.5,1.5]
$$

先 scale：

$$
[0.2,-1.0,0.5,1.5]
\odot
[2.0,0.5,1.0,3.0]
=
[0.4,-0.5,0.5,4.5]
$$

再 shift：

$$
x'_1
=
[0.5,0.5,0.3,4.5]
$$

对于 token 2：

$$
[-0.4,0.8,-1.2,0.3]
\odot
[2.0,0.5,1.0,3.0]
=
[-0.8,0.4,-1.2,0.9]
$$

加上 $\beta$：

$$
x'_2
=
[-0.7,1.4,-1.4,0.9]
$$

对于 token 3：

$$
[1.1,-0.2,0.0,-0.9]
\odot
[2.0,0.5,1.0,3.0]
=
[2.2,-0.1,0.0,-2.7]
$$

再加 shift：

$$
x'_3
=
[2.3,0.9,-0.2,-2.7]
$$

最终：

$$
X'
=
\begin{bmatrix}
0.5 & 0.5 & 0.3 & 4.5\\
-0.7 & 1.4 & -1.4 & 0.9\\
2.3 & 0.9 & -0.2 & -2.7
\end{bmatrix}
$$

这个例子清楚说明：

1. 三个 token 使用**完全同一组** $\gamma,\beta$；
2. 第 1 hidden feature 始终使用 scale $2.0$ 和 shift $0.1$；
3. 第 2 hidden feature 始终使用 scale $0.5$ 和 shift $1.0$；
4. 但由于每个 token 的原始 feature 值不同，调制后仍然保持不同 token 表示。

因此 adaLN 并不是“把所有 token 变得一样”，而是：

> **根据条件，对整段 token 序列施加同一套 feature-wise 变换规则。**

### 6.3.6 不同条件会产生不同的 feature-wise 变换

假设同一 noisy latent 在两个不同 timestep 下：

```text
较早时间步：噪声很强
较晚时间步：latent 已经比较干净
```

条件向量不同：

$$
c_{\mathrm{early}}
\neq
c_{\mathrm{late}}
$$

于是条件网络产生不同的：

$$
\gamma_{\mathrm{early}},
\beta_{\mathrm{early}}
$$

和：

$$
\gamma_{\mathrm{late}},
\beta_{\mathrm{late}}
$$

这意味着 Transformer 即使面对相似 token feature，也会因为当前去噪阶段不同而改变内部 feature 的放大、压缩和偏移方式。

class label 同理：

```text
dog class
   ↓
一组 γ, β

airplane class
   ↓
另一组 γ, β
```

因此 timestep 和 class 不需要直接成为 latent token 的内容，而是可以直接控制 Transformer 每一层“如何解释和处理”这些 token features。

### 6.3.7 adaLN 与 Cross-Attention 的 conditioning 哲学不同

两者不是简单的“不同实现方式”，而是两类非常不同的信息注入机制。

Cross-attention：

$$
\text{latent token}
\rightarrow
\text{主动读取 condition token}
$$

条件改变的是**token 间的信息读取内容**。

adaLN：

$$
\text{condition}
\rightarrow
(\gamma,\beta)
\rightarrow
\text{改变每层 feature transformation}
$$

条件改变的是**网络内部 feature 的变换规则**。

因此可以概括：

```text
Cross-Attention：
“我现在需要从条件里读取什么信息？”

adaLN：
“在当前条件下，我应该怎样缩放/平移内部特征？”
```

论文实验中，adaLN 的额外 Gflops 最低，而且它是比较方案中唯一“对所有 token 应用同一条件调制函数”的机制。

### 6.3.8 DiT 官方实现中的具体形式

> [!note] 来源区分
> 以下 `SiLU + Linear`、`elementwise_affine=False` 和六路参数输出属于作者官方 PyTorch 实现细节；论文正文主要给出结构层面的 adaLN / adaLN-Zero 描述。

论文层面可以概括成：

$$
\text{condition}
\rightarrow
\text{scale / shift}
$$

作者官方 PyTorch 实现中，DiT block 使用：

```text
SiLU
  ↓
Linear(d, 6d)
```

作为条件调制器。

对于 adaLN-Zero，一次输出六组 $d$ 维向量：

$$
\left[
\beta_{\mathrm{msa}},
\gamma_{\mathrm{msa}},
\alpha_{\mathrm{msa}},
\beta_{\mathrm{mlp}},
\gamma_{\mathrm{mlp}},
\alpha_{\mathrm{mlp}}
\right]
=
f_{\mathrm{cond}}(c)
$$

其中：

- $\beta_{\mathrm{msa}},\gamma_{\mathrm{msa}}$：调制 Multi-Head Self-Attention 前的 LayerNorm；
- $\alpha_{\mathrm{msa}}$：门控 Attention residual；
- $\beta_{\mathrm{mlp}},\gamma_{\mathrm{mlp}}$：调制 MLP 前的 LayerNorm；
- $\alpha_{\mathrm{mlp}}$：门控 MLP residual。

官方实现的 LayerNorm 使用：

```text
elementwise_affine=False
```

也就是关闭 LayerNorm 自身固定的逐维 affine 参数，让条件网络产生的 scale/shift 成为主要调制来源。

因此：

```text
vanilla adaLN
= condition-to-affine mapping

adaLN-Zero
= condition-to-affine mapping
+ condition-to-residual-gate mapping
```

# 7. 核心创新：adaLN-Zero

adaLN-Zero 在 adaLN 基础上进一步加入残差分支的逐维门控参数 $\alpha$。

概念上，一个 block 可以写成：

$$
h
=
x+
\alpha_1(c)\odot
\operatorname{MSA}
\left(
\operatorname{adaLN}_1(x;c)
\right)
$$

$$
x'
=
h+
\alpha_2(c)\odot
\operatorname{MLP}
\left(
\operatorname{adaLN}_2(h;c)
\right)
$$

其中：

- MSA = Multi-Head Self-Attention（多头自注意力）
- MLP = Multi-Layer Perceptron（多层感知机 / pointwise feed-forward）
- $\gamma_1,\beta_1,\alpha_1$ 控制注意力支路
- $\gamma_2,\beta_2,\alpha_2$ 控制前馈支路

作者把生成 $\alpha$ 的输出初始化为 0，使训练开始时：

$$
x' \approx x
$$

即整个 DiT block 初始接近恒等映射。这借鉴了 ResNet 和 diffusion U-Net 中“将残差分支末端初始化为 0”的稳定训练策略。

## 7.1 官方代码实现补充

> [!note] 来源区分
> 本小节的结构思想来自论文；`SiLU + Linear`、`elementwise_affine=False`、六路参数输出与具体 zero initialization 细节来自作者官方 PyTorch 实现。

论文正文只给出结构层面的描述。联网核查作者官方实现后，具体形式为：

$$
c=t_{\mathrm{emb}}+y_{\mathrm{emb}}
$$

一个 `SiLU + Linear(d, 6d)` 从条件向量产生六组 $d$ 维参数：

$$
(\beta_{\text{msa}},\gamma_{\text{msa}},\alpha_{\text{msa}},
\beta_{\text{mlp}},\gamma_{\text{mlp}},\alpha_{\text{mlp}})
$$

官方实现的调制函数是：

$$
\operatorname{modulate}(x,\beta,\gamma)
=
x\odot(1+\gamma)+\beta
$$

因此 block 的实际结构可写为：

$$
x
\leftarrow
x+
\alpha_{\text{msa}}
\odot
\operatorname{MSA}
\left(
\operatorname{LN}(x)\odot(1+\gamma_{\text{msa}})
+\beta_{\text{msa}}
\right)
$$

$$
x
\leftarrow
x+
\alpha_{\text{mlp}}
\odot
\operatorname{MLP}
\left(
\operatorname{LN}(x)\odot(1+\gamma_{\text{mlp}})
+\beta_{\text{mlp}}
\right)
$$

官方代码把每个 DiT block 的 `adaLN_modulation` 最后一层 Linear 的权重和 bias 都初始化为 0，因此训练开始时六组调制向量均为 0。此时两条 residual gate 为 0，残差更新为 0，所以整个 block 初始表现为恒等映射；同时 scale/shift 也从零调制开始。

> [!note] 概念来源
> adaLN 的“条件生成逐通道 affine 参数”与 FiLM（Feature-wise Linear Modulation，特征维线性调制）的思想直接相通。FiLM 的核心就是由条件输入产生 feature-wise scale 与 shift；DiT 则把这种条件仿射调制放在 Transformer 的 LayerNorm 之后，并额外加入 residual gate。

# 8. Transformer 主干尺寸

作者沿用 ViT 的 scaling 方式，同时增加深度 $N$、隐藏维度 $d$ 与 attention head 数。

| 模型 | Block 数 $N$ | hidden size $d$ | heads | Gflops（$I=32,p=4$） |
|---|---:|---:|---:|---:|
| DiT-S | 12 | 384 | 6 | 1.4 |
| DiT-B | 12 | 768 | 12 | 5.6 |
| DiT-L | 24 | 1024 | 16 | 19.7 |
| DiT-XL | 28 | 1152 | 16 | 29.1 |

完整设计空间为：

$$
\text{model size}\in\{S,B,L,XL\}
$$

$$
p\in\{8,4,2\}
$$

从而形成 **12 个规模不同的 DiT 模型**。

命名如：

- DiT-XL/2：XLarge 主干，latent patch size = 2
- DiT-B/4：Base 主干，latent patch size = 4

# 9. 输出层 / “Transformer decoder”

论文把 DiT block 后的输出层称为 Transformer decoder，但它**不是 NLP Transformer 中由 masked self-attention + cross-attention 构成的 decoder stack**。

这里实际只是：

1. 最终 LayerNorm（如果使用 adaLN，则仍受条件调制）；
2. 对每个 token 做线性映射；
3. 每个 token 输出：

$$
p\times p\times 2C
$$

其中 $2C$ 对应：

- $C$ 个噪声预测通道；
- $C$ 个协方差预测通道。

4. 最后执行 unpatchify / reshape，恢复空间布局。

因此若输入 latent 是 $32\times32\times4$，最终得到：

- $\hat\epsilon:32\times32\times4$
- $\hat\Sigma:32\times32\times4$

# 10. Classifier-Free Guidance（CFG）

类别条件模型在训练时随机丢弃 class condition，并用一个可学习的 null embedding 代替，使同一模型同时学会：

- conditional prediction：$\epsilon_\theta(x_t,c)$
- unconditional prediction：$\epsilon_\theta(x_t,\varnothing)$

采样时组合二者：

$$
\hat{\epsilon}_\theta(x_t,c)
=
\epsilon_\theta(x_t,\varnothing)
+
s
\left[
\epsilon_\theta(x_t,c)
-
\epsilon_\theta(x_t,\varnothing)
\right]
$$

其中 $s>1$ 是 guidance scale。

直观上：

- $s$ 较大：更强地朝条件类别移动，通常提高样本保真度/精确率；
- 但可能牺牲多样性/召回率。

本文的结果也清楚显示这一 trade-off。

# 11. 实验设置

## 11.1 数据与任务

- 数据集：ImageNet
- 任务：class-conditional image generation
- 分辨率：
  - 256×256
  - 512×512

## 11.2 VAE 与 latent

使用 Stable Diffusion 的预训练 VAE：

- 下采样倍率：8
- 256×256 RGB → $32\times32\times4$
- 512×512 RGB → $64\times64\times4$

VAE 在 DiT 训练过程中冻结。

## 11.3 优化器与训练超参数

所有模型统一使用：

- AdamW
- learning rate：$1\times10^{-4}$
- constant learning rate
- weight decay：0
- batch size：256
- data augmentation：仅 horizontal flip
- 不使用 learning-rate warmup
- 不使用额外 regularization
- Exponential Moving Average（EMA，指数滑动平均）衰减：0.9999
- 所有报告结果使用 EMA 权重

作者强调：不同尺寸、不同 patch size 的模型共用几乎完全相同的训练超参数，没有针对各模型单独调参。

## 11.4 扩散超参数

沿用 ADM：

- $t_{\max}=1000$
- linear variance schedule
- variance 从 $1\times10^{-4}$ 到 $2\times10^{-2}$
- 使用 ADM 的 covariance parameterization
- timestep 与 label embedding 方法也沿用 ADM

## 11.5 评价指标

主要指标：

- FID-50K：Fréchet Inception Distance，50K 样本
- 250 DDPM sampling steps

辅助指标：

- IS：Inception Score
- sFID：spatial FID
- Precision
- Recall

除明确标注 CFG 的结果外，scaling 分析中的 FID 默认**不使用 classifier-free guidance**。

## 11.6 计算平台

- JAX
- TPU-v3 pods
- DiT-XL/2 在 TPU v3-256、global batch size 256 下约 **5.7 iterations/s**

# 12. 实验一：条件注入方式消融

作者固定最强主干 DiT-XL/2，比较四种 block：

| 条件注入 | 约 Gflops | 主要特点 |
|---|---:|---|
| In-context | 119.4 | 把 $t,c$ 作为 token |
| Cross-attention | 137.6 | 额外 cross-attention |
| adaLN | 118.6 | 条件生成 LayerNorm scale/shift |
| adaLN-Zero | 118.6 | adaLN + residual gate + zero initialization |

论文图5显示：

- adaLN-Zero 在训练全过程中 FID 最低；
- 400K step 时，adaLN-Zero 的 FID 几乎只有 in-context 的一半；
- cross-attention 计算量最高，却没有获得最好性能；
- vanilla adaLN 也明显弱于 adaLN-Zero，说明 **初始化策略本身就是重要因素**。

因此后续全部实验统一采用 adaLN-Zero。

# 13. 实验二：扩大模型 vs 减小 patch size

作者训练 12 个模型：

$$
\{S,B,L,XL\}\times\{p=8,4,2\}
$$

并从两个方向增加计算量。

## 13.1 固定 patch size，扩大 Transformer

增加：

- depth
- hidden width
- attention heads

结果：所有 patch size 下，模型越大，FID 越低。

## 13.2 固定模型尺寸，减小 patch size

例如固定 XL，只把 $p$ 从 8 → 4 → 2。

此时：

- 参数量基本不变；
- token 数大幅增加；
- Gflops 大幅增加；
- FID 明显下降。

这组实验是论文最重要的证据之一，因为它说明：

> **生成质量并不是单纯由参数量决定，而与模型实际执行的计算量密切相关。**

# 14. Gflops 与 FID 的 scaling 关系

论文图8把 12 个模型在 400K step 时的 FID-50K 对 forward Gflops 作图，得到相关系数：

$$
\rho\approx -0.93
$$

即 Gflops 越高，FID 越低。

尤其值得注意：

- 不同模型配置只要 Gflops 相似，FID 也往往接近；
- 例如 DiT-S/2 与 DiT-B/4 虽参数配置不同，但计算量相近，因此性能也相近。

因此作者认为 **forward-pass compute 是比 parameter count 更有效的 DiT complexity 指标**。

# 15. 大模型是否更“计算有效”

作者进一步估计总训练计算量：

$$
\text{training compute}
\approx
\text{model Gflops}
\times
\text{batch size}
\times
\text{training steps}
\times 3
$$

其中乘 3 是近似认为 backward pass 约为 forward pass 的两倍。

论文图9表明：

- 小模型训练更久，起初可以继续改善；
- 但到一定训练计算量后，会比“大模型 + 更少训练步数”更低效；
- XL/2 最终会超过 XL/4，即使控制累计训练 Gflops。

因此 scaling 不只是“多训练几步”，而是 **增加每次前向传播的模型计算能力**。

# 16. 256×256 ImageNet 最终结果

作者将 DiT-XL/2 继续训练至 **7M steps**。

| 模型 | FID ↓ | sFID ↓ | IS ↑ | Precision ↑ | Recall ↑ |
|---|---:|---:|---:|---:|---:|
| DiT-XL/2 | 9.62 | 6.85 | 121.50 | 0.67 | **0.67** |
| DiT-XL/2-G, cfg=1.25 | 3.22 | 5.28 | 201.77 | 0.76 | 0.62 |
| **DiT-XL/2-G, cfg=1.50** | **2.27** | **4.60** | **278.24** | **0.83** | 0.57 |

此前 LDM-4-G 的最好 FID 为 3.60；DiT-XL/2-G 将其降至 2.27。

同时可观察 CFG 的典型 trade-off：

- cfg 增大：FID、IS、Precision 改善；
- Recall 从 0.67 → 0.62 → 0.57，说明多样性有所下降。

论文还指出，即便只训练 **2.35M steps**（更接近 ADM 的训练规模），XL/2 仍可达到 FID 2.55，超过此前扩散模型。

# 17. 512×512 ImageNet

512×512 图像经 VAE 后：

$$
64\times64\times4
$$

DiT-XL/2 的 patch size 为 2，因此：

$$
T=(64/2)^2=1024
$$

即每次前向传播处理 **1024 个 latent tokens**。

模型 forward compute：

$$
524.6\ \text{Gflops}
$$

训练 3M iterations，超参数与 256 模型保持一致。

| 模型 | FID ↓ | sFID ↓ | IS ↑ | Precision ↑ | Recall ↑ |
|---|---:|---:|---:|---:|---:|
| DiT-XL/2 | 12.03 | 7.12 | 105.25 | 0.75 | **0.64** |
| DiT-XL/2-G, cfg=1.25 | 4.64 | 5.77 | 174.77 | 0.81 | 0.57 |
| **DiT-XL/2-G, cfg=1.50** | **3.04** | **5.02** | **240.82** | **0.84** | 0.54 |

论文对比的 U-Net 模型计算量：

- ADM：1983 Gflops
- ADM-U：2813 Gflops
- DiT-XL/2：524.6 Gflops

因此作者认为 DiT 在高分辨率下仍具有较好的 compute efficiency。

# 18. 模型计算 vs 采样计算

扩散模型在推理时还可以通过增加 sampling steps 增加 test-time compute。作者的问题是：

> 如果模型本身很小，能否靠“多采样几步”追上大模型？

他们对 12 个 DiT 使用：

$$
[16,32,64,128,256,1000]
$$

个 sampling steps。

代表性例子：

- DiT-L/2，1000 steps：
  - 约 80.7 Tflops / image
  - FID-10K = 25.9
- DiT-XL/2，128 steps：
  - 约 15.2 Tflops / image
  - FID-10K = 23.7

XL/2 用约 **1/5 的采样计算量**，仍获得更好的 FID。

因此：

> **增加采样计算无法弥补主干模型计算不足。**

# 19. 关键贡献

## 19.1 架构贡献

证明标准 Transformer 可以直接替换 diffusion U-Net，形成 Diffusion Transformer（DiT）。

## 19.2 条件注入贡献

系统比较：

- in-context token
- cross-attention
- adaLN
- adaLN-Zero

并证明 adaLN-Zero 是质量/计算效率最优方案。

## 19.3 Scaling 贡献

用两种互补手段增加计算量：

- 增加模型深度和宽度；
- 减小 patch size、增加 token 数；

二者都稳定提高生成质量。

## 19.4 评价复杂度的方法论贡献

指出图像生成架构中 parameter count 不能充分反映计算复杂度，建议使用 Gflops 分析 scaling。

## 19.5 性能贡献

DiT-XL/2 在当时的 class-conditional ImageNet 256/512 生成任务上达到扩散模型 SOTA，并在 512×512 上也表现出较高计算效率。

# 20. 局限

论文自身的结论边界需要注意：

1. **实验主要是 class-conditional ImageNet。**  
   并没有在正文中真正验证 text-to-image；作者只在结论中提出未来可把 DiT 用作 DALL·E 2、Stable Diffusion 等文本条件模型的主干。

2. **仍依赖卷积 VAE。**  
   因此论文证明的是“扩散去噪主干不需要 U-Net”，不是证明整个图像生成系统都不需要卷积。

3. **scaling 结论是经验规律。**  
   Gflops 与 FID 的强相关来自作者测试的 DiT 设计空间，并非理论上保证所有架构增加计算都会改善生成质量。

4. **更高 Gflops 本身意味着更高训练/推理成本。**  
   论文的核心观点是更大的 DiT 更有效，而不是计算成本消失。

5. **条件形式较简单。**  
   类别标签可以压缩成单个向量，因此 adaLN 很自然；对于长文本、空间条件、多模态 token，单个全局调制向量是否充分，本文没有回答。

# 21. 关键概念补充（联网核查）

## 21.1 adaLN 与 FiLM 的关系

FiLM（Feature-wise Linear Modulation）提出了一般性的条件调制：

$$
\operatorname{FiLM}(F|c)
=
\gamma(c)\odot F+\beta(c)
$$

即条件 $c$ 不一定需要和主特征做 attention，而可以直接产生每个 feature channel 的 affine 参数。

DiT 的 adaLN 可以理解为：

```text
条件 c
  ↓
生成 γ, β
  ↓
LayerNorm(x)
  ↓ feature-wise scale + shift
  ↓
Attention / MLP
```

adaLN-Zero 又进一步加入 residual gate $\alpha$。

联网核查来源：Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*, AAAI 2018, DOI 10.1609/aaai.v32i1.11671；以及 DiT 作者官方 `models.py`。

## 21.2 为什么 zero initialization 有意义

若一个深层残差网络在初始化时每层都进行较大的随机变换，深层堆叠可能让优化早期更加困难。

adaLN-Zero 让：

$$
\alpha_{\text{msa}}=\alpha_{\text{mlp}}=0
$$

使初始 block 近似：

$$
f(x)=x
$$

因此深层网络从“恒等映射 + 逐渐学会残差修正”的状态开始。

论文把这一思路与：

- ResNet 的 residual branch zero initialization
- diffusion U-Net 中 residual block 的末端卷积 zero initialization

联系起来。

## 21.3 CFG 的本质

Classifier-Free Guidance 不需要额外分类器。它让一个模型同时学 conditional 与 unconditional score，然后在采样时沿：

$$
\epsilon_\theta(x_t,c)
-
\epsilon_\theta(x_t,\varnothing)
$$

这个“条件方向”外推。

联网核查来源：Ho & Salimans, *Classifier-Free Diffusion Guidance*。

# 22. 相关工作脉络

## 22.1 Vision Transformer：Transformer 如何处理图像

Dosovitskiy et al. 的 ViT 把图像切成 patch，再把 patch 当作 token 输入标准 Transformer。

DiT 直接继承这条路线：

```text
ViT:
image → patch tokens → Transformer → recognition

DiT:
noised latent → patch tokens → Transformer → noise / covariance
```

DiT 的关键变化是把 ViT 从判别任务骨干改造成 diffusion denoiser。

## 22.2 Latent Diffusion Models：为什么不在像素空间运行

Rombach et al. 的 LDM 先用 autoencoder 将图像压缩到低分辨率 latent，再在 latent 中执行 diffusion。

DiT 没有改变这套两阶段思想，只替换：

```text
LDM:
latent → U-Net denoiser

DiT:
latent → Transformer denoiser
```

因此 DiT 可以视为 **LDM 框架 + ViT 风格 backbone** 的组合。

## 22.3 ADM / Improved DDPM：扩散训练 recipe 的来源

DiT 并没有重新设计 diffusion schedule、variance parameterization 或评价流程，而是大量沿用 Nichol & Dhariwal / Dhariwal & Nichol 的扩散模型训练 recipe：

- 预测噪声；
- 学习 covariance；
- 1000-step linear variance schedule；
- timestep / class embedding 方法；
- FID 评价工具链。

这使论文可以把实验重点集中到 backbone architecture，而不是扩散目标本身。

# 23. 论文内部逻辑关系

整篇论文可以压缩为下面这条因果链：

```text
问题：
U-Net 是否是 diffusion 的必要 backbone？
        ↓
选择 LDM 作为低计算量实验平台
        ↓
用 ViT 风格 patch Transformer 替换 U-Net
        ↓
需要解决条件 t / class 如何进入 Transformer
        ↓
比较 4 种 conditioning
        ↓
adaLN-Zero 最好
        ↓
建立 4 种模型尺寸 × 3 种 patch size
        ↓
分别通过“增大网络”和“增加 token”提升 Gflops
        ↓
两条 scaling 路线都降低 FID
        ↓
Gflops 与 FID 强负相关
        ↓
继续放大 DiT-XL/2
        ↓
ImageNet 256/512 达到当时扩散模型 SOTA
```

# 24. 最值得记住的实现细节

> [!summary]
> - 输入不是 RGB，而是冻结 VAE 产生的 latent。
> - 256×256 图像对应 $32\times32\times4$ latent。
> - Patchify 后 token 数 $T=(I/p)^2$。
> - timestep embedding 与 class embedding 相加形成条件向量。
> - 最优 DiT block 使用 adaLN-Zero。
> - 条件向量在每个 block 中生成两套 scale、shift、gate。
> - residual gate zero initialization 使 block 初始近似 identity。
> - 输出层把每个 token 映射为 $p\times p\times2C$，再 unpatchify。
> - $2C$ 分别对应 noise 与 learned covariance。
> - 模型设计空间为 S/B/L/XL × patch 8/4/2。
> - Gflops 比参数量更能解释性能 scaling。
> - 增大 model compute 比单纯增加 sampling steps 更有效。

## 跨文献关联导航（2026-10-15 整理）

> [!note] 文献关系标注
> “方法承接”或“研究延展”表示现有笔记可支持的脉络；“比较”“互补”“类比”仅用于研究组织，不声称作者直接引用或实验证实联合方案。

**总览**：[[Literature/文献关系总览|文献关系总览]]

**所属主题**：[[Literature/Topics/Transformer与物理场算子|Transformer与物理场算子]] · [[Literature/Topics/几何表示与性能条件生成|几何表示与性能条件生成]]

**重点关联**：

- **方法比较**：[[Literature/DiffAirfoil An Efficient Novel Airfoil Sampler Based on Latent Space Diffusion Model for Aerodynamic Shape Optimization|DiffAirfoil An Efficient Novel Airfoil Sampler Based on Latent Space Diffusion Model for Aerodynamic Shape Optimization]] — 二者均使用扩散，但一个用于潜空间翼型采样，一个为图像 DiT。
- **方法比较**：[[Literature/Predicting unsteady hydrodynamic performance of seaplanes based on diffusion models|Predicting unsteady hydrodynamic performance of seaplanes based on diffusion models]] — 扩散生成思想一致，去噪网络和输出对象分别为图像 latent 与水动力时间序列。
- **条件注入对照**：[[Literature/transformer/Unisolver|Unisolver]] — adaLN 条件调制与 PDE 域级/点级条件分解可对比。
- **条件机制参照**：[[Literature/transformer/Towards a Foundation-Model Paradigm for Aerodynamic Prediction in Three-dimensional Design|Towards a Foundation-Model Paradigm for Aerodynamic Prediction in Three-dimensional Design]] — AeroTransformer 的物理工况深层调制可与 DiT 方案对照。
- **架构正交**：[[Literature/transformer/Uformer - A General U-Shaped Transformer for Image Restoration|Uformer - A General U-Shaped Transformer for Image Restoration]] — 多尺度/局部注意力与条件调制是独立设计维度。
