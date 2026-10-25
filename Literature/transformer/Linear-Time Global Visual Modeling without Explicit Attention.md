---
title: "Linear-Time Global Visual Modeling without Explicit Attention"
authors:
  - Ruize He
  - Dongchen Han
  - Gao Huang
year: 2026
venue: "arXiv preprint (cs.CV), version 2, 2026-05-06"
doi: "10.48550/arXiv.2605.01711"
arxiv: "2605.01711v2"
url: "https://arxiv.org/abs/2605.01711"
code: "https://github.com/LeapLabTHU/WeightFormer"
source: "未提供本地 PDF；原文 https://arxiv.org/pdf/2605.01711"
status: read
updated: 2026-10-25
tags:
  - literature
  - transformer
  - dynamic-weights
  - hypernetwork
  - attention
  - MLP
  - vision
  - linear-complexity
  - scientific-machine-learning
---

> [!abstract] 摘要翻译
> Transformer 的全局序列建模通常被归因于显式计算注意力权重，这带来关于序列长度的二次复杂度。本文提出另一种数学视角：注意力可以重写成一个参数由输入动态预测的多层感知机（Multilayer Perceptron，MLP），全局信息由动态生成的参数间接编码。基于这一解释，作者探索无需显式注意力、仅通过输入条件化的权重实现线性复杂度全局建模的可能性；设计了动态线性层和动态深度卷积核的预测机制，并构造 WeightFormer。在图像分类、检测、分割和条件图像生成等视觉实验中取得有竞争力的结果和较好的高分辨率效率。

# 一句话结论

**Self-Attention 对每个输入序列都可等价地视为临时生成一个两层的“Softmax-MLP”；第一层权重为 \(K^\top\)，第二层权重为 \(V\)。** 这一结论是**同一注意力运算的代数重写**，并不意味着原始注意力已经实现线性复杂度，也不意味着整个 Transformer 只有一个 MLP。论文另行提出 WeightFormer，利用固定大小的全局统计预测卷积/线性层参数，才实现关于 token 数近似线性的计算。

> [!important] “一次性 MLP 权重”的准确含义
> 每次前向、针对**当前输入**生成的 \(K,V\) 是临时的有效权重；训练好的 \(W_K,W_V,W_Q\) 仍然是跨样本共享的持久参数。并非训练权重只用一次或真的在每次推理时更新模型参数。且 Softmax 耦合全部隐藏单元的分数，它与通常逐元素 ReLU/GELU 的 MLP 并不完全同构。

## 1. 问题、动机与研究定位

标准注意力会显式构造 \(N\times N\) 的 token 相似度矩阵，约需 \(O(N^2d)\) 运算与 \(O(N^2)\) 注意力权重存储（未计 FlashAttention 一类融合实现）。既有稀疏注意力、低秩近似与核化注意力都在尝试改变这一矩阵的计算形式。

本文换了一个问题：**全局信息混合一定要先生成每个 token 对之间的关系，还是可以将输入压缩为少量动态参数，再由这些参数驱动逐位置变换？** 其目标是验证动态参数化是否可以充当注意力的高效替代，而非宣称已经为每种任务设计了最强的视觉主干。

## 2. 核心推导：Attention = 输入生成权重的 Softmax-MLP

令 \(X\in\mathbb R^{N\times d}\)，并通过持久的投影矩阵产生：

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V .
\]

标准缩放点积注意力：

\[
O=\operatorname{Softmax}\left(\frac{QK^\top}{\sqrt d}\right)V.
\]

对第 \(i\) 个 query：

\[
o_i=\underbrace{\operatorname{Softmax}\left(
\underbrace{q_iK^\top/\sqrt d}_{\text{第一线性层}}
\right)}_{\text{非线性激活}}
\underbrace{V}_{\text{第二线性层}}.
\]

其中 \(K^\top\in\mathbb R^{d\times N}\)、\(V\in\mathbb R^{N\times d}\)。可以严格视为一个**隐藏层宽度等于输入 token 数 \(N\)**、对所有 query 共享本次动态权重的两层网络：

\[
W_{\mathrm{fast}}(X)=\{K^\top,V\},
\qquad
o_i=F_{W_{\mathrm{fast}}(X)}(q_i).
\]

这里的参数由上下文 \(X\) 产生，是“快速/临时权重”（fast or ephemeral weights）的解释，不应与普通训练参数混淆。换一个输入序列 \(X'\)，就得到另一组有效权重。

![论文中图2：显式注意力与动态 MLP 的等价重写](https://arxiv.org/html/2605.01711v2/dyn.png)

*论文中图 2：左侧 \(A=\sigma(QK^\top),O=AV\) 是显式加权聚合；右侧把 \(K^\top\) 和 \(V\) 当作由输入预测的临时 MLP 权重。此处插入 arXiv HTML 原始图，而非自行重绘。*

### 2.1 为什么这个重写本身不能节省计算？

因为“临时 MLP”的隐藏宽度是 \(N\)，所有 \(N\) 个 query 都要经过它。计算 \(QK^\top\) 仍需 \(O(N^2d)\)。这是表示形式的等价，不是推导出一个更快的精确注意力算法。

### 2.2 数学等价不等于对所有任务有效的机制解释

所谓“MLP”在本文中并非只含点级计算：其权重本身编码了**整段输入序列的信息**。因此各 token 仍通过这些动态权重产生非局部依赖。静态逐点 MLP 没有这条输入依赖全局统计的通路，不能据此断言“只用普通 MLP 就等价于 Transformer”。

## 3. 从代数解释到 WeightFormer：固定大小的动态权重

论文提出独立于 token 数 \(N\) 的**全局统计压缩**，再用它为普通线性层或深度卷积核生成动态更新：

\[
W(X)=W_0+\Delta W(X).
\]

其中 \(W_0\) 是训练学习的静态参数，\(\Delta W(X)\) 是当前输入决定的增量。

两种全局统计：

\[
\phi_{\mathrm{pool}}(X)=\operatorname{Pool}(X)
\in\mathbb R^{M\times d},
\qquad
\phi_{\mathrm{corr}}(X)=X^\top X
\in\mathbb R^{d\times d}.
\]

只要 \(M,d\) 与输入点数 \(N\) 无关，统计构造和动态参数应用对 \(N\) 呈线性增长。这是 WeightFormer 的效率来源，不是精确重现原始 \(K^\top,V\)。

### 3.1 动态 Linear：直接用全局统计改写通道变换

最简单的方法是全局平均池化（Global Average Pooling，GAP）后用小 MLP 预测参数：

\[
\Delta W=\operatorname{Reshape}(\operatorname{MLP}(\operatorname{GAP}(X))).
\]

但压成单个向量可能损失细节。作者进一步使用二阶统计 \(X^\top X\)，包括线性、非线性、深层和双侧激活（Bilateral Activation）几种预测形式。例如：

\[
\Delta W=W_1\sigma(W_2X^\top)\sigma(XW_3)W_4.
\]

这个式子先对转置的 token 维特征和原 token 特征分别做可学习投影、非线性，再组合为权重矩阵。默认先在空间维进行 2 倍下采样，减少二阶统计的计算开销。作者发现将 Bilateral Activation 只加到 MLP 的**第一线性层**时，精度与推理成本的折中较好。

### 3.2 动态 Depthwise Convolution：预测每个通道的空间核

对 \(X\in\mathbb R^{d\times H\times W}\)，使用动态深度卷积核：

\[
W(X)=W_0+\Delta W(X),\quad
\Delta W(X)\in\mathbb R^{d\times 3\times3}.
\]

论文对比 GAP、空间自适应池化（Adaptive Average Pooling，AAP）、幅度/方向解耦和卷积核预测网络等方案。最终采用 Spatially Adaptive Prediction：

\[
X'=\operatorname{AAP}(X,(3,3)),\qquad
\Delta W=\operatorname{MLP}(X').
\]

这条路径通过每幅图像的全局概况来决定局部卷积核，因此虽仍执行局部卷积，**结果已经能依赖远处的输入区域**。

### 3.3 WeightFormer 的完整 block

在原有卷积/MLP 式视觉主干中，每隔三个 block 插入一个动态 block；其他 block 保持静态。动态 block 包含：空间自适应预测的 depthwise convolution，以及第一线性层采用 Bilateral Activation 的 MLP；第二线性层不改。

![论文中图4：WeightFormer 架构、动态模块和伪代码](https://arxiv.org/html/2605.01711v2/wfm.png)

*论文中图 4：动态权重发生在 depthwise conv 和第一全连接层；静态层仍然保留。WeightFormer 是无需显式注意力的**替代模型**，不是和原 Transformer 精确同函数的等价实现。*
