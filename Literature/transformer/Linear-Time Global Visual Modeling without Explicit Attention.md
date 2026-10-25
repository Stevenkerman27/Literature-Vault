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


## 4. 实验设置：先验证哪些动态预测有效，再评估完整模型

### 4.1 预测策略消融

作者先在 ImageNet-1K 测试精简的卷积主干，逐一替换其 MLP 或 depthwise convolution。表中吞吐量为论文报告的单卡图像每秒数量，准确率为 Top-1：

| 方案 | 参数量 | FLOPs | 吞吐量 | Top-1 |
|---|---:|---:|---:|---:|
| 静态 CNN 基线 | 5.7M | 1.1G | 4172 | 73.3% |
| 动态 Linear 1，GAP | 10.1M | 1.1G | 3759 | 74.7% |
| 动态 Linear 1，线性相关性 + Pool | 6.8M | 1.2G | 3278 | 76.2% |
| 动态 Linear 1，Bilateral | 6.9M | 1.2G | 3213 | 76.4% |
| 动态 Linear 1+2，Bilateral | 8.2M | 1.3G | 2659 | 76.7% |
| 动态 DWC，Spatially Adaptive | 6.2M | 1.1G | 4051 | 74.6% |
| 动态 Linear 1+DWC，Bilateral + Adaptive | 7.4M | 1.2G | 3201 | **76.8%** |

这不是等参数的单因素实验：动态模块改变了参数量和吞吐量，不能把准确率差全部归因于某一种信息混合性质。它验证的是实际精度/计算量折中。

### 4.2 主要 ImageNet-1K 训练协议

- 数据：1.28M 训练图、50K 验证图、1,000 类。
- 从头训练 300 epochs；AdamW、weight decay 0.05、余弦学习率衰减。
- batch size 2048；20 epochs 线性 warm-up；初始学习率 \(4\times10^{-3}\)。
- RandAugment、Mixup、CutMix、Random Erasing；作者用 8 张 RTX 3090 训练。
- 完整模型的动态 block 按每三个 block 一个的频率稀疏插入。

## 5. 结果：精度、吞吐量、分割与生成

### 5.1 ImageNet-1K 分类（224×224，除特别注明）

| 模型 | 参数 | FLOPs | Top-1 |
|---|---:|---:|---:|
| DeiT-T | 6M | 1.2G | 72.2% |
| WeightFormer-T | 7M | 1.1G | **76.3%** |
| DeiT-S | 22M | 4.6G | 79.8% |
| WeightFormer-S | 27M | 4.4G | **81.3%** |
| ConvNeXt-S (iso.) | 22M | 4.3G | 79.7% |
| WeightFormer-B（448×448） | 27M | 17.7G | 83.4% |

注意最后一行在 **448×448** 输入上得到，不能与其他 224×224 模型当作同分辨率结果对比。WeightFormer-S 参数量也多于 DeiT-S，因此不应把精度差当作严格等参数优势。

### 5.2 高分辨率推理

在 1248×1248 输入（6084 个 token）时，论文报告相比 DeiT：

- 吞吐量为其 **7.7 倍**；
- 每图 GPU 显存使用减少约 **91%**。

这些是论文特定模型、实现、分辨率和 RTX 3090 上的对照结果，并非所有 Transformer、所有显卡的通用倍数。

### 5.3 COCO 检测 / 实例分割

在 COCO 2017、Cascade Mask R-CNN 对照中：

| 指标 | DeiT-T | WeightFormer-T |
|---|---:|---:|
| Box AP | 44.4 | **45.0** |
| Mask AP | 38.1 | **38.3** |
| 总 FLOPs | 594G | **566G** |
| Backbone FLOPs | 106G | **77G** |

结果说明动态权重方法在检测/分割的完整系统中带来的是相对有限的 AP 改进，而 backbone 计算成本下降更明显。

### 5.4 ADE20K 语义分割

使用 UperNet 框架，WeightFormer-T 的 mIoU 为 40.7（DeiT-T 为 39.2），WeightFormer-S 为 45.6（DeiT-S 为 44.0）。S 型 backbone FLOPs 从 DeiT-S 的 35G 降至 27G。

### 5.5 条件图像生成

论文还将动态参数主干用于 ImageNet-1K 类条件生成，对比 Diffusion Transformer（DiT）与 DiG；生成质量用 Fréchet Inception Distance（FID，越低越好）衡量。示例：

| 模型 | 参数 | FLOPs | FID ↓ |
|---|---:|---:|---:|
| DiT-S/2 | 33M | 6.1G | 68.40 |
| WeightFormer-S/2 | 38M | 5.0G | **61.39** |
| DiG-S/2 | 33M | 4.3G | 62.06 |
| DiT-B/2 | 130M | 23.0G | 43.47 |
| WeightFormer-B/2 | 149M | 20.0G | **38.21** |

比较中模型参数量并不完全相同。这些实验支持动态参数化适用于**视觉生成网络的去噪主干**，但不是对航空几何扩散或生成式物理求解的直接证明。

## 6. 动态层数量消融：为什么不是越动态越好？

| 动态 block 数 | Top-1 | 吞吐量（img/s） |
|---:|---:|---:|
| 4 | 75.4% | 3869 |
| 6 | 76.3% | 3515 |
| 8 | 76.2% | 3218 |
| 11 | **76.9%** | 2745 |
| 14 | 76.6% | 2460 |
| 17 | 70.2%* | 2323 |

其中 17 个动态 block 的结果是在训练发散前得到的最佳准确率。需要区分“准确率最高”与“综合效率最优”：作者选择 **6 个动态 block**（约每三个 block 插入一个）作为性能/成本的工程折中，而不是声称它是精度最高。

## 7. 局限与证据边界

1. **代数等价不等于高效实现。** 原始 attention 的“动态 MLP”隐藏宽度随 \(N\) 增大，仍有 \(O(N^2)\) 代价。WeightFormer 是另一种压缩统计后重新设计的架构，**不与原注意力在函数上严格等价**。
2. **Softmax 不是普通逐元素激活。** 它在当前序列各 key 方向归一化，直接耦合隐藏单元；把它叫 MLP 主要强调“两个线性变换夹一个非线性算子”的形式。
3. **所谓“输入一次性权重”不是训练权重更新。** \(K^\top,V\) 依赖本次输入，但 \(W_Q,W_K,W_V\) 依旧是共享训练参数。
4. **视觉任务证据不等于 PDE 证据。** 本文没有不规则非结构网格、变拓扑机翼、Navier–Stokes 求解或反向几何优化的直接实验。
5. **固定统计可能丢失关键信息。** GAP 只有一阶均值；\(X^\top X\) 是全局二阶统计，都可能对小尺度边界层、激波、细长翼端等局部区域不敏感。
6. **归纳偏置不明。** 作者承认动态参数的表达能力与归纳偏置尚未得到充分理论解释；训练稳定性也会受到动态权重预测路径影响。
7. **复杂度有条件。** 对 token 数 \(N\) 线性依赖并不等于对特征宽度 \(d\) 或动态核大小也线性；实验使用固定宽度和固定空间核。
8. **密度和积分权重问题。** 原始视觉输入有规则二维像素网格；直接把 \(X^\top X\) 搬到非均匀 CFD 网格会使密集采样区域贡献更大。面积/体积权重、边界特征和坐标不变性需要另行设计。

## 8. 对物理场与飞行器生成器的可检验启发（本文之外的研究推论）

一种有价值的思路是构造**输入条件化的逐点物理网络**：

\[
s(X)=\left[
\sum_i a_i\phi(x_i),\quad
\sum_i a_i\phi(x_i)\phi(x_i)^\top
\right],
\quad W_{\mathrm{dyn}}=H(s(X),c),
\quad
u_i=F(x_i,c;W_0+W_{\mathrm{dyn}}),
\]

其中 \(a_i\) 是面元面积/体单元体积或规范化积分权重，\(c\) 包含来流工况。这样可以使每个点的预测依赖整个物理域的统计量，又不显式形成 \(N\times N\) 注意力矩阵。

它是**建议实验的候选架构**，不是 WeightFormer 论文已经验证的 PDE 算子。还必须比较：高频压力梯度、激波位置、边界层/翼根细节、网格密度迁移、OOD 构型、升阻力矩积分误差、几何梯度和实际训练时间。特别是全局低阶统计可能不足以传播空间精细的非局部相互作用。

与近期 Transolver 机制审视工作的关系是**概念互补，不是直接继承**：WeightFormer 认为全局输入可压缩为动态网络参数；[[Literature/transformer/Transolver Is a Linear Transformer Revisiting Physics-Attention Through the Lens of Linear Attention|LinearNO]] 与 [[Literature/transformer/Does Transolver really need a Transformer|Does Transolver really need a Transformer?]] 强调 slice/deslice 全局混合与 token self-attention 的功能应分开分析。二者值得用相同物理数据集对比，但不能据现有视觉结果直接判断优劣。

## 9. 与 MiTA 的直接对照（2026-10-25 补充）

- [[Literature/transformer/Mixture-of-Top-k Attention Efficient Attention via Scalable Fast Weights|MiTA]]（arXiv 首版 2026-02-01，NeurIPS 2026）比 WeightFormer 更早直接用「两层 Fast-Weight MLP」解释全注意力，并且进一步构造五维分类法。其算法保留压缩共享专家 + Top-k KV 稀疏专家的显式注意力；WeightFormer 则从全局统计直接生成动态线性层、卷积核。两者共享代数视角，但**不是同一替代结构，更不能按不同论文的视觉指标横向判定通用优劣**。
- 关于“MLP 精确等价”的细节：MiTA 将 softmax 归一化吸收到第二层 \hat V(q) 中，因此这层有效权重也会依赖 query；若保留原 softmax，则它并非普通逐元素激活。该等价不意味着 attention 本身具备线性时间。

## 10. 核心相关研究脉络

- **HyperNetworks**（Ha et al., 2016）：通过一个网络生成另一网络的权重，是输入/任务条件化参数生成的重要先例。本文并非最先提出动态权重，而是给出 attention 的代数重写并系统测试其替代方案。https://arxiv.org/abs/1609.09106
- **Pay Attention to MLPs**（Liu et al., 2021，gMLP）：通过空间门控 MLP 进行序列交互，说明视觉/语言的全局混合不一定需要标准 self-attention。但 gMLP 的空间混合权重与本文动态生成的逐输入权重并不相同。https://arxiv.org/abs/2105.08050
- **Transolver Is a Linear Transformer**（Hu et al., AAAI 2026）：将物理切片注意力重写为特殊线性注意力，并提出 LinearNO。它使用点到 latent slice 的低秩交互；WeightFormer 使用固定维全局统计预测参数，因而是可对比的两种全局信息机制。[[Literature/transformer/Transolver Is a Linear Transformer Revisiting Physics-Attention Through the Lens of Linear Attention]]

## 参考资料

- 原文 v2：https://arxiv.org/html/2605.01711v2
- 原文 PDF：https://arxiv.org/pdf/2605.01711
- 官方代码：https://github.com/LeapLabTHU/WeightFormer
- 数据与全部精确数字见论文 Table 2、3、4、5、6、7，复杂度与基本推导见 Section 3–4。
- 图 2 与图 4 采用 arXiv HTML 原始图片远程引用：尚无本地 PDF 实体，因此未声称完成本地资产提取。


## 架构族导航

通用长序列机制请看 [[Literature/Topics/神经网络架构与高效序列建模|神经网络架构与高效序列建模]]，该页将本论文与 [[Literature/architectures/Transformers are SSMs Generalized Models and Efficient Algorithms Through Structured State Space Duality|Mamba-2 / SSD]]、[[Literature/architectures/Gated Linear Attention Transformers with Hardware-Efficient Training|GLA]] 和 [[Literature/transformer/Mixture-of-Top-k Attention Efficient Attention via Scalable Fast Weights|MiTA]] 区分为动态权重、状态压缩和稀疏路由三种不同机制。
