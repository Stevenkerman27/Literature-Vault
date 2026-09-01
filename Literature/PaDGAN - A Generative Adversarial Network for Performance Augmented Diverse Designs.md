---
title: "PaDGAN: A Generative Adversarial Network for Performance Augmented Diverse Designs"
authors:
  - Wei Chen
  - Faez Ahmed
year: 2020
venue: "IDETC/CIE 2020; journal version in Journal of Mechanical Design 143(3), 031703"
doi: "10.1115/1.4048626"
source: "/mnt/data/2002.11304v5.pdf"
tags:
  - generative-design
  - GAN
  - DPP
  - airfoil
  - design-space-exploration
---

> [!abstract] 摘要（原文翻译）
> 深度生成模型已被证明是自动设计综合与设计空间探索的有效工具，但用于工程设计时面临三个问题：生成设计缺乏多样性、难以显式提升整体性能/质量，以及通常只能在训练数据域内插值而难以产生训练域之外的新颖设计。本文提出一种基于**行列式点过程（Determinantal Point Process, DPP）**的新损失，同时对“多样性”和“质量”进行概率建模，并据此构建 **Performance Augmented Diverse Generative Adversarial Network（PaDGAN）**。在三个二维合成问题和一个真实翼型问题上，PaDGAN 能生成覆盖更广、性能更高的样本；与普通 GAN 相比，平均质量约提高 28%，同时缓解模式坍塌。更重要的是，模型会把生成分布的边界向训练数据之外的高质量区域扩展，说明它不仅能插值，还具有受性能梯度引导的定向外推能力。

## 一句话定位

PaDGAN 不是“给定目标性能生成指定设计”的条件生成器，而是把一个无条件生成器的总体分布**推向“高性能且彼此不同”的区域**：它回答的是“从整个设计空间里多给我一些好且不同的设计”。

## 1. 问题与动机

工程设计中的生成模型不能只追求“像训练数据”。设计者通常还关心：

1. **质量/性能（quality/performance）**：如翼型的 $C_L/C_D$；
2. **覆盖度/多样性（diversity/coverage）**：不能只反复生成一个局部最优附近的相似方案；
3. **新颖性（novelty）**：最好能探索训练数据尚未覆盖、但性能更优的区域。

普通 GAN 只学习 $p_{data}(x)$，没有显式性能目标，且可能发生 mode collapse（模式坍塌）。单独加入性能奖励又容易把所有样本压到同一个高性能峰值。作者因此希望构造一个**同一项损失里同时包含质量和多样性**的目标。

## 2. 核心方法

![[assets/PaDGAN - A Generative Adversarial Network for Performance Augmented Diverse Designs/论文中图1_PaDGAN架构.png]]
*论文中图1：PaDGAN 总体架构。生成样本同时送入判别器与性能估计器，DPP 核把性能和样本间差异结合成额外损失。*

### 2.1 从 DPP 的“质量 × 多样性”分解出发

对样本集合 $S$，DPP 的 $L$-ensemble 核可写为

$$
L_{ij}=q_i\,\phi(i)^T\phi(j)\,q_j,
$$

其中：

- $q_i\ge 0$：第 $i$ 个设计的质量；
- $\phi(i)^T\phi(j)$：样本 $i,j$ 的相似度；
- 对固定集合大小，$\det(L_S)$ 越大，代表该集合兼具更高质量和更低冗余。

可进一步理解为

$$
P_L(S)\propto \prod_{i\in S}q_i^2\;\det(K_S).
$$

第一项偏好高质量样本，第二项偏好互相不相似的样本。

![[assets/PaDGAN - A Generative Adversarial Network for Performance Augmented Diverse Designs/论文中图2_DPP几何直觉.png]]
*论文中图2：DPP 的几何直觉。向量长度由质量决定，夹角反映相似度；更长且不共线的向量张成更大的体积。*

### 2.2 Performance Augmented DPP Loss

对一个生成 batch $B$，作者直接由已知的样本相似度和质量构造 DPP 核：

$$
L_B(i,j)=k(x_i,x_j)\big(q(x_i)q(x_j)\big)^{\gamma_0}.
$$

实验中相似度采用 RBF（Radial Basis Function，径向基函数）核：

$$
k(x_i,x_j)=\exp\left(-0.5\|x_i-x_j\|^2\right).
$$

PaDGAN 的 DPP 损失为

$$
\mathcal L_{PaD}(G)=-\frac{1}{|B|}\log\det(L_B)
=-\frac{1}{|B|}\sum_i\log\lambda_i,
$$

$\lambda_i$ 是 $L_B$ 的特征值。训练目标为

$$
\min_G\max_D V(D,G)+\gamma_1\mathcal L_{PaD}(G).
$$

因此生成器既要骗过判别器以保持“像真实设计”，又要让 batch 的 DPP 行列式尽量大，从而向高质量且互相不同的样本移动。

### 2.3 两个关键超参数的含义

- $\gamma_0$：**DPP 核内部的 quality dial**。$\gamma_0=0$ 时质量项全部退化为 1，只鼓励多样性；$\gamma_0$ 很大时，高质量样本概率质量被放大，模型会更接近“只追求性能”。作者强调简单把 $q$ 乘常数不能改变质量/多样性的相对权重，因为那只是统一缩放体积；指数变换才改变质量分布的动态范围。
- $\gamma_1$：**DPP 损失相对 GAN 损失的总权重**。它决定额外目标对训练的干预强度。

两者不要混淆：$\gamma_0$ 调“DPP 内部质量 vs. 多样性”，$\gamma_1$ 调“DPP 目标 vs. GAN 真实性目标”。

### 2.4 性能估计器与可微梯度

DPP 损失要通过 $q(x)$ 反向传播到生成器，因此需要 $dq/dx$。作者给出三种可能来源：

1. 可微物理求解器（例如伴随方法）；
2. 数值差分；
3. 可微 surrogate model（代理模型），本文翼型实验采用神经网络代理。

这意味着 PaDGAN 的“向高性能方向外推”并非无目标地越过数据边界，而是由性能估计器提供的梯度方向驱动。

### 2.5 代理模型不可靠时的稳定化

训练早期 GAN 会生成不真实样本，而仅在真实数据上训练的性能代理对这些样本可能严重外推失真。作者提出两项启发式稳定化：

**(1) Realisticity-weighted quality**

$$
q(x)=D(x)q'(x),
$$

让判别器认为“不真实”的样本即使代理给出高性能，也不会得到过高质量权重。

**(2) 逐步增加 DPP loss 权重**

$$
\gamma_1=\gamma_1'\left(\frac{t}{T}\right)^p.
$$

训练早期先学真实性，后期再逐渐强化质量和多样性。

## 3. 实验设置

### 3.1 三个二维合成实验

统一采用高斯混合质量函数

$$
q(x)=\sum_{k=1}^{K}\exp\left(-\frac{(x-\mu_k)^T(x-\mu_k)}{2\sigma^2}\right).
$$

- **Example I**：训练数据均匀位于半径 0.25–0.5 的圆环内，质量函数有 6 个峰。目标是同时覆盖六个高质量峰。
- **Example II**：训练数据为 $3\times3$ 九个簇，质量峰只有 4 个。理想模型应主要生成四个高质量簇，同时不要塌到其中一个峰。
- **Example III**：与 Example I 类似，但训练数据被限制在半径 0.325–0.375 的更窄圆环，用于专门检验**向训练边界之外外推**的能力。

比较四个模型：

- GAN：普通 GAN；
- $GAN_D$：PaDGAN 但 $\gamma_0=0$，只优化多样性；
- $GAN_Q$：普通 GAN + 单独的平均质量奖励；
- PaDGAN：同时优化质量和多样性。

合成实验设置：RBF bandwidth = 1.0，PaDGAN $\gamma_0=2,\gamma_1=0.5$，$GAN_Q$ 的质量权重 $\gamma_2=10$。

![[assets/PaDGAN - A Generative Adversarial Network for Performance Augmented Diverse Designs/论文中图6_合成例3外推.png]]
*论文中图6：Example III。$GAN_Q$ 明显模式坍塌；PaDGAN 不仅跨出训练圆环，还主要沿高质量区域向外扩展。*

### 3.2 翼型实验

- 数据：UIUC 翼型数据库，原始约 1,600 个真实翼型；按 BézierGAN 相关预处理与增广得到 **38,802** 个翼型。
- 性能：$C_L/C_D$，用 XFOIL 计算。
- 性能代理：ResNet（Residual Neural Network，残差神经网络），用于提供可微性能估计。
- 生成器：BézierGAN，用可解释低维 Bézier 表示生成翼型。
- PaDGAN 参数：$\gamma_0=2,\gamma_1=0.2$。

### 3.3 评价指标

**多样性分数**：随机子集的相似度矩阵 log-determinant 的平均值；越大（越不负）越多样。

$$
s_{div}=\frac1n\sum_{i=1}^{n}\log\det(L_{S_i}).
$$

**质量分数**：生成样本平均性能

$$
s_{qa}=\frac1{|Y|}\sum_i q(x_i).
$$

合成实验另定义 overall score，同时奖励“落在高质量峰附近”和“在各峰之间均匀分布”。每种模型每个设置训练 10 次，统计随机性。

## 4. 主要结果

### 4.1 合成问题

- $GAN_D$ 多样性最高但质量最低：证明单独“摊开”不等于有效设计探索。
- $GAN_Q$ 质量最高但多样性最差，常塌到一个或少数峰：证明单独“追求性能”会损失方案族。
- PaDGAN 在三个例子中都获得最高 overall score，并且方差更小，说明“质量 × 多样性”的联合建模比两个独立目标更稳定。
- Example III 最关键：PaDGAN 能在训练域外生成样本，而且外推方向与质量梯度一致，不是随机越界。

### 4.2 翼型问题

论文表格给出的十次训练统计：

| 模型 | Diversity | Quality |
|---|---:|---:|
| GAN | $-22.0326\pm1.4982$ | $56.1703\pm66.4480$ |
| PaDGAN | **$-12.5455\pm4.8970$** | **$76.0670\pm43.9802$** |

PaDGAN 的平均性能高于普通 GAN，同时覆盖范围更大。

![[assets/PaDGAN - A Generative Adversarial Network for Performance Augmented Diverse Designs/论文中图10_翼型新颖设计.png]]
*论文中图10：t-SNE（t-Distributed Stochastic Neighbor Embedding）空间中的真实数据、GAN 与 PaDGAN。PaDGAN 在训练样本凸包之外出现新设计，并在其中找到高 $C_L/C_D$ 的新颖翼型区域。*

## 5. 关键贡献

1. 把 DPP 的质量-多样性分解直接改造成可训练的生成模型损失。
2. 通过 $\gamma_0$ 提供质量/多样性的显式 trade-off 控制。
3. 用实验证明“只奖励质量”会模式坍塌，而联合 DPP 目标能保留多个高性能模式。
4. 展示了**受性能梯度引导的设计空间边界扩展**，使生成模型从“拟合历史设计分布”向“主动探索高质量新区域”迈进一步。

## 6. 局限与启发

### 局限

- 依赖**可微且可信的性能估计器**；代理模型在训练域外的错误梯度可能把生成器引向虚假的“高性能”区域。
- 多样性完全由所选相似度核决定。直接在原始高维几何坐标上用欧氏距离/RBF，不一定等价于工程上有意义的“形状差异”。
- DPP log-det 对 batch 内样本整体起作用，计算涉及矩阵特征值/行列式，batch 变大时成本上升且数值稳定性需要注意。
- “外推”并不自动等于物理可行。论文也指出显式约束没有被建模，只是通过逼近训练分布隐式学习约束。
- 它没有“目标性能条件”输入，因此不能直接请求“生成 $C_L/C_D=80$ 的不同翼型”；它只是整体偏向高质量。

### 对生成式工程设计的启发

最值得借鉴的是把“生成分布应该长什么样”直接写进训练目标。对于后续机翼/翼型 latent code 生成，可以把 $q(x)$ 换成气动性能、结构重量、约束裕度或多目标标量化，把 $k(x_i,x_j)$ 换成更符合设计语义的 latent/几何距离。这样生成器不只是复刻数据，还能被引导形成一个“高性能设计先验”。

## 7. 关键概念补充（联网核查）

### DPP 为什么天然适合多样性？

Kulesza & Taskar 的 DPP 综述将 DPP 描述为一种建模**排斥/负相关**的概率过程：一组元素越相似，它们共同被选中的概率越低；同时 $L$ 核可分解成 quality 与 similarity 两部分。这正是 PaDGAN 用 log-det 同时编码“好”和“不同”的理论来源。参考：*Determinantal Point Processes for Machine Learning*, DOI 10.1561/2200000044。

### GDPP 与 PaDGAN 的区别

GDPP（Generative Determinantal Point Process）也用 DPP 缓解 GAN mode collapse，但它的目标是让生成数据的多样性**匹配训练数据**；PaDGAN 则直接**最大化生成样本的多样性**并显式引入工程质量。因此若训练数据本身偏向少数模式，PaDGAN 理论上不必复刻这种偏置。

## 8. 文献脉络（约 3 篇核心工作）

1. **GDPP, Elfeki et al., 2019**：把 DPP 引入生成模型以抵抗 mode collapse；PaDGAN 在此基础上进一步加入工程性能质量，并不要求多样性复制训练集。
2. **BézierGAN, Chen & Fuge**：提供适用于光滑曲线/翼型的低维可解释生成表示；PaDGAN 的翼型实验直接使用 BézierGAN 作为生成器骨架。
3. **PcDGAN, Nobari et al., 2021**：直接继承 PaDGAN 的 DPP 思路，但把“quality”重定义为“满足指定连续性能条件的程度”，从无条件高性能生成扩展为连续条件逆设计。见 [[PcDGAN - A Continuous Conditional Diverse Generative Adversarial Network For Inverse Design]]。

## 9. 关联

- [[PcDGAN - A Continuous Conditional Diverse Generative Adversarial Network For Inverse Design]]
- [[PaDGAN 与 PcDGAN：关系、差异与选择]]
