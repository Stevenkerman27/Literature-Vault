---
title: "PcDGAN: A Continuous Conditional Diverse Generative Adversarial Network For Inverse Design"
authors:
  - Amin Heyrani Nobari
  - Wei Chen
  - Faez Ahmed
year: 2021
venue: "KDD '21: Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining"
doi: "10.1145/3447548.3467414"
source: "/mnt/data/PcDGAN_A_continuous_conditional_diverse_GAN_for_Inverse_Desisgn(1).pdf"
tags:
  - generative-design
  - conditional-GAN
  - inverse-design
  - DPP
  - airfoil
---

> [!abstract] 摘要（原文翻译）
> 工程设计经常需要直接合成满足给定性能要求的新设计。传统流程依赖反复优化和性能评估，速度慢且受初始设计影响。已有 conditional GAN（cGAN，条件生成对抗网络）可根据目标性能直接生成设计，但多数 cGAN 只适合离散类别条件；Continuous conditional GAN（CcGAN）虽然处理连续条件，却在性能标签分布不均匀时表现较差，而且生成样本可能不能覆盖完整设计空间。本文提出 **Performance Conditioned Diverse Generative Adversarial Network（PcDGAN）**：以 singular vicinal loss（单标签邻域损失）解决连续标签稀疏与不均衡，以基于 DPP 的条件多样性损失同时提升设计覆盖，并提出 Lambert Log Exponential Transition Score（LLETS）作为自增强条件质量分数。合成问题和翼型逆设计表明，PcDGAN 相比 CcGAN 在翼型任务中将 conditioning likelihood 提高 69%，并得到更好的设计空间覆盖；在合成问题中条件似然最高提升约 78%。

## 一句话定位

PcDGAN 可以理解为 **PaDGAN + 连续条件建模**：它不再问“哪些设计整体质量高”，而是问“在我指定的连续性能 $y^*$ 下，能不能一次生成很多**都接近 $y^*$、但几何上彼此不同**的设计”。

## 1. 问题与动机

连续性能条件和图像类别条件本质不同。若条件 $y$ 是 $C_L/C_D$：

- $y$ 是连续实数，几乎不可能为每一个精确数值准备大量训练样本；
- 标签分布通常极不均匀，极端高/低性能区域尤其稀疏；
- 逆设计要求比普通条件图像生成更严格：**输出不仅要像真实设计，还必须数值上满足目标性能**；
- 对同一性能，工程上希望得到多个不同形状，而不是一个单一答案。

CcGAN 用 vicinal loss（邻域损失）缓解连续条件数据稀疏，但它仍按训练数据中的标签分布采样，数据密集区训练更频繁；同时没有专门机制强制“同一条件下的多样性”。PcDGAN 针对这两个缺口设计。

## 2. 总体架构

![[assets/PcDGAN - A Continuous Conditional Diverse Generative Adversarial Network For Inverse Design/论文中图1_PcDGAN架构.png]]
*论文中图1：PcDGAN。生成器以噪声和连续条件为输入；性能估计器给出生成样本的预测标签；LLETS 把条件误差转换为 DPP quality；判别器使用 singular vicinal loss；两部分共同训练生成器。*

训练中同时有三股信号：

1. **GAN/邻域对抗信号**：确保输出仍属于真实设计流形；
2. **条件误差信号**：估计器判断生成设计的实际性能是否接近目标；
3. **DPP 多样性信号**：要求同一条件附近的一批样本互相不同。

## 3. 方法详解

### 3.1 从 CcGAN 的 vicinal loss 到 singular vicinal loss

CcGAN 的核心思想来自 Vicinal Risk Minimization（VRM，邻域风险最小化）：当某个连续标签 $y$ 附近没有足够精确匹配样本时，不只用 $y$ 本身，而是在其邻域内使用相近标签样本来训练条件判别器。

CcGAN 有 Hard Vicinal Discriminator Loss（HVDL）和 Soft Vicinal Discriminator Loss（SVDL）。SVDL 对邻域内样本按

$$
w(y_i,y)=\exp[-\nu(y_i-y)^2]
$$

加权，标签越接近当前条件贡献越大。

问题在于 CcGAN 的训练标签仍主要来自**数据自身的经验分布**。若中间区域样本很多、两端很少，训练也会把多数更新花在中间区域。

PcDGAN 的做法有两步：

1. 先在归一化标签范围（0–1）中**均匀采样一个目标值**；
2. 找训练集中与这个值最接近的真实标签 $y_s$，当前训练 step 只围绕这一个 $y_s$ 的邻域计算 vicinal loss，即令所有 $y_j=y_s$。

所以“singular”不是指只有一个训练样本，而是指**每个 step 只聚焦一个条件中心**。这样条件空间本身被近似均匀遍历，不再由数据密度决定训练频率。

### 3.2 为什么单独增加一个性能估计器

传统 cGAN/CcGAN 让判别器同时完成两件事：判断真假 + 感知条件是否匹配。PcDGAN 把“条件满足度”显式交给一个独立的 differentiable performance estimator（可微性能估计器）。

估计器可以是：

- 可微物理模型；
- 神经网络 surrogate；
- 任何能够对生成样本给出连续性能预测并允许梯度回传的模型。

这使“是否满足性能目标”成为一个可单独优化、可量化的信号，而不是完全寄托在判别器内部的隐表示上。

### 3.3 LLETS：把条件误差变成“自增强”质量分数

令

$$
\epsilon=\|y_{condition}-y_{estimated}\|_1,
$$

作者认为直接用 $-L_1$ 或 $-L_2$ 作为 DPP quality 不理想：$L_1$ 梯度恒定，$L_2$ 在误差大时梯度反而大，不符合“已经接近目标时进一步精修”的需求。

因此构造 Lambert Log Exponential Transition Score（LLETS）：

$$
LLETS(\epsilon)=
\begin{cases}
-\dfrac{\ln\epsilon}{a}, & \epsilon>e^{-ae^{W(-1/(2a))}}\\[4pt]
\exp\left(-\dfrac{\epsilon^2}{2\sigma^2}\right), & \epsilon\le e^{-ae^{W(-1/(2a))}}
\end{cases}
$$

其中 $a\ge e/2$ 是 Lambert cutoff，$W$ 是 Lambert W function，$\sigma$ 由连续一阶导数条件确定。论文所有实验取 $a=4.7$。

它想实现四个性质：

1. 不同标签下的 quality 尺度一致；
2. 越接近正确条件，梯度越强化，形成 self-reinforcing；
3. 到非常高的条件精度后梯度重新变平，避免振荡；
4. 只有完美条件误差 $\epsilon=0$ 处梯度归零。

这里 Lambert W 的作用主要是**解析地确定两段函数平滑连接的位置/参数**，不是模型本身在“学习 Lambert W”。

### 3.4 Performance-conditioned DPP loss

PcDGAN 继承 PaDGAN 的 DPP 核：

$$
L_B(i,j)=k(x_i,x_j)\big(q(x_i)q(x_j)\big)^{\gamma_0},
$$

但这里 $q(x)$ 不再表示“设计本身有多高性能”，而是表示**它对当前目标条件满足得有多好**，即由 LLETS 给出。

DPP loss：

$$
\mathcal L_{PcD}(G)=-\frac1{|B|}\log\det(L_B).
$$

于是同一个 batch 想降低损失，必须同时做到：

- $q(x_i)$ 高：每个样本都接近指定条件；
- $k(x_i,x_j)$ 低相关：样本彼此不同。

这正是“conditional diversity”：**相同性能，多种形态**。

### 3.5 为什么 singular vicinal loss 与 DPP 必须配合

如果一个 batch 混合许多相距很远的条件，那么 DPP 会把“性能不同导致形状不同”也误当成多样性。PcDGAN 每一步集中在一个条件邻域，使 DPP 看到的是“基本相同条件下，形状是否不同”。

所以两者分工非常明确：

- singular vicinal loss：把训练局部化到单一性能邻域，并让条件空间采样更均衡；
- DPP：在这个局部条件切片上铺开设计分布；
- LLETS：阻止 DPP 为了扩大多样性而把样本推离目标性能。

总目标为

$$
\mathcal L^{PcDGAN}(D,G)=\mathcal L^{sVDL}(D)+\mathcal L^{\epsilon}(G)+\gamma_1\mathcal L^{PcD}(G).
$$

### 3.6 条件输入机制

CcGAN 的 Improved Label Input（ILI）先训练一个 embedder，把连续标签映射到预训练回归器最后一层特征。PcDGAN 认为这有三个问题：计算昂贵、从标量标签到高维特征可能是一对多、若使用的是物理求解器而非神经网络估计器则根本没有“最后一层特征”。

PcDGAN 改为简单的**可学习 embedding + 线性层**：

- 生成器：条件向量用于 Conditional Batch Normalization（CBN，条件批归一化）；
- 判别器：用 projection discriminator（投影判别器）把条件 embedding 与样本特征作内积。

![[assets/PcDGAN - A Continuous Conditional Diverse Generative Adversarial Network For Inverse Design/论文中图2_条件输入机制.png]]
*论文中图2：生成器通过 CBN 注入连续标签，判别器通过 label projection 注入标签。*

## 4. 实验设置

### 4.1 合成问题

性能函数仍为六峰高斯混合：$K=6,\sigma\approx0.1$，六个峰均匀分布在半径 0.4 的圆周上。对任意给定条件，理想生成器都应在六个 mode 上产生满足该条件的样本。

- **Example 1**：10,000 个数据点在 $[-0.6,0.6]^2$ 均匀分布；考察数据均衡时的条件生成。
- **Example 2**：一半数据保持均匀，另一半集中在某一个峰附近半径 0.2 的圆内；专门制造设计空间严重不均衡，考察 mode collapse 与条件偏置。

PcDGAN 使用精确的性能函数作为 estimator，不需要代理模型。

### 4.2 真实翼型逆设计

- UIUC 数据库，预处理与增广后 **38,802** 个翼型；
- 每个翼型用 **192 个表面点，每点二维坐标**表示，即 $x_i\in\mathbb R^{192\times2}$；
- 条件标签为 $C_L/C_D$，由 XFOIL 计算后归一化到 0–1；
- 标签分布明显不均衡：均值 0.4507，标准差 0.1483；**高于 0.9 的样本仅 0.18%，低于 0.1 的仅 0.5%**；
- estimator：在全部 38,802 个翼型上训练的 ResNet surrogate；训练 surrogate 时也使用均衡 minibatch，避免代理本身向密集标签区偏置；
- 生成器：BézierGAN。

### 4.3 主要超参数与训练细节

- RBF 相似度核 bandwidth = 1；
- 合成：$\gamma_0=3.0,\gamma_1=0.5$；
- 翼型：$\gamma_0=3.0,\gamma_1=0.4$；
- LLETS：$a=4.7$；
- 两种方法都采用 soft vicinal loss；$\kappa,\sigma$ 按 CcGAN 的经验规则选择；
- 翼型中 quality 进一步乘判别器输出 $q(x)=D(x)q'(x)$ 以增强训练稳定性；
- 翼型训练也采用逐步增加 $\gamma_1$ 的 schedule，指数 $p=5$；
- GAN 训练 20,000 steps，batch size 32；Adam 初始学习率 $10^{-4}$，每 2,000 step 乘 0.8。

## 5. 评价指标

### Label Error

目标条件与生成样本预测实际标签之间的平均绝对误差。越小越好。

### Likelihood Score

对生成样本的实际标签做 Gaussian KDE（Kernel Density Estimation，高斯核密度估计），计算**目标条件在该标签分布下的密度**。它回答“随机抽一个生成设计，有多大可能正好满足设计者要求”。越大越好。

作者认为这个指标比均值误差更接近逆设计需求，因为少量极端 outlier 会拉大 MAE，但不一定代表目标附近没有大量有效解。

### Diversity Score

对生成样本随机取子集，以相似度矩阵的 log determinant 评价覆盖。越大（越不负）越好。

评估协议很重：对 0.05–0.95 之间 100 个条件，每个条件生成 1,000 个样本；这一过程重复 10 次，并对每种方法训练 10 个独立模型，以衡量随机性。

## 6. 结果

### 6.1 合成问题

Example 2 最能说明方法作用：CcGAN 倾向在占优势的数据 mode 生成样本，虽然那里更容易满足条件，但牺牲了其他 mode；PcDGAN 即使面对不均衡数据，也能在全部 mode 上生成条件正确的样本。

### 6.2 翼型结果

论文 Table 1：

| 模型 | Likelihood Score ↑ | Label Error ↓ | Diversity Score ↑ |
|---|---:|---:|---:|
| CcGAN | $3.429\pm2.240$ | $0.119\pm0.026$ | $-19.297\pm2.140$ |
| PcDGAN | **$11.181\pm2.686$** | **$0.0284\pm0.004$** | **$-16.567\pm1.509$** |

也就是说 PcDGAN 不只是“平均更准”，而且不同训练 run 的方差更小。

![[assets/PcDGAN - A Continuous Conditional Diverse Generative Adversarial Network For Inverse Design/论文中图5_翼型实验性能对比.png]]
*论文中图5：从低到高整个条件范围，PcDGAN 的 likelihood 更高、label error 更低、diversity 更好且方差更小。*

![[assets/PcDGAN - A Continuous Conditional Diverse Generative Adversarial Network For Inverse Design/论文中图7_tSNE覆盖对比.png]]
*论文中图7：固定条件 0.3，并只保留预测标签在 $0.3\pm0.025$ 内的“有效设计”后，PcDGAN 在 t-SNE 空间中覆盖的区域明显大于 CcGAN，说明提升的多样性不是靠生成条件错误的样本换来的。*

## 7. 关键贡献

1. **连续条件 + 多样性统一建模**：把 PaDGAN 的 DPP quality 从“性能高低”改成“条件满足程度”。
2. **Singular vicinal loss**：从标签空间均匀采样，并每步只围绕一个条件邻域训练，解决非均匀标签分布造成的偏置。
3. **LLETS**：为条件误差设计自增强梯度形状，在接近目标时仍持续强化条件精度。
4. **独立性能估计器**：允许直接把可微物理/代理模型接入条件训练，而不必让判别器独自承担回归信息。
5. **更轻量的 condition input mechanism**：不再依赖 CcGAN 的高成本 embedder。

## 8. 局限与启发

### 局限

- 论文主要处理**单个标量连续条件**。作者明确把高维/多目标条件留作未来工作；条件维度升高后，邻域数据稀疏会更严重。
- 仍依赖可微性能估计器。若 surrogate 在生成分布外不可信，LLETS 的强梯度可能放大代理误差。
- LLETS 的 cutoff $a=4.7$ 是经验选择；该梯度形状是否在其他设计域仍最优，没有理论/消融充分证明。
- DPP 的“多样性”取决于 $k(x_i,x_j)$。对翼型直接在几何表征上使用 RBF，不一定等价于气动/工程语义上的多样性。
- 训练更复杂：需要 GAN + estimator + DPP 矩阵 + 连续条件采样，且对数行列式有数值稳定性要求。
- 它保证的是“对 estimator 的条件满足”，最终高保真仿真/实验仍应作为校验闭环。

### 对工程生成模型的启发

PcDGAN 给出一个很重要的设计范式：**条件不必只作为网络输入，也可以直接进入 loss，成为“生成 batch 的质量定义”**。对于机翼 latent code 生成，如果希望输入目标升阻比/结构重量后输出多个不同 code，可以把 DPP 相似度定义在 latent 或解码后的几何上，把 LLETS quality 定义在 surrogate 预测性能与目标之间，从而把“精确条件 + 多解性”同时写进训练目标。

## 9. 关键概念补充（联网核查）

### Vicinal Risk Minimization（VRM）

Chapelle、Weston、Bottou 与 Vapnik 的 VRM 思路，是用训练样本附近的“邻域分布”替代只在离散经验点上做风险最小化。CcGAN 将这一思想用于连续标签：一个精确条件没有真实样本时，用附近标签样本构造可训练的局部经验风险。PcDGAN 不是推翻 VRM，而是改变**邻域中心的采样策略**和每步覆盖方式。

### Lambert W function

Lambert W 是 $w e^w=z$ 的反函数，即 $w=W(z)$。PcDGAN 用它求解 LLETS 两段函数平滑连接所需的参数。其标准数学定义可见 NIST DLMF §4.13；经典系统论述见 Corless et al., 1996, DOI 10.1007/BF02124750。

### Conditional Batch Normalization（CBN）

CBN 的核心是让条件向量决定 Batch Normalization 中的缩放与偏移参数 $\gamma,\beta$，即条件信息可以在多个卷积层持续调制特征，而不仅是网络输入端拼接一次。原始工作由 de Vries et al. 在 NIPS 2017 系统化提出。

### Projection Discriminator

Miyato & Koyama 的 projection discriminator 不简单把标签拼到判别器特征后，而是让标签 embedding 与样本特征做内积，并将其加入真假评分。这样条件兼容性直接进入判别器能量/打分函数，PcDGAN 用此机制注入连续标签。

## 10. 文献脉络（约 3 篇核心工作）

1. **CcGAN, Ding et al.**：首先系统解决连续标量条件的 GAN 训练，用 HVDL/SVDL 和新标签输入机制处理连续条件稀疏；PcDGAN 的 singular vicinal loss 直接以其 vicinal loss 为基础。
2. **PaDGAN, Chen & Ahmed**：提出 performance-augmented DPP loss，把质量和多样性统一；PcDGAN 几乎完整继承这一 DPP 框架，但把 quality 改造成“条件准确性”。见 [[PaDGAN - A Generative Adversarial Network for Performance Augmented Diverse Designs]]。
3. **GDPP, Elfeki et al.**：更早使用 DPP 正则抑制 mode collapse，提供“DPP 可以直接作为生成模型多样性损失”的方法脉络；PaDGAN/PcDGAN进一步引入工程质量/条件信息。

## 11. 关联

- [[PaDGAN - A Generative Adversarial Network for Performance Augmented Diverse Designs]]
- [[PaDGAN 与 PcDGAN：关系、差异与选择]]
