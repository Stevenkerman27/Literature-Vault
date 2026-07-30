---
title: "DiffAirfoil: An Efficient Novel Airfoil Sampler Based on Latent Space Diffusion Model for Aerodynamic Shape Optimization"
authors: [Zhen Wei, Edouard Dufour, Colin Pelletier, Pascal Fua, Michaël Bauerheim]
year: 2024
venue: "AIAA AVIATION FORUM AND ASCEND 2024"
doi: "10.2514/6.2024-3755"
tags: [paper, aerodynamic-design, airfoil, diffusion-model, generative-design, surrogate-based-optimization, shape-parameterization]
status: read
source: "D:\\3D\\Projects\\Papers\\storage\\RV3D5LJE\\Wei 等 - DiffAirfoil An Efficient Novel Airfoil Sampler Based on Latent Space Diffusion Model for Aerodynami.pdf"
---

# DiffAirfoil: An Efficient Novel Airfoil Sampler Based on Latent Space Diffusion Model for Aerodynamic Shape Optimization

> [!abstract] 摘要直译
> 基于代理模型的优化被广泛用于气动外形优化，其有效性取决于能否对设计空间进行有代表性的采样。然而，传统采样方法难以有效采样高维设计空间。本文提出 DiffAirfoil：一种新的翼型采样方法，其基础是在自动学习得到的潜空间中进行扩散。DiffAirfoil 的数据效率很高，相较于生成对抗网络，只需显著更少的训练几何体。它通过自动参数化保证采样翼型的有效性。我们证明，DiffAirfoil 能生成多样且有效的二维翼型；同时，无需适配或重新训练即可进行条件生成。全面的基准测试表明，本方法在数据效率、实现难度和条件满足性方面具有显著优势。因此，DiffAirfoil 有望提高气动外形优化任务中代理模型的采样效率。

## 问题与动机

代理模型优化（Surrogate-Based Optimization，SBO）的成败首先受训练集限制：如果训练翼型没有覆盖优化过程会经过的形状区域，代理模型即使在已有样本上误差很小，也会在优化器真正关心的区域给出错误梯度或错误最优点。传统的拉丁超立方采样（Latin Hypercube Sampling，LHS）通常在人工参数化系数中均匀取点；当系数很多时，均匀并不等于翼型几何合理，所得样本既可能出现不光滑或不合弦长对齐的轮廓，也高度依赖人为设定各系数的取值范围。

生成对抗网络（Generative Adversarial Network，GAN）能从数据中学习几何分布，却有两个与低样本气动任务直接冲突的问题：生成器和判别器的对抗训练易不稳定，且数据量不足时会发生模式坍塌，即只生成少量相似翼型。已有翼型 GAN 工作通常使用超过 1400 个几何体，而真实设计项目中可直接复用的历史外形往往很少，甚至与目标翼型用途不相近。

本文将“生成新几何”拆成两个较易控制的问题：先用连续、可微的几何模型把翼型限制在一个有效潜空间，再让扩散模型只学习该潜空间的概率分布。这样，扩散模型不必同时学习点的拓扑、轮廓平滑性和弦向对齐；条件约束也可在采样时以梯度形式加入，而非为每一个新约束重新训练条件生成器。

## 方法

![[assets/diffairfoil-latent-space-diffusion-airfoil-sampler/framework-crop-p002.png]]

*图 1。上半部分：无条件潜空间扩散模型由高斯噪声反向去噪得到潜向量，再由几何解码器变形模板翼型。下半部分：每个反向步加入由几何测量误差反传的条件引导，令最终翼型满足目标面积或最大厚度。*

### 1. 自动几何参数化：潜空间模型

作者采用先前提出的潜空间模型（Latent Space Model，LSM）作为翼型参数化器。以 NACA-0012 的离散轮廓 $\hat M=\{\hat V,E\}$ 为固定模板，其中 $\hat V$ 是 $N$ 个顶点、$E$ 是固定边连接关系。一个多层感知机（Multi-Layer Perceptron，MLP）$f_\Theta(z,\hat v)$ 读取潜向量 $z\in\mathbb{R}^{256}$ 和模板顶点坐标，输出该点的二维位移 $\delta v$；全部顶点叠加位移后得到 $M=\{\hat V+f_\Theta(z,\hat V),E\}$。

训练 LSM 时，每个训练翼型都有一个可优化的潜码，网络权重与全部潜码共同更新。损失由 Chamfer distance（CD，倒角距离）重建项和潜码 $L_2$ 正则组成。CD 衡量两组表面点的双向最近距离，正则则使已学习的潜空间平滑、紧凑。因模板的点顺序与边连接在解码时始终不变，生成结果天然保留封闭轮廓、上下表面连续性和统一弦长，而不是再靠后处理修补非法几何。

这一步的角色值得区分：LSM 本身能够把一个已有目标翼型编码为潜码，也能从给定潜码解码几何，但它不提供“应从哪里取潜码”的分布。直接从高斯分布或潜空间包围盒取点仍可能离开训练形状流形，因此后续扩散模型学习的是已优化潜码的真实分布。

### 2. 无条件生成：在潜空间执行扩散

潜空间扩散模型（Latent Space Diffusion Model，LSDM）遵循去噪扩散概率模型（Denoising Diffusion Probabilistic Model，DDPM）。正向过程从训练翼型的潜码 $z_0$ 开始，在 $T=100$ 个马尔可夫步骤中按方差日程 $\beta_t$ 逐步加入高斯噪声，直至 $z_T\sim\mathcal{N}(0,I)$；本文令 $\beta_t$ 从 $10^{-4}$ 线性增长至 $0.02$。反向过程从标准高斯噪声出发，用三隐层、带 Leaky ReLU 激活的 MLP $g_\Phi(z_t,t)$ 预测该时间步的噪声，再据此得到 $z_{t-1}$。

训练不需要让网络直接回归完整轮廓，而是随机抽取时间步 $t$ 和噪声 $\epsilon$，最小化预测噪声与 $\epsilon$ 的平方差。这是由证据下界（Evidence Lower Bound，ELBO）化简得到的常用 DDPM 目标；KL divergence（Kullback-Leibler divergence，KL 散度）项由固定噪声日程处理。推理时重复 100 次反向更新，最后把 $z_0$ 交给冻结的 LSM 解码为翼型。

这一分层使模型规模与数据需求明显小于直接在点云或图像上做扩散：LSDM 只处理 256 维连续潜码，LSM 负责几何先验。论文报告在 NVIDIA V100 图形处理器（Graphics Processing Unit，GPU）上，50 个样本的 LSM 与 LSDM 训练时间分别为 671.9 s 和 716.4 s，单个新翼型采样为 61.7 ms；这些是特定实现和硬件下的计时，不应直接外推至其他网络或三维网格。

### 3. 条件生成：把可微几何约束嵌入每一步去噪

条件潜空间扩散模型（Conditional Latent Space Diffusion Model，CLSDM）没有额外训练一个“条件网络”。给定目标几何量 $\hat C$，例如面积或最大相对厚度，作者将它表示为 LSM 解码结果的可微测量 $c(z)$。在每个反向时间步，将无条件噪声预测修正为

$$g_\Phi(z_t,t)+\nabla_{z_t}\|c(z_t)-\hat C\|^2.$$

直观上，第一项把点推向高概率的有效翼型潜码，第二项沿着“减小条件误差”的方向拉动潜码；二者同时参与去噪。因此条件不是在最终翼型生成后再投影或筛选，而是持续塑造整条生成路径。面积可由 Shoelace formula（鞋带公式）对有序顶点直接计算；最大厚度可由上下表面三次样条的纵坐标差得到。只要新增指标能从 LSM 输出对潜码求导，它可作为插件加入，不要求为新条件再收集标注数据或重训模型。

这也界定了适用边界：方法天然适配面积、厚度、曲率等可微几何量；若条件是不可微的网格质量、离散制造规则、带强数值噪声的失速裕度或一次 CFD 响应，就需要可微替代模型、平滑近似或另行设计引导项，不能直接沿用本文公式。

## 基准实验

### 数据、模型和评价

- 数据来自 UIUC（University of Illinois Urbana-Champaign）翼型数据库。作者随机取 1000、500、250、100 和 50 个翼型组成五组训练集，并在同一 LSM 潜空间内比较 LSDM 与 GAN，以及 CLSDM 与条件生成对抗网络（Conditional GAN，CGAN）。这使对比聚焦于采样器，避免将不同几何表示的差异误归因于生成模型。
- GAN/CGAN 的生成器和判别器均为两隐层 MLP，优化器为 AdamW；LSDM 也用 AdamW，学习率 $10^{-5}$。LSM 权重与潜码的 Adam 学习率分别为 $5\times10^{-4}$ 和 $10^{-3}$。
- 分布真实性采用 Fréchet inception distance（FID，弗雷歇特 Inception 距离）评估。这里不是将翼型渲染为图像，而是先训练 CFD 代理模型，从其末隐层取特征，再比较 UIUC 真样本和生成样本的特征均值与协方差；FID 越小，表示两组分布越接近。
- 探索性分为多样性与新颖性。$D^{10}_{\mathrm{intra}}$ 是每个生成翼型到同批 10 个最近邻的平均 CD，再对样本平均，较大表示生成集内部差异大；$D^{10}_{\mathrm{inter}}$ 是每个生成翼型到数据库中 10 个最近邻的平均 CD，较大表示不会只复刻训练样本。作者也报告面积、最大厚度、弯度弦长比的标准差（Standard Deviation，STD），并以 bootstrap（自助法）估计 95% 置信区间。

### 无条件与条件采样结果

在无条件实验中，LSDM 在所有五种数据量下均能生成弦长归一、表面平滑，且在厚度、弯度、前缘半径和最大厚度位置上有明显变化的翼型。FID、$D^{10}_{\mathrm{intra}}$、$D^{10}_{\mathrm{inter}}$ 与几何 STD 的趋势一致：数据足够时 GAN 和 LSDM 的 FID 接近；训练样本少于 500 后 GAN 的分布匹配显著变差，少于 250 后出现清晰模式坍塌，100 和 50 样本下多样性、新颖性接近失效；LSDM 的各指标基本不随数据减少而显著退化。潜码的 t-distributed Stochastic Neighbor Embedding（t-SNE，t 分布随机邻域嵌入）可视化也显示 LSDM 的样本覆盖训练分布，而 GAN 样本聚成少数小团。

条件实验将翼型面积固定为 0.09。CLSDM 将鞋带公式的面积误差梯度放进采样步骤；CGAN 则把面积标量作为训练和推理输入。CLSDM 在 50、100、250、500、1000 个训练样本下均保持更小的条件绝对误差和更好的覆盖。CGAN 即使样本较多，面积误差仍明显较大，且小样本下同时出现分布偏置和聚团。CGAN 在 50 样本时表面上有时显示更高“新颖性”，但作者指出这主要源于它没有满足面积条件，因而不构成有效优势。

论文并未给出每个图中所有数值的表格、独立随机种子重复次数或显著性检验；“只需 50 个样本”的结论应理解为该 UIUC 子集、该网络大小与该二维任务下的强实证结果，而不是对任意几何数据集的无条件保证。

## 代理优化案例：采样质量是否传递为设计收益

作者在二维 NACA-0012 跨声速优化中检验采样器的工程价值。目标是最大化升阻比 $L/D$，最大厚度固定为 12% 弦长；流动为无黏跨声速，马赫数 $Ma=0.85$、攻角为 $0^\circ$。网格由直接映射模型（Direct Mapping Model，DMM）变形，使用 SU2 的 Euler 方程求解器生成 CFD 数据。

四种代理模型的训练数据策略如下：

| 代理模型 | 训练样本来源 | 设计含义 |
| --- | --- | --- |
| SM#1-1 | 500 个围绕 NACA-0012 的 15 阶 Class/Shape Transformation（CST，类别-形状变换）系数 LHS 样本，较窄的人工范围 | 传统局部扰动；范围设置较保守。 |
| SM#1-2 | 同样 500 个 CST-LHS 样本，但扩大系数范围 | 仅改变人工范围，检验传统方案对调参是否敏感。 |
| SM#2 | UIUC 中随机选取的 50 个历史翼型 | 数据少且多数与 NACA-0012 用途无关。 |
| SM#3 | 用 SM#2 的 50 个翼型训练 CLSDM，再按 12% 最大厚度条件生成的样本 | 目标构型周围的有效、受约束生成数据。 |

所有气动代理均为两隐层、ReLU 激活的 MLP，输入为轮廓 60 点的 $x,y$ 坐标（120 维），输出升力系数 $C_L$ 与阻力系数 $C_D$，用 Adam 训练 500 个 epoch。外形优化仍在 LSM 潜变量中进行：先编码 NACA-0012，再对代理给出的 $C_D$、$C_L$、厚度罚项和潜码正则进行 Adam 优化，学习率 $10^{-4}$、共 2000 步。这种设定同时避免了直接在 120 个坐标中优化而造成的几何失效。

最终结果中，SM#3 的代理预测 $L/D=8.2603$，真实仿真 $L/D=7.7188$，沿优化轨迹的决定系数 $R^2=0.9620$。相比之下，SM#1-1、SM#1-2、SM#2 的真实最终 $L/D$ 分别为 6.6236、0.9195、5.8405，对应 $R^2$ 为 0.3265、-2.1509、0.2041。后三者的代理会把优化推向过大弯度，真实阻力反而上升；SM#3 的形状改动最小且真实性最高。该案例支持的结论不是“生成模型直接找到最优翼型”，而是“条件生成的训练集覆盖更贴近优化轨迹，因而提高后续代理模型在关键区域的可信度”。

## 关键贡献

1. 以 LSM 提供连续、可微且拓扑固定的翼型参数化，再以低维扩散模型学习潜码分布，将有效性和概率采样两个问题解耦。
2. 在 50 至 1000 个 UIUC 翼型的数据稀缺梯度上，同 GAN/CGAN 做了质量、探索性和条件满足性的系统比较，展示小样本下对模式坍塌的鲁棒性。
3. 以可微测量误差的梯度引导反向扩散，实现面积、最大厚度等条件的即插即用控制，不需要为每个新条件重训模型。
4. 将生成样本接入真实的 SBO 闭环，并用四种数据生成策略说明生成先验的价值体现在代理模型的局部有效性，而非仅是视觉上更像训练翼型。

## 局限与启发

- **结论目前限于二维翼型。** 固定拓扑的模板变形在二维封闭曲线中很自然；三维机翼、翼身融合体或含复杂网格拓扑时，LSM 的点对应、网格有效性和潜空间维度都会更困难。论文将三维扩展列为未来工作，尚未给出验证。
- **“有效几何”不等于“工程可用”。** 弦向对齐和平滑轮廓没有覆盖结构强度、制造厚度、前缘曲率、操纵面、网格质量或多工况气动约束。工程使用应把这些作为条件项、硬筛选或独立验证。
- **条件引导依赖可微且尺度合理的测量。** 多个条件之间的梯度量级可能冲突；本文未系统讨论引导权重、条件容差或不可行目标的处理。实际实现需监测条件残差与样本多样性，避免把高概率形状拉向低概率甚至不合理的区域。
- **生成器不是 CFD 替代品。** 它降低的是建立代理训练集时“如何生成可信候选几何”的难度；每个新样本的 $C_L,C_D$ 仍需由 CFD 或可靠气动代理评估。高保真预算应优先投入条件生成的候选及其附近区域，并以主动学习持续回填。
- **可复用原则。** 先将可行几何编码为连续潜空间，再在潜空间建模分布和优化，通常比在手工高维系数或原始点坐标中直接随机取样更易控制。这个原则可与 [[Data-driven surrogate model for aerodynamic design using separable shape tensor method]] 的流形降维、[[SURROOPT A GENERIC SURROGATE-BASED OPTIMIZATION CODE FOR AERODYNAMIC AND MULTIDISCIPLINARY DESIGN]] 的加点策略结合。

## 关联

- 自动参数化前作：[[Automatic Parameterization for Aerodynamic Shape Optimization via Deep Geometric Learning]]、[[Latent Representation of CFD Meshes and Application to 2D Airfoil Aerodynamics]]
- 生成与优化主题：[[Diffusion model]]、[[Generative design]]、[[Airfoil design]]、[[Surrogate-based optimization]]
- 对比思路：[[Generative Adversarial Networks]]、[[Data-driven surrogate model for aerodynamic design using separable shape tensor method]]
