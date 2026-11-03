---
title: "Locality-aware Surrogates for Gradient-based Black-box Optimization"
authors: [Ali Momeni, Stefan Uhlich, Arun Venkitaraman, Chia-Yu Hsieh, Andrea Bonetti, Ryoga Matsuo, Eisaku Ohbuchi, Lorenzo Servadei]
year: 2025
venue: "ICML 2025, Proceedings of Machine Learning Research 267"
doi: "10.48550/arXiv.2501.19161"
tags: [literature, black-box-optimization, surrogate-model, gradient-estimation, GradPIE, ABBO]
status: read
source: "未提供本地 PDF"
url: "https://arxiv.org/abs/2501.19161"
---

> [!abstract] 摘要直译
> 本文针对物理与工程中的不可微黑箱系统，提出用于主动模型驱动黑箱优化的局部性感知代理模型。作者受梯度定理启发，建立梯度对齐与 Gradient Path Integral Equation（GradPIE）损失之间的理论联系，并提出支持离线和在线训练的算法。在耦合非线性振子、模拟电路和光学系统三个任务上，方法在有限黑箱查询预算下改善了优化效率。

## 研究问题

主动黑箱优化（Active Black-box Optimization，ABBO）允许优化过程中继续查询模拟器或物理系统，但真实梯度不可直接获得。常见做法是训练可微代理模型，在反向过程中由代理提供输入梯度，而前向评估仍使用真实黑箱。论文强调：点值预测准确不代表输入梯度准确；优化器真正依赖的是梯度方向，因此代理训练应约束局部变化关系。

设黑箱为 \(F:\mathcal X\subseteq\mathbb R^{D_i}\to\mathbb R^{D_o}\)，目标为
\[
x^\star=\arg\max_{x\in\mathcal X}\psi(F(x)).
\]
给定数据集 \(\mathcal D=\{(x_i,F(x_i))\}_{i=1}^N\)，基础代理模型（Base-model）通常以 MAE 或 MSE 拟合单点输出。

## GradPIE 方法

### 1. 从梯度定理得到点对损失

对可微函数 \(F\) 和连接 \(x,x'\) 的路径 \(\gamma_{x,x'}\)，梯度定理给出
\[
\int_{\gamma_{x,x'}}\nabla F(r)\cdot dr=F(x')-F(x).
\tag{3}
\]
作者将路径积分中的真实梯度替换为代理梯度，右侧保留黑箱真实输出差：
\[
\mathcal L_{\mathrm{GradPIE}}(\theta)=
\mathbb E_{(x,x')\sim\mathcal D^2}
\left\|\int_{\gamma_{x,x'}}\nabla\hat F(r;\theta)\cdot dr-
(F(x')-F(x))\right\|_1.
\tag{4}
\]
再次使用梯度定理可化为
\[
\boxed{
\mathcal L_{\mathrm{GradPIE}}(\theta)=
\mathbb E_{(x,x')\sim\mathcal D^2}
\left\|(\hat F(x';\theta)-\hat F(x;\theta))-
(F(x')-F(x))\right\|_1
}
\tag{5}
\]
因此训练不需要真实梯度标签或额外有限差分查询，只需要已有样本对的输入和真实输出。

### 2. 与梯度对齐的理论联系

MAE 比较单点输出；GradPIE 比较两个输入之间的输出变化。全局常数偏移在差分中抵消，因此 GradPIE 不要求真实函数与代理函数的绝对输出基准一致，而是约束局部相对变化。

论文把式（5）改写为真实系统与代理模型的雅可比矩阵差异沿路径的积分。对足够近的点对 \(\|x-x'\|\leq\epsilon\)，局部近似为
\[
(J[\hat F](x;\theta)-J[F](x))(x'-x).
\tag{8}
\]
在局部样本对分布密度连续等条件下，对邻域内位移方向积分，GradPIE 与雅可比差异的行范数成比例，行范数定义为
\[
\|A\|_{\mathrm{row}}=\sum_{d=1}^{D_o}\|A_{d:}\|_2.
\]
所以在局部近似及分布假设下，最小化 GradPIE 对应于减小真实系统与代理模型的雅可比差异。该推导依赖局部邻域足够小、路径可微和样本对分布密度连续等条件；它不是对任意距离点对的严格等式。

### 3. K 近邻实现

直接对所有样本两两计算式（5）会产生大量点对。作者对每个样本 \(x\) 预计算 \(K\) 个最近邻 \(x'_k\)，训练损失为
\[
\boxed{
\mathcal L_{\mathrm{GradPIE}}(\theta)=
\mathbb E_{x\sim\mathcal D}\left[
\frac1K\sum_{k=1}^{K}
\|(F(x)-F(x'_k))-(\hat F(x;\theta)-\hat F(x'_k;\theta))\|_1
\right].
}
\tag{10}
\]
这就是局部性感知的来源：只用近邻点对学习局部输出变化。\(K\) 是任务相关超参数；CNON 实验显示增加 \(K\) 不保证持续改善，合适值取决于输入维度和问题复杂度。作者也指出可将 L1 改为 MSE 形式。

## 算法流程

- **离线：** 对固定数据集预计算 K 近邻；用 mini-batch 优化 GradPIE，达到损失阈值后停止；固定代理模型，从标准正态分布初始化输入，随后用代理梯度迭代优化。优化期间不再查询黑箱，性能依赖初始数据覆盖。
- **在线：** 从初始数据集开始，每轮在当前优化点附近采样
  \[
  \hat x_i^{(t)}\sim\mathcal N(x^{(t)},\sigma^2I),
  \]
  调用黑箱获得真实输出并加入数据集；用更新后的数据重训代理；按局部候选的真实黑箱输出排序，选择前 \(N_{\mathrm{best}}\) 个样本，再用代理梯度更新输入。重复局部采样、真实查询、重训练和输入更新。

## 实验与结果

### CNON：耦合非线性振子网络

输入为振子初始振幅 \(q^0\)，输出为演化后的状态 \(q^\infty\)。作者用 1,000 组输入—输出训练 MLP，比较 MAE 与 GradPIE 的梯度估计；维度为 7 和 10 时，GradPIE 分别使梯度相对误差降低超过 27% 和 20%，余弦相似度提高超过 10% 和 8%。离线任务使用 \(D_i=7\)，最小化 \(\|q^\infty-\lambda\|_1\)。在线任务使用 \(D_i=10\)，比较 \(N_s=0,1\)。200 次迭代时，\(N_s=0\) 下 Base-model、GradPIE、精确梯度分别为 \(0.131\pm0.041\)、\(0.081\pm0.015\)、\(0.048\pm0.001\)；\(N_s=1\) 下分别为 \(0.114\pm0.030\)、\(0.066\pm0.007\)、\(0.048\pm0.001\)。指标越小越好。

### OpAmp：两级运算放大器

在 10 维设计空间调整晶体管尺寸和补偿电容，最大化
\[
\mathrm{FoM}=1.2\frac{G}{100}+1.6\frac{PM}{90^\circ}
+10\frac{UGBW}{1\ \mathrm{GHz}},
\]
其中 \(G\) 为线性直流增益、\(PM\) 为相位裕度、\(UGBW\) 为单位增益带宽。电路由 NGspice 与 Skywater-PDK 130nm 工艺模拟。实验先随机采样 500 组尺寸；每轮训练代理 100 个 epoch，再对 FoM 最好的 5 个样本使用 Adam 更新一步，共 200 轮。200 轮时 Random、Base-model、GradPIE 的 FoM 均值±标准差为 \(7.628\pm0.835\)、\(8.860\pm0.899\)、\(9.615\pm0.694\)。GradPIE 的 50 轮结果已优于 Base-model 的 200 轮结果。

### OWMS：光学波操纵系统

通过调整 \(60\times60\) 空间光调制器（Spatial Light Modulator，SLM）的相位图，使输出光场匹配目标光场，搜索空间为 3,600 维。入射光为腰半径 70 μm、波长 700 nm 的高斯光束，目标是分成两个高斯光束；传播由 Rayleigh–Sommerfeld 衍射积分描述，使用 TorchOptics 和 FFT 计算。200 轮时 Random、BayOpt、Base-model、GradPIE 的损失均值±标准差分别为 \(0.227\pm0.009\)、\(0.103\pm0.019\)、\(0.117\pm0.013\)、\(0.090\pm0.015\)，越小越好。

### 查询预算与网络结构

论文报告达到基线 200 轮性能所需的黑箱调用预算：CNON 约 \(O(100)\)（\(N_s=0\)）或 \(O(120)\)（\(N_s=1\)，约减少 50%/70%）；OpAmp 约 \(O(1000)\)，约减少 60%；OWMS 约 \(O(750)\) 达到 Base-model、约 \(O(1200)\) 达到 BayOpt 性能，约减少 62%/40%。三个任务的查询预算降低比例均超过 40%。

代理模型均为 MLP：CNON 为 2 层、每层 256 单元；OpAmp 为 4 层 [100,100,100,20]；OWMS 为 5 层 [1000,1000,1000,1000,500]。均使用 GELU、偏置和 Adam；OpAmp、OWMS 使用 LayerNorm，CNON 不使用。

## 讨论与局限

- \(K\) 是任务相关超参数，增加近邻数不保证梯度估计持续改善。
- 实验使用 MLP；作者指出 GradPIE 也可用于其他可训练代理结构，例如生成式模型。
- 在线局部采样采用简单高斯分布；作者提出可研究 Latin Hypercube Sampling（LHS）或由局部梯度信息引导的采样。
- 理论联系依赖可微性和局部近似条件；论文未给出对所有非光滑黑箱系统均成立的梯度对齐保证。
- 论文的主要对照是相同代理结构分别使用 MAE/MSE 训练的 Base-model，并在不同任务中与精确梯度、随机搜索或 BayOpt 作比较。

## 论文中的相关工作

- **Grathwohl et al. (2017), Backpropagation through the void：** 训练代理模型在反向传播阶段提供黑箱梯度估计，是本文代理梯度思路的背景。
- **Shirobokov et al. (2020), Black-box optimization with local generative surrogates（L-GSO）：** 在参数空间局部邻域迭代训练生成式代理近似模拟器；本文同样重视局部代理，但以 GradPIE 约束局部输出差值。
- **Trabucco et al. (2021), Conservative objective models（COMs）：** 通过惩罚分布外输入的过度预测改善离线模型驱动优化；本文聚焦主动优化中代理梯度的局部对齐。

## 核心结论

本文的核心贡献是将代理训练目标从点值拟合改为局部点对差值拟合，并通过梯度定理说明该目标在局部条件下与雅可比对齐相关。KNN 使该损失能够在有限邻域内计算；离线流程使用固定数据训练，在线流程则沿优化路径持续查询并更新代理。三个工程任务的结果表明，GradPIE 在相同类型的代理梯度优化流程中改善了优化表现并降低了达到基线性能所需的黑箱查询预算。
