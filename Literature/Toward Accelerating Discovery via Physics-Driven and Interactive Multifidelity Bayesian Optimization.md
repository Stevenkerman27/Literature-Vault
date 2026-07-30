---
title: "Toward Accelerating Discovery via Physics-Driven and Interactive Multifidelity Bayesian Optimization"
authors: [Arpan Biswas, Mani Valleti, Rama Vasudevan, Maxim Ziatdinov, Sergei V. Kalinin]
year: 2024
venue: "Journal of Computing and Information Science in Engineering"
doi: "10.1115/1.4066856"
tags: [paper, bayesian-optimization, multifidelity, physics-informed, human-in-the-loop, active-learning, materials-discovery]
status: read
source: "D:\\3D\\Projects\\Papers\\storage\\Q9RVWCTY\\Biswas 等 - 2024 - Toward Accelerating Discovery via Physics-Driven and Interactive Multifidelity Bayesian Optimization.pdf"
---

# Toward Accelerating Discovery via Physics-Driven and Interactive Multifidelity Bayesian Optimization

> [!abstract] 摘要直译
> 计算和实验材料发现都需要探索多维且常常不可微的参数空间，例如具有多重相互作用的哈密顿量相图、组合材料库的成分空间、工艺空间和分子嵌入空间。单次评估往往昂贵或耗时，因此穷举网格搜索或随机搜索的数据需求过高。由此，贝叶斯优化（Bayesian Optimization，BO）等主动学习方法受到广泛关注：它围绕人类的学习（发现）目标进行自适应探索。然而，经典 BO 依赖预先定义的优化目标，且平衡探索与利用的策略完全由数据驱动。实际问题中，领域专家能够以部分已知物理规律的形式给出系统先验，而实验过程中的探索策略也常会变化。本文提出一个建立在多保真贝叶斯优化（Multifidelity Bayesian Optimization，MFBO）之上的交互式工作流：先使用经典的数据驱动 MFBO，再扩展到所提出的、物理驱动的结构化 MFBO（structured MFBO，sMFBO），最后扩展为允许人参与的交互式 MFBO（interactive MFBO，iMFBO），以实现自适应且与领域专家目标一致的探索。作者用 Ising 模型生成的高度非光滑多保真模拟数据演示这些方法：自旋-自旋相互作用为参数空间，晶格尺寸为保真度空间，目标为最大化热容。详尽比较显示，物理知识注入和实时人工决策能够改善探索并提高其与真实规律的一致性。配套笔记本可复现实验并应用于其他系统。

## 问题与动机

在材料发现、相图计算或自动化实验中，高保真评价通常很慢，直接进行网格搜索会把预算耗在明显不重要的区域。BO 以高斯过程（Gaussian Process，GP）拟合未知黑箱响应，并以采集函数选择下一次实验，从而在预测均值较高的区域利用、在不确定性大的区域探索。

但经典 BO 有两个不符合真实科研流程的假设。第一，它只从已经采到的数据学习；事实上，研究者往往已经知道一些不完整但有价值的物理结构，例如响应函数有一个相变峰、存在不连续跃迁或某些区域应被避开。第二，采集函数、保真度成本和收敛条件被预先固定；实际实验中，专家会根据中途观察到的现象调整目标或策略。

MFBO 将昂贵的高保真观测与便宜但近似的低保真观测放入同一个 GP。它可以先用低保真模型粗略扫描，再在有希望的区域执行高保真测量。不过，低保真模型可能遗漏关键物理或甚至对应错误的系统；此时，纯数据驱动的跨保真相关性会把搜索带偏。本文的重点是将两类额外信息显式纳入闭环：**可概率化的物理先验**与**按需触发的专家策略干预**。

## 方法总览

![[assets/physics-driven-interactive-multifidelity-bayesian-optimization/imfbo-workflow-p005.png]]

*图 1。iMFBO 工作流：专家设定目标和初始策略，算法依据多保真观测循环更新；在策略变更触发后，专家可调整参数域、代理模型、保真度成本比和停止条件。*

### 1. 经典 MFBO：联合建模“位置”和“保真度”

每个观测写为 $(x_i,f_i,y_i)$：$x_i$ 是设计变量，$f_i\in\{0,1\}$ 表示低、高保真度，$y_i$ 是响应。多保真高斯过程（Multifidelity Gaussian Process，MFGP）的协方差由设计空间核与保真度核相乘：

$$
\operatorname{cov}((x_i,f_i),(x_j,f_j))=\sigma^2 R(x_i,x_j)K_F(f_i,f_j),\qquad
K_F(f_i,f_j)=\exp(-\delta|f_i-f_j|).
$$

其中 $R$ 采用径向基函数（Radial Basis Function，RBF）核或 Matérn 核，$\theta$ 控制每个设计维度的相关长度，$\delta$ 学习高低保真之间的差异。超参数通过马尔可夫链蒙特卡洛（Markov Chain Monte Carlo，MCMC）后验估计。

与单保真 BO 只挑选位置不同，MFBO 的采集函数同时选择 $(x,f)$。高保真候选使用按成本比 $C$ 缩放的期望改进（Expected Improvement，EI）；低保真候选使用其 EI 与高保真 EI 的差值，衡量“在此处补一个低保真点能改变多少高保真决策”。每轮从两类候选中取采集值最大者。因此 $C$ 是关键的策略旋钮：调高会更偏向便宜的低保真采样，调低则更愿意购买高保真信息。

### 2. sMFBO：将物理知识作为 GP 的均值函数

标准 MFGP 常以零均值作为先验；sMFBO 保留数据驱动核，却将均值替换成由领域知识给出的概率模型 $M_f(x;\eta)$。未知物理参数 $\eta$ 与核超参数一起由 MCMC 从数据更新。于是预测可理解为“物理结构的先验趋势 + GP 对偏差的修正”，而不是把物理偏好硬塞进采集函数。

这种表述特别适合“知道现象存在、但不知道精确位置和强度”的场景。论文在一维不连续测试函数中，规定均值在某个未知断点 $c$ 两侧允许不同的偏移量，并给 $a,b,c$ 设先验。即便给入的平滑函数形状不完全正确，模型仍可通过后续高、低保真观测修正；而标准 MFBO 因低保真函数没有断点，会倾向在错误的低保真峰附近反复利用。

在 Ising 问题中，作者将“热容在相变附近单峰”的知识写为带未知峰高、峰位和宽度的高斯型均值函数。这个设计的关键不是声称热容必然精确服从高斯曲线，而是让模型在数据稀缺、非光滑区域尚未被采到时，也能优先把高保真预算导向可能的相变区域。

### 3. iMFBO：把策略修改设计成受控的人机闭环

iMFBO 在每轮更新后监视当前最优值。若连续若干步没有改进，系统向专家发出是否变更策略的提示；专家可拒绝，或选择一项/多项修改：

- **参数空间**：缩小到新关注区，或排除已识别但敏感、无意义的区域。
- **代理模型**：在纯数据驱动 MFGP 与物理驱动结构化 MFGP（structured MFGP，sMFGP）之间切换；也可扩展为多个代理模型的集成。
- **采集函数的保真度成本比**：改变 $C$，重分配低、高保真预算。
- **收敛条件**：增减总迭代次数，或在信息已足够时提前结束。

论文的一维演示说明了这不是让人手动挑下一个点。算法仍以采集函数选点；人只在模型已学到一个不连续区域后，指示系统避开该区、切换回标准 MFGP 并降低成本比，以转向另一个更有价值的最优区域。面对错误先验，交互式流程在第 16 次迭代发现真实最优点，并在第 22 次因采集值已很小而收敛。

## 实验设置

### 合成测试：为什么经典 MFBO 可能失效

作者先使用两个一维、双保真测试函数。连续函数中，初始化 10 个点（7 个低保真、3 个高保真）均远离真实峰值，随后运行 15 轮。MFBO 先用低保真探索，在第 12 轮才靠近目标区，最终以 6 次高保真和 9 次低保真评价找到最优值。

第二个函数的高保真曲线在 $x=7.5$ 有不连续性，而低保真近似仍连续。相同的 15 轮预算下，经典 MFBO 做了 4 次高保真和 11 次低保真采样，却只得到稍有偏差的最优点。原因是低保真代理没有包含跃迁，且对称的保真度核把错误的低保真峰信息传给高保真预测。sMFBO 注入“可能有跃迁”的均值先验后，仅约 6 轮就找到目标区域；iMFBO 再利用专家对已识别断点的判断，将余下预算转去新的有效区域。

### Ising 相变搜索：保真度、成本与目标

实际案例以二维方格 Ising 模型为高保真系统，以自旋-自旋相互作用 $J$ 为输入、热容 $H_c$ 为输出，目标是找到 $H_c$ 最大的相变区域。高保真模型为 $60\times60$ 晶格，每次模拟约 8 分钟；低保真模型为 $20\times20$ 晶格，每次约 30 秒。两种模型均先运行 500 个蒙特卡洛步达到平衡，再以 500 步收集统计量，约化温度固定为 $T=2.7$。

每次多保真试验以 10 个随机初始样本开始（6 个低保真、4 个高保真），然后做 25 次自动探索；文中图 5-6 的非交互结果按总计 35 个后续多保真评价展示。成本比设为 $C=8.6$，来自高低保真计算时间之比及调节系数。GP 核的方差、长度尺度和保真度差异参数均取 $[0.01,1]$ 均匀先验；观测噪声先验根据 $J=1.17$ 处 20 次低保真重复模拟的标准差设为半正态分布。

作者还故意构造了错误低保真情形：高保真仍为方格模型，低保真改成三角晶格并使用 Kawasaki 动力学。后者虽然计算更便宜，却不再是目标系统的正确近似，用于检验物理先验是否能抑制错误的跨保真迁移。

## 结果与解读

| 方法与低保真来源 | 高保真预测均方误差（Mean Squared Error，MSE） | 相对单保真平均 MSE 的改进 | 说明 |
| --- | ---: | ---: | --- |
| 单保真 BO（5 次独立运行均值） | $1.6\times10^{-3}$ | 基线 | 以 8 次高保真运行匹配 iMFBO 的总成本，仍明显欠拟合。 |
| MFBO，方格低保真 | $0.8\times10^{-3}$ | 50% | 正确低保真可减少高保真需要，但非光滑峰区仍难拟合。 |
| MFBO，三角低保真 | $4.0\times10^{-3}$ | -150% | 错误低保真反而比单保真更差；34 次低保真、仅 1 次高保真，显示过度相信廉价代理。 |
| sMFBO，方格低保真 | $0.7\times10^{-3}$ | 56% | 单峰物理均值使模型更早聚焦相变区。 |
| sMFBO，三角低保真 | $0.5\times10^{-3}$ | 68% | 即使低保真模型错误，物理先验仍显著限制错误迁移。 |
| iMFBO，方格低保真 | $0.3\times10^{-3}$ | 81% | 最佳结果；4 次高保真、22 次低保真后，由专家触发最后一次高保真并停止。 |
| iMFBO，三角低保真 | $1.0\times10^{-3}$ | 37.5% | 初期使用标准 MFBO 已受错误低保真影响，后续切换 sMFGP 后仍能优于单保真。 |

非交互的正确低保真案例中，MFBO 共选 5 个高保真、30 个低保真点，MSE 为 $8\times10^{-4}$；sMFBO 则为 4 个高保真、31 个低保真点，MSE 降到 $7\times10^{-4}$。这表明增益并非来自多做昂贵计算，而来自先验改变了采样位置与拟合趋势。单保真 BO 在等成本的 8 次高保真样本下，五种随机初值的 MSE 均值为 $1.6\times10^{-3}$、标准差为 $0.89\times10^{-3}$；随机搜索与拉丁超立方采样（Latin Hypercube Sampling，LHS）也没有优于它。

## 关键贡献

1. 将 MFBO 从“数据 + 成本权衡”扩展为“数据 + 可学习物理先验 + 实时专家决策”的三层闭环。
2. 以概率均值函数而非确定性规则注入物理知识，使峰位、跃迁位置及先验强度能随数据共同后验更新。
3. 把人工角色限制为可审计的策略级选择，而不是人工替代优化器选点，保留了 BO 的自动实验闭环。
4. 在高度非光滑、且可能存在错误低保真模型的 Ising 任务上，量化表明 sMFBO/iMFBO 能避免纯 MFBO 的负迁移。

## 局限与工程启发

- **物理先验的形式仍需专家构建。** 论文展示的断点和单峰均值函数较简单。若先验形式与系统严重不符，尽管 GP 可以修正，早期采样仍可能受偏置影响；应把先验视作带不确定度的假设，而不是硬约束。
- **“交互”尚未评估人的成本与一致性。** 触发规则是连续若干步不改进，且案例中的策略切换由作者预设性地解释。实际部署需要明确谁有权限切换、何时记录理由，以及如何避免不同操作者给出不可复现的干预。
- **保真度关系被设为较简单的对称核。** 论文自己指出，非光滑区域会迫使核尺度变小并抬高全局不确定性；未来可用分区 GP、非平稳核或显式的非对称/因果跨保真模型。
- **多保真不天然优于单保真。** 当低保真模型结构错误时，经典 MFBO 的 MSE 恶化到单保真基线的 2.5 倍。工程实施前应小规模校准高低保真相关性，并保留足量高保真锚点，而不是仅按成本比大量采低保真数据。
- **对气动优化的可迁移意义。** 在 [[SURROOPT A GENERIC SURROGATE-BASED OPTIMIZATION CODE FOR AERODYNAMIC AND MULTIDISCIPLINARY DESIGN]] 一类工作流中，可将低分辨率 CFD、面板法或经验模型作为低保真，将高分辨率雷诺平均 Navier-Stokes（Reynolds-Averaged Navier-Stokes，RANS）作为高保真；单峰、激波位置范围、失速前后区域或已知设计规律可写为均值先验。前提是将这些规律参数化为可校正的概率模型，并让设计师仅在收敛停滞、模型失配或需求变化时修改策略。

## 关联

- 主题：[[Bayesian optimization]]、[[Multi-fidelity modeling]]、[[Physics-informed machine learning]]、[[Human-in-the-loop optimization]]、[[Autonomous experimentation]]
- 气动优化实现：[[SURROOPT A GENERIC SURROGATE-BASED OPTIMIZATION CODE FOR AERODYNAMIC AND MULTIDISCIPLINARY DESIGN]]、[[Data-driven surrogate model for aerodynamic design using separable shape tensor method]]
- 生成式设计与代理模型：[[DiffAirfoil An Efficient Novel Airfoil Sampler Based on Latent Space Diffusion Model for Aerodynamic Shape Optimization]]
