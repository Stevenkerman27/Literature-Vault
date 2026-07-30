---
title: "SURROOPT: A GENERIC SURROGATE-BASED OPTIMIZATION CODE FOR AERODYNAMIC AND MULTIDISCIPLINARY DESIGN"
authors: [Zhong-Hua Han]
year: 2016
venue: "30th Congress of the International Council of the Aeronautical Sciences (ICAS 2016)"
doi: ""
tags: [paper, surrogate-based-optimization, aerodynamic-optimization, multidisciplinary-design-optimization, kriging, infill-sampling]
status: read
source: "D:\\3D\\Projects\\Papers\\storage\\K8A9PBKH\\Han - SURROOPT A GENERIC SURROGATE-BASED OPTIMIZATION CODE FOR AERODYNAMIC AND MULTIDISCIPLINARY DESIGN.pdf"
---

# SURROOPT: A GENERIC SURROGATE-BASED OPTIMIZATION CODE FOR AERODYNAMIC AND MULTIDISCIPLINARY DESIGN

> [!abstract] 摘要直译
> 代理模型优化（SBO）是一类利用代理模型近似昂贵目标函数和约束函数的优化算法，并由此引导新样本点向最优解附近添加和评估。SBO 已被证明对常采用计算流体力学（CFD）等昂贵数值分析的工程设计问题非常有效。尽管 SBO 日益流行，但由于收敛性质仍不充分、所谓“维数灾难”带来的困难，以及作为通用优化算法时功能尚不完整，它很少被当作通用优化算法使用。在过去十年中，研究者持续推进 SBO 的发展，目标是构建能够求解具有平滑、连续设计空间的任意优化问题的高效全局优化算法。本文回顾了作者团队近期在 SBO 开发方面的进展，重点介绍通用优化代码 SurroOpt 的发展，以及其在气动和多学科设计优化中的最新应用。

## 问题与动机

飞行器外形和气动-结构联合设计通常将一次 CFD 或多学科分析（MDA）作为黑箱函数评估。直接使用遗传算法需要成千上万次昂贵计算；而局部梯度法容易受局部最优、噪声或非光滑约束限制。SBO 的基本思路是以少量真实分析样本训练廉价近似模型，在近似模型上反复做子优化，只把最有价值的新点交回高保真求解器。

论文的定位是工程化的软件框架，而不是一种新的单独采样准则。它将四类原本常以研究原型形式存在的选择组合起来：面向确定性数值实验的 DoE、可替换的代理模型、兼顾开发与探索的加点准则、以及可处理约束和多目标问题的低成本子优化。关键问题是如何将这些模块编排为可重启、可并行、可由用户接入任意求解器的通用循环。

## 方法

![[assets/surroopt-generic-surrogate-based-optimization/method-workflow-p006.png]]

*图 8，SurroOpt 的离线建模与在线优化流程。离线阶段由 DoE 驱动并行高保真求解；在线阶段反复拟合代理、选择加点准则、用 GA/BFGS/SQP 求解子问题、评估新样本并更新数据集。*

### 缩写与命名

下表集中定义文中使用的英文缩写；CFD（计算流体力学）按本库约定不再展开。

| 缩写 | 英文全称 | 本文中的含义 |
| --- | --- | --- |
| SBO | Surrogate-Based Optimization | 代理模型优化：用廉价近似模型引导少量昂贵真实评估。 |
| DoE | Design of Experiments | 试验设计：在真实求解预算有限时，选择能尽量均匀覆盖设计空间的初始样本。 |
| LHS / UD / MCD | Latin Hypercube Sampling / Uniform Design / Monte Carlo Design | 拉丁超立方采样 / 均匀设计 / 蒙特卡洛设计，SurroOpt 实现的三种初始 DoE 方法。 |
| MDA | Multidisciplinary Design Analysis | 多学科设计分析：一次完整的耦合气动-结构计算。 |
| GA / BFGS / SQP | Genetic Algorithm / Broyden-Fletcher-Goldfarb-Shanno / Sequential Quadratic Programming | 遗传算法 / 拟牛顿算法 / 序列二次规划，用于廉价代理模型上的子优化。 |
| PRSM / PCE | Polynomial Response Surface Model / Polynomial Chaos Expansion | 多项式响应面模型 / 多项式混沌展开。 |
| GEK / HK | Gradient-Enhanced Kriging / Hierarchical Kriging | 梯度增强 Kriging / 层次 Kriging。 |
| RBF / ANN / SVR | Radial Basis Function / Artificial Neural Network / Support Vector Regression | 径向基函数 / 人工神经网络 / 支持向量回归。 |
| EI / PI / MSE / LCB / TS / MSP | Expected Improvement / Probability of Improvement / Mean Squared Error / Lower Confidence Bound / Target Searching / Minimizing Surrogate Prediction | 期望改进 / 改进概率 / 均方误差 / 下置信界 / 目标搜索 / 最小化代理预测，六种加点准则。 |
| MPI | Message Passing Interface | 消息传递接口，用于多个样本或批量加点的并行计算。 |
| EGO | Efficient Global Optimization | 基于 EI 的经典高效全局优化流程，通常每次只选一个新样本。 |
| MDO | Multidisciplinary Design Optimization | 多学科设计优化。 |
| MDF / IDF / SAND / CO / CSSO | Multidisciplinary Design Feasible / Individual Design Feasible / Simultaneous Analysis and Design / Collaborative Optimization / Concurrent Subspace Optimization | 五种 MDO 架构，分别以不同方式处理学科耦合与一致性。 |
| NSGA-II | Non-dominated Sorting Genetic Algorithm II | 非支配排序遗传算法 II，一种多目标进化算法。 |
| RANS | Reynolds-Averaged Navier-Stokes | 雷诺平均 Navier-Stokes 方程，文中 CFD 的流动求解模型。 |

RAE 2822 与 ONERA M6 是标准气动基准构型名称：前者是翼型，后者是跨声速机翼；它们不是文中算法缩写。G9 与 TNK 是标准优化测试问题的标识：前者为强约束单目标问题，后者为双目标约束问题，均非需要展开的算法缩写。$Ma$ 和 $Re$ 分别表示 Mach number（马赫数）和 Reynolds number（雷诺数）；$C_l$、$C_d$、$C_m$ 分别为 lift coefficient（升力系数）、drag coefficient（阻力系数）、pitching-moment coefficient（俯仰力矩系数）。

### 统一的 SBO 闭环

针对带边界和不等式约束的问题 $\min_{\mathbf{x}} y(\mathbf{x})$，SurroOpt 的一轮迭代可分为：

1. 用 LHS、均匀设计（UD）或 Monte Carlo design（MCD）在设计空间生成初始样本，并由 CFD/MDA 等真实求解器计算目标和所有约束。
2. 在样本数据库上分别训练目标函数 $\hat y(\mathbf{x})$ 和约束 $\hat g_i(\mathbf{x})$ 的代理模型，并调节其超参数。
3. 以某个加点准则定义廉价的子优化问题。遗传算法、Hooke-Jeeves、BFGS 或 SQP 在代理模型上搜索候选点；它们的成本相较一次 CFD 可忽略。
4. 将候选点交给真实分析器，追加 $\{\mathbf{x},y,\mathbf{g}\}$ 至数据库，更新代理模型，然后检查终止条件。

该循环把“主优化”与“子优化”分开：主优化的预算由真实函数评估数决定，子优化则可以使用更激进的混合搜索。终止条件包括连续最佳点的位置/目标改变量、最佳点的代理误差、达到最大真实评估数，以及 EI 最大值足够小。程序保存数据库和状态，`restart=1` 可从中断处继续。

### 模块化的模型与采样

- **DoE。** DoE 是 *Design of Experiments*（试验设计）：不是为了直接找到最优解，而是在真实 CFD/MDA 预算有限时，先选取分布均匀、信息互补的样本来训练初始代理。代码实现 LHS、UD 和 MCD；它们针对无随机噪声的计算机试验，在有限预算下覆盖设计域，这与面向物理实验的全因子或中心复合设计不同。
- **代理模型。** 参数化模型包括二次响应面（PRSM）和多项式混沌（PCE）；非参数模型包括 Kriging、梯度增强 Kriging（GEK）、层次 Kriging（HK）、协同 Kriging、RBF、ANN、SVR。文中强调 Kriging，因为其预测均值和方差可直接服务于不确定性驱动的加点。
- **多保真接口。** 通过加性、乘性或混合 bridge function 组织 variable-fidelity model；因此低保真物理模型不只是预筛选器，也可进入代理建模层。
- **软件边界。** 用户接口负责读写目标、约束及可选梯度；主程序与外部 solver 解耦。Fortran、C/C++、MPI、Python 和 Matlab 均可作为周边接口，MPI 用于并行求解多个样本或一批加点。

### 加点准则：在“当前最好”和“未知区域”之间取舍

SurroOpt 没有固定一种准则，而是将准则作为可组合策略：

| 准则 | 子问题 | 作用与风险 |
| --- | --- | --- |
| MSP | 最小化 $\hat y(\mathbf{x})$ | 纯开发，快速跟随代理最优点，但若初期代理错误会陷入错误局部区域。 |
| EI | 最大化期望改进 | 结合均值与 Kriging 标准差，是开发/探索的折中。 |
| PI | 最大化改进概率 | 倾向最可能的改进，但不计改进幅度，探索性较弱。 |
| MSE | 最大化预测方差 | 纯探索，用于提高全局代理精度而非直接找最优。 |
| LCB | 最小化 $\hat y-A s$ | 参数 $A$ 控制折中；$A\to0$ 近似 MSP，$A$ 很大时趋近纯探索。 |
| TS | 相对预设目标值最大化 EI | 当工程师已有可达性能目标时，将搜索定向到该阈值。 |

对约束 $g_i(\mathbf{x})\le0$，代码将每个 Kriging 约束代理转为满足约束的概率 $P[G_i(\mathbf{x})\le0]$。例如 constrained EI 优化 $EI(\mathbf{x})\prod_i P[G_i(\mathbf{x})\le0]$：一个候选点即使预计能降低阻力，若可行概率很低也会被压低。该设计避免只在代理预测的硬可行区域内搜索；同时仍保留罚函数、SQP 等传统约束处理方式。

论文的实证结论是，单一 MSP、MSE 或 PI 在多峰 Rastrigin 问题上可能失败；将 MSP 与 EI（或 LCB）组合可兼顾局部收敛和全局发现。并行加点则让多台计算资源一次评估多个候选，而不必等待严格串行的 EGO 循环。

### 多目标与多学科扩展

多目标模式既可用线性权重转为单目标，也支持构建 Pareto 前沿。对于多学科优化，SurroOpt 可嵌入 MDF、IDF、SAND、CO 和 CSSO 等架构；其角色仍是替代昂贵的系统级或学科级响应，并把耦合分析产生的目标、约束回写至样本库。因而 MDO 架构负责耦合一致性，SurroOpt 负责在高代价响应上减少试算次数。

## 实验设置与结果

### 解析基准：验证收敛、约束和 Pareto 能力

- **二维 Rastrigin。** 仅用 4 个初始 DoE 点。EI、LCB、MSP+EI 与 MSP+LCB 能找到全局最优；MSP、MSE 和 PI 的单独使用会停在局部最优。这说明软件的“组合准则”不是装饰性功能，而是对多峰问题的稳健性来源。
- **G9 强约束问题。** 7 个变量、4 个约束，可行域仅约占设计域的 0.5%，初始样本为 14 个。MSP+EI 得到目标值 680.64287，接近公开参考值 680.63006，且四个约束均严格满足；直接 GA 需要约 21,000 次函数评估，而最接近的 SurroOpt 组合用约 1,009 次。这里的比较体现代理方法的主要优势是减少昂贵评估，而不是子优化本身更强。
- **TNK 双目标问题。** 30 次重复中，SurroOpt 给出 112 个 Pareto 解，收敛指标 $1.6978\times10^{-3}$、多样性 0.625964，使用 278 次函数评估；NSGA-II 给出相近质量的前沿却需 20,000 次评估。
- **Sellar MDO 基准。** 在 MDF、IDF、SAND 三种架构下均得到约 8.00286 的最优目标。与梯度法相比，SurroOpt 的系统级/学科级调用更少，例如 MDF 为 22 次 MDA、各学科 95 次，而梯度法为 32 次 MDA、各学科 284 次。

### 二维气动翼型

RAE 2822 的优化目标为最小阻力，约束为 $C_l\ge0.824$、$C_m\ge-0.092$ 与截面积不小于基准。工况为 $Ma=0.734$、$Re=6.5\times10^6$，RANS 网格为 $512\times256$。优化采用多轮搜索；最优阻力由基准约 195.00 counts 降至 104.29 counts，同时 $C_l=82.4$ counts、$C_m=-0.0880$、截面积 $0.07794$，满足全部约束。该案例验证约束概率与代理循环能在典型跨声速翼型任务中工作，但文中未给出与其他现代 SBO 实现的同预算对照。

### 三维 ONERA M6 机翼

机翼采用 30 个变量：4 个平面形状变量和两个控制截面共 26 个剖面形状变量。工况为 $Ma=0.8395$、$Re=11.72\times10^6$、攻角 $3.06^\circ$；最小化阻力，同时保持升力、根/梢最大相对厚度与翼面积。初始 DoE 样本没有可行点，但受可行概率调制的加点最终进入可行区。

在约 200 次 RANS 计算后，$C_d$ 从 0.01751 降至 0.01277，降低 27.07%；$C_l$ 几乎不变（0.23803 到 0.23814），根部厚度不变，梢部厚度增加 2.40%，面积增加 0.28%。论文还比较单点 EGO 与每轮同时取 4 点的并行加点，后者在相同累计计算量下收敛更快。优化后的压力云图和两个翼展截面上的压力分布均显示激波减弱。

### 气动-结构联合优化

高亚声速翼身布局在 10 km、$Ma=0.76$ 巡航，机翼面积 105 $m^2$。气动外形和结构布局由 23 个变量描述：巡航攻角、翼展方向线性扭转、后掠/收缩相关量以及 20 个蒙皮厚度变量。MDF 架构内每次高保真评估须进行气动-静力气动弹性松耦合迭代，直至气动力与结构变形均收敛。

用 LHS 生成 200 个初始 MDA 样本后，MSP+EI 再迭代 50 次，总计约 300 次 MDA。最优解将单翼重量从 2030.3 kg 降至 1311.1 kg；升力仍大于 54 t，升阻比从 27.56 变为 27.01（保持不低于 27），最大等效应力基本不变，最大变形从 0.947 m 增至 0.998 m，仍满足不超过 1 m 的约束。这说明该框架能够在高代价耦合分析中处理多个工程约束，但训练 23 维问题仍需要很大的 200 点初始投资。

## 关键贡献

1. 把代理模型优化抽象为可插拔的 DoE、代理、加点、子优化、约束和终止模块，并以用户接口连接外部工程分析器。
2. 将概率可行性乘入 EI、PI、MSE 等加点准则，使探索高性能区域的同时显式考虑约束不确定性。
3. 提供组合与并行加点机制；实验显示 MSP+EI 是比单一准则更稳健的默认选择。
4. 在解析、多目标、气动和气动-结构任务中演示同一框架，可跨 MDF/IDF/SAND 等 MDO 架构使用。

## 局限与启发

- **“通用”受光滑连续设计空间前提限制。** 论文明确面向连续、平滑的黑箱问题；离散拓扑、强非平稳响应、求解失败频繁的 CFD 或大量类别变量需要额外机制。
- **维数灾难尚未被根治。** 实证变量范围是 2 到 52。23 维 MDO 已需 200 个初始样本；对超过 100 个变量及大量约束，作者只将梯度增强和多保真代理列为未来方向。
- **基准比较的预算不完全一致。** G9 与 NSGA-II 的优势十分明显，但不同加点准则的真实评估数跨度很大；气动案例缺少与其他软件在相同 CFD 预算下的严格消融对比。
- **约束可行概率依赖代理校准。** 当约束模型在边界附近失真时，$EI\times P(\mathrm{feasible})$ 可能过度排斥可行边界，或错误接受不可行点。真实工程中仍需对高价值候选执行独立验证。
- **可复用的工程原则。** 不应把“选 Kriging”当作完整 SBO 方案。更关键的是：数据接口要保存可重启状态；加点必须有开发和探索的切换；子优化器应能在廉价代理上混合全局与局部搜索；并行资源应直接体现为批量加点而非仅并行单个 CFD。

## 关联

- 主题：[[Surrogate-based optimization]]、[[Aerodynamic shape optimization]]、[[Multidisciplinary design optimization]]、[[Kriging]]、[[Expected improvement]]
- 后续的几何表示与 SBO 结合：[[Data-driven surrogate model for aerodynamic design using separable shape tensor method]]
- 核心前作脉络：[[Efficient Global Optimization of Expensive Black-Box Functions]]、[[Recent Advances in Surrogate-Based Optimization]]、[[Metamodeling in Multidisciplinary Design Optimization How Far Have We Really Come]]
