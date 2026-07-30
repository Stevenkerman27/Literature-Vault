---
title: "High-dimensional efficient global optimization using both random and supervised embeddings"
authors:
  - "Rémy Priem"
  - "Youssef Diouane"
  - "Nathalie Bartoli"
  - "Sylvain Dubreuil"
  - "Paul Saves"
year: 2023
venue: "AIAA AVIATION 2023 Forum"
doi: "10.2514/6.2023-4448"
source: "/mnt/data/Priem 等 - 2023 - High-dimensional efficient global optimization using both random and supervised embeddings.pdf"
aliases:
  - "EGORSE"
tags:
  - paper
  - Bayesian-optimization
  - high-dimensional-optimization
  - random-embedding
  - supervised-embedding
  - PLS
---

> [!abstract] 摘要翻译
> 贝叶斯优化（Bayesian Optimization，BO）是求解昂贵黑箱优化问题的强大策略之一，但传统 BO 受维数灾难影响，通常只适用于低维问题。本文面向高维优化，引入低维线性嵌入子空间，并在优化过程中自适应学习这些嵌入。提出的方法称为 EGORSE（Efficient Global Optimization coupled with Random and Supervised Embedding，结合随机与监督嵌入的高效全局优化），以自适应方式组合随机线性嵌入和监督线性嵌入。作者在设计变量数为 10–600 的学术算例上，将 EGORSE 与多种先进算法比较。结果表明，EGORSE 在 CPU 时间和昂贵黑箱调用次数两方面均具有解决高维黑箱优化问题的潜力。

## 一句话总结

EGORSE 将高维黑箱函数投影到多个低维子空间中，用监督降维方向提高搜索针对性、用随机方向保留全局探索，再把“投影超立方体中部分点无法映回原空间”的几何问题改写成约束贝叶斯优化，从而显著减少反向映射二次规划的调用次数。

## 问题与动机

目标问题是昂贵、无梯度、仅可查询函数值的高维黑箱优化：

$$
\min_{\mathbf{x}\in\Omega} f(\mathbf{x}),\qquad \Omega=[-1,1]^d,\quad d\gg 20.
$$

论文以多学科设计分析与优化（Multidisciplinary Design Analysis and Optimization，MDAO）为背景：高保真气动、推进和结构模型单次评估昂贵，且常被视作黑箱。标准贝叶斯优化依赖高斯过程（Gaussian Process，GP）代理模型，但高维下存在两类瓶颈：

1. GP 核函数的超参数数量及最大似然估计成本随维数增长；高维空间中样本距离也更容易失去区分度。
2. 采集函数的全局优化本身成为高维、非凸问题，计算代价迅速上升。

现有高维贝叶斯优化（High-Dimensional Bayesian Optimization，HDBO）大体有两条路线：修改 GP 结构，或先降维再做 BO。本文选择第二条路线，但重点解决随机嵌入方法中的两个实际问题：低维搜索边界难以定义，以及候选点映回原空间时需要频繁求解二次规划。

## 核心假设

论文假设目标函数具有较低的**有效维数** $d_e\ll d$：存在矩阵 $\mathbf A\in\mathbb R^{d_e\times d}$ 和低维函数 $f_{\mathbf A}$，使得

$$
f(\mathbf x)=f_{\mathbf A}(\mathbf A\mathbf x).
$$

也就是说，虽然输入有 $d$ 个坐标，函数实际上只沿少数线性组合发生主要变化。EGORSE 的任务不是直接在 $d$ 维空间建模，而是不断构造和更新这些重要方向。

> [!note] 记号一致性
> 论文第 IV.E 节有一处把传递矩阵写成 $\mathbb R^{d\times d_e}$；但由 $\mathbf u=\mathbf A\mathbf x$、式（6）及算法 2 可知，主定义应为 $\mathbf A\in\mathbb R^{d_e\times d}$。本笔记按主公式使用该方向。

## 方法：EGORSE

### 1. 为每种降维方法构造低维子空间

设共有 $T$ 种降维方法 $\mathcal R^{(t)}$。第 $t$ 种方法生成传递矩阵

$$
\mathbf A^{(t)}\in\mathbb R^{d_e\times d},\qquad \mathbf u=\mathbf A^{(t)}\mathbf x.
$$

原始超立方体 $\Omega$ 投影后形成的真实可达集合为

$$
\mathcal A^{(t)}=\{\mathbf u=\mathbf A^{(t)}\mathbf x:\mathbf x\in\Omega\}.
$$

$\mathcal A^{(t)}$ 通常是一个多面体或 zonotope，而不是标准 BO 方便处理的超立方体。因此作者定义包围它的最小轴对齐超立方体 $\mathcal B^{(t)}$。第 $j$ 个低维坐标的范围是

$$
\left[-\sum_{i=1}^{d}|A^{(t)}_{j,i}|,\ \sum_{i=1}^{d}|A^{(t)}_{j,i}|\right].
$$

![[assets/High-dimensional efficient global optimization using both random and supervised embeddings/论文中图1.png]]

**论文中图 1**：白色区域是实际可达集合 $\mathcal A^{(t)}$；黑色区域属于包围盒 $\mathcal B^{(t)}$，但不存在满足 $\mathbf A^{(t)}\mathbf x=\mathbf u$ 且 $\mathbf x\in\Omega$ 的原空间点。

### 2. 两种反向映射

对可达点 $\mathbf u\in\mathcal A^{(t)}$，使用带等式约束的反向映射：

$$
\gamma_B^{(t)}(\mathbf u)=
\arg\min_{\mathbf x\in\Omega}
\left\|\mathbf x-[\mathbf A^{(t)}]^+\mathbf u\right\|_2^2
\quad\text{s.t.}\quad
\mathbf A^{(t)}\mathbf x=\mathbf u,
$$

其中 $[\mathbf A^{(t)}]^+$ 是 Moore–Penrose 伪逆。该问题是二次规划，只在 $\mathbf u\in\mathcal A^{(t)}$ 时可行。

为了让目标函数在整个包围盒 $\mathcal B^{(t)}$ 上都有定义，作者还使用无等式约束的投影：

$$
\gamma_W^{(t)}(\mathbf u)\in
\arg\min_{\mathbf x\in\Omega}
\left\|\mathbf x-[\mathbf A^{(t)}]^+\mathbf u\right\|_2^2.
$$

该映射对所有 $\mathbf u\in\mathcal B^{(t)}$ 都存在，但不同的 $\mathbf u$ 可能映到同一个 $\mathbf x$。

据此定义低维目标：

$$
f^{(t)}(\mathbf u)=
\begin{cases}
f\!\left(\gamma_B^{(t)}(\mathbf u)\right), & \mathbf u\in\mathcal A^{(t)},\\
f\!\left(\gamma_W^{(t)}(\mathbf u)\right), & \mathbf u\notin\mathcal A^{(t)}.
\end{cases}
$$

### 3. 将“不可映回”改写为约束

作者构造约束 $g^{(t)}(\mathbf u)\ge 0$ 来表示低维点是否属于真实可达集合。其思想是：

- 可达区内，用映回原空间后的范数衡量离原始超立方体边界的余量；
- 不可达区内，约束值为负，并随点靠近可达区而趋近于 0；
- 约束归一化到约 $[-1,1]$，平衡可行与不可行样本的建模尺度。

最终在标准超立方体 $\mathcal B^{(t)}$ 中求解约束问题：

$$
\min_{\mathbf u\in\mathcal B^{(t)}} f^{(t)}(\mathbf u)
\quad\text{s.t.}\quad g^{(t)}(\mathbf u)\ge 0.
$$

![[assets/High-dimensional efficient global optimization using both random and supervised embeddings/论文中图3.png]]

**论文中图 3**：左图是扩展后的低维目标及不可行区域；右图是约束 BO 在当前 GP 模型下构造的优化子问题。绿色星号是下一次真实评估点。

#### 相比 RREMBO 的计算优势

RREMBO 在每一次采集函数评估时，都要解二次规划判断候选点是否可达。采集函数优化通常需要成百上千次内部评估，因此该成本随原始维数 $d$ 显著增长。

EGORSE 把可达性作为一个由 GP 学习的约束。二次规划只在约束 BO 最终选出新候选点、准备调用真实目标函数时求解，而不是在采集函数的每次内部评估中求解。这是论文 CPU 时间优势的主要来源。

### 4. 监督嵌入与随机嵌入并用

EGORSE 可组合多种降维器：

- **随机高斯嵌入**：矩阵元素随机生成，不依赖当前数据。优点是方向无偏，能够探索监督模型尚未发现的区域。
- **哈希嵌入**：用稀疏哈希矩阵把原始坐标随机分配到低维坐标，计算更快。
- **偏最小二乘回归**（Partial Least Squares，PLS）：利用输入样本和目标值，寻找与输出协方差最大的正交方向。
- **边缘高斯过程方法**（Marginal Gaussian Process，MGP）：对嵌入矩阵本身设置概率模型，并用拉普拉斯近似估计其后验；EGORSE 使用后验局部最大值 $\hat{\mathbf A}$ 作为传递矩阵。

PLS 提供“当前数据认为最重要”的方向，随机高斯嵌入补充未被观测数据支持的探索方向。实验中最佳配置是 **PLS + Gaussian**。

### 5. 自适应重学子空间

EGORSE 的外层循环如下：

1. 对每种降维方法构建 $\mathbf A^{(t)}$；监督方法使用当前所有已评估样本。
2. 计算包围盒 $\mathcal B^{(t)}$。
3. 构造 $f^{(t)}$ 与 $g^{(t)}$。
4. 在该子空间内运行固定预算的约束 BO。
5. 合并所有子空间产生的新真实评估点，更新全局实验设计（Design of Experiments，DoE）。
6. 下一轮用扩充后的数据重新学习监督子空间。

来自不同子空间的点被共同加入 DoE，因此 PLS 不会只在自身先前选择的方向上自我强化；随机子空间提供的样本可以纠正监督方向。

![[assets/High-dimensional efficient global optimization using both random and supervised embeddings/论文中图4.png]]

**论文中图 4**：扩展设计结构矩阵（eXtended Design Structure Matrix，XDSM）展示 EGORSE 的数据流：构建嵌入矩阵、计算低维边界、构造目标和约束、运行约束 BO、评估并回填全局 DoE。

## 贝叶斯优化子程序与实现细节

每个低维子问题由约束贝叶斯优化（Constrained Bayesian Optimization，CBO）求解：

- 代理模型：高斯过程，由 SMT（Surrogate Modeling Toolbox，代理建模工具箱）构建。
- 采集函数：期望改进（Expected Improvement，EI）。EI 综合 GP 均值和标准差，在利用当前低预测值与探索高不确定性之间折中。
- 采集函数优化：先用 ISRES（Improved Stochastic Ranking Evolution Strategy，改进随机排序进化策略）搜索多峰全局区域，再用 SNOPT（Sparse Nonlinear OPTimizer，稀疏非线性优化器）做基于梯度的局部精修。
- 反向映射二次规划：用 CVXOPT（凸优化工具箱）求解；状态为 `optimal` 时认为 $\mathbf u\in\mathcal A^{(t)}$。
- 约束 BO 实现：SEGO/SEGOMOE。原文未展开 SEGOMOE 的全称，将其作为作者团队的约束全局优化工具箱使用。

## 实验设置

### 实验 1：超参数敏感性分析

#### 测试问题

作者从二维 Modified Branin（MB）函数出发，把输入归一化到 $[-1,1]^2$，再生成随机矩阵 $\mathbf A_d\in\mathbb R^{2\times d}$：

$$
\mathrm{MB}_d(\mathbf x)=f_1(\mathbf A_d\mathbf x),\qquad \mathbf x\in[-1,1]^d.
$$

测试 $d=10$ 和 $d=100$，真实有效维数均为 2；已知全局最优值约为 1.1。

#### 比较的 EGORSE 变体

1. Gaussian：随机高斯矩阵。
2. Hash：随机哈希矩阵。
3. PLS：仅监督 PLS。
4. PLS + Gaussian。
5. MGP：仅监督 MGP。
6. MGP + Gaussian。

#### 预算与统计

- 每个问题、每个变体独立运行 10 次。
- 初始 DoE 大小测试 5、$d$、$2d$ 三种。
- 设定有效维数 $d_e=2$。
- 每次子空间优化最多评估 $20d_e=40$ 个点。
- 双降维器版本运行 10 个外层迭代；单降维器版本把外层迭代翻倍，使总真实评估预算统一为约 800 次。
- 曲线展示 10 次运行中“当前最佳可行目标值”的均值；标准差带在图中缩小为原来的四分之一以提高可读性。

#### 结果

- 在 MB_10 上，六种版本差异较小。初始 DoE 只有 5 点时 Gaussian 略优；DoE 增大后 PLS + Gaussian 略优。
- 在 MB_100 上差异明显，PLS + Gaussian 在三种初始 DoE 大小下都表现出最佳的收敛速度与稳健性折中。
- 对 PLS + Gaussian 而言，初始 DoE 取 $d$ 点优于 5 点和 $2d$ 点：5 点不足以稳定识别方向；$2d$ 点则在优化开始前消耗了过多预算。

![[assets/High-dimensional efficient global optimization using both random and supervised embeddings/论文中图6.png]]

**论文中图 6**：MB_100 上的敏感性分析。右下图显示，PLS + Gaussian 配合 $d$ 点初始 DoE 的整体表现最好。

### 实验 2：与 HDBO 基线比较

#### 基线与配置

| 方法 | 核心机制 | 实验配置 |
|---|---|---|
| EGORSE | PLS + Gaussian，自适应子空间 | 初始 DoE 为 $d$ 点；沿用敏感性实验预算 |
| TuRBO | Trust-Region Bayesian Optimization，信赖域贝叶斯优化 | 5 个信赖域；800 次评估；内部生成 $d$ 个初始点；EI |
| EGO-KPLS | Efficient Global Optimization with Kriging–PLS | 800 次评估；与 EGORSE 相同初始 DoE；2 个 KPLS 主成分；EI |
| RREMBO | 改进低维域与反向映射的随机嵌入 BO | 20 次子空间优化串联；每次 $20d_e$ 次评估；$d_e=2$；EI |
| HESBO | Hashing-Enhanced Subspace Bayesian Optimization，哈希增强子空间 BO | 与 RREMBO 相同预算；$d_e=2$；EI |

每个问题、每种算法独立运行 10 次，同时比较按真实评估次数和按 CPU 时间的收敛曲线。

#### 结果解读

- **MB_10，按评估次数**：TuRBO 和 EGO-KPLS 收敛最快；EGORSE、RREMBO、HESBO 差异较小。
- **MB_100，按评估次数**：EGO-KPLS 最快达到较低目标值，TuRBO 次之；EGORSE 在三种嵌入方法中结果最好。
- **MB_100，按 CPU 时间**：RREMBO、TuRBO、EGO-KPLS 完整运行均超过 8 小时，而 EGORSE 的候选点搜索明显更快。EGORSE 在与 HESBO 相近的时间内获得更低目标值。
- 因而 EGORSE 的主要优势不是“单位黑箱调用总是最优”，而是**内部优化开销随维数增长更慢**。若真实黑箱极昂贵，调用次数可能更重要；若代理和采集优化本身已成为瓶颈，EGORSE 更有吸引力。

![[assets/High-dimensional efficient global optimization using both random and supervised embeddings/论文中图7.png]]

**论文中图 7**：上排按真实评估次数比较，下排按 CPU 时间比较。MB_100 中，EGO-KPLS/TuRBO 的样本效率更好，但 EGORSE 的运行时间优势更明显。

### 实验 3：600 维 Rover 路径规划

#### 问题构造

基础 Rover_60 问题用 30 个二维控制点定义样条轨迹，共 60 个变量。机器人要从起点到终点，目标函数鼓励短路径并惩罚穿越障碍；已知最优值为 $-5$。

作者将 Rover_60 归一化后再嵌入 600 维：

$$
\mathrm{Rover}_{600}(\mathbf x)=\mathrm{Rover}_{60}(\mathbf A_{600}\mathbf x),
\qquad \mathbf A_{600}\in\mathbb R^{60\times 600}.
$$

因此该问题的真实有效维数是 60，而算法仍设 $d_e=2$。

![[assets/High-dimensional efficient global optimization using both random and supervised embeddings/论文中图8.png]]

**论文中图 8**：Rover 路径规划示例。绿色样条从起点到终点，需尽量避开禁止区域。

#### 配置与结果

- 由于作者认为 TuRBO、EGO-KPLS、RREMBO 在 100 维以上耗时不可接受，600 维实验只比较 EGORSE PLS + Gaussian 与 HESBO。
- 初始 DoE 为 $d=600$ 点。
- 每个算法使用 200 次子空间优化，每次 $20d_e=40$ 次评估，即后续预算约 8000 次。
- EGORSE 最终平均目标值略低于 HESBO，标准差也较小；HESBO 运行稍快，但时间差不显著。
- 两者都远未达到已知最优值 $-5$。作者给出的原因是：算法只搜索 2 个有效方向，而真实有效维数为 60；同时 PLS 是全局线性方法，会抹去可能包含全局最优点的局部变化。

![[assets/High-dimensional efficient global optimization using both random and supervised embeddings/论文中图9.png]]

**论文中图 9**：600 维 Rover 上的时间与评估次数曲线。EGORSE 略优于 HESBO，但最终目标值约为 2.39，说明严重的有效维数失配仍未解决。

## 关键贡献

1. **新的低维约束优化表述**：把包围盒中“不可映回原空间”的区域显式建模为约束，避免在采集函数每次内部评估时求解二次规划。
2. **随机与监督嵌入结合**：监督方向提高样本效率，随机方向避免过早锁定错误子空间，并保留更广泛探索。
3. **子空间自适应更新**：每轮使用所有真实评估点重新学习监督方向，不把嵌入固定在初始 DoE 上。
4. **较好的计算可扩展性**：在 100 维测试中，EGORSE 的 CPU 时间显著低于 RREMBO、TuRBO 和 EGO-KPLS；在 600 维测试中仍可运行。

## 局限与启发

### 局限

- **依赖低有效维数假设**：当 $d_e$ 设得远小于真实有效维数时，算法无法覆盖函数主要变化，Rover_600 是直接反例。
- **$d_e$ 需要人工指定**：论文固定取 $d_e=2$，没有自适应估计机制。
- **监督降维是全局线性的**：PLS 依据全局协方差找方向，可能忽略只在局部区域出现的重要变化；随机嵌入只能部分缓解。
- **多数基准是人工升维**：MB_10 和 MB_100 本质上都来自二维函数，天然满足方法假设，不能完全代表真实高维工程问题。
- **样本效率并非始终最佳**：MB_100 按评估次数比较时，EGO-KPLS 和 TuRBO 明显优于 EGORSE；论文“全面优于先进方法”的结论应主要理解为 CPU 可扩展性优势。
- **比较条件不完全统一**：TuRBO 不能使用相同初始 DoE，各算法来自不同工具箱和内部优化器，CPU 时间会同时受到实现质量影响。
- **尚未处理原始昂贵约束**：本文把嵌入可达性转成约束，但原始问题本身仍是无约束黑箱；工程约束扩展被列为未来工作。

### 启发

- 高维 BO 的瓶颈不只有真实函数调用；采集函数优化、GP 训练和几何映射也可能主导总时间。
- “监督方向 + 随机方向”是一种实用的偏差—探索折中：前者利用数据，后者防止监督模型在早期小样本下自信过度。
- 对真实问题，更合理的下一步是联合学习**有效维数、局部子空间和信赖域**，而不是只增加固定全局线性方向。
- 在黑箱极其昂贵的场景，应分别报告样本效率与墙钟时间；EGORSE 在二者上的排序不同。

## 关键概念补充（外部资料）

### 有效维数与随机嵌入

REMBO 的理论出发点是：若函数只依赖某个低维线性子空间，那么随机低维嵌入以较高概率仍能包含一个原问题最优解。随机矩阵并不识别真实方向，而是依靠多次随机尝试覆盖它。EGORSE 继承这一全局探索逻辑，并加入监督方向以加速收敛。[^rembo]

### PLS 与 PCA 的区别

主成分分析（Principal Component Analysis，PCA）只寻找输入方差最大的方向，完全不使用目标值。PLS 则寻找输入投影与输出之间协方差最大的方向，因此更适合“找出哪些变量组合最影响目标”的监督降维。代价是：小样本、噪声或非线性局部结构可能让该方向不稳定。[^pls]

### MGP 的含义

MGP 不把嵌入矩阵当作一个固定待优化参数，而是把矩阵元素视为随机变量，在观测数据条件下估计后验分布。原方法通过拉普拉斯近似对嵌入不确定性进行近似边缘化，以减少嵌入超参数误设的影响；本文最终取后验局部最大值 $\hat{\mathbf A}$ 作为子空间。[^mgp]

### TuRBO 的对照意义

TuRBO 不显式假设全局低维线性子空间，而是在原空间中维护一个或多个自适应信赖域，每个区域拟合局部 GP，并根据连续成功或失败扩张、收缩区域。它通常具有较好的样本效率，但在本文实现中，高维采集优化和多个局部模型带来较高 CPU 成本。[^turbo]

### HESBO 的对照意义

HESBO 使用极稀疏的哈希嵌入：每个原始维度随机映射到一个低维坐标并乘以随机符号。投影和反向计算非常快，但哈希碰撞会把多个重要方向混合在一起。[^hesbo]

## 相关工作脉络（约 3 篇核心文献）

1. **Wang et al., REMBO**：首次系统提出用随机线性嵌入把超高维 BO 转到低维空间，并给出低有效维数下的理论保证。它奠定了本文“随机子空间保留全局发现概率”的基础。[^rembo]
2. **Binois et al., RREMBO/低维域选择**：指出随机嵌入后低维搜索域的边界并不简单，研究最小低维集合和反向映射。EGORSE 直接沿用其 $\gamma_B$ 映射，但通过约束 BO 减少可达性二次规划的调用频率。[^rrembo]
3. **Garnett et al., Active Learning of Linear Embeddings**：把嵌入矩阵纳入 GP 的概率建模并主动学习低维结构，代表监督嵌入路线。EGORSE 的 MGP 版本继承该方法，而性能最好的 PLS 版本则采用更简单的监督方向估计。[^mgp]

## 关联

- [[Bayesian Optimization]]
- [[Gaussian Process]]
- [[Expected Improvement]]
- [[Constrained Bayesian Optimization]]
- [[High-Dimensional Bayesian Optimization]]
- [[Random Embedding]]
- [[Partial Least Squares]]
- [[Trust Region]]
- [[Multidisciplinary Design Analysis and Optimization]]

## 复现清单

- 原始域统一归一化为 $[-1,1]^d$。
- 选择 $d_e$、初始 DoE 大小、外层迭代数、每个子空间的真实评估预算。
- 实现至少两类嵌入：PLS 与随机高斯矩阵。
- 按式（6）计算 $\mathcal B^{(t)}$。
- 用带等式约束二次规划实现 $\gamma_B$，用盒约束最小二乘实现 $\gamma_W$。
- 构造式（9）的可达性约束与式（11）的扩展目标。
- 在每个子空间运行带 EI 的约束 BO，并只在最终候选点处调用反向映射和真实函数。
- 将所有子空间的新样本合并，重新训练 PLS，重复外层循环。
- 同时记录真实函数调用数、采集优化时间、GP 训练时间和总墙钟时间。

[^rembo]: Wang, Z. et al. *Bayesian Optimization in a Billion Dimensions via Random Embeddings*. Journal of Artificial Intelligence Research, 55, 2016. DOI: 10.1613/jair.4806.
[^rrembo]: Binois, M., Ginsbourger, D., Roustant, O. *On the Choice of the Low-Dimensional Domain for Global Optimization via Random Embeddings*. Journal of Global Optimization, 76, 2020, 69–90.
[^mgp]: Garnett, R., Osborne, M. A., Hennig, P. *Active Learning of Linear Embeddings for Gaussian Processes*. UAI 2014, 230–239. arXiv:1310.6740.
[^pls]: Helland, I. S. *On the Structure of Partial Least Squares Regression*. Communications in Statistics—Simulation and Computation, 17(2), 1988, 581–607. DOI: 10.1080/03610918808812681.
[^turbo]: Eriksson, D. et al. *Scalable Global Optimization via Local Bayesian Optimization*. NeurIPS 2019.
[^hesbo]: Nayebi, A., Munteanu, A., Poloczek, M. *A Framework for Bayesian Optimization in Embedded Subspaces*. ICML 2019, PMLR 97:4752–4761.
