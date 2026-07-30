---
title: "Aerodynamic shape optimization of airfoils across rarefied to continuum regimes using deep reinforcement learning"
authors:
  - Guangyu Liu
  - Ziqi Cui
  - Yifan Jiang
  - Linying Zhang
  - Jun Zhang
year: 2025
venue: "Physics of Fluids"
doi: "10.1063/5.0296585"
tags:
  - literature
  - aerodynamic-optimization
  - airfoil
  - deep-reinforcement-learning
  - rarefied-flow
  - multiscale-flow
status: read
source: "D:\\3D\\Projects\\Papers\\storage\\8YJ7LN5B\\Liu 等 - 2025 - Aerodynamic shape optimization of airfoils across rarefied to continuum regimes using deep reinforce.pdf"
---

> [!abstract] 摘要译文
> 可重复使用空天飞机和大气再入飞行器等跨越大高度范围运行的飞行器，会面临从连续介质到稀薄气体的复杂气动环境。传统气动设计通常仅针对单一流动区间优化，因此难以在如此宽广的工况范围内保持稳定性能。这一挑战要求设计框架能够在整个任务包线内适应变化的流动物理。为此，本文提出基于深度强化学习（Deep Reinforcement Learning, DRL）的气动优化框架 CrossAero：它直接在 Knudsen 数（Knudsen number, Kn）的连续设计空间上工作，无需人工选择离散设计点。该能力通过引入 Kn 的概率密度函数（probability density function, PDF）实现；该函数编码不同流动工况的相对重要性，并可按任务需求或各工况驻留时间定制。CrossAero 从完整 Kn 分布直接学习，在统一框架中捕捉跨流域气动效应。作者先用两个单一流域优化任务验证其在不同气动条件下的可靠性；随后为给定 Kn 分布优化固定几何翼型，相比基准翼型实现 5.9% 的阻力降低；最后训练出流动条件化翼型生成器，给定 Kn 即可快速生成近最优设计而无需重新训练。结果表明，该框架在跨流域气动设计中具有稳健性、效率和适应性。

## 问题与动机

高空再入或可重复使用空天飞行任务中，气体稀薄度随高度跨越多个数量级。以 Kn 为刻画，连续流通常为 `Kn < 0.001`，滑移流为 `0.001 < Kn < 0.1`，过渡流为 `0.1 < Kn < 10`，更高 Kn 则接近自由分子流。针对单个 Kn 找到的低阻翼型，在另一流域可能并非好解：低 Kn 时激波和波阻主导，尖锐细长外形有利；Kn 增大后粒子碰撞减少、摩擦阻力相对突出，较厚较短的外形反而可能更优。

常见多点优化把几个代表工况的目标加权，但设计点、权重和约束阈值依赖人工选取。概率化多点优化虽然可按运行概率赋权，仍会用高斯求积把连续分布压缩为有限节点。本文的核心主张是：把任务中的 Kn 视为连续随机变量，在训练时从其分布采样，让策略直接最小化全分布期望阻力；这同时可支持一副固定的折中翼型和按工况即时变形的翼型生成器。

## 方法

### 设计变量、目标与约束

翼型上下表面各以三次 B 样条表示。每条曲线有 9 个控制点，前缘和尾缘固定，余下 7 个控制点仅允许纵坐标变化，因此总设计维数为 14；横坐标采用余弦加密，以加强对前缘和尾缘敏感区域的控制。上表面变量限制为正、下表面限制为负，确保翼型不交叉。

给定翼型参数 $a$，优化目标为

$$
\min_a J = \mathbb{E}_{\mathrm{Kn}\sim P(\mathrm{Kn})}[D(a,\mathrm{Kn})].
$$

这里 $P(\mathrm{Kn})$ 由任务驻留时间或设计优先级给定，$D$ 为阻力。每次生成几何后，作者按 $c(a)=\sqrt{A_0/A(a)}$ 对全部坐标统一缩放，使面积恒等于基准 NACA（National Advisory Committee for Aeronautics）0012 翼型面积 $A_0$。因此几何变化不会通过任意改变尺度来获得不公平的阻力收益；每个 Kn 均相对基准翼型弦长定义，缩放后不重新计算来流 Kn。

### 跨流域气动环境

传统 Navier-Stokes-Fourier（NSF）连续介质模型在过渡和自由分子流中失效；直接模拟蒙特卡洛（Direct Simulation Monte Carlo, DSMC）虽理论上适用于全流域，但在连续流中必须使用小于平均自由程和碰撞时间的网格与步长，代价高。本文使用统一随机粒子（Unified Stochastic Particle, USP）方法及其 SPARTACUS 求解器：在一个时间步内统一分子输运与碰撞，可在保持跨流域精度的同时放宽 DSMC 的时空步长限制。USP 求得翼型阻力，构成强化学习环境的高保真反馈。

### CrossAero 的学习闭环

![[assets/crossaero/crossaero-overview-p007.png]]

每一个环境交互仅对应一个稳态工况：

1. 从指定的 $P(\mathrm{Kn})$ 采样状态 $s_t=\mathrm{Kn}$。
2. 策略网络以 Kn 为输入，输出 14 维翼型动作 $a_t$；随后执行固定面积缩放。
3. USP 计算阻力，奖励定义为 $r_t=-D(a_t,\mathrm{Kn})$，并保存元组 $(s_t,a_t,r_t)$。
4. 累积 $N_{step}$ 个样本后，使用近端策略优化（Proximal Policy Optimization, PPO）的裁剪目标更新策略网络；价值网络以均方误差（Mean Squared Error, MSE）拟合回报，并以优势 $A=R-V$ 指导策略改进。

问题没有跨时间的状态演化，作者把折扣因子设为 0，令每个稳态样本只由自身阻力决定。这里强化学习的实质不是序列控制，而是用 PPO 学习一个从工况到设计变量的随机策略；当 Kn 是固定值时，退化为单工况优化；当 $P$ 是离散 Dirac 加权和时，退化为传统多点问题。

## 实验设置与结果

所有算例均取马赫数（Mach number, Ma）`Ma = 2`，以 NACA 0012 为面积和特征长度参考。每个训练曲线统计 5 个独立随机种子运行的均值与 95% 置信区间（confidence interval, CI）。

### 单工况验证

- 连续流：`Kn = 0.0165`，对应雷诺数（Reynolds number, Re）`Re = 200`；从圆形翼型开始，训练 `80` 个 epoch、每个 `50` 步。初始钝头形成脱体弓形激波，优化后前缘形成两道斜激波，最大无量纲压力从 `6.5` 降至 `3.8`，阻力降低 `43.1%`；约第 45 个 epoch 后外形和阻力趋于收敛。
- 稀薄流：`Kn = 0.5`；从随机翼型开始，训练 `40 x 50` 步。所得双凸、前后缘尖锐的翼型与已有统一稀薄/连续流优化结果相近；最大压力由 `6.60` 降至 `4.35`，阻力降低 `17.7%`。
- 与遗传算法（Genetic Algorithm, GA）在 `Kn=0.5` 的对照中，两者都进行 2000 次评估、各运行 5 次。DRL 的最佳阻力系数为 `0.6043`，低于 GA 的 `0.6137`；平均最佳值为 `0.6052` 对 `0.6166`，达到最佳值的平均步数为 `1400.0` 对 `1478.8`。代价是略高的优化器中央处理器（central processing unit, CPU）时间，`0.196` 对 `0.165` core-hours。

### 固定几何的跨流域设计

任务 Kn 取对数正态（lognormal, LN）分布 $\mathrm{LN}(\ln 0.1,1.0)$，其中中位数为 `0.1`，覆盖从近连续流到稀薄流的宽范围。作者从连续流已训练策略迁移学习：前 30 个 epoch 各 50 步以快速探索，后 30 个 epoch 各 100 步以精修。最终翼型保留连续流最优解的细长轮廓，同时适度增厚以平衡稀薄流表现。

在 500 个按该分布抽取的测试流场上，跨流域翼型的分布加权阻力系数为 `0.4108`，优于连续流专用翼型的 `0.4200`、稀薄流专用翼型的 `0.4208` 和 NACA 0012 的 `0.4363`，相对基准降低 `5.9%`。在两个代表点，它相对 NACA 0012 的阻力也分别降低 `6.2%`（`Kn=0.0165`）和 `11.9%`（`Kn=0.5`），但不会超过各自单点专用设计，这是稳健折中的预期代价。以两个高斯求积节点近似同一任务时，阻力系数为 `0.4150`，略差于连续分布训练的 `0.4108`。

### 条件化翼型生成

最后，作者不再寻找一副固定几何，而让策略网络学习 $a^*(\mathrm{Kn})$。训练 Kn 范围为 `[0.005, 1.0]`，在对数空间均匀采样，且以自由来流动压归一化阻力，避免某一流域因数值尺度更大而主导学习。它从跨流域固定翼型策略迁移，训练 `60` 个 epoch：前 30 个各 80 步、后 30 个各 120 步。

在 `Kn=0.0165` 和 `0.5`，生成翼型相对单工况最优翼型的交并比误差（intersection-over-union error, IoU error）分别仅 `0.04` 和 `0.03`，Hausdorff 距离分别为 `0.0075` 和 `0.0042`，阻力差异仅 `0.06%` 和 `0.02%`。单个翼型推理耗时约 `0.4 ms`。从 `Kn=0.01` 到 `1.0`，生成形状由尖锐细长逐渐变为较厚较短，反映了波阻向摩擦阻力主导机制的转换。

## 贡献与局限

**贡献**：将跨流域优化表述为 Kn 连续分布上的期望风险最小化，而不是手工选点；用同一个 USP 求解器消除 CFD/DSMC 分区耦合带来的接口复杂性；同一 PPO 框架给出固定几何的稳健设计和可即时推理的条件化翼型生成器；并用高斯求积、GA 与单工况解展示其收益。

**局限与启发**：结果局限于二维、`Ma=2`、零攻角和 14 维 B 样条空间，尚未检验三维、非定常、真实任务轨迹或结构/热约束。优势很大程度取决于 $P(\mathrm{Kn})$ 是否代表实际任务，分布设定错误会直接把优化资源投向错误工况。USP 仍是训练内环，未报告总 CFD 成本、网格收敛或跨求解器误差；`0.4 ms` 只表示训练完成后的网络推理，不包括生成训练数据的成本。对于可变形翼面，还需加入连续形变、机构行程和载荷等可制造约束。

## 关联

- [[Aerodynamic Design and Optimization via a Specialized Agentic Generative AI Framework]]：同属人工智能（Artificial Intelligence, AI）驱动气动设计，但后者关注多智能体生成流程，本文的策略直接把流动工况映射到形状。
- [[DiffAirfoil An Efficient Novel Airfoil Sampler Based on Latent Space Diffusion Model for Aerodynamic Shape Optimization]]：扩散模型的形状生成可与 CrossAero 的工况条件化思想对照；本文以物理求解器奖励而非离线数据生成训练信号。
- [[SURROOPT A GENERIC SURROGATE-BASED OPTIMIZATION CODE FOR AERODYNAMIC AND MULTIDISCIPLINARY DESIGN]]：代理模型优化通过近似降低评估成本，CrossAero 则直接以策略复用和迁移学习降低重复优化成本。
- [[Multi-fidelity convolutional neural network surrogate model for aerodynamic optimization based on transfer learning]]：两者都使用迁移学习；本文从简单流域策略迁移到跨流域任务，而该文在不同保真度数据间迁移。
