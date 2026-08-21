---
title: "Controlled gliding and perching through deep-reinforcement-learning"
authors:
  - Guido Novati
  - L. Mahadevan
  - Petros Koumoutsakos
year: 2019
venue: "Physical Review Fluids 4(9), 093902"
doi: "10.1103/PhysRevFluids.4.093902"
source: "/mnt/data/Novati 等 - 2019 - Controlled gliding and perching through deep-reinforcement-learning.pdf"
tags:
  - reinforcement-learning
  - gliding
  - optimal-control
  - flow-control
  - perching
---

> [!abstract] 摘要
> 受控滑翔是自然飞行器和人造飞行器中能量效率最高的运动方式之一。本文证明：即使控制器不知道底层物理方程，也可以通过深度强化学习（Deep Reinforcement Learning, D-RL）学习满足不同最优性指标的滑翔与着陆策略。作者将一个二维受控椭圆刚体的低阶动力学模型与 D-RL 结合，使滑翔体能够以最小控制能量或最短到达时间抵达预定位置并完成指定姿态的栖停（perching）。两类最优轨迹都较平滑，但能量最优策略倾向于较小幅度的控制，而时间最优策略包含更强、更高频的控制。论文进一步研究了椭圆体形状和重量对最优策略的影响，并发现模型自由的强化学习控制比基于模型的最优控制对初始条件和模型参数扰动更鲁棒，仅付出有限的额外计算代价。训练后的策略还能从训练中未见过的起点出发抵达目标，说明该方法有潜力用于在复杂流动环境中工作的自主飞行机器人。

## 一句话概括

这篇论文把一个**受控下落椭圆体的低阶气动力学模型**包装成连续状态、连续动作的强化学习问题，用 **RACER 离策略 Actor-Critic** 学习闭环控制律，并证明同一个框架可以自动发现“bounding（间歇滑翔/翻滚）”和“tumbling（持续翻滚）”两类飞行模式，在精度、鲁棒性和泛化性上表现出相对于传统开环最优控制的优势。

## 问题与动机

传统最优控制（Optimal Control, OC）需要显式动力学模型，并通常为给定初始条件求一条开环最优轨迹。一旦系统受扰偏离该轨迹，就需要重新求解。对真实飞行器而言，流体力难以精确建模，环境还可能存在噪声和不确定性，因此作者关注的问题是：

1. 能否只通过与环境交互，让滑翔体学习到到达指定位置和姿态的控制策略，而不在控制算法中显式使用动力学模型？
2. 最小时间和最小控制能量会产生什么不同的滑翔模式？
3. 滑翔体的形状与密度如何决定最优运动模式？
4. 强化学习得到的闭环策略相对于经典最优控制是否更鲁棒？

论文的生物学背景是无翼滑翔蚂蚁等动物：它们可以通过改变身体姿态或肢体位置产生控制力矩，在下落过程中调节横向位移。工程背景则是微型飞行器（Micro Air Vehicle, MAV）的节能滑翔与栖停。

## 方法

### 1. 二维受控椭圆体动力学模型

作者没有直接求解 Navier-Stokes 方程，而采用基于落纸片/椭圆体实验与数值结果拟合得到的六维常微分方程（Ordinary Differential Equation, ODE）模型。椭圆体长、短半轴分别为 $a,b$，定义

$$
\beta=\frac{b}{a},\qquad \rho^*=\frac{\rho_s}{\rho_f},\qquad I=\beta\rho^*,
$$

其中 $\beta$ 是形状参数，$\rho^*$ 是固体/流体密度比，$I$ 是无量纲转动惯量。长度以 $a$ 无量纲化，速度尺度为 $\sqrt{(\rho_s/\rho_f-1)gb}$。

状态变量为

$$
s=(x,y,\theta,u,v,w),
$$

其中 $(x,y)$ 是质心位置，$\theta$ 是椭圆长轴相对水平线的姿态角，$u,v$ 是速度在椭圆随体坐标系长短轴方向上的分量，$w$ 是角速度。

![[assets/Controlled gliding and perching through deep-reinforcement-learning/论文中图1.png]]

**论文中图1**：二维椭圆体的自由度和控制力矩 $\tau$。固定坐标系用于描述 $(x,y)$，$u,v$ 定义在随体旋转坐标系中。

动力学为

$$
(I+\beta^2)\dot u=(I+1)vw-\Gamma v-\sin\theta-Fu,
$$

$$
(I+1)\dot v=-(I+\beta^2)uw+\Gamma u-\cos\theta-Fv,
$$

$$
\frac14\left[I(1+\beta^2)+\frac12(1-\beta^2)^2\right]\dot w
=-(1-\beta^2)uv-M+\tau,
$$

$$
\dot x=u\cos\theta-v\sin\theta,\qquad
\dot y=u\sin\theta+v\cos\theta,\qquad
\dot\theta=w.
$$

流体力、流体力矩和环量采用经验参数化：

$$
F=\frac1\pi\left[A-B\frac{u^2-v^2}{u^2+v^2}\right]\sqrt{u^2+v^2},
$$

$$
M=0.2(\mu+\nu|w|)w,
$$

$$
\Gamma=\frac{2}{\pi}\left(C_Rw-C_T\frac{uv}{\sqrt{u^2+v^2}}\right),
$$

取 $A=1.4, B=1, \mu=\nu=0.2, C_T=1.2, C_R=\pi$。这些参数对应的动力学主要用于定性再现 $Re\sim O(10^3)$ 时的稳定下落、fluttering（扑动）和 tumbling（翻滚）等运动模式，而不是高保真定量气动力预测。

控制量只有一个：式中的**主动控制力矩 $\tau$**。它可代表动物改变肢体姿态、移动质心或偏转来流所产生的等效力矩。

### 2. 强化学习问题定义

由于 ODE 可以在任意时刻给出完整六维状态，本文将问题视为满足 Markov 性的 Markov Decision Process（MDP，马尔可夫决策过程），因此主实验使用前馈神经网络（Neural Network, NN），而不需要循环神经网络。

核心设置如下：

| 项目 | 设置 |
|---|---|
| 状态 | $s_t=(x,y,\theta,u,v,w)\in\mathbb R^6$ |
| 动作 | 高斯策略采样 $a_t$，再映射为 $\tau_t=\tanh(a_t)\in[-1,1]$ |
| 动作保持时间 | 默认 $\Delta t=0.5$，一个决策周期内 $\tau_t$ 保持常数 |
| 标称起点 | $x_0=y_0=\theta_0=0$ |
| 训练随机初始条件 | $x(0)\sim U[-5,5]$，$\theta(0)\sim U[-\pi/2,\pi/2]$ |
| 目标位置 | $(x_G,y_G)=(100,-50)$ |
| 目标栖停角 | $\theta_G=\pi/4$ |
| episode 终止 | 滑翔体接触地面 |
| 折扣因子 | $\gamma=1$，因为重力保证轨迹有限时结束 |

论文没有在该处进一步说明 $u,v,w$ 是否以及如何随机初始化，因此不能从正文补出其分布。

为允许不同的末端栖停机动，$x=50\sim100$ 区域的地面被向下“挖槽”，其高度写为

$$
y_{\rm ground}=-50-0.4\min(x-50,100-x).
$$

这使智能体可以探索更丰富的末端轨迹，而不是被一条完全平直的地面过早截断。

### 3. RACER：连续动作离策略 Actor-Critic

本文使用 RACER。RACER 是算法名称，原文及其方法论文没有给出需要展开的英文全称；它属于 **off-policy actor-critic（离策略 Actor-Critic）**。

一个三层、每层 128 个单元的 NN 输入状态 $s$，同时输出四个量：

$$
\{m^w(s),\sigma^w(s),V^w(s),l^w(s)\}.
$$

其中前两个量定义高斯策略

$$
\pi_w(a|s)=\frac{1}{\sqrt{2\pi}\sigma^w(s)}
\exp\left[-\frac12\left(\frac{a-m^w(s)}{\sigma^w(s)}\right)^2\right].
$$

$V^w(s)$ 是状态价值，$l^w(s)$ 用于构造一个关于动作的二次型 action-value（动作价值）近似：

$$
Q^w(s,a)=V^w(s)-\frac12[l^w(s)]^2
\left[(a-m^w(s))^2-[\sigma^w(s)]^2\right].
$$

这个参数化有两个关键含义：

- 对给定状态，$Q(s,a)$ 在策略均值 $a=m(s)$ 附近达到最大；
- 不需要再单独建立一个把 $(s,a)$ 同时作为输入的 Critic 网络，因而连续动作下的 $Q$ 估计可以用闭式形式得到。

训练时保存轨迹经验。每个 observation 记录状态、奖励、采样动作，以及采样该动作时的旧行为策略 $\mu_t=\{m_t,\sigma_t\}$。由于网络不断更新，同一条旧经验在训练时对应的当前策略已经变成 $\pi_w$，因此需要重要性采样比率

$$
\rho(s_t,a_t)=\frac{\pi_w(a_t|s_t)}{\mu_t(a_t|s_t)}.
$$

Actor 的梯度本质上由

$$
\rho_t\,[\hat Q(s_t,a_t)-V^w(s_t)]\nabla_w\log\pi_w(a_t|s_t)
$$

加权；Critic 则拟合动作价值目标。

#### Retrace 的作用

正文使用 Retrace 构造离策略的多步 $Q$ 目标，递推形式为

$$
\hat Q_{\rm ret}(s_t,a_t)=r_{t+1}+\gamma V^w(s_{t+1})
+\gamma\min(1,\rho_{t+1})
[\hat Q_{\rm ret}(s_{t+1},a_{t+1})-Q^w(s_{t+1},a_{t+1})].
$$

其直觉是：**旧经验与当前策略越接近，就越充分利用其多步回报；两者差异过大时，就截断重要性权重，避免方差爆炸。** Retrace（Munos 等，2016）被设计为兼顾低方差、离策略安全性和样本利用效率的多步回报估计。

#### 外部补充：RACER 与 ReF-ER 的关系

本文将更多实现细节指向 Ref. 47。该方法论文提出 **ReF-ER（Remember and Forget Experience Replay，记忆与遗忘经验回放）**，并将 RACER 与其结合。ReF-ER 主要解决经验回放中的 distribution shift（分布漂移）：

- 根据当前策略与产生旧经验的行为策略之间的重要性比率，把样本分成 near-policy 与 far-policy；
- 对偏离当前策略过远的经验停止/抑制其梯度；
- 通过 Kullback-Leibler（KL，Kullback-Leibler divergence）项限制策略相对回放记忆中的行为策略变化过快，相当于建立一个 trust region（信赖域）。

因此，RACER 的优势不只是“用了经验回放”，而是**尽量让被重复利用的旧数据仍处于当前策略可可靠学习的局部范围内**。本文正文没有重新列出 Ref. 47 中 ReF-ER 的全部超参数，因此严格复现训练器仍需查阅该方法论文。

### 4. 奖励设计：把最优控制目标写成 RL reward

论文分别优化时间和控制能量。

**时间最优：**

$$
c_t=\Delta t.
$$

**能量最优：**

$$
c_t=\int_{t-1}^{t}\tau^2(t)dt=\tau_{t-1}^2\Delta t.
$$

这里的 $\tau^2$ 只是控制能耗 proxy（代理指标），不是严格物理功率。作者指出，由于转动阻力关系，平均有 $w^2\sim\tau$，实际输入功率 $w\tau$ 大约按 $\tau^{3/2}$ 缩放，因此采用 $\tau^2$ 实际上施加了更严格的强控制惩罚。

普通时刻的 reward 为

$$
r_t=-c_t+|x_G-x_{t-1}|-|x_G-x_t|.
$$

后两项奖励“向目标靠近”。如果轨迹最终准确达到 $x_G$，这部分沿时间求和后只剩下一个由初始距离决定的常数，因此不会改变原本时间/能量目标的最优策略；它的作用主要是让学习过程获得密集反馈，而不是只在终点得到信号。

终止时再加入位置与姿态 bonus：

$$
r_T=-c_T+K\left[e^{-(x_G-x_T)^2}+e^{-10(\theta_G-\theta_T)^2}\right].
$$

姿态项仅在 $95<x_T<105$ 时启用，以避免智能体找到“远离目标但姿态正确”的局部最优解。时间最优训练取 $K_T=50$，能量最优取 $K_E=20$，使终点奖励量级与累计控制代价大致相当。

这也是论文和传统 OC 的关键区别：**OC 可以把边界条件和终端约束直接写进优化问题，RL 只能通过终止条件和 reward shaping（奖励塑形）间接推动策略满足约束。**

## 主要结果

### 1. 自动出现两种飞行模式：bounding 与 tumbling

![[assets/Controlled gliding and perching through deep-reinforcement-learning/论文中图2.png]]

**论文中图2**：上行为 bounding flight，下行为 tumbling flight；左列是 $x-y$ 轨迹，右列是 $u-v$ 相图。

**Bounding flight** 是滑翔与快速翻滚交替的间歇模式：

- 滑翔阶段施加负力矩，维持较小攻角，借助来流偏转产生升力并减慢下沉；
- 随后施加强正力矩快速旋转约 $180^\circ$，产生额外升力并恢复到新的滑翔姿态；
- 其控制具有明显的“低强度维持 + 短促强脉冲”结构。

**Tumbling flight** 更简单：智能体施加近似恒定的力矩，使椭圆体在前进过程中持续旋转并产生平均升力，接近落点时再降低转速以完成指定姿态的着陆。

### 2. 形状和密度决定最优飞行模式

作者在

$$
\rho^*\in[25,800],\qquad \beta\in[0.025,0.4]
$$

上进行参数扫描，并针对每组参数分别训练时间最优和能量最优策略。

![[assets/Controlled gliding and perching through deep-reinforcement-learning/论文中图3.png]]

**论文中图3**：左图给出时间最优策略在参数空间中的 flight pattern；中、右图分别给出最优时间代价和能量代价。

主要规律：

- **轻、细长的椭圆体**倾向于 bounding flight；
- **重、厚的椭圆体**倾向于 tumbling flight；
- 二者的分界大致由
  $$
  I=\rho^*\beta\approx30
  $$
  描述，即转动惯量是比单独的密度或厚度更直接的控制参数；
- 时间最优策略的模式分类最清晰；能量最优策略通常混合两种模式的特征；
- 时间代价和能量代价都随 $\beta$、$\rho^*$ 增大而上升；
- 在 $|\tau|\le1$ 限制下，作者没有找到 $I>160$ 的时间最优可达策略，也没有找到 $I>80$ 的能量最优策略。能量目标对大力矩惩罚更强，因此先丧失可达性。

### 3. 训练后的策略对初始条件和模型误差具有鲁棒性

![[assets/Controlled gliding and perching through deep-reinforcement-learning/论文中图4.png]]

**论文中图4**：左图改变训练中未见过的起始 $x$ 位置，右图对 ODE 气动力参数施加比例 log-normal 噪声。

在 $\beta=0.1,\rho^*=200$ 的时间最优策略下：

- 即使初始横向位置超出训练时的 $[-5,5]$ 区间，策略仍可把滑翔体引导到目标附近；
- 对 $\{A,B,\mu,\nu,C_T,C_R\}$ 同时施加随机比例扰动，噪声标准差取 $\sigma_\xi=0.1,0.2,0.4$；
- 图中每个噪声等级展示 $10^4$ 条轨迹的包络，即使 $\sigma_\xi=0.4$，控制器仍能恢复方向并在目标邻域着陆。

这说明学习到的是**状态反馈的闭环策略**，而不是只记住单条轨迹。

### 4. 与最优控制的比较

作者选择 $\beta=0.1,\rho^*=200$，与 Paoletti & Mahadevan 的最优控制结果比较。对方使用 GPOPS（General Pseudospectral Optimal Control Software，一类高斯伪谱最优控制求解器）求开环轨迹。

![[assets/Controlled gliding and perching through deep-reinforcement-learning/论文中图5.png]]

**论文中图5**：能量最优 RL（蓝）与 OC（黑）的轨迹、角速度和控制力矩。两者都趋向于用较小且近似恒定的力矩维持 steady tumbling。

![[assets/Controlled gliding and perching through deep-reinforcement-learning/论文中图6.png]]

**论文中图6**：时间最优 RL（蓝）与 OC（黑）。两者都呈现类似 bang-bang（控制量频繁触及上下界）的强控制结构，在滑翔与翻滚阶段之间切换。

论文报告：

- RL 的最终能量代价约比 OC **低 2%**；
- RL 的时间代价约比 OC **低 4%**；
- 改变初始角 $\theta_0$ 后，末端位置和姿态误差均约为 $O(10^{-2})$。

但这里不能简单解读为“RL 理论上优于 OC”。作者自己指出，OC 求解器需要对连续时间问题离散化，只能在有限网格精度下得到局部最优解；RL 使用不同的时间离散，因此小幅优势可能来自数值离散差异。更可靠的结论是：**RL 找到了与 OC 定性一致的近最优控制模式，同时得到可直接反馈当前状态的闭环策略。**

### 5. 控制频率会影响学习难度和最优性能

![[assets/Controlled gliding and perching through deep-reinforcement-learning/论文中图8.png]]

**论文中图8**：比较 $\Delta t=0.5,2,16,64$ 时训练过程中的落点、姿态、时间代价和能量代价。

- $\Delta t=64$ 时，每条轨迹只有约 2–3 次决策，单次动作必须非常精确，训练明显更困难；
- 较高动作频率提供更多纠错机会，也能表达更细的时间结构，因此可得到更低的最优时间代价；
- 作者每执行一次 action 就进行一次梯度更新；同时长动作周期需要在一次动作内推进更多底层 ODE 数值积分步，因此在其实现中 $\Delta t=0.5$ 的训练反而比 $\Delta t=64$ 更快。

### 6. 部分可观测时，RNN 能补回一部分隐变量信息，但训练更难

![[assets/Controlled gliding and perching through deep-reinforcement-learning/论文中图9.png]]

**论文中图9**：完整状态前馈 NN、只观察 $\{x,y,\theta\}$ 的前馈 NN、以及只观察 $\{x,y,\theta\}$ 的循环网络比较。

若隐藏速度 $\{u,v,w\}$：

- 只用 $\{x,y,\theta\}$ 的前馈 NN 很难学到精确落点；
- 使用 RNN（Recurrent Neural Network，循环神经网络），具体为 LSTM（Long Short-Term Memory，长短期记忆网络），可以从位置/姿态时间历史中隐式估计速度；
- RNN 最终能够较准确地到达 $x_T$，但需要更多迭代才能学到正确栖停角，时间代价也不如完整状态问题。

这表明**减少输入维度并不一定降低学习难度**：如果被删掉的是保证 Markov 性的状态变量，问题会从完全可观测 MDP 变成部分可观测问题。

### 7. RACER、NAF、PPO 的比较

![[assets/Controlled gliding and perching through deep-reinforcement-learning/论文中图10.png]]

**论文中图10**：RACER、NAF 和 PPO 在时间最优任务上的训练表现。

- RACER 在前约 **1000 条观测轨迹**后就能稳定把落点集中到 $x_G$ 附近；
- NAF（Normalized Advantage Functions，归一化优势函数）也是离策略算法，并使用二次型 $Q(s,a)$，但策略均值由 Critic 梯度而不是 policy gradient（策略梯度）驱动，本文中落点分布较不稳定；
- PPO（Proximal Policy Optimization，近端策略优化）是 on-policy（同策略）Actor-Critic，只用最新轨迹更新，本文中未能把落点分布稳定集中到 $x_G$；作者将其归因于 on-policy $Q$/return 估计的高方差。

训练时策略是随机高斯策略；最终评估时若每一步直接采用均值动作 $m(s)$，可显著提高落点精度。

## 关键贡献

1. 将受控椭圆体滑翔/栖停问题构造成连续状态、连续动作的深度强化学习任务，并在不把动力学方程显式交给控制器的情况下学习近最优闭环策略。
2. 在同一框架下发现时间最优与能量最优控制，并自动恢复了与经典 OC 相似的 bounding / tumbling 控制结构。
3. 给出形状-密度参数空间中的运动模式图，发现 $I=\rho^*\beta\approx30$ 可近似作为 bounding 与 tumbling 的分界。
4. 证明 RL 策略能泛化到训练未见的起点，并对相当大的低阶模型参数误差保持落点鲁棒性。
5. 系统分析动作频率、可观测状态以及 RL 算法选择对精确控制性能的影响，指出“学会一个看起来合理的动作”与“以 $O(10^{-2})$ 精度满足终端约束”是两个不同难度的问题。

## 实验 setup 汇总

### 基准任务

- 二维静止流体中的重力下落椭圆体；
- 目标从约 $(0,0)$ 横向飞至 $(100,-50)$；
- 指定末端角度 $\pi/4$；
- 控制仅为有界力矩 $\tau\in[-1,1]$；
- 默认控制周期 $\Delta t=0.5$；
- 主网络：3 个隐藏层，每层 128 单元；
- 策略：单变量 Gaussian policy（高斯策略）；
- $Q$：由 $V,m,\sigma,l$ 构造的二次型闭式近似；
- 经验：离策略 replay，使用 importance sampling（重要性采样）与 Retrace；
- 时间最优与能量最优分别训练；
- 参数扫描：$\rho^*\in[25,800]$、$\beta\in[0.025,0.4]$；
- 鲁棒性：改变起点，及对 6 个经验气动力参数施加 log-normal 比例噪声；
- 算法消融：RACER / NAF / PPO；
- 可观测性消融：全状态 NN / 位置姿态 NN / 位置姿态 LSTM；
- 控制频率消融：$\Delta t=0.5,2,16,64$。

### 论文没有完整报告的复现细节

本文正文明确给出网络宽度、主要 reward、状态动作定义和控制周期，但将 RACER 的更多实现细节转引到 Ref. 47。因此，若要严格复现，仍需要额外确认 replay memory 大小、mini-batch、优化器/学习率、ReF-ER 相关阈值与 annealing 等设置。论文也没有给出 RL 与 OC 的完整 wall-clock 时间对比，因此“modest additional computational cost”无法仅从本文量化。

## 局限

1. **物理模型低阶。** 环境是二维经验 ODE，而不是三维高保真 CFD/DNS；参数模型只保证对 $Re\sim O(10^3)$ 运动模式的定性一致性。
2. **“model-free”仅指控制算法。** 策略更新不需要显式动力学模型或模型梯度，但训练仍依赖一个可反复调用的 ODE 仿真环境，并不是不需要 simulator。
3. **控制自由度极少。** 智能体只有一个标量力矩，无法表示真实鸟类/MAV 的升力面变形、攻角、舵面、推力等多自由度控制。
4. **终端约束依赖手工 reward shaping。** 与 OC 可直接写等式/不等式约束不同，RL 需要人工设计 $K$、位置奖励和终端 bonus，且这些超参数会改变学习难度。
5. **能量指标只是代理。** $\tau^2$ 不是严格机械功率，因而“energy-optimal”应理解为在该控制代价定义下的最优，而非真实飞行能耗最优。
6. **对 OC 的小幅优势不是全局最优证明。** 2%–4% 的差异可能来自双方离散化精度；RL 本身也只得到近似最优策略。
7. **部分可观测问题明显更难。** 真实飞行器往往不能直接测得完整状态，本文结果显示传感器缺失会显著增加学习难度。
8. **尚无真实飞行实验。** 作者的下一步设想是三维 Direct Numerical Simulation（DNS，直接数值模拟）和无人机栖停/滑翔实验，本文并未完成 sim-to-real 验证。

## 启发

### 对强化学习控制

- **精确终端任务应优先关注 reward/constraint 设计，而不只是换更强的 RL 算法。** 本文的难点不是“能不能往右飞”，而是同时满足落点、姿态和代价最优。
- **离策略经验复用特别适合昂贵物理仿真，但必须控制 off-policyness。** 旧样本与当前策略偏差太大时，经验回放反而可能产生错误梯度；Retrace 与 ReF-ER 都是在解决这一问题。
- **低阶模型预训练 + 闭环反馈可用于提高迁移容错。** 作者的鲁棒性结果支持先在便宜模型中学习反馈规律，再向高保真仿真/实验迁移的路线，但本文还没有真正验证这种 transfer。
- **高动作频率既提高策略表达能力，也提高在线纠错能力。** 对需要精确 terminal state 的任务，动作离散不能只从计算量角度选择。

### 对滑翔/飞行控制

- 运动模式并非一定需要人为预设。给定目标和代价后，RL 可以在物理动力学中自行发现具有明确气动意义的 bounding/tumbling 策略。
- $I=\rho^*\beta$ 近似控制模式边界说明，**形状、密度与可控性通过惯性耦合**；如果未来把可变构型作为设计变量，结构参数和控制策略应联合优化，而不是先定构型再独立设计控制器。
- RL 的主要工程优势不是比 OC 多省 2%–4%，而是**一次离线训练后得到整个状态空间上的反馈控制律**，能够在初始条件和模型误差变化时直接在线反应。

## 关联与文献脉络

### 1. Paoletti & Mahadevan, 2011 — *Planar controlled gliding, tumbling, and descent*

这是本文最直接的前身。它使用基本相同的受控椭圆体 ODE，并通过经典最优控制求能量/时间最优轨迹。本文不是重新发明物理模型，而是把“给定模型求单条最优轨迹”改造成“通过交互学习闭环策略”，并直接用该工作作为 OC benchmark。

### 2. Reddy et al., 2016 — [[Learning to soar in turbulent environments]]

Reddy 等研究的是**利用环境中的热上升气流进行 soaring**，重点是从湍流局部传感线索中学习导航策略；本论文研究的是**通过自身控制力矩改变下落/滑翔动力学**。两者都把飞行动物问题转成 RL，但前者的核心是“感知并利用复杂环境”，本文的核心是“精确连续控制与终端约束”。

### 3. Reddy et al., 2018 — [[Glider soaring via reinforcement learning in the field]]

2018 年工作把 Reddy 2016 的思路推进到真实滑翔机现场实验，使用离散状态/动作与长期采集的 field experience。相比之下，本文完全在低阶仿真中训练，但采用连续动作深度 RL，能处理更高精度的轨迹和姿态控制。两条路线互补：Reddy 强调真实环境下可学习的传感线索，Novati 强调连续闭环控制的精度、鲁棒性和与 OC 的比较。

### 方法侧补充：Novati & Koumoutsakos, 2019 — *Remember and Forget for Experience Replay*

这是本文 RACER 的方法来源，重点解决离策略经验回放中的 policy-behavior distribution shift，并提出 ReF-ER。理解这篇方法论文有助于解释为什么本文在精确连续控制任务中 RACER 比 PPO/NAF 更稳定。

## 我的理解

本文最重要的结论不是“深度 RL 能控制一个椭圆体”，而是它展示了一种适合流体/飞行问题的控制范式：**用足够便宜的动力学模型产生大量轨迹，通过离策略 RL 学出闭环控制律，再依靠反馈对模型误差与初始条件变化进行修正。** 相比直接做一次最优轨迹求解，它把计算从“每个新初值重新优化”转为“离线训练一次，在线快速推理”。

与此同时，这篇论文也清楚暴露了 RL 在工程最优控制中的核心短板：精确约束需要 reward shaping，完整状态假设往往不现实，低阶 simulator 中的鲁棒性不等于真实系统鲁棒性。因此，如果把这一路线扩展到更复杂飞行器，下一步最值得关注的不是单纯增大网络，而是**约束强化学习、部分可观测状态估计、低阶到高保真的策略迁移，以及真实物理能耗定义**。

## 外部补充来源

- Novati, G. & Koumoutsakos, P. (2019), *Remember and Forget for Experience Replay*, ICML / PMLR 97：用于补充 RACER 与 ReF-ER 的关系。
- Munos, R., Stepleton, T., Harutyunyan, A. & Bellemare, M. G. (2016), *Safe and Efficient Off-Policy Reinforcement Learning*, NeurIPS：用于补充 Retrace 的设计目的与直觉。
