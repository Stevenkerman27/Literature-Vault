---
title: "Learning efficient navigation in vortical flow fields"
authors:
  - Peter Gunnarson
  - Ioannis Mandralis
  - Guido Novati
  - Petros Koumoutsakos
  - John O. Dabiri
year: 2021
venue: "Nature Communications"
doi: "10.1038/s41467-021-27015-y"
source: "/mnt/data/Gunnarson 等 - 2021 - Learning efficient navigation in vortical flow fields.pdf"
tags:
  - reinforcement-learning
  - flow-navigation
  - vortical-flow
  - V-RACER
  - robotics
---

> [!abstract] 摘要
> 在存在非定常背景流的环境中，实现高效的点到点导航对于海洋测绘等机器人任务非常重要。实际机器人往往只能感知局部环境，而且流场会随时间变化，因此依赖完整全局流场的最优控制方法并不总是可用。本文使用深度强化学习（Deep Reinforcement Learning, Deep RL）学习固定自航速度游动体在二维非定常流场中的时间高效导航策略：局部环境信息和目标相对位置输入深度神经网络，由网络输出连续游动方向，并使用 V-RACER 与 Remember and Forget Experience Replay（ReF-ER，记忆-遗忘经验回放）训练。结果表明，智能体能够主动利用背景流到达目标，但性能强烈依赖状态中提供的流动信息。与仿生的局部涡量感知相比，局部速度感知表现显著更好，在圆柱尾迹任务中达到接近 100% 的成功率，同时其航行时间接近具有完整全局流场信息的时间最优轨迹。

# 一句话总结

这篇工作的核心不是“强化学习能不能在涡流里导航”，而是：**在只能局部感知的非定常流场中，状态变量的选择决定了 RL 是否真正可用；直接决定下一步平移动力学的局部速度 \((u,v)\) 比看似更“流体力学”的涡量 \(\omega\) 更有信息量。**

# 问题与动机

传统时间最优路径规划（例如 Zermelo 导航问题及其现代数值方法）通常假设可以预先获得完整流场。真实海洋、风场或机器人自身扰动形成的流动往往难以完整测量，并且可能快速变化，因此更现实的机器人需要依赖机载传感器形成**局部闭环导航策略**。

本文重点回答两个问题：

1. 只给机器人局部流动信息，Deep RL 能否在非定常流场中学到鲁棒且接近时间最优的导航策略？
2. **什么局部流动物理量最适合作为 RL 状态？** 仿生的涡量信息是否真的优于直接速度信息？

# 方法

## 1. 圆柱尾迹导航环境

主实验采用二维不可压缩圆柱绕流，雷诺数为

$$
Re=400,
$$

形成非定常 von Kármán 涡街。作者选择该流场是为了兼顾两点：它足够复杂、具有明显非定常涡结构，同时比真实海洋流或完全湍流更容易解释学习到的策略。

游动体被简化为**无质量质点**，位置为

$$
\mathbf X_n=[x_n,y_n],
$$

其运动由自身恒定游速与局部背景流速度共同决定。离散动力学采用前向 Euler：

$$
\mathbf X_{n+1}=\mathbf X_n+\Delta t\left[
U_{\mathrm{swim}}
\begin{pmatrix}
\cos\theta_n\\
\sin\theta_n
\end{pmatrix}
+\mathbf U_{\mathrm{flow}}(x_n,y_n,t_n)
\right].
$$

其中：

- \(\theta_n\)：智能体直接控制的游动方向；
- \(U_{\mathrm{swim}}=0.8U_\infty\)：自航速度仅为自由来流速度的 80%，因此在部分区域无法靠自身速度直接逆流；
- \(\Delta t=0.3D/U_\infty\)；
- \(D\)：圆柱直径；
- \(U_\infty\)：自由来流速度。

若 \(U_{\mathrm{swim}}<0.6U_\infty\)，该任务大多不可解；若 \(U_{\mathrm{swim}}>U_\infty\)，智能体可以直接克服背景流，问题又会过于简单。

![[assets/Learning efficient navigation in vortical flow fields/论文中图1.png]]

**论文中图1：圆柱尾迹导航任务。** 起点和目标分别从两个直径为 \(4D\) 的圆形区域内随机采样，两区域均位于圆柱下游约 \(5D\)，并相对尾迹中心线上下偏置约 \(2.05D\)。每个 episode 还随机选择涡脱落周期中的起始时刻。进入目标点半径 \(D/6\) 的邻域即视为成功。

这种空间和时间随机化非常关键：若固定起点、目标点和流场相位，网络可能把“相对位置”与某个确定流场结构死记硬背，而不是真正形成可推广的局部感知策略。所有 RL 智能体只知道相对目标位移 \((\Delta x,\Delta y)\)，不提供绝对坐标。

## 2. 状态、动作与对照策略

作者比较了四种导航方式：

| 策略 | 状态/信息 | 说明 |
|---|---|---|
| Naive swimmer | 仅目标方向 | 非 RL；始终直接朝目标游，\(\theta=\tan^{-1}(\Delta y/\Delta x)\) |
| Flow-blind RL | \(s=\{\Delta x,\Delta y\}\) | 不感知流场，只根据目标相对位置学习 |
| Vorticity RL | \(s=\{\Delta x,\Delta y,\omega_n,\omega_{n-1}\}\) | 使用当前与上一时刻的局部涡量，试图感知涡量变化 |
| Velocity RL | \(s=\{\Delta x,\Delta y,u,v\}\) | 使用局部背景流两个速度分量 |

动作空间是连续的游动方向 \(\theta\)。因此这不是离散“向上/下/左/右”的路径规划，而是连续控制问题。

Vorticity swimmer 的设计受斑马鱼流感知启发。由于点涡量本质上是速度梯度信息，它可以帮助识别尾迹中的旋涡结构，但对空间均匀的平移流并不敏感；这一点后来直接解释了其性能上限。

## 3. V-RACER 策略网络

本文使用 **V-RACER** 处理连续动作。正文给出的网络配置为一个两层、每层 128 个单元的深度神经网络（记为 **128 × 128**）。网络输入为当前状态，输出连续游动策略，并同时给出用于训练探索的高斯方差。

结合 V-RACER 原始方法论文，可以更准确地理解网络输出：单个神经网络同时近似状态价值 \(V^w(s)\)，并参数化高斯策略

$$
\pi_w(a|s)=\mathcal N\big(m^w(s),\Sigma^w(s)\big),
$$

其中本问题的动作是一维方向 \(\theta\)，因此可以理解为网络输出方向分布的均值与方差，以及状态价值估计。训练属于 **off-policy policy gradient（离策略策略梯度）**：历史交互数据会被重复利用，而不是一次更新后立即丢弃。

### ReF-ER：为什么历史经验还能稳定复用？

V-RACER 配套使用 **Remember and Forget Experience Replay（ReF-ER）**。其核心问题是：经验回放中的旧样本来自旧策略 \(\mu\)，而当前策略已经变成 \(\pi\)；如果两者相差太远，用旧样本更新当前策略会带来高方差甚至错误梯度。

原始 ReF-ER 方法通过重要性权重

$$
\rho=\frac{\pi(a|s)}{\mu(a|s)}
$$

衡量旧经验与当前策略的接近程度，并做两件事：

1. **Forget / Rule 1**：把与当前策略差异过大的 far-policy 样本梯度置零，只用 near-policy 样本更新；
2. **Remember / Rule 2**：用 Kullback-Leibler（KL）散度惩罚限制当前策略过快偏离回放缓存中的历史行为，使经验回放保持有效。

因此，ReF-ER 的作用不是简单“存更多经验”，而是主动控制**经验的离策略程度（off-policyness）**，在数据效率与训练稳定性之间取得平衡。本文没有在正文中展开这些公式，具体算法来自其引用的 ReF-ER/V-RACER 方法论文。

## 4. 奖励函数

每个时间步奖励为

$$
r_n=-\Delta t
+10\left[
\frac{\|\mathbf X_{n-1}-\mathbf X_{\mathrm{target}}\|}{U_{\mathrm{swim}}}
-
\frac{\|\mathbf X_n-\mathbf X_{\mathrm{target}}\|}{U_{\mathrm{swim}}}
\right]
+\mathrm{bonus}.
$$

三部分分别对应：

- **时间惩罚** \(-\Delta t\)：鼓励尽快到达；
- **距离进展奖励**：如果本步比上一步更接近目标，就获得正奖励；系数 10 用于把该项量级与时间项拉到相近范围，作者指出这样显著提高了训练速度和成功率；
- **成功奖励**：到达目标时额外给 \(200\) 个时间单位，约为典型轨迹持续时间的 30 倍。

若游动体离开仿真区域或撞上圆柱，则 episode 判为失败。

总奖励可以望远镜式化简为

$$
r_{\mathrm{total}}
=-T_f
+10\frac{\|\mathbf X_{\mathrm{start}}-\mathbf X_{\mathrm{target}}\|}{U_{\mathrm{swim}}}
+\mathrm{bonus}.
$$

因此，对于给定起点和目标且最终成功的轨迹，第二项是常数，最大化回报基本等价于**最小化到达时间 \(T_f\)**；大额 terminal bonus 则强烈推动策略优先学会“成功到达”。

作者还测试过把距离项改成到目标距离的倒数，但效果更差。

![[assets/Learning efficient navigation in vortical flow fields/论文中图2.png]]

**论文中图2：三类 RL 智能体的训练回报。** 每类智能体训练 20,000 个 episodes，并独立训练 5 次以降低神经网络随机初始化造成的偶然差异。Velocity RL 的回报最快稳定到成功区域附近；Flow-blind 和 Vorticity RL 则长期保留大量失败 episode。

## 5. 与时间最优控制比较

为了判断 RL 是否只是“能到”，还是确实接近时间最优，作者另外构造了一个拥有**完整时空流场先验**的全局路径规划器：

1. 使用 **Rapidly-exploring Random Tree（RRT，快速扩展随机树）** 先找到一条可行控制序列；
2. 使用 MATLAB `fmincon` 做带约束梯度优化，最小化时间步长，从而最小化总到达时间 \(T_f\)；
3. 约束包括：起点固定、每一步都满足游动体动力学、终点进入目标半径 \(D/6\)；
4. 因为梯度法可能落入局部最优，重复 100 次并选择最快解；
5. 再用 level-set 方法验证最快结果的全局最优性。

这个最优控制器知道完整未来流场，但产生的是**开环轨迹**；RL 只依赖局部当前观测，却是闭环反馈策略。因此二者的比较本质上是“全局先验 + 开环最优”与“局部观测 + 闭环学习”的比较。

## 6. 双涡流（double gyre）迁移实验

作者进一步用拓扑明显不同的二维非定常 **double gyre** 流场测试策略迁移。正文引用标准模型而未重新写出方程。其引用的标准形式为：

$$
\psi(x,y,t)=A\sin[\pi f(x,t)]\sin(\pi y),
$$

$$
f(x,t)=a(t)x^2+b(t)x,
\qquad
a(t)=\epsilon\sin(\omega t),
\qquad
b(t)=1-2\epsilon\sin(\omega t),
$$

由流函数得到

$$
u=-\frac{\partial\psi}{\partial y}
=-\pi A\sin[\pi f(x,t)]\cos(\pi y),
$$

$$
v=\frac{\partial\psi}{\partial x}
=\pi A\cos[\pi f(x,t)]\sin(\pi y)\frac{\partial f}{\partial x}.
$$

本文使用无量纲长度 \(L=1\)，参数为

$$
A=\frac{2}{3}U_{\mathrm{swim}},\qquad
\epsilon=0.3,\qquad
\omega=\frac{20\pi U_{\mathrm{swim}}}{3L}.
$$

起点随机采样于右侧 gyre，目标随机采样于左侧 gyre；两个圆形区域直径均为 \(L/2\)，中心分别位于 \((3L/2,L/2)\) 与 \((L/2,L/2)\)。同样随机化起始流场相位。进入目标半径 \(L/50\) 内视为成功。

该部分比较：Naive、在圆柱尾迹上训练的 Velocity RL、以及在 double gyre 上重新训练的 Velocity RL。

# 关键实验结果

## 1. 局部速度感知带来数量级的成功率提升

![[assets/Learning efficient navigation in vortical flow fields/论文中图4.png]]

**论文中图4：四种策略在圆柱尾迹中的轨迹与平均成功率。** 绿色为成功，红色为失败。

| 策略 | 成功率 |
|---|---:|
| Naive | \(1.3\%\pm0.4\%\) |
| Flow-blind RL | \(39.4\%\pm5.8\%\) |
| Vorticity RL | \(47.2\%\pm8.7\%\) |
| Velocity RL | \(99.9\%\pm0.1\%\) |

这些成功率基于 12,500 个测试 episodes，并统计 5 次独立训练产生的标准差。

最直接的结论是：

- 仅靠“始终朝目标游”几乎完全失败，因为 \(U_{\mathrm{swim}}<U_\infty\)，智能体会被主流卷走；
- 即使看不到瞬时流场，RL 也能通过相对位置学到平均流对运动的影响，因此 Flow-blind 明显优于 naive；
- 单点涡量确实提供额外信息，但只带来有限提升；
- 同时感知 \(u,v\) 后，成功率跃升到接近 100%。

## 2. Velocity RL 学会“借尾迹逆流上游”，而不是直冲目标

![[assets/Learning efficient navigation in vortical flow fields/论文中图3.png]]

**论文中图3：Velocity RL 的典型成功轨迹。** 因为自身速度低于自由来流，策略先进入圆柱尾迹中的低速区，在涡结构之间不断调整方向并保持在尾迹内，从而逐步向上游移动；到达足够上游的位置后再转向目标。这是一种明确的**利用背景流结构**的策略，而不是把流场当作纯扰动抵消。

## 3. 为什么速度比涡量更有效？

![[assets/Learning efficient navigation in vortical flow fields/论文中图5.png]]

**论文中图5：固定目标和时刻下的策略方向场。** Flow-blind 基本不随瞬时流场变化；Vorticity RL 会对旋涡做有限调整；Velocity RL 对背景流变化最敏感，尤其能在尾迹外被主流带走前主动转回尾迹。

论文给出的物理解释有两层：

### 第一层：可观测性

涡量是速度梯度的旋转部分。若某一区域是近似均匀平移流，局部涡量可以接近零，但这股流仍然会持续平移游动体。因此**单点涡量无法感知均匀平移背景流**。它能够区分“涡街区域”和“自由来流区域”，所以比完全不看流场好，但信息不够完整。

### 第二层：一步动力学预测

在本文简化动力学中，下一步位置直接包含局部速度：

$$
\mathbf X_{n+1}=\mathbf X_n+\Delta t(\mathbf U_{\mathrm{swim}}+\mathbf U_{\mathrm{flow}}).
$$

因此当前 \((u,v)\) 对 \(\mathbf X_{n+1}\) 有直接决定作用。换句话说，Velocity RL 获得了一个对**下一状态高度预测性的状态变量**。它仍不知道下一时刻的流速，但至少能准确判断“如果现在采取这个方向，下一步会被流带到哪里”。

这也是本文最有普适性的启发：**RL 状态设计应优先保留对状态转移有直接预测力的物理量，而不应只依据仿生直觉或某个看起来更高阶的流体特征。**

## 4. 局部观测的 RL 接近全局时间最优路径

![[assets/Learning efficient navigation in vortical flow fields/论文中图6.png]]

**论文中图6：三组起终点下 RL 与时间最优轨迹。** 黑色为 Velocity RL，红色为拥有完整流场信息的最优控制。

三组示例到达时间为：

| 示例 | RL \(T_f\) | Optimal \(T_f\) | 最优轨迹相对更快 |
|---|---:|---:|---:|
| 1 | 8.80 | 5.38 | 39% |
| 2 | 18.4 | 15.4 | 16% |
| 3 | 33.3 | 25.7 | 23% |

RL 轨迹总体形态与时间最优路径相近，说明策略网络从大量交互中学到了一种隐式的“局部流动 → 未来导航价值”映射。不过局部观测仍有天然限制：例如右侧示例中，最优控制器会更早从另一位置进入尾迹以避开未来高速度区域，而 RL 在初始局部观测中无法预知该远处流动。

## 5. 对随机初值很鲁棒，但跨拓扑流场迁移失败

![[assets/Learning efficient navigation in vortical flow fields/论文中图7.png]]

**论文中图7：double gyre 迁移实验。**

| 策略 | double gyre 成功率 |
|---|---:|
| Naive | \(40.9\%\pm1.1\%\) |
| 圆柱尾迹训练的 Velocity RL | \(4.1\%\pm2.0\%\) |
| double gyre 重新训练的 Velocity RL | \(87.4\%\pm3.1\%\) |

圆柱尾迹策略直接迁移到 double gyre 后甚至显著差于 naive，说明网络学到的不是一个普适“看到速度就知道怎么走”的控制律，而是**与训练流场结构强耦合的局部决策映射**。坐标旋转和缩放也不能解决这一问题。

但在新流场中重新训练后，Velocity RL 又能达到较高成功率，说明方法本身可以适应不同流场，问题在于**策略迁移能力**而不是学习能力。

# 关键贡献

1. **将局部流动传感 + Deep RL 用于非定常二维流场的闭环导航**，并通过随机起点、随机目标和随机流场相位避免策略依赖固定场景记忆。
2. **系统比较不同局部流动观测量**，发现局部速度 \((u,v)\) 远优于仿生涡量 \(\omega\)，圆柱尾迹成功率从约 47% 提升到约 100%。
3. **证明仅依赖局部当前观测的 RL 可以接近拥有完整全局流场信息的时间最优轨迹**。
4. 展示了策略对较大范围初始条件、目标位置和流场相位具有鲁棒性，但也明确暴露出**跨不同流场拓扑的零样本迁移失败**。
5. 从物理角度给出状态设计原则：对下一状态转移直接有预测力的观测量可能比更“高级”或更仿生的派生量更适合 RL。

# 局限

- **动力学过度简化**：游动体是无质量质点，能够瞬时改变方向，没有惯性、姿态、角速度、舵机/推力器带宽或控制延迟。
- **游动体不反作用于流场**：背景流预先给定；真实机器人特别是旋翼、扑翼或鱼形推进器会明显改变周围流动。
- **二维流场**：真实海洋和大气流通常是三维、多尺度且可能包含强湍流。
- **速度感知在模型中异常“理想”**：由于动力学线性叠加，当前局部速度几乎直接决定下一步平移；真实机器人中，速度并不能单独精确决定下一状态，因此 99.9% 的优势未必能原样迁移。
- **目标相对位置被精确给出**：定位误差、通信失效或目标本身运动都没有建模。
- **传感器复杂度有限**：只研究了少数点传感状态。论文提到分布式压力/剪切传感与最优传感器布局可能更强，但没有实现。
- **最优控制比较规模有限**：正文只展示三组典型起终点轨迹；最优控制器本身依赖全时空流场并且是开环，二者信息条件并不对等。
- **跨流场泛化很弱**：圆柱尾迹策略无法直接迁移到 double gyre，需要重新训练。
- **sim-to-real 未验证**：所有主实验均为数值环境。论文提到对传感器噪声具有一定鲁棒性，但具体噪声设置位于补充材料，当前 PDF 正文没有给出详细参数。

# 对后续研究的启发

## 1. 状态设计比“换 RL 算法”可能更关键

本工作最强的 ablation 不是网络结构，而是输入状态：从 \(\omega\) 换成 \((u,v)\) 就把成功率从约 47% 推到约 100%。在流体控制/飞行控制中，值得优先问：

> 哪些传感量直接参与或强预测下一步状态转移？

例如对真实飞行器，局部气流速度之外，还可能需要姿态、角速度、空速、攻角、侧滑角以及执行器状态，从而使观测更接近 Markov state。

## 2. 随机化是防止“记地图”的必要训练设计

随机起点、随机目标、随机流场相位，再加上只提供目标相对位置，实际上构成了一种早期的 **domain randomization / task randomization** 思路。它迫使网络学局部闭环规律，而不是固定绝对位置到动作的查表关系。

## 3. 当前问题仍带有部分可观测性

虽然 Velocity RL 的状态更有预测力，但单个时刻的局部速度并不能确定未来的非定常流场，因此严格来说仍是部分可观测问题。本文仍采用前馈策略网络，没有显式构造历史隐状态。后续可以考虑：

- Recurrent Neural Network（RNN，循环神经网络）或 Transformer 处理观测历史；
- 多点/分布式流动传感；
- 在线系统辨识或 belief state，先估计当前流场相位/流型，再导航。

## 4. 迁移失败说明需要“跨流场训练”，而不仅是局部速度输入

圆柱尾迹到 double gyre 的 4.1% 成功率说明局部速度本身不自动带来通用策略。若希望做更通用的飞行/航行智能体，可以考虑：

- 多种流场联合训练；
- 流场参数随机化；
- meta-RL / context-conditioned policy；
- 利用短历史序列在线识别当前流场 regime；
- 在策略中显式加入动力学模型或局部预测模型。

# 文献脉络

## Colabrese et al. — *Flow navigation by smart microswimmers via reinforcement learning*

较早展示了智能微游动体可以通过 RL 利用涡流进行导航，并讨论了在拓扑相似流场之间的策略迁移。本文在其基础上进一步测试了拓扑明显不同的 double gyre，结果表明这种迁移并不普遍成立。

## Biferale et al. — *Zermelo's problem: optimal point-to-point navigation in 2D turbulent flows using reinforcement learning*

把 RL 与 Zermelo 时间最优导航联系起来，说明 RL 能接近时间最优轨迹。但其设置更接近对重复、确定的湍流快照进行导航，并使用较粗的状态/动作离散。本文进一步处理真正非定常流、随机起终点，并把局部流动传感作为核心变量。

## Novati & Koumoutsakos — *Remember and Forget for Experience Replay*

提供本文使用的 V-RACER / ReF-ER 方法基础。其重点是连续动作离策略强化学习中的经验回放稳定性：通过筛掉与当前策略差异过大的历史样本，并用 KL 散度约束策略漂移，提高数据利用率和稳定性。本文将这一算法用于局部传感驱动的流场导航。

# 关联

- [[Reddy 等 - 2016 - Learning to soar in turbulent environments]]：同样关注利用环境流动而非单纯抵抗流动，强调局部环境信息与策略学习。
- [[Reddy 等 - 2018 - Glider soaring via reinforcement learning in the field]]：真实场景中的局部风场估计 + RL，与本文“局部速度感知优于全局先验依赖”的思路直接相关。
- [[Novati 等 - 2019 - Controlled gliding and perching through deep-reinforcement-learning]]：同样采用深度强化学习解决连续流体/飞行动力学控制，可对比状态、动作和奖励设计。

# 外部概念补充来源

- V-RACER / ReF-ER：Novati, G. & Koumoutsakos, P., *Remember and Forget for Experience Replay*, ICML 2019, PMLR 97:4851–4860。
- Double gyre 标准方程：Shadden, S. C., Lekien, F. & Marsden, J. E., *Definition and properties of Lagrangian coherent structures from finite-time Lyapunov exponents in two-dimensional aperiodic flows*, Physica D 212, 271–304。
