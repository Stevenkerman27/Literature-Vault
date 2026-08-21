---
title: "Efficient navigation in vortical flows based on reinforcement learning and flow field prediction"
authors:
  - Yuanpeng Zhang
  - Shizhan Zheng
  - Chao Xu
  - Shengze Cai
year: 2025
venue: "Ocean Engineering"
doi: "10.1016/j.oceaneng.2025.120937"
source: "/mnt/data/Zhang 等 - 2025 - Efficient navigation in vortical flows based on reinforcement learning and flow field prediction.pdf"
---

> [!abstract] 摘要翻译
> 本文研究复杂、时变流场中自主智能体的导航问题，并采用深度强化学习（Deep Reinforcement Learning, DRL）中的近端策略优化（Proximal Policy Optimization, PPO）求解点到点导航的 Zermelo 问题。困难在于智能体自身游动速度低于周围流速，因此不能简单地逆流抵抗，而必须主动适应并利用流场动力学。作者提出 **Look-Ahead State-Space（LASS，前视状态空间）**，通过让智能体预估若采取若干候选航向后未来可能到达的状态，将未来信息显式加入强化学习状态。前视所需的未来流场既可以使用真实流场，也可以由一个 **长短期记忆网络（Long Short-Term Memory, LSTM）+ 转置卷积网络（Transposed Convolutional Network, TransCNN）** 根据智能体自身历史传感数据预测得到。实验表明，LASS 显著提升了动态流场中的导航成功率和适应性；使用预测流场构造的 PredLASS 与使用真实未来流场的 TrueLASS 性能接近。与基于完整全局流场的最优控制（Optimal Control, OC）相比，OC 的航行时间略短，但规划计算开销大几个数量级，因此 PPO/LASS 更适合在线实时导航。

# 问题与动机

论文研究的是经典 **Zermelo 导航问题**：智能体在随时间和空间变化的背景流中，以固定的相对游动速度控制航向，目标是尽可能高成功率、尽可能短时间地从起点到达目标点。

传统最优控制、图搜索、采样规划等方法在已知全局流场时可以求近似最优路径，但真实海流、风场往往只能被局部测量，而且时变流场的全局预测本身也需要较大计算量。DRL 的优势是训练后只需局部观测即可在线输出控制动作，但已有工作存在明显的**状态信息不足与泛化范围受限**问题。

本文直接继承了 Gunnarson 等人的重要结论：与涡量、压力等局部物理量相比，**局部速度向量 $[u,v]$ 对导航最有用**。但作者指出，仅用

$$
S_t=[\Delta x,\Delta y,u,v]
$$

仍不足以让一个 PPO 策略同时适应更大范围的随机起终点，尤其是在需要从流场两侧双向导航时会出现明显偏置。因此本文的核心问题不是“还能换什么传感量”，而是：

> **能否把局部传感历史转化为“未来若干种动作会导致什么后果”的人工导航线索，从而提高 DRL 的决策可观测性和泛化能力？**

作者的答案是 **流场预测 + LASS + PPO**。

# 方法总览

整个方法可以拆成四层：

1. 智能体实时测得当前位置和局部背景流速：$I_t=[x,y,u,v]$；
2. 利用最近 $m=10$ 步历史 $[I_{t-9},\dots,I_t]$ 预测未来 $n=10$ 步、智能体周围 $128\times128$ 网格上的流速场；
3. 在 8 个预设候选航向下，用预测/真实未来流场向前模拟 10 步，得到 8 个“如果这样走，未来会到哪里、那里的流速是多少”的未来状态；
4. 将当前状态和这 8 个未来状态拼成 LASS，输入 PPO，输出当前真正执行的连续航向 $\theta$。

这里最重要的一点是：**LASS 并不是直接在 8 个预设方向中选动作。**这 8 个方向只是用于生成“前视特征”；PPO 最终动作仍是连续的 $\theta\in[0,2\pi]$。

## 1. 智能体运动学

智能体被简化为不影响流场的二维质点，绝对位置为 $\mathbf p=[x,y]^T$，以固定速度 $U_{swim}$ 相对流体游动，只控制航向角 $\theta$：

$$
\dot{\mathbf p}=\mathbf v_{flow}+U_{swim}
\begin{bmatrix}
\cos\theta\\
\sin\theta
\end{bmatrix}.
$$

离散仿真为：

$$
\mathbf p_{t+1}=\mathbf p_t+\left(\mathbf v_{flow}+U_{swim}
\begin{bmatrix}
\cos\theta\\
\sin\theta
\end{bmatrix}\right)\Delta t.
$$

所有实验都特意令智能体速度小于环境中的最大流速，因此某些区域不能靠“硬顶着流走”通过，必须绕行或借流。

## 2. 基础状态 BaseSS

Baseline State-Space（BaseSS，基础状态空间）为：

$$
S_t=[\Delta x,\Delta y,u,v]_{1\times4},
$$

其中 $[\Delta x,\Delta y]$ 是目标相对当前位置的位移，$[u,v]$ 是当前位置背景流速。

这与 Gunnarson 等工作中的 velocity swimmer 基本一致：状态只包含“目标在哪里”和“我现在被流往哪里推”。问题在于它只描述**当前**，不能直接表示当前流动结构在未来几步会如何影响运动。

## 3. 流场预测网络：LSTM + TransCNN

预测网络只使用智能体自身传感历史，不输入全局流场：

$$
I_t=[\mathbf p,\mathbf v_{flow}]=[x,y,u,v].
$$

输入为长度 $m=10$ 的序列，因此输入张量可理解为 $10\times4$。输出是未来 $n=10$ 个时刻，在智能体周围 $128\times128$ 网格上的二维速度场：

$$
\{u^{pred}_{t:t+9},v^{pred}_{t:t+9}\}.
$$

**论文中图1：流场预测网络结构**

![[assets/Efficient navigation in vortical flows based on reinforcement learning and flow field prediction/fig1.png]]

图 1 的实现逻辑为：

- 两层 LSTM 提取 10 步历史中的时间特征，隐藏维度为 256；
- 对未来不同时间步使用对应的全连接（Fully Connected, FC）分支，将时序特征变成低空间分辨率特征图；
- 之后经 4 层 TransCNN 逐级上采样，最终生成 $128\times128$ 的高分辨率 $u,v$ 场；
- 因此预测器学习的是“局部运动历史/局部流速历史 $\rightarrow$ 附近未来流场”的映射，而不是从一张当前全局流场外推未来。

### 预测网络训练数据

作者在两种流场中分别生成训练集：

- 使用随机动作让智能体在流场中运动并自动采样；
- 如果某些空间区域覆盖不足，再手动控制智能体补充样本；
- 为保证随机轨迹在时间上连续，动作满足

$$
a_t\sim\mathcal N(a_{t-1},(0.1\pi)^2);
$$

- 总数据量超过 150,000 个样本；
- 80% 训练、20% 测试；
- Adam 优化器，初始学习率 $7\times10^{-3}$，训练 200 epochs；
- Appendix A 中最终流场预测 MSE 约在 $10^{-2}$ 或更低量级。

作者还在转置卷积过程中使用均值上采样以平滑重构结果，并在 Double-Gyre 边界处用对称 padding 缓解边界问题。

## 4. Look-Ahead State-Space（LASS）

这是本文最核心的创新。

对于当前 BaseSS $S_t$，预先规定 8 个航向：

$$
\Theta=\{0,0.25\pi,0.5\pi,0.75\pi,\pi,1.25\pi,1.5\pi,1.75\pi\}.
$$

对每个候选方向 $\theta'$，假设未来 $n=10$ 步沿该方向运动，并结合未来流场逐步积分运动学方程，得到第 $t+n$ 步的未来 BaseSS：

$$
S^{\theta'}_{t+n}=[\Delta x',\Delta y',u',v'].
$$

随后拼接：

$$
S^{LASS}_t=
[S_t,
S^0_{t+n},
S^{0.25\pi}_{t+n},
\ldots,
S^{1.75\pi}_{t+n}]_{9\times4}.
$$

也就是从 **4 维当前观测**扩展为 **36 维当前 + 多个反事实未来结果**。

**论文中图2：BaseSS、TrueLASS 与 PredLASS 的构造**

![[assets/Efficient navigation in vortical flows based on reinforcement learning and flow field prediction/fig2.png]]

两种 LASS：

- **TrueLASS**：前视模拟直接使用真实未来流场；只适合仿真/已知流场，是性能上界；
- **PredLASS**：用 LSTM-TransCNN 预测未来流场，再做完全相同的 8 方向前视；这是面向实际部署的版本。

> [!note] 对 LASS 的理解
> 可以把 LASS 看成一种“显式模型前视特征”：它类似很浅的模型预测控制（Model Predictive Control, MPC）思想——先问“几个候选控制会把我带到哪里”，但并不直接优化前视轨迹，而是把这些结果交给 PPO 学习如何使用。换言之，**PPO 负责策略学习，LASS 负责把部分动力学后果显式化**。

## 5. PPO 控制器

PPO 使用 Actor-Critic 结构：

- Actor 根据状态输出连续动作分布的均值和方差；
- Critic 估计状态价值；
- 实验主体网络为 3 个隐藏层，每层 128 个神经元，权重和偏置做正交初始化。

PPO 损失由三部分组成：

$$
L=L^{CLIP}+c_1\operatorname{MSE}(\hat V(s_t),R_t)-c_2H[\pi_\theta].
$$

其中：

- $L^{CLIP}$：用概率比率 clipping 限制一次策略更新幅度；
- value MSE：训练 Critic；
- entropy：防止策略过早变得完全确定，维持探索。

> [!info] PPO clipping 为什么重要（外部概念补充）
> PPO 原论文的核心思想是：采集一批 on-policy 数据后，可以对同一批数据做多轮 minibatch 更新，但通过截断新旧策略概率比率，避免一次梯度更新把策略推得过远。它在 TRPO（Trust Region Policy Optimization，信赖域策略优化）的“限制策略变化”思想和实现简单性之间做了折中。本论文采用的就是这一 clipped surrogate objective。

动作空间仍为单个连续航向角：

$$
A=[\theta],\qquad \theta\in[0,2\pi].
$$

## 6. 奖励函数

单步奖励：

$$
r_n=-\Delta t
-\alpha\left[
\frac{d(p_n,p_{target})-d(p_{n-1},p_{target})}{U_{swim}}
\right]
+r_{success}.
$$

含义：

- $-\Delta t$：每走一步都付时间成本；
- 距离差项：只要比上一时刻更接近目标就获得正向奖励；
- $r_{success}$：到达目标时给予大额成功奖励。

整条轨迹累计后距离变化项望远镜式消去：

$$
r_{total}=-T_{total}+\alpha\frac{D}{U_{swim}}+r_{success}.
$$

因此对**给定起终点的一次成功 episode**，$D$ 是常数，最大化总奖励就等价于进一步压缩导航时间。

# 实验设置

作者选择两个拓扑和动力学特征差异明显的二维时变流场，且都令起点、终点和初始流场相位随机化。

**论文中图3：两个测试流场及随机起终点区域**

![[assets/Efficient navigation in vortical flows based on reinforcement learning and flow field prediction/fig3.png]]

## Double-Gyre

解析二维周期流：

- $A=0.6$，$\epsilon=0.3$，$\omega=6\pi$；
- 最大背景流速约 2.8；
- $U_{swim}=0.9$；
- $dx=dy=0.005$，$dt=0.01$；
- 两个随机采样圆心分别为 $[0.5,0.5]$ 和 $[1.5,0.5]$，半径均为 0.25；
- 每回合从其中一侧随机采样起点，另一侧随机采样目标，因此需要同时学会左右两个方向；
- 起始时间也随机；
- 成功半径 $r=0.02$；
- 最大 400 simulation steps；
- PPO 总训练长度 $4\times10^6$ simulation steps；
- 最终测试 10,000 episodes。

## Cylinder Flow

二维不可压缩圆柱绕流，Reynolds 数 400：

- 无量纲来流速度 1.0，圆柱直径 1.0；
- $U_{swim}=0.9$；
- $dx=dy=0.05$，$dt=0.15$；
- 起终点区域圆心 $[5.0,2.1]$ 与 $[5.0,-2.1]$，半径均为 2.0；
- 从上/下两侧随机互换起终点，必须穿越圆柱尾迹；
- 成功半径 $r=1/6$；
- 最大 400 simulation steps；
- PPO 同样训练到 $4\times10^6$ steps；
- 最终测试 10,000 episodes。

# 核心实验结果

## 1. Double-Gyre：LASS 将成功率从约 69% 提升到约 95%

| 方法 | 策略形式 | 成功率 | 成功轨迹平均时间 | 每次导航计算时间 |
|---|---|---:|---:|---:|
| PPO + BaseSS | online | 69.23% | 1.82 s | 0.134 s |
| PPO + TrueLASS | online | 95.45% | 1.61 s | 0.205 s |
| PPO + PredLASS | online | 94.90% | 1.68 s | 0.864 s |
| Optimal Control | offline | 100% | 1.20 s | 84 s |

主要结论：

- 仅增加“未来后果”状态信息，不改变 PPO 主体结构，成功率提高约 26 个百分点；
- 用预测流场构造 PredLASS 后，成功率只比 TrueLASS 低 0.55 个百分点；
- PredLASS 因多了流场预测推理，单次计算时间比 BaseSS/TrueLASS 高，但仍远低于 OC；
- OC 路径平均更快，但其 84 s 的离线规划时间与 PPO 的亚秒级在线推理不在同一数量级。

> [!warning] 论文中的一个数值不一致
> 正文称 TrueLASS 相比 BaseSS “平均导航时间减少 0.23 s”，但 Table 1 给出的 1.82 s 与 1.61 s 相差 **0.21 s**。这里保留论文原始表格数值，不替作者默默修正。

**论文中图6：预测流场与真实流场，以及由二者产生的前视轨迹**

![[assets/Efficient navigation in vortical flows based on reinforcement learning and flow field prediction/fig6.png]]

图 6 的关键信息不是像素级预测完全一致，而是：**PredLASS 最终得到的 8 条短期前视轨迹与 TrueLASS 很接近**。对控制器而言，这种“决策相关的一致性”比精确复原所有流场数值更重要。

**论文中图8：BaseSS / TrueLASS / PredLASS 与最优控制轨迹比较**

![[assets/Efficient navigation in vortical flows based on reinforcement learning and flow field prediction/fig8.png]]

同一初始条件下，图 8 中 OC 用时 1.40 s；BaseSS 为 2.22 s，TrueLASS 为 1.56 s，PredLASS 为 1.63 s。LASS 不仅提高是否能到达，也明显让策略更接近时间最优路径。

## 2. Cylinder Flow：BaseSS 的“双向偏置”被 LASS 基本消除

BaseSS 在圆柱尾迹中的问题更明显：同样网络结构和超参数下，不同训练会自发偏向“从上往下”或“从下往上”其中一个方向，另一个方向成功率很差。这说明仅靠 $[\Delta x,\Delta y,u,v]$，PPO 容易学成局部经验策略，而不是足够对称、足够可迁移的流场利用规律。

| 方法 | 成功率 | 成功轨迹平均时间 | 每次导航计算时间 |
|---|---:|---:|---:|
| PPO + BaseSS | 64.75% | 19.08 s | 0.128 s |
| PPO + TrueLASS | 98.91% | 17.18 s | 0.148 s |
| PPO + PredLASS | 98.29% | 19.52 s | 0.793 s |

TrueLASS 和 PredLASS 都把成功率提高到约 98–99%。

> [!warning] 第二个数值不一致
> 正文在 Cylinder Flow 部分也写成 TrueLASS “平均导航时间减少 0.23 s”，但 Table 2 的 19.08 s 与 17.18 s 实际相差 **1.90 s**。这是正文与表格的明显不一致。

此外，BaseSS 的平均成功时间和 PredLASS 相近并不代表效率相同：论文指出 BaseSS 主要只能完成目标更靠近中心、路径较短的简单样本，因此其“仅对成功样本统计的平均时间”存在选择偏差；PredLASS 能完成更长、更困难的任务。

**论文中图11：Cylinder Flow 中 PredLASS 的在线导航与局部流场预测**

![[assets/Efficient navigation in vortical flows based on reinforcement learning and flow field prediction/fig11.png]]

图 11 显示智能体在圆柱尾迹低速区附近形成明显的 zig-zag 航迹，并利用预测到的涡结构穿行；右侧真实/预测流场在 $t+1,t+5,t+9$ 的主要空间结构保持一致。

**论文中图12：TrueLASS 与 PredLASS 的双向导航结果**

![[assets/Efficient navigation in vortical flows based on reinforcement learning and flow field prediction/fig12.png]]

示例中上下两个方向均达到 20/20 成功，直观体现 LASS 对 BaseSS 单侧策略偏置的修复。

# 补充实验

## PPO 网络规模消融

作者比较 3×128、3×64、2×128、4×128 等 Actor-Critic 网络。3×128 整体最好：

- Double-Gyre + BaseSS：约 70%；
- Double-Gyre + TrueLASS：约 96%；
- Cylinder + BaseSS：约 66%；
- Cylinder + TrueLASS：约 98%。

一个值得注意的现象是：**使用 TrueLASS 后，不同网络规模之间的性能差距明显缩小。**这支持作者的核心论点——性能瓶颈并不只是网络容量不足，更关键的是状态是否包含足够的决策信息。

## 传感器噪声

作者对实测 $u,v$ 加零均值 Gaussian noise：

- BaseSS 的 Double-Gyre 成功率随噪声增加明显下降；
- PredLASS 在 $\sigma=0.2$ 时仍约为 94% 量级；
- Cylinder PredLASS 在测试噪声范围内仍接近 98%。

这说明前视状态相当于在时间和空间上对单点噪声进行了“动力学整合”，比直接把即时速度交给策略更稳健。

需要注意：Table 4 中 Double-Gyre PredLASS 在 $\sigma=0$ 时列为 95.45%，而主实验 Table 1 中 PredLASS 是 94.90%；论文没有解释这一差异，可能来自不同实验重复或表格记录差异。

## 流场预测误差：bias 比零均值 noise 更危险

作者直接对 TrueLASS 使用的未来流场加入不同误差：

- 零均值 Gaussian noise 对 Double-Gyre 的影响很小；
- Cylinder 直到 $\sigma>0.3$ 才出现明显退化；
- **均值偏移 $\mu$** 的影响明显更严重，尤其 Cylinder 对负偏置非常敏感，例如 $\mu=-0.3$ 时成功率降至 60.68%。

作者据此强调：流场预测不必追求每一点的极高数值精度，更重要的是保留对导航有用的**流动结构和模式**。从控制角度看，零均值误差在多步积分中可能部分抵消，而持续偏置会系统性地把所有 look-ahead 终点推向错误方向，因此对动作排序的破坏更大。

# 与最优控制的比较

OC benchmark 分两步：

1. 使用 **Time-Based Rapidly-exploring Random Tree（TB-RRT，时间扩展快速探索随机树）** 在状态-时间空间 $[x,y,t]$ 中先找一条满足动力学约束的可行轨迹；
2. 再用约束梯度优化不断缩短控制序列，并以到达目标为约束优化终点误差。

由于局部优化可能陷入局部最优，作者独立运行规划器 50 次，选最短结果近似全局最优。

> [!info] TB-RRT 是什么（外部概念补充）
> 普通 RRT 的树节点主要表示系统状态；TB-RRT 进一步把“时间”加入节点，使每个节点成为“某个具体状态在某个具体时刻”的状态-时间点。这样可以在目标/环境随时间变化时规划满足时间约束的轨迹。本论文正是借助这一点，把时变流速纳入每个节点的运动学传播。

OC 的优势是路径质量：它拥有完整真实未来流场，可以直接针对时间最优目标求解。劣势是需要全局信息和大量在线计算。本文的 Double-Gyre 统计中，OC 平均 84 s 才生成一次规划，而 PPO 系列只需约 0.1–0.9 s。因此作者把两者定位为：

- **OC：全局信息 + 离线高质量规划；**
- **PPO/LASS：部分观测 + 训练后在线闭环控制。**

# 关键贡献

1. **提出 LASS。**把当前单点观测扩展成多个候选航向对应的未来状态，使 RL 直接看到动作的短期动力学后果。
2. **把局部历史传感与前视连接起来。**通过 LSTM-TransCNN，只依赖智能体历史 $[x,y,u,v]$ 来预测局部未来流场，从而构造可实际使用的 PredLASS。
3. **显著改善起终点泛化。**同一 PPO 能处理 Double-Gyre 左右双向、Cylinder 上下双向随机起终点，解决 BaseSS 容易形成单侧策略的问题。
4. **PredLASS 接近 TrueLASS。**预测误差没有显著破坏导航成功率，说明面向控制的流场预测不一定需要高精度复原整个场。
5. **展示在线计算优势。**与 OC 相比牺牲少量路径最优性，换来数量级更低的决策计算成本。

# 局限与值得注意的问题

## 1. 动力学仍然非常理想化

智能体是二维质点：

- 无惯性和姿态动力学；
- 航向可以直接控制；
- 固定游速；
- 不改变周围流场；
- 没有执行器饱和、转向速率限制、推进功率模型。

因此当前结果更接近“大尺度流场中的路径决策层”，不能直接视为完整 AUV/无人机飞控策略。

## 2. 每个新流场仍需重新训练

论文明确指出，当前方案对每个新流场都需要重新训练流场预测模型和 PPO。虽然起终点泛化明显增强，但**跨流场泛化没有真正解决**。

这一点与 Gunnarson 的结论形成呼应：Gunnarson 的 cylinder 策略无法直接迁移到 Double-Gyre；本文不是用一个统一策略解决两个流场，而是在两个流场中分别训练，从而把泛化范围从“固定/局部起终点”扩展到了“同一流场内更大起终点区域”。

## 3. PredLASS 的 PPO 训练仍使用 TrueLASS

主实验中作者先训练 PPO + TrueLASS，然后在测试 PredLASS 时使用**同一个 PPO 模型**，只把真实未来流场换成预测流场。这很好地隔离了“预测误差”本身的影响，但也意味着实用部署流程仍依赖训练阶段能够获得高质量真实未来流场/仿真流场。

> [!note] 笔记理解
> 这可以视为一种“训练时特权信息、部署时估计信息”的结构：策略在干净的 TrueLASS 上学会如何解释 look-ahead 特征，然后预测器尽量在部署时复现这些特征。论文没有用“privileged information”这个术语，但从实验流程看，这是理解 TrueLASS→PredLASS 设计很有帮助的视角。

## 4. 预测器实际上在学习特定流场动力学

仅从 10 步 $[x,y,u,v]$ 推出附近未来 $128\times128$ 流场，在任意未知非周期三维环境中非常困难。当前两个环境均为低维、强结构、可重复的周期/准周期流场，因此历史局部观测对全局相位含有较强信息。作者也承认未来需要面向真实季节性海流等场景扩展。

## 5. “时间最优”比较不是同等信息条件

OC 使用完整全局真实流场，RL 只用局部/预测信息；因此两者比较的意义不是证明 RL 已达到同信息条件下的最优，而是展示**在现实信息与计算约束下，RL 能用多少路径质量换取多少实时性**。

# 对后续研究的启发

## 1. LASS 本质上是“把模型预测变成状态编码”

传统 model-based RL 往往让模型直接参与规划；本文采用更轻量的方式：模型只生成若干未来结果，最终控制仍由 model-free PPO 完成。对复杂飞行器/水下航行器，这种方式可能比完整在线 MPC 更容易扩展。

## 2. 预测目标不一定要是完整高分辨率流场

本文先预测 $128\times128\times10$ 的完整局部流场，再压缩成 8 个 $t+n$ 状态。实际上控制器最终只使用这些 look-ahead 结果，因此未来可以考虑直接学习：

$$
\text{history}+\theta'\rightarrow S^{\theta'}_{t+n}
$$

而不是先重建整个流场。这样可能显著降低预测网络计算量，也更符合“task-oriented prediction”。这是论文框架自然导出的改进方向，不是作者已验证的结果。

## 3. 可以把离散 8 方向改成可学习的 query

当前 8 个方向均匀覆盖 $2\pi$，是人工设计。后续可以让模型根据当前状态动态生成最有信息量的 look-ahead action queries，或者用 attention 在更密集的候选轨迹集合中选择。

## 4. 对真实载具，需要把前视状态从位置扩展到完整动力学状态

若考虑惯性、姿态、航速、能耗、侧滑等，未来状态可以从

$$
[\Delta x,\Delta y,u,v]
$$

扩展为例如

$$
[\Delta x,\Delta y,\mathbf v_{body},\psi,\dot\psi,E,\mathbf v_{flow},\ldots],
$$

此时 LASS 就从简单的路径前视变成动力学可达性前视。

# 关联文献脉络

## Gunnarson et al. — Learning efficient navigation in vortical flow fields

这篇是本文最直接的前置工作。其核心发现是：在非定常圆柱尾迹中，使用局部速度 $[u,v]$ 的 RL swimmer 远优于只看位置或局部涡量的 swimmer，并能接近全局最优控制路径；但策略对流场类型和任务分布敏感，cylinder 策略无法直接迁移到 Double-Gyre。

**本文承接点：**保留 $[\Delta x,\Delta y,u,v]$ 作为 BaseSS，不再争论“速度还是涡量”，转而解决“当前局部速度还不够描述未来”的问题，通过 LASS 增加前视线索。

## Mei, Kutz & Brunton — Observability-Based Energy Efficient Path Planning with Background Flow via DRL

该工作把背景流中的 DRL 路径规划与“可观测性”联系起来，强调移动传感和状态信息质量对能效路径规划的重要性。

**本文承接点：**LASS 可以理解为一种非常具体的可观测性增强方式——不是增加更多物理传感器，而是利用动力学模型和预测器把有限历史观测转成未来状态特征。

## Achermann et al. — WindSeer

WindSeer 使用稀疏、带噪风速测量和已知地形，通过神经网络实时重建复杂地形上的三维风场，并展示了在未见地形上的泛化能力。

**本文承接点：**两者都试图解决“全局流场不可直接获得”问题，但路径不同：WindSeer 重点是稀疏测量到空间流场重建；本文只用智能体自己的轨迹历史预测局部未来流场，并把预测结果进一步转成导航前视状态。

# 一句话总结

> **这篇论文最重要的不是把 PPO 换得更复杂，而是把“未来若往不同方向走会怎样”显式编码进状态：LASS 用短期动力学前视弥补局部即时观测的信息不足，流场预测则让这种前视在没有真实未来流场时仍可实现。**

# 外部概念与关联资料

- Schulman et al., *Proximal Policy Optimization Algorithms*: https://arxiv.org/abs/1707.06347
- Gunnarson et al., *Learning efficient navigation in vortical flow fields*: https://doi.org/10.1038/s41467-021-27015-y
- Mei et al., *Observability-Based Energy Efficient Path Planning with Background Flow via Deep Reinforcement Learning*: https://doi.org/10.1109/CDC49753.2023.10383428
- Achermann et al., *WindSeer: real-time volumetric wind prediction over complex terrain aboard a small uncrewed aerial vehicle*: https://doi.org/10.1038/s41467-024-47778-4
- Sintov & Shapiro, *Time-Based RRT Algorithm for Rendezvous Planning of Two Dynamic Systems*: https://doi.org/10.1109/ICRA.2014.6907855
