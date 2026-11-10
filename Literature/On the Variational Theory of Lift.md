---
title: "On the Variational Theory of Lift"
authors:
  - "Haithem E. Taha"
year: 2026
venue: "AIAA Journal"
doi: "10.2514/1.J066884"
url: "https://doi.org/10.2514/1.J066884"
published_online: 2026-11-04
source: "未提供本地 PDF（全文由当前对话附件提供，尚未归档至 Literature-Vault）"
status: read
tags:
  - variational-theory-of-lift
  - potential-flow
  - Kutta-condition
  - Appellian
  - Gauss-principle
  - minimum-pressure-gradient
---

> [!abstract] 摘要译述
> Kutta–Zhukovsky 的经典升力理论主要适用于具有尖锐尾缘的翼型。对于圆钝尾缘，势流方程和无穿透条件仍允许多个环量解，而经典 Kutta 条件不再能通过消除尾缘奇异性选出唯一解。本文从分析力学出发，澄清基于高斯最小约束原理的升力变分理论：回顾最小作用量、高斯原理、Hertz 最小曲率原理、约束力与外加力、虚功与实际功，以及变分法中的边界条件处理；利用 Helmholtz 分解说明不可压缩约束的几何结构，证明在满足无穿透边界条件时，压力力与所有运动学允许的速度场正交，因此压力梯度是维持连续性和无穿透约束的反作用力。将高斯原理推广到连续介质后得到最小压力梯度原理。本文还讨论经典理论与变分理论的适用范围：任何基于稳态、无旋流动的理论都具有可逆性，因而不能用于反向流动构型；在适用范围内，变分理论给出环量和升力随几何参数连续变化的预测。

# 一句话定位

**这是一篇对既有 Variational Theory of Lift（升力变分理论）进行数学澄清、回应学术质疑并界定适用范围的论文，不是最初提出该理论的论文，也不是一篇提出新 CFD 求解器或报告新风洞实验的论文。**

论文的核心思想是：对于不可压缩、稳态、无黏势流，控制方程与无穿透边界条件并不能唯一确定环量；可以在运动学允许的候选流场中，选择使 Steady Appellian 最小的流场，从而确定环量和升力。尖锐尾缘极限下，该选择恢复 Kutta 条件。

## 1. 问题与动机：势流中的环量非唯一性

对二维简单连通物体，经典环量理论可将速度场写为

\[
\mathbf u(\mathbf x)=\mathbf u_0(\mathbf x)+\Gamma\mathbf u_1(\mathbf x),
\]

其中：

- \(\mathbf u_0\) 是无环量基础流场；
- \(\mathbf u_1\) 是单位环量对应的速度场；
- \(\Gamma\) 是待确定的环量。

上述速度场族中的每个成员都可以满足稳态 Euler 方程、不可压缩连续性方程、物体表面无穿透条件和远场条件。因此，仅凭这些方程不能唯一确定 \(\Gamma\)。

二维 Kutta–Joukowski 定理给出

\[
L'=\rho U_\infty\Gamma.
\]

所以环量不唯一就意味着升力不唯一。

对于数学尖锐尾缘，除一个特殊环量外，其余解通常在尾缘产生速度奇异性。要求尾缘速度有界，就能选出 Kutta 环量。对于圆钝尾缘，尾缘不再有必须消除的奇异性，Kutta 条件因而失去唯一选择解的作用。

作者希望寻找一个不依赖数学尖锐尾缘、能够处理平滑几何的闭合原则。

## 2. 分析力学基础

### 2.1 高斯最小约束原理

设粒子受到外加力 \(\mathbf F_i\)，如果不存在约束，其自由加速度为

\[
\mathbf a_i^{\mathrm{free}}=\frac{\mathbf F_i}{m_i}.
\]

如果系统还受到运动学约束，实际加速度必须位于允许的加速度集合中。高斯原理提出：在所有运动学允许的加速度中，自然运动选择使其偏离自由加速度最小的那个。

Jacobi 给出的二次代价为

\[
\boxed{
Z=\frac12\sum_{i=1}^{N}
m_i\left|\mathbf a_i-\frac{\mathbf F_i}{m_i}\right|^2
}
\]

并在加速度约束下最小化。

对于光滑动力学，该优化问题的一阶最优性条件恢复 Newton 方程。因此，高斯原理不是一个任意经验损失，而是 Newton 动力学的一种等价变分表达。

### 2.2 Hertz 最小曲率原理

当系统不存在外加力时，自由运动是匀速直线运动，即测地运动。偏离直线运动的程度可由曲率衡量。因此，高斯最小约束原理在无外加力的特殊情况下退化为 Hertz 最小曲率原理。

这也是论文中“least curvature”和 Appellian 思路的力学来源。

### 2.3 外加力与约束力

作者强调两种力不能混淆：

- **外加力（impressed force）**：其存在和本构关系独立于运动学约束。
- **约束力（constraint force）**：其唯一作用是确保约束成立；移除约束后，该约束力也随之消失。

约束力由运动方程与约束共同决定，而不是仅由约束单独决定。

### 2.4 虚功与实际功

分析力学中，“约束力不做功”通常指约束力不做**虚功**，并非实际运动中的功一定为零。

- 实际功沿真实运动轨迹计算。
- 虚功沿所有允许的虚位移计算。
- 约束力与允许虚位移正交，因此虚功为零。

这一差异是论文回应相关质疑的重要基础。

## 3. 不可压缩流体中的压力：Helmholtz–Leray 分解

对于满足适当正则性条件的向量场，可作正交分解：

\[
\mathbf v=\mathbf w+\nabla f,
\]

其中

\[
\nabla\cdot\mathbf w=0,\qquad
\mathbf w\cdot\mathbf n=0
\quad\text{on }\partial\Omega.
\]

利用分部积分：

\[
\begin{aligned}
\int_\Omega\nabla f\cdot\mathbf w\,d\Omega
&=-\int_\Omega f\nabla\cdot\mathbf w\,d\Omega
+\int_{\partial\Omega}f\mathbf w\cdot\mathbf n\,dS\\
&=0.
\end{aligned}
\]

因此，梯度场与满足散度为零、无穿透边界条件的速度场正交。

不可压缩 Navier–Stokes 方程可整理为

\[
\mathbf f-\mathbf u\cdot\nabla\mathbf u
=
\mathbf u_t+\frac{1}{\rho}\nabla p.
\]

左侧是未施加压力约束时的自由加速度；压力梯度将其投影到满足不可压缩和无穿透约束的允许运动空间。

由此，作者将压力梯度解释为不可压缩约束的反作用力，而不是任意的外加驱动力。

**重要前提：** 内部散度为零还不够；无穿透条件或等价的已规定边界法向速度，是上述 Helmholtz 正交性成立的必要组成部分。

## 4. 从高斯原理到最小压力梯度原理

### 4.1 黏性不可压缩流动

忽略重力和电磁体力时，论文给出的代价函数为

\[
\boxed{
\mathcal A=
\frac12\int_\Omega
\rho\left|
\mathbf u_t+\mathbf u\cdot\nabla\mathbf u
-\nu\nabla^2\mathbf u
\right|^2d\Omega
}
\]

其中 \(\nu\) 是运动黏度。

作者称其为 Principle of Minimum Pressure Gradient（PMPG，最小压力梯度原理）：在每个时刻，在满足连续性和无穿透约束的候选运动中，选择需要最小压力约束力的运动。

对该代价函数施加运动学约束，其一阶最优性条件恢复 Navier–Stokes 方程。

### 4.2 无黏理想流体

当黏性项消失时，得到 Appellian：

\[
\boxed{
\mathcal S=
\frac12\int_\Omega
\rho\left|
\mathbf u_t+\mathbf u\cdot\nabla\mathbf u
\right|^2d\Omega
}
\]

对于稳态流动，\(\mathbf u_t=0\)，因此

\[
\boxed{
S_s[\mathbf u]
=
\frac12\int_\Omega
\rho|\mathbf u\cdot\nabla\mathbf u|^2d\Omega
}
\]

这就是稳态升力变分理论所使用的 Steady Appellian。

在稳态无黏 Euler 方程下：

\[
\nabla p=-\rho(\mathbf u\cdot\nabla)\mathbf u.
\]

因此，在恒定密度不可压缩流体中，最小化 Steady Appellian 等价于最小化压力梯度平方的体积分。

**它不是最小化翼面压力差，也不是直接最小化升力。** 它通过选择整个流场，间接决定环量和升力。

## 5. 升力变分理论的具体实现

作者先定义所有满足稳态 Euler 方程和运动学约束的流场集合 \(\Theta\)：

\[
\min_{\mathbf u\in\Theta}S_s[\mathbf u].
\]

这个问题本来是无限维优化。对于附着流动，若采用经典势流理论给出的候选族

\[
\mathcal U=
\{\mathbf u_0+\Gamma\mathbf u_1\}
\subset\Theta,
\]

则问题可约化为一维环量优化：

\[
\boxed{
\Gamma^*
=
\underset{\Gamma}{\arg\min}
\ S_s[\mathbf u_0+\Gamma\mathbf u_1]
}
\]

由于速度对 \(\Gamma\) 线性，非线性对流加速度对 \(\Gamma\) 为二次，Steady Appellian 因此是关于 \(\Gamma\) 的四次多项式。

作者在所研究的算例中发现每个迎角对应唯一的最小值。

### 尖锐尾缘与圆钝尾缘

论文采用 Modified Zhukovsky Airfoil，以参数 \(D\) 控制尾缘平滑程度：

- \(D=0\)：经典数学尖锐尾缘；
- \(D=0.001\)：接近尖锐尾缘；
- \(D=0.05\)：平滑尾缘；
- \(D=1\)：映射退化为圆柱。

在数学尖锐尾缘极限下，除 Kutta 环量外，其余环量对应的 Steady Appellian 发散。因此，最小化原则恢复 Kutta 条件。

对于平滑尾缘，所有候选流场都不再具有上述尾缘奇异性，最小化原则仍然可以选择唯一环量。

## 6. 论文中的数值例子与证据边界

### 6.1 Figure 6：Modified Zhukovsky Airfoil

论文第 8 页的 Figure 6 比较了不同环量下的归一化 Appellian：

\[
\hat S=\frac{S_s}{\rho U_\infty^4},
\qquad
\hat\Gamma=\frac{\Gamma}{4\pi U_\infty a}.
\]

设置包括：

- 前后非对称参数 \(\tau=0.1\)；
- 近尖锐尾缘 \(D=0.001\)；
- 平滑尾缘 \(D=0.05\)；
- 不同攻角下的环量扫描；
- 将平滑尾缘算例的最优环量与 RANS 预测进行对照。

论文报告：

1. 近尖锐尾缘时，Appellian 的最小值与 Kutta 环量吻合。
2. 圆钝尾缘时，最小 Appellian 对应的环量与 Kutta 预测出现差异。
3. 在所展示的圆钝尾缘算例中，变分理论的环量更接近 RANS 预测。

图中 \(D=0.05\) 对应约为弦长 0.1% 的尾缘半径。

注意：Figure 6 是对文献 [8] 的复现，并非本文新增的独立风洞或 CFD 实验。RANS 是该算例中的数值比较基准，不等同于实验真值。

### 6.2 Figure 8～10：可逆性与反向流动

对于稳态无旋流动，如果 \(\mathbf u\) 是解，则 \(-\mathbf u\) 也满足相应的稳态理想流动方程及运动学条件。

压力取决于速度平方，因此正向与反向流动会得到相同的压力分布。这与实际反向飞行构型通常具有相反压力推力的经验不一致。

作者据此认为，任何建立在稳态无旋假设上的理论，包括经典升力理论和变分升力理论，都不能直接用于反向流动构型。

### 6.3 Figure 15：分离流扩展

论文还讨论了圆柱分离流的扩展思路：在自由流线模型中，以分离角 \(\beta\) 参数化候选流场，再通过 Appellian 选择分离角。

文中给出的先前研究结果为：

\[
\beta^*=83.7^\circ,
\qquad
\beta_{\mathrm{empirical}}=83.85^\circ.
\]

这是引用的先前工作，不是本文首次提出的实验结果。

## 7. 对 Peters–Ormiston 质疑的回应

论文第 IV～VI 节主要处理三个问题：

1. **压力力的性质**：压力梯度由不可压缩约束和无穿透边界条件决定，因此作者认为它属于约束力，而不是可以任意分解的外加力。
2. **变分边界条件**：在变分法中，可以通过限制允许变分直接施加边界条件，不必把每个边界条件都重新写成带 Lagrange 乘子的侧约束。
3. **虚功与实际功**：约束力的“无功”指虚功为零，不意味着实际功必然为零。

作者还强调，运动方程和约束条件本身存在非唯一解时，约束力的某个分量仍可能不唯一；这并不能证明该力属于外加力。

## 8. 适用范围与局限

| 方面 | 论文明确的范围 |
|---|---|
| 基本模型 | 稳态、无黏、不可压、无旋流动 |
| 主要升力应用 | 二维附着流、适当的参数化候选流场 |
| 尾缘 | 可以扩展到平滑尾缘 |
| 反向流动 | 不适用 |
| 前后对称几何 | 在稳态无旋框架下预测零升力 |
| 非定常流动 | 需要扩展的时间相关变分模型 |
| 分离流 | 需要更丰富的候选流场，例如自由流线模型 |
| 黏性作用 | 不能凭稳态闭合理论否定黏性在升力生成中的作用 |

特别需要强调：论文讨论了如何进一步将 PMPG 扩展到非定常流动、分离流和高维数值计算，但并不意味着文中所有扩展都已成为经过完整物理验证的通用求解器。

## 9. 相关研究脉络

| 文献 | 主要贡献 | 与本文关系 |
|---|---|---|
| Gonzalez & Taha, 2022, *A Variational Theory of Lift*, JFM 941, A58 | 首次提出基于 Appellian 的升力变分闭合 | 理论起点 |
| Taha, Gonzalez & Shorbagy, 2023, *A Minimization Principle for Incompressible Fluid Mechanics*, Physics of Fluids 35, 127110 | 建立 PMPG 的连续介质基础 | 将高斯原理推广到流体 |
| Taha & Gonzalez, 2023, *Refining Kutta’s Flow over a Flat Plate: Necessary Conditions for Lift*, AIAA Journal 61, 2060–2068 | 研究平板与前后对称几何 | 讨论稳态无旋理论的对称性限制 |
| Peters & Ormiston, 2025, *Application of Gauss’s Principle to the Classical Airfoil Lift Problem*, AIAA Journal 63, 6–20 | 对变分理论提出质疑 | 本文直接回应的工作 |

## 10. 关键结论

1. Kutta Condition 是升力变分理论在数学尖锐尾缘极限下的特殊情况。
2. 对平滑尾缘，可以使用 Appellian 最小化来选择环量，而不依赖尾缘奇异性消除。
3. 压力梯度在不可压缩、满足无穿透条件的流动中，是约束反作用力；这依赖 Helmholtz–Leray 正交性。
4. 虚功与实际功必须严格区分。
5. 稳态无旋理论具有可逆性，因此不能直接处理反向流动。
6. 稳态升力闭合不能独立解决升力从零到稳态建立的完整非定常过程。

## 关联

- [[Literature/Aerodynamic Design via Control Theory|Aerodynamic Design via Control Theory]]
- [[Literature/Topics/气动优化与设计空间探索|气动优化与设计空间探索]]
- [[Literature/Topics/机翼生成器研究对照与证据边界|机翼生成器研究对照与证据边界]]
- [[Gauss's Principle of Least Constraint]]
- [[Helmholtz–Leray Projection]]
- [[PDE-constrained optimization]]

## 阅读建议

优先读原始 2022 JFM 论文理解升力变分理论的提出，再读 2023 PMPG 论文理解一般不可压缩流体的数学基础，最后回到本文，重点看第 III、V、VII、VIII 节。本文的主要价值在于澄清理论结构、争议和适用边界，而不是提供新的独立实验数据。
