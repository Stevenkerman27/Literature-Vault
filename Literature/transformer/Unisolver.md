---
title: "Unisolver: PDE-Conditional Transformers Towards Universal Neural PDE Solvers"
authors:
  - Hang Zhou
  - Yuezhou Ma
  - Haixu Wu
  - Haowen Wang
  - Mingsheng Long
year: 2025
venue: "Proceedings of the 42nd International Conference on Machine Learning (ICML 2025), PMLR 267"
doi: null
source: "/mnt/data/UniSolver.pdf"
tags:
  - neural-operator
  - PDE
  - transformer
  - conditional-transformer
  - scientific-machine-learning
aliases:
  - Unisolver
---

> [!abstract] 摘要翻译
> 近年来，深度模型逐渐成为求解偏微分方程（Partial Differential Equations, PDEs）的有前景工具，即神经 PDE 求解器。尽管基于仿真数据训练的神经算子或基于物理约束损失训练的模型能够较好地求解 PDE，但它们通常只适用于少量 PDE 实例，例如某一固定方程及有限范围的系数，因此难以泛化到多样 PDE，也就难以成为传统数值求解器的实用替代模型。本文提出 **Unisolver**：一种在多样 PDE 数据上联合训练、并显式以 PDE 信息为条件的 Transformer，目标是迈向能够覆盖广泛 PDE 的通用神经求解器。作者并非单纯扩大数据和参数规模，而是从 PDE 求解过程的数学结构出发，将决定 PDE 解的完整信息拆分为一组 **complete PDE components（完整 PDE 组件）**，并进一步分为 **domain-wise（全域作用）** 与 **point-wise（逐点作用）** 两类深层条件。Unisolver 将这些条件以解耦方式注入 Transformer 隐空间，在三个大规模 PDE 基准上获得一致的最优结果，并表现出较强的零样本泛化能力与可扩展性。

# 一句话概括

**Unisolver 的核心不是“用 Transformer 解 PDE”本身，而是把决定 PDE 解的物理信息系统化为一组完整条件，并按照“全域作用/逐点作用”分别编码，再通过类似条件归一化与门控的机制注入 Transformer。**

# 1. 问题与动机

## 1.1 现有神经 PDE 求解器的两条主线

论文将现有方法概括为两类：

1. **Physics-Informed Neural Networks（PINNs，物理信息神经网络）**
   - 把 PDE 方程本身写进损失函数；
   - 对单个 PDE 可以得到较准确的解；
   - 但遇到新的方程或任务通常需要重新训练，跨 PDE 泛化能力有限。

2. **Neural Operators（神经算子）**
   - 从大量输入—输出场数据中学习函数空间到函数空间的映射；
   - 相比 PINN 更适合直接推理；
   - 但很多方法只输入初始场或历史场，没有完整输入控制解的 PDE 信息。

作者用 Navier–Stokes 方程举例：即使初始流场完全相同，只要黏性系数或外力不同，未来状态就会不同。因此：

> **仅仅让网络从初始场“自己猜”隐含的 PDE 条件，本质上近似于一个逆问题，难度高且容易得到模糊的物理表征。**

## 1.2 “通用 PDE 求解器”的任务定义

设：

- $D\subset\mathbb R^d$：连续物理域；
- $\mathcal M=\{x_1,\dots,x_n\}$：对域的离散网格/采样点；
- $X$：初始条件或历史观测；
- $C_{\mathrm{PDE}}$：控制当前样本的 PDE 组件；
- $Y$：目标解。

Unisolver 要学习

$$
\mathcal G:(X,\mathcal M,C_{\mathrm{PDE}})\rightarrow Y,
$$

并希望在联合训练多个 PDE 后，能直接泛化到新的初始条件、系数、外力、边界条件，甚至未见过的 PDE 配置。

# 2. 核心理论动机：Complete PDE Components

## 2.1 振动弦方程作为动机示例

论文从固定端点的一维振动弦方程出发：

$$
\partial_{tt}u-a^2\partial_{xx}u=f(x,t),
\qquad (x,t)\in(0,L)\times(0,T),
$$

边界条件：

$$
u(0,t)=u(L,t)=0,
$$

初始条件：

$$
u(x,0)=\phi(x),\qquad
\partial_tu(x,0)=\psi(x).
$$

其解析解可写为

$$
u(x,t)=
\frac12\left[\Phi(x+at)+\Phi(x-at)\right]
+\frac{1}{2a}\int_{x-at}^{x+at}\Psi(\xi)\,d\xi
+\frac{1}{2a}\int_0^t d\tau
\int_{x-a(t-\tau)}^{x+a(t-\tau)}
f(\xi,\tau)\,d\xi .
$$

这个表达式直接展示了：

- 初始位移 $\phi$、初始速度 $\psi$ 会影响解；
- 系数 $a$ 在整个区域内以一致方式改变传播速度；
- 外力 $f(x,t)$ 的影响依赖具体空间—时间位置；
- 几何域 $L,T$ 会改变积分范围；
- 边界条件决定了解的延拓/约束方式。

因此，**PDE 的解由多种相互作用的方程组件共同决定，而不同组件对空间位置的作用方式并不相同。**

## 2.2 六类完整 PDE 组件

论文把控制 PDE 解的信息整理为：

| 分组 | PDE 组件 | 典型例子 | 作用方式 |
|---|---|---|---|
| 输入 | Initial condition | 初始场 $u(x,0)$ | 作为待演化状态 |
| Domain-wise | Equation symbols | 方程结构/符号 | 整个区域共享 |
| Domain-wise | Equation coefficients | 黏性、扩散、波速等系数 | 整个区域共享 |
| Domain-wise | Boundary condition type | Dirichlet / Neumann / Robin / Periodic | 整个区域共享其“类型” |
| Point-wise | External force | $f(x,t)$ | 不同位置值不同 |
| Point-wise | Domain geometry | 几何 mask / 网格区域 | 每个位置独立判断 |
| Point-wise | Boundary value function | 边界上的数值函数 | 依赖边界位置 |

### 为什么边界条件要拆成两部分？

边界条件同时包含：

- **类型**：例如 Dirichlet、Neumann、Robin；
- **值**：例如边界上的 $u(x)$、$\partial_n u(x)$ 等函数。

类型是全局语义，因此归入 domain-wise；具体边界值是空间分布的，因此归入 point-wise。

# 3. 方法：Universal Components Embedding

Unisolver 不把所有 PDE 信息粗暴拼接到输入通道，而是先按物理作用方式分别编码成深层条件。

## 3.1 Equation Symbols：用 LLM 编码 PDE 符号结构

作者使用 **LLaMA-3 8B** 对 PDE 的 LaTeX 表达式编码。

例如：

```text
u_{tt} - a^2 u_{xx} = f(x,t)
```

具体流程：

1. 把 PDE 的 LaTeX 字符串送入 LLaMA-3；
2. 取 LLM 最后一层 Transformer block 的输出；
3. 沿 token 序列维做平均；
4. 得到一个 **4096 维方程 embedding**；
5. 再通过 MLP 投影到 Unisolver 所需的隐藏维度。

关键设计：

- prompt 中只保留系数和外力的**符号名字**，例如使用 `a`，而不是把 $a=0.01$ 写入 LLM；
- 目的是让 LLM 负责理解“方程结构/符号语义”；
- 具体数值由其他组件编码器单独处理。

这使“符号结构”和“数值条件”被显式分工。

## 3.2 Domain-wise 其他组件

对于：

- PDE 数值系数；
- 边界条件类型；

作者使用两层线性层，中间使用 **SiLU（Sigmoid Linear Unit）** 激活：

$$
c_{\text{domain}}=\mathrm{Linear}_2\left(
\mathrm{SiLU}(\mathrm{Linear}_1(c))
\right).
$$

其中：

- 系数可看作实值向量；
- 边界类型类似类别标签。

## 3.3 Point-wise 组件

对于：

- 外力场；
- 几何二值 mask；
- 边界值函数；

它们本身就是定义在网格 $\mathcal M$ 上的物理场，因此作者使用与输入场相同的 **patch embedding** 方法，把二维/一维物理场转为 token 序列。

## 3.4 同类条件合并

同一类别中的多个深层条件直接相加：

- 所有 domain-wise 条件合并为 $C_{\text{domain}}$；
- 所有 point-wise 条件合并为 $C_{\text{point}}$。

作者认为，如果把每个条件都完全独立处理，会造成条件过度分离，削弱联合学习不同 PDE 时的表示能力。

# 4. PDE-Conditional Transformer

这是 Unisolver 最关键的网络结构创新。

## 4.1 Subspace Decoupling：把隐藏通道一分为二

设 Transformer 隐藏表示为

$$
X\in\mathbb R^{N_{\text{token}}\times d}.
$$

作者沿通道维将隐藏空间均分为两部分：

- 前一半专门接受 domain-wise 条件；
- 后一半专门接受 point-wise 条件。

从多头注意力角度，可以理解成：

> 一部分 attention heads 主要学习全域 PDE 条件的影响，另一部分 heads 主要学习逐点 PDE 条件的影响。

其目的不是减少交互，而是**降低两种物理条件在同一子空间中相互干扰**。

## 4.2 Deep Condition Aggregation

分别把两类条件通过 MLP 投影到对应半个子空间。

对 domain-wise 条件，因为所有 token 共享同一条件，所以会沿序列长度复制：

$$
\mathrm{repeat}(C_{\text{domain}}).
$$

point-wise 条件本身就是 token 序列，因此无需复制。

对任意条件参数 $\star\in\{\text{scale},\text{shift},\text{select}\}$，条件向量可写成：

$$
I_\star=
\mathrm{Concat}
\left(
\mathrm{MLP}_\star(C_{\text{domain}}).\mathrm{repeat},
\mathrm{MLP}_\star(C_{\text{point}})
\right).
$$

## 4.3 Scale–Shift–Select 条件注入

每个 Transformer block 中，条件既用于：

- **Scale**：缩放归一化后的隐藏表示；
- **Shift**：平移隐藏表示；
- **Select**：对注意力/FFN 输出做软门控。

Self-Attention 分支可概括为：

$$
\widetilde X_n=
I_{\text{select}}^{(A)}
\odot
\mathrm{SA}
\left(
I_{\text{scale}}^{(A)}
\odot
\mathrm{LN}(X_{n-1})
+
I_{\text{shift}}^{(A)}
\right)
+
X_{n-1}.
$$

Feed-Forward 分支为：

$$
X_n=
I_{\text{select}}^{(F)}
\odot
\mathrm{FF}
\left(
I_{\text{scale}}^{(F)}
\odot
\mathrm{LN}(\widetilde X_n)
+
I_{\text{shift}}^{(F)}
\right)
+
\widetilde X_n.
$$

其中：

- $\mathrm{SA}$：Self-Attention（自注意力）；
- $\mathrm{FF}$：Feed-Forward Network（前馈网络）；
- $\mathrm{LN}$：Layer Normalization（层归一化）；
- $\odot$：逐元素乘法。

最后一层 Transformer 输出之后，作者还会再次根据 PDE 条件进行 scale/shift，再线性投影到预测场。

## 4.4 与 FiLM / DiT 条件机制的关系

这部分论文借鉴了条件归一化类思想。

### FiLM

**FiLM（Feature-wise Linear Modulation，逐特征线性调制）** 的核心是让条件网络生成逐通道参数 $\gamma,\beta$，然后对特征做

$$
\gamma\odot x+\beta.
$$

Unisolver 的 `scale + shift` 本质上属于同一类条件调制思想，但它进一步加入 `select` 门控，而且把条件拆成 domain-wise / point-wise 两个子空间。

参考：Perez et al., 2018, *FiLM: Visual Reasoning with a General Conditioning Layer*。

### DiT

**DiT（Diffusion Transformer）** 的 adaLN / adaLN-Zero 结构同样让条件向量生成 LayerNorm 的 scale/shift，并额外生成 residual gate。

Unisolver 的 `scale–shift–select` 与这一设计非常接近，只是条件不再是扩散时间步/类别，而是完整 PDE 组件。

参考：Peebles & Xie, 2023, *Scalable Diffusion Models with Transformers*。

# 5. 完整网络流程

从输入到输出可整理为：

```text
Initial State X
    │
    ├─ Patch Embedding ─────────────────────────────┐
    │                                               │
Equation Symbols ─ LLaMA-3 ─ MLP ─┐                │
PDE Coefficients ───── MLP ────────┼─ Domain-wise ─┤
Boundary Type ───────── MLP ────────┘   Condition   │
                                                    │
External Force ─ Patch Emb ─────────┐               │
Domain Geometry ─ Patch Emb ────────┼─ Point-wise ─┤
Boundary Value ─ Patch Emb ─────────┘   Condition   │
                                                    ▼
                                     PDE-Conditional Transformer × N
                                     [LN → scale/shift → SA → select]
                                     [LN → scale/shift → FFN → select]
                                                    │
                                             De-patch / INR
                                                    │
                                                    ▼
                                                PDE Solution
```

# 6. 1D 任务中的 PolyINR 解码器

在 1D time-dependent PDE benchmark 中，Unisolver 并不是简单 de-patch 到固定时空网格，而是使用改造后的 **PolyINR（Polynomial Implicit Neural Representation，多项式隐式神经表示）** 作为解码器。

INR 的基本思想是学习一个坐标到物理量的连续函数：

$$
(x,t)\mapsto u(x,t).
$$

PolyINR 与常见的 sinusoidal positional encoding 不同，它通过在每层将特征与坐标的仿射变换做逐元素乘法，使网络逐层构造更高阶的多项式坐标交互。

因此在 1D benchmark 中：

- Transformer 负责把初始状态 + PDE 条件编码为全局/局部表示；
- PolyINR 根据任意时空坐标查询最终解。

论文仅在该 1D benchmark 中明确采用这种解码形式。

参考：Singh, Shukla & Turaga, 2023, *Polynomial Implicit Neural Representations for Large Diverse Datasets*。

# 7. 实验设置

## 7.1 三个大规模 benchmark

| Benchmark | 维度 | 分辨率 | 样本数 | 变化的 PDE 组件 | 主要目的 |
|---|---:|---:|---:|---|---|
| HeterNS | 2D + Time | $(64,64,10)$ | 15k | 系数、外力 | 精细测试组件泛化 |
| 1D time-dependent PDEs | 1D + Time | $(256,100)$ | 3M | 方程符号、系数、外力、边界 | 大规模预训练 + zero-shot |
| 2D mixed PDEs | 2D + Time | $(128,128,10)$ | 74.1k | 符号、系数、外力、几何、边界 | 多数据集联合训练 |

论文报告的平均单 A100 等效训练成本约为：

- HeterNS：$\sim 60$ GPU hours（该表是跨模型平均；附录中 Unisolver 本身为 24 GPU hours）；
- 1D：$\sim 3000$ GPU hours；
- 2D mixed：$\sim 800$ GPU hours。

## 7.2 统一评价指标

训练和评价都使用 **Relative L2 Error（相对 $L_2$ 误差）**：

$$
\mathrm{Relative\ L2}(u,\hat u)
=
\frac{\|u-\hat u\|_{L_2}}{\|u\|_{L_2}}.
$$

越低越好。

# 8. HeterNS：异构二维 Navier–Stokes

## 8.1 数据集

二维不可压缩 Navier–Stokes 涡量形式：

$$
\partial_t w + u\cdot\nabla w
=
\nu\Delta w+f(x).
$$

变化组件：

### 黏性系数

训练：

$$
\nu\in
\{10^{-5},5\times10^{-5},10^{-4},5\times10^{-4},10^{-3}\}.
$$

### 外力

$$
f(x)=0.1\left[
\sin\left(\omega\pi(x_1+x_2)\right)
+
\cos\left(\omega\pi(x_1+x_2)\right)
\right],
$$

训练：

$$
\omega\in\{1,2,3\}.
$$

共有 $5\times3=15$ 个 PDE 配置，每个配置 1000 个训练样本，共 15,000 个。

## 8.2 训练配置

Unisolver：

- 参数量：4.1M；
- attention dim：256；
- FFN dim：256；
- patch size：4；
- heads：8；
- head dim：32；
- Transformer depth：8。

训练：

- 300 epochs；
- batch size 60；
- Adam；
- 初始学习率 $5\times10^{-4}$；
- cosine annealing；
- Relative L2 loss；
- 单步预测训练；
- 测试时 autoregressive（自回归）滚动预测。

## 8.3 对比结果

| 设置 | FNO | ViT | 最佳其他方法附近 | Unisolver |
|---|---:|---:|---:|---:|
| Viscosity in-distribution | 0.0210 | 0.0156 | 0.0156 | **0.0098** |
| Viscosity zero-shot | 0.0574 | 0.0514 | 0.0514 | **0.0374** |
| Force in-distribution | 0.0975 | 0.0593 | 0.0593 | **0.0428** |
| Force zero-shot | 0.4316 | 0.3158 | 0.3158 | **0.1053** |

最突出的是 **外力 zero-shot**：

- 其他神经算子即使显式把外力拼接到输入，也很难泛化；
- Unisolver 从 0.3158 降到 0.1053；
- 相对第二名提升约 **66.7%**。

当黏性和外力同时 unseen 时，论文报告 Unisolver 仍平均提升约 **41.3%**。

极端案例 $\omega=0.5$：

- 所有比较模型相对误差均超过 0.5；
- Unisolver 为 **0.098**。

# 9. 1D Time-Dependent PDEs

## 9.1 训练 PDE 家族

训练集来自 PDEformer，通式为：

$$
\partial_tu
+
f_0(u)
+
s(x)
+
\partial_x\left(
f_1(u)-\kappa(x)\partial_xu
\right)=0,
$$

其中

$$
f_i(u)=c_{i1}u+c_{i2}u^2+c_{i3}u^3,
\qquad i=0,1.
$$

每个 $c_{ik}$：

- 以 0.5 概率设为 0；
- 否则从 $[-3,3]$ 均匀采样。

$\kappa(x)$ 和 $s(x)$ 可以是：

- 0；
- 常数；
- 空间函数。

边界条件：

- 1.5M periodic；
- 1.5M non-periodic；
- non-periodic 的两端从 Dirichlet / Neumann / Robin 中随机选择。

因为零系数会导致对应项从 LaTeX prompt 中删除，最终形成 **576 种不同的方程符号形式**。

## 9.2 模型与训练

用于与 PDEformer-L 比较的 Unisolver：

- 19M 参数；
- attention dim 512；
- MLP dim 512；
- 8 Transformer layers；
- 8 heads；
- 8 PolyINR layers。

训练：

- 500 epochs；
- batch size 1024；
- Adam；
- learning rate $6\times10^{-4}$；
- cosine annealing；
- Relative L2。

## 9.3 Zero-shot 测试

使用另一个数据源 PDEBench：

### Burgers

$$
\partial_tu+\partial_x(u^2/2)
=
\frac{\nu}{\pi}\partial_{xx}u.
$$

测试：

$$
\nu=0.1,\ 0.01,\ 0.001.
$$

### Advection

$$
\partial_tu+\beta\partial_xu=0,
\qquad \beta=0.1.
$$

PDEBench 与训练集时空区域不同，因此作者先进行坐标变换，再相应变换 PDE 系数与符号，最后才进行 zero-shot 推理。

## 9.4 结果

| Task | PDEformer | Unisolver | 提升 |
|---|---:|---:|---:|
| In-distribution | 0.0225 | **0.0108** | 52.0% |
| Burgers $\nu=0.1$ | 0.00744 | **0.00513** | 31.0% |
| Burgers $\nu=0.01$ | 0.0144 | **0.00995** | 30.9% |
| Burgers $\nu=0.001$ | 0.0393 | **0.0299** | 23.9% |
| Advection $\beta=0.1$ | 0.0178 | **0.0138** | 22.5% |

四个 zero-shot 任务相对 PDEformer 平均提升约 **27.1%**。

# 10. 2D Mixed PDEs

## 10.1 数据组成

该 benchmark 来自 DPOT，汇总四个 benchmark 的 12 个子数据集：

- FNO Navier–Stokes；
- PDEBench：
  - Compressible Navier–Stokes；
  - Diffusion–Reaction；
  - Shallow-Water；
- PDEArena；
- CFDBench。

覆盖变化：

- 黏性系数；
- Mach number；
- shear viscosity；
- 外力；
- 边界条件；
- 几何。

## 10.2 “组件是否存在”的 binary mask

不同数据集并不拥有完全相同的 PDE 组件。

例如：

- FNO-NS 有变化的 viscosity；
- shallow-water 中不存在这个特定组件。

因此作者为每个组件增加一个二进制存在标记：

- `1`：组件有效；
- `0`：该 benchmark 中不存在该组件。

这一 mask 输入到对应组件编码器，防止模型把“不存在”误解成“数值为 0”。

## 10.3 模型与训练

Unisolver：

- 33M 参数；
- attention dim 768；
- MLP dim 768；
- patch size 8；
- heads 8；
- head dim 96；
- depth 6。

训练：

- batch size 320；
- 1000 epochs；
- AdamW；
- learning rate $10^{-3}$；
- OneCycle scheduler；
- 200 warmup epochs；
- 单步预测训练；
- autoregressive 测试。

## 10.4 结果

Unisolver 在 12 个测试集中的 **11 个优于 DPOT**。

平均 Relative L2：

- DPOT：$5.50\times10^{-2}$；
- Unisolver：$4.54\times10^{-2}$；
- 平均提升：**17.5%**。

在组件主导的任务中尤其明显：

- FNO-NS：系数泛化；
- PDEArena-Force：外力泛化；
- CFDBench：几何泛化。

# 11. 消融实验

## 11.1 LLM embedding 是否真的有用？

作者在 50k 规模的 1D 数据子集上做消融。

结果：

- 去掉 LLM embedding 性能最差；
- 用互相正交的随机向量替代，反而略好于完全去掉；
- LLaMA-3 相比无 LLM 平均提升约 **5.76%**。

这说明模型不仅需要一个“方程类别标记”，更受益于方程符号之间的结构关系。

## 11.2 “错误数学表达”实验

作者额外尝试：

1. 只让 LLM 编码非零项数量；
2. 把乘法改成除法、把多项式阶数改为倒数，故意输入错误数学关系；
3. 人工构造 one-hot 方程结构。

最终完整 LLM 表达在 in-distribution 与四个 zero-shot 任务中都最好。

相对人工表示平均提升约 **4.23%**。

作者谨慎表述为：

> 不能据此断言 LLM “理解”了数学，但至少可以确认它把有用的数学结构编码进了深层表示。

## 11.3 Subspace Decoupling

去掉 domain-wise / point-wise 子空间拆分后：

- zero-shot 性能平均下降约 **5.45%**。

说明两类条件若完全混合，会产生明显干扰。

## 11.4 直接 Concatenate 为什么差？

如果不使用条件调制，而是直接拼接 PDE 组件：

- 平均性能下降约 **21.0%**。

这支持论文的核心判断：

> **“有没有 PDE 信息”不是全部，关键还包括“以什么结构把 PDE 信息注入网络”。**

## 11.5 不完整 PDE 信息

作者在 HeterNS 中以 30% 概率把每个 PDE 组件替换成可学习缺失 token。

结果只有约 49% 样本仍拥有完整组件。

推理时：

- 即使不提供完整组件，Unisolver 多数情况下仍优于 FNO；
- 若提供完整组件，平均进一步提升 **21.6%**。

因此模型可以一定程度处理现实中的 component missing，但完整物理信息仍显著更好。

# 12. Joint Training 与新方程适应

## 12.1 多 PDE 联合训练是否只是“数据更多”？

作者构造三个互不重叠的 1D 子数据集，分别只包含 1、2、3 阶多项式 PDE。

相同模型：

| Polynomial Order | 独立训练 | 联合训练 | 联合训练提升 |
|---|---:|---:|---:|
| 1 | 0.0792 | **0.0555** | 29.9% |
| 2 | 0.1161 | **0.0738** | 36.5% |
| 3 | 0.1236 | **0.0695** | 43.7% |

说明不同 PDE 之间存在可迁移的共享结构，联合训练本身就是有效的 representation learning。

## 12.2 泛化到未见过的方程阶数

预训练：

- 只看最高 2 阶多项式 PDE。

微调：

- 3 阶 PDE；
- 200 epochs。

与从头训练 500 epochs 比较：

| Finetune Samples | From Scratch | Pretrain + Finetune | 提升 |
|---:|---:|---:|---:|
| 5k | 0.3308 | **0.1624** | 50.9% |
| 10k | 0.1913 | **0.1036** | 45.8% |
| 20k | 0.1327 | **0.0891** | 32.9% |

因此 Unisolver 对真正“新方程类型”的主要证据之一是 **快速微调适应能力**，而不仅是所有新方程都能完全 zero-shot 解决。

# 13. 可扩展性与效率

## 13.1 Data Scaling

训练数据从 50k 增加到 3M：

- in-distribution：0.0232 $\rightarrow$ 0.0106；
- Burgers $\nu=0.001$：0.1399 $\rightarrow$ 0.0299。

论文据此认为 Unisolver 有明显的数据 scaling 潜力。

## 13.2 Model Scaling

模型从 3M 增加到 63M 参数：

- in-distribution：0.0342 $\rightarrow$ 0.0156；
- Burgers $\nu=0.001$：0.1188 $\rightarrow$ 0.0889。

整体趋势随模型规模提升而改善。

## 13.3 推理效率

HeterNS 单帧预测，A100，batch size 1：

| 方法 | 时间 / s | 显存 / MB |
|---|---:|---:|
| FNO | 0.0042 | 730 |
| ViT | 0.0045 | 558 |
| Unisolver | **0.0054** | **554** |
| Numerical Solver | 7.26 | 524 |

Unisolver 的推理速度与常规神经算子同一量级，同时比用于生成数据的伪谱数值求解器快约三个数量级。

# 14. 关键贡献

1. **Complete PDE Components**
   - 不再只输入部分系数；
   - 明确把影响 PDE 解的主要信息拆成完整组件集合。

2. **Domain-wise / Point-wise 物理分类**
   - 分类依据来自 PDE 解析结构，而非纯经验网络设计；
   - 不同类别使用不同编码方法。

3. **PDE-Conditional Transformer**
   - 用 scale–shift–select 将 PDE 条件深度注入每个 Transformer block；
   - 比直接 concatenation 更有效。

4. **Subspace Decoupling**
   - 为全域条件与逐点条件分配不同隐藏子空间；
   - 降低两种条件的表示干扰。

5. **显式结合 LLM 的数学先验**
   - LLM 只负责方程符号结构；
   - 数值系数和物理场交给专门编码器。

6. **多 PDE 联合训练与 scaling**
   - 在 3M 级数据上展示 zero-shot、fine-tuning、data/model scaling。

# 15. 局限

## 15.1 论文明确指出：当前只支持规则网格

point-wise 组件采用 patchify，因此当前 Unisolver 主要面向 grid data。

论文认为：

- 大规模 irregular-mesh PDE 数据集不足；
- 不规则网格数据生成代价极高；
- 未来可将 canonical Transformer 替换为 [[Transolver]] 等 geometry-general backbone。

## 15.2 方程表达形式仍然敏感

同一个 PDE 可能有不同的等价写法。

当前 LLM embedding 直接基于 LaTeX，因此：

- 数学等价但字符串不同的表达可能得到不同 embedding；
- 作者建议未来先用 LLM / prompting 把不同写法规范化到统一形式。

## 15.3 “Universal” 仍是 towards universal

论文标题使用的是 **Towards Universal Neural PDE Solvers**。

从实验覆盖来看，模型已经覆盖多种方程、系数、外力、边界和几何，但仍不是对任意 PDE 的严格通用求解器。尤其是：

- 真正完全新型方程的证据更多来自 fine-tuning；
- irregular mesh 尚未验证；
- 多数训练仍依赖大规模高质量数值仿真数据。

# 16. 关联与研究脉络

## [[PDEformer]]

**PDEformer** 尝试把 PDE 符号表达构造成计算图，用 Graph Transformer 聚合符号与数值信息。

与 Unisolver 的差异：

- PDEformer：重点是“把方程表示成结构化图”；
- Unisolver：重点是“完整 PDE 组件 + 物理作用范围分类 + 条件 Transformer”。

论文在 1D 3M 数据 benchmark 上直接比较，Unisolver 用更少参数取得更低误差。

**脉络：**
PDEformer 证明“显式告诉网络 PDE 是什么”有效；Unisolver进一步追问“需要告诉哪些信息，以及这些信息应如何进入网络”。

## [[DPOT]]

**DPOT（Denoising Operator Transformer）** 的核心是扩大多 PDE 训练规模，并通过 denoising + autoregressive operator Transformer 学习跨数据集共享规律。

Unisolver 对它的批评是：

> 单纯依赖大规模数据，希望模型自己从观测场中隐式推断 PDE 物理信息，仍然会形成一个困难的隐式逆问题。

因此：

- DPOT 更强调 scale；
- Unisolver 强调 **scale + explicit PDE conditioning**。

在 2D mixed benchmark 上，Unisolver 平均相对误差从 5.50 降到 4.54。

## [[Transolver]]

**Transolver** 通过 Physics-Attention 把大量 mesh points 聚合为隐含物理状态 token，重点解决：

- general geometry；
- irregular mesh；
- 大规模网格的注意力效率。

Unisolver 与其解决的问题不同：

- Transolver：**空间几何/网格表示问题**；
- Unisolver：**PDE 条件表达与跨 PDE 泛化问题**。

论文在局限部分明确指出，两者可以组合：

> 用 Transolver 替换 Unisolver 当前 canonical Transformer，有望扩展到 irregular mesh。

因此二者更像互补关系，而不是竞争关系。

# 17. 方法脉络总结

可以把这条研究线压缩成：

```text
FNO / Neural Operator
    ↓
从数据学习 PDE operator，但 PDE 条件通常隐式
    ↓
CAPE / PROSE / PITT / PDEformer
    ↓
开始显式输入系数、符号或计算图
    ↓
DPOT / MPP
    ↓
进一步扩大多 PDE 预训练规模
    ↓
Unisolver
    ↓
完整 PDE 组件
+ domain-wise / point-wise 分类
+ LLM equation embedding
+ scale-shift-select conditional Transformer
+ multi-PDE joint training
```

而 Transformer 条件机制本身的来源可理解为：

```text
FiLM
    ↓
feature-wise scale / shift
    ↓
DiT / adaptive LayerNorm
    ↓
scale / shift + residual gate
    ↓
Unisolver
    ↓
用 PDE 条件生成 scale / shift / select
并将 domain-wise / point-wise 分配到不同子空间
```

# 18. 阅读结论

这篇工作的关键价值可以概括为：

> **把 PDE 物理信息从“额外输入特征”提升为 Transformer 的结构化条件变量。**

它认为神经 PDE 求解器要真正跨方程泛化，不能只靠：

- 更大的模型；
- 更多的仿真数据；
- 把系数简单 concatenate 到输入。

而应该同时回答三个问题：

1. **哪些物理信息完整决定当前 PDE？**
2. **这些信息是全局作用还是局部作用？**
3. **条件应该在网络内部以什么机制持续影响表示？**

Unisolver 的答案分别是：

- Complete PDE Components；
- Domain-wise / Point-wise；
- Decoupled Scale–Shift–Select Conditional Transformer。

---

## 外部概念补充来源

- Perez et al., 2018, *FiLM: Visual Reasoning with a General Conditioning Layer*: https://ojs.aaai.org/index.php/AAAI/article/view/11671
- Peebles & Xie, 2023, *Scalable Diffusion Models with Transformers*: https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf
- Singh, Shukla & Turaga, 2023, *Polynomial Implicit Neural Representations for Large Diverse Datasets*: https://openaccess.thecvf.com/content/CVPR2023/html/Singh_Polynomial_Implicit_Neural_Representations_for_Large_Diverse_Datasets_CVPR_2023_paper.html
- Unisolver official PMLR page: https://proceedings.mlr.press/v267/zhou25r.html

## 跨文献关联导航（2026-10-15 整理）

> [!note] 文献关系标注
> “方法承接”或“研究延展”表示现有笔记可支持的脉络；“比较”“互补”“类比”仅用于研究组织，不声称作者直接引用或实验证实联合方案。

**总览**：[[Literature/文献关系总览|文献关系总览]]

**所属主题**：[[Literature/Topics/Transformer与物理场算子|Transformer与物理场算子]] · [[Literature/Topics/机翼生成器研究对照与证据边界|机翼生成器研究对照与证据边界]]

**重点关联**：

- **条件注入对照**：[[Literature/transformer/Scalable Diffusion Models with Transformers|Scalable Diffusion Models with Transformers]] — adaLN 条件调制与 PDE 域级/点级条件分解可对比。
- **方法互补**：[[Literature/transformer/Transolver|Transolver]] — 物理注意力组织与 PDE 条件分层解决不同瓶颈。
