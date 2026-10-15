---
title: "Transolver: A Fast Transformer Solver for PDEs on General Geometries"
authors:
  - Haixu Wu
  - Huakun Luo
  - Haowen Wang
  - Jianmin Wang
  - Mingsheng Long
year: 2024
venue: "Proceedings of the 41st International Conference on Machine Learning (ICML), PMLR 235"
doi: null
source: "/mnt/data/Transolver.pdf"
arxiv: "2402.02366"
updated: "2026-09-06"
successor:
  title: "Transolver-3: Scaling Up Transformer Solvers to Industrial-Scale Geometries"
  year: 2026
  venue: "Proceedings of the 43rd International Conference on Machine Learning (ICML), PMLR 306"
  source: "/mnt/data/2602.04940v2.pdf"
  arxiv: "2602.04940v2"
tags:
  - neural-operator
  - transformer
  - PDE
  - scientific-machine-learning
  - unstructured-mesh
  - physics-attention
  - aerodynamic-surrogate
---

> [!abstract] 摘要翻译
> Transformer 已经在许多领域取得突破，并开始被用于求解偏微分方程（Partial Differential Equation, PDE）。但 PDE 通常被离散为规模很大、几何形状复杂的网格，直接让 Transformer 在海量单点之间建立关系，不仅计算代价高，也难以从这些低层网格点中学习复杂的物理相关性。本文提出 **Transolver**：不直接把“网格点”视为注意力的基本语义单元，而是学习离散几何背后潜在的**内禀物理状态**。核心模块 **Physics-Attention** 会把离散域自适应地软划分为若干可学习、形状自由的 slice，使具有相似特征、并在任务优化中逐渐呈现相似物理状态的网格点被分配到同一 slice；再把每个 slice 聚合成 physics-aware token，在这些 token 上做注意力，最后把更新后的 token 广播回原网格。该机制对复杂几何具有内生的适应能力，在 slice 数固定且远小于网格点数时，对网格规模呈线性复杂度。论文在 6 个标准 PDE 基准上取得一致最优结果，平均相对提升约 22%，并在汽车和翼型的大规模工业级仿真任务上表现出较好的精度、效率、扩展性和分布外泛化能力。

## 一句话理解

**Transolver 的核心创新不是新的 Q/K/V 公式，而是把 tokenization 本身变成可学习过程：先把大量 mesh point 动态软聚合为少量 latent physics tokens，再在这些 token 上做标准 self-attention，最后 DeSlice 回每个 mesh point。**

核心链条：

$$
\boxed{
X\in\mathbb R^{N\times C}
\xrightarrow{\text{soft slicing}}
Z\in\mathbb R^{M\times C}
\xrightarrow{\text{self-attention}}
Z'\in\mathbb R^{M\times C}
\xrightarrow{\text{deslice}}
X'\in\mathbb R^{N\times C}
}
$$

其中通常 $M\ll N$。

> [!tip] 阅读优先级
> 如果只想先真正理解实现，优先搞清楚四个对象：$w_{ij}$、$z_j$、$z'_j$、$x'_i$。附录的积分算子证明是理论解释，不是理解或实现 Transolver 的前置条件。

---

## 1. 问题与动机

### 1.1 为什么普通 Transformer 不适合直接处理 PDE 网格

PDE 数值问题通常先把连续域 $\Omega$ 离散为 $N$ 个网格点。工业问题中的 $N$ 很大，而且点可能是不规则的二维/三维点云、非结构网格或表面-体网格混合结构。

直接把每个网格点当作一个 Transformer token 有两个问题：

1. **计算复杂度**：标准自注意力需要构造 $N\times N$ 的相关矩阵，复杂度随 $N^2$ 增长。
2. **关系学习困难**：PDE 中真正重要的是压力区、尾流、激波、材料挤压区等高阶物理结构，而不是单个离散点本身。即使把 attention 线性化，仍然是在海量单点之间建立关系，作者认为容易得到信息量较低、接近均匀的注意力。

传统视觉 Transformer 的 patchify 也不能直接解决这个问题，因为规则矩形 patch 不适用于任意非结构几何，而且固定空间块未必对应相同物理状态。

### 1.2 作者的核心假设

离散网格只是连续物理空间的采样。虽然网格点很多，但其中可能存在较少数量的、内部相对一致的潜在物理状态，例如汽车前部高压区、后部低压区、翼型激波、尾流、材料高应力区。

因此作者把建模单位从：

$$
\text{mesh points}
$$

提升为：

$$
\text{learned physical states / physics-aware tokens}.
$$

> [!figure] 论文中图 1：不同 PDE/几何上的 learned slices
> ![[assets/Transolver/Figure1.png]]
> 图中展示 Darcy、Elasticity、Airfoil 和 Shape-Net Car 的 learned slices。高权重点会聚合到对应的 latent slice；这些 slice 可以形成多孔介质边界、材料挤压区、翼型激波/尾迹、汽车前后/上下区域等结构。

---

## 2. 核心方法：Physics-Attention

### 2.1 问题形式化

连续输入域为

$$
\Omega\subset\mathbb{R}^{C_g},
$$

离散为 $N$ 个点：

$$
\mathbf g\in\mathbb{R}^{N\times C_g}.
$$

在这些点上还可以有已观测物理量

$$
\mathbf u\in\mathbb{R}^{N\times C_u},
$$

其中 $\mathbf u$ 在部分任务中可以不存在。输入首先经过线性嵌入：

$$
\mathbf x^0=\operatorname{Linear}(\operatorname{Concat}(\mathbf g,\mathbf u)),
\qquad
\mathbf x^0\in\mathbb{R}^{N\times C}.
$$

### 2.2 Step 1：Slice——每个点对 $M$ 个 latent slice 做软分配

对第 $i$ 个网格点 feature

$$
\mathbf x_i\in\mathbb R^{1\times C},
$$

先投影成 $M$ 个 logits：

$$
\mathbf a_i=\operatorname{Project}(\mathbf x_i)\in\mathbb R^{1\times M},
$$

再沿 slice 维做 Softmax：

$$
\mathbf w_i=\operatorname{Softmax}(\mathbf a_i),
\qquad
\sum_{j=1}^{M}w_{ij}=1.
$$

所以

$$
W=[w_{ij}]\in\mathbb R^{N\times M}
$$

就是 **mesh point → latent slice 的 soft assignment matrix**。

例如 $N=6,M=3$ 时，某次 forward 可能得到：

$$
W=
\begin{bmatrix}
0.90&0.08&0.02\\
0.85&0.10&0.05\\
0.10&0.80&0.10\\
0.05&0.90&0.05\\
0.02&0.08&0.90\\
0.05&0.10&0.85
\end{bmatrix}.
$$

直观看：

- 点 1、2 主要属于 slice 1；
- 点 3、4 主要属于 slice 2；
- 点 5、6 主要属于 slice 3。

这很像 **learnable soft clustering**，但它没有独立的聚类损失；分组规则完全由最终 PDE 预测损失端到端训练。

> [!important] Softmax 到底“解决”了什么
> Softmax 本身只严格保证每个点在 $M$ 个 slice 上的权重和为 1，并使分配可微、可端到端训练。论文认为训练后分配会变得更尖锐、低熵；严格来说，**Softmax 本身不保证低熵**，是否接近 one-hot 取决于学到的 logits 大小。真正使 slice 具有任务意义的是 `Project + Softmax + downstream loss` 的联合训练。

### 2.3 $M$ 是谁决定的？

聚合后的 token 数 $M$ 是**人为设定的超参数**，不是模型自己决定。

原论文主设置：

- $C=128$ 时常用 $M=64$；
- $C=256$ 时常用 $M=32$。

论文消融测试 $M\in\{1,8,16,32,64,96,128,256,512,1024\}$。总体趋势：

- $M$ 太小：过度压缩，状态混合；
- $M$ 适中：性能明显改善；
- $M$ 太大：physics domain 被过度碎片化，token attention 也重新变复杂，性能可能波动或下降。

作者经验上认为 $32\sim256$ 是较容易调的区间。

### 2.4 slice 是 sample-dependent 的，不是固定模板

训练完成后，固定的是 `Project` 的参数 $\theta$，而不是固定的 assignment matrix。

对样本 $A$：

$$
W^{(A)}=\operatorname{Softmax}(\operatorname{Project}_\theta(X^{(A)})),
$$

对样本 $B$：

$$
W^{(B)}=\operatorname{Softmax}(\operatorname{Project}_\theta(X^{(B)})).
$$

一般情况下：

$$
\boxed{W^{(A)}\neq W^{(B)}}.
$$

因此**同一个训练好的模型，对两个不同几何/不同 feature 的样本可以形成不同 physics slices**。模型学习的是“如何根据当前 feature 分组”的规则，而不是一张固定的网格分区表。

同时，每个 Transolver layer 都在当前层 feature 上重新计算 slicing：

$$
W^{(l)}=
\operatorname{Softmax}
\left(
\operatorname{Project}^{(l)}(X^{(l-1)})
\right),
$$

所以即使是同一个样本，不同层的 slice 也没有必要相同。

> [!note] 关于工况变化
> 如果 Mach、angle of attack 等工况已经进入并影响当前 hidden feature，那么同一几何在不同工况下也可以产生不同 slice assignment。原论文没有专门做“固定几何、只改变工况并追踪 slice 移动”的独立实验，因此这一点属于架构机制允许的结果，不是论文单独验证的结论。

> [!figure] 论文中图 2：从离散域学习 physics-aware token
> ![[assets/Transolver/Figure2.png]]
> 论文以汽车表面为例说明：slice 不是固定几何块，而是从当前 mesh feature 学出的 soft physical grouping。

### 2.5 Step 2：Aggregate——把每个 slice 压成一个 physics-aware token

第 $j$ 个 token 为归一化加权平均：

$$
\mathbf z_j
=
\frac{\sum_{i=1}^{N}w_{ij}\mathbf x_i}
{\sum_{i=1}^{N}w_{ij}},
\qquad
\mathbf z_j\in\mathbb R^{1\times C}.
$$

最终得到：

$$
Z\in\mathbb R^{M\times C}.
$$

用矩阵写更直观。令

$$
D=\operatorname{diag}(W^T\mathbf 1)\in\mathbb R^{M\times M},
$$

则：

$$
\boxed{Z=D^{-1}W^TX}.
$$

维度为：

$$
(N\times C)
\xrightarrow{W^T}
(M\times C).
$$

### 2.6 Step 3：Token Attention——真正的 Transformer attention 只发生在 $M$ 个 token 上

$$
Q,K,V=\operatorname{Linear}(Z),
$$

$$
Z'
=
\operatorname{Softmax}
\left(
\frac{QK^T}{\sqrt C}
\right)V,
\qquad
Z'\in\mathbb R^{M\times C}.
$$

因此 attention matrix 从普通 full attention 的

$$
N\times N
$$

变成：

$$
M\times M.
$$

如果 $N=32,000,M=32$：

$$
32,000\times32,000
\quad\rightarrow\quad
32\times32.
$$

### 2.7 Step 4：DeSlice——把更新后的状态广播回每个网格点

对第 $i$ 个点：

$$
\mathbf x'_i
=
\sum_{j=1}^{M}w_{ij}\mathbf z'_j.
$$

矩阵形式：

$$
\boxed{X'=WZ'}.
$$

例如某点：

$$
w_i=[0.90,0.08,0.02],
$$

则：

$$
x'_i=0.90z'_1+0.08z'_2+0.02z'_3.
$$

因此同一组 soft assignment 同时完成：

- `mesh → slice` 的聚合；
- `slice → mesh` 的广播。

### 2.8 从矩阵角度看整个 Physics-Attention

忽略 Q/K/V 投影、Softmax 和归一化细节，可以把结构直觉写成：

$$
X'
\approx
W A_s W^T X,
$$

其中

$$
A_s\in\mathbb R^{M\times M}
$$

是 slice-token attention。

因此可以把

$$
A_{\mathrm{eff}}=W A_s W^T
$$

理解成一个**隐式的 mesh-to-mesh 全局关系算子**。Transolver 没有显式 materialize $N\times N$ attention，却通过

$$
N\times M
\rightarrow
M\times M
\rightarrow
M\times N
$$

实现了全局信息交换。

> [!warning] 这只是结构直觉
> 实际实现包含 slice normalization、Q/K/V 线性投影、Softmax、多头机制和 residual，不应把代码实现直接等同于一行 $WA_sW^T$。

### 2.9 多头形式与复杂度

论文采用 multi-head Physics-Attention：沿 channel 维把输入划分到多个子空间，每个 head 分别学习自己的 slice assignment、token attention 和 deslice。

原论文给出复杂度：

$$
\mathcal O(NMC+M^2C).
$$

只要 $M$ 固定且 $M\ll N$，关于网格点数 $N$ 为线性。

> [!figure] 论文中图 3：Transolver layer / Physics-Attention 总体结构
> ![[assets/Transolver/Figure3.png]]
> 每个 head 执行 Slice → Aggregate → token attention → DeSlice，随后接标准 Transformer 的 residual、LayerNorm 和 Feed-Forward Network（FFN）。

---

## 3. 完整 Transolver Layer

单个 layer 与 pre-norm Transformer 基本一致，只把标准 self-attention 换成 Physics-Attention：

$$
\hat{\mathbf x}^{l}
=
\operatorname{PhysicsAttn}
(\operatorname{LayerNorm}(\mathbf x^{l-1}))
+\mathbf x^{l-1},
$$

$$
\mathbf x^{l}
=
\operatorname{FeedForward}
(\operatorname{LayerNorm}(\hat{\mathbf x}^{l}))
+\hat{\mathbf x}^{l}.
$$

**关键点：每层输出仍然是 $N\times C$。**

因此原始 Transolver 的实际路径是：

$$
N
\rightarrow M
\rightarrow N
\rightarrow \text{FFN on }N
$$

然后下一层又重新：

$$
N\rightarrow M\rightarrow N.
$$

这解释了一个容易被“线性 attention”掩盖的问题：**attention 本身被压缩了，但整个网络并没有永久工作在 $M$ 个 token 上。** 这一点后来成为 Transolver-3 重点解决的扩展瓶颈。

---

## 4. 数学证明和本文方法到底是什么关系

### 4.1 Neural Operator 的积分形式

神经算子（Neural Operator）试图学习函数到函数的映射。一个典型非局部更新可写为：

$$
\mathcal G(u)(g^*)
=
\int_\Omega
\kappa(g^*,\xi)u(\xi)d\xi.
$$

直觉是：更新位置 $g^*$ 时，对整个域中位置 $\xi$ 的信息做带核 $\kappa$ 的加权积分。

### 4.2 标准 attention 为什么像离散积分

把积分离散成求和：

$$
y_i\approx\sum_j\kappa_{ij}u_j.
$$

而 attention 是：

$$
y_i=\sum_j a_{ij}v_j.
$$

所以 attention weight 可以被理解成 learnable kernel 的离散化。附录 Lemma A.1 复述了已有工作中“canonical attention 可视作积分算子的 Monte-Carlo 近似”的观点。

### 4.3 Transolver 的理论问题

普通 attention 直接在 mesh domain $\Omega$ 上做：

$$
\text{mesh}\leftrightarrow\text{mesh}.
$$

Transolver 却做：

$$
\Omega
\rightarrow
\Omega_s
\rightarrow
\Omega,
$$

其中 $\Omega_s$ 是 slice domain。

因此作者需要回答：

> attention 不再发生在原始 mesh 上，整个 Physics-Attention 还能不能解释成原始物理域上的 neural integral operator？

Theorem 3.4 给出的答案是：在论文所列假设下可以。

### 4.4 证明对应正文的三步

连续版 slice token：

$$
u_s(\xi_s)
=
\frac{\int_\Omega w_{\xi,\xi_s}u(\xi)d\xi}
{\int_\Omega w_{\xi,\xi_s}d\xi},
$$

对应正文 Eq. (2)。

随后在 slice domain 做 attention/integral，再用 slice weight 映射回 mesh，对应：

$$
\boxed{
\underbrace{\text{DeSlice}}_{\text{Eq. 4}}
\left[
\underbrace{\text{Attention}}_{\text{Eq. 3}}
\left(
\underbrace{\text{Slice Token}}_{\text{Eq. 2}}
\right)
\right].
}
$$

> [!important] 这个证明没有证明什么
> 它**没有**证明 Transolver 一定能精确解 Navier-Stokes，也没有证明 learned slice 一定等于真实物理状态，更没有证明它一定优于其他 neural operator。它主要说明：Physics-Attention 虽然先压到 latent slice domain 再计算，整体仍可被理论上解释为原始域上的 learnable integral operator。

> [!warning] 理论假设边界
> 附录证明使用可数域、slice 权重平滑、mesh/slice domain 映射可构造成微分同胚，并对 Jacobian 作简化等假设。它是结构解释，不是有限 $M$、有限数据下的 PDE 解误差界。

---

## 5. 与 full-resolution Plain Transformer 的直接比较

论文附录 Table 10 在 Darcy 上把 Physics-Attention 换成 canonical full attention，其余架构保持一致：

| Mesh points $N$ | Plain Transformer | Transolver | 相对提升 |
|---:|---:|---:|---:|
| 484 | 0.02017 | 0.02019 | -0.1% |
| 1,681 | 0.0103 | 0.0089 | 13.6% |
| 3,364 | 0.0073 | 0.0058 | 20.5% |
| 7,225 | 0.0081 | 0.0059 | 27.2% |
| 10,609 | OOM | 0.0057 | — |
| 19,881 | OOM | 0.0062 | — |
| 44,521 | OOM | 0.0063 | — |
| 168,921 | OOM | 0.0060 | — |

最小分辨率下二者几乎相同；随着 $N$ 增大，Transolver 既更省资源，Relative L2 也更低。论文据此认为，物理状态聚合不只是“牺牲精度换效率”，在大规模 mesh 上还可能降低关系学习难度。

> [!important] 实验边界
> 这个严格的 Plain Transformer vs Transolver 控制实验主要在 Darcy 上完成，并不是八个 benchmark 全部都有一组完全同构的 full-attention 对照。

---

## 6. 原始 Transolver 的显存瓶颈：为什么 $M$ 很小仍然会吃显存

### 6.1 核心原因不是 $M\times M$ attention，而是每层仍保持 full-resolution representation

例如：

$$
B=4,\quad N=32768,\quad C=256.
$$

一个 full-resolution hidden activation 就有：

$$
4\times32768\times256
=33,554,432
$$

个数；FP32 约 128 MiB。

若 FFN expansion ratio 为 4，中间 feature：

$$
B\times N\times4C
=
4\times32768\times1024,
$$

FP32 单个中间 activation 已约 512 MiB。

与此同时，slice assignment 还需要：

$$
W\in\mathbb R^{N\times M}.
$$

因此 Transolver 主要消除了：

$$
\boxed{N\times N\text{ attention map}}
$$

但没有消除：

$$
\boxed{N\times C\text{ point-wise activations and FFN}}
$$

以及：

$$
\boxed{N\times M\text{ slice weights}}.
$$

### 6.2 不同样本的 $N$ 不同，算法本身是否要求 padding？

**数学上不要求。**

样本 A：

$$
X_A\in\mathbb R^{4\times C}
\rightarrow
W_A^TX_A\in\mathbb R^{M\times C},
$$

样本 B：

$$
X_B\in\mathbb R^{7\times C}
\rightarrow
W_B^TX_B\in\mathbb R^{M\times C}.
$$

只要 $M,C$ 一致，两个样本最终都能得到 $M\times C$ token。

**padding 是 dense GPU batching 的实现问题，不是 Physics-Attention 的数学要求。** 如果为了 `torch.stack` 把短样本 pad 到 $N_{\max}$，必须 mask padded rows；因为全零 feature 经过 `Linear + Softmax` 仍可能产生非零 slice weights，直接参与分母并污染 token aggregation。

正确的 masked aggregation 应类似：

$$
z_j
=
\frac{\sum_i m_iw_{ij}x_i}
{\sum_i m_iw_{ij}},
\qquad m_i\in\{0,1\}.
$$

原始 Transolver 论文没有提出专门的 ragged-batch / packed-tensor 机制；多数 benchmark 在同一数据集内使用固定 mesh count。

---

## 7. 实验设置

### 7.1 八个 benchmark

| 数据集 | 几何 | 维度 / 网格规模 | 输入 | 输出 | 训练 / 测试 |
|---|---|---:|---|---|---:|
| Elasticity | point cloud | 2D / 972 | 材料结构点坐标 | 点内应力 | 1000 / 200 |
| Plasticity | structured mesh | 2D + time / 3,131 | 冲压模具形状 | 未来 20 步网格形变（4 方向） | 900 / 80 |
| Airfoil | structured mesh | 2D / 11,271 | 翼型结构 | Mach number | 1000 / 200 |
| Pipe | structured mesh | 2D / 16,641 | 管道结构 | 水平流速 | 1000 / 200 |
| Navier-Stokes | regular grid | 2D + time / 4,096 | 过去 10 步速度场 | 未来 10 步速度场 | 1000 / 200 |
| Darcy | regular grid | 2D / 7,225（主实验 85×85） | 多孔介质结构 | 流体压力 | 1000 / 200 |
| Shape-Net Car | unstructured mesh | 3D / 32,186 | 车辆几何 + signed distance function + normal vector | 周围速度 + 表面压力 | 789 / 100 |
| AirfRANS | unstructured mesh | 2D / 32,000 | 翼型结构及多工况样本 | 速度、压力等场量；主文重点评估 lift | 800 / 200 |

### 7.2 统一训练与模型配置

主实验统一使用 $L=8$ 层、8 heads。

| 数据集 | Loss | Epoch | Optimizer | Batch | $C$ | $M$ |
|---|---|---:|---|---:|---:|---:|
| Elasticity | Relative L2 | 500 | AdamW | 1 | 128 | 64 |
| Plasticity | Relative L2 | 500 | AdamW | 8 | 128 | 64 |
| Airfoil | Relative L2 | 500 | AdamW | 4 | 128 | 64 |
| Pipe | Relative L2 | 500 | AdamW | 4 | 128 | 64 |
| Navier-Stokes | Relative L2 | 500 | AdamW | 2 | 256 | 32 |
| Darcy | $L_{rL2}+0.1L_g$ | 500 | AdamW | 4 | 128 | 64 |
| Shape-Net Car | $L_v+0.5L_s$ | 200 | Adam | 1 | 256 | 32 |
| AirfRANS | $L_v+L_s$ | 400 | Adam | 1 | 256 | 32 |

所有任务初始学习率均为 $10^{-3}$。

`Project()` 根据几何不同实现为：

- 非结构网格 Elasticity、Shape-Net Car、AirfRANS：point-wise Linear；
- 其余结构/规则网格：$3\times3$ 局部卷积。

### 7.3 指标

物理场 Relative L2：

$$
\text{Relative }L_2
=
\frac{\|u-\hat u\|}{\|u\|}.
$$

Shape-Net Car 与 AirfRANS 还从预测场计算 drag/lift coefficient，并报告系数误差与 Spearman rank correlation。排序相关性对设计优化很重要，因为它反映代理模型能否正确比较不同设计。

---

## 8. 主要实验结果

### 8.1 六个标准 benchmark

| Benchmark | Transolver Relative L2 | 相比第二名相对误差下降 |
|---|---:|---:|
| Elasticity | **0.0064** | **25.6%** |
| Plasticity | **0.0012** | **29.4%** |
| Airfoil | **0.0053** | **10.2%** |
| Pipe | **0.0033** | **29.7%** |
| Navier-Stokes | **0.0900** | **24.7%** |
| Darcy | **0.0057** | **12.3%** |

### 8.2 工业设计任务

| Task | Volume ↓ | Surface ↓ | 系数误差 ↓ | Spearman ↑ |
|---|---:|---:|---:|---:|
| Shape-Net Car | **0.0207** | **0.0745** | $C_D$ **0.0103** | $\rho_D$ **0.9935** |
| AirfRANS | **0.0037** | **0.0142** | $C_L$ **0.1030** | $\rho_L$ **0.9978** |

---

## 9. 消融与模型分析

### 9.1 slice 数 $M$

- $M=1$：退化成全局 pooling，没有 slice-to-slice attention；
- 增大 $M$：一般改善细粒度状态表达；
- $M$ 过大：state fragmentation、token noise 和 $M^2$ attention 成本增加。

例如 Elasticity：$M=1$ 时 0.0148，$M=64$ 为 0.0064，$M=256$ 为 0.0054，而 $M=1024$ 回升到 0.0068。

### 9.2 learned slice vs fixed regular squares

Darcy 即使本来是 regular grid，把 learnable slices 换成固定 $4\times4$ square 仍会从约 0.0059 退化到 0.0088，说明“按任务学习状态”比“固定几何 patch”更符合本文假设。

### 9.3 attention 是否更有选择性

论文比较 attention weights 相对 uniform distribution 的 Kullback-Leibler divergence（KL divergence）：

| Benchmark | Galerkin Transformer | Transolver |
|---|---:|---:|
| Elasticity | 0.3803 | **1.7795** |
| Darcy | 0.2739 | **1.8274** |

更大意味着 attention 分布偏离均匀更多。作者据此认为 physics-token attention 比 massive mesh-point attention 更容易形成尖锐关系。

> [!figure] 论文中图 5：learned slices 与 attention map
> ![[assets/Transolver/Figure5.png]]
> 左侧显示随机重采样网格后仍可形成相似 slice；右侧 Transolver token attention 相比 Galerkin Transformer 的 mesh-point attention 更集中。

---

## 10. 扩展性、OOD 与附录实验

### 10.1 分辨率、数据量、参数规模

Darcy 上作者把分辨率提高到原设置约 25 倍，并增加数据和模型层数。Transolver 在高分辨率下保持较稳定表现，Elasticity 与 Airfoil 随深度从 8 层扩到 40 层还有明显改善。

### 10.2 自适应多尺度 slice

作者尝试前 4 层 $M=64$、后 4 层 $M=32$：

| 配置 | Airfoil | Darcy | 内存 | 时间 |
|---|---:|---:|---:|---:|
| 全部 $M=64$ | 0.0053 | 0.0059 | 0.64 GB | 38.18 s/epoch |
| $64\rightarrow32$ | **0.0050** | **0.0056** | 0.62 GB | 38.10 s/epoch |

说明 $M$ 可以被理解为 physical-state granularity，但系统化 hierarchical/multiscale Transolver 并不是原论文主体。

### 10.3 AirfRANS OOD

训练与测试 Reynolds number / angle of attack 范围不重叠：

| OOD setting | $C_L$ error ↓ | $\rho_L$ ↑ |
|---|---:|---:|
| Reynolds | **0.2996** | **0.9896** |
| Angle of attack | **0.1500** | **0.9950** |

### 10.4 Lagrangian 初步实验

WaterDrop：过去 6 步预测未来 994 步、2,000 particles。

- GNN：MSE 0.0182；
- GNN + Transolver：**0.0069**；
- relative promotion：**62.1%**。

作者没有让 Transolver 完全替代 GNN，因为 Lagrangian particle dynamics 中显式局部邻接仍然重要。

---

## 11. 原始 Transolver 的关键贡献与局限

### 11.1 关键贡献

1. 把 PDE Transformer 的基本 attention 单元从 mesh point 提升为 learned physical-state token。
2. 提出 `Slice → Aggregate → Attention → DeSlice` 的 Physics-Attention。
3. 支持 regular grid、point cloud、structured/unstructured mesh 与 surface-volume hybrid geometry。
4. 在 $M\ll N$ 下把 attention 从 $O(N^2)$ 降到 $O(NMC+M^2C)$。
5. learned tokenization 是 sample-dependent 的，避免固定 patch/topology 假设。

### 11.2 局限

- $M$ 仍需人工设定；
- physics-aware 是 latent semantics，不是显式物理标签；
- 每层仍返回 $N\times C$，FFN 仍在 full-resolution mesh 上执行；
- 仍需生成/保存 $N\times M$ slice weights；
- 原论文没有专门解决 variable-$N$ dense batching；
- multiscale hierarchy 只做了初步实验；
- Lagrangian 场景仍需要局部图结构补充。

其中**full-resolution activation + $N\times M$ slice weights**正是后续 Transolver-3 重点优化的对象。

---

# 12. 2026 后续进展：Transolver-3

> [!abstract] Transolver-3 的核心进展
> Transolver-3 不改变“mesh → physics states → attention → mesh”的核心物理状态思想，而是重新分析原始 Physics-Attention 的真正扩展瓶颈，并通过 **Faster Slice/DeSlice、Geometry Slice Tiling、Geometry Amortized Training、Physical State Caching + Full Mesh Decoding** 把可处理规模推进到超过 $10^8$ cells，最大实验达到约 **160M volume cells**。

## 12.1 它首先指出：原始 Transolver 的主要瓶颈已经不是 attention

原始形式：

$$
x_{\text{proj}}=xW_1,
$$

$$
w=\operatorname{Softmax}(xW_2),
$$

$$
s=(wd^{-1})^Tx_{\text{proj}},
$$

$$
x_{\text{out}}=(ws')W_3.
$$

虽然 slice-domain attention 只有

$$
O(M^2C),
$$

但工业级 $N>10^8$ 时，以下线性项已经非常昂贵：

- Linear1：$O(NC^2)$ time，$O(NC)$ memory；
- slice weights：$O(NCM)$ time，$O(NM)$ memory；
- deslice / Linear3：仍产生 $O(NC)$ 中间量。

论文举例：$N=10^8,C=128$ 时，一个 FP32 的 $N\times C$ activation 就约 **51.2 GB**。

## 12.2 Faster Slice / DeSlice：利用矩阵结合律把大 Linear 移到 latent domain

原 slice：

$$
s=w^T(xW_1)d^{-1}.
$$

利用结合律：

$$
\boxed{
w^T(xW_1)
=(w^Tx)W_1
}
$$

所以先聚合：

$$
w^Tx:\quad
(M\times N)(N\times C)
\rightarrow M\times C,
$$

再在线性层上处理：

$$
(M\times C)(C\times C)
\rightarrow M\times C.
$$

这样 Linear1 不再在 $N$ 个 mesh cells 上执行，而只在 $M$ 个 states 上执行。

DeSlice 同理：

$$
(ws')W_3
=
\boxed{w(s'W_3)}.
$$

先在 $M$ 个 state 上执行 Linear3，再广播到 $N$ 个点。

这个重排**数学等价，不是近似**。Transolver-3 把与 $N$ 相关的主要 time terms 从 5 个降到 3 个，space terms 从 4 个降到 2 个。

> [!figure] Transolver-3 论文中图 2：训练阶段的三项几何扩展机制
> ![[assets/Transolver/Transolver3_Figure2.png]]
> (a) Faster Slice/DeSlice 通过矩阵结合律把投影移到 slice domain；(b) Geometry Slice Tiling 不再 materialize 完整 $N\times M$ 权重；(c) Geometry Amortized Training 每次只从完整工业网格采一个子集训练。

## 12.3 Geometry Slice Tiling：不再一次存完整 $N\times M$ assignment matrix

原始需要：

$$
W\in\mathbb R^{N\times M}.
$$

当

$$
N=160M,\quad M=64,
$$

单个 FP32 $W$ 理论上就需要约 41 GB。

Transolver-3 把 mesh 切成 $T$ 个 tile：

$$
X=
\begin{bmatrix}
X^{(1)}\\
X^{(2)}\\
\vdots\\
X^{(T)}
\end{bmatrix},
$$

逐 tile 计算：

$$
S_{\text{raw}}^{(t)}
=(W^{(t)})^TX^{(t)},
$$

并累计：

$$
S_{\text{raw}}
=
\sum_{t=1}^{T}S_{\text{raw}}^{(t)},
$$

归一化分母也同样累计：

$$
d=
\sum_{t=1}^{T}d^{(t)}.
$$

因此最终 token 与一次性计算完全一致，但峰值 slice-weight memory 从：

$$
O(NM)
$$

变成：

$$
\boxed{O(N_tM)}
$$

其中 $N_t=N/T$ 是单 tile 大小。

训练时每个 tile 还使用 gradient checkpointing，用额外重计算换显存。

论文对 800k cells 的实验：

| tile size | 800k | 200k | 100k | 20k | 10k | 5k |
|---:|---:|---:|---:|---:|---:|---:|
| memory (GB) | 22.16 | 13.62 | 12.28 | 11.27 | 11.09 | 11.04 |
| latency (ms) | 404 | 410 | 429 | 453 | 656 | 1307 |

所以 tiling 是典型的 **memory ↔ compute trade-off**；论文认为该实验中 100k tile 是较平衡的选择。

## 12.4 Geometry Amortized Training：训练时根本不吃完整 $10^8$ mesh

即使前两项优化后，单 GPU single-pass 也只能处理约 2.9M cells。要训练 20M/160M mesh，Transolver-3 每个 iteration 从完整网格 $D_N$ 随机无放回抽样：

$$
D_n\subset D_N,
\qquad
n\sim10^5-10^6.
$$

AhmedML、DrivAerML 主实验采用：

$$
\boxed{n=100K}.
$$

核心假设是：高保真 mesh 是连续物理 manifold 的离散采样；在不同 iteration 暴露不同随机子集，可以把处理完整几何的成本 amortize 到训练过程，而模型被迫学习对具体离散方式更鲁棒的规律。

DrivAerML subset-size 消融：

| subset size | $p_s$ | $u$ | $\tau$ | $p_v$ |
|---:|---:|---:|---:|---:|
| 1k | 4.84 | 5.30 | 7.31 | 7.94 |
| 5k | 4.05 | 4.63 | 6.39 | 6.53 |
| 10k | 3.86 | 4.49 | 6.22 | 6.19 |
| 20k | 3.80 | 4.29 | 6.08 | 5.90 |
| 50k | 3.76 | 4.21 | 6.02 | 5.79 |
| 100k | **3.71** | **4.14** | **5.85** | **5.72** |

越大越准，但即使 1k 也没有训练崩溃。

## 12.5 Physical State Caching：推理时把“全局状态估计”和“逐点场解码”拆开

Transolver-3 推理不是强迫完整 $N$ 点同时进入一个 forward，而分两阶段。

### Stage A：Physical State Caching

把完整 mesh 分成可放入显存的 chunks：

$$
D_N=D_1\cup D_2\cup\cdots\cup D_K.
$$

逐块累计每层 physical states，得到与“无限显存下一次性全 mesh forward”相同的全局 state：

$$
s_{\text{cache}}
=
\{s_{\text{out}}'^{(1)},\ldots,s_{\text{out}}'^{(L)}\}.
$$

每层 cache 只有：

$$
M\times C.
$$

### Stage B：Full Mesh Decoding

对任意 query point $x$，第 $l$ 层只需：

$$
w^{(l)}
=\operatorname{Softmax}(\operatorname{Linear2}(x^{(l)})),
$$

$$
x_{\text{out}}^{(l)}
=w^{(l)}s_{\text{out}}'^{(l)}.
$$

即：

$$
(1\times M)(M\times C)
\rightarrow1\times C.
$$

所以 output mesh 也可以分 chunk 逐步 decode，不要求 160M 点同时常驻 GPU。

> [!figure] Transolver-3 论文中图 3：decoupled inference
> ![[assets/Transolver/Transolver3_Figure3.png]]
> 左侧先流式聚合全网格信息得到 layer-wise physical state cache；右侧再让任意 mesh coordinate 逐层读取 cache，得到 full-resolution field prediction。

> [!important] 与 variable-$N$/padding 的关系
> Transolver-3 实际上弱化了“大 mesh 必须 pad/stack 成统一 $N$”的问题：训练时所有超大网格可以统一随机采固定 $n$ 点；推理时每个样本按自身 $N$ 切成任意数量 chunks。它并不是通用 ragged-batch 库，但从 workflow 上避免了把完整 heterogeneous meshes pad 到同一个全局 $N_{\max}$。

## 12.6 实际扩展到了什么尺度

三个工业级 aerodynamic benchmarks：

| Benchmark | Surface | Volume | 规模/特点 |
|---|---:|---:|---|
| NASA-CRM | ~400K | — | full aircraft surface |
| AhmedML | ~1M | ~20M | high-fidelity automotive, hybrid RANS-LES |
| DrivAerML | ~8M | **~160M** | high-fidelity automotive, dataset ~31 TB |

Transolver-3 的 geometry scaling 把 Transolver family 从几十万/百万级推进到**一亿级以上**。

> [!figure] Transolver-3 论文中图 6：单 GPU raw capacity
> ![[assets/Transolver/Transolver3_Figure6.png]]
> 在不使用 amortized training 的 single-pass 测试里，Faster Slice/DeSlice + Tiling 使单 A100 40GB raw capacity 达约 2.9M cells，约为 Transolver++ 的 1.9×；真正跨到 $10^8$ 依赖后续 amortized training 与 decoupled inference。

## 12.7 主结果

Transolver-3 在三个 benchmark 的 10 个 field metrics 中 9 个最好。DrivAerML：

| Model | $p_s$ | $u$ | $\tau$ | $p_v$ |
|---|---:|---:|---:|---:|
| AB-UPT | 3.82 | 5.93 | 7.29 | 6.08 |
| Transolver++ | 4.12 | 4.70 | 6.42 | 6.70 |
| **Transolver-3** | **3.71** | **4.14** | **5.85** | **5.72** |

DrivAerML integrated coefficients：

$$
R^2(C_D)=0.972,
\qquad
R^2(C_L)=0.985.
$$

论文同时强调 **evaluation resolution 本身会影响 drag/lift 的数值积分精度**：用更多 surface cells 解码预测场，$C_D/C_L$ 的 MAE 持续降低。

> [!figure] Transolver-3 论文中图 9：输入/输出分辨率扩展的收益
> ![[assets/Transolver/Transolver3_Figure9.png]]
> (a) physical-state cache 使用更多 input cells 时，field error 继续下降；(b) 用更多 surface cells 做 full-mesh decoding / integration 时，drag/lift 误差继续降低。

## 12.8 对 discontinuity / shock 的补充实验

AirCraft transonic/supersonic dataset 约 300k points/sample，训练 subset 100k。surface pressure Relative L2：

| Model | $p_s$ error (%) |
|---|---:|
| AB-UPT | 6.93 |
| Transolver | 9.20 |
| Transolver++ | 6.40 |
| **Transolver-3** | **5.80** |

这说明 Transolver-3 的 sampling + physics-state 机制没有表现出“无法处理 shock/discontinuity”的结构性障碍。不过它并没有证明该结构在 shock localization 上一定优于具有显式局部多尺度先验的 U-shaped 方法。

## 12.9 Transolver-3 的局限

论文明确列出：

1. tiling 用额外计算换显存，tile size 需要在 latency 与 memory 间折中；
2. 主要验证 stationary aerodynamics，强瞬态或 tightly coupled multiphysics 尚未充分验证；
3. 未来希望加入 geometry pretraining（如 GeoPT）增强跨工业 topology 泛化。

---

## 13. Transolver 家族的演进逻辑

可以把三代方法理解为连续解决三个瓶颈：

$$
\boxed{
\text{Transolver}
:\quad
N\text{ mesh tokens}
\rightarrow
M\text{ physics tokens}
}
$$

解决：**$N^2$ attention + mesh-point semantic bottleneck**。

$$
\boxed{
\text{Transolver++}
:\quad
\text{multi-GPU distributed scaling}
}
$$

解决：**million-scale mesh 的硬件容量问题**，但主要是 hardware-driven scaling。

$$
\boxed{
\text{Transolver-3}
:\quad
\text{algebraic reordering}
+
\text{tiling}
+
\text{subset training}
+
\text{state caching/query decoding}
}
$$

解决：**$O(N)$ activation、$N\times M$ weights、full-mesh training/inference 的工业级扩展问题**。

最重要的思想变化是：

> 原始 Transolver 试图让一个 layer 高效地在完整 mesh 上运行；Transolver-3 进一步接受“完整工业 mesh 根本不应该同时驻留在一个 forward 中”，转而把全局状态的估计做成可流式累积的 cache，再把 field prediction 变成 query decoding。

---

## 14. 与其他架构思想的联系

### 14.1 与固定 patch / U-shaped hierarchy

两类方法的 inductive bias 不同：

- 固定 patch / U-shaped：按**空间邻接与多尺度几何结构**降采样，适合规则 correspondence、局部高频结构和 skip reconstruction；
- Transolver：按**当前 feature 的 latent physical similarity**动态聚合，适合 irregular mesh、variable topology 和非局部物理关系。

二者并不互斥。一个值得实验的组合是：

$$
\text{local / geometric encoder}
\rightarrow
\text{physics-state bottleneck/cache}
\rightarrow
\text{query or U-shaped decoder}.
$$

这是架构设计启发，不是原论文已经验证的组合。

### 14.2 与 Perceiver / latent operator 思路

Transolver-3 的 `global physical state cache + arbitrary point decoding` 已经非常接近“compact latent representation + query decoder”的 operator 形态；区别在于 latent states 仍然由 physics slicing 的 soft assignment 明确从 mesh 聚合而来，并且每层都保留 physical-state semantics。

---

## 15. 核心研究脉络

### [[Fourier Neural Operator]]

FNO 用 Fourier domain 参数化 non-local integral operator，奠定现代 neural operator 的代表性范式。Transolver 仍是数据驱动 operator learning，但不依赖固定 Fourier basis，而是用 learned physics-state attention 表达非局部关系。

### [[Choose a Transformer - Fourier or Galerkin]]

Galerkin Transformer 从 operator learning 角度讨论 attention 与积分/投影算子的关系。Transolver 继承“attention 可作为 learnable operator”的理论视角，但进一步把 attention 单元从 mesh point 提升到 learned slice token。

### [[GNOT]]

GNOT 等方法通过高效/线性 attention 直接处理 operator learning。Transolver 的区别不是只“换一种线性 attention”，而是**先改变 tokenization 层级**，再做 attention。

研究脉络可概括为：

$$
\text{spectral operator}
\rightarrow
\text{attention operator on mesh points}
\rightarrow
\text{attention operator on learned physical states}
\rightarrow
\text{streamable/cached physical-state operator at industrial scale}.
$$

---

## 16. 最终速记

### 原始 Transolver 四个式子

$$
\boxed{
w_i=\operatorname{Softmax}(\operatorname{Project}(x_i))
}
$$

$$
\boxed{
z_j=
\frac{\sum_iw_{ij}x_i}
{\sum_iw_{ij}}
}
$$

$$
\boxed{
Z'=\operatorname{Attention}(Z)
}
$$

$$
\boxed{
x'_i=\sum_jw_{ij}z'_j
}
$$

### Transolver-3 四个扩展关键词

$$
\boxed{
\text{Faster Slice/DeSlice}
+
\text{Tiling}
+
\text{Amortized Training}
+
\text{Physical State Cache}
}
$$

---

## 17. 参考概念资料

- Kovachki et al., *Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs*, JMLR 2023: https://www.jmlr.org/papers/v24/21-1524.html
- Shuhao Cao, *Choose a Transformer: Fourier or Galerkin*, NeurIPS 2021: https://proceedings.neurips.cc/paper/2021/hash/d0921d442ee91b896ad95059d13df618-Abstract.html
- Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*: https://arxiv.org/abs/2010.08895

## 跨文献关联导航（2026-10-15 整理）

> [!note] 文献关系标注
> “方法承接”或“研究延展”表示现有笔记可支持的脉络；“比较”“互补”“类比”仅用于研究组织，不声称作者直接引用或实验证实联合方案。

**总览**：[[Literature/文献关系总览|文献关系总览]]

**所属主题**：[[Literature/Topics/Transformer与物理场算子|Transformer与物理场算子]] · [[Literature/Topics/气动代理与多保真学习|气动代理与多保真学习]]

**重点关联**：

- **方法承接**：[[Literature/transformer/Spatially-aware transformer operator for real-time aerodynamic evaluations of arbitrary three-dimensional vehicles|Spatially-aware transformer operator for real-time aerodynamic evaluations of arbitrary three-dimensional vehicles]] — SATO 在 Transolver 的 Physics-Attention 上结合局部序列化注意力。
- **方法比较**：[[Literature/transformer/SMART - Scalable Mesh-free Aerodynamic Simulations from Raw Geometries using a Transformer-based Surrogate Model|SMART - Scalable Mesh-free Aerodynamic Simulations from Raw Geometries using a Transformer-based Surrogate Model]] — 不规则网格物理 slice 与原始几何 latent / 任意 query。
- **物理代理对照**：[[Literature/transformer/Towards a Foundation-Model Paradigm for Aerodynamic Prediction in Three-dimensional Design|Towards a Foundation-Model Paradigm for Aerodynamic Prediction in Three-dimensional Design]] — 非结构网格算子与统一表面 patch 主干有不同数据假设。
- **方法互补**：[[Literature/transformer/Unisolver|Unisolver]] — 物理注意力组织与 PDE 条件分层解决不同瓶颈。
- **算子比较**：[[Literature/transformer/Perceiver IO|Perceiver IO]] — 固定 bottleneck/query 与按物理状态动态聚合 slice 不同。
- **全局通信对照**：[[Literature/transformer/Star-Transformer|Star-Transformer]] — 单一 relay node 与多个 physics slice 都降低全连接成本，但归纳偏置不同。
