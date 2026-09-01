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
tags:
  - neural-operator
  - transformer
  - PDE
  - scientific-machine-learning
  - unstructured-mesh
---

> [!abstract] 摘要翻译
> Transformer 已经在许多领域取得突破，并开始被用于求解偏微分方程（Partial Differential Equation, PDE）。但 PDE 通常被离散为规模很大、几何形状复杂的网格，直接让 Transformer 在海量单点之间建立关系，不仅计算代价高，也难以从这些低层网格点中学习复杂的物理相关性。本文提出 **Transolver**：不直接把“网格点”视为注意力的基本语义单元，而是学习离散几何背后潜在的**内禀物理状态**。核心模块 **Physics-Attention** 会把离散域自适应地软划分为若干可学习、形状自由的 slice，使具有相似物理状态的网格点被分配到同一 slice；再把每个 slice 聚合成 physics-aware token，在这些 token 上做注意力，最后把更新后的 token 广播回原网格。该机制对复杂几何具有内生的适应能力，在 slice 数固定且远小于网格点数时，对网格规模呈线性复杂度。论文在 6 个标准 PDE 基准上取得一致最优结果，平均相对提升约 22%，并在汽车和翼型的大规模工业级仿真任务上表现出较好的精度、效率、扩展性和分布外泛化能力。

## 一句话理解

**Transolver 的关键不是发明一种新的 Q/K/V 注意力公式，而是先把大量 mesh point 软聚合成少量“物理状态 token”，在这些 token 上做标准注意力，再映射回 mesh。**

## 1. 问题与动机

### 1.1 为什么普通 Transformer 不适合直接处理 PDE 网格

PDE 数值问题通常先把连续域 $\Omega$ 离散为 $N$ 个网格点。工业问题中的 $N$ 很大，而且点可能是不规则的二维/三维点云、非结构网格或表面-体网格混合结构。

直接把每个网格点当作一个 Transformer token 有两个问题：

1. **计算复杂度**：标准自注意力需要构造 $N\times N$ 的相关矩阵，复杂度随 $N^2$ 增长。
2. **关系学习困难**：PDE 中真正重要的是压力区、尾流、激波、材料挤压区等高阶物理结构，而不是单个离散点本身。即使把注意力线性化，仍然是在海量单点之间建立关系，作者认为容易得到信息量较低、接近均匀的注意力。

传统视觉 Transformer 的 patchify 也不能直接解决这个问题，因为规则矩形 patch 不适用于任意非结构几何，而且固定空间块未必对应相同的物理状态。

### 1.2 作者的核心假设

离散网格只是连续物理空间的采样。虽然网格点很多，但其中存在较少数量的、内部相对一致的潜在物理状态。例如汽车表面可以出现前部高压区、斜面区、后部低压区；翼型周围可出现激波、尾流等结构。

因此作者把建模单位从：

$$
\text{mesh points} \quad\longrightarrow\quad \text{learned physical states}
$$

改为：

$$
N\ \text{mesh points}
\xrightarrow{\text{Slice}}
M\ \text{physics-aware tokens}
\xrightarrow{\text{Attention}}
M\ \text{updated tokens}
\xrightarrow{\text{DeSlice}}
N\ \text{mesh features},
$$

其中 $M\ll N$。

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

其中 $\mathbf u$ 在部分任务中可以不存在。例如：

- 时序流场预测：输入包含网格位置和过去的流速场；
- 汽车/翼型设计：输入主要是几何及几何派生特征，输出表面与周围流场物理量。

输入首先经过线性嵌入：

$$
\mathbf x^0=\operatorname{Linear}(\operatorname{Concat}(\mathbf g,\mathbf u)),
\qquad
\mathbf x^0\in\mathbb{R}^{N\times C}.
$$

### 2.2 Slice：学习每个网格点属于哪些物理状态

对每个网格点特征 $\mathbf x_i\in\mathbb{R}^{1\times C}$，先投影成 $M$ 个 slice 权重：

$$
\mathbf w_i=\operatorname{Softmax}(\operatorname{Project}(\mathbf x_i)),
\qquad
\mathbf w_i\in\mathbb{R}^{1\times M},
$$

并满足

$$
\sum_{j=1}^{M}w_{i,j}=1.
$$

因此 $w_{i,j}$ 表示第 $i$ 个点属于第 $j$ 个 slice 的软分配程度。

论文把第 $j$ 个 slice 写成

$$
\mathbf s_j=\{w_{i,j}\mathbf x_i\}_{i=1}^{N}.
$$

这里不是传统网格分区中的“硬切块”，而是**全局、软分配、可跨空间位置**的聚类式表示：两个空间上距离很远的点，只要学到的特征相似，也可以有很高的同一 slice 权重。

> [!important] 一个容易误解的点
> 文中没有额外加入显式的熵正则项。作者的解释是：沿 $M$ 个 slice 维度做 Softmax 会使分配更有选择性，而端到端任务损失会继续推动 slice 形成对预测有用、内部物理状态较一致的分组。因此这里的“physics-aware”是**通过任务监督涌现出来的潜变量语义**，不是人工提供的物理标签。

### 2.3 Aggregate：把一个 slice 压缩成一个 physics-aware token

第 $j$ 个 slice 中的所有点经过归一化加权聚合：

$$
\mathbf z_j
=
\frac{\sum_{i=1}^{N}w_{i,j}\mathbf x_i}
{\sum_{i=1}^{N}w_{i,j}},
\qquad
\mathbf z_j\in\mathbb{R}^{1\times C}.
$$

因此得到

$$
\mathbf z\in\mathbb{R}^{M\times C}.
$$

这一操作把 $N$ 个 mesh token 压缩成 $M$ 个 physics-aware token。

### 2.4 Token Attention：注意力只在 $M$ 个物理 token 上计算

对 $\mathbf z$ 做标准缩放点积注意力：

$$
\mathbf q,\mathbf k,\mathbf v=\operatorname{Linear}(\mathbf z),
$$

$$
\mathbf z'
=
\operatorname{Softmax}\left(
\frac{\mathbf q\mathbf k^{\mathsf T}}{\sqrt C}
\right)\mathbf v,
\qquad
\mathbf z'\in\mathbb{R}^{M\times C}.
$$

这里注意力矩阵只有 $M\times M$，不再是 $N\times N$。

### 2.5 DeSlice：把物理 token 广播回网格点

使用**与 Slice 相同的权重**把更新后的 token 重组回每个网格点：

$$
\mathbf x'_i
=
\sum_{j=1}^{M}w_{i,j}\mathbf z'_j.
$$

也就是说，同一个 $w_{i,j}$ 同时承担：

- mesh $\rightarrow$ slice/token 的聚合权重；
- token $\rightarrow$ mesh 的广播权重。

这使 Physics-Attention 形成一个非常清楚的三段式：

$$
\boxed{
\text{Slice/Aggregate}
\rightarrow
\text{Token Attention}
\rightarrow
\text{DeSlice/Broadcast}
}
$$

### 2.6 多头形式与复杂度

论文采用多头 Physics-Attention：沿通道维将输入划分到多个子空间，每个 head 分别执行 slice、token attention 和 deslice。

单层总复杂度写为：

$$
\mathcal O(NMC+M^2C).
$$

只要 $M$ 被视为常数且 $M\ll N$，复杂度关于网格点数 $N$ 为线性。

> [!note] 复杂度结论的条件
> “线性复杂度”依赖于 $M$ 不随 $N$ 同阶增长。如果为了更高分辨率而让 $M$ 也显著增长，则 $M^2C$ 的 token attention 项仍会增加。

## 3. Transolver 层与完整网络

单个 Transolver layer 与 pre-norm Transformer 基本一致，只是把标准自注意力换成 Physics-Attention：

$$
\hat{\mathbf x}^{l}
=
\operatorname{PhysicsAttn}(\operatorname{LayerNorm}(\mathbf x^{l-1}))
+\mathbf x^{l-1},
$$

$$
\mathbf x^{l}
=
\operatorname{FeedForward}(\operatorname{LayerNorm}(\hat{\mathbf x}^{l}))
+\hat{\mathbf x}^{l}.
$$

最后对第 $L$ 层输出做线性投影，得到目标物理量预测。

> [!figure] 论文中图 2：从离散域学习 physics-aware token
> ![](https://arxiv.org/html/2402.02366v2/slice_new_compress.png)
> 图中以汽车表面压力为例：空间上的 mesh point 通过软分配形成若干 slice；每个 slice 聚合成一个 token。slice 并不要求是连续的规则区域。

> [!figure] 论文中图 3：Transolver layer / Physics-Attention 总体结构
> ![](https://arxiv.org/html/2402.02366v2/slice_attention_new_compress.png)
> 每个 head 都执行 Slice → Aggregate → token attention → Broadcast/DeSlice，然后与标准 Transformer 一样接 LayerNorm、残差连接和前馈网络（Feed-Forward Network, FFN）。

## 4. 为什么作者认为 slice 会学到“物理状态”

论文给出两层理由：

1. slice 权重直接由 mesh feature 投影得到，因此特征相似的点倾向产生相似的 slice 分配；
2. token 之间随后需要通过 attention 完成任务。如果一个 token 混合了过多不相容的状态，后续注意力难以形成清晰关系，最终预测损失会反向优化 slice 权重。

这与有限元等传统数值方法的计算域划分不同。传统划分主要由几何邻近性和人工数值设计决定；Transolver 的 slice 可以把**空间相隔较远但物理意义相近**的点放在一起。论文举例称，汽车的挡风玻璃、车牌和前灯可在某些 slice 中被共同归为“迎风前部”状态。

> [!figure] 论文中图 1：不同 PDE/几何上的 learned slices
> ![](https://arxiv.org/html/2402.02366v2/intuitive_new_all_pde.png)
> 作者展示 Darcy、Elasticity、Airfoil 和 Shape-Net Car 中的 slice。可视化中出现了多孔介质边界、弹性挤压区、翼型激波/尾流以及汽车前后/上下区域等结构。

## 5. Physics-Attention 与“神经算子”的理论解释

### 5.1 神经算子是什么

普通神经网络主要学习有限维向量到向量的映射；**神经算子（Neural Operator）**试图学习函数空间之间的映射，例如“给定 PDE 系数场/边界条件函数，输出对应的解函数”。Kovachki 等将 neural operator 形式化为非局部积分算子与局部非线性变换的组合；FNO（Fourier Neural Operator）则在傅里叶域中参数化积分核。[联网资料 1][^web-neural-operator]

对 Transolver 而言，这意味着每一层不只是处理一个固定维度向量，而可理解为在近似一个作用于离散场上的积分更新算子。

### 5.2 标准 attention 为什么可以看作积分算子

论文附录先复述 Galerkin Transformer 等工作的观点：对连续输入函数 $u:\Omega\rightarrow\mathbb R^C$，可定义

$$
\mathcal G(u)(g^*)
=
\int_{\Omega}\kappa(g^*,\xi)u(\xi)\,d\xi.
$$

当用离散网格点对积分做 Monte-Carlo 近似，并把核 $\kappa$ 参数化为 query-key 相似度时，就得到类似 softmax attention 的离散求和形式。Galerkin Transformer 的工作也从算子学习角度系统讨论了 attention 与积分/投影算子的联系。[联网资料 2][^web-galerkin]

### 5.3 Physics-Attention 的证明思路

Transolver 不在 mesh domain 上直接做 attention，而是在 slice domain 上做，因此作者的思路是：

1. 构造 mesh domain $\Omega$ 与 slice domain $\Omega_s$ 之间的映射；
2. 把 mesh 上的函数聚合成 slice token 函数 $u_s$；
3. 对积分变量做换元，把 $\Omega$ 上的积分写成 $\Omega_s$ 上的积分；
4. 用 token attention 近似 slice domain 上的积分；
5. 最后通过 DeSlice 把结果映射回 mesh。

附录中用到**微分同胚（diffeomorphism）**：即映射可逆、映射本身和逆映射都光滑，可把它直观理解为“不会撕裂/粘合的光滑坐标变换”。[^web-diffeomorphism]

> [!warning] 对理论结果的阅读边界
> 附录证明依赖可数域、slice 权重平滑、可构造微分同胚以及简化 Jacobian 行列式等假设。它主要说明 Physics-Attention 可以被解释为一种 learnable integral operator；论文并没有给出有限 $M$ 下对真实 PDE 解算子的严格逼近误差界。

## 6. 实验设置

### 6.1 八个 benchmark

| 数据集 | 几何 | 维度 / 网格规模 | 输入 | 输出 | 训练 / 测试 |
|---|---|---:|---|---|---:|
| Elasticity | point cloud | 2D / 972 | 材料结构点坐标 | 点内应力 | 1000 / 200 |
| Plasticity | structured mesh | 2D + time / 3,131 | 冲压模具形状 | 未来 20 步网格形变（4 方向） | 900 / 80 |
| Airfoil | structured mesh | 2D / 11,271 | 翼型结构 | Mach number | 1000 / 200 |
| Pipe | structured mesh | 2D / 16,641 | 管道结构 | 水平流速 | 1000 / 200 |
| Navier-Stokes | regular grid | 2D + time / 4,096 | 过去 10 步速度场 | 未来 10 步速度场 | 1000 / 200 |
| Darcy | regular grid | 2D / 7,225（主实验 85×85） | 多孔介质结构 | 流体压力 | 1000 / 200 |
| Shape-Net Car | unstructured mesh | 3D / 32,186 | 车辆几何 + signed distance function + normal vector | 周围速度 + 表面压力 | 789 / 100 |
| AirfRANS | unstructured mesh | 2D / 32,000 | 翼型几何及工况样本 | 速度、压力、黏性等场量；主文关注 lift | 800 / 200 |

覆盖的物理类型包括：

- 固体力学：Elasticity、Plasticity；
- Navier-Stokes 流体问题：Airfoil、Pipe、Navier-Stokes、Shape-Net Car、AirfRANS；
- Darcy 多孔介质流。

### 6.2 统一训练与模型配置

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

所有任务初始学习率均为 $10^{-3}$。其中：

- $L_v$：体网格（volume）物理场损失；
- $L_s$：表面（surface）物理场损失；
- Darcy 的 $L_g$：遵循 ONO 设置加入的空间梯度正则项。

`Project()` 的实现根据几何有所不同：

- 非结构网格 Elasticity、Shape-Net Car、AirfRANS：单个 point-wise Linear；
- 其余结构/规则网格：$3\times3$ 局部卷积。

实验在单张 NVIDIA A100 上完成，每组重复三次。

### 6.3 指标

物理场使用相对 $L_2$ 误差：

$$
\text{Relative }L_2
=
\frac{\|u-\hat u\|}{\|u\|}.
$$

在 Shape-Net Car 与 AirfRANS 中，还从预测的压力/速度场计算 drag/lift coefficient，并报告其相对误差以及 Spearman rank correlation。后者衡量真实排序和代理模型预测排序的一致程度，因此直接关联设计优化中“能否把更优设计排到前面”。

## 7. 主要实验结果

### 7.1 六个标准 benchmark：全部取得最优结果

| Benchmark | Transolver Relative L2 | 相比第二名的相对误差下降 |
|---|---:|---:|
| Elasticity | **0.0064** | **25.6%** |
| Plasticity | **0.0012** | **29.4%** |
| Airfoil | **0.0053** | **10.2%** |
| Pipe | **0.0033** | **29.7%** |
| Navier-Stokes | **0.0900** | **24.7%** |
| Darcy | **0.0057** | **12.3%** |

论文特别强调 point cloud / structured mesh 上的提升较大，并把 Darcy 的优势归因于：多孔介质边界附近存在复杂 fluid-structure interaction，直接在大量 mesh point 上做线性 attention 难以形成清晰的关系，而 physics-aware token 提供了更高层的关系建模单位。

### 7.2 工业设计任务

| Task | Volume ↓ | Surface ↓ | 系数误差 ↓ | Spearman ↑ |
|---|---:|---:|---:|---:|
| Shape-Net Car | **0.0207** | **0.0745** | $C_D$ **0.0103** | $\rho_D$ **0.9935** |
| AirfRANS | **0.0037** | **0.0142** | $C_L$ **0.1030** | $\rho_L$ **0.9978** |

对设计问题，Spearman 指标很重要：即使场误差不完全为零，只要代理模型能够正确排序不同设计，就仍可能有效指导形状优化。

### 7.3 为什么比 geo-FNO / GNN 类方法更强

论文的解释是：

- geo-FNO 需要把不规则几何变换到规则隐空间网格，傅里叶基的周期边界假设在汽车这种复杂表面-体混合几何中容易退化；
- 图方法擅长局部几何关系，但全局物理关系建模受限；
- Transolver 不依赖固定网格拓扑，而通过 slice 学习全局物理状态并在这些状态之间建模关系。

## 8. 消融与模型分析

### 8.1 slice 数量 $M$

总体趋势是 $M$ 从很小值增加时性能改善，但过大后会波动甚至下降：

- $M=1$ 时 Physics-Attention 退化成全局 pooling，物理状态之间没有 attention，性能显著下降；
- $M$ 增大意味着更细粒度的物理状态；
- $M$ 太大（如 512/1024）会把 physics domain 过度碎片化，token 数增加，注意力可能重新受到噪声干扰；
- 作者经验上认为 $32\sim256$ 比较容易调。

例如 Elasticity：$M=1$ 时误差 0.0148，$M=64$ 为 0.0064，$M=256$ 最低到 0.0054，而 $M=1024$ 回升到 0.0068。

### 8.2 learned slice vs fixed regular squares

即使 Darcy 本来就在规则网格上，把 learnable slice 替换为固定 $4\times4$ 方块也会退化：Darcy 从约 0.0059 变为 0.0088。

这支持了论文的核心观点：重要的划分依据不是几何上固定的局部 patch，而是由任务学习得到的物理状态。

### 8.3 与 full attention 对比

Darcy 上随着分辨率提高，plain Transformer 很快遇到显存问题：

| mesh points | Plain Transformer | Transolver |
|---:|---:|---:|
| 484 | 0.02017 | 0.02019 |
| 1,681 | 0.0103 | 0.0089 |
| 3,364 | 0.0073 | 0.0058 |
| 7,225 | 0.0081 | 0.0059 |
| 10,609 及以上 | OOM | 仍可运行 |

在很小的离散规模，两者相近；随着 $N$ 增大，Transolver 的优势逐渐出现。

### 8.4 attention 是否真的更“有信息量”

论文用注意力权重相对均匀分布的 Kullback-Leibler divergence（KL divergence）进行比较：

| Benchmark | Galerkin Transformer | Transolver |
|---|---:|---:|
| Elasticity | 0.3803 | **1.7795** |
| Darcy | 0.2739 | **1.8274** |

更大的 KL divergence 表示注意力明显偏离均匀分布。作者据此认为，对 physics-aware tokens 做 attention 比对海量 mesh points 做线性 attention 更容易形成尖锐、选择性的相关性。

> [!figure] 论文中图 5：learned slices 与 attention map
> ![](https://arxiv.org/html/2402.02366v2/visual_new_compress.png)
> 左侧显示即使随机删除一部分网格点，slice 仍能捕捉相似的物理区域；右侧显示 Transolver token attention 比 Galerkin Transformer 的 mesh-point attention 更集中。

## 9. 扩展性、OOD 与附录实验

### 9.1 分辨率、数据量和参数规模

Darcy 上作者把：

- 输入分辨率扩大到原设置约 25 倍；
- 训练数据量扩大到约 5 倍；
- 模型层数扩大到约 5 倍。

Transolver 在更高分辨率下保持较稳定的误差，并能从更多数据和更大参数量中获益。

其他 benchmark 的参数扩展实验中：

- Elasticity：8 层约 0.0064，40 层约 0.0047；
- Airfoil：8 层约 0.0053，40 层约 0.0037；
- Plasticity / Pipe 已较接近饱和，继续加深会出现波动。

### 9.2 自适应多尺度 slice

因为每个 slice 平均聚合约 $N/M$ 个点，所以 $M$ 可以理解为物理状态建模的“粒度”。作者尝试前 4 层 $M=64$、后 4 层 $M=32$：

| 配置 | Airfoil | Darcy | 内存 | 时间 |
|---|---:|---:|---:|---:|
| 全部 $M=64$ | 0.0053 | 0.0059 | 0.64 GB | 38.18 s/epoch |
| 多尺度 $64\rightarrow32$ | **0.0050** | **0.0056** | 0.62 GB | 38.10 s/epoch |

作者认为 Airfoil 和 Darcy 本身具有较明显的多尺度特性，因此分层改变 slice 粒度可能有益，但把系统化 multiscale Transolver 留作未来工作。

### 9.3 AirfRANS 的 OOD 泛化

训练集和测试集的 Reynolds number / angle of attack 完全不重叠：

- OOD Reynolds：训练 $[3,5]\times10^6$；测试 $[2,3]\times10^6\cup[5,6]\times10^6$；
- OOD angle：训练 $[-2.5^\circ,12.5^\circ]$；测试 $[-5^\circ,-2.5^\circ]\cup[12.5^\circ,15^\circ]$。

Transolver 结果：

| OOD setting | $C_L$ error ↓ | $\rho_L$ ↑ |
|---|---:|---:|
| Reynolds | **0.2996** | **0.9896** |
| Angle of attack | **0.1500** | **0.9950** |

说明它在这两个工况维度上具有较好的外推排序能力。

### 9.4 Lagrangian 初步实验

主文主要使用 Eulerian 数据。对于跟踪粒子的 Lagrangian 模拟，Transolver 虽然能直接读入散点，但缺失图结构中的显式局部邻接信息。因此作者没有直接替代 GNN，而是在 GNN message passing 旁路并联一个 Transolver layer。

WaterDrop 任务需要基于过去 6 步预测未来 994 步、共 2,000 个粒子：

- GNN：position MSE = 0.0182；
- GNN + Transolver：**0.0069**；
- 相对下降 **62.1%**。

这是论文对“Transolver 可以补充局部 GNN 的全局物理关系”的初步证据。

## 10. 效率

Physics-Attention 的主要收益随网格规模增大而更明显。附录在 $N=32,768$ 时给出的训练效率为：

| Model | GPU memory | time / epoch |
|---|---:|---:|
| Transolver | 2.36 GB | **98.552 s** |
| GNOT | 7.46 GB | 209.923 s |
| ONO | 10.09 GB | 462.459 s |
| OFormer | 2.44 GB | 182.205 s |
| Galerkin Transformer | **2.05 GB** | 129.872 s |

因此 Transolver 并非在每个规模、每个指标上都绝对最小显存，但其运行时间增长较慢，并在大网格上保持了较好的精度-效率折中。

## 11. 关键贡献

1. **改变 PDE Transformer 的 tokenization 层级**：从 mesh-point attention 转为 latent physical-state attention。
2. **Physics-Attention**：提出可学习的 Slice → Aggregate → Attention → DeSlice 机制，可处理规则网格、点云、结构/非结构网格与表面-体混合几何。
3. **复杂度降低**：在 $M\ll N$ 且固定时，由标准 attention 的 $\mathcal O(N^2)$ 降为 $\mathcal O(NMC+M^2C)$。
4. **端到端的几何泛化表示**：slice 不依赖规则 patch 或固定图拓扑，且在随机重采样/破损网格下仍能形成相似状态。
5. **实验覆盖全面**：从 6 个标准 PDE benchmark 到汽车/翼型工业任务，并测试了效率、扩展性、OOD 和 Lagrangian 混合设置。

## 12. 局限与启发

### 12.1 论文中直接体现的局限

- **$M$ 仍是超参数**：最佳 slice 数与目标 PDE 的物理复杂度有关；太小会过度聚合，太大会过度碎片化。
- **多尺度结构尚未系统设计**：附录只做了简单 $64\rightarrow32$ 的分层实验，作者明确留作未来工作。
- **Lagrangian 场景仍需要图局部信息**：论文自己指出单独使用 Transolver 会缺失粒子邻接关系，因此实验采用 GNN + Transolver。
- **工业数据仍有限**：AirfRANS 只有约 1,000 个高保真样本，论文也指出这导致较大的随机波动。
- **AirfRANS 主文没有把 drag estimation 作为重点**：原数据论文指出空气速度难以准确估计，因此本文主要评估 lift 相关指标。
- **基础模型仍是未来目标**：作者把大规模预训练 PDE solver 作为后续方向，而不是本文已经实现的结果。

### 12.2 阅读时需要特别区分的概念

- “physics-aware token” 是**学习得到的潜在表征**，其物理语义主要由可视化、性能与 attention 统计间接支持，并不是由真实“压力区/尾流区”标签监督。
- OOD 实验说明的是对 AirfRANS 中 Reynolds number 和 angle of attack 范围的外推能力，不能直接等价为“跨 PDE 方程族的通用泛化”。
- 理论部分说明 Physics-Attention 可解释为积分算子，但并未证明学习到的 slice 一定与真实物理状态一一对应，也没有提供对真实 PDE 解误差的严格有限样本界。

### 12.3 最值得迁移的架构思想

从模型设计角度，本文最重要的启发是：**当原始 token 数极大且单 token 语义太低层时，可以先学习一个任务驱动的软分组/latent tokenization，再在 latent token 上使用全局 attention。**

这一思想同时解决了两个不同问题：

- computational bottleneck：减少 attention token 数；
- semantic bottleneck：让 attention 作用于更高层、更容易建立关系的状态表示。

## 13. 关联与研究脉络

### [[Fourier Neural Operator]]：从“解单个 PDE”到“学习解算子”

FNO 建立了现代 neural operator 的代表性范式：学习参数函数到解函数的映射，并在 Fourier domain 中高效参数化非局部积分核。它说明 PDE surrogate 可以学习整个问题族，而不仅是一次数值求解。[联网资料 3][^web-fno]

**与本文的关系**：Transolver 仍属于数据驱动 neural operator，但不依赖 Fourier basis；核心非局部算子由 physics-aware token attention 实现，因此更强调复杂几何适应性。

### [[Choose a Transformer - Fourier or Galerkin]]：把 Transformer attention 解释为 operator learning

Galerkin Transformer 把 self-attention 引入 PDE operator learning，并从算子逼近角度讨论 attention 与积分/投影的关系，还设计线性化 attention 以改善计算效率。[联网资料 2][^web-galerkin]

**与本文的关系**：Transolver 继承“attention 可以作为 learnable integral operator”的思想，但认为直接在 mesh point 上计算 attention 仍然语义过低，因此把积分/attention 的离散单元进一步换成 learned slices。

### [[GNOT]]：通用 neural operator Transformer 与线性 attention

GNOT（General Neural Operator Transformer）等工作使用高效/线性 attention 来处理 operator learning，能够覆盖较广的 PDE 输入输出形式。Transolver 在相关工作中把 GNOT、OFormer、ONO 归为“仍直接对大量 mesh point 做 attention”的路线。

**与本文的关系**：研究脉络可以概括为：

$$
\text{FNO: spectral operator}
\rightarrow
\text{Galerkin/GNOT: attention operator on mesh points}
\rightarrow
\text{Transolver: attention operator on learned physical-state tokens}.
$$

这也是 Transolver 相对于前代 PDE Transformer 最核心的结构创新。

## 14. 参考的联网概念资料

[^web-neural-operator]: Kovachki et al., *Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs*, JMLR 2023: https://www.jmlr.org/papers/v24/21-1524.html
[^web-galerkin]: Shuhao Cao, *Choose a Transformer: Fourier or Galerkin*, NeurIPS 2021: https://proceedings.neurips.cc/paper/2021/hash/d0921d442ee91b896ad95059d13df618-Abstract.html
[^web-diffeomorphism]: *Diffeomorphism*, Encyclopedia of Mathematics: https://encyclopediaofmath.org/wiki/Diffeomorphism
[^web-fno]: Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*, arXiv:2010.08895: https://arxiv.org/abs/2010.08895
