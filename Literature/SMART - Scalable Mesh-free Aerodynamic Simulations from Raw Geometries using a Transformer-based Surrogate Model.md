---
title: "SMART: Scalable Mesh-free Aerodynamic Simulations from Raw Geometries using a Transformer-based Surrogate Model"
authors:
  - Jan Hagnberger
  - Mathias Niepert
year: 2026
venue: "Proceedings of the 43rd International Conference on Machine Learning (ICML 2026), PMLR 306"
doi: "10.48550/arXiv.2601.18707"
source: "/mnt/data/SMART.pdf"
arxiv: "2601.18707v2"
tags:
  - neural-surrogate
  - aerodynamics
  - transformer
  - mesh-free
  - neural-operator
  - point-cloud
  - PDE
---

> [!abstract] 原文摘要翻译
> 基于机器学习（Machine Learning, ML）的代理模型已成为复杂几何体物理仿真的高效替代方案。许多现有方法把仿真网格作为额外输入，从而降低预测误差；但为新几何体生成仿真网格本身代价很高。相反，不依赖仿真网格的无网格方法通常误差更大。为解决这一矛盾，本文提出 SMART：一种只使用几何体点云、无需访问仿真网格，就能在任意查询位置预测物理量的神经代理模型。模型把几何信息和仿真参数编码到共享潜空间，使其同时表达结构特征和物理场的参数化特征；随后，物理解码器对编码器各层的中间潜表示进行交叉注意力（cross-attention），将空间查询映射为对应物理量。借助这种跨层交互，模型在网络深度方向上同时推进潜在几何表征和正在形成的物理场。大量实验表明，SMART 在不使用仿真网格的前提下，仍能达到或超过依赖仿真网格的方法，并可扩展到工业规模的空气动力学仿真。

# 一句话结论

SMART 的关键不是简单地“把 CFD 网格拿掉”，而是把 **几何体本身压缩成一个只读、逐层更新的 latent geometry（潜在几何）**，再让每个空间查询点独立地从这些潜在几何中读取信息。这样既保留了 Transformer 对复杂几何关系的建模能力，又避免让查询点之间互相影响，因此可以在推理时把数千万个查询点任意分块、顺序计算，而不改变结果。

# 1. 问题与动机

## 1.1 两种“网格”必须区分

本文反复强调，空气动力学流程里实际存在两类不同对象：

1. **几何网格**：由计算机辅助设计（Computer-Aided Design, CAD）模型导出的三角面片，例如 Standard Tessellation Language（STL）表面网格。它主要描述物体形状，生成相对便宜。
2. **仿真网格**：数值求解器真正用于离散偏微分方程（Partial Differential Equation, PDE）的表面与体网格。它通常会在边界层、尖角、高梯度区等位置进行加密，生成昂贵，而且不同几何体的网格点数和空间分布不同。

SMART 所谓 **mesh-free（无网格）**，准确含义是：

- 训练标签仍然来自数值求解器，因此训练时真实物理量通常只在 CFD 仿真网格点上可用；
- 但这些网格点在 SMART 中只作为“监督采样位置”，**不会作为结构性输入塑造模型的内部状态**；
- 推理新几何体时不需要预先生成 CFD 优化网格，只需要几何点云和任意想查询的位置。

这比“训练和推理从头到尾都没有任何网格”更准确。

## 1.2 学习目标

几何体表示为三维点云

$$
G=\{(x,y,z)_i\}_{i=1}^{N},
$$

仿真参数为

$$
\boldsymbol{\xi}\in\mathbb{R}^{N_p},
$$

例如偏航角、攻角、马赫数等。对于任意空间位置 $\mathbf{p}=(x,y,z)\in\Omega$，目标是学习时间平均后的稳态/统计稳态解函数

$$
\mathbf{u}:\Omega\times\mathcal{G}\times\mathbb{R}^{N_p}\rightarrow\mathbb{R}^{N_c},
$$

即

$$
(\mathbf{p},G,\boldsymbol{\xi})
\mapsto
\mathbf{u}(\mathbf{p},G,\boldsymbol{\xi}).
$$

神经网络 $f_\theta$ 用训练数据近似它：

$$
\mathbf{u}(\mathbf{p},G,\boldsymbol{\xi})
\approx
f_\theta(\mathbf{p},G,\boldsymbol{\xi}).
$$

对于一组查询点

$$
P=\{\mathbf{p}_j\}_{j=1}^{M},
$$

模型需要输出这些位置的压力、速度等物理量。

## 1.3 作者提出的三个设计要求

### D1：效率与精度

代理模型必须显著快于传统数值求解器，同时误差仍可接受。

### D2：任意查询点彼此独立

对于任意包含点 $\mathbf p$ 的查询集合 $P$，必须满足

$$
f_\theta(\mathbf p,G,\boldsymbol\xi)
=
f_\theta(P,G,\boldsymbol\xi)_j.
$$

也就是说，同一个位置的预测不能因为“同时还查询了哪些别的位置”而改变。

这条要求直接排除了查询点之间的 self-attention（自注意力）。如果查询点彼此做 self-attention，那么某一点的表示就会依赖同一批次中的其他点。

### D3：不依赖 CFD 仿真网格

仿真网格不能作为模型输入特征。原因不是网格没用，而是它太有用，以至于模型会学会依赖求解器的离散方式；当测试时换成 CAD 表面点或均匀体采样时，预测会明显恶化。

# 2. SMART 方法

整体模型可以写成

$$
f_\theta(P,G,\boldsymbol{\xi})
=
\mathcal D_\theta
\left(
\tilde P,\,
\underbrace{\mathcal E_\theta(\tilde G,\boldsymbol{\xi})}_{\text{latent geometries}},\,
\boldsymbol{\xi}
\right),
\tag{1}
$$

其中：

- $\tilde G$：编码后的几何点坐标；
- $\tilde P$：编码后的查询坐标；
- $\mathcal E$：geometry encoder（几何编码器）；
- $\mathcal D$：physics decoder（物理解码器）。

SMART 不是只产生一个最终 latent，而是编码器每一层都产生一个中间 latent geometry，解码器的第 $l$ 层直接读取编码器第 $l$ 层的中间表示。

---

## 2.1 Modulated Positional Encoding：可调制位置编码

标准 Transformer 使用固定多频率正弦/余弦基来编码坐标。SMART 认为空气动力学场中不同空间区域的频率需求不同：例如平滑来流主要是低频，而激波、边界附近或尾迹可能需要更高频成分。

因此作者提出 Modulated Positional Encoding（MPE，可调制位置编码）。对坐标 $x$ 的第 $2i$ 个正弦通道：

$$
\operatorname{MPE}(x,2i)
=
\sin\left(
\gamma_{2i}(x)\frac{x}{10000^{2i/d}}
+
\beta_{2i}(x)
\right),
$$

其中

$$
\gamma_{2i}(x),\beta_{2i}(x)=\operatorname{MLP}(x).
$$

Multi-Layer Perceptron（MLP，多层感知机）根据坐标本身产生缩放和偏移：

- $\gamma(x)$ 改变该通道的有效空间频率；
- $\beta(x)$ 改变相位。

余弦通道使用同样的调制方式。

三维坐标分别编码后拼接：

$$
\operatorname{MPE}(x,y,z)
=
\operatorname{MPE}(x)\Vert
\operatorname{MPE}(y)\Vert
\operatorname{MPE}(z).
$$

于是得到

$$
\tilde G\in\mathbb R^{N\times d},
\qquad
\tilde P\in\mathbb R^{M\times d}.
$$

消融实验表明 MPE 有稳定但不算决定性的收益：在 SHIFT-SUV 和 SHIFT-Wing 上，相比标准 Transformer 位置编码，表面/体场误差约改善 2%–3%。

---

## 2.2 Geometry Encoder：把百万级几何点压成固定大小 latent geometry

编码器的目标是

$$
\mathcal E(\tilde G,\boldsymbol\xi)
=
\left(
E_G^{(1)},\ldots,E_G^{(L)}
\right),
\qquad
E_G^{(l)}\in\mathbb R^{K\times d}.
\tag{2}
$$

这里 $K$ 是固定 latent token 数，通常 $N\gg K$。

这一步非常重要：模型的内部几何表示大小由 $K$ 决定，而不再由原始几何点数 $N$ 或最终查询点数 $M$ 决定。

### 2.2.1 先用真实几何点构造 coarse geometry，而不是 learnable tokens

SMART 首先从输入几何中均匀采样 $K$ 个点：

$$
\tilde C=\{\tilde c_k\}_{k=1}^{K},
\qquad
\tilde c_k\sim\mathcal U(\tilde G),
$$

并令

$$
E^{(0)}:=\tilde C\in\mathbb R^{K\times d}.
$$

因此最初的 latent token 本身就有明确的空间位置和几何含义。

这与许多 Perceiver 类架构从一组纯 learnable latent queries 开始不同。本文消融非常强烈地支持这一选择：在 SHIFT-Wing 上，把 coarse geometry 换成 learnable tokens 后，

- 表面误差上升 **181%**；
- 体场误差上升 **148%**。

作者的解释是：真实粗几何在编码开始之前就提供了额外的几何 scaffold（支架），而纯可学习 token 需要从零开始通过注意力建立与几何的对应。

### 2.2.2 每个编码器块重新采样原始几何

第 $l$ 个编码器块先从完整几何中重新均匀采样 $N'$ 个点：

$$
\tilde G_{\text{sub}}
=
\{\tilde g_n\}_{n=1}^{N'},
\qquad
\tilde g_n\sim\mathcal U(\tilde G).
\tag{3}
$$

然后用 latent geometry 作为 Query，用采样后的原始几何作为 Key/Value：

$$
E_G^{(l)}
=
\operatorname{CrossAttn}
\left(
Q=E^{(l-1)},
KV=\tilde G_{\text{sub}}
\right).
\tag{4}
$$

这里的作用可以理解为：每一层都让固定数量的 latent geometry tokens 再次“回头读取”真实几何。

每层重新采样而不是始终使用同一子集，使模型在多层传播过程中能够覆盖非常大的点云，同时避免每层都对全部几何点做注意力。

### 2.2.3 latent geometry 内部再做一次 refinement

得到 $E_G^{(l)}$ 后，模型还进行

$$
E_{CA}^{(l)}
=
\operatorname{CrossAttn}
\left(
Q=E^{(l-1)},
KV=E_G^{(l)}
\right).
$$

这一步把刚从原始几何读取到的信息重新整合回 coarse latent geometry，并建模 latent tokens 之间的空间关系。

### 2.2.4 用 FiLM 风格调制把工况注入 latent geometry

编码器随后使用逐 token 的 MLP：

$$
E^{(l)}
=
W_2
\left(
\sigma(W_1E_{CA}^{(l)}+b_1)
\odot
\gamma(\boldsymbol\xi)
+
\beta(\boldsymbol\xi)
\right)
+b_2,
$$

其中

$$
\gamma(\boldsymbol\xi),\beta(\boldsymbol\xi)
=
\operatorname{MLP}(\boldsymbol\xi).
\tag{5}
$$

这与 Feature-wise Linear Modulation（FiLM，逐特征线性调制）的基本形式一致：

$$
h'=\gamma(z)\odot h+\beta(z).
$$

联网核对 FiLM 原始论文后，可以把这一机制理解为：**不把工况参数简单拼在 token 后面，而是让工况直接控制隐藏特征每个通道的增益与偏置**。因此相同几何体在不同偏航角、攻角或马赫数下可以被映射到不同的物理相关 latent state。

### 2.2.5 每层同时产生两种输出

每个编码器块保留：

1. $E_G^{(l)}$：几何 cross-attention 后的中间 latent geometry，送给对应深度的 decoder；
2. $E^{(l)}$：经过 refinement 与工况调制 MLP 后的表示，送给下一 encoder block。

因此 encoder 不是只在最后输出一个几何表示，而是产生一条逐层演化的 latent geometry 序列。

---

## 2.3 Physics Decoder：查询点只“读” latent geometry，不互相通信

初始查询表示为

$$
D^{(0)}:=\tilde P\in\mathbb R^{M\times d}.
$$

第 $l$ 个 decoder block 使用第 $l$ 个 encoder block 的中间 latent geometry：

$$
D_{CA}^{(l)}
=
\operatorname{CrossAttn}
\left(
Q=D^{(l-1)},
KV=E_G^{(l)}
\right).
\tag{7}
$$

随后再应用与编码器相同思路的 simulation-parameter modulated MLP。

最终

$$
D^{(L)}\in\mathbb R^{M\times d}
$$

经过一个输出 MLP 映射到

$$
\hat{\mathbf u}(P,G,\boldsymbol\xi)
\in
\mathbb R^{M\times N_c}.
$$

### 为什么它严格满足 query independence？

对任意查询点 $\mathbf p_j$：

- 位置编码是 pointwise 的；
- MLP 是 pointwise 的；
- cross-attention 的 Key/Value 只来自固定的 latent geometry；
- 不存在 query-query self-attention；
- query 也不会回写、修改 latent geometry。

所以某个查询点的计算图里没有其他查询点的表示。

因此：

- 调换查询点顺序不会改变对应输出；
- 把一个大查询集合拆成任意小块不会改变结果；
- 单点查询和把该点放进一个百万点批次中得到的结果相同。

这不是普通的“支持任意坐标输入”，而是更强的 **集合组成不变性**。

---

## 2.4 Cross-layer Geometry–Physics Update：SMART 最关键的结构创新

严格 encode-process-decode 架构通常只把最终 encoder 表示 $E^{(L)}$ 交给 decoder。

SMART 则在每一个深度建立配对连接：

$$
E_G^{(1)}\rightarrow D^{(1)},
\quad
E_G^{(2)}\rightarrow D^{(2)},
\quad \ldots,\quad
E_G^{(L)}\rightarrow D^{(L)}.
$$

作者给出两个作用：

### 1. Joint refinement of geometry and physics

随着深度增加：

- encoder 中 latent geometry 持续变化；
- decoder 中 query embedding 逐渐变成 latent physical field。

每层 cross-attention 把这两条演化路径耦合起来，因此物理场不是在“几何编码完成后才开始解码”，而是在每一层都持续受到当前几何表示约束。

### 2. Multi-scale geometric grounding

作者认为浅层可能更偏粗尺度几何，深层逐渐包含更细节的结构；decoder 逐层读取这些表示，相当于同时获得多个表征深度的几何信息。

理解上可以把它类比为 U-Net 的跨层 skip connection，但 SMART 不是特征拼接，而是 **decoder query 对对应 encoder latent 做 cross-attention**。

这是整篇论文最重要的消融项之一。在 SHIFT-Wing 上，如果所有 decoder block 都只读取最终 encoder 表示：

- 表面误差增加 **140%**；
- 体场误差增加 **145%**。

因此论文的实验支持：SMART 的性能不是“有个 latent encoder + query decoder”就能得到的，逐层 geometry-physics coupling 是核心。

---

## 2.5 参数共享

论文方法部分说明 decoder 的 cross-attention 与 MLP 可以和对应 encoder block 的同类层共享权重；实验超参数表中所有 SMART 配置均开启了 **Shared Attention Layer and MLP**。

这意味着 encoder/decoder 不是完全独立的两套变换，而是进一步通过权重共享施加结构约束并降低参数量。

# 3. 训练与推理

## 3.1 训练时不可能一次使用全部 CFD 点

工业数据每个样本可包含数百万到数千万表面/体点。SMART 利用 query independence，在每个训练步骤从完整 CFD 网格中随机抽取查询点，只在这些点上计算损失。

训练目标的随机近似为：

$$
\arg\min_\theta
\mathbb E_{
(G,\boldsymbol\xi,P,\mathbf u)\sim\mathcal D_{\text{train}},
\,
p_1,\ldots,p_{M'}\sim\mathcal U(P)
}
\left[
\mathcal L
\left(
f_\theta(P,G,\boldsymbol\xi),
\mathbf u(P,G,\boldsymbol\xi)
\right)
\bigg|_{P=\{p_1,\ldots,p_{M'}\}}
\right].
$$

只要长期随机采样覆盖整个空间，模型在期望意义下仍能看到完整物理场。

## 3.2 损失：按物理通道分别计算 relative L2

对真实值

$$
Y\in\mathbb R^{M\times N_c}
$$

和预测值

$$
\hat Y\in\mathbb R^{M\times N_c},
$$

单样本 relative L2 loss 为

$$
\mathcal L(\hat Y,Y)
=
\frac{1}{N_c}
\sum_{c=1}^{N_c}
\frac{
\|\hat Y_{\cdot,c}-Y_{\cdot,c}\|_2
}{
\|Y_{\cdot,c}\|_2
}.
\tag{10}
$$

按通道分别归一化的作用是避免量纲/数值幅值较大的通道主导损失，例如 $x$ 向速度幅值通常明显大于横向速度。

训练时分别计算：

- surface loss：表面压力；
- volume loss：体场 $x,y,z$ 三个速度分量；

最终优化两者之和。

## 3.3 推理时任意分块

完整分辨率可能有几千万查询点。由于查询彼此严格独立，可以把 $P$ 分成多个子区域：

$$
P=P_1\cup P_2\cup\cdots\cup P_B,
$$

逐块运行 decoder。

理论上和一次性把所有点输入模型完全等价，但显存只需要容纳当前 query chunk。

这正是 SMART 能在单张 GPU 上对千万级输出场做推理的关键。

# 4. 实验设置

## 4.1 数据集

| 数据集 | 任务/特点 | 样本划分 | 平均几何点 | 平均表面点 | 平均体点 |
|---|---|---:|---:|---:|---:|
| ShapeNetCar | 简化车辆，时间平均外流场 | 789 train / 100 test | 3,682 | 3,682 | 29,498 |
| AhmedML | Ahmed body；高保真 Detached-Eddy Simulation（DES，分离涡模拟），结合 Reynolds-Averaged Navier–Stokes（RANS）与 Large-Eddy Simulation（LES，大涡模拟）思想 | 400 / 100 | 101,405 | 1,069,004 | 21,195,967 |
| SHIFT-SUV | 工业级 SUV 外流；使用 estate、full-scale 子集 | 798 / 200 | 2,548,037 | 2,521,717 | 50,507,366 |
| SHIFT-Wing | NASA Common Research Model 参数化飞机；包含马赫数与攻角；使用自适应网格捕捉跨声速激波 | 800 / 200（只取前 1000 个样本） | 1,623,086 | 3,246,168 | 5,970,264 |

补充：

- AhmedML 单个几何的网格生成约需 30 min，完整高保真模拟约 48 h。
- SHIFT-Wing 的网格是仿真过程中自适应细化得到的，因此若代理模型把这种网格当输入，实际推理新几何时就会陷入“先跑完整 CFD 才能得到代理模型输入”的悖论。

## 4.2 Baselines

论文比较：

- Geometry-Informed Neural Operator（GINO）
- Operator Transformer（OFormer）
- General Neural Operator Transformer（GNOT）
- Transolver
- Latent Neural Operator（LNO）
- Geometry-Preserving Universal Physics Transformer（GP-UPT）
- Mesh-Dependent Anchored-Branched Universal Physics Transformer（MD AB-UPT）
- Mesh-Free Anchored-Branched Universal Physics Transformer（MF AB-UPT）

其中：

- GNOT、Transolver、MD AB-UPT 使用 CFD 仿真网格作为输入；
- 其余方法按本文定义属于 mesh-free。

值得注意的是，论文并非完全照搬所有 baseline 的原始实现，而对部分模型做了面向三维空气动力学的适配，例如：

- OFormer 被作者扩展为 3D；
- OFormer、GNOT、Transolver、LNO 的坐标编码被改为 Transformer 风格位置编码，因为作者发现比各自默认设置误差更低；
- 多个 baseline 使用 adaptive Layer Normalization Zero（adaLN-Zero，自适应零初始化层归一化）注入附加工况参数；
- GP-UPT 由作者根据论文描述重新实现，并直接用几何点云训练；
- MF AB-UPT 用几何表面点作为 surface anchors，并用规则三维网格点作为 volume anchors。

因此结果应理解为“在本文统一实验框架和作者调优后的实现”之间的比较。

## 4.3 主实验训练方式

RQ1 中所有模型在每个样本上随机使用：

- 16k surface query points；
- 16k volume query points。

随后 RQ2 直接把这些训练好的模型用于完整分辨率的百万/千万级查询。

每个模型使用多个不同随机初始化训练，正文报告均值，附录给出标准差。

## 4.4 SMART 超参数

| 数据集 | hidden dim $d$ | latent geometry $K$ | 每层 sampled geometry $N'$ | encoder/decoder blocks | lr | precision | 参数量 |
|---|---:|---:|---:|---:|---:|---|---:|
| ShapeNetCar | 128 | 3,682 | 3,682 | 6 / 6 | $10^{-4}$ | float16 | 2.01M |
| AhmedML | 256 | 2,048 | 16,384 | 6 / 6 | $10^{-4}$ | float16 | 7.94M |
| SHIFT-SUV | 256 | 2,048 | 8,192 | 6 / 6 | $10^{-4}$ | float16 | 7.94M |
| SHIFT-Wing | 256 | 4,096 | 4,096 | 8 / 8 | $5\times10^{-5}$ | float32 | 12.43M |

统一设置：

- batch size = 1；
- 200 epochs；
- optimizer = Lion；
- encoder/decoder attention 与 MLP 开启参数共享；
- SHIFT-Wing 使用 gradient norm clipping = 2.0；
- 其余三个数据集不使用 gradient clipping。

训练硬件为 NVIDIA A100 SXM4，单个 SMART 模型训练约需 3–16 h，具体取决于数据集与精度。

## 4.5 $K$ 应该怎么选？

论文特别说明：latent geometry token 数 $K$ 不应简单跟 CFD 网格点数成比例。

$K$ 更应该由以下因素决定：

- 几何复杂度；
- 物理结构复杂度；
- 有效空间尺度。

例如 SHIFT-Wing 有长距离、相干的 trailing vortices（尾涡），因此使用 $K=4096$；SHIFT-SUV 是较短、快速破碎的钝体尾迹，$K=2048$ 即可。

# 5. 实验结果

所有误差均沿用论文表格的 **Relative L2 Error $\times10^{-2}$** 标度。

## 5.1 RQ1：16k + 16k 子采样分辨率

SMART 在四个数据集的 surface 与 volume 指标上全部取得最佳均值：

| 数据集 | SMART Surface | SMART Volume | 主要现象 |
|---|---:|---:|---|
| ShapeNetCar | 6.7734 | 5.0215 | 小规模数据上仍略优于 GP-UPT / AB-UPT |
| AhmedML | 3.1195 | 5.1511 | 与 MF AB-UPT 很接近，但 surface 更优 |
| SHIFT-SUV | 0.0175 | 6.2627 | 优于 Transolver、MD/MF AB-UPT |
| SHIFT-Wing | 0.0557 | 14.6639 | 优势最明显；MF AB-UPT 的 volume error 达 39.4740 |

SHIFT-Wing 是最有说服力的测试，因为它使用仿真过程中形成的自适应网格；此时“依赖 CFD 网格输入”在真实新样本推理中尤其不可行。

## 5.2 RQ2：直接查询完整分辨率

训练仍然只使用 16k surface + 16k volume 随机点，但测试时查询完整百万/千万级点场。

结果基本维持 RQ1 排名：

- SMART 除 AhmedML volume 略逊于 MF AB-UPT 外，其余指标均最佳；
- GNOT 和 Transolver 在大数据集完整分辨率推理时，在单张 A100 80GB 上发生 out-of-memory；
- query-independent 模型可以分块推理，因此不需要同时把所有查询点放入显存。

SMART 完整分辨率误差：

| 数据集 | Surface | Volume |
|---|---:|---:|
| ShapeNetCar | 6.7734 | 5.0357 |
| AhmedML | 3.2938 | 5.3050 |
| SHIFT-SUV | 0.0175 | 6.2946 |
| SHIFT-Wing | 0.0559 | 14.7187 |

## 5.3 RQ3：彻底换掉 CFD 优化查询网格

这是验证“mesh-free 是否真的有用”的关键实验。

SHIFT-SUV 测试时改成：

- surface：CAD-derived mesh，约 600k 点；
- volume：近似均匀网格，约 2M 点。

训练时模型仍然只见过 CFD 优化网格上的监督点。

结果：

| 模型 | Surface | Volume |
|---|---:|---:|
| SMART | **0.0176** | **5.9543** |
| MF AB-UPT | 0.0181 | 7.3500 |
| Transolver | 0.0622 | 25.0725 |
| GNOT | 0.0765 | 45.9890 |
| MD AB-UPT | 0.1172 | 140.9773 |

MD AB-UPT 的体场误差从正常设置下约 6.6 激增到约 141，直接证明了“把 CFD 网格当结构输入”会形成强烈的 solver-induced mesh bias（求解器网格偏置）。

SMART 基本没有这种崩溃。

## 5.4 RQ4：查询分布发生空间区域偏移

训练时见完整空间域，测试时只查询 SUV 尾部区域，模拟“只关心 spoiler 后方尾迹”的设计场景。

| 模型 | Rear Surface | Rear Volume |
|---|---:|---:|
| SMART | **0.0147** | **5.9319** |
| MF AB-UPT | 0.0156 | 6.2652 |
| Transolver | 0.0321 | 18.0383 |
| MD AB-UPT | 0.0660 | 33.6630 |

结论与 RQ3 一致：真正把查询坐标当独立 probe（探针）的 mesh-free 模型，对测试时查询分布变化更稳健。

## 5.5 气动力积分

论文还用预测表面场计算净气动力

$$
\mathbf F
=
\int_S
\left(
-p\mathbf n+\boldsymbol\tau_w
\right)\,ds,
$$

离散为

$$
\mathbf F
\approx
\sum_{s_i\in S}
\left(
-p(s_i)\mathbf n(s_i)
+
\boldsymbol\tau_w(s_i)
\right)
A(s_i).
$$

由此投影得到 drag / lift。

决定系数（coefficient of determination, $R^2$）：

| 数据集 | Drag $R^2$ | Lift $R^2$ |
|---|---:|---:|
| SHIFT-SUV | 0.94 | 0.86 |
| SHIFT-Wing | 1.00 | 1.00 |

SHIFT-SUV 的较大 lift 区间表现较弱，作者指出该区域训练样本稀少，AB-UPT 中也观察到类似现象。

## 5.6 推理成本

在 SHIFT-Wing：

- 8 encoder + 8 decoder blocks；
- $K=4096$；
- float32；
- 5 million queries；

单张 A100 80GB 上 wall-clock inference time 为

$$
17.11\pm0.0656\ \text{s}.
$$

在 SHIFT-SUV 约 53M mesh points 的完整输出上：

- SMART 只需约 7GB GPU memory；
- Transolver++ 在单张 A100 80GB 上无法运行；
- 作者根据 Transolver++ 论文的 memory scaling 线性外推，估计需要约 20 张 A100 80GB 才能处理该规模。

这里“约 20 张”是作者的外推估计，不是实际实测。

# 6. 消融实验：到底什么最重要？

## 6.1 Latent token 数 $K$

SHIFT-SUV：

| $K$ | Surface | Volume |
|---:|---:|---:|
| 128 | 0.0181 | 6.6090 |
| 256 | 0.0180 | 6.5126 |
| 512 | 0.0178 | 6.4245 |
| 1024 | 0.0176 | 6.2957 |
| 2048 | 0.0175 | 6.2627 |
| 4096 | **0.0174** | **6.2076** |

趋势稳定单调：更大的 latent geometry 提供更多容量，但即使 $K=128$ 也已经可以超过许多 baseline。

## 6.2 MPE

去掉 MPE、改回标准 Transformer positional encoding：

- SHIFT-SUV：surface +2%，volume +3%；
- SHIFT-Wing：surface +2%，volume +2%。

有收益，但不是主要性能来源。

## 6.3 Coarse geometry initialization

换成 learnable latent tokens：

- SHIFT-Wing surface：+181%；
- SHIFT-Wing volume：+148%。

**极重要。**

## 6.4 Encoder geometry cross-attention

去掉每层“回读原始几何”的 cross-attention，在 $K=256$ 的 SHIFT-SUV 上：

- surface：+4%；
- volume：+10%。

说明仅仅保留最初的粗几何还不够，逐层继续读取原始点云能补充细节。

## 6.5 Cross-layer geometry-physics update

decoder 只使用最终 encoder representation：

- SHIFT-Wing surface：+140%；
- SHIFT-Wing volume：+145%。

**与 coarse geometry initialization 并列为最关键结构。**

## 6.6 Uniform sampling vs. Farthest Point Sampling

用 Farthest Point Sampling（FPS，最远点采样）替代 uniform sampling 后误差只变化约 0%–0.6%。

因此作者选择均匀随机采样，因为在百万级点云上明显更便宜。

# 7. 为什么 SMART 能比普通 latent encoder-decoder 更强？

可以把结构拆成四个层级理解：

1. **真实几何作 latent scaffold**  
   latent token 一开始就具有坐标与几何身份，而不是匿名向量。

2. **多层重新读取原始几何**  
   不是一次性把点云压缩完就丢掉，而是每层用新的 $\tilde G_{\text{sub}}$ 更新 latent。

3. **工况通过 FiLM 式通道调制进入网络**  
   工况改变的不只是最后输出，而是 latent feature 的演化轨迹。

4. **decoder 每层读取对应深度的 latent geometry**  
   物理场和几何场在网络深度方向共同演化。

因此 SMART 真正的“processor”不是一个单独的 Transformer bottleneck，而是 **encoder latent geometry 与 decoder latent physics 之间的逐层耦合过程**。

# 8. 局限

论文明确给出以下限制：

1. **只验证外部空气动力学**  
   目前主要是汽车和航空外流场，没有验证叶轮机械、生物流体等其他流体问题。

2. **只处理 time-independent / time-averaged field**  
   没有时间推进机制，不能直接预测瞬态演化。作者把 latent time stepping 作为未来方向。

3. **没有显式满足物理守恒律**  
   当前架构和损失没有强制质量、动量等守恒约束。

4. **随机 query sampling 可能漏掉稀有局部结构**  
   若重要现象只占很小空间区域，均匀随机采样可能欠采样。

附录还暴露了一个重要的经验局限：

5. **跨形状类别零样本泛化较弱**  
   用 SHIFT-SUV estate 训练、直接测试 fastback：

   - subsampled surface error：+84%；
   - subsampled volume error：+167%；
   - full-resolution volume error：+168%。

   论文因此认为，对显著不同的 shape category，仍需要少量 fine-tuning 或更丰富的多类别训练集。

# 9. 对方法的启发

## 9.1 “mesh-free”真正需要的是 representation independence

如果网络内部表征由 query mesh 的密度、拓扑或分布共同构建，那么即使接口形式上接受任意坐标，它也不是真正意义上的 mesh-independent surrogate。

SMART 给出的更严格判断标准是：

> query 只能读取几何/物理全局表示，不能参与构造或修改该表示。

## 9.2 query independence 同时是一个建模约束和系统工程约束

禁止 query-query self-attention 会牺牲一部分显式空间相关性，但换来的好处是：

- 任意空间分块；
- 固定显存；
- 测试时 query distribution shift 更稳健；
- 推理输出分辨率与模型内部状态解耦。

所以这不是纯粹为了“理论漂亮”，而是为千万点工业输出服务的结构性约束。

## 9.3 Latent token 最好有领域语义

消融结果显示，latent geometry 使用真实几何点初始化远胜于匿名 learnable tokens。

这说明在科学机器学习中，一个固定长度 latent bottleneck 并不一定要是抽象向量；如果 latent token 可以继承空间坐标、几何点或物理锚点，它可能获得更强 inductive bias（归纳偏置）。

## 9.4 条件参数最好参与中间特征更新

工况 $\boldsymbol\xi$ 没有只在输入或输出拼接一次，而是逐层调制 MLP。这样做与 FiLM 的思想一致：条件变量控制“网络如何处理特征”，而不是仅作为附加数值被动传播。

# 10. 关键概念补充（联网核对）

## 10.1 FiLM：Feature-wise Linear Modulation

FiLM 的标准形式是

$$
\operatorname{FiLM}(h|z)
=
\gamma(z)\odot h+\beta(z).
$$

条件 $z$ 经过小网络生成每个 feature channel 的 scale 与 shift。原始 FiLM 工作表明，这种简单的逐特征仿射变换可以有效控制深层网络计算。

SMART 使用完全同类的思想，把 simulation parameters 作为 $z$，在 encoder 和 decoder 的 MLP 中调制中间激活。

核心区别于“concat condition”：

- concat：条件只是额外输入；
- FiLM：条件直接改变每个隐藏通道的响应幅值和偏置。

## 10.2 Perceiver 式 latent bottleneck：与 SMART 的相似与不同

Perceiver 的核心是用少量 latent queries 对超大输入做不对称 cross-attention，把高维输入压缩进固定 latent bottleneck。

SMART 与它相似之处：

- 都让固定数量 $K$ 的 latent tokens 从超大输入中读取信息；
- 都避免直接在全部输入点上做全局 self-attention。

但 SMART 的关键不同点是：

- 初始 latent 不是任意 learnable queries，而是从真实几何点中抽样得到；
- encoder 每一层都会再次读取新的原始几何子集；
- decoder 不只看最终 latent，而是读取每一层 intermediate latent geometry。

因此可以把 SMART 理解为一个 **geometry-anchored、multi-stage latent cross-attention architecture**。

# 11. 相关工作与研究脉络

## 11.1 Vectorized Conditional Neural Fields

同一研究团队此前把 PDE 解表示为条件神经场，并强调参数条件化、连续查询和高效推理。

与 SMART 的联系：

- 都把“在任意坐标处查询物理场”作为核心接口；
- 都允许条件参数影响场预测。

重要差别：

- 早期 Vectorized Conditional Neural Fields 会显式建模多个时空 query 之间的 attention dependency；
- SMART 为工业规模稳态 CFD 进一步提出严格 query independence，禁止 query-query self-attention。

从研究方向上看，这是从“可查询的条件神经场”向“可任意分块的大规模物理场解码”推进。

## 11.2 CALM-PDE

同一第一作者后续/相邻工作继续研究“在 compressed latent space 中求解 PDE”，但使用连续、自适应卷积来减少 Transformer attention 在非规则空间上的内存开销。

与 SMART 的共同主线：

- 不在完整高分辨率物理空间中直接进行所有计算；
- 先把信息压缩到更小 latent representation；
- 再从 latent representation 恢复任意空间位置的解。

SMART 把这一思想进一步放到复杂三维 CAD 几何和工业空气动力学中，并采用 geometry-aware cross-attention 形成 latent geometry。

## 11.3 AB-UPT

Anchored-Branched Universal Physics Transformer 是 SMART 最直接的工业 CFD 架构参照之一。

共同点：

- 使用压缩 latent representation；
- 用 anchors / latent tokens 支持百万级 query；
- 强调 query independence；
- 几何与物理表示之间存在持续的信息交换。

SMART 明确指出其 cross-layer geometry-physics interaction 与 AB-UPT 的 anchor attention 思路相近。

主要差异：

- MD AB-UPT 仍从 CFD simulation mesh 中取 anchors，因此存在 mesh dependency；
- AB-UPT 使用 surface / volume 两个 branch；
- SMART 使用统一 branch，直接从 raw geometry latent 解码 surface 与 volume 查询；
- SMART 的 encoder 还会逐层重新采样并读取原始几何。

### 作者研究脉络总结

可以概括为：

**conditional neural field → compressed latent PDE modeling → geometry-anchored latent representation + query-independent industrial CFD decoder**

即逐步把“连续可查询的物理场”与“压缩潜空间计算”结合，并最终针对工业级复杂几何加入 mesh-free 与跨层 geometry-physics coupling。

# 12. 与其他 Transformer PDE 架构的关联

## [[Transolver]]

两者都试图避免在百万点上做普通二次复杂度 self-attention，但路径相反：

- Transolver：把所有空间点聚合成 physics-aware slice tokens，再在 slice tokens 上做 attention，最后映射回空间点；
- SMART：先从几何点云构造固定 latent geometry，query 永远只读取 latent，不参与全局物理表示的更新。

论文实验中，Transolver 在 CFD 优化 mesh 上表现强，但由于 query points 会参与 physics-attention，其预测会依赖 query set 的整体分布；换成非 CFD 网格或局部 query distribution 后误差明显上升，而且完整千万点推理会出现显存问题。

## [[Perceiver IO]]

结构层面最值得对照的是“latent bottleneck + cross-attention”。

SMART 可以看成给 latent bottleneck 加了三种空气动力学特化：

1. latent token 由真实 geometry points 初始化；
2. encoder 逐层再次读取几何；
3. decoder 逐层读取 encoder 的 intermediate latent geometries。

## [[DiT]]

二者都体现了“条件参数不只拼接，而是调制中间计算”的思想：

- Diffusion Transformer（DiT）典型使用 adaptive Layer Normalization（adaLN）进行条件调制；
- SMART 在自己的主干里使用 FiLM 风格的 scale/shift 去调制 MLP 激活。

这两种方法具体作用位置不同，但共同逻辑都是：**让条件决定 feature transformation 本身**。

# 13. 最值得记住的五点

1. **Mesh-free 不等于训练数据没有 mesh；而是 CFD mesh 不参与模型内部表征。**
2. **Query independence 的实现条件很严格：query 不能互相 attention，也不能回写 global latent。**
3. **真实 coarse geometry 初始化 latent tokens 是决定性设计，远胜于纯 learnable tokens。**
4. **decoder 每层读取对应 encoder intermediate latent geometry 的 cross-layer coupling 是第二个决定性设计。**
5. **随机 subsampling + query-independent chunked decoding 使“16k 点训练、5000万点推理”成为可能。**
