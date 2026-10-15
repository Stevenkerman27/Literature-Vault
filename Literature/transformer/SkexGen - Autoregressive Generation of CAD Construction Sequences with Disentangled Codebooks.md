---
title: "SkexGen: Autoregressive Generation of CAD Construction Sequences with Disentangled Codebooks"
authors:
  - Xiang Xu
  - Karl D. D. Willis
  - Joseph G. Lambourne
  - Chin-Yi Cheng
  - Pradeep Kumar Jayaraman
  - Yasutaka Furukawa
year: 2022
venue: "Proceedings of the 39th International Conference on Machine Learning (ICML), PMLR 162"
doi: "10.48550/arXiv.2207.04632"
source: "/mnt/data/SkexGen.pdf"
tags:
  - CAD
  - generative-model
  - autoregressive
  - Transformer
  - VQ-VAE
  - discrete-latent
  - controllable-generation
---

> [!abstract] 摘要译文
> 本文提出 **SkexGen**，一种面向计算机辅助设计（Computer-Aided Design, CAD）构造序列的自回归生成模型，重点处理由 **草图（sketch）+ 拉伸（extrude）** 操作组成的建模过程。模型使用彼此独立的 Transformer 编码器，将构造序列中的**拓扑变化、几何变化和拉伸变化**编码到三个解耦的离散码本中；随后，自回归 Transformer 解码器在给定码本向量的条件下生成具有相应属性的 CAD 构造序列。实验表明，这种解耦码本表示能够生成多样且高质量的 CAD 模型，同时增强用户对生成结果的控制能力，并支持高效的设计空间探索。

## 1. 问题与动机

传统参数化 CAD 模型适合在既定拓扑下调整少量参数，但当设计发生较大的拓扑变化时往往很脆弱，而且建立高质量参数化模板本身就依赖专业设计经验。已有深度生成模型能够直接生成 CAD 构造序列，但“能生成”不等于“可控制”：如果用户只能随机采样完整模型，很难有针对性地保持某些设计属性、只改变另一些属性。

SkexGen 的核心目标因此不是单纯提升随机生成质量，而是把 CAD 变化拆成三个尽可能独立的因素：

- **Topology（拓扑）**：由哪些曲线、环、面等组成，以及它们的组合结构。
- **Geometry（几何）**：曲线和轮廓在二维平面中的具体位置、尺度和形状参数。
- **Extrude（拉伸）**：草图如何被放置、旋转、缩放、拉伸，并通过布尔运算组合成三维实体。

作者希望在生成时可以固定其中一部分，仅重新采样其余部分，从而实现“保持拓扑、改变几何”“保持几何、改变拓扑”“保持草图、改变三维拉伸方式”等受控生成。

## 2. CAD 构造序列表示

### 2.1 层级结构

SkexGen 使用一套显式的 sketch-and-extrude 层级表示：

```text
Sketch-and-Extrude Model
└── Extruded-Sketch × 多个
    ├── Sketch
    │   └── Face × 多个
    │       └── Loop × 多个
    │           └── Curve × 多个
    │               └── Line / Arc / Circle
    └── Extrude + Boolean operation
```

其中：

- **Curve**：最低层曲线原语，包括 line、arc、circle。
- **Loop**：一条闭合路径，可由多个曲线组成；若是 circle，则一个 circle 本身就是闭环。
- **Face**：由一个外环和若干内环（孔）构成的二维区域。
- **Sketch**：由一个或多个 face 构成。
- **Extruded-Sketch**：将 sketch 拉伸后得到的三维体。
- 多个 extruded-sketch 通过 **union / intersection / subtraction** 三种布尔操作组成最终模型。

论文特别强调：相较 DeepCAD，SkexGen 新增了 **Face** 这一层，因此一个 sketch 可以显式包含多个 face，表达能力更强。

> [!figure] 论文中图1：CAD 原语层级与序列表示
> 图1同时展示了 Curve → Loop → Face → Extruded-Sketch 的层级，以及相应 token 序列。该图最关键的意义是说明：SkexGen 不是直接生成网格或点云，而是生成可被 CAD 系统重新解析的构造序列。

### 2.2 五类 token

完整构造序列由五类 token 组成：

1. **Topology token, T**：表示曲线类型 line / arc / circle。
2. **Geometry token, G**：表示二维点坐标。
3. **End-primitive token, E**：表示 curve / loop / face / sketch / extruded-sketch 等原语结束。
4. **Extrusion token**：表示拉伸、位姿、缩放和布尔操作参数。
5. **End-sequence token, End**：表示整个序列结束。

几何坐标被离散到 $64\times 64$ 网格，即 6 bit 精度，共 $64^2=4096$ 个像素位置。

曲线点的序列编码有一个重要压缩规则：

- line 几何上由 2 个端点定义，但序列中只显式存其起点，终点由下一条曲线的起点（或闭环首点）隐式给出，因此占 **1 个 G token**；
- arc 由 3 个点定义，但同理只显式存前两个点，因此占 **2 个 G token**；
- circle 不能通过下一条曲线共享终点，因此显式存 4 个点，占 **4 个 G token**。

这也是后面 sketch decoder 可以**不显式生成 topology token**的原因：一个 curve 中 G token 的数量就足以反推出 line / arc / circle。

## 3. SkexGen 总体架构

SkexGen 分成两个独立训练的分支：

- **Sketch branch**：负责二维草图，进一步分成 topology encoder、geometry encoder 和 sketch decoder。
- **Extrude branch**：负责三维拉伸与布尔操作，由 extrude encoder 和 extrude decoder 构成。

另外还有一个独立的 **code selector**，负责从三个码本中选择彼此兼容的离散代码。

> [!figure] 论文中图2：SkexGen 总体架构
> 左侧 sketch branch 使用两个编码器分别得到 topology code 和 geometry code，再通过 cross-attention 条件化 sketch decoder；右侧 extrude branch 得到 extrude code 并条件化 extrude decoder。两个分支分别生成子序列，最后 merge 成完整 CAD 构造序列。图中未画出的另一个自回归 Transformer decoder 用于 code selection。

### 3.1 Topology Encoder

Topology encoder 只读取与拓扑有关的子序列。其 token 类别为：

- 3 种 topology token：line / arc / circle；
- 3 种较高层级的 end token：loop / face / sketch；
- 1 个 End。

因此输入 one-hot 维度为 $3+3+1=7$。

对第 $i_T$ 个 topology token，其嵌入为：

$$
\mathbf T = \mathbf W^{tp}\mathbf h_T^{tp}+\mathbf p^{(i_T)},
$$

其中 $d_E=256$，$\mathbf W^{tp}\in\mathbb R^{256\times 7}$，$\mathbf p^{(i_T)}$ 是可学习位置编码。

编码器采用标准 Transformer：

- 4 个 layer block；
- 每层 8 个 attention head；
- Layer Normalization；
- Feed-Forward Network（FFN）。

作者借鉴 Vision Transformer 的做法，在输入序列前加入可学习的 **code-token**。其输出向量 $\mathbf Z^e_{tp}$ 不直接作为连续潜变量，而是映射到 topology codebook 中最近的离散向量：

$$
\mathbf Z^Q_{tp}=\mathbf b^{(k)}_{tp},\qquad
k=\arg\min_j\left\|\mathbf Z^e_{tp}-\mathbf b^{(j)}_{tp}\right\|_2^2.
$$

实际模型不是 1 个 code-token，而是 **4 个 topology code-token**，因此每个样本最终得到 4 个 topology code。Topology codebook 大小为 **500**。

### 3.2 Geometry Encoder

Geometry encoder 的输入子序列包含：

- 4096 种离散二维坐标位置；
- 4 种 end-primitive：curve / loop / face / sketch；
- 1 个 End。

因此 token 总类别数为：

$$
4096+4+1=4101.
$$

除了普通 token embedding 外，作者还额外加入独立的 x 坐标和 y 坐标 embedding：

$$
\mathbf G=
\mathbf W^{ge}\mathbf h_G^{ge}
+\mathbf W^x\mathbf h_G^x
+\mathbf W^y\mathbf h_G^y
+\mathbf p^{(i_G)}.
$$

其中 $\mathbf h_G^x,\mathbf h_G^y\in\mathbb R^{64}$ 分别表示离散 x、y 坐标。作者指出，这两个 coordinate embedding 并非必需，但实验中能进一步提升结果。

Geometry encoder 同样使用 Transformer 和 code-token，但只使用 **2 个 code-token**，因此每个样本得到 2 个 geometry code；Geometry codebook 大小为 **1000**。

### 3.3 Sketch Decoder

Sketch decoder 接收：

- 4 个 topology code；
- 2 个 geometry code；
- 已经生成的历史 token。

它自回归生成 geometry token 和 end-primitive token，从而恢复完整 sketch 子序列。

#### 为什么不生成 topology token？

因为 line / arc / circle 分别对应 1 / 2 / 4 个 G token，曲线类型可以根据该 curve 内部的 G 数量反推出。因此 topology code 只作为高层条件，不需要再次显式输出 topology token。

#### Teacher forcing 与 shift-right

训练时 decoder 输入序列向右平移一位，在最前面加入 `STR`（start）token；第 $k$ 个位置根据前 $k-1$ 个 ground-truth token 预测原序列第 $k$ 个 token。这就是标准 teacher forcing。

对 decoder 输出 embedding $\mathbf K\in\mathbb R^{256}$，使用线性层映射到 4101 类并做 softmax：

$$
\mathbf h_K^{out}=\operatorname{softmax}(\mathbf W^{out}\mathbf K),
$$

其中 $\mathbf W^{out}\in\mathbb R^{4101\times 256}$。

#### Code 通过 cross-attention 注入

6 个离散 code 并非简单拼接到 token embedding，而是作为 cross-attention 的条件输入。为了区分 topology code 与 geometry code，作者给两组 code 加入不同的可学习 code-position embedding：

$$
\mathbf Z^{Q(i)}_{tp}+\mathbf p^{(q_{tp})},
\qquad
\mathbf Z^{Q(i)}_{ge}+\mathbf p^{(q_{ge})}.
$$

Sketch decoder 同样使用 4 层、8 头 Transformer，但采用 causal mask，只允许当前位置关注先前 token。

## 4. Extrude Branch

Extrude branch 与 sketch branch 结构相似，但只使用一个 extrude codebook。

每次 extruded-sketch 对应的参数包括：

- **H**：参考平面上下两侧的拉伸距离，共 2 个值；每个值量化为 64 bins。
- **R**：拉伸方向对应的 $3\times 3$ 旋转矩阵，共 9 个值。作者观察 DeepCAD 数据中矩阵元素几乎都属于 $\{-1,0,1\}$，覆盖率超过 99%，因此每个元素按 3 类离散表示。
- **O**：三维平移向量，共 3 个值，每个量化为 64 bins。
- **S**：草图缩放，包括二维缩放中心 2 个值和 uniform scale 1 个值，共 3 个值，每个量化为 64 bins。
- **B**：布尔操作，三类：intersection / union / subtraction。
- **E**：extruded-sketch 结束。

展平后一次 extrusion 操作包含：

```text
H H
R R R R R R R R R
O O O
S S S
B
E
```

共 $2+9+3+3+1+1=19$ 个 token，不计 End。

Extrude token 使用 **72 维 one-hot**：

$$
64\text{(numeric bins)}+3\text{(rotation)}+3\text{(Boolean)}+1\text{(E)}+1\text{(End)}=72.
$$

Extrude encoder：

- 4 个 code-token；
- codebook size = 1000。

Extrude decoder 与 sketch decoder 类似，只是：

- 只接收一类 extrude codebook；
- 不需要再用额外 embedding 区分不同 codebook；
- 输出维度为 72，而不是 4101。

因此整个 CAD 模型最终由 **10 个离散 code** 描述其高层变化因素：

$$
4\text{ topology}+2\text{ geometry}+4\text{ extrude}=10.
$$

## 5. 离散码本训练目标

Sketch branch 联合训练 topology encoder、geometry encoder 与 sketch decoder，主要包含三类损失：

### 5.1 序列重建损失

$$
\mathcal L_{rec}=\sum_K
\operatorname{CrossEntropy}(\mathbf h_K^{out},\mathbf h_K^{gt}).
$$

### 5.2 Codebook loss

以 topology code 为例：

$$
\left\|\operatorname{sg}(\mathbf Z^e_{tp})-\mathbf b_{tp}\right\|_2^2.
$$

`sg` 是 stop-gradient：前向计算保持恒等，反向传播时切断梯度。该项让被选中的 codebook vector 向 encoder 输出靠近。

### 5.3 Commitment loss

$$
\beta\left\|\mathbf Z^e_{tp}-\operatorname{sg}(\mathbf b_{tp})\right\|_2^2,
$$

geometry code 同样有对应项。论文设 $\beta=0.25$。这一项限制 encoder 输出不要在连续空间任意漂移，而要“commit”到某个离散 code 附近。

论文还说明使用指数移动平均（Exponential Moving Average, EMA）更新 codebook，decay rate 为 **0.99**。

Extrude branch 使用同样的 reconstruction + codebook + commitment 训练方式。

## 6. Code Selector：为什么需要第二阶段生成？

三个 encoder 得到的离散 code 不服从 VAE 常见的标准正态先验，因此随机从三个码本独立抽 code 很可能得到**彼此不兼容的组合**。

为此，作者先对所有训练样本编码，得到真实出现过的 topology / geometry / extrude code 组合，再训练一个额外的自回归 Transformer decoder 作为 **code selector**，学习 code index 的联合分布。

以“给定 topology + geometry，预测 extrude code”为例：

1. topology 和 geometry 的 code index 先用 1000 维 one-hot 表示；
2. 通过 $256\times 1000$ 的可学习矩阵投影到 256 维；
3. 已知 code 通过 cross-attention 注入 decoder；
4. decoder 依次自回归预测 4 个 extrude code index；
5. 每一步通过 $1000\times 256$ 线性层 + softmax 得到 1000 个 codebook index 的概率。

这一机制也允许建立不同条件版本，例如：

- 给定 topology，采样 geometry + extrude；
- 给定 geometry，采样 topology + extrude；
- 给定 topology + geometry，只采样 extrude。

这正是 SkexGen 实现可控生成的关键。

## 7. 完整生成流程

SkexGen 的生成分两层：

### 第一层：生成离散 code

Code selector 从三个码本中选择一组彼此兼容的 topology / geometry / extrude codes。

### 第二层：生成 CAD token 序列

- sketch decoder 根据 topology + geometry codes 自回归生成 sketch 子序列；
- extrude decoder 根据 extrude codes 自回归生成 extrude 子序列；
- 两个子序列 merge 成完整 sketch-and-extrude sequence；
- CAD 软件将该序列解析为 Boundary Representation（B-Rep）实体。

测试时两个 decoder 使用 **nucleus sampling（top-p sampling）**：$p=0.95$，temperature = 1.0。

## 8. 实验设置

### 8.1 数据集

使用 DeepCAD 数据集，原始包含 **178,238** 条构造序列，划分为：

- 90% train；
- 5% validation；
- 5% test。

作者分别去除重复 sketch 子序列和 extrude 子序列，并移除无效的 sketch-and-extrude 操作。最终训练集为：

- **74,584** 个 sketch subsequence；
- **86,417** 个 extrude subsequence。

单独进行 sketch generation 实验时，从所有 CAD 构造步骤中抽取 sketch，去重后得到 **114,985** 个训练样本。

### 8.2 网络与训练超参数

| 项目 | 设置 |
|---|---:|
| Transformer blocks | 4 |
| attention heads / block | 8 |
| embedding dim | 256 |
| FFN dim | 512 |
| normalization | pre-LayerNorm |
| dropout | 0.1 |
| optimizer | Adam |
| learning rate | 0.001 |
| batch size | 128 |
| epochs | 500 |
| sketch max length | 200 |
| extrude max length | 100 |
| GPU | RTX A5000 |
| sampling | nucleus sampling, top-p=0.95, T=1.0 |

额外训练技巧：

- linear warm-up；
- gradient clipping；
- 前 **25 epochs 暂停 code quantization**，作者发现这样能稳定 codebook 训练；
- 对 geometry token 坐标加入小幅随机噪声作数据增强。

## 9. 实验指标

### 9.1 2D sketch

- **Fréchet Inception Distance（FID）**：比较真实与生成样本特征分布的均值和协方差，越低越好。这里使用在手绘草图分类任务上预训练的 ResNet-18 提取特征。
- **Unique**：生成样本中仅出现一次的比例，越高越好。
- **Novel**：生成样本中未出现在训练集里的比例，越高通常表示新颖性越强，但无效样本也可能虚高该指标。

### 9.2 3D CAD

基于表面均匀采样 2000 个点并计算 Chamfer distance：

- **Coverage（COV）**：真实样本中能被生成样本覆盖的比例，越高越好。
- **Minimum Matching Distance（MMD）**：生成样本到最近真实样本的平均距离，越低越好。
- **Jensen-Shannon Divergence（JSD）**：真实与生成点分布差异，越低越好。
- 同样统计 Unique 与 Novel。

两个序列若经过 6-bit quantization 后所有 token 完全一致，则视为相同样本。

## 10. 关键实验结果

### 10.1 Sketch 随机生成

| 方法 | FID ↓ | Unique ↑ | Novel ↑ |
|---|---:|---:|---:|
| CurveGen | 109.12 | 93.05% | 83.82% |
| DeepCAD | 75.47 | 98.79% | 97.45% |
| **SkexGen** | **18.56** | 96.02% | 83.54% |

SkexGen 的 FID 显著最低。论文指出 DeepCAD 的 Novel 很高并不意味着真正更好：它会生成较多自交、非 watertight 等无效形状，而这些错误样本因为训练集中不存在，也会被计入“novel”。

作者的视觉观察是 SkexGen 生成的 sketch：

- 更复杂；
- 自交更少；
- 对称性更强；
- 比 CurveGen 更常出现复杂的矩形和圆组合。

### 10.2 3D CAD 随机生成与消融

| 方法 | COV ↑ | MMD ↓ | JSD ↓ | Unique ↑ | Novel ↑ |
|---|---:|---:|---:|---:|---:|
| DeepCAD | 76.8 | 1.68 | 2.01 | 91.0% | 87.0% |
| SkexGen - disentanglement - quantization（近似 VAE） | 74.3 | 1.54 | 0.92 | 97.8% | 91.9% |
| SkexGen - disentanglement（单一码本，近似 VQ-VAE） | 80.4 | 1.55 | 1.12 | 99.8% | 99.3% |
| **SkexGen** | **83.6** | **1.48** | **0.81** | **99.9%** | **99.8%** |

结论非常明确：

1. 仅使用连续潜变量、去掉量化与解耦，性能最差；
2. 加入离散量化、但只保留一个统一 codebook，已经明显改善；
3. 再把 codebook 分成 topology / geometry / extrude 三类，整体指标进一步提升。

因此论文的核心贡献不只是“用了 VQ-VAE”，而是**将离散潜变量按 CAD 语义因素拆分为多个 codebook**。

### 10.3 运行速度

生成 10,000 个样本：

- DeepCAD：约 15 s；
- SkexGen：约 90 s。

SkexGen 因自回归采样明显慢于 DeepCAD，但论文指出它仍比拥有两个相互依赖自回归 decoder 的 CurveGen 快约 2 倍。

## 11. 可控生成与“解耦”验证

> [!figure] 论文中图5：固定 topology code 或 geometry code 的条件生成
> 左半部分每一行固定 topology code，只改变 geometry / extrude；因此拓扑结构保持相似，而尺寸、位置等几何属性改变。右半部分固定 geometry code，整体尺度与布局相似，但外轮廓可能从 line 变为 circle 或 arc。

作者按照 $\beta$-VAE 中的思路定量评估解耦：

1. 生成一对样本；
2. 固定 topology / geometry / extrude 中的一类 code，另外两类随机采样；
3. 将两者编码后的 latent 表示做平均 pairwise difference；
4. 训练一个小型 Transformer classifier 判断“究竟固定的是哪一类 code”。

SkexGen 的分类准确率达到：

$$
99.8\pm 0.1\%.
$$

这表明三类 code 的变化模式高度可区分。

## 12. 设计探索应用

### 12.1 Latent interpolation

对两个已有 CAD 模型：

1. encoder 分别提取 10 个未量化的连续 code $\{\mathbf Z^e\}$；
2. 在连续 code 空间中线性插值；
3. 再量化到最近 codebook vector；
4. 用 decoder 生成中间 CAD 模型。

为了让插值路径更一致，作者在 decoder 生成阶段不使用 nucleus sampling，而是每一步取最高概率 token。

> [!figure] 论文中图7：设计插值
> 插值过程中既可能出现连续几何变化，也可能突然发生离散拓扑变化。例如轮廓从多边形逐渐变成包含多个圆的结构。作者明确指出，这种插值通常并不“平滑”，因为拓扑变化本质上是离散的。

### 12.2 Code mixing

可以直接从两个模型中复制不同类别的 code：

```text
Topology code ← 模型 A
Geometry code ← 模型 B
Extrude code ← A 或 B / 重新采样
```

然后让 decoder 生成混合模型。

> [!figure] 论文中图8：Topology / Geometry / Extrude code mixing
> 图8展示了“拓扑来自 A、几何来自 B”的混合结果。例如参考拓扑中有多个圆柱，而参考几何提供方形排布，生成结果就会出现“多个圆柱按方形阵列排列”的组合。

这比普通 latent interpolation 更直接地体现了“设计因素可组合”的价值。

## 13. 关键概念补充（联网核对）

### 13.1 Vector-Quantised Variational AutoEncoder（VQ-VAE）

VQ-VAE 的核心不是让 encoder 输出连续高斯潜变量，而是让 encoder 输出先与一个有限 codebook 做最近邻匹配，再把最近的 codebook vector 送入 decoder。原始 VQ-VAE 工作指出，离散潜变量可以避免强大 autoregressive decoder 忽略 latent 的 posterior collapse 问题。

SkexGen 继承了 VQ-VAE 的三个关键机制：

- 最近邻量化；
- codebook / commitment 约束；
- stop-gradient 处理不可微量化路径。

但它做了一个面向 CAD 的关键改造：**不是一个统一码本，而是把 topology、geometry、extrude 分开量化**。

### 13.2 Nucleus sampling / top-p sampling

Nucleus sampling 不固定保留 top-k 个 token，而是在每一步寻找一个最小候选集合，使其累积概率达到阈值 $p$，然后只在这部分动态“概率核心”里采样。这样既保留随机性，又丢掉长尾中非常低概率、通常更不可靠的 token。

SkexGen 测试时取 $p=0.95$，即每一步从累计概率达到 95% 的动态候选集合中采样。

### 13.3 “Disentanglement” 在本文中的实际含义

这里的 disentanglement 不是严格证明三组潜变量在统计上完全独立，而是人为规定三种输入子序列与三套独立 encoder / codebook，使不同类别 code **主要负责不同 CAD 因素**，再通过条件生成和分类实验验证这种分工是否形成。

因此它更准确地说是一种**结构化、语义分组的离散潜变量设计**。

## 14. 局限与启发

### 14.1 论文明确或由其设置直接体现的局限

1. **自回归生成较慢**：10,000 个 CAD 样本约 90 s，明显慢于 DeepCAD。
2. **插值并不平滑**：因为 code quantization 和拓扑变化均是离散的，latent interpolation 会出现跳变。
3. **建模语言范围有限**：模型只覆盖 sketch-and-extrude + Boolean operation，并不等于完整工业 CAD 命令集。
4. **几何精度受离散化限制**：二维坐标只使用 6-bit、$64\times64$ 网格。
5. **Extrude 表示依赖 DeepCAD 数据分布**：旋转矩阵元素被简化为 $\{-1,0,1\}$，作者说明该近似在当前数据中覆盖超过 99%，但这不是任意 CAD 位姿的一般表示。
6. **解耦指标验证的是“类别可区分”**：99.8% 分类准确率说明三类 code 具有高度不同的变化模式，但不能单独证明它们在所有设计语义上完全独立。

### 14.2 方法上的启发

- **把全局设计因素先离散化，再做细粒度自回归生成**：code selector 决定“生成什么类型的设计”，token decoder 决定“具体如何写出 CAD 序列”。
- **条件控制不一定要直接注入连续性能向量**：也可以先学习结构化 latent code，再对 code 的联合分布建模。
- **多码本比单一码本更适合具有明确语义因子的工程对象**：当 topology、geometry、operation 本身具有不同物理/几何含义时，强制分支化可以增强可解释控制。
- **cross-attention 是很自然的条件注入方式**：历史 token 作为 autoregressive 主序列，而全局 code 作为另一组 memory，由 decoder 按需查询。

## 15. 关联工作与作者脉络

### 15.1 TurtleGen / Engineering Sketch Generation（Willis et al., 2021）

TurtleGen 使用 Transformer 自回归生成工程草图曲线序列，是 SkexGen 的直接表示基础之一。SkexGen 延续“把 CAD 看成语言序列”的思路，但进一步扩展到 face 层级、三维 extrude，并加入离散潜变量实现可控生成。

### 15.2 DeepCAD（Wu et al., 2021）

DeepCAD 提供大规模 sketch-and-extrude 数据集，并直接生成 CAD 构造命令序列，是本文最重要的基线。SkexGen 既使用 DeepCAD 数据，也修改其表示：新增 face primitive，从而允许一个 sketch 包含多个 face；同时把直接序列生成改成“解耦离散 code + 自回归 decoder”的两阶段生成。

### 15.3 VQ-VAE（van den Oord et al., 2017）

VQ-VAE 提供了离散 codebook、最近邻量化、commitment loss 与 stop-gradient 的技术基础。SkexGen 的创新点不是重新提出向量量化，而是把它用于高度结构化的 CAD 构造序列，并将潜空间拆成 topology / geometry / extrude 三类语义码本。

## 16. 关系总结

可以把 SkexGen 看成以下三条研究线的交汇：

```text
CAD as Language / Transformer sequence generation
                │
                ├── TurtleGen / DeepCAD
                │
                ▼
      Sketch-and-Extrude token sequence
                │
                │ + VQ-VAE discrete latent bottleneck
                ▼
  Topology / Geometry / Extrude disentangled codebooks
                │
                │ + autoregressive code selector
                ▼
 controllable CAD generation / interpolation / code mixing
```

最值得记住的一点是：**SkexGen 并非仅用 Transformer 直接预测 CAD token，而是先把设计压缩成 10 个具有语义分工的离散 code，再让不同 decoder 在这些 code 的条件下生成完整构造序列。** 这种分层设计同时服务于生成质量、组合合法性和用户控制。

## 17. 外部概念核对来源

- van den Oord, Vinyals, Kavukcuoglu, *Neural Discrete Representation Learning*, NeurIPS 2017：VQ-VAE、离散潜变量、向量量化与 commitment loss。
- Holtzman et al., *The Curious Case of Neural Text Degeneration*, ICLR 2020：nucleus sampling / top-p sampling。
- Higgins et al., *beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*, ICLR 2017：disentanglement 的定量评估思路。

## 跨文献关联导航（2026-10-15 整理）

> [!note] 文献关系标注
> “方法承接”或“研究延展”表示现有笔记可支持的脉络；“比较”“互补”“类比”仅用于研究组织，不声称作者直接引用或实验证实联合方案。

**总览**：[[Literature/文献关系总览|文献关系总览]]

**所属主题**：[[Literature/Topics/几何表示与性能条件生成|几何表示与性能条件生成]] · [[Literature/Topics/整机设计与工程智能体|整机设计与工程智能体]]

**重点关联**：

- **表示比较**：[[Literature/PcDGAN - A Continuous Conditional Diverse Generative Adversarial Network For Inverse Design|PcDGAN - A Continuous Conditional Diverse Generative Adversarial Network For Inverse Design]] — 连续性能条件生成与离散 CAD 拓扑/几何码本分层。
- **表示比较**：[[Literature/SeqXY2SeqZ-Structure Learning for 3D Shapes by Sequentially Predicting 1D Occupancy Segments from 2D Coordinates|SeqXY2SeqZ-Structure Learning for 3D Shapes by Sequentially Predicting 1D Occupancy Segments from 2D Coordinates]] — 两者都用序列表达工程几何，但 token 语义不同。
- **几何语义对照**：[[Literature/Deformation-driven shape correspondence via shape recognition|Deformation-driven shape correspondence via shape recognition]] — 构造序列解耦与跨形状部件对应探索不同几何结构层级。
- **表示原则**：[[Literature/transformer/How Far Should Tokenization Go - Predictive Effectiveness and Relational Losslessness|How Far Should Tokenization Go - Predictive Effectiveness and Relational Losslessness]] — token 接口是否预先固定关系会影响上下文建模。
- **离散连续构型对照**：[[Literature/A comparison of methodologies for generating design-requirement-driven conceptual aircraft designs|A comparison of methodologies for generating design-requirement-driven conceptual aircraft designs]] — 整机配置变量与 CAD 拓扑/几何表达都需处理混合变量。
