---
title: Towards a Foundation-Model Paradigm for Aerodynamic Prediction in Three-dimensional Design
authors:
  - Yunjia Yang
  - Babak Gholami
  - Caglar Gurbuz
  - Mohammad Rashed
  - Nils Thuerey
year: 2026
venue: arXiv preprint (arXiv:2604.18062v1)
doi:
source: file:///D:/3D/Projects/Papers/storage/LJIYHCX6/Yang%20%E7%AD%89%20-%202026%20-%20Towards%20a%20Foundation-Model%20Paradigm%20for%20Aerodynamic%20Prediction%20in%20Three-dimensional%20Design.pdf
tags:
  - aerodynamics
  - surrogate-model
  - transformer
  - foundation-model
  - transfer-learning
  - wing-design
  - CFD
---

> [!abstract] 摘要
> 精确的机器学习气动预测模型对于加速外形优化很重要，但三维复杂构型的高质量训练数据代价极高。本文提出一种面向设计代理模型的“两阶段基础模型范式”：先在几何多样、但几何细节适度简化的大规模数据上预训练，再用少量高细节、任务相关样本进行微调。为此，论文开发了适合大规模训练的 Transformer 气动代理模型 **AeroTransformer**。在跨声速机翼问题上，模型先用近 3 万个流场样本的 SuperWing 数据集预训练，再迁移到围绕 Common Research Model（CRM）机翼构造的高自由度局部设计域。使用约 450 个任务样本时，微调后的模型表面流场误差达到约 **0.36%**，相对从头训练降低约 **84.2%**。论文还系统研究了模型规模、预训练数据量、批量大小、辅助气动力损失、下游数据量、训练步数以及参数高效微调策略，并据此给出有限数据与有限计算预算下的训练建议。

# 一句话结论

这篇论文的核心不是“用 Transformer 替代 CFD”，而是把三维气动代理模型的构建方式改写为：**用一次昂贵但可复用的广域预训练学习通用几何—流动物理映射，再用少量局部高保真数据恢复具体设计任务所需的细节精度。**

# 1. 问题与动机

传统气动代理模型通常围绕某个基准构型，在很局部的设计空间内生成训练数据并重新训练一个模型。这种策略有两个问题：

1. **高保真数据生成成本可能抵消代理模型带来的优化收益。** 三维跨声速 RANS 计算本身昂贵，多点优化、鲁棒优化还需要大量重复气动评估。
2. **“广覆盖”和“高精度”存在矛盾。** 如果想让一个模型覆盖很大的三维设计空间，就需要极大的几何多样性；但真实工业几何又具有大量局部自由度，直接在这个高维空间中做高保真采样几乎不可承受。

论文因此提出一个很明确的设计原则：

- **预训练阶段优先几何多样性，允许适度牺牲几何细节；**
- **微调阶段优先局部几何细节，只覆盖具体设计任务附近的小设计域。**

也就是说，预训练负责学习“大的气动规律”，微调负责恢复“任务局部的细节”。

![[assets/Towards a Foundation-Model Paradigm for Aerodynamic Prediction in Three-dimensional Design/Figure 1.png]]

论文中图1给出了完整范式：广域设计空间通过低维参数化采样形成 foundation dataset；针对具体基准构型，在局部区域通过更细致的形变或参数化生成 task dataset；预训练模型再被微调为多个任务专用代理模型。

# 2. AeroTransformer：模型如何实现

## 2.1 总体输入输出

论文研究的是“二维翼面嵌入三维空间”的情况，机翼表面可以统一映射到单块结构化网格。

输入包含：

- 几何表面网格
  $$
  g\in\mathbb{R}^{H\times W\times 3}
  $$
  实验中为 $256\times128\times3$ 的单元中心坐标；
- 工况向量
  $$
  c=[Ma,\alpha]
  $$
  其中 $Ma$ 为马赫数，$\alpha$ 为攻角。

论文给出两种输出模型：

- **ATsurf**：预测表面流场，输出 $C_p$ 与壁面摩擦系数 $C_f$；由于 $C_f$ 是切向矢量，分为流向和展向两个分量，因此实际输出是 3 个通道：$C_p,C_{f,\tau},C_{f,z}$。
- **ATcoef**：直接预测整体气动系数，如 $C_L,C_D,C_{M,z}$。

作者最终认为：**以表面流场为主输出，再通过积分得到气动系数，比直接回归气动系数更可靠。**

## 2.2 Backbone：PDE-Transformer 的层级 U 形结构

![[assets/Towards a Foundation-Model Paradigm for Aerodynamic Prediction in Three-dimensional Design/Figure 3.png]]

论文中图3是 AeroTransformer 的核心结构图。蓝色部分来自 PDE-Transformer，红色部分是本文针对气动任务的改造。

### Mesh patching

输入不是图像像素，而是结构化翼面网格坐标。论文使用卷积 patch embedding：

- patch size：$p_H=p_W=4$；
- 原始 $256\times128$ 网格被分为 $64\times32=2048$ 个初始 token；
- 每个 patch 通过共享卷积核映射到隐藏维度。

### 层级下采样—潜空间—上采样

ATsurf 使用 U 形层级架构：

- 两个 down-sampling stage；
- 一个 latent stage；
- 两个 up-sampling stage；
- 各 stage 中 Transformer block 数量为 **2 / 5 / 8 / 5 / 2**，合计 **22 个 block**；
- 下采样时空间分辨率减半、通道维数增大；上采样反向恢复分辨率；
- 对称层之间使用 skip connection，把高分辨率局部特征重新注入解码路径。

这种设计的目的不是单纯模仿 U-Net，而是解决全分辨率 self-attention 的 token 数量过大问题：早期保留细节，深层逐步聚合全局信息，然后再恢复表面流场。

### Shifted-window attention

模型采用 Swin Transformer 风格的：

- **W-MSA：Window-based Multi-head Self-Attention，窗口多头自注意力**；
- **SW-MSA：Shifted-Window Multi-head Self-Attention，移位窗口多头自注意力**。

窗口大小 $w=8$。注意力只在局部窗口内部计算，因此显著降低计算与显存开销；相邻 block 将窗口平移 $w/2$，使原本处于不同窗口的 token 发生信息交换。

论文还使用基于相对位置的偏置，而不是简单的绝对位置编码。

## 2.3 工况注入：不是只把 Ma 和 AoA 拼到输入

一个非常重要的改动是 **operating condition injection**。

最直接的做法是把 $Ma$ 和 $\alpha$ 扩展到每个网格点，再与几何坐标拼接。但作者认为，条件信息在深层网络中可能被逐渐“遗忘”。因此本文借鉴 adaptive Layer Normalization Zero（adaLN-Zero，自适应层归一化零初始化）式条件调制：

1. $[Ma,\alpha]$ 先经过一个小型多层感知机（MLP）得到工况 embedding；
2. 对每一个 Transformer block，都从该 embedding 回归出独立的 scale / shift；
3. Layer Normalization 后的 token 在进入 attention 或 feed-forward network（FFN）前被缩放和平移；
4. attention 和 FFN 分支的输出也再乘条件相关的缩放量。

因此工况不是一个“一次性输入”，而是**逐层持续调制**网络内部特征。

实验中，这个设计在相同约 1M 参数量下，相对普通 AeroTransformer 又带来了约 4% 的误差改善。

> [!note] 关于 adaLN-Zero 的边界
> 本文明确说其条件注入“受 adaLN-Zero 启发”，但没有明确说明 AeroTransformer 的调制层是否完全采用原始 DiT 中的 zero initialization。因此不能仅凭名称假设本文复现了原始 adaLN-Zero 的全部初始化细节。

## 2.4 ATsurf 和 ATcoef 的差别

### ATsurf

完整使用 U 形 backbone。最后的卷积层把每个 token 展开成 $N_{var}\times p_H\times p_W$，再 reshape 回原始翼面网格上的 3 通道流场。

### ATcoef

只保留下采样部分和 latent stage，不再进行上采样。latent token 再通过 **attention pooling** 压缩为单一向量，最后用全连接层输出 $C_L,C_D,C_{M,z}$。

从结构上看，ATcoef 更像一个几何—工况到标量性能的 encoder-only 回归器。

# 3. 为什么论文最终选择“先预测流场，再积分气动力”

从预测表面流场得到整体气动力时，论文利用表面压力和摩擦进行积分。形式上可写为：

$$
[C_L,C_D]=R_\alpha\,C_F
$$

其中总表面力由各网格单元的压力、摩擦、法向量与面积累积得到。这样，局部流场和整体气动力之间存在明确的几何/物理积分关系。

## 3.1 直接回归气动系数容易过拟合

在预训练数据上的对比非常明显：

| 方法 | 测试 $\delta C_L$ | 测试 $\delta C_D$ | 测试 $\delta C_{M,z}$ |
|---|---:|---:|---:|
| Random Forest | 43.86×10⁻³ | 36.49×10⁻⁴ | 58.96×10⁻³ |
| LGBM | 19.65×10⁻³ | 16.77×10⁻⁴ | 25.46×10⁻³ |
| ATcoef | 7.19×10⁻³ | 5.98×10⁻⁴ | 8.45×10⁻³ |
| **ATsurf + 积分** | **1.31×10⁻³** | **1.19×10⁻⁴** | **1.55×10⁻³** |

ATcoef 在训练集上已经和 ATsurf 接近，但测试误差明显放大，说明直接预测少数几个标量更容易学习到数据集局部相关性。反之，要求模型先解释完整表面流场，相当于施加了一个高维物理结构约束，作者把它视为一种 regularization（正则化）。

## 3.2 点对点流场误差最小，不等于积分气动力误差最小

基础表面流场损失为逐单元均方误差：

$$
\mathcal L_{surf}=MSE(\hat C_p,\hat C_f;C_p,C_f)
$$

但 $C_L$、$C_D$ 是所有单元误差经过**有符号、带几何权重**的累积。

以升力为例，压力误差的贡献近似为：

$$
\Delta C_L\propto \sum_i (\hat C_{p,i}-C_{p,i})
(R_\alpha n_i)A_i
$$

因此会出现两种现象：

- 根部单元面积/弦长更大，同样的 $C_p$ 误差对整体升力影响更大；
- 对阻力而言，正负局部误差可能相互抵消，所以平均绝对点误差更小的模型，整体 $C_D$ 反而可能更差。

附录 D 专门用局部敏感度 $\partial C_L/\partial C_{p,i}$ 和 $\partial C_D/\partial C_{p,i}$ 展示了这一点。

## 3.3 辅助气动力损失

因此作者在 ATsurf 上进一步增加：

$$
\mathcal L_{coef}=MSE(\hat C_L,C_L)+MSE(\hat C_D,C_D)
$$

最终：

$$
\mathcal L=\mathcal L_{surf}+\lambda\mathcal L_{coef}
$$

测试结果：

- $\lambda=0.1$ 时整体最均衡；
- 表面误差从 $(0.157,0.120,0.147)\%$ 改善到 $(0.154,0.119,0.146)\%$；
- $C_L$ 误差从 1.31×10⁻³ 降到 1.11×10⁻³；
- $C_{M,z}$ 从 1.55×10⁻³ 降到 1.33×10⁻³；
- $\lambda=0.5$ 的 $C_L$ 略好，但表面流场略差。

很有意思的是：辅助气动力损失不仅提升整体系数，还可能改善流场。这说明全局积分监督能够抑制空间上不一致的误差分布。

论文还尝试了 **ConFIG（Conflict-Free Training）** 来自动组合多个损失梯度，但效果明显差于手动选择 $\lambda$。作者推测原因是这里两个梯度本身大多同向，ConFIG 的“化解梯度冲突”优势并没有真正发挥出来。

# 4. 机翼几何表示与网格

![[assets/Towards a Foundation-Model Paradigm for Aerodynamic Prediction in Three-dimensional Design/Figure 4.png]]

论文中图4给出了跨声速机翼的参数化。全局平面形参数包括：

- 前缘后掠角 $\Lambda_{LE}$；
- 展弦比 $AR$；
- 梢根比 $TR=c_{tip}/c_{root}$；
- kink 位置 $\eta_k$；
- 根部调整比例 $\kappa$。

此外还有沿展向分布的：

- 剖面翼型；
- 上反角/局部前缘高度 $y_{LE}$；
- 扭转角 $\alpha_{tw}$。

![[assets/Towards a Foundation-Model Paradigm for Aerodynamic Prediction in Three-dimensional Design/Figure 5.png]]

论文中图5展示了统一网格表示。所有样本被投影到相同的结构化参考表面网格：

- 展向方向使用 129 个截面，因此单元中心为 128；
- 每个截面沿翼型周向离散，上下表面采用固定归一化弦向位置；
- 最终模型使用 $256\times128$ 个 cell-centered 网格点；
- 翼尖约 0.3% 展长被移除以保证拓扑一致，积分气动力时再插值恢复，对整体系数的影响约 0.1%。

这一步非常关键：它把三维机翼表面变成类似“规则二维数组”的数据，使 PDE-Transformer/Swin 风格的 2D 层级操作成为可能。

# 5. 两阶段数据集设计

![[assets/Towards a Foundation-Model Paradigm for Aerodynamic Prediction in Three-dimensional Design/Figure 6.png]]

论文中图6很好地体现了两个数据集并不是“同一个参数空间里一个大、一个小”，而是**参数化保真度本身不同**。

## 5.1 预训练：SuperWing

| 项目 | 设置 |
|---|---|
| 不同机翼几何 | 4239 |
| 有效流场样本 | 28856 |
| 几何自由度 | 37 |
| 每个机翼工况 | 随机采样 8 个 |
| 马赫数 | $Ma\in[0.75,0.90]$ |
| 攻角 | $\alpha\in[2^\circ,12^\circ]$ |
| Reynolds 数 | 20 million，固定 |
| 自由来流温度 | 300 K，固定 |

全局参数做广范围均匀采样：

- $\Lambda_{LE}\in[25^\circ,40^\circ]$
- $AR\in[8,11]$
- $TR\in[0.15,0.40]$
- $\eta_k\in[36\%,42\%]$
- $\kappa\in[10\%,110\%]$

剖面细节则被刻意压缩：

- 使用**一个基准翼型**，以 20 个 Class-Shape Transformation（CST，类-形状变换）系数描述；
- 主要通过展向最大厚度和弯度变化形成不同截面；
- 厚度、弯度、扭转等展向分布使用少量 B-spline 控制点表示。

所以 SuperWing 的目的不是穷举真实机翼细节，而是用可承受的数据量覆盖尽可能广的主导几何变化。

## 5.2 微调：CRM 局部高细节数据

| 项目 | 设置 |
|---|---|
| 不同机翼几何 | 288 |
| 有效流场样本 | 2145 |
| 几何自由度 | 153 |
| 全局平面形 | 固定为 CRM |
| 控制截面 | 7 个展向位置 |
| 每截面翼型 | 20 个 CST 系数，围绕 CRM 对应系数 ±40% 扰动 |
| 上反/高度扰动 | ±0.05 $c_{root}$ |
| 扭转角 | $[-3^\circ,0^\circ]$ |
| 马赫数 | 与 SuperWing 相同 |
| 攻角 | $[-2^\circ,4^\circ]$，CRM 安装角为 6.71° |

因此微调集反而具有更高自由度：**153 vs 37**。它只是不再覆盖广泛的平面形，而是在 CRM 附近把局部翼型细节放开。

附录 A 的几何降维也支持这一点：

| 数据集 | PCA 99% 能量所需模态 | PCA 99.9% 能量所需模态 |
|---|---:|---:|
| SuperWing | 5 | 11 |
| CRM 微调集 | 11 | 42 |

这说明 CRM 局部设计域虽然“范围小”，但局部几何结构更复杂。

作者还用 t-SNE 做了互相投影：从 SuperWing 的全局视角看，CRM 集中在一个局部区域；但从 CRM 的局部细节视角看，SuperWing 只覆盖其中一部分细致变化。因此这个迁移并不是简单的同分布插值。

# 6. CFD 与训练实验设置

## 6.1 CFD

- RANS 求解器：ADflow；
- 体网格：pyHyp；
- 多重网格策略：“3w”；
- 每个算例最多 4000 iteration；
- 残差低于 $10^{-10}$ 判定收敛；
- 数据生成硬件：160 个 Intel Xeon Gold 5320 CPU 核心。

## 6.2 流场误差

每个流场通道先计算平均绝对误差，再用该样本对应真实场的 max-min 范围归一化。论文将三通道平均定义为 surface flow error（SFE）：

$$
SFE=\frac{1}{3}(\delta C_p+\delta C_{f,\tau}+\delta C_{f,z})
$$

气动系数则直接报告 $C_L,C_D,C_{M,z}$ 的平均绝对误差。

## 6.3 预训练优化设置

- optimizer：AdamW；
- one-cycle learning-rate schedule；
- 最大学习率 $LR_{max}=10^{-3}$；
- 前半程从 $0.04LR_{max}$ 上升到 $LR_{max}$，后半程降到 $0.001LR_{max}$；
- gradient clipping：梯度范数超过阈值时缩放；
- S-size 默认 clipping norm = 1.0；
- L-size 为稳定训练改为 0.5；
- 模型规模实验统一训练 585.6k optimization steps；
- 主要 scaling 实验 batch size = 64。

论文测试了 exponential moving average（EMA，指数移动平均）式 gradient clipping，但效果反而更差。

## 6.4 模型规模

初始 hidden dimension：

- S：16；
- M：32；
- L：64；
- attention heads：8；
- FFN/MLP ratio：4。

> [!warning] 论文内部的参数量数字存在不一致
> 第 3.2.3 节文字称 S/M/L 约对应 1.0M / 3.8M / **10.5M** 参数；但后续 Fig. 9、Table 5 和实验讨论使用 L-size ≈ **14.5M / 14.52M**。本文笔记保留这个差异，不自行替作者校正。

# 7. 预训练阶段结果

## 7.1 架构对比

在约 1M 参数、batch size 4、90%/10% train-test split、三次交叉验证下：

| 模型 | $\delta C_p$ | $\delta C_{f,\tau}$ | $\delta C_{f,z}$ | 峰值显存 |
|---|---:|---:|---:|---:|
| U-Net | 0.911% | 0.548% | 0.635% | 1.03 GB |
| ViT | 0.353% | 0.267% | 0.313% | 4.61 GB |
| Transolver | 0.401% | 0.309% | 0.357% | 10.71 GB |
| AeroTransformer | 0.279% | 0.209% | 0.253% | 0.62 GB |
| **AeroTransformer + 工况注入** | **0.264%** | **0.198%** | **0.238%** | **0.64 GB** |

AeroTransformer 相比 ViT 的主要优势不是只来自“更深”，而是层级 token 压缩 + 局部注意力显著降低了显存，因此可以实际扩展 batch size 和模型规模。

## 7.2 batch size

固定 optimization steps 时，batch size 从 4 增至 64，测试误差持续下降，且大致与 batch size 的对数呈近线性改善；超过 64 后收益变小。代价是训练时间和显存近似线性增加。

这意味着本文实际强调的是：**显存节省可以被重新投资到更大的 batch，从而换取更稳定梯度和更好最终性能。**

## 7.3 scaling

随着预训练数据量和模型规模增加，测试 SFE 都下降。最佳 L-size 模型在预训练分布的 held-out 测试集上达到约 **0.110% SFE**。

同时：

- 模型从 S 扩到 L，训练时间和峰值显存大约翻倍；
- 但单个几何、9 个工况的推理时间仍约为 30 ms 量级，增长不大；
- 数据增加的收益逐渐饱和，但部分原因是所有设置用了相同训练步数，大数据版本实际上每个样本看到的 epoch 更少。

# 8. 下游 CRM 任务：预训练到底值不值

这是整篇论文最关键的实验。

作者只取 CRM 数据集的大约 20%，即约 **450 个流场样本 / 58 个机翼几何**，并做 10-fold cross-validation。每个 fold 中，测试折固定，然后从其余数据中再取 20% 训练模型。完整参数微调只进行 5.6k steps，约 49.6 分钟。

![[assets/Towards a Foundation-Model Paradigm for Aerodynamic Prediction in Three-dimensional Design/Figure 10.png]]

论文中图10的结果：

| 方法 | SFE | $\delta C_L$ | $\delta C_D$ |
|---|---:|---:|---:|
| 从头训练 | 2.28% | 38.10×10⁻³ | 27.85×10⁻⁴ |
| Zero-shot 预训练模型 | 0.86% | 14.66×10⁻³ | 8.87×10⁻⁴ |
| **预训练 + 450 样本微调** | **0.36%** | **4.06×10⁻³** | **3.36×10⁻⁴** |

结论非常清楚：

- **不做任何微调的 zero-shot 模型已经比同样规模、用 450 个样本从头训练的模型好很多；**
- 微调后 SFE 相对从头训练下降约 **84.2%**；
- 相比 zero-shot，微调又削掉大约 2/3 误差。

因此预训练的价值不是“让优化更快收敛到同一个解”，而是**改变了小数据下可达到的误差下限**。

论文进一步固定 450 样本、不断增加训练步数。从头训练即使增加到非常大的优化预算，SFE 仍饱和在约 1.7%；微调模型则在 0.3% 左右饱和。这支持作者的判断：性能差异来自迁移知识，而不只是更好的 initialization。

# 9. 下游数据怎么花最划算

![[assets/Towards a Foundation-Model Paradigm for Aerodynamic Prediction in Three-dimensional Design/Figure 14.png]]

论文中图14研究从几十个到约两千个任务样本的数据规模。

核心发现：

1. task-specific 样本增加时，微调性能近似持续改善；
2. 只有 $O(10)$ 个样本时，已经能做到约 0.5% 量级 SFE；
3. $O(10^3)$ 个样本时可以降到约 **0.231%**；
4. 同样样本预算很小时，**优先增加不同几何形状，而不是在少数几何上增加更多工况**，效果更好。

作者解释：预训练模型对 operating condition 已经具有较强泛化能力，所以微调数据最稀缺的方向反而是新的几何变化。

这对设计数据集构造有一个非常实用的结论：**小预算阶段先铺开 shape diversity，再补同一 shape 的多工况密度。**

# 10. 数据生成和 GPU 微调之间如何分预算

作者定义了一个时间成本比：

$$
\gamma=\frac{\text{生成10个CFD样本的时间}}{\text{1000个训练step的时间}}
$$

在论文硬件环境中：

- 2 个节点、每节点 80 CPU 核时，$\gamma\approx4.6$；
- 如果增加到 16 个同配置节点，$\gamma\approx0.57$。

![[assets/Towards a Foundation-Model Paradigm for Aerodynamic Prediction in Three-dimensional Design/Figure 16.png]]

论文中图16说明：

- $\gamma$ 大，即 CFD 数据很贵时，额外预算更适合花在**增加训练步数**；
- $\gamma$ 小，即 CFD 可以高度并行时，额外预算更适合花在**增加下游样本**。

这其实是整篇论文中很有工程价值的一点：最优数据量不是一个固定数字，而取决于“CPU/HPC 数据生成”和“GPU 训练”之间的相对价格。

# 11. 参数高效微调

论文比较三类方案：

1. 全参数微调；
2. 只更新 attention 中的 query/key/value projection；
3. 在 attention 的 query/value projection 上使用 **Low-Rank Adaptation（LoRA，低秩适配）**。

LoRA 将权重更新写为：

$$
W'=W+\frac{\alpha}{r}W_1W_2
$$

其中 $r\ll d$，原始 $W$ 冻结，只训练两个低秩矩阵。本文使用 $\alpha=2r$，并测试 $r=4,16$。

450 样本下：

| 微调方式 | 可训练参数 | $\delta C_p$ | $\delta C_D$ |
|---|---:|---:|---:|
| 全参数 | 14.52M (100%) | 0.435% | 3.36×10⁻⁴ |
| 仅 attention | 2.19M (15.0%) | 0.465% | 3.45×10⁻⁴ |
| LoRA-16 | 0.24M (1.7%) | 0.499% | 3.66×10⁻⁴ |
| LoRA-4 | 0.06M (0.4%) | 0.567% | 4.15×10⁻⁴ |

只微调 attention 仅多约 6.1% 的表面流场误差，却把可训练参数压到 15%。而在非常少的数据（$O(10^1)-O(10^2)$）下，只更新 attention 反而可能比全参数微调更好，因为过拟合风险更小。

![[assets/Towards a Foundation-Model Paradigm for Aerodynamic Prediction in Three-dimensional Design/Figure 17.png]]

LoRA 虽然进一步把参数压到 1.7% 甚至 0.4%，但精度下降更明显。作者因此认为，这个下游气动问题需要的适配自由度高于极低秩更新能够表达的范围。

# 12. 论文真正证明了什么

本文验证了几个具有“foundation model 风格”的性质：

1. **预训练数据越多，下游越好；**
2. **预训练模型越强，下游越好；**
3. 即使预训练几何参数化更粗，仍可以迁移到参数化更细的局部设计域；
4. 下游只需要远少于从头训练所需的数据；
5. 预训练模型可以 zero-shot 使用；
6. 预训练权重可以通过全参数、attention-only 或 LoRA 等不同成本的方式适配。

但它并没有证明“已经得到通用气动基础模型”。论文自己明确限定：当前工作只研究跨声速机翼，而且所有几何仍要求能映射到统一的二维结构化翼面网格。

# 13. 局限

## 13.1 几何通用性仍有限

模型输入依赖统一 $256\times128$ 结构化表面网格。因此对于：

- 翼身组合；
- 发动机短舱；
- 多部件高升力构型；
- 汽车等复杂拓扑；

当前网络不能直接照搬。论文提出未来可用 projection-based 或 voxel-based embedding，但没有在本文验证。

## 13.2 物理条件覆盖有限

实验固定 Reynolds 数和自由来流温度，只变化 $Ma$ 和 $\alpha$。因此“跨工况泛化”主要指这两个变量，不代表对 Reynolds、温度、湍流模型等都有基础模型能力。

## 13.3 预训练数据仍然昂贵

SuperWing 近 3 万个 RANS 流场，论文称用了两个 HPC 节点约 4 个月生成。作者的论点是这属于一次性、可并行投资，而不是说其成本低。

## 13.4 下游验证只有一个主要局部域

本文最重要的微调实验围绕 CRM。要证明真正的“多任务可复用基础模型”，还需要多个彼此明显不同的下游基准构型和设计问题。

## 13.5 代理模型精度不等于优化可靠性

论文强调设计用途，也展示 WebWing 交互应用，但核心实验主要评价预测误差，并没有系统展示大量完整气动优化任务中：

- optimizer 是否因 surrogate bias 走到错误区域；
- 最优解的 CFD 回代误差；
- 约束可行性和鲁棒性；
- 超出微调局部域后的失效边界。

因此“适合优化”在本文中主要由高精度和已有研究动机支持，而不是被完整优化 benchmark 直接证明。

# 14. 对气动生成式设计/代理建模的启发

## 14.1 预训练几何不必与最终设计变量完全一致

这是本文最值得借鉴的思想之一。预训练集使用 37 自由度的简化参数化，微调集却是 153 自由度。模型迁移仍然有效。

这意味着建立可复用气动模型时，不一定需要提前定义未来所有设计任务的最终参数化。更可行的方法是：

- 预训练阶段先保证主导形态变化和气动物理覆盖；
- 真正进入某个设计问题后，再用更高维、更细的局部参数化补充。

## 14.2 “几何多样性”和“工况多样性”的边际价值不同

小样本微调时，多覆盖不同 geometry 比同一 geometry 多覆盖工况更重要。对于后续生成式外形设计，这暗示数据预算应首先保证潜在设计流形上的几何覆盖，而不是把大量 CFD 都堆在少数外形的密集 AoA/Ma 网格上。

## 14.3 预测中间物理场比直接预测性能标量更有泛化性

ATcoef 的过拟合和 ATsurf 的稳定泛化说明：如果最终优化目标来自一个明确的物理积分或后处理，最好让网络先预测“足够丰富的中间物理量”，再从中计算目标。

对生成式设计来说，这比只训练“几何 code → $C_L,C_D$”更有价值，因为流场还能：

- 提供局部约束；
- 解释 shock、分离、摩擦分布；
- 作为后续多任务性能计算的共享表征；
- 给优化器提供更强的物理一致性。

## 14.4 全局物理量可以作为高维场监督的结构化正则项

$L_{surf}+\lambda L_{coef}$ 的结果表明，局部点误差和全局积分误差并不是替代关系。把整体气动力加进 loss，相当于告诉网络“哪些空间误差真正重要”。这比单纯对网格点 uniform weighting 更接近设计目标。

# 15. 相关工作与研究脉络（简化）

## 15.1 Fast Buffet-Onset Prediction and Optimization Method Based on Pretrained Flowfield Prediction Model

这项较早工作已经使用“预训练流场模型 + 下游气动设计/优化”的思想，重点是二维翼型和 buffet-onset 相关问题。它建立了一个关键前提：**预训练流场代理模型不只是做快速 CFD 替代，还可以直接进入气动优化工作流。** 当前论文把这一思想扩展到了更高维、更昂贵的三维跨声速机翼。

## 15.2 Transferable Machine Learning Model for the Aerodynamic Prediction of Swept Wings / Rapid Aerodynamic Prediction of Swept Wings via Physics-Embedded Transfer Learning

这一阶段的工作开始系统研究三维后掠翼上的 transfer learning，说明不同机翼之间存在可迁移的气动表征。当前论文进一步把“从一个任务迁到另一个任务”的 transfer learning 提升成“广域大数据预训练 + 多个潜在下游任务”的 foundation-style 框架。

## 15.3 SuperWing: a Comprehensive Transonic Wing Dataset for Data-Driven Aerodynamic Design

SuperWing 是当前工作的直接数据基础。它通过简化但广覆盖的 37 自由度参数化构造了 4239 个机翼、近 3 万个流场，使“大规模预训练”在三维跨声速机翼上成为可能。当前论文的关键推进在于：**不再把 SuperWing 只视为一个训练数据集，而是把它视为共享预训练资产。**

整体研究脉络可以概括为：

**二维预训练流场代理 → 三维后掠翼迁移学习 → 大规模三维数据集 → 可复用的 foundation-model 式预训练/微调范式。**

# 16. 关键概念补充（联网资料）

> [!info] 这一节是外部资料补充，不是论文正文直接给出的解释。

## 16.1 Shifted Window 为什么能降低复杂度

Swin Transformer 的核心思想是把全局 $N$ 个 token 的 attention，改成固定大小局部窗口内计算。全局 self-attention 的主要计算随 token 数近似二次增长，而固定窗口 attention 对图像/网格面积近似线性增长。然后相邻 block 把窗口平移半个窗口，使上一个 block 中彼此分隔的窗口在下一层发生交叉连接。

原始来源：Liu et al., *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*, arXiv:2103.14030。

## 16.2 adaLN-Zero 的直觉

Adaptive Layer Normalization（AdaLN）不是只做固定的 LayerNorm，而是让条件向量生成 scale 和 shift，从而控制每层特征：

$$
AdaLN(h,c)=\gamma(c)\odot LN(h)+\beta(c)
$$

Diffusion Transformer（DiT）中的 adaLN-Zero 进一步让调制/残差门控在初始化时接近零，使网络初始行为更接近恒等映射，再逐步学习条件控制。后续分析工作也发现 zero initialization 对稳定性和性能很关键。

原始来源：Peebles & Xie, *Scalable Diffusion Models with Transformers*, arXiv:2212.09748。后续机制分析可参考 *Unveiling the Secret of AdaLN-Zero in Diffusion Transformer*, arXiv:2608.09438。

但再次强调：本文 AeroTransformer 只明确写“受 adaLN-Zero 启发”，并未完整声明采用原始 DiT 的所有初始化细节。

## 16.3 LoRA 为什么能少训练很多参数

LoRA 假设从预训练模型到下游模型所需的权重变化 $\Delta W$ 具有较低的有效秩，因此不直接学习完整 $d\times k$ 更新矩阵，而学习：

$$
\Delta W=AB,\quad A\in\mathbb R^{d\times r},\quad B\in\mathbb R^{r\times k},\quad r\ll\min(d,k)
$$

这样参数量从 $dk$ 降到 $r(d+k)$。本文只给 query 和 value projection 加 LoRA，而 key projection 冻结。

原始来源：Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685。

## 16.4 ConFIG 在做什么

ConFIG 面向多损失训练。若不同 loss 的梯度方向互相冲突，简单加权可能导致一个任务改善、另一个任务恶化。ConFIG 构造一个更新方向，使其与每个 loss-specific gradient 都保持正点积，从几何上避免“直接反着某个目标走”。

本文尝试用它处理 $L_{surf}$ 与 $L_{coef}$，但没有优于手动 $\lambda$。这反而说明这里的主要问题可能不是严重梯度冲突，而更像是不同误差指标对空间区域的权重不同。

来源：Liu, Chu & Thuerey, *ConFIG: Towards Conflict-Free Training of Physics Informed Neural Networks*, ICLR 2025。

# 17. 复现时最值得记录的超参数

```text
Input surface mesh: 256 × 128 × 3
Operating conditions: [Ma, AoA]
Output ATsurf: Cp, Cf_streamwise, Cf_spanwise
Patch size: 4 × 4
Initial tokens: 64 × 32 = 2048
Window size: 8 × 8
Stages: 2 / 5 / 8 / 5 / 2 Transformer blocks
Total blocks: 22
Attention heads: 8
MLP ratio: 4
Hidden dim S/M/L: 16 / 32 / 64
Pretraining optimizer: AdamW
Pretraining LRmax: 1e-3
One-cycle: 0.04 LRmax -> LRmax -> 0.001 LRmax
Gradient clipping: norm 1.0; L-size main runs 0.5
Main scaling batch size: 64
Main scaling training steps: 585.6k
Auxiliary aero loss weight: lambda ≈ 0.1 preferred
Fine-tuning LRmax: 1e-4
Fine-tuning initial LR factor: 0.005 LRmax
LoRA: Q and V only, alpha = 2r, r = 4 or 16
```

# 18. 最终评价

这篇论文最重要的贡献不是某一个 Transformer 模块，而是把三维气动代理模型的经济性问题说清楚并做了相对完整的实验验证：**预训练数据负责“广”，微调数据负责“细”；模型结构必须足够可扩展，才能真正把一次性预训练成本摊到多个下游任务。**

从实验结果看，这一范式在 CRM 局部设计域上确实显著提高了数据效率，而且 zero-shot、scaling、parameter-efficient fine-tuning 等现象都符合基础模型范式的预期。不过当前模型仍是“跨声速机翼基础模型的雏形”，距离任意三维气动构型的统一 foundation model 还有明显的几何表示和数据覆盖问题。
