---
title: "AI-Based Generative Algorithms applied to the design of Blended Wing Body Aircraft"
authors: [Marta A. Martín, Andrés Mateo-Gabín, Thomas Wagenaar, Gonzalo Rubio, Sven A. Lanzan Ferran]
year: 2025
venue: "AIAA Aviation Forum and ASCEND 2025"
doi: "10.2514/6.2025-3292"
url: "https://doi.org/10.2514/6.2025-3292"
source: "本轮对话上传 PDF；未归档到 Literature-Vault"
status: read
tags: [literature, blended-wing-body, BWB, DDPM, diffusion-model, 1D-U-Net, conditional-generation, aerodynamic-design]
---

> [!abstract] 摘要译述
> 本文提出一种用于生成满足指定气动目标的翼身融合体（Blended Wing Body，BWB）几何的生成建模框架。方法将去噪扩散概率模型（Denoising Diffusion Probabilistic Model，DDPM）与定制的 1D U-Net 结合，并以升力、阻力和俯仰力矩系数作为条件。BWB 几何由一个紧凑的 24 维参数向量描述，包含平面形状和翼型截面特征；数据通过 GEMSEO 协调 DUST 与 NeuralFoil 的中等保真气动分析生成。实验显示，模型能够生成具有几何多样性、且气动系数接近目标的 BWB 构型。不过训练没有加入结构或可制造性约束，因此结果应视为气动概念设计，而不是完整工程可行方案。

# 一句话定位

**该论文将 BWB 的 24 维结构化几何向量视为一维信号，用条件 DDPM + 1D U-Net 学习气动可行几何分布，并根据目标 $(C_L,C_D,C_M)$ 反向去噪生成多个方案。**

它与 EDEN 的性能加权离散构型分布、deepSPACE 的 Design Style 搜索、DiffGeo 的能量引导几何潜扩散不同：本论文把气动系数直接作为去噪网络的条件输入，但生成对象仍是一个固定 BWB 参数化空间中的连续几何向量。

## 1. 研究问题与整体流程

作者关注 BWB 的早期概念设计。其几何空间高维、非凸，局部几何变化会影响升力、阻力和俯仰力矩。传统参数扫描和优化方法需要反复调用气动求解器；已有 MLP、VAE、GAN 方法则可能出现多样性不足、mode collapse，或难以捕捉几何参数之间的局部/全局相关性。

论文流程分为四步：

1. 用全局平面形状参数与局部 CST 翼型系数构造 24 维几何向量；
2. 通过 GEMSEO 协调 DUST 和 NeuralFoil，为样本生成气动系数；
3. 训练以目标气动系数为条件的 DDPM，噪声预测器使用 1D U-Net；
4. 输入目标 $(C_L,C_D,C_M)$，从高斯噪声逐步去噪得到新的 BWB 几何，再通过 DUST 重新评价。

## 2. BWB 几何参数化：24 维向量

### 2.1 CST 翼型表示

翼型表面使用 Class-Shape Transformation（CST，类别—形状变换）表示：

$$
y(x)=C(x)\,S(x).
$$

其中 $x\in[0,1]$ 是归一化弦向坐标；class function 控制前缘与后缘的端点行为，shape function 由 Bernstein 多项式展开：

$$
C(x)=x^{N_1}(1-x)^{N_2},
\qquad N_1=0.5,\quad N_2=1.0,
$$

$$
S(x)=\sum_{i=0}^{N}A_i B_i^N(x),
$$

$$
B_i^N(x)=\binom Ni x^i(1-x)^{N-i}.
$$

本文采用对称翼型假设，只显式参数化上表面：

$$
y_U(x)=C(x)S_U(x),
\qquad
y_L(x)=-y_U(x).
$$

这样可以减少自由度，并使生成翼型保持上下表面对称。

### 2.2 展向分段与全局参数

BWB 沿展向分成五个区域，并在四个主要截面位置定义局部几何。每个区域包含：

- 后掠角 $\Lambda$；
- 上反角 $\Gamma$；
- 区域展向长度 $b$。

每个截面位置包含：

- 弦长 $c$；
- 扭转角 $\theta$；
- CST 翼型系数 $A_i$。

相邻截面之间，平面几何参数和 CST 系数均采用线性插值，构造连续展向变化。

### 2.3 设计变量与物理边界

论文表 2 给出的参数范围如下：

| 类别 | 参数 | 符号 | 下界 | 上界 | 单位 |
|---|---|---:|---:|---:|---|
| 弦长 | Root chord | $c_1$ | 20.0 | 30.0 | m |
| 弦长 | Mid chord | $c_3$ | 5.0 | 10.0 | m |
| 弦长 | Tip chord | $c_4$ | 3.0 | 4.0 | m |
| 弦长 | Winglet chord | $c_5$ | 1.5 | 3.0 | m |
| 展长 | Root span | $b_1$ | 5.0 | 8.0 | m |
| 展长 | Transition span | $b_2$ | 3.0 | 5.0 | m |
| 展长 | Wing span | $b_3$ | 13.0 | 19.0 | m |
| 展长 | Winglet span | $b_4$ | 1.5 | 2.5 | m |
| 扭转 | Transition twist | $\theta_2$ | 2.0 | 8.0 | ° |
| 扭转 | Wing twist | $\theta_3$ | -2.0 | 4.0 | ° |
| 扭转 | Winglet twist | $\theta_5$ | -5.0 | 5.0 | ° |
| 后掠 | Root sweep | $\Lambda_1$ | 40.0 | 60.0 | ° |
| 后掠 | Transition/Wing sweep | $\Lambda_{23}$ | 20.0 | 40.0 | ° |
| 后掠 | Winglet sweep | $\Lambda_4$ | 35.0 | 60.0 | ° |
| 上反 | Root dihedral | $\Gamma_1$ | 0.0 | 10.0 | ° |
| 上反 | Wing dihedral | $\Gamma_3$ | 0.0 | 10.0 | ° |
| CST | Root/Transition profile | $A_{u1}$ | $[0,0,0,0]$ | $[0.2,0.2,0.2,0.2]$ | - |
| CST | Wing/Winglet profile | $A_{u3}$ | $[0,0,0,0]$ | $[0.2,0.2,0.2,0.2]$ | - |

作者还施加了若干耦合约束，包括 root twist、transition sweep、transition dihedral、winglet dihedral 和 CST 系数共享。原文表 1 中 “Wing twist” 对应的等式在 PDF 中显示为 $\theta_3=\theta_3$，看起来是排版/符号错误；这里不擅自改写为其他截面关系。论文称所有变量归一化到 $[0,1]$，同时将该过程描述为 z-score normalization；这两种术语通常指向不同的变换，原文没有进一步澄清具体实现。

## 3. 数据集与中等保真气动评价

GEMSEO 负责组织参数采样、几何约束和两个学科模块：

- **DUST：** 基于 Vortex Lattice Method（VLM，涡格法）的快速三维升力面分析，输出 $C_L,C_D,C_M$。
- **NeuralFoil：** 用深度学习近似 XFoil 的二维翼型极线，为 DUST 提供各展向截面的气动极线。

翼型极线计算条件：

| 参数 | 数值 |
|---|---:|
| 巡航 Mach 数 | 0.80 |
| 自由来流速度 | 238.0 m/s |
| 巡航高度 | FL350 |
| 压力 | 23.8 kPa |
| 温度 | 216.7 K |
| 密度 | 0.364 kg/m³ |
| 动力黏度 | $1.46\times10^{-5}$ Pa·s |
| 攻角扫描 | $-180^\circ$ 至 $+180^\circ$ |

每个截面的 Reynolds 数根据局部弦长计算。论文指出，攻角并未作为单独设计变量输入；有效气动入射角由几何向量中的局部扭转分布共同决定。

数据描述存在一个需要保留的数量口径问题：数据章节称初始生成 1,500 个样本，过滤掉求解不收敛、非物理系数和 NaN 后按 70%/20%/10% 划分；另一处又称最终数据集包含“数千”个 BWB 几何，但没有给出最终精确样本数。因此不能将 1,500 直接当成过滤后的最终数据集规模。

## 4. DDPM：前向加噪与反向去噪

令干净的 24 维几何向量为 $x_0$，扩散步为 $t$，噪声日程为 $\beta_t$。前向过程逐步添加高斯噪声：

$$
q(x_t\mid x_{t-1})
=
\mathcal N
\left(
x_t;
\sqrt{1-\beta_t}\,x_{t-1},
\beta_t I
\right).
$$

定义：

$$
\alpha_t=1-\beta_t,
\qquad
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s.
$$

由此可以直接采样任意时间步的噪声几何：

$$
x_t
=
\sqrt{\bar\alpha_t}\,x_0
+
\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
$$

反向过程学习条件分布：

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal N
\left(
x_{t-1};
\mu_\theta(x_t,t,c),
\Sigma_\theta(x_t,t)
\right),
$$

其中条件向量为：

$$
c=(C_L,C_D,C_M).
$$

论文采用噪声预测参数化，反向均值为：

$$
\mu_\theta(x_t,t,c)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-
\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t,c)
\right).
$$

### 4.1 训练目标

每次训练：

1. 从数据集采样干净几何 $x_0$；
2. 均匀采样扩散步 $t$；
3. 采样高斯噪声 $\epsilon$；
4. 根据 $\bar\alpha_t$ 构造 $x_t$；
5. 让网络预测添加的噪声。

训练损失为：

$$
\mathcal L_{\mathrm{simple}}
=
\mathbb E_{x_0,\epsilon,t}
\left[
\left\|
\epsilon-\epsilon_\theta(x_t,t,c)
\right\|_2^2
\right].
$$

### 4.2 条件生成采样

从标准高斯噪声开始：

$$
x_T\sim\mathcal N(0,I).
$$

然后对 $t=T,\ldots,1$ 迭代预测噪声并更新 $x_t$，最终得到 $x_0$。同一目标条件下，不同随机噪声路径可以产生多个几何方案，这正是模型保留同条件多样性的来源。

## 5. 1D U-Net：几何向量上的噪声预测器

网络输入为：

$$
x_t\in\mathbb R^{1\times24},
\qquad
c\in\mathbb R^3.
$$

作者把 24 个几何参数看作一维结构化信号，而不是把它们当成彼此独立的标量。

### 5.1 时间与气动条件编码

- 时间步 $t$ 先通过 sinusoidal position embedding；
- 时间嵌入再经过两层 MLP，使用 Swish 激活；
- 气动条件 $c=(C_L,C_D,C_M)$ 通过线性层投影；
- 时间嵌入与气动条件嵌入融合为 32 维联合嵌入 $e$；
- 该嵌入通过 Feature-wise Linear Modulation（FiLM，特征级线性调制）注入网络各处。

### 5.2 编码器—解码器结构

编码器包含三个下采样 block，每个 block 由以下模块构成：

1. FiLM 调制的 residual block；
2. self-attention；
3. stride convolution，下采样并使通道数翻倍。

瓶颈处还有一个 residual block 和一个 self-attention 层。解码器采用转置卷积逐步上采样，结合 residual block 与 encoder-decoder skip connections 恢复几何信息。Residual block 内使用 Group Normalization 和 Swish。最后通过 1D convolution 将特征映射回 24 维噪声预测。

**论文中图 9（PDF 第 13 页）**是最关键的网络结构图：它展示 $x_t$、时间 $t$ 和气动条件 $c$ 如何进入 U-Net，并通过 FiLM 在编码器、瓶颈和解码器中调制特征。

### 5.3 训练配置

| 超参数 | 论文报告值 |
|---|---:|
| Base channels | 64 |
| Time + condition embedding | 32 |
| Noise variance $\beta$ | $10^{-3}$ 至 0.02 |
| Residual blocks | 7 |
| Self-attention layers | 2 |
| Diffusion steps $T$ | 1,000 |
| Batch size | 16 |
| Learning rate | $2\times10^{-4}$ |
| Epochs | 2,000 |

原文训练策略段落称使用 AdamW 并将 gradient clipping max norm 设为 1.0，但表 4 的 Optimizer 行写的是 Adam。该处存在内部不一致，不能在没有代码佐证时擅自选定其中一个版本。

训练曲线分析也有口径差异：正文指出验证损失在约 750 epoch 后逐步上升，并认为长时间训练出现 overfitting；随后又称最终 checkpoint 选在约 1,000 epoch；结论部分则称没有观察到 overfitting。阅读结果时应保留这一矛盾，而不能简单概括为“2000 epoch 无过拟合”。

## 6. 生成质量与气动目标精度

作者用 100 个测试构型，将条件输入的目标系数与生成几何经 DUST 重新计算的真实系数比较。

| 指标 | $C_L$ | $C_D$ | $C_M$ |
|---|---:|---:|---:|
| MSE | $4.52\times10^{-4}$ | $1.93\times10^{-4}$ | $1.31\times10^{-4}$ |
| RMSE | $2.13\times10^{-2}$ | $1.39\times10^{-2}$ | $1.14\times10^{-2}$ |
| RMdSE | $1.54\times10^{-2}$ | $8.80\times10^{-3}$ | $8.23\times10^{-3}$ |
| MAPE | 6.80% | 11.84% | 13.49% |
| $R^2$ | 0.9427 | 0.7722 | 0.7441 |

升力系数的回归精度最好；阻力和俯仰力矩对几何细节更敏感，相关性较低。作者还给出绝对误差 ECDF：

- 约 80% 的 $C_L$ 预测误差低于 0.04；
- 约 90% 的 $C_D$ 预测误差低于 0.02；
- 约 85% 的 $C_M$ 预测误差低于 0.02。

同一目标 $(C_L,C_D,C_M)$ 下生成多个 BWB 几何，结果显示不同几何可以满足相近的气动目标，说明模型能够表示同条件多解，而不是只记忆一个确定性映射。

## 7. 局限与解释边界

1. **设计空间固定。** 24 维参数化描述的是一个预先定义的 BWB 几何族，不是任意飞机布局或任意部件拓扑生成。
2. **只约束气动系数。** 训练没有施加结构强度、重量、可制造性、系统稳定性或内部容积约束。
3. **中等保真分析。** DUST 与 NeuralFoil 适合快速概念评估，但不能将其结果等同于完整高保真 CFD 或飞行验证。
4. **数据量口径不清。** 正文的 1,500 初始样本与“数千”最终数据集描述没有给出一致的精确口径。
5. **训练配置存在文字/表格冲突。** Optimizer 的 Adam/AdamW 以及 validation loss/overfitting 叙述应视为原文报告的不一致。
6. **目标条件不是硬约束保证。** $C_L,C_D,C_M$ 是模型条件，最终仍需重新运行气动分析并筛选误差较大的候选。

## 8. 与其他生成式设计方法的关系

| 方法 | 几何表示 | 条件方式 | 生成对象 |
|---|---|---|---|
| [[Literature/Configuration Selection via Self-Supervised Performance-Weighted Generative Neural Networks|EDEN]] | 离散设计选择树 | MDAO 性能加权概率分布 | 离散构型候选 |
| [[Literature/deepSPACE Generative AI for Configuration Design Space Exploration|deepSPACE]] | 构型参数 + Design Style | 需求与 style 联合条件 | 跨风格构型 |
| BWB 条件 DDPM（本文） | 24 维 BWB 参数向量 + CST | $(C_L,C_D,C_M)$ 直接注入 U-Net | 固定 BWB 族内连续几何 |
| [[Literature/Aerodynamic Shape Design Space Exploration with Deep Latent Diffusion Model|DiffGeo]] | 模板网格形变 + 自动 latent | 采样时能量梯度引导 | 翼型与单个 3D 叶片几何 |

相关方法脉络：

- Ho et al., *Denoising Diffusion Probabilistic Models*（2020）：DDPM 的前向加噪、反向去噪和噪声预测训练基础。
- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*（2015）：U 型 encoder-decoder 与 skip connection 的结构基础。
- Kulfan, *Universal Parametric Geometry Representation Method*（2008）：CST 几何参数化的基础。
- Wagenaar et al., *Generative Aerodynamic Design with Diffusion Probabilistic Models*（2024）：作者相关的翼型扩散生成工作脉络。

## 9. 对 GRASS-plane 的研究启发（项目推论）

- **可借鉴：** 将整机气动目标作为条件嵌入生成网络，可作为任务条件接口的参考；FiLM 提供了一种把条件信息注入多层特征的实现方式。
- **可借鉴：** 1D U-Net 说明当设计变量具有稳定顺序和局部相关性时，可以把参数向量当作一维信号处理。
- **不能直接照搬：** 本文的输入长度固定为 24，不能证明其可处理每架飞机不同的翼面数量、每翼不同的截面数或离散装配拓扑。
- **待验证假设：** 对 GRASS 的每个翼面局部截面序列使用共享 1D U-Net/FiLM 模块，是否能改善局部几何相关性；再与现有自回归变长截面解码器进行等数据量消融。
- **物理验收：** 气动系数条件生成仍需由真实求解器重新评价，不能用条件误差或代理 loss 替代结构、配平和制造可行性检查。

## 10. 核心图片定位

- **论文中图 1（PDF 第 4 页）**：DDPM 前向加噪与反向去噪的整体流程。
- **论文中图 3（PDF 第 7 页）**：几何向量扩散训练与采样流程。
- **论文中图 4（PDF 第 8 页）**：BWB 分段几何参数化，标出弦长、展向区域、扭转和翼型参数。
- **论文中图 5（PDF 第 10 页）**：GEMSEO 协调 DUST 与 NeuralFoil 的数据生成流程。
- **论文中图 9（PDF 第 13 页）**：1D U-Net 架构及时间/气动条件 FiLM 注入。
- **论文中图 14（PDF 第 17 页）**：相同气动目标下生成的多种 BWB 几何。
- **论文中图 15–16（PDF 第 18–19 页）**：目标与真实气动系数回归及绝对误差累计分布。

## 关联

- [[Literature/Configuration Selection via Self-Supervised Performance-Weighted Generative Neural Networks|EDEN]]：性能加权的离散构型候选生成。
- [[Literature/deepSPACE Generative AI for Configuration Design Space Exploration|deepSPACE]]：需求与 design style 条件下的跨构型探索。
- [[Literature/Aerodynamic Shape Design Space Exploration with Deep Latent Diffusion Model|DiffGeo]]：小样本几何潜空间扩散与能量引导。
- [[Literature/generative_design/aircraft/BlendedNet++ A Dataset and Benchmark for Field-Resolved Aerodynamics and Inverse Design of Blended Wing Body Aircraft|BlendedNet++]]：固定 BWB 参数族中的场级代理与条件扩散逆设计对照。
- [[Literature/Topics/几何表示与性能条件生成|几何表示与性能条件生成]]。
- [[Literature/Topics/跨布局飞行器生成与新型网络架构|跨布局飞行器生成与新型网络架构]]。
