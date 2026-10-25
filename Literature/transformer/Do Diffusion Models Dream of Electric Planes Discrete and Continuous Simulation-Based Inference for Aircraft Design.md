---
title: "\"Do Diffusion Models Dream of Electric Planes?\" Discrete and Continuous Simulation-Based Inference for Aircraft Design"
authors:
  - Aurelien Ghiglino
  - Daniel Elenius
  - Anirban Roy
  - Ramneet Kaur
  - Manoj Acharya
  - Colin Samplawski
  - Brian Matejek
  - Susmit Jha
  - Juan J. Alonso
  - Adam D. Cobb
year: 2026
venue: "ICML 2026, Proceedings of Machine Learning Research 306:34698–34727"
doi: null
arxiv: "2603.13284v1"
url: "https://proceedings.mlr.press/v306/ghiglino26a.html"
code: "https://github.com/SRI-CSL/mixed-masked-diffusion"
source: "未提供本地 PDF；全文依据 https://arxiv.org/html/2603.13284v1"
status: read
updated: 2026-10-25
tags: [literature, aircraft-design, topology-generation, eVTOL, discrete-diffusion, diffusion-transformer, variable-length, simulation-based-inference]
---

> [!abstract] 原文摘要译述
> 本文在真实的概念飞机设计环境中研究模拟驱动推断（Simulation-Based Inference，SBI）：给定目标飞行性能，不输出唯一优化解，而是推断满足性能的设计分布。eVTOL 同时包含离散的机翼/旋翼/推进拓扑和随拓扑变化的连续几何与系统参数，无法简单用一个固定维连续扩散处理。作者提出层级式 MixeDiT–MaskeDiT：首先联合建模离散拓扑与连续性能观测，再根据采样拓扑，以拓扑掩码的 Transformer 扩散模型生成该拓扑有效的连续参数。作者用 SUAVE 概念设计仿真和物理/几何筛选构建 30,276 个设计、144 种 eVTOL 拓扑的基准，验证该方法可在不同条件下生成多样飞机并重现合理的工程趋势。

# 一句话理解

**这篇论文真正回答了“如何在同一个生成模型里生成机翼数、推进布局等离散拓扑，以及各拓扑不同数量的连续设计参数”——不是对一类固定翼面不断插值，也不是仅生成点云外形。**

## 1. 与既有飞机生成研究的区别

[[Literature/A comparison of methodologies for generating design-requirement-driven conceptual aircraft designs|A comparison of methodologies...]] 比较 VAE、归一化流与扩散在**固定维窄体客机特征**上的采样，其主要实验甚至是无条件生成。这里将拓扑视为生成变量 \(\tau\)，连续变量 \(\theta\) 的维度 \(D_\tau\) 随飞机拓扑而改变；并对给定性能观测 \(x\) 直接做后验生成：

\[
p(\tau,\theta\mid x)
=p(\theta\mid\tau,x)\,p(\tau\mid x).
\]

这正是跨布局的关键数学分解。

## 2. 数据和设计对象

- eVTOL 概念级设计由树状概率程序采样，包含最多两片主翼、平尾、垂尾、翼上升力旋翼组、前向推进器和可选机鼻推进器等。
- 按组件是否存在，可形成 **144 种拓扑**；性能观测向量 \(x\) 维数为 **10**，含升阻、质量和电池状态等高层性能。
- 最大连续参数槽位在不同章节的记法不同：方法正文的**有效最大连续参数为 126**，Simformer 固定 token 模板中**最大槽位为 136**；不能把两个数字混同。具体一个拓扑只激活其相应参数。
- 原始符号程序采样 **300,000** 个候选，经过几何/结构检查、稳定性判断与 SUAVE 求解收敛筛选，留下 **30,276** 个有效样本。
- SUAVE 属于概念级分析工具，主要依赖低阶气动/经验关系，**不是高保真 RANS 或整机试飞真实性验证**。作者明确指出，在罕见拓扑和强分离流动下误差会更大；训练集也仍包括实际无法飞行的非常规设计。

## 3. 模型 A：MixeDiT 产生拓扑 \(\tau\)

MixeDiT（Mixed Diffusion Transformer）同时接收离散拓扑和连续性能变量的扩散状态：

\[
(\mathbf p_\tau,\epsilon_x)
=s_\phi(\tau_{t_\tau},x_{t_x},t_\tau,t_x).
\]

主要设计：
1. 拓扑采用 Riemannian Diffusion Language Modeling（RDLM）的连续流形表示，将离散分布嵌入概率单纯形/几何空间，最终通过 argmax 还原离散类别。
2. 性能量使用连续扩散头预测噪声。
3. 将离散与连续数据联合建模；若某项性能已知，则对该项条件保持不扩散，通过指定条件从模型中采样相应拓扑。
4. Transformer 负责不同构型 token 与不同观测维度的依赖建模；这里的关键创新是**混合离散/连续分布建模**，不是新型物理注意力算子。

## 4. 模型 B：MaskeDiT 在给定拓扑下生成 \(\theta\)

MaskeDiT 基于 Simformer 的 Transformer 扩散思想，将单个飞机的连续参数表示为一组变量 token。传统 Simformer 假定这些 token 全都存在；作者引入拓扑存在掩码：

\[
M_\tau\in\{0,1\}^{d_{\max}},
\qquad
M_{\tau,i}=1 \iff \text{第 }i\text{ 个参数存在}.
\]

**条件掩码** \(M_C\) 另行表示哪些性能/参数已知，控制观测量与目标量之间的信息传播。两类掩码均用于 self-attention 以及训练损失：不存在的组件**不贡献 loss**。

模型在每个训练 step 随机采样条件集合（联合、似然、后验或 Bernoulli 选择），用有效 token 上的归一化去噪均方误差训练：

\[
\mathcal L=
\frac{\sum_iM_i\|\epsilon_{\omega,i}-\epsilon_i\|^2}{\sum_iM_i}.
\]

实际文中使用带条件掩码的 score/noise matching 和 EMA，推理使用反向随机微分方程的 Euler–Maruyama 离散积分。**掩码只是使不存在变量不参与，而非保证生成飞机的全部连续几何约束自动成立**。

## 5. 两阶段完整生成

1. 用户提供全部或部分高层性能 \(x^*\)；
2. MixeDiT 采样 \(\tau\sim p(\tau\mid x^*)\)；
3. 根据 \(\tau\) 生成存在参数 mask；
4. MaskeDiT 采样 \(\theta\sim p(\theta\mid\tau,x^*)\)；
5. 按树状数据模型解码构型，并对隐藏截面进行必要的随机补全；
6. 使用 SUAVE 和几何/结构规则重新评价，比较后验预测与条件的匹配。

该链路直接给出**跨拓扑的多解分布**，同时清楚暴露了“离散结构先验是否完整”“连续参数先验是否合理”“物理验证保真度是否足够”三个限制。

## 6. 论文实验与数值

### 拓扑条件分布

作者令除总质量外的观测保持均值，改变质量为均值 \(\pm2\sigma\)：
- 高质量组**91.6%** 生成样本含第二片翼；
- 低质量组仅 **0.4%** 含第二片翼。

这说明条件能影响**离散拓扑分布**，而不是只影响同一拓扑的连续尺度。

### 参数后验分布验证

对训练测试中样本超过 200 的 **10 种拓扑**，MaskeDiT 与为单拓扑训练的 Neural Likelihood Estimation（NLE）比较：

| 指标 | MaskeDiT | 单拓扑 NLE |
|---|---:|---:|
| MMD（越小越好） | 0.004 ± 0.001 | **0.002 ± 0.001** |
| Joint C2ST（越接近 0.5 越好） | 0.777 ± 0.040 | **0.744 ± 0.023** |
| 平均边缘 C2ST | 0.614 ± 0.016 | **0.596 ± 0.010** |

因此**不能宣称 MaskeDiT 在每项分布指标上胜过 NLE**；它的优势是用同一个带 mask 的模型覆盖不同拓扑，而 NLE 通常要为每个拓扑单独建模。

### 物理趋势与速度

- 采样结果能恢复电池质量与飞机质量、机翼面积、旋翼尺寸之间的合理关联；
- 不同拓扑对应不同翼展和阻力关系，作者用多个固定条件/改变条件案例进行分析；
- 完整生成链在论文批量实验中，相对运行 100 个 SUAVE 评估样本报告 **至少 10× 的 wall-clock 加速**。但该比较不包含离线数据产生与训练阶段。

## 7. 最主要局限

- 最大两翼等来自预先编写的构型语法，不是对任意尚未定义的飞机拓扑开放的无界搜索。
- 图像/几何补全使用概率先验，会引入额外随机性；两个共享高层参数的设计在低层细节上可不相同。
- 主要依赖低保真 SUAVE，不能把性能后验精度解释为 Navier–Stokes 或详细结构真实性。
- 设计空间中存在大量不合理飞机，数据先验与后续工程规则对模型输出质量影响很大。
- 使用一套具有上下界的组件 token 模板，尚不能证明对超出预定义最大组件数量的泛化。

## 8. 与其他论文的关系

- **本论文是明显更强的跨布局拓扑建模证据**：[[Literature/A comparison of methodologies for generating design-requirement-driven conceptual aircraft designs]] 主要处理固定维、近似固定族的概念客机表格样本。
- **另一种跨构型路线**：[[Literature/基于生成式模型的三维气动外形设计框架]] 共享三维点云潜空间并根据气动条件生成，**不显式枚举机翼/发动机拓扑树**。
- **互补的整机模块协调**：[[Literature/GLUE Coordinating Pre-Trained Generative Models for System-Level Design|GLUE]] 连接已训练的机翼/机身/内部布局生成器，但其主要实验的拓扑仍固定。
- **条件输入互补**：[[Literature/基于扩散Transformer的多约束气动外形生成式设计|多约束 DiT]] 在固定参数维的情况下处理缺失/任意组合条件，与拓扑变长 mask 回答不同问题。

## 参考资料

- [ICML 2026 会议版本](https://proceedings.mlr.press/v306/ghiglino26a.html)
- [arXiv HTML，含方法公式、实验表与附录](https://arxiv.org/html/2603.13284)
- [官方代码](https://github.com/SRI-CSL/mixed-masked-diffusion)
