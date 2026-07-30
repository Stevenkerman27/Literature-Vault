---
title: "Data-driven surrogate model for aerodynamic design using separable shape tensor method"
authors: [Bo Pang, Yang Zhang, Junlin Li, Xudong Wang, Min Chang, Junqiang Bai]
year: 2024
venue: "Chinese Journal of Aeronautics"
doi: "10.1016/j.cja.2024.03.014"
tags: [paper, aerodynamic-design, surrogate-based-optimization, shape-parameterization, grassmann-manifold, dimensionality-reduction]
status: read
source: "D:\\3D\\Projects\\Papers\\storage\\9JN85TC4\\Pang 等 - 2024 - Data-driven surrogate model for aerodynamic design using separable shape tensor method.pdf"
---

# Data-driven surrogate model for aerodynamic design using separable shape tensor method

> [!abstract] 摘要直译
> 随着设计变量维度增加以及约束日益复杂，代理模型优化（SBO）的有效性受到限制。传统的线性和非线性降维算法主要以各种形式分解由设计变量或目标函数构成的数学矩阵；在此过程中，无法保证设计空间的平滑性，且还需要在优化中加入额外约束函数，从而增加计算成本。本文提出一种新的参数化方法，同时改善 SBO 的这两个问题。该参数化通过在 Grassmann 子流形内解耦仿射变换（缩放、旋转、剪切和平移），使翼型的物理信息能够在高维空间中被分别表示。在此基础上，采用主测地分析（PGA）实现几何控制、压缩设计空间、减少设计变量数量和维度，并提高代理优化过程中的预测性能。为进行比较，论文以保留 95% 能量定义降维空间，并以跨声速 RAE 2822 为示例。该方法显著提高了代理模型的优化效率，同时能够有效施加几何约束。在三维问题中，它可以同时设计飞机不同部件的平面形状及高阶扰动变形。应用于 ONERA M6 机翼时，优化后升阻比达到 18.09，较基准构型提高 27.25%；传统代理模型优化仅提高 17.97%，由此展示了该方法的优势。

## 问题与动机

代理模型优化以 Kriging 等近似模型替代昂贵的 CFD，再通过无梯度全局搜索寻找气动外形。但样本数量会随设计变量维数迅速增长。传统 CST 或 Hicks-Henne 参数化直接扰动坐标/多项式系数，把局部弯度变化和整体缩放、旋转、剪切混在同一空间；随机样本因而常违反面积、厚度等几何约束，传统 SBO 还需训练额外的约束代理模型。

本文的策略是先按物理含义拆分自由度，再对真正描述“高阶形状”的部分降维：将仿射变换作为独立、可控制的布局量，而不是与局部外形扰动竞争同一组优化变量。

## 方法

![[assets/data-driven-surrogate-model-separable-shape-tensor/method-overview-p009.png]]

*图 13，Grassmannian-SBO：初始设计空间经 DOE、极分解和对数空间映射得到 Grassmann 设计空间；PGA 压缩后进行 DOE、CFD、代理模型、遗传算法和补点迭代。*

### 形状-仿射分离

将有序采样的二维翼型写成坐标矩阵 $X \in \mathbb{R}^{n \times 2}$。大尺度仿射变形为 $MX+b$，其中 $b$ 表示平移，$M \in GL(2)$ 包含缩放、剪切和旋转。作者采用 Landmark Affine Standardization（LAS），先去中心化并标准化协方差，再以 SVD 得到 $\tilde X$。

$\tilde X$ 表示去除了平移和一般线性变换影响的高阶形状；平均仿射矩阵 $\bar M$ 则保留尺度、姿态和布局信息。对机翼，$M$ 的非对角项可以单独控制截面扭转。因此仿射量没有被删除，而是从形状模态中解耦，按需要作为额外设计变量使用。

### Grassmann 流形上的 PGA

LAS 表示是 Grassmann 流形上的点，不应直接以欧氏直线做 PCA 插值。论文先求样本的 Karcher 内蕴均值，将数据通过对数映射投到均值处的切空间；随后在切空间中做 SVD，按累计 95% 能量保留主测地方向。生成候选外形时，再经指数映射回流形，并与 $\bar M$ 组合恢复物理坐标。

这种 PGA 保留流形上的测地结构，低维变量更接近训练数据覆盖的可行形状集合。它解决的不是单纯压缩坐标，而是避免普通 POD/PCA 在多基准翼型间插值时走入未定义的形状区域。

### Grassmannian-SBO

低维 PGA 系数空间中使用 Latin hypercube sampling 生成初始样本，RANS CFD 给出气动响应，Kriging 以预测均值和 Expected Improvement 指导补点，遗传算法负责搜索。常规 SBO 会为面积、最大厚度等约束分别建立 Kriging 模型；本文通过在流形子空间采样，使候选形状天然接近训练数据的几何尺度，从实验上免去这些额外约束模型。

“不需要额外约束”并不代表所有工程约束被严格满足，而是面积、厚度等与训练集分布一致的约束无需额外代理近似；结构、制造或未编码的气动稳定性仍须显式验证。

## 实验结果

### RAE 2822 跨声速翼型

- 工况为 $Ma=0.73$、$Re=6.5 \times 10^6$、攻角 $2.3^\circ$，优化目标为升阻比。
- CST 空间使用 18 个变量；保留 95% 能量后的 Grassmann 空间使用 8 个变量。
- 两种方法均以 30 个初始 CFD 样本训练。常规 SBO 还设置面积不超过 1.5%、厚度不超过 4.5% 的约束代理模型。
- Grassmannian-SBO 在累计 70 次 CFD 评估时达到 $K=75.83$；常规 SBO 需 158 次才达到近似的 $K=75.80$。两者后期 Kriging 相对误差约为 1.58% 与 1.60%。
- 流形方案的面积、厚度变化分别约小于 1% 和 2%；常规方案的中间样本可达约 6% 和 7%。

### ONERA M6 三维机翼

| 方法 | 设计变量 | 最终升阻比 | 相对基准提升 |
| --- | ---: | ---: | ---: |
| 基准 ONERA M6 | - | 14.219 | - |
| 常规 SBO | 48 | 16.774 | 17.97% |
| Grassmannian-SBO | 24 | 17.534 | 23.31% |
| Grassmannian-SBO-2（含 2 个扭转变量） | 24 + 2 | 18.093 | 27.25% |

流形方案将三个翼展截面的 48 个 CST 变量降至 24 个；第二种方案再独立调节两个截面扭转角。优化后上表面激波减弱，加入负扭转后进一步降低激波阻力。实验中，流形方案面积变化保持在 2% 以内、厚度变化在 6% 以内；论文报告常规 SBO 的中间构型可出现超过 11% 的几何偏离。

## 关键贡献

1. 将高阶形状扰动与仿射变换分离，使整体布局/尺度控制不再完全耦合于局部气动造型。
2. 用 Grassmann 流形上的 PGA 替代欧氏 PCA/POD，在保留内蕴几何的同时压缩设计变量。
3. 将该表示接入 Kriging-EI-遗传算法 SBO，在实验中同时降低变量维度和几何约束代理模型的训练负担。
4. 展示了把仿射分量作为独立布局变量扩展到三维机翼扭转优化的方式。

## 局限与启发

- **可行性是近似的。** 子空间仅使样本靠近训练分布，不能替代结构强度、制造规则、局部曲率等硬约束。
- **依赖一致的点对应。** LAS/PGA 要求翼型地标点的语义和顺序一致；不同网格或拓扑需先重采样与配准。
- **95% 能量不等价于气动最优。** 小方差模式仍可能影响激波、分离和载荷，工程应用应检查截断模式的目标敏感性。
- **对比收益来源耦合。** 性能提升同时来自降维、可行域收缩和少训练约束模型，不能单独归因于 PGA。

可复用的设计原则是：先区分哪些几何自由度应被搜索、哪些变化只应被单独控制，再在与数据结构一致的空间中压缩变量。该思路可与主动学习、多保真代理和生成式几何先验结合。

## 关联

- 基础方法：[[Separable shape tensors for aerodynamic design]]
- 主题：[[Aerodynamic shape optimization]]、[[Surrogate-based optimization]]、[[Grassmann manifold]]、[[Principal geodesic analysis]]
