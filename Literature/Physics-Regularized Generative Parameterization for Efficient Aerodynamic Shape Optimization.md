---
title: "Physics-Regularized Generative Parameterization for Efficient Aerodynamic Shape Optimization"
authors: [Xiangyu Gu, Xinwei Wang, Mengying Ma, Jichao Li]
year: 2026
venue: "AIAA Journal"
doi: "10.2514/1.J066536"
url: "https://doi.org/10.2514/1.J066536"
source: "未提供本地 PDF；依据公开期刊摘要/作者页面核对：https://www.researchgate.net/publication/404156220_Physics-Regularized_Generative_Parameterization_for_Efficient_Aerodynamic_Shape_Optimization"
status: abstract-reviewed
tags: [literature, physics-regularization, generative-parameterization, aerodynamic-shape-optimization, blended-wing-body]
---

> [!abstract] 摘要译述
> 本文提出物理正则化生成参数化方法，将数据驱动气动建模与生成学习结合，以构造紧凑的气动形状设计空间。方法先通过深度学习几何过滤构造模态设计空间，再使用自演化气动建模和自适应采样引入物理正则，随后以优化采样生成满足几何与性能约束的多样可行形状，并在该数据集上训练生成模型。作者在 BWB 构型上验证了设计变量压缩及代理优化收益。

# 方法脉络（摘要可支持的范围）

1. Deep-learning-based geometric filtering：从高维几何中提取紧凑模态设计空间。
2. Self-evolving aerodynamic modeling：配合 adaptive sampling 逐步改善气动模型，并将物理信息纳入采样过程。
3. Optimization-based sampling：在几何和性能约束下生成多样可行形状。
4. Generative parameterization：使用物理正则化数据集训练生成模型，获得用于形状优化的低维 latent space。

作者报告 BWB 案例中相较已有几何过滤方法减少设计变量，代理优化成本下降超过 55%，快速数据驱动优化中的阻力预测误差由 32.5% 降至 3.2%。这些数字来自公开摘要/二级摘要页面，需全文核验其基线、误差定义和实验设置后才能用于定量比较。

# 对 GRASS 的直接启发

这项工作与 GRASS 的交替工程闭环最接近：生成模型不只拟合历史几何，还要通过性能模型、自适应采样和可行性筛选形成物理正则化数据分布。

但不能仅凭摘要判断其训练流程是否与 GRASS 的“生成候选—VSPAERO 真值标注—Transolver replay 微调”相同。需要全文比较生成器是否更新、代理模型是否更新、真值数据如何回流、是否保留历史分布，以及是否采用独立验证集。

GRASS 的差异化实验应重点证明：
- 可变翼面数量和变长截面是否由同一生成器处理；
- 新生成候选的真值标签如何改善代理模型；
- 双验证集是否限制灾难性遗忘；
- 物理评估成本与代理误差是否共同下降，而非只报告训练 loss。

# 局限与证据边界

当前状态为 abstract-reviewed，未取得全文。不能补写具体网络架构、训练轮次、采样算法、完整约束清单或置信区间。摘要中的 BWB 验证不等于跨离散布局验证。

# 关联

- [[Literature/Topics/跨布局飞行器生成与新型网络架构]]
- [[Literature/Topics/几何表示与性能条件生成]]
- [[Literature/BlendedNet++ A Dataset and Benchmark for Field-Resolved Aerodynamics and Inverse Design of Blended Wing Body Aircraft]]
