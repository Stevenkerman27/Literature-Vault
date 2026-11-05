---
title: "Automated Parameterization for Aerodynamic Shape Optimization via Deep Geometric Learning"
authors: [Zhen Wei, Aobo Yang, Jichao Li, Michaël Bauerheim, Rhea P. Liem, Pascal Fua]
year: 2025
venue: "AIAA Journal, 63(12), 5430–5447"
doi: "10.2514/1.J065203"
url: "https://doi.org/10.2514/1.J065203"
source: "未提供本地 PDF；作者接受稿：https://spiral.imperial.ac.uk/entities/publication/0edd6ce1-6083-4066-a8d5-2a677383f74a"
status: abstract-reviewed
tags: [literature, deep-geometric-learning, geometric-parameterization, aerodynamic-shape-optimization, mesh-deformation]
---

> [!abstract] 摘要译述
> 本文提出 DeepGeo，一种基于神经网络的深度几何映射方法，用于自动建立气动形状优化参数化。它希望减少传统参数化对人工调参和专业经验的依赖，同时允许较大形状变形并保持全局表面平滑。该方法集成 CFD 网格变形，在二维圆形、CRM 机翼和 BWB 飞机案例中验证，取得与先进自由形变方法相当的优化结果。

# 方法与实验（公开摘要范围）

DeepGeo 的目标不是从飞机任务条件直接生成任意布局，而是自动学习一种可用于形状修改的几何参数化，并把几何变形传递给 CFD 网格。公开摘要列出的三个案例分别为二维圆形、CRM 机翼和 BWB 外形，使用相同参数化设置验证适用性。

# 与 GRASS 的关系

DeepGeo 可作为连续几何参数化/解码层的对照工作。GRASS 的主要问题更靠前：决定部件集合与数量，并为每个翼面生成变长截面几何，再处理整机装配。DeepGeo 的多案例不应误读为一个模型已经学会可变数量部件拓扑。

可比较的实验包括几何重建质量、变形平滑性、设计空间可达范围，以及将其几何参数化替换 CST/翼面解码器后的下游物理性能。

# 局限与证据边界

当前依据期刊摘要和机构库元数据建立，尚未在本笔记中逐页核对作者接受稿的完整公式和实验表。因此具体网络结构、网格节点数和优化耗时不作推断。

# 关联

- [[Literature/Topics/几何表示与性能条件生成]]
- [[Literature/Topics/跨布局飞行器生成与新型网络架构]]
- [[Literature/Scalable Parameterization of Aerodynamic Shapes Using Deep-Learning-Based Geometric Compression]]
