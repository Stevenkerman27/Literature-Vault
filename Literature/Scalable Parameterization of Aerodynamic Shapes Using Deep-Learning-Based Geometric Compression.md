---
title: "Scalable Parameterization of Aerodynamic Shapes Using Deep-Learning-Based Geometric Compression"
authors: [Tom Bamford, Andy Keane, David J. J. Toal]
year: 2026
venue: "AIAA Journal"
doi: "10.2514/1.J066018"
url: "https://doi.org/10.2514/1.J066018"
source: "未提供本地 PDF；依据公开摘要核对：https://www.researchgate.net/publication/403917926_Scalable_Parameterization_of_Aerodynamic_Shapes_Using_Deep-Learning-Based_Geometric_Compression"
status: abstract-reviewed
tags: [literature, geometric-compression, implicit-neural-representation, meta-learning, aerodynamic-shape-parameterization]
---

> [!abstract] 摘要译述
> 本文针对二维和三维气动形状的可扩展参数化，使用深度学习压缩单个几何，并通过元学习拟合相应几何分布。其表示从离散网格转向连续神经隐式几何场，使设计变量数量可减少一个数量级或更多，同时适应不同几何场分辨率和空间维度。

# 方法与研究问题

摘要描述的主要思路是：
- 将几何从离散网格描述转为连续神经隐式场；
- 使用深度学习压缩单个几何对象；
- 通过 meta-learning 拟合几何对象对应的分布；
- 在二维翼型和三维机翼参数化上，与其他深度学习参数化及离散场表示比较。

它主要研究“如何以紧凑、可扩展的变量表示连续几何”，而非直接预测飞机部件数量或装配拓扑。

# 与 GRASS 的关系

对 GRASS 的截面几何表示有参考价值：连续隐式表示可能降低固定 CST 参数族的表达限制，并允许不同空间分辨率下使用同一几何表示。

但它不能直接替代 GRASS 的部件级离散生成逻辑。连续几何场不会自动提供翼面数量、部件语义、翼根连接、翼盒结构或机身装配约束。若用于 GRASS，应将其定位为几何解码器候选，而非完整整机生成框架。

# 局限与证据边界

当前仅核对公开摘要，未取得全文。具体隐式场网络、元学习目标、训练集规模、基准和定量表格尚未核实；不推断实现细节。

# 关联

- [[Literature/Topics/几何表示与性能条件生成]]
- [[Literature/Topics/跨布局飞行器生成与新型网络架构]]
- [[Literature/Automated Parameterization for Aerodynamic Shape Optimization via Deep Geometric Learning]]
