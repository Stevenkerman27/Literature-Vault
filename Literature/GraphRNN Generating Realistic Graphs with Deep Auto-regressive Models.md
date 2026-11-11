---
title: "GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Models"
authors: ["Jiaxuan You", "Rex Ying", "Xiang Ren", "William Hamilton", "Jure Leskovec"]
year: 2018
venue: "Proceedings of the 35th International Conference on Machine Learning, PMLR 80:5708–5717"
doi: ""
tags: [graph-generation, autoregressive-model, variable-size-graphs, structure-generation]
status: abstract-reviewed
source: "未提供本地 PDF；https://proceedings.mlr.press/v80/you18a.html"
url: "https://proceedings.mlr.press/v80/you18a.html"
---

# 核心摘要

> [!abstract] 原文摘要翻译
> 图生成需要处理图结构的非唯一性、高维性，以及边之间复杂的非局部依赖。GraphRNN 提出深度自回归模型，将图生成拆分为节点生成与边生成两个序列过程：每一步都以已经生成的图结构为条件。作者报告该方法能学习多样化图分布，并在其基准中扩展到比此前深度模型大约 50 倍的图。

## 1. 方法（摘要级）

GraphRNN 将一个图的生成因子分解为逐步生成节点及其连接关系的条件概率。它把图的非固定大小和非局部依赖转化为自回归条件建模问题。当前只核对了公开摘要与 PMLR 元数据，尚未核验 BFS 排序、edge-level RNN、node-level RNN 等全文实现细节，故不在此补写。

## 2. 实验与局限

摘要报告在图生成基准上优于比较基线，并能扩展到更大图；具体数据集、指标定义、模型参数与数值尚待全文核验。图的节点排序会影响自回归序列，因而结构生成需要处理等价排列/规范排序问题。

## 3. 对 GRASS 的启发（研究假设）

可把飞机部件装配看成带类型节点与关系边的图，但 GRASS 当前契约把所有翼面根部直接连机身，并没有任意 parent pointer。GraphRNN 适合作为“拓扑可变时如何因子分解生成”的比较基线，不意味着应立即把现有简单 ADD_WING/STOP 替换成完整图生成器。

## 4. 关联

- [[Literature/architectures/Universal Approximation of Functions on Sets]]
- [[Literature/Topics/跨布局飞行器生成与新型网络架构]]
- [[Literature/ShapeAssembly learning to generate programs for 3D shape structure synthesis]]
