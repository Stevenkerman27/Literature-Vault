---
title: "ShapeAssembly: Learning to Generate Programs for 3D Shape Structure Synthesis"
type: literature-note
status: abstract-reviewed
created: 2026-11-11
updated: 2026-11-11
tags: [program-generation, hierarchical-generation, assembly, 3d-shape, latent-variable-model]
---

# ShapeAssembly: Learning to Generate Programs for 3D Shape Structure Synthesis

- **Authors:** R. Jones et al.
- **Venue:** ACM Transactions on Graphics 39(6), 2020
- **Source:** https://doi.org/10.1145/3414685.3417812
- **Evidence:** publisher abstract checked; full paper and supplementary material not yet audited.

## 中文摘要

ShapeAssembly 将三维结构表示为领域专用的装配程序。程序通过声明部件代理并描述部件之间的连接/对称关系来构造形状；程序结构包含离散操作，连续自由变量控制具体几何。作者从 PartNet 数据中提取程序，并训练层级序列 VAE 生成新的装配程序。论文报告生成结构具有较好的合理性和物理有效性，并可通过可微程序解释器拟合点云。

## 方法核心

与直接生成点云或网格不同，模型生成的是“如何组装形状”的程序。可以将其抽象为：
[
p(	au,	hetamid c)=p(	aumid c),p(	hetamid	au,c),
]
其中 (	au) 是离散程序/拓扑，(	heta) 是程序中的连续参数。该分解是对论文方法的概括表达，不应视为论文逐字给出的概率公式。

## 可支持的结论

- 离散结构程序与连续几何参数可以共同描述复杂形状族。
- 层级序列 VAE 可以学习生成结构程序。
- 可微程序解释器使结构程序能够与几何拟合任务连接。

## 局限与不可推出结论

- ShapeAssembly 的 cuboid proxy 与其领域程序不等于飞机气动几何。
- “物理有效”是论文任务与数据集定义下的结论，不等于满足航空器结构、气动、稳定性或适航要求。
- 不能从该论文推出 GRASS 应采用 VAE，更不能推出其 latent 空间具备跨翼数外推保证。

## 对 GRASS 的启发

它与飞机生成的对应关系很直接：飞机可以先生成装配程序/组件图，再生成连续部件参数。建议把离散层限定在少量、可审计的布局操作（组件类型、翼数、连接、镜像、角色），连续层保留 CST、机身参数、位置和尺度。

这为 GRASS 提供一个可解释的数学分层：拓扑错误可以在几何解码前拒绝；连续参数则在固定拓扑分支内优化。每个拓扑分支可单独讨论连续函数逼近，不必声称跨离散拓扑边界存在一个连续映射。

## 相关文献

- [[Literature/Topics/跨布局飞行器生成与新型网络架构|跨布局飞行器生成专题]]
- [[Literature/GraphRNN Generating Realistic Graphs with Deep Auto-regressive Models|GraphRNN]]
- [[Literature/generative/PolyGen An Autoregressive Generative Model of 3D Meshes|PolyGen]]
