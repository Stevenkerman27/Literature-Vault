---
title: "PolyGen: An Autoregressive Generative Model of 3D Meshes"
type: literature-note
status: abstract-reviewed
created: 2026-11-11
updated: 2026-11-11
tags: [autoregressive-generation, 3d-mesh, vertices, faces, transformer]
---

# PolyGen: An Autoregressive Generative Model of 3D Meshes

- **Authors:** Charlie Nash, Yaroslav Ganin, S. M. Ali Eslami, Peter Battaglia
- **Venue:** ICML 2020, PMLR 119:7220–7229
- **Source:** https://proceedings.mlr.press/v119/nash20a.html
- **Evidence:** publisher abstract checked; detailed tokenization, factorization and experiment tables not yet audited.

## 中文摘要

PolyGen 直接对三维多边形网格建模，使用自回归 Transformer 依次生成顶点和面，而不是先生成体素或隐式形状。模型支持类别、体素、图像等条件，并通过概率生成表达不确定性。

## 方法核心

将网格写成离散序列，按顺序生成顶点，再生成面连接。概率分解让每一步都依赖之前已生成的几何/拓扑信息。其优势是直接建模 mesh 的连接结构，而不仅是连续坐标。

## 可支持的结论

- 顶点和面连接可以作为自回归生成对象。
- 概率序列建模能够表达多种可能的三维网格。
- 论文报告了可用网格生成和条件重建实验。

## 局限与不可推出结论

- 摘要不足以说明顶点排序、面序列化和网格合法性的全部处理细节。
- 自回归生成不自动保证水密、流形、CAD 可编辑或工程约束。
- 逐顶点生成可能与 GRASS 的参数化翼面/CST 几何表示不匹配。
- 不应仅因其使用 Transformer 就将其视为跨布局生成器的最佳架构。

## 对 GRASS 的启发

PolyGen 的价值在于“拓扑本身可以是生成变量”。但 GRASS 的目标更接近受限组件程序，而不是任意三角网格：翼数、翼面角色、连接关系、每翼截面数应先被显式表示，连续几何再由物理参数化解码。

建议把 PolyGen 作为 mesh-level 生成的边界对照，而不是第一候选实现。

## 相关文献

- [[Literature/Topics/跨布局飞行器生成与新型网络架构|跨布局飞行器生成专题]]
- [[Literature/GraphRNN Generating Realistic Graphs with Deep Auto-regressive Models|GraphRNN]]
- [[Literature/ShapeAssembly learning to generate programs for 3D shape structure synthesis|ShapeAssembly]]
