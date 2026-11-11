---
title: "Grounding Continuous Representations in Geometry: Equivariant Neural Fields"
type: literature-note
status: abstract-reviewed
created: 2026-11-11
updated: 2026-11-11
tags: [equivariance, neural-fields, geometry, conditional-generation, latent-representation]
---

# Grounding Continuous Representations in Geometry: Equivariant Neural Fields

- **Authors:** David Wessels, David Knigge, Riccardo Valperga, Samuele Papa, Sharvaree Vadgama, Efstratios Gavves, Erik Bekkers
- **Venue:** ICLR 2025
- **Source:** https://proceedings.iclr.cc/paper_files/paper/2025/hash/95a704bd2fdf8ef8242b4adcc7ce3c93-Abstract-Conference.html
- **arXiv:** https://arxiv.org/abs/2406.05753
- **Evidence:** publisher abstract and arXiv abstract checked; full mathematical construction not yet audited.

## 中文摘要

本文提出 Equivariant Neural Fields（ENFs），针对条件神经场中 latent 缺少显式几何结构的问题，引入几何感知的 cross-attention，使潜变量成为带空间位置和方向信息的特征点集。其目标是让潜表示与连续场之间具有一致的变换规律：几何场发生变换时，latent 也按相应规律变换，反之亦然。作者将这种结构用于分类、分割、预测、重建和生成等任务。

## 方法核心

传统 conditional neural field 通常用一个全局 latent 向共享 neural field 解码器提供条件。ENF 则让 latent 本身带有几何位置/方向，使用 geometry-informed cross-attention 将 latent 点集映射到连续场。其核心不是单纯增加注意力层，而是明确规定 latent 与空间变换之间的关系。

## 可支持的结论

- 纯全局 latent 可能缺少精细几何推理所需的局部性与方向信息。
- 几何等变 latent 可以在表示中保留局部几何模式。
- 作者报告了跨多个任务的实验改进，包括生成建模。

## 局限与不可推出结论

- 摘要不提供完整群作用、特征表示类型、定理假设或具体误差界。
- 等变性只保证对指定变换的结构一致性，不自动保证气动、结构或装配合法。
- 飞机设计包含带语义的部件角色、尺度和边界条件；不能把任意 SE(3) 等变直接套用到带固定重力/来流/机身轴向的任务。

## 对 GRASS 的启发（研究假设）

对于每个翼面，考虑用“沿展向坐标排列的几何 latent field”而不是只有一个翼面全局向量。组件级 latent 表示布局/角色，截面级 latent 表示局部形状；跨翼交互在组件 latent 上进行，局部几何由坐标条件解码。

值得测试的对称性不是“所有空间旋转都不改变问题”，而是项目真实成立的变换，例如半模镜像约束、坐标重参数化，以及同一翼面采样站位密度变化。必须先写明物理输入和输出在变换下如何变化，再设计等变层。

## 与 GRASS 的关系

当前翼面由有限截面参数化，随后重采样到固定物理网格。ENF 提供的研究方向是让潜在几何表达与实际空间坐标绑定，从而检验跨不同截面数、采样分辨率和翼面局部形状的泛化。它不要求立刻替换现有 CST/截面表示。

## 相关文献

- [[Literature/Topics/跨布局飞行器生成与新型网络架构|跨布局飞行器生成专题]]
- [[Literature/architectures/Neural Operator Learning Maps Between Function Spaces|Neural Operator]]
