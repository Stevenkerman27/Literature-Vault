---
title: "Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs"
type: literature-note
status: abstract-reviewed
created: 2026-11-11
updated: 2026-11-11
tags: [neural-operator, universal-approximation, function-spaces, discretization-invariance, PDE]
---

# Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs

- **Authors:** Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar
- **Venue:** JMLR 24(89), 2023
- **Source:** https://jmlr.org/papers/v24/21-1524.html
- **Evidence:** journal abstract checked; full 97-page paper is available but theorem details have not yet been independently audited.

## 中文摘要

本文将神经网络从有限维向量映射推广到函数空间之间的映射，提出 Neural Operator。其结构由线性积分算子和非线性激活组合而成，并证明在论文条件下可以逼近连续非线性算子。论文还强调离散化不变性：同一组参数可以作用于不同离散分辨率的函数表示。

## 数学对象

传统网络学习 (f:mathbb R^n	omathbb R^m)。神经算子学习的是函数到函数的映射：
[
mathcal G:mathcal X	omathcal Y,qquad u(cdot)mapsto v(cdot).
]
这对 PDE 解算器很自然，因为输入几何、边界条件和材料场经过离散采样，输出仍然是空间场。

## 可支持的结论

- Neural Operator 框架具有连续算子通用逼近定理。
- 论文提出图、多极图、低秩和 Fourier 等参数化。
- 参数共享的算子形式旨在跨不同离散分辨率工作。

## 局限与不可推出结论

- 算子通用逼近不是离散拓扑生成的通用逼近。
- 不同网格上的离散化不变性不等于不同飞机布局之间的分布泛化。
- 定理不保证有限训练样本下的精度、训练收敛、稳定性或真实 CFD 精度。
- 对 GRASS 而言，翼数变化是离散结构变化，不能只用“不同网格分辨率”解释。

## 对 GRASS 的启发

应把两个学习任务分开：

1. **生成器：** 条件与随机变量到离散组件拓扑及连续几何的映射。
2. **气动代理：** 已给定几何与工况到局部压力/载荷场的函数映射。

第二个任务适合神经算子理论；第一个任务需要图/集合/序列生成理论。二者可通过可微几何接口连接，但不能共享一个未经证明的“通用逼近”结论。

## 与 Transolver 的联系

《Does Transolver Really Need a Transformer?》把 slice-deslice 看作低秩非局部积分算子，并使用 averaging neural operator 的理论解释其通用性。该结论针对连续场算子，不是飞机拓扑或条件生成分布。对 GRASS 的直接价值是分析代理中的全局信息交换机制，而非直接证明生成器能力。

## 相关文献

- [[Literature/Topics/跨布局飞行器生成与新型网络架构|跨布局飞行器生成专题]]
- [[Literature/architectures/Does Transolver really need a Transformer|Does Transolver Really Need a Transformer]]
