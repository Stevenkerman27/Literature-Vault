---
title: "Universal Approximation of Functions on Sets"
authors: ["Edward Wagstaff", "Fabian B. Fuchs", "Martin Engelcke", "Michael A. Osborne", "Ingmar Posner"]
year: 2022
venue: "Journal of Machine Learning Research, 23(151):1–56"
doi: ""
tags: [deep-sets, set-functions, permutation-invariance, universal-approximation, representation-theory]
status: read
source: "未提供本地 PDF；https://arxiv.org/abs/2107.01959"
url: "https://jmlr.org/papers/v23/21-0730.html"
arxiv: "2107.01959"
---

# 核心摘要

> [!abstract] 原文摘要翻译
> 对集合函数建模，也就是对输入排列保持不变的函数，一直是机器学习中的重要问题。Deep Sets 是一种常用方法，已知它可以通用逼近连续集合函数。本文进一步分析这一结论成立所需的潜在空间维数：只有潜在维度足够高时，通用表示/逼近保证才成立；如果维度再低一维，就存在某些分段仿射函数，使 Deep Sets 的最坏情况误差不优于常数基线。作者还把 Deep Sets 解释为 Janossy pooling 范式中最受限制的一种形式，并讨论其对其他集合学习架构的意义。

## 1. 问题与动机

集合没有天然顺序。飞机中的翼面、部件或组件关系若只因数据文件里的排列不同就产生不同预测，模型学习到的可能是人为编号，而非物理关系。集合函数要求输入排列不变；集合到集合的映射则通常要求排列等变（输入如何重排，输出对应地重排）。

Deep Sets 的典型形式为：

\[
f(X)=\rho\left(\sum_{x\in X}\phi(x)\right).
\]

共享编码器 \(\phi\) 对每个元素独立编码，再通过求和等对称聚合得到全局表示，最后由 \(\rho\) 输出。求和保证排列不变，但单纯的逐元素编码与池化并没有显式地建立元素之间的成对关系。

## 2. 方法与理论结果

论文将 Deep Sets、self-attention 等方法放入更一般的 \(k\)-ary Janossy pooling 框架中。完整 Janossy pooling 对输入元素的排列分别应用同一函数，再对结果聚合；它具有很强的表达能力，但枚举排列的成本很高。Deep Sets 可看作仅使用一元元素编码再做对称池化的受限形式。

本文最重要的结论不是“sum pooling 永远万能”，而是给出潜在维数的必要/充分边界。对含 \(M\) 个标量元素的连续集合函数，连续 Deep Sets 的潜在空间维数至少需要达到 \(M\) 才能保证通用表示/逼近；论文也给出充分条件 \(M+1\)，并说明低于必要边界时存在无法被连续 sum-decomposition 逼近的函数。维度刚好达到下界只是理论保证，不意味着训练容易或样本效率好。

## 3. 实验与证据边界

这是以数学分析为主的论文，核心证据是表示与逼近定理及其构造性/拓扑论证，不是某个航空数据集上的性能榜单。论文明确指出：即使维数满足定理，所需 \(\phi,\rho\) 也未必容易学到；提高 latent 维度在实践中可能更有效。

## 4. 局限

- 标量集合的精确维数结论不能不加条件地直接套到多维几何向量集合。
- 通用逼近不等于有限样本可学习、优化可收敛或 OOD 泛化。
- 对跨部件复杂关系，纯 sum pooling 没有显式 pairwise relational mechanism；attention 或图消息传递可提供更合适的归纳偏置，但其必要性要由目标函数结构决定。

## 5. 对 GRASS 的启发（项目推论，非论文结论）

GRASS 的多翼面不应仅仅“把每个翼面编码后求和”。翼面间的干扰、相对位置、翼身接头、局部碰撞等依赖成对或高阶关系。更合理的对照是：

1. **Set baseline**：共享 wing encoder + sum/mean pooling + global decoder，作为排列不变基线。
2. **Relational set**：翼面 token 先做 pairwise interaction（attention 或 edge message passing），再做全局聚合。
3. **Hierarchical set**：每个 wing 内部是有序截面序列，wing 之间是无序集合；不可把两层的顺序语义混为一谈。

建议做合法翼面顺序随机置换测试：同一飞机的部件排列变化后，整机物理输出应不变，逐翼输出应按同样置换变化。单纯对训练数据做随机换序只能作为增强，不是架构等变性的数学保证。

## 6. 关联

- [[Literature/Topics/跨布局飞行器生成与新型网络架构]]
- [[Literature/architectures/Are Transformers Universal Approximators of Sequence-to-Sequence Functions]]
- [[Literature/GraphRNN Generating Realistic Graphs with Deep Auto-regressive Models]]
- [[Literature/Topics/神经网络架构与高效序列建模]]
