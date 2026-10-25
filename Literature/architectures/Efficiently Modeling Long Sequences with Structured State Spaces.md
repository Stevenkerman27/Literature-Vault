---
title: "Efficiently Modeling Long Sequences with Structured State Spaces"
authors: [Albert Gu, Karan Goel, Christopher Ré]
year: 2022
venue: "ICLR 2022"
doi: null
arxiv: "2111.00396"
url: "https://arxiv.org/abs/2111.00396"
source: "未提供本地 PDF；原文 https://arxiv.org/pdf/2111.00396"
tags: [paper, state-space-model, S4, long-sequence, architecture]
status: read
updated: 2026-10-25
---

> [!abstract] 摘要译述
> 长序列中的依赖关系给循环网络、卷积网络和 Transformer 带来不同的效率瓶颈。本文提出结构化状态空间序列模型 S4：在连续状态空间模型的基础上，通过 Normal Plus Low-Rank（NPLR，正规矩阵加低秩修正）参数化，将计算转换为高效而数值稳定的 Cauchy 核计算，同时保留长程记忆能力。作者在 Long Range Arena（LRA）、图像与语音序列建模上展示其优越性，构成后续 Selective SSM 与 Mamba 的重要数学前身。

# 核心定位

**S4 的突破是让固定参数的连续状态空间系统既能作为卷积并行训练，又能作为状态递归推理。** 它并未做输入内容驱动的选择性遗忘：这正是 [[Literature/architectures/Mamba Linear-Time Sequence Modeling with Selective State Spaces|Mamba]] 后来改进的地方。

## 1. 方程：连续 ODE、离散递推和卷积同一算子

用输入 \(u(t)\)、隐藏状态 \(x(t)\) 和输出 \(y(t)\) 写连续模型：

\[
x'(t)=Ax(t)+Bu(t), \quad y(t)=Cx(t)+Du(t).
\]

经时间步长 \(\Delta\) 离散，得到：

\[
x_k=\bar A x_{k-1}+\bar B u_k,\qquad
y_k=Cx_k+Du_k .
\]

忽略直接项，可将循环展开成卷积核：

\[
\bar K=(C\bar B,C\bar A\bar B,\dots,C\bar A^{L-1}\bar B),
\qquad y=\bar K*u .
\]

只要参数 \(A,B,C,\Delta\) 对时间固定（Linear Time-Invariant，LTI），整段序列可用快速卷积并行处理；自回归推理可缓存固定大小状态。这是与必须维护不断增长 KV cache 的标准 causal softmax attention 的不同之处。

## 2. 为什么 S4 需要 NPLR？

HiPPO（High-order Polynomial Projection Operators）给出用于压缩连续历史的结构化状态矩阵，但直接计算长卷积核代价高且数值不稳定。S4 使用：

\[
A=V(\Lambda-PQ^*)V^*
\]

即将 \(A\) 写成可对角化的正规部分与低秩修正，并通过 Woodbury 恒等式与 Cauchy kernel 计算频域传递函数。方法关键是**避免直接为每个时间步计算高次矩阵幂**，又保留 HiPPO 初始化的长程记忆归纳偏置。

NPLR、Cauchy 核和稳定离散化是论文真实创新，不是简单地“把 Attention 换成一维卷积”。

## 3. 模型和实验配置

- **LRA**：包括长文本、图像等长程依赖序列及约 16K 长度的 Path-X；与 RNN、Transformer 和高效序列模型比较准确率与算力。
- **Sequential CIFAR-10**：将 32×32 图像展平为长度 1024 的序列，且不借助图像增强或辅助损失，检验长依赖分类。
- **WikiText-103、语音和图像生成**：检验通用序列建模，而不是只在单一合成任务上验证。
- **计算对照**：比较 S4 不同结构化参数化的卷积核生成、递归计算和长序列效率。

作者报告 Sequential CIFAR-10 约 **91%** 准确率，LRA 各任务达到当时领先水平并解决此前模型难以通过的 Path-X；生成速度在论文给定比较条件下约 **60 倍**。这些不是通用所有硬件的速度指标。

## 4. 边界与启发

- LTI 意味着卷积核不随输入 token 内容改变；对需要 selective copying、上下文触发的条件路由，其能力有限。
- HiPPO 初始化和核算法复杂，推理与训练需分别采用优化实现；后续对角 SSM（S4D）与 Mamba 优化了不同的维度。
- **PDE 时间演化的可迁移性只是项目假说**：动力学隐藏状态与连续流体 ODE 的形式相似，不代表已有证明 S4 能解决三维稳态气动场。空间离散点的排列、几何边界和守恒是额外挑战。

## 相关文献

- **直接技术后继**：[[Literature/architectures/Mamba Linear-Time Sequence Modeling with Selective State Spaces|Mamba]] 引入输入依赖的 \(\Delta,B,C\)，牺牲固定卷积核而获得选择性。
- **理论桥梁**：[[Literature/architectures/Transformers are SSMs Generalized Models and Efficient Algorithms Through Structured State Space Duality|Mamba-2 / SSD]] 将特殊 SSM 写成结构化 masked attention。
- **机制对照**：[[Literature/architectures/Hyena Hierarchy Towards Larger Convolutional Language Models|Hyena]] 同样利用长卷积，但额外叠加数据依赖门控。

原文：https://arxiv.org/html/2111.00396
