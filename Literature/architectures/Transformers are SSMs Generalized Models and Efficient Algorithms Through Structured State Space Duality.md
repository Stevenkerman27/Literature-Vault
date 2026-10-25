---
title: "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"
authors: [Tri Dao, Albert Gu]
year: 2024
venue: "ICML 2024, PMLR 235:10041–10071"
doi: null
arxiv: "2405.21060"
url: "https://arxiv.org/abs/2405.21060"
code: "https://github.com/state-spaces/mamba"
source: "未提供本地 PDF；原文 https://arxiv.org/pdf/2405.21060"
tags: [paper, Mamba-2, state-space-duality, structured-attention, semiseparable, sequence-architecture]
status: read
updated: 2026-10-25
---

> [!abstract] 摘要译述
> 本文证明多类结构化状态空间模型（Structured State Space Model，SSM）与一类结构化 Masked Attention 可通过半可分矩阵（Semiseparable Matrix）统一理解，称为 State Space Duality（SSD，状态空间对偶）。该对偶不仅是理论重写，还给出面向 GPU 矩阵乘法单元的高效分块算法。以此为基础提出 Mamba-2，论文报告其核心 SSD 层比 Mamba-1 的优化扫描快 2–8 倍，并保持具有竞争力的语言建模性能。

# 一句话解释

**Mamba-2 建立了“某些线性注意力 = 某些结构化 SSM”的严格数学桥梁，而非证明全部 Softmax Transformer 都等价于 Mamba。** 它用略受约束的状态转移结构换取 GPU 张量核心（Tensor Core）友好的批量矩阵乘法。

## 1. 从递推展开到结构化 Attention

选择性 SSM 的递归为：

\[
h_t=A_th_{t-1}+B_tx_t,\quad y_t=C_t^\top h_t .
\]

对过去时刻 \(s\le t\) 展开：

\[
y_t=\sum_{s\le t}
C_t^\top (A_tA_{t-1}\cdots A_{s+1})B_sx_s .
\]

从矩阵角度，系数构成一个因果下三角矩阵 \(M\)。若 \(A_t=a_tI\)（每个 head 内标量乘单位阵），则：

\[
M_{ts}
=\underbrace{\prod_{j=s+1}^{t}a_j}_{L_{ts}}
\,C_t^\top B_s .
\]

因此：

\[
Y=(L\odot CB^\top)X .
\]

其形式与掩码核注意力类似：Query、Key 向量决定特征相似度，\(L\) 则承载因果和时间衰减。这类 \(L\) 具有 1-semiseparable 结构（远离对角线的非重叠块秩受限）。

> [!important] 严格边界
> SSD 适用于一类**有限维特征映射的结构化 Masked Attention**，而不是普通 Softmax Attention 的任意权重矩阵。不能把题目 “Transformers are SSMs” 解读为“标准 Transformer 结构和 Mamba 在所有情况下完全相同”。

## 2. SSD 算法为什么快？

作者将序列切成多个 chunk。对角块内进行矩阵乘法，处理短范围交互；块间低秩耦合则通过压缩 state 传播。其计算复用 GPU 对 GEMM（通用矩阵乘法）的高度优化，避免逐 token recurrent scan 难以填满 Tensor Cores 的问题。

论文的 Theorem 6.1 在 head width 与 state size 都为 \(N\) 的设置下，给出训练 FLOPs \(O(TN^2)\)、单步递推 FLOPs \(O(N)\)、缓存状态 \(O(N^2)\) 的量级表述（依原文约定分析），强调对**序列长度 \(T\)** 线性，而不是对所有宽度也线性。

## 3. 与 Mamba-1 的结构差别

Mamba-1 的状态转移可以按通道拥有更一般的对角矩阵；Mamba-2 的 SSD 限制每个 head 的状态转移为 \(a_t I\)，以牺牲部分参数化自由度交换更易矩阵化的训练算法。与此同时 Mamba-2 支持较大状态扩展，在有限计算预算内有利于状态容量。

## 4. 论文实验

- **算法 microbenchmark**：SSD 与 Mamba-1 的 fused selective scan、FlashAttention-2 对照，改变序列长度与状态扩展维数。
- **语言模型**：比较 Mamba-2、Mamba-1 与 Attention 基线的规模、困惑度和效率。
- **核心速度数字**：SSD 内核在较大状态规模下比 Mamba-1 fused scan 快 **2–8×**；相对 FlashAttention-2 在作者基准的 2K 序列长度附近出现交叉，16K 时报告约 6× 层级内核速度。这不是整个大模型端到端训练/推理时间的统一倍数。

## 5. 局限与启发

- 限制 \(A_t=a_tI\) 不如任意对角状态转移通用；SSD 也不是所有注意力函数的完全等价实现。
- 仍有固定维状态的信息压缩瓶颈。对于要求精确随机访问过去所有局部细节的任务，不能仅凭线性复杂度替代 Attention。
- 针对网格表面预测必须额外证明序列排列、积分测度和坐标不变性；语言上的 1D 因果递推不自动具备无序点集的对称性。

## 相关研究

- **直接前作**：[[Literature/architectures/Mamba Linear-Time Sequence Modeling with Selective State Spaces|Mamba]]。
- **理论旁支**：[[Literature/architectures/Gated Linear Attention Transformers with Hardware-Efficient Training|GLA]] 与 [[Literature/architectures/Retentive Network A Successor to Transformer for Large Language Models|RetNet]] 使用可递推矩阵记忆与衰减。
- **直接后继**：[[Literature/architectures/Mamba-3 Improved Sequence Modeling using State Space Principles|Mamba-3]]。
- **跨任务比较**：[[Literature/transformer/Transolver Is a Linear Transformer Revisiting Physics-Attention Through the Lens of Linear Attention|LinearNO]] 为物理网格的低秩跨点混合，不应被当作 SSD 特例。

官方论文：https://proceedings.mlr.press/v235/dao24a.html
