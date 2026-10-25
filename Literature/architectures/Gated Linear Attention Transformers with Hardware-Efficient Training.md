---
title: "Gated Linear Attention Transformers with Hardware-Efficient Training"
authors: [Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim]
year: 2024
venue: "ICML 2024, PMLR 235:56501–56523"
doi: null
arxiv: "2312.06635"
url: "https://arxiv.org/abs/2312.06635"
source: "未提供本地 PDF；原文 https://arxiv.org/pdf/2312.06635"
tags: [paper, linear-attention, gated-linear-attention, GLA, recurrent, hardware-aware]
status: read
updated: 2026-10-25
---

> [!abstract] 摘要译述
> 线性注意力可以在训练时利用矩阵计算，在推理时以矩阵值隐藏状态递归执行，但传统实现可能比高度优化的 Softmax Attention 更慢，且表达能力有限。本文先给出 FlashLinearAttention 的 GPU 数据移动优化算法，再设计 Gated Linear Attention（GLA），通过随输入改变的门控使记忆状态能够选择性遗忘，并提出分块并行训练方法。在中等规模语言模型中，GLA 取得与 LLaMA 风格 Transformer、RetNet、Mamba 等竞争性表现及较好的长度泛化。

# 核心思想

**GLA 是“线性 Attention + 输入依赖遗忘矩阵”，而不是标准 Softmax Attention 的精确数学等价实现。** 它通过矩阵状态承担快速权重（fast weights）的角色，又针对硬件瓶颈专门设计 chunkwise kernel。

## 1. 从线性注意力到递归矩阵记忆

经典核化线性注意力（忽略分母归一化的简化写法）：

\[
S_t=S_{t-1}+k_tv_t^\top,\qquad o_t=q_t^\top S_t.
\]

如果永远只加新 Key–Value 外积，旧信息难以忘记，长序列可能发生干扰与污染。

GLA 引入逐通道数据依赖门控：

\[
S_t=G_t\odot S_{t-1}+k_tv_t^\top,\qquad
o_t=q_t^\top S_t,
\]

其中 \(G_t\) 包含由当前输入计算、用于选择性保留或衰减各状态维度的因子。**这个符号是核心机制的抽象形式**；实际论文使用特定可分解门控参数化、归一化和硬件适配。

## 2. 训练和推理的三种形式

- **Recurrent form**：不断更新固定大小矩阵状态，推理时免去随上下文增长的完整 KV cache。
- **Parallel form**：将递归展开为矩阵乘法，但完整掩码和数值稳定处理可能开销较大。
- **Chunkwise form**：块内使用高效矩阵乘法，块间递归积累历史状态；作者以 FlashLinearAttention / Triton 内核减少显存搬运，兼顾并行性与数值稳定。

作者特别强调：理论 FLOPs 少并不意味着真实 GPU 一定快，核函数、IO、warp occupancy、张量核心利用率共同决定运行时间。

## 3. 实验设置

- **340M 参数**：基于中等规模语言模型对照约 **15B tokens**；
- **1.3B 参数**：对照约 **100B tokens**；
- 比较 LLaMA 架构 Transformer、RetNet、Mamba 等，考察验证 loss、回忆/推理任务、训练吞吐和长度外推。
- 论文报告 GLA 在 **2K** 上训练，可较稳定推广到超过 **20K** 的序列而困惑度没有显著恶化（限定于其实验）；某些更长序列上 throughput 比 Mamba 更高。

这些是该篇同一训练协议中的实验结果，不等于在所有语言任务均严格优于 Softmax Transformer。

## 4. 限制与工程含义

- 矩阵状态只保留累计统计，不会像完整原始 Key–Value 列表那样支持任意精确回访；强检索任务仍可能受固定容量限制。
- 内容门控是逐 token 的选择性，不是对 CFD 网格的几何归纳偏置。
- 论文比较的是语言序列，不提供升力/阻力或不规则点云上的证据。
- 对 CFD 时间步流场，GLA 是潜在的时间记忆架构；对空间网格需额外处理无序性、积分权重和边界条件。

## 5. 与其他论文的联系

- [[Literature/architectures/Retentive Network A Successor to Transformer for Large Language Models|RetNet]]：从位置尺度衰减向数据依赖门控扩展，是机制层面的延展，不一定是代码继承。
- [[Literature/architectures/Gated Delta Networks Improving Mamba2 with Delta Rule|Gated DeltaNet]]：增加目标键的误差修正式写入，解决单纯外积写入的精度限制。
- [[Literature/architectures/Transformers are SSMs Generalized Models and Efficient Algorithms Through Structured State Space Duality|Mamba-2 / SSD]]：矩阵状态与结构化注意力的另一种实现。
- [[Literature/transformer/Mixture-of-Top-k Attention Efficient Attention via Scalable Fast Weights|MiTA]]：GLA 将上下文压成固定维状态；MiTA 对部分原始 KV 进行精确访问，并保留压缩共享分支。

官方论文：https://proceedings.mlr.press/v235/yang24ab.html
