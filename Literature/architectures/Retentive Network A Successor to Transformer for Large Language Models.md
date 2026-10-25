---
title: "Retentive Network: A Successor to Transformer for Large Language Models"
authors: [Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, Furu Wei]
year: 2023
venue: "arXiv preprint"
doi: null
arxiv: "2307.08621"
url: "https://arxiv.org/abs/2307.08621"
source: "未提供本地 PDF；原文 https://arxiv.org/pdf/2307.08621"
tags: [paper, attention, retention, recurrent, long-sequence, linear-attention]
status: read
updated: 2026-10-25
---

> [!abstract] 摘要译述
> 为解决 Transformer 训练并行性与推理 KV cache 成本之间的张力，RetNet 提出 Retention 操作。它能够以三种不同计算模式表示相同的衰减式序列关系：训练时的并行表示、增量解码的递归表示，以及超长序列的分块递归表示。作者在语言建模中比较了规模规律、zero/few-shot 性能、训练吞吐和解码效率，提出一条不完全依赖标准 Softmax Attention 的语言架构路线。

# 核心观点

**RetNet 的创新不是“去掉注意力即可递归”，而是从具有衰减的 Key–Value 聚合中构造三种计算等价的实现形态。** 标准 Attention 存每个过去 token 的 KV；Retention 将过去信息压缩成固定维矩阵状态。

## 1. 并行式与递归式

论文的 Retention 可简化为具有因果衰减核的矩阵乘法：

\[
Y=(QK^\top\odot D)V,\quad
D_{ij}=\begin{cases}\gamma^{i-j}&i\ge j\\0&i<j.\end{cases}
\]

其中 \(Q,K\) 还包括与相对位置有关的复数旋转相位（论文中以 \(\Theta\) 表示）。省略这些项的式子只说明计算结构。

同一操作的递归形式：

\[
S_t=\gamma S_{t-1}+k_t^\top v_t,\qquad
o_t=q_t S_t .
\]

这与单纯向历史位置分配一组 Softmax 权重不同：它对 Key–Value 外积做指数衰减累积。头间可采用不同 \(\gamma\)，形成多尺度 retention。

## 2. 三种计算范式

- **Parallel**：训练时构造块内或完整衰减掩码，以矩阵乘法一次性处理 token。
- **Recurrent**：推理时将 \(S_{t-1}\) 作为固定维状态缓存，每个新 token 更新状态，不保留无限增长的原始 KV cache。
- **Chunkwise recurrent**：块内并行，块间递归传播状态，适合长序列的训练与分块推理。

> [!important] 复杂度边界
> 论文的 **O(1) inference** 指的是每个新 token 的状态缓存不随已处理长度增长，并非单个 token 的实际计算 FLOPs 为常数，与 head width、head 数和 batch 都无关；训练阶段完整并行形式也可能需要二次时间/空间，分块形式才用于规避大规模完整矩阵。

## 3. 架构与实验

作者在 RetNet 中使用 Gated Multi-Scale Retention，配合前馈网络、归一化和残差。不同 retention head 使用不同的记忆衰减尺度，使模型同时表达短程与长程依赖。

- 数据和任务：自回归语言建模、不同模型规模的 loss scaling、zero-shot 与 few-shot 下游任务。
- 资源指标：GPU 显存、推理吞吐、不同输入长度的延迟；与使用 KV cache 的 Transformer 对照。
- 论文报告在 7B 参数、8K 输入上下文的特定实验中，解码吞吐约 **8.4×**、显存减少约 **70%**。不能把这组数字外推至所有批量、框架和硬件。

## 4. 局限及必要区分

- RetNet 属于因果**序列**架构，Retention 的时间衰减与位置编码依赖 token 顺序，不能直接对任意重排的点云保持等变。
- 固定矩阵状态会压缩历史，适合语言序列压缩，但难以保证对所有过去细节的精确随机检索。
- 三种形式在理论上对应同一 Retention 关系，不等于单个训练推理 kernel 在数值精度与吞吐量上总完全一致。
- 对物理场只能作为“时间上的记忆 / 分块计算”候选，而非稳态三维流场空间邻接的直接替代。

## 5. 文献脉络

- [[Literature/architectures/Efficiently Modeling Long Sequences with Structured State Spaces|S4]]：同属递归/卷积视角，状态构造不同。
- [[Literature/architectures/Gated Linear Attention Transformers with Hardware-Efficient Training|GLA]]：在 RetNet 式固定衰减之外增加数据相关门控。
- [[Literature/architectures/Transformers are SSMs Generalized Models and Efficient Algorithms Through Structured State Space Duality|Mamba-2 / SSD]]：从结构化矩阵角度统一某些 SSM 与 masked attention；不是 RetNet 的直接版本继承。
- [[Literature/transformer/Mixture-of-Top-k Attention Efficient Attention via Scalable Fast Weights|MiTA]]：MiTA 仍路由原始 KV 子集，Retention 更倾向固定维状态压缩，是不同信息保存策略。

原文：https://arxiv.org/html/2307.08621
