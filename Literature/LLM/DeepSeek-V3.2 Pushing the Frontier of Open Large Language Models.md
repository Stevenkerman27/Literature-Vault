---
title: "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models"
authors:
  - DeepSeek-AI
year: 2025
venue: "arXiv technical report"
doi: "10.48550/arXiv.2512.02556"
arxiv: "2512.02556"
url: "https://arxiv.org/abs/2512.02556"
source: "未提供本地 PDF；原文 https://arxiv.org/pdf/2512.02556"
tags: [paper, LLM, sparse-attention, DSA, MLA, GRPO, agent]
status: read
updated: 2026-11-11
---

> [!abstract] 摘要译述
> DeepSeek-V3.2 旨在同时提升计算效率、推理和 Agent 能力。架构方面提出 DeepSeek Sparse Attention（DSA，DeepSeek 稀疏注意力），用轻量索引器为每个 Query 选择少量历史 Token，降低长上下文注意力成本；训练方面扩大强化学习计算，并构建大规模 Agent 任务合成流水线。本文的重要价值在于把稀疏注意力从推理期技巧推进为经过继续预训练适配的模型架构。

# 核心定位

**DSA 不是简单地把 Attention 矩阵随机置零，而是学习一个索引器，对历史 Token 进行内容相关的 Top-K 选择，再让主注意力只处理被选中的历史位置。**

## 1. 问题与动机

在长度为 L 的序列上，标准全注意力的主要计算复杂度随 L 平方增长。即使采用 MLA 压缩 KV Cache，长上下文时仍然需要处理大量历史 KV 条目。

稀疏注意力可以减少每个 Query 实际读取的历史位置，但直接稀疏化会遇到两个问题：

- 如何判断哪些历史 Token 对当前 Query 重要？
- 如何让索引器足够轻量，不让“寻找相关 Token”的开销抵消主注意力节省？

DSA 的目标是用低成本 Indexer 学习稀疏选择，并通过继续预训练让主模型适应稀疏访问模式。

## 2. 方法

### 2.1 Lightning Indexer

DSA 包含 Lightning Indexer 和 Fine-Grained Token Selection 两个组件。对当前 Query Token h_t 与历史 Token h_s，Indexer 计算相关性分数：

\[
I_{t,s}=\sum_j w^I_{t,j}\operatorname{ReLU}
\left((q^I_{t,j})^\top k^I_s\right).
\]

其中索引 Query、Key 和权重由隐藏状态投影得到。Indexer 使用少量 Head，并采用 ReLU 以提升吞吐。随后对每个 Query 选择分数最高的 Top-K 历史 Token，主注意力只在选中集合上计算。

### 2.2 在 MLA 下实例化 DSA

论文基于 MLA 的 MQA 模式实现 DSA：每个 Token 的潜在 KV 条目由所有 Query Head 共享。这样 Indexer 选出的历史位置与压缩后的 KV 表示结合，避免先恢复完整多头 KV 再做稀疏筛选。

注意：这里的 MQA 是 MLA 的一种运行模式，不应把它与“普通 MQA + DSA”完全等同。

### 2.3 Dense Warm-up 与 Sparse Training

为了让 Indexer 学会近似原有稠密注意力的选择分布，训练分为两阶段：

1. **Dense Warm-up**：保持主模型 Dense Attention，只训练 Indexer。将原注意力各 Head 的分数求和并沿序列维度 L1 归一化，作为目标分布；用 KL Divergence 训练 Indexer。
2. **Sparse Training**：引入 Top-K Token 选择，并联合优化模型参数，使主模型适应稀疏访问模式；同时继续对选中 Token 集合上的注意力分布施加对齐约束。

论文报告 warm-up 阶段训练 1,000 步，每步使用 16 条 128K 序列，总计约 2.1B Token。

### 2.4 复杂度边界

DSA 主注意力部分从 O(L²) 降为 O(LK)，其中 K 远小于 L。但 Lightning Indexer 本身仍然需要对历史位置打分，复杂度仍为 O(L²)，只是其 Head 数和维度较小，并且可用 FP8 实现。

因此准确表述应是：**DSA 将主注意力的二次计算压缩为稀疏计算；索引器仍保留二次扫描，但其常数项显著更小。** 不应宣称整个算法严格变成纯 O(LK)。

## 3. 实验设置与结果

- **基础模型**：DeepSeek-V3.1-Terminus，已扩展到 128K 上下文。
- **继续预训练**：先进行 Dense Warm-up，再进行 Sparse Training；训练数据分布与此前 128K 长上下文扩展数据对齐。
- **性能对照**：比较 DeepSeek-V3.2-Exp 与 DeepSeek-V3.1-Terminus 的标准基准、人工偏好和长上下文任务。
- **结果**：作者报告长序列计算效率显著改善，短上下文和长上下文任务没有观察到明显性能退化；在 AA-LCR 等独立长上下文评估中，推理模式得分有所提高。
- **Agent 后训练**：报告合成 1,800 多个不同环境和 85,000 个复杂 Prompt，用于强化学习训练 Agent 能力。

## 4. 关键贡献

1. 用学习型 Indexer 选择历史 Token，构造内容相关的稀疏注意力。
2. 把 DSA 与 MLA 的压缩 KV 表示结合，避免稀疏化与 KV Cache 设计割裂。
3. 通过 Dense Warm-up 和 Sparse Training 两阶段训练，让主模型适应新的稀疏访问模式。
4. 明确报告主注意力与 Indexer 的不同复杂度，便于分析端到端效率边界。

## 5. 局限与启发

- Indexer 仍保留 O(L²) 扫描；它的常数较小，但不是免费操作。
- Top-K 选择质量决定主注意力能否保留关键历史信息；选择遗漏会造成不可逆的信息损失。
- 训练需要 Dense Warm-up 和 Sparse Training，不能假设把稠密模型推理时替换成 DSA 就能无损运行。
- 论文的效率结果依赖专门的 Indexer Kernel、FP8 和部署优化；需要在目标硬件上分别测试 Prefill 与 Decode。
- 对研究复现而言，应同时报告选择召回率、任务性能、Indexer 开销、主注意力开销和端到端延迟。

## 6. 关联

- **直接继承**：[[Literature/LLM/DeepSeek-V2 A Strong Economical and Efficient Mixture-of-Experts Language Model|DeepSeek-V2]] 提供 MLA 低秩 KV Cache 表示。
- **直接继承**：[[Literature/LLM/DeepSeek-V3 Technical Report|DeepSeek-V3]] 提供模型基础、MoE 和训练基础设施。
- **机制相似但实现不同**：Native Sparse Attention（NSA）同样研究硬件友好的稀疏注意力，但其分块压缩、选择与局部注意力设计不能与 DSA 混为一谈。
- **直接扩展**：[[Literature/LLM/DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence|DeepSeek-V4]] 在 DSA 基础上引入 CSA，将 KV 序列压缩与稀疏选择串联。

## 7. 图片与原件

仓库当前未提供本地 PDF，因此暂不嵌入论文截图。建议查看原文 Figure 1 / Figure 2（Indexer 与 DSA 结构）及 Section 2.1（Dense Warm-up、Sparse Training）。
