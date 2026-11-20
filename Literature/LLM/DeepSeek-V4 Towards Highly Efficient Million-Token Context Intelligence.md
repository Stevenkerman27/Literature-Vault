---
title: "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence"
authors:
  - DeepSeek-AI
year: 2026
venue: "arXiv technical report"
doi: "10.48550/arXiv.2606.19348"
arxiv: "2606.19348"
url: "https://arxiv.org/abs/2606.19348"
source: "未提供本地 PDF；原文 https://arxiv.org/pdf/2606.19348"
tags: [paper, LLM, million-context, CSA, HCA, mHC, MoE, Muon]
status: read
updated: 2026-11-11
---

> [!abstract] 摘要译述
> 本文介绍 DeepSeek-V4 系列预览模型，包括 1.6T 总参数、49B 激活参数的 V4-Pro，以及 284B 总参数、13B 激活参数的 V4-Flash，两者支持 100 万 Token 上下文。相较 V3，V4 保留 DeepSeekMoE 和 Multi-Token Prediction，同时引入 Compressed Sparse Attention（CSA，压缩稀疏注意力）、Heavily Compressed Attention（HCA，重度压缩注意力）、Manifold-Constrained Hyper-Connections（mHC，流形约束超连接）和 Muon 优化器。作者报告在 1M 上下文下，V4-Pro 的单 Token 推理 FLOPs 为 V3.2 的 27%，KV Cache 为 10%。

# 核心定位

**DeepSeek-V4 不是抛弃 Transformer 的新范式，而是在 Transformer 主干上联合升级 Attention、残差流拓扑、优化器和推理基础设施，以支持百万 Token 上下文。**

## 1. 问题与动机

当上下文长度进入数十万到百万 Token，注意力访问和 KV Cache 会成为主要成本。单纯减少 KV Head 或压缩每个 Token 的 KV 向量仍不够，因为历史序列长度本身仍然很大。

V4 采用分层思路：

- 先沿序列维压缩 KV 条目。
- 对压缩后的条目再进行稀疏选择，减少实际读取数量。
- 对部分层使用更激进的 KV 压缩，保留另一种低成本全局信息通道。
- 同时改善残差流跨层传播，避免更复杂的网络拓扑破坏训练稳定性。

## 2. 方法

### 2.1 继承 DeepSeek-V3 的设计

V4 保留 Transformer 主干、DeepSeekMoE 和 MTP。MoE 仍由共享专家与细粒度路由专家构成；MTP 配置沿用 V3。V4 的变化重点不是重新发明 MoE，而是改善超长上下文 Attention 与残差流结构。

### 2.2 Compressed Sparse Attention（CSA）

CSA 将 KV Cache 的序列维压缩与 DSA 稀疏选择串联：

1. 将每 m 个 KV 条目压缩成一个潜在条目。
2. 对压缩后的 KV 使用 Lightning Indexer。
3. 对每个 Query 只选取 Top-K 个压缩 KV 条目执行核心注意力。

因此它同时降低“历史条目数量”和“每个 Query 访问的条目数”。论文中 CSA 的压缩率记为 m；在给定配置中 m=4。

### 2.3 Heavily Compressed Attention（HCA）

HCA 对 KV Cache 进行更激进的压缩：每 m' 个 KV 条目合并成一个条目，其中 m' 远大于 CSA 的 m。与 CSA 不同，HCA 对压缩后的表示执行 Dense Attention，而不是再做 DSA Top-K 选择。

CSA 和 HCA 在层间交错使用，形成 Hybrid Attention。这样既保留稀疏注意力对相关历史片段的选择能力，也通过高度压缩的全局通道提供更广范围的信息。

### 2.4 Manifold-Constrained Hyper-Connections（mHC）

V4 将标准残差连接扩展为多流残差结构，并采用 mHC 约束残差流之间的映射矩阵。mHC 的核心约束是双随机矩阵：矩阵元素非负，每行和每列之和都为 1。其详细方法见 [[Literature/architectures/mHC Manifold-Constrained Hyper-Connections|mHC 独立论文笔记]]。

### 2.5 Muon 与训练 / 推理系统

V4 使用 Muon 优化器，并进一步优化专家并行通信计算重叠、TileLang Kernel、批次不变与确定性 Kernel、长上下文并行，以及 KV Cache 管理。MoE 专家权重在后训练中还使用 FP4 量化感知训练。模型架构效率不能与这些系统优化割裂评价。

## 3. 实验设置与结果

- **模型规模**：V4-Pro 1.6T 总参数、49B 激活参数；V4-Flash 284B 总参数、13B 激活参数。
- **预训练数据**：V4-Flash 训练 32T Token，V4-Pro 训练 33T Token；语料强调长文档、数学、代码、多语言和高质量网页数据。
- **上下文长度**：原生支持 1M Token。
- **架构配置示例**：V4-Flash 使用 43 层、隐藏维度 4096；CSA 压缩率 m=4、Top-K=512；HCA 压缩率 m'=128；注意力层采用 CSA/HCA 交错配置。MoE 层包含 1 个共享专家和 256 个路由专家，每 Token 激活 6 个路由专家。
- **效率对比**：在 1M Token 上下文设置下，V4-Pro 相较 V3.2 的单 Token 推理 FLOPs 为 27%，KV Cache 为 10%；V4-Flash 分别报告为 10% 和 7%。
- **模型评估**：包含知识、推理、代码、Agent 和百万 Token 长上下文任务；论文报告 V4-Pro-Max 在多个开放模型基准上具有竞争力。

上述效率数字是作者在指定模型和 1M 上下文条件下的报告值，不能脱离精度、硬件、Batch、Prefill/Decode 和 KV Cache 管理策略直接外推。

## 4. 关键贡献

1. CSA 将 KV 序列压缩和 DSA 稀疏选择组合，降低注意力读取成本。
2. HCA 以更高压缩率构造稠密全局信息通道，与 CSA 形成混合注意力。
3. mHC 改善多流残差连接的信号传播稳定性。
4. 将 MoE、长上下文注意力、优化器和系统 Kernel 协同设计，支持百万 Token 场景。

## 5. 局限与启发

- CSA 的压缩和 Top-K 选择都可能丢失局部细节；HCA 的高压缩表示也需要验证其对精确检索和细粒度定位任务的影响。
- 百万 Token 窗口不等于模型能均匀、准确地利用每一个 Token；必须区分可输入长度、可检索长度和复杂推理质量。
- 论文为预览版技术报告，部分效率指标是作者自己的硬件与系统测量；独立复现仍需要统一的硬件、精度和服务配置。
- mHC 的收益与 CSA/HCA 的收益来自不同模块，应通过独立消融分离。
- 对实际系统而言，KV Cache 的显存、带宽、共享前缀复用、磁盘缓存和 Prefill 成本需要分别测量。

## 6. 关联

- **直接继承**：[[Literature/LLM/DeepSeek-V3 Technical Report|DeepSeek-V3]] 提供 DeepSeekMoE、MLA 和 MTP 基础。
- **直接继承并扩展**：[[Literature/LLM/DeepSeek-V3.2 Pushing the Frontier of Open Large Language Models|DeepSeek-V3.2]] 提供 DSA；V4 的 CSA 将 KV 压缩与 DSA 组合。
- **直接继承 / 宏观拓扑设计**：[[Literature/architectures/mHC Manifold-Constrained Hyper-Connections|mHC]] 是 V4 的残差流结构基础。
- **机制对照**：[[Literature/LLM/Mamba-3 Improved Sequence Modeling using State Space Principles|Mamba-3]] 通过递归状态更新降低序列成本；CSA/HCA 仍然属于 Transformer Attention 的压缩与稀疏化路线，并非 SSM。
- **硬件协同方向**：[[Literature/LLM/LatentMoE Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts|LatentMoE]] 进一步研究 MoE 专家计算与通信的低维化。

## 7. 图片与原件

仓库当前未提供本地 PDF，因此暂不嵌入论文截图。建议查看原文 Figure 2（整体架构）、Figure 3（CSA）和 mHC 方法图。
