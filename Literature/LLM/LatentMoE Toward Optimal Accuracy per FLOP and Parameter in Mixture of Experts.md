---
title: "LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts"
authors:
  - Venmugil Elango
  - Nidhi Bhatia
  - Roger Waleffe
  - Rasoul Shafipour
  - Tomer Asida
  - Abhinav Khattar
  - Nave Assaf
  - Maximilian Golub
  - Joey Guman
  - Tiyasa Mitra
  - Ritchie Zhao
  - Ritika Borkar
  - Ran Zilberstein
  - Mostofa Patwary
  - Mohammad Shoeybi
  - Bita Rouhani
year: 2026
venue: "arXiv preprint"
doi: "10.48550/arXiv.2601.18089"
arxiv: "2601.18089"
url: "https://arxiv.org/abs/2601.18089"
source: "未提供本地 PDF；原文 https://arxiv.org/pdf/2601.18089"
tags: [paper, LLM, MoE, latent-space, hardware-software-co-design, inference-efficiency]
status: read
updated: 2026-11-11
---

> [!abstract] 摘要译述
> 本文从硬件与软件协同设计角度重新研究 MoE，目标是在给定推理计算和参数成本下最大化模型精度。作者指出，传统 MoE 的专家权重读取和跨设备 Token 通信都发生在模型完整隐藏维度上，可能受内存带宽和通信成本限制。LatentMoE 先把 Token 表示投影到共享低维潜在空间，再在潜在空间中完成路由专家计算和 Token Dispatch；作者通过理论分析与最高 95B 参数规模的实验，报告其在 Accuracy per FLOP 和 Accuracy per Parameter 上优于标准 MoE 设计。

# 核心定位

**LatentMoE 的核心不是“把隐藏层缩小一点”，而是将专家路由、专家计算和专家通信从完整隐藏维度解耦，搬到低维潜在空间，并把省下的成本用于增加专家数量与 Top-K。**

## 1. 问题与动机

MoE 常被描述为“总参数很大、每 Token 只激活少量专家，所以推理计算更便宜”。但实际部署不只受 FLOPs 限制：

- 专家权重需要从显存读取，内存带宽可能成为瓶颈。
- Token 被路由到不同设备时，需要跨 GPU / 节点交换激活，All-to-All 通信可能成为瓶颈。
- 如果只降低激活专家数或专家中间维度，虽然计算量下降，却可能降低每 Token 可用的非线性容量和专家组合多样性。

因此论文把 MoE 效率拆成 Accuracy per FLOP 与 Accuracy per Parameter 两个维度，并显式考虑低延迟推理下的参数读取和通信成本。

## 2. 方法

### 2.1 在潜在空间执行路由专家计算

设模型隐藏维度为 d，LatentMoE 使用投影矩阵将输入压缩到低维潜在空间 l：

\[
x\in\mathbb R^d
\xrightarrow{W_{\downarrow}}
z\in\mathbb R^\ell
\xrightarrow{\text{Router + Experts}}
z'
\xrightarrow{W_{\uparrow}}
y\in\mathbb R^d,\qquad \ell<d.
\]

路由专家的权重和 Token Dispatch / Aggregation 都在 l 维潜在空间执行。若压缩比例为：

\[
\alpha=\frac{d}{\ell},
\]

则路由通信量和专家权重读取成本相对标准 MoE 可按约 1/α 的比例下降。

### 2.2 将节省的成本用于更多专家组合

LatentMoE 不只保留计算节省，还提出 Accuracy-Oriented 配置：把总路由专家数 N 和每 Token 激活专家数 K 同时扩大 α 倍。

直觉是：在通信量和专家权重读取成本大致不变的情况下，更多专家和更大的 Top-K 可以增加 Token 可选择的专家组合数，并提高每 Token 的有效非线性预算。

论文同时保留共享专家在原始隐藏维度 d 上计算，因为共享专家不构成主要的路由通信和权重读取瓶颈。也就是说，低维化主要针对路由专家路径，而不是把整个模型所有计算都压缩到潜在空间。

### 2.3 效率配置与精度配置

论文区分两种设计取向：

- Efficiency-Oriented：主要利用潜在空间降低通信和权重读取成本。
- Accuracy-Oriented：将省下的成本投入更多专家和更高 Top-K，以提升模型质量，同时维持近似相同的推理成本。

因此 LatentMoE 的关键变量是潜在维度 l、压缩比例 α、总专家数 N 和激活专家数 K 的联合选择，而不是单独调小专家维度。

## 3. 实验设置与结果

- **设计目标**：比较不同 MoE 结构在 Accuracy per FLOP 和 Accuracy per Parameter 上的 Pareto Frontier。
- **规模探索**：设计空间探索最高覆盖约 95B 总参数、8B 激活参数的模型配置。
- **训练数据**：报告超过 1T Token 的训练跨度。
- **消融分析**：研究潜在维度压缩比例、专家数、激活专家数和模型质量之间的权衡。
- **推理评估**：区分离线高吞吐和在线低延迟两类部署情境，分析权重加载、通信和计算瓶颈。
- **主要结论**：论文报告在压缩比例 α≤4 时模型质量可得到较好保持，并采用 α=4 作为后续实验的默认配置；Accuracy-Oriented LatentMoE 在近似相同通信与内存带宽成本下，通过扩大 N 和 K 提升精度。

## 4. 关键贡献

1. 将 MoE 的设计目标从单纯减少 FLOPs 扩展为同时优化 Accuracy per FLOP 与 Accuracy per Parameter。
2. 将专家路由和专家计算解耦于完整隐藏维度，降低专家权重读取和跨设备通信。
3. 把低维化节省的成本重新投入专家数量与 Top-K，提升专家组合多样性。
4. 将硬件吞吐、延迟、内存带宽和通信成本纳入 MoE 架构搜索，而不是只看参数量和理论 FLOPs。

## 5. 局限与启发

- 潜在维度不能无限压缩：如果低于任务所需的有效特征秩，关键信息会丢失，模型质量可能下降。
- α=4 是论文实验中较合适的配置，不应视为所有模型、数据和硬件的普适最优值。
- 低维专家路径增加 Down / Up Projection；其收益取决于投影开销能否被更低的权重读取与通信成本覆盖。
- 高 Top-K 会提高专家组合容量，但也可能增加路由调度、Kernel Launch 和负载均衡压力；需要端到端测量。
- 论文的 Accuracy per Parameter 与 Accuracy per FLOP 仍不足以完整刻画真实服务成本，实际部署还应报告 P50/P99 延迟、吞吐、峰值显存、跨节点流量和批量大小。

## 6. 关联

- **直接对照**：[[Literature/LLM/DeepSeekMoE Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models|DeepSeekMoE]] 通过细粒度切分和共享专家提高专家专门化；LatentMoE 进一步把路由专家路径放到低维潜在空间。
- **直接对照**：[[Literature/LLM/DeepSeek-V2 A Strong Economical and Efficient Mixture-of-Experts Language Model|DeepSeek-V2]] 将 MoE 与 MLA 组合，分别降低 FFN 激活计算和 KV Cache；LatentMoE 主要优化 MoE 权重读取与 Token Dispatch。
- **模型应用**：论文指出该架构已被 Nemotron-3 Super / Ultra 等模型采用；具体模型配置和部署结果应结合对应技术报告核对。
- **共同主题**：[[Literature/LLM/DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence|DeepSeek-V4]] 也体现架构与硬件协同设计，但其 CSA/HCA 主要解决长上下文 Attention / KV 成本，LatentMoE 主要解决专家路径的权重读取和通信成本。

## 7. 图片与原件

仓库当前未提供本地 PDF，因此暂不嵌入论文截图。建议查看原文 Figure 1（Standard MoE 与 LatentMoE 结构对比）和 Table 1（效率配置与精度配置成本比较）。
