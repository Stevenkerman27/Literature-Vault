---
title: "Gated Delta Networks: Improving Mamba2 with Delta Rule"
authors: [Songlin Yang, Jan Kautz, Ali Hatamizadeh]
year: 2025
venue: "ICLR 2025"
doi: null
arxiv: "2412.06464"
url: "https://arxiv.org/abs/2412.06464"
source: "未提供本地 PDF；原文 https://arxiv.org/pdf/2412.06464"
tags: [paper, gated-delta-network, GDN, linear-attention, delta-rule, associative-memory]
status: read
updated: 2026-10-25
---

> [!abstract] 摘要译述
> 线性注意力与 Mamba 类递归架构虽然推理高效，但在精确关联检索和长上下文记忆上仍可能不足。作者观察到两种机制互补：门控遗忘（gating）适合快速清除不需要的信息，Delta Rule（差分规则）能够对某个 Key 对应的旧记忆进行定向修正。Gated DeltaNet（GDN）将二者结合为 gated delta 更新，并给出硬件高效的分块并行算法，在语言、常识推理、真实与合成检索、长上下文测试中超过该文比较的 Mamba-2、DeltaNet 等模型；还研究与局部 Attention / Mamba-2 的混合结构。

# 核心问题

**固定维状态不是单纯越大越好：重要的是能够忘记全局旧信息，同时只替换与当前 Key 冲突的旧 Value。** Gated DeltaNet 正是区分了“全部衰减”与“定向覆盖”，不是把 Softmax 权重的名字改成 Delta。

## 1. 三种矩阵记忆更新

令 Key 为 \(k_t\in\mathbb R^{d_k}\)，Value 为 \(v_t\in\mathbb R^{d_v}\)，矩阵状态为 \(S_t\in\mathbb R^{d_v\times d_k}\)。

**普通线性注意力外积累加**：

\[
S_t=S_{t-1}+v_tk_t^\top .
\]

问题：旧 Key–Value 可能不断叠加，难以改写。

**Mamba-2 式标量门控遗忘**（简化）：

\[
S_t=\alpha_tS_{t-1}+v_tk_t^\top .
\]

问题：遗忘是全局性的，可能一起抹去仍有价值的信息。

**Delta Rule**：

\[
S_t=S_{t-1}+
\beta_t(v_t-S_{t-1}k_t)k_t^\top .
\]

其中 \(S_{t-1}k_t\) 是旧状态对该 Key 的当前预测；本次只写入预测误差，而不是把新 Value 整体叠加。可以理解为 Key 定位的定向记忆修正。

**Gated DeltaNet**：

\[
\boxed{S_t=\alpha_tS_{t-1}
+\beta_t(v_t-\alpha_tS_{t-1}k_t)k_t^\top}.
\]

它结合全局遗忘系数 \(\alpha_t\) 与 Key 定位的覆盖力度 \(\beta_t\)。这与原文的
\(S_t=S_{t-1}[\alpha_t(I-\beta_t k_tk_t^\top)]+\beta_tv_tk_t^\top\)
是同一式子。实际网络还包含归一化、输入投影和残差等模块。

## 2. 为什么还需要专用训练算法？

逐 token 的 Delta 更新包含矩阵–向量运算，不能充分利用 GPU 的高吞吐矩阵乘法。作者利用 WY 表示（矩阵变换的低秩分解）和分块累计，将 chunk 内多次 key-specific 更新改写为矩阵乘法，同时 chunk 间保持递归状态。因而保留线性状态推理的优势，又能高效训练长序列。

## 3. 实验设置与结果范围

- 语言模型比较包含 400M 与 **1.3B** 等量级；
- 1.3B 的主要公平对照使用 FineWeb-Edu **100B tokens**、相同 optimizer / data / tokenization 配置；
- 评估语言模型 perplexity、commonsense、needle-in-a-haystack（单针检索）、长度外推和长上下文基准；
- 比较 Mamba-1、Mamba-2、RetNet、HGRN2、DeltaNet 和 GDN，以及加入 sliding window attention 的混合结构；
- 论文对等规模对照中报告 GDN 在多数相关语言与检索任务超过上述线性模型，尤其凸显 Delta Rule 对目标键更新的好处。不能把这一结论理解为全面超过所有 Transformer 规模或所有真实任务。

## 4. 局限与工程边界

- 固定维矩阵状态仍可能遭遇多个相近 Key 的干扰；Delta 修正是定向写入机制，不等于永久精确数据库。
- 与 Mamba-3 的后续比较应使用相同训练语料与架构设定。[[Literature/architectures/Mamba-3 Improved Sequence Modeling using State Space Principles|Mamba-3]] 是 2026 年另一个改进方向，不是本文作者已测试的模型。
- 对无序 CFD 网格，无论 Key 来自坐标还是物理特征，使用顺序递推都会引入点排列影响，需要新的对称机制。
- 这篇主要是语言与长文本结果，不是雷诺数迁移、激波或可微气动计算的直接证据。

## 5. 文献关系

- **直接演进**：[[Literature/architectures/Gated Linear Attention Transformers with Hardware-Efficient Training|GLA]]（学习遗忘门控）+ DeltaNet 的 key-specific 写入 → Gated DeltaNet（本论文）。
- **竞争/数学对照**：[[Literature/architectures/Transformers are SSMs Generalized Models and Efficient Algorithms Through Structured State Space Duality|Mamba-2]] 有结构化状态对偶，GDN 在该基线上引入更有针对性的写入。
- **直接后续比较**：[[Literature/architectures/Mamba-3 Improved Sequence Modeling using State Space Principles|Mamba-3]] 在共同数据预算下对比 GDN。
- **与可微物理场的关系**：若要将长时间序列上的外力、控制和历史流场编码成紧凑状态，可作为候选；对稳态空间网格无需强行使用时序模型。

原文：https://arxiv.org/html/2412.06464
