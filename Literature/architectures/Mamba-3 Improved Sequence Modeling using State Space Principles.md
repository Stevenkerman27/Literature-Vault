---
title: "Mamba-3: Improved Sequence Modeling using State Space Principles"
authors:
  - Aakash Lahoti
  - Kevin Y. Li
  - Berlin Chen
  - Caitlin Wang
  - Aviv Bick
  - J. Zico Kolter
  - Tri Dao
  - Albert Gu
year: 2026
venue: "ICLR 2026"
doi: null
arxiv: "2603.15569"
url: "https://arxiv.org/abs/2603.15569"
source: "未提供本地 PDF；原文 https://arxiv.org/pdf/2603.15569"
tags: [paper, Mamba-3, state-space-model, recurrent, MIMO, complex-state, inference-efficient]
status: read
updated: 2026-10-25
---

> [!abstract] 摘要译述
> 线性序列模型具有固定大小的状态与低内存推理优势，但在状态跟踪、检索和语言质量上仍可能落后于全注意力模型，而且某些理论上线性的更新并没有用好实际 GPU。Mamba-3 从推理效率出发，引入指数调整梯形（Exponential-Trapezoidal）状态离散化、具有复数动力学的状态更新，以及多输入多输出（Multi-Input Multi-Output，MIMO）状态建模。作者在检索、状态跟踪与 1.5B 规模语言模型基准中报告相对 Mamba-2 和 Gated DeltaNet 的改进。

# 核心定位

**Mamba-3 不是 Mamba-2 改一个 kernel，而是将状态转移的数值离散化、状态相位表达和跨通道混合联合重设计。** 它也提醒研究者：Mamba-2 的主要目标是训练吞吐，而 Mamba-3 更多考虑推理时的算力、带宽和状态容量。

## 1. 新的离散化：Exponential-Trapezoidal

作者将以前实现中的离散化解释为 Exponential-Euler，并提出新的梯形更新：

\[
h_t=e^{\Delta_t A_t}h_{t-1}
+(1-\lambda_t)\Delta_t e^{\Delta_t A_t}B_{t-1}x_{t-1}
+\lambda_t\Delta_tB_tx_t,
\quad \lambda_t\in[0,1] .
\]

与只读本时刻状态输入 \(B_tx_t\) 的 Euler 更新相比，该式能结合上一时刻与当前时刻的输入；当 \(\lambda_t=1\) 时退化为旧式更新，\(\lambda_t=\tfrac12\) 时包含标准梯形思想。作者指出它等价于在 **SSM state-input 上执行宽度 2 的输入依赖卷积**，与普通 block 外的 Conv1D 不同。

### 精度声明边界

论文证明，在标准稳定性条件并且 \(\lambda_t=\frac12+O(\Delta_t)\) 的约束下具有相应更高阶误差性质；实际消融发现**不施加该约束**反而有利于学习效果。不得将连续数值积分阶数无条件等同于语言模型预测精度。

## 2. 复数状态为什么重要？

相较纯实值衰减 \(a_t\)，复数状态转移可以表示旋转/相位与衰减的组合，因此可保留更丰富的顺序、周期和状态跟踪信息。作者还给出与旋转位置编码（Rotary Position Embedding，RoPE）相关的等价理解。

## 3. MIMO：提高运算强度而不线性增加缓存

Mamba-1/2 的简化 SISO（Single-Input Single-Output）结构对多个通道相对独立处理。Mamba-3 引入 \(R\) 个输入/输出组的投影与状态读写，使有限缓存状态能用更高运算密度服务计算。论文在主实验中尝试 \(R=4\) 的 MIMO，并调整 FFN 内层维度以保持与 SISO 比较时的参数量近似一致。

作者宣称 MIMO 增加计算 FLOPs，而对内存搬运的影响较小，因此在 bandwidth-bound 的 autoregressive decode 中有潜在收益；其有效性取决于硬件、实现和比较设置。

## 4. 论文实验与数值

- 语言主表：多个模型规模；在 1.5B 等规模上，统一使用 FineWeb-Edu **100B tokens** 预训练、Llama-3.1 tokenizer、上下文长度 2K。
- 对照：Transformer、Mamba-2、Gated DeltaNet（GDN）和 Mamba-3 SISO/MIMO。
- 1.5B 规模的下游综合准确率：GDN **55.8**、Mamba-2 **55.7**、Mamba-3 SISO **56.4**、Mamba-3 MIMO **57.6**。相较最好基线的提升：SISO **+0.6 个百分点**，MIMO 在 SISO 上再 **+1.2 个百分点**。
- 同一规模困惑度：Mamba-3 SISO **10.35**、MIMO **10.24**；论文报告 MIMO 用状态大小 64 可匹配 Mamba-2 状态大小 128 的 perplexity。
- 另外分析 Needle-in-a-Haystack（NIAH）检索、状态跟踪、状态维度缩减与 GPU 推理。论文明确承认在部分真实检索任务上 Transformer 仍有优势，尤其是非结构化的精确查询。

这些提升是该论文的训练预算、训练数据、特定 eval suite 的结果，不自动外推到所有 token 长度和领域数据。

## 5. 关键边界与联动

- 固定维状态仍会压缩历史；复数动力学与 MIMO 不会神奇地消除所有精确随机访问的限制。
- 其设计动机偏**自回归语言推理**，不是无序表面网格的三维场回归；状态空间模型中“时间”是一维序列，CFD 网格中的点排序需要额外处理。
- Mamba-3 是作者独立的 ICLR 2026 论文，不能仅阅读 2024 年 Mamba-2 就认为掌握最新 Mamba 谱系。

## 相关文献

- [[Literature/architectures/Mamba Linear-Time Sequence Modeling with Selective State Spaces|Mamba-1]]：输入选择性更新。
- [[Literature/architectures/Transformers are SSMs Generalized Models and Efficient Algorithms Through Structured State Space Duality|Mamba-2 / SSD]]：对偶与训练高效实现。
- [[Literature/architectures/Gated Delta Networks Improving Mamba2 with Delta Rule|Gated DeltaNet]]：结合门控遗忘与定向 delta 写入，是本文的重要竞争基线。
- [[Literature/architectures/MambaVision A Hybrid Mamba-Transformer Vision Backbone|MambaVision]]：将旧式 Mamba 用于 2D 视觉的混合方法，不应误认为使用了 Mamba-3。

来源：[arXiv HTML](https://arxiv.org/html/2603.15569) · [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html)
