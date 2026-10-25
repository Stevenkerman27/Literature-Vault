---
title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
authors: [Albert Gu, Tri Dao]
year: 2024
venue: "Conference on Language Modeling (COLM 2024); arXiv first submitted 2023-12-01"
doi: null
arxiv: "2312.00752"
url: "https://arxiv.org/abs/2312.00752"
code: "https://github.com/state-spaces/mamba"
source: "未提供本地 PDF；原文 https://arxiv.org/pdf/2312.00752"
tags: [paper, Mamba, selective-state-space-model, recurrent, linear-time, sequence-model]
status: read
updated: 2026-10-25
---

> [!abstract] 摘要译述
> 固定的状态空间序列模型在语言等离散输入上的内容推理能力受到限制。Mamba 把 SSM 参数设计为输入的函数，使模型能够选择性保留、遗忘或写入信息；并针对选择性 SSM 无法使用单一时不变卷积核的问题，提出 GPU 硬件感知的并行 selective scan。模型去除了传统 Transformer 的注意力与独立 MLP 子层，在语言、音频和基因组任务中报告良好效果，同时保持序列长度线性伸缩。

# 关键定位

**Mamba 不是稀疏 Transformer，也不是线性注意力换了名字；它使用固定大小的递归隐藏状态，按当前 token 决定状态如何更新。** 作为序列模型，它针对固定卷积核无法按内容进行选择性压缩的问题；作为网络架构，它用选择性 SSM 取代 Attention/FFN 组合。

## 1. 从 S4 的 LTI 到 Selective SSM

普通离散状态空间模型：

\[
h_t=\bar A h_{t-1}+\bar Bx_t,\qquad
y_t=Ch_t .
\]

[[Literature/architectures/Efficiently Modeling Long Sequences with Structured State Spaces|S4]] 中 \(\bar A,\bar B,C\) 固定，因此整个序列对应一个固定卷积核。Mamba 改为：

\[
B_t=s_B(x_t),\quad C_t=s_C(x_t),\quad
\Delta_t=\operatorname{softplus}(s_\Delta(x_t)),
\]

并由 \(\Delta_t,A,B_t\) 离散化得到 \(\bar A_t,\bar B_t\)：

\[
h_t=\bar A_t h_{t-1}+\bar B_t x_t,\qquad y_t=C_t h_t .
\]

- \(\Delta_t\) 控制时间尺度/忘记率；
- \(B_t\) 决定当前输入以什么方向写入状态；
- \(C_t\) 决定如何从状态读取输出。

**输入选择性**使当前 token 可以影响未来长距离的信息传播。固定参数卷积无法直接实现这种效果，因为现在的核取决于完整输入序列。

## 2. 三项架构和算法设计

### 2.1 Selective copying 与 induction

论文用 Selective Copying（只记住被标记的少量 token，过滤噪声）和 Induction Heads 风格的关联检索任务诊断内容选择。它们是构造的合成测试，不能直接等价于自然语言全面推理能力。

### 2.2 Hardware-aware Selective Scan

不能再把卷积核一次性缓存，作者采用并行 scan 处理按时间变化的线性递推，使用内核融合、减少高带宽显存（HBM）读写、重计算中间状态等策略。形式上可递归处理每一步，也可通过结合律作树形并行扫描。算法最终复杂度随长度 \(L\) 线性增长。

### 2.3 Mamba Block

Block 先进行输入线性投影、短局部卷积、SiLU 激活，再进入 selective SSM；另一条门控投影调制输出，最后投影回通道宽度。所谓“没有 MLP block”并非没有线性层，而是没有传统 Transformer 的单独两层 FFN 子层。

## 3. 实验与结果

- **语言**：从约 130M 到 2.8B 参数，比较 Transformer 与其他 subquadratic 模型的预训练困惑度及下游 zero/few-shot 任务。
- **DNA / 基因组**：长度扩展实验检验 SSM 处理超长序列。
- **音频**：长上下文建模与自回归波形生成。
- **长序列效率**：对选择性 scan 与完整 decoder throughput 进行测量。

论文报告 Mamba-3B 量级模型与约 2 倍参数的 Transformer 在其指定语言基准上具有竞争力；自回归推理在相似规模与特定硬件/批量设置下吞吐可达 Transformer 的 **4–5 倍**。这是作者的特定 benchmark 结果，不能断言全部规模或任务都达到五倍速度。

## 4. 局限和开放问题

- 固定大小状态代表压缩历史，**无法像完整 KV cache 一样随时精确重读每个过去 token**；准确检索大量随机细节可能受限。
- 状态递推依赖 token 顺序，不能直接用于无序的非结构表面网格；视觉模型通常需要二维扫描、分层局部特征或结合 Attention。
- 线性时间不保证任何实际短序列都更快，scan 内核与张量并行方式决定吞吐。
- 输入选择机制使其比 S4 更适合离散内容，但不自动带来 PDE 守恒、几何不变性和 OOD 泛化。

## 5. 关键联系

- **数学前作**：[[Literature/architectures/Efficiently Modeling Long Sequences with Structured State Spaces|S4]] → Mamba（LTI → input-selective）。
- **后续数学重写**：[[Literature/architectures/Transformers are SSMs Generalized Models and Efficient Algorithms Through Structured State Space Duality|Mamba-2 / SSD]] 用 semiseparable 矩阵连接选择性 SSM 与结构化 masked attention。
- **后续推理设计**：[[Literature/architectures/Mamba-3 Improved Sequence Modeling using State Space Principles|Mamba-3]] 从状态离散化、复数动力学与 MIMO 改善状态能力。
- **同一问题的替代方案**：[[Literature/architectures/Gated Linear Attention Transformers with Hardware-Efficient Training|GLA]] 用门控矩阵状态保留/遗忘信息，[[Literature/architectures/Retentive Network A Successor to Transformer for Large Language Models|RetNet]] 提供并行和递归两种实现。
- **空间迁移案例**：[[Literature/architectures/MambaVision A Hybrid Mamba-Transformer Vision Backbone|MambaVision]] 需要额外二维层级与高层 Attention，而不只是替换层名称。

参考原文：https://arxiv.org/html/2312.00752
