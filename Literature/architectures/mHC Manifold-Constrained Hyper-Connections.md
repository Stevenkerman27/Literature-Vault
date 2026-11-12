---
title: "mHC: Manifold-Constrained Hyper-Connections"
authors:
  - Zhenda Xie
  - Yixuan Wei
  - Huanqi Cao
  - Chenggang Zhao
  - Chengqi Deng
  - Jiashi Li
  - Damai Dai
  - Huazuo Gao
  - Mingyu Xu
  - Kuai Yu
  - Liang Zhao
  - Shangyan Zhou
  - Zhean Xu
  - Zhengyan Zhang
  - Wangding Zeng
  - Shengding Hu
  - Yuqing Wang
  - Jingyang Yuan
  - Lean Wang
  - Wenfeng Liang
year: 2026
venue: "ICML 2026; Proceedings of Machine Learning Research 306"
doi: "10.48550/arXiv.2512.24880"
arxiv: "2512.24880"
url: "https://arxiv.org/abs/2512.24880"
source: "未提供本地 PDF；原文 https://arxiv.org/pdf/2512.24880"
tags: [paper, LLM, residual-connection, hyper-connections, manifold, Birkhoff-polytope, architecture-topology]
status: read
updated: 2026-11-11
---

> [!abstract] 摘要译述
> Hyper-Connections（HC）通过扩展残差流宽度和增加连接方式提升模型表达能力，但无约束的残差映射会破坏标准残差连接的恒等映射性质，造成大规模训练不稳定和扩展困难，同时增加显存访问开销。本文提出 Manifold-Constrained Hyper-Connections（mHC），把 HC 的残差映射投影到特定流形上，以恢复恒等映射相关的稳定性质，并结合 Kernel Fusion、重计算和流水线通信重叠降低系统开销。作者在语言模型预训练实验中展示了更好的稳定性和可扩展性。

# 核心定位

**mHC 属于网络宏观拓扑（Macro-Architecture）创新：它研究不同层、不同残差流之间如何传播和混合信息，而不是改变 Attention 或 FFN 内部的计算。**

## 1. 问题与动机

标准残差连接为：

\[
x_{l+1}=x_l+F(x_l,W_l).
\]

恒等项 x_l 使浅层信号能够直接传递到深层，避免每一层都必须重新学习完整的信息通路。

Hyper-Connections（HC）把残差流扩展为 n 条并行流，并使用三个可学习映射：

- H_pre：把 nC 维残差流聚合为当前层函数 F 的 C 维输入。
- H_post：把 F 的输出重新写回 n 条残差流。
- H_res：直接混合不同残差流。

问题在于 H_res 是无约束矩阵。多层复合后，残差流的总映射可能偏离恒等映射，造成前向信号放大或衰减，以及反向梯度不稳定。论文在 27B 模型中观察到 HC 的训练 Loss 和梯度范数出现明显不稳定，复合映射的最大增益接近 3000。

此外，n 条残差流会增加读写和流水线通信成本。即使 FLOPs 增长有限，内存访问也可能成为瓶颈。

## 2. 方法

### 2.1 将残差映射约束到双随机矩阵流形

mHC 的核心是要求残差映射 H_res 满足：

\[
H_{\mathrm{res}}\mathbf 1=\mathbf 1,\qquad
\mathbf 1^\top H_{\mathrm{res}}=\mathbf 1^\top,\qquad
H_{\mathrm{res}}\geq 0.
\]

即每个元素非负，行和与列和均为 1。这样的矩阵称为双随机矩阵，其集合构成 Birkhoff Polytope（Birkhoff 多面体）。

双随机约束带来三项性质：

1. **非扩张性**：谱范数不超过 1，降低残差映射不断放大信号的风险。
2. **复合封闭性**：两个双随机矩阵相乘仍为双随机矩阵，因此跨多层的残差流映射保持约束。
3. **凸组合解释**：双随机矩阵可视为置换矩阵的凸组合，残差流在保持稳定性的同时仍可交换和混合信息。

当 n=1 时，双随机约束退化为标量 1，恢复标准恒等残差映射。

### 2.2 参数化与 Sinkhorn-Knopp 投影

模型先根据多流输入计算动态映射和静态 Bias，得到未约束矩阵 H_res。随后通过指数映射保证元素为正，再使用 Sinkhorn-Knopp 迭代交替归一化行和列：

\[
M^{(0)}=\exp(\widetilde H_{\mathrm{res}}),\qquad
M^{(t)}=T_r(T_c(M^{(t-1)})).
\]

迭代后得到近似双随机矩阵。论文实践中使用 20 次迭代。

此外，H_pre 和 H_post 也施加非负约束，减少正负系数混合导致的信号抵消。

### 2.3 系统实现

mHC 的数学约束会带来额外 Kernel 和内存访问开销，因此论文同时设计：

- **Kernel Fusion**：融合 RMSNorm、映射计算和残差合并，减少重复读取和 Kernel Launch。
- **Recomputing**：前向后不保留部分 mHC 中间激活，在反向时重新计算，降低显存峰值。
- **DualPipe 通信重叠**：调整流水线阶段边界和执行流，让残差流跨 Stage 通信尽可能与计算重叠。
- **TileLang**：用于实现专用混合精度 Kernel。

作者报告扩展率 n=4 时，经过系统优化后，额外训练时间开销约 6.7%。

## 3. 实验设置与结果

- **模型架构**：使用受 DeepSeek-V3 启发的 MoE 语言模型。
- **主要规模**：27B 参数模型，HC 与 mHC 的残差流扩展率均为 n=4。
- **Scaling 实验**：另训练 3B、9B 模型，训练数据量与参数规模成比例；另用 3B 模型在固定 1T Token 语料上研究 Token Scaling。
- **基线**：标准 Residual、Hyper-Connections（HC）和 mHC。
- **下游基准**：BBH、DROP、GSM8K、HellaSwag、MATH、MMLU、PIQA、TriviaQA。
- **27B 结果**：mHC 在多数基准上超过 HC 和标准残差基线；例如 BBH 为 51.0，HC 为 48.9，Baseline 为 43.8；DROP 为 53.9，HC 为 51.6，Baseline 为 47.0。
- **稳定性**：mHC 的 Loss 和梯度范数更稳定；作者报告其复合映射最大增益约 1.6，而 HC 接近 3000。
- **系统开销**：n=4 时额外训练时间约 6.7%。

## 4. 关键贡献

1. 将残差流混合矩阵约束到双随机矩阵流形，在允许信息交换的同时恢复稳定的恒等映射性质。
2. 把网络拓扑的数学约束与 Kernel Fusion、重计算、流水线通信重叠联合设计。
3. 在 27B 规模实验中同时考察 Loss 稳定性、梯度范数、下游任务和系统开销。
4. 将架构研究从 Attention / FFN 等模块内部设计扩展到层间残差流的宏观连接拓扑。

## 5. 局限与启发

- Sinkhorn-Knopp 使用有限迭代次数，实践中得到的是近似双随机矩阵；约束误差会影响理论性质的精确程度。
- 多流残差会增加状态读写、参数和通信成本；6.7% 开销依赖专门的系统优化，不能假设未经优化的实现也有相同效率。
- 主要实验采用 DeepSeek 风格 MoE 语言模型；是否同样适用于 Dense Transformer、视觉模型或不同深度 / 宽度配置，需要独立验证。
- 双随机约束保证的是非扩张和混合性质，不等于所有信息都无损保留；过强混合可能影响不同流的独立表征能力。
- 对项目研究而言，应将残差流拓扑与 Attention / FFN 创新分开消融，测量梯度传播、训练稳定性、额外 I/O 和任务性能。

## 6. 关联

- **直接继承**：Hyper-Connections（HC）提出多流残差结构，mHC 在其残差映射上增加流形约束。
- **理论基础**：ResNet 的 Identity Mappings 解释标准残差连接为什么有助于深层信号传播；mHC 的目标是在多流结构下恢复类似性质。
- **模型应用**：[[Literature/LLM/DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence|DeepSeek-V4]] 将 mHC 用于大规模 Transformer 主干。
- **机制区分**：[[Literature/LLM/DeepSeekMoE Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models|DeepSeekMoE]] 改变 FFN 内部的参数路由；mHC 改变层间残差流拓扑。二者位于不同架构层次，可组合而非互相替代。

## 7. 图片与原件

仓库当前未提供本地 PDF，因此暂不嵌入论文截图。建议查看原文 Figure 1（Residual / HC / mHC 对比）、Figure 2（训练稳定性）和 Figure 5（27B 下游结果）。
