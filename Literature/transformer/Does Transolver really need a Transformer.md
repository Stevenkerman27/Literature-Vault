---
title: "Does Transolver really need a Transformer?"
authors:
  - Shizheng Wen
  - Siddhartha Mishra
year: 2026
venue: "arXiv preprint, cs.LG, v1 (2026-09-26)"
doi: null
arxiv: "2609.32525v1"
url: "https://arxiv.org/abs/2609.32525"
code: "https://github.com/Shizheng-Wen/flashslice"
source: "未提供本地 PDF；本笔记依据 arXiv 原文 HTML，原文 PDF 见 https://arxiv.org/pdf/2609.32525"
tags:
  - literature
  - neural-operator
  - transformer
  - physics-attention
  - linear-attention
  - PDE
  - scientific-machine-learning
  - aerodynamics
status: read
updated: 2026-10-24
---

> [!abstract] 摘要翻译
> 本文对 Transolver 的 Physics-Attention 进行系统的实证与理论分析。作者在九个三维流体与工业气动基准上发现，将 slice token 之间的 self-attention 替换为逐 token 的固定线性映射，预测误差基本不变；但移除 slice/deslice 全局混合，或只在网络开头执行一次 slice/deslice，都会显著恶化结果。作者利用 averaging neural operator 的通用逼近理论解释这种现象，并提出 FlashSlice：一种类似 FlashAttention 的融合内核，在不显式存储完整 slice-weight 张量的情况下流式执行 slice/deslice，可降低显存并加速训练。

# 一句话理解

**Transolver 的关键可能不是 slice token 之间的 Transformer，而是每层反复执行的全局点—slice—点信息通路，以及这条通路与全分辨率点特征和逐点 MLP 的交替。** 论文没有证明“attention 都没用”，而是区分了两件事：slice token 的自注意力可以删，点与 slice 之间的全局混合不能轻易删。

![论文中图 1：Transolver 机制消融及九个基准上的误差变化](https://arxiv.org/html/2609.32525v1/F1_money.png)

*图 1。作者对比完整 Transolver、去除 token self-attention、移除整个全局混合、解绑投影、冻结 slice 权重和只做一次 slice/deslice 的结果。原文图直接嵌入。*

## 研究问题与方法视角

原始 Transolver 的 Physics-Attention 每层执行以下操作：

1. 从当前点特征计算每个点属于 \(G\) 个 slice 的软权重。
2. 按权重对全分辨率点特征加权平均，形成 \(G\) 个 slice token。
3. 在 slice token 之间执行 self-attention。
4. 把更新后的 slice token 广播回所有点，并与原点特征做残差连接。
5. 在全分辨率点特征上执行逐点两层 MLP，再进入下一层。

关键结构有两个：slice 权重每层根据当前特征重新计算；同一组权重同时用于 slice 和 deslice。点级表示并不会永久压缩到 \(G\) 个 token，而是贯穿整个网络。

## 理论视角：slice/deslice 是低秩非局部算子

在连续域中，可以把 slice 权重视为一组数据依赖的函数 \(s_g(x)\)，满足 \(\sum_g s_g(x)=1\)。slice 先对全域特征积分聚合，deslice 再把聚合结果映射回空间位置。因此一层可写成类似：

\[
(\mathcal{K}v)(x)=
\sum_{g,h=1}^{G}
s_g(x) A_{gh}
\frac{\int_\Omega s_h(y)v(y)\,d\mu(y)}
{\int_\Omega s_h(y)\,d\mu(y)}.
\]

其中 \(A\) 是作用于 slice token 的核心映射。完整 Transolver 中，\(A\) 由 token self-attention 产生；NoTokenAttn 变体将其替换为常数核心和逐 token 的可学习通道映射，token 之间不再互相读取，但所有点仍可经由 slice/deslice 交换全局信息。

这一区分非常重要：**删除 token self-attention 并没有删除非局部性。** 每个点的输出仍然依赖全域点对 slice 汇总的贡献，再通过 deslice 读取这些汇总量。

作者借助 averaging neural operator（ANO，平均神经算子）的通用逼近结果说明：在目标算子连续、输入集合紧致、逐点网络宽度可任意增大的理论条件下，只有全局平均和逐点编码/解码也能逼近连续算子。因此，token self-attention 并不是实现通用算子逼近的必要条件。该理论不保证有限宽度、有限数据和有限训练预算下的误差，也不能直接推出具体基准上 attention 删除后的误差幅度。

## 消融设计

论文在 Transolver 的统一主干上逐一改变模块，尽量隔离每项设计的作用。

| 变体 | 修改 | 检验的问题 |
|---|---|---|
| Baseline | 原始 Physics-Attention | 参照模型 |
| NoTokenAttn | token self-attention 替换为逐 token 线性映射 | token 之间的内容相关交互是否必要 |
| MlpOnly | 删除整个 slice/deslice 子层，只保留逐点残差 MLP | 仅靠逐点网络能否替代全局通信 |
| MlpOnly-Wide | 加宽 MlpOnly，使参数量达到基线的 112% | MlpOnly 的损失是否仅由容量不足造成 |
| Untied | deslice 使用独立投影与温度 | slice/deslice 权重共享是否必要 |
| FrozenSlice | 所有层复用第一层的 slice 权重 | 是否必须每层重新计算 slice 权重 |
| SliceOnce | 只 slice 一次，随后在压缩 token 上运行 Transformer，最后 deslice 一次 | 是否能像 latent bottleneck 模型那样永久压缩点表示 |

主干使用 8 层、隐藏宽度 256、8 个 head、每个 head 32 个 slice、MLP ratio 为 2、GELU 激活且不使用 dropout，完整模型约 3.87M 参数。所有变体使用同一数据流程和相同训练步数；Baseline 与 NoTokenAttn 使用三个随机种子，其余消融通常使用一个种子。指标为最佳验证集相对 \(L^1\) 误差，按样本和输出组归一化后平均，越低越好。

## 实验设置

九个数据集配置涵盖瞬态流场和工业气动：

- **Taylor–Green vortex**：三维涡旋演化，输入包含当前时刻的局部场状态，预测未来状态。
- **SHIFT-Wing surface / volume**：跨几何与变化来流条件的跨声速机翼表面或体积场预测。
- **DrivAerNet++ surface / volume**：固定来流条件下的汽车表面或体积场。
- **SHIFT-SUV surface / volume**：固定工况，主要考察几何变化对表面或体积场的影响。
- **DrivAerML surface / volume**：工业汽车气动，单个体积样本可达约 \(1.3\times10^8\) 个点。

预测输出包括压力、速度向量或壁面剪切向量。对大网格数据，训练时每次取随机点子集；作者测得评估指标受子采样影响的相对变化不超过 0.18%，远小于各架构消融带来的误差变化。

## 主要结果：准确率来自哪里？

### 1. 删除 token self-attention：基本不变

| 基准 | Baseline | NoTokenAttn | MlpOnly | SliceOnce |
|---|---:|---:|---:|---:|
| Taylor–Green | 0.0756 | **0.0745** | 0.0786 | 0.188 |
| SHIFT-Wing surface | 0.0580 | 0.0579 | 0.139 | 0.155 |
| SHIFT-Wing volume | 0.127 | 0.127 | 0.408 | 0.793 |
| DrivAerNet++ surface | 0.193 | **0.188** | 0.294 | 0.435 |
| DrivAerNet++ volume | 0.163 | **0.159** | 0.315 | 0.386 |
| SHIFT-SUV surface | 0.157 | 0.157 | 0.252 | 0.449 |
| SHIFT-SUV volume | 0.0601 | 0.0624 | 0.197 | 0.408 |
| DrivAerML surface | 0.089 | 0.096 | 0.501 | 0.579 |
| DrivAerML volume | 0.110 | 0.112 | 0.497 | 0.745 |

NoTokenAttn 在七个配置上相对完整模型的变化位于 −2.6% 到 +1.8% 之间，另两个配置的误差分别高 3.8% 与 7.9%。四个配置中 NoTokenAttn 是最优或并列最优。这表明在这些任务和训练预算下，token 之间的内容相关 self-attention 并非主要性能来源。

### 2. 删除全局混合：误差显著恶化

MlpOnly 的误差达到完整模型的 1.04–5.6 倍。即使将逐点 MLP 加宽到基线参数量的 112%，也只能恢复极少的性能差距。在 DrivAerNet++ 上，扩大后的 MlpOnly 仍分别比完整模型差约 1.6 倍（表面）和 1.9 倍（体积）。

这符合气动问题的非局部特征：某一点的压力不仅由该点局部几何描述决定，还受整机几何、来流和全局流场相互作用影响。论文进一步发现，移除全局混合时，压力误差通常比速度或壁面剪切误差恶化更明显。

### 3. 只在开头做一次 slice/deslice：最差

SliceOnce 的误差达到完整模型的 2.2–6.8 倍，且在九个配置中均明显变差。它保留了 token self-attention，参数量甚至更大，却把后续网络深度限制在压缩后的 \(G\) 个 token 中，失去全分辨率点流与每层反复全局交互的能力。

作者从理论上将这解释为非局部信息预算的差异：每层都进行 slice/deslice 时，各层可以基于更新后的特征重新汇总信息；SliceOnce 则只通过一次 \(G\times c\) 的压缩瓶颈传递全局信息。

### 4. 是否必须每层重算 slice 权重？

FrozenSlice 复用第一层的 slice 权重，误差代价从 −0.5% 到 +17%，主要在体积场任务上变大。因此论文支持“全局混合要反复发生”，但没有证明“每层都必须重新学习一套 slice 分配”在所有任务上都是必要条件。

## 与 LinearNO 的关系：一个关键分歧

Hu 等人的 [[Literature/transformer/Transolver Is a Linear Transformer Revisiting Physics-Attention Through the Lens of Linear Attention|LinearNO]] 认为，解绑 Q/K 投影之后，slice-token self-attention 可以删除。Wen 与 Mishra 进一步在 SHIFT-SUV surface 上做了“是否使用 token attention × 是否共享投影”的 2×2 控制实验：

- 四种配置误差差异不超过 1.05%。
- 删除 attention 的变化约为 +0.54%。
- 解绑投影的变化约为 −0.51%。
- 两因素交互约为 −0.41%。

这些变化与 Baseline 三个种子的波动幅度（约 ±0.47%）相近。作者据此认为，在该实验中，解绑投影并不是 attention 可删除的必要条件。它不是严格的统计等效性检验，也不能证明两种方案在所有数据集、宽度和训练设置下完全相同。

作者还报告，在其对照实现中，LinearNO 使用沿点维的额外 softmax 归一化；在发布的 latent width 设置下，每次训练迭代约慢 6.8 倍，但准确率没有变化。这个时间结论只适用于论文给定实现和配置，不能泛化到所有硬件与实现。

## FlashSlice：优化全局混合的实现成本

论文同时提出 FlashSlice，用 Triton 融合内核流式执行 slice/deslice，不把完整 \(N\times G\) 的权重张量写入高带宽显存（HBM）。内核在点块中临时计算 membership 权重，并在反向传播中重算，换取更低显存占用。作者报告：

- 标准配置训练加速约 1.35–1.7 倍，显存最多减少 25%。
- 大 slice 数设置下训练可加速约 4–6.5 倍，显存减少 85–96%。
- 两种内核路径均不使用原子加法，作者报告其结果位级确定，并通过数值对照验证输出与参数梯度。

这项工程贡献与“attention 是否必要”是两个不同问题：即使 token self-attention 可以删除，slice/deslice 仍然是主要全局混合路径，其内存读写依然值得优化。

## 关键贡献

1. 在九个三维流体与工业气动配置上系统消融 Transolver 的三个组成部分：slice/deslice、token self-attention、全分辨率逐点 MLP。
2. 实证指出 token self-attention 可被逐 token 线性映射替代，而全局 slice/deslice 及其与点流的交替是关键。
3. 用 averaging neural operator 的理论解释 attention-free 结构的通用逼近能力，并解释逐点 MLP 和只压缩一次的瓶颈为何不足。
4. 提出 FlashSlice 内核，降低 slice/deslice 的内存占用与运行时间。

## 局限与证据边界

- 这是一篇 2026 年 9 月提交的 arXiv 预印本，不应与经过长期独立复现的结论等同。
- Baseline 和 NoTokenAttn 报告三个种子的均值与标准差，其他消融大多只有单种子；一些差距较小的比较应谨慎解释。
- 通用逼近定理依赖连续算子、紧致输入集和无界宽度等理论条件，不直接保证实际有限模型的泛化误差。
- 九个配置虽包含多个工业气动任务，但并不覆盖所有 PDE 家族、所有网格分布、强瞬态多物理场或真实飞行试验。
- 结论是“slice token 之间的 self-attention 在这些实验中可删”，不是“全局交互、几何耦合或所有形式的 attention 都无用”。

## 与相关工作的关系

- [[Literature/transformer/Transolver|Transolver]]：原始 Physics-Attention 与每层的点—slice—点全局通信。
- [[Literature/transformer/Transolver Is a Linear Transformer Revisiting Physics-Attention Through the Lens of Linear Attention|Transolver Is a Linear Transformer / LinearNO]]：先解绑 Q/K，再移除 token self-attention；本文对“解绑是否必要”提出不同的消融证据。
- [[Literature/Topics/Transformer与物理场算子|Transformer 与物理场算子]]：把物理切片、低秩非局部算子、局部特征和查询解码放在同一架构谱系中比较。

## 参考资料

- Wen and Mishra, *Does Transolver really need a Transformer?*, arXiv:2609.32525, 2026. [论文](https://arxiv.org/abs/2609.32525) · [FlashSlice 代码](https://github.com/Shizheng-Wen/flashslice)
- Hu et al., *Transolver Is a Linear Transformer: Revisiting Physics-Attention Through the Lens of Linear Attention*, AAAI 2026. [论文](https://arxiv.org/abs/2511.06294)
- Lanthaler, Li and Stuart, *Nonlocality and Nonlinearity Implies Universality in Operator Learning*, 2023.
