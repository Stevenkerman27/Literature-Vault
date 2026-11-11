---
title: "Does Transolver really need a Transformer?"
authors: ["Shizheng Wen", "Siddhartha Mishra"]
year: 2026
venue: "arXiv preprint, arXiv:2609.32525"
doi: ""
tags: [transolver, neural-operator, averaging-neural-operator, universal-approximation, attention]
status: read
source: "未提供本地 PDF；https://arxiv.org/abs/2609.32525"
url: "https://arxiv.org/abs/2609.32525"
arxiv: "2609.32525"
---

# 核心摘要

> [!abstract] 原文摘要翻译
> Transolver 系列神经算子使用 physics-attention：将非结构网格点软分配到少量 slice token，在 token 之间做 self-attention，再将结果广播回网格点。作者在九个三维流体基准上进行消融，发现把 token attention 替换为常数线性映射，精度几乎不变；移除全局 slicing/deslicing，或只执行一次，则性能显著下降。作者借助 averaging neural operator 理论证明：slicing/deslicing 与逐点 MLP 已足以通用逼近连续算子，attention 在该架构中不是通用逼近所必需的。论文还提出 FlashSlice 降低显存和计算成本。

## 1. 问题与动机

Transolver 中包含三种机制：全分辨率点流、slice/deslice 全局耦合、slice token 间的 attention。以往改进常集中于 attention 变体，但没有清楚分离哪种机制真正贡献精度。本文通过结构消融与算子理论分析回答这一问题。

## 2. 数学结构

将每层的 physics-attention 看作低秩非局部积分算子。slice 权重构成依赖当前特征的 partition of unity；每个 slice 聚合点特征，再经 token mixing，最后把聚合信息广播回原始点流。关键不是 token 间一定要做复杂 attention，而是每层都要在局部点表示与全局聚合表示之间交换信息。

论文借助 Averaging Neural Operator（ANO）通用逼近定理：对有界 Lipschitz 区域上的连续算子，在紧输入集合上，存在足够宽的 pointwise lift 网络 R、全局平均编码和 pointwise decoder Q，使算子误差任意小。证明分两步：
1. 用有限个连续泛函对目标算子做有限秩展开；
2. 用对输入/坐标的逐点非线性提升后做全局平均，近似这些泛函。
解码器 Q 在全分辨率点上逐点作用，坐标信息可通过残差流保留。

在 Transolver 的特例中，slice logits 取相同值即可使各 slice 表示相同的全局平均；常数 token mixing 就能退化为 ANO 耦合。因此通用逼近不依赖 token self-attention。相反，若完全移除 slice/deslice，模型只剩 pointwise MLP，无法逼近一般非局部算子；若只做一次聚合后在 token 空间深层计算，则缺少每层持续的全分辨率 pointwise decoder，表达效率受限。

## 3. 实验结果

论文在九个 3D 流体基准上比较完整模型和消融模型：
- 将 token self-attention 替换为常数线性映射，误差相对完整模型约在 -2.6% 到 +7.9% 范围内；
- 移除 slice/deslice，误差增加约 1.04–5.6 倍；
- 只做一次 slice/deslice，误差增加约 2.2–6.8 倍。
- FlashSlice 把 slice/deslice 融合成流式 kernel，标准配置下训练约快 1.35–1.7 倍，显存最多降低约 25%；大 slice 数配置下速度约快 4–6.5 倍、显存降低约 85–96%。

## 4. 局限与适用边界

- 这是**物理场算子**的表达能力结论，不是离散飞机拓扑或生成分布的通用逼近定理。
- ANO 的理论宽度 \(J(\epsilon)\) 随精度要求可能无界；定理不保证 GRASS 当前固定 slice 数和有限 hidden size 已处于有效逼近区间。
- 九个 benchmark 的结果不能直接证明 GRASS 的整机 Transolver 在不同布局、不同翼面数、不同物理标签质量下也有同样结论。
- 本文没有证明 attention 对所有数据集都无用；它证明的是在所研究的 Transolver 结构与任务中，attention 可被常数 mixing 替代而不明显损害精度。

## 5. 对 GRASS 的启发

这篇论文直接指导的是 GRASS 的**气动代理**而非生成器：应把 slice/deslice 的跨面全局耦合、全分辨率点流与 token mixing 分开消融；不要仅因模型叫 Transolver 就继续增加 attention 复杂度。对生成器的间接启发是方法论：识别承担表达能力的必要算子，再证明其充分性，避免把 Transformer 当作默认答案。

## 6. 关联

- [[Literature/Topics/Transformer与物理场算子]]
- [[Literature/Topics/跨布局飞行器生成与新型网络架构]]
- [[Literature/architectures/Universal Approximation of Functions on Sets]]
- [[Literature/architectures/Are Transformers Universal Approximators of Sequence-to-Sequence Functions]]
