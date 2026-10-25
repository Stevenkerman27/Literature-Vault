---
title: "Hyena Hierarchy: Towards Larger Convolutional Language Models"
authors: [Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y. Fu, Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon, Christopher Ré]
year: 2023
venue: "ICML 2023, PMLR 202:28043–28078"
doi: null
arxiv: "2302.10866"
url: "https://arxiv.org/abs/2302.10866"
source: "未提供本地 PDF；原文 https://arxiv.org/pdf/2302.10866"
tags: [paper, Hyena, convolution, long-range, gated-convolution, sequence-model]
status: read
updated: 2026-10-25
---

> [!abstract] 摘要译述
> 标准 Attention 的二次复杂度使长上下文昂贵。Hyena 用隐式参数化的长卷积（long convolution）与数据驱动门控交替构造次二次复杂度的 token mixer，力图在没有显式稠密 Attention 的情况下保持长程推理能力。作者研究合成记忆/归纳任务、语言建模和高分辨率视觉分类，在 8K、64K 等长度下报告计算收益，并展示对多种长卷积参数化与门控深度的消融。

# 核心思想

**Hyena 不通过保存全部 Query–Key 相似度进行全局交互，而是用固定/隐式生成的长卷积滤波器对完整序列混合，再用当前位置的特征对卷积结果进行输入依赖门控。** 这是“长卷积 + 门控”路线，与 S4（SSM 生成卷积核）既相似又不同。

## 1. Hyena Operator

给定长度 \(L\) 的输入序列 \(x\)，先用多路线性投影产生数据控制信号 \(v,x^{(1)},x^{(2)},\ldots\)。交替执行：

\[
z^{(0)}=v,\qquad
z^{(i+1)}=x^{(i)}\odot(h^{(i)}*z^{(i)}),
\]

其中 \(*\) 表示全序列卷积，\(\odot\) 为逐元素 gating，\(h^{(i)}\) 为由隐式网络产生的长滤波器。这是一种**概念性简化写法**：原文还包含短卷积、顺序安排、位置编码、门控和 output projection 等实现细节。

长卷积核通常由隐式 MLP 按相对位置索引生成，而不需直接训练长度为 \(L\) 的稠密向量。计算卷积可以使用 FFT，量级通常为 \(O(L\log L)\)；即使不显式构造 \(L^2\) 注意力矩阵，所需滤波器仍然依赖序列长度。

## 2. 为什么数据门控重要？

纯 LTI 长卷积会把相同卷积滤波器作用于整个序列，缺乏“遇到当前 token 时按内容选择”的能力。Hyena 以多阶门控叠加序列传播，每一层都可以按当前特征调节经过长卷积的信号；与 [[Literature/architectures/Mamba Linear-Time Sequence Modeling with Selective State Spaces|Mamba]] 的输入相关 SSM 参数不同，Hyena 主要通过门控调制长卷积路径。

## 3. 实验设置与结果

- **合成回忆/推理**：长达数千至数十万 token 的合成序列，比较原始长卷积、状态空间模型和 Attention。
- **语言**：WikiText-103、The Pile 的语言模型预训练/评估，比较训练 compute 与 perplexity。
- **高分辨率图像分类**：探索长卷积主干在图像序列化后的有效性。
- 论文报告在其合成回忆任务中比部分 SSM/隐式序列算子提升超过 50 个准确率百分点；在 2K 语言序列上以约低 20% 的训练计算量接近 Transformer；在 8K/64K 等特定算子测速中，比优化 Attention 分别快约 **2× / 100×**。这些并非全模型在任意任务/硬件上的通用速度结论。

## 4. 与 S4、Mamba、Attention 的区别

| 结构 | 全局混合途径 | 长程内容选择机制 |
|---|---|---|
| [[Literature/architectures/Efficiently Modeling Long Sequences with Structured State Spaces|S4]] | 时不变 SSM 对应的卷积 | 没有输入依赖的核心状态系数 |
| Hyena | 隐式生成的长卷积 | 多级逐位置乘法门控 |
| [[Literature/architectures/Mamba Linear-Time Sequence Modeling with Selective State Spaces|Mamba]] | 输入依赖状态更新 | \(\Delta_t,B_t,C_t\) 选择性 |
| Standard Attention | Query–Key 对每个历史 token 的相似度 | Query 内容直接检索历史 |

## 5. 对物理场和工程生成的边界

- 长卷积可用于规则时间/空间网格序列，而非自动支持任意非结构点云。将空间网格序列化会引入人为顺序、位置信息和边界跨越问题。
- FFT 效率取决于硬件、卷积长度和 batch；本论文的语言基准不等于气动误差验证。
- 对有冲击波/尖锐边界的物理场，长卷积可能平滑局部极值，需要设计局部分支和保真度校验。

### 关联

[[Literature/architectures/MambaVision A Hybrid Mamba-Transformer Vision Backbone|MambaVision]] 选择在低层保留 CNN 局部归纳偏置；与 Hyena 一样说明效率不只来自稀疏化注意力。[[Literature/transformer/Mixture-of-Top-k Attention Efficient Attention via Scalable Fast Weights|MiTA]] 保留少量原始 KV 以精细检索，而 Hyena 不显式读取 KV 列表。

原文：https://proceedings.mlr.press/v202/poli23a.html
