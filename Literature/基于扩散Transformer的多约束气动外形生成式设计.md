---
title: "基于扩散Transformer的多约束气动外形生成式设计"
title_en: "Multi-Constraint Generative Design of Aerodynamic Shapes Based on Diffusion Transformers"
authors: [张一鸣, 田锋, 马浩元, 胡宁, 覃建秀, 陈思员]
year: 2026
venue: "航空学报（Acta Aeronautica et Astronautica Sinica），网络发表 2026-07-24"
doi: "10.7527/S1000-6893.2026.33671"
url: "https://hkxb.buaa.edu.cn/CN/10.7527/S1000-6893.2026.33671"
source: "未提供本地 PDF；目前按期刊官网公开摘要阅读，缺少可核对的全文参数与实验表"
status: abstract-reviewed
updated: 2026-10-25
tags: [literature, aircraft-generative-design, diffusion-transformer, DiT, missing-conditions, multi-condition, Fourier-embedding]
---

> [!abstract] 摘要译述
> 飞行器概念设计经常只得到部分性能和尺寸指标，不能要求用户每次填写固定完整条件向量。本文提出以扩散 Transformer（Diffusion Transformer，DiT）为核心的连续参数化气动外形生成器：将升阻比、弹径、弹翼根弦长等连续标量条件经多频 Fourier 嵌入编码，再在 token 序列维度上与扩散输入拼接，使条件直接参与 Self-Attention。同时使用联合随机丢弃策略，结合全局条件屏蔽和逐条件独立屏蔽，使推理能够根据任意子集条件生成对应连续外形参数。

# 一句话理解

**本论文研究的是“条件子集不固定，但生成几何参数仍连续且属于给定表示族”的逆向设计。** 它不能直接证明支持翼数、发动机数和连接图等离散拓扑变化；这与跨拓扑扩散的任务不同。

## 1. 模型输入输出

- 输入：初始噪声/带噪连续几何参数、扩散时间步、提供的若干目标条件。
- 条件示例：升阻比 \(L/D\)、弹径和翼根弦长等来自不同量纲的工程标量；在推理时允许不完整。
- 输出：连续参数化的飞行器气动外形变量，并非三维 CAD 实体、直接的网格点云或 CFD 场。
- 核心：Diffusion Transformer 负责去噪并建立几何 token 与条件 token 之间的交互。官网摘要未提供具体 \(\epsilon\)-prediction 或 \(v\)-prediction 头定义，不能仅凭 DiT 名称推定。

## 2. Multi-frequency Fourier Feature Embedder

标准 Fourier 特征编码可写成（**以下是方法族示意，不是论文精确频率取值**）：

\[
\gamma(c)=[\sin(2\pi f_1c),\cos(2\pi f_1c),\ldots,\sin(2\pi f_mc),\cos(2\pi f_mc)].
\]

本文的条件不是全部先相加到扩散时间嵌入，而是成为 token 序列的一部分：

\[
X_{\mathrm{DiT}}
=[X_{\mathrm{geometry\ noise}};\,E(c_1);\ldots;E(c_k)].
\]

这让 Self-Attention 能够根据不同几何变量对不同性能条件分配关注，而不强迫所有条件经一次全局加和后丢失独立性。作者在官方摘要中指出，它避免传统加法式条件融合中条件归一化和时间嵌入的相互干扰。

## 3. Missing-Condition Training

训练时对连续条件随机屏蔽：
- **全局丢弃**：一起移除所有条件，使主干学无条件样本生成行为；
- **逐条件独立丢弃**：随机移除不同标量，使模型经历不同观测子集。

在概念层面，训练允许模型近似学习

\[
p_\theta(x_{\mathrm{shape}}\mid\{c_i:i\in\mathcal O\}),\quad\mathcal O\subseteq\{1,\dots,k\}.
\]

该技术与 classifier-free guidance 有共享的随机无条件训练思想，但本文还增加逐项条件丢弃；不能直接把二者当作同一个算法。论文声称推理阶段可使用任意数量与组合的给定条件，并产出匹配目标的外形参数，但还需要全文的成功率和真值复核才能评价在严格工程公差下的能力。

## 4. 与已读论文的区别

| 方法 | 主要可变对象 | 已解决的问题 |
|---|---|---|
| [[Literature/Range-GAN Range-Constrained Generative Adversarial Network for Conditioned Design Synthesis|Range-GAN]] | 连续属性的区间 | 尽量生成落在允许范围内并覆盖区间 |
| 本文：多约束 DiT | 条件观测的子集 | 用户仅给部分性能/尺寸指标也能生成 |
| [[Literature/transformer/Do Diffusion Models Dream of Electric Planes Discrete and Continuous Simulation-Based Inference for Aircraft Design|MixeDiT–MaskeDiT]] | 离散拓扑与变长几何参数 | 条件如何影响组件数量、拓扑选择 |
| [[Literature/GLUE Coordinating Pre-Trained Generative Models for System-Level Design|GLUE]] | 已训练组件 latent 的组合 | 组件接口、性能约束和整机多解如何协调 |

对整机模型，可考虑把任务指标作为条件 token、把组件存在性作为 topology mask、把每个组件的连续参数交给可微几何解码器，然后用代理验证。这只是拟议组合，**不是本文已经证明的端到端系统**。

## 5. 实验与证据边界

官方期刊页只提供元数据与较详细的中英文摘要，目前尚不足以核对：
- 数据量、具体连续几何参数维数与训练划分；
- Transformer 深度、宽度、attention heads、噪声日程、条件 token 的类型编码；
- 全局和逐项条件 dropout 的概率；
- 固定条件/部分条件下的逐指标 MAE、约束联合满足率和真实 CFD 复核量；
- 与 adaLN、FiLM、简单 concat、条件 U-Net 的公平配对消融。

所以本笔记状态是 **abstract-reviewed（摘要已核实，全文待补）**，只总结官网明确支持的模块与机制，不填入未经核实的定量成绩。

## 6. 值得后续全文深读的问题

1. 条件缺失时是 mask、null token 还是实际删去 token？
2. 是否在训练中遇到所有可能的条件组合？未见组合能否外推？
3. 条件类型、数值尺度、时间步嵌入在 LayerNorm 和 Self-Attention 中如何保留区分？
4. 生成参数是否满足硬的几何/物理约束？有无独立高保真验证？
5. 如果输出改成可变组件集合，能否与拓扑扩散形成一致的生成接口？

## 原文入口

[《航空学报》论文页、完整公开摘要、DOI](https://hkxb.buaa.edu.cn/CN/10.7527/S1000-6893.2026.33671)
