---
title: "Are Transformers Universal Approximators of Sequence-to-Sequence Functions?"
authors: ["Chulhee Yun", "Srinadh Bhojanapalli", "Ankit Singh Rawat", "Sashank J. Reddi", "Sanjiv Kumar"]
year: 2020
venue: "International Conference on Learning Representations (ICLR)"
doi: ""
tags: [transformer, universal-approximation, permutation-equivariance, sequence-modeling]
status: abstract-reviewed
source: "未提供本地 PDF；https://openreview.net/forum?id=ByxRM0Ntvr"
url: "https://research.google/pubs/are-transformers-universal-approximators-of-sequence-to-sequence-functions/"
---

# 核心摘要

> [!abstract] 原文摘要翻译
> 尽管 Transformer 已广泛用于自然语言处理，其表达能力仍需要理论刻画。作者证明：Transformer 能通用逼近具有紧支撑的连续排列等变序列到序列函数。进一步加入位置编码后，模型可在紧致域上通用逼近任意连续序列到序列函数。证明还揭示 self-attention 与逐位置前馈网络在通用逼近中的不同作用：固定宽度的 self-attention 层能够计算上下文映射，是通用逼近的重要组成。

## 1. 问题与动机

Transformer 的经验成功并不自动说明它适合任意结构化生成任务。本文研究的是给定序列输入、输出同长度序列的连续映射，以及在置换输入时输出相应置换的等变性质。

## 2. 方法与理论结论（摘要级）

论文给出 Transformer 的通用逼近结论：
- 无位置编码时，模型对应排列等变的序列到序列映射；
- 使用位置编码后，可解除该结构限制，在紧致输入域上逼近一般连续序列映射；
- self-attention 负责构造上下文相关映射，FFN 负责逐位置非线性变换，两者作用不同。

本笔记目前只核对了公开摘要和论文元数据，尚未核对完整证明中的层数、宽度、头数、激活函数等技术假设，因此不把摘要结论扩写成更强的无条件定理。

## 3. 实验与证据边界

公开摘要将贡献定位为表达能力理论分析，并提及基于证明启发的替代架构实验；当前尚未核验论文全文中的具体实验设置与数字。

## 4. 局限

- 该定理说明在特定假设下存在足够强的 Transformer，不说明常用小模型在有限数据下容易训练。
- 位置编码会把排列等变问题变成顺序敏感问题；对物理上无角色的翼面，随意使用绝对 slot embedding 可能引入不必要的编号依赖。
- 通用逼近不保证对未见布局组合的 compositional generalization。

## 5. 对 GRASS 的启发（项目推论，非论文结论）

GRASS 必须先决定 wing list 是“有语义的序列”还是“无语义的集合”。如果 wing 顺序只是数据存储顺序，模型应当具有排列等变性，或通过 canonicalization/对称损失严格处理；不能仅依靠位置编码的通用逼近能力。若 ADD_WING 的生成顺序被定义为真正的自回归决策顺序，则顺序是生成过程的一部分，但最终物理结果仍应对同一组翼面的重排不变。

## 6. 关联

- [[Literature/architectures/Universal Approximation of Functions on Sets]]
- [[Literature/Topics/跨布局飞行器生成与新型网络架构]]
- [[Literature/GraphRNN Generating Realistic Graphs with Deep Auto-regressive Models]]
