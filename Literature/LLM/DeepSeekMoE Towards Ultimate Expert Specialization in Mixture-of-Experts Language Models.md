---
title: "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models"
authors:
  - Damai Dai
  - Chengqi Deng
  - Chenggang Zhao
  - R.X. Xu
  - Huazuo Gao
  - Deli Chen
  - Jiashi Li
  - Wangding Zeng
  - Xingkai Yu
  - Y. Wu
  - Zhenda Xie
  - Y.K. Li
  - Panpan Huang
  - Fuli Luo
  - Chong Ruan
  - Zhifang Sui
  - Wenfeng Liang
year: 2024
venue: "ACL 2024 (Long Papers)"
doi: "10.18653/v1/2024.acl-long.70"
arxiv: "2401.06066"
url: "https://arxiv.org/abs/2401.06066"
source: "未提供本地 PDF；原文 https://arxiv.org/pdf/2401.06066"
tags: [paper, LLM, MoE, conditional-computation, expert-specialization, sparse-FFN]
status: read
updated: 2026-11-11
---

> [!abstract] 摘要译述
> 传统 Mixture-of-Experts（MoE，混合专家）通常从 N 个专家中为每个 Token 选择 Top-K 个专家，但这种粗粒度划分容易造成专家之间知识重叠，难以形成明确分工。本文提出 DeepSeekMoE，通过细粒度专家切分和共享专家隔离两项设计提升专家专门化程度：把原有专家切成更多小专家，同时按比例增加激活专家数以保持计算量；另外固定一部分专家始终参与计算，集中承载不同上下文中的共通知识。作者在 2B 和 16B 模型上进行实验，报告在相近或更低计算量下取得有竞争力的语言建模和下游任务表现。

# 核心定位

**DeepSeekMoE 的主要贡献不是“把 FFN 换成多个专家”——这是早期 MoE 已经做过的事；它重新设计了专家的粒度和共享结构，使 Token 的专家组合更灵活，并减少路由专家之间的冗余。**

## 1. 问题与动机

Dense Transformer 的 FFN 对每个 Token 都执行同一套参数。MoE 将 FFN 替换为多个专家网络，由 Router 根据 Token 表示选择部分专家，从而使总参数量可以增长，而每个 Token 的激活计算量不必同比增长。

但标准 Top-K MoE 存在两个问题：

1. **专家粒度过粗**：一个专家需要同时承载多个知识模式，不同专家之间容易学到重叠功能。
2. **通用知识重复存储**：不同领域 Token 都需要的通用变换可能在多个路由专家中重复学习，浪费专家容量。

因此，论文把目标从“增加专家数量”进一步推进到“让专家更容易形成互补分工”。

## 2. 方法

### 2.1 标准 MoE 作为基线

给定第 l 层 Token 表示 u_t，Router 计算它与每个专家的亲和度 s_{i,t}，Softmax 归一化后选取 Top-K：

\[
s_{i,t}=\operatorname{Softmax}_{i}(u_t^\top e_i),\qquad
g_{i,t}=
\begin{cases}
s_{i,t},&i\in\operatorname{TopK}(s_t),\\
0,&\text{otherwise}.
\end{cases}
\]

MoE 输出是被选中专家输出的门控加权和，再与残差相加。标准 MoE 的问题不是 Top-K 本身，而是每个专家的参数块相对较大，路由组合空间有限。

### 2.2 Fine-Grained Expert Segmentation：细粒度专家切分

假设基线有 N 个专家，每个 Token 激活 K 个。DeepSeekMoE 将每个专家拆成 m 个更小的专家，因此专家总数变成 mN，同时把激活数量提高到 mK。

关键约束是：**切分专家后按比例提高激活数，保持总激活参数量和计算预算大致不变。** 这不是单纯把专家变小、减少计算，而是把相同计算预算分配给更多可组合的专家子模块。

组合空间会显著扩大。论文举例：N=16、K=2 时，标准 Top-2 有 C(16,2)=120 种组合；若每个专家切成 4 份，变为 64 个细粒度专家并激活 8 个，组合数变为 C(64,8)，约 44 亿种。更多组合意味着 Router 可以为不同 Token 构造更细的功能组合。

### 2.3 Shared Expert Isolation：共享专家隔离

在路由专家之外，设置 K_s 个始终激活的共享专家：

\[
h_t=u_t+\sum_{i=1}^{K_s}E_i^{(s)}(u_t)
+\sum_{i=1}^{N_r}g_{i,t}E_i^{(r)}(u_t).
\]

共享专家承担不同领域和上下文都需要的共通知识；路由专家则主要学习差异化、领域相关的知识。这样做的目的不是让共享专家“更强”，而是减少通用模式在多个路由专家之间的重复编码，为路由专家释放专门化空间。

### 2.4 负载均衡

论文同时考虑 Expert-Level Balance Loss 和 Device-Level Balance Loss：前者避免少数专家过载，后者避免专家并行部署时少数设备承接过多 Token。两者分别对应模型路由均衡和系统通信均衡，不能只看 Router 的统计分布而忽略设备侧的负载。

## 3. 实验设置与结果

- **2B 规模验证**：在 12 个零样本或少样本基准上评估，覆盖语言建模、常识推理、阅读理解、代码生成和闭卷问答。
- **对比对象**：标准 GShard MoE、参数规模更大的 GShard 2.9B，以及相同总参数规模的 Dense 模型。
- **消融实验**：分别比较共享专家隔离、专家切分粒度和共享专家/路由专家比例。
- **16B 扩展实验**：在约 2T Token 语料上训练，并与 DeepSeek 7B、LLaMA2 7B 等模型比较。
- **主要结论**：DeepSeekMoE 2B 超过同规模 GShard 2B，并接近参数和计算量约为其 1.5 倍的 GShard 2.9B；16B 版本以约 40% 的计算量达到与强 7B Dense 模型相近的表现。专家切分更细时，多个基准上的表现呈持续改善趋势。

这些结果支持“细粒度组合 + 共享专家”有助于专门化，但不意味着专家已经获得人类可解释、互不重叠的语义分工。

## 4. 关键贡献

1. 把 MoE 的设计重点从专家数量转向专家粒度和可组合性。
2. 通过共享专家承载共通知识，减少路由专家重复学习。
3. 用消融和专家分析验证两项架构设计的作用，而不只展示最终模型分数。
4. 为 DeepSeek-V2/V3 后续模型提供了 FFN 稀疏化的架构基础。

## 5. 局限与启发

- 专家切分提高路由组合空间，但实际收益仍受 Router 质量、负载均衡和专家并行通信约束。
- 细粒度专家并不自动保证语义专门化；需要结合专家激活分布、跨域泛化和消融来验证。
- 论文的 2B/16B 实验不能单独证明所有模型规模、训练语料和硬件条件下都存在相同收益。
- 对工程实现而言，专家参数量不是唯一成本：Token Dispatch、All-to-All 通信、设备负载和显存带宽同样重要。

## 6. 关联

- **直接继承**：[[Literature/LLM/DeepSeek-V2 A Strong Economical and Efficient Mixture-of-Experts Language Model|DeepSeek-V2]] 将 DeepSeekMoE 与 MLA 结合，形成面向训练和推理效率的完整模型设计。
- **直接继承**：[[Literature/LLM/DeepSeek-V3 Technical Report|DeepSeek-V3]] 保留细粒度路由专家与共享专家，并进一步改进负载均衡。
- **对照基线**：GShard、Switch Transformer 是理解 Top-K/Top-1 路由与稀疏专家扩展的基础工作；它们不是 DeepSeekMoE 的同义实现。
- **后续硬件协同设计**：[[Literature/LLM/LatentMoE Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts|LatentMoE]] 把专家计算和通信搬到低维潜在空间，继续优化 MoE 的实际推理成本。

## 7. 图片与原件

仓库当前未提供该论文的本地 PDF，因此暂不嵌入论文截图或猜测图片路径。可从上述 arXiv / ACL 原文获取图 1（架构）和图 3（消融结果）。
