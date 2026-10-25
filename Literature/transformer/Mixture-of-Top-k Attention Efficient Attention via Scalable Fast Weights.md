---
title: "Mixture-of-Top-k Attention: Efficient Attention via Scalable Fast Weights"
authors:
  - Qishuai Wen
  - Zhiyuan Huang
  - Xianghan Meng
  - Wei He
  - Chun-Guang Li
year: 2026
venue: "NeurIPS 2026 (accepted; arXiv v6 dated 2026-09-27)"
doi: "10.48550/arXiv.2602.01219"
arxiv: "2602.01219v6"
url: "https://arxiv.org/abs/2602.01219"
code: "https://github.com/QishuaiWen/MiTA"
source: "未提供本地 PDF；原始论文 v6：https://arxiv.org/pdf/2602.01219v6"
status: read
updated: 2026-10-25
tags:
  - literature
  - transformer
  - fast-weights
  - sparse-attention
  - mixture-of-experts
  - top-k
  - linear-complexity
  - dynamic-weights
  - vision
---

> [!abstract] 摘要译述（依据 arXiv v6）
> 标准 Self-Attention 可以解释为两层 fast-weight 多层感知机（Multi-Layer Perceptron，MLP）：权重来自当前序列的 Key/Value，隐藏层宽度等于上下文 token 数 \(N\)，因此难以扩展至长序列。以此为出发点，论文把高效注意力统一为两类 fast-weight 扩展策略：Routing（路由，只读取一部分原始权重）和 Compression（压缩，将大量原始权重提炼成小型共享模块）；提出涵盖策略、专家类型、专家数量、专家构造和路由拓扑的五维分类。新方法 Mixture-of-Top-k Attention（MiTA）先通过少量 landmark queries 探测全体 Key/Value，将其聚合为一个全局共享专家，同时为每个 landmark 检索 top-k 的原始 Key/Value，构成可变形且可复用的稀疏专家；各原始 query 一直访问共享专家，另外稀疏访问所选专家。在 ImageNet-1K、ADE20K 和 Long Range Arena 上验证了准确率与运行效率，并分析了 token-pruning 现象与不同注意力算法之间的迁移。

# 一句话结论

**MiTA 将 Attention 解释为“输入生成的一次性 MLP”，但其算法创新并不是替换成普通 MLP，而是把长度为 \(N\) 的动态 Key/Value 权重拆成：全局压缩的共享专家 + \(m\) 个由内容决定的 top-k 稀疏专家，然后让查询只读其中很小一部分。**

这是与 [[Literature/transformer/Linear-Time Global Visual Modeling without Explicit Attention|WeightFormer]] 不同的 fast-weight 路线：MiTA **仍然执行真实的注意力**，只是把每个 query 访问的 Key/Value 缩减为 \(m+ks\) 个，而 WeightFormer 用全局统计直接预测卷积/MLP 的参数。MiTA 在 2026-02-01 首次公开，WeightFormer 直到 2026-05 才公开，因而如果检索关键词是“一次性 MLP 权重”，MiTA 更早且更直接命中这种说法。

## 1. 核心数学观察：Self-Attention 相当于 Fast-Weight MLP

令一个 query 为 \(q\in\mathbb R^d\)，序列的 Key/Value 按列排成 \(K,V\in\mathbb R^{d\times N}\)。标准注意力：

\[
\operatorname{Attn}(q,K,V)
=V\,\operatorname{softmax}\!\left(K^\top q/\sqrt d\right).
\]

普通两层 MLP 写成：

\[
\operatorname{MLP}_\sigma(x)
=W_2\,\sigma(W_1^\top x+b_1)+b_2.
\]

将 Attention 的 Softmax 归一化展开：

\[
\operatorname{Attn}(q,K,V)
=
\underbrace{\frac{V}{\sum_{j=1}^N
\exp(k_j^\top q/\sqrt d)}}_{\widehat V(q)}
\exp\!\left(
\underbrace{K^\top/\sqrt d}_{\widehat K^\top}
q\right).
\]

此时类似一个两层网络：输入是 \(q\)，第一层权重来自 \(K\)，第二层权重来自 \(V\)，隐藏单元数是 \(N\)，非线性可写为指数函数。

> [!important] 等价关系的严格边界
> \(\widehat V(q)\) 含有 Softmax 分母，因此**会依赖当前 query**。若不吸收分母，则非线性应理解为跨隐藏单元耦合的 Softmax，而不是普通逐元素激活。这是对 Attention 运算的一种代数重写，不应误写为“存在一套不依赖 query 的普通固定 ReLU-MLP 参数使两者恒等”。“一次性”只表示 Key/Value 形成的有效权重随当前上下文而变；长期训练参数 \(W_Q,W_K,W_V\) 仍然共享且持续存在。
>
> 对所有 \(N\) 个 query 运行上述 \(N\)-宽度网络，仍需 \(O(N^2d)\) 量级计算。**仅把 Attention 称作 MLP 不会自动降低复杂度。**

## 2. 五维 Fast-Weight Scaling 分类

作者认为各种高效注意力可以按如下五个维度讨论：

| 维度 | 要回答的问题 | 代表取值 |
|---|---|---|
| Scaling strategy | 通过什么降低动态权重开销？ | Compression、Routing、两者组合 |
| Expert count | 需要多少组可重用快权重？ | 1、\(m\)、\(N\) |
| Expert type | 每组快权重对应什么计算模块？ | Linear、MLP、任意状态更新模块 |
| Expert construction | 如何从原始 Key/Value 构造专家？ | 核化、投影、空间局部块、内容检索、test-time training |
| Routing topology | 原始 query 与专家如何连边？ | 全部共享、块级路由、逐 query 路由、\(N\)-to-\(m\) |

### Compression（压缩）

例如 Linear Attention / Agent Attention / Linformer：不让每个 query 访问全部 \(N\) 对 Key/Value，而是把上下文表示为小尺寸的全局模块。优点是全局统计保留且查询成本低；问题是可能丢掉细节，尤其是少数重要 token。

### Routing（路由）

例如 MoBA、窗口化 Sparse Attention、per-query Top-k Attention：只允许 query 读取被选中的原始 Key/Value。优点是保留被访问 token 的精细原值；局限是只访问局部子集，可能缺少全局摘要。此外：

- 固定连续块（rigid block expert）依赖位置先验，不一定与语义相关。
- 每个 query 都单独从全体 \(N\) 个 token 计算 Top-k，虽然最终只读取 \(k\) 个，预筛选仍可能昂贵，并产生 \(N\) 套很难复用的索引。

### MiTA 的组合动机

用 \(m\ll N\) 个 Landmark Query 建立**可复用、内容相关的 Top-k 专家**，又始终给每个 query 提供压缩后的全局信息，组合两个策略的优点。

![论文中图1：Fast-Weight 扩展与 Routing、Compression 两条路线](https://arxiv.org/html/2602.01219v6/fast_weight_scaling_v5.png)

*论文中图 1：原始全注意力产生宽度随 \(N\) 增长的动态网络；图右表示按块 Routing（MoBA）与共享模块 Compression（TTT）的两种扩展方法。图片直接引用 arXiv 原图。*

## 3. MiTA 算法：Landmark、共享专家、可变形专家

设原始输入长度为 \(N\)，Landmark Query 数为 \(m\)，每个专家包含 \(k\) 对原始 Key/Value，某 query 访问 \(s\) 个稀疏专家。

### Step 1：从 Query 提取 Landmark Queries

在视觉任务上，沿空间均匀划分窗口，对 Query 做平均池化：

\[
\widetilde Q=\operatorname{AdaptiveAvgPool}(Q,m)
\in\mathbb R^{d\times m}.
\]

每个 Landmark Query 是对一个原始区域的粗汇总，但随后检索到的 Key/Value **不受该区域限制**；这正是专家“可变形”（Deformable）的含义。

### Step 2：每个 Landmark 在全局 Key 中检索 Top-k

\[
S=K^\top\widetilde Q/\sqrt d
\in\mathbb R^{N\times m},
\qquad
I_i=\operatorname{TopK}(S_{:,i},k).
\]

对第 \(i\) 个 Landmark：

\[
K^{(i)}=K_{:,I_i},\qquad
V^{(i)}=V_{:,I_i}\in\mathbb R^{d\times k}.
\]

共得到 \(m\) 个内容依赖的 Top-k 专家。与给每个原始 query 建一组 Top-k 不同，**这个 Top-k 集合由多个 query 共享**；所建专家数从 \(N\) 缩为 \(m\)。

### Step 3：Landmark 同时压缩全局信息

每个 Landmark 对全部 Key/Value 做 Cross-Attention：

\[
\widetilde v_i
=\operatorname{Attn}(\widetilde q_i,K,V).
\]

共享专家使用：

\[
\widetilde{\mathcal E}(q)
=\operatorname{Attn}(q,\widetilde Q,\widetilde V).
\]

这里 \(\widetilde Q\) 在最终注意力中充当**压缩后的 Key**，\(\widetilde V\) 充当压缩后的 Value。计算 Landmark 对全部 Key 的相似度可以复用 Step 2 中的 \(S\)。

### Step 4：每个原始 Query 选择 \(s\) 个稀疏专家

对 query \(q\)，用 \(q^\top\widetilde Q\) 作为 Router 的 logits。记其 Top-s 个专家编号为 \(e_1(q),\ldots,e_s(q)\)。

最终构造：

\[
K^*=[\widetilde Q,\ K^{(e_1(q))},\ldots,K^{(e_s(q))}],
\]

\[
V^*=[\widetilde V,\ V^{(e_1(q))},\ldots,V^{(e_s(q))}],
\]

\[
\boxed{
\operatorname{MiTA}(q)
=
\operatorname{Attn}(q,K^*,V^*)
}.
\]

所以一个 query 实际只访问：

\[
\boxed{m+ks\ \text{对 Key/Value}}
\]

而不是全部 \(N\) 对。**作者不是先分别计算共享专家和稀疏专家的独立输出再简单平均**，而是将两组 Key/Value 拼接，在同一 Softmax 中进行最终注意力；实现时可利用 online Softmax 和 FlashAttention 分开计算后精确合成。

![MiTA 方法概览：官方仓库图](https://raw.githubusercontent.com/QishuaiWen/MiTA/main/figures/mita.png)

*MiTA 官方代码仓库的机制概览。用于辅助理解共享压缩专家与可复用稀疏专家；这里不冒称它是经本地 PDF 裁切的论文原图。*

### 一组具体尺寸

ImageNet-1K 分类中，作者选取：

\[
m=25,\quad k=25,\quad s=1.
\]

224×224 输入、16×16 Patch 大致对应 \(N=196\) 个 patch token，每个 query 只访问 \(25+25=50\) 对。语义分割采用 \(m=k=49\)，每个 query 读取 98 对，显著少于完整注意力的 1024 或 1600 对。长序列 LRA 任务采用 \(m=k=128,s=1\)，每个 query 读取 256 对。

### 复杂度和实现瓶颈

固定 \(m,k,s\) 时：

- Landmark 检索/全局压缩：\(O(Nmd)\)；
- 每个原始 query 的共享专家 + 路由专家注意力：\(O(N(m+ks)d)\)；
- Router 查询匹配与 Top-k 索引还涉及排序、gather 与内存访问。

因此随着 \(N\) 增长，这些主体算术运算关于 \(N\) 近似线性。**不能忽略 Landmark 对全体 token 的 \(Nm\) 次筛选，也不能假设 \(m,k\) 随 \(N\) 增大时仍严格线性。** 论文明确指出随机 gather 的访存可能使实现变成 memory-bound；为了加速，将 query 按专家编号分组，执行变长批次 FlashAttention。

论文当前主要实现 \(s=1\)；任一 query 都访问共享专家 + 一个稀疏专家，\(s>1\) 是架构允许、但没有充分实证覆盖的扩展。

## 4. 实验配置与主要结果

论文在 ImageNet-1K 分类、ADE20K 语义分割和 Long Range Arena（LRA）长序列基准中评估，而不是在 PDE 或 CFD 数据集上评估。视觉模型同时测试无额外组件与带 Depth-Wise Convolution（DWC，逐通道卷积）、Gate（数据依赖门控）等组件的变体。

### 4.1 ImageNet-1K：区分原始 MiTA 与带额外组件的版本

| 模型 | 参数 | FLOPs | Top-1（%） |
|---|---:|---:|---:|
| DeiT-T（标准注意力） | 5.7M | 1.2G | 72.2 |
| Linear-DeiT-T | 5.7M | 1.1G | 68.0 |
| Agent-DeiT-T | 5.7M | 1.1G | 70.3 |
| MiTA-DeiT-T（仅替换 Attention） | 5.7M | 1.1G | 71.1 |
| MiTA-DeiT-T + DWC | 5.7M | 1.1G | **73.4** |
| DeiT-S（标准注意力） | 22M | 4.6G | 79.9 |
| MiTA-DeiT-S（仅替换 Attention） | 22M | 4.4G | 79.8 |
| MiTA-DeiT-S + DWC | 22M | 4.4G | 80.6 |
| MiTA-DeiT-S + DWC + Gate | 22M | 4.7G | **81.2** |

**重要对照：** 不带额外卷积/门控的 MiTA-T（71.1）并未超过标准 DeiT-T（72.2），MiTA-S（79.8）也略低于标准 DeiT-S（79.9）。不能直接宣称“原生 MiTA 注意力严格胜过全注意力”。作者使用 DWC / Gate 之后，准确率才超过相应 DeiT 基线；这需要在消融中单独说明。

此外，MiTA-ViT-5-S 的 81.3% 和加入 DWC 后的 81.7% 仍低于 ViT-5-S 的 82.2%，但计算量较低（约 4.5G vs 4.7G）。

### 4.2 ADE20K：省计算，但替换预训练注意力会掉精度

论文在 512²（Tiny/Small/Base）或 640²（Large）分辨率测试语义分割，使用 CBT 分割头；MiTA 是直接替换注意力，并**没有以 MiTA 原生预训练 backbone**：

| Backbone | 原始 ViT + CBT：mIoU / FLOPs | MiTA + CBT：mIoU / FLOPs |
|---|---:|---:|
| Tiny | 39.1 / 12G | 36.5 / 7G |
| Small | 45.8 / 33G | 44.5 / 25G |
| Base | 49.3 / 111G | 48.1 / 95G |
| Large | 53.3 / 622G | 50.5 / 508G |

论文报告最大 FLOPs 降低约 42%，但 mIoU 下跌 1.2–2.8 点。**“更省算力”得到支持；“完全不损失密集预测能力”没有得到支持。**

### 4.3 长序列 LRA：效率增益最大

Long Range Arena 含 ListOps（2K）、Text（4K）、Retrieval（4K）、Image（1K）、Pathfinder（1K）五个任务。使用 RTX 4090 24 GB，沿用统一训练超参数，报告：

| 方法 | 五任务平均准确率 | 总训练时间 |
|---|---:|---:|
| Standard Attention | 58.19 | 10.6 小时 |
| Agent Attention | 58.22 | 1.9 小时 |
| MoBA | 56.74 | 4.6 小时 |
| MiTA Route-only | 58.81 | 5.5 小时 |
| MiTA（Compression + Routing） | **58.91** | **2.4 小时** |

相对标准 Attention，MiTA 的总训练时间减少约 77%，平均准确率基本持平。**不过 Agent Attention 的平均准确率与 MiTA 接近，且在这个表中的总训练时间更短（1.9 vs 2.4 小时）**。因此这不是“在所有基线上速度和精度都第一”的证据。

## 5. 消融实验：到底哪一部分有用？

在 ImageNet-1K Tiny 模型、相同训练设置下，论文 Appendix A 报告：

| 变体 | Top-1 |
|---|---:|
| Compression + Routing（默认） | **71.1%** |
| 只有 Compression | 70.3% |
| 只有 Routing | 70.8% |
| \(m=16,k=16\) | 70.0% |
| \(m=25,k=25\)（默认） | 71.1% |
| \(m=25,k=36\) | **71.7%** |
| \(m=36,k=25\) | 71.2% |
| \(m=36,k=36\) | **71.7%** |

作者认为路由分支比压缩分支更关键：相对压缩-only，路由-only 有较高准确率；但两者结合又优于单独使用一种。提高 \(k\) 的增益通常高于等量提高 \(m\)，支持先确定 \(mk/N\) 的预算，再探索 \(k>m\) 的建议。

Landmark 选择方式：2D 平均池化（71.1%）优于随机选择（70.6%）、可学习参数（66.3%）和卷积（67.0%）；Token Merging（ToMe）略高（71.3%），但会增加实现复杂性与预处理成本。

### 涌现的 Token Pruning

作者可视化专家所选 Key/Value 的位置，发现后面层级会集中在类别相关区域，其他 token 较少被选为稀疏专家。但未被路由选中的 token 仍可能进入共享压缩分支，所以这里是**对稀疏精细检索的隐式剪枝**，而不是物理删除所有无关 token，也不能据此断言模型完全忽略背景。

![MiTA 官方仓库的 token-pruning 可视化](https://raw.githubusercontent.com/QishuaiWen/MiTA/main/figures/token_pruning.png)

### 算法迁移（Algorithmic Generalization）

作者测试了训练时使用标准注意力、推理时直接换成 MiTA（不重新训练）的情况：论文报告能保留原准确率的 95% 以上，但不等于逐 token 输出数值近似，也不意味着直接切换后精度完全无损。在从 ImageNet-21K 预训练模型微调时，MiTA-S 达到 80.9%，比 Standard Attention 的 81.2% 低 0.3 点；MiTA-L 85.5%，比 Standard Attention 的 85.9% 低 0.4 点。

## 6. 贡献与证据边界

### 主要贡献

1. 在 Fast-Weight 视角下，用五维分类统一组织 Routing、Compression 及其混合的高效注意力。
2. 将固定块专家替换为 Landmark 通过全局 Key 评分产生的 **Query-aware deformable Top-k 专家**。
3. 使 Top-k KV 集合由 \(N\) 个 query 独有变成 \(m\) 个专家共享，同时始终保留全局压缩分支。
4. 给出具体 FlashAttention/online Softmax 实现与 ImageNet、ADE20K、LRA 的实证，并考察隐式剪枝与算法替换的泛化。

### 不应该过度解读的结论

- **不是“标准注意力完全被证明多余”。** MiTA 依然在两个分支上运行注意力，其输出仍由 Query–Key 相似度控制；只是访问更少 Key/Value。
- **不是无条件线性。** Landmark–全序列评分需要 \(Nm\) 次点积，最终访问 \(m+ks\) 对；只有当 \(m,k,s\) 受控时才具备所述规模优势。
- **不是任意任务的精确替代。** MiTA 近似完整 Attention 的信息访问；ImageNet 原生模型、语义分割和微调实验里确有可观测的准确率损失。
- **存在不规则内存访问。** Top-k gather、Expert 分组、长度不同的 FlashAttention 与分支合并会产生实际开销，算术 FLOPs 不能完全代表训练/推理速度。
- **没有大语言模型逐 token decoding 实验。** 论文在 Limitations 中明确指出计算预算限制了 LLM decode 阶段验证。
- **没有 PDE/CFD 实证。** 视觉 token 的二维局部平均池化与非结构网格的密度/面积测度不一致，不能直接当作保网格不变的神经算子。

## 7. 对物理场代理建模的研究启发（属于本文之外的推论）

MiTA 给出一种可以测试的“低频全局总结 + 局部/稀有重要细节精确检索”组合。类比到翼身融合体、整机气动或非结构 CFD：

- **Global branch**：通过面积/体积加权的 Landmark 或 learned slices 汇总整机非局部信息（来流、翼身耦合、压力整体分布）。
- **Sparse routed branch**：使用几何与流动特征相似度检索可能跨空间距离关联的关键点，例如激波上下游、尾迹、翼根干扰区；这只是待验证假设。
- **Query decoder**：在每个面元输出压力/剪切，保留局部几何输入和 residual pointwise 处理。

必须单独检查：节点换序不变性、面积/体积加权、边界条件、采样密度变化、局部高频误差、跨构型 OOD、升阻积分误差、几何梯度和真实 GPU wall-clock。直接将二维像素的平均池化套用到随机点顺序的非结构网格，会引入没有物理意义的排序偏置。

与前面阅读的论文关系：

| 论文 | 全局信息机制 | 与 MiTA 的区别 |
|---|---|---|
| [[Literature/transformer/Transolver|Transolver]] | 每层 learned slice → slice token self-attention → deslice | 以软物理聚合为中心，不显式为每个 slice 选 top-k 原始 KV |
| [[Literature/transformer/Transolver Is a Linear Transformer Revisiting Physics-Attention Through the Lens of Linear Attention|LinearNO]] | Q/K 独立映射 + 低秩全局混合 | 去掉 slice token self-attention，不使用 MiTA 式精细 KV 稀疏检索 |
| [[Literature/transformer/Does Transolver really need a Transformer|NoTokenAttn / FlashSlice]] | 多层反复的全局 slice/deslice | 实证区分全局混合与 token self-attention，但未测试 MiTA 机制 |
| [[Literature/transformer/Linear-Time Global Visual Modeling without Explicit Attention|WeightFormer]] | 全局统计预测动态 MLP / Conv 参数 | 不构造可路由的 KV 专家，主要是动态网络参数化替代注意力 |

**可检验而非已证实的方向：** 使用物理切片作为 Landmark，保持 Transolver 的全局混合，再为每个切片加入少量关键网格点 KV 检索，与没有路由分支的模型做严格配对消融。这种结合尚未在本文或上述 Transolver 研究中直接验证。

## 8. 重点前作与后续阅读

- Schlag, Irie & Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers*, ICML 2021：线性注意力与 Fast-Weight Programming 的早期联系。https://arxiv.org/abs/2102.11174
- Lu et al., *MoBA: Mixture of Block Attention for Long-Context LLMs*, 2025：将固定连续块视为可路由 KV 专家，是 MiTA 所批评和改进的“刚性专家”路线。https://arxiv.org/abs/2502.13189
- Yuan et al., *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*, 2025：同时使用压缩与稀疏注意力的代表工作。https://arxiv.org/abs/2502.11089
- [[Literature/transformer/Linear-Time Global Visual Modeling without Explicit Attention|WeightFormer]]：同样从动态权重解释 Attention，但提出了另一种不显式运行 Attention 的视觉架构。

## 原文与复现入口

- 最新原文（2026-09-27，v6）：https://arxiv.org/html/2602.01219v6
- PDF：https://arxiv.org/pdf/2602.01219v6
- 官方 GitHub：https://github.com/QishuaiWen/MiTA
- 作者官方示例提供 MiTA-DeiT、MiTA-ViT-5、MiTA-Segmenter；需要注意具体 FlashAttention 版本和 GPU 环境。
- 本笔记的实验数字与公式据 v6 正文 Table 1–5、Appendix Table 6–7；不将早期 v1/v3 版本标题或结果混入最新版本。


## 新增分类入口：与 SSM/递归记忆比较

参见 [[Literature/Topics/神经网络架构与高效序列建模|神经网络架构与高效序列建模]]：[[Literature/architectures/Gated Linear Attention Transformers with Hardware-Efficient Training|GLA]] 与 [[Literature/architectures/Gated Delta Networks Improving Mamba2 with Delta Rule|Gated DeltaNet]] 将过去压入固定维矩阵状态；本论文 MiTA 仍通过少量可复用专家精确读取原始 Key/Value，属于**压缩 + 稀疏路由**的混合方案。相同 Fast-Weight 解释不等于相同存储/检索能力。
