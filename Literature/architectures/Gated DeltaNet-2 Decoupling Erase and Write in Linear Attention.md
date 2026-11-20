---
title: "Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention"
authors:
  - Ali Hatamizadeh
  - Yejin Choi
  - Jan Kautz
year: 2026
venue: "arXiv preprint (v2: 2026-11-06)"
doi: "10.48550/arXiv.2605.22791"
arxiv: "2605.22791"
url: "https://arxiv.org/abs/2605.22791"
source: "未提供本地 PDF；原文 https://arxiv.org/pdf/2605.22791"
tags: [paper, linear-attention, recurrent-memory, delta-rule, gated-delta-net, Mamba, retrieval]
status: read
updated: 2026-11-11
---

> [!abstract] 摘要译述
> 线性注意力用固定大小的递归状态替代随上下文增长的 KV Cache，将序列混合复杂度降为线性时间、解码内存降为常数级。但固定状态需要承载越来越多的 Key–Value 关联，容易发生记忆干扰。既有 Delta Rule 模型在写入新 Value 前先读取并修正旧记忆；Gated DeltaNet 和 Kimi Delta Attention（KDA）进一步引入遗忘机制，但仍用同一个标量门控同时控制 Key 侧擦除和 Value 侧写入。本文提出 Gated DeltaNet-2，将两者拆成独立的通道级 Erase Gate 与 Write Gate，并给出可高效并行训练的 chunkwise WY 算法和 gate-aware backward。作者在 1.3B / 100B Token 实验及 31.6B 混合 latent-MoE 实验中报告语言建模、常识推理和检索方面的改进。

# 核心定位

**Gated DeltaNet-2 进一步细化了递归记忆的编辑机制：不是只决定“忘多少”，而是分别决定旧 Key 关联中哪些方向需要擦除，以及新 Value 中哪些方向值得写入。**

## 1. 问题与动机

线性注意力将历史信息压缩进固定大小的状态矩阵。与标准 Softmax Attention 不同，它不保存所有历史 Token 的 K/V，而是不断更新一个有限容量的关联记忆。

主要更新机制的差别：

- 普通线性注意力：不断累加外积，旧关联难以主动删除。
- Mamba-2：使用输入依赖的衰减系数，对旧状态进行全局遗忘。
- DeltaNet：先读取当前 Key 对应的旧 Value，再按预测误差定向覆盖。
- Gated DeltaNet：同时使用全局衰减与 Delta Rule 定向修正。
- KDA：把衰减扩展为 Key 通道级，但 Delta 更新强度仍是标量。
  
Gated DeltaNet-2 观察到，擦除和写入发生在状态矩阵的不同轴上：擦除决定 Key 侧哪些方向的旧关联需要被移除，写入决定 Value 侧哪些方向的新内容需要被保留。将两者绑定为一个标量会限制记忆更新的表达能力。

## 2. 方法

### 2.1 固定大小递归记忆

本文采用的状态方向为：

\[
S_t\in\mathbb R^{d_k\times d_v},\qquad
o_t=S_t^\top q_t.
\]

普通线性注意力按外积累加：

\[
S_t=S_{t-1}+k_tv_t^\top.
\]

Delta Rule 则先计算旧状态对当前 Key 的预测，再只写入误差：

\[
S_t=(I-\beta_t k_tk_t^\top)S_{t-1}
+\beta_t k_tv_t^\top.
\]

当 Key 归一化时，Delta Rule 可以对当前 Key 对应的关联进行定向覆盖，而不必把整个状态一起清空。

### 2.2 分离 Erase Gate 与 Write Gate

Gated DeltaNet-2 分别预测：

\[
b_t\in[0,1]^{d_k},\qquad
w_t\in[0,1]^{d_v}.
\]

其中 b_t 是 Key 侧的通道级擦除门控，w_t 是 Value 侧的通道级写入门控。定义：

\[
e_t=b_t\odot k_t,\qquad
z_t=w_t\odot v_t,\qquad
D_t=\operatorname{Diag}(\alpha_t).
\]

先对旧状态进行通道级衰减，再通过擦除方向读取旧内容，最后写入修正：

\[
\bar S_t=D_tS_{t-1},\qquad
r_t=\bar S_t^\top e_t,
\]

\[
\boxed{
S_t=\bar S_t+k_t(z_t-r_t)^\top
}
\]

等价形式为：

\[
S_t=
\left(I-k_t(b_t\odot k_t)^\top\right)
D_tS_{t-1}
+k_t(w_t\odot v_t)^\top.
\]

三个机制承担不同职责：

- Channel-wise Decay：对 Key 维度的旧记忆进行不同速率的衰减。
- Erase Gate b_t：选择当前读取旧记忆时需要擦除的 Key 通道。
- Write Gate w_t：选择新 Value 中需要写入状态的通道。

当 b_t 和 w_t 都退化为同一个标量 β_t 时，模型恢复 KDA；如果衰减也退化为标量 α_t，则恢复 Gated DeltaNet。因此它把已有方法作为受限特例包含在更一般的参数化中。

### 2.3 Fast-Weight 视角

状态 S_t 可视为快速变化的联想记忆。每个 Token 到来时：

1. 衰减旧记忆；
2. 用经过 Erase Gate 过滤的 Key 方向读取旧内容；
3. 将旧内容与经过 Write Gate 过滤的新 Value 比较；
4. 用 Delta 修正更新记忆。

这相当于在线求解一个局部目标：尽量让新状态靠近衰减后的旧状态，同时修正当前 Key 对应的预测，使其朝向门控后的目标 Value。

### 2.4 Chunkwise WY 与 Gate-Aware Backward

逐 Token 递归不利于 GPU 高吞吐矩阵计算。论文把状态更新转化为 WY 形式，在 Chunk 内用矩阵乘法并行计算，Chunk 之间仍保留递归状态。

通道级衰减被吸收到非对称擦除因子中；反向传播则显式处理 Erase / Write Gate 对 WY 逆变换的影响，避免把通道级门控错误地简化成一个事后标量缩放。

## 3. 实验设置与结果

### 3.1 1.3B 受控比较

- **规模与数据**：各递归模型约 1.3B 参数，在 100B FineWeb-Edu Token 上训练。
- **训练设置**：AdamW，峰值学习率 4e-4，Weight Decay 0.1，Gradient Clipping 1.0，Cosine LR；1B Token Warm-up；全局 Batch 0.5M Token；训练序列长度 4K。
- **混合模型**：另比较与 Sliding Window Attention 结合的 Hybrid 版本，窗口为 2K。
- **基线**：Mamba-2、Gated DeltaNet、KDA、Mamba-3 SISO / MIMO，以及 Transformer / Hybrid Attention。
- **语言与常识**：Gated DeltaNet-2 在 Recurrent 与 Hybrid 设置中均取得更高的综合平均表现。
- **合成检索**：在 RULER 的 Single Needle 与 Multi-Key Needle 任务上，尤其在固定状态需要区分多个关联的场景中取得优势。
- **真实检索**：在 SWDE、SQuAD、FDA、TriviaQA、Natural Questions 和 DROP 等任务上，Hybrid 版本的整体检索平均表现较强。

### 3.2 31.6B Hybrid Latent-MoE 扩展实验

作者进一步训练 31.6B 总参数、3.2B 激活参数的 Hybrid Latent-MoE 模型，使用 300B Token，并包含少量 Full Global Attention 层。实验固定 Attention 层位置、递归状态大小、激活参数量和训练配方，只替换递归 Token Mixer。

Gated DeltaNet-2 在十项知识、代码、数学和常识基准中的九项排名第一，并在四个类别平均分上领先 KDA 和 Mamba-2。相对 KDA 的 Overall Average 提升 1.42 分，相对 Mamba-2 提升 2.53 分。

这项实验比单独的 1.3B 对照更有说服力，但仍是特定 Hybrid Latent-MoE 配置，不能直接推断任意大模型都获得同样幅度的收益。

### 3.3 消融与吞吐

- Erase / Write 两个门控都保留通道级结构时效果最好；只保留 Erase Gate 的通道结构通常保留了大部分收益，只保留 Write Gate 的收益较弱。
- 去掉 Delta Correction 后，检索指标明显下降，说明收益不只是来自两个额外门控。
- 单 H100 的 Hybrid 1.3B 训练吞吐随序列长度从 38.0 降到 36.1 K Token/s，论文报告相较 KDA 只有较小的常数开销。

## 4. 关键贡献

1. 将 Delta Rule 的擦除方向与写入方向解耦为两个通道级门控。
2. 把 KDA 和 Gated DeltaNet 统一为更一般更新规则的受限特例。
3. 通过非对称擦除因子和 Gate-Aware WY Backward 保留高效 Chunkwise 并行训练。
4. 不仅进行 1.3B 受控比较，还在 31.6B Hybrid Latent-MoE 模型上验证递归 Mixer 的替换收益。

## 5. 局限与启发

- 1.3B 实验的训练长度为 4K；长上下文优势主要通过 RULER 和检索任务观察，不能直接等同于百万 Token 通用上下文能力。
- 31.6B 实验包含全局 Attention 层，收益来自混合架构中的递归模块替换，不是纯 Gated DeltaNet-2 独立主干。
- 论文明确指出高阶排列组合等任务仍存在绝对准确率不足；固定状态记忆仍受容量与关联干扰限制。
- 通道级门控带来额外 Kernel 和推理延迟；需要同时比较吞吐、单 Token Decode 延迟、状态读写带宽和长序列检索质量。
- 对无序 CFD 网格，递归状态更新可能依赖 Token / 网格点排列。若将其用于空间场建模，必须设计对置换、网格重编号和局部拓扑变化稳健的机制，不能直接把网格点排序当成自然时间序列。

## 6. 文献关系

- **直接继承**：[[Literature/architectures/Gated Delta Networks Improving Mamba2 with Delta Rule|Gated DeltaNet]] 将全局遗忘与 Delta Rule 定向覆盖结合；本文进一步拆分 Key 侧擦除和 Value 侧写入。
- **数学对照**：[[Literature/architectures/Transformers are SSMs Generalized Models and Efficient Algorithms Through Structured State Space Duality|Mamba-2 / SSD]] 使用状态空间对偶和数据依赖衰减；本文在 Delta Rule 快权重记忆路径上扩展。
- **并行路线**：[[Literature/LLM/Mamba-3 Improved Sequence Modeling using State Space Principles|Mamba-3]] 改进 SSM 离散化、复数状态与 MIMO；两者不是同一个更新规则。
- **共同基准比较**：本文直接比较 Mamba-2、Gated DeltaNet、KDA、Mamba-3 SISO / MIMO，适合与已有 Gated DeltaNet、Mamba-3 笔记对读。
- **系统组合**：31.6B 实验将该递归 Mixer 放入 Hybrid Latent-MoE 模型，说明递归状态模块可以与 MoE 和 Full Attention 层组合。

## 7. 图片与原件

仓库当前未提供本地 PDF，因此暂不嵌入论文截图。建议查看原文 Figure 1（状态更新机制对比）、Figure 2（H100 吞吐）、Table 2–5（语言、RULER 检索与 31.6B Hybrid Latent-MoE 结果）。
