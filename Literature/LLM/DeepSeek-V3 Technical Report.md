---
title: "DeepSeek-V3 Technical Report"
authors:
  - DeepSeek-AI
year: 2024
venue: "arXiv technical report"
doi: "10.48550/arXiv.2412.19437"
arxiv: "2412.19437"
url: "https://arxiv.org/abs/2412.19437"
source: "未提供本地 PDF；原文 https://arxiv.org/pdf/2412.19437"
tags: [paper, LLM, MoE, MLA, MTP, load-balancing, FP8, distributed-training]
status: read
updated: 2026-11-11
---

> [!abstract] 摘要译述
> DeepSeek-V3 是一个总参数量 671B、每个 Token 激活 37B 参数的 MoE 语言模型。它沿用 DeepSeek-V2 的 Multi-head Latent Attention（MLA）和 DeepSeekMoE，并引入辅助损失无关的负载均衡策略与 Multi-Token Prediction（MTP，多 Token 预测）训练目标。论文同时报告 FP8 混合精度训练、DualPipe 计算通信重叠和跨节点 All-to-All 优化。模型在 14.8T Token 上预训练，完整训练报告成本为 2.788M H800 GPU 小时。

# 核心定位

**DeepSeek-V3 不是只靠 MoE 参数规模取胜。它展示了架构、优化器数值精度、并行通信和 GPU kernel 必须联合设计，才能把稀疏模型的理论计算优势转化为真实训练效率。**

## 1. 问题与动机

DeepSeek-V3 继承 DeepSeek-V2 的 MLA 与 DeepSeekMoE，但大规模 MoE 训练仍有三个工程瓶颈：

1. Router 负载不均会造成部分专家过载、其他专家空闲，降低专家并行效率。
2. 传统辅助负载均衡损失可能与语言建模目标冲突，过强时损害模型质量和专家专门化。
3. FP8 低精度计算、跨节点专家通信和激活存储都需要精细的数值与系统设计。

论文还希望通过 MTP 在每个位置预测多个未来 Token，增加训练监督信号密度，并为推测解码提供辅助预测头。

## 2. 方法

### 2.1 DeepSeekMoE + Auxiliary-Loss-Free Load Balancing

FFN 保留共享专家和路由专家。路由专家的亲和度由 Token 表示和专家向量计算。与传统依赖较大辅助损失来平衡负载的方式不同，DeepSeek-V3 为每个专家维护一个动态 Bias b_i：

- 选择专家时，用亲和度分数加 Bias 决定 Top-K。
- 计算专家输出的门控权重时，仍使用原始亲和度分数，而不是加 Bias 后的分数。
- 每个训练步统计专家负载：过载专家的 Bias 减小，负载不足专家的 Bias 增大。
- 另保留一个权重很小的 sequence-wise balance loss，避免单条序列内部出现极端失衡。

这种策略将“选哪个专家”和“专家输出贡献多少”分开：Bias 主要负责路由均衡，不直接改变输出加权分数。

### 2.2 Node-Limited Routing 与 No Token-Dropping

Node-Limited Routing 限制每个 Token 最多发送到 M 个节点，以控制跨节点通信范围，并让通信与计算尽可能重叠。由于负载均衡足够有效，论文报告训练和推理阶段均不需要丢弃 Token。

### 2.3 Multi-Token Prediction（MTP）

普通自回归语言模型在位置 t 只预测下一个 Token。MTP 将预测目标扩展到多个未来位置：

- 第 k 个 MTP 模块包含共享 Embedding、共享 Output Head、一个 Transformer Block 和投影矩阵。
- 多个 MTP 模块按深度顺序预测后续 Token，而不是多个互相独立的输出头并行猜测。
- 每个预测深度保留完整因果链。
- MTP 的辅助损失与主语言建模损失联合训练；推理时可将 MTP 头用于推测解码。

因此 MTP 既是训练目标设计，也可以成为推理加速的辅助结构；它不等于改变主模型的自回归因果约束。

### 2.4 FP8 混合精度训练

大部分高计算密度 GEMM 使用 FP8，但对数值敏感的 Embedding、Output Head、MoE Gating、Normalization 和 Attention 等模块保留较高精度。论文采用细粒度 Tile-wise / Block-wise Quantization，并提高累加精度；MoE 激活缓存和通信也使用低精度格式。

作者在约 1T Token 的较小模型实验中比较 FP8 与 BF16，报告相对 Loss 误差持续低于 0.25%。这项验证与 671B 主模型的训练报告应区分：前者是低精度数值可行性的实验，后者是系统规模训练结果。

### 2.5 DualPipe 与通信计算重叠

通过流水线调度让计算和跨节点通信重叠，减少专家并行 All-to-All 的等待时间。该方法是算法、分布式训练框架与 GPU 资源分配共同优化的结果，不是一个可脱离集群实现独立复现的模型层模块。

## 3. 实验设置与结果

- **模型规模**：671B 总参数，每 Token 激活 37B 参数。
- **预训练数据**：14.8T 高质量、多样化 Token。
- **训练阶段**：预训练后进行 SFT 和 RL；后训练还从 DeepSeek-R1 系列蒸馏推理能力。
- **效率报告**：预训练 2.664M H800 GPU 小时；上下文扩展 119K GPU 小时；后训练 5K GPU 小时；合计 2.788M H800 GPU 小时。作者按每 GPU 小时 2 美元估算官方训练成本为 5.576M 美元，且明确不含此前架构、算法和数据研究消融成本。
- **架构消融**：比较辅助损失无关策略、序列级负载均衡以及 MTP 对 Loss 和下游任务的影响。
- **精度实验**：在较小模型和约 1T Token 训练中比较 FP8/BF16 Loss。
- **系统实验**：分析 DualPipe、跨节点 All-to-All、Prefill / Decode 部署和低精度存储通信。

## 4. 关键贡献

1. 用动态专家 Bias 实现辅助损失无关的负载均衡，减少负载约束对专家专门化的干扰。
2. 用 MTP 增加未来 Token 预测监督，并提供推测解码接口。
3. 证明 FP8 混合精度可以用于超大规模预训练，同时通过分组量化和高精度累加控制数值误差。
4. 把 MoE 的通信成本作为架构设计的一部分，利用 DualPipe 等机制提升计算通信重叠。

## 5. 局限与启发

- 2.788M GPU 小时只统计论文定义的官方训练阶段，不包括此前研发和消融成本；成本比较必须核对统计口径。
- 训练效率依赖 H800 集群、网络拓扑、通信实现和专用 kernel，不能直接移植成普通单机的速度预期。
- Auxiliary-Loss-Free 不等于完全不需要任何辅助约束：论文仍使用极小的序列级辅助损失防止单序列极端失衡。
- MTP 的训练收益和推测解码收益是两个不同问题，应分别测量准确率、接受率、吞吐和额外计算成本。

## 6. 关联

- **直接继承**：[[Literature/LLM/DeepSeek-V2 A Strong Economical and Efficient Mixture-of-Experts Language Model|DeepSeek-V2]] 提供 MLA 与 DeepSeekMoE 的基础架构。
- **直接继承**：[[Literature/LLM/DeepSeekMoE Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models|DeepSeekMoE]] 提供细粒度专家与共享专家机制。
- **训练范式合流**：[[Literature/LLM/DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning|DeepSeek-R1]] 的长链推理能力被蒸馏到 V3 后训练阶段；R1 的贡献主要属于 RL 后训练，而非 V3 的主干架构。
- **直接继承并扩展**：[[Literature/LLM/DeepSeek-V3.2 Pushing the Frontier of Open Large Language Models|DeepSeek-V3.2]] 在 V3 系列基础上加入稀疏注意力和更大规模 Agentic RL。
- **直接继承并扩展**：[[Literature/LLM/DeepSeek-V4 Towards Highly Efficient Million-Token Context Intelligence|DeepSeek-V4]] 保留 DeepSeekMoE 和 MTP，并改进长上下文注意力与残差流拓扑。

## 7. 图片与原件

仓库当前未提供本地 PDF，因此暂不嵌入论文截图。建议优先查看原文 Figure 2（MoE / MTP）、Figure 6（FP8 混合精度框架）和 DualPipe 相关图。
