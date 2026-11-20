---
title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
authors:
  - DeepSeek-AI
year: 2024
venue: "arXiv preprint"
doi: "10.48550/arXiv.2405.04434"
arxiv: "2405.04434"
url: "https://arxiv.org/abs/2405.04434"
source: "未提供本地 PDF；原文 https://arxiv.org/pdf/2405.04434"
tags: [paper, LLM, MoE, MLA, KV-cache, RoPE, inference-efficiency]
status: read
updated: 2026-11-11
---

> [!abstract] 摘要译述
> 本文提出 DeepSeek-V2，一个总参数量 236B、每个 Token 激活约 21B 参数、支持 128K 上下文的混合专家语言模型。模型采用两项核心设计：Multi-head Latent Attention（MLA，多头潜在注意力）通过低秩联合压缩 Key 和 Value，减少推理时的 KV Cache；DeepSeekMoE 通过稀疏专家计算降低训练成本。相较 DeepSeek 67B，论文报告训练成本降低 42.5%、KV Cache 减少 93.3%，最大生成吞吐提高至 5.76 倍。

# 核心定位

**DeepSeek-V2 的关键价值在于把“参数稀疏化”和“注意力缓存压缩”放进同一个模型架构：MoE 降低每 Token 的 FFN 激活计算，MLA 降低自回归解码时随上下文增长的 KV Cache。**

## 1. 问题与动机

大模型推理有两类彼此不同的成本：

- FFN 参数计算成本：Dense 模型每个 Token 都经过完整 FFN，模型总参数规模与激活计算量强绑定。
- 自回归解码的 KV Cache 成本：每生成一个 Token，都需要保存此前 Token 的 Key/Value；上下文越长、层数越多、KV Head 越多，缓存越大。

MoE 主要缓解第一类成本，GQA/MQA 等方法主要缓解第二类成本。DeepSeek-V2 进一步提出 MLA，以低秩潜在表示压缩 KV，而不只是减少 KV Head 数量。

## 2. 方法

### 2.1 Multi-head Latent Attention（MLA）

标准多头注意力在每个位置缓存各层的 Key 和 Value。MLA 将隐藏状态 h_t 投影到较低维的 KV 潜在向量：

\[
c_t^{KV}=W^{DKV}h_t.
\]

再由该潜在向量恢复各注意力头所需的内容 Key 和 Value：

\[
k_t^C=W^{UK}c_t^{KV},\qquad
v_t^C=W^{UV}c_t^{KV}.
\]

**推理时主要缓存 c_t^{KV}，而不是逐头保存完整 K/V 张量。** 因此缓存维度主要由潜在维度决定，而不再与 Query Head 数量线性绑定。

MLA 还引入 Decoupled RoPE（解耦旋转位置编码）：把携带内容信息的压缩 Key 与携带位置旋转信息的 Key 分开处理，再在注意力计算中组合。这样避免对压缩潜在向量直接施加 RoPE 后破坏低秩压缩结构。

### 2.2 DeepSeekMoE

FFN 采用细粒度路由专家与共享专家结构，继承 [[Literature/LLM/DeepSeekMoE Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models|DeepSeekMoE]] 的专家切分和共享专家隔离思路。

每个 Token 只激活一部分路由专家，减少实际 FFN 计算。为适配专家并行训练，论文还采用 Device-Limited Routing：限制一个 Token 可被发送到的设备数量，避免跨节点通信失控；并使用专家级、设备级和通信级负载均衡约束。

### 2.3 两项设计解决不同瓶颈

| 设计 | 主要压缩对象 | 主要收益 |
|---|---|---|
| DeepSeekMoE | 每个 Token 激活的 FFN 参数 | 降低训练和前向计算成本 |
| MLA | 解码阶段需要保留的 KV 状态 | 降低 KV Cache 显存和带宽开销 |
| Device-Limited Routing | Token Dispatch 的设备范围 | 控制 All-to-All 通信成本 |

不要把 MLA 误解成另一种 MoE：MLA 改变 Attention 的 K/V 表示和缓存形式，MoE 改变 FFN 的参数激活路径。

## 3. 实验设置与结果

- **模型规模**：DeepSeek-V2 总参数 236B，每 Token 激活约 21B 参数；支持 128K 上下文。
- **预训练数据**：8.1T Token 的高质量、多来源语料。
- **训练阶段**：预训练之后进行 Supervised Fine-Tuning（SFT，监督微调）和 Reinforcement Learning（RL，强化学习）。
- **评估任务**：语言建模、知识理解、常识推理、阅读理解、数学、代码和标准化考试等。
- **效率比较**：以 DeepSeek 67B 为主要效率参照，论文报告训练成本降低 42.5%、KV Cache 降低 93.3%、最大生成吞吐达到 5.76 倍。
- **小模型验证**：同时发布 DeepSeek-V2-Lite，约 15.7B 总参数、2.4B 激活参数，便于研究 MLA 与 DeepSeekMoE。

这些数字是论文所述模型、部署实现和测试条件下的报告结果，不应直接等同于任意硬件上的端到端加速比。

## 4. 关键贡献

1. 用低秩联合压缩把每个 Token 的 KV 状态映射到潜在向量，改变 KV Cache 的存储表示。
2. 将内容 Key 与 RoPE 位置 Key 解耦，使低秩缓存压缩与相对位置信息兼容。
3. 将 DeepSeekMoE 的稀疏 FFN 与 MLA 组合，分别处理训练计算和推理缓存瓶颈。
4. 通过设备受限路由和通信均衡把架构设计与分布式系统成本联系起来。

## 5. 局限与启发

- MLA 减少缓存，但仍需在 Attention 计算时使用潜在表示恢复各头所需的 K/V；节省的显存与新增投影计算之间存在权衡。
- MLA 的实际吞吐收益依赖 kernel、批量大小、上下文长度和内存带宽，不能仅凭缓存尺寸推断所有场景的延迟。
- MoE 的总参数量不等于每 Token 激活量；部署时还要考虑专家权重常驻、专家并行通信和负载不均衡。
- 对需要复现的实现，应分别测量 Prefill 和 Decode，不应把两者的瓶颈混为一谈。

## 6. 关联

- **直接继承**：[[Literature/LLM/DeepSeekMoE Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models|DeepSeekMoE]] 提供细粒度专家与共享专家设计。
- **直接继承**：[[Literature/LLM/DeepSeek-V3 Technical Report|DeepSeek-V3]] 沿用 MLA 与 DeepSeekMoE，并继续改进负载均衡和训练基础设施。
- **机制对照**：GQA/MQA 通过共享 KV Head 降低缓存；MLA 通过低秩潜在表示压缩 KV，二者都降低缓存但机制不同。
- **后续注意力稀疏化**：[[Literature/LLM/DeepSeek-V3.2 Pushing the Frontier of Open Large Language Models|DeepSeek-V3.2]] 在 MLA 表示上进一步加入 DeepSeek Sparse Attention。

## 7. 图片与原件

仓库当前未提供该论文的本地 PDF，因此暂不嵌入论文截图。建议精读原文 Figure 1（整体架构与效率对比）和 Section 2.1 / Appendix D（MLA 结构与注意力消融）。
