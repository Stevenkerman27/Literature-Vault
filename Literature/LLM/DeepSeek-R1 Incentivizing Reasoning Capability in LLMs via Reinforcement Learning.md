---
title: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
authors:
  - DeepSeek-AI
year: 2025
venue: "Nature 2025; arXiv preprint"
doi: "10.1038/s41586-025-09422-z"
arxiv: "2501.12948"
url: "https://arxiv.org/abs/2501.12948"
source: "未提供本地 PDF；原文 https://arxiv.org/pdf/2501.12948"
tags: [paper, LLM, reasoning, reinforcement-learning, GRPO, post-training, test-time-compute]
status: read
updated: 2026-11-11
---

> [!abstract] 摘要译述
> 本文研究如何通过强化学习激励大语言模型的推理能力。DeepSeek-R1-Zero 直接从 DeepSeek-V3-Base 出发进行大规模强化学习，不以人工标注的推理轨迹进行监督微调作为前置步骤。训练中逐渐出现自我反思、验证和动态策略调整等推理行为。为改善可读性、语言一致性和整体可用性，DeepSeek-R1 在冷启动长链思维数据基础上采用多阶段训练。论文的核心贡献属于后训练与强化学习范式，而不是提出新的 Transformer 主干架构。

# 核心定位

**DeepSeek-R1 的研究价值在于展示：对可验证推理任务，奖励驱动的强化学习可以诱导模型发展更长、更有结构的推理过程。它不是一种新的 Attention、FFN 或 MoE 架构。**

## 1. 问题与动机

传统推理模型通常依赖大量人工编写或筛选的 Chain-of-Thought（CoT，思维链）轨迹进行监督微调。但高质量推理轨迹成本高，且人工示范可能限制模型探索其他有效策略。

论文探索另一种路线：对数学、代码等具有可验证结果的任务，直接对模型生成的完整回答给出结果奖励，让模型通过 Reinforcement Learning（RL，强化学习）探索解题策略。

## 2. 方法

### 2.1 DeepSeek-R1-Zero：先 RL，再观察推理行为

R1-Zero 以 DeepSeek-V3-Base 为基础，不先使用人工标注的推理轨迹进行 SFT。训练主要依赖基于规则的奖励：

- 数学任务：答案与标准结果匹配时给奖励。
- 代码任务：通过测试用例执行和结果验证提供奖励。
- 格式奖励：鼓励模型按要求输出思考和最终答案结构。

这种奖励主要监督最终结果，而不是逐步判断每一步推理是否正确。论文观察到模型在训练过程中逐渐出现反思、回溯和调整策略等行为。

### 2.2 Group Relative Policy Optimization（GRPO）

GRPO 是本文使用的核心策略优化算法。对同一个问题 q，从旧策略采样一组 G 个回答，得到对应奖励 r_i。组内优势估计为：

\[
A_i=\frac{r_i-\operatorname{mean}(r_1,\ldots,r_G)}
{\operatorname{std}(r_1,\ldots,r_G)}.
\]

再使用类似 PPO 的裁剪策略目标和 KL 约束更新策略。与 PPO 的典型实现不同，GRPO 不额外训练一个 Critic / Value Model，而是用同一问题下多个回答的组内奖励估计优势。

直观理解：不是要求模型学一个额外网络来估计“这个状态值多少钱”，而是比较同一道题的多个候选回答谁相对更好，再用组内相对优势更新策略。

### 2.3 DeepSeek-R1：冷启动与多阶段训练

R1 并不是简单复制 R1-Zero 的纯 RL 流程。它增加了面向可读性和通用能力的训练阶段：

1. **Cold Start**：收集少量长 CoT 数据，对基础模型进行初始微调，改善推理结构、语言一致性和用户可读性。
2. **Reasoning-oriented RL**：在推理任务上继续强化学习，优化正确性和推理能力。
3. **Rejection Sampling + SFT**：从模型生成结果中筛选高质量样本，并与非推理数据组成约 800K 样本的数据集进行监督微调。
4. **Second RL Stage**：继续强化推理表现，同时通过有用性和安全性奖励改善通用对话行为。

论文还提供基于 Qwen 和 Llama 的多个蒸馏模型，说明长链推理轨迹也可以作为训练数据迁移给较小的 Dense 模型。

## 3. 实验设置与结果

- **基础模型**：DeepSeek-V3-Base；论文也在更小规模模型上进行实验和蒸馏。
- **R1-Zero RL**：每道题采样 16 个回答，最大生成长度随训练阶段由 32,768 增至 65,536 Token；总计 10,400 个训练步骤。
- **奖励任务**：数学、代码、STEM 和逻辑等可验证任务；另对有用性和无害性使用奖励模型。
- **训练资源**：R1-Zero 使用 64×8 张 H800 GPU，约 198 小时；R1 主训练约 80 小时；数据生成另使用约 5K H800 GPU 小时。作者报告的成本统计口径应与预训练成本区分。
- **AIME 2024**：R1-Zero 的 pass@1 从训练初始的 15.6% 提升到 77.9%；使用 Self-Consistency 解码后达到 86.7%。
- **蒸馏**：提供 1.5B、7B、8B、14B、32B、70B 等 Dense 模型，展示推理轨迹对较小模型的迁移价值。

## 4. 关键贡献

1. 展示可验证奖励可以在不预先提供人工推理轨迹的条件下，诱导出自我反思和验证行为。
2. 使用 GRPO 省去独立 Value Model，以组内相对奖励估计优势。
3. 区分 R1-Zero 的纯 RL 探索与 R1 的冷启动、多阶段训练，展示了可读性和能力之间的工程权衡。
4. 将长链推理能力通过数据蒸馏迁移到更小的 Dense 模型。

## 5. 局限与启发

- **奖励可验证性是前提**：数学和代码更容易提供自动验证；开放式写作、复杂科学解释等任务很难仅靠最终答案奖励判断过程质量。
- **Reward Hacking**：模型可能学会利用奖励函数漏洞；论文也讨论了模型奖励过度优化导致的投机行为。
- **R1-Zero 可读性不足**：纯 RL 产生的长推理过程可能出现语言混合、格式不稳定等问题，说明能力和可读性不是同一个优化目标。
- **推理长度不是正确性的充分条件**：更长的思考过程会增加推理时计算开销，必须结合正确率、延迟、Token 成本和任务难度评估。
- **不能将 GRPO 误认为架构创新**：GRPO 改变策略优化过程，不改变 Transformer 的 Attention / FFN 计算图。

## 6. 关联

- **基础模型**：[[Literature/LLM/DeepSeek-V3 Technical Report|DeepSeek-V3]] 提供 R1 所基于的主干模型及部分预训练基础。
- **算法来源**：DeepSeekMath（arXiv:2402.03300）是理解 GRPO 动机、数学推理 RL 和奖励设计的重要前置论文。
- **后续扩展**：[[Literature/LLM/DeepSeek-V3.2 Pushing the Frontier of Open Large Language Models|DeepSeek-V3.2]] 继续扩大 GRPO 训练规模，并扩展到工具使用和 Agentic Task。
- **研究分界**：MoE、MLA、DSA、mHC 改变模型计算结构；R1 改变模型训练和推理行为。两条路线可以组合，但不应在文献分类中混为一类。

## 7. 图片与原件

仓库当前未提供本地 PDF，因此暂不嵌入论文截图。建议优先查看原文 Figure 1（R1-Zero 训练曲线）、GRPO 算法图和多阶段训练流程图。
