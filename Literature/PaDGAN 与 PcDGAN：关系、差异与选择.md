---
title: "PaDGAN 与 PcDGAN：关系、差异与选择"
authors:
  - "comparison note"
year: 2026
venue: "Obsidian synthesis note"
doi: "N/A"
source: "/mnt/data/2002.11304v5.pdf; /mnt/data/PcDGAN_A_continuous_conditional_diverse_GAN_for_Inverse_Desisgn(1).pdf"
tags:
  - generative-design
  - comparison
  - PaDGAN
  - PcDGAN
---

## 1. 最核心的关系

两篇论文不是两个平行方法，而是很清楚的**方法继承链**：

> **PaDGAN 先提出“用 DPP 同时优化质量与多样性”；PcDGAN 再把其中的“质量”改写成“给定连续条件的满足程度”，并补上连续标签不均衡训练机制。**

可以把 PaDGAN 的核写成

$$
L_{ij}=k(x_i,x_j)[q_{perf}(x_i)q_{perf}(x_j)]^{\gamma_0},
$$

其中 $q_{perf}$ 越高表示设计性能本身越好。

PcDGAN 保留相同骨架，但变为

$$
L_{ij}=k(x_i,x_j)[q_{cond}(x_i;y^*)q_{cond}(x_j;y^*)]^{\gamma_0},
$$

其中

$$
q_{cond}(x;y^*)=LLETS(|f(x)-y^*|).
$$

因此真正发生变化的不是 DPP 数学结构，而是**quality 的语义**：

- PaDGAN：绝对质量，“越高越好”；
- PcDGAN：相对目标的条件质量，“越接近我指定的 $y^*$ 越好”。

## 2. 为什么 PcDGAN 不能简单地叫“conditional PaDGAN”

如果只给 PaDGAN 的生成器拼一个条件 $y^*$，仍然有三个问题：

1. **连续标签稀疏**：精确 $y^*$ 往往没有训练样本；
2. **标签不均衡**：极端性能区域在数据中很少，普通条件训练更新不足；
3. **同一条件下多样性定义不清**：如果一个 batch 混着不同条件，DPP 会把“条件不同”造成的形状差异误认为 conditional diversity。

所以 PcDGAN 必须额外引入：

- vicinal loss：允许用邻域样本训练连续条件；
- singular label sampling：每步只聚焦一个条件切片，并均匀覆盖整个标签范围；
- LLETS：把“离目标多远”变成适合 DPP 与反向传播的 quality；
- 新 condition input mechanism：把连续条件有效注入 G 与 D。

换句话说，PcDGAN 是**PaDGAN 的目标函数继承 + CcGAN 的连续条件机制改造**。

## 3. 相同点

| 维度 | 共同点 |
|---|---|
| 生成骨架 | 都在 GAN 上增加额外 DPP loss |
| DPP 核 | 都采用 similarity × quality 的分解 |
| 多样性 | 都通过 $-\log\det(L_B)$ 鼓励 batch 内样本张开 |
| 相似度 | 实验都使用 RBF kernel，bandwidth 1 |
| 性能模型 | 都允许可微物理模型或神经网络 surrogate |
| 翼型 backbone | 都使用 UIUC 数据、XFOIL、ResNet surrogate 与 BézierGAN |
| 稳定化 | 翼型中都使用 $D(x)$ 加权 quality，并采用逐步增加 DPP loss 权重的 schedule |
| 核心价值观 | 都认为工程生成模型不应只“拟合训练分布”，而应主动形成更有用的设计分布 |

尤其值得注意：PcDGAN 论文中翼型设置大量沿用 PaDGAN，包括 38,802 个增广翼型、BézierGAN、ResNet estimator、RBF DPP kernel，以及 realisticity-weighted quality/逐步增大 DPP 权重。这说明 PcDGAN 是非常直接的工程延伸。

## 4. 不同点

| 维度 | PaDGAN | PcDGAN |
|---|---|---|
| 问题类型 | 无条件/整体高性能生成 | 连续性能条件逆设计 |
| 输入 | 随机噪声 $z$ | 随机噪声 $z$ + 目标条件 $y^*$ |
| “质量”定义 | $q(x)$ = 设计性能 | $q(x;y^*)$ = 条件满足度 |
| 目标 | 高性能 + 多样 | 指定性能下准确 + 多样 |
| 关键新增机制 | DPP quality-diversity loss | LLETS + singular vicinal loss + condition embedding |
| 条件空间 | 不存在 | 连续标量，且需处理数据稀疏/不均衡 |
| 典型输出 | “给我很多高 $C_L/C_D$ 翼型” | “给我很多 $C_L/C_D\approx y^*$ 的不同翼型” |
| 新颖性重点 | 强调向训练域外高质量区域外推 | 更强调条件切片内覆盖与插补；不是以最大性能外推为主目标 |
| 对比基线 | GAN、$GAN_D$、$GAN_Q$ | CcGAN |
| 评价 | quality/diversity/overall | likelihood/label error/diversity |

## 5. “高质量”与“满足条件”是两个完全不同的优化方向

这是理解两篇论文最关键的一点。

假设翼型性能 $f(x)=C_L/C_D$，训练范围归一化为 0–1。

### PaDGAN

如果 $q(x)=f(x)$，那么生成器总会被推向 $f(x)\to1$。即使你想要一个中等性能但结构更稳健的方案，它也不会主动生成，因为“中等性能”在其目标里就是低质量。

### PcDGAN

给定 $y^*=0.4$，则最好的样本是 $f(x)=0.4$，而不是 $f(x)=1$。一个 $f(x)=0.95$ 的翼型在 PcDGAN 看来反而**条件质量很差**。

所以：

> PaDGAN 是**优化型生成**；PcDGAN 是**逆映射型生成**。

前者在学“哪里好”，后者在学“给定性能值，对应哪些设计”。

## 6. DPP 在两篇论文里的功能也有细微变化

### PaDGAN：跨高性能 mode 分散

DPP 要阻止样本全部挤在同一个性能峰，让模型在多个高性能区域都产生样本。其多样性是**全局设计空间多样性**。

### PcDGAN：固定条件切片内分散

DPP 在一个 singular vicinal neighborhood 中工作，要求“性能几乎相同的样本，其几何仍应不同”。其多样性是**条件多样性（conditional diversity）**。

这两个概念不能混为一谈：全局看起来很分散的条件 GAN，可能只是因为不同条件对应不同形状；真正困难的是**固定同一个条件后仍有多个解**。

## 7. 为什么 PcDGAN 要引入 LLETS，而 PaDGAN 不需要

PaDGAN 的 quality 通常天然是“越大越好”的标量，如 $C_L/C_D$，可以直接进入 DPP。

PcDGAN 的原始信号却是误差

$$
\epsilon=|f(x)-y^*|,
$$

它是“越小越好”。简单用 $-\epsilon$ 不但可能产生负质量，而且梯度性质不适合 DPP 的质量加权。因此必须先把误差映射成非负、尺度一致、梯度形状合适的 quality。LLETS 本质上就是这个**error-to-quality transfer function**。

## 8. 数据不均衡问题：PaDGAN 与 PcDGAN 的关注点不同

PaDGAN 主要担心的是**设计空间 mode imbalance**：GAN 会只覆盖少数几何 mode。DPP 直接鼓励样本在 $x$ 空间分散。

PcDGAN 还额外面对**标签空间 imbalance**：例如翼型数据中 $y>0.9$ 只有 0.18%。如果仍按经验标签分布训练，极端条件几乎没有梯度更新。singular vicinal sampling 正是针对 $y$ 空间做近似均衡采样。

所以 PcDGAN 实际同时处理两个空间：

- $y$ 空间：训练频率均衡；
- $x$ 空间：DPP 覆盖多样。

## 9. 对实际工程任务应该选哪个？

### 适合 PaDGAN

当需求是：

- “帮我生成一批尽可能高性能、但不要长得都一样的设计”；
- “我想扩大高质量设计库/初始种群”；
- “后面还会做优化，希望生成器先把采样分布推向高性能区”。

PaDGAN 更自然。

### 适合 PcDGAN

当需求是：

- “我已经知道性能要求，直接反求设计”；
- “同一个性能目标希望得到多个备选方案”；
- “条件是连续值，且训练数据在某些条件区很稀疏”。

PcDGAN 更合适。

### 二者可能组合的进一步方向

实际设计常同时有“必须满足条件”和“条件满足后还要尽量优”。例如：

- 指定升力 $C_L=C_L^*$；
- 在此约束下最小化 $C_D$；
- 同时保持多个不同翼型家族。

这时可以把 DPP quality 设计为复合项，例如

$$
q(x)=q_{condition}(x;C_L^*)^\alpha\cdot q_{objective}(x;C_D)^\beta\cdot q_{feasible}(x),
$$

再配合 DPP similarity。概念上就是把 PcDGAN 的“条件质量”和 PaDGAN 的“绝对性能质量”合并。这一点论文没有实现，但从两篇方法的共同数学骨架看是自然延伸。

## 10. 对 latent-code 生成式设计尤其值得注意的地方

如果你的生成对象不是直接几何，而是后续由 decoder 解码的 latent code $z_{shape}$，这两篇论文仍然可以分层使用：

- realism/feasibility 由 decoder 与训练数据流形保证；
- performance estimator 可以接在 decoder 后面对真实几何评估；
- DPP similarity 可以定义在 latent code、解码几何或学习到的 shape feature 上；
- 如果 latent 欧氏距离与几何差异不一致，**不应直接默认在 latent 上用 RBF 就等价于设计多样性**；最好验证距离的语义，或在解码后的几何/特征空间计算 kernel。

对于条件生成 latent code，PcDGAN 的最大价值不是 GAN 本身，而是它给出了一个通用目标：

> **同一连续需求下，生成多个性能都正确、形态又互不重复的候选。**

## 11. 两篇论文的逻辑脉络图

```text
普通 GAN
  │
  ├─ 问题：mode collapse；只拟合数据，不关心工程性能
  │
  ▼
PaDGAN
  ├─ DPP similarity → diversity
  ├─ performance q(x) → quality
  └─ 目标：高性能 + 多样 + 可向高质量域外扩展
       │
       ├─ 但不能指定“我要性能恰好等于 y*”
       │
       ▼
CcGAN 的连续条件思想
  ├─ vicinal loss
  └─ continuous label input
       │
       ▼
PcDGAN
  ├─ singular vicinal loss → 均衡覆盖连续条件空间
  ├─ estimator + LLETS → 精确条件 quality
  ├─ DPP → 固定条件下的多解性
  └─ 目标：连续条件逆设计 + conditional diversity
```

## 12. 关联笔记

- [[PaDGAN - A Generative Adversarial Network for Performance Augmented Diverse Designs]]
- [[PcDGAN - A Continuous Conditional Diverse Generative Adversarial Network For Inverse Design]]
