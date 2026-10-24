---
title: "Transolver Is a Linear Transformer: Revisiting Physics-Attention Through the Lens of Linear Attention"
authors:
  - Wenjie Hu
  - Sidun Liu
  - Peng Qiao
  - Zhenglun Sun
  - Yong Dou
year: 2026
venue: "Proceedings of the AAAI Conference on Artificial Intelligence, 40(1), 408–416"
doi: "10.1609/aaai.v40i1.37003"
arxiv: "2511.06294v3"
url: "https://arxiv.org/abs/2511.06294"
code: "https://github.com/HiPRL/LinearNO"
source: "未提供本地 PDF；本笔记依据 arXiv 原文 HTML，原文 PDF 见 https://arxiv.org/pdf/2511.06294"
tags:
  - literature
  - neural-operator
  - transformer
  - linear-attention
  - physics-attention
  - PDE
  - scientific-machine-learning
  - aerodynamics
status: read
updated: 2026-10-24
---

> [!abstract] 摘要翻译
> Transolver 使用 Physics-Attention，将大量网格点软分配到少量 slice，聚合成 token，在 slice 之间执行自注意力，再把结果映射回网格点。本文指出，Physics-Attention 可以重写为一种特殊的线性注意力；实验还发现，移除 slice token 之间的自注意力在许多任务上并不降低误差，甚至会改善结果。作者据此提出 Linear Attention Neural Operator（LinearNO）：先放宽 slice 与 deslice 投影的共享约束，再移除 slice-token self-attention。LinearNO 在六个标准偏微分方程基准上取得有竞争力的结果，平均减少 40.0% 的参数量与 36.2% 的计算量；在 AirfRANS 和 ShapeNet-Car 两个工业级气动数据集上也取得更好的报告结果。

# 一句话理解

**这篇论文的核心不是“把 Transolver 换成另一种更快的 attention kernel”，而是先将 Physics-Attention 解释为低秩线性注意力，再提出删去 slice-token self-attention 的 LinearNO。** 但 LinearNO 仍然保留点到 latent slice、latent slice 回到点的全局信息通路；它不是纯逐点 MLP。

## 问题与动机

Transolver 的 Physics-Attention 通过 \(N\) 个网格点与 \(M\) 个 slice 之间的软分配降低全局交互的成本，其中通常 \(M\ll N\)。原始模块先把点特征聚合成 \(M\) 个 slice token，再在这 \(M\) 个 token 上执行标准自注意力，最后将更新后的 token 广播回 \(N\) 个点。

作者提出两个问题：

1. 如果 slice 和 deslice 本身已经构成一种低秩的全局信息传播，slice token 之间的自注意力是否真的必要？
2. 原始 Physics-Attention 中 slice 与 deslice 使用同一个可学习投影，这种对称约束是否会限制表达能力？

作者观察到原始 Transolver 在部分实验中产生较为均匀、区分度不足的 slice 权重，并假设共享的 Q/K 投影可能加剧这一问题。第二点是作者提出的机制解释，需要与后续独立消融结果一起阅读，不能直接视为唯一原因。

## 方法：从 Physics-Attention 到 LinearNO

### 1. 重写原始 Physics-Attention

令点特征为 \(H\in\mathbb{R}^{N\times C}\)，slice 数为 \(M\)。原始 Transolver 先计算点到 slice 的权重：

\[
W=\operatorname{Softmax}_{M}(\operatorname{Linear}^{Q/K}(H)),
\qquad W\in\mathbb{R}^{N\times M}.
\]

将特征投影为 \(V=\operatorname{Linear}^{V}(H)\)，再通过加权平均聚合成 slice token：

\[
Z=D^{-1}W^\top V,\qquad
D=\operatorname{diag}(W^\top\mathbf{1}).
\]

原始模块随后计算 \(Z'=\operatorname{SelfAttention}(Z)\)，再反切片：

\[
H'=WZ'.
\]

因此原始模块的结构可以概括为：

\[
H\;(N\text{ 个点})
\rightarrow Z\;(M\text{ 个 slice})
\rightarrow \operatorname{SelfAttention}(Z)
\rightarrow H'\;(N\text{ 个点}).
\]

### 2. 广义化：分开学习 Q 与 K 的投影

作者将原来共享的投影拆为独立的 \(\operatorname{Linear}^{Q}\) 和 \(\operatorname{Linear}^{K}\)，分别构造：

\[
\varphi(Q)=\operatorname{Softmax}_{M}(\operatorname{Linear}^{Q}(H)),
\]

\[
\psi^\top(K)=
\operatorname{Softmax}_{N}(\operatorname{Linear}^{K}(H)^\top).
\]

这里 \(\varphi(Q)\in\mathbb{R}^{N\times M}\)，而 \(\psi^\top(K)\in\mathbb{R}^{M\times N}\)。两侧分别在 slice 维和点维归一化，解除原始 slice/deslice 的共享权重约束。作者认为这种不对称的归一化和独立投影有助于形成更丰富的权重模式。

### 3. 简化：删除 slice-token self-attention

在解除对称约束后，作者移除 slice token 之间的 self-attention，得到：

\[
\boxed{
\operatorname{LinearNO}(H)=
\varphi(Q)\left[\psi^\top(K)\operatorname{Linear}^{V}(H)\right]
}
\]

计算顺序是先计算 \(M\times N\) 与 \(N\times C\) 的乘积，再将 \(M\times C\) 的 latent 表示映射回 \(N\times C\)。它没有 \(M\times M\) 的 token self-attention 矩阵。

在 \(M\) 固定时，原始 Physics-Attention 的主要复杂度可写为 \(O(NM+M^2)\)，LinearNO 则为 \(O(NM)\)。这里“线性”指关于网格点数 \(N\) 的复杂度；若 \(M\) 随 \(N\) 增长，仍不能无条件称为对所有维度均线性。

![论文中图 2：Physics-Attention 与 LinearNO 的结构对比](https://arxiv.org/html/2511.06294v3/pipeline.png)

*图 2。上方是 Transolver 的 slice + token self-attention + deslice；下方将 Q/K 投影分开，并删除 token self-attention。原文图直接嵌入，未用重绘图替代。*

## 理论解释

作者证明，在网格逐步细化、采样点服从空间测度、特征函数有界等假设下，LinearNO 在概率意义上收敛到连续积分核算子的 Monte Carlo 近似。直觉上，\(\psi^\top(K)V\) 聚合全域信息，\(\varphi(Q)\) 决定每个输出点如何读取 latent 汇总。

该结果支持 LinearNO 属于神经算子（Neural Operator）范畴，并解释其离散化行为；**它不是有限样本下的误差上界，也不意味着对任意几何、网格或工况都优于 Transolver**。

## 实验设置与结果

### 1. 六个标准 PDE 基准

论文使用 Airfoil、Pipe、Plasticity、Navier–Stokes（NS）、Darcy 与 Elasticity 六个标准任务。网格规模从 Elasticity 的 972 个点，到 Burgers 的 16,384 个点不等；Airfoil、Pipe、Plasticity 采用结构网格，NS、Darcy、Burgers 使用笛卡尔网格，Elasticity 使用非结构网格。主文表格报告相对 \(L^2\) 误差，越低越好。

| 任务 | Transolver 相对 \(L^2\) | LinearNO 相对 \(L^2\) |
|---|---:|---:|
| Airfoil | 0.0053 | **0.0049** |
| Pipe | 0.0030 | **0.0024** |
| Plasticity | 0.0013 | **0.0011** |
| NS | 0.0882 | **0.0699** |
| Darcy | 0.0055 | **0.0050** |
| Elasticity | 0.0065 | **0.0050** |

这些 Transolver 结果中部分为作者复现值；Transolver++ 的部分结果则是根据论文描述复现，因为原始实现不可用。比较应理解为作者报告的基准结果，而不是所有模型在完全相同代码库下重新训练的独立基准。

### 2. 工业级气动任务

作者进一步测试 AirfRANS 与 ShapeNet-Car，既预测物体表面与周围流场，也评估积分气动系数。表中数值来自论文的相对 \(L^2\) 误差；\(\rho\) 为 Spearman 秩相关系数，越接近 1 表示设计排序越一致。

| 数据集 / 指标 | Transolver | LinearNO |
|---|---:|---:|
| AirfRANS，体积场 | 0.0023 | **0.0011** |
| AirfRANS，表面场 | 0.0085 | **0.0077** |
| AirfRANS，升力系数 \(C_L\) | 0.1230 | **0.0491** |
| AirfRANS，升力排序 \(\rho_L\) | 0.9978 | **0.9992** |
| ShapeNet-Car，体积场 | 0.0221 | **0.0194** |
| ShapeNet-Car，表面场 | 0.0785 | **0.0754** |
| ShapeNet-Car，阻力系数 \(C_D\) | 0.0117 | **0.0106** |
| ShapeNet-Car，阻力排序 \(\rho_D\) | 0.9914 | **0.9925** |

作者还在 AirfRANS 上设置了训练与测试的雷诺数（Reynolds number）和迎角范围差异，考察分布外泛化。此结果仍属于特定数据集划分与物理模型生成的数据分布，不能等同于对真实飞行试验或任意新构型的验证。

### 3. 计算效率

在六个标准 PDE 任务上，作者报告 LinearNO 平均减少 **40.0% 参数量**和 **36.2% FLOPs（浮点运算量）**。这一效率提升来自删除 slice token 之间的 \(M\times M\) 自注意力，并让聚合与反聚合以线性注意力形式直接相乘，而不是来自移除点到 slice 的全局混合。

## 消融实验：哪些机制真正有用？

论文的关键消融是：即使保留原始 Physics-Attention 的共享投影，删除 slice self-attention 也常常不损害准确率；因此“必须先解绑 Q/K，才能去掉 slice attention”的解释并不充分。作者报告，去掉 slice attention 在多数设置中会改善误差；但改善幅度依任务而变，不能概括为每个任务都必然更好。

作者的两步消融表明，**仅做投影解绑并不稳定地产生全部收益，解绑与删除 self-attention 的组合效果最好**。在六个基准上，完整 LinearNO 的相对 \(L^2\) 误差分别为 0.0049、0.0024、0.0011、0.0699、0.0050 和 0.0050。

## 关键贡献

1. 将 Transolver 的 Physics-Attention 写成线性注意力的特殊形式，揭示它与低秩核积分算子的关系。
2. 通过分开 Q/K 投影、删除 latent token self-attention，得到 LinearNO。
3. 在六个标准 PDE 基准和两个工业级气动数据集上报告精度优势，同时降低参数量与计算量。
4. 给出离散网格细化极限下的积分核算子解释，为该结构的算子学习属性提供理论支撑。

## 局限与证据边界

- **不是“slice/deslice 没用”。** LinearNO 仍然需要全局聚合与反聚合；它删除的是 slice token 之间的 self-attention。
- **投影解绑的解释并非最终定论。** 2026 年后续论文在九个三维流体与气动基准上进一步控制“是否使用 token attention”和“是否共享投影”两个因素，发现共享投影、无 token attention 的版本同样可以接近基线。见 [[Literature/transformer/Does Transolver really need a Transformer|Does Transolver really need a Transformer?]]。
- **复杂度声明有条件。** \(O(NM)\) 依赖 slice 数 \(M\) 的取值；此外，真实训练时间还受点级特征、MLP、内存读写与实现影响。
- **跨论文误差不可直接横比。** 数据划分、输入特征、训练预算与指标不同；应优先比较同一论文内、相同设置下的消融。
- **对气动设计的意义是结构假设，不是自动成立的结论。** 需要在目标几何、工况、积分气动力和梯度质量上重新验证。

## 与相关工作的关系

- [[Literature/transformer/Transolver|Transolver]]：前作的 Physics-Attention 是本文被重写与简化的对象。
- [[Literature/transformer/Does Transolver really need a Transformer|Does Transolver really need a Transformer?]]：后续工作在更多工业气动任务中检验“token attention 是否必要”，并对投影解绑的必要性提出不同解释。
- [[Literature/Topics/Transformer与物理场算子|Transformer 与物理场算子]]：从物理切片、低秩全局通信、局部细节与任意查询的角度比较不同算子结构。

## 参考资料

- Hu et al., *Transolver Is a Linear Transformer: Revisiting Physics-Attention Through the Lens of Linear Attention*, AAAI 2026. [论文](https://arxiv.org/abs/2511.06294) · [官方代码](https://github.com/HiPRL/LinearNO)
- Wu et al., *Transolver: A Fast Transformer Solver for PDEs on General Geometries*, ICML 2024. [论文](https://arxiv.org/abs/2402.02366)
