---
title: "Lost in the Middle at Birth: An Exact Theory of Transformer Position Bias"
authors:
  - "Borun D Chowdhury"
year: 2026
venue: "arXiv preprint arXiv:2603.10123 [cs.LG]"
doi: "10.48550/arXiv.2603.10123"
source: "/mnt/data/U.pdf"
tags:
  - transformer
  - long-context
  - position-bias
  - causal-attention
  - residual-connection
  - jacobian
  - rope
---

> [!abstract] 原文摘要翻译
> “Lost in the Middle（迷失在中间）”现象指：大型语言模型从上下文开头和结尾检索信息时表现较好，而对位于上下文中部的信息表现显著下降，形成典型的 U 形性能曲线。既有解释常把这种现象归因于训练后形成的 Softmax 注意力伪影，或者 Rotary Position Embedding（RoPE，旋转位置嵌入）等位置编码带来的距离衰减。本文提出一个更强且更精确的结论：**U 形位置偏置在模型初始化时就已经存在，在任何训练发生之前、甚至在位置编码尚未产生有效影响时就存在。**作者把多层因果注意力建模为 Cesàro 矩阵的迭代幂，并在连续极限下推导出精确的闭式影响密度。因果掩码使梯度影响在提示开头出现对数发散的“Primacy Tail（首因尾部）”，而残差连接在最后一个 token 处产生一个孤立的 $O(1)$ “Recency Delta（近因 Delta 锚点）”。两端之间形成一个量级约为 $O\!\left(1/(H-1)!\right)$ 的中部“dead zone（死区）”，其中 $H$ 为网络深度，因此中间位置的信息检索和学习在结构上更困难。实验显示，未训练的 Qwen2 与 GPT-2 架构在 Step 0 就出现 U 形 Jacobian（雅可比）影响曲线，而且是否使用 RoPE 几乎不改变这一形状；进一步比较随机初始化与预训练模型发现，标准预训练虽然会学习局部的信息路由尖峰，却没有消除宏观 U 形谷底。本文并不声称这种偏置不可克服，也不声称修改 RoPE 没有价值，而是给出一个可解析的架构基线，以明确后续干预真正需要对抗的结构性偏置。

## 一句话结论

**Decoder-only Transformer 中，“因果掩码 + 残差连接”本身就会在初始化时产生“开头强—中间弱—结尾强”的 U 形信息路由先验：因果路径的层层复合制造首因偏置，纯残差高速通路制造末 token 的近因锚点，中间 token 则主要依赖被反复稀释的混合路径。**

这篇文章最重要的价值不是再一次证明“Lost in the Middle 存在”，而是把它从一个经验现象改写为一个**可以写出闭式公式的网络拓扑问题**。

---

# 1. 问题与动机

长上下文语言模型存在一个稳定的经验现象：相关信息放在上下文开头或结尾时容易被利用，放在中部时性能明显下降。过去主要有两类解释：

1. **首因偏置来自训练后的 Attention Sink（注意力汇）**：模型学会把多余注意力质量倾倒到前几个 token 上。
2. **近因偏置与中部长距离退化来自位置编码**：例如 RoPE 对相对距离的处理可能让模型更偏好局部、较近的 token。

本文的问题更基础：

> 如果完全不训练 Transformer，仅仅随机初始化权重，这个 U 形偏置是否已经存在？

如果答案是“是”，那么训练后的注意力模式和位置编码最多是在一个已有的结构性基线上继续塑形，而不是 U 形现象的唯一来源。

本文最终把问题转化为：**从最后一个 token 的输出往输入反向传播时，不同输入位置有多少条、什么强度的梯度/信息路由路径能够到达最终状态？**

---

# 2. 核心贡献

## 2.1 初始化时就存在 U 形位置偏置

作者证明并实验验证：随机初始化、尚未训练的深层 decoder-only Transformer 已经具有明显的 U 形输入位置影响曲线。

- 开头 token：由于因果图中的上游地位，经过多层后拥有大量复合路径，形成 **Primacy Tail（首因尾部）**。
- 中间 token：既没有最早 token 的组合路径优势，也无法走完整的纯残差高速通路，形成 **factorial dead zone（阶乘级死区）**。
- 最后 token：可以每层都只沿残差连接直接传到输出，形成 **Recency Anchor（近因锚点）**。

## 2.2 给出离散与连续两套精确公式

作者把初始化时的因果注意力写成 Cesàro 矩阵 $M$，并推导 $M^H$ 以及包含残差后的 $N^H$ 的精确形式；再令序列长度 $L\to\infty$，得到连续位置 $x\in(0,1]$ 上的闭式影响密度。

## 2.3 将 RoPE 从“结构性起因”中剥离

初始化时，query/key 近似各向同性随机向量。RoPE 本质上是正交旋转，因此不会改变随机内积的分布；于是它不会破坏初始化时因果 Softmax 的位置交换对称性。实验上，RoPE 与 No-RoPE 的 Step 0 Jacobian 曲线几乎一致。

## 2.4 解释为什么标准训练没有自然填平中部谷底

训练后模型确实会出现内容相关的局部尖峰，例如文档边界会被学成信息路由锚点，但宏观 U 形包络仍然存在。作者进一步提出：中部位置不仅前向影响弱，**用于学习这些位置的反向梯度也同样弱**，可视为位置相关的有效学习率

$$
\eta(x) \propto \rho_H(x).
$$

因此，中部位置在标准 next-token prediction（下一 token 预测）目标下更新得更慢，优化器更容易利用两端已经“好走”的路径。

---

# 3. 方法论：从完整 Transformer 化简到可解析的路由模型

## 3.1 研究对象：最后一个 token 对所有输入位置的 Jacobian

设输入序列长度为 $L$，最终关心最后位置 $L$ 的隐藏状态或 logits。本文用输入-输出 Jacobian 的位置切片衡量输入位置 $j$ 对最终预测的局部影响。

直观上，若

$$
\left\|\frac{\partial h_L^{(H)}}{\partial h_j^{(0)}}\right\|
$$

很小，那么改变输入位置 $j$ 对最终状态的一阶影响也很小；反向传播到这个位置的梯度也更弱。

> [!important] Jacobian 不是“检索准确率”本身
> 本文明确限定：Jacobian norm（雅可比范数）描述的是局部敏感度和梯度路由强度，并不等价于实际 QA / retrieval accuracy（检索准确率）。它是用来刻画架构先验的代理量。

## 3.2 为什么可以先忽略 FFN 与归一化层

完整 pre-norm Transformer block 中，真正直接混合不同 token 位置的是 attention；Feed-Forward Network（FFN，前馈网络）、RMSNorm / LayerNorm 都是逐 token 操作，因此其 Jacobian 在“位置维”上是 block-diagonal（块对角）的。

本文的化简原则是：

- **保留**：残差连接、因果 attention 的跨位置混合、Value-Output 投影的整体增益。
- **去掉/折叠**：RMSNorm 的位置无关缩放、FFN 的逐位置变换。

这并不意味着 FFN 对实际模型完全没影响，而是说它**不会单独创建从位置 $j$ 到位置 $i$ 的横向路由边**；所以如果只研究“位置拓扑的宏观形状”，可以先把它从 toy model（玩具模型）中去掉。

化简后的单层形式为

$$
h^{(1)}=(I+\tilde\alpha M)h^{(0)},
$$

经过 $H$ 层：

$$
h^{(H)}=(I+\tilde\alpha M)^H h^{(0)}.
$$

为了只比较位置分布的形状，作者进一步归一化

$$
\alpha=\frac{\tilde\alpha}{1+\tilde\alpha},
$$

并定义

$$
N=(1-\alpha)I+\alpha M.
$$

这样 $N$ 每行和为 1，可以把 $(N^H)_{L,j}$ 直接理解成最后 token 从源位置 $j$ 获得的归一化结构影响。

---

# 4. 初始化时为什么因果注意力变成 Cesàro 矩阵

## 4.1 初始化的均匀因果 attention

随机初始化时，query 与 key 尚未学到内容相关结构。本文在宽网络/期望意义下使用

$$
q_i\cdot k_j \approx 0,
$$

因此对第 $i$ 个 token，可见的历史位置 $1,\dots,i$ 近似等权：

$$
A_{ij}\approx \frac{1}{i},\qquad j\le i.
$$

于是位置路由矩阵成为下三角矩阵

$$
M_{ij}=\begin{cases}
\frac{1}{i}, & j\le i,\\
0, & j>i.
\end{cases}
$$

这就是本文所称的 **Cesàro matrix（Cesàro 矩阵）**。

从工程直觉看，它非常简单：**第 $i$ 行就是“对前 $i$ 个 token 做前缀平均”**。多层堆叠相当于反复做因果前缀平均，而反复平均并不会保持位置对称性——早期 token 会进入更多后续前缀，再被下一层继续平均，因此影响会逐层积累。

## 4.2 精确离散解

纯因果 attention、无残差时，经过 $H$ 层后，从输入位置 $j$ 路由到最后位置 $L$ 的影响是 $(M^H)_{L,j}$。本文给出闭式表达：

$$
(M^H)_{L,j}
=
\binom{L-1}{j-1}
\sum_{m=j}^{L}
(-1)^{m-j}
\binom{L-j}{m-j}
\left(\frac{1}{m}\right)^H.
$$

加入残差后：

$$
N=(1-\alpha)I+\alpha M.
$$

由于 $I$ 与 $M$ 可交换，直接用二项式展开：

$$
N^H
=
\sum_{r=0}^{H}
\binom{H}{r}
\alpha^r(1-\alpha)^{H-r}M^r.
$$

这个式子特别重要，因为它把所有可能路径分成：

- $r=0$：一层 attention 都不走，只走残差；
- $r=1$：只在一层发生因果混合；
- $r=2$：在两层发生因果混合；
- ……
- $r=H$：每层都走因果 attention。

因此可以把残差 Transformer 理解成**不同“因果混合次数”的路径集合的二项式叠加**。

---

# 5. 连续极限：U 形的闭式结构

令离散位置归一化为

$$
x=\frac{j}{L}\in(0,1].
$$

当 $L\to\infty$，Cesàro 矩阵变为连续因果平均算子

$$
(\mathcal{M}h)(y)=\int_0^y \frac{1}{y}h(x)\,dx.
$$

作者把这个高度简化的连续模型戏称为 **Spherical-Cow Operator**：这里的 “spherical cow（球形奶牛）” 是理论建模中“为了得到可解析结果而主动极度简化现实系统”的比喻。

## 5.1 只有因果掩码：Primacy Tail

无残差时，$H$ 层连续影响密度为

$$
\rho_H^{(M)}(x)
=
\frac{1}{(H-1)!}
\left(\ln\frac{1}{x}\right)^{H-1}.
$$

当 $x\to0$：

$$
\ln\frac{1}{x}\to\infty,
$$

因此最前面的 token 影响发散，形成首因偏置。

### 直觉

第一个 token 位于所有后续 token 的因果上游。层数越深，越多“先被后面的 token 聚合，再被更后面的 token 聚合”的组合路径能够把它的信息带到最后位置。因此即使每一层单独都是均匀 attention，多层复合后也会自动把影响向左端集中。

> [!note] 一层没有 Lost in the Middle
> 当 $H=1$ 时，$\rho_1^{(M)}(x)=1$，纯因果 attention 的位置影响是平的。U 形不是“单层 mask”立即造成的，而是**深度下的因果复合 + 残差高速通路**共同造成的。

## 5.2 加入残差：Recency Anchor

加入残差后的连续密度为

$$
\rho_H^{(N)}(x)
=
(1-\alpha)^H\delta(1-x)
+
\sum_{r=1}^{H}
\binom{H}{r}
(1-\alpha)^{H-r}\alpha^r
\frac{1}{(r-1)!}
\left(\ln\frac{1}{x}\right)^{r-1}.
$$

其中

$$
(1-\alpha)^H\delta(1-x)
$$

就是最后位置的 **Recency Anchor**。

这里的 $\delta(1-x)$ 是 Dirac delta（狄拉克 Delta）：它不是普通意义上的“无限大函数”，而是在连续模型中用来表示“所有有限质量集中在一个点”的广义函数。对应到有限长度 Transformer，它表示一个非常具体的离散事实：**最后 token 可以在每一层都走 identity residual，不经过任何跨 token 平均，因此存在一条不被 Softmax 稀释的直接路径。**

## 5.3 中间为什么最弱：hybrid path 的稀释

中间 token 无法走 $r=0$ 的纯残差路径到最终位置，因为纯残差不会改变 token index；它必须至少某一层通过 attention 把信息送到更靠后的 token。

因此中部信息主要依赖 **hybrid paths（混合路径）**：

- 某些层走残差；
- 某些层走 causal attention；
- 一旦走 causal attention，就要承担前缀平均带来的 $1/i$ 稀释；
- 多次因果混合后，连续核中出现 $(r-1)!$ 的分母。

作者把这种结构性中部抑制概括为约

$$
O\!\left(\frac{1}{(H-1)!}\right)
$$

量级的 dead zone。更精确地说，完整残差模型是不同 $r$ 的项叠加；每个经历 $r$ 次因果混合的连续路径项都有 $(r-1)!$ 的阶乘抑制，而中间位置又缺少左右两端各自的特殊增强机制，所以宏观上形成深谷。

---

# 6. 从完整 attention Jacobian 到 Cesàro 路由核

单头 attention 的完整逐层 Jacobian 被拆成三部分：

$$
\frac{\partial h_i^{(l+1)}}{\partial h_j^{(l)}}
=
\delta_{ij}I_d
+
A_{ij}W_VW_O
+
\left(
\sum_{k\le i}
\frac{\partial A_{ik}}{\partial h_j^{(l)}}v_k
\right)W_O.
$$

分别是：

1. **Residual pathway（残差路径）**：$\delta_{ij}I_d$；
2. **Value Pathway（值路径）**：$A_{ij}W_VW_O$；
3. **Score Pathway（打分路径）**：输入改变 query/key，进而改变 attention score 和 Softmax 权重。

## 6.1 为什么初始化时只保留 Value Pathway

随机初始化、宽度 $d_k\to\infty$ 时，本文论证 Score Pathway 相比 Value Pathway 多一个

$$
O(d_k^{-1/2})
$$

抑制，因此在宽网络极限下消失。

于是

$$
\frac{\partial h_i^{(l+1)}}{\partial h_j^{(l)}}
\approx
\delta_{ij}I_d+A_{ij}W_VW_O.
$$

$W_VW_O$ 对所有位置对 $(i,j)$ 都相同，只负责 feature dimension（特征维）的混合和整体增益，不负责决定“哪个位置更强”。因此端到端 Jacobian 的位置形状可写成

$$
\rho(j)
\approx
\gamma^H (N^H)_{L,j},
\qquad
\gamma=\|W_VW_O\|.
$$

取对数后

$$
\log_{10}\rho(j)
=
H\log_{10}\gamma
+
\log_{10}(N^H)_{L,j}.
$$

第一项与位置无关，所以 U 形的形状完全由 $(N^H)_{L,j}$ 决定。

## 6.2 多头 attention 为什么不改变结论

初始化时各个 head 共享同样的均匀因果位置分布 $A_{ij}\approx1/i$，区别只在各自独立的 $W_V^hW_O^h$ 随机增益。

因此多头不是产生另一种位置拓扑，而是把**同一种 Cesàro 位置结构叠加多次**。作者认为随着 head 数增加，随机增益平均后，经验 Jacobian 反而会更集中到理论曲线附近，偏差约按

$$
O\!\left(\frac{1}{\sqrt{H_{\text{heads}}}}\right)
$$

缩小。

---

# 7. 为什么 RoPE 在初始化时不起结构作用

Rotary Position Embedding（RoPE，旋转位置嵌入）把 query/key 乘上与位置有关的正交旋转：

$$
S_{ij}
=
\frac{(R_iq_i)^\top(R_jk_j)}{\sqrt{d_k}}
=
\frac{q_i^\top R(i-j)k_j}{\sqrt{d_k}}.
$$

初始化时 $q_i,k_j$ 近似来自各向同性随机分布。正交旋转不会改变各向同性随机向量的分布，因此

$$
q_i^\top R(i-j)k_j
$$

与没有旋转时的随机内积分布相同。

因此本文的结论不是“RoPE 永远没有距离效应”，而是：

> **RoPE 不能解释 Step 0 时已经存在的 U 形拓扑。距离相关的 RoPE 偏置属于训练后可能出现的附加效应，而不是初始化 U 形的必要条件。**

这一区分非常关键，也与论文最后的 scope 相一致：修改 RoPE 仍可能改善训练后的长上下文利用，但它不是在修复最底层的 causal-residual topology（因果-残差拓扑）。

---

# 8. 论文中图 1：初始化时理论与实际网络一致

![[assets/Lost in the Middle at Birth - An Exact Theory of Transformer Position Bias/论文中图1.png]]

图 1 使用一个未训练的深层 Qwen2 架构直接测量输入-输出 Jacobian：

- **左列**：只有 causal attention、无 residual 时，理论和实测都表现为强烈的左端 Primacy Tail，但没有右端锚点。
- **中列**：加入 residual 后，理论和实测都出现“左端高 + 中间谷底 + 最右端尖峰”的完整非对称 U 形。
- **右列**：完整模型启用 RoPE 后，曲线与无 RoPE 情况基本重合。

实验配置：

- 24 层；
- hidden dimension $d=896$；
- 序列长度 $L=2048$；
- SwiGLU；
- RMSNorm；
- 随机高斯权重；
- 200 个 NaturalQuestions（NQ）样本；
- 残差归一化混合参数拟合为 $\alpha\approx0.881$。

理论与 Step 0 实测的匹配指标：

- Spearman rank correlation（斯皮尔曼秩相关）：$\rho=0.99$；
- Wasserstein distance（Wasserstein 距离）：$W=0.02$；
- RoPE vs. No-RoPE 初始化曲线的 Spearman 相关同样约为 $0.99$。

这张图是全文最核心的证据：**U 形不是训练后才出现，而是随机初始化的网络已经具有。**

---

# 9. 实验如何测 Jacobian：不是完整 Frobenius norm

如果对整个词表 logits $\ell$ 计算输入位置 $j$ 的完整 Jacobian：

$$
J_j=\frac{\partial \ell}{\partial e_j},
$$

其自然标量指标是 Frobenius norm（Frobenius 范数）

$$
\rho_F(j)=\|J_j\|_F.
$$

但这在大词表下计算很昂贵。本文采用单次反向传播的 scalar probe（标量探针）代理：令固定探针向量 $u$ 为全 1，定义

$$
s=u^\top\ell,
$$

然后一次 backward 得到

$$
g_j=\frac{\partial s}{\partial e_j}=J_j^\top u,
$$

实际画的是

$$
\rho(j)=\|g_j\|_2.
$$

作者承认 $\|J_j^\top u\|_2$ **不等于** $\|J_j\|_F$。其合理性来自结构分解：在本文关注的初始化拓扑中，可近似写成

$$
J_j\approx (N^H)_{L,j}C,
$$

其中 $C$ 与位置无关，因此无论用 Frobenius norm 还是这个 scalar-probe L2 norm，位置依赖都只通过同一个 $(N^H)_{L,j}$ 进入，宏观曲线形状相同。

---

# 10. 论文中图 2：预训练学会局部路由，但没有消除宏观谷底

![[assets/Lost in the Middle at Birth - An Exact Theory of Transformer Position Bias/论文中图2.png]]

这组实验比较完全相同架构的两种状态：

- 左列：随机初始化；
- 右列：完成预训练。

并设计两类输入：

### Vanilla context

每个序列来自单个 NQ 文档，截断到 2048 token。

### Chunked context

每个序列由多个**不同文档的 300-token 片段**直接拼接：

- 不插 separator token（分隔符）；
- 边界固定在 0、300、600、……；
- 对 200 个样本求统计；
- 图中阴影为 p16–p84 percentile band（第 16–84 百分位区间）。

### 结果

随机初始化模型完全不关心内容边界，因此 Vanilla 与 Chunked 都呈平滑 Cesàro U 形。

预训练模型则出现两层结构：

1. **宏观层**：U 形仍然存在；
2. **局部层**：Chunked 条件下在每个 300-token 边界出现尖峰。

这说明训练后的 Score Pathway 能学习“这个位置很像文档边界，应当形成信息锚点”，但它是在既有 U 形骨架之上添加局部 routing spike（路由尖峰），而不是把整个中间区域拉平。

论文还报告峰谷比在对数尺度上大约从初始化的 $10^2$ 增加到预训练后的 $10^3$，即标准训练并没有压缩这个谷底。

> [!note] 局部 U 形
> 作者进一步提出，只要某个子区间被 chunk boundary、delimiter 或 segment marker 之类的 attention anchor 划分出来，相同的 causal-residual 几何可以在该子区间内重新出现，因此这种先验具有一定的“尺度自由”特征。

---

# 11. 论文中图 3：训练前 100 步中，Score Pathway 开始活动，但谷底不抬升

![[assets/Lost in the Middle at Birth - An Exact Theory of Transformer Position Bias/论文中图3.png]]

这项 micro-training（微型训练）实验用于观察模型从 Step 0 开始发生什么。

## 11.1 模型与数据

- Qwen2-0.5B 架构，约 494M 参数；
- 24 层；
- 14Q/2KV Grouped-Query Attention（GQA，分组查询注意力）；
- $d=896$；
- SwiGLU；
- RMSNorm；
- RoPE $\theta=10^6$；
- 序列长度 $L=2048$；
- 不加载任何预训练权重。

从 NQ validation set 中取前 60 个样本：

- 0–49：训练池；
- 50–59：held-out evaluation（留出评估），训练期间绝不使用。

## 11.2 优化设置

- 训练 100 steps；
- AdamW；
- learning rate 从 $5\times10^{-4}$ 通过 cosine schedule 衰减到 $5\times10^{-5}$；
- weight decay = 0.1；
- gradient clipping = 1.0；
- batch size = 1；
- 每一步随机选择训练样本并截取连续 2048-token window；
- 标准 next-token cross-entropy（下一 token 交叉熵）损失。

## 11.3 Jacobian 评估

每次参数更新之前，以及最终第 100 步后，分别对 10 个 Vanilla 和 10 个 Chunked 留出序列计算完整位置 Jacobian profile，因此得到两个

$$
(10,101,2048)
$$

数组。

## 11.4 结果

- Step 0：平滑的 Cesàro U 形。
- 大约 50 steps 内：出现越来越明显的局部尖峰，说明训练后的 Score Pathway 开始参与内容相关 routing。
- 但宏观 U 形 envelope（包络）保持。
- 把最右端 Recency Anchor 归一化为 1 后，中部谷底并没有被抬高，反而相对变深。

这支持作者的解释：标准优化器会优先利用已经存在的几何高速路径，而不会自发付出足够代价去“填平”阶乘抑制的中间区域。

---

# 12. 跨架构验证

附录进一步测试 GPT-2 Small 与 GPT-2 Medium：

- GPT-2 Small：12 层；
- GPT-2 Medium：24 层；
- 与 Qwen2 不同，它们使用 LayerNorm、absolute positional encoding（绝对位置编码）与 GELU。

尽管归一化、激活函数和位置编码类型不同，初始化时仍出现同类非对称 U 形；预训练后宏观形状也继续存在。

这部分实验的目的不是证明所有可能 Transformer 变体都一定如此，而是说明该现象并非 Qwen2 + RoPE + RMSNorm 的偶然组合。

---

# 13. 三篇核心相关工作与研究脉络

## 13.1 Lost in the Middle：先发现经验 U 形

最早系统展示长上下文位置敏感性的工作之一，通过 multi-document question answering（多文档问答）和 key-value retrieval（键值检索）控制“相关信息所在位置”，观察到开头/结尾性能高、中间性能低的 U 形曲线。

**它回答的是：现象是否存在。**

本文接着追问：**为什么即使还没训练，这个形状就可能已经存在？**

## 13.2 Attention Sinks：发现早期 token 会吸收大量注意力

StreamingLLM 工作发现，很多语言模型会把很强的注意力分配给最早若干 token，即使这些 token 并不具备对应语义重要性，并将其称为 Attention Sink。

**它强调的是：早期位置确实会成为注意力汇。**

本文进一步把“首因偏置”解释成因果图的结构结果：即便没有训练，反复的均匀 causal averaging 也会让早期 token 占据更多复合路径。因此训练后的 Attention Sink 可以看作在一个先天 Primacy Tail 上进一步被利用和放大。

## 13.3 Residual-aware position bias：把 residual 纳入位置偏置理论

与本文最接近的理论工作把 residual connection 加入 cumulative attention rollout（累积注意力展开），指出：

- causal masking 产生 primacy；
- residual connection 防止所有影响完全坍缩到第一个 token，并产生 recency；
- 有限深度下自然得到 U 形。

本文与其关键区别在于：**不依赖从训练后模型测得的 residual mixing 参数来建立“架构先验”，而是在随机初始化条件下直接推导离散闭式解与连续密度。**

### 脉络总结

可以把这一研究链条压缩为：

**经验发现 U 形 → 发现早期 Attention Sink → 图论解释 causal primacy → residual-aware 理论补出 recency → 本文把整个 U 形推进到“Step 0 精确闭式基线”。**

---

# 14. 关键概念补充（外部资料）

> [!info] 本节不是论文原文，而是为理解论文补充的外部背景。

## 14.1 Cesàro matrix 为什么恰好对应初始化因果 attention

经典 Cesàro summation（Cesàro 求和）的一阶形式本质上就是对一个序列的前缀做 arithmetic average（算术平均）。用矩阵写，就是第 $i$ 行只在前 $i$ 个位置非零，并以相同权重做平均。

所以本文不是随意找了一个数学矩阵去拟合 attention，而是因为随机初始化的 causal Softmax 正好把第 $i$ 个 token 的可见前缀近似平均，天然得到 Cesàro 矩阵。

参考：Encyclopedia of Mathematics, “Cesàro summation methods”。

## 14.2 Dirac delta 在这里到底表示什么

Dirac delta $\delta(x-a)$ 的关键性质是

$$
\int f(x)\delta(x-a)\,dx=f(a).
$$

它用于表示“质量集中在一个点”。因此式子里的 $\delta(1-x)$ 不应读成“最后 token 的真实梯度是无穷大”，而应读成：**在从离散位置转到连续密度之后，最后一个 token 那条离散的 residual-only 质量没有被摊到连续区间，而是保留成一个原子质量。**

参考：Wolfram MathWorld, “Delta Function”。

## 14.3 “Spherical cow” 不是新算法

“Spherical cow” 是物理学里形容极端理想化 toy model 的幽默说法。本文把连续极限算子称为 Spherical-Cow Operator，强调它是把复杂 Transformer 剥离到“只剩位置路由骨架”的理想模型，而不是一个需要实现的新网络层。

参考：Wikipedia, “Spherical cow”。

## 14.4 Wasserstein distance 在图 1 中的含义

Wasserstein distance 是比较两个概率分布形状差异的一类距离，可用 optimal transport（最优传输）的“搬运质量成本”理解。这里它与 Spearman correlation 配合：

- Spearman 主要检查位置排序是否一致；
- Wasserstein 更关注整体分布形状是否接近。

因此 $\rho=0.99$ 与 $W=0.02$ 一起表明理论曲线不仅“趋势一致”，整体位置质量分布也很接近实验。

---

# 15. 局限与需要谨慎理解的地方

## 15.1 精确理论只覆盖初始化时的线性 Value Pathway

论文真正能给出闭式解析的核心前提是：初始化时 Score Pathway 在宽度极限下被 $O(d_k^{-1/2})$ 抑制。模型训练以后，attention score 已经是强烈 data-dependent（数据相关）的非线性函数，作者没有给出训练后 Softmax 的闭式上界，而是转为经验测量 Jacobian。

因此：

**“初始化时的 U 形是可证明的”** 和 **“任何训练后的 Transformer 都无法克服 U 形”** 是两件不同的事。本文只支持前者，并用实验显示标准预训练通常没有克服它。

## 15.2 Jacobian surrogate 与真实检索性能之间仍有距离

Jacobian 衡量局部一阶敏感度，并不直接等于模型能否从中间位置成功“读出答案”。尤其训练后网络有非线性 gating、attention score 重排和 feature interaction，局部梯度只能作为机制指标之一。

## 15.3 scalar probe 不是完整 Frobenius Jacobian

实验为了计算可行性，用全 1 logit probe 的单次 backward 代替完整 Frobenius norm。作者给出了“位置核可因子化”的理由来证明宏观形状应保持，但这仍依赖其结构近似。

## 15.4 对 RoPE 工程工作的表述需要区分“初始化起因”和“训练后作用”

正文某处使用了很强的措辞，认为单纯 flatten relative positional encoding（展平相对位置编码衰减）误解了问题几何；但论文自己的 scope 又明确表示并不声称 RoPE 修改无用。

更稳妥的读法是：

- **RoPE 不是 Step 0 U 形的必要原因；**
- **RoPE 仍可能是训练后长上下文退化的一部分，因此修改它仍可能有工程收益；**
- 如果只改 RoPE，而不改变 causal-residual 路由或训练目标，理论上仍保留本文指出的架构基线。

## 15.5 实验覆盖面有限

主要实证集中在：

- Qwen2-0.5B；
- GPT-2 Small / Medium；
- NaturalQuestions；
- 标准 next-token pretraining / 微型训练。

论文没有系统测试 aggressive position-targeted fine-tuning（强位置定向微调）、专门的中部 curriculum、显式中部 loss reweighting 等干预能把谷底抬到什么程度。

---

# 16. 启发

## 16.1 对长上下文训练：问题可能首先是“梯度到不了”，而不只是“注意力没学好”

如果位置 $x$ 的前向影响和反向梯度都带有同一个 $\rho_H(x)$ 因子，那么标准全局学习率无法弥补位置差异。一个中部 token 即使携带重要监督信号，也可能因为结构路径弱而学得更慢。

论文直接建议的方向包括：

- explicit middle-context curriculum learning（显式中部上下文课程学习）；
- targeted loss weighting（按位置定向加权 loss）；
- over-sampling needle-in-a-haystack data（过采样“草堆找针”式样本）。

这些方法的共同点不是修改一个位置编码公式，而是**主动给中部位置更大的训练压力**。

## 16.2 对架构设计：应关注“跨层位置路由图”而不是单层 attention map

单层均匀 attention 看起来完全公平，但 $M^H$ 并不公平。由此可见：

> 判断长上下文位置偏置时，只看某一层 attention heatmap 可能不够；真正相关的是跨层 residual + attention 的累计 path topology（路径拓扑）。

这对任何想设计长上下文 Transformer 变体的工作都有启发：应该分析端到端 rollout / Jacobian / path count，而不只是局部 attention score。

## 16.3 对“减少 Lost in the Middle”的设计目标

从本文理论出发，可以把改进策略分为三个层级：

1. **改变训练目标**：直接增加中部位置的有效梯度；
2. **改变路由结构**：让中间 token 拥有不需要多次前缀平均稀释的高速路径；
3. **改变局部注意力偏置**：例如位置编码、attention calibration 等。

本文最核心的立场是：第 3 类方法即使有效，也是在一个更底层的 causal-residual baseline 上工作；如果想“根治”而不是只缓解，需要考虑前两层。

---

# 17. 关联

- [[Lost in the Middle]]
- [[Attention Sink]]
- [[Long Context Transformer]]
- [[Causal Mask]]
- [[Residual Connection]]
- [[RoPE]]
- [[Jacobian]]
- [[Attention Rollout]]
- [[Grouped-Query Attention]]
- [[NaturalQuestions]]

## 与已有 Transformer 理解的直接连接

- 多头注意力本身不是本文 U 形的来源；初始化时所有 head 的位置核相同，多头主要降低随机波动。
- FFN / SwiGLU 不直接混合 token 位置，因此本文把它们视为“纵向 feature transform”，而把 attention + residual 视为“横向 position routing”。
- RoPE 改的是 query-key score 几何，但在随机各向同性初始化时，正交旋转不改变 score 分布，所以不会改变 Step 0 的位置核。
- 训练开始后 Score Pathway 被激活，模型才真正开始学习 content-dependent routing；因此可以把“初始化 Cesàro 骨架”和“训练后 Score Pathway 长出的局部尖峰”分别理解为**先验骨架**与**任务相关路由**。

---

# 18. 最终理解框架

可以用下面四步记住全文：

1. **初始化 attention 近似前缀平均**：$A_{ij}\approx1/i$，所以是 Cesàro 矩阵 $M$。
2. **深度复合产生 primacy**：$M^H$ 使早期 token 通过大量因果路径反复进入后续状态，连续极限得到 $\left(\ln 1/x\right)^{H-1}/(H-1)!$。
3. **残差产生 recency**：最后 token 有一条完全不经过 attention 稀释的 residual-only path，在连续密度中表现为 $\delta(1-x)$。
4. **中部两头不占**：既没有早期 token 的组合路径优势，也没有最后 token 的纯残差锚点，只能依赖被 Softmax 前缀平均稀释的 hybrid paths，于是形成 U 形谷底；标准训练能在其上学习局部尖峰，但通常不会自然填平整个谷底。

这就是本文所谓 **“Lost in the Middle at Birth”**：模型还没学会任何东西之前，中间位置就已经处在更不利的信息与梯度路由几何中。
