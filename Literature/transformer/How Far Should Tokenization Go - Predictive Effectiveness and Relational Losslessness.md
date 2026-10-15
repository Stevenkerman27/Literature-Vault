---
title: "How Far Should Tokenization Go? Predictive Effectiveness and Relational Losslessness"
authors:
  - Yi Wang
year: 2026
venue: "arXiv (cs.LG)"
doi: "10.48550/arXiv.2608.18025"
source: "/mnt/data/2608.18025v2.pdf"
arxiv: "2608.18025v2"
tags:
  - tokenization
  - predictive-codelength
  - transformer
  - symbolic-music
  - representation-learning
  - information-theory
---

> [!abstract] 原文摘要翻译
> GPT 风格模型依赖有限、可复用的 token 词表，因此 token 接口本身就是现代序列建模的重要组成部分。符号音乐天然由离散音符事件以及和弦、动机、乐句等重复结构组成，看起来非常适合 token 化；但离开自然语言后，每个领域都必须自行决定 token 接口应该表示什么，以及 token 化应该推进到什么程度。本文以**预测码长（predictive codelength）**为共同判据，提出 **Effectiveness–Losslessness Framework（有效性–无损性框架）**。其中，**Fact–Token Boundary（事实–Token 边界）**决定哪些由观测本身确定的结构应进入静态 token 接口；进入接口后，可以再通过 BPE（Byte Pair Encoding，字节对编码）等方式对载体进行可逆重编码。**Token–State Boundary（Token–状态边界）**则规定 token 化应该在何处停止：依赖上下文才能确定的关系，不应由 tokenizer 预先固定，而应留给模型的 contextual state（上下文状态）计算。作者在符号音乐上进行多随机种子受控实验，并在独立语料上复现时间坐标干预方向。显式音乐时间、调性框架规范化和音高分解均降低预测码长；固定五度圈几何反而增加预测码长；BPE 虽使载体序列长度减少 56.8%，却在所有随机种子上增加预测码长。论文据此主张：tokenization 的质量不能由“是否可逆”或“序列是否更短”单独判断，而应由**预测计算被分配在静态表示、载体编码和上下文状态中的位置**来判断。

> [!note] 图像说明
> 本笔记重新依据论文 PDF 的渲染页面核对了图 1、图 2 和图 4 的布局、颜色、箭头、误差棒和数值。当前运行时仍不能把用户上传 PDF 的原始图像字节直接导出到 Obsidian 包，因此 `assets/` 中是根据论文原图逐项复刻的高保真图形资产，而不是原 PDF 的像素级裁切；正文中的图号与论文一致。

# 一句话结论

**Tokenization 本质上是在分配预测计算：把“由观测确定、且能减轻有限模型预测负担”的结构提前放进静态表示；把“必须依赖上下文才能确定”的关系留给 Transformer 的状态计算。** 序列更短、表示可逆、包含领域先验，都不能单独推出 tokenizer 更好。

# 1. 问题与动机

论文追问的不是“哪一种 tokenizer 最好”，而是更基础的问题：

> **Tokenization 应该表示什么？应该做到哪一步停止？**

作者认为，现有 tokenizer 比较经常把不同层面的干预混在一起：

1. **Coordinate construction（坐标构造）**：改变模型面对的坐标，例如把序列位置改成真实音乐时间，把绝对音高拆成音级和音区；
2. **Carrier coding（载体编码）**：不改变所表示的事实，只改变这些事实如何被序列化，例如 BPE 合并符号；
3. **Relational commitment（关系承诺）**：在上下文出现前就预先固定某种关系几何，例如规定音级必须服从五度圈距离；
4. 同时，tokenization 还可能间接改变词表大小、每个事实对应的 autoregressive step 数、参数量和计算量。

如果只比较最终准确率、困惑度或 token 数量，就无法区分提升究竟来自哪一个操作。因此本文提出统一账本：在固定数据源、预测器族和资源预算后，比较某一表示操作是否降低**每个声明事实的预测码长**。

# 2. Effectiveness–Losslessness Framework

## 2.1 四层处理链

论文把从原始领域数据到预测状态的过程写成：

$$
x_{\mathrm{raw}}
\xrightarrow{A}x
\xrightarrow{R}z_{1:N}
\xrightarrow{C}u_{1:T}
\xrightarrow{F_\theta}h_{1:T}.
$$

其中：

- $A$：定义真正被建模的 **declared fact surface（声明事实表面）** $x$；
- $R$：把事实转换成模型可见的 **coordinate carriers（坐标载体）** $z_{1:N}$；
- $C$：把坐标载体序列化/组合成 token sequence $u_{1:T}$；
- $F_\theta$：上下文模型，根据历史 token 构造 $h_t=F_\theta(u_{<t})$，再由 $q_\theta(u_t\mid h_t)$ 预测当前 token。

![[assets/How Far Should Tokenization Go - Predictive Effectiveness and Relational Losslessness/Figure 1 - Effectiveness-Losslessness Framework.png]]

*论文中图 1：Effectiveness–Losslessness Framework。蓝色虚线区域表示 static representation，橙色虚线区域表示 contextual computation。`fixed relation g` 从 token sequence 分叉并以橙色虚线进入 contextual state，表示 tokenizer 侧提前固定关系的干预路径。*

这张图最重要的不是流程，而是两条边界：

- **Fact–Token / Effectiveness**：哪些观测决定的结构值得提前放到模型接口；
- **Token–State / Losslessness**：什么时候应该停止静态编码，把关系留给模型状态。

## 2.2 统一评价量：资源受限预测码长

对数据分布 $P$、预测器族 $\mathcal Q$、资源预算 $\mathcal B$，论文定义：

$$
\mathcal{L}_{P,\mathcal Q,\mathcal B}(R,C)=
\frac{
\inf_{q\in\mathcal Q_{\mathcal B}}
\mathbb E_{x\sim P}\left[
-\sum_{t=1}^{T}\log_2 q(u_t\mid u_{<t})
\right]
+\mathbb E_{x\sim P}[\ell_{\mathrm{side}}(x)]
}{
\mathbb E_{x\sim P}[N_{\mathrm{fact}}(x)]
},
\qquad u=C(R(x)).
$$

逐项解释：

- $-\log_2 q(u_t\mid u_{<t})$：模型给当前目标分配的 surprise，以 bit 计；
- 对整条序列求和，相当于用模型概率给数据做预测编码；
- $\ell_{\mathrm{side}}$：为了可逆恢复而必须额外保存的 side information（旁信息）；
- 分母 $N_{\mathrm{fact}}$ 是原始声明事实数，而不是 token 数，所以不同 carrier 长度可以在同一单位上比较；
- $\mathcal B$ 固定模型容量、上下文、训练 exposure、优化过程与 checkpoint 选择策略。

论文特别强调：这不是完整的 **MDL（Minimum Description Length，最小描述长度）**。这里不计算模型参数本身的描述长度，而是一个**resource-bounded predictive data codelength（资源受限预测数据码长）**。因此一切结论都必须写成“在给定 $(P,\mathcal Q,\mathcal B)$ 下”。

## 2.3 三类干预对应三个指标

在匹配干预中，分别定义：

$$
G_{\mathrm{coord}}(R_0\rightarrow R_1)
=\mathcal L(R_0,C)-\mathcal L(R_1,C),
$$

$$
G_{\mathrm{carrier}}(C_0\rightarrow C_1)
=\mathcal L(R,C_0)-\mathcal L(R,C_1),
$$

$$
E_{\mathrm{rel}}(g)
=\mathcal L(R_g,C)-\mathcal L(R_{\mathrm{open}},C).
$$

含义：

- $G_{\mathrm{coord}}>0$：新坐标使模型更容易预测；
- $G_{\mathrm{carrier}}>0$：新载体编码使模型更容易预测；
- $E_{\mathrm{rel}}>0$：固定关系引入了额外预测码长，即 **relational excess**。

论文的方法论要求是：**只能对 matched within-family delta 做机制解释。** 跨 family 的绝对码长可以描述系统高低，但不能用来判断某个具体操作为什么有效。

# 3. Fact–Token Boundary：什么应该进入 token 接口？

## 3.1 Predictive Effectiveness Principle

如果一个由观测事实确定的操作，在匹配资源约束下带来：

$$
G_{\mathrm{coord}}>0,
$$

则它是 **predictively effective（预测有效）** 的，适合进入静态模型接口。

核心观点是：

> **Information preservation does not imply computational equivalence.**

两种表示可以携带完全相同的信息，但有限容量模型访问这些信息的难度可以不同。因此“可逆”只能说明事实没有丢失，不能说明表示对当前 Transformer 同样容易。

## 3.2 Decoupling：把耦合坐标拆成可单独寻址的因子

对事实 $x=(a,b)$：

$$
R_{\mathrm{dec}}(a,b)=\bigl(r_A(a),r_B(b)\bigr),
$$

并要求可恢复：

$$
\operatorname{Rec}_{\mathrm{dec}}(R_{\mathrm{dec}}(a,b))=(a,b).
$$

但是 decoupling **不假设统计独立**：

$$
P(A,B)\neq P(A)P(B)
$$

完全允许成立。它只是取消“两个因素必须通过一个耦合坐标才能访问”的表示约束，统计依赖仍由上下文模型学习。

论文中的 pitch class/register 分解就是典型 decoupling。

## 3.3 Denesting 与 canonicalization

对由观测决定的变换 $\phi$：

$$
R_\phi(x)=\bigl(\phi(x),s_\phi(x)\bigr),
$$

其中 $s_\phi(x)$ 是恢复原数据需要的旁信息，并满足：

$$
\operatorname{Rec}_\phi(R_\phi(x))=x.
$$

其中一种重要情况是 **canonicalization（规范化/典范化）**：把已知变换轨道中的样本对齐到一个统一参考框架，消除 nuisance variation，同时保留逆变换信息。

但作者同样不把“确定性/可逆”当成充分条件：只有实测 $G_{\mathrm{coord}}>0$ 才说明这个 canonicalization 对当前 learner 真正有帮助。

# 4. Carrier Coding：更短的序列不等于更低的预测负担

当坐标 $R(x)$ 已经固定后，$C$ 只改变这些坐标如何被 token 序列承载。BPE 属于这一层。

论文明确写出：

$$
|C_1(z)|<|C_0(z)|
\not\Rightarrow
\mathcal L(R,C_1)<\mathcal L(R,C_0).
$$

因此必须区分：

- **carrier compaction**：token 数更少；
- **predictive compression**：有限模型实际需要的预测 bit 更少。

本文的 J/K 实验给出强反例：BPE 大幅缩短序列，但预测码长反而显著增加。

# 5. Token–State Boundary：什么应该留给 Transformer 状态？

## 5.1 Relational Losslessness Principle

假设两个对象 $x_i,x_j$ 的“有用关系”依赖上下文 $\mathcal C$：

$$
\rho^{\mathcal C}_{ij}=f(x_i,x_j,\mathcal C),
$$

且存在两个上下文：

$$
\exists\,\mathcal C_1,\mathcal C_2:
\rho^{\mathcal C_1}_{ij}\neq\rho^{\mathcal C_2}_{ij}.
$$

如果 tokenizer 在上下文出现前就固定一个：

$$
g_{ij}=g(x_i,x_j),
$$

那么它就做了 **static relational commitment（静态关系承诺）**。

作者把只保留身份/坐标、但不提前规定这种几何的接口称为 **relation-open interface（关系开放接口）**。

即使固定关系表示仍然完全可逆：

$$
R_g^{-1}(R_g(x))=x,
$$

也不能推出：

$$
E_{\mathrm{rel}}(g)\le 0.
$$

因此本文所谓 relational losslessness 不是“字节级可逆”，而是**不因为过早固定上下文相关关系而给预测器增加负担**。

# 6. 符号音乐中的具体实现

## 6.1 声明事实与事件流

每个观察事件写为：

$$
n_i=(t_i,o_i,p_i,d_i),
$$

其中：

- $t_i$：event type，Note 或 REST；
- $o_i$：onset，起始时刻；
- $p_i$：pitch，音高；
- $d_i$：duration，持续时间。

模型只声明这些 score-level facts。演奏表现、MIDI 控制器、音色、micro-timing 等不属于声明事实。

所有非鼓音符被放进一个**不分声部的单事件流**。同一 onset 的音符仍保持为独立事件，并按 MIDI pitch 升序序列化；这个顺序不代表旋律优先级、声部身份或语义重要性。

Decoder 对每个事件采用：

$$
\mathrm{TYPE}\rightarrow\mathrm{TIME}\rightarrow\mathrm{PITCH}\rightarrow\mathrm{DURATION}.
$$

这是 exact-event joint distribution 的 chain-rule factorization（链式分解），不是音乐语义层级。

## 6.2 音乐时间坐标

普通 sequence index 只表示序列化顺序，不等价于真实音乐时间。作者把 onset 映射成：

$$
\tau_i=\phi(o_i)=
(\text{bar progress},\text{beat/bar/multibar phases}).
$$

附录进一步说明，C/D/E/H/I 使用：

- directed bar progress；
- within-beat phase；
- beat-within-bar phase；
- bar、4-bar、16-bar 多尺度 phase。

其中 directed component 用来消除周期 phase 的歧义，因此联合时间坐标在支持范围内可逆：

$$
\tau_i=\phi(o_i),\qquad \phi^{-1}(\tau_i)=o_i.
$$

D/E/H/I 还加入由保留时间坐标确定的 pairwise attention bias：

- signed onset displacement；
- within-bar displacement；
- signed bar distance；
- signed beat distance。

这些关系不需要乐句、和弦、声部或动机语义，因此作者仍把它们放在 Fact–Token 一侧。

## 6.3 Pitch factorization

绝对音高可双射分解为：

$$
c_i=p_i\bmod 12,
$$

$$
r_i=\left\lfloor\frac{p_i}{12}\right\rfloor,
$$

所以：

$$
p_i\longleftrightarrow(c_i,r_i).
$$

其中 $c_i$ 为 pitch class（音级），$r_i$ 为 register（音区）。H arm 使用二者的 learned categorical embeddings，不插入任何固定 pitch-class distance kernel，因此它是本文主要的 relation-open pitch interface。

## 6.4 Tonal canonicalization

F/G family 在同一 high-confidence tonal subset 上比较。G 对整首歌估计一个确定性 shift $s$：

$$
\tilde p_i=p_i-s,
$$

模型预测 canonical pitch $\tilde p_i$，最后在模型外恢复：

$$
p_i=\tilde p_i+s.
$$

重要约束：

- $s$ 不提供给 Transformer；
- tonic、mode、scale degree、harmonic function 都不进入模型；
- $s\in\{-5,\ldots,6\}$，共有 12 种可能，主账本使用固定 4 bit side code；
- 主结果保守地按每个独立 8-bar window 收 4 bit；另做 source-song amortization 审计。

所以 G 的收益不是“给了模型 key 标签”，而是**把一个可逆的全局移调自由度从学习问题里移出**。

## 6.5 固定音高几何：relation commitment

### Chromatic circle

$$
\gamma_{\mathrm{chr}}(p_i)=
\left[
\sin\frac{2\pi(p_i\bmod12)}{12},
\cos\frac{2\pi(p_i\bmod12)}{12}
\right].
$$

### Circle of fifths

先定义：

$$
f_i=\bigl(7(p_i\bmod12)\bigr)\bmod12,
$$

再映射：

$$
\gamma_{\mathrm{fif}}(p_i)=
\left[
\sin\frac{2\pi f_i}{12},
\cos\frac{2\pi f_i}{12}
\right].
$$

两个映射在 12 个音级上都保持单射，因此没有丢失 pitch-class identity；问题在于 phase inner product 会在 contextual attention 之前就固定“谁应该靠近谁”。这正是论文要检验的 relational commitment。

# 7. 主实验 setup

## 7.1 Pop1K7 数据与切分

| split | songs | windows |
|---|---:|---:|
| train | 1,378 | 18,460 |
| validation | 184 | 2,011 |
| sealed test | 185 | 1,979 |

关键数据完整性约束：**先按 source-song lineage 切分，再 windowing**。论文审计 exact、transposition-invariant、rhythm/interval duplicate component，跨 split 数量均为 0。

## 7.2 Transformer 与训练预算

共享 causal Transformer：

- 16 Transformer layers；
- width = 64；
- 4 attention heads；
- FFN（Feed-Forward Network，前馈网络）width = 128；
- dropout = 0；
- deterministic complete 8-bar windows；
- 最多 2,048 note tokens；
- AdamW；
- learning rate：$3\times10^{-4}\rightarrow3\times10^{-5}$ cosine schedule + warmup；
- batch size = 128 songs；
- 三个预注册种子：20260819、20260820、20260821；
- 最大预算 2,000 target-equivalent epochs；
- full-split exposure：3,650,482,000；
- matched-subset exposure：2,664,430,000；
- checkpoint 只按 validation native micro code 选择，tie 取更早 exposure；
- sealed test 仅在 checkpoint 锁定后访问一次。

不同 representation arm 的 active trainable parameter count 略有变化，论文把它视为完整接口干预的一部分，因此只允许 matched family 内做机制归因。

## 7.3 A–K 干预臂

| Arm | Active interface | 作用 |
|---|---|---|
| A | raw pitch identity + generic sequence position | 时间坐标基线 |
| B | A + learned absolute onset/bar lookup | 测绝对时间 exposure |
| C | raw pitch + unary ordered/multiscale metric time，替代 generic/absolute position | 测多尺度时间坐标 |
| D | C + pairwise onset/meter/bar/beat relations | 完整时间坐标 arm |
| E | D time + chromatic-circle phase + register | fixed chromatic relation |
| F | matched subset 上 original absolute pitch | canonicalization 基线 |
| G | whole-song tonal canonicalization，逆 shift 留在模型外 | canonicalization |
| H | categorical pitch class + categorical register + D time | pitch factorization / relation-open |
| I | fixed circle-of-fifths phase + register + D time | fixed fifths relation |
| J | event-preserving REMI-like BAR/POSITION/pitch-or-REST/duration stream | carrier 基线 |
| K | J + train-only reversible BPE | carrier recoding |

允许机制解释的 matched deltas：

- $G_{\mathrm{coord}}$：A–D，D–H，F–G；
- $E_{\mathrm{rel}}$：D–E，H–I；
- $G_{\mathrm{carrier}}$：J–K。

J–D、K–D 只能描述整套系统的码长差异，不能解释某一单独机制，因为它们同时改变了 event/field autoregression、decoder factorization、Transformer positions per fact 和 compute per fact。

# 8. 实验结果

![[assets/How Far Should Tokenization Go - Predictive Effectiveness and Relational Losslessness/Figure 2 - Pop1K7 clean-test codelength.png]]

*论文中图 2：Pop1K7 clean-test 每声明 musical fact 的预测码长。图 (a) 是 A–D 时间坐标；图 (b) 同时展示 D/H factorization 与 D/E、H/I relational excess；图 (c) 是 F/G canonicalization。误差棒为三个预注册随机种子的 sample SD。*

## 8.1 时间坐标是最强的 coordinate gain

逐步增益：

- A→B：$0.694$ bits/fact；
- B→C：$0.096$ bits/fact；
- C→D：$0.059$ bits/fact。

总计：

$$
G_{\mathrm{coord}}(A\rightarrow D)=0.84809\ \mathrm{bits/fact},
$$

相当于 **11.92%** 的预测码长下降，三个 seed 都支持 D。

更重要的是，A→D 的提升并不只来自 Time head。附录 D 的分解：

| factor | D−A (bits/fact) |
|---|---:|
| Type | -0.021304 |
| Time | -0.286924 |
| Pitch | -0.340213 |
| Duration | -0.199646 |
| **Total** | **-0.848087** |

只有约 $0.287$ bits/fact 来自 Time，本身约占三分之一；其余主要来自 Pitch 和 Duration。说明显式时间坐标改变的是整个 joint conditional distribution，而不是只让时间戳容易预测。

## 8.2 Pitch factorization：小但为正

D→H：

$$
G_{\mathrm{coord}}(D\rightarrow H)=0.03171\ \mathrm{bits/fact}.
$$

两个 seed 改善，一个近似持平。主要增益位于 Pitch 项：

$$
\Delta \mathrm{Pitch}=-0.02556\ \mathrm{bits/fact}.
$$

论文的结论不是“pitch class/register 永远优于 absolute pitch”，而是：**对当前受限 Transformer，两个可单独寻址的可逆因素略比单一绝对 pitch ID 更易利用。**

## 8.3 Tonal canonicalization：计入 side information 后仍有收益

F/G clean-test ledger：

| 账本 | F | G | $G_{coord}(F\to G)$ |
|---|---:|---:|---:|
| model code only | 6.48686 | 6.36174 | 0.12512 |
| 4 bits / 8-bar window | 6.48686 | 6.39631 | **0.09055** |
| 4 bits / source song | 6.48686 | 6.36478 | 0.12208 |

主结果采用最保守的 per-window side charge：

$$
G_{\mathrm{coord}}(F\rightarrow G)=0.09055\ \mathrm{bits/fact}.
$$

三个 seed 均改善。这表明：**可逆去除一个 nuisance degree of freedom，可能比让有限模型自己学习跨移调等价性更省码长。**

## 8.4 固定五度圈几何产生明显 relational excess

H 为 relation-open pitch-class identity，I 把 pitch class 映射到固定 circle-of-fifths phase。结果：

$$
E_{\mathrm{rel}}(\mathrm{fifths})
=\mathcal L(I)-\mathcal L(H)
=0.09831\ \mathrm{bits/fact}.
$$

三个 seed 全部退化，其中 Pitch 项贡献：

$$
0.08428\ \mathrm{bits/fact}.
$$

这构成 Token–State Boundary 的核心实验证据：五度圈本身当然是有意义的音乐结构，但**把它固定成所有上下文共享的模型输入几何**，会妨碍当前 learner 的预测。

D→E 的固定 chromatic geometry 只产生：

$$
E_{\mathrm{rel}}=0.01032\ \mathrm{bits/fact},
$$

且 seed 方向混合。因此作者只说“没有可靠收益”，而不把它升级为稳定有害结论。

## 8.5 BPE：序列缩短 56.8%，预测码长反而增加

J/K carrier control：

| Interface | serialized targets/fact | held-out bits/fact |
|---|---:|---:|
| J: uncompressed stream | 3.563 | $5.96305\pm0.10235$ |
| K: J + reversible BPE | 1.538 | $6.77367\pm0.05643$ |

token target 数减少：

$$
\frac{3.563-1.538}{3.563}\approx56.8\%.
$$

但：

$$
G_{\mathrm{carrier}}(J\rightarrow K)
=\mathcal L(J)-\mathcal L(K)
=-0.81061\ \mathrm{bits/fact}.
$$

三个 seed 的 K−J penalty：

- +0.95195；
- +0.82678；
- +0.65312 bits/fact。

这直接证明：**carrier compaction ≠ predictive compression**。

## 8.6 Figure 3：为什么作者不在同一个 epoch 上硬比较

论文附录 B 的 Figure 3 给出 A–D、F–G、D–H–I 三组 validation dynamics。每组都训练到相同最大预算 2,000 target-equivalent epochs；细线是独立 seed，粗线是三 seed 均值，空心 marker 是各 seed 按 validation 独立选出的 checkpoint。

这张图的目的不是再提供一个性能排名，而是排除“只是优化速度不同”的混淆。论文最终使用**统一最大预算内的最小 validation codelength**来锁 checkpoint，再做一次 sealed-test evaluation，而不是在一个任意共同 epoch 上比较。

## 8.7 独立 ComMU 方向性复现

ComMU 4/4：9,299 首歌，其中：

- train 8,652；
- validation 323；
- sealed test 324。

A 与 D 只改变 musical-time interface，其余目标、decoder、optimizer、batch/context policy、三 seed 与 checkpoint rule 匹配。

clean test：

- A：$12.33993\pm0.14311$ bits/fact；
- D：$12.22855\pm0.18073$ bits/fact；
- mean D−A = **−0.11139 bits/fact**。

三个 paired seed 全部支持 D。但所有 final curves 在 epoch 4.5→5 仍在改善，因此论文谨慎称为**directional replication（方向性复现）**，而不是收敛后的能力极限。

## 8.8 上下文关系 probe

![[assets/How Far Should Tokenization Go - Predictive Effectiveness and Relational Losslessness/Figure 4 - Context-relation intervention.png]]

*论文中图 4：secondary context-relation probe。替换左侧或右侧上下文都会显著降低 same-song relation accuracy，左右都替换后接近 50% chance。*

准确率：

- full context：75.93%；
- left replaced：64.51%；
- right replaced：63.56%；
- both replaced：50.87%。

另一个固定 target 的 predictive code probe：

| condition | bits/event | change from Full |
|---|---:|---:|
| FULL | 3.5394 | — |
| LEFT | 3.8083 | +0.2689 |
| RIGHT | 3.8117 | +0.2723 |
| SHUFFLED | 4.2084 | +0.6690 |

说明 state 不只是使用“有多少上下文”，而是使用**有序 content–time binding**。但作者明确强调：该 probe 使用不同 source/task/representation，只是 state-side supporting evidence，不能与 A–I 的主干预数值合并。

# 9. 关键概念补充（外部资料）

> [!info] 本节不是论文新增实验，而是为理解论文所做的外部概念补充。

## 9.1 Predictive V-information：为什么可逆表示仍可能有不同“可用信息”

Xu et al. 在 ICLR 2020 的 *A Theory of Usable Information Under Computational Constraints* 中提出 **predictive $\mathcal V$-information**。传统 Shannon mutual information 默认观察者可以进行任意复杂的解码；$\mathcal V$-information 则把预测器限制在某个函数族 $\mathcal V$，因此衡量的是“给定计算能力下，信息对这个预测器是否可用”。

这解释本文的核心逻辑：

- $R_1$ 与 $R_2$ 可以都完全可逆；
- 但对固定有限 Transformer，$R_1$ 可能更容易通过简单预测路径访问，因此 held-out codelength 更低。

也就是说，本文的 predictive effectiveness 实质上是在比较**representation 对受限 learner 的 computational accessibility（计算可达性）**。

外部参考：Xu et al., ICLR 2020, arXiv:2002.10689。

## 9.2 BPE 为什么不能只用压缩率解释

BPE（Byte Pair Encoding）最初源于压缩思路：不断合并高频相邻符号对。在语言模型里，它同时会改变：

- 序列长度；
- vocabulary size；
- 单 token entropy；
- 每个原始事实分配到多少 autoregressive steps；
- output head 参数量和计算位置。

Schmidt et al. 在 EMNLP 2024 的 *Tokenization Is More Than Compression* 中直接检验“更少 token 是否自动带来更好 LM”，结果是否定的。他们把 tokenizer 拆成 pre-tokenization、vocabulary construction、segmentation 三个阶段，并表明最短 segmentation 不保证最好 downstream performance。

本文进一步用 J/K matched intervention 把这一观点转化成严格的 codelength 控制实验：**carrier 变短 56.8%，但 predictive codelength 增加 0.811 bits/fact。**

外部参考：Schmidt et al., EMNLP 2024, DOI `10.18653/v1/2024.emnlp-main.40`。

## 9.3 Tokenizer 是统计/计算映射，而不只是“预处理脚本”

Gastaldi et al. 在 ICLR 2025 的 *The Foundations of Tokenization: Statistical and Computational Concerns* 中用 stochastic maps 形式化 tokenizer，并分析一致性、歧义、有限性和 sequentiality 等性质。

这条研究线把 tokenizer 提升为一个会改变学习问题本身的表示映射。本文沿着这一方向进一步区分：

1. coordinate construction；
2. reversible carrier recoding；
3. fixed relational commitment。

然后用统一预测码长分别给这三类操作记账。

外部参考：Gastaldi et al., ICLR 2025。

# 10. 文献脉络：3 篇核心相关工作

```text
Xu et al., 2020
受限模型下“可用信息”取决于 predictor family
        │
        └──> 本文：用 resource-bounded predictive codelength
              衡量表示是否更容易被当前 learner 利用

Gastaldi et al., 2025
把 tokenizer 形式化为独立的统计/计算映射
        │
        └──> 本文：进一步拆成 coordinate / carrier / relation 三类操作

Schmidt et al., 2024
“token 更少”不自动意味着 LM 更好
        │
        └──> 本文：J/K matched control 给出
              carrier compaction ≠ predictive compression 的直接证据
```

研究脉络可以概括为：

**计算受限的信息可用性** → **tokenizer 的独立理论地位** → **压缩长度不是效果充分解释** → 本文提出**两条边界 + 三个操作级码长指标**来决定预测计算应该放在哪里。

# 11. 论文主要贡献

1. **统一评价判据**：用每个声明事实的 predictive codelength 比较 tokenization 操作，而不是只看 recoverability、token count 或 downstream accuracy；
2. **Fact–Token Boundary**：观测决定的结构只有在实际降低有限模型预测负担时才应进入静态接口；
3. **Token–State Boundary**：依赖上下文的关系应尽量留给 contextual state，而不是 tokenizer 预先固定；
4. **Carrier coding 单独记账**：BPE 等可逆重编码不能与坐标设计混为一谈；
5. **操作级 matched interventions**：时间坐标、canonicalization、pitch factorization 有益；fixed fifths geometry 有害；BPE 序列压缩不等于预测压缩；
6. **实验协议严格**：source-song split、预注册 seeds、validation-only selection、sealed test、side-information ledger、独立 ComMU replication 都明确写入 frozen protocol。

# 12. 局限与证据边界

论文主动限制了结论范围：

- 所有码长都是相对于具体 $P,\mathcal Q,\mathcal B$ 的 resource-bounded 结果，不是无限容量模型或 unrestricted coding optimum；
- 不是所有 fixed relation 都有害，只测试了 chromatic / fifths 两种 pitch geometry；
- 不是所有 BPE 都有害，只能说明当前 J/K intervention 下 BPE 失败；
- F/G 使用 high-confidence tonal subset，不能与 full-split D 做绝对机制排名；
- ComMU 只建立方向性复现，训练曲线尚未收敛；
- context probe 只说明 state 对 ordered context 敏感，不能推断 hidden state 一定显式编码“和弦/乐句/终止式/声部”等可解释变量；
- exact recovery 只针对声明的 Note/REST fields 与必要 side information，不等价于恢复原始 MIDI 全部字节、controller、instrumentation、micro-timing 或 acoustic performance；
- 未研究主观听感、生成质量以及更大规模模型的 scaling behavior。

# 13. 对 Transformer 几何编码与工程建模的启发

这套框架并不依赖音乐领域，可以直接转译到几何/物理序列建模中。

## 13.1 Reconstruction loss 不是几何 code 的充分评价

一个 latent/code 能低误差重建几何，只证明 recoverability 较好，不保证它是下游气动代理或生成模型最容易使用的表示。

更接近本文的评价方式是固定模型和训练预算，比较：

$$
\text{held-out NLL / physical fact}
$$

或等价 predictive codelength，判断不同 geometry interface 的**计算可用性**。

## 13.2 全局几何参数与局部截面参数可做 factorization，但不意味着独立

例如全局 span/chord/pose 与逐截面归一化参数 $(span_i,chord_i,twist_i,CST_i)$ 可以像 pitch class/register 一样拆成可单独寻址的因素。

关键点是：**factorization 不等于 independence assumption**。Transformer 仍然可以在 attention/state 中学习全局尺度与局部截面之间的强依赖。

## 13.3 观测决定的几何关系更适合 Fact–Token 一侧

候选包括：

- 归一化 spanwise coordinate；
- 截面真实相对位置；
- 可逆的姿态/尺度 canonicalization；
- 由几何直接计算的 signed displacement；
- 明确、无上下文歧义的拓扑邻接。

这些类似论文的 musical-time coordinate：可以显式提供，然后用 matched predictive test 决定是否真的有效，而不是仅凭“物理上合理”就固定。

## 13.4 工况相关的气动耦合更接近 Token–State 一侧

如果两个截面/表面 patch 的“重要关系”会随着 Mach、AoA、Reynolds number、全局外形或其他工况变化，那么提前写死一个固定 relational geometry，可能类似本文的 circle-of-fifths commitment。

更稳妥的策略是保留足够几何身份与物理坐标，让 self-attention / cross-attention 根据当前工况形成 contextual relation；只有经过 matched experiment 证明稳定有效的固定关系，才值得进入 tokenizer 或 attention bias。

## 13.5 不要为了缩短序列盲目 merge geometry tokens

如果把多个截面、控制点、patch 合并成更大的 token，不应只比较 token count 或 FLOPs，还应同时比较：

- 每个原始几何自由度/物理事实的 predictive NLL；
- 气动代理误差；
- 条件生成可控性；
- 同等训练 exposure 下的收敛；
- carrier vocabulary 扩大带来的 output entropy/参数变化。

J/K 的结果说明：**把多个容易的局部预测步骤合并成一个更大的复合 token，可能让每一步更难预测，最终总码长更高。**

# 14. 最值得记住的设计准则

> [!important] Tokenization 的两条边界
> **Fact–Token Boundary：** 把“由观测决定，并且经实验证明能降低预测码长”的结构显式化。  
> **Token–State Boundary：** 把“需要上下文才能确定”的组织关系留给模型状态计算。

> [!tip] 可直接复用的实验流程
> 1. 先定义真正需要保持的 declared facts；  
> 2. 把每个候选改动归类为 coordinate / carrier / relation；  
> 3. 设计 matched intervention，只改变一个操作；  
> 4. 固定 predictor family、context、training exposure、checkpoint rule；  
> 5. side information 必须纳入账本；  
> 6. 以每个原始 fact 的 held-out predictive codelength 比较；  
> 7. 不从“可逆”“token 更少”“包含领域先验”直接推出表示更好。

## 跨文献关联导航（2026-10-15 整理）

> [!note] 文献关系标注
> “方法承接”或“研究延展”表示现有笔记可支持的脉络；“比较”“互补”“类比”仅用于研究组织，不声称作者直接引用或实验证实联合方案。

**总览**：[[Literature/文献关系总览|文献关系总览]]

**所属主题**：[[Literature/Topics/Transformer与物理场算子|Transformer与物理场算子]] · [[Literature/Topics/几何表示与性能条件生成|几何表示与性能条件生成]]

**重点关联**：

- **表示原则**：[[Literature/transformer/SkexGen - Autoregressive Generation of CAD Construction Sequences with Disentangled Codebooks|SkexGen - Autoregressive Generation of CAD Construction Sequences with Disentangled Codebooks]] — token 接口是否预先固定关系会影响上下文建模。
- **表示原则**：[[Literature/SeqXY2SeqZ-Structure Learning for 3D Shapes by Sequentially Predicting 1D Occupancy Segments from 2D Coordinates|SeqXY2SeqZ-Structure Learning for 3D Shapes by Sequentially Predicting 1D Occupancy Segments from 2D Coordinates]] — 稀疏占用段编码与 token 关系复杂度的取舍。
- **理论互补**：[[Literature/transformer/Lost in the Middle at Birth - An Exact Theory of Transformer Position Bias|Lost in the Middle at Birth - An Exact Theory of Transformer Position Bias]] — token 关系预编码与因果路由位置偏差是不同的表征问题。
