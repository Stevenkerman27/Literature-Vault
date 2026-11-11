---
title: "Configuration Selection via Self-Supervised, Performance-Weighted Generative Neural Networks"
authors: [Jordan T. Smart, Juan J. Alonso]
year: 2023
venue: "AIAA SciTech Forum 2023"
doi: "10.2514/6.2023-0669"
url: "https://doi.org/10.2514/6.2023-0669"
source: "本轮对话上传 PDF；未归档到 Literature-Vault"
status: read
tags: [literature, configuration-design, mixed-integer-optimization, generative-design, VAE, performance-weighting, aerospace]
---

> [!abstract] 摘要译述
> 构型选择是航空航天概念设计中的核心任务，但长期主要依赖人工构思和经验迭代。本文将构型选择形式化为 NP-complete 的混合整数优化问题，并提出 Experimental Design Engineering Network（EDEN，实验设计工程网络）作为一类生成式神经网络。作者先将离散设计候选嵌入连续潜空间，再利用多学科分析与优化（MDAO）评价构型，构造性能加权概率分布；网络通过重构该分布学习生成候选。与只学习训练数据中“常见且可行”的组合不同，EDEN 的训练目标显式偏向高性能设计，因此希望以有限的分析预算生成更有希望的候选，而不是保证全局最优解。

# 一句话定位

**EDEN 将黑箱混合整数构型设计中的“搜索哪些离散组合”转化为“学习一个向高性能候选倾斜的生成概率分布”。** 论文的关键不是提出一种全新的 VAE 主干，而是将 MDAO 性能评价构造成训练分布，并把这个分布嵌入 VAE 的重构目标。

## 1. 问题定义与研究动机

构型设计变量同时包含：

- 离散选择：例如发动机数量、推进系统类型、电池化学体系、材料选择；
- 连续变量：例如尺寸、位置、形状参数；
- 黑箱性能与约束：设计必须经过物理分析工具才能得到可行性和性能，通常没有可用的解析梯度。

作者将问题写为混合整数非线性优化：

$$
\begin{aligned}
\min_{\Theta}\quad & f(\Theta)\\
\text{s.t.}\quad & g(\Theta)\leq s,\\
& \Theta_i\in\mathbb Z,\quad i\in I.
\end{aligned}
$$

其中 $f(\Theta)$ 是非线性性能评价，$g(\Theta)$ 是约束函数，$\Theta_i$ 中由索引集合 $I$ 指定的变量必须取整数。

论文强调：即使把问题限制为混合整数线性规划（MILP），一般情形仍是 NP-complete。对于航空航天设计，连续松弛还可能失去物理意义：例如不能合理评价一架具有 2.37 台发动机的飞机。许多分析程序还是 black-box MINLP：设计者可以输入一个完整构型并得到分析结果，却无法假设设计空间连续、凸或可微。

传统路线的不足：

1. MILP/MINLP 求解器依赖可处理的函数结构或连续松弛；
2. Bayesian/convex optimization 仍需要有意义的中间点评价；
3. 遗传算法、模拟退火等随机搜索可保留离散构型，但时间与计算预算缺少可预先量化的性能保证；
4. 普通生成模型主要学习“数据中哪些组合出现过”，并不自然偏向性能最好的组合。

EDEN 的定位是一个生成式启发式方法：降低生成高质量候选的成本，但不恢复严格的全局最优性保证。

## 2. 设计候选的树形数值表示

系统的详细物理网格可能因分析方法而异。例如，同一机翼的 CFD 网格与涡格法（VLM）面元表示并不相同。EDEN 因而不直接编码完整物理网格，而是优先编码构型中的离散设计选择。

作者把离散选择组织为一棵有根树：

- 每个内部节点代表一次设计选择；
- 每条分支代表该选择的一个可选项；
- 叶节点代表一个完整离散构型；
- 选择顺序由树的层级确定。

对于每层都有相同分支数 $r$、总共有 $h$ 层的平衡树，设计总数为：

$$
N=r^h.
$$

论文图 2 的示例树为 rank $r=3$、height $h=7$，因此 $N=3^7=2187$。材料选择案例则有 9 种材料、7 个独立结构位置：

$$
N=9^7=4{,}782{,}969\approx 4.8\times 10^6.
$$

对于不平衡树，作者用最大分支数 $r^*$ 与最大树高 $h^*$ 构造上界，并把每个叶节点编码为长度 $h^*$ 的整数向量。每个整数表示对应层的分支选择；不足最大深度的构型在尾部补零，以便所有构型共享相同张量形状。

论文用最多抛 5 次硬币、目标为得到 3 次正面作类比：

- 抛出一次正面、一次反面后停止：$[1,2,0,0,0]$；
- 连续三次正面后停止：$[1,1,1,0,0]$。

**表示能力边界：** 这种表示能够统一不同深度的离散选择树，但它本身不是一个通用 CAD 网格表示，也没有直接规定多部件装配几何、部件之间的连续接口或整机物理一致性。

## 3. MDAO 性能评价如何变成生成概率分布

EDEN 首先对训练候选进行性能分析。对于最小化问题，作者定义能量函数：

$$
E(\Theta)=
\begin{cases}
f(\Theta), & \Theta\text{ 可行},\\
+\infty, & \Theta\text{ 不可行}.
\end{cases}
$$

再通过 softargmax / Boltzmann 形式将性能转换为归一化概率：

$$
p(\Theta\mid f(\Theta))
=
\frac{\exp[-E(\Theta)]}{Z},
\qquad
Z=\sum_{\Theta'}\exp[-E(\Theta')].
$$

这一步是 EDEN 的核心。若只用普通最大似然估计（MLE），网络学习的是训练集中不同特征组合出现的频率；引入性能权重后，训练目标变成拟合“哪些组合更可能具有较好的性能”。

因此，训练样本的数量不再直接决定生成器偏好的强弱：一个在原始样本中很少出现、但性能很好的设计，也可以在加权分布中获得较高概率质量。

### 3.1 性能权重的直观例子

论文使用一个旋翼结构材料选择问题说明这种变化：

- 旋翼半径为 4 ft，设计推力为 500 lbf；
- 7 个结构位置分别从 9 种材料中选择；
- 设计总数约为 $4.8\times10^6$；
- 结构件按 limit-load sizing criterion 估算尺寸和重量。

原始样本的重量分布存在明显长尾：多数候选集中在约 10 kg 附近，少数接近最优的候选低于 5 kg，较差样本可达到约 350 kg。原始样本中只有约 8% 的候选低于 5 kg，而性能加权分布把几乎全部概率质量集中到低于 5 kg 的区域。

**这里的性能权重是对已有 MDAO 评价结果的再分配，不是未经评价就能知道哪个候选最优。** 因而生成质量仍依赖训练数据覆盖和分析模型可信度。

## 4. VAE 架构与 PLBO 训练目标

论文以变分自编码器（VAE）作为 EDEN 的示例架构。

- Encoder：将候选设计向量映射为高斯潜变量分布，$\mathrm{En}(\Theta)\rightarrow\mathcal N(\mu,\Sigma)$；
- Latent sampling：从该高斯分布中采样潜变量 $z$；
- Decoder：根据 $z$ 输出候选构型上的概率分布，$\mathrm{De}(z)\rightarrow\hat p(\Theta\mid z)$。

候选向量采用 one-hot 表示，以便对离散构型的解码概率进行比较。

### 4.1 KL 正则项

作者将编码器输出与标准高斯分布进行比较：

$$
\mathcal L_R
=
D_{\mathrm{KL}}
\left(
\mathcal N(\mu,\Sigma)\,\|\,\mathcal N(0,I)
\right).
$$

该项避免编码器只记忆训练候选，而是要求潜空间保持可采样的连续结构。论文将其解释为一种“noising”/正则机制：相似候选在潜空间中形成可探索的局部区域。

### 4.2 性能加权解码项

对训练候选 $\Theta_i$，解码器应为该候选分配概率质量，且奖励强度与其性能权重相关。按照论文式（7）的 one-hot 目标项，可将其写为：

$$
\mathcal L_D
=
-p(\Theta_i\mid f(\Theta_i))
\,\hat p(\Theta_i\mid z).
$$

这里保留了论文式（7）展示的负概率质量形式；虽然正文将其描述为 cross entropy，但该式展示的目标项没有标准交叉熵中的 $\log$，不应擅自替换成普通 NLL。

总目标称为 Performance Lower Bound Objective（PLBO）：

$$
\mathcal L_{\mathrm{PLBO}}
=
\mathcal L_R+\mathcal L_D.
$$

训练使用自动微分计算网络参数梯度，并通过 ADAM 随机梯度下降更新。该损失的关键不是简单地重构原始样本，而是让解码器把更多概率质量分配给高性能样本。

## 5. 数据预算与实验结果

作者将重点放在中等规模设计空间：约 $10^8$ 到 $10^{13}$ 个可能构型。论文讨论的分析预算为每天可评价约 0.1%–1% 的设计空间，并将演示训练集限制在不超过 $10^5$ 个样本。作者认为，对于小于 $10^7$ 的设计空间，训练专用 EDEN 模型通常不划算。

旋翼材料选择案例展示了：

1. 原始候选集的性能分布较分散；
2. 根据 MDAO 结果构造的性能加权分布将概率集中在低重量区域；
3. VAE 学习该分布后，可以低成本重复生成高性能候选；
4. PLBO 随训练 epoch 总体下降，说明网络能拟合性能加权目标。

论文图 6 的训练历史显示约 2,800 个 epoch 的 PLBO 变化，但作者没有报告跨不同旋翼尺寸、载荷或新性能目标的系统泛化实验。作者指出，可以将生成结果与新设计空间采样得到的性能加权分布比较 cross entropy，以判断是否需要重新训练。

## 6. 局限与解释边界

1. **不是全局最优性证明。** EDEN 是候选生成启发式，不给出可证明的全局最优保证。
2. **训练仍依赖 MDAO 数据。** 性能加权只能重排已评价候选的信息，无法凭空识别完全未覆盖的高性能区域。
3. **潜空间结构由表示与训练数据共同决定。** 树形编码统一了离散选择长度，但不自动表达任意 CAD 装配、连续几何接口或跨部件物理耦合。
4. **示例范围有限。** 主要演示是旋翼材料选择问题，不是完整飞机跨布局生成。
5. **分布迁移需重新验证。** 改变旋翼尺寸、推力或性能目标后，作者建议用新采样分布检验生成器，而不是默认旧模型仍然适用。

## 7. 与其他生成式构型方法的关系

| 方法 | 主要学习对象 | 生成偏好 / 条件机制 | 与 EDEN 的区别 |
|---|---|---|---|
| [[Literature/deepSPACE Generative AI for Configuration Design Space Exploration|deepSPACE]] | 配置、需求与 design style 的联合关系 | 固定需求下沿低维 style 维度探索多种构型 | deepSPACE 把设计空间组织为 style slice；EDEN 直接拟合性能加权构型分布 |
| [[Literature/AI-Based Generative Algorithms applied to the design of Blended Wing Body Aircraft|BWB 条件 DDPM]] | 24 维 BWB 几何参数向量 | 以 $C_L,C_D,C_M$ 为条件的扩散去噪 | BWB 生成连续参数化的固定 BWB 几何，不是通用离散构型树 |
| [[Literature/Aerodynamic Shape Design Space Exploration with Deep Latent Diffusion Model|DiffGeo]] | 固定模板上的几何 latent manifold | 采样时通过可微能量梯度引导几何约束 | DiffGeo 的重点是几何有效性与能量条件；EDEN 的重点是性能加权的离散构型分布 |

## 8. 对 GRASS-plane 的研究启发（项目推论）

- **可借鉴：** 对 ADD_WING/STOP、推进系统类型等离散选择，树形/序列式表示提供了“离散决策先统一编码，再对候选分布建模”的思路。
- **可借鉴：** 真实气动或多学科分析得到的性能标签可以转化为候选采样权重，而不必只用均匀经验数据训练生成器。
- **不能直接照搬：** EDEN 的树形整数向量没有展示翼面空间布局、连续截面几何、机身连接与多部件装配约束。
- **待验证假设：** 在 GRASS 的整机候选池上，比较普通 MLE、按约束可行性加权、按多目标性能加权三种训练分布，观察高性能候选命中率、可行率和多样性是否同时改善。

## 9. 核心图片定位

- **论文中图 1（PDF 第 3 页）**：整体系统架构，显示 MDAO 性能评价与图形化候选表示如何形成生成网络训练数据。
- **论文中图 2（PDF 第 4 页）**：平衡设计空间树，说明离散构型如何编码为固定长度整数向量。
- **论文中图 3–5（PDF 第 6–7 页）**：原始样本、近最优样本及累计分布的对照，展示性能加权如何将概率集中到低重量区域。
- **论文中图 6（PDF 第 9 页）**：PLBO 训练历史。

## 10. 参考文献脉络

- Nair et al., *Solving Mixed Integer Programs Using Neural Networks*（2021）：神经网络作为混合整数规划求解启发式的相关基础。
- Kingma & Welling, *Auto-Encoding Variational Bayes*（2013）：VAE 的潜变量概率建模与重参数化基础。
- Browne et al., *A Survey of Monte Carlo Tree Search Methods*（2012）：树搜索如何学习不同分支的选择分布；EDEN 认为其只需学习叶节点构型分布时可以更直接。

## 关联

- [[Literature/deepSPACE Generative AI for Configuration Design Space Exploration|deepSPACE]]：同一研究脉络中从性能加权概率分布走向 design-style 低维搜索。
- [[Literature/AI-Based Generative Algorithms applied to the design of Blended Wing Body Aircraft|BWB 条件 DDPM]]：固定 BWB 参数空间中的气动目标条件生成。
- [[Literature/Aerodynamic Shape Design Space Exploration with Deep Latent Diffusion Model|DiffGeo]]：几何潜空间扩散与能量引导。
- [[Literature/Topics/气动优化与设计空间探索|气动优化与设计空间探索]]。
- [[Literature/Topics/跨布局飞行器生成与新型网络架构|跨布局飞行器生成与新型网络架构]]。
