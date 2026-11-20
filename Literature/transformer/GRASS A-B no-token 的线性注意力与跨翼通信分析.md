---
title: "GRASS A/B no-token：线性注意力形式与跨翼通信瓶颈"
authors: ["GRASS 项目研究笔记"]
year: 2026
venue: "项目源码分析；非发表论文"
doi: ""
type: research-note
status: source-derived-and-synthetic-checked
updated: 2026-11-16
source: "GRASS-plane@8fd2f543ccf35df16f897a823ef95c00adc2b472；参考论文全文 HTML，未提供本地 PDF"
tags: [neural-operator, linear-attention, cross-wing-interaction, rank, GRASS]
---

> [!abstract] 本文摘要（项目分析，不是论文摘要翻译）
> A/B no-token 共享动态低秩聚合—回传的基本思想，但不是通过调整参数就能互换的同一个层。A 在全机面元上建立一个全局低秩核；B 的局部核按翼分块，跨翼通信经过每翼一个隐藏向量，并将同一个更新广播到该翼所有面元。下面给出精确因子分解、单层跨翼 Jacobian 秩界、源码对应和隔离合成验证。结论不包含训练优劣或物理精度认证。

关联：[[Literature/Topics/Transformer与物理场算子|物理场算子专题]] · [[Literature/transformer/Transolver Is a Linear Transformer Revisiting Physics-Attention Through the Lens of Linear Attention|LinearNO 原论文笔记]] · [[Literature/transformer/Does Transolver really need a Transformer|NoTokenAttn 消融]]。
关系等级：本文与 LinearNO 是“分析方法借鉴 + 有条件代数对应”；改进建议为“项目假设”，不是论文原结论。

## 1. 问题、范围与原始来源

这里的 A/B 是整机气动代理，不是飞机几何生成器。代码基线固定为 `8fd2f543ccf35df16f897a823ef95c00adc2b472`。

源码：
- [wing_mesh_surrogate.py](https://github.com/Stevenkerman27/GRASS-plane/blob/8fd2f543ccf35df16f897a823ef95c00adc2b472/wing_mesh_surrogate.py)：PhysicsAttention、MeshBlock。
- [wing_mesh_no_token.py](https://github.com/Stevenkerman27/GRASS-plane/blob/8fd2f543ccf35df16f897a823ef95c00adc2b472/wing_mesh_no_token.py)：只替换 slice-token 自注意力。
- [wing_mesh_multisheet.py](https://github.com/Stevenkerman27/GRASS-plane/blob/8fd2f543ccf35df16f897a823ef95c00adc2b472/wing_mesh_multisheet.py)：A 全机打包、B 逐翼局部层与 GlobalSheetInteraction。
- [wing_mesh_linearno.py](https://github.com/Stevenkerman27/GRASS-plane/blob/8fd2f543ccf35df16f897a823ef95c00adc2b472/wing_mesh_linearno.py)：已有 LA 面积加权 LinearNO。
- 参考论文：[全文 v3](https://arxiv.org/html/2511.06294v3)，方法 Eq.1–9 与附录 F；论文完整总结沿用上面的现有笔记，不重复归档。

数学分析采用 eval 模式、有效面元面积严格正、正常非下溢 slice mass。padding 从数学域剔除；数值测试另外覆盖 padding。浮点实现的 tiny clamp 若激活，会改变精确归一化恒等式。设翼数为 W、总面元数 N、隐藏维数 d、每头 slice 数 m。

## 2. A no-token 的精确低秩核

令 X 为进入一个注意力分支的 LayerNorm 隐藏特征，S(X)∈R^{N×m} 是单头 slice 权重，每行 softmax 后和为 1。记面积矩阵 A_a=diag(a)，D=diag(S^T a)。

聚合与回传：

$$
T=D^{-1}S^T A_a X,\qquad Y=S\,\mathrm{ChannelMap}(T).
$$

源码中的 value 投影、token_map、output 投影都是通道方向仿射变换，不混合不同 slice。把它们合并，含多头及偏置的精确表达为：

$$
Y=\sum_h K_h(X)X M_h+\mathbf1 b^T,\qquad K_h=S_hD_h^{-1}S_h^T A_a.
$$

这里 M_h 是各通道矩阵的乘积。token_map 可以吸收到每头 value 投影，偏置同步合并；这保持前向函数，却不保证优化轨迹、正则化或训练速度相同。

### 冻结当前路由后的空间核性质

$$
\operatorname{rank}(K_h)\le m,\quad K_h\mathbf1=\mathbf1,\quad a^TK_h=a^T,
$$

$$
A_aK_h=(A_aS_h)D_h^{-1}(A_aS_h)^T\succeq0.
$$

因此 a_i K_ij=a_j K_ji；A_a^{1/2}K_hA_a^{-1/2} 为半正定对称矩阵。这个共享读写的核具有面积加权平均性质，但通常不是幂等投影。

这些是每头空间混合核的性质，不是完整多头网络、气动响应或物理互易性定理。S 依赖输入，完整微分包含 (dK)X；不能用 rank(K)≤m 宣称整个网络 Jacobian 也只有秩 m。残差、LayerNorm（层归一化）和前馈网络进一步使整网非线性。

### A 为什么仍然跨翼通信

按翼将 S 分块为 S_1,…,S_W。第 v 翼到第 u 翼的混合核为：

$$
K_{uv}=S_uD_G^{-1}S_v^TA_v,\qquad D_G=\sum_w\operatorname{diag}(S_w^Ta_w).
$$

通常 K_uv 非零，且不同接收面元有不同的行。全局每个 slice 的平均，等于各翼同编号局部 slice 平均按其 slice 面积质量加权后的结果。删除 slice-token 自注意力，并未删除这种跨翼聚合。

## 3. B 的精确形式：局部算子 + 每翼摘要交换

每片翼先运行共享参数的完整局部 MeshBlock（包括注意力、残差与逐点前馈）：

$$
\widetilde H_w=F(H_w).
$$

各翼局部注意力核都是 S_w D_w^{-1}S_w^T A_w；在全机面元空间上，这一阶段是块对角通信。

令 P∈R^{W×N} 为每翼面积均值算子：P_wj=a_j/Σ_{k∈w}a_k（j属于w），其他位置为0。令 R∈R^{N×W} 为翼归属指示矩阵。对非空翼，PR=I。

源码先形成 U=P Htilde，每翼只有一个 d 维向量。随后：

$$
L(U)=L_0+\operatorname{MHA}(\operatorname{LN}_lL_0,\operatorname{LN}_sU,\operatorname{LN}_sU),
$$

$$
C(U)=U+\operatorname{MHA}(\operatorname{LN}_sU,\operatorname{LN}_lL(U),\operatorname{LN}_lL(U)),
$$

$$
\boxed{H_B^+=F_{local}(H)+R\,C(PF_{local}(H)).}
$$

MHA 是 Multi-Head Attention（多头注意力）。L_0 是默认8个可学习 latent。B no-token 仍保留上述两次跨翼 MHA；no-token 只作用于翼内 slice-token 自注意力。

## 4. 可以严格说明的本质差异

### 4.1 输入压缩与输出广播是两个独立瓶颈

B 跨翼模块接收的总信息只有 Wd 个标量。8个latent是在每翼压成一个向量之后才使用的，增加latent数量不能恢复已经被 P 消除的信息。

在局部处理后的接口上，若 PΔ=0，则 C(P(Htilde+Δ))=C(PHtilde)。这是精确恒等式。不能把它偷换为“原始几何均值一样就无法区分”：局部网络可以先把高阶几何信息编码进各隐藏通道。

输出端，R 将同一向量广播给同翼所有面元。因此，固定本层输入的目标翼隐藏特征，其他翼的变化对目标翼本层输出的增量在空间上完全相同。A 的 K_uv 没有这一约束。

这不表示 B 最终只能预测均匀载荷：下一层的逐点非线性，以及最终输出头，可以将共同上下文与不同局部状态结合，形成空间差异。

### 4.2 单层跨翼 Jacobian 秩界

固定面积及分组，对 u≠v，记 p_v 为第v翼归一化面积向量：

$$
J^B_{u\leftarrow v}=(\mathbf1_{N_u}\otimes I_d)\frac{\partial C_u}{\partial U_v}(p_v^T\otimes I_d)J_{F_v},
$$

$$
\boxed{\operatorname{rank}(J^B_{u\leftarrow v})\le d.}
$$

这包括 B 一个完整局部层后的全局交换，且是单层边界处的结论。它不是“秩为1”，不是多层网络的深度无关上界，也不是对全部几何变量端到端 Jacobian 的无条件声明。真实几何变化还会改变面积、输入编码和各层状态。

A 允许多种跨翼空间模式；在下面同宽单层夹具中可见跨翼 Jacobian 秩超过 d。不能由此推出 A 在所有任务、参数预算和深度下严格优于 B。

### 4.3 翼面分组本身是否参与模型

保持所有面元特征、面积、相对位置字段及积分权不变，只重新分组：A 仍看到同一 packed 输入；B 的局部块与 P 改变，输出一般改变。

这与置换等变不同。同步重排翼列表或翼内面元时，两种模型都可保持对应的等变性；B 不是因为列表有序就自动依赖翼编号。也不能把重新计算每翼 relative_position 后的输入变化误称为仅改变分组。

### 4.4 单翼时也不是自动相等

W=1 时 B 仍加 C(U)。即使将两次 MHA 输出全部置零，也仍有 C(U)=U，因此多加了面积均值。要做严格 local-only 消融，必须旁路整个 GlobalSheetInteraction，不能只把注意力矩阵或 projection 清零。

## 5. 为什么不能直接照搬论文附录 F 把 B 化成 A

原文附录 F 使用固定可学习锚点作为读取/写回的 query 或 key，可以将这些锚点与固定投影矩阵相乘并吸收进参数。

GRASS B 的第一次读取确实使用固定 L_0；但第二次注意力的 key/value 是第一步更新后的 L(U)，而且经过 LayerNorm。它们依赖当前输入，不能再次作为固定参数吸收。无论如何重写，这个模块仍只看到 U=PHtilde，并只通过 R 广播输出。因此同属低秩汇总机制，不等于与 A 有相同的参数化和通信限制。

## 6. 四点例子与隔离数值验证

用两个翼各两个点，面积全为1，固定路由：

$$
S=\begin{bmatrix}.9&.1\\.1&.9\\.9&.1\\.1&.9\end{bmatrix},\quad v=(0,0,1,-1)^T.
$$

A 的 K=SS^T/2 给第一个翼的输出 (0.32,-0.32)。第二个翼的均值是0，但不同位置的正负扰动仍通过不同 slice 传给第一个翼。

B 的每翼平均交换无法区分该向量和全零向量。此例限定在固定路由的标量混合、B的局部处理后接口；不是把非线性局部网络省掉后对整网下结论。

实际执行了独立 Python 核心公式转写及数值检验，不是直接导入完整仓库。环境：Python3.13.5、PyTorch2.10.0+cpu、float64、seed271828；命令 `python verify_operators.py`，退出码0，19/19项检查通过。无训练数据、checkpoint、真实气动物理计算。

| 检查 | 结果 |
|---|---:|
| A源码对应公式 vs 显式 K 的前向最大误差 | 2.22e-16 |
| 上述两种表达的输入梯度误差 | 1.67e-16 |
| token_map 合并进 value 的误差 | 5.55e-17 |
| B同翼广播一致性误差 | 2.22e-16 |
| B源翼零均值隐藏扰动对目标翼的变化 | 2.22e-16 |
| 同宽d=8、m=3夹具：A跨翼完整单层Jacobian秩 | 24 |
| 相同夹具：B跨翼完整单层Jacobian秩 | 6，满足≤8 |
| 只改变分组：A最大差异 | 0 |
| 只改变分组：B最大差异 | 0.5474 |

数值秩使用绝对阈值1e-9。幅值随随机初始化与夹具而变；数值检查支持推导，但不替代理论证明，也不是项目回归或精度验收。

## 7. 对 GRASS 的研究建议

不能只归结为“B更弱”。A全机共享m个slice，B每翼m个，W翼总计Wm个局部summary；B给每片翼独立局部容量，跨翼模块对每翼均值而非全机面积进行信息汇总。A的局部slice可能受大翼面积主导，B则有每翼归一化的不同偏置。这是可检验机制，不是已证明的精度优劣。

优先做两个互相正交的消融：

1. B每翼summary数：1→r，但暂时保留同翼广播。测试输入压缩瓶颈。
2. 保持summary构造不变，将同翼广播改为面元自身特征产生query、逐面元读取全局summary。测试输出空间分辨率瓶颈。

两者组合再与A比较，保持数据、split、输入特征、损失、初始化策略、训练预算一致，并同时做同宽与参数量匹配对照。应关注局部载荷误差、整机积分和跨翼几何扰动的导数，而不只看总体训练loss。

已有LA应保留为第三个机制基线：它对全机packed面元使用独立query/key，面积加权key在面元维归一，query在latent维归一。它不是“只有共享参数，没有跨翼交互”。A-no-token和LA的差别是共享读写/归一化约束，不只是都删了token自注意力。

本轮未修改GRASS或Dpanel代码，未训练。笔记记录的是结构分析和小型代数证据，不是对模型优劣的经验结论。

## 8. 文档与源码的差异

当前 docs/aircraft_surrogate.md 中“单翼时B全局latent无事可做”及LA“跨片耦合只能靠共享权重”的文字，不能当作上述源码的精确等价描述；前者仍加summary残差，后者在全机面元上做全局K聚合。本笔记以固定SHA源码为准，并显式保留这项文档不一致记录。
