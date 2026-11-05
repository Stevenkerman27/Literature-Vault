---
title: "Aerodynamic Shape Design Space Exploration with Deep Latent Diffusion Model"
authors: [Zhen Wei, Edouard Dufour, Colin Pelletier, Michaël Bauerheim, Pascal Fua]
year: 2026
venue: "AIAA Journal"
doi: "10.2514/1.J066320"
arxiv: "2609.00812"
url: "https://doi.org/10.2514/1.J066320"
source: "未提供本地 PDF；作者 postprint：https://borowiecki.dev/pdf/2609.00812"
status: read
tags: [literature, aerodynamic-design, latent-diffusion, generative-design, shape-parameterization, airfoil, turbomachinery]
---

> [!abstract] 摘要译述
> 本文提出 DiffGeo，一种面向数据稀缺气动设计空间探索的潜空间扩散生成框架。它将自动几何参数化的潜空间模型与扩散采样器结合，生成新颖、几何有效且可控的设计。实验包含二维翼型生成基准、代理优化数据生成，以及三维涡轮机械叶片原型。作者报告该方法在至少少一个数量级的数据条件下仍能生成高质量、多样化样本，并通过基于能量的条件引导加入复杂几何约束。

# 一句话定位

**DiffGeo 是“固定几何模板上的可学习形变表示 + 潜空间扩散采样 + 可微能量引导”，不是离散飞机组件拓扑生成器，也不是生成器与气动代理交替更新的联合训练框架。**

本文是 2024 年 AIAA AVIATION 论文 DiffAirfoil 的期刊扩展。期刊版增加了 3D 涡轮叶片原型和更完整的几何条件引导验证。

## 1. 几何表示：Latent Space Model

作者使用 Latent Space Model（LSM，潜空间模型）作为自动几何参数化器。LSM 属于 auto-decoder：训练数据中的每个形状有一个待优化 latent code，网络权重与所有 latent code 联合优化。

给定模板网格 \(\hat M=\{\hat V,E\}\)，其中顶点和连接关系固定，MLP 读取 latent \(z\) 与模板顶点坐标，预测顶点位移：

\[
\delta v=f_\Theta(z,\hat v),\qquad
M=\{\hat V+f_\Theta(z,\hat V),E\}.
\]

因此，几何表示的顶点数和连接关系由模板决定；模型学习的是模板上的连续形变，而不是新组件或新连接拓扑。

LSM 的重建目标为双向 Chamfer Distance（CD，倒角距离）加 latent 二范数正则：

\[
\mathcal L_{LSM}=\mathcal L_{CD}(V,S)+w_z\|z\|^2.
\]

实现参数：latent 维数 \(d=256\)；解码 MLP 4 层，隐藏宽度 [256, 256, 512, 512]；LSM 权重 Adam 学习率 \(5\times10^{-4}\)，每个 latent code 学习率 \(10^{-3}\)，正则权重 \(w_z=10^{-4}\)。

## 2. 无条件生成：Latent Space Diffusion Model

Latent Space Diffusion Model（LSDM）在 LSM 学到的 latent code 分布上训练去噪扩散模型。前向过程对真实 latent 加高斯噪声，反向网络预测噪声；训练采用标准 noise-matching MSE。

实现为 5 层 MLP，隐藏宽度 [256, 512, 512, 512, 256]，Leaky ReLU；扩散步数 \(T=1000\)，\(\beta\) 从 \(10^{-4}\) 线性增加到 0.02；AdamW 学习率 \(10^{-5}\)。

## 3. 条件生成：能量引导，而非条件网络重训

给定可微目标或约束，作者定义能量函数：

\[
E(V)=\|O(V)\|^2+
\sum_i\rho_i\|C_i^E(V)\|^2+
\sum_j\gamma_j\|\max(0,C_j^I(V))\|^2.
\]

在每个反向扩散步骤，将能量对当前 latent 的梯度加入 score，引导采样朝低能量区域移动：

\[
\nabla_{z^t}\log\hat p(z^t)
=
-\frac{1}{\sqrt{1-\bar\alpha_t}}g_\phi(z^t,t)
-\xi\nabla_{z^t}E(z^t).
\]

这使几何条件可以在推理阶段替换或组合，不需要为每种新条件重训基础扩散模型。对难以在有限扩散步数内满足的条件，作者提出 enhanced conditional sampling：完成一次引导去噪后，将 latent 重新加噪到指定时间步，再重复引导采样。2D 实验使用 3 次增强循环、重加噪级别 \(T^*=300\)；3D 扭转分布引导使用 \(T^*=650\)。

**重要边界：** 论文明确表示当前研究主要聚焦几何设计目标，直接将气动性能目标纳入扩散引导是未来工作。其代理优化案例是先用最大厚度条件生成训练几何，再训练气动代理并进行 latent 优化；不能将其描述为“扩散过程直接由气动性能梯度引导”。

## 4. 实验一：二维翼型生成

- 数据：UIUC 翼型数据库，分别抽取 50、100、250、500、1000 个样本。
- 比较：LSDM、共享同一 LSM latent 空间的 GAN，以及变分 auto-decoder（VAD）。
- 评价：基于气动代理隐藏层特征的 FID、生成集内部/相对训练集的 10-NN Chamfer 距离，以及面积、厚度、弯度等统计量。
- 结果：LSDM 在极小样本下比 GAN/VAD 更稳定；GAN 在 50–100 样本时出现明显分布质量和多样性退化。论文没有把这些结果推广为任意三维组件拓扑生成能力。

## 5. 实验二：条件生成数据改善代理优化

该案例的重点不是生成器直接输出最优翼型，而是检验“更有针对性的几何采样能否改善代理训练集”。

任务为 NACA-0012 二维翼型优化，目标最大化升阻比 \(L/D\)，最大厚度固定为 12% 弦长，流动为无黏跨声速，\(Ma=0.85\)、攻角 \(0^\circ\)。几何通过 DeepGeo 变形模板 CFD 网格，使用 SU2 Euler 求解器生成升力和阻力标签。

四种代理训练集策略：
1. SM#1-1：500 个窄范围 CST-LHS 几何；
2. SM#1-2：500 个更宽范围 CST-LHS 几何；
3. SM#2：从 UIUC 随机取 50 个历史翼型；
4. SM#3：以同一批 50 个翼型训练 DiffGeo，再以最大厚度 12% 的能量条件生成 500 个翼型。

气动代理为两隐层 ReLU MLP，输入为 60 个轮廓点的 120 维坐标，输出 \(C_L,C_D\)，训练 500 epochs。优化器在 LSM latent 上运行 Adam 2000 步，学习率 \(10^{-4}\)，使用代理预测与厚度/latent 正则优化。

| 代理训练集 | 最终代理预测 L/D | 最终 SU2 L/D | 优化轨迹 R² |
|---|---:|---:|---:|
| SM#1-1 | 6.4322 | 6.6236 | 0.3265 |
| SM#1-2 | 6.7252 | 6.9370 | 0.7995 |
| SM#2 | 4.8718 | 5.8405 | 0.2041 |
| SM#3（DiffGeo） | **8.2603** | **7.7188** | **0.9620** |

作者强调该案例是低数据代理辅助优化的流程验证，不是生产级跨声速优化认证。其重要启发是：生成样本的价值在于覆盖优化器实际经过的局部区域，而不只是几何多样性。

## 6. 实验三：三维轴流压气机转子叶片

- 数据：75 个叶片，由 6 个基准叶片做凸组合插值得到；本质上是一个很小、原本处于低维线性子空间的数据集。
- LSM 模板：75 个叶片的顶点平均形状，4000 个顶点。
- 无条件生成：生成 200 个叶片，与原训练集比较几何统计和凸包外推程度。
- 条件生成：用能量引导约束 hub/tip 厚度或展向扭转分布；扭转分布引导生成 300 个候选，其中 222 个有效，约 74% 成功率。
- CFD：通过 DeepGeo 变形 Rotor 37 的约 40k 单元模板网格，使用 DAFoam RANS 求解器评估叶片通道。

该实验表明 DiffGeo 能在固定叶片模板表示下进行非线性几何变化和高维几何条件引导；它没有改变叶片组件数量，也没有生成多个部件之间的装配关系。

## 7. 对 GRASS 的关系与创新边界

| 研究维度 | DiffGeo | GRASS 目标 |
|---|---|---|
| 几何表示 | 模板网格 + 连续形变 latent | 结构化整机 + 各翼变长截面几何 |
| 离散组件数量 | 固定模板 | ADD_WING/STOP 预测翼面数量 |
| 截面长度 | 由几何模板/参数化决定 | 每翼独立截面数 |
| 条件控制 | 对可微几何能量做采样引导 | 任务条件 + 整机物理约束 |
| 物理数据回流 | 条件采样生成数据，再训练气动代理 | 计划在外环中以真实整机标签更新代理，并反馈生成器 |
| 系统装配 | 单一形状/叶片 | 多翼面、机身、翼身连接与整机质量/配平 |

不能把“latent diffusion + 几何条件引导”作为 GRASS 的原创性主张。更有区分度的问题是：如何在同一个整机生成器中联合处理离散翼面数量、每翼变长截面几何、翼面布局与装配关系，并通过持续物理数据回流维持生成器和代理的适用域。

**但需要准确区分当前实现与目标方法：** 当前整机交替外环中生成器保持冻结，外环生成候选、调用 VSPAERO 标注、再微调 Transolver；真正的生成器参数反馈尚需补齐后，才能称为生成器与代理双向交替更新。当前生成器另有监督、代理物理微调和配平校准训练阶段，这与上述外环不是同一个过程。

## 8. 局限

1. LSM 使用固定模板顶点和连接关系，不能据此声称支持离散组件拓扑或任意部件数量。
2. 3D 案例是单个轴流压气机转子叶片，不是整机飞机布局生成。
3. 性能代理主要在扩散生成后参与训练集构建与 latent 优化；论文没有证明气动性能梯度直接进入扩散采样过程。
4. 几何能量引导较强时可能产生无效样本；3D 扭转引导案例有约 26% 无效候选，需要筛选。
5. 2D 气动优化使用 Euler 无黏流动和局部 NACA-0012 优化域，不能解读为高保真生产级认证。

## 9. 文献关联

- [[Literature/DiffAirfoil An Efficient Novel Airfoil Sampler Based on Latent Space Diffusion Model for Aerodynamic Shape Optimization]]：DiffGeo 的 2024 年会议前作。
- [[Literature/Topics/几何表示与性能条件生成]]：潜空间几何与能量条件生成。
- [[Literature/Topics/跨布局飞行器生成与新型网络架构]]：跨布局组件数量与结构化整机生成的对照。
- [[Literature/Physics-Regularized Generative Parameterization for Efficient Aerodynamic Shape Optimization]]：物理正则化生成参数化。
- [[Literature/GLUE Coordinating Pre-Trained Generative Models for System-Level Design]]：系统级组件协调。
