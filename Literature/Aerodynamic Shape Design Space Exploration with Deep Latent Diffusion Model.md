---
title: "Aerodynamic Shape Design Space Exploration with Deep Latent Diffusion Model"
authors: [Zhen Wei, Edouard Dufour, Colin Pelletier, Michaël Bauerheim, Pascal Fua]
year: 2026
venue: "AIAA Journal"
doi: "10.2514/1.J066320"
arxiv: "2609.00812"
url: "https://doi.org/10.2514/1.J066320"
source: "未提供本地 PDF；依据作者 postprint 摘要与公开项目说明核对：https://arxiv.org/abs/2609.00812"
status: abstract-reviewed
tags: [literature, aerodynamic-design, latent-diffusion, generative-design, shape-parameterization, airfoil, turbomachinery]
---

> [!abstract] 摘要译述
> 本文提出 DiffGeo，一种面向数据稀缺气动设计空间探索的潜空间扩散生成框架。它将自动几何参数化的潜空间模型与扩散采样器结合，生成新颖、几何有效且可控的设计。实验包含二维翼型生成基准、代理优化数据生成，以及三维涡轮机械叶片原型。作者报告该方法在至少少一个数量级的数据条件下仍能生成高质量、多样化样本，并通过基于能量的条件引导加入复杂设计约束。

# 研究问题与方法

DiffGeo 将几何表示与概率采样分开：先训练几何潜空间模型，将几何压缩到连续 latent；再训练扩散模型学习 latent 分布，并从噪声反向采样。性能/几何条件通过能量引导作用于采样过程，使生成过程可以复用几何表示而不必为每个新条件重新训练完整生成器。

公开摘要列出的验证任务为：
1. 二维翼型：在小样本下与 GAN、VAE 比较样本质量、多样性和约束满足；
2. 代理优化：条件采样产生任务相关的翼型数据，改善代理训练和优化表现；
3. 三维涡轮叶片：从少量参考设计生成几何合理且性能较高的叶片原型。

作者提供了 2D airfoil 代码与预训练权重仓库：https://github.com/kfxw/DiffGeo 。公开仓库说明了自编码器训练、latent 导出、扩散训练、无条件/条件采样的复现流程。

# 与跨布局飞机生成的关系

DiffGeo 与 [[Literature/DiffAirfoil An Efficient Novel Airfoil Sampler Based on Latent Space Diffusion Model for Aerodynamic Shape Optimization]] 属于同一研究脉络，但扩展了几何任务和条件生成。它与 GRASS 的共同点是生成几何并将采样接入代理优化；关键差异在于目前公开摘要和 2D 代码不能证明其支持整机多翼面数量、部件连接图、每翼变长截面序列或翼身装配。

因此不能将“latent diffusion + 气动条件”作为 GRASS 的原创性主张。更有区分度的问题是：如何在一个整机结构化生成器中联合处理离散部件数量、每部件变长几何、空间装配关系和闭环工程标签。

# 局限与证据边界

本笔记只核对公开摘要和项目说明，未取得并逐页阅读 AIAA Journal 全文。摘要不足以支持网络层数、损失公式、数据集规模、具体采样步数和定量结果的详细复述；以上不作推断。公开代码重点为二维翼型，不能据此声称论文已经实现任意三维飞机拓扑生成。

# 关联

- [[Literature/DiffAirfoil An Efficient Novel Airfoil Sampler Based on Latent Space Diffusion Model for Aerodynamic Shape Optimization]]：潜空间扩散的前作。
- [[Literature/Topics/几何表示与性能条件生成]]：几何表示与性能引导。
- [[Literature/Topics/跨布局飞行器生成与新型网络架构]]：跨布局与结构化组件生成的比较入口。
