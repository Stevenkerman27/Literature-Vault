---
title: "MambaVision: A Hybrid Mamba-Transformer Vision Backbone"
authors: [Ali Hatamizadeh, Jan Kautz]
year: 2025
venue: "CVPR 2025, pp. 25261–25270"
doi: null
arxiv: "2407.08083"
url: "https://arxiv.org/abs/2407.08083"
code: "https://github.com/NVlabs/MambaVision"
source: "未提供本地 PDF；原文 https://arxiv.org/pdf/2407.08083"
tags: [paper, mamba, vision, hybrid-transformer, hierarchical-backbone, spatial-modeling]
status: read
updated: 2026-10-25
---

> [!abstract] 摘要译述
> 一维 Mamba/SSM 具有高效长序列建模能力，但图像具有非因果、二维局部和多尺度结构。MambaVision 提出四阶段层级视觉主干：前两阶段使用卷积做高分辨率局部特征提取，后两阶段使用重设计的 MambaVision Mixer 与 Transformer Attention。作者通过混合位置和比例消融发现，将 Attention 放在较高层有利于长程空间关联；模型在 ImageNet-1K、COCO 和 ADE20K 上取得有竞争力的精度与吞吐量。

# 核心定位

**它不是“纯 Mamba 已经证明替代了视觉 Transformer”，相反，论文的主要证据之一是高层加入 Self-Attention 会改善空间依赖与精度。** 对物理场，这支持探索“局部几何编码 → 选择性状态混合 → 少量全局注意力/查询”的混合路线，但该组合尚非 CFD 实证。

## 1. 从 1D Mamba 到二维视觉架构

Mamba 原型使用沿 token 顺序的递归状态，因果短卷积用于局部预处理。在图像中，这种方向性不是天然的空间归纳偏置。MambaVision 做两项改造：

1. 将 causal Conv1D 改为非因果卷积，以便同时利用扫描顺序两侧局部信息；
2. 在使用选择性 SSM 的一条通路之外，增加不经 SSM 的对称局部卷积分支，减少序列化扫描的空间信息损失。

可把混合后的基本输出概括为：

\[
u=\operatorname{SiLU}(\operatorname{Conv}(X_x)),\quad
v=\operatorname{SiLU}(\operatorname{Conv}(X_z)),
\]

\[
z=\operatorname{OutProj}([\operatorname{SelectiveScan}(u);v]).
\]

这是简化结构表达；具体投影和 reshape 参见论文 Algorithm 1。

## 2. 四阶段 Macro Architecture

- Stem：两个 stride-2 的 3×3 CNN，将原图降采样为约 \(H/4\times W/4\)。
- Stage 1–2：卷积残差块，在高分辨率阶段高效提取局部特征。
- Stage 3–4：Token mixer block，分别使用 MambaVision Mixer 和 Self-Attention。每个阶段前半的 block 使用 MambaVision，后半使用自注意力。
- 各阶段通过 stride-2 卷积下采样，形成尺度层级。

## 3. 实验设置与结果

ImageNet-1K 224×224 分类、MS COCO 目标检测/实例分割、ADE20K 语义分割。吞吐量对照在论文指定的 A100、batch size 128 等条件测量。主要 ImageNet 结果：

| 模型 | 参数量 | FLOPs | Top-1 |
|---|---:|---:|---:|
| MambaVision-T | 31.8M | 4.4G | 82.3% |
| MambaVision-S | 50.1M | 7.5G | 83.3% |
| MambaVision-B | 97.7M | 15.0G | 84.2% |
| MambaVision-L | 227.9M | 34.9G | 85.0% |

上述为不同资源档位，不能直接以 T、S、B、L 的精度差归因于某个模块。论文通过 hybrid pattern、Token Mixer 设计等消融论证高层 Attention 的作用；检测/分割结果也显示与相近规模视觉 backbone 有竞争力。

## 4. 为什么混合结构是研究重点？

纯 1D 扫描 SSM 会受到扫描顺序和方向性的影响；局部卷积提供空间高频归纳偏置，而高层 Attention 能直接建立远处 patch 间的匹配关系。层级 downsampling 让 Attention 放在更少 token 的后层，控制二次开销。这是三个模块在**不同分辨率和信息需求**下的分工，而不是简单堆叠新算子名称。

## 5. 局限与气动启发

- 验证在规则图像像素/patch 上进行，并没有无序非结构 CFD 面元实验；面积权重与网格拓扑需要另设计。
- 高层加入 Attention 证明混合可能有效，不能据此断言所有模型都应使用 Attention。
- 对翼根/激波/尾迹等局部细节，应关注下采样是否丢失信息；对全局升阻积分，应同时考虑局部场与物理积分的一致性。
- 论文使用的是当时 MambaVision Mixer，不等于 2026 [[Literature/architectures/Mamba-3 Improved Sequence Modeling using State Space Principles|Mamba-3]] 的复数 MIMO SSM。

## 关联

- **空间适配**：[[Literature/architectures/Mamba Linear-Time Sequence Modeling with Selective State Spaces|Mamba]] → MambaVision。
- **视觉机制比较**：[[Literature/transformer/Uformer - A General U-Shaped Transformer for Image Restoration|Uformer]] 同样进行空间局部/多尺度，但主干机制和任务不同。
- **物理场类比（未验证继承）**：[[Literature/transformer/Spatially-aware transformer operator for real-time aerodynamic evaluations of arbitrary three-dimensional vehicles|SATO]] 为物理场加入局部 patch 和全局 learned slice；两者都在解决局部细节与全局通信分工，但一个是视觉骨干、一个是 CFD 代理，不能横比精度。
- **工程备选**：[[Literature/transformer/Transolver|Transolver]] 以可学习物理切片替代固定扫描顺序，在不规则面元上有不同的几何表示假设。

来源：[CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.html) · [原文](https://arxiv.org/html/2407.08083) · [代码](https://github.com/NVlabs/MambaVision)
