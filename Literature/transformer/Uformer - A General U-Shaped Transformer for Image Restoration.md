---
title: "Uformer: A General U-Shaped Transformer for Image Restoration"
authors:
  - Zhendong Wang
  - Xiaodong Cun
  - Jianmin Bao
  - Wengang Zhou
  - Jianzhuang Liu
  - Houqiang Li
year: 2022
venue: "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)"
doi: "10.1109/CVPR52688.2022.01716"
source: "/mnt/data/Uformer.pdf"
tags:
  - transformer
  - image-restoration
  - u-net
  - window-attention
  - encoder-decoder
---

> [!abstract] 摘要译文
> 本文提出 **Uformer**，一种面向图像恢复的有效且高效的 Transformer 架构。它以 Transformer block 构建层次化的编码器—解码器网络。Uformer 有两个核心设计：其一是 **Locally-enhanced Window Transformer（LeWin，局部增强窗口 Transformer）block**，以非重叠窗口内的自注意力替代全局自注意力，从而显著降低高分辨率特征图上的计算复杂度，同时结合局部上下文；其二是可学习的 **multi-scale restoration modulator（多尺度恢复调制器）**，以多尺度空间偏置的形式调节 Uformer 解码器多层中的特征。该调制器只引入很少的额外参数和计算量，却能增强不同恢复任务中的细节恢复能力。借助这两个设计，Uformer 能同时建模局部与较长程依赖。论文在图像去噪、运动去模糊、散焦去模糊和去雨等任务上进行了实验；在不依赖多阶段框架或复杂损失函数的情况下，Uformer 获得了优于或可比于当时最先进方法的结果。

## 一句话定位

Uformer 的本质是：**保留 U-Net 的多尺度 U 形编码器—解码器与跳连结构，把主要卷积处理单元替换为“窗口自注意力 + 卷积增强 FFN”的 LeWin Transformer block，并在解码器中加入轻量的可学习空间偏置调制器。**

## 1. 问题与动机

图像恢复需要从噪声、模糊、雨纹等退化图像中恢复干净图像。论文认为当时主流卷积网络有两个结构性问题：

1. **长程依赖建模能力有限。** 卷积天然偏向局部邻域，虽然可通过堆叠扩大感受野，但显式建模远距离像素关系较弱。
2. **直接使用标准 Transformer 又太贵。** 若把高分辨率特征图的每个空间位置都作为 token，global self-attention（全局自注意力）的计算量随 token 数平方增长，难以直接用于高分辨率恢复。
3. **恢复任务又强依赖局部上下文。** 一个退化像素的恢复通常依赖相邻像素的纹理、边缘和结构，因此仅依赖全局关系也不理想。

因此论文不是简单地“把 U-Net 改成 Transformer”，而是同时解决 **高分辨率注意力成本** 与 **局部结构建模** 两个问题。

## 2. 整体架构：U-shaped hierarchical Transformer


### 2.1 输入投影

给定退化图像

$$
\mathbf I\in\mathbb R^{3\times H\times W},
$$

先经过一个 $3\times3$ 卷积与 LeakyReLU，提取低层特征：

$$
\mathbf X_0\in\mathbb R^{C\times H\times W}.
$$

这里 $C$ 是基础特征通道数。

### 2.2 编码器

编码器包含 $K$ 个 stage。每个 stage 包含：

- 若干个 LeWin Transformer blocks；
- 一个 down-sampling（下采样）层。

下采样时，先把 token 重新整理为空间特征图，再用 **$4\times4$、stride 2 的卷积**：

- 空间尺寸减半；
- 通道数翻倍。

因此第 $l$ 个编码 stage 输出满足

$$
\mathbf X_l\in
\mathbb R^{2^l C\times \frac{H}{2^l}\times\frac{W}{2^l}}.
$$

默认 $K=4$，因此最深层空间分辨率为 $H/16\times W/16$，通道数为 $16C$。

### 2.3 Bottleneck

编码器末端还有一个由多个 LeWin blocks 组成的 bottleneck。

论文的关键解释是：**窗口大小在各层保持不变，而特征图不断下采样，因此深层一个固定大小窗口对应到原图上的区域越来越大。** 所以即使每个 attention 只在局部窗口内部计算，到了低分辨率 bottleneck，窗口也能覆盖原图上很大的有效感受野；当窗口大小等于当前特征图大小时，等价于对该层执行全局 attention。

### 2.4 解码器与 skip connection

解码器同样包含 $K$ 个 stage，每个 stage 包含：

1. **$2\times2$、stride 2 的转置卷积**进行上采样；
2. 通道数减半、空间尺寸扩大一倍；
3. 与对应编码器 stage 的特征进行 **channel-wise concatenation（通道维拼接）**；
4. 将拼接后的特征送入若干 LeWin Transformer blocks。

补充材料还比较了三种 skip connection：

| 方案 | SIDD PSNR | 结论 |
|---|---:|---|
| Concat-Skip | **39.77 dB** | 最终采用 |
| Cross-Skip | 39.75 dB | 接近但参数更多 |
| ConcatCross-Skip | 39.73 dB | 未带来收益 |

因此最终 Uformer 采用最简单的 **concatenation-based skip connection**。

### 2.5 输出与全局残差

经过全部解码 stage 后，将 token 恢复为 2D 特征图，再经 $3\times3$ 卷积预测残差图：

$$
\mathbf R\in\mathbb R^{3\times H\times W}.
$$

最终恢复图像为

$$
\mathbf I'=\mathbf I+\mathbf R.
$$

也就是说网络学习的不是完整干净图像，而是从退化输入到干净图像之间的 **residual（残差）**。

## 3. LeWin Transformer block

LeWin = **Locally-enhanced Window Transformer**。它是 Uformer 的核心基本单元，由两部分构成：

1. **W-MSA：Window-based Multi-head Self-Attention（窗口多头自注意力）**；
2. **LeFF：Locally-enhanced Feed-Forward Network（局部增强前馈网络）**。

其 pre-norm + residual 形式为：

$$
\mathbf X'_l=
\operatorname{W\!\!-\!MSA}\left(\operatorname{LN}(\mathbf X_{l-1})\right)
+\mathbf X_{l-1},
$$

$$
\mathbf X_l=
\operatorname{LeFF}\left(\operatorname{LN}(\mathbf X'_l)\right)
+\mathbf X'_l.
$$

其中 LN 是 **Layer Normalization（层归一化）**。

### 3.1 W-MSA：只在窗口内部做 attention

设输入二维特征为

$$
\mathbf X\in\mathbb R^{C\times H\times W}.
$$

将其切分为互不重叠的 $M\times M$ 窗口，共有

$$
N=\frac{HW}{M^2}
$$

个窗口。第 $i$ 个窗口展平后为

$$
\mathbf X^i\in\mathbb R^{M^2\times C}.
$$

对每个窗口独立执行 multi-head self-attention（多头自注意力）。若共有 $k$ 个 head，单头维度为 $d_k=C/k$，第 $k$ 个 head 使用各自的投影矩阵 $W_k^Q,W_k^K,W_k^V$。

attention 写为：

$$
\operatorname{Attention}(\mathbf Q,\mathbf K,\mathbf V)
=
\operatorname{SoftMax}
\left(
\frac{\mathbf Q\mathbf K^T}{\sqrt{d_k}}+\mathbf B
\right)\mathbf V,
$$

其中 $\mathbf B$ 是可学习的 **relative position bias（相对位置偏置）**；论文将其参数表写为

$$
\hat{\mathbf B}\in
\mathbb R^{(2M-1)\times(2M-1)}.
$$

直观上，相对位置偏置不是告诉模型“这个 token 是第 37 个”，而是让 attention score 额外知道两个空间位置之间的相对位移，例如“右边 2 格、下边 1 格”。固定窗口使这种相对位移查表非常紧凑。

### 3.2 为什么窗口 attention 能降低复杂度

全局 self-attention 在 $H\times W$ 个 token 上的主要复杂度为

$$
\mathcal O(H^2W^2C).
$$

窗口 attention 中，每个窗口有 $M^2$ 个 token，窗口数为 $HW/M^2$，因此复杂度变为

$$
\mathcal O\left(
\frac{HW}{M^2}\cdot M^4C
\right)
=
\mathcal O(M^2HWC).
$$

当 $M$ 固定时，它对图像像素数 $HW$ 近似呈线性增长，而不是平方增长。

### 3.3 shifted window

论文还在每个 stage 的偶数 LeWin block 中测试了 **shifted window（移位窗口）**。补充材料给出的消融为：

- 不使用 window shift：39.76 dB；
- 使用 window shift：39.77 dB。

提升只有 **0.01 dB**，但论文最终仍把 shift 作为默认设置。

这说明 Uformer 的主要收益并不依赖 shifted window，本身的 U 形多尺度层次已经提供了较强的跨区域信息整合能力。

## 4. LeFF：把空间局部性放进 FFN

标准 Transformer 的 Feed-Forward Network（FFN，前馈网络）对每个 token 独立执行两层线性变换，它不会直接利用“相邻 token 在二维空间中相邻”这一先验。

Uformer 的 LeFF 流程是：

1. 对每个 token 做线性投影，扩展通道维；
2. 将 token 按原空间位置 reshape 为 2D feature map；
3. 施加 $3\times3$ **depth-wise convolution（逐通道卷积）**；
4. 再 flatten 回 token；
5. 用另一层线性投影把通道降回输入维度；
6. 每个 linear/convolution 后使用 **GELU（Gaussian Error Linear Unit，高斯误差线性单元）** 激活。

因此 LeWin block 中形成了明确分工：

- **W-MSA**：负责窗口内内容依赖与较长距离关系；
- **LeFF 中的 depth-wise convolution**：显式注入邻域局部结构。

论文的 locality 消融尤其重要：

| locality 注入位置 | SIDD PSNR | RealBlur-R / J PSNR |
|---|---:|---:|
| 都不加 | 39.74 | 36.15 / 28.99 |
| 加到 self-attention | 39.72 | — |
| 加到 LeFF | **39.77** | **36.22 / 29.06** |
| attention 与 LeFF 都加 | — | 36.19 / 28.85 |

结论是：**把局部归纳偏置放在 FFN 比直接改 attention 更合适。**

## 5. Multi-scale Restoration Modulator

不同退化类型有不同的扰动模式，例如 blur、noise、rain。为了增强恢复细节，论文在 **decoder 的多个尺度**加入轻量 modulator。

### 5.1 形式

在每个 LeWin Transformer block 内，一个 modulator 是可学习张量：

$$
\mathbf M\in\mathbb R^{M\times M\times C},
$$

其中：

- $M$：当前 W-MSA 的窗口尺寸；
- $C$：当前 feature map 的通道数。

它被作为一个 **shared bias（共享偏置）**，加到该层所有非重叠窗口上，然后再进入 self-attention。带调制器的 W-MSA 在图中记为 MW-MSA。

### 5.2 为什么叫 multi-scale

“multi-scale”不是说单个 modulator 自己内部有多个尺度，而是因为 **decoder 的不同 stage 分辨率和通道数不同，每个 stage 都有自己的 modulator**。因此网络在多个空间尺度上分别进行可学习的特征校准。

### 5.3 它不是显式退化条件输入

从论文定义看，modulator 是训练得到的参数张量，并没有接收“噪声强度”“模糊核”或任务标签作为输入。因此这里的 modulation 是 **静态可学习特征偏置**，不是根据样本条件动态生成的 conditioning。

论文所说的“适用于不同恢复任务”，指的是这套结构能够分别用于去噪、去模糊、去雨等任务，而不是一个单模型在推理时根据退化标签动态切换任务。

### 5.4 消融结果

| 任务 | 无 modulator | 有 modulator | 提升 |
|---|---:|---:|---:|
| GoPro motion deblurring | 29.11 | 29.57 | **+0.46 dB** |
| SIDD denoising | 39.86 | 39.89 | +0.03 dB |
| SPAD deraining | 47.43 | 47.84 | **+0.41 dB** |

modulator 对不同退化的贡献差异明显：对运动模糊和雨纹更显著，对 SIDD 去噪较小。

## 6. 损失函数

训练采用 Charbonnier loss：

$$
\mathcal L(\mathbf I',\hat{\mathbf I})
=
\sqrt{\|\mathbf I'-\hat{\mathbf I}\|_2^2+\epsilon^2},
$$

其中

$$
\epsilon=10^{-3}.
$$

$\hat{\mathbf I}$ 是 ground truth。

Charbonnier loss 可以看作平滑版的 $L_1$ 型鲁棒损失：在误差接近 0 时通过 $\epsilon$ 避免绝对值在零点不可导，同时相比纯 $L_2$ 对大误差不那么敏感。论文没有使用额外的感知损失、GAN 损失或复杂多阶段损失。

## 7. 模型规模与架构变体

主文定义三种规模：

| Variant | 基础通道 $C$ | Encoder depths | Decoder |
|---|---:|---|---|
| Uformer-T | 16 | {2, 2, 2, 2} | 镜像 encoder |
| Uformer-S | 32 | {2, 2, 2, 2} | 镜像 encoder |
| Uformer-B | 32 | {1, 2, 8, 8} | 镜像 encoder |

默认窗口大小为 **$8\times8$**，编码器/解码器 stage 数 $K=4$。

> [!info] 主文与官方实现的边界
> 主文没有把 bottleneck depth、每个 stage 的 head 数和 FFN expansion ratio 全部列出。官方公开实现中，默认 `mlp_ratio=4`；Uformer-B 的完整 depth 配置为 `[1,2,8,8,2,8,8,2,1]`，中间 `2` 对应 bottleneck；head 数为 `[1,2,4,8,16,16,8,4,2]`。这些属于官方实现补充，而不是主文表格直接给出的实验参数。

## 8. 训练与实验设置

### 8.1 通用训练设置

- Optimizer：AdamW（带解耦 weight decay 的 Adam）
- $\beta=(0.9,0.999)$
- weight decay：0.02
- 初始 learning rate：$2\times10^{-4}$
- cosine decay 到：$10^{-6}$
- 数据增强：随机水平翻转；随机旋转 $90^\circ,180^\circ,270^\circ$
- LeWin window size：$8\times8$
- 默认 encoder/decoder stage 数：$K=4$
- 评价指标：
  - **PSNR（Peak Signal-to-Noise Ratio，峰值信噪比）**
  - **SSIM（Structural Similarity Index Measure，结构相似性指标）**
- 除 deraining 外，PSNR/SSIM 在 RGB 空间计算；deraining 按既有工作在 YCbCr 的 Y 通道计算。

### 8.2 各任务训练设置（补充材料）

#### Denoising

- 训练集：SIDD（Smartphone Image Denoising Dataset）
- 随机 crop：$128\times128$
- batch size：32
- epoch：250
- SIDD evaluation：$256\times256$ patches
- DND（Darmstadt Noise Dataset）evaluation：$512\times512$ patches，在线服务器评测
- DND 直接使用在 SIDD 上训练的模型，不重新训练。

#### Motion deblurring

- 只在 GoPro 训练；再直接测试 GoPro、HIDE、RealBlur-R、RealBlur-J
- 训练 crop：$256\times256$
- batch size：32
- validation：中心 crop $256\times256$
- epoch：3000
- 最终测试：full-resolution image

#### Defocus deblurring

- 数据集：DPD
- 按官方 patch segmentation 方案切成 $512\times512$ patch
- patch 重叠率：60%
- 使用 Sobel filter 计算 sharpness energy，丢弃 sharpness 最低的 30% patch
- epoch：160
- batch size：4
- 测试：full-resolution image

#### Deraining

- 数据集：SPAD
- 训练集：超过 64k 张 $256\times256$ 图像
- 测试集：1k 张 $512\times512$ 图像
- 2 GPUs
- mini-batch：16
- epoch：10

#### Supplementary demoireing

补充材料还在 TIP18 上验证 demoireing：

- 对带边框图像取中心比例 $[0.15,0.85]$ 区域；
- resize 到 $256\times256$；
- 使用 bilinear interpolation；
- 训练 250 epochs。

该实验属于补充材料扩展，不是主文主要四类恢复实验的核心结论。

## 9. 主要实验结果

### 9.1 Real noise removal

| Dataset | Uformer-B PSNR | SSIM | 与此前最好结果比较 |
|---|---:|---:|---|
| SIDD | **39.89 dB** | **0.960** | 比 NBNet 高 0.14 dB |
| DND | **40.04 dB** | 0.956 | 比 NBNet 高 0.09 dB |

与 IPT（Image Processing Transformer）相比：

| Model | GMACs | Params | SIDD PSNR | DND PSNR |
|---|---:|---:|---:|---:|
| IPT | ~380G | 115.31M | 39.10 | 39.62 |
| Uformer-B | **89.46G** | **50.88M** | **39.89** | **40.04** |

Uformer 不需要 IPT 那种大规模预训练，同时计算量和参数量显著更低。

### 9.2 Motion blur removal

Uformer-B：

- GoPro：**33.06 dB / 0.967**
- HIDE：30.90 dB / **0.953**
- RealBlur-R：**36.19 dB / 0.956**
- RealBlur-J：**29.09 dB / 0.886**

其中 HIDE 的 PSNR 略低于 MPRNet 的 30.96 dB，但 SSIM 更高；RealBlur-R/J 上则取得最优结果，说明仅在 GoPro 上训练后仍具有较强 real-scene generalization。

### 9.3 Defocus blur removal

DPD：

- Uformer-B：**26.28 dB / 0.891**
- 前一最好 KPAC：25.24 dB / 0.842

PSNR 提升 **1.04 dB**。

### 9.4 Real rain removal

SPAD：

- Uformer-B：**47.84 dB / 0.9925**
- 前一最好 SPAIR：44.10 dB / 0.9872

PSNR 提升 **3.74 dB**，是论文最显著的任务级提升之一。

## 10. 消融实验：真正支持了哪些设计判断

### 10.1 Transformer block vs. convolution ResBlock

论文把 Uformer 的 LeWin blocks 全部换成卷积 ResBlocks，构造计算规模相近的 UNet-T/S/B。

| Model | GMACs | Params | SIDD PSNR |
|---|---:|---:|---:|
| UNet-T | 15.49G | 9.50M | 39.62 |
| Uformer-T | **12.00G** | **5.23M** | **39.66** |
| UNet-S | 34.76G | 21.38M | 39.65 |
| Uformer-S | 43.86G | **20.63M** | **39.77** |
| UNet-B | 86.97G | 53.58M | 39.71 |
| Uformer-B | 89.46G | **50.88M** | **39.89** |

结论：LeWin block 的提升随模型规模增大而更明显，但并非数量级跃升；Base 级别约提升 0.18 dB。

### 10.2 U-shaped hierarchy vs. single-scale ViT

单尺度 ViT baseline：

- hidden dimension 256；
- patch size $16\times16$；
- 12 个标准 Transformer blocks；
- SIDD PSNR 仅 **38.51 dB**。

显著低于 Uformer，说明恢复任务中 **层次化多尺度结构本身非常关键**，不能把提升简单归因于“使用了 Transformer”。

### 10.3 locality 应放在哪里

LeFF 中加入 depth-wise convolution 有正收益，而把 locality 直接加入 self-attention 并没有稳定收益。论文据此选择 **attention 负责关系建模、FFN 负责局部空间增强** 的分工。

### 10.4 modulator

modulator 在 GoPro 与 SPAD 上收益明显，在 SIDD 上收益很小，说明它并不是所有退化上都同样有效，但确实能够用很少额外计算改善某些结构化退化的恢复细节。

## 11. 关键贡献

1. **U-shaped Transformer restoration architecture**：把 Transformer 真正放进高分辨率、多尺度 U-Net 式恢复框架，而不是只在最低分辨率处加少量 attention。
2. **LeWin block**：用 local-window attention 把复杂度从 $\mathcal O(H^2W^2C)$ 降到 $\mathcal O(M^2HWC)$，同时用 LeFF 中的 depth-wise convolution 补回局部空间归纳偏置。
3. **Multi-scale restoration modulator**：在 decoder 多尺度注入可学习空间 bias，成本很低，对 motion blur 和 rain 等任务带来明显收益。
4. **实验层面的论证较完整**：不仅与 SOTA 比较，还分别消融了 Transformer/卷积、层次结构/单尺度 ViT、locality 注入位置、modulator、window shift 与 skip connection。

## 12. 局限与启发

> [!warning] 论文没有单独的 limitation section
> 以下首先区分作者直接证明的结论与从其结构/实验可保守得到的观察，不把后者当作作者原话。

### 12.1 可由论文直接支持的限制性观察

- **窗口 attention 仍然是局部的。** 长程依赖主要依靠多尺度下采样后的更大有效感受野，以及可选 shifted window，而不是高分辨率层直接进行全局 attention。
- **shifted window 的边际收益很小。** 补充材料只提升 0.01 dB，说明跨窗口交互并非 Uformer 性能的主要来源。
- **modulator 的收益任务相关。** 在 SIDD 上仅 +0.03 dB，但 GoPro +0.46 dB、SPAD +0.41 dB。
- **Base 模型并不轻量。** Uformer-B 仍有 50.88M 参数、89.46 GMACs；论文所说的效率主要是相对于 global attention、IPT 和同规模恢复模型。
- **“general” 指架构可迁移到不同恢复任务，不等于一个统一多任务模型。** 各任务仍分别训练/评估。

### 12.2 方法论启发

- Transformer 与卷积并不是二选一：**把 global/content interaction 与 local inductive bias 分别放到不同子模块**，可能比在 attention 内部同时解决所有问题更有效。
- 对 dense prediction / restoration，**hierarchy + skip connection** 的价值与 attention 机制本身同等重要；单尺度 ViT baseline 明显失败。
- 一个固定窗口在多尺度网络中的“物理感受野”会随下采样扩大，因此 **局部算子 + 层次结构** 可以间接形成全局建模能力。
- decoder 上的轻量 feature modulation 可以成为恢复细节的独立设计维度，而不必改动主干 attention 公式。

## 13. 关联文献与研究脉络

### U-Net

提供 U-shaped encoder-decoder、multi-scale hierarchy 与 encoder-decoder skip connection。Uformer 保留了这一总体拓扑，仅把主要特征处理单元换为 Transformer block。

### Swin Transformer

提供了 hierarchical vision Transformer、window-based attention、shifted window 与 relative position bias 的关键思想。Uformer 采用窗口 attention 与相对位置偏置，并测试 shifted window，但目标从 high-level vision backbone 转向 low-level image restoration。

### LocalViT

其核心思想之一是在 Transformer FFN 中加入局部卷积，使 token 在保持 attention 全局/关系建模能力的同时显式利用局部空间邻域。Uformer 的 LeFF 沿着这一方向，将 depth-wise convolution 放进 FFN，并通过消融表明这一位置比直接增强 self-attention 的 locality 更有效。

### IPT

IPT 是较早将标准 Transformer 系统用于图像处理/恢复的工作，但依赖大规模合成数据预训练和 multi-task learning。Uformer 则强调从架构层面提高高分辨率恢复效率，不依赖大规模预训练，并以更低参数量与 GMACs 获得更高的 SIDD/DND 结果。

### 脉络总结

```text
U-Net: 多尺度 U 形拓扑 + skip connection
        │
        ├──────────────┐
        │              │
Swin: 窗口 attention   LocalViT: FFN 中注入局部卷积
        │              │
        └──────┬───────┘
               ↓
            Uformer
     LeWin + U-shaped hierarchy
               │
               └── decoder multi-scale modulator

IPT 则构成另一条“Transformer 用于 image restoration”的直接前驱，
Uformer 主要在效率、层次结构与无需大规模预训练方面与其形成对照。
```

## 14. 阅读时最容易混淆的几点

1. **W-MSA 不是“每个窗口一个独立参数的 attention”。** Q/K/V 投影参数在所有窗口间共享，只是 attention 的 token 交互范围被限制在各窗口内部。
2. **multi-scale modulator 不是条件向量。** 它是 decoder 各层的可学习空间 bias；没有从样本条件动态生成。
3. **LeFF 的 depth-wise convolution 不负责跨通道混合。** 跨通道混合主要由前后的 linear / $1\times1$ projection 完成；$3\times3$ depth-wise convolution 主要负责每个通道内部的局部空间混合。
4. **Uformer 的“较长程依赖”主要来自层次化。** 高分辨率层仍只看局部窗口；随着分辨率下降，同样 $8\times8$ 窗口覆盖原图更大区域。
5. **skip connection 对图像恢复是信息保真通道。** 编码器中的低层纹理可以直接传给解码器，而不要求 bottleneck 重新编码全部细节，这与图像恢复任务的目标高度一致。

## 跨文献关联导航（2026-10-15 整理）

> [!note] 文献关系标注
> “方法承接”或“研究延展”表示现有笔记可支持的脉络；“比较”“互补”“类比”仅用于研究组织，不声称作者直接引用或实验证实联合方案。

**总览**：[[Literature/文献关系总览|文献关系总览]]

**所属主题**：[[Literature/Topics/Transformer与物理场算子|Transformer与物理场算子]]

**重点关联**：

- **多尺度类比**：[[Literature/transformer/Towards a Foundation-Model Paradigm for Aerodynamic Prediction in Three-dimensional Design|Towards a Foundation-Model Paradigm for Aerodynamic Prediction in Three-dimensional Design]] — 两者可用 U 形层级思想，任务/模块不同。
- **架构正交**：[[Literature/transformer/Scalable Diffusion Models with Transformers|Scalable Diffusion Models with Transformers]] — 多尺度/局部注意力与条件调制是独立设计维度。
