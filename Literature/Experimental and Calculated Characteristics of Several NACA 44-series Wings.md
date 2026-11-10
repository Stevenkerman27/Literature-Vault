---
title: "Experimental and Calculated Characteristics of Several NACA 44-series Wings with Aspect Ratios of 8, 10, and 12 and Taper Ratios of 2.5 and 3.5"
authors:
  - Robert H. Neely
  - Thomas V. Bollech
  - Gertrude C. Westrick
  - Robert R. Graham
year: 1947
venue: "NACA Technical Note 1270"
doi: null
url: "https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19930081871.pdf"
source: "未提供本地 PDF；NASA NTRS 原文链接 https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19930081871.pdf"
tags: [literature, aerodynamics, panel-method, validation, wind-tunnel, finite-wing, lifting-line, NACA-44]
status: abstract-reviewed
updated: 2026-11-10
---

> [!abstract] 摘要译述（仅依据 NASA NTRS 公共摘要，尚未核验原文全文）
> 作者针对七副 NACA 44 系列无后掠梯形翼，比较按二维翼型数据计算的有限翼气动特性与风洞试验，目标是评价既有计算方法，并研究展弦比、梢根比及根部厚度弦长比对结果的影响。计算包含基于线性剖面升力曲线的传统升力线理论，以及允许非线性剖面升力曲线的版本；摘要还提到 Jones 边缘速度因子修正。测试翼的展弦比为 8、10、12，梢根比的原文记法为 2.5 和 3.5。

## 1. 为什么值得关注

这是**有真实有限翼风洞数据的三维验证基准**，不是只能检验数学自洽的椭圆翼解析例。它对 [[Literature/Topics/势流与面元求解器验证|势流与面元求解器验证]] 的价值在于：固定机翼几何及雷诺数/马赫数/迎角后，能够检查求解器给出的升力斜率与俯仰力矩是否接近外部独立参考。

不能把升力线计算曲线当作风洞真值；该报告本来就在比较二者。

## 2. 已由摘要核实的设置

| 项目 | 已核实的信息 |
| --- | --- |
| 研究对象 | 七副 NACA 44-series 无后掠梯形翼 |
| 展弦比 | 8、10、12 |
| 梢根比 | 2.5、3.5（注意原报告采用 root/tip 表示；与常用 taper ratio = tip/root 要区分） |
| 计算路线 | 线性翼型曲线的升力线理论；考虑二维剖面非线性的版本 |
| 额外修正 | Jones edge-velocity factor |
| 对照来源 | 风洞试验及二维剖面数据 |

**未核实，不得直接抄用作验收真值**：七副翼的逐一厚度/弯度/弦长参数、风洞雷诺数与马赫数、压力测点数据、升力/力矩曲线数字、风洞壁面修正及失速角。这些需要拿到全文和原始数据后逐项摘录。

## 3. 对 Dpanel 可执行的验证方案（本项目建议，非论文结论）

1. 按原文逐一重建翼型截面、根/尖弦、展长、几何面积与参考力矩点。先检查几何，不启动气动求解。
2. 固定 `M \ll 1`、小迎角、附着流工况。用独立升力线数值解对照 `dC_L/d\alpha`，再与报告风洞数据对照。
3. 比较 `C_L(\alpha)`、`C_{m,c/4}(\alpha)` 与 `\Gamma(y)`（最后一项仅在原文提供可转换的分布数据时使用）；进行弦向、展向、前后缘/翼尖局部加密。
4. 分开保存近场压力积分、远场 Trefftz 诱导阻力及其差值；不可把实测总阻力当作无粘求解器的诱导阻力目标。
5. 报告 source/geometry SHA、工况、网格、闭包（pressure / potential_jump）、原始曲线、残差、误差与失败原因。严格区分物理失配和风洞粘性/压缩性因素。

关联工程： [Dpanel#49](https://github.com/Stevenkerman27/Dpanel/issues/49)（三角面元与外部物理基准）、[Dpanel#38](https://github.com/Stevenkerman27/Dpanel/issues/38)（升力斜率口径）。

## 4. 已知局限与下一步全文核验

- **来源限制**：本笔记只核对 NTRS 摘要与元数据，未读完整报告。这里没有引用任何原文曲线的具体数值或绘制论文图。
- NACA 44-series 是有弯度、有限厚度的实体翼，不会严格满足零厚薄翼的二维 `2\pi` 极限。
- 风洞阻力包含粘性分量；无粘势流只能在合理假设下预测升力、压力以及尾迹诱导阻力，不能拟合总阻力与失速。
- 后续：核对实际表格和图编号，提取明确工况下的 `C_L\)-\(\alpha`、\(C_m\)-\(\alpha\) 数据并建立逐点引用；根据原图需要，再使用仓库 `tools/extract-pdf-figure.ps1` 导入准确编号的图片。

## 5. 文献关系（关系类型明确）

- **理论对照而非直接继承**：[[Prandtl Lifting Line Theory]] —— TN-1270 将升力线计算与有限翼实验并列比较。
- **可比较的另一实验基准**：NASA TM 4074（Ladson，1988），NACA 0012 二维风洞数据，重点校核二维翼型；它与 TN-1270 的三维有限翼目标不同。
- **验证证据选择**：NASA TM 100019（McCroskey，1987），系统讨论 NACA 0012 多风洞实验间的差异，提醒不能随意选一个实验曲线当成无误差真值。

## 原始来源

- [NASA NTRS, NACA-TN-1270](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19930081871.pdf)
- [NASA TM 4074](https://ntrs.nasa.gov/citations/19880019495)
- [NASA TM 100019](https://ntrs.nasa.gov/citations/19880002254)
