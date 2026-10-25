# Literature-Vault 神经网络架构分类与关系审查（2026-10-25）

## 审查范围与结论

本次按仓库 main 的实际文件树和主要索引检查，重点是目录是否一致、论文机制之间的关系是否可追溯、Obsidian 路径链接是否可导航，**不等同于所有论文的科学结论已被独立复现**。

审查时（加入九篇新论文后）的事实：
- Literature 下有 **56 个顶层/子目录 Markdown 笔记**：根目录 30、transformer 子目录 17、architectures 子目录 9；另有 **8 个 Topics 专题页面**。其中至少 3 个 Markdown 是综合/导航页而非单篇原始研究（文献关系总览、PaDGAN 与 PcDGAN 比较、Transformer 架构创新综述），因此不应把 56 称作“56 篇独立论文”。
- 8 个 Topics 页面含约 **201 次 Obsidian wiki 链接引用**；进行“目标完整路径或文件 basename 是否在仓库中”的静态检查后，拟新增本审查页之外，没有检测到对应的缺失目标。该检查不能代替 Obsidian 对大小写、alias、锚点和插件的运行时解析，也没有穷举每一篇正文的所有引用。
- 旧「文献关系总览」中的“39 篇独立论文笔记”数字已过时，应该删去静态总数，改用动态检索或明确日期。
- 旧「几何表示与性能条件生成」存在 Range-GAN 表格行落在 Markdown 表头之前的排版错误，应修复。
- 旧 WeightFormer 笔记出现重复的“## 9”章节编号，应修复。

## 分类设计评估

### 已经合理的部分

- 使用 Obsidian Markdown + YAML front matter + wiki-links，利于迭代知识图谱与检索。
- 「Topics」已经按生成设计、气动代理、Transformer 物理算子、优化、强化学习、整机设计等**研究问题**组织，这是正确的横向维度。
- Transolver → SATO 的明确架构承接、PaDGAN → PcDGAN 的方法继承、Transolver ↔ SMART 的替代方案等，很多跨论文关系已有清晰描述，并不是从零开始。
- 已有文章经常区分论文实验证据和项目级猜想，这是应当保留的风格。

### 当前不够合理的部分

1. **路径混合了不同分类轴**：transformer 目录既含 Token Attention 的实现、Swin/DiT/Uformer 这样的视觉/生成范式、SkexGen 的 CAD token 生成、又含 SATO/Transolver 领域代理。文件夹名不能推断机制。
2. **序列建模主线缺失**：过去没有 S4 → Mamba → Mamba-2 SSD → Mamba-3 的完整独立笔记；Attention → linear attention → recurrent matrix memory → gated delta 也缺少图谱。
3. **跨论文“关系标签”不统一**：数学等价、模型衍生、相似机制、性能对照有时仅被一行“关联”归并。尤其“Transformer 等价于 MLP”“SSM 等价于 Attention”容易被过度泛化。
4. **阅读路径偏应用**：物理场专题是非常好的领域入口，但不能承担所有通用语言/视觉/序列架构的教学导航。
5. **索引维护方式不稳定**：人工追加“当前笔记”若不说明是精选列表，容易被用户理解成全量目录；静态论文数量也会过期。

## 本次实施的最小破坏修复

- 建立 **Literature/architectures/**：仅用于通用序列架构基础论文，新增 S4、Mamba-1、Mamba-2/SSD、Mamba-3、RetNet、GLA、Gated DeltaNet、Hyena、MambaVision 九篇完整中文笔记。
- 增加 [[Literature/Topics/神经网络架构与高效序列建模|神经网络架构与高效序列建模]] Map of Content（内容地图），按**信息传播机制**而非论文所属领域组织，并在关系图中明确实线/虚线含义。
- 修改 [[Literature/Topics/Transformer与物理场算子|Transformer 与物理场算子]] 的上层入口，使物理算子 MOC 与通用序列 MOC 正交。
- 让 [[Literature/文献关系总览|文献关系总览]] 指向新 MOC，删除陈旧的“39 篇”静态统计。
- 修正 Range-GAN 表格行放置与 WeightFormer 重复章节编号；仅做局部文档质量调整，不更改其原有科学结论。
- 更新文献整理约定：允许根目录/transformer 历史路径和 architectures 新路径并存，**保留旧 Obsidian 路径链接**，不进行大规模移动。

## 关系边的五种证据标签（后续统一沿用）

| 标签 | 断言要求 | 示例 |
|---|---|---|
| **方法承接** | 后作明确复用并修改前作机制 | S4 → Mamba；Mamba → Mamba-2 → Mamba-3；Transolver → SATO |
| **数学等价（有条件）** | 说明成立的精确算子类别、参数化和假设 | Mamba-2 的 SSM 与特定 structured masked attention 的 SSD 对偶 |
| **机制合流** | 结合来自不同工作的构件，不把所有上游叫作同系列版本 | GLA 的 gating + Delta Rule → Gated DeltaNet |
| **可比较替代** | 回答相同工程问题、计算与数据条件可能不同 | MiTA ↔ WeightFormer；MambaVision ↔ SATO（跨视觉/PDE 的类比） |
| **项目研究假说** | 具有架构启发，但原论文未给出目标 CFD/飞行器验证 | Mamba 的时序状态记忆用于不规则翼面点云；MiTA 的路由专家用于激波细节检索 |

## 针对气动几何生成的一条重要分界

通用序列模型主要处理 **有自然顺序** 的时间/语言；高精度 CFD 网格可能是 **无自然顺序** 的不规则空间点集。不能把 Selective Scan、Hyena 长卷积直接作用于随意排列的面元并称为 mesh-independent neural operator。跨域候选必须验证：

- 点重新排列是否影响物理预测；
- 面元面积/体积权重和积分测度是否一致；
- 机翼布局/几何/工况分布外泛化；
- 局部压力尖峰、翼根干扰、激波、尾迹的高频误差；
- 积分气动力/力矩与几何梯度；
- 真实端到端训练与推理速度，不仅 FLOPs。

## 后续推荐而非已完成工作

1. 核实目录外的所有单篇正文链接，补充 CI 脚本对 wiki-links 的持续检查，并排除明确允许的待建占位页。
2. 逐渐修正文献 source 字段：没有本地 PDF 的新笔记记录 arXiv/出版页，避免虚构 D 盘文件；旧笔记仍按实际原始 PDF 路径管理。
3. 若目录规模继续增大，再考虑将旧 transformer/ 内的论文移动到更精确的存储目录；移动时必须批量更新所有 Wiki links、图片相对路径和反向引用，并保留迁移映射，**本次不做破坏性移动**。
4. 若补充 RWKV、原始 DeltaNet、FlashAttention、VMamba、Native Sparse Attention，应先标明与已收录论文的直接继承/理论区别，而不是只累加孤立摘要。
