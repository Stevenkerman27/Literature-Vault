# 文献笔记模块

## 目的

`Literature/` 用于存放已阅读论文的 Obsidian 笔记。每篇笔记聚焦论文的研究问题、方法、实验结果、局限和可复用启发，并通过 wiki 链接关联前作、数据集和研究主题。

## 约定

- 路径：`Literature/<论文标题>.md`
- 文件开头使用 YAML front matter，至少包含 `title`、`authors`、`year`、`venue`、`doi`、`tags`、`status`、`source`。
- 若已有真实本地原文 PDF，source 保存对应绝对路径；若原文未归档，在 source 明确写“未提供本地 PDF”，同时用 url / arxiv / doi 指向可核验原文。**禁止虚构本地路径**。作者、年份、会议/期刊和 DOI 均保存在 front matter，不在正文重复。
- 正文使用中文总结；开头以 Obsidian `abstract` callout 提供原文摘要的翻译，并至少包含：问题与动机、方法、关键贡献、实验结果、局限与启发、关联。文献研究部分简化为3篇左右的核心相关文章并总结作者的脉络。需要完整总结论文方法论和实验设置，做到让人看总结的笔记就能明白具体实现和实验的setup。
- 推荐以 `[[...]]` 链接前作、数据集和主题；链接目标尚未创建时可保留为待补全的空链接。
- 英文缩写必须有全称解释。CFD这种太基础的不需要

When a figure materially explains the method, export it with `tools/extract-pdf-figure.ps1` to `Literature/assets/<paper-slug>/` and embed it with an Obsidian image link. See `docs/pdf-figure-extraction.md`.


## 研究入口（不是全量论文目录）

- [[Literature/文献关系总览|文献关系总览]]：从科研问题进入各个专题。
- [[Literature/Topics/神经网络架构与高效序列建模|神经网络架构与高效序列建模]]：S4、Mamba、SSD、RetNet、GLA、GDN、Hyena、MiTA、WeightFormer 等的机制关系。
- [[Literature/Topics/Transformer与物理场算子|Transformer与物理场算子]]：具体 CFD/PDE 代理与几何表示接口。
- [[Literature/Topics/几何表示与性能条件生成|几何表示与性能条件生成]]：CAD、翼型、布局条件生成与多样性。

新增论文应将链接放到最相关的专题；不要把“手工追加当前笔记”当作全量索引。

## 跨论文关系标注

至少区分 **直接方法承接**、**有假设的数学等价**、**机制合流**、**可比较替代**、**项目未验证假说**，不得仅因两者都使用 Transformer/注意力就声称直接继承。详见 [[docs/literature-architecture-taxonomy-audit|架构分类审查]]。
