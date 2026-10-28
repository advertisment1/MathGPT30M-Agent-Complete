# 数学语料及许可证

- 主数据：HuggingFaceTB/finemath `finemath-4plus`，HF 数据卡标注 ODC-BY；高质量数学教学网页、含 Markdown/LaTeX。3+ 约 34B tokens，4+ 约 9.6B tokens。注意数据许可与每个源网页可能有各自权利要求。
- 可选：`finemath-3plus`，覆盖更广但质量门槛较低。
- 研究生数学：斯坦福课程推荐书单不等于可自由下载/用于模型训练的版权许可。**不要抓取受版权保护的教材全文**。可添加明确 CC-BY/CC0/公有领域或获授权的讲义与开放教材（人工审查许可证、保留 attribution）。
- 建议补充专题：实分析、复分析、线性代数、抽象代数、拓扑、微分几何、概率测度、偏微分方程；把来源、许可证、下载日期写入 manifest。
- 项目目前没有自动抓取斯坦福教材的脚本，避免把课程书单误当训练授权。
- 严格评估：按文档哈希切分；需另外检查网页近重复和 benchmark contamination。数学推理能力不等于 next-token loss，后续应另设独立问题集人工评估。

数据卡：https://huggingface.co/datasets/HuggingFaceTB/finemath
斯坦福书单：https://ulo.stanford.edu/textbooks
