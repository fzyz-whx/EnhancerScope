# 数据卡 — DeepSTARR 增强子活性数据集

> M1 交付物。所有数字均来自实测（脚本与 notebook 可复现），无任何编造。

## 来源与出处

| 项 | 内容 |
|---|---|
| 数据集 | `GenerTeam/DeepSTARR-enhancer-activity`（HuggingFace，经 hf-mirror.com 下载） |
| 上游 | DeepSTARR 官方 Zenodo record [5502060](https://zenodo.org/records/5502060)（HF 卡片自述"仅做格式调整"） |
| 论文 | de Almeida et al., *Nature Genetics* 2022 — DeepSTARR（果蝇 S2 细胞 STARR-seq） |
| 许可 | HF 数据集页未标注 license（如实记录：仅研究用途使用） |

## 规模与字段（实测）

| split | 行数 | 说明 |
|---|---|---|
| train | 402,278 | 清洗后（原始 402,296，过滤 18 条含 N 序列） |
| valid | 40,570 | 全部来自 chr2R |
| test | 41,186 | 全部来自 chr2R |

字段：`id`（`chr臂_起_止_链_peak类别`）、`sequence`（249bp，仅 ACGT）、
`Dev_log2_enrichment` / `Hk_log2_enrichment`（连续 log2 富集，双任务目标）、
`*_scaled` / `*_quantile_normalized` 变体列，以及 `label` 列——**实测为
`[Dev_scaled, Hk_scaled]` 二元组 numpy 数组列（非类别标签，value_counts 会挂死）**。
本项目的 M2/M3 以原始 `*_log2_enrichment` 为主目标，选择理由见 docs/experiments.md。

## 划分方案（防泄漏，实测断言）

- **染色体级划分**：train 覆盖 11 条臂（chr2L/2LHet/2RHet/3L/3LHet/3R/3RHet/4/X/XHet/YHet），
  **整条 chr2R 臂 held out**（valid + test 全部取自 chr2R）。
- 同源序列（>80% 相似）几乎必然落在同一染色体臂内，因此染色体级划分是防泄漏的硬保证。
- **valid/test 坐标级零重叠（实测）**：valid 取 chr2R 左半（坐标 4,329–10,573,449），
  test 取右半（10,574,436–21,146,449），双向区间重叠率 **0.0%**（numpy 前缀最大值法，精确计算）。
- `scripts/verify_split.py` 断言：249bp ✓ / 仅 ACGT ✓ / 目标有限 ✓ / **三 split 的 id 零交集** ✓ /
  **train 与 valid/test 染色体零重叠** ✓（防泄漏验证全绿）。

## 清洗规则（确定性）

- 过滤含 `N` 的序列：18 条（0.004%），**全部来自异染色质臂**（chrYHet / chr2RHet）——
  dm6 参考基因组未测序区域，每条 N 数成对出现（正反链 twin）。valid/test 本身无 N。
- 规则由 `scripts/clean_data.py` 实现：同一份 raw 输入永远得到一致输出（可复现，无需入库原始数据）。

## EDA 结论（notebooks/01_eda_deepstarr.ipynb，已执行嵌入输出）

1. **活性分布**：Dev / Hk 两任务均右偏（强增强子长尾），三 split 形状一致 → 染色体级划分未造成分布漂移。
2. **序列长度**：全部 249bp（每个 split 唯一长度数 = 1）。
3. **GC 含量**：近正态（~45%），split 间一致 → k-mer baseline 不会因 GC 偏差占便宜。
4. **Dev vs Hk**：中等相关（见 notebook 第 4 节输出）→ 双任务共享部分序列语法但各有特异 motif，
   是 M4 可解释性分析的切入点。

## 已知局限（如实）

- 许可未在 HF 页标注；上游 Zenodo 为公开学术数据，仅研究用途。
- valid/test 同取自 chr2R 但坐标零重叠（左半/右半分割，实测 0.0%）；与官方 DeepSTARR
  划分口径的差异（官方为 train chr2L/3L/3R + valid chr2R + test chrX？）需在 M2 用
  论文对照确认，差异会记录在 devlog。
- 异染色质臂序列质量差（N 富集）已被过滤，但也意味着模型对异染色质区无代表性。
