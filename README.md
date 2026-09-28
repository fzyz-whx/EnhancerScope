# EnhancerScope 🔬

> 增强子活性预测 × 基因组语言模型 —— LoRA 微调 DNABERT-2 / HyenaDNA，严格基准 + 可解释性 + 浏览器 demo + MCP 工具。

**状态：M0 仓库奠基进行中。** 项目故事线、benchmark 表、在线 demo 随里程碑推进逐步补全。

## 项目简介（占位）

在 MPRA 增强子活性数据上，系统对比四个层次的方法：k-mer+GBDT → 轻量 CNN → 基因组语言模型零样本 → LoRA 微调，配合防泄漏的染色体级划分与可解释性分析。最终把最优模型做成浏览器里人人可用的交互工具，并封装为 LLM agent 的 MCP 插件。

## Benchmark（test split，Spearman ρ，越大越好）

| 模型 | Dev（发育型） | Hk（管家型） | 说明 |
|---|---|---|---|
| ① k-mer(k=6)+GC → LightGBM | 0.581 | 0.519 | 单种子 42；词袋无位置信息 |
| ② DeepSTARR 风格 CNN | **0.639 ± 0.003** | 0.566 ± 0.005 | 从零训练，3 种子 |
| ③ DNABERT-2 零样本（embed+Ridge，不微调） | 0.428 | 0.363 | 探针 train 抽样 5 万 |
| ④ **LoRA 微调 DNABERT-2** | 0.6214 ± 0.0088 | **0.5722 ± 0.0044** | 仅微调 0.50% 参数，47min/run，显存峰值 2.67GB |
| ⑤ LoRA 微调 HyenaDNA | 0.2845 ± 0.0021 | 0.2486 ± 0.0008 | **负结果**，归因见 docs/experiments.md §4.1 |

**核心结论**：LoRA 微调把 DNABERT-2 的零样本（0.428/0.363）大幅提升到 0.6214/0.5722（Dev +0.19 / Hk +0.21），
并在 Hk 上超过从零训练的全量 CNN——只微调 0.5% 参数的预训练模型与全量从零训练的 CNN 互有胜负，
这是基因组语言模型预训练价值的直接证据。HyenaDNA（3.3M）的失败归因如实记录在 experiments.md。

复现：`uv run --group ml python scripts/baseline_kmer_gbm.py && uv run --group ml python scripts/baseline_cnn.py && uv run --group ml python scripts/baseline_zeroshot.py`（固定种子，结果追加 `results/baseline.csv`）。各模型局限见 CSV `notes` 列与 docs/devlog.md。

排序符合预期：k-mer 词袋（无位置信息）< CNN（局部 motif 语法）< 零样本垫底（不微调的 embedding+线性探针）——M3 的 LoRA 微调将以此为靶线。与官方 DeepSTARR 论文（test Pearson ~0.93）的差距如实归因：早停急 + 超参未完全复现，M2 定位是 baseline 而非 SOTA 复现。

## 快速开始（占位）

```bash
uv sync
uv run enhancerscope-demo
```

## 路线图

- [x] M0 仓库奠基（本里程碑：骨架 / uv 环境 / CI / pre-commit / 首个 PR）
- [ ] M1 数据勘察与数据卡（EDA + 防泄漏划分）
- [ ] M2 Baseline 三件套（k-mer+GBDT / CNN / 零样本）
- [ ] M3 主实验：LoRA 微调 DNABERT-2 与 HyenaDNA
- [ ] M4 可解释性分析（归因 + in-silico 突变 + motif）
- [ ] M5 浏览器交互层（ONNX + 前端 + GitHub Pages）
- [ ] M6 MCP 工具层（FastMCP server + 测试）
- [ ] M7 发布打磨（README 终稿 / 博客 / Release）

## 文档

- [开发日志](docs/devlog.md) —— 每个里程碑的坑与解法
- 数据卡 / 实验记录 / 决策记录随 M1 起逐步补充
