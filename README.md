# EnhancerScope 🔬

> 增强子活性预测 × 基因组语言模型 —— LoRA 微调 DNABERT-2 / HyenaDNA，严格基准 + 可解释性 + 浏览器 demo + MCP 工具。

**状态：M0 仓库奠基进行中。** 项目故事线、benchmark 表、在线 demo 随里程碑推进逐步补全。

## 项目简介（占位）

在 MPRA 增强子活性数据上，系统对比四个层次的方法：k-mer+GBDT → 轻量 CNN → 基因组语言模型零样本 → LoRA 微调，配合防泄漏的染色体级划分与可解释性分析。最终把最优模型做成浏览器里人人可用的交互工具，并封装为 LLM agent 的 MCP 插件。

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
