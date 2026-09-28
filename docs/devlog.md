# Devlog — EnhancerScope

> 开发日志：每个里程碑记录「做了什么 / 遇到什么坑 / 怎么解决 / 下一步」。
> 面试深挖弹药库——所有数字必须可溯源，所有坑必须写清楚来龙去脉。

## 2026-09-28 · M0 开工前执行计划（M0-M1）

### M0（仓库奠基）计划
1. `git init`（main 分支）+ `.gitignore` + `.gitattributes`
2. uv + `pyproject.toml`：src 布局、hatchling 构建、`[project.scripts]` 提供 demo 命令
3. 目录骨架：`src/ tests/ scripts/ notebooks/ configs/ docs/ data/ results/`
4. demo 脚本：`enhancerscope-demo` 打印包版本 / Python 版本 / 平台 / torch 可用性
   —— 对应 DoD「新环境 `uv sync` 后一条命令跑通」
5. pre-commit：ruff + black
6. GitHub Actions CI（CPU）：ruff check + black --check + pytest
7. 推 GitHub（`fzyz-whx/EnhancerScope`，public），走 **首个 PR** 合并进 main

### M1（数据勘察与数据卡）计划
- 候选数据集（按 goal 优先级）：
  ① 人类 MPRA（Kircher et al. 2019 饱和 MPRA / Tewhey 2016 / VISTA 衍生，标准：可下载、≥1 万条、连续活性值）
  ② 受阻则 fallback 到 DeepSTARR（果蝇 STARR-seq，~50 万条，社区复现广泛）
  ③ 分类兜底 GUE 基准
- EDA：活性分布 / 序列长度 / GC 含量
- **防泄漏划分：按染色体划分 train/val/test，禁止随机划分**（同源序列泄漏是增强子预测最常见的翻车点）
- 产出：`docs/datacard.md` + EDA notebook + 固定 split 文件入库

### 环境风险预判（开工前已知）
- 本机 RTX 5060 Laptop = Blackwell 架构（sm_120），PyTorch 需要 **cu128 轮子**（PyTorch ≥2.7）；M0/M1/M2 不需要 torch，M3 微调时再装，装法写进 docs。
- HuggingFace 直连不稳：统一走 `HF_ENDPOINT=https://hf-mirror.com` 镜像，写进 README FAQ。
- ruff / black 版本在 CI 与 pre-commit 两侧若不一致会导致 lint 结果漂移 → dev 依赖与 pre-commit rev 锁同一版本。
- 本机 8GB 显存：M3 必须 LoRA + bf16 + gradient checkpointing + 梯度累积，序列长度 ≤512bp。

## 2026-09-28 · M0 完成记录

**做了什么**：GitHub 公开仓库（CI 全绿）、uv+pyproject（src布局/hatchling/锁版本）、目录骨架、demo CLI（`uv sync` 后 `uv run enhancerscope-demo` 一条命令跑通，DoD 达成）、pre-commit（ruff+black 与 dev 依赖锁同版本）、GitHub Actions CI（CPU: ruff+black+pytest，16s 全绿）、**首个 PR #1 走完整流程合并进 main**。

**遇到什么坑（3个，全是这台机器的网络）**：
1. **TUN 模式反而坏事**：TUN 开着时 git/gh 全挂（SSL_ERROR_SYSCALL / EOF），关掉后走系统代理端口 7897 反而通——已按用户机器实情固化：`git config --global http.https://github.com.proxy http://127.0.0.1:7897`。
2. **Go 程序的 HTTP/2 被节点掐**：curl 正常但 gh/git 挂 → 根因是 h2 经代理节点被干扰。修复：`git config --global http.version HTTP/1.1` + `GODEBUG=http2client=0`（gh 必须 per-command 设）。
3. **Clash 节点不稳**：直连 GitHub 完全不通；经外部控制器（127.0.0.1:9097，密钥 set-your-secret）API 实测后把主组「一元机场」切到「自动选择」(URLTest 自动故障转移) → 流量稳定。

**下一步**：M1 数据勘察——确认人类 MPRA / DeepSTARR / GUE 三个候选的真实可下载性，选定后下载→EDA→按染色体防泄漏划分→数据卡。

## 2026-09-28 · M1 完成记录

### 数据源决定（按 goal 优先级）
1. **首选 ① 人类 MPRA（Kircher 2019 饱和 MPRA）：不可用**。Zenodo 经两个代理节点均不可达
   （10/10 + 5/5 失败，EOF）；HF 上仅有 `gonzalobenegas/sat_mut_mpra`（只有 test.parquet，
   无法训练）。kircherlab 的 GitHub 仓是 Shiny 前端，不含原始数据。
2. **按 goal 规则 fallback 到 ② DeepSTARR**（果蝇，GenerTeam/DeepSTARR-enhancer-activity，
   上游为官方 Zenodo 5502060）。HF 卡片自述"仅格式调整"。
   —— **此 fallback 无需批准（goal 明示），特此在 devlog 顶部记录原因。**

### 数据实测事实（全部可复现）
- 总量 484,034 = train 402,278 / valid 40,570 / test 41,186；序列 249bp；双任务连续目标
  `Dev_log2_enrichment` / `Hk_log2_enrichment`（±8 log2 长尾）。
- **染色体级防泄漏**：train 11 臂，chr2R 整臂 held out；**valid/test 坐标零重叠**
  （valid=chr2R 左半 4,329–10,573,449，test=右半 10,574,436–21,146,449，双向 0.0%，
  numpy 前缀最大值精确计算）。
- 清洗：过滤 18 条含 N 序列（全部来自 chrYHet/chr2RHet 异染色质臂，N 数成对出现）。
- `scripts/verify_split.py` 把上述全部变成断言（防泄漏验证全绿），不再是口头声明。

### 坑与解法
1. **Zenodo 完全不可达**（代理节点问题）→ 记录原因后按规则 fallback，不硬刚。
2. **label 列是 numpy 数组列**（每行 [Dev_scaled, Hk_scaled] 二元组，非类别标签）——
   `value_counts()` 直接挂死。教训：先查 dtype 再统计。
3. **后台 python 孤儿进程**（TaskStop 只杀 shell 不杀子进程）占 674MB 内存拖慢一切 →
   `taskkill /PID x /F` 后恢复。教训：长任务用 `.venv/Scripts/python.exe` 直跑并设 timeout。
4. **`uv run` 在断网时会卡在环境校验** → 网络不稳时用 venv python 直跑。

### 下一步（M2）
k-mer+GBDT / 轻量 CNN / 零样本 LM 三条 baseline，统一 Spearman/Pearson/RMSE，结果落
results/baseline.csv。

## 2026-09-28 · M2 开工前计划

**目标**：固定划分上三条 baseline——① k-mer+LightGBM（CPU）② DeepSTARR 风格轻量 CNN（GPU）③ gDNA LM 零样本（embedding+线性探针，不微调）。统一指标 Spearman/Pearson/RMSE，产出 results/baseline.csv + README benchmark 表。

**关键决策（预判）**：
- torch 装 cu128（RTX 5060=Blackwell sm_120，PyTorch≥2.7 才支持），~2.5GB 下载与写代码并行
- ③ 零样本打分：伪对数似然需 249×3 次前向 × 4 万序列 = 不可行；用 **embedding(均值池化)+Ridge 探针**（goal 明示允许），train 侧抽样 5 万条提特征（8GB 显存约束，如实记录）
- CNN 用 3 个种子报均值±std；GBDT/Ridge 确定性单种子
- 防泄漏红线：所有模型只见 train split，chr2R 的 valid/test 仅用于评估
