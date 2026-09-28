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
