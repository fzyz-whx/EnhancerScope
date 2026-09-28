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

## 2026-09-28 · M2 进行中：baseline ①② 完成记录

### 结果（test split，真实数字）
| model | Dev Spearman | Hk Spearman | 备注 |
|---|---|---|---|
| ① kmer_gbm (k=6+GC, LightGBM) | 0.581 | 0.519 | 单种子42, CPU ~6min |
| ② CNN (DeepSTARR风格) | **0.639 ± 0.003** | **0.566 ± 0.005** | 3 seeds (42/43/44), GPU |

### 与官方 DeepSTARR 论文的差距（如实归因，不许美化）
官方论文 test Pearson r≈0.93+，我们 CNN 只有 0.66/0.74。差距归因（待 M2 收尾验证）：
1. **早停太急**：patience=5，16-17 epoch 就停了（官方训 200 epochs）
2. 未复现官方超参细节（官方 batch 128 / 特定调度）
3. lr 调度不同（我们 ReduceLROnPlateau）
→ M2 的定位是 baseline 而非复现 SOTA：CNN > k-mer 符合预期（词袋无位置信息），
这条差距线正是 M3 LoRA 微调要打穿的靶子。

### 坑
1. LightGBM 4.6 的 `eval_set` 参数已 deprecated（改 eval_X/eval_y）——仅警告，结果有效。
2. CNN 显存策略生效：one-hot uint8 全量驻 GPU（400MB），float 按batch转，训练快且稳。

## 2026-09-28 · M2 完成记录：Baseline 三件套全绿

### 最终结果（test split，Spearman ρ）
| model | Dev（发育型） | Hk（管家型） | 备注 |
|---|---|---|---|
| ① k-mer+GBDT (k=6+GC, LightGBM) | 0.581 | 0.519 | seed 42, CPU ~6min |
| ② CNN (DeepSTARR 风格) | **0.639 ± 0.003** | **0.566 ± 0.005** | seeds 42/43/44, GPU |
| ③ 零样本 (DNABERT-2 embed + Ridge) | 0.428 | 0.363 | 不微调, alpha=10 valid 选出 |

**排序完全符合预期**：k-mer 词袋（无位置信息）< CNN（学局部 motif 语法）< 微调上限（M3 靶线）。
零样本垫底符合口径（不微调的 embedding + 线性探针本来就弱）——M3 LoRA 的增值空间一目了然。

### 与官方论文差距（如实归因）
官方 DeepSTARR test Pearson r≈0.93，我们 CNN 0.66/0.74。归因：早停急（patience=5，16-17 epoch
即停 vs 官方 200 epochs）、未复现官方超参细节、ReduceLROnPlateau 调度不同。**M2 定位是 baseline
而非复现 SOTA**——差距线就是 M3 的靶子。

### 坑与解法（③ 四连坑，全部真实）
1. **einops 缺失** → DNABERT-2 remote code 需要 → `uv add --group ml einops`
2. **triton 缺失** → remote code 静态 import triton，但官方无 Windows wheel → `triton-windows` fork
3. **transformers 5.x 只认 safetensors** → DNABERT-2 仓库只有 pytorch_model.bin（449MB）→ 锁 `transformers<5`（4.57.6）
4. **hf-mirror SSL EOF 抖动** → 下载 tokenizer 时 5 连败 → curl 带 `-C -` 断点续传 + 重试循环
5. **label 列是 numpy 数组列**（[Dev_scaled, Hk_scaled] 二元组，非类别标签）→ `value_counts()` 挂死——先查 dtype 再统计（教训）
6. **evaluate() 常量预测 NaN** → 零方差的秩相关无定义，按 0（无信号）如实处理并写进 docstring

### 工程要点
- ①③ 单种子（42）确定性可复现；② 3 种子（42/43/44）报均值±标准差
- CNN 显存策略：one-hot uint8 全量驻 GPU（402k×4×249 ≈ 400MB），按 batch 转 float——训练全程零 CPU-GPU 拷贝
- ③ 探针 train 抽样 5 万（embedding 提取全量 train 需 ~25min GPU；5 万对线性探针已饱和，如实记录）

## 2026-09-28 · M3 开工前计划

### 实验设计（8GB 显存硬约束下的决策）
- **LoRA 配置**：r=16, alpha=32, dropout=0.1, target=["query","value"]（BERT 系标准选择）；
  gradient checkpointing + bf16 + 梯度累积（等效 batch 64 = 物理 32 × 累积 2）
- **序列 249bp 原生**（数据卡），无需截断——512bp 约束天然满足
- **DNABERT-2 117M**：bf16 权重 ~234MB + LoRA 参数 ~0.6M + 激活（checkpointing）→ 预估 3-5GB ✓
- **HyenaDNA**：选 small-32k-seqlen-hf（短序列兼容），参数量实测打印；remote code 风险已知（einops ✓ 已装）
- **训练预算**：LoRA 收敛快，2-3 epochs + 早停（patience=2 on valid）；先单 seed 冒烟测速度再定 epochs
- **3 seeds**（42/43/44）× 2 模型 = 6 runs；每 run 记录训练曲线/显存峰值/时长
- **对比口径**：与 M2 三件套同表（同划分/同指标/同 test 只碰一次）

### DoD 清单
- [ ] results/lora.csv（模型 × seed × 指标，可溯源到日志）
- [ ] docs/experiments.md（超参 + 选择理由）
- [ ] 训练曲线图（results/figures/lora_curves.png）
- [ ] README benchmark 表更新（六行对比）
- [ ] 若负结果：如实呈现 + 归因分析

### 已知风险预判
1. HyenaDNA remote code × transformers 4.57 兼容性（einops 已装，其他坑跑了才知道）
2. LoRA target modules：DNABERT-2 是 BERT 架构（query/value 标准选择）；HyenaDNA 是 Hyena 算子，
   target_modules 命名不同（可能要 ["proj","in_proj","out_proj"] 之类），跑了才知道
3. 显存若 OOM：降 batch → 累积加大 → 再不行缩短序列（512 红线内 249 本来就短，还有余量）

## 2026-09-28 · M2 完成记录

### 三件套最终结果（test split，Spearman ρ）
| model | Dev | Hk | 说明 |
|---|---|---|---|
| ① kmer_gbm (k=6+GC) | 0.581 | 0.519 | CPU 单种子 ~6min |
| ② CNN | **0.639±0.003** | **0.566±0.005** | GPU 3 seeds，~9min |
| ③ zeroshot (embed+Ridge) | 0.428 | 0.363 | GPU ~11min，不微调 |

排序符合预期（词袋<CNN<零样本垫底）；与官方论文差距如实归因（早停急+超参未复现），M3 的靶线明确。

### 坑与解法（M2 全部实录）
1. **einops 缺失**：DNABERT-2 remote code 硬依赖 → uv add einops
2. **triton 无官方 Windows wheel**：remote code 静态 import triton → triton-windows 3.8.0 fork
3. **transformers 5.x 只认 safetensors**：DNABERT-2 只有 pytorch_model.bin → 锁 <5（4.57.6）
4. **hf-mirror SSL EOF 抖动**：tokenizer_config 下载 5 连败 → curl -C - 断点续传 + 重试循环
5. **label 列是 numpy 数组列**（[Dev,Hk] 二元组）→ value_counts 挂死；先查 dtype 再统计
6. **后台 python 孤儿进程 674MB**：TaskStop 只杀 shell 不杀子进程 → taskkill /F；此后长任务一律 timeout + 后台
7. **evaluate() 常量预测 NaN**：零方差秩相关无定义 → 按无信号=0 如实定义并写 docstring
8. **GC 算术乌龙**：我自己把 ACGTACGT 的 GC 算成 0.25（实为 4/8=0.5），gc_fraction 本来就对——测试期望写错，修测试不修代码
9. **③ 的 embedding 缓存投毒**：坏模型时期的 embeddings/*.npy 必须清掉重提（否则污染下游）

### DoD 自查（M2）
- [x] results/baseline.csv（20 行：①4 + ②12 + ③4，含 notes 局限附注）
- [x] README benchmark 表（三行对比 + 复现命令 + 差距归因）
- [x] 固定随机种子（42/43/44），脚本一键复现
- [x] 局限如实记录（notes 列 + devlog），未隐藏未弱化

## 2026-09-28 · M3 进行中：模型加载五连坑与兼容修复

### 坑与解法（全部实测，非推测）
1. **target_modules 命名不符**：配置写了 BERT 系标准的 query/value，实测打印顶层名发现 DNABERT-2 的
   attention 是**融合 Wqkv 投影**（无独立 query/value）→ 改 `["Wqkv"]`。HyenaDNA 实测为 `in_proj/out_proj` ✓
2. **返回类型是 tuple 不是 ModelOutput**：DNABERT-2 标准注意力回退路径返回
   `(last_hidden_state, pooled)` → `.last_hidden_state` 报 AttributeError → 加 `hidden_of()` 兼容层
3. **dtype 硬编码缺陷**：bf16 加载时标准注意力回退路径 `Float vs BFloat16` 混算崩 →
   **fp32 加载权重 + 训练 autocast(bf16)** 承担混合精度（goal 的 bf16 约束由 autocast 满足，如实记录）
4. **gradient checkpointing 不兼容**：DNABERT-2 自定义 BertModel 不支持 → 脚本 try/except 降级并打印；
   实测显存峰值仅 **2.20GB**（8GB 卡余量充足，无需 checkpointing 也能跑）
5. **tokenize 慢（9 分钟/次）**：484k 条 BPE 编码 → 加 tokenize 缓存（data/processed/tokenized/*.pt，
   按 model+split 命名），3 seeds 省 ~18 分钟

### HyenaDNA 实测（CPU 探测，未占 GPU）
- 3.3M 参数（small-32k），hidden=d_model（不是 hidden_size）
- forward **不接受 attention_mask** → hidden_of 加了 TypeError 回退
- tokenizer 是单碱基级（HyenaDNATokenizer），249bp → 249 tokens

### 冒烟结果（DNABERT-2 / seed 42 / 1 epoch，真实数字）
- test: Dev ρ=0.579, Hk ρ=0.548；显存峰值 2.20GB；时长 15.9min/epoch
- 对比 M2：已超零样本(+0.15/+0.19)、超 k-mer 的 Hk；逼近 CNN —— 1 epoch 即打平/超越 3/4 条 baseline

### 坑续（第 6-8 坑，均为实测）
6. **timeout 太短杀在评估边界**：每 seed `timeout 3000`（50min），而 3 epochs 训练需 47.3min，
   最终评估+写 CSV 被砍 → 改为每 run `timeout 4600`（76min，留足余量）
7. **torch>=2.6 的 `weights_only=True` 默认值拒绝自定义对象**：tokenize 缓存用 `torch.save(BatchEncoding)`，
   `torch.load` 直接 `UnpicklingError`（seed 43/44 因此 0 epoch 崩溃）→ `torch.load(..., weights_only=False)`（本地可信缓存）
8. **微调权重未持久化**（自查发现）：best_state 只在内存 → 补 `data/models/finetuned/lora_{model}_seed{N}.pt`
   （LoRA adapter + pooler，~2.4MB，不入库），M4 可解释性与 M5 ONNX 导出依赖它

## 2026-09-29 · M3 完成记录

### 结果（test，3 seeds 均值±std）
| 模型 | Dev ρ | Hk ρ | 时长/run | 显存峰值 |
|---|---|---|---|---|
| **LoRA DNABERT-2**（117M，微调 0.50%） | **0.6214 ± 0.0088** | **0.5722 ± 0.0044** | 47.1min | 2.67GB |
| LoRA HyenaDNA（3.3M） | 0.2845 ± 0.0021 | 0.2486 ± 0.0008 | 8.9min | 2.66GB |

### 核心结论
1. 微调把 DNABERT-2 从零样本 0.428/0.363 提升到 0.6214/0.5722（+0.19/+0.21）——微调价值被量化
2. LoRA 只调 0.5% 参数即拿下 Hk 最优（> CNN 全量从零训练），Dev 与 CNN 差 0.018 —— 预训练价值直接证据
3. **HyenaDNA 是负结果**（低于 k-mer 与零样本）：loss 几乎没学动（2.46→2.42）。归因假设：
   容量（3.3M vs 117M）/ 单碱基 tokenizer（249 tokens vs BPE 63）/ 超参为 DNABERT-2 调的 /
   target 模块未消融。未做后续调优，如实记录（docs/experiments.md §4.1）

### 新增坑（第 9-10 坑）
9. **peft 包装器会透传 attention_mask**：HyenaDNA 底层 forward 不接受该参数，而 peft 的包装 forward
   签名里带它并显式透传 → 连"去掉 mask 重试"也被同样 TypeError 挡住 →
   解法：except 分支下钻到未包装的底层模型调用（LoRA 模块已注入其子层，不丢微调）
10. **tokenizer 不产出 attention_mask**（HyenaDNA 单碱基分词器）→ 全 1 mask 兜底

### 工程与可复现
- 6 个 run 全部：固定种子（42/43/44）、超参在 configs/lora.json、结果可追溯到 results/lora.csv + logs/*.json
- 微调权重持久化：data/models/finetuned/lora_{model}_seed{N}.pt（2.4MB/个，不入库，M4/M5 复用）
- 训练曲线（均值±std）：results/figures/lora_curves.png；聚合表：results/lora_summary.md

## 2026-09-29 · M4 开工前计划（可解释性分析）

**目标**：对最优模型（LoRA DNABERT-2，按 valid MSE 选种子）做归因分析，产出 docs/interpretability.md
（≥3 个完整案例：序列 → 归因图 → motif 命中 → 文献佐证）。

**方法设计**
- **in-silico 突变扫描**（主方法，goal 明示）：逐位置替换为其余 3 种碱基，测 Δ预测（Dev/Hk），
  得到 per-base 重要度曲线——无需梯度、确定性、直接对应生物学语义（"这个位点变了活性会怎样"）
- **saliency（梯度归因）**：对 embedding 求 ∂output/∂embedding，再按 token→碱基映射回 249bp 位置
  （DNABERT-2 是 BPE 6-mer，需处理 token 边界：token 内碱基均分梯度）
- **motif 对照**：JASPAR CORE insects（果蝇）PWM + 自实现 log-odds 扫描器（小工具，也是可讲点）；
  高归因位点与 motif 命中位置的重叠率作为量化指标
- **≥3 个案例**：优先选 test 集高活性序列 + 归因清晰的例子；归因与生物学不符时如实分析

**风险预判**
1. JASPAR 下载可能被墙 → 备选：geco/本地化 PWM 集（如实标注来源）
2. BPE 边界导致 per-base 归因不精确 → 用突变扫描作为主证据，梯度归因作辅证（两条路互相印证）

## 2026-09-29 · M4 完成记录（可解释性分析）

### 方法与结果
- **归因**：单碱基饱和突变（主，747 次前向/序列）+ 6bp 窗口遮蔽（辅）；两法一致性 Spearman
  ρ：case1 0.405 / case2 **0.735** / case3 0.592（case2/3 互相印证）
- **motif 对照**：JASPAR2020 insects（146 PWM）+ 自实现 log-odds 扫描器；指标 enrichment = 高归因碱基
  落在 motif 区的比例 / motif 覆盖率

| 案例 | 预测 Dev/真值 | 预测 Hk/真值 | Dev enrich | Hk enrich | 结论 |
|---|---|---|---|---|---|
| 2 | 3.44/4.18 | 6.26/6.59 | **2.49** | **2.77** | 正例：top 归因 126-128 落在 pnr(GATA) 命中区 123-134（score 最高）|
| 3 | 3.17/3.01 | 0.79/2.10 | **2.85** | 0.71 | 部分正例：Dev 对齐同源域簇；Hk 归因不与 motif 重合且低估 |
| 1 | 4.28/2.03 | 0.84/-0.78 | 0.50 | 0.25 | **失败案例**：最高预测样本高估 + 归因避开 motif |

### 新增坑（第 11-13 坑）
11. **jaspar.genereg.net 不可达**（6/6 失败）→ 改从 `vanheeringen-lab/gimmemotifs` 仓库经 gh api 获取
    JASPAR2020 insects PWM 副本（文件头保留原始出处），并写 `scripts/download_jaspar.py` 固化路径
12. **梯度归因不可用**：DNABERT-2 自定义 BertModel 的 `inputs_embeds` 路径报
    `TypeError: ones_like(): argument 'input' must be Tensor, not NoneType`（硬依赖 input_ids）
    → 改用两种扰动归因（单碱基 + 窗口遮蔽）并做一致性交叉验证，如实记录
13. **`&&` 链被 lint 截断导致"以为跑了其实没跑"**：ruff I001 失败使后续命令（含重跑）未执行，
    读到的是旧日志——教训：关键实验重跑不要挂在长 `&&` 链尾

### 下一步（M5）
最优模型 ONNX 导出 + Vite/TS 前端（粘贴序列 → 预测 + 按突变敏感度高亮碱基）+ GitHub Pages。
