# 预期面试问答（Interview Q&A）

> 全部答案基于本项目真实跑出的数字与代码，证据位置已标注。**答不上来的地方如实说"未验证"** ——
> 承认边界比编造更能通过追问。建议面试前把"证据"列里的文件打开过一遍。

---

### Q1. 用一句话介绍这个项目？

在果蝇 S2 细胞的 STARR-seq 增强子活性数据上（48 万条、249bp、连续活性值），系统对比了
k-mer+GBDT / CNN / 基因组语言模型零样本 / LoRA 微调四类方法，**只微调 0.50% 参数**的
DNABERT-2 在 Hk 任务上超过从零训练的全量 CNN，并把最优模型做成浏览器端可用的工具与 MCP 插件。
核心是**防泄漏评估 + 可复现**：所有数字都能从 `results/` 里的脚本重跑出来。

### Q2. DNABERT-2 和 HyenaDNA 的架构区别？为什么一个能微调成功一个失败？

- DNABERT-2：**BERT 式 Transformer**，BPE 分词（6-mer 词表 4096），ALiBi 位置编码，
  attention 是**融合的 Wqkv 投影**（没有独立 query/value —— 这也是我配置 LoRA target_modules
  时被实测纠正的点）。117M 参数。
- HyenaDNA：**隐式长卷积（Hyena 算子）**替代 attention，单碱基分词，3.3M 参数。
- 结果：DNABERT-2 LoRA → test ρ 0.6214/0.5722；HyenaDNA LoRA → 0.2845/0.2486（**负结果**）。
- 归因（写在 experiments.md §4.1，是**假设不是结论**）：容量差 35 倍 + 单碱基分词使序列变长
  4 倍（249 tokens vs 63）+ 超参是为 DNABERT-2 调的、Hyena 算子的 in_proj/out_proj 未必是最优适配位置。
- 证据：`docs/experiments.md`、`results/lora.csv`、`results/logs/lora_*.json`。

### Q3. LoRA 的原理是什么？为什么它能以 0.5% 的参数逼近全参微调？

预训练权重 W 冻结，只学一个低秩增量 ΔW = BA（B: d×r, A: r×d，r≪d），前向是 Wx + BAx。
假设是微调所需的变化位于低维子空间。因为只存/训 B、A，参数量从 d² 降到 2dr。
本项目 r=16、alpha=32（alpha/r 的缩放是社区惯例），可训参数 589,824 / 117.7M = 0.50%。
代价是容量上限（没做 r 的消融，已列在局限里）。
证据：`configs/lora.json`、`docs/experiments.md` §1.2。

### Q4. 为什么强调"按染色体划分"而不是随机划分？

增强子序列有大量同源重复。随机划分会让近似重复的序列同时出现在训练与测试里，指标虚高。
所以我按染色体划分：train 覆盖 11 条臂，**整条 chr2R 臂 held out**，并且实测 valid(chr2R 左半)
与 test(右半) 之间**坐标零重叠（0.0%，用 numpy 前缀最大值法精确算的）**、三份 split 的 id 零交集。
`scripts/verify_split.py` 把这几条写成断言，CI 里能跑；还有 sha256 校验清单防漂移。
证据：`docs/datacard.md`、`scripts/verify_split.py`、`data/manifests/splits.sha256`。

### Q5. 你的 CNN 为什么打不过论文里的 DeepSTARR（0.93 vs 你 0.66）？

如实归因，不美化：**早停太急**（patience=5，16-17 个 epoch 就停了，官方训 200 epochs）、
超参未完全复现（官方 batch 128 等细节）、学习率调度不同（我用 ReduceLROnPlateau）。
另外本项目定位是"透明可复现的对比基准"，不是 SOTA 复现——我把差距写进 README 而不是藏起来。
证据：`docs/experiments.md` §3、`results/baseline.csv`。

### Q6. 零样本基线为什么那么弱（0.428）？你怎么定义"零样本"？

零样本 = **不微调**：用 DNABERT-2 提 embedding（最后一层隐状态均值池化），只训练一个 Ridge
线性探针（在 valid 上选 alpha，test 只碰一次）。弱是合理的：117M 模型在 249bp 上的表示没有
针对这个回归任务适配，线性可分性有限。这个数字的价值在于它是 LoRA 的对照下界。
另外伪对数似然（pseudo-likelihood）打分在 4 万条序列 × 249 位上是不可行的（~3000 万次前向），
所以选了 embedding+探针这一口径。
证据：`scripts/baseline_zeroshot.py`、`results/baseline.csv`。

### Q7. 可解释性你是怎么做的？为什么不用梯度？

用**单碱基饱和突变**（主）：每位替换为其余 3 种碱基（747 次前向/序列），取 |Δ预测|；
再用**6bp 窗口遮蔽**做辅证，两法一致性 Spearman ρ = 0.405/0.735/0.592（三个案例）。
不用梯度的原因有两条，都很具体：① DNABERT-2 自定义代码的 `inputs_embeds` 路径直接报
`TypeError: ones_like(): argument 'input' must be Tensor, not NoneType`（硬依赖 input_ids）；
② BPE 分词下 token 跨多个碱基，梯度到碱基需要额外映射假设。扰动法精确、确定、模型无关。
证据：`src/enhancerscope/explain.py`、`docs/interpretability.md` §1。

### Q8. 你的模型真的学到了生物学吗？给我证据。

两个案例给证据、一个案例给反例。**案例 2**（test 序列 `chr2R_13586410_13586658_-`）：
模型 top 归因位置 126/127/128（Dev 与 Hk 重合）**正落在 `pnr`（pannier，GATA 家族）的
JASPAR 命中区 123–134 内（score 14.6，全序列最高）**；命中集合以同源域（cut/lbe/lbl）+ GATA 为主，
与果蝇发育增强子的经典组合语法一致（Arnosti & Kulkarni 2005；pnr 见 Ramain et al. 1993）。
量化：top-10% 高归因碱基落在 motif 区的比例是随机基线的 **2.49/2.77 倍**。
**反例案例 1**：最高预测样本反而高估 2.25 log2 且归因避开 motif 区（enrichment 0.50/0.25），
我给了三个可检验假设（STARR-seq 质粒语境差异 / motif-syntax 敏感度 / 回归长尾）。
证据：`docs/interpretability.md`、`results/interpretability/case{1,2,3}.json`。

### Q9. ONNX 导出你做了什么保证？怎么证明导出没出错？

**数值一致性闸门**：导出后拿 96 条真实 test 序列，对比 PyTorch 与 onnxruntime 的输出。
LoRA DNABERT-2（fp32，481MB）：max|Δ| = 7e-6，Pearson 1.0/1.0；
CNN（0.81MB）：max|Δ| = 9.5e-7，Pearson 1.0/1.0；ONNX 模型的 test 指标 ρ=0.632/0.569
与训练脚本的 0.639/0.566 一致。闸门不通过就不发布（脚本返回非 0）。
证据：`results/onnx_parity.json`、`scripts/export_onnx.py`。

### Q10. 为什么网页上跑的不是最优模型？

因为 481MB 的下载对网页 demo 不可接受。int8 量化被 ORT 量化器在 481MB 模型上的
shape-inference 形状冲突卡住（我试了三条路线，报错都记在 devlog），fp16 在 CPU/WASM 上
**加载即失败**（类型不匹配）。所以网页用 CNN（0.81MB）——它是基准里 Dev 侧最优的模型，
而 LoRA 的 ONNX 作为 Release 资产仍然提供。**这个取舍写在 webapp/README 与页面文案里**。
浏览器实测：模型加载 0.2–0.4s，单序列推理 2–15ms，747 变体敏感度 511–572ms（都 <3s）。
证据：devlog M5、`webapp/README.md`、`results/onnx_parity.json`。

### Q11. MCP 是什么？你的 MCP server 怎么验证？

MCP（Model Context Protocol）是让 LLM agent 调用外部工具的标准协议。我的 server 用 FastMCP
暴露三个工具：`predict_activity` / `explain_sequence` / `batch_scan`，一条命令启动
（`uv run enhancerscope-mcp`，stdio）。验证分两层：① 14 个单元测试（含入参 schema 断言与
回归值）在 CI 跑；② **协议层冒烟**真的拉起进程走 JSON-RPC：initialize → tools/list → tools/call。
端到端演示见 `docs/assets/mcp_demo.txt`。
证据：`src/enhancerscope/mcp_server.py`、`scripts/mcp_stdio_check.py`、`tests/test_mcp_tools.py`。

### Q12. 这个项目里最难的工程问题是什么？

不是模型，是**受限网络下的可复现性**：TUN 模式让 git/gh 全挂；Go 程序的 HTTP/2 被代理节点干扰；
Zenodo 不可达；hf-mirror 间歇性 SSL EOF。逐条解决的：git 强制 HTTP/1.1 + 代理；
`GODEBUG=http2client=0`；Clash 控制器实测节点延迟再切换；大文件一律 `curl -C -` 断点续传 +
重试循环（468MB 的 DNABERT-2 权重就是这么下完的）。全部记在 devlog（20+ 条坑）。
这些在面试里比"我调了个模型"更能说明工程能力。

### Q13. 如果没有 8GB 显存限制，你会怎么改这个项目？

三条：① 全参微调对照（验证 LoRA 与全参的差距）；② 更大模型（HyenaDNA medium/large、
Nucleotide Transformer 500M）做规模消融；③ LoRA 秩消融 r∈{4,8,16,32} + alpha 调参，
并把 epochs 提到 10+（当前 valid loss 在第 3 个 epoch 仍在下降，见曲线图）。
证据：`results/figures/lora_curves.png`、`docs/experiments.md` §4.2。

### Q14. 你的评价指标为什么用 Spearman？

这是回归任务且目标右偏（强增强子长尾）。Spearman 衡量单调排序一致性——对"哪些序列更可能是
强增强子"这类排序问题最贴切，对异常值不敏感；同时报 Pearson 和 RMSE 给出绝对误差视角。
DeepSTARR 论文也用 Spearman/Pearson，便于横向对照。
证据：`src/enhancerscope/metrics.py`、`results/baseline.csv`。

### Q15. 数据质量上你发现了什么问题，怎么处理的？

三个：① **18 条含 N 的序列**，全部来自异染色质臂（chrYHet/chr2RHet，dm6 未测序区），
且 N 的个数成对出现（正反链 twin）→ 确定性过滤（0.004%），规则写在 `scripts/clean_data.py`；
② **`label` 列是 numpy 数组列**（每行 `[Dev_scaled, Hk_scaled]` 二元组），不是类别标签——
我最初当成标签列做 `value_counts()` 直接把进程挂死，后来先查 dtype 才弄清；
③ 划分口径与官方论文可能不同，作为"已知局限"写进数据卡，M2 计划里本来要做文献对照（未完成，如实列出）。
证据：`docs/datacard.md`、`scripts/clean_data.py`。

### Q16. 如果让你重新做这个项目，你会改什么？

① 先把**环境与网络**摸清再动模型（我在 M0-M2 花了大量时间在与模型无关的网络问题上，
plan 里应该单列一个 milestone）；② `parse_fasta` 那种跨平台 bug 应该靠"本地 + CI 双跑"
更早暴露（我是 CI 第一次跑 MCP 测试才抓到的，而且当时还误合并了 PR——流程失误也记在 devlog）；
③ 更早锁定 ONNX 导出路线（M5 的风险预判是对的，但 int8 我试得太久，应该在第二次失败时就切方案）。

### Q17. 你的项目里"和 AI agent 协作"具体是怎么体现的？

全程用 agent 做实现与调试：调研（多个并行调研 agent 产出选型报告）→ 代码/实验/文档的实现 →
反复的失败-诊断-修复循环（比如 DNABERT-2 的 5 个兼容坑、ONNX 量化的 3 条路线、
HyenaDNA 的调用签名问题）。**但所有关键决策与验收是我定的**：数据集 fallback 的规则、
防泄漏划分、闸门阈值、页面用 CNN 的取舍、负结果的如实呈现。agent 负责把"可验证的东西"做出来，
我负责判断什么是可信的、什么该写进简历。
