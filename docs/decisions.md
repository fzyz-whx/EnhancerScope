# 决策记录（Decisions）

> 格式：**问题 → 选项 → 决定 → 理由 → 代价/风险**。每条都能在 devlog 或结果文件中找到对应证据。
> 这是面试深挖时的"设计决策剧本"。

---

## D1. 数据集：为什么最终用的是果蝇 DeepSTARR，而不是人类 MPRA？

- **选项**：① 人类 MPRA（Kircher 2019 饱和 MPRA / Tewhey 2016 / VISTA 衍生）；② DeepSTARR（果蝇 S2 细胞 STARR-seq，约 48 万条）；③ GUE 基准（分类）。
- **决定**：② DeepSTARR。
- **理由**：本机网络环境（见 D7）下 **Zenodo 全程不可达**（10/10 + 5/5 次失败），而人类 MPRA 的原始数据托管在 Zenodo；HuggingFace 上仅有 `gonzalobenegas/sat_mut_mpra`（只有 test 分片，无法训练）；kircherlab 的 GitHub 仓库是 Shiny 前端而非数据。DeepSTARR 有 **HF 镜像（GenerTeam/DeepSTARR-enhancer-activity）**可直接下载，且提供**连续活性值**（非分类）、规模 48 万、社区复现广泛。
- **代价**：题材从人类变为果蝇；丧失"疾病相关调控元件"的叙事。**如实记录在 devlog 与数据卡，不假装是人类数据**。划分口径与官方论文未必一致这一点也写进了数据卡（已知局限）。

## D2. 划分策略：为什么坚持按染色体划分（chr2R 整臂 held-out）？

- **选项**：① 随机划分（最省事）；② 按染色体划分；③ 按序列相似度聚类划分（最严）。
- **决定**：②，并实测断言（`scripts/verify_split.py`）。
- **理由**：增强子序列存在大量同源重复（转座子、串联重复）。随机划分会把"近似重复"同时放进训练与测试，得到虚高的指标——这是增强子预测类论文最常见的翻车点。实测结果：train 覆盖 11 条臂（402,278 条），**整条 chr2R 臂 held out**（valid 40,570 + test 41,186），且 **valid/test 之间坐标零重叠（0.0%）**——数据内建划分已经是染色体级的，我们只是把它验证成断言并加了 sha256 校验清单。
- **代价**：指标低于随机划分的"好看数字"（例如 CNN test ρ=0.639 而非 0.7+）；换来的是**可辩护的评估**——面试时这是加分项而不是减分项。

## D3. 主实验：为什么用 LoRA 而不用全参微调？

- **选项**：① 全参微调（效果上限高）；② LoRA（参数高效）；③ 只做零样本/线性探针（最省）。
- **决定**：② LoRA。
- **理由**：**8GB 显存硬约束**——DNABERT-2 全参微调需要 >10GB（权重+梯度+优化器状态）；LoRA 只训练 0.50% 参数（589,824/117.7M），实测**显存峰值 2.67GB**，可以稳定跑 3 个种子。同时它给出了本项目最有价值的对照：同一模型的零样本（0.428/0.363）vs LoRA 微调（0.6214/0.5722）。
- **代价**：低秩容量可能限制上限（未做 r∈{4,8,32} 消融，如实列在 experiments.md 的局限里）。

## D4. 微调精度：为什么是 fp32 权重 + autocast(bf16)？

- **选项**：① bf16 加载 + bf16 前向（最省显存）；② fp32 全流程；③ fp32 权重 + autocast（混合）。
- **决定**：③。
- **理由**：DNABERT-2 的 remote code 在**标准注意力回退路径存在 dtype 硬编码缺陷**——bf16 加载后前向会报 `expected scalar type Float but found BFloat16`（devlog M3 有完整栈）。fp32 权重绕开缺陷，autocast 仍让前向走 bf16（goal 的混合精度约束由 autocast 承担）。
- **代价**：显存比纯 bf16 高一些（实测 2.67GB 完全可接受）；这是**第三方代码缺陷导致的技术妥协**，如实写在 experiments.md §1.4。

## D5. 浏览器端：为什么页面跑 CNN 而不是最"优"的 LoRA 模型？

- **选项**：① 页面加载 LoRA DNABERT-2（ONNX 481MB）；② int8 量化后加载（目标 ~120MB）；③ 页面加载 CNN（0.81MB），LoRA 的 ONNX 另存为 Release 资产。
- **决定**：③，且**两个模型的 ONNX 都导出并做了数值一致性验证**。
- **理由**：481MB 的下载对网页 demo 不可接受；int8 路线被 ORT 量化器在 481MB 模型上的 shape-inference 形状冲突卡住（devlog M5 第 14 坑），fp16 路线在 CPU/WASM 上**加载即失败**（第 15 坑）。CNN 不是"替身"：它是本项目基准里 **Dev 侧最优**的模型，且输入是 4×249 one-hot —— 浏览器端零分词依赖。
- **代价**：网页端的模型弱于 LoRA（Dev 2.64 vs 3.44 on 案例 2 序列）；**在 webapp/README 与页面文案里明确写出这一点**，不假装页面跑的是最优模型。

## D6. 可解释性：为什么用扰动法而不是梯度归因（saliency/IG）？

- **选项**：① 梯度归因（快、常见）；② 单碱基饱和突变（慢但精确）；③ 窗口遮蔽。
- **决定**：②（主）+ ③（辅），并做两法一致性交叉验证；梯度路线**尝试后放弃并记录报错**。
- **理由**：DNABERT-2 自定义代码的 `inputs_embeds` 路径报 `TypeError: ones_like(): argument 'input' must be Tensor, not NoneType`（硬依赖 input_ids），无法安全地对 embedding 求梯度；且 DNABERT-2 是 BPE 分词，一个 token 跨多个碱基，梯度到碱基的映射需要额外假设。扰动法则**精确、确定、模型无关**，且两法一致性（case2 ρ=0.735）本身就是归因稳健性的证据。
- **代价**：747 次前向/序列的计算量（GPU 上几秒级，浏览器端 0.5s）。

## D7. 网络受限环境下的工程处理（本机 TUN/系统代理/节点不稳）

- **现象**：TUN 模式反而让 git/gh 全挂；代理节点对 GitHub API 成功率低；Zenodo 完全不可达；hf-mirror 间歇性 SSL EOF。
- **决定**（逐条实测得出，全部记录在 devlog）：
  1. git 固定走系统代理端口（`http.https://github.com.proxy`）+ **强制 HTTP/1.1**；
  2. Go 程序（gh）的 HTTP/2 被节点干扰 → `GODEBUG=http2client=0`；
  3. Clash 外部控制器（127.0.0.1:9097）用于**实测节点延迟后切换**，而不是凭面板数字；
  4. 大文件下载一律 **curl `-C -` 断点续传 + 重试循环**（DNABERT-2 权重 468MB 靠这个下完的）；
  5. 下载脚本与运行分离（`scripts/download_*.py`），权重不入库。
- **代价**：脚本复杂了一些；换来的是一套"在受限网络下仍然可复现"的工程实践——这本身是面试可讲的内容。

## D8. 依赖分层：为什么 torch 在 `ml` 组，而 fastmcp/onnxruntime 在主依赖？

- **决定**：训练/推理的 ML 栈（torch/transformers/lightgbm/sklearn/peft）放 `[dependency-groups] ml`；MCP 层需要的 `fastmcp`、`onnxruntime` 放主依赖。
- **理由**：CI 要"跑 CPU 单测 + lint"（goal 硬约束），不能被 2.5GB 的 torch 拖慢；但 **MCP 工具是一等交付物，必须被 CI 真实测试**（不是跳过/标记 xfail）。onnxruntime CPU 版体积可控，且工具测试真的会跑一遍 0.81MB 的模型。
- **代价**：本地开发要记得 `uv sync --all-groups`；README 已写明。
