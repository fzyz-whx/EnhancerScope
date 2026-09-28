# M3 实验记录：LoRA 微调基因组语言模型

> 数字全部来自真实运行，可追溯到 `results/lora.csv`、`results/logs/lora_*.json`（每 run 的曲线/显存/时长）
> 与 `results/lora_summary.md`（聚合表）。6 个 run = 2 模型 × 3 种子。

## 1. 实验设计

### 1.1 模型与目标
| 模型 | 参数量 | 架构要点 | 出处 |
|---|---|---|---|
| DNABERT-2-117M | 117M | BPE 分词（6-mer 词表 4096）+ ALiBi 位置编码 + **融合 Wqkv 投影** | Zhou et al., ICLR 2024 |
| HyenaDNA small-32k | 3.3M | 隐式长卷积（Hyena 算子）+ 单碱基分词（249bp → 249 tokens） | Nguyen et al., NeurIPS 2023 |

任务：249bp 序列 → 预测 Dev/Hk 两个增强子活性（log2 富集），回归，MSE 损失，输出 2 维（双任务单头）。

### 1.2 为什么 LoRA（而不是全参微调）
- **显存硬约束**：RTX 5060 Laptop 8GB。LoRA 可训参数占比 **0.50%**（589,824/117.7M，DNABERT-2），
  实测显存峰值 **2.67GB**——全参微调在此卡上不可行，LoRA 是唯一可行且有效的路径。
- **论文依据**：LoRA（Hu et al., ICLR 2022）以低秩更新逼近全参微调，小数据上更抗过拟合。

### 1.3 超参与选择理由（完整列表见 configs/lora.json）
| 超参 | 取值 | 理由 |
|---|---|---|
| LoRA r / alpha | 16 / 32 | 论文常用起点；alpha/r=2 缩放为社区惯例 |
| lora_dropout | 0.1 | 轻正则 |
| target_modules | DNABERT-2: `Wqkv`；HyenaDNA: `in_proj`/`out_proj` | **实测打印顶层模块名确定**（DNABERT-2 无独立 query/value，是融合投影——预设值猜错被脚本拦下，devlog 第 1 坑） |
| 学习率 | 3e-4 | LoRA 常用区间（比全参微调高一个量级） |
| 物理 batch / 累积 | 32 / 2（等效 64） | 8GB 折中；等效 batch 与 CNN baseline 可比 |
| max_epochs / patience | 3 / 2 | LoRA 收敛快；早停防过拟合 |
| 序列长度 | 249bp 原生（goal 约束 ≤512bp） | 数据本身长度，无截断 |

### 1.4 精度策略（如实记录的妥协）
DNABERT-2 自定义代码在标准注意力回退路径存在 **dtype 硬编码缺陷**（bf16 前向触发
`Float vs BFloat16` 混算崩溃）。采用 **fp32 权重 + autocast(bf16) 训练**：前向在 bf16 下进行
（goal 的混合精度约束由 autocast 承担），权重驻留 fp32 绕开缺陷。HyenaDNA 无此问题。
另：DNABERT-2 自定义 BertModel **不支持 gradient checkpointing**（脚本降级并打印）；因显存峰值仅
2.67GB，未启用也不影响可行性。

### 1.5 防泄漏契约（与全流程一致）
- 只用 `train` 训练（chr2L/3L/3R/X/4 等 11 臂）
- `valid`（chr2R 左半）仅用于早停
- `test`（chr2R 右半，与 valid 坐标零重叠）**只在选出最优权重后评估一次**

## 2. 结果（test split，3 种子均值 ± 标准差）

| 模型 | Dev ρ | Dev Pearson | Dev RMSE | Hk ρ | Hk Pearson | Hk RMSE | 时长/run | 显存峰值 |
|---|---|---|---|---|---|---|---|---|
| **LoRA DNABERT-2** | **0.6214 ± 0.0088** | 0.6393 | 1.1952 | **0.5722 ± 0.0044** | 0.7490 | 1.1386 | 47.1 min | 2.67 GB |
| LoRA HyenaDNA | 0.2845 ± 0.0021 | 0.2872 | 1.4763 | 0.2486 ± 0.0008 | 0.2756 | 1.6618 | 8.9 min | 2.66 GB |

种子间波动：DNABERT-2 的 Dev σ=0.0088 / Hk σ=0.0044（稳定）；HyenaDNA 的 σ 更小但均值极低。
原始数据：`results/lora.csv`（含每 seed 每 split 每任务的完整指标）；训练曲线：`results/figures/lora_curves.png`。

## 3. 与 M2 baseline 的对比（test Spearman ρ）

| 方法 | Dev | Hk | 备注 |
|---|---|---|---|
| ① k-mer(6)+GC → LightGBM | 0.581 | 0.519 | 无位置信息 |
| ② DeepSTARR 风格 CNN | **0.639** ± 0.003 | 0.566 ± 0.005 | 从零训练，400k 全量 |
| ③ DNABERT-2 零样本（embed+Ridge） | 0.428 | 0.363 | 不微调 |
| ④ **LoRA DNABERT-2**（本里程碑） | 0.6214 ± 0.0088 | **0.5722 ± 0.0044** | 微调 0.5% 参数，47min/run |
| ⑤ LoRA HyenaDNA（本里程碑） | 0.2845 ± 0.0021 | 0.2486 ± 0.0008 | **负结果**，见 §4.1 |

**结论（如实）**：
1. **LoRA 微调显著优于同模型零样本**：DNABERT-2 从 0.428/0.363 提升到 0.6214/0.5722
   （Dev +0.19 / Hk +0.21）——微调的价值被直接量化。
2. **LoRA DNABERT-2 拿下 Hk 最优**（0.5722 > CNN 0.566），Dev 次优（0.6214 < CNN 0.639，差 0.018）。
   一个只微调 0.5% 参数的预训练模型，与从零训练的全量 CNN 打平/互有胜负——这正是基因组语言模型
   预训练价值的直接证据，也是本项目"值得讲"的核心结论。
3. 与官方 DeepSTARR 论文（test Pearson ≈0.93）仍有差距（我们 0.64/0.75）：超参未完全复现、
   仅 3 epochs、LoRA 低秩容量限制——**本项目定位是透明可复现的对比基准，不是 SOTA 复现**。

## 4. 负结果归因与未竟事项（如实）

### 4.1 HyenaDNA 微调为何失败（0.285/0.249）
现象：训练 loss 仅从 2.46 降到 ~2.42（几乎没学动），valid loss 始终高于 k-mer baseline 水平。
归因假设（按可能性排序，均为可检验的假设而非结论）：
1. **容量限制**：3.3M 参数（比 DNABERT-2 小 35 倍），而生成式预训练获得的序列语法表示需要
   足够容量才能适配下游回归；
2. **分词粒度**：单碱基 tokenizer → 249 tokens/序列（DNABERT-2 的 BPE 仅 63），长字符序列的
   序列建模对小模型更难；
3. **超参不适配**：LoRA 只调 98,304 个参数（in_proj/out_proj），lr=3e-4 与 3 epochs 是为
   DNABERT-2 调的，未必适配 Hyena 算子；
4. **目标模块选择**：Hyena 算子中 in_proj/out_proj 可能不是最有效的适配位置（未做消融）。
**下一步值得尝试**（未做，如实记录）：换 hyenadna-medium/large、调 LR、增 epochs、加 target 模块消融。

### 4.2 其他局限
- LoRA 秩 r=16 未做消融（r∈{4,8,16,32}）；alpha 未调。
- 3 epochs 上限：DNABERT-2 的 valid loss 在 epoch 2 仍在下降（见曲线图），更多轮次可能更高。
- 未跑全参微调对照（8GB 显存不可行，已在 §1.2 说明）。
- 聚合脚本的曲线图为英文标签（matplotlib 默认字体无中文，避免豆腐块）。
