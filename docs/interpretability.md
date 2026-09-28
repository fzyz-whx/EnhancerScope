# M4 可解释性分析：模型在看哪里？

> 对象：**LoRA 微调 DNABERT-2**（`best_valid_mse` 最优种子自动选出，即 seed 42；见 `src/enhancerscope/model.py::load_finetuned`）
> 原始数字：`results/interpretability/case{1,2,3}.json` + `summary.csv`；图：`results/figures/interpret_case{1,2,3}.png`
> 复现：`uv run --group ml python scripts/interpret_case_studies.py`（固定 seed=42 抽样）

## 1. 方法：两种扰动归因 + JASPAR 对照

| 方法 | 做什么 | 为什么这样选 |
|---|---|---|
| **单碱基饱和突变**（主） | 249 个位置逐一替换为其余 3 种碱基（3×249=747 次前向/序列），Δ预测 = 突变 − 野生型 | 精确、确定、模型无关；直接对应生物学提问"这个位点换了，活性怎么变" |
| **窗口遮蔽**（辅） | 6bp 窗口以 `N` 遮蔽（步长 3），测 Δ预测 | 破坏整段语法（而非单点），与单碱基法机制互补 |
| **梯度归因（saliency / IG）** | 对 embedding 求 ∂output/∂embedding | **不可用，如实记录**：DNABERT-2 自定义 `BertModel` 在 `inputs_embeds` 路径报 `TypeError: ones_like(): argument 'input' must be Tensor, not NoneType`（其代码硬依赖 `input_ids`）。且 BPE 分词下一个 token 跨多个碱基，需额外映射假设 |

**两法一致性**（窗口遮蔽重要度 vs 同窗口内单碱基最大重要度，Spearman ρ）：
case1 ρ=**0.405**、case2 ρ=**0.735**、case3 ρ=**0.592**——case2/3 两法高度一致（归因稳健），
case1 一致性低本身就是信号（见案例 1 分析）。

**JASPAR 对照**：JASPAR2020 CORE + UNVALIDATED **insects** 集合（146 个 PWM；经
`vanheeringen-lab/gimmemotifs` 仓库获取，文件头保留原始出处与日期；`jaspar.genereg.net`
在本机网络不可达，见 devlog M4）。自实现 log-odds 扫描器（`src/enhancerscope/motifs.py`），
阈值 = 0.8 × 该 motif 最大可能得分。

**量化指标**：`top_precision`（高归因碱基落在 motif 命中区内的比例）、
`enrichment` = top_precision / motif 覆盖率（>1 表示富集，= 模型注意力与已知 motif 对齐）。

## 2. 三个案例

### 案例 2：模型与已知发育增强子语法对齐（最清晰的正例）
- **序列**：`chr2R_13586410_13586658_-`（chr2R 右半 test 集，抽样中 Hk 预测最高）
- **预测 vs 真值**：Dev 3.44 / 真 4.18；Hk **6.26 / 真 6.59**（高活性增强子，预测方向与量级均正确）
- **归因**：top 位置 Dev = Hk = **[126, 127, 128, 76, 73, 129]**（两任务贡献位置几乎完全重合，
  提示共享的序列语法）
- **motif 命中**：`MA0536.1_pnr`（pannier，**GATA 家族**）**123–134，score 14.6（全序列最高）**——
  **top 归因位置 126/127/128 正落在该命中区内**；另有 `MA0218.1_ct`（cut，同源域）37–43、
  `MA0231.1_lbe` / `MA0232.1_lbl`（ladybird，同源域）96–102 与 192–198、`MA0170.1_C15`（GATA 家族）218–225
- **重叠指标**：Dev enrichment **2.49**、Hk enrichment **2.77**（top-10% 归因碱基落在 motif 区的比例是随机基线的 2.5–2.8 倍）
- **文献佐证**：果蝇发育增强子的经典语法是 **GATA 因子 + 同源域（homeodomain）因子的组合编码**。
  `pnr`（pannier）编码 GATA 家族转录因子（Ramain et al., EMBO J 1993；Heitzler et al., Development 1996）；
  `ct`/`lbe`/`lbl` 为同源域因子（Blochlinger et al., Nature 1988；Jagla et al., Development 1997）；
  组合编码/flexible billboard 的增强子语法见 Arnosti & Kulkarni, J Cell Biochem 2005。
  JASPAR 数据库本身见 Fornes et al., NAR 2020。
- **结论**：模型不是"记住"序列，而是在 GATA + 同源域 motif 的位置上做因果性判断——这是本项目最想展示的证据。

### 案例 3：Dev 侧对齐、Hk 侧不对齐（部分正例）
- **序列**：`chr2R_11878583_11878831_+`（Dev top-50 中 motif 富集最高者）
- **预测 vs 真值**：Dev **3.17 / 真 3.01**（准确）；Hk 0.79 / 真 2.10（低估）
- **归因**：Dev top = [123, 133, 124, 122, 165]；Hk top = [241, 202, 242, 235, 133]（两任务位置几乎不重叠）
- **motif 命中**：`MA0210.1_ara`（arachnid）、`MA0217.1_caup`（caupolican）、`MA0222.1_exd`（extradenticle）、
  `MA0233.1_mirr`（mirror）、`MA0199.1_Optix`、`MA0185.1_Deaf1`
- **重叠指标**：Dev enrichment **2.85**（最高）、Hk enrichment **0.71**（低于随机基线）
- **解读（如实）**：Dev 与 Hk 是共享主干、不同头部的双任务模型；本案例中 Dev 头把注意力集中在一簇
  同源域 motif 上（ara/caup/mirr/exd 均为同源域或同源域辅因子，`exd` 是同源域蛋白的关键辅因子
  ——Mann & Chan, Curr Opin Genet Dev 1996），而 Hk 头的高归因位置（241/202/242/235）**不与任何
  已知 motif 重合**——它对 Hk 的低估（0.79 vs 真 2.10）可能正源于此（未捕捉到驱动管家型活性的语法）。

### 案例 1：过度预测 + 归因与 motif 不对齐（诚实的失败案例）
- **序列**：`chr2R_18268125_18268373_+`（抽样中 Dev 预测最高）
- **预测 vs 真值**：Dev **4.28 / 真 2.03**（严重高估）；Hk 0.84 / 真 **-0.78**（方向都反了）
- **motif 命中**：18 个（含 `MA0165.1_Abd-B`（Hox）、`ara`、`caup`、`mirr`、`Dbx`、`Deaf1` 等）
- **重叠指标**：Dev enrichment **0.50**、Hk **0.25**（低于基线——归因反而避开 motif 命中区）
- **两法一致性**：ρ=0.405（三种信号同时变弱：预测错、归因弱、两法不一致）
- **如实分析可能原因**：
  1. **STARR-seq 的测量语境**：这是质粒报告实验，与基因组语境（染色质可及性、甲基化等）解耦——
     含 Hox/同源域 motif 的序列在基因组中可能被压制，但在质粒上不是；
  2. **孤立 motif 匹配不等于语法**：模型可能对"像增强子"的局部特征给高分，而缺少对 motif 间距/朝向
     （syntax）的敏感度——与本案例归因避开 motif 区一致；
  3. **回归长尾**：Dev 真值分布右偏（数据卡），高活性区长尾样本少，模型在高分区倾向于过度外推。
- **为什么这条反而重要**：它说明我们的评估没有只挑好看的案例；同时给出可检验的下一步
  （对高预测样本做 held-out 校准分析、或在 loss 中对长尾加权）。

## 3. 小结（可讲的三句话）

1. **模型学到了生物学**：两个案例的高归因位置与 JASPAR 果蝇 motif 重叠富集 2.5–2.9 倍，
   且命中集中在同源域 + GATA 家族——与果蝇发育增强子已知的组合语法一致。
2. **可解释性方法本身是工程**：梯度归因因自定义模型代码不可用（记录了具体报错），改用
   单碱基饱和突变（主）+ 窗口遮蔽（辅）两种扰动法，并在 case2/3 上互相印证（ρ=0.735/0.592）。
3. **失败案例同样有价值**：最高预测样本反而高估 2.25 log2 且归因与 motif 不对齐，指向
   STARR-seq 语境差异与 motif-syntax 敏感度两个可检验假设。

## 4. 局限（如实）

- JASPAR 用 **2020 insects**（非 2024 版）：本机无法访问 jaspar.genereg.net，via gimmemotifs 获取。
- 只做了 3 个案例（抽样 2000 条中选），未做全 test 集的系统性归因统计。
- 均匀背景（0.25）的 log-odds 是简化；果蝇基因组 GC ~42%。
- 阈值启发式（0.8 × 最大得分）未与 FPR 校准。
- 未做 motif 消融实验（把命中区替换为随机序列看 Δ 预测）——这是最直接的下一个实验。
