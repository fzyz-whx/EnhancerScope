"""生成 M1 EDA notebook（真实代码单元格，随后用 nbconvert 执行嵌入输出）。"""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

cells = [
    nbf.v4.new_markdown_cell(
        "# DeepSTARR 数据勘察（M1）\n\n"
        "**数据**：`GenerTeam/DeepSTARR-enhancer-activity`（HF 镜像），源自官方 Zenodo 5502060，"
        "仅格式调整。果蝇 S2 细胞 STARR-seq，249bp，双任务连续目标"
        "（`Dev_log2_enrichment` 发育型 / `Hk_log2_enrichment` 管家型）。\n\n"
        "**防泄漏事实（scripts/verify_split.py 已断言）**：train 含 11 条染色体臂，"
        "整条 chr2R 臂 held out（valid+test 全部取自 chr2R）；18 条含 N 序列"
        "（异染色质臂未测序区）已由 scripts/clean_data.py 过滤。"
    ),
    nbf.v4.new_code_cell(
        "from pathlib import Path\n\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "FIG = Path('../results/figures')\n"
        "FIG.mkdir(parents=True, exist_ok=True)\n"
        "SPLITS = ('train', 'valid', 'test')\n"
        "COLORS = {'train': '#4878CF', 'valid': '#EE854A', 'test': '#60AC45'}\n"
        "TARGETS = ('Dev_log2_enrichment', 'Hk_log2_enrichment')\n\n"
        "df = {s: pd.read_parquet(f'../data/processed/{s}.parquet') for s in SPLITS}\n"
        "for s in SPLITS:\n"
        "    print(f'{s:>6}: {len(df[s]):>8,} 行 | 列: {list(df[s].columns)}')"
    ),
    nbf.v4.new_markdown_cell("## 1. 活性分布（双任务连续目标）"),
    nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)\n"
        "for ax, col, name in zip(axes, TARGETS, ('发育型 Dev', '管家型 Hk')):\n"
        "    for s in SPLITS:\n"
        "        ax.hist(df[s][col], bins=80, alpha=0.45, density=True, label=s, color=COLORS[s])\n"
        "    ax.set_title(f'{name} log2 富集倍数分布')\n"
        "    ax.set_xlabel('log2 enrichment')\n"
        "    ax.legend()\n"
        "axes[0].set_ylabel('密度')\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG / '01_activity_dist.png', dpi=130, bbox_inches='tight')\n"
        "fig"
    ),
    nbf.v4.new_markdown_cell(
        "**解读**：两任务活性均右偏（强增强子长尾），train/valid/test 三 split 形状一致——"
        "染色体级划分未造成分布漂移。"
    ),
    nbf.v4.new_markdown_cell("## 2. 序列长度与 GC 含量"),
    nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
        "lens = {s: df[s]['sequence'].str.len().unique() for s in SPLITS}\n"
        "axes[0].bar(list(lens), [len(v) for v in lens.values()], color=[COLORS[s] for s in SPLITS])\n"
        "axes[0].set_title('序列长度唯一值个数（全部应为 1，即 249bp）')\n"
        "axes[0].set_ylabel('唯一长度数')\n"
        "for s in SPLITS:\n"
        "    gc = df[s]['sequence'].str.count('[GC]') / df[s]['sequence'].str.len()\n"
        "    axes[1].hist(gc, bins=60, alpha=0.45, density=True, label=s, color=COLORS[s])\n"
        "axes[1].set_title('GC 含量分布')\n"
        "axes[1].set_xlabel('GC fraction')\n"
        "axes[1].legend()\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG / '02_length_gc.png', dpi=130, bbox_inches='tight')\n"
        "fig"
    ),
    nbf.v4.new_markdown_cell(
        "**解读**：长度统一 249bp（每个 split 唯一长度数 = 1）；GC 含量近正态（~45%），"
        "split 间一致——k-mer 类 baseline 不会因 GC 偏差占便宜。"
    ),
    nbf.v4.new_markdown_cell("## 3. 染色体 × split（防泄漏结构）"),
    nbf.v4.new_code_cell(
        "chrom = pd.DataFrame(\n"
        "    {s: f['id'].str.split('_').str[0].value_counts() for s, f in df.items()}\n"
        ").fillna(0).astype(int)\n"
        "fig, ax = plt.subplots(figsize=(11, 4))\n"
        "bottom = np.zeros(len(chrom))\n"
        "for s in SPLITS:\n"
        "    ax.bar(chrom.index, chrom[s], bottom=bottom, label=s, color=COLORS[s])\n"
        "    bottom += chrom[s].to_numpy()\n"
        "ax.set_yscale('log')\n"
        "ax.set_title('染色体 x split（对数轴）：chr2R 整臂 held out')\n"
        "ax.set_ylabel('序列数 (log)')\n"
        "ax.legend()\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG / '03_chr_split.png', dpi=130, bbox_inches='tight')\n"
        "fig"
    ),
    nbf.v4.new_markdown_cell(
        "**解读**：train 覆盖 11 条臂（chr2L/3L/3R/X/4 及异染色质），**chr2R 完整 held out**"
        "（valid 40,570 + test 41,186 全部来自 chr2R）——同源序列（>80% 相似）几乎必然落在"
        "同臂内，染色体级划分是防泄漏的硬保证。"
    ),
    nbf.v4.new_markdown_cell("## 4. Dev 与 Hk 任务相关性"),
    nbf.v4.new_code_cell(
        "samp = df['train'].sample(20000, random_state=0)\n"
        "r = samp['Dev_log2_enrichment'].corr(samp['Hk_log2_enrichment'])\n"
        "fig, ax = plt.subplots(figsize=(5.5, 5))\n"
        "ax.scatter(samp['Dev_log2_enrichment'], samp['Hk_log2_enrichment'], s=3, alpha=0.25, color='#4878CF')\n"
        "ax.set_xlabel('Dev log2 enrichment')\n"
        "ax.set_ylabel('Hk log2 enrichment')\n"
        "ax.set_title(f'Dev vs Hk（train 抽样 2 万, Pearson r={r:.2f}）')\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG / '04_dev_vs_hk.png', dpi=130, bbox_inches='tight')\n"
        "print(f'Pearson r = {r:.3f}')\n"
        "fig"
    ),
    nbf.v4.new_markdown_cell(
        "**解读**：双任务相关性中等——发育型与管家型增强子共享部分序列语法但各有特异 motif，"
        "这正是双头模型（DeepSTARR 式）有意义的依据，也是 M4 可解释性分析的切入点。"
    ),
]
nb["cells"] = cells

out = Path("notebooks/01_eda_deepstarr.ipynb")
nbf.write(nb, out)
print(f"notebook 已生成: {out} ({len(cells)} 个单元格)")
