"""M3 结果聚合：训练曲线图（多种子均值±std）与 model × seed × 指标汇总表。

用法: uv run --group ml python scripts/summarize_lora.py
产出:
  results/figures/lora_curves.png   — train/valid loss 曲线（按模型，多 seed）
  results/lora_summary.md           — 汇总表（test split，均值±标准差，供 README/experiments.md 引用）
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "results" / "logs"
FIG = ROOT / "results" / "figures"
CSV = ROOT / "results" / "lora.csv"


def load_curves() -> dict[str, list[dict]]:
    curves: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(LOGS.glob("lora_*_seed*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        curves[d["model"]].append(d)
    return curves


def plot(curves: dict[str, list[dict]]) -> Path | None:
    if not curves:
        print("没有曲线数据（results/logs/lora_*.json 不存在）")
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.2), squeeze=False)
    for ax, (model, runs) in zip(axes[0], curves.items()):
        for phase, ls in (("train_loss", "-"), ("valid_loss", "--")):
            max_ep = max(len(r["curve"]) for r in runs)
            mat = np.full((len(runs), max_ep), np.nan)
            for i, r in enumerate(runs):
                for e, row in enumerate(r["curve"]):
                    mat[i, e] = row[phase]
            mean, std = np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)
            x = np.arange(max_ep)
            ax.plot(x, mean, ls, marker="o", label=phase.replace("_loss", ""))
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)
        ax.set_title(f"{model}（{len(runs)} seeds）")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE loss")
        ax.legend()
    fig.suptitle("M3 LoRA 训练曲线（均值 ± 标准差）")
    fig.tight_layout()
    out = FIG / "lora_curves.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"曲线图: {out}")
    return out


def summary() -> Path | None:
    if not CSV.exists():
        print("results/lora.csv 不存在")
        return None
    df = pd.read_csv(CSV)
    test = df[df["split"] == "test"]
    rows = []
    for (model, task), g in test.groupby(["model", "task"]):
        rows.append(
            {
                "model": model,
                "task": task,
                "n_seeds": len(g),
                "spearman_mean": g["spearman"].mean(),
                "spearman_std": g["spearman"].std(ddof=0),
                "pearson_mean": g["pearson"].mean(),
                "rmse_mean": g["rmse"].mean(),
            }
        )
    out_df = pd.DataFrame(rows).sort_values(
        ["task", "spearman_mean"], ascending=[True, False]
    )
    out = ROOT / "results" / "lora_summary.md"
    try:
        table = out_df.to_markdown(index=False, floatfmt=".4f")
    except ImportError:  # tabulate 未安装时退化为纯文本表
        table = "```\n" + out_df.to_string(index=False) + "\n```"
    lines = ["# M3 LoRA 汇总（test split）\n", table]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_df.to_string(index=False))
    print(f"汇总: {out}")
    return out


def main() -> int:
    curves = load_curves()
    print(f"发现 {sum(len(v) for v in curves.values())} 个 run，模型: {list(curves)}")
    plot(curves)
    summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
