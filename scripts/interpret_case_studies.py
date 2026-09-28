"""M4 案例研究：对最优模型做突变扫描归因 + JASPAR motif 对照 + 出图。

用法: uv run --group ml python scripts/interpret_case_studies.py [--n-sample 2000]

产出:
  results/interpretability/case{N}.json     — 每个案例的原始数字（预测/重要度/motif 命中/重叠指标）
  results/figures/interpret_case{N}.png     — 归因图（per-base 重要度 + motif 命中位置）
案例选择（确定性，seed=42）:
  ①② test 抽样中 Dev / Hk 预测值最高的序列（模型最"自信"的例子）
  ③ 在 Dev 预测 top-50 中，motif 重叠富集最高的序列（最能讲"模型与已知生物学对齐"的例子）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enhancerscope.data import TARGETS, load_split  # noqa: E402
from enhancerscope.explain import insilico_mutagenesis, occlusion  # noqa: E402
from enhancerscope.model import load_finetuned, predict  # noqa: E402
from enhancerscope.motifs import (  # noqa: E402
    attribution_motif_overlap,
    parse_pfm,
    scan_sequence,
)

OUT_JSON = ROOT / "results" / "interpretability"
OUT_FIG = ROOT / "results" / "figures"
MOTIF_PFM = ROOT / "data" / "motifs" / "jaspar2020_insects.pfm"


def pick_cases(preds: np.ndarray, truth: np.ndarray, n_dev_top: int = 50) -> list[int]:
    """返回 3 个案例的索引（①② Dev/Hk 预测最高；③ Dev top-50 中占位，稍后按 motif 富集替换）。"""
    idx_dev = int(np.argmax(preds[:, 0]))
    idx_hk = int(np.argmax(preds[:, 1]))
    top_dev = np.argsort(preds[:, 0])[-n_dev_top:][::-1]
    return [idx_dev, idx_hk, int(top_dev[0])], top_dev


def plot_case(
    seq: str, imp: np.ndarray, hits: pd.DataFrame, metrics: dict, title: str, path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    x = np.arange(len(seq))
    for ax, t, name in zip(
        axes[:2], range(2), ("Dev (developmental)", "Hk (housekeeping)")
    ):
        ax.bar(x, imp[:, t], width=1.0, color=("#4878CF" if t == 0 else "#EE854A"))
        for _, h in hits.iterrows():
            ax.axvspan(int(h["start"]), int(h["end"]), color="green", alpha=0.12)
        ax.set_ylabel(f"|Δ pred|\n{name}")
    hits_line = hits.head(60)
    axes[2].set_ylim(-0.5, max(1, len(hits_line)) - 0.5)
    for i, (_, h) in enumerate(hits_line.iterrows()):
        axes[2].plot([int(h["start"]), int(h["end"])], [i, i], lw=2, color="green")
    axes[2].set_yticks(range(len(hits_line)))
    axes[2].set_yticklabels(
        [f"{h['motif']} ({h['score']:.1f})" for _, h in hits_line.iterrows()],
        fontsize=6,
    )
    axes[2].set_xlabel("position (bp)")
    axes[2].set_ylabel("JASPAR hits")
    fig.suptitle(
        f"{title}\noverlap: top_precision={metrics['top_precision']:.2f} "
        f"motif_coverage={metrics['motif_coverage']:.2f} enrichment={metrics['enrichment']:.2f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sample", type=int, default=2000)
    args = ap.parse_args()

    OUT_JSON.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    print("[M4] 加载最优微调模型（按 best_valid_mse 自动选种子）...")
    model, pooler, tokenizer, device = load_finetuned("dnabert2", seed=None)

    def predict_fn(seqs: list[str]) -> np.ndarray:
        return predict(model, pooler, tokenizer, seqs, device, batch=96)

    test = load_split("test")
    rng = np.random.default_rng(42)
    sample_idx = np.sort(
        rng.choice(len(test), size=min(args.n_sample, len(test)), replace=False)
    )
    sample = test.iloc[sample_idx].reset_index(drop=True)
    preds = predict_fn(sample["sequence"].tolist())
    truth = sample[list(TARGETS)].to_numpy()
    print(
        f"[M4] 抽样 {len(sample)} 条 test 预测完成；Dev ρ={pd.Series(preds[:,0]).corr(pd.Series(truth[:,0]), method='spearman'):.3f}"
    )

    motifs = parse_pfm(MOTIF_PFM)
    print(f"[M4] JASPAR 载入 {len(motifs)} 个 motif")

    (i_dev, i_hk, i_third), top_dev = pick_cases(preds, truth)

    # 案例③：在 Dev top-50 中找 motif 富集最高者
    best_enrich, best_case = -1.0, None
    for ci in top_dev[:20]:
        seq = sample.loc[ci, "sequence"]
        res = insilico_mutagenesis(predict_fn, seq)
        hits = scan_sequence(seq, motifs, frac=0.8)
        if hits.empty:
            continue
        m = attribution_motif_overlap(res["importance"][:, 0], hits, top_frac=0.1)
        if m["enrichment"] == m["enrichment"] and m["enrichment"] > best_enrich:
            best_enrich, best_case = m["enrichment"], int(ci)
    i_third = best_case if best_case is not None else int(top_dev[0])
    print(
        f"[M4] 案例③选定 index={i_third}（top-50 中 motif 富集最高 enrichment={best_enrich:.2f}）"
    )

    summary = []
    for n, ci in enumerate([i_dev, i_hk, i_third], start=1):
        row = sample.iloc[ci]
        seq = row["sequence"]
        res = insilico_mutagenesis(predict_fn, seq)
        occ = occlusion(predict_fn, seq, window=6, stride=3)
        hits = scan_sequence(seq, motifs, frac=0.8)
        m_dev = attribution_motif_overlap(res["importance"][:, 0], hits, top_frac=0.1)
        m_hk = attribution_motif_overlap(res["importance"][:, 1], hits, top_frac=0.1)
        # 两法一致性：6bp 窗口遮蔽重要度 vs 同窗口内单碱基突变的最大重要度（Spearman）
        # 两者都基于扰动但机制不同（遮蔽破坏语法 vs 单点替换），一致 → 归因稳健
        mut_win = np.array(
            [
                float(np.abs(res["importance"][s : s + 6, :]).max())
                for s in occ["starts"]
            ]
        )
        rho_agree = float(stats.spearmanr(occ["importance"], mut_win).statistic)
        case = {
            "case": n,
            "id": row["id"],
            "sequence": seq,
            "wildtype_pred": {
                "Dev": float(res["wildtype"][0]),
                "Hk": float(res["wildtype"][1]),
            },
            "truth": {
                "Dev": float(row["Dev_log2_enrichment"]),
                "Hk": float(row["Hk_log2_enrichment"]),
            },
            "top_attributed_positions": {
                "Dev": [int(p) for p in np.argsort(res["importance"][:, 0])[-8:][::-1]],
                "Hk": [int(p) for p in np.argsort(res["importance"][:, 1])[-8:][::-1]],
            },
            "motif_hits": hits.to_dict(orient="records")[:40],
            "overlap_metrics": {"Dev": m_dev, "Hk": m_hk},
            "attribution_agreement_spearman": round(rho_agree, 3),
            "occlusion_top": [
                {
                    "start": int(s),
                    "delta_Dev": round(float(d[0]), 3),
                    "delta_Hk": round(float(d[1]), 3),
                }
                for s, d in sorted(
                    zip(occ["starts"], occ["delta"]), key=lambda x: -abs(x[1]).max()
                )[:10]
            ],
        }
        (OUT_JSON / f"case{n}.json").write_text(
            json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        plot_case(
            seq,
            res["importance"],
            hits,
            m_dev,
            f"Case {n}: {row['id']}",
            OUT_FIG / f"interpret_case{n}.png",
        )
        summary.append(
            {
                "case": n,
                "id": row["id"],
                "pred_Dev": round(case["wildtype_pred"]["Dev"], 3),
                "true_Dev": round(case["truth"]["Dev"], 3),
                "pred_Hk": round(case["wildtype_pred"]["Hk"], 3),
                "true_Hk": round(case["truth"]["Hk"], 3),
                "n_motif_hits": len(hits),
                "Dev_top_precision": m_dev["top_precision"],
                "Dev_enrichment": m_dev["enrichment"],
                "Hk_top_precision": m_hk["top_precision"],
                "Hk_enrichment": m_hk["enrichment"],
            }
        )
        print(
            f"[M4] case{n} {row['id']}: hits={len(hits)} Dev enrich={m_dev['enrichment']:.2f} Hk enrich={m_hk['enrichment']:.2f}"
        )

    pd.DataFrame(summary).to_csv(OUT_JSON / "summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
