"""JASPAR PWM 解析与 log-odds 扫描器（自实现，无外部依赖——原理可讲、结果可复现）。

数据来源：`data/motifs/jaspar2020_insects.pfm`（JASPAR2020 CORE + UNVALIDATED insects，
经 vanheeringen-lab/gimmemotifs 仓库获取，文件头保留了原始出处与日期；
用 `scripts/download_jaspar.py` 可重新获取）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASES = "ACGT"
_LUT = np.full(256, -1, dtype=np.int8)
for _i, _b in enumerate(BASES):
    _LUT[ord(_b)] = _i


def parse_pfm(path: Path | str) -> dict[str, np.ndarray]:
    """解析 gimmemotifs/JASPAR `.pfm`（`>ID_NAME` 后接 L 行 4 列频率）→ {id: (L,4) float}。"""
    motifs: dict[str, np.ndarray] = {}
    name: str | None = None
    rows: list[list[float]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            if name is not None and rows:
                motifs[name] = np.asarray(rows, dtype=np.float64)
            name = line[1:].split()[0].strip()
            rows = []
        else:
            vals = [float(x) for x in line.split()]
            if len(vals) == 4:
                rows.append(vals)
    if name is not None and rows:
        motifs[name] = np.asarray(rows, dtype=np.float64)
    return motifs


def log_odds(pfm: np.ndarray, background: float = 0.25) -> np.ndarray:
    """频率矩阵 → log2-odds（伪计数 1e-3 防 log(0)；均匀背景 0.25，见 README 的简化说明）。"""
    p = np.clip(np.asarray(pfm, dtype=np.float64), 1e-3, None)
    p = p / p.sum(axis=1, keepdims=True)
    return np.log2(p / background)


def scan(seq: str, lo: np.ndarray, threshold: float) -> list[tuple[int, float]]:
    """在 seq 上扫描 log-odds PWM，返回 [(start, score)]（score ≥ threshold）。"""
    width = lo.shape[0]
    if len(seq) < width:
        return []
    codes = _LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
    hits: list[tuple[int, float]] = []
    cols = np.arange(width)
    for s in range(len(seq) - width + 1):
        w = codes[s : s + width]
        if (w < 0).any():
            continue
        score = float(lo[cols, w].sum())
        if score >= threshold:
            hits.append((s, score))
    return hits


def scan_sequence(
    seq: str, motifs: dict[str, np.ndarray], frac: float = 0.8
) -> pd.DataFrame:
    """全 motif 扫描：阈值 = frac × 该 motif 的最大可能得分（常用启发式）。

    返回 DataFrame[motif, start, end, score]。
    """
    hits: list[dict] = []
    for mid, pfm in motifs.items():
        lo = log_odds(pfm)
        threshold = frac * float(lo.max(axis=1).sum())
        for s, score in scan(seq, lo, threshold):
            hits.append(
                {
                    "motif": mid,
                    "start": s,
                    "end": s + lo.shape[0],
                    "score": round(score, 3),
                }
            )
    return (
        pd.DataFrame(hits).sort_values("start").reset_index(drop=True)
        if hits
        else pd.DataFrame(columns=["motif", "start", "end", "score"])
    )


def attribution_motif_overlap(
    importance: np.ndarray, hits: pd.DataFrame, top_frac: float = 0.1
) -> dict[str, float]:
    """量化指标：高归因位置（top_frac 分位）与 motif 命中位置的重叠。

    - hit_coverage: motif 覆盖的碱基中被判为高归因的比例（召回视角）
    - top_precision: 高归因碱基落在 motif 覆盖区内的比例（精确视角）
    - enrichment: top_precision / motif 覆盖率（>1 表示富集）
    """
    n = len(importance)
    covered = np.zeros(n, dtype=bool)
    for _, h in hits.iterrows():
        covered[int(h["start"]) : min(int(h["end"]), n)] = True
    k = max(1, int(round(n * top_frac)))
    top_idx = np.argsort(np.abs(importance))[-k:]
    top_mask = np.zeros(n, dtype=bool)
    top_mask[top_idx] = True
    cov_rate = covered.mean()
    top_precision = float((top_mask & covered).sum() / k)
    hit_coverage = float((covered & top_mask).sum() / max(1, covered.sum()))
    return {
        "motif_coverage": round(float(cov_rate), 4),
        "top_precision": round(top_precision, 4),
        "hit_coverage": round(hit_coverage, 4),
        "enrichment": (
            round(top_precision / cov_rate, 3) if cov_rate > 0 else float("nan")
        ),
    }
