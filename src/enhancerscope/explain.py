"""M4 归因方法：in-silico 突变扫描（主）与窗口遮蔽遮挡（辅）。

为何以突变扫描为主：DNABERT-2 是 BPE 分词（一个 token 跨多个碱基），梯度归因需要
token→碱基的额外映射假设；而逐碱基饱和突变是**精确、确定、模型无关**的 per-base 归因，
且直接对应生物学提问"这个位点的碱基换了，活性会怎样变"。
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from enhancerscope.motifs import BASES


def insilico_mutagenesis(
    predict_fn: Callable[[list[str]], np.ndarray], seq: str
) -> dict:
    """逐位置替换为其余 3 种碱基，测 Δ 预测。

    返回 {"wildtype": (2,), "delta": (L, 4, 2), "importance": (L, 2)}
    - delta[p, b, t] = pred_t(突变 p→b) − pred_t(野生型)，被替换的碱基记 nan
    - importance[p, t] = max_b |delta[p, b, t]|（该位点对任务 t 的最大影响）
    """
    wt = predict_fn([seq])[0].astype(np.float64)
    length = len(seq)
    variants: list[str] = []
    meta: list[tuple[int, int]] = []
    for p, ref in enumerate(seq):
        for bi, b in enumerate(BASES):
            if b != ref:
                variants.append(seq[:p] + b + seq[p + 1 :])
                meta.append((p, bi))
    preds = predict_fn(variants).astype(np.float64)
    delta = np.full((length, 4, wt.shape[0]), np.nan)
    for (p, bi), pred in zip(meta, preds):
        delta[p, bi, :] = pred - wt
    importance = np.nanmax(np.abs(delta), axis=1)
    return {"wildtype": wt, "delta": delta, "importance": importance}


def occlusion(
    predict_fn: Callable[[list[str]], np.ndarray],
    seq: str,
    window: int = 6,
    stride: int = 3,
) -> dict:
    """窗口遮蔽：把 [s, s+window) 替换为 'N'×window，测 Δ 预测（辅证方法）。"""
    wt = predict_fn([seq])[0].astype(np.float64)
    starts = list(range(0, max(1, len(seq) - window + 1), stride))
    variants = [
        seq[:s] + "N" * min(window, len(seq) - s) + seq[s + window :] for s in starts
    ]
    preds = predict_fn(variants).astype(np.float64)
    delta = preds - wt
    return {
        "wildtype": wt,
        "starts": starts,
        "delta": delta,
        "importance": np.abs(delta).max(axis=1),
    }
