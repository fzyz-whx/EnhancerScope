"""统一回归指标：Spearman ρ / Pearson r / RMSE（M2 三条 baseline 与 M3 微调共用）。

约定：所有指标只对 finite 值计算（数据卡已断言目标列有限，此处是第二道防线）；
同一函数贯穿全部模型，保证 README benchmark 表的数字口径一致。
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """计算 Spearman ρ / Pearson r / RMSE，过滤任一侧非有限值的样本对。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"形状不一致: {y_true.shape} vs {y_pred.shape}")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        raise ValueError("有限样本不足 2 对，无法计算指标")
    yt, yp = y_true[mask], y_pred[mask]
    spearman = float(stats.spearmanr(yt, yp).statistic)
    pearson = float(stats.pearsonr(yt, yp).statistic)

    def no_signal(x: float) -> float:
        # 零方差（如常量预测）时相关系数无定义（NaN），按 0（无信号）处理并如实上报
        return 0.0 if np.isnan(x) else x

    return {
        "spearman": no_signal(spearman),
        "pearson": no_signal(pearson),
        "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
    }
