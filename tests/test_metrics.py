"""metrics.evaluate 的已知答案测试（CI CPU 上跑，不依赖 torch）。"""

import numpy as np
import pytest

from enhancerscope.metrics import evaluate


def test_perfect_prediction() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = evaluate(y, y)
    assert m["spearman"] == pytest.approx(1.0)
    assert m["pearson"] == pytest.approx(1.0)
    assert m["rmse"] == 0.0


def test_reversed_ranking() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = evaluate(y, y[::-1].copy())
    assert m["spearman"] == -1.0
    assert m["pearson"] == -1.0


def test_constant_prediction() -> None:
    y = np.array([1.0, 2.0, 3.0])
    m = evaluate(y, np.full(3, 2.0))
    assert m["spearman"] == 0.0  # 无秩区分
    assert m["rmse"] > 0.0


def test_nan_pairs_dropped() -> None:
    y = np.array([1.0, 2.0, np.nan, 4.0])
    p = np.array([1.0, 2.0, 3.0, 4.0])
    m = evaluate(y, p)
    assert m["spearman"] == pytest.approx(1.0)  # NaN 对被过滤后完全一致


def test_shape_mismatch_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="形状不一致"):
        evaluate(np.zeros(3), np.zeros(4))
