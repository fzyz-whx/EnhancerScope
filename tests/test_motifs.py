"""motifs 模块的已知答案测试（CI CPU 上跑，无 torch 依赖）。"""

import numpy as np
import pandas as pd

from enhancerscope.motifs import (
    attribution_motif_overlap,
    log_odds,
    parse_pfm,
    scan,
    scan_sequence,
)


def _write_pfm(tmp_path):
    p = tmp_path / "mini.pfm"
    # 两个 motif：M1 偏好 AAAA（4 列均为 A）；M2 偏好 GGGG
    p.write_text(
        "\n".join(
            [
                "# 注释行",
                ">MA0001.1_testA",
                "0.97 0.01 0.01 0.01",
                "0.97 0.01 0.01 0.01",
                "0.97 0.01 0.01 0.01",
                "0.97 0.01 0.01 0.01",
                ">MA0002.1_testG",
                "0.01 0.01 0.97 0.01",
                "0.01 0.01 0.97 0.01",
                "0.01 0.01 0.97 0.01",
                "0.01 0.01 0.97 0.01",
            ]
        ),
        encoding="utf-8",
    )
    return p


def test_parse_pfm_counts(tmp_path) -> None:
    motifs = parse_pfm(_write_pfm(tmp_path))
    assert set(motifs) == {"MA0001.1_testA", "MA0002.1_testG"}
    assert motifs["MA0001.1_testA"].shape == (4, 4)


def test_log_odds_prefers_consensus() -> None:
    lo = log_odds(np.array([[0.97, 0.01, 0.01, 0.01]]))
    assert lo[0, 0] > lo[0, 1]  # A 列分数 > C 列


def test_scan_finds_expected_site() -> None:
    lo = log_odds(np.array([[0.97, 0.01, 0.01, 0.01]] * 4))
    hits = scan("TTTTAAAAGGGG", lo, threshold=3.0)
    assert [s for s, _ in hits] == [4]  # AAAA 起始于 4


def test_scan_no_hit_below_threshold() -> None:
    lo = log_odds(np.array([[0.97, 0.01, 0.01, 0.01]] * 4))
    assert scan("TTTTTTTT", lo, threshold=3.0) == []


def test_scan_sequence_returns_dataframe(tmp_path) -> None:
    motifs = parse_pfm(_write_pfm(tmp_path))
    hits = scan_sequence("GGGGAAAATTTT", motifs, frac=0.5)
    assert isinstance(hits, pd.DataFrame)
    assert {"motif", "start", "end", "score"} <= set(hits.columns)
    assert len(hits) >= 1


def test_attribution_motif_overlap_metric() -> None:
    importance = np.array([0, 0, 5, 5, 0, 0, 0, 0], dtype=float)  # 高归因在 2-3
    hits = pd.DataFrame([{"motif": "M", "start": 2, "end": 4, "score": 1.0}])
    m = attribution_motif_overlap(importance, hits, top_frac=0.25)  # top2 = 位置 2,3
    assert m["top_precision"] == 1.0
    assert m["enrichment"] > 1
