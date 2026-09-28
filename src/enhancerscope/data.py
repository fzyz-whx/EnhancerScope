"""数据加载与编码（唯一数据入口：M1 数据卡的 data/processed/*.parquet）。

防泄漏契约：任何模型只允许 load_split("train") 训练；
valid/test 仅用于评估（染色体级划分见 docs/datacard.md）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed")
SPLITS = ("train", "valid", "test")
TARGETS = ("Dev_log2_enrichment", "Hk_log2_enrichment")
SEQ_COL = "sequence"
ID_COL = "id"

# ASCII → 碱基索引（A=0, C=1, G=2, T=3；其余字符 → -1）
_LOOKUP = np.full(256, -1, dtype=np.int8)
for _i, _b in enumerate("ACGT"):
    _LOOKUP[ord(_b)] = _i


def load_split(split: str, data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """加载清洗后的 split；split 名必须是 train/valid/test。"""
    if split not in SPLITS:
        raise ValueError(f"split 必须是 {SPLITS} 之一，收到 {split!r}")
    return pd.read_parquet(Path(data_dir) / f"{split}.parquet")


def to_codes(sequences: pd.Series | list[str]) -> np.ndarray:
    """序列 → (N, L) int8 碱基索引矩阵，非 ACGT 字符为 -1。"""
    seq_list = list(sequences)
    joined = "".join(seq_list).encode("ascii")
    codes = np.frombuffer(joined, dtype=np.uint8).reshape(len(seq_list), -1)
    return _LOOKUP[codes].astype(np.int8)


def one_hot(sequences: pd.Series | list[str]) -> np.ndarray:
    """序列 → (N, 4, L) uint8 one-hot（向量化；非 ACGT 字符 → 全零通道）。"""
    codes = to_codes(sequences).astype(np.int64)
    n, length = codes.shape
    out = (codes[None, :, :] == np.arange(4)[:, None, None]).transpose(1, 0, 2)
    return out.astype(np.uint8).reshape(n, 4, length)


def chrom_arm(df: pd.DataFrame) -> pd.Series:
    """从 id 列（chr臂_起_止_链_类别）提取染色体臂，防泄漏验证用。"""
    return df[ID_COL].str.split("_").str[0]
