"""特征工程：one-hot 编码与 k-mer 词袋（M2 baseline ① 的输入）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from enhancerscope.data import to_codes

KMER_BASES = "ACGT"


def kmer_vocab(k: int) -> list[str]:
    """长度 k 的全部 ACGT 词（按字典序）。"""
    if k <= 0:
        raise ValueError(f"k 必须为正整数，收到 {k}")
    vocab = [""]
    for _ in range(k):
        vocab = [p + b for p in vocab for b in KMER_BASES]
    return vocab


def kmer_counts(sequences: pd.Series | list[str], k: int = 6) -> sparse.csr_matrix:
    """k-mer 词频矩阵 (N, 4^k) float32 CSR。

    实现：碱基 → 2bit 索引，k-mer 索引用 k 位四进制编码（rolling polynomial），
    np.add.at 累加计数后转 CSR。含非 ACGT 字符的窗口跳过（数据卡：清洗后无 N）。

    局限（如实）：词袋无位置信息——k-mer baseline 丢失 motif 顺序语法，
    这正是它与 CNN / 语言模型的差距来源之一，benchmark 表里如实记录。
    """
    codes = to_codes(sequences).astype(np.int64)
    n, length = codes.shape
    if k > length:
        raise ValueError(f"k={k} 超过序列长度 {length}")
    n_windows = length - k + 1
    idx = np.zeros((n, n_windows), dtype=np.int64)
    valid = np.ones((n, n_windows), dtype=bool)
    for j in range(k):
        window = codes[:, j : j + n_windows]
        idx = idx * 4 + window
        valid &= window >= 0
    idx[~valid] = 0  # 无效窗口占位，稍后用 mask 权重清零

    rows = np.repeat(np.arange(n), n_windows)
    cols = idx.ravel()
    weights = valid.ravel().astype(np.float32)
    mat = sparse.coo_matrix((weights, (rows, cols)), shape=(n, 4**k), dtype=np.float32)
    counts = mat.tocsr()
    counts.sum_duplicates()
    return counts


def gc_fraction(sequences: pd.Series | list[str]) -> np.ndarray:
    """GC 含量 (N,) float32。"""
    codes = to_codes(sequences)
    gc = ((codes == 1) | (codes == 2)).sum(axis=1)
    total = (codes >= 0).sum(axis=1).clip(min=1)
    return (gc / total).astype(np.float32)


def kmer_features(sequences: pd.Series | list[str], k: int = 6) -> sparse.csr_matrix:
    """baseline ① 的输入：k-mer 词频 + GC 含量（拼接为 (N, 4^k + 1) CSR）。"""
    counts = kmer_counts(sequences, k=k)
    gc = sparse.csr_matrix(gc_fraction(sequences).reshape(-1, 1))
    return sparse.hstack([counts, gc], format="csr", dtype=np.float32)
