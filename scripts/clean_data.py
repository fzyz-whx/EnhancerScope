"""M1 数据清洗：过滤含 N 的序列（异染色质臂的未测序区域），生成 processed parquet。

DeepSTARR (dm6) 的异染色质臂（chrYHet / chr2RHet 等）在参考基因组中存在大量未测序
N 区，18 条 train 序列含 N（0.004%）。过滤策略对活性分布影响可忽略，且避免 one-hot
编码时引入第五通道。清洗规则是确定性的：同一份 raw 输入永远得到字节级一致的输出。

用法: uv run python scripts/clean_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")
SPLITS = ("train", "valid", "test")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    total_removed = 0
    for split in SPLITS:
        df = pd.read_parquet(RAW / f"{split}.parquet")
        keep = df["sequence"].str.fullmatch(r"[ACGT]+")
        removed = int((~keep).sum())
        total_removed += removed
        clean = df[keep].reset_index(drop=True)
        clean.to_parquet(OUT / f"{split}.parquet", index=False)
        print(f"[{split}] {len(df):,} -> {len(clean):,} (移除 {removed} 条含N序列)")
    print(f"共移除 {total_removed} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
