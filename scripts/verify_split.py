"""防泄漏划分验证（M1 DoD）：把数据卡声明变成可断言的事实。

验证项：
1. 三个 split 序列长度均为 249bp（DeepSTARR 标准）
2. 碱基仅含 ACGT
3. 目标列全为有限数值
4. id 在三个 split 间无交集
5. 染色体级防泄漏：train 染色体与 valid/test 染色体无交集

用法: uv run python scripts/verify_split.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DATA = Path("data/processed")
SPLITS = ("train", "valid", "test")
SEQ_LEN = 249
TARGETS = ("Dev_log2_enrichment", "Hk_log2_enrichment")


def main() -> int:
    failures: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    id_sets: dict[str, set[str]] = {}

    for split in SPLITS:
        df = pd.read_parquet(DATA / f"{split}.parquet")
        frames[split] = df
        id_sets[split] = set(df["id"])
        seq = df["sequence"]

        bad_len = int((seq.str.len() != SEQ_LEN).sum())
        if bad_len:
            failures.append(f"[{split}] {bad_len} 条序列长度 != {SEQ_LEN}bp")

        bad_base = int((~seq.str.fullmatch(r"[ACGT]+")).sum())
        if bad_base:
            failures.append(f"[{split}] {bad_base} 条含 ACGT 以外字符")

        for col in TARGETS:
            if not pd.to_numeric(df[col], errors="coerce").notna().all():
                failures.append(f"[{split}] {col} 含非数值")

    chrom = (
        pd.DataFrame(
            {s: f["id"].str.split("_").str[0].value_counts() for s, f in frames.items()}
        )
        .fillna(0)
        .astype(int)
    )
    print("== 染色体 x split 分布 ==")
    print(chrom.to_string())

    tr_chroms = set(chrom.index[chrom["train"] > 0])
    for s in ("valid", "test"):
        overlap = tr_chroms & set(chrom.index[chrom[s] > 0])
        if overlap:
            failures.append(f"染色体泄漏: train 与 {s} 共享 {sorted(overlap)}")

    for a in SPLITS:
        for b in SPLITS:
            if a < b and (n := len(id_sets[a] & id_sets[b])):
                failures.append(f"id 交集泄漏: {a} ∩ {b} = {n} 条")

    print(
        f"\ntrain/valid/test = {len(frames['train']):,}/{len(frames['valid']):,}/{len(frames['test']):,}"
    )
    if failures:
        print("防泄漏验证失败:")
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    print("防泄漏验证全部通过 ✓ (249bp / ACGT / 目标有限 / id 无交集 / 染色体不重叠)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
