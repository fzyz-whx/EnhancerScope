"""M2 baseline ①: k-mer 词袋 + LightGBM（CPU，单种子，确定性）。

用法: uv run --group ml python scripts/baseline_kmer_gbm.py
产出: results/baseline.csv（追加 model=kmer_gbm 行）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enhancerscope.data import TARGETS, load_split  # noqa: E402
from enhancerscope.features import kmer_features  # noqa: E402
from enhancerscope.metrics import evaluate  # noqa: E402

RESULTS = Path("results/baseline.csv")
CONFIG = json.loads(Path("configs/baselines.json").read_text(encoding="utf-8"))


def append_rows(rows: list[dict]) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    header = not RESULTS.exists()
    df.to_csv(RESULTS, mode="a", header=header, index=False)
    print(f"已追加 {len(rows)} 行到 {RESULTS}")


def main() -> int:
    cfg = CONFIG["kmer_gbm"]
    seed = CONFIG["seed"]
    print(f"[kmer_gbm] k={cfg['k']} seed={seed}，构建特征中...")
    train = load_split("train")
    valid = load_split("valid")
    test = load_split("test")

    feats = {
        s: kmer_features(df["sequence"], k=cfg["k"])
        for s, df in [("train", train), ("valid", valid), ("test", test)]
    }
    print(f"[kmer_gbm] 特征维度: {feats['train'].shape}")

    rows: list[dict] = []
    for task in TARGETS:
        model = lgb.LGBMRegressor(
            objective=cfg["objective"],
            n_estimators=cfg["max_rounds"],
            learning_rate=cfg["learning_rate"],
            num_leaves=cfg["num_leaves"],
            feature_fraction=cfg["feature_fraction"],
            bagging_fraction=cfg["bagging_fraction"],
            bagging_freq=cfg["bagging_freq"],
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(
            feats["train"],
            train[task],
            eval_set=[(feats["valid"], valid[task])],
            callbacks=[
                lgb.early_stopping(cfg["early_stopping"], verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        best_iter = model.best_iteration_ or cfg["max_rounds"]
        for split, df in (("valid", valid), ("test", test)):
            pred = model.predict(feats[split], num_iteration=best_iter)
            m = evaluate(df[task].to_numpy(), pred)
            rows.append(
                {
                    "model": "kmer_gbm",
                    "task": task,
                    "split": split,
                    "seed": seed,
                    **m,
                    "notes": f"k={cfg['k']}词袋+GC; best_iter={best_iter}; {cfg['note']}",
                }
            )
            print(f"[kmer_gbm] {task} {split}: {m}")
    append_rows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
