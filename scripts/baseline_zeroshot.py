"""M2 baseline ③: 基因组语言模型零样本打分（不微调）。

embedding 均值池化 + Ridge 线性探针。DNABERT-2 (zhihan1996/DNABERT-2-117M) 仅作
特征提取器，任何权重更新都没有发生——这就是 goal 定义的"零样本"口径。

用法: uv run --group ml python scripts/baseline_zeroshot.py
环境: HF_ENDPOINT=https://hf-mirror.com（国内镜像，见 README FAQ）
产出: results/baseline.csv（追加 model=zeroshot_ridge 行）
缓存: data/processed/embeddings/{split}.npy（train 为 5 万抽样，见 configs）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enhancerscope.data import TARGETS, load_split  # noqa: E402
from enhancerscope.metrics import evaluate  # noqa: E402

RESULTS = Path("results/baseline.csv")
CONFIG = json.loads(Path("configs/baselines.json").read_text(encoding="utf-8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def append_rows(rows: list[dict]) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    header = not RESULTS.exists()
    df.to_csv(RESULTS, mode="a", header=header, index=False)
    print(f"已追加 {len(rows)} 行到 {RESULTS}")


@torch.no_grad()
def embed(model, tokenizer, sequences: list[str], batch_size: int) -> np.ndarray:
    """DNABERT-2 最后一层隐状态均值池化 → (N, 768) float32。"""
    model.eval()
    out = []
    for i in range(0, len(sequences), batch_size):
        batch = tokenizer(
            sequences[i : i + batch_size], return_tensors="pt", padding=True
        ).to(DEVICE)
        raw = model(**batch)
        # DNABERT-2 标准注意力回退返回 tuple (last_hidden_state, pooled)，需兼容
        hidden = (
            raw[0] if isinstance(raw, tuple) else raw.last_hidden_state
        )  # (B, T, H)
        mask = batch["attention_mask"].unsqueeze(-1)  # (B, T, 1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        out.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(out)


def get_embeddings(
    model,
    tokenizer,
    split: str,
    sequences: list[str],
    cache_dir: Path,
    sample: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{split}.npy"
    if cache.exists():
        print(f"[zeroshot] 命中缓存 {cache}")
        return np.load(cache)
    if sample is not None and sample < len(sequences):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(sequences), size=sample, replace=False)
        idx.sort()
        sequences = [sequences[i] for i in idx]
        print(f"[zeroshot] train 抽样 {sample:,}/{len(sequences):,} (seed={seed})")
    print(f"[zeroshot] 提取 {len(sequences):,} 条 embedding...")
    emb = embed(model, tokenizer, sequences, CONFIG["zeroshot"]["embed_batch_size"])
    np.save(cache, emb)
    print(f"[zeroshot] 缓存至 {cache} {emb.shape}")
    return emb


def main() -> int:
    from sklearn.linear_model import Ridge
    from transformers import AutoModel, AutoTokenizer

    cfg = CONFIG["zeroshot"]
    local_dir = Path("data/models/dnabert2")
    if (local_dir / "pytorch_model.bin").exists():
        model_name = str(local_dir)
        print(f"[zeroshot] 使用本地模型: {model_name}（离线加载，零运行时网络依赖）")
    else:
        model_name = cfg["model"]
        print(f"[zeroshot] 本地模型不存在，回退在线加载: {model_name}")
    print(f"[zeroshot] 加载 {model_name} (device={DEVICE})")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(
        DEVICE
    )  # fp32：117M 仅 ~470MB 显存；且标准注意力回退路径混 dtype 会炸（Half vs Float）

    splits = {s: load_split(s) for s in ("train", "valid", "test")}
    cache_dir = Path("data/processed/embeddings")
    embs = {
        "train": get_embeddings(
            model,
            tokenizer,
            "train",
            splits["train"]["sequence"].tolist(),
            cache_dir,
            sample=cfg["train_probe_sample"],
            seed=CONFIG["seed"],
        ),
        "valid": get_embeddings(
            model, tokenizer, "valid", splits["valid"]["sequence"].tolist(), cache_dir
        ),
        "test": get_embeddings(
            model, tokenizer, "test", splits["test"]["sequence"].tolist(), cache_dir
        ),
    }

    # train 侧的 y 需与抽样后的行对齐：重新按同一 seed 抽样
    rng = np.random.default_rng(CONFIG["seed"])
    n_train = len(splits["train"])
    idx = np.sort(
        rng.choice(n_train, size=min(cfg["train_probe_sample"], n_train), replace=False)
    )
    y_train = splits["train"].loc[idx, list(TARGETS)].to_numpy(dtype=np.float32)

    # 探针超参在 valid 上选（不碰 test）
    rows: list[dict] = []
    for alpha in cfg["probe_alphas"]:
        model_probe = Ridge(alpha=alpha)
        model_probe.fit(embs["train"], y_train)
        vp = model_probe.predict(embs["valid"])
        vs = np.mean(
            [
                evaluate(splits["valid"][t].to_numpy(), vp[:, i])["spearman"]
                for i, t in enumerate(TARGETS)
            ]
        )
        print(f"[zeroshot] alpha={alpha} valid平均spearman={vs:.4f}")

    best_alpha = min(
        cfg["probe_alphas"],
        key=lambda a: -np.mean(
            [
                evaluate(
                    splits["valid"][t].to_numpy(),
                    Ridge(alpha=a)
                    .fit(embs["train"], y_train)
                    .predict(embs["valid"])[:, i],
                )["spearman"]
                for i, t in enumerate(TARGETS)
            ]
        ),
    )
    print(f"[zeroshot] 最优 alpha={best_alpha}（valid 上选，未碰 test）")
    final = Ridge(alpha=best_alpha).fit(embs["train"], y_train)

    for split, df in (("valid", splits["valid"]), ("test", splits["test"])):
        preds = final.predict(embs[split])
        for ti, task in enumerate(TARGETS):
            m = evaluate(df[task].to_numpy(), preds[:, ti])
            rows.append(
                {
                    "model": "zeroshot_ridge",
                    "task": task,
                    "split": split,
                    "seed": CONFIG["seed"],
                    **m,
                    "notes": f"DNABERT-2均值池化+Ridge(alpha={best_alpha}); 不微调; train探针抽样{cfg['train_probe_sample']}; {cfg['note']}",
                }
            )
            print(f"[zeroshot] {task} {split}: {m}")
    append_rows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
