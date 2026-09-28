"""M2 baseline ②: DeepSTARR 风格轻量 CNN（PyTorch，GPU，3 种子报均值±标准差）。

用法: uv run --group ml python scripts/baseline_cnn.py
产出: results/baseline.csv（追加 model=cnn 行，每 seed 每任务每 split 一行）

防泄漏契约：只在 train split 训练，valid（chr2R 左半）早停，test（chr2R 右半）
仅在选好最优权重后评估一次。
显存策略（8GB 约束）：全量 one-hot uint8 驻留 GPU（402k×4×249 ≈ 400MB），
训练时按 batch 转 float32——避免 CPU-GPU 拷贝成为瓶颈。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enhancerscope.data import TARGETS, load_split, one_hot  # noqa: E402
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


class DeepSTARRCNN(nn.Module):
    """DeepSTARR 风格 4 卷积块 + 2 全连接，输出 2 维（Dev/Hk 双任务）。"""

    def __init__(self) -> None:
        super().__init__()

        def block(cin: int, cout: int, k: int) -> list[nn.Module]:
            return [nn.Conv1d(cin, cout, k, padding="same"), nn.ReLU(), nn.MaxPool1d(2)]

        self.encoder = nn.Sequential(
            *block(4, 64, 7),
            *block(64, 120, 5),
            *block(120, 120, 5),
            *block(120, 60, 5),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(60 * (249 // 16), 60),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(60, 2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encoder(x))


@torch.no_grad()
def predict(model: DeepSTARRCNN, x: Tensor, batch: int = 4096) -> np.ndarray:
    out = []
    for i in range(0, x.shape[0], batch):
        out.append(model(x[i : i + batch].float()).cpu().numpy())
    return np.concatenate(out)


def train_one_seed(
    seed: int,
    cfg: dict,
    tensors: dict[str, Tensor],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """训练单种子，返回 (valid 预测, test 预测)，形状 (N, 2)。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DeepSTARRCNN().to(DEVICE)
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    loss_fn = nn.MSELoss()

    x_train, y_train = tensors["x_train"], tensors["y_train"]
    x_valid, y_valid = tensors["x_valid"], tensors["y_valid"]
    n = x_train.shape[0]
    best_loss, best_state, bad = float("inf"), None, 0
    for epoch in range(cfg["max_epochs"]):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        total = 0.0
        for i in range(0, n, cfg["batch_size"]):
            idx = perm[i : i + cfg["batch_size"]]
            xb = x_train[idx].float()
            yb = y_train[idx]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vl = 0.0
            for i in range(0, x_valid.shape[0], cfg["batch_size"] * 4):
                xb = x_valid[i : i + cfg["batch_size"] * 4].float()
                vl += loss_fn(
                    model(xb), y_valid[i : i + cfg["batch_size"] * 4]
                ).item() * len(xb)
            valid_loss = vl / x_valid.shape[0]
        sched.step(valid_loss)
        if valid_loss < best_loss - 1e-5:
            best_loss, bad = valid_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(
                    f"  seed={seed} epoch={epoch} 早停 (best valid MSE {best_loss:.4f})"
                )
                break
        if epoch % 5 == 0:
            print(
                f"  seed={seed} epoch={epoch} train={total / n:.4f} valid={valid_loss:.4f}"
            )

    model.load_state_dict(best_state)
    model.eval()
    # 保存最优权重：M5 浏览器 demo 用（CNN 是纯卷积，ONNX/WASM 友好；devlog M5 记录了取舍）
    ckpt_dir = Path(__file__).resolve().parents[1] / "data" / "models" / "finetuned"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": best_state, "seed": seed, "best_valid_mse": best_loss},
        ckpt_dir / f"cnn_seed{seed}.pt",
    )
    preds_valid = predict(model, tensors["x_valid"])
    preds_test = predict(model, tensors["x_test"])
    return preds_valid, preds_test


def main() -> int:
    cfg = CONFIG["cnn"]
    print(f"[cnn] device={DEVICE} seeds={cfg['seeds']}")
    torch.backends.cudnn.benchmark = True

    train = load_split("train")
    valid = load_split("valid")
    test = load_split("test")
    tensors = {
        "x_train": torch.from_numpy(one_hot(train["sequence"])).to(DEVICE),
        "x_valid": torch.from_numpy(one_hot(valid["sequence"])).to(DEVICE),
        "x_test": torch.from_numpy(one_hot(test["sequence"])).to(DEVICE),
        "y_train": torch.tensor(
            train[list(TARGETS)].to_numpy(), dtype=torch.float32, device=DEVICE
        ),
        "y_valid": torch.tensor(
            valid[list(TARGETS)].to_numpy(), dtype=torch.float32, device=DEVICE
        ),
    }
    print(f"[cnn] 数据驻留 GPU: x_train {tuple(tensors['x_train'].shape)} uint8")

    rows: list[dict] = []
    for seed in cfg["seeds"]:
        preds_valid, preds_test = train_one_seed(seed, cfg, tensors)
        for split, preds, df in (
            ("valid", preds_valid, valid),
            ("test", preds_test, test),
        ):
            for ti, task in enumerate(TARGETS):
                m = evaluate(df[task].to_numpy(), preds[:, ti])
                rows.append(
                    {
                        "model": "cnn",
                        "task": task,
                        "split": split,
                        "seed": seed,
                        **m,
                        "notes": f"DeepSTARR风格4conv+2dense; {cfg['note']}",
                    }
                )
                print(f"[cnn] seed={seed} {task} {split}: {m}")
    append_rows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
