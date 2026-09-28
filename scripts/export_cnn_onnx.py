"""M5：CNN 基线 → ONNX（浏览器 demo 用的可运行模型）。

为什么浏览器端用 CNN 而不是 LoRA DNABERT-2（如实记录的工程取舍，devlog M5）：
- LoRA 模型已成功导出 ONNX 且数值一致（fp32，max|Δ|≈7e-6，见 results/onnx_parity.json），
  但体积 481MB（int8 量化被导出图/量化器的形状冲突卡住，fp16 在 CPU/WASM 上无法加载）；
- CNN 是纯卷积（~6MB），onnxruntime-web 的 WASM 后端可以真正跑起来，且**无需 JS 侧分词**
  （输入是 4×249 one-hot，浏览器里三行代码即可构造）。
- CNN 在本项目基准里是 Dev 侧最优（0.639），并非"随便挑的替身"。

用法: uv run --group ml python scripts/export_cnn_onnx.py [--seed 42]
产出: webapp/public/cnn.onnx（含 int8 尝试结果）+ 更新 results/onnx_parity.json 的 cnn 字段
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from enhancerscope.data import TARGETS, load_split, one_hot  # noqa: E402
from enhancerscope.metrics import evaluate  # noqa: E402

OUT = ROOT / "webapp" / "public" / "cnn.onnx"
PARITY = ROOT / "results" / "onnx_parity.json"


class OneHotWrapper(torch.nn.Module):
    """把 (B,4,249) one-hot 输入直接映射到 (B,2) 输出——浏览器只需构造 one-hot。"""

    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from baseline_cnn import (
        CONFIG,
        DeepSTARRCNN,
        train_one_seed,
    )  # 复用 M2 的网络与训练循环

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, valid, test = load_split("train"), load_split("valid"), load_split("test")
    tensors = {
        "x_train": torch.from_numpy(one_hot(train["sequence"])).to(device),
        "x_valid": torch.from_numpy(one_hot(valid["sequence"])).to(device),
        "x_test": torch.from_numpy(one_hot(test["sequence"])).to(device),
        "y_train": torch.tensor(
            train[list(TARGETS)].to_numpy(), dtype=torch.float32, device=device
        ),
        "y_valid": torch.tensor(
            valid[list(TARGETS)].to_numpy(), dtype=torch.float32, device=device
        ),
    }
    print(f"[cnn-onnx] 训练 seed={args.seed}（与 M2 相同配置）...")
    t0 = time.time()
    net = DeepSTARRCNN().to(device)
    # 复用 M2 的训练函数：内部会保存 data/models/finetuned/cnn_seed{N}.pt
    train_one_seed(args.seed, CONFIG["cnn"], tensors)
    ckpt = ROOT / "data" / "models" / "finetuned" / f"cnn_seed{args.seed}.pt"
    net.load_state_dict(
        torch.load(ckpt, weights_only=False, map_location=device)["state_dict"]
    )
    net.eval()
    print(f"[cnn-onnx] 训练完成 {time.time() - t0:.0f}s；权重 {ckpt.name}")

    net = net.to("cpu").eval()  # ONNX 导出要求模型与 dummy 同设备，一律 CPU 导出
    wrapper = OneHotWrapper(net).eval()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, 4, 249), dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(OUT),
        input_names=["onehot"],
        output_names=["activity"],
        dynamic_axes={"onehot": {0: "batch"}, "activity": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    # 合并 external data 为单文件（浏览器只需抓一个文件）
    import onnx as _onnx

    _m = _onnx.load(str(OUT), load_external_data=True)
    _onnx.save_model(_m, str(OUT), save_as_external_data=False)
    data_file = OUT.with_suffix(".onnx.data")
    if data_file.exists():
        data_file.unlink()
    print(f"[cnn-onnx] 导出 {OUT} ({OUT.stat().st_size / 1e6:.2f}MB)")

    # 一致性闸门 + 延迟实测（含 test 指标复核）
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    idx = rng.choice(len(test), 60, replace=False)
    seqs = test["sequence"].iloc[idx].tolist()
    x = torch.from_numpy(
        one_hot(seqs)
    ).float()  # 模型输入为 float32（训练时按 batch 转换）
    with torch.no_grad():
        y_torch = wrapper(x).numpy()
    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    y_onnx = sess.run(None, {"onehot": x.numpy()})[0]
    from scipy import stats

    parity = {
        "max_abs_diff": float(np.abs(y_torch - y_onnx).max()),
        "pearson_Dev": float(stats.pearsonr(y_torch[:, 0], y_onnx[:, 0]).statistic),
        "pearson_Hk": float(stats.pearsonr(y_torch[:, 1], y_onnx[:, 1]).statistic),
    }
    ok = (
        parity["max_abs_diff"] < 1e-3
        and min(parity["pearson_Dev"], parity["pearson_Hk"]) > 0.999
    )
    print(f"[cnn-onnx] 一致性: {parity} → {'通过 ✓' if ok else '未通过 ✗'}")

    # 单序列 CPU 延迟（ONNX Runtime，作为浏览器 WASM 的下限参考）
    single = x[:1].numpy()
    for _ in range(3):
        sess.run(None, {"onehot": single})
    t1 = time.time()
    for _ in range(20):
        sess.run(None, {"onehot": single})
    lat_ms = (time.time() - t1) / 20 * 1000
    print(f"[cnn-onnx] 单序列 CPU 延迟（ORT Python，WASM 上界参考）: {lat_ms:.1f}ms")

    # test 指标复核（导出模型 vs 训练脚本口径一致性）
    x_test = one_hot(test["sequence"]).astype(np.float32)
    preds = np.concatenate(
        [
            sess.run(None, {"onehot": x_test[i : i + 512]})[0]
            for i in range(0, len(x_test), 512)
        ]
    )
    metrics = {
        t: evaluate(test[t].to_numpy(), preds[:, i]) for i, t in enumerate(TARGETS)
    }
    print(f"[cnn-onnx] ONNX 模型 test 指标: {json.dumps(metrics, ensure_ascii=False)}")

    result = json.loads(PARITY.read_text(encoding="utf-8")) if PARITY.exists() else {}
    result["cnn"] = {
        "parity": parity,
        "gate_pass": ok,
        "size_mb": OUT.stat().st_size / 1e6,
        "single_latency_ms_cpu": round(lat_ms, 1),
        "test_metrics_onnx": metrics,
    }
    PARITY.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
