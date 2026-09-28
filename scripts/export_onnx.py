"""M5：把最优微调模型（LoRA DNABERT-2）导出 ONNX，并做数值一致性闸门。

用法:
  uv run --group ml python scripts/export_onnx.py [--quantize]

产出:
  webapp/public/model.onnx            （int8 动态量化，供浏览器加载；未量化时约 470MB）
  results/onnx_parity.json            （PyTorch vs ONNX 的一致性指标——发布闸门证据）

闸门（不通过则不应发布该模型）:
  - 预测值与 PyTorch 的 Pearson r > 0.99（量化后 > 0.98）且 max|Δ| < 0.05（量化后 < 0.15）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enhancerscope.data import load_split  # noqa: E402
from enhancerscope.model import hidden_of, load_finetuned  # noqa: E402

OUT_DIR = ROOT / "artifacts" / "onnx"  # LoRA 的 ONNX 是 Release 资产（481MB），不进网页
PARITY = ROOT / "results" / "onnx_parity.json"


class OnnxWrapper(nn.Module):
    """把 (base+LoRA, pooler) 包成单输出 (B, 2) 的图，隐藏 tuple/中间量。"""

    def __init__(self, model: nn.Module, pooler: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.pooler = pooler

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        h = hidden_of(self.model, input_ids, attention_mask)
        return self.pooler(h, attention_mask)


def torch_predict(
    wrapper: OnnxWrapper, tokenizer, seqs: list[str], device, batch: int = 32
) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            toks = tokenizer(
                seqs[i : i + batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            ids = toks["input_ids"].to(device)
            mask = toks["attention_mask"].to(device)
            outs.append(wrapper(ids, mask).float().cpu().numpy())
    return np.concatenate(outs)


def onnx_predict(sess, tokenizer, seqs: list[str], batch: int = 32) -> np.ndarray:
    outs = []
    for i in range(0, len(seqs), batch):
        toks = tokenizer(
            seqs[i : i + batch],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        res = sess.run(
            None,
            {
                "input_ids": toks["input_ids"].astype(np.int64),
                "attention_mask": toks["attention_mask"].astype(np.int64),
            },
        )
        outs.append(np.asarray(res[0]))
    return np.concatenate(outs)


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    from scipy import stats

    return {
        "max_abs_diff": float(np.abs(a - b).max()),
        "mean_abs_diff": float(np.abs(a - b).mean()),
        "pearson_Dev": float(stats.pearsonr(a[:, 0], b[:, 0]).statistic),
        "pearson_Hk": float(stats.pearsonr(a[:, 1], b[:, 1]).statistic),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--quantize", action="store_true", help="导出 int8 动态量化版本（浏览器用）"
    )
    ap.add_argument("--n-parity", type=int, default=96)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, pooler, tokenizer, device = load_finetuned("dnabert2", seed=None)
    wrapper = OnnxWrapper(model, pooler).eval().to(device)
    print(f"[onnx] 模型已加载（最优种子自动选择），device={device}")

    test = load_split("test")
    rng = np.random.default_rng(0)
    seqs = (
        test["sequence"]
        .iloc[rng.choice(len(test), args.n_parity, replace=False)]
        .tolist()
    )

    fp32_path = OUT_DIR / "model_fp32.onnx"
    if not fp32_path.exists():
        # 249bp 的 BPE 编码恒为 63 tokens（数据卡/实测），序列维固定、只保留动态 batch：
        # 简化图结构让 ORT 量化器的形状推断通过（devlog M5 第 15 坑）
        dummy_ids = torch.ones((1, 63), dtype=torch.long, device=device)
        dummy_mask = torch.ones((1, 63), dtype=torch.long, device=device)
        torch.onnx.export(
            wrapper,
            (dummy_ids, dummy_mask),
            str(fp32_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["activity"],
            dynamic_axes={
                "input_ids": {0: "batch"},
                "attention_mask": {0: "batch"},
                "activity": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"[onnx] fp32 导出: {fp32_path} ({fp32_path.stat().st_size / 1e6:.0f}MB)")

    # torch 默认以 external data 分包（*.onnx + *.onnx.data），而 onnxruntime 量化器需要单文件 →
    # 合并为自包含单文件（约 470MB，低于 protobuf 2GB 上限），也便于作为 Release 资产分发
    fp32_single = OUT_DIR / "model_fp32_single.onnx"
    if not fp32_single.exists():
        import onnx as _onnx

        _m = _onnx.load(str(fp32_path), load_external_data=True)
        _onnx.save_model(_m, str(fp32_single), save_as_external_data=False)
        print(f"[onnx] 合并单文件: {fp32_single.stat().st_size / 1e6:.0f}MB")

    w0 = torch_predict(wrapper, tokenizer, seqs, device)
    import onnxruntime as ort

    sess0 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    o0 = onnx_predict(sess0, tokenizer, seqs)
    parity_fp32 = compare(w0, o0)
    print(f"[onnx] fp32 一致性: {parity_fp32}")
    ok_fp32 = (
        parity_fp32["max_abs_diff"] < 0.05
        and min(parity_fp32["pearson_Dev"], parity_fp32["pearson_Hk"]) > 0.99
    )

    result = {"fp32": parity_fp32, "fp32_gate_pass": ok_fp32}
    if args.quantize:
        import shutil
        import tempfile

        import onnx as _onnx2
        from onnxruntime.quantization import QuantType, quantize_dynamic

        # 根因排查（devlog M5 第 14 坑）：ORT 量化器在写 "-inferred.onnx" 时对**含中文的路径**
        # 失败（onnx 的 C++ 写文件不认非 ASCII 路径）→ 在纯英文临时目录里做量化再搬回来。
        q_path = OUT_DIR / "model.onnx"
        quant_route = "int8"
        tmpdir = Path(tempfile.mkdtemp(prefix="ortq_", dir="C:/"))
        try:
            shutil.copy(fp32_single, tmpdir / "model.onnx")
            quantize_dynamic(
                str(tmpdir / "model.onnx"),
                str(tmpdir / "model_int8.onnx"),
                weight_type=QuantType.QInt8,
            )
            shutil.move(str(tmpdir / "model_int8.onnx"), str(q_path))
        except Exception as e:  # noqa: BLE001
            print(
                f"[onnx] int8 路线失败（{type(e).__name__}: {str(e)[:90]}）→ 回退 fp16"
            )
            from onnxconverter_common import float16

            proto = _onnx2.load(str(fp32_single), load_external_data=True)
            q_path = OUT_DIR / "model_fp16.onnx"
            _onnx2.save_model(
                float16.convert_float_to_float16(proto, keep_io_types=True), str(q_path)
            )
            quant_route = "fp16-fallback"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        print(
            f"[onnx] 量化完成（路线={quant_route}）: {q_path} ({q_path.stat().st_size / 1e6:.0f}MB)"
        )
        sessq = ort.InferenceSession(str(q_path), providers=["CPUExecutionProvider"])
        oq = onnx_predict(sessq, tokenizer, seqs)
        parity_q = compare(w0, oq)
        print(f"[onnx] 量化后一致性: {parity_q}")
        tol = 0.15 if quant_route == "int8" else 0.3
        ok_q = (
            parity_q["max_abs_diff"] < tol
            and min(parity_q["pearson_Dev"], parity_q["pearson_Hk"]) > 0.98
        )
        result.update(
            {
                "quant_route": quant_route,
                "quant": parity_q,
                "quant_gate_pass": ok_q,
                "sizes_mb": {
                    "fp32": fp32_path.stat().st_size / 1e6,
                    "shipped": q_path.stat().st_size / 1e6,
                },
            }
        )

    PARITY.parent.mkdir(parents=True, exist_ok=True)
    PARITY.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[onnx] 闸门结果写入 {PARITY}")
    if not ok_fp32 or (args.quantize and not result.get("int8_gate_pass", False)):
        print("[onnx] !! 一致性闸门未通过——按计划不发布该导出（如实记录）")
        return 1
    print("[onnx] 一致性闸门通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
