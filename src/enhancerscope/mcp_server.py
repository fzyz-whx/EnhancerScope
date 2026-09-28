"""M6：EnhancerScope 的 MCP 工具层（FastMCP）。

给任意 LLM agent 用的三个工具：
- `predict_activity(sequence)`   单序列活性预测（Dev/Hk）
- `explain_sequence(sequence, top_k)`  逐碱基突变敏感度（与 M4 归因方法同源）
- `batch_scan(fasta, top_k)`      批量扫描（FASTA 文本或文件路径）→ 汇总报告

模型：`webapp/public/cnn.onnx`（0.81MB，已入库；与 PyTorch 数值一致 max|Δ|<1e-6，
证据见 results/onnx_parity.json）→ 克隆仓库即可用，无需下载权重。

启动（一条命令，stdio 传输）：
    uv run enhancerscope-mcp
接入任意 MCP client（Claude Desktop / Cline / ZCode 等）见 README 的 5 分钟指南。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from fastmcp import FastMCP

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "webapp" / "public" / "cnn.onnx"
SEQ_LEN = 249
BASES = "ACGT"

SESSION: ort.InferenceSession | None = None


# ---------------------------------------------------------------- 核心实现（纯函数，便于测试）


def normalize_sequence(raw: str) -> tuple[str, dict[str, Any]]:
    """清洗序列：大写、去空白；<249 补 N、>249 截断。返回 (序列, 处理说明)。"""
    seq = "".join(raw.split()).upper()
    if not seq:
        raise ValueError("序列为空")
    bad = sorted({c for c in seq if c not in BASES})
    if bad:
        raise ValueError(f"只接受 A/C/G/T，发现非法字符: {''.join(bad)}")
    info: dict[str, Any] = {"input_length": len(seq)}
    if len(seq) < SEQ_LEN:
        seq = seq + "N" * (SEQ_LEN - len(seq))
        info["padded_to"] = SEQ_LEN
    elif len(seq) > SEQ_LEN:
        seq = seq[:SEQ_LEN]
        info["truncated_to"] = SEQ_LEN
    return seq, info


def get_session(model_path: Path | str = DEFAULT_MODEL) -> ort.InferenceSession:
    global SESSION
    if SESSION is None:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"模型不存在: {model_path}（先跑 scripts/export_cnn_onnx.py）"
            )
        SESSION = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
    return SESSION


def one_hot(seqs: list[str]) -> np.ndarray:
    """(N,4,249) float32 one-hot（非 ACGT → 全零列）。"""
    out = np.zeros((len(seqs), 4, SEQ_LEN), dtype=np.float32)
    idx = {b: i for i, b in enumerate(BASES)}
    for n, seq in enumerate(seqs):
        for p, ch in enumerate(seq):
            if ch in idx:
                out[n, idx[ch], p] = 1.0
    return out


def run_model(seqs: list[str], model_path: Path | str = DEFAULT_MODEL) -> np.ndarray:
    """批量前向 → (N,2) [Dev, Hk]。"""
    sess = get_session(model_path)
    out = sess.run(None, {"onehot": one_hot(seqs)})[0]
    return np.asarray(out, dtype=np.float64)


def predict_activity_impl(
    sequence: str, model_path: Path | str = DEFAULT_MODEL
) -> dict[str, Any]:
    seq, info = normalize_sequence(sequence)
    pred = run_model([seq], model_path)[0]
    return {
        "dev_log2": round(float(pred[0]), 4),
        "hk_log2": round(float(pred[1]), 4),
        "model": "DeepSTARR-style CNN (ONNX, 0.81MB)",
        "preprocessing": info,
    }


def explain_sequence_impl(
    sequence: str, top_k: int = 10, model_path: Path | str = DEFAULT_MODEL
) -> dict[str, Any]:
    """逐碱基饱和突变：每位替换为其余 3 种碱基，取 |Δ预测| 最大值（与 M4 同源）。"""
    if top_k < 1 or top_k > SEQ_LEN:
        raise ValueError(f"top_k 必须在 1..{SEQ_LEN}")
    seq, info = normalize_sequence(sequence)
    base = run_model([seq], model_path)[0]
    variants: list[str] = []
    owners: list[int] = []
    for p, ref in enumerate(seq):
        for b in BASES:
            if b != ref:
                variants.append(seq[:p] + b + seq[p + 1 :])
                owners.append(p)
    preds = run_model(variants, model_path)
    dev = np.zeros(SEQ_LEN)
    hk = np.zeros(SEQ_LEN)
    for i, p in enumerate(owners):
        dev[p] = max(dev[p], abs(preds[i, 0] - base[0]))
        hk[p] = max(hk[p], abs(preds[i, 1] - base[1]))
    top_dev = np.argsort(dev)[-top_k:][::-1]
    top_hk = np.argsort(hk)[-top_k:][::-1]
    return {
        "dev_log2": round(float(base[0]), 4),
        "hk_log2": round(float(base[1]), 4),
        "top_sensitive_positions": {
            "dev": [
                {"pos": int(p), "base": seq[p], "abs_delta": round(float(dev[p]), 4)}
                for p in top_dev
            ],
            "hk": [
                {"pos": int(p), "base": seq[p], "abs_delta": round(float(hk[p]), 4)}
                for p in top_hk
            ],
        },
        "method": "single-base saturation mutagenesis (747 variants), same as docs/interpretability.md",
        "preprocessing": info,
    }


def parse_fasta(text_or_path: str) -> list[dict[str, str]]:
    """FASTA 文本或文件路径 → [{"id","sequence"}]。

    路径探测要防御：把整段 FASTA 文本当路径传给 Path.exists() 在 Linux 上会抛
    OSError(ENAMETOOLONG)——CI 抓到的跨平台 bug（devlog M6 第 21 坑）。
    """
    text = text_or_path
    if chr(10) not in text_or_path and len(text_or_path) < 4096:
        try:
            p = Path(text_or_path)
            if p.is_file():
                text = p.read_text(encoding="utf-8")
        except OSError:
            pass
    records: list[dict[str, str]] = []
    cur_id: str | None = None
    chunks: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur_id is not None:
                records.append({"id": cur_id, "sequence": "".join(chunks)})
            cur_id, chunks = line[1:].strip() or f"seq{len(records) + 1}", []
        elif cur_id is None:
            cur_id, chunks = f"seq{len(records) + 1}", [line]
        else:
            chunks.append(line)
    if cur_id is not None:
        records.append({"id": cur_id, "sequence": "".join(chunks)})
    if not records:
        raise ValueError("未解析到任何序列（需要 FASTA 文本或文件路径）")
    return records


def batch_scan_impl(
    fasta: str, top_k: int = 3, model_path: Path | str = DEFAULT_MODEL
) -> dict[str, Any]:
    records = parse_fasta(fasta)
    seqs = []
    for r in records:
        seq, _ = normalize_sequence(r["sequence"])
        seqs.append(seq)
    preds = run_model(seqs, model_path)
    rows = []
    for r, pred in zip(records, preds):
        rows.append(
            {
                "id": r["id"],
                "dev_log2": round(float(pred[0]), 4),
                "hk_log2": round(float(pred[1]), 4),
                "verdict": (
                    "强增强子候选"
                    if max(pred) > 3
                    else ("弱/无活性" if max(pred) < 0 else "中等")
                ),
            }
        )
    order_dev = sorted(range(len(rows)), key=lambda i: -rows[i]["dev_log2"])
    order_hk = sorted(range(len(rows)), key=lambda i: -rows[i]["hk_log2"])
    return {
        "n_sequences": len(rows),
        "per_sequence": rows,
        "top_dev": [
            rows[i]["id"] for i in order_dev[: max(1, top_k if top_k > 0 else 3)]
        ],
        "top_hk": [
            rows[i]["id"] for i in order_hk[: max(1, top_k if top_k > 0 else 3)]
        ],
        "model": "DeepSTARR-style CNN (ONNX, 0.81MB)",
    }


# ---------------------------------------------------------------- FastMCP 注册


def build_server() -> FastMCP:
    mcp = FastMCP(
        "EnhancerScope",
        instructions=(
            "增强子活性预测工具集（果蝇 S2 细胞 STARR-seq 数据训练的 CNN）。"
            "输入为 DNA 序列（249bp，A/C/G/T）；返回发育型(Dev)与管家型(Hk)增强子活性的 log2 富集预测。"
            "做碱基级解释用 explain_sequence；批量分析用 batch_scan。"
        ),
    )

    @mcp.tool
    def predict_activity(sequence: str) -> dict:
        """预测单条 DNA 序列的增强子活性（Dev/Hk, log2 富集）。序列应为 249bp 的 A/C/G/T。"""
        return predict_activity_impl(sequence)

    @mcp.tool
    def explain_sequence(sequence: str, top_k: int = 10) -> dict:
        """逐碱基突变敏感度归因：返回影响最大的 top_k 个位点（单碱基饱和突变 |Δ预测|）。"""
        return explain_sequence_impl(sequence, top_k=top_k)

    @mcp.tool
    def batch_scan(fasta: str, top_k: int = 3) -> dict:
        """批量扫描多条序列：fasta 可以是 FASTA 文本或文件路径；返回逐条预测与排名汇总。"""
        return batch_scan_impl(fasta, top_k=top_k)

    return mcp


def main() -> None:
    """入口：stdio 传输（MCP client 通过标准输入输出与 agent 通信）。"""
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
