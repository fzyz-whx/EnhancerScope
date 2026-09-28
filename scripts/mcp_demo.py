"""M6 演示：模拟一个 agent 收到"帮我分析这 5 条增强子序列"后的完整工具调用流程。

流程（对应 DoD 的"agent 收到请求 → 自动调工具 → 产出汇总报告"）：
  1) batch_scan   —— 一次调用拿到 5 条序列的预测与排名
  2) explain_sequence —— 对排名第一的序列做逐碱基突变归因
  3) 汇总报告     —— 打印为 markdown，可直接贴进对话

用法: uv run python scripts/mcp_demo.py
产出: docs/assets/mcp_demo.txt（演示记录入库）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enhancerscope.data import load_split  # noqa: E402
from enhancerscope.mcp_server import (  # noqa: E402
    batch_scan_impl,
    explain_sequence_impl,
)

OUT = ROOT / "docs" / "assets" / "mcp_demo.txt"


def main() -> int:
    test = load_split("test")
    rng = __import__("numpy").random.default_rng(7)
    idx = rng.choice(len(test), 5, replace=False)
    records = [
        (
            test.iloc[i]["id"],
            test.iloc[i]["sequence"],
            test.iloc[i]["Dev_log2_enrichment"],
            test.iloc[i]["Hk_log2_enrichment"],
        )
        for i in idx
    ]
    fasta = "".join(f">{rid}\n{seq}\n" for rid, seq, _, _ in records)

    log: list[str] = []
    log.append("# MCP 端到端演示记录（scripts/mcp_demo.py）\n")
    log.append(
        '模拟请求：*"帮我分析这 5 条增强子序列，看看哪条最可能是强增强子，并解释第一名为什么强。"*\n'
    )
    log.append("## ① agent 调用工具 `batch_scan`\n")
    log.append("入参：FASTA（5 条 test 序列，真实数据）\n")

    scan = batch_scan_impl(fasta, top_k=3)
    log.append("```json")
    log.append(
        json.dumps(
            {k: v for k, v in scan.items() if k != "per_sequence"},
            ensure_ascii=False,
            indent=2,
        )
    )
    log.append("```\n")

    top_id = scan["top_hk"][0]
    top_seq = next(seq for rid, seq, _, _ in records if rid == top_id)
    log.append(f"## ② agent 对排名第一的 `{top_id}` 调用 `explain_sequence`\n")
    expl = explain_sequence_impl(top_seq, top_k=5)
    log.append("```json")
    log.append(
        json.dumps(
            {k: v for k, v in expl.items() if k != "preprocessing"},
            ensure_ascii=False,
            indent=2,
        )
    )
    log.append("```\n")

    log.append("## ③ agent 汇总报告\n")
    log.append("| 序列 | Dev 预测 | Hk 预测 | 真实 Dev | 真实 Hk | 判定 |")
    log.append("|---|---|---|---|---|---|")
    truth = {rid: (d, h) for rid, _, d, h in records}
    for row in scan["per_sequence"]:
        d, h = truth[row["id"]]
        log.append(
            f"| {row['id']} | {row['dev_log2']:.2f} | {row['hk_log2']:.2f} | {d:.2f} | {h:.2f} | {row['verdict']} |"
        )
    log.append("")
    log.append(
        f"**结论**：`{top_id}` 的 Hk 预测最高（{scan['per_sequence'][[r['id'] for r in scan['per_sequence']].index(top_id)]['hk_log2']:.2f}）；"
        f"归因显示最敏感的位点是 {expl['top_sensitive_positions']['hk'][:3]}。"
        "（预测值与真实标签逐条对照——这批序列模型预测方向正确的不在少数，但也存在过度预测，方法与局限见 docs/interpretability.md。）"
    )

    text = "\n".join(log)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n演示记录已写入 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
