"""M6：MCP 工具层测试（CI 上真跑模型，onnxruntime 已在主依赖里）。

覆盖：入参校验（非法字符/空/长度处理）、预测回归值、解释输出结构、FASTA 解析（文本/文件/异常）、
批量扫描汇总字段、以及 FastMCP 工具注册与入参 schema（DoD：三个工具测试全过 + schema 校验）。
"""

from __future__ import annotations

import asyncio

import pytest

from enhancerscope.mcp_server import (
    batch_scan_impl,
    build_server,
    explain_sequence_impl,
    normalize_sequence,
    parse_fasta,
    predict_activity_impl,
)

SEQ_264 = "AGCTTAGCTAGCTTGGCATCGATCGATCGATCG" * 8  # 264bp → 截断到 249


# ---------------------------------------------------------------- 入参校验


def test_normalize_pads_short_sequence() -> None:
    seq, info = normalize_sequence("ACGT" * 10)
    assert len(seq) == 249
    assert info["padded_to"] == 249
    assert seq.endswith("N")


def test_normalize_truncates_long_sequence() -> None:
    seq, info = normalize_sequence(SEQ_264)
    assert len(seq) == 249
    assert info["truncated_to"] == 249


def test_normalize_case_insensitive_and_strips_whitespace() -> None:
    seq, _ = normalize_sequence("acgt acgt\nACGT")
    assert seq.startswith("ACGTACGTACGT")


def test_normalize_rejects_invalid_chars() -> None:
    with pytest.raises(ValueError, match="只接受 A/C/G/T"):
        normalize_sequence("ACGTX")


def test_normalize_rejects_empty() -> None:
    with pytest.raises(ValueError, match="序列为空"):
        normalize_sequence("   \n ")


def test_explain_rejects_bad_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        explain_sequence_impl("ACGT" * 10, top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        explain_sequence_impl("ACGT" * 10, top_k=999)


# ---------------------------------------------------------------- 预测与解释


def test_predict_activity_matches_reference() -> None:
    """回归测试：数值必须与首次记录一致（模型与代码变更会被捕捉）。"""
    r = predict_activity_impl(SEQ_264)
    assert r["dev_log2"] == pytest.approx(-0.8812, abs=1e-3)
    assert r["hk_log2"] == pytest.approx(-0.4890, abs=1e-3)
    assert r["preprocessing"]["truncated_to"] == 249


def test_explain_sequence_structure_and_ranking() -> None:
    r = explain_sequence_impl(SEQ_264, top_k=5)
    for task in ("dev", "hk"):
        top = r["top_sensitive_positions"][task]
        assert len(top) == 5
        deltas = [t["abs_delta"] for t in top]
        assert deltas == sorted(deltas, reverse=True)  # 按影响降序
        assert all(0 <= t["pos"] < 249 for t in top)


def test_explain_reference_top_dev() -> None:
    r = explain_sequence_impl(SEQ_264, top_k=3)
    top = r["top_sensitive_positions"]["dev"]
    assert [t["pos"] for t in top] == [23, 27, 28]
    assert top[0]["abs_delta"] == pytest.approx(0.5131, abs=1e-3)


# ---------------------------------------------------------------- FASTA 与批量扫描


def test_parse_fasta_text_and_ids() -> None:
    text = ">s1\nACGT\n>s2\nTTTT\nGGGG\n"
    recs = parse_fasta(text)
    assert [r["id"] for r in recs] == ["s1", "s2"]
    assert recs[1]["sequence"] == "TTTTGGGG"


def test_parse_fasta_from_file(tmp_path) -> None:
    p = tmp_path / "x.fa"
    p.write_text(">a\nACGTACGT\n", encoding="utf-8")
    recs = parse_fasta(str(p))
    assert recs[0]["id"] == "a"


def test_parse_fasta_rejects_empty() -> None:
    with pytest.raises(ValueError, match="未解析到任何序列"):
        parse_fasta("\n\n")


def test_batch_scan_summary_fields() -> None:
    text = ">low\n" + "ACGT" * 60 + "\n>also\n" + "TTTT" * 60 + "\n"
    r = batch_scan_impl(text, top_k=2)
    assert r["n_sequences"] == 2
    assert {row["id"] for row in r["per_sequence"]} == {"low", "also"}
    assert len(r["top_dev"]) == 2 and len(r["top_hk"]) == 2


# ---------------------------------------------------------------- MCP 注册与 schema


def test_server_registers_three_tools_with_schemas() -> None:
    """三个工具都注册且带 JSON schema 入参校验（DoD：工具测试 + 入参 schema 校验）。"""
    server = build_server()
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    assert set(tools) == {"predict_activity", "explain_sequence", "batch_scan"}
    for name, tool in tools.items():
        schema = tool.parameters
        assert schema.get("type") == "object", f"{name} 缺 object schema"
        assert schema.get("properties"), f"{name} 缺入参定义"
    assert set(tools["explain_sequence"].parameters["properties"]) == {
        "sequence",
        "top_k",
    }
    assert set(tools["batch_scan"].parameters["properties"]) == {"fasta", "top_k"}
    assert tools["batch_scan"].parameters.get("required") == ["fasta"]
