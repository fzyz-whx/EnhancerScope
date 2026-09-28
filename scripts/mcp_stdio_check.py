"""M6 端到端冒烟：真的把 MCP server 拉起来，走 stdio JSON-RPC 握手 → tools/list → tools/call。

这比"函数级单测"更接近 DoD 的"一条命令启动 server + 三个工具可用"：
验证的是**协议层**（initialize / notifications/initialized / tools/list / tools/call）。

用法: uv run python scripts/mcp_stdio_check.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = (
    str(ROOT / ".venv" / "Scripts" / "python.exe")
    if (ROOT / ".venv").exists()
    else sys.executable
)
SEQ = "AGCTTAGCTAGCTTGGCATCGATCGATCGATCG" * 8


def rpc(proc: subprocess.Popen, payload: dict) -> dict:
    assert proc.stdin and proc.stdout
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("server 提前退出")
        msg = json.loads(line)
        if msg.get("id") == payload.get("id"):
            return msg


def main() -> int:
    proc = subprocess.Popen(
        [PY, "-m", "enhancerscope.mcp_server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        cwd=str(ROOT),
    )
    try:
        init = rpc(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "stdio-check", "version": "0"},
                },
            },
        )
        print("initialize →", init["result"]["serverInfo"])
        assert proc.stdin
        proc.stdin.write(
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        )
        proc.stdin.flush()

        tools = rpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = [t["name"] for t in tools["result"]["tools"]]
        print("tools/list →", names)
        assert set(names) == {"predict_activity", "explain_sequence", "batch_scan"}

        call = rpc(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "predict_activity", "arguments": {"sequence": SEQ}},
            },
        )
        content = json.loads(call["result"]["content"][0]["text"])
        print("tools/call predict_activity →", content)
        assert "dev_log2" in content and "hk_log2" in content
        print("\nstdio 端到端冒烟通过 ✓（initialize / tools/list / tools/call 全通）")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
