"""模型权重下载脚本（goal 规则：数据/权重不入库，用下载脚本 + 数据卡管理）。

从 huggingface.co 官方源经代理下载 DNABERT-2-117M 全部文件到 data/models/dnabert2/，
每个文件独立重试 + curl -C - 断点续传（代理节点不稳，EOF 率高，见 devlog M1/M2）。
下载完成后 baseline_zeroshot.py 优先从本地目录离线加载，运行时零网络依赖。

用法: uv run --group ml python scripts/download_model.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = "zhihan1996/DNABERT-2-117M"
FILES = [
    "config.json",
    "configuration_bert.py",
    "bert_layers.py",
    "bert_padding.py",
    "flash_attn_triton.py",
    "generation_config.json",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "LICENSE",
]
OUT = Path("data/models/dnabert2")
BASE = f"https://huggingface.co/{REPO}/resolve/main"
PROXY = "http://127.0.0.1:7897"
ATTEMPTS = 20


def fetch(name: str) -> bool:
    dest = OUT / name
    if dest.exists() and dest.stat().st_size > 0 and name != "pytorch_model.bin":
        return True  # 小文件信任存在性；大文件靠 -C - 续传校验
    url = f"{BASE}/{name}"
    for attempt in range(1, ATTEMPTS + 1):
        cmd = [
            "curl",
            "-sL",
            "--fail",
            "-C",
            "-",
            "--max-time",
            "600",
            "--retry",
            "3",
            "-x",
            PROXY,
            url,
            "-o",
            str(dest),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            size = dest.stat().st_size / 1e6
            print(f"  ✓ {name} ({size:.1f}MB, 第{attempt}次尝试)")
            return True
        print(f"  ✗ {name} 第{attempt}次失败: {r.stderr.strip()[:80]}")
    return False


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    failed = [f for f in FILES if not fetch(f)]
    if failed:
        print(f"下载失败: {failed}")
        return 1
    print("全部文件下载完成 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
