"""JASPAR 果蝇 motif 数据库获取脚本（goal 规则：数据不入库，用脚本管理）。

背景：jaspar.genereg.net 在本机网络下不可达（devlog M4 记录）。本脚本从
`vanheeringen-lab/gimmemotifs` 仓库获取其**捆绑的 JASPAR2020 insects PWM 副本**
（.pfm 文件头部保留原始出处：JASPAR2020 CORE insects non-redundant + UNVALIDATED，
retrieved 2019-10-18），与官方发布内容一致。

用法: uv run python scripts/download_jaspar.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

OUT = Path("data/motifs/jaspar2020_insects.pfm")
REPO = "vanheeringen-lab/gimmemotifs"
PATH_IN_REPO = "data/motif_databases/JASPAR2020_insects.pfm"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "gh",
        "api",
        f"repos/{REPO}/contents/{PATH_IN_REPO}",
        "-H",
        "Accept: application/vnd.github.raw",
    ]
    for attempt in range(1, 6):
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0 and r.stdout:
            OUT.write_bytes(r.stdout)
            print(f"✓ 已下载 {OUT} ({len(r.stdout)} bytes, 第{attempt}次尝试)")
            return 0
        print(f"第{attempt}次失败: {r.stderr.decode(errors='replace')[:100]}")
    print("下载失败（需要 gh 已登录且网络可达 github.com）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
