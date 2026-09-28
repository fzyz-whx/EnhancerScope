"""环境自检 demo：打印项目与运行环境信息（M0 DoD：uv sync 后一条命令跑通）。"""

from __future__ import annotations

import platform
import sys

from enhancerscope import __version__


def main() -> int:
    """打印环境信息，返回退出码 0。"""
    print("EnhancerScope 环境 demo")
    print(f"  包版本 : {__version__}")
    print(f"  Python : {sys.version.split()[0]}")
    print(f"  平台   : {platform.platform()}")
    try:
        import torch
    except ImportError:
        print("  torch  : 未安装（M0 预期如此；M3 微调时装 cu128 版）")
    else:
        print(
            f"  torch  : {torch.__version__} (CUDA 可用: {torch.cuda.is_available()})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
