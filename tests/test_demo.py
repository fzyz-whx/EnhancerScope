"""M0 冒烟测试：包元数据与 demo CLI。"""

from enhancerscope import __version__
from enhancerscope.demo import main


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_demo_exit_zero(capsys) -> None:
    assert main() == 0
    out = capsys.readouterr().out
    assert "EnhancerScope" in out
    assert __version__ in out
