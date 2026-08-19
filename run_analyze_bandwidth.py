"""双击运行：分析 LQR 控制器在各个物理通道上的闭环带宽与模态频率。

Double-click runner: analyze controller bandwidth and modal frequencies across all physical loops.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from wheel_leg_lqr.analyze_bandwidth import main  # noqa: E402


if __name__ == "__main__":
    main()
    try:
        input("Press Enter to close...")
    except EOFError:
        pass
