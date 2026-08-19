"""双击运行：执行 LQR QR 权重矩阵自动寻优脚本。

Double-click runner: run the automatic LQR QR optimizer.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from wheel_leg_lqr.optimize_qr import main  # noqa: E402


if __name__ == "__main__":
    main()
    try:
        input("Press Enter to close...")
    except EOFError:
        pass
