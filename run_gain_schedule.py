"""双击运行：计算并写出唯一的 K(L1, L2) 张量积增益调度产物。

Double-click runner: compute and emit the project's L1/L2 gain-schedule artifacts.
Artifacts are written to data/generated/ in this copy.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
_OUTPUT_DIR = _ROOT / "data" / "generated"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from wheel_leg_lqr.gain_schedule import main  # noqa: E402


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *sys.argv[1:], "--output-dir", str(_OUTPUT_DIR)]
    main()
    print(f"L1/L2 artifacts written to: {_OUTPUT_DIR}")
    try:
        input("Press Enter to close...")
    except EOFError:
        pass
