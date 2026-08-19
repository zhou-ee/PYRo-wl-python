"""双击运行：符号结构能控性检查并转储矩阵产物。

Double-click runner: run the symbolic structural controllability check and
dump the nominal input matrix and the generic controllability matrix.

产物（nominal_input_matrix.txt、generic_controllability_matrix.txt）写入本副本
data/generated/。Artifacts are written to data/generated/ in this copy.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
_OUTPUT_DIR = _ROOT / "data" / "generated"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from wheel_leg_lqr.symbolic_controllability import main  # noqa: E402


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--output-dir", str(_OUTPUT_DIR)]
    main()
    print(f"artifacts written to: {_OUTPUT_DIR}")
    try:
        input("Press Enter to close...")
    except EOFError:
        pass
