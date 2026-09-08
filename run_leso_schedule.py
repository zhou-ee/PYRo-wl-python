"""Double-click runner for the POD/Chebyshev six-disturbance LESO schedule."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
_OUTPUT_DIR = _ROOT / "data" / "generated" / "leso"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from wheel_leg_lqr.leso_schedule import main  # noqa: E402


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *sys.argv[1:], "--output-dir", str(_OUTPUT_DIR)]
    main()
    print(f"LESO artifacts written to: {_OUTPUT_DIR}")
    try:
        input("Press Enter to close...")
    except EOFError:
        pass
