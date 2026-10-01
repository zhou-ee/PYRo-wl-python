"""Generate contact-free two-hip LQR artifacts from an uninstalled checkout."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from wheel_leg_lqr.airborne_lqr import main  # noqa: E402


if __name__ == "__main__":
    main()
