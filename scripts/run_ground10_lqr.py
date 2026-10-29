"""Run the independent ten-state model from an uninstalled checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wheel_leg_lqr.cli.ground10_lqr import main

if __name__ == "__main__":
    main()
