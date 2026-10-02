"""Generate the forward model-acceleration schedule without installing the package."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from wheel_leg_lqr.model_acceleration import main  # noqa: E402


if __name__ == "__main__":
    main()
