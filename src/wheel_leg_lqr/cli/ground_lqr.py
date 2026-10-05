from __future__ import annotations
from ..artifacts.io import default_output_dir
import argparse
from pathlib import Path
from ..artifacts.io import resolve_output_dir
from ..scheduling.ground_lqr import compute_gain_schedule
from ..artifacts.ground_lqr import save_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=float, default=0.01, help="Grid step in meters (default: 0.01).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir("ground_lqr"),
        help="Directory for generated L1/L2 artifacts (default: outputs/ground_lqr).",
    )
    parser.add_argument(
        "--exhaustive-validation",
        action="store_true",
        help="Run the slow 5-second saturation simulation at every grid node.",
    )
    args = parser.parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    result = compute_gain_schedule(
        step=args.step,
        exhaustive_validation=args.exhaustive_validation,
    )
    save_result(result, output_dir)
    print(
        f"nodes={len(result.nodes)}, "
        f"max_fit_error={result.max_abs_gain_error:.6e}, "
        f"rms_fit_error={result.rms_gain_error:.6e}"
    )
    print(
        "max spectral radius: "
        f"raw={result.max_raw_spectral_radius:.9f}, "
        f"fitted={result.max_fitted_spectral_radius:.9f}"
    )
    print(
        "saturation validation: "
        f"{result.saturation_validation_node_count}/{len(result.nodes)} nodes"
    )

if __name__ == "__main__":
    main()
