"""Command-line entry point for the variable-leg-length gain fit."""

from __future__ import annotations

import argparse
from pathlib import Path

from .gain_schedule import fit_gain_schedule, save_gain_fit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit cubic K(L) coefficients and emit a C++ constexpr array."
    )
    parser.add_argument(
        "--header",
        type=Path,
        default=Path("artifacts/k_poly_coef.hpp"),
        help="output C++ header path",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/k_poly_fit.json"),
        help="output JSON validation report path",
    )
    args = parser.parse_args()

    result = fit_gain_schedule()
    save_gain_fit(result, header_path=args.header, report_path=args.report)

    print(f"saved header: {args.header.resolve()}")
    print(f"saved report: {args.report.resolve()}")
    print(
        f"L range: {result.leg_lengths[0]:.6f} ... "
        f"{result.leg_lengths[-1]:.6f} m, "
        f"samples={result.leg_lengths.size}"
    )
    print(f"K polynomial degree: {result.coefficients.shape[2] - 1}")
    print(f"max absolute fit error: {result.max_abs_error:.9e}")
    print(f"RMS fit error: {result.rms_error:.9e}")


if __name__ == "__main__":
    main()
