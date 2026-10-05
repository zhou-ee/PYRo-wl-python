from __future__ import annotations
from ..artifacts.io import default_output_dir
import argparse
from pathlib import Path
from ..models.airborne import config
from ..scheduling.airborne_lqr import compute_airborne_schedule
from ..artifacts.airborne_lqr import save_airborne_schedule
from ..artifacts.io import resolve_output_dir as _resolve_output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=float, default=0.01, help="L1/L2 grid spacing in metres")
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--sample-time", type=float, default=config.SAMPLE_TIME)
    parser.add_argument(
        "--q-diag", type=float, nargs=6,
        metavar=("QB1", "QDB1", "QB2", "QDB2", "QTHETA", "QDTHETA"),
        help="Six physical-state Q diagonal weights; defaults to airborne.config Q_DIAG",
    )
    parser.add_argument(
        "--r-diag", type=float, nargs=2, metavar=("RTP1", "RTP2"),
        help="Two hip-input R diagonal weights; defaults to airborne.config R_DIAG",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=default_output_dir("airborne_lqr")
    )
    args = parser.parse_args()
    result = compute_airborne_schedule(
        step=args.step, degree=args.degree, sample_time=args.sample_time,
        q_diag=args.q_diag, r_diag=args.r_diag,
    )
    output_dir = _resolve_output_dir(args.output_dir)
    save_airborne_schedule(result, output_dir)
    print(
        f"nodes={len(result.nodes)}, validation={result.validation_node_count}, "
        f"max_relative_gain_error={result.max_relative_gain_error:.6e}, "
        f"max_controllable_radius={result.max_float32_shape_radius:.9f}"
    )
    print(f"Airborne LQR artifacts written to: {output_dir}")

if __name__ == "__main__":
    main()
