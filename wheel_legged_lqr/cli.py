# 命令行入口 / Command-line entry point

from __future__ import annotations

# This fallback allows both `python -m ...` and direct execution of this file.
#  cli.py
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from wheel_legged_lqr import config
    from wheel_legged_lqr.config import ConfigurationError
    from wheel_legged_lqr.lqr import compute_lqr
else:
    from . import config
    from .config import ConfigurationError
    from .lqr import compute_lqr

import argparse
import json
from pathlib import Path

import numpy as np

def main() -> None:
    """Run one fixed-geometry calculation / """
    parser = argparse.ArgumentParser(
        description="Fixed document-coordinate dynamics and LQR /  LQR"
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/lqr_fixed.npz"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--beta", type=float, default=None, help="override config beta / override beta")
    parser.add_argument("--gamma", type=float, default=None, help="override config gamma / override gamma")
    parser.add_argument("--ts", type=float, default=None, help="override config Ts /  config ")
    args = parser.parse_args()

    try:
        result = compute_lqr(beta=args.beta, gamma=args.gamma, sample_time=args.ts)
    except ConfigurationError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Save matrices and metadata
    np.savez_compressed(
        args.output,
        leg_length=result.parameters.leg_length,
        l_wp=result.parameters.l_wp,
        l_ph=result.parameters.l_ph,
        sample_time=result.sample_time,
        state=result.state, inputs=result.inputs,
        Q=result.q, R=result.r,
        A=result.A, B=result.B, Ad=result.Ad, Bd=result.Bd,
        P=result.lqr.P, K=result.lqr.K,
        closed_loop_poles=result.lqr.closed_loop_eigenvalues,
        controllability_rank=result.controllability_rank,
    )

    report = {
        "leg_length": result.parameters.leg_length,
        "l_wp": result.parameters.l_wp,
        "l_ph": result.parameters.l_ph,
        "beta": float(result.state[2]),
        "sample_time": result.sample_time,
        "controllability_rank": result.controllability_rank,
        "closed_loop_pole_radius": float(np.max(np.abs(result.lqr.closed_loop_eigenvalues))),
    }
    report_path = args.report or args.output.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"saved: {args.output.resolve()}")
    print(f"geometry: L={result.parameters.leg_length:.6f} m, L_wp={result.parameters.l_wp:.6f} m, L_ph={result.parameters.l_ph:.6f} m")
    print(f"A shape={result.A.shape}, B shape={result.B.shape}, K shape={result.lqr.K.shape}")
    print("K =")
    print(result.lqr.K)
    print(f"controllability rank={result.controllability_rank}")
    print(f"closed-loop pole radius={np.max(np.abs(result.lqr.closed_loop_eigenvalues)):.9f}")

if __name__ == "__main__":
    main()
