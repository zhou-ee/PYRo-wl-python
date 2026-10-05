from __future__ import annotations
from ..config import DEFAULT_ROBOT
from ..models.ground.trim import compute_equilibrium_beta
from ..artifacts.io import default_output_dir
import argparse
from pathlib import Path
import numpy as np
from ..models.ground import config
from ..scheduling.rdob_node import DEFAULT_BANDWIDTH_HZ, compute_rdob_node
from ..scheduling.rdob import compute_rdob_schedule
from ..artifacts.rdob_node import export_rdob
from ..artifacts.rdob import export_rdob_schedule


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("schedule", "node"), default="schedule",
                        help="default: independent L1/L2 schedule; node: single trim")
    parser.add_argument("--l1", type=float, default=None, help="single-node mode only")
    parser.add_argument("--l2", type=float, default=None, help="single-node mode only")
    parser.add_argument("--min-length", type=float, default=DEFAULT_ROBOT.min_leg_length)
    parser.add_argument("--max-length", type=float, default=DEFAULT_ROBOT.max_leg_length)
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--fit-tolerance", type=float, default=0.005,
                        help="maximum per-matrix relative Frobenius error; default 0.005 = 0.5%%")
    parser.add_argument("--channel-tolerance", type=float, default=0.01,
                        help="maximum matched input-map relative Frobenius error; default 0.01 = 1%%")
    parser.add_argument("--sample-time", type=float, default=config.SAMPLE_TIME)
    parser.add_argument("--bandwidth-hz", type=float, nargs=6, default=DEFAULT_BANDWIDTH_HZ,
                        metavar="HZ", help="Tw1 Tw2 Tp1 Tp2 FL1 FL2; defaults produce the nominal RDOB step pole")
    parser.add_argument("--damping-diagonal", type=float, nargs=7, default=None,
                        metavar="D", help="optional diagonal D0 in generalized-coordinate order")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir("rdob"))
    args = parser.parse_args(argv)
    if args.mode == "schedule" and (args.l1 is not None or args.l2 is not None):
        parser.error("--l1/--l2 require --mode node; scheduling uses independent --min-length/--max-length bounds")
    try:
        common = dict(bandwidth_hz=np.asarray(args.bandwidth_hz), sample_time=args.sample_time,
                      damping=None if args.damping_diagonal is None else np.diag(args.damping_diagonal))
        if args.mode == "node":
            node = compute_rdob_node(
                (config.TRIM_HEIGHT / np.cos(compute_equilibrium_beta(config.TRIM_HEIGHT))) if args.l1 is None else args.l1,
                (config.TRIM_HEIGHT / np.cos(compute_equilibrium_beta(config.TRIM_HEIGHT))) if args.l2 is None else args.l2, **common)
            output = export_rdob(node, args.output_dir)
        else:
            schedule = compute_rdob_schedule(
                lower=args.min_length, upper=args.max_length, step=args.step,
                relative_tolerance=args.fit_tolerance,
                matched_rate_tolerance=args.channel_tolerance, **common)
            output = export_rdob_schedule(schedule, args.output_dir)
    except ValueError as error:
        parser.error(str(error))
    print(f"RDOB artifacts written to: {output}")

if __name__ == "__main__":
    main()
