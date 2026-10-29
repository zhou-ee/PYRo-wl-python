"""Compute the independent fixed-length ten-state ground model and LQR."""

import argparse
from pathlib import Path

import numpy as np

from ..artifacts.ground10_lqr import save_node, save_result
from ..artifacts.io import default_output_dir
from ..config import DEFAULT_ROBOT
from ..scheduling.ground10_lqr import compute_gain_node, compute_gain_schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("node", "schedule"), default="schedule")
    nominal = (DEFAULT_ROBOT.min_leg_length + DEFAULT_ROBOT.max_leg_length) / 2
    parser.add_argument("--l1", type=float, default=nominal)
    parser.add_argument("--l2", type=float, default=nominal)
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument(
        "--output-dir", type=Path, default=default_output_dir("ground10_lqr")
    )
    args = parser.parse_args()
    if args.mode == "node":
        node = compute_gain_node(args.l1, args.l2)
        save_node(node, args.output_dir)
        print(
            f"A={node.linearization.A.shape}, B={node.linearization.B.shape}, K={node.gain.shape}"
        )
        print(
            f"controllability_rank={node.controllability_rank}, radius={np.max(np.abs(node.closed_loop_poles)):.9f}"
        )
    else:
        result = compute_gain_schedule(step=args.step, degree=args.degree)
        save_result(result, args.output_dir)
        print(
            f"training_nodes={len(result.nodes)}, holdout_nodes={result.metrics['holdout_node_count']}"
        )
        print(f"max_float32_radius={result.metrics['max_float32_spectral_radius']:.9f}")
        print(f"saturated_recovery={result.metrics['saturation_passed']}")
    print(f"output_dir={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
