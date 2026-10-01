"""Inspect the instantaneous free-flight pitch response to two axial leg forces."""

from __future__ import annotations

import argparse

import numpy as np

from . import config
from .model import compute_leg_force_pitch_effect


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l1", type=float, default=0.28, help="Left leg length, m")
    parser.add_argument("--l2", type=float, default=0.28, help="Right leg length, m")
    parser.add_argument(
        "--body-com-x", type=float, default=0.03,
        help="Body COM ahead of hip centre, m (default: 0.03)",
    )
    parser.add_argument("--beta1-deg", type=float, default=0.0)
    parser.add_argument("--beta2-deg", type=float, default=0.0)
    parser.add_argument("--theta-deg", type=float, default=0.0)
    parser.add_argument("--fl1", type=float, default=100.0, help="Positive extension force, N")
    parser.add_argument("--fl2", type=float, default=100.0, help="Positive extension force, N")
    args = parser.parse_args()
    angles = dict(
        beta_1=np.deg2rad(args.beta1_deg),
        beta_2=np.deg2rad(args.beta2_deg),
        theta=np.deg2rad(args.theta_deg),
    )
    effect = compute_leg_force_pitch_effect(
        args.l1, args.l2, body_com_x_from_hip=args.body_com_x, **angles
    )
    centered = compute_leg_force_pitch_effect(
        args.l1, args.l2, body_com_x_from_hip=0.0, **angles
    )
    forces = np.array([args.fl1, args.fl2], dtype=float)
    if not np.all(np.isfinite(forces)):
        parser.error("FL values must be finite")
    print(f"body COM x from hip: {args.body_com_x:.6g} m")
    print(f"body Iyy: {config.PARAMETERS.body_pitch_inertia:.9g} kg m^2")
    print(f"body moment arms [FL1, FL2]: {effect.body_moment_arm.tolist()} m")
    print(f"body-only pitch moment: {effect.body_moment_arm @ forces:.9g} N m")
    print(
        "body-only pitch acceleration: "
        f"{effect.body_pitch_acceleration_per_newton @ forces:.9g} rad/s^2"
    )
    print(
        "free-flight pitch acceleration per newton [FL1, FL2]: "
        f"{effect.pitch_acceleration_per_newton.tolist()} rad/s^2/N"
    )
    print(
        "free-flight pitch acceleration at requested forces: "
        f"{effect.pitch_acceleration_per_newton @ forces:.9g} rad/s^2"
    )
    print(
        "change relative to body COM x=0: "
        f"{(effect.pitch_acceleration_per_newton - centered.pitch_acceleration_per_newton) @ forces:.9g} rad/s^2"
    )
    print("Instantaneous zero-velocity result; leg travel and force duration are not integrated.")


if __name__ == "__main__":
    main()
