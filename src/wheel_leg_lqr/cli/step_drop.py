from __future__ import annotations
import argparse
from ..simulation.step_drop import simulate_step_drop
from ..models.contact.model import _WHEEL_1, _WHEEL_2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-height", type=float, default=0.18)
    parser.add_argument(
        "--down-leg", type=int, choices=(_WHEEL_1, _WHEEL_2), default=_WHEEL_1
    )
    parser.add_argument("--horizon", type=float, default=3.0)
    args = parser.parse_args()
    result = simulate_step_drop(
        step_height=args.step_height,
        down_leg=args.down_leg,
        horizon_s=args.horizon,
    )
    print(
        f"target leg lengths: {result.target_lengths[0]:.6f}, "
        f"{result.target_lengths[1]:.6f} m"
    )
    print(f"impact time: {result.impact_time!r} s")
    print(f"impact wheel vertical speed: {result.impact_speed!r} m/s")
    print(
        "lower-contact normal impulse through termination: "
        f"{result.impact_impulse!r} N s"
    )
    print(f"peak contact force: {result.peak_contact_force:.6g} N")
    print(
        "mechanical-stop impulses: "
        f"{result.mechanical_stop_impulse[0]:.6g}, "
        f"{result.mechanical_stop_impulse[1]:.6g} N s"
    )
    print(f"peak mechanical-stop force: {result.peak_mechanical_stop_force:.6g} N")
    print(f"input saturation samples: {result.saturation_steps}")
    print(f"settling time: {result.settling_time!r} s")
    print(f"converged: {result.converged}")
    print(f"terminated time: {result.terminated_time:.6f} s")
    print(f"failure reason: {result.failure_reason!r}")

if __name__ == "__main__":
    main()
