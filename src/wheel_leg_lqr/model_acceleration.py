"""Generate the forward model-acceleration schedule with total degree <= 3."""

from __future__ import annotations

import argparse

from .scheduling.model_acceleration_schedule import (
    compute_model_acceleration_schedule, export_model_acceleration_schedule,
    extract_model_acceleration_coefficients, predict_model_acceleration,
)

__all__ = [
    "compute_model_acceleration_schedule", "export_model_acceleration_schedule",
    "extract_model_acceleration_coefficients", "predict_model_acceleration", "main",
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=float, default=0.01, help="training grid step in m (default 0.01)")
    parser.add_argument("--random-holdouts", type=int, default=64, help="random pairs, each also mirrored")
    parser.add_argument("--max-error", type=float, default=0.02, help="hard per-channel peak-normalized max error")
    parser.add_argument("--rms-target", type=float, default=0.005, help="advisory peak-normalized RMS target")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    print("Extracting ten coefficients; fitting ten total-degree cubic terms per channel.", flush=True)
    schedule = compute_model_acceleration_schedule(
        step=args.step, random_holdouts=args.random_holdouts,
        max_error=args.max_error, rms_target=args.rms_target,
    )
    output = export_model_acceleration_schedule(schedule, args.output_dir)
    metrics = schedule.diagnostics["float32_all"]
    maximum = max(item["max_peak_normalized"] for item in metrics.values())
    print(f"Generated {output}; 100 float32 coefficients (400 bytes); max error {100*maximum:.4f}%.")
    missed = schedule.diagnostics["rms_target_exceeded_channels"]
    if missed:
        print(f"RMS advisory target {100*args.rms_target:g}% exceeded: {', '.join(missed)}; see report.")


if __name__ == "__main__":
    main()
