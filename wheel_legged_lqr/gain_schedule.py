"""Compute and fit a cubic gain schedule K(L)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial

from . import config
from .lqr import compute_lqr


@dataclass(frozen=True)
class GainFitResult:
    """Results of the variable-leg-length gain fitting process."""

    leg_lengths: np.ndarray
    gains: np.ndarray
    coefficients: np.ndarray
    fitted_gains: np.ndarray
    max_abs_error: float
    rms_error: float


def make_leg_length_grid(
    start: float = config.L_SWEEP_START,
    stop: float = config.L_SWEEP_STOP,
    step: float = config.L_SWEEP_STEP,
) -> np.ndarray:
    """Create an inclusive, evenly spaced leg-length grid in metres."""
    if step <= 0.0:
        raise ValueError("leg-length sweep step must be positive")
    if stop < start:
        raise ValueError("leg-length sweep stop must not be smaller than start")

    count = int(round((stop - start) / step)) + 1
    values = start + step * np.arange(count, dtype=float)
    values[-1] = stop
    return values


def fit_gain_schedule(
    *,
    leg_lengths: np.ndarray | None = None,
    degree: int = config.K_POLY_DEGREE,
) -> GainFitResult:
    """Compute K at each L and fit each K entry with a polynomial in L.

    Coefficients are returned in ascending power order:
    c[0] + c[1]*L + c[2]*L**2 + c[3]*L**3.
    """
    if degree < 0:
        raise ValueError("polynomial degree must be non-negative")

    lengths = make_leg_length_grid() if leg_lengths is None else np.asarray(leg_lengths, dtype=float)
    if lengths.ndim != 1 or lengths.size < degree + 1:
        raise ValueError("the leg-length grid must contain at least degree + 1 samples")

    state = config.make_working_state()
    inputs = config.make_working_input()
    q = config.make_q()
    r = config.make_r()

    gains = []
    for length in lengths:
        parameters = config.make_parameters_for_leg_length(float(length))
        result = compute_lqr(
            sample_time=config.Ts,
            parameters=parameters,
            state=state,
            inputs=inputs,
            q=q,
            r=r,
        )
        gains.append(result.lqr.K)

    gain_values = np.asarray(gains, dtype=float)  # [n_L, 2, 6]
    coefficients = np.empty((2, 6, degree + 1), dtype=float)
    fitted = np.empty_like(gain_values)

    for input_index in range(2):
        for state_index in range(6):
            # Polynomial.fit is numerically better conditioned on metre-scale L.
            polynomial = Polynomial.fit(
                lengths,
                gain_values[:, input_index, state_index],
                degree,
            ).convert()
            coefficients[input_index, state_index, :] = polynomial.coef
            fitted[:, input_index, state_index] = polynomial(lengths)

    error = fitted - gain_values
    return GainFitResult(
        leg_lengths=lengths,
        gains=gain_values,
        coefficients=coefficients,
        fitted_gains=fitted,
        max_abs_error=float(np.max(np.abs(error))),
        rms_error=float(np.sqrt(np.mean(error**2))),
    )


def render_cpp_header(
    coefficients: np.ndarray,
    *,
    degree: int,
    variable_name: str = "K_POLY_COEF",
) -> str:
    """Render coefficients as a C++ constexpr float array."""
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (2, 6, degree + 1):
        raise ValueError(
            f"expected coefficient shape (2, 6, {degree + 1}), "
            f"got {coefficients.shape}"
        )

    lines = [
        "#pragma once",
        "",
        "constexpr int K_POLY_DEGREE = %d;" % degree,
        "constexpr float %s[2][6][K_POLY_DEGREE + 1] = {" % variable_name,
    ]
    for input_index in range(2):
        lines.append("    {")
        for state_index in range(6):
            values = ", ".join(
                f"{float(value):.10e}f"
                for value in coefficients[input_index, state_index]
            )
            comma = "," if state_index < 5 else ""
            lines.append(f"        {{{values}}}{comma}")
        comma = "," if input_index == 0 else ""
        lines.append(f"    }}{comma}")
    lines.append("};")
    lines.append("")
    return "\n".join(lines)


def save_gain_fit(
    result: GainFitResult,
    *,
    header_path: Path,
    report_path: Path,
) -> None:
    """Save the C++ header and a JSON validation report."""
    header_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    header_path.write_text(
        render_cpp_header(
            result.coefficients,
            degree=result.coefficients.shape[2] - 1,
        ),
        encoding="utf-8",
    )

    report = {
        "polynomial_degree": int(result.coefficients.shape[2] - 1),
        "coefficient_order": "constant, L, L^2, L^3",
        "leg_length_start_m": float(result.leg_lengths[0]),
        "leg_length_stop_m": float(result.leg_lengths[-1]),
        "sample_count": int(result.leg_lengths.size),
        "max_abs_fit_error": result.max_abs_error,
        "rms_fit_error": result.rms_error,
        "Q_diag": config.Q_diag,
        "R_diag": config.R_diag,
        "coefficients": result.coefficients.tolist(),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


