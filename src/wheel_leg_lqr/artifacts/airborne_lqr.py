from __future__ import annotations
from .metadata import artifact_metadata
import json
from pathlib import Path
import numpy as np
from ..models.airborne.model import AIRBORNE_INPUT_NAMES, AIRBORNE_STATE_NAMES
from ..scheduling.airborne_lqr import AirborneScheduleFit
from .io import resolve_output_dir as _resolve_output_dir, write_text as _write_text


def render_airborne_header(result: AirborneScheduleFit) -> str:
    """Render a self-contained C++ coefficient header, separate from ground LQR."""
    coefficients = result.gain_coefficients.astype(np.float32)
    degree = coefficients.shape[2] - 1
    center = (result.parameters.min_leg_length + result.parameters.max_leg_length) / 2
    scale = (result.parameters.max_leg_length - result.parameters.min_leg_length) / 2
    lines = [
        "#ifndef WHEEL_LEG_AIRBORNE_LQR_COEF_H",
        "#define WHEEL_LEG_AIRBORNE_LQR_COEF_H",
        "#include <cstdint>",
        "",
        "// Contact-free two-hip LQR. State error is in radians and radians/s.",
        "// u = -K(L1,L2) * (x - x_ref); clamp each T_p to its actuator limit.",
        "// Two full-plant poles remain at 1: free-flight angular momentum is conserved.",
        f"// Q diag [beta1, dbeta1, beta2, dbeta2, theta, dtheta]: {result.q_diag.tolist()}",
        f"// R diag [Tp1, Tp2]: {result.r_diag.tolist()}",
        "namespace airborne_lqr {",
        "constexpr std::uint32_t STATE_DIM = 6;",
        "constexpr std::uint32_t INPUT_DIM = 2;",
        f"constexpr std::uint32_t K_POLY_DEGREE = {degree};",
        f"constexpr float LENGTH_CENTER_M = {center:.9g}f;",
        f"constexpr float LENGTH_HALF_RANGE_M = {scale:.9g}f;",
        "// lambda_i = clamp((Li - LENGTH_CENTER_M) / LENGTH_HALF_RANGE_M, -1, 1).",
        "// K[i][j] = sum_p sum_q K_COEF[i][j][p][q] * lambda_1^p * lambda_2^q.",
        "enum State : std::uint8_t { BETA_1, DOT_BETA_1, BETA_2, DOT_BETA_2, THETA, DOT_THETA };",
        "enum Input : std::uint8_t { T_P1, T_P2 };",
        "constexpr float K_COEF[INPUT_DIM][STATE_DIM]",
        "    [K_POLY_DEGREE + 1][K_POLY_DEGREE + 1] = {",
    ]
    for input_index in range(2):
        lines.append("    {")
        for state_index in range(6):
            lines.append("        {")
            for p in range(degree + 1):
                row = ", ".join(
                    f"{float(value):.9e}f" for value in coefficients[input_index, state_index, p]
                )
                lines.append(f"            {{{row}}},")
            lines.append("        },")
        lines.append("    },")
    lines.extend(["};", "}  // namespace airborne_lqr", "", "#endif", ""])
    return "\n".join(lines)


def render_airborne_summary(result: AirborneScheduleFit) -> str:
    return "\n".join(
        [
            "# Airborne two-hip LQR gain schedule",
            "",
            f"- Grid: {len(result.l1_grid)} x {len(result.l2_grid)} = {len(result.nodes)} nodes",
            f"- Sample time: {result.sample_time:.6g} s",
            "- State order: `beta_1, dbeta_1, beta_2, dbeta_2, theta, dtheta`",
            "- Input order: `T_p1, T_p2`",
            f"- Q diagonal: `{result.q_diag.tolist()}`",
            f"- R diagonal: `{result.r_diag.tolist()}`",
            f"- Maximum node absolute fit error: {result.max_abs_gain_error:.6e}",
            f"- Maximum validation relative gain error: {result.max_relative_gain_error:.6e}",
            f"- Maximum raw controllable pole radius: {result.max_raw_shape_radius:.9f}",
            f"- Maximum float32 fitted controllable pole radius: {result.max_float32_shape_radius:.9f}",
            f"- Float32 validation points: {result.validation_node_count} (grid and cell centers)",
            "- The two global-rotation poles remain at 1; all three absolute angles cannot be independently held in flight.",
            "",
            "See `docs/06_airborne_lqr.md` for the plant derivation and use constraints.",
        ]
    )


def save_airborne_schedule(result: AirborneScheduleFit, output_dir: Path) -> None:
    """Write the C++ header, NumPy arrays, and validation report."""
    output_dir = _resolve_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_text(output_dir / "airborne_lqr_coef.h", render_airborne_header(result))
    _write_text(output_dir / "README.md", render_airborne_summary(result))
    np.savez_compressed(
        output_dir / "airborne_lqr_schedule.npz",
        l1_grid=result.l1_grid,
        l2_grid=result.l2_grid,
        state_names=np.array(AIRBORNE_STATE_NAMES),
        input_names=np.array(AIRBORNE_INPUT_NAMES),
        gains=np.stack([node.gain for node in result.nodes]),
        fitted_gains=result.fitted_gains,
        gain_coefficients=result.gain_coefficients,
        q_diag=result.q_diag,
        r_diag=result.r_diag,
        sample_time=np.array(result.sample_time),
    )
    report = {
        **artifact_metadata(result.parameters, sample_time=result.sample_time, q_diag=result.q_diag.tolist(), r_diag=result.r_diag.tolist()),
        "state_names": AIRBORNE_STATE_NAMES,
        "input_names": AIRBORNE_INPUT_NAMES,
        "length_domain_m": [
            result.parameters.min_leg_length,
            result.parameters.max_leg_length,
        ],
        "sample_time_s": result.sample_time,
        "q_diag": result.q_diag.tolist(),
        "r_diag": result.r_diag.tolist(),
        "polynomial_degree_per_axis": result.gain_coefficients.shape[2] - 1,
        "grid_nodes": len(result.nodes),
        "validation_nodes": result.validation_node_count,
        "max_abs_node_gain_fit_error": result.max_abs_gain_error,
        "max_relative_float32_validation_gain_error": result.max_relative_gain_error,
        "max_raw_controllable_spectral_radius": result.max_raw_shape_radius,
        "max_float32_controllable_spectral_radius": result.max_float32_shape_radius,
        "neutral_global_rotation_poles": 2,
        "body_mass_kg": result.parameters.body_mass,
    }
    _write_text(output_dir / "airborne_lqr_fit.json", json.dumps(report, indent=2))
