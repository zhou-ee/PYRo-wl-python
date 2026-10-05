from __future__ import annotations
from ..models.ground.trim import fit_equilibrium_beta
from .metadata import artifact_metadata
from .io import resolve_output_dir
import json
from pathlib import Path
import numpy as np
from ..models.ground import config
from ..scheduling.ground_lqr import GainScheduleFit, _float32_fit_metrics
from .io import write_text


def render_active_fl_u0_block(coefficients: np.ndarray) -> str:
    """Render the active motor-side F_L equilibrium feedforward."""

    coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
    if coefficients.size != 4:
        raise ValueError("active F_L U0 must contain four cubic coefficients")
    fl_c0, fl_c1, fl_c2, fl_c3 = (
        f"{value:.10e}f" for value in coefficients
    )
    return "\n".join(
        [
            "// Active motor F_L feedforward after passive gas-spring support.",
            "// The plant model and K assume total virtual F_L; the supplied",
            "// gas-spring polynomial is already equivalent F_L.",
            "// f_l_active = f_l_total - F_gas(L)",
            "constexpr float FL_U0_POLY_COEF[U0_POLY_DEGREE + 1] = {",
            f"    {fl_c0}, {fl_c1}, {fl_c2},",
            f"    {fl_c3}}};",
        ]
    )


def _render_lqr_body(
    result: GainScheduleFit,
    *,
    include_extended_state_dim: bool,
    state_namespace: str,
    input_namespace: str,
    leg_state_name: str,
    dot_leg_state_name: str,
) -> str:
    """Render LQR declarations without an include guard or ``#include``."""

    coefficients = result.gain_coefficients
    input_coefficients = result.input_coefficients
    q_vals = list(result.nodes[0].settings.q_diag)
    r_vals = list(result.nodes[0].settings.r_diag)
    parameters = result.nodes[0].parameters
    center = (parameters.min_leg_length + parameters.max_leg_length) / 2
    half_range = (parameters.max_leg_length - parameters.min_leg_length) / 2

    lines = [
        "/*",
        ' *  "Q_diag":',
    ]
    for i in range(0, len(q_vals), 2):
        pair = ",".join(
            f"{v:.1f}" if float(v).is_integer() else f"{v}"
            for v in q_vals[i : i + 2]
        )
        prefix = "[" if i == 0 else ""
        suffix = "]" if i + 2 >= len(q_vals) else ","
        lines.append(f" *  {prefix}{pair}{suffix}")

    lines.append(' *  "R_diag":')
    for i in range(0, len(r_vals), 2):
        pair = ",".join(
            f"{v:.1f}" if float(v).is_integer() else f"{v}"
            for v in r_vals[i : i + 2]
        )
        prefix = "[" if i == 0 else ""
        suffix = "]" if i + 2 >= len(r_vals) else ","
        lines.append(f" *  {prefix}{pair}{suffix}")

    deg1 = coefficients.shape[2] - 1
    u0_deg = input_coefficients.shape[1] - 1

    dimension_lines = [
        f"{'constexpr uint32_t STATE_DIM':<68} = {len(config.STATE_NAMES)};",
        f"{'constexpr uint32_t INPUT_DIM':<68} = {len(config.INPUT_NAMES)};",
    ]
    if include_extended_state_dim:
        dimension_lines.append(
            f"{'constexpr uint32_t EXT_STATE_DIM':<68} = "
            f"{len(config.STATE_NAMES) + len(config.INPUT_NAMES)};"
        )
    dimension_lines.extend(
        [
            f"{'constexpr uint32_t K_POLY_DEGREE':<68} = {deg1};",
            f"{'constexpr uint32_t U0_POLY_DEGREE':<68} = {u0_deg};",
            f"{'constexpr uint32_t BETA_TRIM_POLY_DEGREE':<68} = {len(fit_equilibrium_beta(result.nodes[0].parameters)) - 1};",
        ]
    )
    lines.extend(
        [
            " */",
            "",
            *dimension_lines,
            "",
            "// L1/L2 tensor-product polynomial coefficients over normalized leg lengths.",
            f"// lambda_1 = clamp((L1 - {center:.8g}f) / {half_range:.8g}f, -1.0f, 1.0f)",
            f"// lambda_2 = clamp((L2 - {center:.8g}f) / {half_range:.8g}f, -1.0f, 1.0f)",
            "// K[input][state] = sum C[input][state][p][q] * lambda_1^p * lambda_2^q",
            "constexpr float K_POLY_COEF[INPUT_DIM][STATE_DIM]"
            "[K_POLY_DEGREE + 1][K_POLY_DEGREE + 1] = {",
        ]
    )

    for in_idx in range(len(config.INPUT_NAMES)):
        lines.append("    {")
        for st_idx in range(len(config.STATE_NAMES)):
            lines.append("        {")
            for p in range(deg1 + 1):
                row = coefficients[in_idx, st_idx, p]
                c0, c1, c2, c3 = (f"{v:.10e}f" for v in row)
                lines.append(f"            {{{c0}, {c1}, {c2},")
                lines.append(f"             {c3}}},")
            lines.append("        },")
        lines.append("    },")
    lines.append("};")
    lines.append("")

    lines.extend(
        [
            render_active_fl_u0_block(input_coefficients[4, :, 0]),
            "",
        ]
    )

    beta_c0, beta_c1, beta_c2, beta_c3 = (
        f"{v:.10e}f" for v in fit_equilibrium_beta(result.nodes[0].parameters)
    )
    lines.extend(
        [
            "constexpr float BETA_TRIM_POLY_COEF[BETA_TRIM_POLY_DEGREE + 1] = {",
            f"    {beta_c0}, {beta_c1}, {beta_c2},",
            f"    {beta_c3}}};",
            "",
            f"namespace {state_namespace}",
            "{",
            "enum : uint8_t",
            "{",
            "    X = 0,",
            "    DOT_X,",
            "    PSI,",
            "    DOT_PSI,",
            "    THETA,",
            "    DOT_THETA,",
            "    PHI,",
            "    DOT_PHI,",
            f"    {leg_state_name},",
            f"    {dot_leg_state_name},",
            "    BETA_1,",
            "    DOT_BETA_1,",
            "    BETA_2,",
            "    DOT_BETA_2,",
            "};",
            "}",
            "",
            f"namespace {input_namespace}",
            "{",
            "enum : uint8_t",
            "{",
            "    T_W1 = 0,",
            "    T_W2,",
            "    T_P1,",
            "    T_P2,",
            "    F_L1,",
            "    F_L2,",
            "};",
            "}",
        ]
    )
    return "\n".join(lines)


def render_firmware_lqr_body(result: GainScheduleFit) -> str:
    """Render the LQR section used by the combined firmware ``coef.h``."""

    return _render_lqr_body(
        result,
        include_extended_state_dim=True,
        state_namespace="state_def",
        input_namespace="input_def",
        leg_state_name="L",
        dot_leg_state_name="DOT_L",
    )


def render_header(result: GainScheduleFit) -> str:
    """生成与下位机完全兼容的 L1/L2 张量积增益调度 C/C++ 头文件。"""

    body = _render_lqr_body(
        result,
        include_extended_state_dim=False,
        state_namespace="lqr_state_def",
        input_namespace="lqr_input_def",
        leg_state_name="L_BAR",
        dot_leg_state_name="DOT_L_BAR",
    )
    return "\n".join(
        [
            "#ifndef __LQR_COEF_H__",
            "#define __LQR_COEF_H__",
            '#include "stdint.h"',
            "",
            body,
            "",
            "#endif",
            "",
        ]
    )


def render_summary(result: GainScheduleFit) -> str:
    """生成 L1/L2 增益调度综合 Markdown 报告。"""
    max_float_radius, max_float_gain_error = _float32_fit_metrics(result)
    lines = [
        "# L1/L2 Gain Schedule Summary: K(L1, L2)",
        "",
        f"- Domain: `L1 in [{result.l1_grid[0]:g}, {result.l1_grid[-1]:g}] m`, `L2 in [{result.l2_grid[0]:g}, {result.l2_grid[-1]:g}] m`",
        f"- Grid Resolution: `{len(result.l1_grid)} x {len(result.l2_grid)} = {len(result.nodes)}` nodes",
        "- Polynomial Model: Bivariate Tensor-Product `Degree (3, 3)` (16 coefficients per gain channel)",
        f"- Max Gain Fit Error: `{result.max_abs_gain_error:.6e}`",
        f"- RMS Gain Fit Error: `{result.rms_gain_error:.6e}`",
        f"- Max Raw Discrete Spectral Radius: `{result.max_raw_spectral_radius:.9f}`",
        f"- Max Fitted Discrete Spectral Radius: `{result.max_fitted_spectral_radius:.9f}`",
        f"- Max Float32 Discrete Spectral Radius: `{max_float_radius:.9f}`",
        f"- Max Float32 Gain Rounding Error: `{max_float_gain_error:.6e}`",
        "- Controlled generalized coordinate: `L_bar=(L1+L2)/2`, with derived height not included in the LQR state",
        "- Gas spring: `A/B/K` use total virtual `F_L`; exported `FL_U0` is active motor force (`F_total-F_gas`)",
        "- State order: `x, dx, psi, dpsi, theta, dtheta, phi, dphi, L_bar, dL_bar, beta1, dbeta1, beta2, dbeta2`",
        f"- Saturated recovery peak state error: `{result.saturation_validation.max_state_abs:.6e}`",
        f"- Saturated recovery final state error: `{result.saturation_validation.max_final_state_abs:.6e}`",
        f"- Saturated recovery steps: `{result.saturation_validation.saturation_steps}`",
        f"- Saturated recovery validation nodes: `{result.saturation_validation_node_count} / {len(result.nodes)}`",
        "",
        "## Closed-Loop Stability Verification",
        "",
        f"All {len(result.nodes)} nodes across the full L1/L2 operating envelope have closed-loop spectral radii strictly below 1.0 under float32 precision.",
        "",
    ]
    return "\n".join(lines)


def render_polynomial_matrices(result: GainScheduleFit) -> str:
    """生成 L1/L2 张量积系数矩阵可视化文档。"""
    parameters = result.nodes[0].parameters
    center = (parameters.min_leg_length + parameters.max_leg_length) / 2
    half_range = (parameters.max_leg_length - parameters.min_leg_length) / 2
    lines = [
        "# K(L1, L2) L1/L2 Tensor-Product Polynomial Coefficient Matrices",
        "",
        f"Evaluated via: `lambda_1 = clamp((L1 - {center:g})/{half_range:g}, -1, 1)`, `lambda_2 = clamp((L2 - {center:g})/{half_range:g}, -1, 1)`",
        "`K[i, j] = sum_{p=0}^3 sum_{q=0}^3 C[i, j, p, q] * lambda_1^p * lambda_2^q`",
        "",
    ]
    deg1 = result.gain_coefficients.shape[2] - 1
    for in_idx, in_name in enumerate(config.INPUT_NAMES):
        lines.append(f"## Input {in_name} (`u[{in_idx}]`)")
        lines.append("")
        for st_idx, st_name in enumerate(config.STATE_NAMES):
            lines.append(f"### `K[{in_name}, {st_name}]` (4x4 Matrix for `p, q in 0..3`)")
            lines.append("")
            lines.append("| p \\ q | q=0 (1) | q=1 (lam2) | q=2 (lam2^2) | q=3 (lam2^3) |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for p in range(deg1 + 1):
                row = result.gain_coefficients[in_idx, st_idx, p]
                row_str = " | ".join(f"`{v:+.6e}`" for v in row)
                p_label = f"p={p} (lam1^{p})" if p > 0 else "p=0 (1)"
                lines.append(f"| **{p_label}** | {row_str} |")
            lines.append("")
    return "\n".join(lines)


def save_result(result: GainScheduleFit, output_dir: Path) -> None:
    """将 L1/L2 增益调度产物保存至指定输出目录。"""
    max_float_radius, _ = _float32_fit_metrics(result)
    if max_float_radius >= 1.0:
        raise ValueError("float32 fitted gain contains an unstable closed-loop node in L1/L2 domain")

    output_dir = resolve_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text(output_dir / "lqr_coef.h", render_header(result))
    write_text(output_dir / "K_l1_l2_polynomial_matrices.md", render_polynomial_matrices(result))
    write_text(output_dir / "README.md", render_summary(result))
    np.savez_compressed(
        output_dir / "l1_l2_gain_nodes.npz",
        l1_grid=result.l1_grid,
        l2_grid=result.l2_grid,
        gains=np.array([n.gain for n in result.nodes]),
        fitted_gains=result.fitted_gains,
        gain_coefficients=result.gain_coefficients,
        input_coefficients=result.input_coefficients,
    )
    report = {
        **artifact_metadata(result.nodes[0].parameters, sample_time=result.nodes[0].settings.sample_time, q_diag=list(result.nodes[0].settings.q_diag), r_diag=list(result.nodes[0].settings.r_diag)),
        "length_domain_m": [result.nodes[0].parameters.min_leg_length, result.nodes[0].parameters.max_leg_length],
        "sample_count": len(result.nodes),
        "polynomial_model": "Tensor-Product Degree (3, 3)",
        "max_abs_gain_fit_error": result.max_abs_gain_error,
        "rms_gain_fit_error": result.rms_gain_error,
        "max_raw_closed_loop_spectral_radius": result.max_raw_spectral_radius,
        "max_fitted_closed_loop_spectral_radius": result.max_fitted_spectral_radius,
        "max_float32_fitted_closed_loop_spectral_radius": max_float_radius,
        "saturated_recovery": {
            "validation_node_count": result.saturation_validation_node_count,
            "total_node_count": len(result.nodes),
            "passed": result.saturation_validation.passed,
            "max_state_abs": result.saturation_validation.max_state_abs,
            "max_final_state_abs": result.saturation_validation.max_final_state_abs,
            "max_commanded_input_abs": result.saturation_validation.max_commanded_input_abs,
            "saturation_steps": result.saturation_validation.saturation_steps,
            "worst_case": result.saturation_validation.worst_case,
        },
    }
    (output_dir / "l1_l2_gain_fit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
