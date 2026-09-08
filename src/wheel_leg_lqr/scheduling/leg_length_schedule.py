"""计算并拟合 K(L1, L2)：覆盖全腿长范围的 L1/L2 张量积增益调度。

Compute and fit L1/L2 tensor-product gain scheduling K(L1, L2) over the full
operating envelope (L1, L2) in [0.18, 0.38]^2 m^2, then emit the C++ header,
polynomial matrix report, NumPy archive, and JSON summary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from ..core import config
from .. import resolve_output_dir
from ..core.model import StaticLinearization, linearize_static_trim
from ..control import SaturationValidation, validate_saturated_closed_loop
from ..control.lqr import controllability_rank, discretize_state_space, solve_discrete_lqr
from ..artifacts.io import write_text
from .polynomial import evaluate_tensor_polynomial, fit_tensor_coefficients, tensor_design_matrix

__all__ = [
    "GainNode",
    "GainScheduleFit",
    "compute_active_equilibrium_input",
    "compute_gain_node",
    "compute_gain_schedule",
    "evaluate_fitted_gain",
    "main",
    "make_leg_length_grid",
    "render_active_fl_u0_block",
    "render_firmware_lqr_body",
    "render_header",
    "render_polynomial_matrices",
    "render_summary",
    "save_result",
]


@dataclass(frozen=True)
class GainNode:
    l1: float
    l2: float
    height: float  # Derived vertical height; not an LQR state.
    linearization: StaticLinearization
    Ad: np.ndarray
    Bd: np.ndarray
    gain: np.ndarray
    closed_loop_poles: np.ndarray
    controllability_rank: int


@dataclass(frozen=True)
class GainScheduleFit:
    l1_grid: np.ndarray
    l2_grid: np.ndarray
    nodes: tuple[GainNode, ...]
    gain_coefficients: np.ndarray  # Shape: (INPUT_DIM, STATE_DIM, DEG1+1, DEG2+1)
    input_coefficients: np.ndarray  # Shape: (INPUT_DIM, DEG1+1, DEG2+1)
    fitted_gains: np.ndarray  # Shape: (N_NODES, INPUT_DIM, STATE_DIM)
    max_abs_gain_error: float
    rms_gain_error: float
    max_raw_spectral_radius: float
    max_fitted_spectral_radius: float
    saturation_validation: SaturationValidation
    saturation_validation_node_count: int


def make_leg_length_grid(
    start: float = config.PARAMETERS.min_leg_length,
    stop: float = config.PARAMETERS.max_leg_length,
    step: float = 0.01,
) -> np.ndarray:
    count = int(round((stop - start) / step)) + 1
    grid = np.linspace(start, stop, count)
    return grid


def _discretize(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return discretize_state_space(A, B)


def _controllability_rank(A: np.ndarray, B: np.ndarray) -> int:
    return controllability_rank(A, B)


def compute_gain_node(l1: float, l2: float) -> GainNode:
    """在单个 (L1, L2) 坐标处求解自平衡配平、线性化、离散化与离散 LQR。"""
    b1 = config.compute_equilibrium_beta_from_length(l1)
    b2 = config.compute_equilibrium_beta_from_length(l2)
    linearization = linearize_static_trim(
        l1,
        l2,
        beta_1=b1,
        beta_2=b2,
    )
    Ad, Bd = _discretize(linearization.A, linearization.B)
    gain, _, poles = solve_discrete_lqr(Ad, Bd)
    return GainNode(
        l1=float(l1),
        l2=float(l2),
        height=float(linearization.geometry.height),
        linearization=linearization,
        Ad=Ad,
        Bd=Bd,
        gain=gain,
        closed_loop_poles=poles,
        controllability_rank=_controllability_rank(Ad, Bd),
    )


def compute_active_equilibrium_input(node: GainNode) -> np.ndarray:
    """Return motor-side trim input after passive gas-spring support.

    The linear plant and LQR gain continue to use total virtual ``F_L``.
    Only the exported motor feedforward subtracts the already-equivalent
    passive gas-spring force on each leg.
    """

    active_input = node.linearization.equilibrium_input.copy()
    active_input[4] -= config.compute_gas_spring_force(node.l1)
    active_input[5] -= config.compute_gas_spring_force(node.l2)
    return active_input


def _build_tensor_design_matrix(
    lam1: np.ndarray,
    lam2: np.ndarray,
    deg1: int = 3,
    deg2: int = 3,
) -> np.ndarray:
    """构建 L1/L2 张量积多项式设计矩阵 X，形状为 (N, (deg1+1)*(deg2+1))。"""
    return tensor_design_matrix(lam1, lam2, deg1, deg2)


def _fit_tensor_coefficients(
    lam1: np.ndarray,
    lam2: np.ndarray,
    values: np.ndarray,
    deg1: int = 3,
    deg2: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """拟合多通道 L1/L2 张量积多项式系数。

    values 形状: (N_SAMPLES, ...)
    返回 coefficients 形状: (..., deg1+1, deg2+1)
    fitted 形状: (N_SAMPLES, ...)
    """
    return fit_tensor_coefficients(lam1, lam2, values, deg1, deg2)


def evaluate_fitted_gain(
    coefficients: np.ndarray,
    l1: float,
    l2: float,
    l_nominal: float = 0.28,
    l_half_range: float = 0.10,
    deg1: int = 3,
    deg2: int = 3,
) -> np.ndarray:
    """使用嵌套秦九韶算法计算给定 (L1, L2) 处的 L1/L2 拟合增益矩阵。"""
    lam1 = np.clip((l1 - l_nominal) / l_half_range, -1.0, 1.0)
    lam2 = np.clip((l2 - l_nominal) / l_half_range, -1.0, 1.0)

    # coefficients: (INPUT_DIM, STATE_DIM, deg1+1, deg2+1)
    return evaluate_tensor_polynomial(coefficients, lam1, lam2)


def compute_gain_schedule(
    step: float = 0.01,
    deg1: int = 3,
    deg2: int = 3,
    *,
    exhaustive_validation: bool = False,
) -> GainScheduleFit:
    """在完整 (L1, L2) 网格上求解并拟合张量积增益调度矩阵。"""
    l1_grid = make_leg_length_grid(step=step)
    l2_grid = make_leg_length_grid(step=step)

    nodes_list: list[GainNode] = []
    lam1_list: list[float] = []
    lam2_list: list[float] = []

    l_nominal = 0.28
    l_half_range = 0.10

    for l1 in l1_grid:
        for l2 in l2_grid:
            node = compute_gain_node(l1, l2)
            nodes_list.append(node)
            lam1_list.append((l1 - l_nominal) / l_half_range)
            lam2_list.append((l2 - l_nominal) / l_half_range)

    nodes = tuple(nodes_list)
    lam1_arr = np.array(lam1_list, dtype=float)
    lam2_arr = np.array(lam2_list, dtype=float)

    raw_gains = np.array([node.gain for node in nodes])
    raw_inputs = np.array([compute_active_equilibrium_input(node) for node in nodes])

    gain_coefficients, fitted_gains = _fit_tensor_coefficients(
        lam1_arr, lam2_arr, raw_gains, deg1, deg2
    )
    input_coefficients, _ = _fit_tensor_coefficients(
        lam1_arr, lam2_arr, raw_inputs, deg1, deg2
    )

    fitted_radii = []
    for node, fitted_gain in zip(nodes, fitted_gains):
        closed_loop = node.Ad - node.Bd @ fitted_gain
        poles = np.linalg.eigvals(closed_loop)
        fitted_radii.append(float(np.max(np.abs(poles))))

    error = fitted_gains - raw_gains
    float32_fitted_gains = _evaluate_float32_fitted_gains(
        nodes, gain_coefficients, deg1, deg2
    )
    if exhaustive_validation:
        validation_indices = np.arange(len(nodes), dtype=int)
    else:
        validation_indices = _representative_validation_indices(l1_grid, l2_grid)
    validation_nodes = tuple(nodes[index] for index in validation_indices)
    validation_gains = float32_fitted_gains[validation_indices]
    saturation_validation = _validate_fitted_saturation(
        validation_nodes,
        validation_gains,
    )
    if not saturation_validation.passed:
        raise ValueError(
            "saturated closed-loop recovery failed at "
            f"{saturation_validation.worst_case}"
        )
    return GainScheduleFit(
        l1_grid=l1_grid,
        l2_grid=l2_grid,
        nodes=nodes,
        gain_coefficients=gain_coefficients,
        input_coefficients=input_coefficients,
        fitted_gains=fitted_gains,
        max_abs_gain_error=float(np.max(np.abs(error))),
        rms_gain_error=float(np.sqrt(np.mean(error**2))),
        max_raw_spectral_radius=max(
            float(np.max(np.abs(node.closed_loop_poles))) for node in nodes
        ),
        max_fitted_spectral_radius=max(fitted_radii),
        saturation_validation=saturation_validation,
        saturation_validation_node_count=len(validation_nodes),
    )


def _representative_validation_indices(
    l1_grid: np.ndarray,
    l2_grid: np.ndarray,
) -> np.ndarray:
    """Return a 5 x 5 spread of nodes for the slow saturation simulations.

    The selected coordinates include both operating-envelope boundaries and
    three interior slices.  This samples asymmetry in both L1 and L2 while
    keeping the default workflow substantially faster than exhaustive
    per-node simulation.
    """
    positions_1 = np.linspace(0, len(l1_grid) - 1, num=min(5, len(l1_grid)), dtype=int)
    positions_2 = np.linspace(0, len(l2_grid) - 1, num=min(5, len(l2_grid)), dtype=int)
    indices = {
        index_1 * len(l2_grid) + index_2
        for index_1 in positions_1
        for index_2 in positions_2
    }
    return np.asarray(sorted(indices), dtype=int)


def _validate_fitted_saturation(
    nodes: tuple[GainNode, ...], fitted_gains: np.ndarray
) -> SaturationValidation:
    """Validate the gain actually exported by the L1/L2 polynomial fit."""
    max_state_abs = 0.0
    max_final_state_abs = 0.0
    max_commanded_input_abs = 0.0
    total_saturation_steps = 0
    passed = True
    worst_case = "none"
    for node, gain in zip(nodes, fitted_gains):
        validation = validate_saturated_closed_loop(
            node.Ad,
            node.Bd,
            gain,
            node.linearization.equilibrium_input,
        )
        max_state_abs = max(max_state_abs, validation.max_state_abs)
        max_final_state_abs = max(max_final_state_abs, validation.max_final_state_abs)
        max_commanded_input_abs = max(
            max_commanded_input_abs, validation.max_commanded_input_abs
        )
        total_saturation_steps += validation.saturation_steps
        if not validation.passed:
            passed = False
            worst_case = (
                f"L1={node.l1:.6f}, L2={node.l2:.6f}: "
                f"{validation.worst_case}"
            )
    return SaturationValidation(
        passed=passed,
        max_state_abs=max_state_abs,
        max_final_state_abs=max_final_state_abs,
        max_commanded_input_abs=max_commanded_input_abs,
        saturation_steps=total_saturation_steps,
        worst_case=worst_case,
    )


def _evaluate_float32_fitted_gains(
    nodes: tuple[GainNode, ...],
    coefficients: np.ndarray,
    deg1: int,
    deg2: int,
) -> np.ndarray:
    """Evaluate the same float32 tensor polynomial used by the header."""
    coefficients = np.asarray(coefficients, dtype=np.float32)
    gains = []
    for node in nodes:
        lam1 = np.float32(np.clip((node.l1 - 0.28) / 0.10, -1.0, 1.0))
        lam2 = np.float32(np.clip((node.l2 - 0.28) / 0.10, -1.0, 1.0))
        p_powers = np.array([lam1**p for p in range(deg1 + 1)], dtype=np.float32)
        q_powers = np.array([lam2**q for q in range(deg2 + 1)], dtype=np.float32)
        gain = np.einsum("ijpq,p,q->ij", coefficients, p_powers, q_powers)
        gains.append(gain.astype(float))
    return np.asarray(gains, dtype=float)


def _float32_fit_metrics(
    result: GainScheduleFit,
) -> tuple[float, float]:
    fitted_float = result.gain_coefficients.astype(np.float32)
    l_nominal = 0.28
    l_half_range = 0.10
    deg1 = result.gain_coefficients.shape[2] - 1
    deg2 = result.gain_coefficients.shape[3] - 1

    radii = []
    gain_errors = []
    for node in result.nodes:
        lam1 = np.float32(
            np.clip((node.l1 - l_nominal) / l_half_range, -1.0, 1.0)
        )
        lam2 = np.float32(
            np.clip((node.l2 - l_nominal) / l_half_range, -1.0, 1.0)
        )
        p_pow = np.array([lam1**p for p in range(deg1 + 1)], dtype=np.float32)
        q_pow = np.array([lam2**q for q in range(deg2 + 1)], dtype=np.float32)
        gain_f32 = np.einsum("ijpq,p,q->ij", fitted_float, p_pow, q_pow)

        closed_loop = (
            node.Ad.astype(np.float32)
            - node.Bd.astype(np.float32) @ gain_f32
        )
        poles = np.linalg.eigvals(closed_loop)
        radii.append(float(np.max(np.abs(poles))))
        gain_errors.append(
            float(np.max(np.abs(gain_f32.astype(float) - node.gain)))
        )

    return max(radii), max(gain_errors)


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
    q_vals = config.Q_DIAG.tolist()
    r_vals = config.R_DIAG.tolist()

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
    deg2 = coefficients.shape[3] - 1
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
            f"{'constexpr uint32_t BETA_TRIM_POLY_DEGREE':<68} = {len(config.LEG_COM_BETA_EQ_POLY) - 1};",
        ]
    )
    lines.extend(
        [
            " */",
            "",
            *dimension_lines,
            "",
            "// L1/L2 tensor-product polynomial coefficients over normalized leg lengths.",
            "// lambda_1 = clamp((L1 - 0.28f) / 0.10f, -1.0f, 1.0f)",
            "// lambda_2 = clamp((L2 - 0.28f) / 0.10f, -1.0f, 1.0f)",
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
        f"{v:.10e}f" for v in config.LEG_COM_BETA_EQ_POLY
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
        "- Domain: `L1 in [0.18, 0.38] m`, `L2 in [0.18, 0.38] m`",
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
    lines = [
        "# K(L1, L2) L1/L2 Tensor-Product Polynomial Coefficient Matrices",
        "",
        "Evaluated via: `lambda_1 = clamp((L1 - 0.28)/0.10, -1, 1)`, `lambda_2 = clamp((L2 - 0.28)/0.10, -1, 1)`",
        "`K[i, j] = sum_{p=0}^3 sum_{q=0}^3 C[i, j, p, q] * lambda_1^p * lambda_2^q`",
        "",
    ]
    deg1 = result.gain_coefficients.shape[2] - 1
    deg2 = result.gain_coefficients.shape[3] - 1
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
        "domain": "[0.18, 0.38] x [0.18, 0.38] m^2",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=float, default=0.01, help="Grid step in meters (default: 0.01).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generated"),
        help="Directory for generated L1/L2 artifacts (default: data/generated).",
    )
    parser.add_argument(
        "--exhaustive-validation",
        action="store_true",
        help="Run the slow 5-second saturation simulation at every grid node.",
    )
    args = parser.parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    result = compute_gain_schedule(
        step=args.step,
        exhaustive_validation=args.exhaustive_validation,
    )
    save_result(result, output_dir)
    print(
        f"nodes={len(result.nodes)}, "
        f"max_fit_error={result.max_abs_gain_error:.6e}, "
        f"rms_fit_error={result.rms_gain_error:.6e}"
    )
    print(
        "max spectral radius: "
        f"raw={result.max_raw_spectral_radius:.9f}, "
        f"fitted={result.max_fitted_spectral_radius:.9f}"
    )
    print(
        "saturation validation: "
        f"{result.saturation_validation_node_count}/{len(result.nodes)} nodes"
    )


if __name__ == "__main__":
    main()
