"""Fit the contact-free two-hip LQR gain over the L1/L2 length envelope."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

from . import config
from .model import (
    AIRBORNE_INPUT_NAMES,
    AIRBORNE_STATE_NAMES,
    AirborneLinearization,
    linearize_airborne,
)


@dataclass(frozen=True)
class AirborneGainNode:
    l1: float
    l2: float
    linearization: AirborneLinearization
    Ad: np.ndarray
    Bd: np.ndarray
    Ad_shape: np.ndarray
    Bd_shape: np.ndarray
    gain: np.ndarray  # [T_p1, T_p2] x six interleaved physical states.
    shape_poles: np.ndarray


@dataclass(frozen=True)
class AirborneScheduleFit:
    l1_grid: np.ndarray
    l2_grid: np.ndarray
    sample_time: float
    q_diag: np.ndarray
    r_diag: np.ndarray
    nodes: tuple[AirborneGainNode, ...]
    gain_coefficients: np.ndarray  # [input, state, p, q].
    fitted_gains: np.ndarray
    max_abs_gain_error: float
    max_relative_gain_error: float
    max_raw_shape_radius: float
    max_float32_shape_radius: float
    validation_node_count: int


def _resolve_output_dir(output_dir: Path) -> Path:
    """Keep generated data out of the repository's frozen reference snapshot."""
    path = Path(output_dir).resolve()
    repository_root = Path(__file__).resolve().parents[3]
    snapshot_root = repository_root.parent / "host_code"
    try:
        path.relative_to(snapshot_root)
    except ValueError:
        return path
    raise ValueError(f"output-dir must not point into the frozen snapshot {snapshot_root}")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip("\n") + "\n", encoding="utf-8", newline="\n")


def make_leg_length_grid(step: float) -> np.ndarray:
    """Return the airborne model's own inclusive leg-length grid."""
    p = config.PARAMETERS
    count = int(round((p.max_leg_length - p.min_leg_length) / step)) + 1
    if count < 2:
        raise ValueError("step is too large for the airborne leg-length interval")
    return np.linspace(p.min_leg_length, p.max_leg_length, count)


def _discretize(A: np.ndarray, B: np.ndarray, sample_time: float) -> tuple[np.ndarray, np.ndarray]:
    n, m = B.shape
    Ad, Bd, _, _, _ = cont2discrete(
        (A, B, np.zeros((1, n)), np.zeros((1, m))), sample_time, method="zoh"
    )
    return np.asarray(Ad), np.asarray(Bd)


def _controllability_rank(A: np.ndarray, B: np.ndarray) -> int:
    blocks = [B]
    for _ in range(1, A.shape[0]):
        blocks.append(A @ blocks[-1])
    return int(np.linalg.matrix_rank(np.hstack(blocks)))


def _discrete_lqr(
    Ad: np.ndarray, Bd: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    cost = solve_discrete_are(Ad, Bd, Q, R)
    gain = np.linalg.solve(R + Bd.T @ cost @ Bd, Bd.T @ cost @ Ad)
    poles = np.linalg.eigvals(Ad - Bd @ gain)
    return gain, poles


def _fit_tensor_gains(
    lam1: np.ndarray, lam2: np.ndarray, gains: np.ndarray, degree: int
) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack(
        [lam1**p * lam2**q for p in range(degree + 1) for q in range(degree + 1)]
    )
    flat, *_ = np.linalg.lstsq(design, gains.reshape(len(gains), -1), rcond=None)
    coefficients = flat.reshape(degree + 1, degree + 1, *gains.shape[1:])
    coefficients = np.transpose(coefficients, (2, 3, 0, 1))
    fitted = (design @ flat).reshape(gains.shape)
    return coefficients, fitted


def _normalization() -> tuple[float, float]:
    p = config.PARAMETERS
    return (
        (p.min_leg_length + p.max_leg_length) / 2.0,
        (p.max_leg_length - p.min_leg_length) / 2.0,
    )


def compute_airborne_gain_node(
    l1: float,
    l2: float,
    *,
    sample_time: float = config.SAMPLE_TIME,
    q_diag: np.ndarray | None = None,
    r_diag: np.ndarray | None = None,
) -> AirborneGainNode:
    """Solve discrete LQR on four controllable shape states at one node."""
    plant = linearize_airborne(l1, l2, q_diag=q_diag, r_diag=r_diag)
    Ad, Bd = _discretize(plant.A, plant.B, sample_time)
    Ad_shape, Bd_shape = _discretize(plant.A_shape, plant.B_shape, sample_time)
    if _controllability_rank(Ad_shape, Bd_shape) != 4:
        raise ValueError(f"airborne hip shape is not controllable at ({l1}, {l2})")
    shape_gain, shape_poles = _discrete_lqr(Ad_shape, Bd_shape, plant.Q_shape, plant.R)
    gain = shape_gain @ plant.state_projection
    return AirborneGainNode(
        l1=float(l1),
        l2=float(l2),
        linearization=plant,
        Ad=Ad,
        Bd=Bd,
        Ad_shape=Ad_shape,
        Bd_shape=Bd_shape,
        gain=gain,
        shape_poles=shape_poles,
    )


def evaluate_airborne_gain(
    coefficients: np.ndarray,
    l1: float,
    l2: float,
    *,
    dtype: type = float,
) -> np.ndarray:
    """Evaluate the exported tensor polynomial with float64 or MCU float32."""
    coefficients = np.asarray(coefficients, dtype=dtype)
    if coefficients.ndim != 4 or coefficients.shape[:2] != (2, 6):
        raise ValueError("coefficients must have shape (2, 6, degree1+1, degree2+1)")
    center, scale = _normalization()
    lam1 = dtype(np.clip((float(l1) - center) / scale, -1.0, 1.0))
    lam2 = dtype(np.clip((float(l2) - center) / scale, -1.0, 1.0))
    # Nested Horner order matches the C++ example in the documentation.
    result = np.zeros((2, 6), dtype=dtype)
    for p in range(coefficients.shape[2] - 1, -1, -1):
        row = np.zeros((2, 6), dtype=dtype)
        for q in range(coefficients.shape[3] - 1, -1, -1):
            row = row * lam2 + coefficients[:, :, p, q]
        result = result * lam1 + row
    return result


def _shape_radius(node: AirborneGainNode, gain: np.ndarray) -> float:
    shape_gain = gain @ node.linearization.state_lift
    poles = np.linalg.eigvals(node.Ad_shape - node.Bd_shape @ shape_gain)
    return float(np.max(np.abs(poles)))


def compute_airborne_schedule(
    *,
    step: float = 0.01,
    degree: int = config.POLYNOMIAL_DEGREE,
    sample_time: float = config.SAMPLE_TIME,
    q_diag: np.ndarray | None = None,
    r_diag: np.ndarray | None = None,
) -> AirborneScheduleFit:
    """Fit K(L1,L2) and verify float32 gains on grid and cell centers."""
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be positive and finite")
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    grid = make_leg_length_grid(step)
    if len(grid) <= degree:
        raise ValueError("the leg-length grid needs at least degree+1 points per axis")
    if not np.isfinite(sample_time) or sample_time <= 0.0:
        raise ValueError("sample_time must be positive and finite")
    nodes = tuple(
        compute_airborne_gain_node(
            l1, l2, sample_time=sample_time, q_diag=q_diag, r_diag=r_diag
        )
        for l1 in grid
        for l2 in grid
    )
    center, scale = _normalization()
    lam1 = np.array([(node.l1 - center) / scale for node in nodes])
    lam2 = np.array([(node.l2 - center) / scale for node in nodes])
    raw_gains = np.stack([node.gain for node in nodes])
    coefficients, fitted = _fit_tensor_gains(lam1, lam2, raw_gains, degree)
    coefficients_float32 = coefficients.astype(np.float32)
    midpoint_grid = (grid[:-1] + grid[1:]) / 2.0
    validation_nodes = [
        *nodes,
        *(
            compute_airborne_gain_node(
                l1, l2, sample_time=sample_time, q_diag=q_diag, r_diag=r_diag
            )
            for l1 in midpoint_grid
            for l2 in midpoint_grid
        ),
    ]
    max_radius = 0.0
    max_relative_error = 0.0
    for node in validation_nodes:
        gain = evaluate_airborne_gain(
            coefficients_float32, node.l1, node.l2, dtype=np.float32
        ).astype(float)
        max_radius = max(max_radius, _shape_radius(node, gain))
        max_relative_error = max(
            max_relative_error,
            float(np.linalg.norm(gain - node.gain) / np.linalg.norm(node.gain)),
        )
    if max_radius >= 1.0:
        raise ValueError("float32 fitted airborne LQR is unstable on the validation grid")
    return AirborneScheduleFit(
        l1_grid=grid,
        l2_grid=grid.copy(),
        sample_time=float(sample_time),
        q_diag=nodes[0].linearization.q_diag.copy(),
        r_diag=nodes[0].linearization.r_diag.copy(),
        nodes=nodes,
        gain_coefficients=coefficients,
        fitted_gains=fitted,
        max_abs_gain_error=float(np.max(np.abs(fitted - raw_gains))),
        max_relative_gain_error=max_relative_error,
        max_raw_shape_radius=max(float(np.max(np.abs(n.shape_poles))) for n in nodes),
        max_float32_shape_radius=max_radius,
        validation_node_count=len(validation_nodes),
    )


def render_airborne_header(result: AirborneScheduleFit) -> str:
    """Render a self-contained C++ coefficient header, separate from ground LQR."""
    coefficients = result.gain_coefficients.astype(np.float32)
    degree = coefficients.shape[2] - 1
    center, scale = _normalization()
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
        "state_names": AIRBORNE_STATE_NAMES,
        "input_names": AIRBORNE_INPUT_NAMES,
        "length_domain_m": [
            config.PARAMETERS.min_leg_length,
            config.PARAMETERS.max_leg_length,
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
        "body_mass_kg": config.PARAMETERS.body_mass,
    }
    _write_text(output_dir / "airborne_lqr_fit.json", json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=float, default=0.01, help="L1/L2 grid spacing in metres")
    parser.add_argument("--degree", type=int, default=config.POLYNOMIAL_DEGREE)
    parser.add_argument("--sample-time", type=float, default=config.SAMPLE_TIME)
    parser.add_argument(
        "--q-diag", type=float, nargs=6,
        metavar=("QB1", "QDB1", "QB2", "QDB2", "QTHETA", "QDTHETA"),
        help="Six physical-state Q diagonal weights; defaults to airborne.config Q_DIAG",
    )
    parser.add_argument(
        "--r-diag", type=float, nargs=2, metavar=("RTP1", "RTP2"),
        help="Two hip-input R diagonal weights; defaults to airborne.config R_DIAG",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/generated/airborne_lqr")
    )
    args = parser.parse_args()
    result = compute_airborne_schedule(
        step=args.step, degree=args.degree, sample_time=args.sample_time,
        q_diag=args.q_diag, r_diag=args.r_diag,
    )
    output_dir = _resolve_output_dir(args.output_dir)
    save_airborne_schedule(result, output_dir)
    print(
        f"nodes={len(result.nodes)}, validation={result.validation_node_count}, "
        f"max_relative_gain_error={result.max_relative_gain_error:.6e}, "
        f"max_controllable_radius={result.max_float32_shape_radius:.9f}"
    )
    print(f"Airborne LQR artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
