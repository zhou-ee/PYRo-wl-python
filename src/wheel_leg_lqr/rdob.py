"""Host-side RDOB L1/L2 scheduling and optional single-trim diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import resolve_output_dir
from .artifacts.io import write_text
from .control.rdob import (
    RdobDesign,
    design_rdob,
    trapezoidal_step_constants,
)
from .core import config
from .core.model import StaticLinearization, linearize_static_trim


# Match the disturbance pole used by the deployed LESO.  RDOB uses a
# trapezoidal update, so invert that discretization instead of interpreting
# the LESO pole through a continuous-time exponential approximation.
REFERENCE_LESO_DISTURBANCE_POLE = 0.992
_DEFAULT_RATE_RAD_S = (
    2.0 * (1.0 - REFERENCE_LESO_DISTURBANCE_POLE)
    / (config.SAMPLE_TIME * (1.0 + REFERENCE_LESO_DISTURBANCE_POLE))
)
_DEFAULT_CHANNEL_BANDWIDTH_HZ = float(_DEFAULT_RATE_RAD_S / (2.0 * np.pi))
# Keep the public default as six explicit channel entries.  The values match
# today, but each channel remains independently configurable through the CLI.
DEFAULT_BANDWIDTH_HZ = (
    _DEFAULT_CHANNEL_BANDWIDTH_HZ,  # T_w1
    _DEFAULT_CHANNEL_BANDWIDTH_HZ,  # T_w2
    _DEFAULT_CHANNEL_BANDWIDTH_HZ,  # T_p1
    _DEFAULT_CHANNEL_BANDWIDTH_HZ,  # T_p2
    _DEFAULT_CHANNEL_BANDWIDTH_HZ,  # f_l1
    _DEFAULT_CHANNEL_BANDWIDTH_HZ,  # f_l2
)
REFERENCE_LESO_DISTURBANCE_LIMIT_RATIO = 0.1
DEFAULT_DISTURBANCE_LIMITS = tuple(
    float(value)
    for value in REFERENCE_LESO_DISTURBANCE_LIMIT_RATIO * config.INPUT_UPPER_LIMITS
)


@dataclass(frozen=True)
class RdobNode:
    model: StaticLinearization
    observer: RdobDesign


def compute_rdob_node(
    l1: float, l2: float, *,
    bandwidth_hz: np.ndarray | float | tuple[float, ...] = DEFAULT_BANDWIDTH_HZ,
    sample_time: float = config.SAMPLE_TIME,
    damping: np.ndarray | None = None,
) -> RdobNode:
    """Use existing static mechanics; D0 defaults to zero (not identified).

    The default gives all six RDOB channels the deployed LESO disturbance pole
    at the nominal sample time.  All six channels are observed regardless of
    downstream compensation selection.
    """
    for length in (l1, l2):
        if not np.isfinite(length):
            raise ValueError("leg lengths must be finite")
    bandwidth = np.asarray(bandwidth_hz, dtype=float)
    if bandwidth.ndim > 1 or (bandwidth.ndim == 1 and bandwidth.shape != (6,)):
        raise ValueError("bandwidth_hz must be a scalar or six values")
    if not np.all(np.isfinite(bandwidth)) or np.any(bandwidth <= 0):
        raise ValueError("bandwidth_hz must be positive and finite")
    model = linearize_static_trim(l1, l2)
    observer = design_rdob(
        model.mass, model.stiffness, model.input_matrix,
        rates_rad_s=2.0 * np.pi * bandwidth, sample_time=sample_time,
        damping=damping,
    )
    return RdobNode(model, observer)


def rdob_diagnostics(design: RdobDesign) -> dict[str, float | int]:
    """Algebraic diagnostics; these do not certify the compensated closed loop."""
    s = design.input_matrix
    projector = np.eye(config.N_Q) - s @ design.input_left_inverse
    identity_error = design.input_left_inverse @ s - np.eye(config.N_U)
    dynamic_error = (design.gamma @ np.linalg.solve(design.mass, s)
                     - np.diag(design.rates))
    gamma32 = design.gamma.astype(np.float32).astype(float)
    return {
        "observer_dynamic_states": config.N_U,
        "input_rank": int(np.linalg.matrix_rank(s)),
        "input_condition_number_coordinate_dependent": design.input_condition_number,
        "left_inverse_error_inf": float(np.linalg.norm(identity_error, ord=np.inf)),
        "matched_rate_identity_error_inf": float(np.linalg.norm(dynamic_error, ord=np.inf)),
        "orthogonal_force_rejection_error_inf": float(np.linalg.norm(
            design.input_left_inverse @ projector, ord=np.inf)),
        "auxiliary_step_spectral_radius": float(np.max(np.abs(design.step_poles))),
        "negative_auxiliary_step_poles": int(np.count_nonzero(design.step_poles < 0)),
        "gamma_float32_relative_error": float(np.linalg.norm(gamma32 - design.gamma)
                                             / np.linalg.norm(design.gamma)),
    }


def _cpp_array(name: str, values: np.ndarray) -> str:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(np.abs(array) > np.finfo(np.float32).max):
        raise ValueError(f"{name} is not representable as finite float32")
    rounded = array.astype(np.float32)
    shape = "".join(f"[{size}]" for size in array.shape)
    def row(items: np.ndarray) -> str:
        return ", ".join(f"{float(value):.9e}f" for value in items)
    if array.ndim == 1:
        body = "    " + row(rounded)
    elif array.ndim == 2:
        body = "\n".join("    {" + row(items) + "}," for items in rounded)
    else:
        raise ValueError("C++ export accepts vectors and matrices only")
    return f"static constexpr float {name}{shape} = {{\n{body}\n}};"


def export_rdob(node: RdobNode, output_dir: str | Path) -> Path:
    """Export one frozen trim, with explicit units/timing and no firmware edits."""
    output = resolve_output_dir(output_dir)
    design, model = node.observer, node.model
    gas_force = np.asarray([
        config.compute_gas_spring_force(length)
        for length in model.geometry.leg_lengths
    ])
    active_trim_input = model.equilibrium_input.copy()
    active_trim_input[4:] -= gas_force
    _, step_endpoint_weight, _, step_velocity_scale = (
        trapezoidal_step_constants(design.rates, design.sample_time)
    )
    arrays = {
        name: value for name in design.__dataclass_fields__
        if isinstance(value := getattr(design, name), np.ndarray)
    }
    arrays.update(
        trim_position=model.geometry.q,
        trim_total_input=model.equilibrium_input,
        trim_active_input=active_trim_input,
        trim_gas_spring_force=gas_force,
        leg_lengths=model.geometry.leg_lengths,
        step_endpoint_weight=step_endpoint_weight,
        step_velocity_scale=step_velocity_scale,
    )
    metadata = {
        "schema_version": 1,
        "observer": "six-state input-equivalent RDOB",
        "scope": "frozen static trim; no scheduled stability certification",
        "position_order": list(config.STATE_NAMES[::2]),
        "velocity_order": list(config.STATE_NAMES[1::2]),
        "input_order": list(config.INPUT_NAMES),
        "input_units": ["N*m"] * 4 + ["N"] * 2,
        "position_units": ["m", "rad", "rad", "rad", "m", "rad", "rad"],
        "velocity_units": ["m/s", "rad/s", "rad/s", "rad/s", "m/s", "rad/s", "rad/s"],
        "rates_units": "rad/s; rates = 2*pi*bandwidth_hz",
        "bandwidth_hz": (design.rates / (2 * np.pi)).tolist(),
        "sample_time_s": design.sample_time,
        "kappa_g": model.geometry.kappa_g,
        "trim_height_m": model.geometry.height,
        "damping_is_zero": bool(np.all(design.damping == 0)),
        "discretization": "trapezoidal z update; q/v endpoints and interval-held applied input",
        "input_convention": "actual total generalized inputs minus trim_total_input; includes compensation",
        "gas_spring_convention": "add current modeled passive leg forces to active actuator forces before subtracting trim_total_input",
        "diagnostics": rdob_diagnostics(design),
        "arrays": {name: value.tolist() for name, value in arrays.items()},
    }
    header = [
        "// Generated frozen-trim RDOB coefficients. C++11 or later.",
        "// q/v order: " + ", ".join(config.STATE_NAMES[::2]),
        "// Input order: " + ", ".join(config.INPUT_NAMES),
        "// q/v are measured deltas from this SAME trim (v0 = 0).",
        "// u is actual TOTAL input minus kTrimTotalInput over [k,k+1).",
        "// Reconstruct: d[i] = z[i] + sum_j kGamma[i][j]*v[j].",
        "// Initialize: z[i] = initial_d[i] - sum_j kGamma[i][j]*v[j].",
        "// At the next sample, using OLD z and input from the elapsed interval:",
        "// z_next[i] = kStepPoles[i]*z[i] + kStepInputGain[i]*u[i]",
        "//   + sum_j kStepPositionGain[i][j]*(q_old[j]+q_new[j])",
        "//   + sum_j kStepVelocityGain[i][j]*(v_old[j]+v_new[j]);",
        "// d_next = z_next + kGamma*v_new. Compensation is a separate stage.",
        "#pragma once", "namespace wheel_leg_rdob {",
        "static constexpr unsigned kPositionDimension = 7;",
        "static constexpr unsigned kInputDimension = 6;",
        f"static constexpr float kSampleTime = {design.sample_time:.9e}f;",
    ]
    for name, value in (
        ("kRatesRadPerSecond", design.rates), ("kGamma", design.gamma),
        ("kContinuousPositionGain", design.position_gain),
        ("kContinuousVelocityGain", design.velocity_gain),
        ("kStepPoles", design.step_poles),
        ("kStepEndpointWeight", step_endpoint_weight),
        ("kStepPositionGain", design.step_position_gain),
        ("kStepVelocityGain", design.step_velocity_gain),
        ("kStepVelocityScale", step_velocity_scale),
        ("kStepInputGain", design.step_input_gain),
        ("kTrimPosition", model.geometry.q),
        ("kTrimTotalInput", model.equilibrium_input),
        ("kTrimActiveInput", active_trim_input),
    ):
        header.append(_cpp_array(name, value))
    header.append("} // namespace wheel_leg_rdob")
    # Build/validate all content before writing any artifact.
    json_text = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "rdob.npz", **arrays, sample_time=design.sample_time)
    write_text(output / "rdob.json", json_text)
    write_text(output / "rdob_coefficients.h", "\n\n".join(header))
    report = [
        "# 六阶输入等效 RDOB 配平点设计", "",
        f"- L1/L2: {model.geometry.leg_lengths.tolist()} m",
        f"- 采样周期: {design.sample_time:g} s",
        f"- 六路带宽: {metadata['bandwidth_hz']} Hz",
        "- 离散 pole/endpoint weight/input gain/velocity scale 均按六通道数组导出。",
        "- 输入顺序: " + ", ".join(config.INPUT_NAMES),
        "- 当前输出仅适用于此固定配平点，不是腿长调度拟合。",
        "- 连续误差极点为 -lambda；离散系数采用梯形积分，不是联合系统精确 ZOH。",
        "- D0 为零。" if metadata["damping_is_zero"] else "- D0 使用调用者提供的阻尼矩阵。",
        "- 观测全部六路；Tw/Tp/FL 补偿开关与增益在观测器外独立设置。",
        "- 左伪逆仅消去 ker(S0.T) 中的广义力；一般外力仍有输入空间投影。",
        "- 输入使用总广义力。主动腿力应加当前气弹簧模型力后再减总配平值。",
        "- 初始化 z = initial_d - Gamma*v；切换系数需重新基准化。",
        "- 带宽是离线起点，未验证硬件噪声、延迟、接触切换或闭环稳定裕度。",
        "", "## 数值诊断", "", "```json",
        json.dumps(metadata["diagnostics"], indent=2), "```",
    ]
    write_text(output / "rdob_report.md", "\n".join(report))
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("schedule", "node"), default="schedule",
                        help="default: independent L1/L2 schedule; node: single trim")
    parser.add_argument("--l1", type=float, default=None, help="single-node mode only")
    parser.add_argument("--l2", type=float, default=None, help="single-node mode only")
    parser.add_argument("--min-length", type=float, default=config.PARAMETERS.min_leg_length)
    parser.add_argument("--max-length", type=float, default=config.PARAMETERS.max_leg_length)
    parser.add_argument("--step", type=float, default=0.02)
    parser.add_argument("--fit-tolerance", type=float, default=0.005,
                        help="maximum per-matrix relative Frobenius error; default 0.005 = 0.5%%")
    parser.add_argument("--channel-tolerance", type=float, default=0.01,
                        help="maximum matched input-map relative Frobenius error; default 0.01 = 1%%")
    parser.add_argument("--sample-time", type=float, default=config.SAMPLE_TIME)
    parser.add_argument("--bandwidth-hz", type=float, nargs=6, default=DEFAULT_BANDWIDTH_HZ,
                        metavar="HZ", help="Tw1 Tw2 Tp1 Tp2 FL1 FL2; default matches the deployed LESO disturbance pole")
    parser.add_argument("--damping-diagonal", type=float, nargs=7, default=None,
                        metavar="D", help="optional diagonal D0 in generalized-coordinate order")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "schedule" and (args.l1 is not None or args.l2 is not None):
        parser.error("--l1/--l2 require --mode node; scheduling uses independent --min-length/--max-length bounds")
    try:
        common = dict(bandwidth_hz=np.asarray(args.bandwidth_hz), sample_time=args.sample_time,
                      damping=None if args.damping_diagonal is None else np.diag(args.damping_diagonal))
        if args.mode == "node":
            node = compute_rdob_node(
                config.TRIM_L_BAR if args.l1 is None else args.l1,
                config.TRIM_L_BAR if args.l2 is None else args.l2, **common)
            output = export_rdob(node, args.output_dir)
        else:
            from .scheduling.rdob_schedule import compute_rdob_schedule, export_rdob_schedule
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
