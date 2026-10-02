"""Ten forward-acceleration coefficients, scheduled by a total-degree cubic.

The existing static model owns the trims and D0=0 linearization.  Only the
forward acceleration row is exported; LQR/RDOB continue to own runtime q0/u0.
Measured-length scheduling makes delta_L_bar=0, so its coefficient is omitted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .. import resolve_output_dir
from ..artifacts.io import write_text
from ..core import config
from ..core.model import StaticLinearization, linearize_static_trim
from .polynomial import (
    TOTAL_CUBIC_TERMS, evaluate_total_cubic_polynomial,
    fit_total_cubic_coefficients,
)


COEFFICIENT_NAMES = (
    "k_theta", "k_phi", "k_beta1", "k_beta2",
    "b_w1", "b_w2", "b_p1", "b_p2", "b_f1", "b_f2",
)
STATE_POSITION_COLUMNS = np.array([2, 3, 5, 6])  # theta, phi, beta1, beta2
MIRROR_CHANNELS = np.array([0, 1, 3, 2, 5, 4, 7, 6, 9, 8])
MIRROR_SIGNS = np.array([1, -1, 1, 1, 1, 1, 1, 1, 1, 1])
MIRROR_TERMS = np.array([TOTAL_CUBIC_TERMS.index((j, i))
                         for i, j in TOTAL_CUBIC_TERMS])


def extract_model_acceleration_coefficients(model: StaticLinearization) -> np.ndarray:
    """Solve M0.T*r=e_x for the inverse's first row, without forming an inverse."""
    selector = np.zeros(config.N_Q)
    selector[0] = 1.0
    row = np.linalg.solve(model.mass.T, selector)
    state = -row @ model.stiffness
    if np.max(np.abs(state[:2])) > 1e-8 or np.any(model.A[1, 1::2] != 0.0):
        raise ValueError("forward model requires H[x,psi]=0 and D0=0")
    return np.concatenate((state[STATE_POSITION_COLUMNS], row @ model.input_matrix))


def predict_model_acceleration(
    coefficients: np.ndarray,
    position: np.ndarray,
    total_input: np.ndarray,
    trim_position: np.ndarray,
    trim_total_input: np.ndarray,
    *, previous_disturbance: np.ndarray | None = None,
    dtype=np.float32,
) -> float:
    """Compute k*delta_q + b*(u_total-u0_total+d_previous), in m/s².

    Positions have the canonical seven-coordinate order.  Inputs are actual
    total generalized inputs (motor plus passive gas-spring leg forces).
    Trims must come from the SAME LQR/RDOB scheduling point.  No velocities
    enter this D0=0 local model.  Passing no disturbance selects the fallback.
    Continuous measured-length scheduling assumes delta_L_bar=0; only
    theta, phi, beta1, beta2 enter the position term.
    """
    c = np.asarray(coefficients, dtype=dtype).reshape(len(COEFFICIENT_NAMES))
    q = np.asarray(position, dtype=dtype).reshape(config.N_Q)
    q0 = np.asarray(trim_position, dtype=dtype).reshape(config.N_Q)
    u = np.asarray(total_input, dtype=dtype).reshape(config.N_U)
    u0 = np.asarray(trim_total_input, dtype=dtype).reshape(config.N_U)
    du = u - u0
    if previous_disturbance is not None:
        du = du + np.asarray(previous_disturbance, dtype=dtype).reshape(config.N_U)
    return float(np.sum(c[:4] * (q[STATE_POSITION_COLUMNS] - q0[STATE_POSITION_COLUMNS]), dtype=dtype)
                 + np.sum(c[4:] * du, dtype=dtype))


def _normalization_constants(lower: float, upper: float, dtype):
    # Match the existing RDOB mapping, including float32 rounding at each step.
    cast = np.dtype(dtype).type
    low, high = cast(lower), cast(upper)
    center = cast(cast(0.5) * cast(low + high))
    inverse_half = cast(cast(1.0) / cast(cast(0.5) * cast(high - low)))
    return center, inverse_half


@dataclass(frozen=True)
class ModelAccelerationSchedule:
    lower: float
    upper: float
    coefficients: np.ndarray
    coefficients64: np.ndarray
    training_points: np.ndarray
    holdout_points: np.ndarray
    reference_coefficients: np.ndarray
    trim_positions: np.ndarray
    trim_total_inputs: np.ndarray
    kappa_g: np.ndarray
    diagnostics: dict

    def evaluate(self, l1: float, l2: float, *, dtype=np.float32) -> np.ndarray:
        """Return ten coefficients; reject extrapolation outside the fitted domain."""
        for length in (l1, l2):
            if not np.isfinite(length) or not self.lower <= length <= self.upper:
                raise ValueError("leg lengths must be finite and inside the fitted domain")
        cast = np.dtype(dtype).type
        center, inverse_half = _normalization_constants(self.lower, self.upper, dtype)
        x = cast(np.clip((cast(l1) - center) * inverse_half, -1.0, 1.0))
        y = cast(np.clip((cast(l2) - center) * inverse_half, -1.0, 1.0))
        table = self.coefficients if np.dtype(dtype) == np.dtype(np.float32) else self.coefficients64
        return evaluate_total_cubic_polynomial(table, x, y, dtype=dtype)


def _sampling_points(lower: float, upper: float, step: float, random_holdouts: int):
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be positive and finite")
    intervals = int(round((upper - lower) / step))
    if intervals < 3 or not np.isclose(intervals * step, upper - lower, rtol=0, atol=1e-12):
        raise ValueError("step must divide the domain into at least three intervals")
    if random_holdouts < 0:
        raise ValueError("random_holdouts must be nonnegative")
    axis = np.linspace(lower, upper, intervals + 1)
    training = np.array([(l1, l2) for l1 in axis for l2 in axis])
    dense_axis = np.linspace(lower, upper, 2 * intervals + 1)
    # Include both the cell centers and points with just one midpoint coordinate.
    holdout = np.array([(l1, l2) for i, l1 in enumerate(dense_axis)
                        for j, l2 in enumerate(dense_axis) if i % 2 or j % 2])
    random = np.random.default_rng(20261002).uniform(lower, upper, (random_holdouts, 2))
    return training, np.vstack((holdout, random, random[:, ::-1]))


def _metrics(reference: np.ndarray, predicted: np.ndarray, scales: np.ndarray) -> dict:
    error = np.asarray(predicted, dtype=float) - reference
    rms = np.sqrt(np.mean(error**2, axis=0))
    maximum = np.max(np.abs(error), axis=0)
    return {name: {
        "reference_peak": float(scales[i]),
        "rms_absolute": float(rms[i]), "max_absolute": float(maximum[i]),
        "rms_peak_normalized": float(rms[i] / scales[i]),
        "max_peak_normalized": float(maximum[i] / scales[i]),
    } for i, name in enumerate(COEFFICIENT_NAMES)}


def _check_mirror(points: np.ndarray, values: np.ndarray, scales: np.ndarray,
                  tolerance: float) -> dict:
    lookup = {tuple(point): i for i, point in enumerate(points)}
    reflected = values[[lookup[tuple(point[::-1])] for point in points]]
    expected = reflected[:, MIRROR_CHANNELS] * MIRROR_SIGNS
    mirror_error = float(np.max(np.abs(values - expected) / scales))
    diagonal = values[points[:, 0] == points[:, 1]]
    diagonal_errors = [np.abs(diagonal[:, 1]) / scales[1]]
    for left, right in ((2, 3), (4, 5), (6, 7), (8, 9)):
        diagonal_errors.append(np.abs(diagonal[:, left] - diagonal[:, right])
                               / max(scales[left], scales[right]))
    diagonal_error = float(np.max(diagonal_errors))
    if mirror_error > tolerance or diagonal_error > tolerance:
        raise ValueError(f"mirror symmetry failed: {mirror_error=:.6e}, {diagonal_error=:.6e}")
    return {"max_peak_normalized": mirror_error,
            "diagonal_max_peak_normalized": diagonal_error, "tolerance": tolerance}


def compute_model_acceleration_schedule(
    *, step: float = 0.01, random_holdouts: int = 64,
    max_error: float = 0.02, rms_target: float = 0.005,
) -> ModelAccelerationSchedule:
    """Fit ten total cubics, enforce mirror parity, and validate the float32 table.

    Maximum error is a hard export limit.  RMS is an advisory target from the
    supplied design document, reported explicitly when a cubic cannot meet it.
    Errors are normalized by each channel's peak over all reference samples.
    """
    if not np.isfinite(max_error) or max_error <= 0 or not np.isfinite(rms_target) or rms_target <= 0:
        raise ValueError("error limits must be positive and finite")
    lower, upper = config.PARAMETERS.min_leg_length, config.PARAMETERS.max_leg_length
    training, holdout = _sampling_points(lower, upper, step, random_holdouts)
    points = np.vstack((training, holdout))
    values, positions, inputs, offsets = [], [], [], []
    for l1, l2 in points:
        model = linearize_static_trim(l1, l2)
        values.append(extract_model_acceleration_coefficients(model))
        positions.append(model.geometry.q)
        inputs.append(model.equilibrium_input)
        offsets.append(model.geometry.kappa_g)
    values = np.asarray(values)
    scales = np.maximum(np.max(np.abs(values), axis=0), 1e-12)
    # Check the physical model BEFORE constraining the polynomial table.
    raw_symmetry = _check_mirror(points, values, scales, tolerance=1e-7)
    center, inverse_half = _normalization_constants(lower, upper, np.float64)
    normalized = (training - center) * inverse_half
    table, _ = fit_total_cubic_coefficients(normalized[:, 0], normalized[:, 1], values[:len(training)])
    # Project onto the known symmetry: transpose x/y powers and swap channels.
    # This keeps all ten channels and every total-degree <= 3 term.
    mirrored_table = MIRROR_SIGNS[:, None] * table[MIRROR_CHANNELS][:, MIRROR_TERMS]
    table = 0.5 * (table + mirrored_table)
    schedule = ModelAccelerationSchedule(
        lower, upper, table.astype(np.float32), table, training, holdout, values,
        np.asarray(positions), np.asarray(inputs), np.asarray(offsets), {},
    )
    fitted64 = np.asarray([schedule.evaluate(*point, dtype=np.float64) for point in points])
    fitted32 = np.asarray([schedule.evaluate(*point) for point in points])
    diagnostics = schedule.diagnostics
    diagnostics.update(
        raw_mirror=raw_symmetry,
        float32_mirror=_check_mirror(points, fitted32, scales, tolerance=1e-6),
        max_error_limit=max_error, rms_advisory_target=rms_target,
        training_count=len(training), holdout_count=len(holdout),
        error_normalization="per-channel peak absolute reference value over all samples; floor 1e-12",
        float64_all=_metrics(values, fitted64, scales),
        float32_all=_metrics(values, fitted32, scales),
        float32_training=_metrics(values[:len(training)], fitted32[:len(training)], scales),
        float32_holdout=_metrics(values[len(training):], fitted32[len(training):], scales),
        float32_rounding=_metrics(fitted64, fitted32, scales),
    )
    failed = [name for name, metrics in diagnostics["float32_all"].items()
              if metrics["max_peak_normalized"] > max_error]
    if failed:
        raise ValueError("float32 maximum fit error exceeds export limit: " + ", ".join(failed))
    diagnostics["maximum_error_limit_passed"] = True
    diagnostics["rms_target_exceeded_channels"] = [
        name for name in COEFFICIENT_NAMES
        if max(diagnostics[key][name]["rms_peak_normalized"]
               for key in ("float32_all", "float32_training", "float32_holdout")) > rms_target
    ]
    return schedule


def render_model_acceleration_header(schedule: ModelAccelerationSchedule) -> str:
    """Generate only uppercase constants and a 10x10 float coefficient table."""
    center, inverse_half = _normalization_constants(schedule.lower, schedule.upper, np.float32)
    lines = [
        "// Generated ground-contact forward model acceleration schedule; C++11, float32.",
        "// Total degree <= 3; terms: 1,y,y2,y3,x,xy,xy2,x2,x2y,x3.",
        "// x/y = (L1/L2 - MODEL_ACCEL_LENGTH_CENTER) * MODEL_ACCEL_LENGTH_INV_SCALE.",
        "// Clamp normalized x/y to [-1,1] at domain endpoints; do not extrapolate.",
        "// q0 and TOTAL u0 come from the SAME LQR/RDOB trim, not this header.",
        "// Rows: k_theta,k_phi,k_beta1,k_beta2,b_w1,b_w2,b_p1,b_p2,b_f1,b_f2.",
        "// Continuous measured-length scheduling: delta_L_bar=0; no length-deviation channel.",
        "// Applied inputs include passive gas-spring forces; d is the previous sample.",
        "#pragma once", "", "namespace wheel_leg_model_acceleration {", "",
        f"static constexpr float MODEL_ACCEL_LENGTH_MIN = {float(np.float32(schedule.lower)):.9e}f;",
        f"static constexpr float MODEL_ACCEL_LENGTH_MAX = {float(np.float32(schedule.upper)):.9e}f;",
        f"static constexpr float MODEL_ACCEL_LENGTH_CENTER = {float(center):.9e}f;",
        f"static constexpr float MODEL_ACCEL_LENGTH_INV_SCALE = {float(inverse_half):.9e}f;",
        "", "static constexpr float MODEL_ACCEL_COEF[10][10] = {",
    ]
    for name, row in zip(COEFFICIENT_NAMES, schedule.coefficients):
        lines.append(f"    // {name}")
        lines.append("    {" + ", ".join(f"{float(value):.9e}f" for value in row) + "},")
    lines.extend(("};", "", "} // namespace wheel_leg_model_acceleration"))
    return "\n".join(lines)


def _render_report(schedule: ModelAccelerationSchedule) -> str:
    d = schedule.diagnostics
    lines = [
        "# 前向模型加速度调度生成报告", "",
        f"调度域：`[{schedule.lower:g}, {schedule.upper:g}]² m²`；训练 {len(schedule.training_points)} 点，"
        f"独立验证 {len(schedule.holdout_points)} 点（加密网格与固定种子随机点及其镜像）。",
        "", "10 个系数均采用二维总阶 ≤ 3 的 10 项多项式，共 100 个 float32，系数表 400 字节。",
        "项序：`1, y, y², y³, x, xy, xy², x², x²y, x³`。未使用三次全张量拟合。", "",
        "误差按各通道在全部参考样本上的峰值归一；不是逐点相对误差，也不是加速度输出误差。"
        "原始绝对误差、参考峰值与 float64/float32 对照见 JSON。", "",
        "| 系数 | 训练 RMS (%) | 验证 RMS (%) | 全部最大误差 (%) | float32 额外最大误差 (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in COEFFICIENT_NAMES:
        train = d["float32_training"][name]["rms_peak_normalized"]
        holdout = d["float32_holdout"][name]["rms_peak_normalized"]
        maximum = d["float32_all"][name]["max_peak_normalized"]
        rounding = d["float32_rounding"][name]["max_peak_normalized"]
        lines.append(f"| {name} | {100*train:.6f} | {100*holdout:.6f} | {100*maximum:.6f} | {100*rounding:.6f} |")
    lines.extend(("", f"全部通道通过最大误差硬门限 `{100*d['max_error_limit']:g}%`。"))
    missed = d["rms_target_exceeded_channels"]
    if missed:
        lines.extend(("", f"**RMS 建议目标未全部达到**：`{', '.join(missed)}` 超过文档建议的 "
                      f"`{100*d['rms_advisory_target']:g}%`。保持用户要求的最高总阶三次，并明确保留此结果；"
                      "是否满足融合精度需求仍取决于运行状态和输入扰动。"))
    else:
        lines.extend(("", f"全部通道达到 RMS 建议目标 `{100*d['rms_advisory_target']:g}%`。"))
    lines.extend((
        "", "## 镜像与对称线检查", "",
        f"原始模型镜像最大峰值归一误差：`{d['raw_mirror']['max_peak_normalized']:.6e}`；",
        f"float32 求值镜像最大误差：`{d['float32_mirror']['max_peak_normalized']:.6e}`；",
        f"float32 对称线最大误差：`{d['float32_mirror']['diagonal_max_peak_normalized']:.6e}`。",
        "全部自动断言通过。先检查原始模型，再对系数表施加镜像约束；未用拟合约束掩盖模型不对称。",
        "", "## 工作点与输入约定", "",
        "复用 `core.model.linearize_static_trim`，D0=0；φ 是机体 roll，β1/β2 是虚拟腿摆角。",
        "θ0=φ0=0，L_bar0=(L1+L2)/2，βi0 由现有静力学配平计算，通常不为零。",
        "输入为实际施加的总广义输入，包含主动腿力与气弹簧力；先组成总输入，再减总配平输入 u0。",
        "可选 RDOB 输入为上一拍扰动估计 d(k−1)；不传则采用原始总输入。",
        "当调度变量每拍取实际 L1/L2 且 L_bar 也取其均值时，δL_bar=0；"
        "因此不提取、拟合或计算腿长偏差通道；模型保留 4 个状态项和 6 个输入项。",
        "", "NPZ 按 `training_points` 后接 `holdout_points` 保存参考系数、q0、总 u0、"
        "主动 u0、气弹簧力及 κg；C++ 头文件只保存系数和归一化常量，不另拟合配平量。",
        "", "这些检查验证静态线性化与调度拟合；不代表实机动态融合或 KF/RDOB 闭环验证。",
    ))
    return "\n".join(lines)


def export_model_acceleration_schedule(schedule: ModelAccelerationSchedule,
                                       output_dir: str | Path) -> Path:
    output = resolve_output_dir(output_dir)
    points = np.vstack((schedule.training_points, schedule.holdout_points))
    gas_forces = np.array([[config.compute_gas_spring_force(length) for length in pair]
                           for pair in points])
    active_trim = schedule.trim_total_inputs.copy()
    active_trim[:, 4:] -= gas_forces
    arrays = {
        "coefficients": schedule.coefficients, "coefficients64": schedule.coefficients64,
        "terms": np.asarray(TOTAL_CUBIC_TERMS),
        "training_points": schedule.training_points, "holdout_points": schedule.holdout_points,
        "reference_coefficients": schedule.reference_coefficients,
        "trim_positions": schedule.trim_positions, "trim_total_inputs": schedule.trim_total_inputs,
        "trim_active_inputs": active_trim, "trim_gas_spring_forces": gas_forces,
        "kappa_g": schedule.kappa_g,
    }
    center, inverse_half = _normalization_constants(schedule.lower, schedule.upper, np.float32)
    parameters = {name: value.tolist() if isinstance(value, np.ndarray) else value
                  for name, value in asdict(config.PARAMETERS).items()}
    metadata = {
        "schema_version": 2, "model": "ddot_x = k*delta_q_active + b*(u_total-u0_total+d_previous); D0=0; delta_L_bar=0",
        "fit": "total_degree_cubic", "maximum_total_degree": 3,
        "domain_m": [[schedule.lower, schedule.upper]] * 2,
        "normalization_float32": {"center": float(center), "inverse_half_range": float(inverse_half)},
        "coefficient_names": list(COEFFICIENT_NAMES), "terms": [list(term) for term in TOTAL_CUBIC_TERMS],
        "coefficients_float32": schedule.coefficients.tolist(),
        "table_shape": list(schedule.coefficients.shape), "table_bytes": schedule.coefficients.nbytes,
        "position_order": list(config.STATE_NAMES[::2]), "input_order": list(config.INPUT_NAMES),
        "active_position_columns": STATE_POSITION_COLUMNS.tolist(),
        "active_position_order": [config.STATE_NAMES[2 * i] for i in STATE_POSITION_COLUMNS],
        "length_deviation_assumption": "zero: continuous scheduling at measured L1/L2 and their mean",
        "trim_source": "existing core.model static trim; runtime SAME LQR/RDOB q0 and total u0",
        "trim_data_order": "training_points, then holdout_points; detailed workpoints are in NPZ",
        "input_convention": "actual total input including passive gas force; subtract total u0 once",
        "rdob_convention": "optional previous-sample input-equivalent disturbance, not same-sample feedback",
        "physical_parameters": parameters,
        "leg_com_lwp_poly": list(config.LEG_COM_LWP_POLY),
        "leg_com_x_poly": list(config.LEG_COM_X_POLY),
        "gas_spring_force_poly": list(config.GAS_SPRING_FORCE_POLY_COEF),
        "diagnostics": schedule.diagnostics,
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "model_acceleration_schedule.npz", **arrays,
             lower=schedule.lower, upper=schedule.upper,
             normalization_center=center, normalization_inverse_half=inverse_half)
    write_text(output / "model_acceleration_schedule.json",
               json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))
    write_text(output / "model_acceleration_schedule.h", render_model_acceleration_header(schedule))
    write_text(output / "model_acceleration_schedule_report.md", _render_report(schedule))
    return output
