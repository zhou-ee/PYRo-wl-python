from __future__ import annotations
from .cpp import cpp_array as _cpp_array
from ..models.leg_geometry import compute_gas_spring_force
from .metadata import artifact_metadata
import json
from pathlib import Path
import numpy as np
from .io import resolve_output_dir
from .io import write_text
from ..control.rdob import trapezoidal_step_constants
from ..models.ground import config
from ..scheduling.rdob_node import RdobNode, rdob_diagnostics


def export_rdob(node: RdobNode, output_dir: str | Path) -> Path:
    """Export one frozen trim, with explicit units/timing and no firmware edits."""
    output = resolve_output_dir(output_dir)
    design, model = node.observer, node.model
    gas_force = np.asarray([
        compute_gas_spring_force(length, node.parameters)
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
        **artifact_metadata(node.parameters, sample_time=design.sample_time),
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
