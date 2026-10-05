from __future__ import annotations
from .cpp import cpp_array as _cpp_array
import json
from .metadata import artifact_metadata
import numpy as np
from ..artifacts.io import resolve_output_dir
from ..artifacts.io import write_text
from ..control.rdob import trapezoidal_step_constants
from ..models.ground import config
from ..scheduling.rdob import GAMMA_TERMS, TOTAL_CUBIC_TERMS, runtime_accounting


def render_rdob_schedule_header(schedule):
    """Emit a data-only C++11 float32 contract for caller-owned evaluation."""
    poles, weights, input_gain, velocity_scale = trapezoidal_step_constants(
        schedule.rates, schedule.sample_time, dtype=np.float32,
    )
    lines = [
        "// Generated independent L1/L2 RDOB schedule; C++11, float32.",
        "// Input/channel order: " + ", ".join(config.INPUT_NAMES) + ".",
        "// All rate and step constants remain six-element arrays even when equal.",
        "// Gamma coefficients: [input*7+q][normalized-L1 power][normalized-L2 power].",
        "// Bq coefficients use normalized x=L1 and y=L2 coordinates:",
        "// [input*5+(q-2)][1,y,y2,y3,x,xy,xy2,x2,x2y,x3].",
        "// kStepVelocityScale multiplies Gamma; with D=0, step_v_gain[i][j]",
        "// equals kStepVelocityScale[i]*Gamma[i][j].",
        "// This header intentionally provides no Coefficients struct or evaluator.",
        "#pragma once",
        "namespace wheel_leg_rdob_schedule {",
        f"static constexpr float kLowerLength = {schedule.lower:.9e}f;",
        f"static constexpr float kUpperLength = {schedule.upper:.9e}f;",
        f"static constexpr float kSampleTime = {schedule.sample_time:.9e}f;",
        _cpp_array("kRates", schedule.rates),
        _cpp_array("kStepPoles", poles),
        _cpp_array("kStepEndpointWeight", weights),
        _cpp_array("kStepInputGain", input_gain),
        _cpp_array("kStepVelocityScale", velocity_scale),
        _cpp_array("kGammaCoefficients", schedule.fits["gamma"].coefficients.reshape(42, 4, 4)),
        _cpp_array("kPositionGainCoefficients", schedule.fits["position_gain"].coefficients.reshape(30, 10)),
    ]
    damping = schedule.fits.get("damping_gain")
    if damping is not None:
        lines.append(_cpp_array("kDampingGainCoefficients", damping.coefficients.reshape(42, 10)))
    lines.append("} // namespace wheel_leg_rdob_schedule")
    return "\n\n".join(lines)


def export_rdob_schedule(schedule, output_dir):
    output = resolve_output_dir(output_dir)
    step_poles, step_endpoint_weight, step_input_gain, step_velocity_scale = (
        trapezoidal_step_constants(
            schedule.rates, schedule.sample_time, dtype=np.float32,
        )
    )
    arrays = {
        "rates": schedule.rates,
        "step_poles": step_poles,
        "step_endpoint_weight": step_endpoint_weight,
        "step_input_gain": step_input_gain,
        "step_velocity_scale": step_velocity_scale,
        "damping": schedule.damping,
        "training_points": schedule.training_points,
        "holdout_points": schedule.holdout_points,
        "gamma_terms": np.asarray(GAMMA_TERMS),
        "position_gain_terms": np.asarray(TOTAL_CUBIC_TERMS),
    }
    for name, fit in schedule.fits.items(): arrays[name+"_coefficients"] = fit.coefficients
    matrix_metadata = {name: {"kind": fit.kind, "terms": [list(term) for term in fit.terms],
                              "output_shape": list(fit.output_shape), "coefficient_count": fit.coefficient_count, **fit.metrics}
                       for name, fit in schedule.fits.items()}
    metadata = {**artifact_metadata(schedule.parameters, sample_time=schedule.sample_time), "schema_version": 2, "parameters": ["L1", "L2"], "domain_m": [[schedule.lower, schedule.upper]]*2,
                "fit": "Gamma direct tensor cubic; Bq direct total-degree cubic; trims supplied by LQR",
                 "trim_source": "caller-provided LQR q0 and total u0; RDOB does not fit or export trim arrays",
                 "bandwidth_hz": (schedule.rates/(2*np.pi)).tolist(), "sample_time_s": schedule.sample_time,
                 "position_order": list(config.STATE_NAMES[::2]), "input_order": list(config.INPUT_NAMES),
                 "step_constants": {"layout": "six-channel arrays in input_order, even when values are equal",
                                    "velocity_scale_definition": "-step_endpoint_weight * rates; multiplies Gamma when D=0"},
                 "input_convention": "actual TOTAL generalized input; subtract PREVIOUS LQR trim_total_input",
                "matrices": matrix_metadata, "diagnostics": schedule.diagnostics, "runtime_accounting": runtime_accounting(schedule),
                "arrays": {name: value.tolist() for name, value in arrays.items()}}
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output/"rdob_schedule.npz", **arrays, lower=schedule.lower, upper=schedule.upper, sample_time=schedule.sample_time)
    write_text(output/"rdob_schedule.json", json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))
    write_text(output/"rdob_schedule.h", render_rdob_schedule_header(schedule))
    report = ["# RDOB L1/L2 二维调度", "", "Gamma 使用三次全张量直接拟合；Bq 使用总阶三次直接拟合。",
              "q0 与 total u0 由 LQR 调度提供，RDOB 不重复拟合或导出它们。",
              "生成头文件只包含常量表，不包含 Coefficients 结构体或 evaluate()。", "",
              "| 矩阵 | 多项式 | 系数个数 | float32 最大相对 Frobenius 误差 (%) |", "|---|---|---:|---:|"]
    report += [f"| {name} | {fit.kind} | {fit.coefficient_count} | {100*fit.metrics['float32_all']['max_relative_frobenius']:.6g} |" for name, fit in schedule.fits.items()]
    report += ["", "## 六通道离散常量", "",
               "| 通道 | 带宽 (Hz) | rate (rad/s) | z pole | endpoint weight | input gain | velocity scale |",
               "|---|---:|---:|---:|---:|---:|---:|"]
    bandwidths = schedule.rates/(2*np.pi)
    report += [
        f"| {name} | {bandwidths[i]:.9g} | {schedule.rates[i]:.9g} | "
        f"{step_poles[i]:.9g} | {step_endpoint_weight[i]:.9g} | "
        f"{step_input_gain[i]:.9g} | {step_velocity_scale[i]:.9g} |"
        for i, name in enumerate(config.INPUT_NAMES)
    ]
    report += ["", "## 验证", "", "```json", json.dumps(schedule.diagnostics, indent=2), "```", "", "## 存储与运算计数", "", "```json", json.dumps(runtime_accounting(schedule), indent=2), "```"]
    write_text(output/"rdob_schedule_report.md", "\n".join(report))
    return output
