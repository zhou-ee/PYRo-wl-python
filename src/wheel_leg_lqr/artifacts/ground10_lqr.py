"""Ten-state exports with their own namespace, dimensions and model contract."""

import json

import numpy as np

from ..models.ground10.config import COORDINATE_NAMES, INPUT_NAMES, STATE_NAMES
from .cpp import cpp_array
from .io import resolve_output_dir, write_text
from .metadata import artifact_metadata


def _metadata(parameters, settings):
    return {
        **artifact_metadata(
            parameters,
            sample_time=settings.sample_time,
            q_diag=settings.q_diag,
            r_diag=settings.r_diag,
        ),
        "model": "ground10_fixed_length_mean_hip_height",
        "coordinate_names": list(COORDINATE_NAMES),
        "state_names": list(STATE_NAMES),
        "input_names": list(INPUT_NAMES),
        "lengths_are_external_parameters": True,
        "assumptions": [
            "L1/L2 frozen before differentiation; no roll or axial dynamics",
            "independent planar legs with mean hip height; local balance approximation",
            "kinematic closures and inverse matrices from the 14-state model are not used",
        ],
    }


def render_header(result):
    p = result.parameters
    degree = result.gain_coefficients.shape[-1] - 1
    beta_degree = len(result.beta_coefficients) - 1
    lines = [
        "// Independent fixed-length, mean-hip-height ground balance approximation.",
        "// States: " + ", ".join(STATE_NAMES),
        "// Inputs: " + ", ".join(INPUT_NAMES),
        "// L1/L2 are external schedule parameters. Recompute for the selected COM mode.",
        f"// Physical fingerprint: {p.fingerprint()}",
        "// Apply u = u0 - K * (state - trim_state). Torques are in N*m.",
        "#pragma once",
        "#include <cmath>",
        "namespace wheel_leg_ground10 {",
        "static constexpr int kStateDim = 10;",
        "static constexpr int kInputDim = 4;",
        f"static constexpr int kDegree = {degree};",
        f"static constexpr int kBetaDegree = {beta_degree};",
        f"static constexpr float kSampleTime = {result.settings.sample_time:.9e}f;",
        f"static constexpr float kLowerLength = {p.min_leg_length:.9e}f;",
        f"static constexpr float kUpperLength = {p.max_leg_length:.9e}f;",
        f"static constexpr float kLengthCenter = {(p.min_leg_length + p.max_leg_length) / 2:.9e}f;",
        f"static constexpr float kLengthScale = {(p.max_leg_length - p.min_leg_length) / 2:.9e}f;",
        cpp_array("kGainCoefficients", result.gain_coefficients),
        cpp_array("kInputCoefficients", result.input_coefficients),
        cpp_array("kBetaCoefficients", result.beta_coefficients),
        "inline float clampLength(float length) {",
        "    return std::fmax(kLowerLength, std::fmin(kUpperLength, length));",
        "}",
        "inline float normalize(float length) {",
        "    return std::fmax(-1.0f, std::fmin(1.0f, (length-kLengthCenter)/kLengthScale));",
        "}",
        "inline void evaluateGain(float L1, float L2, float (&gain)[kInputDim][kStateDim]) {",
        "    const float x = normalize(L1), y = normalize(L2);",
        "    for (int i=0; i<kInputDim; ++i) for (int j=0; j<kStateDim; ++j) {",
        "        float value = 0.0f;",
        "        for (int a=kDegree; a>=0; --a) {",
        "            float row = 0.0f;",
        "            for (int b=kDegree; b>=0; --b) row = row*y + kGainCoefficients[i][j][a][b];",
        "            value = value*x + row;",
        "        }",
        "        gain[i][j] = value;",
        "    }",
        "}",
        "inline float evaluateBeta(float length) {",
        "    length = clampLength(length);",
        "    float beta = 0.0f;",
        "    for (int a=kBetaDegree; a>=0; --a) beta = beta*length + kBetaCoefficients[a];",
        "    return beta;",
        "}",
        "inline void evaluateTrim(float L1, float L2, float (&state)[kStateDim], float (&input)[kInputDim]) {",
        "    const float x = normalize(L1), y = normalize(L2);",
        "    for (int i=0; i<kStateDim; ++i) state[i] = 0.0f;",
        "    state[6] = evaluateBeta(L1); state[8] = evaluateBeta(L2);",
        "    for (int i=0; i<kInputDim; ++i) {",
        "        float value = 0.0f;",
        "        for (int a=kDegree; a>=0; --a) {",
        "            float row = 0.0f;",
        "            for (int b=kDegree; b>=0; --b) row = row*y + kInputCoefficients[i][a][b];",
        "            value = value*x + row;",
        "        }",
        "        input[i] = value;",
        "    }",
        "}",
        "} // namespace wheel_leg_ground10",
    ]
    return "\n".join(lines)


def save_result(result, output_dir):
    output_dir = resolve_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text(output_dir / "ground10_lqr_coef.h", render_header(result))
    nodes = result.nodes
    np.savez_compressed(
        output_dir / "ground10_gain_nodes.npz",
        grid=result.grid,
        mass=np.stack([n.linearization.mass for n in nodes]),
        A=np.stack([n.linearization.A for n in nodes]),
        B=np.stack([n.linearization.B for n in nodes]),
        q0=np.stack([n.linearization.geometry.q for n in nodes]),
        u0=np.stack([n.linearization.equilibrium_input for n in nodes]),
        gains=np.stack([n.gain for n in nodes]),
        gain_coefficients=result.gain_coefficients,
        input_coefficients=result.input_coefficients,
        beta_coefficients=result.beta_coefficients,
    )
    report = {
        **_metadata(result.parameters, result.settings),
        "metrics": result.metrics,
    }
    write_text(
        output_dir / "ground10_gain_fit.json",
        json.dumps(report, indent=2, allow_nan=False),
    )
    write_text(
        output_dir / "README.md",
        "\n".join(
            [
                "# 十状态地面 LQR 调度",
                "",
                "状态：`" + ", ".join(STATE_NAMES) + "`。",
                "输入：`" + ", ".join(INPUT_NAMES) + "`。",
                "固定腿长、无 roll，采用独立双腿平均髋高的局部平衡近似。",
                "L1/L2 是调度参数；高度和横滚支撑由本模型之外的机构/控制维持。",
                "使用 `u = u0 - K*(state-trim_state)`；头文件含 float32 Horner 求值器。",
                "头文件平衡角为三次拟合值，误差见 JSON；NPZ 中 q0/u0 为精确配平值。",
                "",
                "```json",
                json.dumps(result.metrics, indent=2, allow_nan=False),
                "```",
            ]
        ),
    )


def save_node(node, output_dir):
    output_dir = resolve_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        **_metadata(node.parameters, node.settings),
        "L1": node.l1,
        "L2": node.l2,
        "controllability_rank": node.controllability_rank,
        "closed_loop_spectral_radius": float(np.max(np.abs(node.closed_loop_poles))),
        "trim_coordinates": node.linearization.geometry.q.tolist(),
        "trim_input": node.linearization.equilibrium_input.tolist(),
        "mass": node.linearization.mass.tolist(),
        "A": node.linearization.A.tolist(),
        "B": node.linearization.B.tolist(),
        "Ad": node.Ad.tolist(),
        "Bd": node.Bd.tolist(),
        "K": node.gain.tolist(),
    }
    write_text(
        output_dir / "ground10_node.json", json.dumps(report, indent=2, allow_nan=False)
    )
