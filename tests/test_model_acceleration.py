import json
import shutil
import subprocess

import numpy as np
import pytest

from wheel_leg_lqr.core import config
from wheel_leg_lqr.core.model import (
    evaluate_gravity, evaluate_input_matrix, evaluate_mass, linearize_static_trim,
)
from wheel_leg_lqr.scheduling.model_acceleration_schedule import (
    COEFFICIENT_NAMES, MIRROR_CHANNELS, MIRROR_SIGNS, MIRROR_TERMS,
    compute_model_acceleration_schedule, export_model_acceleration_schedule,
    extract_model_acceleration_coefficients, predict_model_acceleration,
)
from wheel_leg_lqr.scheduling.polynomial import (
    TOTAL_CUBIC_TERMS, evaluate_total_cubic_polynomial, fit_total_cubic_coefficients,
)


@pytest.fixture(scope="module")
def schedule():
    return compute_model_acceleration_schedule(step=0.02, random_holdouts=8)


@pytest.mark.parametrize("lengths", [(.28, .28), (.21, .35), (.38, .18)])
def test_row_extraction_matches_existing_interleaved_state_model(lengths):
    model = linearize_static_trim(*lengths)
    c = extract_model_acceleration_coefficients(model)
    assert COEFFICIENT_NAMES == ("k_theta", "k_phi", "k_beta1", "k_beta2",
                                "b_w1", "b_w2", "b_p1", "b_p2", "b_f1", "b_f2")
    np.testing.assert_allclose(c, np.r_[model.A[1, [4, 6, 10, 12]], model.B[1]], atol=1e-12)
    np.testing.assert_array_equal(model.A[1, :4], 0.0)
    np.testing.assert_array_equal(model.A[1, 1::2], 0.0)
    assert model.geometry.q[4] == sum(lengths) / 2
    assert np.max(np.abs(model.geometry.q[5:])) > 0.01


def test_total_cubic_uses_only_ten_terms_and_reproduces_known_polynomial():
    assert TOTAL_CUBIC_TERMS == ((0, 0), (0, 1), (0, 2), (0, 3),
                                 (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0))
    assert all(i + j <= 3 for i, j in TOTAL_CUBIC_TERMS)
    points = np.array([(x, y) for x in np.linspace(-1, 1, 4) for y in np.linspace(-1, 1, 4)])
    expected = np.random.default_rng(42).normal(size=(10, 10))
    basis = np.array([[x**i * y**j for i, j in TOTAL_CUBIC_TERMS] for x, y in points])
    actual, fitted = fit_total_cubic_coefficients(points[:, 0], points[:, 1], basis @ expected.T)
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    np.testing.assert_allclose(fitted, basis @ expected.T, atol=1e-13)
    x, y = .47, -.6
    np.testing.assert_allclose(evaluate_total_cubic_polynomial(actual, x, y),
                               expected @ np.array([x**i * y**j for i, j in TOTAL_CUBIC_TERMS]),
                               atol=1e-13)


def test_independent_validation_and_float32_accuracy(schedule):
    assert schedule.coefficients.shape == (10, 10)
    assert schedule.coefficients.dtype == np.float32
    assert schedule.coefficients.nbytes == 400
    assert len(schedule.training_points) == 121
    assert len(schedule.holdout_points) == 336
    training = {tuple(point) for point in schedule.training_points}
    assert all(tuple(point) not in training for point in schedule.holdout_points)
    assert schedule.diagnostics["maximum_error_limit_passed"]
    # A total cubic does not meet the advisory RMS goal for every channel.
    assert "k_phi" in schedule.diagnostics["rms_target_exceeded_channels"]
    for metrics in schedule.diagnostics["float32_all"].values():
        assert metrics["max_peak_normalized"] < .02
    for metrics in schedule.diagnostics["float32_rounding"].values():
        assert metrics["max_peak_normalized"] < 2e-5


def test_mirror_is_preserved_by_table_and_float32_evaluation(schedule):
    np.testing.assert_array_equal(
        schedule.coefficients,
        MIRROR_SIGNS[:, None] * schedule.coefficients[MIRROR_CHANNELS][:, MIRROR_TERMS],
    )
    scale = np.max(np.abs(schedule.reference_coefficients), axis=0)
    for l1, l2 in ((.18, .38), (.231, .357), (.28, .28), (.38, .38)):
        c = schedule.evaluate(l1, l2)
        expected = schedule.evaluate(l2, l1)[MIRROR_CHANNELS] * MIRROR_SIGNS
        assert np.max(np.abs(c - expected) / scale) < 1e-6
        if l1 == l2:
            assert abs(c[1]) < 1e-6 * scale[1]
            np.testing.assert_allclose(c[[2, 4, 6, 8]], c[[3, 5, 7, 9]], atol=1e-6)
    # Equal mean lengths do not imply equal coefficients in asymmetric geometry.
    assert np.linalg.norm(schedule.evaluate(.18, .38) - schedule.evaluate(.28, .28)) > .1


def test_trim_deltas_and_previous_rdob_input(schedule):
    model = linearize_static_trim(.22, .34)
    c = schedule.evaluate(.22, .34)
    q0, u0 = model.geometry.q, model.equilibrium_input
    assert predict_model_acceleration(c, q0, u0, q0, u0) == 0.0
    dq, du, previous_d = np.linspace(-.001, .001, 7), np.arange(6.) * .03, np.arange(6.) * .02
    expected = c[:4] @ dq[[2, 3, 5, 6]] + c[4:] @ (du + previous_d)
    actual = predict_model_acceleration(c, q0 + dq, u0 + du, q0, u0,
                                        previous_disturbance=previous_d, dtype=np.float64)
    assert actual == pytest.approx(expected, abs=1e-12)
    dq[:2] += 100.0
    dq[4] += 100.0  # The continuously scheduled mean-length channel is omitted.
    assert predict_model_acceleration(c, q0 + dq, u0 + du, q0, u0,
                                     previous_disturbance=previous_d, dtype=np.float64) == actual
    fallback = predict_model_acceleration(c, q0 + dq, u0 + du, q0, u0, dtype=np.float64)
    assert actual - fallback == pytest.approx(c[4:] @ previous_d, abs=1e-12)


def test_passive_force_must_be_added_before_subtracting_total_trim(schedule):
    model = linearize_static_trim(.28, .28)
    q0, total_u0 = model.geometry.q, model.equilibrium_input
    gas = np.array([config.compute_gas_spring_force(length) for length in (.28, .28)])
    active_u0 = total_u0.copy()
    active_u0[4:] -= gas
    reconstructed_total = active_u0.copy()
    reconstructed_total[4:] += gas
    c = schedule.evaluate(.28, .28)
    assert predict_model_acceleration(c, q0, reconstructed_total, q0, total_u0) == 0.0
    assert abs(predict_model_acceleration(c, q0, active_u0, q0, total_u0)) > .01


def test_small_signal_coefficients_against_independent_nonlinear_dynamics():
    model = linearize_static_trim(.23, .35)
    c = extract_model_acceleration_coefficients(model)
    q0, u0, kappa = model.geometry.q, model.equilibrium_input, model.geometry.kappa_g

    def forward_acceleration(q, u):
        return np.linalg.solve(evaluate_mass(q, kappa),
                               evaluate_input_matrix(q, kappa) @ u - evaluate_gravity(q, kappa))[0]

    for i in range(10):
        dq, du = np.zeros(7), np.zeros(6)
        step = 1e-5 if i < 4 else 1e-3
        if i < 4:
            dq[[2, 3, 5, 6][i]] = step
        else:
            du[i - 4] = step
        slope = (forward_acceleration(q0 + dq, u0 + du)
                 - forward_acceleration(q0 - dq, u0 - du)) / (2 * step)
        assert slope == pytest.approx(c[i], rel=2e-5, abs=1e-7)


@pytest.mark.parametrize("lengths", [(.17, .28), (.28, .39), (float("nan"), .28)])
def test_extrapolation_is_rejected(schedule, lengths):
    with pytest.raises(ValueError, match="domain"):
        schedule.evaluate(*lengths)


def test_grid_and_hard_export_limit():
    with pytest.raises(ValueError, match="divide"):
        compute_model_acceleration_schedule(step=.03)
    with pytest.raises(ValueError, match="positive"):
        compute_model_acceleration_schedule(step=0)
    with pytest.raises(ValueError, match="maximum fit error"):
        compute_model_acceleration_schedule(step=.05, random_holdouts=0, max_error=1e-8)


def test_asymmetric_physical_model_is_rejected_before_fit(monkeypatch):
    from dataclasses import replace
    import wheel_leg_lqr.scheduling.model_acceleration_schedule as module
    model = linearize_static_trim(.28, .28)

    def asymmetric_model(l1, l2):
        stiffness = model.stiffness.copy()
        stiffness[0, 2] += l1
        return replace(model, stiffness=stiffness)

    monkeypatch.setattr(module, "linearize_static_trim", asymmetric_model)
    with pytest.raises(ValueError, match="mirror symmetry"):
        compute_model_acceleration_schedule(step=.05, random_holdouts=0)


def test_export_keeps_workpoints_and_cpp_data_only_contract(tmp_path, schedule):
    export_model_acceleration_schedule(schedule, tmp_path)
    metadata = json.loads((tmp_path / "model_acceleration_schedule.json").read_text(encoding="utf-8"))
    assert metadata["fit"] == "total_degree_cubic"
    assert metadata["schema_version"] == 2
    assert metadata["table_shape"] == [10, 10]
    assert metadata["table_bytes"] == 400
    assert metadata["active_position_columns"] == [2, 3, 5, 6]
    assert metadata["active_position_order"] == ["theta", "phi", "beta_1", "beta_2"]
    header = (tmp_path / "model_acceleration_schedule.h").read_text(encoding="utf-8")
    assert "MODEL_ACCEL_COEF[10][10]" in header
    assert "struct " not in header and "evaluate(" not in header
    with np.load(tmp_path / "model_acceleration_schedule.npz") as data:
        np.testing.assert_array_equal(data["coefficients"], schedule.coefficients)
        np.testing.assert_array_equal(data["trim_positions"], schedule.trim_positions)
        np.testing.assert_allclose(data["trim_active_inputs"][:, 4:] + data["trim_gas_spring_forces"],
                                   data["trim_total_inputs"][:, 4:])
    assert "RMS 建议目标未全部达到" in (tmp_path / "model_acceleration_schedule_report.md").read_text(encoding="utf-8")


def test_exported_header_evaluates_identically_in_cpp(tmp_path, schedule):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler is not installed")
    export_model_acceleration_schedule(schedule, tmp_path)
    points = ((.18, .38), (.38, .18), (.28, .28), (.231, .357), (.38, .38))
    source = tmp_path / "check_model.cpp"
    rows = ", ".join("{" + f"{l1:.9g}f, {l2:.9g}f" + "}" for l1, l2 in points)
    source.write_text('''#include "model_acceleration_schedule.h"
#include <cstdio>
using namespace wheel_leg_model_acceleration;
int main() {
    const float points[][2] = {''' + rows + '''};
    for (const auto& point : points) {
        float x = (point[0] - MODEL_ACCEL_LENGTH_CENTER) * MODEL_ACCEL_LENGTH_INV_SCALE;
        float y = (point[1] - MODEL_ACCEL_LENGTH_CENTER) * MODEL_ACCEL_LENGTH_INV_SCALE;
        if (x < -1.0f) x = -1.0f; if (x > 1.0f) x = 1.0f;
        if (y < -1.0f) y = -1.0f; if (y > 1.0f) y = 1.0f;
        for (const auto& c : MODEL_ACCEL_COEF) {
            const float row0 = ((c[3]*y+c[2])*y+c[1])*y+c[0];
            const float row1 = (c[6]*y+c[5])*y+c[4];
            const float row2 = c[8]*y+c[7];
            const float value = ((c[9]*x+row2)*x+row1)*x+row0;
            std::printf("%.9g\\n", value);
        }
    }
}
''', encoding="utf-8")
    binary = tmp_path / "check_model.exe"
    subprocess.run([compiler, "-std=c++11", "-O2", "-ffp-contract=off", str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    actual = subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout
    actual = np.array([float(value) for value in actual.split()], dtype=np.float32).reshape(-1, 10)
    expected = np.array([schedule.evaluate(*point) for point in points])
    np.testing.assert_array_equal(actual, expected)
