import json
from dataclasses import replace
import shutil
import subprocess

import numpy as np
import pytest

from wheel_leg_lqr.control.rdob import initialize_rdob, reconstruct_disturbance, step_rdob
from wheel_leg_lqr.rdob import DEFAULT_DISTURBANCE_LIMITS, compute_rdob_node, main
from wheel_leg_lqr.scheduling.rdob_schedule import (
    GAMMA_TERMS, TOTAL_CUBIC_TERMS, compute_rdob_schedule, export_rdob_schedule,
    frozen_closed_loop_matrix, render_rdob_schedule_header, runtime_accounting,
    step_scheduled_rdob,
)


@pytest.fixture(scope="module")
def schedule():
    return compute_rdob_schedule()


def test_direct_schedule_covers_independent_lengths_and_preserves_fixed_bq_zeros(schedule):
    assert len(schedule.training_points) == 121
    assert len(schedule.holdout_points) == 164
    assert GAMMA_TERMS == tuple((i, j) for i in range(4) for j in range(4))
    assert TOTAL_CUBIC_TERMS == ((0, 0), (0, 1), (0, 2), (0, 3),
                                 (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0))
    for l1, l2 in ((.18, .18), (.38, .38), (.18, .38), (.33, .21)):
        fitted = schedule.evaluate(l1, l2)
        exact = compute_rdob_node(l1, l2).observer
        assert np.linalg.norm(fitted.gamma-exact.gamma)/np.linalg.norm(exact.gamma) < .005
        assert np.linalg.norm(fitted.position_gain-exact.position_gain)/np.linalg.norm(exact.position_gain) < .005
        np.testing.assert_array_equal(fitted.position_gain[:, :2], 0.)
    assert np.linalg.norm(schedule.evaluate(.20, .24).gamma-schedule.evaluate(.30, .34).gamma) > 1.


def test_direct_fit_and_downstream_contract(schedule):
    assert schedule.fits["gamma"].kind == "tensor_cubic"
    assert schedule.fits["gamma"].coefficient_count == 42*16
    assert schedule.fits["position_gain"].kind == "total_cubic"
    assert schedule.fits["position_gain"].coefficient_count == 30*10
    for fit in schedule.fits.values():
        assert fit.metrics["float32_all"]["max_relative_frobenius"] <= .005
    assert "damping_gain" not in schedule.fits
    assert schedule.diagnostics["max_matched_rate_relative_error"] <= .01
    assert schedule.diagnostics["max_raw_closed_loop_radius"] < 1.
    assert schedule.diagnostics["max_fitted_closed_loop_radius"] < 1.
    cost = runtime_accounting(schedule)
    assert cost["evaluation_multiply_upper_bound"] == 904
    assert cost["evaluation_add_subtract_upper_bound"] == 904
    assert cost["float32_storage_bytes"] == 4020
    assert cost["generated_runtime_struct_bytes"] == 0
    assert cost["runtime_scheduled_matrix_float_bytes"] == 288


def test_piecewise_update_uses_lqr_trim_then_rebases(schedule):
    previous_lengths, next_lengths = (.22, .24), (.31, .34)
    previous, following = schedule.evaluate(*previous_lengths), schedule.evaluate(*next_lengths)
    trim = compute_rdob_node(*previous_lengths).model
    q = trim.geometry.q + np.linspace(-.001, .001, 7)
    v, q_next, v_next = np.linspace(-.01, .01, 7), None, None
    q_next = q + schedule.sample_time*v
    v_next = v + .0001
    estimate = np.arange(6.)*.01
    z = initialize_rdob(previous, v, estimate)
    total_input = trim.equilibrium_input + np.arange(6.)*.05
    expected_z, expected_d = step_rdob(previous, z, q-trim.geometry.q, v,
                                      total_input-trim.equilibrium_input,
                                      q_next-trim.geometry.q, v_next)
    next_z, next_d = step_scheduled_rdob(schedule, z, previous_lengths, next_lengths,
                                         q, v, total_input, q_next, v_next,
                                         trim.geometry.q, trim.equilibrium_input)
    np.testing.assert_allclose(next_d, expected_d, atol=1e-12)
    np.testing.assert_allclose(reconstruct_disturbance(following, next_z, v_next), expected_d)
    assert np.linalg.norm(expected_z-next_z) > .01


def test_piecewise_update_rebases_moving_trim_after_internal_limit(schedule):
    previous_lengths, next_lengths = (.22, .24), (.31, .34)
    previous = schedule.evaluate(*previous_lengths)
    following = schedule.evaluate(*next_lengths)
    trim = compute_rdob_node(*previous_lengths).model
    old_trim_velocity = np.array([1.0, 6.5, 0.0, 0.0, 0.3, 0.0, 0.0])
    new_trim_velocity = np.array([-0.5, -2.0, 0.0, 0.0, -0.1, 0.0, 0.0])
    position = trim.geometry.q.copy()
    velocity = old_trim_velocity.copy()
    limits = np.asarray(DEFAULT_DISTURBANCE_LIMITS)
    z = initialize_rdob(previous, velocity-old_trim_velocity, 2.0*limits)

    next_z, estimate = step_scheduled_rdob(
        schedule, z, previous_lengths, next_lengths,
        position, velocity, trim.equilibrium_input,
        position, velocity, trim.geometry.q, trim.equilibrium_input,
        previous_trim_velocity=old_trim_velocity,
        next_trim_velocity=new_trim_velocity,
    )

    np.testing.assert_allclose(estimate, limits)
    np.testing.assert_allclose(
        next_z,
        limits-following.gamma @ (velocity-new_trim_velocity),
    )
    np.testing.assert_allclose(
        reconstruct_disturbance(
            following, next_z, velocity-new_trim_velocity,
        ),
        limits,
    )


def test_closed_loop_validation_matrix_matches_explicit_step(schedule):
    from wheel_leg_lqr.control.lqr import discretize_state_space, solve_discrete_lqr
    node, observer = compute_rdob_node(.26, .32), schedule.evaluate(.26, .32)
    ad, bd = discretize_state_space(node.model.A, node.model.B)
    gain, _, _ = solve_discrete_lqr(ad, bd)
    compensation = np.diag([1., 1., 0., 0., 1., 1.])
    joint = frozen_closed_loop_matrix(ad, bd, gain, observer, compensation)
    state, z = np.linspace(-.01, .01, 14), np.linspace(-.02, .02, 6)
    actual_u = -gain @ state-compensation @ reconstruct_disturbance(observer, z, state[1::2])
    next_state = ad @ state+bd @ actual_u
    next_z, _ = step_rdob(observer, z, state[::2], state[1::2], actual_u, next_state[::2], next_state[1::2])
    np.testing.assert_allclose(joint @ np.concatenate((state, z)), np.concatenate((next_state, next_z)), atol=1e-12)


@pytest.mark.parametrize("lengths", [(.17, .28), (.28, .39), (float("nan"), .28)])
def test_invalid_schedule_points_are_rejected(schedule, lengths):
    with pytest.raises(ValueError):
        schedule.evaluate(*lengths)


def test_invalid_grid_and_fixed_structure_failure(monkeypatch):
    with pytest.raises(ValueError, match="divide"):
        compute_rdob_schedule(step=.03)
    with pytest.raises(ValueError, match="positive"):
        compute_rdob_schedule(step=0)
    import wheel_leg_lqr.scheduling.rdob_schedule as module
    node = compute_rdob_node(.28, .28)
    def changed_node(l1, l2, **kwargs):
        poles = node.observer.step_poles.copy()
        poles[0] += (l1-l2)*.01
        return replace(node, observer=replace(node.observer, step_poles=poles))
    monkeypatch.setattr(module, "compute_rdob_node", changed_node)
    with pytest.raises(ValueError, match="fixed RDOB structure violated"):
        compute_rdob_schedule(step=.1, random_holdouts=0)


def test_cli_defaults_to_direct_schedule(monkeypatch, tmp_path, schedule):
    import wheel_leg_lqr.scheduling.rdob_schedule as module
    calls = []
    def compute(**kwargs):
        calls.append(kwargs)
        return schedule
    monkeypatch.setattr(module, "compute_rdob_schedule", compute)
    main(["--output-dir", str(tmp_path)])
    assert calls[0]["relative_tolerance"] == .005
    assert calls[0]["matched_rate_tolerance"] == .01
    assert np.asarray(calls[0]["bandwidth_hz"]).shape == (6,)
    assert (tmp_path/"rdob_schedule.h").exists()
    independent = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    main(["--output-dir", str(tmp_path/"independent"), "--bandwidth-hz",
          *(str(value) for value in independent)])
    np.testing.assert_array_equal(calls[1]["bandwidth_hz"], independent)
    with pytest.raises(SystemExit):
        main(["--l1", ".28", "--output-dir", str(tmp_path)])


@pytest.mark.parametrize("with_damping", [False, True])
def test_schedule_export_is_deterministic_and_cpp_matches_float32(schedule, tmp_path, with_damping):
    if with_damping:
        schedule = compute_rdob_schedule(damping=np.eye(7)*.1)
    export_rdob_schedule(schedule, tmp_path)
    before = (tmp_path/"rdob_schedule.h").read_bytes()
    export_rdob_schedule(schedule, tmp_path)
    assert before == (tmp_path/"rdob_schedule.h").read_bytes()
    metadata = json.loads((tmp_path/"rdob_schedule.json").read_text(encoding="utf-8"))
    assert metadata["trim_source"].startswith("caller-provided LQR")
    header = before.decode("utf-8")
    assert ("kDampingGainCoefficients" in header) == with_damping
    assert "struct Coefficients" not in header
    assert "inline bool evaluate(" not in header
    assert metadata["step_constants"]["layout"].startswith("six-channel arrays")
    with np.load(tmp_path/"rdob_schedule.npz") as saved:
        for key, name in (
            ("rates", "kRates"),
            ("step_poles", "kStepPoles"),
            ("step_endpoint_weight", "kStepEndpointWeight"),
            ("step_input_gain", "kStepInputGain"),
            ("step_velocity_scale", "kStepVelocityScale"),
        ):
            block = header.split("static constexpr float " + name + "[")[1].split(";")[0].split("=", 1)[1]
            actual = np.asarray([
                token.removesuffix("f")
                for token in block.replace("{", " ").replace("}", " ").replace(",", " ").split()
            ], dtype=np.float32)
            assert actual.shape == (6,)
            np.testing.assert_array_equal(actual, saved[key].astype(np.float32))
            np.testing.assert_array_equal(
                saved[key], np.asarray(metadata["arrays"][key], dtype=saved[key].dtype),
            )
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("optional C++11 check requires g++")
    source = tmp_path/"schedule_check.cpp"
    source.write_text(r'''
#include "rdob_schedule.h"
#include <cstdio>
using namespace wheel_leg_rdob_schedule;
static float evaluate_gamma(const float c[4][4], float x1, float x2) {
    const float r0=((c[0][3]*x2+c[0][2])*x2+c[0][1])*x2+c[0][0];
    const float r1=((c[1][3]*x2+c[1][2])*x2+c[1][1])*x2+c[1][0];
    const float r2=((c[2][3]*x2+c[2][2])*x2+c[2][1])*x2+c[2][0];
    const float r3=((c[3][3]*x2+c[3][2])*x2+c[3][1])*x2+c[3][0];
    return ((r3*x1+r2)*x1+r1)*x1+r0;
}
static float evaluate_total_cubic(const float c[10], float x1, float x2) {
    const float r0=((c[3]*x2+c[2])*x2+c[1])*x2+c[0];
    const float r1=(c[6]*x2+c[5])*x2+c[4];
    const float r2=c[8]*x2+c[7];
    return ((c[9]*x1+r2)*x1+r1)*x1+r0;
}
int main() {
    const float points[][2]={{.18f,.18f},{.38f,.38f},{.23f,.34f},{.34f,.23f}};
    for (const auto& point : points) {
        const float center=0.5f*(kLowerLength+kUpperLength);
        const float inverse_half=1.0f/(0.5f*(kUpperLength-kLowerLength));
        const float x1=(point[0]-center)*inverse_half;
        const float x2=(point[1]-center)*inverse_half;
        for (unsigned n=0; n<42; ++n)
            std::printf("%.9g ", static_cast<double>(evaluate_gamma(kGammaCoefficients[n],x1,x2)));
        for (unsigned n=0; n<30; ++n)
            std::printf("%.9g ", static_cast<double>(evaluate_total_cubic(kPositionGainCoefficients[n],x1,x2)));
        std::puts("");
    }
}
''', encoding="utf-8")
    executable = tmp_path/"schedule_check.exe"
    subprocess.run([compiler, "-std=c++11", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                    str(source), "-o", str(executable)], check=True, capture_output=True, text=True, timeout=60)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=10)
    actual = np.fromstring(result.stdout, sep=" ", dtype=np.float32).reshape(4, -1)
    for row, lengths in zip(actual, ((.18, .18), (.38, .38), (.23, .34), (.34, .23))):
        point = schedule.evaluate(*lengths, dtype=np.float32)
        expected = np.concatenate((point.gamma.reshape(-1), point.position_gain[:, 2:].reshape(-1)))
        np.testing.assert_array_equal(row, expected)
