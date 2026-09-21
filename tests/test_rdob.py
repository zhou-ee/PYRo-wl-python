import json
import re
import shutil
import subprocess

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from wheel_leg_lqr.control.rdob import (
    design_rdob, initialize_rdob, rdob_derivative,
    reconstruct_disturbance, step_rdob, trapezoidal_step_constants,
)
from wheel_leg_lqr.rdob import (
    DEFAULT_BANDWIDTH_HZ, REFERENCE_LESO_DISTURBANCE_POLE, compute_rdob_node,
    export_rdob, main, rdob_diagnostics,
)


@pytest.fixture
def mechanics():
    rng = np.random.default_rng(371)
    factor = rng.normal(size=(7, 7))
    mass = factor.T @ factor + np.eye(7)
    directions, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    inputs = directions[:, :6] @ np.diag([1., 2., 3., 4., 5., 6.])
    stiffness = np.diag(np.arange(1., 8.))
    damping = np.diag(np.arange(0.1, 0.8, 0.1))
    return mass, stiffness, inputs, damping


def make_design(mechanics, dt=0.001):
    mass, stiffness, inputs, damping = mechanics
    return design_rdob(mass, stiffness, inputs, damping=damping,
                       rates_rad_s=np.arange(3., 9.), sample_time=dt)


def test_default_rdob_poles_match_deployed_leso():
    design = compute_rdob_node(0.28, 0.28).observer
    assert len(DEFAULT_BANDWIDTH_HZ) == 6
    np.testing.assert_allclose(
        design.step_poles, REFERENCE_LESO_DISTURBANCE_POLE,
        rtol=0.0, atol=2e-16,
    )


def test_step_constants_keep_independent_six_channel_layout():
    rates = 2.0*np.pi*np.asarray([0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
    poles, weights, input_gains, velocity_scales = (
        trapezoidal_step_constants(rates, 0.001, dtype=np.float32)
    )
    for values in (poles, weights, input_gains, velocity_scales):
        assert values.shape == (6,)
        assert len(np.unique(values)) == 6
    half = (np.float32(0.5)*np.float32(0.001)*rates.astype(np.float32)).astype(np.float32)
    denominator = (np.float32(1.0)+half).astype(np.float32)
    np.testing.assert_array_equal(poles, ((np.float32(1.0)-half)/denominator).astype(np.float32))
    np.testing.assert_array_equal(velocity_scales, (-weights*rates.astype(np.float32)).astype(np.float32))


def test_acceleration_elimination_and_input_channel_separation(mechanics):
    design = make_design(mechanics)
    rng = np.random.default_rng(41)
    q, v = rng.normal(size=(2, 7))
    z, u, disturbance = rng.normal(size=(3, 6))
    acceleration = np.linalg.solve(
        design.mass,
        design.input_matrix @ (u + disturbance)
        - design.damping @ v - design.stiffness @ q,
    )
    actual_derivative = rdob_derivative(design, z, q, v, u) + design.gamma @ acceleration
    estimate = reconstruct_disturbance(design, z, v)
    np.testing.assert_allclose(actual_derivative, design.rates * (disturbance - estimate),
                               atol=2e-12, rtol=2e-12)
    for channel in range(6):
        force = design.input_matrix[:, channel]
        observed = design.gamma @ np.linalg.solve(design.mass, force)
        np.testing.assert_allclose(observed, np.diag(design.rates)[:, channel], atol=1e-13)


def test_only_orthogonal_unmatched_force_is_rejected(mechanics):
    design = make_design(mechanics)
    s = design.input_matrix
    complement = np.eye(7) - s @ design.input_left_inverse
    normal_force = complement @ np.arange(1., 8.)
    assert np.linalg.norm(normal_force) > 0.1
    matched_part = np.arange(1., 7.)
    for force, expected in (
        (normal_force, np.zeros(6)),
        (normal_force + s @ matched_part, design.rates * matched_part),
    ):
        measured_rate = design.gamma @ np.linalg.solve(design.mass, force)
        np.testing.assert_allclose(measured_rate, expected, atol=1e-12)


def test_continuous_observer_tracks_all_six_constant_disturbances(mechanics):
    design = make_design(mechanics)
    rng = np.random.default_rng(99)
    q0, v0 = rng.normal(scale=0.05, size=(2, 7))
    initial = rng.normal(scale=0.1, size=6)
    disturbance = np.array([0.1, -0.2, 0.3, -0.4, 2., -3.])
    z0 = initialize_rdob(design, v0, initial)
    def derivative(t, state):
        q, v, z = state[:7], state[7:14], state[14:]
        applied = np.sin(t * np.arange(1., 7.))
        acceleration = np.linalg.solve(
            design.mass, design.input_matrix @ (applied + disturbance)
            - design.damping @ v - design.stiffness @ q,
        )
        return np.concatenate((v, acceleration, rdob_derivative(design, z, q, v, applied)))
    solution = solve_ivp(derivative, (0., 1.), np.concatenate((q0, v0, z0)),
                         rtol=1e-10, atol=1e-12, t_eval=np.linspace(0., 1., 21))
    assert solution.success
    estimated = solution.y[14:] + design.gamma @ solution.y[7:14]
    expected = disturbance[:, None] + (initial - disturbance)[:, None] * np.exp(
        -design.rates[:, None] * solution.t)
    np.testing.assert_allclose(estimated, expected, rtol=1e-8, atol=2e-9)


def test_discrete_update_input_timing_and_compensation_sign(mechanics):
    mass, _, inputs, _ = mechanics
    design = design_rdob(mass, np.zeros((7, 7)), inputs,
                         rates_rad_s=np.arange(3., 9.), sample_time=0.002)
    disturbance = np.array([0.2, -0.1, -0.3, 0.4, 2., -1.])
    q, v = np.zeros((2, 7))
    z = initialize_rdob(design, v)
    estimate = np.zeros(6)
    for index in range(1000):
        # Actual input includes compensation plus a varying control demand.
        applied = -estimate + np.sin(index * 0.03) * np.arange(0.1, 0.7, 0.1)
        acceleration = np.linalg.solve(mass, inputs @ (applied + disturbance))
        q_next = q + design.sample_time * v + 0.5 * design.sample_time**2 * acceleration
        v_next = v + design.sample_time * acceleration
        z, estimate = step_rdob(design, z, q, v, applied, q_next, v_next)
        q, v = q_next, v_next
    expected = disturbance * (1. - design.step_poles**1000)
    np.testing.assert_allclose(estimate, expected, atol=2e-12)


def test_sampled_implementation_converges_to_continuous_solution(mechanics):
    # Independently propagate the actual coupled mechanics by exact matrix
    # exponential, not by the same integrator used in the observer.
    mass, stiffness, inputs, damping = mechanics
    disturbance = np.arange(1., 7.) * 0.1
    applied = np.arange(1., 7.) * -0.05
    a = np.block([[np.zeros((7, 7)), np.eye(7)],
                  [-np.linalg.solve(mass, stiffness), -np.linalg.solve(mass, damping)]])
    b = np.vstack((np.zeros((7, 6)), np.linalg.solve(mass, inputs)))
    errors = []
    for dt in (0.01, 0.005):
        design = make_design(mechanics, dt)
        transition = expm(np.block([[a, b @ (applied + disturbance)[:, None]],
                                    [np.zeros((1, 15))]]) * dt)
        state = np.linspace(-0.02, 0.02, 14)
        z = initialize_rdob(design, state[7:])
        for _ in range(round(0.5 / dt)):
            next_state = (transition @ np.append(state, 1.))[:14]
            z, estimate = step_rdob(design, z, state[:7], state[7:], applied,
                                    next_state[:7], next_state[7:])
            state = next_state
        expected = disturbance * (1. - np.exp(-design.rates * 0.5))
        errors.append(np.linalg.norm(estimate - expected))
    assert errors[1] < errors[0] * 0.3  # second-order convergence


def test_initialization_and_coefficient_rebase_have_no_estimate_jump(mechanics):
    first = make_design(mechanics)
    mass, stiffness, inputs, damping = mechanics
    second = design_rdob(mass * 1.1, stiffness, inputs, damping=damping,
                         rates_rad_s=17., sample_time=0.001)
    velocity = np.linspace(-0.2, 0.2, 7)
    z = initialize_rdob(first, velocity)
    np.testing.assert_allclose(reconstruct_disturbance(first, z, velocity), 0., atol=1e-14)
    estimate = np.arange(6.)
    new_z = initialize_rdob(second, velocity, estimate)
    np.testing.assert_allclose(reconstruct_disturbance(second, new_z, velocity), estimate)
    with pytest.raises(ValueError):
        first.gamma[0, 0] = 0.


@pytest.mark.parametrize("l1,l2", [(0.28, 0.28), (0.18, 0.38), (0.38, 0.18)])
def test_real_robot_trim_design_and_state_order(l1, l2):
    node = compute_rdob_node(l1, l2)
    design = node.observer
    metrics = rdob_diagnostics(design)
    assert metrics["input_rank"] == 6
    assert metrics["left_inverse_error_inf"] < 1e-11
    assert metrics["matched_rate_identity_error_inf"] < 1e-10
    assert metrics["orthogonal_force_rejection_error_inf"] < 1e-11
    assert metrics["auxiliary_step_spectral_radius"] < 1.
    np.testing.assert_allclose(node.model.B[1::2],
                               np.linalg.solve(design.mass, design.input_matrix))
    np.testing.assert_allclose(node.model.A[1::2, ::2],
                               -np.linalg.solve(design.mass, design.stiffness))


@pytest.mark.parametrize("case", ["rank", "mass", "symmetry", "rates", "nan", "time", "rcond", "damping"])
def test_invalid_design_is_rejected(mechanics, case):
    mass, stiffness, inputs, damping = [value.copy() for value in mechanics]
    kwargs = dict(rates_rad_s=5., sample_time=0.001, damping=damping)
    if case == "rank":
        inputs[:, -1] = inputs[:, 0]
    elif case == "mass":
        mass = -mass
    elif case == "symmetry":
        mass[0, 1] += 1.
    elif case == "rates":
        kwargs["rates_rad_s"] = np.ones(4)
    elif case == "nan":
        kwargs["rates_rad_s"] = np.nan
    elif case == "time":
        kwargs["sample_time"] = 0.
    elif case == "rcond":
        kwargs["pinv_rcond"] = np.nan
    elif case == "damping":
        kwargs["damping"] = np.zeros((6, 6))
    with pytest.raises(ValueError):
        design_rdob(mass, stiffness, inputs, **kwargs)


def test_nonfinite_runtime_measurements_are_rejected(mechanics):
    design = make_design(mechanics)
    velocity = np.zeros(7)
    velocity[3] = np.nan
    with pytest.raises(ValueError):
        initialize_rdob(design, velocity)
    with pytest.raises(ValueError):
        reconstruct_disturbance(design, np.zeros(6), np.zeros(14))


def test_export_cli_and_float32_coefficients(tmp_path):
    main(["--mode", "node", "--l1", "0.24", "--l2", "0.32", "--output-dir", str(tmp_path)])
    metadata = json.loads((tmp_path / "rdob.json").read_text(encoding="utf-8"))
    assert metadata["position_order"] == ["x", "psi", "theta", "phi", "l_bar", "beta_1", "beta_2"]
    assert metadata["input_order"] == ["T_w1", "T_w2", "T_p1", "T_p2", "f_l1", "f_l2"]
    arrays = metadata["arrays"]
    np.testing.assert_allclose(np.asarray(arrays["trim_total_input"])[4:],
                               np.asarray(arrays["trim_active_input"])[4:]
                               + arrays["trim_gas_spring_force"])
    header = (tmp_path / "rdob_coefficients.h").read_text(encoding="utf-8")
    with np.load(tmp_path / "rdob.npz") as saved:
        for key, name in (("gamma", "kGamma"), ("step_position_gain", "kStepPositionGain"),
                          ("step_velocity_gain", "kStepVelocityGain"),
                          ("step_input_gain", "kStepInputGain"), ("step_poles", "kStepPoles"),
                          ("step_endpoint_weight", "kStepEndpointWeight"),
                          ("step_velocity_scale", "kStepVelocityScale")):
            block = header.split("static constexpr float " + name + "[")[1].split(";")[0].split("=", 1)[1]
            numbers = re.findall(r"([-+]?\d+\.\d+e[-+]\d+)f", block)
            actual = np.array(numbers, dtype=np.float32).reshape(saved[key].shape)
            np.testing.assert_array_equal(actual, saved[key].astype(np.float32))
            np.testing.assert_array_equal(saved[key], arrays[key])


def test_cli_requires_explicit_output_directory():
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2


def test_exported_cpp_header_compiles_and_matches_python_update(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("optional C++11 coefficient integration check requires g++")
    node = compute_rdob_node(0.24, 0.32)
    export_rdob(node, tmp_path)
    source = tmp_path / "check.cpp"
    source.write_text(r'''
#include "rdob_coefficients.h"
#include <cstdio>
using namespace wheel_leg_rdob;
int main() {
    float q[7], v[7], z[6] = {}, d[6] = {}, u[6];
    for (int j = 0; j < 7; ++j) {
        q[j] = 0.001f * (j + 1);
        v[j] = 0.002f * (j - 3);
    }
    for (int i = 0; i < 6; ++i) {
        u[i] = 0.01f * (i - 2);
        for (int j = 0; j < 7; ++j) z[i] -= kGamma[i][j] * v[j];
    }
    for (int k = 0; k < 100; ++k) {
        float q_next[7], v_next[7], z_next[6];
        for (int j = 0; j < 7; ++j) {
            q_next[j] = q[j] + kSampleTime * v[j];
            v_next[j] = v[j] + kSampleTime * 0.001f * (j - 2);
        }
        for (int i = 0; i < 6; ++i) {
            z_next[i] = kStepPoles[i]*z[i] + kStepInputGain[i]*u[i];
            for (int j = 0; j < 7; ++j) {
                z_next[i] += kStepPositionGain[i][j]*(q[j]+q_next[j]);
                z_next[i] += kStepVelocityGain[i][j]*(v[j]+v_next[j]);
            }
            d[i] = z_next[i];
            for (int j = 0; j < 7; ++j) d[i] += kGamma[i][j]*v_next[j];
        }
        for (int j = 0; j < 7; ++j) { q[j] = q_next[j]; v[j] = v_next[j]; }
        for (int i = 0; i < 6; ++i) z[i] = z_next[i];
    }
    for (int i = 0; i < 6; ++i) std::printf("%.9g\n", static_cast<double>(d[i]));
}
''', encoding="utf-8")
    executable = tmp_path / "check.exe"
    subprocess.run([compiler, "-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror",
                    str(source), "-o", str(executable)], check=True,
                   capture_output=True, text=True, timeout=60)
    result = subprocess.run([str(executable)], check=True, capture_output=True,
                            text=True, timeout=10)
    q = 0.001 * np.arange(1., 8.)
    v = 0.002 * np.arange(-3., 4.)
    u = 0.01 * np.arange(-2., 4.)
    design = node.observer
    z = initialize_rdob(design, v)
    for _ in range(100):
        q_next = q + design.sample_time * v
        v_next = v + design.sample_time * 0.001 * np.arange(-2., 5.)
        z, estimate = step_rdob(design, z, q, v, u, q_next, v_next)
        q, v = q_next, v_next
    np.testing.assert_allclose(np.fromstring(result.stdout, sep=" "), estimate,
                               rtol=1e-4, atol=1e-4)
