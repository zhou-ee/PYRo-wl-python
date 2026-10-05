from tests.reference_parameters import (
    REFERENCE_AIRBORNE_Q,
    REFERENCE_AIRBORNE_R,
    REFERENCE_GROUND,
    REFERENCE_ROBOT,
)
from dataclasses import replace
import importlib

import numpy as np
import pytest

from wheel_leg_lqr.models.airborne.model import (
    AIRBORNE_STATE_NAMES,
    GLOBAL_ROTATION,
    HIP_INPUT,
    airborne_generalized_mass,
    airborne_rotational_mass,
    compute_leg_force_pitch_effect,
)
from wheel_leg_lqr.models.airborne.reduction import (
    build_shape_model as linearize_airborne,
)
from wheel_leg_lqr.scheduling.airborne_lqr import (
    _controllability_rank,
    compute_airborne_gain_node,
    compute_airborne_schedule,
    evaluate_airborne_gain,
)
from wheel_leg_lqr.artifacts.airborne_lqr import (
    render_airborne_header,
    save_airborne_schedule,
)
from wheel_leg_lqr.models.airborne.config import FULL_STATE_INDICES


def test_free_flight_mass_and_internal_torque_invariant():
    for l1, l2 in ((0.18, 0.38), (0.28, 0.28), (0.38, 0.18)):
        plant = linearize_airborne(
            l1,
            l2,
            parameters=REFERENCE_ROBOT,
            q_diag=REFERENCE_AIRBORNE_Q,
            r_diag=REFERENCE_AIRBORNE_R,
        )
        np.testing.assert_allclose(plant.mass, plant.mass.T, atol=1e-14)
        assert np.min(np.linalg.eigvalsh(plant.mass)) > 0.0
        np.testing.assert_allclose(GLOBAL_ROTATION @ HIP_INPUT, 0.0, atol=1e-14)
        np.testing.assert_allclose(
            GLOBAL_ROTATION @ plant.mass @ plant.B[1::2], 0.0, atol=1e-13
        )
        np.testing.assert_allclose(
            plant.state_projection @ plant.state_lift, np.eye(4), atol=1e-12
        )
        np.testing.assert_allclose(
            plant.state_projection @ plant.B, plant.B_shape, atol=1e-12
        )
    assert AIRBORNE_STATE_NAMES == (
        "beta_1",
        "dbeta_1",
        "beta_2",
        "dbeta_2",
        "theta",
        "dtheta",
    )
    assert FULL_STATE_INDICES == (10, 11, 12, 13, 4, 5)


def test_global_orientation_does_not_change_free_flight_mass():
    # Physical common rotation: beta_i decreases as theta increases.
    delta = 0.35
    reference = airborne_rotational_mass(
        0.24, 0.33, beta_1=0.08, beta_2=-0.06, parameters=REFERENCE_ROBOT
    )
    rotated = airborne_rotational_mass(
        0.24,
        0.33,
        beta_1=0.08 - delta,
        beta_2=-0.06 - delta,
        theta=delta,
        parameters=REFERENCE_ROBOT,
    )
    np.testing.assert_allclose(rotated, reference, atol=1e-12)


def test_forward_body_com_makes_leg_force_pitch_coupling():
    centered = compute_leg_force_pitch_effect(
        0.28, 0.28, body_com_x_from_hip=0.0, parameters=REFERENCE_ROBOT
    )
    forward = compute_leg_force_pitch_effect(
        0.28, 0.28, body_com_x_from_hip=0.03, parameters=REFERENCE_ROBOT
    )
    np.testing.assert_allclose(forward.body_moment_arm, [0.03, 0.03], atol=1e-14)
    np.testing.assert_allclose(
        forward.body_pitch_acceleration_per_newton,
        [0.03 / REFERENCE_ROBOT.body_pitch_inertia] * 2,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        forward.pitch_acceleration_per_newton,
        [0.06898561622458194] * 2,
        rtol=1e-8,
    )
    assert np.all(
        forward.pitch_acceleration_per_newton > centered.pitch_acceleration_per_newton
    )
    np.testing.assert_allclose(
        forward.generalized_mass[:3, :3],
        airborne_rotational_mass(
            0.28,
            0.28,
            parameters=replace(REFERENCE_ROBOT, body_com_from_hip=(0.03, 0.0, 0.032)),
        ),
        atol=1e-12,
    )
    assert airborne_generalized_mass(0.28, 0.28, parameters=REFERENCE_ROBOT).shape == (
        5,
        5,
    )
    # Equal-and-opposite leg forces cancel pitch at this symmetric trim.
    assert forward.pitch_acceleration_per_newton @ np.array(
        [100.0, -100.0]
    ) == pytest.approx(0.0)
    force_map = np.zeros((5, 2))
    force_map[3:, :] = np.eye(2)
    common_rotation = np.array([-1.0, -1.0, 1.0, 0.0, 0.0])
    acceleration_map = np.linalg.solve(forward.generalized_mass, force_map)
    np.testing.assert_allclose(
        common_rotation @ forward.generalized_mass @ acceleration_map,
        [0.0, 0.0],
        atol=1e-12,
    )


def test_only_four_shape_states_can_be_stabilized():
    node = compute_airborne_gain_node(
        0.28,
        0.28,
        parameters=REFERENCE_ROBOT,
        q_diag=REFERENCE_AIRBORNE_Q,
        r_diag=REFERENCE_AIRBORNE_R,
        sample_time=REFERENCE_GROUND.sample_time,
    )
    assert _controllability_rank(node.Ad, node.Bd) == 4
    assert _controllability_rank(node.Ad_shape, node.Bd_shape) == 4
    assert np.max(np.abs(node.shape_poles)) < 1.0
    full_poles = np.linalg.eigvals(node.Ad - node.Bd @ node.gain)
    assert np.count_nonzero(np.isclose(full_poles, 1.0, atol=1e-9)) == 2
    assert np.count_nonzero(np.abs(full_poles) < 1.0 - 1e-6) == 4

    # A small pitch error is corrected by moving the legs, while total
    # angular momentum remains zero throughout the contact-free response.
    state = np.array([0.0, 0.0, 0.0, 0.0, np.deg2rad(10.0), 0.0])
    initial_shape_error = np.linalg.norm(node.linearization.state_projection @ state)
    peak_torque = 0.0
    for _ in range(2000):
        torque = np.clip(-node.gain @ state, -60.0, 60.0)
        peak_torque = max(peak_torque, float(np.max(np.abs(torque))))
        state = node.Ad @ state + node.Bd @ torque
        assert abs(GLOBAL_ROTATION @ node.linearization.mass @ state[1::2]) < 1e-9
    assert peak_torque < 60.0
    assert 0.0 < state[4] < np.deg2rad(10.0)
    assert state[0] < 0.0 and state[2] < 0.0
    assert (
        np.linalg.norm(node.linearization.state_projection @ state)
        < 0.05 * initial_shape_error
    )


def _reachable_rest_angles(node, initial_angles):
    mass = node.linearization.mass
    rotation = GLOBAL_ROTATION
    gamma_row = rotation @ mass / (rotation @ mass @ rotation)
    return np.linalg.solve(
        np.vstack((node.linearization.shape_projection, gamma_row)),
        np.array([0.0, 0.0, gamma_row @ initial_angles]),
    )


def test_airborne_q_shifts_pitch_leg_compromise():
    initial = np.array([0.0, 0.0, np.deg2rad(10.0)])
    default_q = REFERENCE_AIRBORNE_Q.copy()
    pitch_q = default_q.copy()
    pitch_q[4] *= 10.0
    leg_q = default_q.copy()
    leg_q[[0, 2]] *= 10.0
    default = compute_airborne_gain_node(
        0.28,
        0.28,
        q_diag=default_q,
        parameters=REFERENCE_ROBOT,
        r_diag=REFERENCE_AIRBORNE_R,
        sample_time=REFERENCE_GROUND.sample_time,
    )
    pitch_priority = compute_airborne_gain_node(
        0.28,
        0.28,
        q_diag=pitch_q,
        parameters=REFERENCE_ROBOT,
        r_diag=REFERENCE_AIRBORNE_R,
        sample_time=REFERENCE_GROUND.sample_time,
    )
    leg_priority = compute_airborne_gain_node(
        0.28,
        0.28,
        q_diag=leg_q,
        parameters=REFERENCE_ROBOT,
        r_diag=REFERENCE_AIRBORNE_R,
        sample_time=REFERENCE_GROUND.sample_time,
    )
    rest_default = _reachable_rest_angles(default, initial)
    rest_pitch = _reachable_rest_angles(pitch_priority, initial)
    rest_leg = _reachable_rest_angles(leg_priority, initial)
    assert abs(rest_pitch[2]) < abs(rest_default[2]) < abs(rest_leg[2])
    assert abs(rest_pitch[0]) > abs(rest_default[0]) > abs(rest_leg[0])
    assert not np.allclose(default.gain, pitch_priority.gain)
    assert np.max(np.abs(pitch_priority.shape_poles)) < 1.0
    with pytest.raises(ValueError, match="q_diag"):
        linearize_airborne(
            0.28,
            0.28,
            q_diag=[1, 1, 1, 1, 0, 1],
            parameters=REFERENCE_ROBOT,
            r_diag=REFERENCE_AIRBORNE_R,
        )


def test_ground_weights_cannot_change_airborne_gain(monkeypatch):
    baseline = compute_airborne_gain_node(
        0.24,
        0.32,
        parameters=REFERENCE_ROBOT,
        q_diag=REFERENCE_AIRBORNE_Q,
        r_diag=REFERENCE_AIRBORNE_R,
        sample_time=REFERENCE_GROUND.sample_time,
    )
    ground_config = importlib.import_module("wheel_leg_lqr.models.ground.config")
    monkeypatch.setattr(ground_config, "Q_DIAG", ground_config.Q_DIAG * 10.0)
    monkeypatch.setattr(ground_config, "R_DIAG", ground_config.R_DIAG * 10.0)
    independent = compute_airborne_gain_node(
        0.24,
        0.32,
        parameters=REFERENCE_ROBOT,
        q_diag=REFERENCE_AIRBORNE_Q,
        r_diag=REFERENCE_AIRBORNE_R,
        sample_time=REFERENCE_GROUND.sample_time,
    )
    np.testing.assert_allclose(
        independent.linearization.mass, baseline.linearization.mass
    )
    np.testing.assert_allclose(independent.gain, baseline.gain)


@pytest.fixture(scope="module")
def coarse_schedule():
    return compute_airborne_schedule(
        step=0.04,
        parameters=REFERENCE_ROBOT,
        q_diag=REFERENCE_AIRBORNE_Q,
        r_diag=REFERENCE_AIRBORNE_R,
        sample_time=REFERENCE_GROUND.sample_time,
    )


def test_float32_schedule_covers_asymmetric_lengths(coarse_schedule):
    result = coarse_schedule
    assert result.gain_coefficients.shape == (2, 6, 4, 4)
    assert result.validation_node_count > len(result.nodes)
    assert result.max_float32_shape_radius < 1.0
    assert result.max_relative_gain_error < 0.01
    for lengths in ((0.18, 0.38), (0.31, 0.23), (0.38, 0.18)):
        node = compute_airborne_gain_node(
            *lengths,
            parameters=REFERENCE_ROBOT,
            q_diag=REFERENCE_AIRBORNE_Q,
            r_diag=REFERENCE_AIRBORNE_R,
            sample_time=REFERENCE_GROUND.sample_time,
        )
        fitted = evaluate_airborne_gain(
            result.gain_coefficients.astype(np.float32),
            *lengths,
            dtype=np.float32,
            lower=REFERENCE_ROBOT.min_leg_length,
            upper=REFERENCE_ROBOT.max_leg_length,
        )
        assert fitted.shape == (2, 6)
        assert np.linalg.norm(fitted - node.gain) / np.linalg.norm(node.gain) < 0.01
    np.testing.assert_allclose(
        evaluate_airborne_gain(
            result.gain_coefficients,
            0.01,
            0.99,
            lower=REFERENCE_ROBOT.min_leg_length,
            upper=REFERENCE_ROBOT.max_leg_length,
        ),
        evaluate_airborne_gain(
            result.gain_coefficients,
            0.18,
            0.38,
            lower=REFERENCE_ROBOT.min_leg_length,
            upper=REFERENCE_ROBOT.max_leg_length,
        ),
        atol=1e-10,
    )


def test_export_contains_only_airborne_states_and_hip_inputs(coarse_schedule, tmp_path):
    header = render_airborne_header(coarse_schedule)
    assert "namespace airborne_lqr" in header
    assert "BETA_1, DOT_BETA_1, BETA_2, DOT_BETA_2, THETA, DOT_THETA" in header
    assert "T_P1, T_P2" in header
    assert "T_W1" not in header and "F_L1" not in header
    save_airborne_schedule(coarse_schedule, tmp_path)
    archive = np.load(tmp_path / "airborne_lqr_schedule.npz")
    assert FULL_STATE_INDICES == (10, 11, 12, 13, 4, 5)
    np.testing.assert_array_equal(archive["q_diag"], REFERENCE_AIRBORNE_Q)
    np.testing.assert_array_equal(archive["r_diag"], REFERENCE_AIRBORNE_R)
    assert archive["gain_coefficients"].shape == (2, 6, 4, 4)
    assert (tmp_path / "airborne_lqr_fit.json").exists()


def test_schedule_requires_enough_nodes():
    with pytest.raises(ValueError, match="degree"):
        compute_airborne_schedule(
            step=0.1,
            degree=3,
            parameters=REFERENCE_ROBOT,
            q_diag=REFERENCE_AIRBORNE_Q,
            r_diag=REFERENCE_AIRBORNE_R,
            sample_time=REFERENCE_GROUND.sample_time,
        )
