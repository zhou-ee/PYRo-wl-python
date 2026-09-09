import numpy as np
import pytest

from wheel_leg_lqr.control.leso import (
    DEFAULT_DISTURBANCE_RETENTION_POLE,
    DEFAULT_FL_DISTURBANCE_BANDWIDTH_HZ,
    DEFAULT_FL_STATE_BANDWIDTH_HZ,
    DEFAULT_HIP_STATE_BANDWIDTH_HZ,
    DEFAULT_TORQUE_DISTURBANCE_BANDWIDTH_HZ,
    DEFAULT_UNMATCHED_STATE_POLE,
    DEFAULT_WHEEL_STATE_BANDWIDTH_HZ,
    bounded_disturbance_update,
    build_augmented_model,
    design_leso_correction,
    discrete_pole_from_bandwidth,
    observer_error_matrix,
    observer_error_spectral_radius,
    step_leso_split,
)
from wheel_leg_lqr.core import config
from wheel_leg_lqr.gain_schedule import compute_gain_node
from wheel_leg_lqr.numeric_config import N_U, N_X
from wheel_leg_lqr.scheduling.leso_schedule import INPUT_EXCHANGE, STATE_EXCHANGE


def test_analytic_leso_has_the_requested_augmented_poles():
    node = compute_gain_node(0.23, 0.34)
    correction = design_leso_correction(node.Ad, node.Bd)

    assert np.linalg.matrix_rank(node.Bd) == N_U
    poles = np.sort_complex(
        np.linalg.eigvals(
            observer_error_matrix(
                node.Ad,
                node.Bd,
                correction.state_gain,
                correction.disturbance_gain,
            )
        )
    )
    expected = np.sort_complex(
        np.concatenate(
            [
                np.full(N_X - N_U, correction.unmatched_state_pole),
                correction.state_poles,
                correction.disturbance_poles,
            ]
        )
    )
    np.testing.assert_allclose(poles, expected, atol=2.0e-10, rtol=0.0)
    assert observer_error_spectral_radius(
        node.Ad,
        node.Bd,
        correction.state_gain,
        correction.disturbance_gain,
    ) < 1.0
    np.testing.assert_allclose(
        correction.state_poles[:2],
        discrete_pole_from_bandwidth(DEFAULT_WHEEL_STATE_BANDWIDTH_HZ),
    )
    np.testing.assert_allclose(
        correction.state_poles[2:4],
        discrete_pole_from_bandwidth(DEFAULT_HIP_STATE_BANDWIDTH_HZ),
    )
    np.testing.assert_allclose(
        correction.state_poles[4:],
        discrete_pole_from_bandwidth(DEFAULT_FL_STATE_BANDWIDTH_HZ),
    )
    np.testing.assert_allclose(
        correction.disturbance_poles[:4],
        discrete_pole_from_bandwidth(
            DEFAULT_TORQUE_DISTURBANCE_BANDWIDTH_HZ
        ),
    )
    np.testing.assert_allclose(
        correction.disturbance_poles[4:],
        discrete_pole_from_bandwidth(DEFAULT_FL_DISTURBANCE_BANDWIDTH_HZ),
    )
    assert correction.unmatched_state_pole == DEFAULT_UNMATCHED_STATE_POLE
    assert (
        correction.disturbance_retention_pole
        == DEFAULT_DISTURBANCE_RETENTION_POLE
    )

    fused_state_gain = (
        node.Ad
        - correction.unmatched_state_pole * np.eye(N_X)
        - node.Bd
        @ np.diag(correction.matched_correction_scales)
        @ correction.disturbance_gain
    )
    np.testing.assert_allclose(
        correction.state_gain,
        fused_state_gain,
        atol=2.0e-12,
        rtol=2.0e-12,
    )


def test_split_update_matches_the_augmented_observer_equation():
    node = compute_gain_node(0.21, 0.37)
    correction = design_leso_correction(node.Ad, node.Bd)
    augmented_state, augmented_input, output = build_augmented_model(
        node.Ad, node.Bd
    )
    observer_gain = np.vstack(
        [correction.state_gain, correction.disturbance_gain]
    )
    rng = np.random.default_rng(20260902)
    estimated_state = rng.normal(size=N_X)
    estimated_disturbance = rng.normal(size=N_U)
    measured_state = rng.normal(size=N_X)
    applied_input = rng.normal(size=N_U)

    next_state, next_disturbance = step_leso_split(
        estimated_state,
        estimated_disturbance,
        measured_state,
        applied_input,
        node.Ad,
        node.Bd,
        correction.state_gain,
        correction.disturbance_gain,
    )
    augmented_estimate = np.concatenate(
        [estimated_state, estimated_disturbance]
    )
    expected = (
        augmented_state @ augmented_estimate
        + augmented_input @ applied_input
        + observer_gain @ (measured_state - output @ augmented_estimate)
    )
    np.testing.assert_allclose(
        np.concatenate([next_state, next_disturbance]),
        expected,
        atol=2.0e-13,
        rtol=2.0e-13,
    )


def test_bounded_disturbance_update_limits_shock_and_prevents_windup():
    estimated = np.asarray([0.3, -0.3, 5.9, -5.9, 29.9, -29.9])
    raw_increment = np.asarray([1.0, -1.0, 4.0, -4.0, 20.0, -20.0])
    limits = np.asarray([0.33, 0.33, 6.0, 6.0, 30.0, 30.0])
    maximum_increments = np.asarray(
        [0.005, 0.005, 0.09, 0.09, 0.225, 0.225]
    )

    next_disturbance, accepted_increment = bounded_disturbance_update(
        estimated,
        raw_increment,
        limits,
        maximum_increments,
    )
    retained = DEFAULT_DISTURBANCE_RETENTION_POLE * estimated
    expected = np.clip(
        retained
        + np.clip(raw_increment, -maximum_increments, maximum_increments),
        -limits,
        limits,
    )
    np.testing.assert_allclose(next_disturbance, expected)
    np.testing.assert_allclose(
        accepted_increment, next_disturbance - retained
    )
    assert np.all(np.abs(next_disturbance) <= limits)
    assert np.all(np.abs(accepted_increment) <= maximum_increments + 1.0e-15)


def test_leg_velocity_impact_is_limited_before_reaching_state_correction():
    node = compute_gain_node(0.20, 0.20)
    correction = design_leso_correction(node.Ad, node.Bd)
    residual = np.zeros(N_X)
    residual[9] = 1.0
    raw_increment = correction.disturbance_gain @ residual
    limits = 0.10 * config.INPUT_UPPER_LIMITS
    maximum_increments = (1.0 - correction.disturbance_poles) * limits

    _, accepted_increment = bounded_disturbance_update(
        np.zeros(N_U),
        raw_increment,
        limits,
        maximum_increments,
    )

    # A 1 m/s leg-velocity residual asks for roughly 8.9 N in one update,
    # while firmware may accept only 0.225 N.  The same accepted value is fed
    # to the fused state correction, so the rejected shock cannot bypass it.
    assert abs(raw_increment[4]) > 35.0 * maximum_increments[4]
    assert abs(accepted_increment[4]) == pytest.approx(maximum_increments[4])


@pytest.mark.parametrize("retention", [0.0, 1.0, np.nan])
def test_leso_rejects_invalid_disturbance_retention(retention):
    node = compute_gain_node(0.25, 0.31)
    with pytest.raises(ValueError, match="disturbance_retention_pole"):
        design_leso_correction(
            node.Ad,
            node.Bd,
            disturbance_retention_pole=retention,
        )


def test_analytic_design_preserves_left_right_exchange_symmetry():
    direct = compute_gain_node(0.22, 0.35)
    exchanged = compute_gain_node(0.35, 0.22)
    direct_correction = design_leso_correction(direct.Ad, direct.Bd)
    exchanged_correction = design_leso_correction(exchanged.Ad, exchanged.Bd)

    np.testing.assert_allclose(
        exchanged.Ad,
        STATE_EXCHANGE @ direct.Ad @ STATE_EXCHANGE.T,
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(
        exchanged.Bd,
        STATE_EXCHANGE @ direct.Bd @ INPUT_EXCHANGE.T,
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(
        exchanged_correction.state_gain,
        STATE_EXCHANGE @ direct_correction.state_gain @ STATE_EXCHANGE.T,
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    np.testing.assert_allclose(
        exchanged_correction.disturbance_gain,
        INPUT_EXCHANGE
        @ direct_correction.disturbance_gain
        @ STATE_EXCHANGE.T,
        atol=2.0e-9,
        rtol=2.0e-10,
    )
