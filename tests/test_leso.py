import numpy as np

from wheel_leg_lqr.control.leso import (
    build_augmented_model,
    design_leso_correction,
    observer_error_matrix,
    observer_error_spectral_radius,
    step_leso_split,
)
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
        np.asarray(
            [correction.unmatched_state_pole] * (N_X - N_U)
            + [correction.state_pole] * N_U
            + [correction.disturbance_pole] * N_U
        )
    )
    np.testing.assert_allclose(poles, expected, atol=2.0e-10, rtol=0.0)
    assert observer_error_spectral_radius(
        node.Ad,
        node.Bd,
        correction.state_gain,
        correction.disturbance_gain,
    ) < 1.0


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
