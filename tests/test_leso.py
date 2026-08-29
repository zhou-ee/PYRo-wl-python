import numpy as np

from wheel_leg_lqr import numeric_config as config
from wheel_leg_lqr.control.leso import (
    DEFAULT_LESO_DESIGN,
    N_AUGMENTED,
    LesoEstimate,
    build_augmented_model,
    design_leso_gain,
    initialize_leso,
    observer_error_spectral_radius,
    step_leso_split,
)
from wheel_leg_lqr.gain_schedule import compute_gain_node


def _nominal_design():
    node = compute_gain_node(0.28, 0.28)
    gain = design_leso_gain(node.Ad, node.Bd)
    return node.Ad, node.Bd, gain


def test_six_disturbance_augmented_model_is_observable_and_stable():
    physical_ad, physical_bd, gain = _nominal_design()
    augmented_a, augmented_b, output = build_augmented_model(
        physical_ad,
        physical_bd,
    )
    observability = np.vstack(
        [output @ np.linalg.matrix_power(augmented_a, power) for power in range(N_AUGMENTED)]
    )

    assert augmented_a.shape == (20, 20)
    assert augmented_b.shape == (20, 6)
    assert output.shape == (14, 20)
    assert gain.shape == (20, 14)
    assert np.linalg.matrix_rank(observability) == 20
    assert observer_error_spectral_radius(physical_ad, physical_bd, gain) < 1.0


def test_split_update_matches_full_augmented_matrix_update():
    physical_ad, physical_bd, gain = _nominal_design()
    augmented_a, augmented_b, output = build_augmented_model(
        physical_ad,
        physical_bd,
    )
    rng = np.random.default_rng(42)
    estimate = LesoEstimate(
        state=rng.normal(scale=0.05, size=config.N_X),
        disturbance=rng.normal(scale=0.02, size=config.N_U),
    )
    measured_state = rng.normal(scale=0.01, size=config.N_X)
    applied_delta = rng.normal(scale=0.1, size=config.N_U)

    result = step_leso_split(
        estimate,
        measured_state,
        applied_delta,
        physical_ad,
        physical_bd,
        gain,
    )
    augmented_estimate = np.concatenate([estimate.state, estimate.disturbance])
    expected = (
        augmented_a @ augmented_estimate
        + augmented_b @ applied_delta
        + gain @ (measured_state - output @ augmented_estimate)
    )

    np.testing.assert_allclose(result.estimate.state, expected[: config.N_X])
    np.testing.assert_allclose(result.estimate.disturbance, expected[config.N_X :])


def test_initialization_and_physical_disturbance_limits():
    physical_ad, physical_bd, gain = _nominal_design()
    estimate = initialize_leso(np.zeros(config.N_X))
    large_measurement = np.ones(config.N_X)
    limits = np.array([0.2, 0.2, 1.0, 1.0, 5.0, 5.0])

    result = step_leso_split(
        estimate,
        large_measurement,
        np.zeros(config.N_U),
        physical_ad,
        physical_bd,
        gain,
        disturbance_limits=limits,
    )

    assert np.all(np.abs(result.estimate.physical_disturbance()) <= limits + 1e-12)


def test_all_six_disturbances_converge_in_compensated_closed_loop():
    node = compute_gain_node(0.28, 0.28)
    design = DEFAULT_LESO_DESIGN
    gain = design_leso_gain(node.Ad, node.Bd, design)
    true_disturbance = np.array([0.098, -0.065, 2.4, -1.8, 15.0, -12.0])
    state = np.zeros(config.N_X)
    estimate = initialize_leso(np.zeros(config.N_X))

    for _ in range(2000):
        physical_input = -node.gain @ state - estimate.disturbance
        result = step_leso_split(
            estimate,
            state,
            physical_input,
            node.Ad,
            node.Bd,
            gain,
        )
        state = node.Ad @ state + node.Bd @ (
            physical_input + true_disturbance
        )
        estimate = result.estimate

    np.testing.assert_allclose(
        estimate.disturbance,
        true_disturbance,
        atol=1e-5,
    )
    assert np.max(np.abs(state)) < 1e-3
