from tests.reference_parameters import (
    REFERENCE_GROUND,
    REFERENCE_ROBOT,
)
from wheel_leg_lqr.models.ground.trim import compute_equilibrium_beta
import numpy as np

from wheel_leg_lqr.models.ground import config
from wheel_leg_lqr.scheduling.ground_lqr import compute_gain_node
from wheel_leg_lqr.models.ground.config import default_initial_errors
from wheel_leg_lqr.simulation.saturation import (
    simulate_saturated_closed_loop,
    simulate_saturated_closed_loop_batch,
    validate_saturated_closed_loop,
)


def test_saturation_uses_perturbation_input_after_absolute_clipping():
    Ad = np.eye(config.N_X)
    Bd = np.zeros((config.N_X, config.N_U))
    Bd[0, 0] = 1.0
    gain = np.zeros((config.N_U, config.N_X))
    gain[0, 0] = 1.0
    equilibrium_input = np.zeros(config.N_U)
    equilibrium_input[0] = 0.5
    initial_error = np.zeros(config.N_X)
    initial_error[0] = 2.0

    states, commands, saturation_steps = simulate_saturated_closed_loop(
        Ad,
        Bd,
        gain,
        equilibrium_input,
        initial_error,
        horizon_s=0.001,
        sample_time=0.001,
        lower_limits=np.full(config.N_U, -1.0),
        upper_limits=np.full(config.N_U, 1.0),
    )

    # Absolute command is clipped to -1.0; the perturbation applied to Bd is
    # -1.0 - u0 = -1.5, so the next error is 0.5 rather than 1.0.
    np.testing.assert_allclose(commands[0, 0], -1.0)
    np.testing.assert_allclose(states[-1, 0], 0.5)
    assert saturation_steps == 1


def test_batched_simulation_matches_individual_simulations():
    Ad = np.eye(config.N_X) * 0.95
    Bd = np.zeros((config.N_X, config.N_U))
    Bd[: config.N_U] = np.eye(config.N_U) * 0.01
    gain = np.zeros((config.N_U, config.N_X))
    gain[:, : config.N_U] = np.eye(config.N_U)
    equilibrium_input = np.zeros(config.N_U)
    initial_errors = np.vstack([case[1] for case in default_initial_errors()[:2]])

    batch_states, batch_commands, batch_steps = simulate_saturated_closed_loop_batch(
        Ad,
        Bd,
        gain,
        equilibrium_input,
        initial_errors,
        horizon_s=0.01,
        sample_time=0.001,
        lower_limits=REFERENCE_ROBOT.input_lower_limits,
        upper_limits=REFERENCE_ROBOT.input_upper_limits,
    )
    for index, initial_error in enumerate(initial_errors):
        states, commands, saturation_steps = simulate_saturated_closed_loop(
            Ad,
            Bd,
            gain,
            equilibrium_input,
            initial_error,
            horizon_s=0.01,
            sample_time=0.001,
            lower_limits=REFERENCE_ROBOT.input_lower_limits,
            upper_limits=REFERENCE_ROBOT.input_upper_limits,
        )
        np.testing.assert_allclose(batch_states[index], states)
        np.testing.assert_allclose(batch_commands[index], commands)
        assert batch_steps[index] == saturation_steps


def test_nominal_gain_passes_saturated_recovery_and_hits_limits():
    node = compute_gain_node(
        (0.28 / np.cos(compute_equilibrium_beta(0.28, parameters=REFERENCE_ROBOT))),
        (0.28 / np.cos(compute_equilibrium_beta(0.28, parameters=REFERENCE_ROBOT))),
        parameters=REFERENCE_ROBOT,
        settings=REFERENCE_GROUND,
    )
    validation = validate_saturated_closed_loop(
        node.Ad,
        node.Bd,
        node.gain,
        node.linearization.equilibrium_input,
        cases=(default_initial_errors()[0],),
        sample_time=0.001,
        lower_limits=REFERENCE_ROBOT.input_lower_limits,
        upper_limits=REFERENCE_ROBOT.input_upper_limits,
    )

    assert validation.passed
    assert validation.saturation_steps > 0
    assert validation.max_final_state_abs < 0.01
