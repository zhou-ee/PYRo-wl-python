import numpy as np
import pytest

from wheel_leg_lqr import numeric_config as config
from wheel_leg_lqr.gain_schedule import compute_gain_node
from wheel_leg_lqr.numeric_model import (
    evaluate_height,
    evaluate_input_matrix,
    evaluate_lengths,
    evaluate_mass,
    evaluate_wheel_positions_heading,
    evaluate_wheel_velocity_maps_heading,
    linearize_static_trim,
    make_trim_geometry,
)


def test_lbar_is_the_controlled_mean_actual_leg_length():
    geometry = make_trim_geometry(config.TRIM_L_BAR, config.TRIM_L_BAR)
    assert geometry.q[4] == pytest.approx(config.TRIM_L_BAR)
    lengths = evaluate_lengths(geometry.q, geometry.kappa_g)
    np.testing.assert_allclose(np.mean(lengths), geometry.q[4], atol=1e-12)
    np.testing.assert_allclose(geometry.height, config.TRIM_HEIGHT, atol=1e-12)
    np.testing.assert_allclose(
        evaluate_height(geometry.q, geometry.kappa_g), geometry.height, atol=1e-12
    )


def test_lbar_coordinate_reconstructs_leg_lengths_at_asymmetric_trim():
    geometry = make_trim_geometry(0.36, 0.24, beta_1=0.0, beta_2=0.0)
    np.testing.assert_allclose(geometry.leg_lengths, [0.36, 0.24], atol=1e-12)
    np.testing.assert_allclose(np.mean(geometry.leg_lengths), geometry.q[4], atol=1e-12)


def test_body_inertia_enforces_left_right_symmetry():
    inertia = config.PARAMETERS.body_inertia
    assert inertia[0, 1] == 0.0
    assert inertia[1, 0] == 0.0
    assert inertia[1, 2] == 0.0
    assert inertia[2, 1] == 0.0
    assert np.min(np.linalg.eigvalsh(inertia)) > 0.0


def test_wheel_input_limit_matches_lower_level_current_limit():
    expected_torque_limit = 0.21777611 * 15.0
    np.testing.assert_allclose(
        config.WHEEL_TORQUE_LIMIT_NM, expected_torque_limit, atol=1e-15
    )
    np.testing.assert_allclose(
        config.INPUT_LOWER_LIMITS[:2], [-expected_torque_limit] * 2, atol=1e-15
    )
    np.testing.assert_allclose(
        config.INPUT_UPPER_LIMITS[:2], [expected_torque_limit] * 2, atol=1e-15
    )


def test_height_028_and_beta_zero_reach_requested_endpoints():
    negative = make_trim_geometry(0.18, 0.38, beta_1=0.0, beta_2=0.0)
    positive = make_trim_geometry(0.38, 0.18, beta_1=0.0, beta_2=0.0)
    assert negative.feasible
    assert positive.feasible
    np.testing.assert_allclose(negative.leg_lengths, [0.18, 0.38], atol=1e-12)
    np.testing.assert_allclose(positive.leg_lengths, [0.38, 0.18], atol=1e-12)


def test_mass_and_input_matrices_have_expected_properties():
    geometry = make_trim_geometry(0.28, 0.28, beta_1=0.0, beta_2=0.0)
    mass = evaluate_mass(geometry.q, geometry.kappa_g)
    input_matrix = evaluate_input_matrix(geometry.q, geometry.kappa_g)
    np.testing.assert_allclose(mass, mass.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(mass)) > 0.0
    assert np.linalg.matrix_rank(input_matrix) == 6
    expected_yaw = config.PARAMETERS.wheel_half_track / config.PARAMETERS.wheel_radius
    np.testing.assert_allclose(input_matrix[1, :2], [-expected_yaw, expected_yaw])


def test_wheel_height_constraint_is_preserved():
    geometry = make_trim_geometry(0.34, 0.22, beta_1=0.0, beta_2=0.0)
    wheels = evaluate_wheel_positions_heading(geometry.q, geometry.kappa_g)
    expected = np.array(
        [
            geometry.kappa_g,
            -geometry.kappa_g,
        ]
    )
    np.testing.assert_allclose(wheels[2], expected, atol=1e-12)


def test_heading_positions_are_relative_to_wheel_midpoint():
    geometry = make_trim_geometry(0.34, 0.22, beta_1=0.0, beta_2=0.0)
    q0 = geometry.q.copy()
    q1 = geometry.q.copy()
    q1[0] = 3.7
    q1[1] = 0.63
    wheels = evaluate_wheel_positions_heading(q1, geometry.kappa_g)
    midpoint = np.mean(wheels, axis=1)
    np.testing.assert_allclose(midpoint, [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(
        wheels,
        evaluate_wheel_positions_heading(q0, geometry.kappa_g),
        atol=1e-12,
    )


def test_heading_velocity_map_contains_path_and_yaw_transport_columns():
    geometry = make_trim_geometry(0.28, 0.28, beta_1=0.0, beta_2=0.0)
    q = geometry.q.copy()
    q[1] = 0.63
    maps = evaluate_wheel_velocity_maps_heading(q, geometry.kappa_g)
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    for velocity_map in maps:
        np.testing.assert_allclose(velocity_map[:, 0], e1, atol=1e-12)
        np.testing.assert_allclose(e2 @ velocity_map[:, 0], 0.0, atol=1e-12)
    expected_yaw_columns = np.array(
        [
            [-config.PARAMETERS.wheel_half_track, 0.0, 0.0],
            [config.PARAMETERS.wheel_half_track, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(maps[:, :, 1], expected_yaw_columns, atol=1e-12)


def test_heading_velocity_map_reconstructs_world_components():
    geometry = make_trim_geometry(0.32, 0.24, beta_1=0.0, beta_2=0.0)
    q = geometry.q.copy()
    q[1] = 0.63
    maps_heading = evaluate_wheel_velocity_maps_heading(q, geometry.kappa_g)
    c, s = np.cos(q[1]), np.sin(q[1])
    heading_to_world = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    heading_world = np.array([c, s, 0.0])
    for velocity_map_heading in maps_heading:
        velocity_map_world = heading_to_world @ velocity_map_heading
        np.testing.assert_allclose(
            velocity_map_world[:, 0], heading_world, atol=1e-12
        )
        np.testing.assert_allclose(
            velocity_map_world.T @ velocity_map_world,
            velocity_map_heading.T @ velocity_map_heading,
            atol=1e-12,
        )


def test_yaw_column_is_heading_frame_transport_velocity():
    geometry = make_trim_geometry(0.32, 0.24, beta_1=0.0, beta_2=0.0)
    q = geometry.q.copy()
    q[1:] = [0.81, 0.12, -0.08, 0.29, 0.04, -0.03]
    positions = evaluate_wheel_positions_heading(q, geometry.kappa_g)
    maps = evaluate_wheel_velocity_maps_heading(q, geometry.kappa_g)
    e3 = np.array([0.0, 0.0, 1.0])
    for index in range(2):
        np.testing.assert_allclose(
            maps[index, :, 1],
            np.cross(e3, positions[:, index]),
            atol=1e-12,
        )


def test_absolute_yaw_does_not_change_isotropic_horizontal_mechanics():
    geometry = make_trim_geometry(0.32, 0.24, beta_1=0.0, beta_2=0.0)
    q0 = geometry.q.copy()
    q1 = geometry.q.copy()
    q1[1] = 1.17
    np.testing.assert_allclose(
        evaluate_mass(q0, geometry.kappa_g),
        evaluate_mass(q1, geometry.kappa_g),
        atol=1e-11,
    )
    np.testing.assert_allclose(
        evaluate_input_matrix(q0, geometry.kappa_g),
        evaluate_input_matrix(q1, geometry.kappa_g),
        atol=1e-11,
    )


def test_requested_beta_point_is_rejected_as_non_equilibrium():
    with pytest.raises(ValueError, match="not a static trim"):
        linearize_static_trim(0.28, 0.28, beta_1=0.1, beta_2=0.1)


def test_true_static_node_is_controllable_and_lqr_stable():
    node = compute_gain_node(
        config.TRIM_L_BAR,
        config.TRIM_L_BAR,
    )
    assert node.controllability_rank == 14
    assert np.max(np.abs(node.closed_loop_poles)) < 1.0
    assert np.max(np.abs(node.linearization.equilibrium_residual)) < 1e-10
    np.testing.assert_allclose(
        node.linearization.equilibrium_input[:4],
        [0.0, 0.0, 0.0, 0.0],
        atol=1e-8,
    )
    # The two leg thrust forces should be equal and support the total mass.
    assert np.isclose(
        node.linearization.equilibrium_input[4],
        node.linearization.equilibrium_input[5],
        atol=1e-8,
    )
    total_mass = (
        config.PARAMETERS.body_mass
        + 2.0 * config.PARAMETERS.leg_mass
        + 2.0 * config.PARAMETERS.wheel_mass
    )
    assert node.linearization.equilibrium_input[4] == pytest.approx(
        total_mass * config.PARAMETERS.gravity / 2.0,
        rel=0.1,
    )
