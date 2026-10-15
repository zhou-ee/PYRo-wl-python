"""A line-COM mode must change physics and trim together in every plant."""

from dataclasses import replace
import json
import re

import numpy as np
import pytest

from tests.reference_parameters import REFERENCE_GROUND, REFERENCE_ROBOT
from wheel_leg_lqr.artifacts.ground_lqr import render_header, save_result
from wheel_leg_lqr.models.airborne.model import airborne_generalized_mass
from wheel_leg_lqr.models.contact.model import _mass_and_input, _raw_kinematics
from wheel_leg_lqr.models.ground.trim import (
    compute_equilibrium_beta,
    compute_equilibrium_beta_from_length,
    fit_equilibrium_beta,
)
from wheel_leg_lqr.models.leg_geometry import leg_com
from wheel_leg_lqr.scheduling.ground_lqr import compute_gain_node, compute_gain_schedule


LINE_ROBOT = replace(REFERENCE_ROBOT, leg_com_fore_aft_enabled=False)
ZERO_CURVE_ROBOT = replace(REFERENCE_ROBOT, leg_com_x_poly=(0.0,) * 4)


@pytest.mark.parametrize("length", [0.18, 0.28, 0.38])
def test_disabled_com_stays_on_line_with_length_dependent_axial_position(length):
    lwp, x_com = leg_com(length, LINE_ROBOT)
    assert x_com == 0.0
    assert lwp == leg_com(length, REFERENCE_ROBOT)[0]
    assert lwp == pytest.approx(
        np.polynomial.polynomial.polyval(length, REFERENCE_ROBOT.leg_com_lwp_poly)
    )
    assert leg_com(length, REFERENCE_ROBOT)[1] != 0.0
    assert compute_equilibrium_beta_from_length(length, LINE_ROBOT) == 0.0
    assert compute_equilibrium_beta(length, LINE_ROBOT) == 0.0
    np.testing.assert_array_equal(fit_equilibrium_beta(LINE_ROBOT), np.zeros(4))
    assert leg_com(0.18, LINE_ROBOT)[0] != leg_com(0.38, LINE_ROBOT)[0]


def test_ground_switch_changes_physics_and_cache_restores_enabled_model():
    def node(robot):
        return compute_gain_node(
            0.24, 0.32, parameters=robot, settings=REFERENCE_GROUND
        )

    enabled = node(REFERENCE_ROBOT)
    disabled = node(LINE_ROBOT)
    zero_curve = node(ZERO_CURVE_ROBOT)
    restored = node(REFERENCE_ROBOT)
    for name in ("mass", "A", "B", "equilibrium_input", "stiffness"):
        np.testing.assert_array_equal(
            getattr(disabled.linearization, name),
            getattr(zero_curve.linearization, name),
        )
        np.testing.assert_array_equal(
            getattr(restored.linearization, name), getattr(enabled.linearization, name)
        )
    assert (
        np.linalg.norm(enabled.linearization.mass - disabled.linearization.mass) > 1e-4
    )
    np.testing.assert_array_equal(disabled.linearization.geometry.q[5:7], [0.0, 0.0])
    assert np.max(np.abs(disabled.linearization.equilibrium_residual)) < 1e-8
    np.testing.assert_array_equal(disabled.gain, zero_curve.gain)
    assert np.max(np.abs(disabled.closed_loop_poles)) < 1.0


@pytest.mark.parametrize("lengths", [(0.18, 0.38), (0.28, 0.28), (0.38, 0.18)])
def test_airborne_switch_zeroes_offset_and_length_derivative(lengths):
    posture = dict(beta_1=0.12, beta_2=-0.08, theta=0.05)
    disabled = airborne_generalized_mass(*lengths, parameters=LINE_ROBOT, **posture)
    expected = airborne_generalized_mass(
        *lengths, parameters=ZERO_CURVE_ROBOT, **posture
    )
    enabled = airborne_generalized_mass(*lengths, parameters=REFERENCE_ROBOT, **posture)
    np.testing.assert_array_equal(disabled, expected)
    assert np.min(np.linalg.eigvalsh(disabled)) > 0.0
    assert np.linalg.norm(disabled - enabled) > 1e-4


def test_contact_switch_places_com_on_line_even_with_tilted_legs():
    q = np.array([0.0, 0.0, 0.05, -0.03, 0.28, 0.12, -0.08, 0.02, 0.04])
    lengths, positions = _raw_kinematics(q, LINE_ROBOT)
    for index, beta in enumerate(q[5:7]):
        displacement = positions[index + 2] - positions[index]
        normal = [np.cos(beta), 0.0, np.sin(beta)]
        direction = [-np.sin(beta), 0.0, np.cos(beta)]
        assert displacement @ normal == pytest.approx(0.0, abs=1e-14)
        assert displacement @ direction == pytest.approx(
            leg_com(lengths[index], LINE_ROBOT)[0]
        )
    disabled = _mass_and_input(q, LINE_ROBOT)
    expected = _mass_and_input(q, ZERO_CURVE_ROBOT)
    for actual, reference in zip(disabled[:3], expected[:3]):
        np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(disabled[3].maps, expected[3].maps)
    assert np.linalg.norm(disabled[0] - _mass_and_input(q, REFERENCE_ROBOT)[0]) > 1e-4


def test_disabled_schedule_and_export_share_zero_trim_and_snapshot(tmp_path):
    result = compute_gain_schedule(
        step=0.04, parameters=LINE_ROBOT, settings=REFERENCE_GROUND
    )
    assert result.max_fitted_spectral_radius < 1.0
    assert result.saturation_validation.passed
    for node in result.nodes:
        np.testing.assert_array_equal(node.linearization.geometry.q[5:7], [0.0, 0.0])
    header = render_header(result)
    block = re.search(r"BETA_TRIM_POLY_COEF\[[^\n]+?= \{(.*?)\};", header, re.S)
    assert block is not None
    coefficients = [
        float(value.strip().removesuffix("f")) for value in block[1].split(",")
    ]
    np.testing.assert_array_equal(coefficients, np.zeros(4))
    save_result(result, tmp_path)
    report = json.loads((tmp_path / "l1_l2_gain_fit.json").read_text(encoding="utf-8"))
    assert report["physical_parameters"]["leg_com_fore_aft_enabled"] is False
    assert report["physical_fingerprint"] == LINE_ROBOT.fingerprint()


def test_cad_source_curve_retained_but_equilibrium_follows_switch():
    pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    from wheel_leg_lqr.calibration.leg_com import fit_leg_com_data

    enabled = fit_leg_com_data(parameters=REFERENCE_ROBOT)
    disabled = fit_leg_com_data(parameters=LINE_ROBOT)
    for key in ("L_wp", "x_com", "poly_x", "poly_wp", "alpha_P", "alpha_H"):
        np.testing.assert_array_equal(disabled[key], enabled[key])
    np.testing.assert_array_equal(disabled["beta_eq"], np.zeros_like(disabled["L"]))
    np.testing.assert_array_equal(disabled["poly_beta_eq"], np.zeros(4))


def test_switch_preserves_source_curve_and_is_in_snapshot_and_cache_key():
    assert LINE_ROBOT.leg_com_x_poly == REFERENCE_ROBOT.leg_com_x_poly
    assert LINE_ROBOT.snapshot()["leg_com_fore_aft_enabled"] is False
    assert LINE_ROBOT.fingerprint() != REFERENCE_ROBOT.fingerprint()
    assert LINE_ROBOT != REFERENCE_ROBOT
    assert LINE_ROBOT.effective_leg_com_x_poly == (0.0,) * 4
    assert REFERENCE_ROBOT.effective_leg_com_x_poly == REFERENCE_ROBOT.leg_com_x_poly


@pytest.mark.parametrize("value", [0, 1, "False", None])
def test_switch_rejects_non_boolean_values(value):
    with pytest.raises(ValueError, match="leg_com_fore_aft_enabled must be a bool"):
        replace(REFERENCE_ROBOT, leg_com_fore_aft_enabled=value)
