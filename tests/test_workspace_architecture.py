"""Behavioral checks for shared hardware, isolated weights, and portable outputs."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import ast
import os
import subprocess
import sys

import numpy as np
import pytest

from wheel_leg_lqr.config import DEFAULT_ROBOT
from wheel_leg_lqr.models.ground.model import GroundModel
from wheel_leg_lqr.models.airborne.model import linearize_airborne
from wheel_leg_lqr.models.contact.model import _mass_and_input
from wheel_leg_lqr.models.ground.config import DEFAULT_GROUND
from wheel_leg_lqr.scheduling.ground_lqr import compute_gain_node
from wheel_leg_lqr.scheduling.airborne_lqr import compute_airborne_gain_node
from wheel_leg_lqr.control.lqr import discretize_state_space, solve_discrete_lqr
from wheel_leg_lqr.control.rdob import design_rdob, initialize_rdob, step_rdob
from wheel_leg_lqr.artifacts.io import (
    default_output_dir,
    resolve_output_dir,
    project_root,
)
from wheel_leg_lqr.analysis.airborne_fl_pitch import analyze_leg_force_pitch
from wheel_leg_lqr.models.airborne.model import compute_leg_force_pitch_effect
from wheel_leg_lqr.scheduling.ground_lqr import evaluate_fitted_gain


def test_configured_body_inertia_respects_fore_aft_and_left_right_symmetry():
    inertia = np.asarray(DEFAULT_ROBOT.body_inertia)
    for reflection in (np.diag([-1.0, 1.0, 1.0]), np.diag([1.0, -1.0, 1.0])):
        np.testing.assert_array_equal(reflection @ inertia @ reflection.T, inertia)
    # Constant pure yaw of a symmetric, upright body needs no pitch/roll bias.
    omega = np.array([0.0, 0.0, 12.0])
    np.testing.assert_array_equal(np.cross(omega, inertia @ omega), np.zeros(3))


def test_symmetric_nominal_ground_removes_yaw_body_coupling_but_keeps_joint_coupling():
    robot = replace(DEFAULT_ROBOT, leg_com_fore_aft_enabled=False)
    length = (robot.min_leg_length + robot.max_leg_length) / 2
    node = compute_gain_node(length, length, parameters=robot)
    mass = node.linearization.mass
    np.testing.assert_allclose(mass[1, [2, 3]], 0.0, atol=1e-12)
    assert abs(mass[2, 5]) > 1e-3
    assert np.min(np.linalg.eigvalsh(mass)) > 0.0
    assert np.max(np.abs(node.closed_loop_poles)) < 1.0


@pytest.mark.parametrize("mass,com_x", [(12.0, 0.0), (23.0, 0.045)])
def test_pitch_analysis_uses_supplied_physical_config(mass, com_x):
    robot = replace(DEFAULT_ROBOT, body_mass=mass, body_com_from_hip=(com_x, 0.0, 0.04))
    nominal = (robot.min_leg_length + robot.max_leg_length) / 2
    configured = analyze_leg_force_pitch(parameters=robot)
    explicit = compute_leg_force_pitch_effect(
        nominal, nominal, parameters=robot, body_com_x_from_hip=com_x
    )
    np.testing.assert_array_equal(
        configured.generalized_mass, explicit.generalized_mass
    )
    np.testing.assert_allclose(configured.body_moment_arm, [com_x, com_x], atol=1e-14)


def test_ground_gain_evaluation_uses_parameter_domain():
    robot = replace(DEFAULT_ROBOT, min_leg_length=0.21, max_leg_length=0.41)
    coefficients = np.zeros((6, 14, 2, 2))
    coefficients[:, :, 1, 0] = 2.0
    coefficients[:, :, 0, 1] = 3.0
    np.testing.assert_allclose(
        evaluate_fitted_gain(coefficients, 0.21, 0.41, parameters=robot), 1.0
    )
    np.testing.assert_allclose(
        evaluate_fitted_gain(coefficients, 0.41, 0.21, parameters=robot), -1.0
    )


def test_shared_mass_changes_all_plants_and_cache_switches_back():
    lower, upper = DEFAULT_ROBOT.min_leg_length, DEFAULT_ROBOT.max_leg_length
    l1, l2 = lower + 0.3 * (upper - lower), lower + 0.7 * (upper - lower)
    base = GroundModel(DEFAULT_ROBOT).linearize(l1, l2)
    changed = replace(DEFAULT_ROBOT, body_mass=DEFAULT_ROBOT.body_mass * 1.1)
    other = GroundModel(changed).linearize(l1, l2)
    restored = GroundModel(DEFAULT_ROBOT).linearize(l1, l2)
    assert np.linalg.norm(other.mass - base.mass) > 0.1
    np.testing.assert_array_equal(restored.mass, base.mass)
    np.testing.assert_array_equal(restored.A, base.A)
    assert (
        np.linalg.norm(
            linearize_airborne(l1, l2, parameters=changed).mass
            - linearize_airborne(l1, l2).mass
        )
        > 1e-5
    )
    q = np.r_[base.geometry.q, base.geometry.kappa_g, 0.0]
    original_contact = _mass_and_input(q)[0]
    changed_contact = _mass_and_input(q, changed)[0]
    assert np.linalg.norm(changed_contact - original_contact) > 0.1


def test_ground_weights_change_gain_without_changing_physics_or_airborne():
    lower, upper = DEFAULT_ROBOT.min_leg_length, DEFAULT_ROBOT.max_leg_length
    l1, l2 = lower + 0.3 * (upper - lower), lower + 0.7 * (upper - lower)
    ground = compute_gain_node(l1, l2)
    air = compute_airborne_gain_node(l1, l2)
    settings = replace(
        DEFAULT_GROUND, q_diag=tuple(np.asarray(DEFAULT_GROUND.q_diag) * 2)
    )
    changed = compute_gain_node(l1, l2, settings=settings)
    np.testing.assert_array_equal(changed.linearization.A, ground.linearization.A)
    np.testing.assert_array_equal(changed.linearization.mass, ground.linearization.mass)
    assert np.linalg.norm(changed.gain - ground.gain) > 0.1
    np.testing.assert_array_equal(compute_airborne_gain_node(l1, l2).gain, air.gain)


def test_physical_snapshot_is_immutable_hashable_and_validated():
    robot = replace(
        DEFAULT_ROBOT, body_com_from_hip=np.asarray(DEFAULT_ROBOT.body_com_from_hip)
    )
    assert isinstance(robot.body_com_from_hip, tuple)
    assert hash(robot) == hash(DEFAULT_ROBOT)
    with pytest.raises(FrozenInstanceError):
        robot.body_mass = 2.0
    with pytest.raises(TypeError):
        robot.body_inertia[0][0] = 1.0
    with pytest.raises(ValueError, match="finite"):
        replace(robot, body_mass=float("nan"))
    assert (
        replace(robot, body_mass=robot.body_mass * 1.1).fingerprint()
        != robot.fingerprint()
    )


def test_generic_control_algorithms_on_two_coordinates():
    ad, bd = discretize_state_space(np.zeros((2, 2)), np.eye(2), sample_time=0.01)
    gain, _, poles = solve_discrete_lqr(ad, bd, np.eye(2), np.eye(2))
    assert gain.shape == (2, 2)
    assert np.max(np.abs(poles)) < 1
    observer = design_rdob(
        np.eye(2), np.eye(2), np.eye(2), rates_rad_s=8.0, sample_time=0.01
    )
    zero = np.zeros(2)
    z = initialize_rdob(observer, zero)
    next_z, disturbance = step_rdob(observer, z, zero, zero, zero, zero, zero)
    np.testing.assert_array_equal(next_z, zero)
    np.testing.assert_array_equal(disturbance, zero)


def test_output_resolution_from_another_working_directory(monkeypatch, tmp_path):
    root = project_root()
    monkeypatch.chdir(tmp_path)
    assert default_output_dir("rdob") == root / "outputs/rdob"
    assert resolve_output_dir("custom") == tmp_path / "custom"
    with pytest.raises(ValueError, match="reference"):
        resolve_output_dir(root / "data/reference/nested")


def test_lower_components_do_not_import_cli_or_exporters():
    package = Path(__file__).resolve().parents[1] / "src/wheel_leg_lqr"
    for directory in ("models", "control", "scheduling", "simulation"):
        for path in (package / directory).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    assert "cli" not in module.split("."), path
                    assert "artifacts" not in module.split("."), path
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert (
                            ".cli" not in alias.name and ".artifacts" not in alias.name
                        ), path


@pytest.mark.parametrize(
    "module",
    [
        "ground_lqr",
        "airborne_lqr",
        "rdob",
        "symbolic",
        "bandwidth",
        "qr_optimizer",
        "step_drop",
        "airborne_fl_pitch",
        "leg_com",
    ],
)
def test_cli_help_from_another_directory(module, tmp_path):
    source = Path(__file__).resolve().parents[1] / "src"
    environment = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-m", f"wheel_leg_lqr.cli.{module}", "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
