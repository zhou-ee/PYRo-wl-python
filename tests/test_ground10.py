"""Independent energy, trim, coupling isolation, control and export checks."""

from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from tests.reference_parameters import REFERENCE_ROBOT
from wheel_leg_lqr.artifacts.ground10_lqr import render_header, save_result
from wheel_leg_lqr.models.ground10.config import (
    COORDINATE_NAMES,
    INPUT_NAMES,
    STATE_NAMES,
    Ground10Settings,
)
from wheel_leg_lqr.models.ground10.model import (
    Ground10Model,
    evaluate_gravity,
    evaluate_input_matrix,
    evaluate_kinematics,
    evaluate_mass,
    evaluate_potential,
)
from wheel_leg_lqr.models.leg_geometry import leg_com
from wheel_leg_lqr.scheduling.ground10_lqr import (
    compute_gain_node,
    compute_gain_schedule,
    evaluate_polynomial,
)


REFERENCE_SETTINGS = Ground10Settings(
    sample_time=0.001,
    q_diag=(80.0, 25.0, 40.0, 8.0, 400.0, 25.0, 120.0, 6.0, 120.0, 6.0),
    r_diag=(2.8, 2.8, 0.1, 0.1),
)


def test_state_input_contract():
    assert COORDINATE_NAMES == ("x", "psi", "theta", "beta_1", "beta_2")
    assert STATE_NAMES == (
        "x",
        "dx",
        "psi",
        "dpsi",
        "theta",
        "dtheta",
        "beta_1",
        "dbeta_1",
        "beta_2",
        "dbeta_2",
    )
    assert INPUT_NAMES == ("T_w1", "T_w2", "T_p1", "T_p2")


@pytest.mark.parametrize("lengths", [(0.18, 0.38), (0.28, 0.28), (0.38, 0.18)])
@pytest.mark.parametrize("enabled", [True, False])
def test_trim_and_closed_loop_across_domain(lengths, enabled):
    robot = replace(REFERENCE_ROBOT, leg_com_fore_aft_enabled=enabled)
    node = compute_gain_node(*lengths, parameters=robot, settings=REFERENCE_SETTINGS)
    lin = node.linearization
    assert lin.mass.shape == (5, 5)
    assert lin.A.shape == (10, 10) and lin.B.shape == (10, 4)
    assert node.gain.shape == (4, 10)
    np.testing.assert_allclose(lin.mass, lin.mass.T, atol=1e-14)
    assert np.min(np.linalg.eigvalsh(lin.mass)) > 0
    assert node.controllability_rank == 10
    assert np.max(np.abs(node.closed_loop_poles)) < 1
    assert np.max(np.abs(lin.equilibrium_residual)) < 1e-10
    np.testing.assert_allclose(lin.equilibrium_input, 0, atol=1e-12)
    for length, beta in zip(lengths, lin.geometry.q[3:]):
        lwp, x_com = leg_com(length, robot)
        assert beta == pytest.approx(
            np.arctan(
                robot.leg_mass
                * x_com
                / (robot.body_mass * length / 2 + robot.leg_mass * lwp)
            )
        )


def _reference_positions(q, l1, l2, robot):
    """Direct construction, independent of symbolic positions and Jacobians."""
    _, _, theta, b1, b2 = q
    betas, lengths = np.array([b1, b2]), np.array([l1, l2])
    displacement = np.mean(lengths * np.sin(betas))
    height = np.mean(lengths * np.cos(betas))
    wheels, legs = [], []
    for sign, length, beta in zip((1, -1), lengths, betas):
        lwp, x_com = leg_com(length, robot)
        wx = -displacement + length * np.sin(beta)
        wheels.append([wx, sign * robot.wheel_half_track, 0.0])
        legs.append(
            [
                wx - lwp * np.sin(beta) + x_com * np.cos(beta),
                sign * robot.hip_half_track,
                lwp * np.cos(beta) + x_com * np.sin(beta),
            ]
        )
    cx, cy, cz = robot.body_com_from_hip
    body = [
        -displacement + cx * np.cos(theta) + cz * np.sin(theta),
        cy,
        height - cx * np.sin(theta) + cz * np.cos(theta),
    ]
    return np.array([*wheels, *legs, body])


def test_mass_matches_independent_kinetic_energy_with_frozen_lengths():
    robot, l1, l2 = REFERENCE_ROBOT, 0.23, 0.34
    q = np.array([0.4, 0.25, 0.13, -0.09, 0.07])
    positions = _reference_positions(q, l1, l2, robot)
    jacobians = np.empty((5, 3, 5))
    for index in range(5):
        delta = np.eye(5)[index] * 1e-6
        jacobians[:, :, index] = (
            _reference_positions(q + delta, l1, l2, robot)
            - _reference_positions(q - delta, l1, l2, robot)
        ) / 2e-6
    jacobians[:, :, 0] = [1.0, 0.0, 0.0]
    jacobians[:, :, 1] = np.cross([0.0, 0.0, 1.0], positions)
    masses = [robot.wheel_mass] * 2 + [robot.leg_mass] * 2 + [robot.body_mass]
    mass = sum(m * jac.T @ jac for m, jac in zip(masses, jacobians))
    for jac in jacobians[:2]:
        mass += (
            robot.wheel_rolling_inertia
            / robot.wheel_radius**2
            * np.outer(jac[0], jac[0])
        )
    for index, length in enumerate((l1, l2)):
        lwp = leg_com(length, robot)[0]
        mass[3 + index, 3 + index] += robot.leg_mass * (
            length**2 / 12 + (lwp - length / 2) ** 2
        )
    omega = np.zeros((3, 5))
    omega[:, 1] = [-np.sin(q[2]), 0.0, np.cos(q[2])]
    omega[:, 2] = [0.0, 1.0, 0.0]
    mass += omega.T @ np.asarray(robot.body_inertia) @ omega
    kin = evaluate_kinematics(q, l1, l2, parameters=robot)
    np.testing.assert_allclose(kin.positions, positions, atol=1e-14)
    np.testing.assert_allclose(kin.velocity_maps, jacobians, atol=1e-10)
    np.testing.assert_allclose(
        evaluate_mass(q, l1, l2, parameters=robot), mass, atol=1e-9
    )
    np.testing.assert_allclose(kin.positions[:2, 2], 0.0, atol=0.0)


def test_gravity_and_linearization_match_independent_finite_differences():
    robot, l1, l2 = REFERENCE_ROBOT, 0.24, 0.32
    lin = Ground10Model(robot).linearize(l1, l2)
    q = lin.geometry.q
    for index in range(5):
        delta = np.eye(5)[index] * 1e-6
        gradient = (
            evaluate_potential(q + delta, l1, l2, parameters=robot)
            - evaluate_potential(q - delta, l1, l2, parameters=robot)
        ) / 2e-6
        assert gradient == pytest.approx(
            evaluate_gravity(q, l1, l2, parameters=robot)[index], abs=2e-8
        )

        def acceleration(posture):
            mass = evaluate_mass(posture, l1, l2, parameters=robot)
            force = evaluate_input_matrix(
                posture, l1, l2, parameters=robot
            ) @ lin.equilibrium_input - evaluate_gravity(
                posture, l1, l2, parameters=robot
            )
            return np.linalg.solve(mass, force)

        column = (acceleration(q + delta) - acceleration(q - delta)) / 2e-6
        np.testing.assert_allclose(column, lin.A[1::2, 2 * index], atol=2e-7)


def test_input_virtual_work_has_only_four_torque_channels():
    q, l1, l2 = np.array([0.0, 0.0, 0.05, 0.12, -0.08]), 0.24, 0.32
    robot = REFERENCE_ROBOT
    inputs = evaluate_input_matrix(q, l1, l2, parameters=robot)
    expected = np.zeros((5, 4))
    expected[0, :2] = 1 / robot.wheel_radius
    expected[1, :2] = np.array([-1, 1]) * robot.wheel_half_track / robot.wheel_radius
    expected[2, 2:] = 1
    for leg, (length, beta) in enumerate(zip((l1, l2), q[3:])):
        for wheel in range(2):
            expected[3 + leg, wheel] = ((leg == wheel) - 0.5) * length * np.cos(
                beta
            ) / robot.wheel_radius + (leg == wheel)
        expected[3 + leg, 2 + leg] = 1
    np.testing.assert_allclose(inputs, expected, atol=1e-14)


def test_no_com_curve_derivatives_enter_frozen_length_dynamics():
    l1, l2 = 0.24, 0.32
    robot = REFERENCE_ROBOT
    # This polynomial is zero at both lengths but changes both curve slopes.
    delta = 0.4 * np.array([l1 * l2, -l1 - l2, 1.0, 0.0])
    changed = replace(
        robot,
        leg_com_lwp_poly=tuple(np.asarray(robot.leg_com_lwp_poly) + delta),
        leg_com_x_poly=tuple(np.asarray(robot.leg_com_x_poly) - delta),
    )
    original = Ground10Model(robot).linearize(l1, l2)
    other = Ground10Model(changed).linearize(l1, l2)
    for name in ("mass", "stiffness", "A", "B"):
        np.testing.assert_allclose(
            getattr(original, name), getattr(other, name), atol=2e-12
        )


def test_roll_and_axial_parameters_cannot_feed_removed_dofs_back():
    robot = REFERENCE_ROBOT
    inertia = np.asarray(robot.body_inertia).copy()
    inertia[0, 0] *= 4
    inertia[0, 2] = inertia[2, 0] = 0.0
    changed = replace(
        robot,
        body_inertia=tuple(map(tuple, inertia)),
        gas_spring_force_poly=(1.0, 2.0, 3.0, 4.0),
        leg_force_limit_n=1.0,
    )
    original = Ground10Model(robot).linearize(0.24, 0.32)
    other = Ground10Model(changed).linearize(0.24, 0.32)
    np.testing.assert_array_equal(original.A, other.A)
    np.testing.assert_array_equal(original.B, other.B)


def test_result_differs_from_slicing_fourteen_state_inverse_dynamics():
    from wheel_leg_lqr.models.ground.model import GroundModel

    full = GroundModel(REFERENCE_ROBOT).linearize(0.24, 0.32)
    reduced = Ground10Model(REFERENCE_ROBOT).linearize(0.24, 0.32)
    indices = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13]
    assert np.linalg.norm(reduced.A - full.A[np.ix_(indices, indices)]) > 1e-3
    assert np.linalg.norm(reduced.B - full.B[indices, :4]) > 1e-3


def test_forward_body_com_is_balanced_and_shared_config_changes_physics():
    robot = replace(REFERENCE_ROBOT, body_com_from_hip=(0.03, 0.0, 0.032))
    lin = Ground10Model(robot).linearize(0.22, 0.36)
    np.testing.assert_allclose(lin.equilibrium_input[:2], 0.0, atol=1e-12)
    np.testing.assert_allclose(
        lin.equilibrium_input[2:],
        [-robot.body_mass * robot.gravity * 0.03 / 2] * 2,
        atol=1e-12,
    )
    heavier = Ground10Model(replace(robot, body_mass=robot.body_mass * 1.2)).linearize(
        0.22, 0.36
    )
    assert np.linalg.norm(heavier.mass - lin.mass) > 1


def test_own_qr_changes_gain_without_changing_mass_or_requiring_ground14(monkeypatch):
    ground = importlib.import_module("wheel_leg_lqr.models.ground.model")

    def forbidden(*args, **kwargs):
        raise AssertionError("14-state model must not be used")

    monkeypatch.setattr(ground, "linearize_static_trim", forbidden)
    base = compute_gain_node(
        0.24, 0.32, parameters=REFERENCE_ROBOT, settings=REFERENCE_SETTINGS
    )
    other = compute_gain_node(
        0.24,
        0.32,
        parameters=REFERENCE_ROBOT,
        settings=replace(
            REFERENCE_SETTINGS, q_diag=tuple(np.asarray(REFERENCE_SETTINGS.q_diag) * 2)
        ),
    )
    np.testing.assert_array_equal(base.linearization.mass, other.linearization.mass)
    assert np.linalg.norm(base.gain - other.gain) > 1


@pytest.fixture(scope="module", params=[True, False])
def schedule(request):
    return compute_gain_schedule(
        step=0.04,
        parameters=replace(REFERENCE_ROBOT, leg_com_fore_aft_enabled=request.param),
        settings=REFERENCE_SETTINGS,
    )


def test_schedule_and_exports_are_independent_ten_state_artifacts(schedule, tmp_path):
    assert schedule.gain_coefficients.shape == (4, 10, 4, 4)
    assert schedule.input_coefficients.shape == (4, 4, 4)
    assert schedule.metrics["max_float32_spectral_radius"] < 1
    assert schedule.metrics["saturation_passed"]
    assert schedule.metrics["training_node_count"] == 36
    assert schedule.metrics["holdout_node_count"] == 25
    save_result(schedule, tmp_path)
    report = json.loads(
        (tmp_path / "ground10_gain_fit.json").read_text(encoding="utf-8")
    )
    assert (
        report["physical_parameters"]["leg_com_fore_aft_enabled"]
        == schedule.parameters.leg_com_fore_aft_enabled
    )
    assert len(report["state_names"]) == 10 and len(report["input_names"]) == 4
    assert report["physical_fingerprint"] == schedule.parameters.fingerprint()
    with np.load(tmp_path / "ground10_gain_nodes.npz") as arrays:
        assert arrays["A"].shape == (36, 10, 10)
        assert arrays["B"].shape == (36, 10, 4)
    if not schedule.parameters.leg_com_fore_aft_enabled:
        np.testing.assert_array_equal(schedule.beta_coefficients, np.zeros(4))


def test_cpp_evaluates_gain_and_trim_like_python(schedule, tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ not available")
    (tmp_path / "ground10.h").write_text(render_header(schedule), encoding="utf-8")
    source = r"""
#include "ground10.h"
#include <iomanip>
#include <iostream>
int main() {
    float l1, l2;
    while (std::cin >> l1 >> l2) {
        float k[4][10], x[10], u[4];
        wheel_leg_ground10::evaluateGain(l1, l2, k);
        wheel_leg_ground10::evaluateTrim(l1, l2, x, u);
        std::cout << std::setprecision(10);
        for (auto &row : k) for (float value : row) std::cout << value << ' ';
        for (float value : x) std::cout << value << ' ';
        for (float value : u) std::cout << value << ' ';
        std::cout << '\n';
    }
}
"""
    path = tmp_path / "evaluate.cpp"
    path.write_text(source, encoding="utf-8")
    executable = tmp_path / "evaluate.exe"
    built = subprocess.run(
        [
            compiler,
            "-std=c++11",
            "-O0",
            "-ffp-contract=off",
            str(path),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stderr
    points = [(0.18, 0.38), (0.28, 0.28), (0.38, 0.18), (0.243, 0.317), (0.1, 0.5)]
    run = subprocess.run(
        [str(executable)],
        input="\n".join(f"{a} {b}" for a, b in points),
        capture_output=True,
        text=True,
        check=True,
    )
    rows = np.fromstring(run.stdout, sep=" ").reshape(len(points), 54)
    for (l1, l2), row in zip(points, rows):
        expected = evaluate_polynomial(
            schedule.gain_coefficients,
            l1,
            l2,
            parameters=schedule.parameters,
            dtype=np.float32,
        )
        np.testing.assert_allclose(
            row[:40].reshape(4, 10), expected, rtol=2e-7, atol=2e-6
        )
        beta = [
            np.polynomial.polynomial.polyval(
                np.clip(length, 0.18, 0.38), schedule.beta_coefficients
            )
            for length in (l1, l2)
        ]
        np.testing.assert_allclose(row[40:50][[6, 8]], beta, atol=1e-7)
        u0 = evaluate_polynomial(
            schedule.input_coefficients,
            l1,
            l2,
            parameters=schedule.parameters,
            dtype=np.float32,
        )
        np.testing.assert_allclose(row[50:], u0, atol=1e-7)


def test_length_domain_and_settings_validate_inputs():
    for lengths in [(0.0, 0.28), (float("nan"), 0.28), (0.1, 0.28)]:
        with pytest.raises(ValueError):
            Ground10Model(REFERENCE_ROBOT).linearize(*lengths)
    with pytest.raises(ValueError, match="degree"):
        compute_gain_schedule(degree=-1)
    with pytest.raises(ValueError, match="grid"):
        compute_gain_schedule(step=0.1)
    with pytest.raises(ValueError, match="q_diag"):
        Ground10Settings(q_diag=(1.0,) * 14)
    with pytest.raises(ValueError, match="r_diag"):
        Ground10Settings(r_diag=(1.0,) * 6)


def test_fresh_process_and_cli_do_not_import_fourteen_state_components(tmp_path):
    package = Path(__file__).resolve().parents[1] / "src"
    environment = dict(os.environ, PYTHONPATH=str(package), PYTHONDONTWRITEBYTECODE="1")
    script = "from wheel_leg_lqr.scheduling.ground10_lqr import compute_gain_node; compute_gain_node(0.24,0.32); import sys; assert 'wheel_leg_lqr.models.ground.model' not in sys.modules; assert 'wheel_leg_lqr.models.ground.config' not in sys.modules"
    checked = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stderr
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "wheel_leg_lqr.cli.ground10_lqr",
            "--mode",
            "node",
            "--l1",
            "0.24",
            "--l2",
            "0.32",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    saved = json.loads((tmp_path / "ground10_node.json").read_text(encoding="utf-8"))
    assert np.asarray(saved["A"]).shape == (10, 10)
    assert np.asarray(saved["B"]).shape == (10, 4)
