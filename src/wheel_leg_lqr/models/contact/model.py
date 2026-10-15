from __future__ import annotations
from ..leg_geometry import ground_inertia_com_coefficients
from ..ground.config import N_U
from ...config import DEFAULT_ROBOT, RobotParameters
from dataclasses import dataclass
import numpy as np


N_FREE_Q = 9
Q7 = slice(0, 7)
_WHEEL_1 = 0
_WHEEL_2 = 1


@dataclass(frozen=True)
class _Kinematics:
    """Particle positions and maps in the heading frame for nine coordinates."""

    lengths: np.ndarray
    positions: np.ndarray  # Wheel 1, wheel 2, leg 1 COM, leg 2 COM, body COM.
    maps: np.ndarray  # Shape (5, 3, 9).
    length_jacobian: np.ndarray  # Shape (2, 9).


def _raw_kinematics(q: np.ndarray, parameters: RobotParameters = DEFAULT_ROBOT) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate particles in the original heading frame plus free kappa/z_c."""
    _, _, theta, phi, l_bar, beta_1, beta_2, kappa, z_c = np.asarray(
        q, dtype=float
    ).reshape(N_FREE_Q)
    p = parameters
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, c_phi, -s_phi], [0.0, s_phi, c_phi]]
    )
    rotation_y = np.array(
        [[c_theta, 0.0, s_theta], [0.0, 1.0, 0.0], [-s_theta, 0.0, c_theta]]
    )
    body_to_heading = rotation_y @ rotation_x
    geometry_offset = p.hip_half_track * c_theta * s_phi - kappa
    sec_1, sec_2 = 1.0 / np.cos(beta_1), 1.0 / np.cos(beta_2)
    height = (2.0 * l_bar - geometry_offset * (sec_1 - sec_2)) / (sec_1 + sec_2)
    lengths = np.array(
        [
            (height + geometry_offset) / np.cos(beta_1),
            (height - geometry_offset) / np.cos(beta_2),
        ],
        dtype=float,
    )
    betas = (beta_1, beta_2)
    signs = (1.0, -1.0)
    bar_s = 0.5 * sum(
        length * np.sin(beta) for length, beta in zip(lengths, betas)
    )
    hip_mid = np.array([-bar_s, 0.0, height], dtype=float)
    wheel_positions: list[np.ndarray] = []
    leg_positions: list[np.ndarray] = []
    for sign, length, beta in zip(signs, lengths, betas):
        hip = hip_mid + body_to_heading @ np.array(
            [0.0, sign * p.hip_half_track, 0.0]
        )
        direction = np.array([-np.sin(beta), 0.0, np.cos(beta)])
        lower_pivot = hip - length * direction
        wheel_positions.append(
            lower_pivot
            + np.array([0.0, sign * (p.wheel_half_track - p.hip_half_track), z_c])
        )
        l_wp = np.polynomial.polynomial.polyval(length, p.leg_com_lwp_poly)
        x_com = np.polynomial.polynomial.polyval(length, p.effective_leg_com_x_poly)
        normal = np.array([np.cos(beta), 0.0, np.sin(beta)])
        leg_positions.append(
            lower_pivot
            + l_wp * direction
            + x_com * normal
            + np.array([0.0, 0.0, z_c])
        )
    body = (
        hip_mid
        + body_to_heading @ np.asarray(p.body_com_from_hip)
        + np.array([0.0, 0.0, z_c])
    )
    return lengths, np.asarray(
        [*wheel_positions, *leg_positions, body], dtype=float
    )


def _kinematics(q: np.ndarray, parameters: RobotParameters = DEFAULT_ROBOT) -> _Kinematics:
    q = np.asarray(q, dtype=float).reshape(N_FREE_Q)
    lengths, positions = _raw_kinematics(q, parameters)
    maps = np.empty((5, 3, N_FREE_Q), dtype=float)
    length_jacobian = np.empty((2, N_FREE_Q), dtype=float)
    for index in range(N_FREE_Q):
        step = 1e-6 * max(1.0, abs(q[index]))
        plus, minus = q.copy(), q.copy()
        plus[index] += step
        minus[index] -= step
        lengths_plus, positions_plus = _raw_kinematics(plus, parameters)
        lengths_minus, positions_minus = _raw_kinematics(minus, parameters)
        maps[:, :, index] = (positions_plus - positions_minus) / (2.0 * step)
        length_jacobian[:, index] = (lengths_plus - lengths_minus) / (2.0 * step)
    # The position vectors use a heading frame. Restore translational and yaw
    # transport terms which are implicit in that representation.
    maps[:, :, 0] = np.array([1.0, 0.0, 0.0])
    maps[:, :, 1] += np.cross(np.array([0.0, 0.0, 1.0]), positions)
    return _Kinematics(lengths, positions, maps, length_jacobian)


def _mass_and_input(
    q: np.ndarray,
    parameters: RobotParameters = DEFAULT_ROBOT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, _Kinematics]:
    """Return free-flight M, gravity, S and kinematics.

    The static source model has no C(q,qdot)qdot term. This extension retains
    that convention instead of introducing an unvalidated velocity model.
    """
    kin = _kinematics(q, parameters)
    p = parameters
    mass = np.zeros((N_FREE_Q, N_FREE_Q), dtype=float)
    for wheel_map in kin.maps[:2]:
        rolling = wheel_map[0]
        mass += p.wheel_mass * wheel_map.T @ wheel_map
        mass += (
            p.wheel_rolling_inertia / p.wheel_radius**2
            * np.outer(rolling, rolling)
        )
    for leg_map in kin.maps[2:4]:
        mass += p.leg_mass * leg_map.T @ leg_map
    mass += p.body_mass * kin.maps[4].T @ kin.maps[4]
    for index, length in enumerate(kin.lengths):
        l_ph = length - np.polynomial.polynomial.polyval(
            length, ground_inertia_com_coefficients(parameters)
        )
        inertia = p.leg_mass * length**2 / 12.0 + p.leg_mass * (
            length / 2.0 - l_ph
        ) ** 2
        mass[5 + index, 5 + index] += inertia
    mass = 0.5 * (mass + mass.T)
    particle_masses = np.array(
        [p.wheel_mass, p.wheel_mass, p.leg_mass, p.leg_mass, p.body_mass]
    )
    gravity = p.gravity * np.einsum(
        "p,pj->j", particle_masses, kin.maps[:, 2, :]
    )
    inputs = np.zeros((N_FREE_Q, N_U), dtype=float)
    for index, wheel_map in enumerate(kin.maps[:2]):
        inputs[:, index] = wheel_map[0] / p.wheel_radius
        inputs[5 + index, index] += 1.0
    for index in range(2):
        inputs[2, 2 + index] = 1.0
        inputs[5 + index, 2 + index] = 1.0
        inputs[:, 4 + index] = kin.length_jacobian[index]
    return mass, gravity, inputs, kin


def _compliant_acceleration(
    q: np.ndarray,
    velocity: np.ndarray,
    control: np.ndarray,
    ground: np.ndarray,
    *,
    parameters: RobotParameters = DEFAULT_ROBOT,
    ground_stiffness: float,
    ground_damping: float,
    stop_stiffness: float,
    stop_damping: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, _Kinematics]:
    """Return acceleration with unilateral spring-damper ground and leg stops."""
    mass, gravity, inputs, kin = _mass_and_input(q, parameters)
    generalized_contact = np.zeros(N_FREE_Q, dtype=float)
    wheel_forces = np.zeros(2, dtype=float)
    stop_forces = np.zeros(2, dtype=float)
    for wheel in (_WHEEL_1, _WHEEL_2):
        gap = float(kin.positions[wheel, 2] - ground[wheel])
        normal_speed = float(kin.maps[wheel, 2] @ velocity)
        if gap < 0.0:
            wheel_forces[wheel] = max(
                0.0, -ground_stiffness * gap - ground_damping * normal_speed
            )
            generalized_contact += kin.maps[wheel, 2] * wheel_forces[wheel]
    for leg in (_WHEEL_1, _WHEEL_2):
        upper_gap = parameters.max_leg_length - kin.lengths[leg]
        upper_map = -kin.length_jacobian[leg]
        lower_gap = kin.lengths[leg] - parameters.min_leg_length
        lower_map = kin.length_jacobian[leg]
        if upper_gap < 0.0:
            stop_forces[leg] = max(
                0.0,
                -stop_stiffness * upper_gap - stop_damping * (upper_map @ velocity),
            )
            generalized_contact += upper_map * stop_forces[leg]
        elif lower_gap < 0.0:
            stop_forces[leg] = max(
                0.0,
                -stop_stiffness * lower_gap - stop_damping * (lower_map @ velocity),
            )
            generalized_contact += lower_map * stop_forces[leg]
    acceleration = np.linalg.solve(
        mass, inputs @ control - gravity + generalized_contact
    )
    return acceleration, wheel_forces, stop_forces, kin
