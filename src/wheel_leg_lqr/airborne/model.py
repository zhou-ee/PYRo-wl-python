"""Contact-free planar pitch/leg model for the two hip motors.

The angles have the same signs as the 14-state model: beta_i is the absolute
leg angle, opposite in physical rotation sign to body pitch theta.  A positive
T_p_i therefore has positive generalized force in both beta_i and theta.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from . import config


AIRBORNE_STATE_NAMES = (
    "beta_1", "dbeta_1", "beta_2", "dbeta_2", "theta", "dtheta"
)
AIRBORNE_INPUT_NAMES = ("T_p1", "T_p2")
HIP_INPUT = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
# A rigid rotation changes theta and each beta in opposite directions.
GLOBAL_ROTATION = np.array([-1.0, -1.0, 1.0])


@dataclass(frozen=True)
class AirborneLinearization:
    l1: float
    l2: float
    mass: np.ndarray  # Translation-eliminated mass in [beta_1, beta_2, theta].
    shape_mass: np.ndarray
    shape_lift: np.ndarray  # y = shape_lift @ r + GLOBAL_ROTATION * gamma.
    shape_projection: np.ndarray  # Weighted controllable coordinate z = H @ y.
    A: np.ndarray  # [beta_1, dbeta_1, beta_2, dbeta_2, theta, dtheta].
    B: np.ndarray  # [T_p1, T_p2].
    A_shape: np.ndarray  # [z_1, dz_1, z_2, dz_2].
    B_shape: np.ndarray
    state_lift: np.ndarray  # 6 x 4 mapping from shape states to physical states.
    state_projection: np.ndarray  # 4 x 6 mapping from physical to shape states.
    q_diag: np.ndarray  # Physical six-state diagonal Q.
    r_diag: np.ndarray  # Two-input diagonal R.
    Q_shape: np.ndarray
    R: np.ndarray


def _poly(coef: tuple[float, ...], length: float) -> float:
    return float(np.polynomial.polynomial.polyval(length, coef))


def _poly_derivative(coef: tuple[float, ...], length: float) -> float:
    return float(np.polynomial.polynomial.polyval(
        length, np.polynomial.polynomial.polyder(coef)
    ))


def airborne_generalized_mass(
    l1: float,
    l2: float,
    *,
    beta_1: float = 0.0,
    beta_2: float = 0.0,
    theta: float = 0.0,
    parameters: config.AirborneParameters | None = None,
) -> np.ndarray:
    """Return five-coordinate free-flight mass after eliminating X/Z translation.

    Coordinates are [beta_1, beta_2, theta, L1, L2].  The wheel masses
    translate with the legs, while freely spinning wheel rotor inertia is
    excluded.  Derivatives of the fitted leg COM enter the length columns.
    The leg's local pitch inertia uses a uniform-rod estimate adjusted by
    the CAD centre-of-mass offset; it is an airborne-model assumption.
    """
    lengths = (float(l1), float(l2))
    angles = (float(beta_1), float(beta_2))
    if not all(np.isfinite(value) for value in (*lengths, *angles, theta)):
        raise ValueError("lengths and angles must be finite")
    if min(lengths) <= 0.0:
        raise ValueError("leg lengths must be positive")

    p = config.PARAMETERS if parameters is None else parameters
    total_mass = p.body_mass + 2.0 * (p.leg_mass + p.wheel_mass)
    translation_coupling = np.zeros((2, 5), dtype=float)
    relative_mass = np.zeros((5, 5), dtype=float)

    body_x = p.body_com_x_from_hip
    body_z = p.body_com_z_from_hip
    body_jacobian = np.zeros((2, 5), dtype=float)
    body_jacobian[:, 2] = [
        -body_x * np.sin(theta) + body_z * np.cos(theta),
        -body_x * np.cos(theta) - body_z * np.sin(theta),
    ]
    translation_coupling += p.body_mass * body_jacobian
    relative_mass += p.body_mass * body_jacobian.T @ body_jacobian
    relative_mass[2, 2] += p.body_pitch_inertia

    for index, (length, beta) in enumerate(zip(lengths, angles)):
        l_wp = _poly(p.leg_com_lwp_poly, length)
        x_com = _poly(p.leg_com_x_poly, length)
        l_wp_prime = _poly_derivative(p.leg_com_lwp_poly, length)
        x_com_prime = _poly_derivative(p.leg_com_x_poly, length)
        leg_from_hip = length - l_wp
        leg_from_hip_prime = 1.0 - l_wp_prime
        wheel_jacobian = np.zeros((2, 5), dtype=float)
        wheel_jacobian[:, index] = [length * np.cos(beta), length * np.sin(beta)]
        wheel_jacobian[:, 3 + index] = [np.sin(beta), -np.cos(beta)]
        leg_jacobian = np.zeros((2, 5), dtype=float)
        leg_jacobian[:, index] = [
            leg_from_hip * np.cos(beta) - x_com * np.sin(beta),
            leg_from_hip * np.sin(beta) + x_com * np.cos(beta),
        ]
        leg_jacobian[:, 3 + index] = [
            leg_from_hip_prime * np.sin(beta) + x_com_prime * np.cos(beta),
            -leg_from_hip_prime * np.cos(beta) + x_com_prime * np.sin(beta),
        ]
        for mass, jacobian in (
            (p.wheel_mass, wheel_jacobian),
            (p.leg_mass, leg_jacobian),
        ):
            translation_coupling += mass * jacobian
            relative_mass += mass * jacobian.T @ jacobian
        relative_mass[index, index] += p.leg_mass * (
            length**2 / 12.0 + (l_wp - length / 2.0) ** 2
        )

    mass = relative_mass - translation_coupling.T @ translation_coupling / total_mass
    mass = (mass + mass.T) / 2.0
    if np.min(np.linalg.eigvalsh(mass)) <= 0.0:
        raise ValueError("airborne generalized mass must be positive definite")
    return mass


def airborne_rotational_mass(
    l1: float,
    l2: float,
    *,
    beta_1: float = 0.0,
    beta_2: float = 0.0,
    theta: float = 0.0,
    parameters: config.AirborneParameters | None = None,
) -> np.ndarray:
    """Return the [beta_1, beta_2, theta] mass with frozen leg lengths."""
    return airborne_generalized_mass(
        l1, l2, beta_1=beta_1, beta_2=beta_2, theta=theta,
        parameters=parameters,
    )[:3, :3]


@dataclass(frozen=True)
class LegForcePitchEffect:
    generalized_mass: np.ndarray
    pitch_acceleration_per_newton: np.ndarray  # [F_L1, F_L2], rad/s^2/N.
    body_moment_arm: np.ndarray  # Body-only pitch moment per newton, m.
    body_pitch_acceleration_per_newton: np.ndarray


def compute_leg_force_pitch_effect(
    l1: float,
    l2: float,
    *,
    body_com_x_from_hip: float = 0.03,
    beta_1: float = 0.0,
    beta_2: float = 0.0,
    theta: float = 0.0,
) -> LegForcePitchEffect:
    """Compute instantaneous free-flight pitch acceleration from axial F_L.

    Positive F_L extends a leg: it pushes the lower assembly away from the
    hip and exerts the opposite axial force on the body at the hip.  This is
    an input-map calculation at zero generalized velocities, not a prediction
    of a sustained force pulse or a leg-length controller.
    """
    if not np.isfinite(body_com_x_from_hip):
        raise ValueError("body_com_x_from_hip must be finite")
    p = replace(config.PARAMETERS, body_com_x_from_hip=float(body_com_x_from_hip))
    mass = airborne_generalized_mass(
        l1, l2, beta_1=beta_1, beta_2=beta_2, theta=theta, parameters=p
    )
    leg_force_input = np.zeros((5, 2), dtype=float)
    leg_force_input[3:, :] = np.eye(2)
    pitch_acceleration = np.linalg.solve(mass, leg_force_input)[2]

    body_com_x = p.body_com_x_from_hip * np.cos(theta) + p.body_com_z_from_hip * np.sin(theta)
    body_com_z = -p.body_com_x_from_hip * np.sin(theta) + p.body_com_z_from_hip * np.cos(theta)
    betas = np.array([beta_1, beta_2], dtype=float)
    moment_arms = body_com_x * np.cos(betas) + body_com_z * np.sin(betas)
    return LegForcePitchEffect(
        generalized_mass=mass,
        pitch_acceleration_per_newton=pitch_acceleration,
        body_moment_arm=moment_arms,
        body_pitch_acceleration_per_newton=moment_arms / p.body_pitch_inertia,
    )


def _positive_diagonal(
    values: np.ndarray | None,
    default: np.ndarray,
    size: int,
    name: str,
) -> np.ndarray:
    diagonal = np.asarray(default if values is None else values, dtype=float)
    if diagonal.shape != (size,) or not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise ValueError(f"{name} must contain {size} positive finite diagonal weights")
    return diagonal.copy()


def linearize_airborne(
    l1: float,
    l2: float,
    *,
    q_diag: np.ndarray | None = None,
    r_diag: np.ndarray | None = None,
) -> AirborneLinearization:
    """Linearize the free-floating plant about zero pitch/leg angles and rates."""
    mass = airborne_rotational_mass(l1, l2)
    q_diag = _positive_diagonal(q_diag, config.Q_DIAG, 6, "q_diag")
    r_diag = _positive_diagonal(r_diag, config.R_DIAG, 2, "r_diag")
    angle_q = np.diag(q_diag[[0, 2, 4]])
    velocity_q = np.diag(q_diag[[1, 3, 5]])
    input_r = np.diag(r_diag)

    # r = [theta+beta_1, theta+beta_2] are the controllable hip angles.
    # gamma is the angular-momentum coordinate, unaffected by T_p.
    gamma_row = GLOBAL_ROTATION @ mass / (GLOBAL_ROTATION @ mass @ GLOBAL_ROTATION)
    shape_lift = np.linalg.solve(
        np.vstack((HIP_INPUT.T, gamma_row)),
        np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
    )
    shape_mass = shape_lift.T @ mass @ shape_lift
    # H maps an arbitrary physical angle error to the best reachable shape
    # error under the configured beta/theta position weights.
    shape_projection = np.linalg.solve(
        shape_lift.T @ angle_q @ shape_lift,
        shape_lift.T @ angle_q,
    )

    A = np.zeros((6, 6), dtype=float)
    A[0, 1] = A[2, 3] = A[4, 5] = 1.0
    B = np.zeros((6, 2), dtype=float)
    B[1::2, :] = np.linalg.solve(mass, HIP_INPUT)
    A_shape = np.zeros((4, 4), dtype=float)
    A_shape[0, 1] = A_shape[2, 3] = 1.0
    B_shape = np.zeros((4, 2), dtype=float)
    B_shape[1::2, :] = np.linalg.inv(shape_mass)

    state_lift = np.zeros((6, 4), dtype=float)
    state_lift[0::2, 0::2] = shape_lift
    state_lift[1::2, 1::2] = shape_lift
    state_projection = np.zeros((4, 6), dtype=float)
    state_projection[0::2, 0::2] = shape_projection
    state_projection[1::2, 1::2] = shape_projection
    Q_shape = np.zeros((4, 4), dtype=float)
    Q_shape[0::2, 0::2] = shape_lift.T @ angle_q @ shape_lift
    Q_shape[1::2, 1::2] = shape_lift.T @ velocity_q @ shape_lift
    Q_shape = (Q_shape + Q_shape.T) / 2.0
    return AirborneLinearization(
        l1=float(l1),
        l2=float(l2),
        mass=mass,
        shape_mass=shape_mass,
        shape_lift=shape_lift,
        shape_projection=shape_projection,
        A=A,
        B=B,
        A_shape=A_shape,
        B_shape=B_shape,
        state_lift=state_lift,
        state_projection=state_projection,
        q_diag=q_diag,
        r_diag=r_diag,
        Q_shape=Q_shape,
        R=input_r,
    )
