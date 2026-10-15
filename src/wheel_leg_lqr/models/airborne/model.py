"""Contact-free planar pitch/leg model for the two hip motors.

The angles have the same signs as the 14-state model: beta_i is the absolute
leg angle, opposite in physical rotation sign to body pitch theta.  A positive
T_p_i therefore has positive generalized force in both beta_i and theta.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ...config import DEFAULT_ROBOT, RobotParameters


from .config import AIRBORNE_STATE_NAMES as AIRBORNE_STATE_NAMES, AIRBORNE_INPUT_NAMES as AIRBORNE_INPUT_NAMES
__all__ = [
    "AIRBORNE_STATE_NAMES", "AIRBORNE_INPUT_NAMES", "HIP_INPUT", "GLOBAL_ROTATION",
    "airborne_generalized_mass", "airborne_rotational_mass", "LegForcePitchEffect",
    "compute_leg_force_pitch_effect", "AirbornePlant", "linearize_airborne",
]
HIP_INPUT = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
# A rigid rotation changes theta and each beta in opposite directions.
GLOBAL_ROTATION = np.array([-1.0, -1.0, 1.0])


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
    parameters: RobotParameters | None = None,
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

    p = DEFAULT_ROBOT if parameters is None else parameters
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
        x_com = _poly(p.effective_leg_com_x_poly, length)
        l_wp_prime = _poly_derivative(p.leg_com_lwp_poly, length)
        x_com_prime = _poly_derivative(p.effective_leg_com_x_poly, length)
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
    parameters: RobotParameters | None = None,
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
    body_com_x_from_hip: float | None = None,
    beta_1: float = 0.0,
    beta_2: float = 0.0,
    theta: float = 0.0,
    parameters: RobotParameters = DEFAULT_ROBOT,
) -> LegForcePitchEffect:
    """Compute instantaneous free-flight pitch acceleration from axial F_L.

    Positive F_L extends a leg: it pushes the lower assembly away from the
    hip and exerts the opposite axial force on the body at the hip.  This is
    an input-map calculation at zero generalized velocities, not a prediction
    of a sustained force pulse or a leg-length controller.
    """
    body_com_x_from_hip = parameters.body_com_x_from_hip if body_com_x_from_hip is None else body_com_x_from_hip
    if not np.isfinite(body_com_x_from_hip):
        raise ValueError("body_com_x_from_hip must be finite")
    p = replace(parameters, body_com_from_hip=(float(body_com_x_from_hip), *parameters.body_com_from_hip[1:]))
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

@dataclass(frozen=True)
class AirbornePlant:
    l1: float
    l2: float
    mass: np.ndarray
    A: np.ndarray
    B: np.ndarray

def linearize_airborne(l1, l2, *, parameters: RobotParameters = DEFAULT_ROBOT):
    """Physical six-state plant only; no Q/R or weighted projection."""
    mass = airborne_rotational_mass(l1, l2, parameters=parameters)
    A = np.zeros((6, 6))
    A[0, 1] = A[2, 3] = A[4, 5] = 1.0
    B = np.zeros((6, 2))
    B[1::2] = np.linalg.solve(mass, HIP_INPUT)
    return AirbornePlant(float(l1), float(l2), mass, A, B)
