


"""基于用户文档的动力学 / Dynamics based on the user document."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parameters import RobotParameters

STATE_NAMES = ("x", "dx", "beta", "dbeta", "gamma", "dgamma")
INPUT_NAMES = ("T_w", "T_p")


@dataclass(frozen=True)
class KinematicAccelerations:
    """COM accelerations / """

    xp: float
    yp: float
    xb: float
    yb: float


@dataclass(frozen=True)
class ConstraintForces:
    """Internal forces / """

    n_pb: float
    p_pb: float
    n_wp: float
    p_wp: float


def _as_vector(value, size: int, name: str) -> np.ndarray:
    """Convert and validate a vector / """
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size != size:
        raise ValueError(f"{name} must have {size} elements, got {result.size}")
    return result


def _kinematic_accelerations(
    state: np.ndarray,
    qdd: np.ndarray,
    p: RobotParameters,
) -> KinematicAccelerations:
    """Evaluate the document kinematics / 

    Coordinate definitions / 

    ``x_p = x - L_wp sin(beta)``, ``y_p = L_wp cos(beta)``;
    ``x_b = x - L sin(beta) + L_c sin(gamma)``;
    ``y_b = L cos(beta) + L_c cos(gamma)``.
    """
    _, _, beta, dbeta, gamma, dgamma = state
    ddx, ddbeta, ddgamma = qdd
    leg_length = p.leg_length

    # Leg COM acceleration / 
    xp = ddx + p.l_wp * np.sin(beta) * dbeta**2 - p.l_wp * np.cos(beta) * ddbeta
    yp = -p.l_wp * np.cos(beta) * dbeta**2 - p.l_wp * np.sin(beta) * ddbeta

    # Body COM acceleration / 
    xb = (
        ddx
        + leg_length * np.sin(beta) * dbeta**2
        - p.l_c * np.sin(gamma) * dgamma**2
        - leg_length * np.cos(beta) * ddbeta
        + p.l_c * np.cos(gamma) * ddgamma
    )
    yb = (
        -leg_length * np.cos(beta) * dbeta**2
        - p.l_c * np.cos(gamma) * dgamma**2
        - leg_length * np.sin(beta) * ddbeta
        - p.l_c * np.sin(gamma) * ddgamma
    )
    return KinematicAccelerations(xp=xp, yp=yp, xb=xb, yb=yb)


def constraint_forces(
    state,
    qdd,
    p: RobotParameters,
) -> ConstraintForces:
    """Evaluate ``N/P`` forces from the document / """
    state = _as_vector(state, 6, "state")
    qdd = _as_vector(qdd, 3, "qdd")
    acceleration = _kinematic_accelerations(state, qdd, p)

    # These four equations are copied from the new document.
    # 
    n_pb = p.mb * acceleration.xb
    p_pb = p.mb * (p.g + acceleration.yb)
    n_wp = n_pb + p.mp * acceleration.xp
    p_wp = (p.mb + p.mp) * p.g + p.mb * acceleration.yb + p.mp * acceleration.yp
    return ConstraintForces(n_pb=n_pb, p_pb=p_pb, n_wp=n_wp, p_wp=p_wp)


def _moments(
    state: np.ndarray,
    qdd: np.ndarray,
    p: RobotParameters,
) -> tuple[float, float, ConstraintForces]:
    """Evaluate leg/body moments / """
    _, _, beta, _, gamma, _ = state
    forces = constraint_forces(state, qdd, p)

    # Leg moment about P /  P 
    leg_moment = (
        p.l_wp * (forces.p_wp * np.sin(beta) + forces.n_wp * np.cos(beta))
        + p.l_ph * (forces.p_pb * np.sin(beta) + forces.n_pb * np.cos(beta))
    )

    # Body moment with clockwise-positive gamma / gamma 
    body_moment = p.l_c * (
        forces.p_pb * np.sin(gamma) - forces.n_pb * np.cos(gamma)
    )
    return leg_moment, body_moment, forces


def _residual(
    qdd: np.ndarray,
    state: np.ndarray,
    inputs: np.ndarray,
    p: RobotParameters,
) -> np.ndarray:
    """Return residual of the three document equations / """
    ddx, ddbeta, ddgamma = qdd
    tw, tp = inputs
    leg_moment, body_moment, forces = _moments(state, qdd, p)
    effective_wheel_mass = p.mw + p.iw / p.rw**2

    # Residual = left minus right / 
    return np.array(
        [
            effective_wheel_mass * ddx + forces.n_wp - tw / p.rw,
            # Corrected document sign: +T_p acts on the leg.
            p.ip * ddbeta - (tp + tw + leg_moment),
            # Corrected document sign: +T_p acts on the body.
            p.ib * ddgamma - (tp + body_moment),
        ],
        dtype=float,
    )


def _solve_acceleration(
    state: np.ndarray,
    inputs: np.ndarray,
    p: RobotParameters,
) -> np.ndarray:
    """Solve the implicit acceleration equation / 

    The residual is affine in ``qdd``. Unit-vector probing constructs the
    mass matrix without introducing a second symbolic implementation.
    /  ``qdd`` 
    
    """
    zero = np.zeros(3, dtype=float)
    residual_at_zero = _residual(zero, state, inputs, p)
    mass_matrix = np.column_stack(
        [_residual(np.eye(3)[index], state, inputs, p) - residual_at_zero for index in range(3)]
    )
    return np.linalg.solve(mass_matrix, -residual_at_zero)


def accelerations(state, inputs, p: RobotParameters) -> np.ndarray:
    """Return ``[ddx, ddbeta, ddgamma]`` / """
    state = _as_vector(state, 6, "state")
    inputs = _as_vector(inputs, 2, "inputs")
    return _solve_acceleration(state, inputs, p)


def state_derivative(state, inputs, p: RobotParameters) -> np.ndarray:
    """Evaluate ``dot(X)=f(X,U)`` / """
    state = _as_vector(state, 6, "state")
    acceleration = accelerations(state, inputs, p)
    return np.array(
        [state[1], acceleration[0], state[3], acceleration[1], state[5], acceleration[2]],
        dtype=float,
    )


