"""Weighted shape coordinates for the contact-free hip controller."""
from dataclasses import dataclass
import numpy as np
from ...config import DEFAULT_ROBOT, RobotParameters
from . import config
from .model import HIP_INPUT, GLOBAL_ROTATION, linearize_airborne
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

def build_shape_model(
    l1: float,
    l2: float,
    *,
    parameters: RobotParameters = DEFAULT_ROBOT,
    q_diag: np.ndarray | None = None,
    r_diag: np.ndarray | None = None,
) -> AirborneLinearization:
    """Linearize the free-floating plant about zero pitch/leg angles and rates."""
    plant = linearize_airborne(l1, l2, parameters=parameters)
    mass = plant.mass
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

    A, B = plant.A, plant.B
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
