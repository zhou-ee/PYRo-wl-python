from __future__ import annotations
from ..config import DEFAULT_ROBOT, RobotParameters
from ..control.lqr import discretize_state_space, controllability_rank, solve_discrete_lqr
from .polynomial import fit_tensor_coefficients
from .domain import make_leg_length_grid as inclusive_grid
from dataclasses import dataclass
import numpy as np
from ..models.airborne import config
from ..models.airborne.reduction import AirborneLinearization, build_shape_model


@dataclass(frozen=True)
class AirborneGainNode:
    l1: float
    l2: float
    linearization: AirborneLinearization
    Ad: np.ndarray
    Bd: np.ndarray
    Ad_shape: np.ndarray
    Bd_shape: np.ndarray
    gain: np.ndarray  # [T_p1, T_p2] x six interleaved physical states.
    shape_poles: np.ndarray
    parameters: RobotParameters = DEFAULT_ROBOT

@dataclass(frozen=True)
class AirborneScheduleFit:
    l1_grid: np.ndarray
    l2_grid: np.ndarray
    sample_time: float
    q_diag: np.ndarray
    r_diag: np.ndarray
    nodes: tuple[AirborneGainNode, ...]
    gain_coefficients: np.ndarray  # [input, state, p, q].
    fitted_gains: np.ndarray
    max_abs_gain_error: float
    max_relative_gain_error: float
    max_raw_shape_radius: float
    max_float32_shape_radius: float
    validation_node_count: int
    parameters: RobotParameters = DEFAULT_ROBOT

def make_leg_length_grid(step, parameters=DEFAULT_ROBOT):
    return inclusive_grid(parameters.min_leg_length, parameters.max_leg_length, step)


def _discretize(A, B, sample_time):
    return discretize_state_space(A, B, sample_time=sample_time)


def _controllability_rank(A, B):
    return controllability_rank(A, B)


def _discrete_lqr(Ad, Bd, Q, R):
    gain, _, poles = solve_discrete_lqr(Ad, Bd, Q, R)
    return gain, poles


def _fit_tensor_gains(lam1, lam2, gains, degree):
    return fit_tensor_coefficients(lam1, lam2, gains, degree, degree)


def _normalization(parameters: RobotParameters = DEFAULT_ROBOT) -> tuple[float, float]:
    p = parameters
    return (
        (p.min_leg_length + p.max_leg_length) / 2.0,
        (p.max_leg_length - p.min_leg_length) / 2.0,
    )


def compute_airborne_gain_node(
    l1: float,
    l2: float,
    *,
    parameters: RobotParameters = DEFAULT_ROBOT,
    sample_time: float = config.SAMPLE_TIME,
    q_diag: np.ndarray | None = None,
    r_diag: np.ndarray | None = None,
) -> AirborneGainNode:
    """Solve discrete LQR on four controllable shape states at one node."""
    plant = build_shape_model(l1, l2, parameters=parameters, q_diag=q_diag, r_diag=r_diag)
    Ad, Bd = _discretize(plant.A, plant.B, sample_time)
    Ad_shape, Bd_shape = _discretize(plant.A_shape, plant.B_shape, sample_time)
    if _controllability_rank(Ad_shape, Bd_shape) != 4:
        raise ValueError(f"airborne hip shape is not controllable at ({l1}, {l2})")
    shape_gain, shape_poles = _discrete_lqr(Ad_shape, Bd_shape, plant.Q_shape, plant.R)
    gain = shape_gain @ plant.state_projection
    return AirborneGainNode(
        l1=float(l1),
        l2=float(l2),
        linearization=plant,
        Ad=Ad,
        Bd=Bd,
        Ad_shape=Ad_shape,
        Bd_shape=Bd_shape,
        gain=gain,
        shape_poles=shape_poles, parameters=parameters,
    )


def evaluate_airborne_gain(
    coefficients: np.ndarray,
    l1: float,
    l2: float,
    *,
    dtype: type = float,
    lower: float = DEFAULT_ROBOT.min_leg_length,
    upper: float = DEFAULT_ROBOT.max_leg_length,
) -> np.ndarray:
    """Evaluate the exported tensor polynomial with float64 or MCU float32."""
    coefficients = np.asarray(coefficients, dtype=dtype)
    if coefficients.ndim != 4 or coefficients.shape[:2] != (2, 6):
        raise ValueError("coefficients must have shape (2, 6, degree1+1, degree2+1)")
    center, scale = (lower + upper) / 2, (upper - lower) / 2
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("length domain requires finite lower < upper")
    lam1 = dtype(np.clip((float(l1) - center) / scale, -1.0, 1.0))
    lam2 = dtype(np.clip((float(l2) - center) / scale, -1.0, 1.0))
    # Nested Horner order matches the C++ example in the documentation.
    result = np.zeros((2, 6), dtype=dtype)
    for p in range(coefficients.shape[2] - 1, -1, -1):
        row = np.zeros((2, 6), dtype=dtype)
        for q in range(coefficients.shape[3] - 1, -1, -1):
            row = row * lam2 + coefficients[:, :, p, q]
        result = result * lam1 + row
    return result


def _shape_radius(node: AirborneGainNode, gain: np.ndarray) -> float:
    shape_gain = gain @ node.linearization.state_lift
    poles = np.linalg.eigvals(node.Ad_shape - node.Bd_shape @ shape_gain)
    return float(np.max(np.abs(poles)))


def compute_airborne_schedule(
    *,
    parameters: RobotParameters = DEFAULT_ROBOT,
    step: float = 0.01,
    degree: int = 3,
    sample_time: float = config.SAMPLE_TIME,
    q_diag: np.ndarray | None = None,
    r_diag: np.ndarray | None = None,
) -> AirborneScheduleFit:
    """Fit K(L1,L2) and verify float32 gains on grid and cell centers."""
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be positive and finite")
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    grid = make_leg_length_grid(step, parameters)
    if len(grid) <= degree:
        raise ValueError("the leg-length grid needs at least degree+1 points per axis")
    if not np.isfinite(sample_time) or sample_time <= 0.0:
        raise ValueError("sample_time must be positive and finite")
    nodes = tuple(
        compute_airborne_gain_node(
            l1, l2, parameters=parameters, sample_time=sample_time, q_diag=q_diag, r_diag=r_diag
        )
        for l1 in grid
        for l2 in grid
    )
    center, scale = _normalization(parameters)
    lam1 = np.array([(node.l1 - center) / scale for node in nodes])
    lam2 = np.array([(node.l2 - center) / scale for node in nodes])
    raw_gains = np.stack([node.gain for node in nodes])
    coefficients, fitted = _fit_tensor_gains(lam1, lam2, raw_gains, degree)
    coefficients_float32 = coefficients.astype(np.float32)
    midpoint_grid = (grid[:-1] + grid[1:]) / 2.0
    validation_nodes = [
        *nodes,
        *(
            compute_airborne_gain_node(
                l1, l2, parameters=parameters, sample_time=sample_time, q_diag=q_diag, r_diag=r_diag
            )
            for l1 in midpoint_grid
            for l2 in midpoint_grid
        ),
    ]
    max_radius = 0.0
    max_relative_error = 0.0
    for node in validation_nodes:
        gain = evaluate_airborne_gain(
            coefficients_float32, node.l1, node.l2, dtype=np.float32, lower=parameters.min_leg_length, upper=parameters.max_leg_length
        ).astype(float)
        max_radius = max(max_radius, _shape_radius(node, gain))
        max_relative_error = max(
            max_relative_error,
            float(np.linalg.norm(gain - node.gain) / np.linalg.norm(node.gain)),
        )
    if max_radius >= 1.0:
        raise ValueError("float32 fitted airborne LQR is unstable on the validation grid")
    return AirborneScheduleFit(
        l1_grid=grid,
        l2_grid=grid.copy(),
        sample_time=float(sample_time),
        q_diag=nodes[0].linearization.q_diag.copy(),
        r_diag=nodes[0].linearization.r_diag.copy(),
        nodes=nodes,
        gain_coefficients=coefficients,
        fitted_gains=fitted,
        max_abs_gain_error=float(np.max(np.abs(fitted - raw_gains))),
        max_relative_gain_error=max_relative_error,
        max_raw_shape_radius=max(float(np.max(np.abs(n.shape_poles))) for n in nodes),
        max_float32_shape_radius=max_radius,
        validation_node_count=len(validation_nodes),
        parameters=parameters,
    )
