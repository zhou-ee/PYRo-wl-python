from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..control.rdob import RdobDesign, design_rdob
from ..models.ground import config
from ..models.ground.model import StaticLinearization, linearize_static_trim
from ..config import DEFAULT_ROBOT, RobotParameters

from .config import (
    DEFAULT_BANDWIDTH_HZ, DEFAULT_DISTURBANCE_LIMITS,
    DEFAULT_DISTURBANCE_LIMIT_RATIO, DEFAULT_RDOB_STEP_POLE,
)

__all__ = [
    'DEFAULT_BANDWIDTH_HZ', 'DEFAULT_DISTURBANCE_LIMITS',
    'DEFAULT_DISTURBANCE_LIMIT_RATIO', 'DEFAULT_RDOB_STEP_POLE',
    'RdobNode', 'compute_rdob_node', 'rdob_diagnostics',
]


@dataclass(frozen=True)
class RdobNode:
    model: StaticLinearization
    observer: RdobDesign
    parameters: RobotParameters = DEFAULT_ROBOT


def compute_rdob_node(
    l1: float, l2: float, *,
    parameters: RobotParameters = DEFAULT_ROBOT,
    bandwidth_hz: np.ndarray | float | tuple[float, ...] = DEFAULT_BANDWIDTH_HZ,
    sample_time: float = config.SAMPLE_TIME,
    damping: np.ndarray | None = None,
) -> RdobNode:
    """Use existing static mechanics; D0 defaults to zero (not identified).

    The default gives all six RDOB channels ``DEFAULT_RDOB_STEP_POLE`` at the
    nominal sample time.  All six channels are observed regardless of downstream
    compensation selection.
    """
    for length in (l1, l2):
        if not np.isfinite(length):
            raise ValueError("leg lengths must be finite")
    bandwidth = np.asarray(bandwidth_hz, dtype=float)
    if bandwidth.ndim > 1 or (bandwidth.ndim == 1 and bandwidth.shape != (6,)):
        raise ValueError("bandwidth_hz must be a scalar or six values")
    if not np.all(np.isfinite(bandwidth)) or np.any(bandwidth <= 0):
        raise ValueError("bandwidth_hz must be positive and finite")
    model = linearize_static_trim(l1, l2, parameters=parameters)
    observer = design_rdob(
        model.mass, model.stiffness, model.input_matrix,
        rates_rad_s=2.0 * np.pi * bandwidth, sample_time=sample_time,
        damping=damping,
    )
    return RdobNode(model, observer, parameters)


def rdob_diagnostics(design: RdobDesign) -> dict[str, float | int]:
    """Algebraic diagnostics; these do not certify the compensated closed loop."""
    s = design.input_matrix
    projector = np.eye(config.N_Q) - s @ design.input_left_inverse
    identity_error = design.input_left_inverse @ s - np.eye(config.N_U)
    dynamic_error = (design.gamma @ np.linalg.solve(design.mass, s)
                     - np.diag(design.rates))
    gamma32 = design.gamma.astype(np.float32).astype(float)
    return {
        "observer_dynamic_states": config.N_U,
        "input_rank": int(np.linalg.matrix_rank(s)),
        "input_condition_number_coordinate_dependent": design.input_condition_number,
        "left_inverse_error_inf": float(np.linalg.norm(identity_error, ord=np.inf)),
        "matched_rate_identity_error_inf": float(np.linalg.norm(dynamic_error, ord=np.inf)),
        "orthogonal_force_rejection_error_inf": float(np.linalg.norm(
            design.input_left_inverse @ projector, ord=np.inf)),
        "auxiliary_step_spectral_radius": float(np.max(np.abs(design.step_poles))),
        "negative_auxiliary_step_poles": int(np.count_nonzero(design.step_poles < 0)),
        "gamma_float32_relative_error": float(np.linalg.norm(gamma32 - design.gamma)
                                             / np.linalg.norm(design.gamma)),
    }
