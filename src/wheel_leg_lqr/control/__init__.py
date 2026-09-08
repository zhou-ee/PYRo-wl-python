"""Reusable control primitives."""

from .leso import (
    DEFAULT_DISTURBANCE_POLES,
    DEFAULT_FL_DISTURBANCE_BANDWIDTH_HZ,
    DEFAULT_FL_STATE_BANDWIDTH_HZ,
    DEFAULT_STATE_POLES,
    DEFAULT_TORQUE_DISTURBANCE_BANDWIDTH_HZ,
    DEFAULT_TORQUE_STATE_BANDWIDTH_HZ,
    DEFAULT_UNMATCHED_STATE_BANDWIDTH_HZ,
    DEFAULT_UNMATCHED_STATE_POLE,
    LesoCorrection,
    build_augmented_model,
    design_leso_correction,
    discrete_pole_from_bandwidth,
    observer_error_spectral_radius,
    step_leso_split,
)
from .lqr import controllability_rank, discretize_state_space, solve_discrete_lqr
from .saturation import (
    SaturationValidation,
    default_initial_errors,
    simulate_saturated_closed_loop,
    simulate_saturated_closed_loop_batch,
    validate_saturated_closed_loop,
)

__all__ = [
    "DEFAULT_DISTURBANCE_POLES",
    "DEFAULT_FL_DISTURBANCE_BANDWIDTH_HZ",
    "DEFAULT_FL_STATE_BANDWIDTH_HZ",
    "DEFAULT_STATE_POLES",
    "DEFAULT_TORQUE_DISTURBANCE_BANDWIDTH_HZ",
    "DEFAULT_TORQUE_STATE_BANDWIDTH_HZ",
    "DEFAULT_UNMATCHED_STATE_BANDWIDTH_HZ",
    "DEFAULT_UNMATCHED_STATE_POLE",
    "LesoCorrection",
    "SaturationValidation",
    "build_augmented_model",
    "controllability_rank",
    "default_initial_errors",
    "design_leso_correction",
    "discrete_pole_from_bandwidth",
    "discretize_state_space",
    "observer_error_spectral_radius",
    "simulate_saturated_closed_loop",
    "simulate_saturated_closed_loop_batch",
    "solve_discrete_lqr",
    "step_leso_split",
    "validate_saturated_closed_loop",
]
