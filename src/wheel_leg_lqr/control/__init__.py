"""Reusable control primitives."""

from .leso import (
    DEFAULT_LESO_DESIGN,
    N_AUGMENTED,
    N_DISTURBANCE,
    LesoDesignConfig,
    LesoEstimate,
    LesoStepResult,
    build_augmented_model,
    design_leso_gain,
    initialize_leso,
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
    "DEFAULT_LESO_DESIGN",
    "N_AUGMENTED",
    "N_DISTURBANCE",
    "LesoDesignConfig",
    "LesoEstimate",
    "LesoStepResult",
    "SaturationValidation",
    "build_augmented_model",
    "controllability_rank",
    "default_initial_errors",
    "design_leso_gain",
    "discretize_state_space",
    "initialize_leso",
    "observer_error_spectral_radius",
    "simulate_saturated_closed_loop",
    "simulate_saturated_closed_loop_batch",
    "solve_discrete_lqr",
    "step_leso_split",
    "validate_saturated_closed_loop",
]
