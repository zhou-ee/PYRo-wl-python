"""Reusable control primitives."""

from .lqr import controllability_rank, discretize_state_space, solve_discrete_lqr
from .rdob import (
    RdobDesign,
    design_rdob,
    initialize_rdob,
    rdob_derivative,
    reconstruct_disturbance,
    step_rdob,
    trapezoidal_step_constants,
)
from .saturation import (
    SaturationValidation,
    default_initial_errors,
    simulate_saturated_closed_loop,
    simulate_saturated_closed_loop_batch,
    validate_saturated_closed_loop,
)

__all__ = [
    "RdobDesign",
    "SaturationValidation",
    "controllability_rank",
    "default_initial_errors",
    "design_rdob",
    "discretize_state_space",
    "initialize_rdob",
    "rdob_derivative",
    "reconstruct_disturbance",
    "simulate_saturated_closed_loop",
    "simulate_saturated_closed_loop_batch",
    "solve_discrete_lqr",
    "step_rdob",
    "trapezoidal_step_constants",
    "validate_saturated_closed_loop",
]
