"""Reusable control primitives."""

from .lqr import controllability_rank, discretize_state_space, solve_discrete_lqr
from .saturation import (
    SaturationValidation,
    default_initial_errors,
    simulate_saturated_closed_loop,
    simulate_saturated_closed_loop_batch,
    validate_saturated_closed_loop,
)

__all__ = [
    "SaturationValidation",
    "controllability_rank",
    "default_initial_errors",
    "discretize_state_space",
    "simulate_saturated_closed_loop",
    "simulate_saturated_closed_loop_batch",
    "solve_discrete_lqr",
    "validate_saturated_closed_loop",
]
