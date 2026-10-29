"""Independent four-input/ten-state LQR nodes and frozen-length schedules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import DEFAULT_ROBOT, RobotParameters
from ..control.lqr import (
    controllability_rank,
    discretize_state_space,
    solve_discrete_lqr,
)
from ..models.ground10.config import Ground10Settings, default_initial_errors
from ..models.ground10.model import (
    StaticLinearization,
    compute_equilibrium_beta_from_length,
    linearize_static_trim,
)
from ..simulation.saturation import validate_saturated_closed_loop
from .domain import make_leg_length_grid
from .polynomial import fit_tensor_coefficients


@dataclass(frozen=True)
class GainNode:
    l1: float
    l2: float
    linearization: StaticLinearization
    Ad: np.ndarray
    Bd: np.ndarray
    gain: np.ndarray
    closed_loop_poles: np.ndarray
    controllability_rank: int
    parameters: RobotParameters
    settings: Ground10Settings


@dataclass(frozen=True)
class GainScheduleFit:
    grid: np.ndarray
    nodes: tuple[GainNode, ...]
    gain_coefficients: np.ndarray
    input_coefficients: np.ndarray
    beta_coefficients: np.ndarray
    metrics: dict
    parameters: RobotParameters
    settings: Ground10Settings


def input_limits(parameters=DEFAULT_ROBOT):
    upper = np.array(
        [parameters.wheel_torque_limit_nm] * 2 + [parameters.hip_torque_limit_nm] * 2
    )
    return -upper, upper


def compute_gain_node(l1, l2, *, parameters=DEFAULT_ROBOT, settings=None):
    settings = settings or Ground10Settings()
    lin = linearize_static_trim(l1, l2, parameters=parameters)
    Ad, Bd = discretize_state_space(lin.A, lin.B, sample_time=settings.sample_time)
    rank = controllability_rank(Ad, Bd)
    if rank != 10:
        raise ValueError(f"ten-state model is not controllable at ({l1}, {l2})")
    gain, _, poles = solve_discrete_lqr(Ad, Bd, settings.make_q(), settings.make_r())
    if not np.all(np.isfinite(gain)) or np.max(np.abs(poles)) >= 1:
        raise ValueError(f"ten-state LQR is unstable at ({l1}, {l2})")
    return GainNode(
        float(l1), float(l2), lin, Ad, Bd, gain, poles, rank, parameters, settings
    )


def evaluate_polynomial(
    coefficients, l1, l2, *, parameters=DEFAULT_ROBOT, dtype=np.float64
):
    """Clamped nested Horner evaluation; this order is also used by the header."""
    if not np.all(np.isfinite([l1, l2])):
        raise ValueError("L1/L2 must be finite")
    c = np.asarray(coefficients, dtype=dtype)
    center = dtype((parameters.min_leg_length + parameters.max_leg_length) / 2)
    scale = dtype((parameters.max_leg_length - parameters.min_leg_length) / 2)
    x = dtype(np.clip((dtype(l1) - center) / scale, -1, 1))
    y = dtype(np.clip((dtype(l2) - center) / scale, -1, 1))
    result = np.zeros(c.shape[:-2], dtype=dtype)
    for i in range(c.shape[-2] - 1, -1, -1):
        row = np.zeros_like(result)
        for j in range(c.shape[-1] - 1, -1, -1):
            row = row * y + c[..., i, j]
        result = result * x + row
    return result


def compute_gain_schedule(
    *, step=0.02, degree=3, parameters=DEFAULT_ROBOT, settings=None
):
    settings = settings or Ground10Settings()
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    grid = make_leg_length_grid(
        parameters.min_leg_length, parameters.max_leg_length, step
    )
    if len(grid) <= degree:
        raise ValueError("grid needs at least degree+1 points per axis")
    nodes = tuple(
        compute_gain_node(l1, l2, parameters=parameters, settings=settings)
        for l1 in grid
        for l2 in grid
    )
    center = (parameters.min_leg_length + parameters.max_leg_length) / 2
    scale = (parameters.max_leg_length - parameters.min_leg_length) / 2
    x = np.array([(n.l1 - center) / scale for n in nodes])
    y = np.array([(n.l2 - center) / scale for n in nodes])
    raw = np.stack([n.gain for n in nodes])
    coefficients, fitted = fit_tensor_coefficients(x, y, raw, degree, degree)
    inputs, _ = fit_tensor_coefficients(
        x,
        y,
        np.stack([n.linearization.equilibrium_input for n in nodes]),
        degree,
        degree,
    )
    middle = (grid[:-1] + grid[1:]) / 2
    holdouts = tuple(
        compute_gain_node(l1, l2, parameters=parameters, settings=settings)
        for l1 in middle
        for l2 in middle
    )
    radii, errors = [], []
    for node in (*nodes, *holdouts):
        gain = evaluate_polynomial(
            coefficients, node.l1, node.l2, parameters=parameters, dtype=np.float32
        ).astype(float)
        radii.append(float(np.max(np.abs(np.linalg.eigvals(node.Ad - node.Bd @ gain)))))
        errors.append(
            float(np.linalg.norm(gain - node.gain) / np.linalg.norm(node.gain))
        )
    if not np.all(np.isfinite(radii)) or max(radii) >= 1:
        raise ValueError(
            "ten-state float32 fit is unstable on the training/holdout grid"
        )
    lower, upper = input_limits(parameters)
    # Use a spread of up to 25 nodes for the slower five-second clipped-input check.
    indices = np.unique(np.linspace(0, len(grid) - 1, min(5, len(grid)), dtype=int))
    validations = []
    for i in indices:
        for j in indices:
            node = nodes[i * len(grid) + j]
            gain = evaluate_polynomial(
                coefficients, node.l1, node.l2, parameters=parameters, dtype=np.float32
            ).astype(float)
            validation = validate_saturated_closed_loop(
                node.Ad,
                node.Bd,
                gain,
                node.linearization.equilibrium_input,
                cases=default_initial_errors(),
                sample_time=settings.sample_time,
                lower_limits=lower,
                upper_limits=upper,
            )
            if not validation.passed:
                raise ValueError(
                    f"ten-state saturated recovery failed at ({node.l1}, {node.l2}): {validation.worst_case}"
                )
            validations.append(validation)
    beta_lengths = np.linspace(
        parameters.min_leg_length, parameters.max_leg_length, 201
    )
    beta_values = [
        compute_equilibrium_beta_from_length(length, parameters)
        for length in beta_lengths
    ]
    beta_coefficients = np.polynomial.polynomial.polyfit(beta_lengths, beta_values, 3)
    metrics = {
        "training_node_count": len(nodes),
        "holdout_node_count": len(holdouts),
        "max_raw_spectral_radius": max(
            float(np.max(np.abs(n.closed_loop_poles))) for n in nodes
        ),
        "max_float32_spectral_radius": max(radii),
        "max_abs_gain_fit_error": float(np.max(np.abs(fitted - raw))),
        "max_relative_gain_fit_error": max(errors),
        "max_beta_fit_error_rad": float(
            np.max(
                np.abs(
                    np.polynomial.polynomial.polyval(beta_lengths, beta_coefficients)
                    - beta_values
                )
            )
        ),
        "saturation_validation_node_count": len(validations),
        "saturation_passed": True,
        "saturation_max_final_state_abs": max(
            v.max_final_state_abs for v in validations
        ),
    }
    return GainScheduleFit(
        grid,
        nodes,
        coefficients,
        inputs,
        beta_coefficients,
        metrics,
        parameters,
        settings,
    )
