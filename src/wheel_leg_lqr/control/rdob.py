"""Input-equivalent reduced-order disturbance observer.

All vectors use physical SI coordinates. Position and velocity are seven-vectors
in ``config.STATE_NAMES[::2]`` order, NOT the interleaved 14-state layout.
The design is frozen at a static trim; inputs are actual applied *total* inputs
minus that trim's total equilibrium input. No acceleration measurement is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class RdobUpdateCoefficients(Protocol):
    """Runtime-only contract shared by exact and scheduled coefficients."""

    rates: np.ndarray
    gamma: np.ndarray
    position_gain: np.ndarray
    velocity_gain: np.ndarray
    sample_time: float
    step_poles: np.ndarray
    step_position_gain: np.ndarray
    step_velocity_gain: np.ndarray
    step_input_gain: np.ndarray


def trapezoidal_step_constants(
    rates_rad_s: np.ndarray | float,
    sample_time: float,
    *,
    dtype=np.float64,
    channel_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the six-channel constants for the trapezoidal ``z`` update.

    The returned arrays are, in order: auxiliary-state poles, endpoint
    weights, held-input gains, and the ``Gamma`` velocity scales.  Keeping all
    four quantities as six-vectors preserves the per-input-channel bandwidth
    contract even when the configured bandwidths currently happen to match.
    For zero mechanical damping, ``step_velocity_gain`` is exactly
    ``velocity_scale[:, None] * gamma``.
    """
    scalar_type = np.dtype(dtype).type
    rates = np.asarray(rates_rad_s, dtype=dtype)
    if rates.ndim == 0:
        if channel_count is None or channel_count < 1:
            raise ValueError('scalar rates require a positive channel_count')
        rates = np.full(channel_count, rates, dtype=dtype)
    if rates.ndim != 1 or len(rates) < 1:
        raise ValueError('rates_rad_s must be a nonempty vector')
    N_U = len(rates)
    if rates.shape != (N_U,) or not np.all(np.isfinite(rates)) or np.any(rates <= 0):
        raise ValueError(f"rates_rad_s must contain {N_U} positive finite values")
    if not np.isfinite(sample_time) or sample_time <= 0:
        raise ValueError("sample_time must be positive and finite")
    step = scalar_type(sample_time)
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"sample_time is not representable as positive finite {np.dtype(dtype).name}")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        half_rate_step = (scalar_type(0.5) * step * rates).astype(dtype)
        denominator = (scalar_type(1.0) + half_rate_step).astype(dtype)
        step_z_poles = ((scalar_type(1.0) - half_rate_step) / denominator).astype(dtype)
        endpoint_weights = (scalar_type(0.5) * step / denominator).astype(dtype)
        step_input_gains = (-step * rates / denominator).astype(dtype)
        velocity_scales = (-endpoint_weights * rates).astype(dtype)
    values = (step_z_poles, endpoint_weights, step_input_gains, velocity_scales)
    if any(not np.all(np.isfinite(value)) for value in values):
        raise ValueError("RDOB step constants overflow; reduce the rates or sample time")
    if np.any(np.abs(step_z_poles) >= 1):
        raise ValueError("sample_time and rates produce numerically marginal poles")
    return values


def _array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


@dataclass(frozen=True)
class RdobDesign:
    """Continuous design and trapezoidal sampled implementation coefficients.

    ``z_dot = -rates*z + position_gain*q + velocity_gain*v - rates*u``.
    ``d_hat = z + gamma*v``. ``rates`` are rad/s (not Hz).

    Discrete update uses both endpoints of measured q/v and the applied input
    held over their interval. The six ``z`` entries are the only observer
    dynamic states; storing previous measurements is an additional buffer.
    """

    mass: np.ndarray
    damping: np.ndarray
    stiffness: np.ndarray
    input_matrix: np.ndarray
    input_left_inverse: np.ndarray
    rates: np.ndarray
    gamma: np.ndarray
    position_gain: np.ndarray
    velocity_gain: np.ndarray
    sample_time: float
    step_poles: np.ndarray
    step_position_gain: np.ndarray
    step_velocity_gain: np.ndarray
    step_input_gain: np.ndarray
    input_condition_number: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, np.ndarray):
                value = value.copy()
                value.setflags(write=False)
                object.__setattr__(self, name, value)


def design_rdob(
    mass: np.ndarray,
    stiffness: np.ndarray,
    input_matrix: np.ndarray,
    *,
    rates_rad_s: np.ndarray | float,
    sample_time: float,
    damping: np.ndarray | None = None,
    pinv_rcond: float = 1e-12,
) -> RdobDesign:
    """Design for M*qdd + D*qd + K*q = S*(u+d), S has full column rank.

    Use an SVD left inverse rather than normal-equation inversion. A matrix
    that loses a column at the requested SVD tolerance is rejected rather than
    silently producing a truncated inverse that mixes disturbance channels.
    The Euclidean projection refers to the supplied generalized-force
    coordinates; rescaling those coordinates changes the unmatched projection.
    """
    mass = np.asarray(mass, dtype=float)
    input_matrix = np.asarray(input_matrix, dtype=float)
    if mass.ndim != 2 or mass.shape[0] != mass.shape[1] or not len(mass):
        raise ValueError('mass must be a nonempty square matrix')
    N_Q = mass.shape[0]
    if input_matrix.ndim != 2 or input_matrix.shape[0] != N_Q or not 0 < input_matrix.shape[1] <= N_Q:
        raise ValueError('input_matrix must have shape (coordinates, channels <= coordinates)')
    N_U = input_matrix.shape[1]
    mass = _array(mass, (N_Q, N_Q), "mass")
    stiffness = _array(stiffness, (N_Q, N_Q), "stiffness")
    input_matrix = _array(input_matrix, (N_Q, N_U), "input_matrix")
    damping = _array(
        np.zeros((N_Q, N_Q)) if damping is None else damping,
        (N_Q, N_Q), "damping",
    )
    if not np.allclose(mass, mass.T, rtol=1e-10, atol=1e-12):
        raise ValueError("mass must be symmetric positive definite")
    try:
        np.linalg.cholesky(mass)
    except np.linalg.LinAlgError as error:
        raise ValueError("mass must be symmetric positive definite") from error
    rates = np.asarray(rates_rad_s, dtype=float)
    if rates.ndim == 0:
        rates = np.full(N_U, float(rates))
    rates = _array(rates, (N_U,), "rates_rad_s")
    if np.any(rates <= 0):
        raise ValueError("rates_rad_s must be positive")
    if not np.isfinite(sample_time) or sample_time <= 0:
        raise ValueError("sample_time must be positive and finite")
    if not np.isfinite(pinv_rcond) or not 0 < pinv_rcond < 1:
        raise ValueError("pinv_rcond must lie strictly inside (0, 1)")
    singular_values = np.linalg.svd(input_matrix, compute_uv=False)
    if singular_values[-1] <= pinv_rcond * singular_values[0]:
        raise ValueError("input_matrix must have full column rank at pinv_rcond")
    left_inverse = np.linalg.pinv(input_matrix, rcond=pinv_rcond)
    gamma = rates[:, None] * (left_inverse @ mass)
    position_gain = rates[:, None] * (left_inverse @ stiffness)
    velocity_gain = rates[:, None] * (left_inverse @ damping - gamma)

    # Trapezoidal integration of the continuous z dynamics. This is not an
    # exact-ZOH discretization of the robot/observer coupled system.
    step_poles, endpoint_weight, step_input_gain, _ = (
        trapezoidal_step_constants(rates, sample_time)
    )
    step_position_gain = endpoint_weight[:, None] * position_gain
    step_velocity_gain = endpoint_weight[:, None] * velocity_gain
    values = (gamma, position_gain, velocity_gain, step_poles,
              step_position_gain, step_velocity_gain, step_input_gain)
    if any(not np.all(np.isfinite(value)) for value in values):
        raise ValueError("RDOB coefficients overflow; rescale the model or rates")
    if np.any(np.abs(step_poles) >= 1):
        raise ValueError("sample_time and rates produce numerically marginal poles")
    return RdobDesign(
        mass, damping, stiffness, input_matrix, left_inverse, rates,
        gamma, position_gain, velocity_gain, float(sample_time), step_poles,
        step_position_gain, step_velocity_gain, step_input_gain,
        float(singular_values[0] / singular_values[-1]),
    )


def reconstruct_disturbance(
    design: RdobUpdateCoefficients, z: np.ndarray, delta_velocity: np.ndarray,
) -> np.ndarray:
    """Reconstruct six disturbances at the SAME timestamp as z and velocity."""
    N_U, N_Q = design.gamma.shape
    return (_array(z, (N_U,), "z")
            + design.gamma @ _array(delta_velocity, (N_Q,), "delta_velocity"))


def initialize_rdob(
    design: RdobUpdateCoefficients, delta_velocity: np.ndarray,
    initial_disturbance: np.ndarray | None = None,
) -> np.ndarray:
    """Initialize/rebase z so d_hat equals a requested estimate (default zero).

    For coefficient switching, reconstruct with the OLD design, then initialize
    with the NEW design and the same physical measurement/estimate. This avoids
    an algebraic jump; it is not a proof of scheduled observer stability.
    """
    N_U, N_Q = design.gamma.shape
    disturbance = _array(
        np.zeros(N_U) if initial_disturbance is None else initial_disturbance,
        (N_U,), "initial_disturbance",
    )
    return disturbance - design.gamma @ _array(
        delta_velocity, (N_Q,), "delta_velocity"
    )


def rdob_derivative(
    design: RdobUpdateCoefficients, z: np.ndarray, delta_position: np.ndarray,
    delta_velocity: np.ndarray, applied_input_delta: np.ndarray,
) -> np.ndarray:
    """Continuous acceleration-free derivative, suitable for offline ODEs."""
    N_U, N_Q = design.gamma.shape
    return (
        -design.rates * _array(z, (N_U,), "z")
        + design.position_gain @ _array(delta_position, (N_Q,), "delta_position")
        + design.velocity_gain @ _array(delta_velocity, (N_Q,), "delta_velocity")
        - design.rates * _array(applied_input_delta, (N_U,), "applied_input_delta")
    )


def step_rdob(
    design: RdobUpdateCoefficients, z: np.ndarray,
    previous_delta_position: np.ndarray, previous_delta_velocity: np.ndarray,
    applied_input_delta: np.ndarray,
    next_delta_position: np.ndarray, next_delta_velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance from k to k+1, returning (z[k+1], d_hat[k+1]).

    Call when the k+1 measurement arrives, using the input actually applied
    during [k,k+1). That input includes compensation and actuator limitations.
    All measurements are relative to the SAME frozen trim. No observer leak,
    compensation gain, or saturation is implicitly inserted here.
    """
    N_U, N_Q = design.gamma.shape
    q0 = _array(previous_delta_position, (N_Q,), "previous_delta_position")
    v0 = _array(previous_delta_velocity, (N_Q,), "previous_delta_velocity")
    q1 = _array(next_delta_position, (N_Q,), "next_delta_position")
    v1 = _array(next_delta_velocity, (N_Q,), "next_delta_velocity")
    next_z = (
        design.step_poles * _array(z, (N_U,), "z")
        + design.step_position_gain @ (q0 + q1)
        + design.step_velocity_gain @ (v0 + v1)
        + design.step_input_gain
        * _array(applied_input_delta, (N_U,), "applied_input_delta")
    )
    return next_z, reconstruct_disturbance(design, next_z, v1)


__all__ = [
    "RdobDesign", "RdobUpdateCoefficients", "design_rdob", "initialize_rdob", "rdob_derivative",
    "reconstruct_disturbance", "step_rdob", "trapezoidal_step_constants",
]
