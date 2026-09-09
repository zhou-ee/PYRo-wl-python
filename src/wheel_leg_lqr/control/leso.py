"""Full-state discrete LESO for matched, input-equivalent disturbances.

The firmware uses the split observer

``x_hat+ = G x_hat + H (u + d_hat) + L_x (y - x_hat)``
``d_hat+ = a_d d_hat + L_d (y - x_hat)``.

All quantities stay in their physical SI coordinates.  The observer design is
an analytic multi-input pole placement for a full-state measurement; it avoids
the non-unique, symmetry-breaking result that a generic MIMO ``place`` call can
return at mirrored left/right operating points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import config


DEFAULT_WHEEL_STATE_BANDWIDTH_HZ = 100.0
DEFAULT_HIP_STATE_BANDWIDTH_HZ = 80.0
# Backward-compatible aggregate name; new scheduling code separates Tw and Tp.
DEFAULT_TORQUE_STATE_BANDWIDTH_HZ = DEFAULT_HIP_STATE_BANDWIDTH_HZ
DEFAULT_TORQUE_DISTURBANCE_BANDWIDTH_HZ = 2.4
DEFAULT_FL_STATE_BANDWIDTH_HZ = 25.0
DEFAULT_FL_DISTURBANCE_BANDWIDTH_HZ = 1.2
DEFAULT_UNMATCHED_STATE_BANDWIDTH_HZ = 25.0
DEFAULT_DISTURBANCE_LEAK_BANDWIDTH_HZ = 0.2


def discrete_pole_from_bandwidth(
    bandwidth_hz: float,
    sample_time: float = config.SAMPLE_TIME,
) -> float:
    """Map a continuous first-order bandwidth to a discrete pole."""

    if not np.isfinite(bandwidth_hz) or bandwidth_hz <= 0.0:
        raise ValueError("bandwidth_hz must be positive and finite")
    if not np.isfinite(sample_time) or sample_time <= 0.0:
        raise ValueError("sample_time must be positive and finite")
    return float(np.exp(-2.0 * np.pi * bandwidth_hz * sample_time))


_DEFAULT_WHEEL_STATE_POLE = discrete_pole_from_bandwidth(
    DEFAULT_WHEEL_STATE_BANDWIDTH_HZ
)
_DEFAULT_HIP_STATE_POLE = discrete_pole_from_bandwidth(
    DEFAULT_HIP_STATE_BANDWIDTH_HZ
)
_DEFAULT_FL_STATE_POLE = discrete_pole_from_bandwidth(
    DEFAULT_FL_STATE_BANDWIDTH_HZ
)
_DEFAULT_TORQUE_DISTURBANCE_POLE = discrete_pole_from_bandwidth(
    DEFAULT_TORQUE_DISTURBANCE_BANDWIDTH_HZ
)
_DEFAULT_FL_DISTURBANCE_POLE = discrete_pole_from_bandwidth(
    DEFAULT_FL_DISTURBANCE_BANDWIDTH_HZ
)
DEFAULT_STATE_POLES = np.asarray(
    [_DEFAULT_WHEEL_STATE_POLE] * 2
    + [_DEFAULT_HIP_STATE_POLE] * 2
    + [_DEFAULT_FL_STATE_POLE] * 2,
    dtype=float,
)
DEFAULT_DISTURBANCE_POLES = np.asarray(
    [_DEFAULT_TORQUE_DISTURBANCE_POLE] * 4
    + [_DEFAULT_FL_DISTURBANCE_POLE] * 2,
    dtype=float,
)
DEFAULT_UNMATCHED_STATE_POLE = discrete_pole_from_bandwidth(
    DEFAULT_UNMATCHED_STATE_BANDWIDTH_HZ
)
DEFAULT_DISTURBANCE_RETENTION_POLE = discrete_pole_from_bandwidth(
    DEFAULT_DISTURBANCE_LEAK_BANDWIDTH_HZ
)
DEFAULT_STATE_POLES.setflags(write=False)
DEFAULT_DISTURBANCE_POLES.setflags(write=False)


@dataclass(frozen=True)
class LesoCorrection:
    """Observer correction matrices and their pole-placement contract."""

    state_gain: np.ndarray
    disturbance_gain: np.ndarray
    state_poles: np.ndarray
    disturbance_poles: np.ndarray
    disturbance_retention_pole: float
    unmatched_state_pole: float
    matched_state_coefficients: np.ndarray
    disturbance_couplings: np.ndarray
    matched_correction_scales: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "state_poles",
            "disturbance_poles",
            "matched_state_coefficients",
            "disturbance_couplings",
            "matched_correction_scales",
        ):
            values = np.asarray(getattr(self, name), dtype=float).copy()
            values.setflags(write=False)
            object.__setattr__(self, name, values)


def _validate_discrete_model(
    state_matrix: np.ndarray,
    input_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    state_matrix = np.asarray(state_matrix, dtype=float)
    input_matrix = np.asarray(input_matrix, dtype=float)
    if state_matrix.shape != (config.N_X, config.N_X):
        raise ValueError(
            f"state_matrix must have shape ({config.N_X}, {config.N_X})"
        )
    if input_matrix.shape != (config.N_X, config.N_U):
        raise ValueError(
            f"input_matrix must have shape ({config.N_X}, {config.N_U})"
        )
    if not np.all(np.isfinite(state_matrix)) or not np.all(
        np.isfinite(input_matrix)
    ):
        raise ValueError("LESO model matrices must contain only finite values")
    return state_matrix, input_matrix


def _validate_retention_pole(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(
            "disturbance_retention_pole must lie strictly inside (0, 1)"
        )
    return value


def build_augmented_model(
    state_matrix: np.ndarray,
    input_matrix: np.ndarray,
    disturbance_retention_pole: float = DEFAULT_DISTURBANCE_RETENTION_POLE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``Ae``, ``Be`` and ``Ce`` for offline checks.

    The six expanded states are slowly leaking input-equivalent disturbances.
    Runtime code uses the split form and does not materialize these matrices.
    """

    state_matrix, input_matrix = _validate_discrete_model(
        state_matrix, input_matrix
    )
    disturbance_retention_pole = _validate_retention_pole(
        disturbance_retention_pole
    )
    augmented_state = np.block(
        [
            [state_matrix, input_matrix],
            [
                np.zeros((config.N_U, config.N_X), dtype=float),
                disturbance_retention_pole
                * np.eye(config.N_U, dtype=float),
            ],
        ]
    )
    augmented_input = np.vstack(
        [input_matrix, np.zeros((config.N_U, config.N_U), dtype=float)]
    )
    output = np.hstack(
        [
            np.eye(config.N_X, dtype=float),
            np.zeros((config.N_X, config.N_U), dtype=float),
        ]
    )
    return augmented_state, augmented_input, output


def design_leso_correction(
    state_matrix: np.ndarray,
    input_matrix: np.ndarray,
    *,
    state_poles: np.ndarray | float | None = None,
    disturbance_poles: np.ndarray | float | None = None,
    disturbance_retention_pole: float = DEFAULT_DISTURBANCE_RETENTION_POLE,
    unmatched_state_pole: float = DEFAULT_UNMATCHED_STATE_POLE,
    pinv_rcond: float = 1.0e-12,
) -> LesoCorrection:
    """Design a deterministic, mirror-equivariant LESO correction.

    Let ``H+`` be the Moore-Penrose inverse of the full-column-rank ``H`` and
    let ``a_d`` be the disturbance-retention pole.  For each input channel
    ``i``, choose

    ``alpha_i = state_pole_i + disturbance_pole_i - a_d`` and
    ``gamma_i = state_pole_i disturbance_pole_i - a_d alpha_i``.

    Let ``Alpha=diag(alpha_i)``, ``Gamma=diag(gamma_i)`` and let
    ``p_u`` be the pole assigned to the state-error complement of ``range(H)``.
    The correction is

    ``L_x = G - p_u I - H (Alpha-p_u I) H+`` and
    ``L_d = Gamma H+``.

    Every matched disturbance channel then has its requested state and
    disturbance poles.  The remaining ``N_X-N_U`` state-error directions have
    pole ``p_u``.  Equal parameters for each left/right pair preserve exchange
    symmetry.
    """

    state_matrix, input_matrix = _validate_discrete_model(
        state_matrix, input_matrix
    )
    def channel_poles(
        values: np.ndarray | float | None,
        defaults: np.ndarray,
        name: str,
    ) -> np.ndarray:
        if values is None:
            result = defaults.copy()
        else:
            result = np.asarray(values, dtype=float)
            if result.ndim == 0:
                result = np.full(config.N_U, float(result), dtype=float)
            else:
                result = result.reshape(-1).copy()
        if result.shape != (config.N_U,):
            raise ValueError(f"{name} must contain {config.N_U} values")
        if not np.all(np.isfinite(result)) or np.any(result <= 0.0) or np.any(
            result >= 1.0
        ):
            raise ValueError(f"{name} must lie strictly inside (0, 1)")
        return result

    state_poles = channel_poles(
        state_poles, DEFAULT_STATE_POLES, "state_poles"
    )
    disturbance_poles = channel_poles(
        disturbance_poles,
        DEFAULT_DISTURBANCE_POLES,
        "disturbance_poles",
    )
    disturbance_retention_pole = _validate_retention_pole(
        disturbance_retention_pole
    )
    if not np.isfinite(unmatched_state_pole) or not (
        0.0 < unmatched_state_pole < 1.0
    ):
        raise ValueError("unmatched_state_pole must lie strictly inside (0, 1)")
    if pinv_rcond <= 0.0:
        raise ValueError("pinv_rcond must be positive")
    if np.linalg.matrix_rank(input_matrix) != config.N_U:
        raise ValueError("input_matrix must have full disturbance-channel rank")

    matched_coefficients = (
        state_poles + disturbance_poles - disturbance_retention_pole
    )
    couplings = (
        state_poles * disturbance_poles
        - disturbance_retention_pole * matched_coefficients
    )
    if np.any(couplings <= 0.0):
        raise ValueError(
            "observer poles and disturbance retention produce a "
            "non-positive disturbance coupling"
        )
    input_inverse = np.linalg.pinv(input_matrix, rcond=pinv_rcond)
    matched_offset = matched_coefficients - unmatched_state_pole
    state_error_matrix = (
        unmatched_state_pole * np.eye(config.N_X)
        + input_matrix @ np.diag(matched_offset) @ input_inverse
    )
    state_gain = state_matrix - state_error_matrix
    disturbance_gain = np.diag(couplings) @ input_inverse
    correction_scales = matched_offset / couplings

    return LesoCorrection(
        state_gain=np.asarray(state_gain, dtype=float),
        disturbance_gain=np.asarray(disturbance_gain, dtype=float),
        state_poles=state_poles,
        disturbance_poles=disturbance_poles,
        disturbance_retention_pole=float(disturbance_retention_pole),
        unmatched_state_pole=float(unmatched_state_pole),
        matched_state_coefficients=matched_coefficients,
        disturbance_couplings=couplings,
        matched_correction_scales=correction_scales,
    )


def observer_error_matrix(
    state_matrix: np.ndarray,
    input_matrix: np.ndarray,
    state_gain: np.ndarray,
    disturbance_gain: np.ndarray,
    disturbance_retention_pole: float = DEFAULT_DISTURBANCE_RETENTION_POLE,
) -> np.ndarray:
    """Return the homogeneous augmented estimation-error transition matrix."""

    state_matrix, input_matrix = _validate_discrete_model(
        state_matrix, input_matrix
    )
    disturbance_retention_pole = _validate_retention_pole(
        disturbance_retention_pole
    )
    state_gain = np.asarray(state_gain, dtype=float)
    disturbance_gain = np.asarray(disturbance_gain, dtype=float)
    if state_gain.shape != (config.N_X, config.N_X):
        raise ValueError("state_gain has the wrong shape")
    if disturbance_gain.shape != (config.N_U, config.N_X):
        raise ValueError("disturbance_gain has the wrong shape")
    return np.block(
        [
            [state_matrix - state_gain, input_matrix],
            [
                -disturbance_gain,
                disturbance_retention_pole * np.eye(config.N_U),
            ],
        ]
    )


def observer_error_spectral_radius(
    state_matrix: np.ndarray,
    input_matrix: np.ndarray,
    state_gain: np.ndarray,
    disturbance_gain: np.ndarray,
    disturbance_retention_pole: float = DEFAULT_DISTURBANCE_RETENTION_POLE,
) -> float:
    """Return ``rho(Ae - Le Ce)`` for one frozen scheduling point."""

    poles = np.linalg.eigvals(
        observer_error_matrix(
            state_matrix,
            input_matrix,
            state_gain,
            disturbance_gain,
            disturbance_retention_pole,
        )
    )
    return float(np.max(np.abs(poles)))


def step_leso_split(
    estimated_state: np.ndarray,
    estimated_disturbance: np.ndarray,
    measured_state: np.ndarray,
    applied_input_delta: np.ndarray,
    state_matrix: np.ndarray,
    input_matrix: np.ndarray,
    state_gain: np.ndarray,
    disturbance_gain: np.ndarray,
    disturbance_retention_pole: float = DEFAULT_DISTURBANCE_RETENTION_POLE,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the same split LESO recurrence used by the firmware."""

    state_matrix, input_matrix = _validate_discrete_model(
        state_matrix, input_matrix
    )
    estimated_state = np.asarray(estimated_state, dtype=float).reshape(-1)
    disturbance_retention_pole = _validate_retention_pole(
        disturbance_retention_pole
    )
    estimated_disturbance = np.asarray(
        estimated_disturbance, dtype=float
    ).reshape(-1)
    measured_state = np.asarray(measured_state, dtype=float).reshape(-1)
    applied_input_delta = np.asarray(applied_input_delta, dtype=float).reshape(-1)
    if estimated_state.size != config.N_X or measured_state.size != config.N_X:
        raise ValueError("state vectors must contain N_X values")
    if (
        estimated_disturbance.size != config.N_U
        or applied_input_delta.size != config.N_U
    ):
        raise ValueError("input/disturbance vectors must contain N_U values")

    innovation = measured_state - estimated_state
    next_state = (
        state_matrix @ estimated_state
        + input_matrix @ (applied_input_delta + estimated_disturbance)
        + np.asarray(state_gain, dtype=float) @ innovation
    )
    next_disturbance = (
        disturbance_retention_pole * estimated_disturbance
        + np.asarray(disturbance_gain, dtype=float) @ innovation
    )
    return next_state, next_disturbance


def bounded_disturbance_update(
    estimated_disturbance: np.ndarray,
    raw_increment: np.ndarray,
    disturbance_limits: np.ndarray,
    maximum_increments: np.ndarray,
    *,
    disturbance_retention_pole: float = DEFAULT_DISTURBANCE_RETENTION_POLE,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the firmware's slew limit, leakage and anti-windup clamp.

    The returned effective increment is the innovation actually accepted after
    both clamps.  The state correction must use this same value so that a shock
    rejected by the disturbance integrator is not reintroduced through the
    fused ``L_x`` path.
    """

    estimated = np.asarray(estimated_disturbance, dtype=float).reshape(-1)
    raw = np.asarray(raw_increment, dtype=float).reshape(-1)
    limits = np.asarray(disturbance_limits, dtype=float).reshape(-1)
    increments = np.asarray(maximum_increments, dtype=float).reshape(-1)
    if not all(
        values.shape == (config.N_U,)
        for values in (estimated, raw, limits, increments)
    ):
        raise ValueError("disturbance vectors must contain N_U values")
    if (
        not np.all(np.isfinite(estimated))
        or not np.all(np.isfinite(raw))
        or not np.all(np.isfinite(limits))
        or not np.all(np.isfinite(increments))
        or np.any(limits <= 0.0)
        or np.any(increments <= 0.0)
    ):
        raise ValueError("disturbance limits and values must be finite")
    retained = _validate_retention_pole(disturbance_retention_pole) * estimated
    slew_limited = np.clip(raw, -increments, increments)
    next_disturbance = np.clip(retained + slew_limited, -limits, limits)
    return next_disturbance, next_disturbance - retained


__all__ = [
    "DEFAULT_DISTURBANCE_LEAK_BANDWIDTH_HZ",
    "DEFAULT_DISTURBANCE_RETENTION_POLE",
    "DEFAULT_DISTURBANCE_POLES",
    "DEFAULT_FL_DISTURBANCE_BANDWIDTH_HZ",
    "DEFAULT_FL_STATE_BANDWIDTH_HZ",
    "DEFAULT_HIP_STATE_BANDWIDTH_HZ",
    "DEFAULT_STATE_POLES",
    "DEFAULT_TORQUE_DISTURBANCE_BANDWIDTH_HZ",
    "DEFAULT_TORQUE_STATE_BANDWIDTH_HZ",
    "DEFAULT_WHEEL_STATE_BANDWIDTH_HZ",
    "DEFAULT_UNMATCHED_STATE_BANDWIDTH_HZ",
    "DEFAULT_UNMATCHED_STATE_POLE",
    "LesoCorrection",
    "bounded_disturbance_update",
    "build_augmented_model",
    "design_leso_correction",
    "discrete_pole_from_bandwidth",
    "observer_error_matrix",
    "observer_error_spectral_radius",
    "step_leso_split",
]
