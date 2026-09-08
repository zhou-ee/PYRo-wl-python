"""Full-state discrete LESO for matched, input-equivalent disturbances.

The firmware uses the split observer

``x_hat+ = G x_hat + H (u + d_hat) + L_x (y - x_hat)``
``d_hat+ = d_hat + L_d (y - x_hat)``.

All quantities stay in their physical SI coordinates.  The observer design is
an analytic multi-input pole placement for a full-state measurement; it avoids
the non-unique, symmetry-breaking result that a generic MIMO ``place`` call can
return at mirrored left/right operating points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import config


# Conservative deployment tuning for the 1 kHz firmware loop.  The slower
# state pole rejects encoder/IMU high-frequency noise, while the disturbance
# pole keeps matched-disturbance compensation focused on low-frequency loads.
DEFAULT_STATE_POLE = 0.80
DEFAULT_DISTURBANCE_POLE = 0.992


@dataclass(frozen=True)
class LesoCorrection:
    """Observer correction matrices and their pole-placement contract."""

    state_gain: np.ndarray
    disturbance_gain: np.ndarray
    state_pole: float
    disturbance_pole: float
    unmatched_state_pole: float
    disturbance_coupling: float


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


def build_augmented_model(
    state_matrix: np.ndarray,
    input_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``Ae``, ``Be`` and ``Ce`` for offline checks.

    The six expanded states are constant input-equivalent disturbances.
    Runtime code uses the split form and does not materialize these matrices.
    """

    state_matrix, input_matrix = _validate_discrete_model(
        state_matrix, input_matrix
    )
    augmented_state = np.block(
        [
            [state_matrix, input_matrix],
            [
                np.zeros((config.N_U, config.N_X), dtype=float),
                np.eye(config.N_U, dtype=float),
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
    state_pole: float = DEFAULT_STATE_POLE,
    disturbance_pole: float = DEFAULT_DISTURBANCE_POLE,
    pinv_rcond: float = 1.0e-12,
) -> LesoCorrection:
    """Design a deterministic, mirror-equivariant LESO correction.

    Let ``H+`` be the Moore-Penrose inverse of the full-column-rank ``H``.
    With

    ``alpha = state_pole + disturbance_pole - 1`` and
    ``gamma = (1-state_pole) (1-disturbance_pole)``, choose

    ``L_x = G - alpha I`` and ``L_d = gamma H+``.

    Every matched disturbance channel then has the requested two observer
    poles.  The remaining ``N_X-N_U`` state-error directions have pole
    ``alpha``.  For the deployment defaults these poles are 0.8, 0.992 and
    0.792 respectively.  The construction commutes with orthogonal signed
    left/right permutations, unlike a non-unique generic MIMO pole placement.
    """

    state_matrix, input_matrix = _validate_discrete_model(
        state_matrix, input_matrix
    )
    if not (0.0 < state_pole < 1.0):
        raise ValueError("state_pole must lie strictly inside (0, 1)")
    if not (0.0 < disturbance_pole < 1.0):
        raise ValueError("disturbance_pole must lie strictly inside (0, 1)")
    if pinv_rcond <= 0.0:
        raise ValueError("pinv_rcond must be positive")
    if np.linalg.matrix_rank(input_matrix) != config.N_U:
        raise ValueError("input_matrix must have full disturbance-channel rank")

    unmatched_pole = state_pole + disturbance_pole - 1.0
    if not (-1.0 < unmatched_pole < 1.0):
        raise ValueError("the implied unmatched state-error pole is unstable")
    coupling = (1.0 - state_pole) * (1.0 - disturbance_pole)
    input_inverse = np.linalg.pinv(input_matrix, rcond=pinv_rcond)
    state_gain = state_matrix - unmatched_pole * np.eye(config.N_X)
    disturbance_gain = coupling * input_inverse

    return LesoCorrection(
        state_gain=np.asarray(state_gain, dtype=float),
        disturbance_gain=np.asarray(disturbance_gain, dtype=float),
        state_pole=float(state_pole),
        disturbance_pole=float(disturbance_pole),
        unmatched_state_pole=float(unmatched_pole),
        disturbance_coupling=float(coupling),
    )


def observer_error_matrix(
    state_matrix: np.ndarray,
    input_matrix: np.ndarray,
    state_gain: np.ndarray,
    disturbance_gain: np.ndarray,
) -> np.ndarray:
    """Return the homogeneous augmented estimation-error transition matrix."""

    state_matrix, input_matrix = _validate_discrete_model(
        state_matrix, input_matrix
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
            [-disturbance_gain, np.eye(config.N_U)],
        ]
    )


def observer_error_spectral_radius(
    state_matrix: np.ndarray,
    input_matrix: np.ndarray,
    state_gain: np.ndarray,
    disturbance_gain: np.ndarray,
) -> float:
    """Return ``rho(Ae - Le Ce)`` for one frozen scheduling point."""

    poles = np.linalg.eigvals(
        observer_error_matrix(
            state_matrix,
            input_matrix,
            state_gain,
            disturbance_gain,
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
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the same split LESO recurrence used by the firmware."""

    state_matrix, input_matrix = _validate_discrete_model(
        state_matrix, input_matrix
    )
    estimated_state = np.asarray(estimated_state, dtype=float).reshape(-1)
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
    next_disturbance = estimated_disturbance + np.asarray(
        disturbance_gain, dtype=float
    ) @ innovation
    return next_state, next_disturbance


__all__ = [
    "DEFAULT_DISTURBANCE_POLE",
    "DEFAULT_STATE_POLE",
    "LesoCorrection",
    "build_augmented_model",
    "design_leso_correction",
    "observer_error_matrix",
    "observer_error_spectral_radius",
    "step_leso_split",
]
