"""Physical-coordinate six-input LESO design and split-form updates.

The plant model is the perturbation model

    dx[k+1] = Ad dx[k] + Bd (du[k] + d[k]),

where ``du`` is the *applied* actuator command after subtracting the static
trim input.  All six actuator channels are expanded as constant, matched
disturbances.  The runtime update exploits the augmented model's block
structure and never constructs a 20-by-20 matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_discrete_are

from ..core import config


N_DISTURBANCE = config.N_U
N_AUGMENTED = config.N_X + N_DISTURBANCE


def _readonly_vector(values: np.ndarray, size: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1).copy()
    if vector.size != size:
        raise ValueError(f"{name} must contain {size} values")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True)
class LesoDesignConfig:
    """Physical-unit covariance choices for the offline steady-state LQE.

    The measurement noise defaults are conservative engineering estimates.
    They must eventually be replaced by variances measured from stationary
    robot logs.  Every covariance is expressed in the corresponding physical
    state or actuator units; no state or input normalization is performed.
    """

    measurement_noise_std: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                1.0e-3,
                2.0e-2,
                2.0e-3,
                2.0e-2,
                1.0e-3,
                2.0e-2,
                1.0e-3,
                2.0e-2,
                5.0e-4,
                1.0e-2,
                1.0e-3,
                2.0e-2,
                1.0e-3,
                2.0e-2,
            ],
            dtype=float,
        )
    )
    state_process_variance: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                1.0e-10,
                9.0e-10,
                2.5e-11,
                2.5e-9,
                6.25e-12,
                2.5e-9,
                6.25e-12,
                2.5e-9,
                1.0e-12,
                1.0e-10,
                2.5e-11,
                2.5e-9,
                2.5e-11,
                2.5e-9,
            ],
            dtype=float,
        )
    )
    disturbance_process_variance: np.ndarray = field(
        default_factory=lambda: 1.0e-6
        * np.maximum(
            np.abs(config.INPUT_LOWER_LIMITS),
            np.abs(config.INPUT_UPPER_LIMITS),
        )
        ** 2
    )

    def __post_init__(self) -> None:
        vectors = (
            ("measurement_noise_std", self.measurement_noise_std, config.N_X),
            ("state_process_variance", self.state_process_variance, config.N_X),
            (
                "disturbance_process_variance",
                self.disturbance_process_variance,
                N_DISTURBANCE,
            ),
        )
        for name, values, size in vectors:
            object.__setattr__(self, name, _readonly_vector(values, size, name))
        if np.any(self.measurement_noise_std <= 0.0):
            raise ValueError("measurement noise standard deviations must be positive")
        if np.any(self.state_process_variance < 0.0) or np.any(
            self.disturbance_process_variance <= 0.0
        ):
            raise ValueError("LESO process variances must be non-negative and disturbances positive")


DEFAULT_LESO_DESIGN = LesoDesignConfig()


@dataclass(frozen=True)
class LesoEstimate:
    """Physical-state and input-equivalent disturbance estimates."""

    state: np.ndarray
    disturbance: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "state",
            _readonly_vector(self.state, config.N_X, "estimated state"),
        )
        object.__setattr__(
            self,
            "disturbance",
            _readonly_vector(
                self.disturbance,
                N_DISTURBANCE,
                "estimated disturbance",
            ),
        )

    def physical_state(self) -> np.ndarray:
        """Return the estimate in its native physical units."""
        return self.state.copy()

    def physical_disturbance(self) -> np.ndarray:
        """Return input-equivalent disturbances in N m or N."""
        return self.disturbance.copy()


@dataclass(frozen=True)
class LesoStepResult:
    estimate: LesoEstimate
    innovation: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "innovation",
            _readonly_vector(self.innovation, config.N_X, "LESO innovation"),
        )


def _validate_discrete_model(Ad: np.ndarray, Bd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Ad = np.asarray(Ad)
    Bd = np.asarray(Bd)
    if not np.issubdtype(Ad.dtype, np.floating):
        Ad = Ad.astype(float)
    if not np.issubdtype(Bd.dtype, np.floating):
        Bd = Bd.astype(float)
    if Ad.shape != (config.N_X, config.N_X):
        raise ValueError(f"Ad must have shape ({config.N_X}, {config.N_X})")
    if Bd.shape != (config.N_X, config.N_U):
        raise ValueError(f"Bd must have shape ({config.N_X}, {config.N_U})")
    if not np.all(np.isfinite(Ad)) or not np.all(np.isfinite(Bd)):
        raise ValueError("Ad and Bd must contain only finite values")
    return Ad, Bd


def build_augmented_model(
    Ad: np.ndarray,
    Bd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build full matrices for offline analysis; runtime uses split updates."""
    Ad, Bd = _validate_discrete_model(
        Ad,
        Bd,
    )
    dtype = np.result_type(Ad.dtype, Bd.dtype)
    augmented_a = np.block(
        [
            [Ad, Bd],
            [
                np.zeros((N_DISTURBANCE, config.N_X), dtype=dtype),
                np.eye(N_DISTURBANCE, dtype=dtype),
            ],
        ]
    )
    augmented_b = np.vstack(
        [Bd, np.zeros((N_DISTURBANCE, config.N_U), dtype=dtype)]
    )
    output = np.hstack(
        [
            np.eye(config.N_X, dtype=dtype),
            np.zeros((config.N_X, N_DISTURBANCE), dtype=dtype),
        ]
    )
    return augmented_a, augmented_b, output


def design_leso_gain(
    Ad: np.ndarray,
    Bd: np.ndarray,
    design: LesoDesignConfig = DEFAULT_LESO_DESIGN,
) -> np.ndarray:
    """Solve the steady-state discrete LQE and return predictor-form gain.

    The returned gain is used in
    ``Xhat[k+1] = Ae Xhat[k] + Be du[k] + L (y[k] - Ce Xhat[k])``.
    """
    Ad, Bd = _validate_discrete_model(Ad, Bd)
    augmented_a, _, output = build_augmented_model(Ad, Bd)
    process_covariance = np.diag(
        np.concatenate(
            [design.state_process_variance, design.disturbance_process_variance]
        )
    )
    measurement_covariance = np.diag(design.measurement_noise_std**2)
    covariance = solve_discrete_are(
        augmented_a.T,
        output.T,
        process_covariance,
        measurement_covariance,
    )
    innovation_covariance = (
        output @ covariance @ output.T + measurement_covariance
    )
    cross_covariance = augmented_a @ covariance @ output.T
    gain = np.linalg.solve(
        innovation_covariance.T,
        cross_covariance.T,
    ).T
    return np.asarray(gain, dtype=float)


def observer_error_spectral_radius(
    Ad: np.ndarray,
    Bd: np.ndarray,
    gain: np.ndarray,
) -> float:
    """Return ``rho(Ae - L Ce)`` for one frozen scheduling point."""
    gain = np.asarray(gain)
    if gain.shape != (N_AUGMENTED, config.N_X):
        raise ValueError(
            f"gain must have shape ({N_AUGMENTED}, {config.N_X})"
        )
    augmented_a, _, output = build_augmented_model(Ad, Bd)
    poles = np.linalg.eigvals(augmented_a - gain @ output)
    return float(np.max(np.abs(poles)))


def initialize_leso(
    measured_state: np.ndarray,
) -> LesoEstimate:
    """Initialize state estimate from measurement and disturbances at zero."""
    measured_state = _readonly_vector(
        measured_state,
        config.N_X,
        "measured state",
    )
    return LesoEstimate(
        state=measured_state,
        disturbance=np.zeros(N_DISTURBANCE, dtype=float),
    )


def step_leso_split(
    estimate: LesoEstimate,
    measured_state: np.ndarray,
    applied_input_delta: np.ndarray,
    Ad: np.ndarray,
    Bd: np.ndarray,
    gain: np.ndarray,
    *,
    disturbance_limits: np.ndarray | None = None,
) -> LesoStepResult:
    """Advance the six-disturbance LESO without augmented matrix products.

    ``applied_input_delta`` must be the actuator command after saturation and
    rate limiting, minus the current equilibrium input ``u0``.
    """
    Ad, Bd = _validate_discrete_model(
        Ad,
        Bd,
    )
    gain = np.asarray(gain, dtype=float)
    if gain.shape != (N_AUGMENTED, config.N_X):
        raise ValueError(
            f"gain must have shape ({N_AUGMENTED}, {config.N_X})"
        )
    measured_state = _readonly_vector(
        measured_state,
        config.N_X,
        "measured state",
    )
    applied_input_delta = _readonly_vector(
        applied_input_delta,
        config.N_U,
        "applied input delta",
    )
    innovation = measured_state - estimate.state
    state_gain = gain[: config.N_X]
    disturbance_gain = gain[config.N_X :]
    next_state = (
        Ad @ estimate.state
        + Bd @ (applied_input_delta + estimate.disturbance)
        + state_gain @ innovation
    )
    next_disturbance = estimate.disturbance + disturbance_gain @ innovation
    if disturbance_limits is not None:
        limits = _readonly_vector(
            disturbance_limits,
            config.N_U,
            "disturbance limits",
        )
        if np.any(limits <= 0.0):
            raise ValueError("disturbance limits must be positive")
        next_disturbance = np.clip(
            next_disturbance,
            -limits,
            limits,
        )
    return LesoStepResult(
        estimate=LesoEstimate(next_state, next_disturbance),
        innovation=innovation,
    )


__all__ = [
    "DEFAULT_LESO_DESIGN",
    "N_AUGMENTED",
    "N_DISTURBANCE",
    "LesoDesignConfig",
    "LesoEstimate",
    "LesoStepResult",
    "build_augmented_model",
    "design_leso_gain",
    "initialize_leso",
    "observer_error_spectral_radius",
    "step_leso_split",
]
