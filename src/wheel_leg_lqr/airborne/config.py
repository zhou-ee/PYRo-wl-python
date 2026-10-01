"""Physical data and controller weights owned by the airborne model.

These values are copied from the current CAD/robot parameter set.  They are
intentionally not imported from the ground-contact fourteen-state controller.
Reidentify or update this module separately when hardware parameters change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AirborneParameters:
    wheel_mass: float = 0.57
    leg_mass: float = 1.232
    body_mass: float = 18.9255
    body_pitch_inertia: float = 0.582892610
    body_com_x_from_hip: float = 0.0
    body_com_z_from_hip: float = 0.032
    min_leg_length: float = 0.18
    max_leg_length: float = 0.38
    # CAD leg COM position relative to the lower pivot, ascending powers of L.
    leg_com_lwp_poly: tuple[float, float, float, float] = (
        0.05806884, 0.37602793, 0.79724260, -0.73809850
    )
    leg_com_x_poly: tuple[float, float, float, float] = (
        -0.07916397, 0.03685243, -0.12886076, 0.69820022
    )

    def __post_init__(self) -> None:
        positive = (
            self.wheel_mass,
            self.leg_mass,
            self.body_mass,
            self.body_pitch_inertia,
            self.min_leg_length,
            self.max_leg_length,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("airborne physical masses, inertia, and lengths must be positive")
        if self.max_leg_length <= self.min_leg_length:
            raise ValueError("airborne max_leg_length must exceed min_leg_length")
        offsets_and_fits = (
            self.body_com_x_from_hip,
            self.body_com_z_from_hip,
            *self.leg_com_lwp_poly,
            *self.leg_com_x_poly,
        )
        if not all(np.isfinite(value) for value in offsets_and_fits):
            raise ValueError("airborne COM offsets and fits must be finite")


PARAMETERS = AirborneParameters()
SAMPLE_TIME = 0.001
POLYNOMIAL_DEGREE = 3
HIP_TORQUE_LIMIT_NM = 60.0

# State order: beta_1, dbeta_1, beta_2, dbeta_2, theta, dtheta.
Q_DIAG = np.array([1200.0, 6.0, 1200.0, 6.0, 400.0, 25.0], dtype=float)
# Input order: T_p1, T_p2.
R_DIAG = np.array([0.1, 0.1], dtype=float)


def make_q() -> np.ndarray:
    return np.diag(Q_DIAG)


def make_r() -> np.ndarray:
    return np.diag(R_DIAG)
