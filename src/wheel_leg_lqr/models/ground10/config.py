"""Ten-state layout and controller settings, independent of ground/config.py."""

from dataclasses import dataclass

import numpy as np

N_Q = 5
N_X = 10
N_U = 4
COORDINATE_NAMES = ("x", "psi", "theta", "beta_1", "beta_2")
STATE_NAMES = (
    "x",
    "dx",
    "psi",
    "dpsi",
    "theta",
    "dtheta",
    "beta_1",
    "dbeta_1",
    "beta_2",
    "dbeta_2",
)
INPUT_NAMES = ("T_w1", "T_w2", "T_p1", "T_p2")

SAMPLE_TIME = 0.001
Q_DIAG = (80.0, 25.0, 40.0, 8.0, 400.0, 25.0, 120.0, 6.0, 120.0, 6.0)
R_DIAG = (2.8, 2.8, 0.1, 0.1)


@dataclass(frozen=True)
class Ground10Settings:
    sample_time: float = SAMPLE_TIME
    q_diag: tuple[float, ...] = Q_DIAG
    r_diag: tuple[float, ...] = R_DIAG

    def __post_init__(self):
        object.__setattr__(self, "q_diag", tuple(float(v) for v in self.q_diag))
        object.__setattr__(self, "r_diag", tuple(float(v) for v in self.r_diag))
        if not np.isfinite(self.sample_time) or self.sample_time <= 0:
            raise ValueError("sample_time must be positive and finite")
        for name, size in (("q_diag", N_X), ("r_diag", N_U)):
            values = np.asarray(getattr(self, name))
            if (
                values.shape != (size,)
                or not np.all(np.isfinite(values))
                or np.any(values <= 0)
            ):
                raise ValueError(f"{name} must contain {size} positive finite values")

    def make_q(self):
        return np.diag(self.q_diag)

    def make_r(self):
        return np.diag(self.r_diag)


def default_initial_errors():
    symmetric = np.zeros(N_X)
    symmetric[[6, 8]] = 0.5
    differential = np.zeros(N_X)
    differential[[6, 8]] = [0.5, -0.5]
    combined = np.zeros(N_X)
    combined[[0, 2, 4, 6, 8]] = [0.1, np.deg2rad(5), np.deg2rad(5), 0.25, -0.25]
    return (
        ("symmetric_leg_swing", symmetric),
        ("differential_leg_swing", differential),
        ("combined_body_and_leg_error", combined),
    )


DEFAULT_GROUND10 = Ground10Settings()
