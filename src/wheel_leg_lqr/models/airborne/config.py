FULL_STATE_INDICES = (10, 11, 12, 13, 4, 5)
"""Six-state airborne controller settings; shares root physical data."""
import numpy as np
SAMPLE_TIME = 0.001

# State order: beta_1, dbeta_1, beta_2, dbeta_2, theta, dtheta.
Q_DIAG = np.array([1200.0, 6.0, 1200.0, 6.0, 400.0, 25.0], dtype=float)
# Input order: T_p1, T_p2.
R_DIAG = np.array([0.1, 0.1], dtype=float)


def make_q() -> np.ndarray:
    return np.diag(Q_DIAG)


def make_r() -> np.ndarray:
    return np.diag(R_DIAG)

AIRBORNE_STATE_NAMES = (
    "beta_1", "dbeta_1", "beta_2", "dbeta_2", "theta", "dtheta"
)
AIRBORNE_INPUT_NAMES = ("T_p1", "T_p2")
