"""Small, dependency-isolated helpers shared by controller analyses."""

from __future__ import annotations

import control
import numpy as np
from scipy.signal import cont2discrete

from ..core import config


def discretize_state_space(A: np.ndarray, B: np.ndarray, *, sample_time: float = config.SAMPLE_TIME) -> tuple[np.ndarray, np.ndarray]:
    """Discretize a continuous state-space pair using zero-order hold."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if B.ndim != 2 or B.shape[0] != A.shape[0]:
        raise ValueError("A and B must have the same state dimension")
    if sample_time <= 0:
        raise ValueError("sample_time must be positive")
    output = np.zeros((1, A.shape[0]), dtype=float)
    feedthrough = np.zeros((1, B.shape[1]), dtype=float)
    Ad, Bd, _, _, _ = cont2discrete((A, B, output, feedthrough), float(sample_time), method="zoh")
    return np.asarray(Ad, dtype=float), np.asarray(Bd, dtype=float)


def controllability_rank(A: np.ndarray, B: np.ndarray) -> int:
    """Return the numerical rank of the Kalman controllability matrix."""
    return int(np.linalg.matrix_rank(np.asarray(control.ctrb(A, B), dtype=float)))


def solve_discrete_lqr(Ad: np.ndarray, Bd: np.ndarray, Q: np.ndarray | None = None, R: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the configured discrete LQR problem and normalize outputs."""
    Q = config.make_q() if Q is None else np.asarray(Q, dtype=float)
    R = config.make_r() if R is None else np.asarray(R, dtype=float)
    gain, cost, poles = control.dlqr(Ad, Bd, Q, R)
    return np.asarray(gain, dtype=float), np.asarray(cost, dtype=float), np.asarray(poles)

