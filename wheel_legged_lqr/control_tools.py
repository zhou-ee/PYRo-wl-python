# 使用控制库的离散化与 LQR / Library-based discretization and LQR
# 离散化与 LQR / Discretization and LQR

from __future__ import annotations

from dataclasses import dataclass

import control as control_lib
import numpy as np
from scipy.signal import cont2discrete

def discretize_zoh(A: np.ndarray, B: np.ndarray, sample_time: float) -> tuple[np.ndarray, np.ndarray]:
    """Use scipy exact ZOH /  scipy """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if sample_time <= 0.0:
        raise ValueError("sample_time must be positive")
    n = A.shape[0]
    C = np.zeros((1, n), dtype=float)
    D = np.zeros((1, B.shape[1]), dtype=float)
    Ad, Bd, _, _, _ = cont2discrete((A, B, C, D), sample_time, method="zoh")
    return Ad, Bd

def controllability_rank(A: np.ndarray, B: np.ndarray) -> int:
    """Use python-control ctrb /  python-control  ctrb"""
    controllability = control_lib.ctrb(np.asarray(A, dtype=float), np.asarray(B, dtype=float))
    return int(np.linalg.matrix_rank(controllability))

@dataclass(frozen=True)
class LQRResult:
    """Library LQR result /  LQR """

    P: np.ndarray
    K: np.ndarray
    closed_loop_eigenvalues: np.ndarray

    @property
    def stable(self) -> bool:
        """Check unit-circle stability / """
        return bool(np.max(np.abs(self.closed_loop_eigenvalues)) < 1.0)

def discrete_lqr(Ad: np.ndarray, Bd: np.ndarray, Q: np.ndarray, R: np.ndarray) -> LQRResult:
    """Call python-control dlqr /  python-control  dlqr"""
    # dlqr returns K, Riccati matrix S, and closed-loop poles E.
    # dlqr  KRiccati  S  E
    K, P, eigenvalues = control_lib.dlqr(Ad, Bd, Q, R)
    return LQRResult(
        P=np.asarray(P, dtype=float),
        K=np.asarray(K, dtype=float),
        closed_loop_eigenvalues=np.asarray(eigenvalues),
    )
