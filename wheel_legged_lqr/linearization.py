"""文档坐标系数值线性化 / Numerical linearization in document coordinates."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .dynamics import state_derivative
from .parameters import RobotParameters


def _central_difference(
    func: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
    *,
    abs_step: float,
    rel_step: float,
) -> np.ndarray:
    """Calculate a central-difference Jacobian /  Jacobian"""
    point = np.asarray(point, dtype=float).reshape(-1)
    output = np.asarray(func(point), dtype=float).reshape(-1)
    jacobian = np.empty((output.size, point.size), dtype=float)

    # Perturb one variable at a time / 
    for index in range(point.size):
        step = abs_step + rel_step * max(1.0, abs(point[index]))
        plus = point.copy()
        minus = point.copy()
        plus[index] += step
        minus[index] -= step
        jacobian[:, index] = (
            np.asarray(func(plus), dtype=float) - np.asarray(func(minus), dtype=float)
        ) / (2.0 * step)
    return jacobian


def linearize(
    state,
    inputs,
    p: RobotParameters,
    *,
    abs_step: float = 1e-7,
    rel_step: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return continuous ``A`` and ``B`` /  AB 

    This numerically evaluates the Jacobians of the exact document model,
    corresponding to ``A=df/dX`` and ``B=df/dU``.
    /  Jacobian 
    ``A=df/dX``  ``B=df/dU``
    """
    state = np.asarray(state, dtype=float).reshape(6)
    inputs = np.asarray(inputs, dtype=float).reshape(2)
    A = _central_difference(
        lambda value: state_derivative(value, inputs, p),
        state,
        abs_step=abs_step,
        rel_step=rel_step,
    )
    B = _central_difference(
        lambda value: state_derivative(state, value, p),
        inputs,
        abs_step=abs_step,
        rel_step=rel_step,
    )
    return A, B
