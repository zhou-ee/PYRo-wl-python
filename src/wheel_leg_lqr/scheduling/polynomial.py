"""Normalized polynomial helpers for L1/L2 schedules."""

from __future__ import annotations

import numpy as np


# Same ordering as the existing RDOB Bq total-degree cubic schedule.
TOTAL_CUBIC_TERMS = tuple((i, j) for i in range(4) for j in range(4 - i))


def fit_total_cubic_coefficients(
    coordinate_1: np.ndarray, coordinate_2: np.ndarray, values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit 1,y,y²,y³,x,xy,xy²,x²,x²y,x³; no terms with i+j > 3."""
    x = np.asarray(coordinate_1, dtype=float).reshape(-1)
    y = np.asarray(coordinate_2, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float)
    if x.size != y.size or values.shape[0] != x.size:
        raise ValueError("coordinates and values must have the same sample count")
    design = np.column_stack([x**i * y**j for i, j in TOTAL_CUBIC_TERMS])
    flat, _, rank, _ = np.linalg.lstsq(design, values.reshape(x.size, -1), rcond=None)
    if rank != len(TOTAL_CUBIC_TERMS):
        raise ValueError("sample grid cannot identify all ten total-cubic terms")
    coefficients = flat.T.reshape((*values.shape[1:], len(TOTAL_CUBIC_TERMS)))
    return coefficients, (design @ flat).reshape(values.shape)


def evaluate_total_cubic_polynomial(
    coefficients: np.ndarray, coordinate_1: float, coordinate_2: float,
    *, dtype=np.float64,
) -> np.ndarray:
    """Evaluate the ten-term cubic in explicit Horner order and precision."""
    cast = np.dtype(dtype).type
    c = np.asarray(coefficients, dtype=dtype)
    if c.shape[-1] != len(TOTAL_CUBIC_TERMS):
        raise ValueError("a total cubic must have ten coefficients per channel")
    x, y = cast(coordinate_1), cast(coordinate_2)
    row0 = ((c[..., 3] * y + c[..., 2]) * y + c[..., 1]) * y + c[..., 0]
    row1 = (c[..., 6] * y + c[..., 5]) * y + c[..., 4]
    row2 = c[..., 8] * y + c[..., 7]
    return ((c[..., 9] * x + row2) * x + row1) * x + row0


def tensor_design_matrix(coordinate_1: np.ndarray, coordinate_2: np.ndarray, degree_1: int, degree_2: int) -> np.ndarray:
    """Build a tensor-product design matrix in ascending power order."""
    coordinate_1 = np.asarray(coordinate_1, dtype=float).reshape(-1)
    coordinate_2 = np.asarray(coordinate_2, dtype=float).reshape(-1)
    if coordinate_1.size != coordinate_2.size:
        raise ValueError("both coordinates must have the same sample count")
    return np.column_stack([coordinate_1**p * coordinate_2**q for p in range(degree_1 + 1) for q in range(degree_2 + 1)])


def fit_tensor_coefficients(coordinate_1: np.ndarray, coordinate_2: np.ndarray, values: np.ndarray, degree_1: int, degree_2: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit independent tensor-product polynomials for all trailing channels."""
    design = tensor_design_matrix(coordinate_1, coordinate_2, degree_1, degree_2)
    values = np.asarray(values, dtype=float)
    if values.shape[0] != design.shape[0]:
        raise ValueError("coordinates and values must have the same sample count")
    output_shape = values.shape[1:]
    coefficients = np.empty((*output_shape, degree_1 + 1, degree_2 + 1), dtype=float)
    fitted = np.empty_like(values)
    for index in np.ndindex(output_shape):
        flat, *_ = np.linalg.lstsq(design, values[(slice(None), *index)], rcond=None)
        coefficients[index] = flat.reshape(degree_1 + 1, degree_2 + 1)
        fitted[(slice(None), *index)] = design @ flat
    return coefficients, fitted


def evaluate_tensor_polynomial(coefficients: np.ndarray, coordinate_1: float, coordinate_2: float) -> np.ndarray:
    """Evaluate a tensor coefficient tensor in ascending power order."""
    coefficients = np.asarray(coefficients)
    powers_1 = np.array([coordinate_1**p for p in range(coefficients.shape[-2])])
    powers_2 = np.array([coordinate_2**q for q in range(coefficients.shape[-1])])
    return np.einsum("...pq,p,q->...", coefficients, powers_1, powers_2)
