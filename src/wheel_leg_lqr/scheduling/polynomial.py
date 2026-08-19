"""Normalized polynomial helpers for the L1/L2 tensor-product schedule."""

from __future__ import annotations

import numpy as np


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
