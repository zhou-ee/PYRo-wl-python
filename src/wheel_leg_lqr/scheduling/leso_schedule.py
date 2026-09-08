"""POD/Chebyshev schedules for the active firmware's split LESO matrices.

The generated runtime contract fills ``G``, ``H``, ``L_x`` and ``L_d`` inside
``wl_chassis_t::_fit_params``.  ``G-I``, ``H`` and ``L_d`` are fitted as three
separate complete dense blocks.  ``L_x = G - alpha I`` is kept exact instead of
being fitted a second time.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .. import resolve_output_dir
from ..artifacts.io import write_text
from ..control.leso import (
    DEFAULT_DISTURBANCE_POLE,
    DEFAULT_STATE_POLE,
    design_leso_correction,
    observer_error_spectral_radius,
)
from ..control.lqr import discretize_state_space, solve_discrete_lqr
from ..core import config
from ..core.model import linearize_static_trim
from .leg_length_schedule import (
    GainScheduleFit,
    compute_gain_schedule,
    make_leg_length_grid,
    render_firmware_lqr_body,
)
from .pod_chebyshev import (
    SymmetricPodChebyshevSchedule,
    fit_symmetric_pod_chebyshev,
    parity_terms,
)
from .polynomial import fit_tensor_coefficients


DEFAULT_TOTAL_DEGREE = 4
DEFAULT_G_EVEN_RANK = 3
DEFAULT_G_ODD_RANK = 3
DEFAULT_H_EVEN_RANK = 3
DEFAULT_H_ODD_RANK = 3
DEFAULT_LD_EVEN_RANK = 3
DEFAULT_LD_ODD_RANK = 3
DEFAULT_COMPENSATION_CUTOFF_HZ = 15.0
DEFAULT_FL_COMPENSATION_GAIN = 0.30

G_SCHEDULED_COLUMN_OFFSET = 4
G_SCHEDULED_COLUMN_COUNT = config.N_X - G_SCHEDULED_COLUMN_OFFSET
NOMINAL_LENGTH = 0.5 * (
    config.PARAMETERS.min_leg_length + config.PARAMETERS.max_leg_length
)
LENGTH_HALF_RANGE = 0.5 * (
    config.PARAMETERS.max_leg_length - config.PARAMETERS.min_leg_length
)

_FIXED_STRUCTURE_ATOL = 1.0e-10
_EXCHANGE_SYMMETRY_RTOL = 2.0e-10
_DEFAULT_G_ERROR_BUDGET = 1.0e-3
_DEFAULT_H_ERROR_BUDGET = 1.0e-3
_DEFAULT_LD_ERROR_BUDGET = 5.0e-3
_DEFAULT_OBSERVER_RADIUS_LIMIT = 0.995
_DEFAULT_CLOSED_LOOP_RADIUS_LIMIT = 0.9995


def _signed_permutation(indices: list[int], signs: list[int]) -> np.ndarray:
    return np.diag(np.asarray(signs, dtype=float)) @ np.eye(len(indices))[
        np.asarray(indices, dtype=int)
    ]


STATE_EXCHANGE = _signed_permutation(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 10, 11],
    [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1],
)
INPUT_EXCHANGE = _signed_permutation([1, 0, 3, 2, 5, 4], [1] * config.N_U)
G_COLUMN_EXCHANGE = STATE_EXCHANGE[
    G_SCHEDULED_COLUMN_OFFSET:, G_SCHEDULED_COLUMN_OFFSET:
]


@dataclass(frozen=True)
class MatrixFitMetrics:
    train_max_relative_error: float
    train_rms_relative_error: float
    train_max_absolute_error: float
    holdout_max_relative_error: float
    holdout_rms_relative_error: float
    holdout_max_absolute_error: float
    float32_train_max_relative_error: float
    float32_holdout_max_relative_error: float


@dataclass(frozen=True)
class ObserverMetrics:
    raw_train_max_radius: float
    raw_holdout_max_radius: float
    fitted_train_max_radius: float
    fitted_holdout_max_radius: float
    float32_train_max_radius: float
    float32_holdout_max_radius: float

    @property
    def minimum_float32_margin(self) -> float:
        return 1.0 - max(
            self.float32_train_max_radius,
            self.float32_holdout_max_radius,
        )


@dataclass(frozen=True)
class DownstreamMetrics:
    float32_train_max_radius: float
    float32_holdout_max_radius: float

    @property
    def minimum_float32_margin(self) -> float:
        return 1.0 - max(
            self.float32_train_max_radius,
            self.float32_holdout_max_radius,
        )


@dataclass(frozen=True)
class LesoScheduleFit:
    l1_grid: np.ndarray
    l2_grid: np.ndarray
    holdout_points: np.ndarray
    g_schedule: SymmetricPodChebyshevSchedule
    h_schedule: SymmetricPodChebyshevSchedule
    ld_schedule: SymmetricPodChebyshevSchedule
    g_metrics: MatrixFitMetrics
    h_metrics: MatrixFitMetrics
    ld_metrics: MatrixFitMetrics
    observer_metrics: ObserverMetrics
    downstream_metrics: DownstreamMetrics
    state_pole: float
    disturbance_pole: float
    unmatched_state_pole: float
    disturbance_coupling: float
    compensation_cutoff_hz: float
    fl_compensation_gain: float
    max_training_symmetry_residual: float
    max_holdout_symmetry_residual: float
    max_fixed_structure_residual: float

    def matrices(
        self,
        l1: float,
        l2: float,
        *,
        dtype: np.dtype | type = np.float64,
        firmware_order: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``G``, ``H``, ``L_x`` and ``L_d`` at one length pair."""

        dtype = np.dtype(dtype)
        dynamic = _assemble_g_dynamic(
            self.g_schedule.evaluate(
                l1, l2, dtype=dtype, firmware_order=firmware_order
            ),
            dtype=dtype,
        )
        state_matrix = np.eye(config.N_X, dtype=dtype) + dynamic
        input_matrix = self.h_schedule.evaluate(
            l1, l2, dtype=dtype, firmware_order=firmware_order
        )
        disturbance_gain = self.ld_schedule.evaluate(
            l1, l2, dtype=dtype, firmware_order=firmware_order
        )
        state_gain = state_matrix.copy()
        diagonal = np.diag_indices(config.N_X)
        state_gain[diagonal] -= dtype.type(self.unmatched_state_pole)
        return state_matrix, input_matrix, state_gain, disturbance_gain


def _fixed_g_dynamic_columns(
    *, dtype: np.dtype | type = np.float64
) -> np.ndarray:
    dtype = np.dtype(dtype)
    fixed = np.zeros((config.N_X, G_SCHEDULED_COLUMN_OFFSET), dtype=dtype)
    fixed[0, 1] = dtype.type(config.SAMPLE_TIME)
    fixed[2, 3] = dtype.type(config.SAMPLE_TIME)
    return fixed


def _assemble_g_dynamic(
    scheduled_columns: np.ndarray,
    *,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    dtype = np.dtype(dtype)
    scheduled_columns = np.asarray(scheduled_columns, dtype=dtype)
    expected = (config.N_X, G_SCHEDULED_COLUMN_COUNT)
    if scheduled_columns.shape[-2:] != expected:
        raise ValueError(f"scheduled G-I block must have trailing shape {expected}")
    dynamic = np.zeros(
        (*scheduled_columns.shape[:-2], config.N_X, config.N_X), dtype=dtype
    )
    dynamic[..., :, :G_SCHEDULED_COLUMN_OFFSET] = _fixed_g_dynamic_columns(
        dtype=dtype
    )
    dynamic[..., :, G_SCHEDULED_COLUMN_OFFSET:] = scheduled_columns
    return dynamic


def _compute_raw_matrices(
    points: np.ndarray,
    *,
    state_pole: float,
    disturbance_pole: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_matrices = []
    input_matrices = []
    disturbance_gains = []
    for l1, l2 in np.asarray(points, dtype=float):
        linearization = linearize_static_trim(
            float(l1),
            float(l2),
            beta_1=config.compute_equilibrium_beta_from_length(float(l1)),
            beta_2=config.compute_equilibrium_beta_from_length(float(l2)),
        )
        state_matrix, input_matrix = discretize_state_space(
            linearization.A, linearization.B
        )
        correction = design_leso_correction(
            state_matrix,
            input_matrix,
            state_pole=state_pole,
            disturbance_pole=disturbance_pole,
        )
        state_matrices.append(state_matrix)
        input_matrices.append(input_matrix)
        disturbance_gains.append(correction.disturbance_gain)
    return (
        np.asarray(state_matrices),
        np.asarray(input_matrices),
        np.asarray(disturbance_gains),
    )


def _holdout_points(
    grid: np.ndarray,
    *,
    random_count: int,
    seed: int,
) -> np.ndarray:
    if random_count < 0:
        raise ValueError("random_count must be non-negative")
    midpoints = 0.5 * (grid[:-1] + grid[1:])
    midpoint_grid = np.asarray(
        [(l1, l2) for l1 in midpoints for l2 in midpoints], dtype=float
    )
    boundary = np.asarray(
        [
            *[(grid[0], value) for value in midpoints],
            *[(grid[-1], value) for value in midpoints],
            *[(value, grid[0]) for value in midpoints],
            *[(value, grid[-1]) for value in midpoints],
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    random = rng.uniform(
        config.PARAMETERS.min_leg_length,
        config.PARAMETERS.max_leg_length,
        size=(random_count, 2),
    )
    # Include every random point's exchanged partner so the physical symmetry
    # assumption is checked on all holdout snapshots as well.
    points = np.vstack([midpoint_grid, boundary, random, random[:, ::-1]])
    rounded = np.round(points, decimals=14)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    return points[np.sort(unique_indices)]


def _exchange_residual_and_project(
    points: np.ndarray,
    matrices: tuple[np.ndarray, ...],
    transforms: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> tuple[tuple[np.ndarray, ...], float]:
    lookup = {
        tuple(np.round(point, decimals=13)): index
        for index, point in enumerate(points)
    }
    projected = [np.asarray(stack, dtype=float).copy() for stack in matrices]
    maximum_residual = 0.0
    visited: set[tuple[int, int]] = set()
    for index, point in enumerate(points):
        mirror_index = lookup.get(tuple(np.round(point[::-1], decimals=13)))
        if mirror_index is None:
            raise ValueError("snapshot set is not closed under parameter exchange")
        pair = tuple(sorted((index, mirror_index)))
        if pair in visited:
            continue
        visited.add(pair)
        for matrix_index, (row_transform, column_transform) in enumerate(
            transforms
        ):
            direct = projected[matrix_index][index]
            mirrored_back = (
                row_transform
                @ projected[matrix_index][mirror_index]
                @ column_transform.T
            )
            residual = float(np.max(np.abs(direct - mirrored_back)))
            scale = max(1.0, float(np.max(np.abs(direct))))
            maximum_residual = max(maximum_residual, residual / scale)
            average = 0.5 * (direct + mirrored_back)
            projected[matrix_index][index] = average
            projected[matrix_index][mirror_index] = (
                row_transform @ average @ column_transform.T
            )
    if maximum_residual > _EXCHANGE_SYMMETRY_RTOL:
        raise ValueError(
            "left/right exchange symmetry residual "
            f"{maximum_residual:.6e} exceeds {_EXCHANGE_SYMMETRY_RTOL:.6e}"
        )
    return tuple(projected), maximum_residual


def _matrix_fit_metrics(
    train_actual: np.ndarray,
    train_fitted: np.ndarray,
    holdout_actual: np.ndarray,
    holdout_fitted: np.ndarray,
    float_train_fitted: np.ndarray,
    float_holdout_fitted: np.ndarray,
) -> MatrixFitMetrics:
    def errors(actual: np.ndarray, fitted: np.ndarray) -> tuple[np.ndarray, float]:
        difference = np.asarray(fitted, dtype=float) - np.asarray(actual, dtype=float)
        denominator = np.linalg.norm(actual.reshape(len(actual), -1), axis=1)
        if np.any(denominator <= np.finfo(float).eps):
            raise ValueError("relative-error denominator is zero")
        relative = np.linalg.norm(difference.reshape(len(actual), -1), axis=1) / denominator
        return relative, float(np.max(np.abs(difference)))

    train_error, train_absolute = errors(train_actual, train_fitted)
    holdout_error, holdout_absolute = errors(holdout_actual, holdout_fitted)
    float_train_error, _ = errors(train_actual, float_train_fitted)
    float_holdout_error, _ = errors(holdout_actual, float_holdout_fitted)
    return MatrixFitMetrics(
        train_max_relative_error=float(np.max(train_error)),
        train_rms_relative_error=float(np.sqrt(np.mean(train_error**2))),
        train_max_absolute_error=train_absolute,
        holdout_max_relative_error=float(np.max(holdout_error)),
        holdout_rms_relative_error=float(np.sqrt(np.mean(holdout_error**2))),
        holdout_max_absolute_error=holdout_absolute,
        float32_train_max_relative_error=float(np.max(float_train_error)),
        float32_holdout_max_relative_error=float(np.max(float_holdout_error)),
    )


def _evaluate_schedule_stack(
    schedule: SymmetricPodChebyshevSchedule,
    points: np.ndarray,
    *,
    dtype: np.dtype | type,
    firmware_order: bool,
) -> np.ndarray:
    return np.asarray(
        [
            schedule.evaluate(
                float(l1),
                float(l2),
                dtype=dtype,
                firmware_order=firmware_order,
            )
            for l1, l2 in points
        ]
    )


def _observer_radii(
    state_matrices: np.ndarray,
    input_matrices: np.ndarray,
    disturbance_gains: np.ndarray,
    unmatched_pole: float,
) -> np.ndarray:
    radii = []
    for state_matrix, input_matrix, disturbance_gain in zip(
        state_matrices, input_matrices, disturbance_gains
    ):
        state_gain = state_matrix - unmatched_pole * np.eye(config.N_X)
        radii.append(
            observer_error_spectral_radius(
                state_matrix,
                input_matrix,
                state_gain,
                disturbance_gain,
            )
        )
    return np.asarray(radii)


def _fit_firmware_lqr_gain(
    train_points: np.ndarray,
    state_matrices: np.ndarray,
    input_matrices: np.ndarray,
) -> np.ndarray:
    """Fit the current host model's cubic L1/L2 reference gain law."""

    raw_gains = np.asarray(
        [
            solve_discrete_lqr(state_matrix, input_matrix)[0]
            for state_matrix, input_matrix in zip(
                state_matrices, input_matrices
            )
        ]
    )
    coordinates = (train_points - NOMINAL_LENGTH) / LENGTH_HALF_RANGE
    coefficients, _ = fit_tensor_coefficients(
        coordinates[:, 0],
        coordinates[:, 1],
        raw_gains,
        config.POLYNOMIAL_DEGREE,
        config.POLYNOMIAL_DEGREE,
    )
    return coefficients


def _evaluate_firmware_lqr_gain(
    coefficients: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Evaluate K with the nested ascending-polynomial float32 order."""

    coefficients = np.asarray(coefficients, dtype=np.float32)
    degree_1 = coefficients.shape[-2] - 1
    degree_2 = coefficients.shape[-1] - 1
    lower = np.float32(config.PARAMETERS.min_leg_length)
    upper = np.float32(config.PARAMETERS.max_leg_length)
    center = np.float32(np.float32(0.5) * np.float32(upper + lower))
    half_range = np.float32(np.float32(0.5) * np.float32(upper - lower))
    inverse_half_range = np.float32(np.float32(1.0) / half_range)
    gains = []
    for l1, l2 in np.asarray(points, dtype=float):
        coordinate_1 = np.float32(
            np.clip(
                np.float32(np.float32(np.float32(l1) - center) * inverse_half_range),
                -1.0,
                1.0,
            )
        )
        coordinate_2 = np.float32(
            np.clip(
                np.float32(np.float32(np.float32(l2) - center) * inverse_half_range),
                -1.0,
                1.0,
            )
        )
        gain = np.empty(coefficients.shape[:2], dtype=np.float32)
        for input_index in range(gain.shape[0]):
            for state_index in range(gain.shape[1]):
                p_terms = np.empty(degree_1 + 1, dtype=np.float32)
                for order_1 in range(degree_1 + 1):
                    value = coefficients[
                        input_index, state_index, order_1, degree_2
                    ]
                    for order_2 in range(degree_2 - 1, -1, -1):
                        value = np.float32(
                            np.float32(value * coordinate_2)
                            + coefficients[
                                input_index,
                                state_index,
                                order_1,
                                order_2,
                            ]
                        )
                    p_terms[order_1] = value
                value = p_terms[degree_1]
                for order_1 in range(degree_1 - 1, -1, -1):
                    value = np.float32(
                        np.float32(value * coordinate_1) + p_terms[order_1]
                    )
                gain[input_index, state_index] = value
        gains.append(gain.astype(float))
    return np.asarray(gains)


def _downstream_closed_loop_radii(
    raw_state_matrices: np.ndarray,
    raw_input_matrices: np.ndarray,
    gains: np.ndarray,
    fitted_state_matrices: np.ndarray,
    fitted_input_matrices: np.ndarray,
    fitted_disturbance_gains: np.ndarray,
    unmatched_pole: float,
    compensation_filter_alpha: float,
    compensation_gains: np.ndarray,
) -> np.ndarray:
    """Return radii of the plant, LESO and filtered disturbance compensation."""

    radii = []
    zero_input_state = np.zeros((config.N_U, config.N_X))
    zero_input_input = np.zeros((config.N_U, config.N_U))
    disturbance_identity = np.eye(config.N_U)
    filter_keep = 1.0 - compensation_filter_alpha
    compensation_gains = np.asarray(compensation_gains, dtype=float)
    if compensation_gains.shape != (config.N_U,):
        raise ValueError(
            f"compensation_gains must have shape ({config.N_U},)"
        )
    compensation_matrix = np.diag(compensation_gains)
    observer_disturbance_matrix = (
        disturbance_identity
        - compensation_filter_alpha * compensation_matrix
    )
    for raw_g, raw_h, gain, fitted_g, fitted_h, fitted_ld in zip(
        raw_state_matrices,
        raw_input_matrices,
        gains,
        fitted_state_matrices,
        fitted_input_matrices,
        fitted_disturbance_gains,
    ):
        fitted_lx = fitted_g - unmatched_pole * np.eye(config.N_X)
        raw_h_compensation = raw_h @ compensation_matrix
        fitted_h_compensation = fitted_h @ compensation_matrix
        closed_loop = np.block(
            [
                [
                    raw_g,
                    -raw_h @ gain,
                    -compensation_filter_alpha * raw_h_compensation,
                    -filter_keep * raw_h_compensation,
                ],
                [
                    fitted_lx,
                    fitted_g - fitted_h @ gain - fitted_lx,
                    fitted_h @ observer_disturbance_matrix,
                    -filter_keep * fitted_h_compensation,
                ],
                [
                    fitted_ld,
                    -fitted_ld,
                    disturbance_identity,
                    zero_input_input,
                ],
                [
                    zero_input_state,
                    zero_input_state,
                    compensation_filter_alpha * disturbance_identity,
                    filter_keep * disturbance_identity,
                ],
            ]
        )
        radii.append(float(np.max(np.abs(np.linalg.eigvals(closed_loop)))))
    return np.asarray(radii)


def compute_leso_schedule(
    step: float = 0.01,
    *,
    total_degree: int = DEFAULT_TOTAL_DEGREE,
    g_even_rank: int = DEFAULT_G_EVEN_RANK,
    g_odd_rank: int = DEFAULT_G_ODD_RANK,
    h_even_rank: int = DEFAULT_H_EVEN_RANK,
    h_odd_rank: int = DEFAULT_H_ODD_RANK,
    ld_even_rank: int = DEFAULT_LD_EVEN_RANK,
    ld_odd_rank: int = DEFAULT_LD_ODD_RANK,
    state_pole: float = DEFAULT_STATE_POLE,
    disturbance_pole: float = DEFAULT_DISTURBANCE_POLE,
    holdout_random_count: int = 64,
    holdout_seed: int = 20260902,
    g_error_budget: float = _DEFAULT_G_ERROR_BUDGET,
    h_error_budget: float = _DEFAULT_H_ERROR_BUDGET,
    ld_error_budget: float = _DEFAULT_LD_ERROR_BUDGET,
    observer_radius_limit: float = _DEFAULT_OBSERVER_RADIUS_LIMIT,
    closed_loop_radius_limit: float = _DEFAULT_CLOSED_LOOP_RADIUS_LIMIT,
    compensation_cutoff_hz: float = DEFAULT_COMPENSATION_CUTOFF_HZ,
    fl_compensation_gain: float = DEFAULT_FL_COMPENSATION_GAIN,
) -> LesoScheduleFit:
    """Fit and validate all matrices consumed by the firmware LESO chain."""

    nyquist_hz = 0.5 / config.SAMPLE_TIME
    if not (0.0 < compensation_cutoff_hz < nyquist_hz):
        raise ValueError(
            "compensation_cutoff_hz must lie strictly between zero and Nyquist"
        )
    if not np.isfinite(fl_compensation_gain) or not (
        0.0 <= fl_compensation_gain <= 1.0
    ):
        raise ValueError("fl_compensation_gain must lie in [0, 1]")

    grid = make_leg_length_grid(step=step)
    train_points = np.asarray(
        [(l1, l2) for l1 in grid for l2 in grid], dtype=float
    )
    holdout_points = _holdout_points(
        grid, random_count=holdout_random_count, seed=holdout_seed
    )
    train_raw = _compute_raw_matrices(
        train_points,
        state_pole=state_pole,
        disturbance_pole=disturbance_pole,
    )
    holdout_raw = _compute_raw_matrices(
        holdout_points,
        state_pole=state_pole,
        disturbance_pole=disturbance_pole,
    )
    transforms = (
        (STATE_EXCHANGE, STATE_EXCHANGE),
        (STATE_EXCHANGE, INPUT_EXCHANGE),
        (INPUT_EXCHANGE, STATE_EXCHANGE),
    )
    train_raw, train_symmetry = _exchange_residual_and_project(
        train_points, train_raw, transforms
    )
    holdout_raw, holdout_symmetry = _exchange_residual_and_project(
        holdout_points, holdout_raw, transforms
    )
    train_g, train_h, train_ld = train_raw
    holdout_g, holdout_h, holdout_ld = holdout_raw

    identity = np.eye(config.N_X)
    train_g_dynamic = train_g - identity
    holdout_g_dynamic = holdout_g - identity
    fixed = _fixed_g_dynamic_columns()
    fixed_residual = max(
        float(
            np.max(
                np.abs(
                    train_g_dynamic[:, :, :G_SCHEDULED_COLUMN_OFFSET] - fixed
                )
            )
        ),
        float(
            np.max(
                np.abs(
                    holdout_g_dynamic[:, :, :G_SCHEDULED_COLUMN_OFFSET] - fixed
                )
            )
        ),
    )
    if fixed_residual > _FIXED_STRUCTURE_ATOL:
        raise ValueError(
            "verified fixed G-I prefix changed: maximum residual "
            f"{fixed_residual:.6e} exceeds {_FIXED_STRUCTURE_ATOL:.6e}"
        )
    train_g_block = train_g_dynamic[:, :, G_SCHEDULED_COLUMN_OFFSET:]
    holdout_g_block = holdout_g_dynamic[:, :, G_SCHEDULED_COLUMN_OFFSET:]

    common = dict(
        nominal_length=NOMINAL_LENGTH,
        half_range=LENGTH_HALF_RANGE,
        total_degree=total_degree,
    )
    g_schedule = fit_symmetric_pod_chebyshev(
        train_g_block,
        train_points,
        STATE_EXCHANGE,
        G_COLUMN_EXCHANGE,
        even_rank=g_even_rank,
        odd_rank=g_odd_rank,
        **common,
    )
    h_schedule = fit_symmetric_pod_chebyshev(
        train_h,
        train_points,
        STATE_EXCHANGE,
        INPUT_EXCHANGE,
        even_rank=h_even_rank,
        odd_rank=h_odd_rank,
        **common,
    )
    ld_schedule = fit_symmetric_pod_chebyshev(
        train_ld,
        train_points,
        INPUT_EXCHANGE,
        STATE_EXCHANGE,
        even_rank=ld_even_rank,
        odd_rank=ld_odd_rank,
        **common,
    )

    schedules = (g_schedule, h_schedule, ld_schedule)
    train_fitted = tuple(
        _evaluate_schedule_stack(
            schedule, train_points, dtype=np.float64, firmware_order=False
        )
        for schedule in schedules
    )
    holdout_fitted = tuple(
        _evaluate_schedule_stack(
            schedule, holdout_points, dtype=np.float64, firmware_order=False
        )
        for schedule in schedules
    )
    float_train_fitted = tuple(
        _evaluate_schedule_stack(
            schedule, train_points, dtype=np.float32, firmware_order=True
        ).astype(float)
        for schedule in schedules
    )
    float_holdout_fitted = tuple(
        _evaluate_schedule_stack(
            schedule, holdout_points, dtype=np.float32, firmware_order=True
        ).astype(float)
        for schedule in schedules
    )

    train_g_fit = _assemble_g_dynamic(train_fitted[0])
    holdout_g_fit = _assemble_g_dynamic(holdout_fitted[0])
    float_train_g_fit = _assemble_g_dynamic(float_train_fitted[0])
    float_holdout_g_fit = _assemble_g_dynamic(float_holdout_fitted[0])
    g_metrics = _matrix_fit_metrics(
        train_g_dynamic,
        train_g_fit,
        holdout_g_dynamic,
        holdout_g_fit,
        float_train_g_fit,
        float_holdout_g_fit,
    )
    h_metrics = _matrix_fit_metrics(
        train_h,
        train_fitted[1],
        holdout_h,
        holdout_fitted[1],
        float_train_fitted[1],
        float_holdout_fitted[1],
    )
    ld_metrics = _matrix_fit_metrics(
        train_ld,
        train_fitted[2],
        holdout_ld,
        holdout_fitted[2],
        float_train_fitted[2],
        float_holdout_fitted[2],
    )

    correction = design_leso_correction(
        train_g[0],
        train_h[0],
        state_pole=state_pole,
        disturbance_pole=disturbance_pole,
    )
    unmatched_pole = correction.unmatched_state_pole
    observer_metrics = ObserverMetrics(
        raw_train_max_radius=float(
            np.max(_observer_radii(train_g, train_h, train_ld, unmatched_pole))
        ),
        raw_holdout_max_radius=float(
            np.max(
                _observer_radii(
                    holdout_g, holdout_h, holdout_ld, unmatched_pole
                )
            )
        ),
        fitted_train_max_radius=float(
            np.max(
                _observer_radii(
                    identity + train_g_fit,
                    train_fitted[1],
                    train_fitted[2],
                    unmatched_pole,
                )
            )
        ),
        fitted_holdout_max_radius=float(
            np.max(
                _observer_radii(
                    identity + holdout_g_fit,
                    holdout_fitted[1],
                    holdout_fitted[2],
                    unmatched_pole,
                )
            )
        ),
        float32_train_max_radius=float(
            np.max(
                _observer_radii(
                    identity + float_train_g_fit,
                    float_train_fitted[1],
                    float_train_fitted[2],
                    float(np.float32(unmatched_pole)),
                )
            )
        ),
        float32_holdout_max_radius=float(
            np.max(
                _observer_radii(
                    identity + float_holdout_g_fit,
                    float_holdout_fitted[1],
                    float_holdout_fitted[2],
                    float(np.float32(unmatched_pole)),
                )
            )
        ),
    )

    gain_coefficients = _fit_firmware_lqr_gain(
        train_points, train_g, train_h
    )
    float_train_gains = _evaluate_firmware_lqr_gain(
        gain_coefficients, train_points
    )
    float_holdout_gains = _evaluate_firmware_lqr_gain(
        gain_coefficients, holdout_points
    )
    compensation_filter_alpha = float(
        np.float32(
            1.0
            - np.exp(
                -2.0
                * np.pi
                * compensation_cutoff_hz
                * config.SAMPLE_TIME
            )
        )
    )
    compensation_gains = np.ones(config.N_U, dtype=np.float32)
    compensation_gains[4:] = np.float32(fl_compensation_gain)
    downstream_metrics = DownstreamMetrics(
        float32_train_max_radius=float(
            np.max(
                _downstream_closed_loop_radii(
                    train_g,
                    train_h,
                    float_train_gains,
                    identity + float_train_g_fit,
                    float_train_fitted[1],
                    float_train_fitted[2],
                    float(np.float32(unmatched_pole)),
                    compensation_filter_alpha,
                    compensation_gains,
                )
            )
        ),
        float32_holdout_max_radius=float(
            np.max(
                _downstream_closed_loop_radii(
                    holdout_g,
                    holdout_h,
                    float_holdout_gains,
                    identity + float_holdout_g_fit,
                    float_holdout_fitted[1],
                    float_holdout_fitted[2],
                    float(np.float32(unmatched_pole)),
                    compensation_filter_alpha,
                    compensation_gains,
                )
            )
        ),
    )

    budget_checks = (
        ("G-I", g_metrics.float32_holdout_max_relative_error, g_error_budget),
        ("H", h_metrics.float32_holdout_max_relative_error, h_error_budget),
        ("L_d", ld_metrics.float32_holdout_max_relative_error, ld_error_budget),
    )
    for name, error, budget in budget_checks:
        if budget <= 0.0:
            raise ValueError("matrix error budgets must be positive")
        if error > budget:
            raise ValueError(
                f"float32 holdout {name} relative error {error:.6e} "
                f"exceeds budget {budget:.6e}"
            )
    maximum_radius = max(
        observer_metrics.fitted_train_max_radius,
        observer_metrics.fitted_holdout_max_radius,
        observer_metrics.float32_train_max_radius,
        observer_metrics.float32_holdout_max_radius,
    )
    if maximum_radius >= observer_radius_limit:
        raise ValueError(
            f"fitted LESO radius {maximum_radius:.9f} exceeds "
            f"deployment limit {observer_radius_limit:.9f}"
        )
    maximum_closed_loop_radius = max(
        downstream_metrics.float32_train_max_radius,
        downstream_metrics.float32_holdout_max_radius,
    )
    if maximum_closed_loop_radius >= closed_loop_radius_limit:
        raise ValueError(
            f"downstream closed-loop radius {maximum_closed_loop_radius:.9f} "
            f"exceeds deployment limit {closed_loop_radius_limit:.9f}"
        )

    return LesoScheduleFit(
        l1_grid=grid,
        l2_grid=grid.copy(),
        holdout_points=holdout_points,
        g_schedule=g_schedule,
        h_schedule=h_schedule,
        ld_schedule=ld_schedule,
        g_metrics=g_metrics,
        h_metrics=h_metrics,
        ld_metrics=ld_metrics,
        observer_metrics=observer_metrics,
        downstream_metrics=downstream_metrics,
        state_pole=float(state_pole),
        disturbance_pole=float(disturbance_pole),
        unmatched_state_pole=unmatched_pole,
        disturbance_coupling=correction.disturbance_coupling,
        compensation_cutoff_hz=float(compensation_cutoff_hz),
        fl_compensation_gain=float(fl_compensation_gain),
        max_training_symmetry_residual=train_symmetry,
        max_holdout_symmetry_residual=holdout_symmetry,
        max_fixed_structure_residual=fixed_residual,
    )


def _firmware_float(value: float) -> str:
    return f"{float(np.float32(value)):.9e}f"


def _append_matrix(
    lines: list[str], declaration: str, matrix: np.ndarray
) -> None:
    lines.append(f"    static constexpr float {declaration} = {{")
    for row in np.asarray(matrix, dtype=np.float32):
        lines.append(
            "        {" + ", ".join(_firmware_float(value) for value in row) + "},"
        )
    lines.extend(["    };", ""])


def _append_tensor(
    lines: list[str], declaration: str, tensor: np.ndarray
) -> None:
    lines.append(f"    static constexpr float {declaration} = {{")
    for matrix in np.asarray(tensor, dtype=np.float32):
        lines.append("        {")
        for row in matrix:
            lines.append(
                "            {"
                + ", ".join(_firmware_float(value) for value in row)
                + "},"
            )
        lines.append("        },")
    lines.extend(["    };", ""])


def _append_schedule_constants(
    lines: list[str],
    name: str,
    schedule: SymmetricPodChebyshevSchedule,
    rows: str,
    columns: str,
) -> None:
    _append_matrix(lines, f"{name}_MEAN[{rows}][{columns}]", schedule.mean_matrix)
    _append_tensor(
        lines,
        f"{name}_EVEN_MODES[LESO_POD_RANK][{rows}][{columns}]",
        schedule.even_modes,
    )
    _append_tensor(
        lines,
        f"{name}_ODD_MODES[LESO_POD_RANK][{rows}][{columns}]",
        schedule.odd_modes,
    )
    _append_matrix(
        lines,
        f"{name}_EVEN_COEFFICIENTS[LESO_POD_RANK]"
        "[LESO_EVEN_TERM_COUNT]",
        schedule.even_coefficients,
    )
    _append_matrix(
        lines,
        f"{name}_ODD_COEFFICIENTS[LESO_POD_RANK]"
        "[LESO_ODD_TERM_COUNT]",
        schedule.odd_coefficients,
    )


def _render_combined_firmware_block(result: LesoScheduleFit) -> str:
    """Render the common source used by the split firmware artifacts."""

    schedules = (result.g_schedule, result.h_schedule, result.ld_schedule)
    degrees = {schedule.total_degree for schedule in schedules}
    if len(degrees) != 1:
        raise ValueError("G, H and L_d firmware schedules must share one degree")
    degree = result.g_schedule.total_degree
    if degree < 1:
        raise ValueError("firmware evaluation requires Chebyshev degree >= 1")
    ranks = {
        rank
        for schedule in schedules
        for rank in (schedule.even_rank, schedule.odd_rank)
    }
    if 0 in ranks:
        raise ValueError("firmware schedules require non-zero even and odd ranks")
    if len(ranks) != 1:
        raise ValueError("firmware requires equal G/H/L_d even/odd ranks")
    pod_rank = result.g_schedule.even_rank
    even_terms = result.g_schedule.even_terms
    odd_terms = result.g_schedule.odd_terms
    if any(not np.array_equal(schedule.even_terms, even_terms) for schedule in schedules):
        raise ValueError("G, H and L_d firmware schedules must share even terms")
    if any(not np.array_equal(schedule.odd_terms, odd_terms) for schedule in schedules):
        raise ValueError("G, H and L_d firmware schedules must share odd terms")
    off_diagonal_terms = even_terms[even_terms[:, 0] != even_terms[:, 1]]
    if not np.array_equal(off_diagonal_terms, odd_terms):
        raise ValueError(
            "odd firmware terms must be the off-diagonal subsequence of even terms"
        )
    lines = [
        "    // BEGIN GENERATED LESO POD-CHEBYSHEV FIT",
        "    // Generated from the current 14-state/6-input model; do not hand-edit.",
        "    // G-I, H and L_d use separate dense POD bases. L_x = G - alpha*I.",
        f"    constexpr uint32_t LESO_CHEBYSHEV_DEGREE = {degree};",
        f"    constexpr uint32_t LESO_EVEN_TERM_COUNT = {len(even_terms)};",
        f"    constexpr uint32_t LESO_ODD_TERM_COUNT = {len(odd_terms)};",
        f"    constexpr uint32_t LESO_G_COLUMN_OFFSET = {G_SCHEDULED_COLUMN_OFFSET};",
        f"    constexpr uint32_t LESO_G_COLUMN_COUNT = {G_SCHEDULED_COLUMN_COUNT};",
        f"    constexpr uint32_t LESO_POD_RANK = {pod_rank};",
        f"    constexpr float LESO_SAMPLE_TIME = {_firmware_float(config.SAMPLE_TIME)};",
        f"    constexpr float LESO_STATE_POLE = {_firmware_float(result.state_pole)};",
        f"    constexpr float LESO_DISTURBANCE_POLE = {_firmware_float(result.disturbance_pole)};",
        f"    constexpr float LESO_UNMATCHED_POLE = {_firmware_float(result.unmatched_state_pole)};",
        f"    constexpr float LESO_COMPENSATION_FILTER_CUTOFF_HZ = {_firmware_float(result.compensation_cutoff_hz)};",
        "    constexpr float LESO_COMPENSATION_FILTER_ALPHA = "
        f"{_firmware_float(1.0 - np.exp(-2.0 * np.pi * result.compensation_cutoff_hz * config.SAMPLE_TIME))};",
        "    // Per-input execution gains: Tw1, Tw2, Tp1, Tp2, Fl1, Fl2.",
        "    constexpr float LESO_COMPENSATION_GAIN[INPUT_DIM] = {",
        "        1.000000000e+00f, 1.000000000e+00f,",
        "        1.000000000e+00f, 1.000000000e+00f,",
        "        "
        f"{_firmware_float(result.fl_compensation_gain)}, "
        f"{_firmware_float(result.fl_compensation_gain)},",
        "    };",
        "",
        "    static constexpr uint8_t LESO_TERMS[LESO_EVEN_TERM_COUNT][2] = {",
    ]
    for order_1, order_2 in even_terms:
        lines.append(f"        {{{int(order_1)}, {int(order_2)}}},")
    lines.extend(["    };", ""])
    _append_schedule_constants(
        lines,
        "LESO_G",
        result.g_schedule,
        "STATE_DIM",
        "LESO_G_COLUMN_COUNT",
    )
    _append_schedule_constants(
        lines, "LESO_H", result.h_schedule, "STATE_DIM", "INPUT_DIM"
    )
    _append_schedule_constants(
        lines, "LESO_LD", result.ld_schedule, "INPUT_DIM", "STATE_DIM"
    )
    lines.extend(
        [
            "    float leso_chebyshev[2][LESO_CHEBYSHEV_DEGREE + 1] = {};",
            "    leso_chebyshev[0][0] = 1.0f;",
            "    leso_chebyshev[1][0] = 1.0f;",
            "    leso_chebyshev[0][1] = norm_L1;",
            "    leso_chebyshev[1][1] = norm_L2;",
            "    for (uint32_t order = 2; order <= LESO_CHEBYSHEV_DEGREE; ++order)",
            "    {",
            "        for (uint32_t axis = 0; axis < 2; ++axis)",
            "        {",
            "            const float product = leso_chebyshev[axis][1] *",
            "                                  leso_chebyshev[axis][order - 1];",
            "            leso_chebyshev[axis][order] =",
            "                product + product - leso_chebyshev[axis][order - 2];",
            "        }",
            "    }",
            "",
            "    float leso_even_basis[LESO_EVEN_TERM_COUNT] = {};",
            "    float leso_odd_basis[LESO_ODD_TERM_COUNT] = {};",
            "    uint32_t odd_term = 0;",
            "    for (uint32_t term = 0; term < LESO_EVEN_TERM_COUNT; ++term)",
            "    {",
            "        const uint32_t p = LESO_TERMS[term][0];",
            "        const uint32_t q = LESO_TERMS[term][1];",
            "        const float direct = leso_chebyshev[0][p] * leso_chebyshev[1][q];",
            "        if (p == q)",
            "        {",
            "            leso_even_basis[term] = direct;",
            "        }",
            "        else",
            "        {",
            "            const float exchanged =",
            "                leso_chebyshev[0][q] * leso_chebyshev[1][p];",
            "            leso_even_basis[term] = direct + exchanged;",
            "            leso_odd_basis[odd_term++] = direct - exchanged;",
            "        }",
            "    }",
            "",
        ]
    )
    lines.extend(
        [
            "    float leso_g_even_values[LESO_POD_RANK] = {};",
            "    float leso_g_odd_values[LESO_POD_RANK] = {};",
            "    float leso_h_even_values[LESO_POD_RANK] = {};",
            "    float leso_h_odd_values[LESO_POD_RANK] = {};",
            "    float leso_ld_even_values[LESO_POD_RANK] = {};",
            "    float leso_ld_odd_values[LESO_POD_RANK] = {};",
            "    for (uint32_t mode = 0; mode < LESO_POD_RANK; ++mode)",
            "    {",
            "        for (uint32_t term = 0; term < LESO_EVEN_TERM_COUNT; ++term)",
            "        {",
            "            const float basis = leso_even_basis[term];",
            "            leso_g_even_values[mode] +=",
            "                LESO_G_EVEN_COEFFICIENTS[mode][term] * basis;",
            "            leso_h_even_values[mode] +=",
            "                LESO_H_EVEN_COEFFICIENTS[mode][term] * basis;",
            "            leso_ld_even_values[mode] +=",
            "                LESO_LD_EVEN_COEFFICIENTS[mode][term] * basis;",
            "        }",
            "        for (uint32_t term = 0; term < LESO_ODD_TERM_COUNT; ++term)",
            "        {",
            "            const float basis = leso_odd_basis[term];",
            "            leso_g_odd_values[mode] +=",
            "                LESO_G_ODD_COEFFICIENTS[mode][term] * basis;",
            "            leso_h_odd_values[mode] +=",
            "                LESO_H_ODD_COEFFICIENTS[mode][term] * basis;",
            "            leso_ld_odd_values[mode] +=",
            "                LESO_LD_ODD_COEFFICIENTS[mode][term] * basis;",
            "        }",
            "    }",
            "",
            "    for (uint32_t row = 0; row < STATE_DIM; ++row)",
            "    {",
            "        for (uint32_t column = 0; column < STATE_DIM; ++column)",
            "        {",
            "            _ctx.data.G(row, column) = (row == column) ? 1.0f : 0.0f;",
            "        }",
            "    }",
            "    _ctx.data.G(state_def::X, state_def::DOT_X) = LESO_SAMPLE_TIME;",
            "    _ctx.data.G(state_def::PSI, state_def::DOT_PSI) = LESO_SAMPLE_TIME;",
            "    for (uint32_t row = 0; row < STATE_DIM; ++row)",
            "    {",
            "        for (uint32_t scheduled_column = 0;",
            "             scheduled_column < LESO_G_COLUMN_COUNT; ++scheduled_column)",
            "        {",
            "            float value = LESO_G_MEAN[row][scheduled_column];",
            "            for (uint32_t mode = 0; mode < LESO_POD_RANK; ++mode)",
            "            {",
            "                value += leso_g_even_values[mode] *",
            "                         LESO_G_EVEN_MODES[mode][row][scheduled_column];",
            "            }",
            "            for (uint32_t mode = 0; mode < LESO_POD_RANK; ++mode)",
            "            {",
            "                value += leso_g_odd_values[mode] *",
            "                         LESO_G_ODD_MODES[mode][row][scheduled_column];",
            "            }",
            "            _ctx.data.G(row, scheduled_column + LESO_G_COLUMN_OFFSET) += value;",
            "        }",
            "",
            "        for (uint32_t column = 0; column < INPUT_DIM; ++column)",
            "        {",
            "            float value = LESO_H_MEAN[row][column];",
            "            for (uint32_t mode = 0; mode < LESO_POD_RANK; ++mode)",
            "            {",
            "                value += leso_h_even_values[mode] * LESO_H_EVEN_MODES[mode][row][column];",
            "            }",
            "            for (uint32_t mode = 0; mode < LESO_POD_RANK; ++mode)",
            "            {",
            "                value += leso_h_odd_values[mode] * LESO_H_ODD_MODES[mode][row][column];",
            "            }",
            "            _ctx.data.H(row, column) = value;",
            "        }",
            "",
            "        for (uint32_t column = 0; column < STATE_DIM; ++column)",
            "        {",
            "            _ctx.data.L_x(row, column) = _ctx.data.G(row, column);",
            "        }",
            "        _ctx.data.L_x(row, row) -= LESO_UNMATCHED_POLE;",
            "    }",
            "",
            "    for (uint32_t row = 0; row < INPUT_DIM; ++row)",
            "    {",
            "        for (uint32_t column = 0; column < STATE_DIM; ++column)",
            "        {",
            "            float value = LESO_LD_MEAN[row][column];",
            "            for (uint32_t mode = 0; mode < LESO_POD_RANK; ++mode)",
            "            {",
            "                value += leso_ld_even_values[mode] * LESO_LD_EVEN_MODES[mode][row][column];",
            "            }",
            "            for (uint32_t mode = 0; mode < LESO_POD_RANK; ++mode)",
            "            {",
            "                value += leso_ld_odd_values[mode] * LESO_LD_ODD_MODES[mode][row][column];",
            "            }",
            "            _ctx.data.L_d(row, column) = value;",
            "        }",
            "    }",
            "    // END GENERATED LESO POD-CHEBYSHEV FIT",
        ]
    )
    return "\n".join(lines)


_FIRMWARE_RUNTIME_FIRST_LINE = (
    "    float leso_chebyshev[2][LESO_CHEBYSHEV_DEGREE + 1] = {};"
)


def render_firmware_coefficients(result: LesoScheduleFit) -> str:
    """Render static LESO data for insertion in ``coef.h``."""

    combined = _render_combined_firmware_block(result)
    static_part, separator, _ = combined.partition(_FIRMWARE_RUNTIME_FIRST_LINE)
    if not separator:
        raise RuntimeError("firmware runtime split marker is missing")
    source_lines = static_part.rstrip().splitlines()
    if not source_lines or "BEGIN GENERATED LESO" not in source_lines[0]:
        raise RuntimeError("firmware coefficient source marker is missing")
    body = []
    for line in source_lines[1:]:
        unindented = line[4:] if line.startswith("    ") else line
        body.append(unindented.replace("static constexpr", "constexpr", 1))
    return "\n".join(
        [
            "// BEGIN GENERATED LESO POD-CHEBYSHEV COEFFICIENTS",
            *body,
            "// END GENERATED LESO POD-CHEBYSHEV COEFFICIENTS",
        ]
    )


def render_firmware_fit_block(result: LesoScheduleFit) -> str:
    """Render only the dynamic evaluation performed in ``_fit_params``."""

    combined = _render_combined_firmware_block(result)
    _, separator, runtime_tail = combined.partition(_FIRMWARE_RUNTIME_FIRST_LINE)
    if not separator:
        raise RuntimeError("firmware runtime split marker is missing")
    old_end = "    // END GENERATED LESO POD-CHEBYSHEV FIT"
    runtime, end_marker, trailing = runtime_tail.rpartition(old_end)
    if not end_marker or trailing.strip():
        raise RuntimeError("firmware runtime end marker is missing")
    return "\n".join(
        [
            "    // BEGIN GENERATED LESO POD-CHEBYSHEV RUNTIME FIT",
            "    // Static dimensions, modes and coefficients are in coef.h.",
            separator + runtime.rstrip(),
            "    // END GENERATED LESO POD-CHEBYSHEV RUNTIME FIT",
        ]
    )


def render_combined_coef_header(
    lqr_result: GainScheduleFit,
    leso_result: LesoScheduleFit,
) -> str:
    """Render one firmware ``coef.h`` containing the unchanged LQR fit and LESO data."""

    return "\n".join(
        [
            "#ifndef __COEF_H__",
            "#define __COEF_H__",
            '#include "stdint.h"',
            "",
            render_firmware_lqr_body(lqr_result).rstrip(),
            "",
            render_firmware_coefficients(leso_result).rstrip(),
            "",
            "#endif",
            "",
        ]
    )


def _basis_operation_counts(total_degree: int) -> tuple[int, int]:
    even = parity_terms(total_degree, 1)
    recurrence_multiplies = 2 * max(0, total_degree - 1)
    recurrence_additions = 4 * max(0, total_degree - 1)
    even_off_diagonal = int(np.count_nonzero(even[:, 0] != even[:, 1]))
    basis_multiplies = len(even) + even_off_diagonal
    basis_additions = 2 * even_off_diagonal
    return (
        recurrence_multiplies + basis_multiplies,
        recurrence_additions + basis_additions,
    )


def _runtime_counts(result: LesoScheduleFit) -> dict[str, int]:
    schedules = (result.g_schedule, result.h_schedule, result.ld_schedule)
    basis_multiplies, basis_additions = _basis_operation_counts(
        result.g_schedule.total_degree
    )
    modal_multiplies = sum(schedule.modal_multiply_count for schedule in schedules)
    reconstruction_multiplies = sum(
        schedule.reconstruction_multiply_count for schedule in schedules
    )
    reconstruction_additions = (
        reconstruction_multiplies
        + config.N_X * G_SCHEDULED_COLUMN_COUNT
        + config.N_X
    )
    modal_additions = modal_multiplies
    observer_matvec_multiplies = (
        config.N_X * config.N_X
        + 2 * config.N_X * config.N_U
        + config.N_X * config.N_X
        + config.N_U * config.N_X
    )
    storage_floats = sum(schedule.storage_float_count for schedule in schedules)
    dense_lookup_floats = len(result.l1_grid) * len(result.l2_grid) * (
        config.N_X * G_SCHEDULED_COLUMN_COUNT
        + config.N_X * config.N_U
        + config.N_U * config.N_X
    )
    stack_floats = (
        2 * (result.g_schedule.total_degree + 1)
        + len(result.g_schedule.even_terms)
        + len(result.g_schedule.odd_terms)
        + sum(schedule.rank for schedule in schedules)
    )
    compensation_filter_multiplies = 3 * config.N_U
    compensation_filter_additions = 3 * config.N_U
    return {
        "fit_multiplies": int(
            basis_multiplies + modal_multiplies + reconstruction_multiplies
        ),
        "fit_additions": int(
            basis_additions + modal_additions + reconstruction_additions
        ),
        "observer_matvec_multiplies": int(observer_matvec_multiplies),
        "compensation_filter_multiplies": int(compensation_filter_multiplies),
        "compensation_filter_additions": int(compensation_filter_additions),
        "storage_floats": int(storage_floats),
        "storage_bytes": int(storage_floats * np.dtype(np.float32).itemsize),
        "dense_lookup_storage_floats": int(dense_lookup_floats),
        "dense_lookup_storage_bytes": int(
            dense_lookup_floats * np.dtype(np.float32).itemsize
        ),
        "stack_floats": int(stack_floats),
        "stack_bytes": int(stack_floats * np.dtype(np.float32).itemsize),
    }


def render_summary(result: LesoScheduleFit) -> str:
    counts = _runtime_counts(result)
    lines = [
        "# Six-input LESO POD–Chebyshev Schedule",
        "",
        f"- Domain: `L1,L2 in [{result.l1_grid[0]:.2f}, {result.l1_grid[-1]:.2f}] m`",
        f"- Training nodes: `{len(result.l1_grid)} x {len(result.l2_grid)} = {len(result.l1_grid) * len(result.l2_grid)}`",
        f"- Independent midpoint/boundary/random holdouts: `{len(result.holdout_points)}`",
        f"- G/H/L_d total Chebyshev degree: `{result.g_schedule.total_degree}`; independent even/odd terms: `{len(result.g_schedule.even_terms)} / {len(result.g_schedule.odd_terms)}`",
        f"- G-I scheduled block: `14 x {G_SCHEDULED_COLUMN_COUNT}`; ranks even/odd `{result.g_schedule.even_rank}/{result.g_schedule.odd_rank}`",
        f"- H scheduled block: `14 x 6`; ranks even/odd `{result.h_schedule.even_rank}/{result.h_schedule.odd_rank}`",
        f"- L_d scheduled block: `6 x 14`; ranks even/odd `{result.ld_schedule.even_rank}/{result.ld_schedule.odd_rank}`",
        "- Rank policy: deployment ranks are selected against the float32 holdout-error and observer-pole budgets; retained energy is diagnostic only.",
        f"- Retained POD energy G/H/L_d: `{result.g_schedule.retained_energy:.9%}` / `{result.h_schedule.retained_energy:.9%}` / `{result.ld_schedule.retained_energy:.9%}`",
        f"- Chebyshev design condition (max): `{max(schedule.even_design_condition for schedule in (result.g_schedule, result.h_schedule, result.ld_schedule)):.6f}` / `{max(schedule.odd_design_condition for schedule in (result.g_schedule, result.h_schedule, result.ld_schedule)):.6f}` (even/odd)",
        f"- Observer poles: matched fast `{result.state_pole:.6f}`, disturbance `{result.disturbance_pole:.6f}`, unmatched state `{result.unmatched_state_pole:.6f}`",
        f"- Disturbance-compensation low-pass: `{result.compensation_cutoff_hz:.3f} Hz` (1 kHz alpha `{1.0 - np.exp(-2.0 * np.pi * result.compensation_cutoff_hz * config.SAMPLE_TIME):.9f}`)",
        f"- Compensation execution gains Tw/Tp/Fl: `1.0 / 1.0 / {result.fl_compensation_gain:.3f}`; observer feedback still uses the full raw disturbance estimate.",
        "- Exchange maps: state `[x,dx,-psi,-dpsi,theta,dtheta,-phi,-dphi,L,dL,beta2,dbeta2,beta1,dbeta1]`; input `[Tw2,Tw1,Tp2,Tp1,Fl2,Fl1]`.",
        f"- Verified fixed structure: only columns `{G_SCHEDULED_COLUMN_OFFSET}..{config.N_X - 1}` of `G-I` are scheduled; `G[0,1]` and `G[2,3]` remain exactly `Ts`.",
        "",
        "| Matrix | Train max rel. | Holdout max rel. | Float32 holdout max rel. | Holdout max abs. |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in (
        ("G-I", result.g_metrics),
        ("H", result.h_metrics),
        ("L_d", result.ld_metrics),
    ):
        lines.append(
            f"| {name} | {metrics.train_max_relative_error:.6e} | "
            f"{metrics.holdout_max_relative_error:.6e} | "
            f"{metrics.float32_holdout_max_relative_error:.6e} | "
            f"{metrics.holdout_max_absolute_error:.6e} |"
        )
    observer = result.observer_metrics
    downstream = result.downstream_metrics
    lines.extend(
        [
            "",
            f"- Raw observer max radius, train/holdout: `{observer.raw_train_max_radius:.9f}` / `{observer.raw_holdout_max_radius:.9f}`",
            f"- Fitted observer max radius, train/holdout: `{observer.fitted_train_max_radius:.9f}` / `{observer.fitted_holdout_max_radius:.9f}`",
            f"- Float32 observer max radius, train/holdout: `{observer.float32_train_max_radius:.9f}` / `{observer.float32_holdout_max_radius:.9f}`",
            f"- Minimum float32 observer pole margin: `{observer.minimum_float32_margin:.9f}`",
            f"- Float32 plant + current-model reference LQR + LESO max radius, train/holdout: `{downstream.float32_train_max_radius:.9f}` / `{downstream.float32_holdout_max_radius:.9f}`",
            f"- Minimum downstream closed-loop pole margin: `{downstream.minimum_float32_margin:.9f}`",
            f"- Fixed-structure residual: `{result.max_fixed_structure_residual:.6e}`",
            f"- Raw exchange-symmetry residual, train/holdout: `{result.max_training_symmetry_residual:.6e}` / `{result.max_holdout_symmetry_residual:.6e}`",
            f"- Runtime fit cost: `{counts['fit_multiplies']}` multiplies, `{counts['fit_additions']}` additions",
            f"- Existing LESO matrix-vector cost: `{counts['observer_matvec_multiplies']}` multiplies per update",
            f"- Compensation low-pass and injection cost: `{counts['compensation_filter_multiplies']}` multiplies, `{counts['compensation_filter_additions']}` additions per update",
            f"- Coefficient storage: `{counts['storage_floats']}` float32 = `{counts['storage_bytes']}` bytes",
            f"- Dense 21x21 lookup equivalent: `{counts['dense_lookup_storage_bytes']}` bytes; coefficient reduction `{counts['dense_lookup_storage_bytes'] / counts['storage_bytes']:.2f}x`",
            f"- Fit scratch stack: `{counts['stack_floats']}` float32 = `{counts['stack_bytes']}` bytes",
            "",
            "The downstream check refits K from this host model; it does not certify a separately maintained firmware K table. The observer and fit use physical SI coordinates; only the two Chebyshev parameters are mapped to `[-1,1]`. Hardware timing and sensor-noise tuning still require real-robot validation.",
            "",
        ]
    )
    return "\n".join(lines)


def save_result(
    result: LesoScheduleFit,
    output_dir: Path,
    *,
    lqr_result: GainScheduleFit | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text(output_dir / "LESO_README.md", render_summary(result))
    write_text(
        output_dir / "leso_coefficients.inc",
        render_firmware_coefficients(result),
    )
    write_text(output_dir / "fit_params_leso.inc", render_firmware_fit_block(result))
    if lqr_result is not None:
        write_text(
            output_dir / "coef.h",
            render_combined_coef_header(lqr_result, result),
        )
    np.savez_compressed(
        output_dir / "leso_schedule.npz",
        l1_grid=result.l1_grid,
        l2_grid=result.l2_grid,
        holdout_points=result.holdout_points,
        state_pole=result.state_pole,
        disturbance_pole=result.disturbance_pole,
        unmatched_state_pole=result.unmatched_state_pole,
        compensation_cutoff_hz=result.compensation_cutoff_hz,
        compensation_filter_alpha=(
            1.0
            - np.exp(
                -2.0
                * np.pi
                * result.compensation_cutoff_hz
                * config.SAMPLE_TIME
            )
        ),
        fl_compensation_gain=result.fl_compensation_gain,
        compensation_gains=np.asarray(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                result.fl_compensation_gain,
                result.fl_compensation_gain,
            ],
            dtype=np.float32,
        ),
        total_degree=result.g_schedule.total_degree,
        g_mean=result.g_schedule.mean_matrix,
        g_even_modes=result.g_schedule.even_modes,
        g_odd_modes=result.g_schedule.odd_modes,
        g_even_coefficients=result.g_schedule.even_coefficients,
        g_odd_coefficients=result.g_schedule.odd_coefficients,
        h_mean=result.h_schedule.mean_matrix,
        h_even_modes=result.h_schedule.even_modes,
        h_odd_modes=result.h_schedule.odd_modes,
        h_even_coefficients=result.h_schedule.even_coefficients,
        h_odd_coefficients=result.h_schedule.odd_coefficients,
        ld_mean=result.ld_schedule.mean_matrix,
        ld_even_modes=result.ld_schedule.even_modes,
        ld_odd_modes=result.ld_schedule.odd_modes,
        ld_even_coefficients=result.ld_schedule.even_coefficients,
        ld_odd_coefficients=result.ld_schedule.odd_coefficients,
    )
    counts = _runtime_counts(result)
    report = {
        "domain_m": [float(result.l1_grid[0]), float(result.l1_grid[-1])],
        "training_grid": [len(result.l1_grid), len(result.l2_grid)],
        "holdout_count": len(result.holdout_points),
        "total_degree": result.g_schedule.total_degree,
        "term_count": {
            "even": len(result.g_schedule.even_terms),
            "odd": len(result.g_schedule.odd_terms),
        },
        "ranks": {
            "g": [result.g_schedule.even_rank, result.g_schedule.odd_rank],
            "h": [result.h_schedule.even_rank, result.h_schedule.odd_rank],
            "ld": [result.ld_schedule.even_rank, result.ld_schedule.odd_rank],
        },
        "retained_energy": {
            "g": result.g_schedule.retained_energy,
            "h": result.h_schedule.retained_energy,
            "ld": result.ld_schedule.retained_energy,
        },
        "float32_holdout_max_relative_error": {
            "g_dynamic": result.g_metrics.float32_holdout_max_relative_error,
            "h": result.h_metrics.float32_holdout_max_relative_error,
            "ld": result.ld_metrics.float32_holdout_max_relative_error,
        },
        "observer": {
            "state_pole": result.state_pole,
            "disturbance_pole": result.disturbance_pole,
            "unmatched_state_pole": result.unmatched_state_pole,
            "float32_train_max_radius": result.observer_metrics.float32_train_max_radius,
            "float32_holdout_max_radius": result.observer_metrics.float32_holdout_max_radius,
            "minimum_float32_margin": result.observer_metrics.minimum_float32_margin,
        },
        "compensation_filter": {
            "cutoff_hz": result.compensation_cutoff_hz,
            "alpha": 1.0
            - np.exp(
                -2.0
                * np.pi
                * result.compensation_cutoff_hz
                * config.SAMPLE_TIME
            ),
            "execution_gains": {
                "wheel_torque": 1.0,
                "hip_torque": 1.0,
                "leg_force": result.fl_compensation_gain,
            },
        },
        "downstream_closed_loop": {
            "controller": "current-host-model cubic L1/L2 reference LQR",
            "float32_train_max_radius": result.downstream_metrics.float32_train_max_radius,
            "float32_holdout_max_radius": result.downstream_metrics.float32_holdout_max_radius,
            "minimum_float32_margin": result.downstream_metrics.minimum_float32_margin,
        },
        "validation": {
            "fixed_structure_residual": result.max_fixed_structure_residual,
            "training_exchange_symmetry_residual": result.max_training_symmetry_residual,
            "holdout_exchange_symmetry_residual": result.max_holdout_symmetry_residual,
        },
        "runtime": counts,
    }
    (output_dir / "leso_schedule.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--degree", type=int, default=DEFAULT_TOTAL_DEGREE)
    parser.add_argument("--state-pole", type=float, default=DEFAULT_STATE_POLE)
    parser.add_argument(
        "--disturbance-pole",
        type=float,
        default=DEFAULT_DISTURBANCE_POLE,
    )
    parser.add_argument(
        "--compensation-cutoff-hz",
        type=float,
        default=DEFAULT_COMPENSATION_CUTOFF_HZ,
    )
    parser.add_argument(
        "--fl-compensation-gain",
        type=float,
        default=DEFAULT_FL_COMPENSATION_GAIN,
    )
    parser.add_argument("--holdout-random-count", type=int, default=64)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/generated/leso")
    )
    args = parser.parse_args()
    result = compute_leso_schedule(
        step=args.step,
        total_degree=args.degree,
        state_pole=args.state_pole,
        disturbance_pole=args.disturbance_pole,
        compensation_cutoff_hz=args.compensation_cutoff_hz,
        fl_compensation_gain=args.fl_compensation_gain,
        holdout_random_count=args.holdout_random_count,
    )
    lqr_result = compute_gain_schedule(step=args.step)
    output_dir = resolve_output_dir(args.output_dir)
    save_result(result, output_dir, lqr_result=lqr_result)
    print(render_summary(result))


__all__ = [
    "DEFAULT_COMPENSATION_CUTOFF_HZ",
    "DEFAULT_FL_COMPENSATION_GAIN",
    "DEFAULT_TOTAL_DEGREE",
    "G_SCHEDULED_COLUMN_COUNT",
    "G_SCHEDULED_COLUMN_OFFSET",
    "INPUT_EXCHANGE",
    "DownstreamMetrics",
    "LesoScheduleFit",
    "MatrixFitMetrics",
    "ObserverMetrics",
    "STATE_EXCHANGE",
    "compute_leso_schedule",
    "render_combined_coef_header",
    "main",
    "render_firmware_coefficients",
    "render_firmware_fit_block",
    "render_summary",
    "save_result",
]


if __name__ == "__main__":
    main()
