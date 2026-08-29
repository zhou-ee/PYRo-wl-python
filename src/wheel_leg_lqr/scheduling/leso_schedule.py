"""Sparse L1/L2 scheduling for the physical-unit six-disturbance LESO model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .. import resolve_output_dir
from ..artifacts.io import write_text
from ..control.leso import (
    DEFAULT_LESO_DESIGN,
    N_AUGMENTED,
    LesoDesignConfig,
    LesoEstimate,
    LesoStepResult,
    design_leso_gain,
    observer_error_spectral_radius,
)
from ..core import config
from .leg_length_schedule import GainNode, compute_gain_node, make_leg_length_grid
from .polynomial import fit_tensor_coefficients


_STATE_MIRROR = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 10, 11],
    dtype=int,
)
_INPUT_MIRROR = np.array([1, 0, 3, 2, 5, 4], dtype=int)


@dataclass(frozen=True)
class SparseTensorSchedule:
    """A fixed matrix plus tensor-polynomial values at selected entries."""

    baseline: np.ndarray
    entries: np.ndarray
    coefficients: np.ndarray
    nominal_length: float = 0.28
    half_range: float = 0.10
    zero_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        baseline = np.asarray(self.baseline, dtype=float).copy()
        entries = np.asarray(self.entries, dtype=int).reshape(-1, 2).copy()
        coefficients = np.asarray(self.coefficients, dtype=float).copy()
        if baseline.ndim != 2:
            raise ValueError("schedule baseline must be a matrix")
        if coefficients.ndim != 3 or coefficients.shape[0] != len(entries):
            raise ValueError("one tensor coefficient matrix is required per entry")
        if len(entries):
            if (
                np.any(entries[:, 0] < 0)
                or np.any(entries[:, 0] >= baseline.shape[0])
                or np.any(entries[:, 1] < 0)
                or np.any(entries[:, 1] >= baseline.shape[1])
            ):
                raise ValueError("scheduled entry lies outside the baseline matrix")
            if len({tuple(entry) for entry in entries}) != len(entries):
                raise ValueError("scheduled entries must be unique")
        if self.half_range <= 0.0:
            raise ValueError("schedule half_range must be positive")
        if self.zero_tolerance < 0.0:
            raise ValueError("schedule zero_tolerance must be non-negative")
        baseline.setflags(write=False)
        entries.setflags(write=False)
        coefficients.setflags(write=False)
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "coefficients", coefficients)

    def polynomial_coordinates(self, l1: float, l2: float) -> tuple[float, float]:
        lam1 = np.clip(
            (float(l1) - self.nominal_length) / self.half_range,
            -1.0,
            1.0,
        )
        lam2 = np.clip(
            (float(l2) - self.nominal_length) / self.half_range,
            -1.0,
            1.0,
        )
        return float(lam1), float(lam2)

    def scheduled_values(
        self,
        l1: float,
        l2: float,
        *,
        dtype: np.dtype | type = np.float64,
    ) -> np.ndarray:
        dtype = np.dtype(dtype)
        lam1, lam2 = self.polynomial_coordinates(l1, l2)
        coefficients = self.coefficients.astype(dtype, copy=False)
        powers_1 = np.array(
            [dtype.type(lam1) ** p for p in range(coefficients.shape[-2])],
            dtype=dtype,
        )
        powers_2 = np.array(
            [dtype.type(lam2) ** q for q in range(coefficients.shape[-1])],
            dtype=dtype,
        )
        return np.einsum("kpq,p,q->k", coefficients, powers_1, powers_2)

    def evaluate(
        self,
        l1: float,
        l2: float,
        *,
        dtype: np.dtype | type = np.float64,
    ) -> np.ndarray:
        matrix = self.baseline.astype(dtype, copy=True)
        if len(self.entries):
            matrix[self.entries[:, 0], self.entries[:, 1]] = self.scheduled_values(
                l1,
                l2,
                dtype=dtype,
            )
        return matrix

    def matvec(
        self,
        l1: float,
        l2: float,
        vector: np.ndarray,
        *,
        dtype: np.dtype | type = np.float64,
    ) -> np.ndarray:
        """Multiply without materializing a scheduled dense matrix."""
        dtype = np.dtype(dtype)
        vector = np.asarray(vector, dtype=dtype).reshape(-1)
        if vector.size != self.baseline.shape[1]:
            raise ValueError(
                f"vector must contain {self.baseline.shape[1]} values"
            )
        baseline = self.baseline.astype(dtype, copy=False)
        fixed_mask = np.abs(baseline) > self.zero_tolerance
        if len(self.entries):
            fixed_mask[self.entries[:, 0], self.entries[:, 1]] = False
        fixed_entries = np.argwhere(fixed_mask)
        result = np.zeros(self.baseline.shape[0], dtype=dtype)
        if len(fixed_entries):
            fixed_rows = fixed_entries[:, 0]
            fixed_columns = fixed_entries[:, 1]
            np.add.at(
                result,
                fixed_rows,
                baseline[fixed_rows, fixed_columns] * vector[fixed_columns],
            )
        if len(self.entries):
            rows = self.entries[:, 0]
            columns = self.entries[:, 1]
            scheduled_products = self.scheduled_values(
                l1,
                l2,
                dtype=dtype,
            ) * vector[columns]
            np.add.at(result, rows, scheduled_products)
        return result

    @property
    def fixed_multiply_count(self) -> int:
        fixed_mask = np.abs(self.baseline) > self.zero_tolerance
        if len(self.entries):
            fixed_mask[self.entries[:, 0], self.entries[:, 1]] = False
        return int(np.count_nonzero(fixed_mask))

    @property
    def multiply_count(self) -> int:
        return self.fixed_multiply_count + len(self.entries)


@dataclass(frozen=True)
class LesoScheduleFit:
    l1_grid: np.ndarray
    l2_grid: np.ndarray
    ad_schedule: SparseTensorSchedule
    bd_schedule: SparseTensorSchedule
    fixed_gain: np.ndarray
    design: LesoDesignConfig
    gain_reference: tuple[float, float]
    max_ad_dynamic_relative_error: float
    max_bd_relative_error: float
    max_raw_observer_spectral_radius: float
    max_fitted_observer_spectral_radius: float
    max_float32_observer_spectral_radius: float
    max_selected_ad_fit_error: float
    max_selected_bd_fit_error: float

    def matrices(
        self,
        l1: float,
        l2: float,
        *,
        dtype: np.dtype | type = np.float64,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.ad_schedule.evaluate(l1, l2, dtype=dtype),
            self.bd_schedule.evaluate(l1, l2, dtype=dtype),
        )


def _mirror_closed_entries(
    matrices: np.ndarray,
    baseline: np.ndarray,
    target_count: int,
    row_mirror: np.ndarray,
    column_mirror: np.ndarray,
) -> np.ndarray:
    if target_count < 0:
        raise ValueError("scheduled entry count must be non-negative")
    if target_count == 0:
        return np.empty((0, 2), dtype=int)
    variation = np.max(np.abs(matrices - baseline), axis=0)
    groups: dict[tuple[tuple[int, int], ...], float] = {}
    for row in range(baseline.shape[0]):
        for column in range(baseline.shape[1]):
            entry = (row, column)
            mirrored = (int(row_mirror[row]), int(column_mirror[column]))
            group = tuple(sorted({entry, mirrored}))
            groups[group] = max(groups.get(group, 0.0), *(variation[item] for item in group))
    selected: set[tuple[int, int]] = set()
    for group, score in sorted(groups.items(), key=lambda item: item[1], reverse=True):
        if score <= np.finfo(float).eps:
            break
        selected.update(group)
        if len(selected) >= target_count:
            break
    return np.asarray(sorted(selected), dtype=int).reshape(-1, 2)


def _fit_sparse_schedule(
    matrices: np.ndarray,
    points: np.ndarray,
    baseline: np.ndarray,
    target_count: int,
    row_mirror: np.ndarray,
    column_mirror: np.ndarray,
    degree_1: int,
    degree_2: int,
) -> tuple[SparseTensorSchedule, np.ndarray, float]:
    entries = _mirror_closed_entries(
        matrices,
        baseline,
        target_count,
        row_mirror,
        column_mirror,
    )
    lam1 = (points[:, 0] - 0.28) / 0.10
    lam2 = (points[:, 1] - 0.28) / 0.10
    if len(entries) == 0:
        schedule = SparseTensorSchedule(
            baseline=baseline,
            entries=entries,
            coefficients=np.empty((0, degree_1 + 1, degree_2 + 1)),
        )
        fitted_matrices = np.broadcast_to(baseline, matrices.shape).copy()
        return schedule, fitted_matrices, 0.0
    selected_values = matrices[:, entries[:, 0], entries[:, 1]]
    coefficients, fitted_values = fit_tensor_coefficients(
        lam1,
        lam2,
        selected_values,
        degree_1,
        degree_2,
    )
    schedule = SparseTensorSchedule(
        baseline=baseline,
        entries=entries,
        coefficients=coefficients,
    )
    fitted_matrices = np.broadcast_to(baseline, matrices.shape).copy()
    fitted_matrices[:, entries[:, 0], entries[:, 1]] = fitted_values
    selected_fit_error = float(np.max(np.abs(fitted_values - selected_values)))
    return schedule, fitted_matrices, selected_fit_error


def _max_relative_error(
    actual: np.ndarray,
    fitted: np.ndarray,
    *,
    offset: np.ndarray | None = None,
) -> float:
    if offset is not None:
        actual = actual - offset
        fitted = fitted - offset
    actual_flat = actual.reshape(actual.shape[0], -1)
    fitted_flat = fitted.reshape(fitted.shape[0], -1)
    denominator = np.linalg.norm(actual_flat, axis=1)
    if np.any(denominator <= np.finfo(float).eps):
        raise ValueError("relative error denominator is zero")
    errors = np.linalg.norm(actual_flat - fitted_flat, axis=1) / denominator
    return float(np.max(errors))


def _make_nodes(
    step: float,
    nodes: Sequence[GainNode] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[GainNode, ...]]:
    if nodes is None:
        l1_grid = make_leg_length_grid(step=step)
        l2_grid = make_leg_length_grid(step=step)
        nodes = tuple(
            compute_gain_node(float(l1), float(l2))
            for l1 in l1_grid
            for l2 in l2_grid
        )
        return l1_grid, l2_grid, nodes
    nodes = tuple(nodes)
    if not nodes:
        raise ValueError("nodes must not be empty")
    l1_grid = np.unique([node.l1 for node in nodes])
    l2_grid = np.unique([node.l2 for node in nodes])
    if len(nodes) != len(l1_grid) * len(l2_grid):
        raise ValueError("nodes must form a complete L1/L2 tensor grid")
    return l1_grid, l2_grid, nodes


def compute_leso_schedule(
    step: float = 0.01,
    *,
    scheduled_ad_entries: int = 20,
    scheduled_bd_entries: int = 20,
    degree_1: int = 3,
    degree_2: int = 3,
    design: LesoDesignConfig = DEFAULT_LESO_DESIGN,
    nodes: Sequence[GainNode] | None = None,
) -> LesoScheduleFit:
    """Fit sparse physical-unit Ad/Bd schedules and validate one fixed LESO gain."""
    l1_grid, l2_grid, nodes = _make_nodes(step, nodes)
    points = np.asarray([(node.l1, node.l2) for node in nodes], dtype=float)
    physical_ad = np.asarray([node.Ad for node in nodes])
    physical_bd = np.asarray([node.Bd for node in nodes])
    if any(np.linalg.matrix_rank(matrix) < config.N_U for matrix in physical_bd):
        raise ValueError("Bd loses disturbance-channel rank inside the scheduling domain")

    reference_index = int(np.argmin(np.sum((points - np.array([0.28, 0.28])) ** 2, axis=1)))
    reference_ad = physical_ad[reference_index]
    reference_bd = physical_bd[reference_index]
    fixed_gain = design_leso_gain(
        reference_ad,
        reference_bd,
        design,
    )

    ad_schedule, fitted_ad, ad_fit_error = _fit_sparse_schedule(
        physical_ad,
        points,
        reference_ad,
        scheduled_ad_entries,
        _STATE_MIRROR,
        _STATE_MIRROR,
        degree_1,
        degree_2,
    )
    bd_schedule, fitted_bd, bd_fit_error = _fit_sparse_schedule(
        physical_bd,
        points,
        reference_bd,
        scheduled_bd_entries,
        _STATE_MIRROR,
        _INPUT_MIRROR,
        degree_1,
        degree_2,
    )

    raw_radii = np.asarray(
        [
            observer_error_spectral_radius(ad, bd, fixed_gain)
            for ad, bd in zip(physical_ad, physical_bd)
        ]
    )
    fitted_radii = np.asarray(
        [
            observer_error_spectral_radius(ad, bd, fixed_gain)
            for ad, bd in zip(fitted_ad, fitted_bd)
        ]
    )
    float_radii = []
    gain_float = fixed_gain.astype(np.float32)
    for l1, l2 in points:
        ad_float = ad_schedule.evaluate(l1, l2, dtype=np.float32)
        bd_float = bd_schedule.evaluate(l1, l2, dtype=np.float32)
        float_radii.append(
            observer_error_spectral_radius(ad_float, bd_float, gain_float)
        )
    max_float_radius = float(np.max(float_radii))
    if max_float_radius >= 1.0:
        raise ValueError("float32 sparse LESO schedule is unstable inside the grid")

    identity = np.eye(config.N_X, dtype=float)
    return LesoScheduleFit(
        l1_grid=l1_grid,
        l2_grid=l2_grid,
        ad_schedule=ad_schedule,
        bd_schedule=bd_schedule,
        fixed_gain=fixed_gain,
        design=design,
        gain_reference=tuple(float(value) for value in points[reference_index]),
        max_ad_dynamic_relative_error=_max_relative_error(
            physical_ad,
            fitted_ad,
            offset=identity,
        ),
        max_bd_relative_error=_max_relative_error(physical_bd, fitted_bd),
        max_raw_observer_spectral_radius=float(np.max(raw_radii)),
        max_fitted_observer_spectral_radius=float(np.max(fitted_radii)),
        max_float32_observer_spectral_radius=max_float_radius,
        max_selected_ad_fit_error=ad_fit_error,
        max_selected_bd_fit_error=bd_fit_error,
    )


def step_scheduled_leso(
    schedule: LesoScheduleFit,
    estimate: LesoEstimate,
    measured_state: np.ndarray,
    applied_input_delta: np.ndarray,
    l1: float,
    l2: float,
    *,
    disturbance_limits: np.ndarray | None = None,
) -> LesoStepResult:
    """Advance the observer with sparse scheduled matrix-vector products."""
    measured_state = np.asarray(measured_state, dtype=float).reshape(-1)
    applied_input_delta = np.asarray(applied_input_delta, dtype=float).reshape(-1)
    if measured_state.size != config.N_X:
        raise ValueError(f"measured_state must contain {config.N_X} values")
    if applied_input_delta.size != config.N_U:
        raise ValueError(f"applied_input_delta must contain {config.N_U} values")
    innovation = measured_state - estimate.state
    combined_input = applied_input_delta + estimate.disturbance
    next_state = (
        schedule.ad_schedule.matvec(l1, l2, estimate.state)
        + schedule.bd_schedule.matvec(l1, l2, combined_input)
        + schedule.fixed_gain[: config.N_X] @ innovation
    )
    next_disturbance = (
        estimate.disturbance
        + schedule.fixed_gain[config.N_X :] @ innovation
    )
    if disturbance_limits is not None:
        limits = np.asarray(disturbance_limits, dtype=float).reshape(-1)
        if limits.size != config.N_U or np.any(limits <= 0.0):
            raise ValueError("disturbance_limits must contain six positive values")
        next_disturbance = np.clip(
            next_disturbance,
            -limits,
            limits,
        )
    return LesoStepResult(
        estimate=LesoEstimate(next_state, next_disturbance),
        innovation=innovation,
    )


def render_leso_summary(result: LesoScheduleFit) -> str:
    lines = [
        "# Six-Disturbance LESO Sparse Schedule",
        "",
        f"- Domain: `L1 in [{result.l1_grid[0]:.2f}, {result.l1_grid[-1]:.2f}] m`, "
        f"`L2 in [{result.l2_grid[0]:.2f}, {result.l2_grid[-1]:.2f}] m`",
        f"- Grid: `{len(result.l1_grid)} x {len(result.l2_grid)}`",
        f"- Augmented dimensions: `{N_AUGMENTED} states, {config.N_X} measurements`",
        f"- Scheduled physical-unit Ad entries: `{len(result.ad_schedule.entries)}`",
        f"- Scheduled physical-unit Bd entries: `{len(result.bd_schedule.entries)}`",
        f"- Ad sparse matvec multiplies: `{result.ad_schedule.multiply_count}`",
        f"- Bd sparse matvec multiplies: `{result.bd_schedule.multiply_count}`",
        f"- Split observer total matrix-vector multiplies: "
        f"`{result.ad_schedule.multiply_count + result.bd_schedule.multiply_count + N_AUGMENTED * config.N_X}`",
        f"- Fixed LQE gain reference: `{result.gain_reference}`",
        f"- Max Ad-I relative error: `{result.max_ad_dynamic_relative_error:.6e}`",
        f"- Max Bd relative error: `{result.max_bd_relative_error:.6e}`",
        f"- Max raw observer spectral radius: `{result.max_raw_observer_spectral_radius:.9f}`",
        f"- Max fitted observer spectral radius: `{result.max_fitted_observer_spectral_radius:.9f}`",
        f"- Max float32 observer spectral radius: `{result.max_float32_observer_spectral_radius:.9f}`",
        "",
        "The measurement-noise defaults are engineering estimates and must be replaced "
        "with stationary robot log statistics before deployment.",
        "",
    ]
    return "\n".join(lines)


def save_leso_result(result: LesoScheduleFit, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text(output_dir / "LESO_README.md", render_leso_summary(result))
    np.savez_compressed(
        output_dir / "leso_schedule.npz",
        measurement_noise_std=result.design.measurement_noise_std,
        state_process_variance=result.design.state_process_variance,
        disturbance_process_variance=result.design.disturbance_process_variance,
        fixed_gain=result.fixed_gain,
        ad_baseline=result.ad_schedule.baseline,
        ad_entries=result.ad_schedule.entries,
        ad_coefficients=result.ad_schedule.coefficients,
        bd_baseline=result.bd_schedule.baseline,
        bd_entries=result.bd_schedule.entries,
        bd_coefficients=result.bd_schedule.coefficients,
    )
    report = {
        "domain": [float(result.l1_grid[0]), float(result.l1_grid[-1])],
        "grid": [len(result.l1_grid), len(result.l2_grid)],
        "gain_reference": [float(value) for value in result.gain_reference],
        "scheduled_ad_entries": len(result.ad_schedule.entries),
        "scheduled_bd_entries": len(result.bd_schedule.entries),
        "max_ad_dynamic_relative_error": result.max_ad_dynamic_relative_error,
        "max_bd_relative_error": result.max_bd_relative_error,
        "max_raw_observer_spectral_radius": result.max_raw_observer_spectral_radius,
        "max_fitted_observer_spectral_radius": result.max_fitted_observer_spectral_radius,
        "max_float32_observer_spectral_radius": result.max_float32_observer_spectral_radius,
    }
    (output_dir / "leso_schedule.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--ad-entries", type=int, default=20)
    parser.add_argument("--bd-entries", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generated/leso"),
    )
    args = parser.parse_args()
    result = compute_leso_schedule(
        step=args.step,
        scheduled_ad_entries=args.ad_entries,
        scheduled_bd_entries=args.bd_entries,
    )
    output_dir = resolve_output_dir(args.output_dir)
    save_leso_result(result, output_dir)
    print(render_leso_summary(result))


__all__ = [
    "LesoScheduleFit",
    "SparseTensorSchedule",
    "compute_leso_schedule",
    "main",
    "render_leso_summary",
    "save_leso_result",
    "step_scheduled_leso",
]


if __name__ == "__main__":
    main()
