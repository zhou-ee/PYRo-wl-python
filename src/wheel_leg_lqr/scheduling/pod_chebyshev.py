"""Dense, exchange-symmetric POD schedules with total-degree Chebyshev scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def parity_terms(total_degree: int, parity: int) -> np.ndarray:
    """Return independent Chebyshev order pairs for one exchange parity.

    Even functions use ``i <= j`` and the paired sum; odd functions use
    ``i < j`` and the paired difference.  This stores each constrained
    coefficient only once.
    """

    if total_degree < 0:
        raise ValueError("total_degree must be non-negative")
    if parity not in (-1, 1):
        raise ValueError("parity must be +1 or -1")
    terms = []
    for degree in range(total_degree + 1):
        for order_1 in range(degree + 1):
            order_2 = degree - order_1
            if parity == 1 and order_1 <= order_2:
                terms.append((order_1, order_2))
            elif parity == -1 and order_1 < order_2:
                terms.append((order_1, order_2))
    return np.asarray(terms, dtype=np.int8).reshape(-1, 2)


def _chebyshev_values(
    coordinate: float,
    degree: int,
    dtype: np.dtype | type,
) -> np.ndarray:
    dtype = np.dtype(dtype)
    cast = dtype.type
    values = np.empty(degree + 1, dtype=dtype)
    values[0] = cast(1.0)
    if degree >= 1:
        values[1] = cast(coordinate)
    for order in range(2, degree + 1):
        product = cast(values[1] * values[order - 1])
        values[order] = cast(cast(product + product) - values[order - 2])
    return values


def _basis_row(
    coordinate_1: float,
    coordinate_2: float,
    terms: np.ndarray,
    parity: int,
    dtype: np.dtype | type,
) -> np.ndarray:
    dtype = np.dtype(dtype)
    cast = dtype.type
    degree = int(np.max(terms)) if len(terms) else 0
    values_1 = _chebyshev_values(coordinate_1, degree, dtype)
    values_2 = _chebyshev_values(coordinate_2, degree, dtype)
    basis = np.empty(len(terms), dtype=dtype)
    for index, (order_1, order_2) in enumerate(terms):
        first = cast(values_1[order_1] * values_2[order_2])
        if order_1 == order_2:
            basis[index] = first
        else:
            second = cast(values_1[order_2] * values_2[order_1])
            basis[index] = cast(first + second) if parity == 1 else cast(first - second)
    return basis


def parity_design_matrix(
    points: np.ndarray,
    total_degree: int,
    parity: int,
    *,
    nominal_length: float,
    half_range: float,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Build the constrained total-degree Chebyshev design matrix."""

    if half_range <= 0.0:
        raise ValueError("half_range must be positive")
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    terms = parity_terms(total_degree, parity)
    rows = []
    for length_1, length_2 in points:
        coordinate_1 = np.clip(
            (float(length_1) - nominal_length) / half_range, -1.0, 1.0
        )
        coordinate_2 = np.clip(
            (float(length_2) - nominal_length) / half_range, -1.0, 1.0
        )
        rows.append(
            _basis_row(coordinate_1, coordinate_2, terms, parity, dtype)
        )
    return np.asarray(rows, dtype=dtype).reshape(len(points), len(terms))


def _validated_transform(transform: np.ndarray, size: int, name: str) -> np.ndarray:
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size})")
    identity = np.eye(size)
    if not np.allclose(transform.T @ transform, identity, atol=1.0e-12):
        raise ValueError(f"{name} must be orthogonal")
    if not np.allclose(transform @ transform, identity, atol=1.0e-12):
        raise ValueError(f"{name} must be an involution")
    return transform


def _project_stack(
    matrices: np.ndarray,
    row_transform: np.ndarray,
    column_transform: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "ab,nbc,dc->nad",
        row_transform,
        matrices,
        column_transform,
    )


def _symmetry_modes(
    projected_snapshots: np.ndarray,
    rank: int,
    parity: int,
    row_transform: np.ndarray,
    column_transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    flat = projected_snapshots.reshape(len(projected_snapshots), -1)
    _, singular_values, right_vectors = np.linalg.svd(flat, full_matrices=False)
    if singular_values.size and singular_values[0] > 0.0:
        threshold = (
            max(flat.shape) * np.finfo(float).eps * singular_values[0]
        )
        available_rank = int(np.count_nonzero(singular_values > threshold))
    else:
        available_rank = 0
    if rank < 0 or rank > available_rank:
        raise ValueError(
            f"requested parity-{parity:+d} POD rank {rank} exceeds "
            f"available snapshot rank {available_rank}"
        )
    if rank == 0:
        return (
            np.empty((0, *projected_snapshots.shape[1:]), dtype=float),
            singular_values,
            available_rank,
        )

    # The SVD is performed inside one symmetry eigenspace.  Project and
    # re-orthogonalize explicitly so nearly degenerate singular subspaces keep
    # definite parity and deterministic modes.
    modes: list[np.ndarray] = []
    for raw_mode in right_vectors[:rank].reshape(rank, *projected_snapshots.shape[1:]):
        mirrored = row_transform @ raw_mode @ column_transform.T
        mode = 0.5 * (raw_mode + parity * mirrored)
        for previous in modes:
            mode -= np.vdot(previous, mode).real * previous
        norm = float(np.linalg.norm(mode))
        if norm <= np.finfo(float).eps:
            raise ValueError("symmetry projection destroyed a selected POD mode")
        mode /= norm
        largest = np.unravel_index(np.argmax(np.abs(mode)), mode.shape)
        if mode[largest] < 0.0:
            mode = -mode
        modes.append(mode)
    result = np.asarray(modes, dtype=float)
    gram = result.reshape(rank, -1) @ result.reshape(rank, -1).T
    if not np.allclose(gram, np.eye(rank), atol=1.0e-10):
        raise ValueError("symmetry-separated POD modes are not orthonormal")
    return result, singular_values, available_rank


@dataclass(frozen=True)
class SymmetricPodChebyshevSchedule:
    """One complete dense matrix/block represented by parity-split POD modes."""

    mean_matrix: np.ndarray
    even_modes: np.ndarray
    odd_modes: np.ndarray
    even_coefficients: np.ndarray
    odd_coefficients: np.ndarray
    even_singular_values: np.ndarray
    odd_singular_values: np.ndarray
    retained_energy: float
    even_retained_energy: float
    odd_retained_energy: float
    even_design_condition: float
    odd_design_condition: float
    total_degree: int
    nominal_length: float
    half_range: float

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean_matrix, dtype=float).copy()
        even_modes = np.asarray(self.even_modes, dtype=float).copy()
        odd_modes = np.asarray(self.odd_modes, dtype=float).copy()
        even_coefficients = np.asarray(self.even_coefficients, dtype=float).copy()
        odd_coefficients = np.asarray(self.odd_coefficients, dtype=float).copy()
        even_singular_values = np.asarray(
            self.even_singular_values, dtype=float
        ).copy()
        odd_singular_values = np.asarray(
            self.odd_singular_values, dtype=float
        ).copy()
        if mean.ndim != 2:
            raise ValueError("mean_matrix must be two-dimensional")
        matrix_shape = mean.shape
        if even_modes.shape[1:] != matrix_shape or odd_modes.shape[1:] != matrix_shape:
            raise ValueError("all POD modes must match mean_matrix")
        expected_even = (len(even_modes), len(self.even_terms))
        expected_odd = (len(odd_modes), len(self.odd_terms))
        if even_coefficients.shape != expected_even:
            raise ValueError(f"even_coefficients must have shape {expected_even}")
        if odd_coefficients.shape != expected_odd:
            raise ValueError(f"odd_coefficients must have shape {expected_odd}")
        if not 0.0 <= self.retained_energy <= 1.0 + 1.0e-12:
            raise ValueError("retained_energy must lie in [0, 1]")
        if self.half_range <= 0.0:
            raise ValueError("half_range must be positive")
        for array in (
            mean,
            even_modes,
            odd_modes,
            even_coefficients,
            odd_coefficients,
            even_singular_values,
            odd_singular_values,
        ):
            array.setflags(write=False)
        object.__setattr__(self, "mean_matrix", mean)
        object.__setattr__(self, "even_modes", even_modes)
        object.__setattr__(self, "odd_modes", odd_modes)
        object.__setattr__(self, "even_coefficients", even_coefficients)
        object.__setattr__(self, "odd_coefficients", odd_coefficients)
        object.__setattr__(self, "even_singular_values", even_singular_values)
        object.__setattr__(self, "odd_singular_values", odd_singular_values)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.mean_matrix.shape)

    @property
    def even_rank(self) -> int:
        return int(len(self.even_modes))

    @property
    def odd_rank(self) -> int:
        return int(len(self.odd_modes))

    @property
    def rank(self) -> int:
        return self.even_rank + self.odd_rank

    @property
    def even_terms(self) -> np.ndarray:
        return parity_terms(self.total_degree, 1)

    @property
    def odd_terms(self) -> np.ndarray:
        return parity_terms(self.total_degree, -1)

    @property
    def storage_float_count(self) -> int:
        rows, columns = self.shape
        return int(
            rows * columns * (1 + self.rank)
            + self.even_coefficients.size
            + self.odd_coefficients.size
        )

    @property
    def reconstruction_multiply_count(self) -> int:
        rows, columns = self.shape
        return int(rows * columns * self.rank)

    @property
    def modal_multiply_count(self) -> int:
        return int(self.even_coefficients.size + self.odd_coefficients.size)

    def polynomial_coordinates(
        self,
        l1: float,
        l2: float,
        *,
        dtype: np.dtype | type = np.float64,
        firmware_order: bool = False,
    ) -> tuple[float, float]:
        if firmware_order:
            dtype = np.dtype(dtype)
            cast = dtype.type
            lower = cast(self.nominal_length - self.half_range)
            upper = cast(self.nominal_length + self.half_range)
            half = cast(cast(0.5) * cast(upper - lower))
            center = cast(cast(0.5) * cast(upper + lower))
            inverse_half = cast(cast(1.0) / half)

            def firmware_coordinate(length: float) -> float:
                coordinate = cast(cast(cast(length) - center) * inverse_half)
                coordinate = max(cast(-1.0), min(cast(1.0), coordinate))
                return float(coordinate)

            return firmware_coordinate(l1), firmware_coordinate(l2)

        coordinate_1 = np.clip(
            (float(l1) - self.nominal_length) / self.half_range, -1.0, 1.0
        )
        coordinate_2 = np.clip(
            (float(l2) - self.nominal_length) / self.half_range, -1.0, 1.0
        )
        return float(coordinate_1), float(coordinate_2)

    def _modal_values(
        self,
        l1: float,
        l2: float,
        *,
        dtype: np.dtype | type,
        firmware_order: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        dtype = np.dtype(dtype)
        cast = dtype.type
        coordinate_1, coordinate_2 = self.polynomial_coordinates(
            l1,
            l2,
            dtype=dtype,
            firmware_order=firmware_order,
        )

        def evaluate_part(parity: int, coefficients: np.ndarray) -> np.ndarray:
            terms = self.even_terms if parity == 1 else self.odd_terms
            basis = _basis_row(
                coordinate_1, coordinate_2, terms, parity, dtype
            )
            coefficients_target = coefficients.astype(dtype, copy=False)
            if not firmware_order:
                return coefficients_target @ basis
            values = np.zeros(len(coefficients_target), dtype=dtype)
            for mode in range(len(coefficients_target)):
                for term in range(len(basis)):
                    product = cast(coefficients_target[mode, term] * basis[term])
                    values[mode] = cast(values[mode] + product)
            return values

        return (
            evaluate_part(1, self.even_coefficients),
            evaluate_part(-1, self.odd_coefficients),
        )

    def evaluate(
        self,
        l1: float,
        l2: float,
        *,
        dtype: np.dtype | type = np.float64,
        firmware_order: bool = False,
    ) -> np.ndarray:
        """Reconstruct the dense block in host or exact firmware order."""

        dtype = np.dtype(dtype)
        cast = dtype.type
        even_values, odd_values = self._modal_values(
            l1, l2, dtype=dtype, firmware_order=firmware_order
        )
        mean = self.mean_matrix.astype(dtype, copy=False)
        even_modes = self.even_modes.astype(dtype, copy=False)
        odd_modes = self.odd_modes.astype(dtype, copy=False)
        if not firmware_order:
            return (
                mean
                + np.einsum("r,rij->ij", even_values, even_modes)
                + np.einsum("r,rij->ij", odd_values, odd_modes)
            )
        result = mean.copy()
        for row in range(self.shape[0]):
            for column in range(self.shape[1]):
                value = result[row, column]
                for mode in range(self.even_rank):
                    product = cast(even_values[mode] * even_modes[mode, row, column])
                    value = cast(value + product)
                for mode in range(self.odd_rank):
                    product = cast(odd_values[mode] * odd_modes[mode, row, column])
                    value = cast(value + product)
                result[row, column] = value
        return result


def fit_symmetric_pod_chebyshev(
    matrices: np.ndarray,
    points: np.ndarray,
    row_transform: np.ndarray,
    column_transform: np.ndarray,
    *,
    total_degree: int,
    even_rank: int,
    odd_rank: int,
    nominal_length: float,
    half_range: float,
) -> SymmetricPodChebyshevSchedule:
    """Fit one complete dense matrix/block with explicit exchange symmetry."""

    matrices = np.asarray(matrices, dtype=float)
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if matrices.ndim != 3 or len(matrices) != len(points):
        raise ValueError("matrices must be an N-by-rows-by-columns snapshot stack")
    if not np.all(np.isfinite(matrices)):
        raise ValueError("POD snapshots must contain only finite values")
    row_transform = _validated_transform(
        row_transform, matrices.shape[1], "row_transform"
    )
    column_transform = _validated_transform(
        column_transform, matrices.shape[2], "column_transform"
    )

    mean = np.mean(matrices, axis=0)
    mean = 0.5 * (mean + row_transform @ mean @ column_transform.T)
    centered = matrices - mean
    mirrored = _project_stack(centered, row_transform, column_transform)
    even_snapshots = 0.5 * (centered + mirrored)
    odd_snapshots = 0.5 * (centered - mirrored)

    even_modes, even_singular, _ = _symmetry_modes(
        even_snapshots,
        even_rank,
        1,
        row_transform,
        column_transform,
    )
    odd_modes, odd_singular, _ = _symmetry_modes(
        odd_snapshots,
        odd_rank,
        -1,
        row_transform,
        column_transform,
    )
    if even_rank and odd_rank:
        cross_gram = even_modes.reshape(even_rank, -1) @ odd_modes.reshape(
            odd_rank, -1
        ).T
        if not np.allclose(cross_gram, 0.0, atol=1.0e-10):
            raise ValueError("even and odd POD subspaces are not orthogonal")

    def fit_part(parity: int, modes: np.ndarray) -> tuple[np.ndarray, float]:
        terms = parity_terms(total_degree, parity)
        design = parity_design_matrix(
            points,
            total_degree,
            parity,
            nominal_length=nominal_length,
            half_range=half_range,
        )
        if len(points) < len(terms) or np.linalg.matrix_rank(design) < len(terms):
            raise ValueError(
                f"training grid cannot support parity-{parity:+d} "
                "total-degree Chebyshev basis"
            )
        condition = float(np.linalg.cond(design))
        if not len(modes):
            return np.empty((0, len(terms)), dtype=float), condition
        scores = np.einsum("nij,kij->nk", centered, modes)
        coefficients = np.linalg.lstsq(design, scores, rcond=None)[0].T
        return coefficients, condition

    even_coefficients, even_condition = fit_part(1, even_modes)
    odd_coefficients, odd_condition = fit_part(-1, odd_modes)
    even_energy = float(np.sum(even_singular**2))
    odd_energy = float(np.sum(odd_singular**2))
    total_energy = even_energy + odd_energy
    captured_even = float(np.sum(even_singular[:even_rank] ** 2))
    captured_odd = float(np.sum(odd_singular[:odd_rank] ** 2))
    retained = (
        (captured_even + captured_odd) / total_energy if total_energy else 1.0
    )

    return SymmetricPodChebyshevSchedule(
        mean_matrix=mean,
        even_modes=even_modes,
        odd_modes=odd_modes,
        even_coefficients=even_coefficients,
        odd_coefficients=odd_coefficients,
        even_singular_values=even_singular,
        odd_singular_values=odd_singular,
        retained_energy=float(retained),
        even_retained_energy=(captured_even / even_energy if even_energy else 1.0),
        odd_retained_energy=(captured_odd / odd_energy if odd_energy else 1.0),
        even_design_condition=even_condition,
        odd_design_condition=odd_condition,
        total_degree=total_degree,
        nominal_length=float(nominal_length),
        half_range=float(half_range),
    )


__all__ = [
    "SymmetricPodChebyshevSchedule",
    "fit_symmetric_pod_chebyshev",
    "parity_design_matrix",
    "parity_terms",
]
