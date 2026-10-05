from __future__ import annotations
from ..config import DEFAULT_ROBOT, RobotParameters
from dataclasses import dataclass
import numpy as np
from ..control.lqr import discretize_state_space, solve_discrete_lqr
from ..control.rdob import initialize_rdob, reconstruct_disturbance, step_rdob, trapezoidal_step_constants
from ..models.ground import config
from ..scheduling.rdob_node import DEFAULT_BANDWIDTH_HZ, RdobNode, compute_rdob_node
from .config import DEFAULT_DISTURBANCE_LIMIT_RATIO

_CONFIGURED_DISTURBANCE_LIMITS = object()

GAMMA_TERMS = tuple((i, j) for i in range(4) for j in range(4))
# Row-major by x power, then ascending y power.  This is the exported firmware
# contract and permits direct nested-loop or Horner evaluation without a term
# index table.
TOTAL_CUBIC_TERMS = tuple((i, j) for i in range(4) for j in range(4-i))
POSITION_ACTIVE_COLUMNS = slice(2, 7)


def _normalize(l1, l2, lower, upper, dtype):
    cast = np.dtype(dtype).type
    low, high = cast(lower), cast(upper)
    center = cast(cast(0.5) * cast(low + high))
    inverse_half = cast(cast(1.0) / cast(cast(0.5) * cast(high - low)))
    x1 = cast(cast(cast(l1) - center) * inverse_half)
    x2 = cast(cast(cast(l2) - center) * inverse_half)
    return cast(np.clip(x1, -1.0, 1.0)), cast(np.clip(x2, -1.0, 1.0))


def _design(points, terms, lower, upper):
    return np.asarray(
        [[x1**i * x2**j for i, j in terms]
         for x1, x2 in (_normalize(*point, lower, upper, np.float64) for point in points)],
        dtype=float,
    )


def _metrics(actual, predicted):
    difference = predicted - actual
    actual_flat, difference_flat = actual.reshape(len(actual), -1), difference.reshape(len(actual), -1)
    relative = np.linalg.norm(difference_flat, axis=1) / np.maximum(np.linalg.norm(actual_flat, axis=1), 1e-12)
    return {"max_relative_frobenius": float(np.max(relative)),
            "rms_relative_frobenius": float(np.sqrt(np.mean(relative**2))),
            "max_absolute": float(np.max(np.abs(difference)))}


@dataclass(frozen=True)
class DirectPolynomialFit:
    kind: str
    terms: tuple[tuple[int, int], ...]
    coefficients: np.ndarray
    output_shape: tuple[int, ...]
    metrics: dict

    @property
    def coefficient_count(self):
        return int(self.coefficients.size)

    def evaluate_normalized(self, x1, x2, dtype=np.float64):
        coefficients = self.coefficients.astype(dtype)
        if self.kind == "tensor_cubic":
            c = coefficients.reshape(-1, 4, 4)
            rows = []
            for i in range(4):
                value = c[:, i, 3].copy()
                for j in (2, 1, 0):
                    value = (value*x2 + c[:, i, j]).astype(dtype)
                rows.append(value)
            value = rows[3].copy()
            for i in (2, 1, 0):
                value = (value*x1 + rows[i]).astype(dtype)
            return value.reshape(self.output_shape)
        if self.kind == "total_cubic":
            c = coefficients.reshape(-1, 10)
            row0 = ((c[:, 3]*x2+c[:, 2])*x2+c[:, 1])*x2+c[:, 0]
            row1 = (c[:, 6]*x2+c[:, 5])*x2+c[:, 4]
            row2 = c[:, 8]*x2+c[:, 7]
            return (((c[:, 9]*x1+row2)*x1+row1)*x1+row0).astype(dtype).reshape(self.output_shape)
        raise ValueError(f"unsupported direct polynomial kind: {self.kind}")


def _fit_direct(values, train_count, points, lower, upper, *, kind, terms):
    values = np.asarray(values, dtype=float)
    coefficients = np.linalg.lstsq(_design(points, terms, lower, upper)[:train_count],
                                   values[:train_count].reshape(train_count, -1), rcond=None)[0].T
    fit = DirectPolynomialFit(kind, terms, coefficients, values.shape[1:], {})
    prediction = np.asarray([
        fit.evaluate_normalized(*_normalize(*point, lower, upper, np.float32), np.float32)
        for point in points
    ])
    fit.metrics.update(training=_metrics(values[:train_count], prediction[:train_count]),
                       holdout=_metrics(values[train_count:], prediction[train_count:]),
                       float32_all=_metrics(values, prediction))
    return fit


@dataclass(frozen=True)
class RdobSchedulePoint:
    """Runtime coefficients; LQR supplies q0 and total u0 separately."""
    rates: np.ndarray
    gamma: np.ndarray
    position_gain: np.ndarray
    velocity_gain: np.ndarray
    sample_time: float
    step_poles: np.ndarray
    step_position_gain: np.ndarray
    step_velocity_gain: np.ndarray
    step_input_gain: np.ndarray


@dataclass(frozen=True)
class RdobSchedule:
    lower: float
    upper: float
    rates: np.ndarray
    sample_time: float
    damping: np.ndarray
    training_points: np.ndarray
    holdout_points: np.ndarray
    fits: dict[str, DirectPolynomialFit]
    diagnostics: dict
    parameters: RobotParameters = DEFAULT_ROBOT

    def evaluate(self, l1, l2, *, dtype=np.float64):
        for length in (l1, l2):
            if not np.isfinite(length) or not self.lower <= length <= self.upper:
                raise ValueError(f"L1 and L2 must both lie in [{self.lower}, {self.upper}]")
        x1, x2 = _normalize(l1, l2, self.lower, self.upper, dtype)
        gamma = self.fits["gamma"].evaluate_normalized(x1, x2, dtype)
        position = np.zeros((6, 7), dtype=dtype)
        position[:, POSITION_ACTIVE_COLUMNS] = self.fits["position_gain"].evaluate_normalized(x1, x2, dtype)
        damping = (self.fits["damping_gain"].evaluate_normalized(x1, x2, dtype)
                   if "damping_gain" in self.fits else np.zeros((6, 7), dtype=dtype))
        rates, dt = self.rates.astype(dtype), np.dtype(dtype).type(self.sample_time)
        velocity = (damping-rates[:, None]*gamma).astype(dtype)
        poles, weight, input_gain, velocity_scale = trapezoidal_step_constants(
            rates, dt, dtype=dtype,
        )
        step_velocity = (
            weight[:, None]*damping+velocity_scale[:, None]*gamma
            if "damping_gain" in self.fits
            else velocity_scale[:, None]*gamma
        ).astype(dtype)
        return RdobSchedulePoint(rates, gamma, position, velocity, float(dt), poles,
                                 (weight[:, None]*position).astype(dtype),
                                 step_velocity, input_gain)


def _snapshot(node: RdobNode):
    observer = node.observer
    return {"gamma": observer.gamma, "position_gain": observer.position_gain,
            "damping_gain": observer.rates[:, None]*(observer.input_left_inverse @ observer.damping)}


def frozen_closed_loop_matrix(ad, bd, gain, observer, compensation):
    velocity_map = np.eye(14)[1::2]
    feedback = -gain-compensation @ observer.gamma @ velocity_map
    xp, xz = ad+bd @ feedback, -bd @ compensation
    endpoint_gain = np.empty((6, 14))
    endpoint_gain[:, ::2], endpoint_gain[:, 1::2] = observer.step_position_gain, observer.step_velocity_gain
    input_gain = np.diag(observer.step_input_gain)
    return np.block([[xp, xz],
                     [endpoint_gain @ (np.eye(14)+xp)+input_gain @ feedback,
                      np.diag(observer.step_poles)+endpoint_gain @ xz-input_gain @ compensation]])


def compute_rdob_schedule(*, lower=None, upper=None,
                          step=.02, relative_tolerance=.005, random_holdouts=24,
                          matched_rate_tolerance=.01, bandwidth_hz=DEFAULT_BANDWIDTH_HZ,
                          sample_time=config.SAMPLE_TIME, damping=None, parameters: RobotParameters = DEFAULT_ROBOT):
    """Fit the fixed direct-polynomial blocks over the full L1/L2 square."""
    lower = parameters.min_leg_length if lower is None else lower
    upper = parameters.max_leg_length if upper is None else upper
    if not all(np.isfinite(value) for value in (lower, upper, step, relative_tolerance, matched_rate_tolerance)):
        raise ValueError("schedule bounds and tolerances must be finite")
    lower = parameters.min_leg_length if lower is None else lower
    upper = parameters.max_leg_length if upper is None else upper
    if not parameters.min_leg_length <= lower < upper <= parameters.max_leg_length:
        raise ValueError("schedule bounds must lie inside the configured leg-length domain")
    if step <= 0 or relative_tolerance <= 0 or matched_rate_tolerance <= 0:
        raise ValueError("step and tolerances must be positive")
    count = int(round((upper-lower)/step))
    if count < 1 or not np.isclose(lower+count*step, upper, rtol=0., atol=1e-12):
        raise ValueError("step must divide the leg-length interval exactly")
    axis = np.linspace(lower, upper, count+1)
    train = np.asarray([(a, b) for a in axis for b in axis])
    mids = .5*(axis[:-1]+axis[1:])
    holdout = [(a, b) for a in mids for b in mids]
    holdout += [(a, b) for a in (lower, upper) for b in mids]
    holdout += [(a, b) for a in mids for b in (lower, upper)]
    holdout += np.random.default_rng(610).uniform(lower, upper, (random_holdouts, 2)).tolist()
    holdout = np.asarray(holdout)
    points = np.concatenate((train, holdout))
    nodes = [compute_rdob_node(*point, parameters=parameters, bandwidth_hz=bandwidth_hz, sample_time=sample_time, damping=damping)
             for point in points]
    reference = nodes[0].observer
    snapshots = [_snapshot(node) for node in nodes]
    for node, snapshot in zip(nodes, snapshots):
        for name in ("rates", "step_poles", "step_input_gain"):
            if not np.array_equal(getattr(node.observer, name), getattr(reference, name)):
                raise ValueError(f"fixed RDOB structure violated: {name}")
        if np.all(reference.damping == 0) and np.any(snapshot["damping_gain"] != 0):
            raise ValueError("zero damping structure violated")
    gamma = np.asarray([snapshot["gamma"] for snapshot in snapshots])
    position = np.asarray([snapshot["position_gain"][:, POSITION_ACTIVE_COLUMNS] for snapshot in snapshots])
    fits = {"gamma": _fit_direct(gamma, len(train), points, lower, upper, kind="tensor_cubic", terms=GAMMA_TERMS),
            "position_gain": _fit_direct(position, len(train), points, lower, upper, kind="total_cubic", terms=TOTAL_CUBIC_TERMS)}
    if not np.all(reference.damping == 0):
        values = np.asarray([snapshot["damping_gain"] for snapshot in snapshots])
        fits["damping_gain"] = _fit_direct(values, len(train), points, lower, upper, kind="total_cubic", terms=TOTAL_CUBIC_TERMS)
    for name, fit in fits.items():
        if fit.metrics["float32_all"]["max_relative_frobenius"] > relative_tolerance:
            raise ValueError(f"{name} direct polynomial exceeds matrix tolerance {relative_tolerance:g}")
    schedule = RdobSchedule(float(lower), float(upper), reference.rates, reference.sample_time,
                            reference.damping, train, holdout, fits, {}, parameters=parameters)
    metrics = _validate_schedule(schedule, points, nodes, matched_rate_tolerance)
    schedule.diagnostics.update(schedule_family="Gamma direct tensor cubic; Bq direct total-degree cubic; trims supplied by LQR",
                                relative_fit_budget=relative_tolerance, matched_rate_budget=matched_rate_tolerance,
                                error_definition="maximum per-point relative Frobenius norm; not elementwise percentage",
                                training_count=len(train), holdout_count=len(holdout),
                                gamma_terms=[list(term) for term in GAMMA_TERMS],
                                position_gain_terms=[list(term) for term in TOTAL_CUBIC_TERMS],
                                compensation_checks=["Tw+FL full gain", "all six full gain"], **metrics)
    return schedule


def _validate_schedule(schedule, points, nodes, matched_rate_tolerance):
    reference = nodes[0].observer
    metrics = {"max_matched_rate_relative_error": 0., "max_raw_closed_loop_radius": 0., "max_fitted_closed_loop_radius": 0.}
    for point, node in zip(points, nodes):
        fitted = schedule.evaluate(*point, dtype=np.float32)
        effective = fitted.gamma @ np.linalg.solve(node.observer.mass, node.observer.input_matrix)
        error = np.linalg.norm(effective-np.diag(reference.rates))/np.linalg.norm(reference.rates)
        metrics["max_matched_rate_relative_error"] = max(metrics["max_matched_rate_relative_error"], float(error))
        a = node.model.A.copy()
        a[1::2, 1::2] = -np.linalg.solve(node.observer.mass, node.observer.damping)
        ad, bd = discretize_state_space(a, node.model.B, sample_time=schedule.sample_time)
        gain, _, _ = solve_discrete_lqr(ad, bd, config.make_q(), config.make_r())
        for compensation in (np.diag([1., 1., 0., 0., 1., 1.]), np.eye(6)):
            for key, observer in (("max_raw_closed_loop_radius", node.observer), ("max_fitted_closed_loop_radius", fitted)):
                radius = float(np.max(np.abs(np.linalg.eigvals(frozen_closed_loop_matrix(ad, bd, gain, observer, compensation)))))
                metrics[key] = max(metrics[key], radius)
    if metrics["max_matched_rate_relative_error"] > matched_rate_tolerance:
        raise ValueError(f"fitted RDOB matched-rate error {metrics['max_matched_rate_relative_error']:.6g} exceeds {matched_rate_tolerance:g}")
    if max(metrics["max_raw_closed_loop_radius"], metrics["max_fitted_closed_loop_radius"]) >= 1.:
        raise ValueError("RDOB failed frozen compensated closed-loop stability validation")
    return metrics


def step_scheduled_rdob(
    schedule, z, previous_lengths, next_lengths, previous_position,
    previous_velocity, applied_total_input, next_position, next_velocity,
    previous_trim_position, previous_trim_total_input, *,
    previous_trim_velocity=None, next_trim_velocity=None,
    disturbance_limits=_CONFIGURED_DISTURBANCE_LIMITS,
):
    """Advance with the old trim, limit d_hat, then rebase to the new trim.

    The deployment-reference default limits each disturbance channel to the
    configured fraction of its actuator limit. Pass ``disturbance_limits=None``
    for the unsaturated mathematical observer. Omitting trim velocities preserves
    the static-trim behavior of the original interface.
    """
    previous, following = schedule.evaluate(*previous_lengths), schedule.evaluate(*next_lengths)
    old_trim_velocity = np.zeros(config.N_Q) if previous_trim_velocity is None else np.asarray(
        previous_trim_velocity, dtype=float,
    )
    new_trim_velocity = old_trim_velocity if next_trim_velocity is None else np.asarray(
        next_trim_velocity, dtype=float,
    )
    previous_velocity = np.asarray(previous_velocity, dtype=float)
    next_velocity = np.asarray(next_velocity, dtype=float)
    for name, value in (
        ("previous_velocity", previous_velocity),
        ("next_velocity", next_velocity),
        ("previous_trim_velocity", old_trim_velocity),
        ("next_trim_velocity", new_trim_velocity),
    ):
        if value.shape != (config.N_Q,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain {config.N_Q} finite values")
    next_z, estimate = step_rdob(previous, z,
                                 np.asarray(previous_position)-np.asarray(previous_trim_position),
                                 previous_velocity-old_trim_velocity,
                                 np.asarray(applied_total_input)-np.asarray(previous_trim_total_input),
                                 np.asarray(next_position)-np.asarray(previous_trim_position),
                                 next_velocity-old_trim_velocity)
    if disturbance_limits is _CONFIGURED_DISTURBANCE_LIMITS:
        disturbance_limits = DEFAULT_DISTURBANCE_LIMIT_RATIO * schedule.parameters.input_upper_limits
    if disturbance_limits is not None:
        limits = np.asarray(disturbance_limits, dtype=float)
        if limits.shape != (config.N_U,) or not np.all(np.isfinite(limits)) or np.any(limits <= 0):
            raise ValueError(f"disturbance_limits must contain {config.N_U} positive finite values")
        estimate = np.clip(estimate, -limits, limits)
    next_delta_velocity = next_velocity-new_trim_velocity
    rebased = initialize_rdob(following, next_delta_velocity, estimate)
    return rebased, reconstruct_disturbance(following, rebased, next_delta_velocity)


def _zero_damping(schedule):
    return np.all(schedule.damping == 0) and "damping_gain" not in schedule.fits


def runtime_accounting(schedule):
    damping = schedule.fits.get("damping_gain")
    damping_mult = 0 if damping is None else 42*9
    # Three scalars (bounds and sample time), plus five six-channel arrays:
    # rates, z poles, endpoint weights, input gains, and velocity scales.
    fixed_scalar_count = 3+5*config.N_U
    stored = (schedule.fits["gamma"].coefficient_count
              + schedule.fits["position_gain"].coefficient_count
              + (0 if damping is None else damping.coefficient_count)
              + fixed_scalar_count)
    scheduled_matrix_float_count = 42+30+(0 if damping is None else 42)
    return {"stored_float_count": stored, "float32_storage_bytes": 4*stored,
            "evaluation_multiply_upper_bound": 4+42*15+30*9+damping_mult,
            "evaluation_add_subtract_upper_bound": 4+42*15+30*9+damping_mult,
            "evaluation_divide_upper_bound": 1, "evaluation_named_array_scratch_bytes": 0,
            "zero_damping_storage_and_evaluation_eliminated": damping is None,
            "generated_runtime_struct_bytes": 0,
            "runtime_scheduled_matrix_float_bytes": 4*scheduled_matrix_float_count,
            "rdob_dynamic_state_bytes": 24,
            "step_and_reconstruct_multiplications": 12+30+6+42+6+42,
            "step_and_reconstruct_add_subtractions": 6+30+42+14+42,
            "coefficient_rebase_multiplications": 42, "coefficient_rebase_add_subtractions": 42,
            "note": "Data-only header; source-level upper bounds assume raw Gamma/Bq matrices, row-wise endpoint scaling, and caller-owned loops."}
