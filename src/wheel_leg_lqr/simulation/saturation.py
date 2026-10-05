"""Dimension-independent saturated perturbation-model simulation."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SaturationValidation:
    passed: bool
    max_state_abs: float
    max_final_state_abs: float
    max_commanded_input_abs: float
    saturation_steps: int
    worst_case: str

def simulate_saturated_closed_loop_batch(Ad, Bd, gain, equilibrium_input, initial_errors, *,
                                        sample_time, lower_limits, upper_limits, horizon_s=5.0):
    Ad, Bd, gain = (np.asarray(v, dtype=float) for v in (Ad, Bd, gain))
    if Ad.ndim != 2 or Ad.shape[0] != Ad.shape[1] or not len(Ad):
        raise ValueError('Ad must be a nonempty square matrix')
    n = Ad.shape[0]
    if Bd.ndim != 2 or Bd.shape[0] != n or not Bd.shape[1]:
        raise ValueError('Bd must have shape (states, inputs)')
    m = Bd.shape[1]
    equilibrium_input = np.asarray(equilibrium_input, dtype=float).reshape(-1)
    errors = np.atleast_2d(np.asarray(initial_errors, dtype=float)).copy()
    lower_limits, upper_limits = (np.asarray(v, dtype=float).reshape(-1) for v in (lower_limits, upper_limits))
    if gain.shape != (m, n) or equilibrium_input.shape != (m,) or errors.ndim != 2 or errors.shape[1] != n or len(errors) == 0:
        raise ValueError('gain, equilibrium_input or initial_errors dimension mismatch')
    if lower_limits.shape != (m,) or upper_limits.shape != (m,) or np.any(lower_limits > upper_limits):
        raise ValueError('input limits must contain ordered per-input bounds')
    if not all(np.all(np.isfinite(v)) for v in (Ad, Bd, gain, equilibrium_input, errors, lower_limits, upper_limits)):
        raise ValueError('simulation inputs must be finite')
    if not np.isfinite(sample_time) or sample_time <= 0 or not np.isfinite(horizon_s) or horizon_s <= 0:
        raise ValueError('sample_time and horizon_s must be positive and finite')
    steps = max(1, int(round(horizon_s / sample_time)))
    states = np.empty((len(errors), steps + 1, n))
    commands = np.empty((len(errors), steps, m))
    counts = np.zeros(len(errors), dtype=int)
    states[:, 0] = errors
    for index in range(steps):
        command = equilibrium_input - errors @ gain.T
        saturated = np.clip(command, lower_limits, upper_limits)
        counts += np.any(command != saturated, axis=1)
        commands[:, index] = saturated
        errors = errors @ Ad.T + (saturated - equilibrium_input) @ Bd.T
        states[:, index + 1] = errors
    return states, commands, counts

def simulate_saturated_closed_loop(Ad, Bd, gain, equilibrium_input, initial_error, **kwargs):
    states, commands, counts = simulate_saturated_closed_loop_batch(
        Ad, Bd, gain, equilibrium_input, initial_error, **kwargs)
    return states[0], commands[0], int(counts[0])

def validate_saturated_closed_loop(Ad, Bd, gain, equilibrium_input, *, cases, sample_time,
                                 lower_limits, upper_limits, horizon_s=5.0,
                                 max_state_abs_limit=10.0, final_state_abs_limit=0.01):
    equilibrium_input = np.asarray(equilibrium_input)
    if np.any(equilibrium_input < lower_limits) or np.any(equilibrium_input > upper_limits):
        return SaturationValidation(False, float('inf'), float('inf'), float('inf'), 0,
                                    'equilibrium_input_out_of_limits')
    states, commands, counts = simulate_saturated_closed_loop_batch(
        Ad, Bd, gain, equilibrium_input, np.array([c[1] for c in cases]), sample_time=sample_time,
        lower_limits=lower_limits, upper_limits=upper_limits, horizon_s=horizon_s)
    maxima = np.max(np.abs(states), axis=(1, 2))
    final = np.max(np.abs(states[:, -1]), axis=1)
    valid = (np.all(np.isfinite(states), axis=(1, 2)) & np.all(np.isfinite(commands), axis=(1, 2))
             & (maxima <= max_state_abs_limit) & (final <= final_state_abs_limit))
    worst = int(np.flatnonzero(~valid)[0]) if not np.all(valid) else int(np.argmax(maxima))
    return SaturationValidation(bool(np.all(valid)), float(np.max(maxima)), float(np.max(final)),
                                float(np.max(np.abs(commands))), int(np.sum(counts)), cases[worst][0])
