"""Actuator-saturated closed-loop checks for exported LQR gains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import config


@dataclass(frozen=True)
class SaturationValidation:
    """Summary of one gain's bounded-input recovery tests."""

    passed: bool
    max_state_abs: float
    max_final_state_abs: float
    max_commanded_input_abs: float
    saturation_steps: int
    worst_case: str


def default_initial_errors() -> tuple[tuple[str, np.ndarray], ...]:
    """Return a small operational envelope that exercises actuator limits."""
    cases: list[tuple[str, np.ndarray]] = []

    symmetric_swing = np.zeros(config.N_X, dtype=float)
    symmetric_swing[10] = 0.5
    symmetric_swing[12] = 0.5
    cases.append(("symmetric_leg_swing", symmetric_swing))

    differential_swing = np.zeros(config.N_X, dtype=float)
    differential_swing[10] = 0.5
    differential_swing[12] = -0.5
    cases.append(("differential_leg_swing", differential_swing))

    combined = np.zeros(config.N_X, dtype=float)
    combined[0] = 0.10
    combined[4] = np.deg2rad(5.0)
    combined[6] = np.deg2rad(5.0)
    combined[8] = 0.02
    combined[10] = 0.25
    combined[12] = -0.25
    cases.append(("combined_body_and_leg_error", combined))
    return tuple(cases)


def simulate_saturated_closed_loop(
    Ad: np.ndarray,
    Bd: np.ndarray,
    gain: np.ndarray,
    equilibrium_input: np.ndarray,
    initial_error: np.ndarray,
    *,
    horizon_s: float = 5.0,
    sample_time: float = config.SAMPLE_TIME,
    lower_limits: np.ndarray = config.INPUT_LOWER_LIMITS,
    upper_limits: np.ndarray = config.INPUT_UPPER_LIMITS,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Simulate ``e[k+1] = Ad e[k] + Bd (u_sat - u0)``.

    The linear model is a perturbation model around the equilibrium input, so
    the saturated absolute command must be converted back to an input delta
    before applying ``Bd``.
    """
    Ad = np.asarray(Ad, dtype=float)
    Bd = np.asarray(Bd, dtype=float)
    gain = np.asarray(gain, dtype=float)
    equilibrium_input = np.asarray(equilibrium_input, dtype=float).reshape(-1)
    error = np.asarray(initial_error, dtype=float).reshape(-1).copy()
    lower_limits = np.asarray(lower_limits, dtype=float).reshape(-1)
    upper_limits = np.asarray(upper_limits, dtype=float).reshape(-1)

    if Ad.shape != (config.N_X, config.N_X):
        raise ValueError(f"Ad must have shape ({config.N_X}, {config.N_X})")
    if Bd.shape != (config.N_X, config.N_U):
        raise ValueError(f"Bd must have shape ({config.N_X}, {config.N_U})")
    if gain.shape != (config.N_U, config.N_X):
        raise ValueError(f"gain must have shape ({config.N_U}, {config.N_X})")
    if equilibrium_input.size != config.N_U:
        raise ValueError(f"equilibrium_input must have {config.N_U} values")
    if error.size != config.N_X:
        raise ValueError(f"initial_error must have {config.N_X} values")
    if (
        lower_limits.size != config.N_U
        or upper_limits.size != config.N_U
        or np.any(lower_limits > upper_limits)
    ):
        raise ValueError("input limits must contain ordered six-channel bounds")

    steps = max(1, int(round(float(horizon_s) / float(sample_time))))
    states = np.empty((steps + 1, config.N_X), dtype=float)
    commands = np.empty((steps, config.N_U), dtype=float)
    states[0] = error
    saturation_steps = 0

    for index in range(steps):
        command = equilibrium_input - gain @ error
        saturated_command = np.clip(command, lower_limits, upper_limits)
        if np.any(command < lower_limits) or np.any(command > upper_limits):
            saturation_steps += 1
        commands[index] = saturated_command
        error = Ad @ error + Bd @ (saturated_command - equilibrium_input)
        states[index + 1] = error

    return states, commands, saturation_steps


def simulate_saturated_closed_loop_batch(
    Ad: np.ndarray,
    Bd: np.ndarray,
    gain: np.ndarray,
    equilibrium_input: np.ndarray,
    initial_errors: np.ndarray,
    *,
    horizon_s: float = 5.0,
    sample_time: float = config.SAMPLE_TIME,
    lower_limits: np.ndarray = config.INPUT_LOWER_LIMITS,
    upper_limits: np.ndarray = config.INPUT_UPPER_LIMITS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate several initial errors together using batched matrix products."""
    Ad = np.asarray(Ad, dtype=float)
    Bd = np.asarray(Bd, dtype=float)
    gain = np.asarray(gain, dtype=float)
    equilibrium_input = np.asarray(equilibrium_input, dtype=float).reshape(-1)
    errors = np.asarray(initial_errors, dtype=float)
    if errors.ndim == 1:
        errors = errors.reshape(1, -1)
    if errors.ndim != 2 or errors.shape[1] != config.N_X:
        raise ValueError(f"initial_errors must have shape (n, {config.N_X})")
    lower_limits = np.asarray(lower_limits, dtype=float).reshape(-1)
    upper_limits = np.asarray(upper_limits, dtype=float).reshape(-1)
    if Ad.shape != (config.N_X, config.N_X):
        raise ValueError(f"Ad must have shape ({config.N_X}, {config.N_X})")
    if Bd.shape != (config.N_X, config.N_U):
        raise ValueError(f"Bd must have shape ({config.N_X}, {config.N_U})")
    if gain.shape != (config.N_U, config.N_X):
        raise ValueError(f"gain must have shape ({config.N_U}, {config.N_X})")
    if equilibrium_input.size != config.N_U:
        raise ValueError(f"equilibrium_input must have {config.N_U} values")
    if (
        lower_limits.size != config.N_U
        or upper_limits.size != config.N_U
        or np.any(lower_limits > upper_limits)
    ):
        raise ValueError("input limits must contain ordered six-channel bounds")

    steps = max(1, int(round(float(horizon_s) / float(sample_time))))
    states = np.empty((errors.shape[0], steps + 1, config.N_X), dtype=float)
    commands = np.empty((errors.shape[0], steps, config.N_U), dtype=float)
    states[:, 0] = errors
    saturation_steps = np.zeros(errors.shape[0], dtype=int)
    for index in range(steps):
        command = equilibrium_input - errors @ gain.T
        saturated_command = np.clip(command, lower_limits, upper_limits)
        saturation_steps += np.any(command != saturated_command, axis=1)
        commands[:, index] = saturated_command
        errors = errors @ Ad.T + (saturated_command - equilibrium_input) @ Bd.T
        states[:, index + 1] = errors
    return states, commands, saturation_steps


def validate_saturated_closed_loop(
    Ad: np.ndarray,
    Bd: np.ndarray,
    gain: np.ndarray,
    equilibrium_input: np.ndarray,
    *,
    cases: tuple[tuple[str, np.ndarray], ...] | None = None,
    horizon_s: float = 5.0,
    max_state_abs_limit: float = 10.0,
    final_state_abs_limit: float = 0.01,
) -> SaturationValidation:
    """Check bounded-input recovery for the standard operating envelope."""
    equilibrium_input = np.asarray(equilibrium_input, dtype=float).reshape(-1)
    if np.any(equilibrium_input < config.INPUT_LOWER_LIMITS) or np.any(
        equilibrium_input > config.INPUT_UPPER_LIMITS
    ):
        return SaturationValidation(
            passed=False,
            max_state_abs=float("inf"),
            max_final_state_abs=float("inf"),
            max_commanded_input_abs=float("inf"),
            saturation_steps=0,
            worst_case="equilibrium_input_out_of_limits",
        )

    if cases is None:
        cases = default_initial_errors()

    max_state_abs = 0.0
    max_final_state_abs = 0.0
    max_commanded_input_abs = 0.0
    total_saturation_steps = 0
    worst_case = "none"
    passed = True

    case_names = tuple(case[0] for case in cases)
    initial_errors = np.asarray([case[1] for case in cases], dtype=float)
    states, commands, saturation_steps = simulate_saturated_closed_loop_batch(
        Ad,
        Bd,
        gain,
        equilibrium_input,
        initial_errors,
        horizon_s=horizon_s,
    )
    for case_index, case_name in enumerate(case_names):
        case_states = states[case_index]
        case_commands = commands[case_index]
        case_max_state = float(np.max(np.abs(case_states)))
        case_final_state = float(np.max(np.abs(case_states[-1])))
        case_max_command = float(np.max(np.abs(case_commands)))
        previous_max_state = max_state_abs
        max_state_abs = max(max_state_abs, case_max_state)
        max_final_state_abs = max(max_final_state_abs, case_final_state)
        max_commanded_input_abs = max(max_commanded_input_abs, case_max_command)
        total_saturation_steps += int(saturation_steps[case_index])

        case_passed = bool(
            np.all(np.isfinite(case_states))
            and np.all(np.isfinite(case_commands))
            and case_max_state <= max_state_abs_limit
            and case_final_state <= final_state_abs_limit
        )
        if not case_passed:
            passed = False
            if worst_case == "none":
                worst_case = case_name
        elif passed and case_max_state > previous_max_state:
            worst_case = case_name

    return SaturationValidation(
        passed=passed,
        max_state_abs=max_state_abs,
        max_final_state_abs=max_final_state_abs,
        max_commanded_input_abs=max_commanded_input_abs,
        saturation_steps=total_saturation_steps,
        worst_case=worst_case,
    )
