from __future__ import annotations
from ..models.contact.config import DEFAULT_CONTACT
from ..models.ground.trim import compute_equilibrium_beta_from_length
from ..config import DEFAULT_ROBOT, RobotParameters
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq
from ..control.rdob import initialize_rdob, reconstruct_disturbance, step_rdob
from ..models.ground import config
from ..scheduling.rdob_node import DEFAULT_BANDWIDTH_HZ, compute_rdob_node
from ..scheduling.ground_lqr import compute_gain_node
from ..models.contact.model import N_FREE_Q, Q7, _WHEEL_1, _WHEEL_2, _kinematics, _compliant_acceleration


@dataclass(frozen=True)
class StepDropResult:
    """Metrics and traces from one release-impact-recovery screening run."""

    step_height: float
    down_leg: int
    target_lengths: np.ndarray
    target_kappa: float
    target_midpoint_height: float
    time: np.ndarray
    state_error: np.ndarray
    leg_lengths: np.ndarray
    commands: np.ndarray
    disturbance_estimate: np.ndarray
    contact_forces: np.ndarray
    mechanical_stop_forces: np.ndarray
    impact_time: float | None
    impact_speed: float | None
    impact_impulse: float | None
    mechanical_stop_impulse: np.ndarray
    peak_contact_force: float
    peak_mechanical_stop_force: float
    saturation_steps: int
    settling_time: float | None
    converged: bool
    terminated_time: float
    failure_reason: str | None


def target_lengths_for_step(
    step_height: float,
    *,
    mean_leg_length: float | None = None,
    down_leg: int = _WHEEL_1,
    parameters: RobotParameters = DEFAULT_ROBOT,
) -> np.ndarray:
    """Solve leg lengths that retain the mean on a two-level support."""
    if mean_leg_length is None:
        mean_leg_length = (parameters.min_leg_length + parameters.max_leg_length) / 2
    if down_leg not in (_WHEEL_1, _WHEEL_2):
        raise ValueError("down_leg must be 0 (leg 1) or 1 (leg 2)")
    if not np.isfinite(step_height) or step_height < 0.0:
        raise ValueError("step_height must be finite and non-negative")
    lower, upper = parameters.min_leg_length, parameters.max_leg_length
    lo = max(lower, 2.0 * mean_leg_length - upper)
    hi = min(upper, 2.0 * mean_leg_length - lower)
    if not lo <= mean_leg_length <= hi:
        raise ValueError("mean_leg_length is incompatible with the configured leg limits")

    def differential(longer: float) -> float:
        shorter = 2.0 * mean_leg_length - longer
        return (
            longer * np.cos(compute_equilibrium_beta_from_length(longer, parameters))
            - shorter * np.cos(compute_equilibrium_beta_from_length(shorter, parameters))
        )

    maximum = differential(hi)
    if step_height > maximum + 1e-12:
        raise ValueError(
            f"step_height {step_height:.6g} m exceeds the {maximum:.6g} m "
            "available at this mean leg length"
        )
    longer = mean_leg_length if step_height == 0.0 else brentq(
        lambda length: differential(length) - step_height, mean_leg_length, hi
    )
    shorter = 2.0 * mean_leg_length - longer
    return np.array(
        [longer, shorter] if down_leg == _WHEEL_1 else [shorter, longer]
    )


def _interleaved_error(position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    state = np.empty(config.N_X, dtype=float)
    state[::2], state[1::2] = position, velocity
    return state


def _inside_numerical_envelope(q: np.ndarray, velocity: np.ndarray) -> bool:
    """Reject a runaway before it poisons the finite-difference dynamics."""
    return bool(
        np.all(np.isfinite(q))
        and np.all(np.isfinite(velocity))
        and np.max(np.abs(q)) < 10.0
        and np.max(np.abs(velocity)) < 1e4
    )


def simulate_step_drop(
    *,
    step_height: float = 0.18,
    down_leg: int = _WHEEL_1,
    parameters: RobotParameters = DEFAULT_ROBOT,
    mean_leg_length: float | None = None,
    horizon_s: float = 3.0,
    compensation: np.ndarray | None = None,
    bandwidth_hz: np.ndarray = DEFAULT_BANDWIDTH_HZ,
    settling_tolerance: float = 1e-3,
    integration_substeps: int = DEFAULT_CONTACT.integration_substeps,
    ground_stiffness: float = DEFAULT_CONTACT.ground_stiffness,
    ground_damping: float = DEFAULT_CONTACT.ground_damping,
    stop_stiffness: float = DEFAULT_CONTACT.stop_stiffness,
    stop_damping: float = DEFAULT_CONTACT.stop_damping,
    settings: config.GroundSettings | None = None,
) -> StepDropResult:
    """Run one wheel release, lower-contact impact and post-impact recovery."""
    if horizon_s <= 0.0 or settling_tolerance <= 0.0:
        raise ValueError("horizon_s and settling_tolerance must be positive")
    if integration_substeps < 1:
        raise ValueError("integration_substeps must be at least one")
    if min(ground_stiffness, ground_damping, stop_stiffness, stop_damping) <= 0.0:
        raise ValueError("all contact stiffness and damping values must be positive")
    settings = settings or config.GroundSettings()
    step = settings.sample_time
    if mean_leg_length is None:
        mean_leg_length = (parameters.min_leg_length + parameters.max_leg_length) / 2
    count = int(round(horizon_s / step))
    if count < 1 or not np.isclose(count * step, horizon_s, atol=1e-12):
        raise ValueError("horizon_s must be an integer number of control samples")
    target_lengths = target_lengths_for_step(
        step_height, mean_leg_length=mean_leg_length, down_leg=down_leg, parameters=parameters
    )
    initial_node = compute_gain_node(mean_leg_length, mean_leg_length, parameters=parameters, settings=settings)
    target_node = compute_gain_node(*target_lengths, parameters=parameters, settings=settings)
    observer = compute_rdob_node(
        *target_lengths, bandwidth_hz=bandwidth_hz, parameters=parameters, sample_time=step
    ).observer
    selector = (
        np.diag([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
        if compensation is None
        else np.asarray(compensation, dtype=float)
    )
    if selector.shape != (config.N_U, config.N_U):
        raise ValueError("compensation must have shape (6, 6)")

    target_q7 = target_node.linearization.geometry.q
    q = np.zeros(N_FREE_Q, dtype=float)
    q[Q7] = initial_node.linearization.geometry.q
    velocity = np.zeros(N_FREE_Q, dtype=float)
    ground = np.array([-step_height, 0.0], dtype=float)
    if down_leg == _WHEEL_2:
        ground = ground[::-1]
    z = initialize_rdob(observer, velocity[Q7])
    feedforward = target_node.linearization.equilibrium_input
    time = np.arange(count + 1) * step
    state_error = np.empty((count + 1, config.N_X), dtype=float)
    leg_lengths = np.empty((count + 1, 2), dtype=float)
    commands = np.empty((count, config.N_U), dtype=float)
    estimates = np.empty((count + 1, config.N_U), dtype=float)
    contact_forces = np.zeros((count, 2), dtype=float)
    mechanical_stop_forces = np.zeros((count, 2), dtype=float)
    state_error[0] = _interleaved_error(q[Q7] - target_q7, velocity[Q7])
    leg_lengths[0] = _kinematics(q, parameters).lengths
    estimates[0] = reconstruct_disturbance(observer, z, velocity[Q7])
    impact_time: float | None = None
    impact_speed: float | None = None
    impact_impulse: float | None = None
    mechanical_stop_impulse = np.zeros(2, dtype=float)
    saturation_steps = 0

    completed = count
    failure_reason: str | None = None
    substep = step / integration_substeps
    for index in range(count):
        if not _inside_numerical_envelope(q, velocity):
            completed = index
            failure_reason = (
                "state diverged beyond the nonlinear contact model's numerical "
                "envelope"
            )
            break
        q_error = q[Q7] - target_q7
        v_error = velocity[Q7]
        state = _interleaved_error(q_error, v_error)
        estimate = reconstruct_disturbance(observer, z, v_error)
        commanded = feedforward - target_node.gain @ state - selector @ estimate
        control = np.clip(
            commanded, parameters.input_lower_limits, parameters.input_upper_limits
        )
        commands[index] = control
        saturation_steps += int(np.any(control != commanded))

        for local_index in range(integration_substeps):
            try:
                acceleration, wheel_force, stop_force, kin = _compliant_acceleration(
                    q,
                    velocity,
                    control,
                    ground,
                    parameters=parameters,
                    ground_stiffness=ground_stiffness,
                    ground_damping=ground_damping,
                    stop_stiffness=stop_stiffness,
                    stop_damping=stop_damping,
                )
            except np.linalg.LinAlgError:
                completed = index
                failure_reason = "mass matrix became singular during contact integration"
                break
            velocity_next = velocity + substep * acceleration
            q_next = q + substep * velocity_next
            next_kin = _kinematics(q_next, parameters)
            if impact_time is None:
                gap_before = kin.positions[down_leg, 2] - ground[down_leg]
                gap_after = next_kin.positions[down_leg, 2] - ground[down_leg]
                if gap_before > 0.0 and gap_after <= 0.0:
                    fraction = gap_before / (gap_before - gap_after)
                    impact_time = float(
                        index * step + (local_index + fraction) * substep
                    )
                    impact_speed = float(
                        kin.maps[down_leg, 2]
                        @ (velocity + fraction * (velocity_next - velocity))
                    )
            if impact_time is not None:
                impact_impulse = float(
                    (0.0 if impact_impulse is None else impact_impulse)
                    + wheel_force[down_leg] * substep
                )
            mechanical_stop_impulse += stop_force * substep
            contact_forces[index] = np.maximum(contact_forces[index], wheel_force)
            mechanical_stop_forces[index] = np.maximum(
                mechanical_stop_forces[index], stop_force
            )
            if not _inside_numerical_envelope(q_next, velocity_next):
                completed = index
                failure_reason = (
                    "state diverged during compliant contact/stop integration"
                )
                break
            q, velocity = q_next, velocity_next
        if failure_reason is not None:
            break
        next_q_error = q[Q7] - target_q7
        next_v_error = velocity[Q7]
        z, next_estimate = step_rdob(
            observer,
            z,
            q_error,
            v_error,
            control - feedforward,
            next_q_error,
            next_v_error,
        )
        state_error[index + 1] = _interleaved_error(next_q_error, next_v_error)
        leg_lengths[index + 1] = _kinematics(q, parameters).lengths
        estimates[index + 1] = next_estimate

    time = time[: completed + 1]
    state_error = state_error[: completed + 1]
    leg_lengths = leg_lengths[: completed + 1]
    commands = commands[:completed]
    estimates = estimates[: completed + 1]
    contact_forces = contact_forces[:completed]
    mechanical_stop_forces = mechanical_stop_forces[:completed]
    envelope = np.max(np.abs(state_error), axis=1)
    outside = np.flatnonzero(envelope > settling_tolerance)
    settling_time = (
        0.0
        if not len(outside)
        else (
            float(time[outside[-1] + 1])
            if outside[-1] + 1 < len(time)
            else None
        )
    )
    return StepDropResult(
        step_height=float(step_height),
        down_leg=int(down_leg),
        target_lengths=target_lengths,
        target_kappa=float(target_node.linearization.geometry.kappa_g),
        target_midpoint_height=-0.5 * float(step_height),
        time=time,
        state_error=state_error,
        leg_lengths=leg_lengths,
        commands=commands,
        disturbance_estimate=estimates,
        contact_forces=contact_forces,
        mechanical_stop_forces=mechanical_stop_forces,
        impact_time=impact_time,
        impact_speed=impact_speed,
        impact_impulse=impact_impulse,
        mechanical_stop_impulse=mechanical_stop_impulse,
        peak_contact_force=float(np.max(np.abs(contact_forces))),
        peak_mechanical_stop_force=float(np.max(np.abs(mechanical_stop_forces))),
        saturation_steps=saturation_steps,
        settling_time=settling_time,
        converged=failure_reason is None and settling_time is not None,
        terminated_time=float(time[-1]),
        failure_reason=failure_reason,
    )
