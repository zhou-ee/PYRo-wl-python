"""Hybrid contact screening for one wheel descending a vertical step.

The seven-coordinate static-trim model fixes the ground height difference and
omits the world vertical position of the wheel midpoint. Neither omission is
valid while a wheel is airborne. This analysis restores both coordinates,
releases the descending wheel, and models lower-ground and mechanical-stop
impacts with unilateral spring-damper contacts.

It is a conservative screening model. It retains the repository's
configuration-dependent mass, gravity, input geometry, controller, RDOB,
input limits, contact timing and accumulated contact impulse. It deliberately omits tyre
compliance, wheel-edge geometry, horizontal friction, actuator dynamics,
sensor delay, and velocity-quadratic terms absent from the source static model.
It can rule out a transition that clearly cannot recover; it cannot certify
hardware impact safety.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from ..control.rdob import initialize_rdob, reconstruct_disturbance, step_rdob
from ..core import config
from ..rdob import DEFAULT_BANDWIDTH_HZ, compute_rdob_node
from ..scheduling.leg_length_schedule import compute_gain_node


N_FREE_Q = 9
Q7 = slice(0, 7)
_WHEEL_1 = 0
_WHEEL_2 = 1


@dataclass(frozen=True)
class _Kinematics:
    """Particle positions and maps in the heading frame for nine coordinates."""

    lengths: np.ndarray
    positions: np.ndarray  # Wheel 1, wheel 2, leg 1 COM, leg 2 COM, body COM.
    maps: np.ndarray  # Shape (5, 3, 9).
    length_jacobian: np.ndarray  # Shape (2, 9).


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


def _raw_kinematics(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate particles in the original heading frame plus free kappa/z_c."""
    _, _, theta, phi, l_bar, beta_1, beta_2, kappa, z_c = np.asarray(
        q, dtype=float
    ).reshape(N_FREE_Q)
    p = config.PARAMETERS
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, c_phi, -s_phi], [0.0, s_phi, c_phi]]
    )
    rotation_y = np.array(
        [[c_theta, 0.0, s_theta], [0.0, 1.0, 0.0], [-s_theta, 0.0, c_theta]]
    )
    body_to_heading = rotation_y @ rotation_x
    geometry_offset = p.hip_half_track * c_theta * s_phi - kappa
    sec_1, sec_2 = 1.0 / np.cos(beta_1), 1.0 / np.cos(beta_2)
    height = (2.0 * l_bar - geometry_offset * (sec_1 - sec_2)) / (sec_1 + sec_2)
    lengths = np.array(
        [
            (height + geometry_offset) / np.cos(beta_1),
            (height - geometry_offset) / np.cos(beta_2),
        ],
        dtype=float,
    )
    betas = (beta_1, beta_2)
    signs = (1.0, -1.0)
    bar_s = 0.5 * sum(
        length * np.sin(beta) for length, beta in zip(lengths, betas)
    )
    hip_mid = np.array([-bar_s, 0.0, height], dtype=float)
    wheel_positions: list[np.ndarray] = []
    leg_positions: list[np.ndarray] = []
    for sign, length, beta in zip(signs, lengths, betas):
        hip = hip_mid + body_to_heading @ np.array(
            [0.0, sign * p.hip_half_track, 0.0]
        )
        direction = np.array([-np.sin(beta), 0.0, np.cos(beta)])
        lower_pivot = hip - length * direction
        wheel_positions.append(
            lower_pivot
            + np.array([0.0, sign * (p.wheel_half_track - p.hip_half_track), z_c])
        )
        l_wp = np.polynomial.polynomial.polyval(length, config.LEG_COM_LWP_POLY)
        x_com = np.polynomial.polynomial.polyval(length, config.LEG_COM_X_POLY)
        normal = np.array([np.cos(beta), 0.0, np.sin(beta)])
        leg_positions.append(
            lower_pivot
            + l_wp * direction
            + x_com * normal
            + np.array([0.0, 0.0, z_c])
        )
    body = (
        hip_mid
        + body_to_heading @ np.asarray(p.body_com_from_hip)
        + np.array([0.0, 0.0, z_c])
    )
    return lengths, np.asarray(
        [*wheel_positions, *leg_positions, body], dtype=float
    )


def _kinematics(q: np.ndarray) -> _Kinematics:
    q = np.asarray(q, dtype=float).reshape(N_FREE_Q)
    lengths, positions = _raw_kinematics(q)
    maps = np.empty((5, 3, N_FREE_Q), dtype=float)
    length_jacobian = np.empty((2, N_FREE_Q), dtype=float)
    for index in range(N_FREE_Q):
        step = 1e-6 * max(1.0, abs(q[index]))
        plus, minus = q.copy(), q.copy()
        plus[index] += step
        minus[index] -= step
        lengths_plus, positions_plus = _raw_kinematics(plus)
        lengths_minus, positions_minus = _raw_kinematics(minus)
        maps[:, :, index] = (positions_plus - positions_minus) / (2.0 * step)
        length_jacobian[:, index] = (lengths_plus - lengths_minus) / (2.0 * step)
    # The position vectors use a heading frame. Restore translational and yaw
    # transport terms which are implicit in that representation.
    maps[:, :, 0] = np.array([1.0, 0.0, 0.0])
    maps[:, :, 1] += np.cross(np.array([0.0, 0.0, 1.0]), positions)
    return _Kinematics(lengths, positions, maps, length_jacobian)


def _mass_and_input(
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, _Kinematics]:
    """Return free-flight M, gravity, S and kinematics.

    The static source model has no C(q,qdot)qdot term. This extension retains
    that convention instead of introducing an unvalidated velocity model.
    """
    kin = _kinematics(q)
    p = config.PARAMETERS
    mass = np.zeros((N_FREE_Q, N_FREE_Q), dtype=float)
    for wheel_map in kin.maps[:2]:
        rolling = wheel_map[0]
        mass += p.wheel_mass * wheel_map.T @ wheel_map
        mass += (
            p.wheel_rolling_inertia / p.wheel_radius**2
            * np.outer(rolling, rolling)
        )
    for leg_map in kin.maps[2:4]:
        mass += p.leg_mass * leg_map.T @ leg_map
    mass += p.body_mass * kin.maps[4].T @ kin.maps[4]
    for index, length in enumerate(kin.lengths):
        l_ph = length - np.polynomial.polynomial.polyval(
            length, (0.0581, 0.3760, 0.7972, -0.7381)
        )
        inertia = p.leg_mass * length**2 / 12.0 + p.leg_mass * (
            length / 2.0 - l_ph
        ) ** 2
        mass[5 + index, 5 + index] += inertia
    mass = 0.5 * (mass + mass.T)
    particle_masses = np.array(
        [p.wheel_mass, p.wheel_mass, p.leg_mass, p.leg_mass, p.body_mass]
    )
    gravity = p.gravity * np.einsum(
        "p,pj->j", particle_masses, kin.maps[:, 2, :]
    )
    inputs = np.zeros((N_FREE_Q, config.N_U), dtype=float)
    for index, wheel_map in enumerate(kin.maps[:2]):
        inputs[:, index] = wheel_map[0] / p.wheel_radius
        inputs[5 + index, index] += 1.0
    for index in range(2):
        inputs[2, 2 + index] = 1.0
        inputs[5 + index, 2 + index] = 1.0
        inputs[:, 4 + index] = kin.length_jacobian[index]
    return mass, gravity, inputs, kin


def target_lengths_for_step(
    step_height: float,
    *,
    mean_leg_length: float = 0.28,
    down_leg: int = _WHEEL_1,
) -> np.ndarray:
    """Solve leg lengths that retain the mean on a two-level support."""
    if down_leg not in (_WHEEL_1, _WHEEL_2):
        raise ValueError("down_leg must be 0 (leg 1) or 1 (leg 2)")
    if not np.isfinite(step_height) or step_height < 0.0:
        raise ValueError("step_height must be finite and non-negative")
    lower, upper = config.PARAMETERS.min_leg_length, config.PARAMETERS.max_leg_length
    lo = max(lower, 2.0 * mean_leg_length - upper)
    hi = min(upper, 2.0 * mean_leg_length - lower)
    if not lo <= mean_leg_length <= hi:
        raise ValueError("mean_leg_length is incompatible with the configured leg limits")

    def differential(longer: float) -> float:
        shorter = 2.0 * mean_leg_length - longer
        return (
            longer * np.cos(config.compute_equilibrium_beta_from_length(longer))
            - shorter * np.cos(config.compute_equilibrium_beta_from_length(shorter))
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


def _compliant_acceleration(
    q: np.ndarray,
    velocity: np.ndarray,
    control: np.ndarray,
    ground: np.ndarray,
    *,
    ground_stiffness: float,
    ground_damping: float,
    stop_stiffness: float,
    stop_damping: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, _Kinematics]:
    """Return acceleration with unilateral spring-damper ground and leg stops."""
    mass, gravity, inputs, kin = _mass_and_input(q)
    generalized_contact = np.zeros(N_FREE_Q, dtype=float)
    wheel_forces = np.zeros(2, dtype=float)
    stop_forces = np.zeros(2, dtype=float)
    for wheel in (_WHEEL_1, _WHEEL_2):
        gap = float(kin.positions[wheel, 2] - ground[wheel])
        normal_speed = float(kin.maps[wheel, 2] @ velocity)
        if gap < 0.0:
            wheel_forces[wheel] = max(
                0.0, -ground_stiffness * gap - ground_damping * normal_speed
            )
            generalized_contact += kin.maps[wheel, 2] * wheel_forces[wheel]
    for leg in (_WHEEL_1, _WHEEL_2):
        upper_gap = config.PARAMETERS.max_leg_length - kin.lengths[leg]
        upper_map = -kin.length_jacobian[leg]
        lower_gap = kin.lengths[leg] - config.PARAMETERS.min_leg_length
        lower_map = kin.length_jacobian[leg]
        if upper_gap < 0.0:
            stop_forces[leg] = max(
                0.0,
                -stop_stiffness * upper_gap - stop_damping * (upper_map @ velocity),
            )
            generalized_contact += upper_map * stop_forces[leg]
        elif lower_gap < 0.0:
            stop_forces[leg] = max(
                0.0,
                -stop_stiffness * lower_gap - stop_damping * (lower_map @ velocity),
            )
            generalized_contact += lower_map * stop_forces[leg]
    acceleration = np.linalg.solve(
        mass, inputs @ control - gravity + generalized_contact
    )
    return acceleration, wheel_forces, stop_forces, kin


def simulate_step_drop(
    *,
    step_height: float = 0.18,
    down_leg: int = _WHEEL_1,
    mean_leg_length: float = 0.28,
    horizon_s: float = 3.0,
    compensation: np.ndarray | None = None,
    bandwidth_hz: np.ndarray = DEFAULT_BANDWIDTH_HZ,
    settling_tolerance: float = 1e-3,
    integration_substeps: int = 5,
    ground_stiffness: float = 5.0e3,
    ground_damping: float = 2.0e2,
    stop_stiffness: float = 1.0e4,
    stop_damping: float = 3.0e2,
) -> StepDropResult:
    """Run one wheel release, lower-contact impact and post-impact recovery."""
    if horizon_s <= 0.0 or settling_tolerance <= 0.0:
        raise ValueError("horizon_s and settling_tolerance must be positive")
    if integration_substeps < 1:
        raise ValueError("integration_substeps must be at least one")
    if min(ground_stiffness, ground_damping, stop_stiffness, stop_damping) <= 0.0:
        raise ValueError("all contact stiffness and damping values must be positive")
    step = config.SAMPLE_TIME
    count = int(round(horizon_s / step))
    if count < 1 or not np.isclose(count * step, horizon_s, atol=1e-12):
        raise ValueError("horizon_s must be an integer number of control samples")
    target_lengths = target_lengths_for_step(
        step_height, mean_leg_length=mean_leg_length, down_leg=down_leg
    )
    initial_node = compute_gain_node(mean_leg_length, mean_leg_length)
    target_node = compute_gain_node(*target_lengths)
    observer = compute_rdob_node(
        *target_lengths, bandwidth_hz=bandwidth_hz, sample_time=step
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
    leg_lengths[0] = _kinematics(q).lengths
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
            commanded, config.INPUT_LOWER_LIMITS, config.INPUT_UPPER_LIMITS
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
            next_kin = _kinematics(q_next)
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
        leg_lengths[index + 1] = _kinematics(q).lengths
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-height", type=float, default=0.18)
    parser.add_argument(
        "--down-leg", type=int, choices=(_WHEEL_1, _WHEEL_2), default=_WHEEL_1
    )
    parser.add_argument("--horizon", type=float, default=3.0)
    args = parser.parse_args()
    result = simulate_step_drop(
        step_height=args.step_height,
        down_leg=args.down_leg,
        horizon_s=args.horizon,
    )
    print(
        f"target leg lengths: {result.target_lengths[0]:.6f}, "
        f"{result.target_lengths[1]:.6f} m"
    )
    print(f"impact time: {result.impact_time!r} s")
    print(f"impact wheel vertical speed: {result.impact_speed!r} m/s")
    print(
        "lower-contact normal impulse through termination: "
        f"{result.impact_impulse!r} N s"
    )
    print(f"peak contact force: {result.peak_contact_force:.6g} N")
    print(
        "mechanical-stop impulses: "
        f"{result.mechanical_stop_impulse[0]:.6g}, "
        f"{result.mechanical_stop_impulse[1]:.6g} N s"
    )
    print(f"peak mechanical-stop force: {result.peak_mechanical_stop_force:.6g} N")
    print(f"input saturation samples: {result.saturation_steps}")
    print(f"settling time: {result.settling_time!r} s")
    print(f"converged: {result.converged}")
    print(f"terminated time: {result.terminated_time:.6f} s")
    print(f"failure reason: {result.failure_reason!r}")


if __name__ == "__main__":
    main()
