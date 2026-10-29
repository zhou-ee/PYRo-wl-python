"""Five-coordinate energy model with externally frozen L1/L2 and no roll.

This is a mean-hip-height local balance approximation. It is derived on its
own coordinate manifold; no 14-state matrices, constraints or inverses enter.
Lengths are parameters before differentiating energies and virtual work.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import sympy as sp

from ...config import DEFAULT_ROBOT, RobotParameters
from ..leg_geometry import leg_com
from .config import N_Q, N_U, N_X


@dataclass(frozen=True)
class GeometryAtTrim:
    l1: float
    l2: float
    q: np.ndarray
    height: float

    @property
    def leg_lengths(self):
        return np.array([self.l1, self.l2])


@dataclass(frozen=True)
class StaticLinearization:
    geometry: GeometryAtTrim
    equilibrium_input: np.ndarray
    equilibrium_residual: np.ndarray
    mass: np.ndarray
    stiffness: np.ndarray
    input_matrix: np.ndarray
    A: np.ndarray
    B: np.ndarray


@dataclass(frozen=True)
class Kinematics:
    # Particle order: wheel 1, wheel 2, leg COM 1, leg COM 2, body COM.
    positions: np.ndarray  # (5 particles, 3 heading-frame components)
    velocity_maps: np.ndarray  # (5, 3, 5 coordinates)
    height: float  # Mean hip height above wheel centers; excludes wheel radius.


@dataclass(frozen=True)
class _CompiledModel:
    mass: object
    gravity: object
    potential: object
    input_matrix: object
    stiffness: object
    positions: object
    velocity_maps: object
    height: object


@lru_cache(maxsize=8)
def _compile_model(parameters: RobotParameters = DEFAULT_ROBOT):
    p = parameters
    x, psi, theta, beta_1, beta_2 = sp.symbols("x psi theta beta_1 beta_2", real=True)
    l1, l2 = sp.symbols("L1 L2", positive=True)
    q = sp.Matrix([x, psi, theta, beta_1, beta_2])
    betas, lengths = (beta_1, beta_2), (l1, l2)
    e1, e3 = sp.Matrix([1, 0, 0]), sp.Matrix([0, 0, 1])
    # The heading-frame origin follows the horizontal wheel midpoint.
    bar_s = sum(length * sp.sin(beta) for length, beta in zip(lengths, betas)) / 2
    height = sum(length * sp.cos(beta) for length, beta in zip(lengths, betas)) / 2
    c, s = sp.cos(theta), sp.sin(theta)
    rotation_y = sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    hip_mid = sp.Matrix([-bar_s, 0, height])

    def curve(coefficients, length):
        return sum(
            sp.Float(value) * length**power for power, value in enumerate(coefficients)
        )

    def velocity_map(position):
        jacobian = position.jacobian(q)
        jacobian[:, 0] = e1
        jacobian[:, 1] = e3.cross(position)
        return jacobian

    wheels, legs = [], []
    for sign, length, beta in zip((1, -1), lengths, betas):
        wheel_x = -bar_s + length * sp.sin(beta)
        wheel = sp.Matrix([wheel_x, sign * sp.Float(p.wheel_half_track), 0])
        lwp = curve(p.leg_com_lwp_poly, length)
        x_com = curve(p.effective_leg_com_x_poly, length)
        # Independent per-leg planar geometry; no constraint-induced dL/dbeta.
        leg = sp.Matrix(
            [
                wheel_x - lwp * sp.sin(beta) + x_com * sp.cos(beta),
                sign * sp.Float(p.hip_half_track),
                lwp * sp.cos(beta) + x_com * sp.sin(beta),
            ]
        )
        wheels.append(wheel)
        legs.append(leg)
    body = hip_mid + rotation_y * sp.Matrix([sp.Float(v) for v in p.body_com_from_hip])
    positions = [*wheels, *legs, body]
    maps = [velocity_map(position) for position in positions]
    masses = [p.wheel_mass, p.wheel_mass, p.leg_mass, p.leg_mass, p.body_mass]
    mass = sp.zeros(N_Q)
    for particle_mass, jacobian in zip(masses, maps):
        mass += sp.Float(particle_mass) * jacobian.T * jacobian
    for jacobian in maps[:2]:
        rolling = e1.T * jacobian
        mass += (
            sp.Float(p.wheel_rolling_inertia / p.wheel_radius**2) * rolling.T * rolling
        )
    for index, length in enumerate(lengths):
        lwp = curve(p.leg_com_lwp_poly, length)
        # Same rod approximation, evaluated with the full configured COM curve.
        inertia = sp.Float(p.leg_mass) * (length**2 / 12 + (lwp - length / 2) ** 2)
        mass[3 + index, 3 + index] += inertia
    angular_map = sp.zeros(3, N_Q)
    angular_map[:, 1] = sp.Matrix([-s, 0, c])
    angular_map[:, 2] = sp.Matrix([0, 1, 0])
    inertia = sp.Matrix([[sp.Float(v) for v in row] for row in p.body_inertia])
    mass += angular_map.T * inertia * angular_map
    potential = sum(
        sp.Float(m * p.gravity) * position[2] for m, position in zip(masses, positions)
    )
    gravity = sp.Matrix([sp.diff(potential, coordinate) for coordinate in q])
    inputs = sp.zeros(N_Q, N_U)
    for index, jacobian in enumerate(maps[:2]):
        column = jacobian.T * e1 / sp.Float(p.wheel_radius)
        column[3 + index] += 1
        inputs[:, index] = column
        inputs[2, 2 + index] = 1
        inputs[3 + index, 2 + index] = 1
    u = sp.Matrix(sp.symbols("T_w1 T_w2 T_p1 T_p2", real=True))
    stiffness = (gravity - inputs * u).jacobian(q)
    arguments = (*q, l1, l2)

    def compile_value(value, args=arguments):
        return sp.lambdify(args, value, modules="numpy", cse=True)

    return _CompiledModel(
        mass=compile_value(mass),
        gravity=compile_value(gravity),
        potential=compile_value(potential),
        input_matrix=compile_value(inputs),
        stiffness=compile_value(stiffness, (*arguments, *u)),
        positions=compile_value(sp.Matrix.hstack(*positions)),
        velocity_maps=compile_value(sp.Matrix.hstack(*maps)),
        height=compile_value(height),
    )


def _lengths(l1, l2):
    lengths = np.asarray([l1, l2], dtype=float)
    if (
        lengths.shape != (2,)
        or not np.all(np.isfinite(lengths))
        or np.any(lengths <= 0)
    ):
        raise ValueError("L1/L2 must be positive and finite")
    return tuple(lengths)


def _arguments(q, l1, l2):
    q = np.asarray(q, dtype=float)
    if q.shape != (N_Q,) or not np.all(np.isfinite(q)):
        raise ValueError("q must contain five finite coordinates")
    return (*q, *_lengths(l1, l2))


def evaluate_mass(q, l1, l2, *, parameters=DEFAULT_ROBOT):
    value = np.asarray(
        _compile_model(parameters).mass(*_arguments(q, l1, l2)), dtype=float
    )
    return (value + value.T) / 2


def evaluate_gravity(q, l1, l2, *, parameters=DEFAULT_ROBOT):
    return np.asarray(
        _compile_model(parameters).gravity(*_arguments(q, l1, l2)), dtype=float
    ).reshape(N_Q)


def evaluate_potential(q, l1, l2, *, parameters=DEFAULT_ROBOT):
    return float(_compile_model(parameters).potential(*_arguments(q, l1, l2)))


def evaluate_input_matrix(q, l1, l2, *, parameters=DEFAULT_ROBOT):
    return np.asarray(
        _compile_model(parameters).input_matrix(*_arguments(q, l1, l2)), dtype=float
    )


def evaluate_kinematics(q, l1, l2, *, parameters=DEFAULT_ROBOT):
    compiled, args = _compile_model(parameters), _arguments(q, l1, l2)
    return Kinematics(
        positions=np.asarray(compiled.positions(*args), dtype=float).T,
        velocity_maps=np.asarray(compiled.velocity_maps(*args), dtype=float)
        .reshape(3, 5, N_Q)
        .transpose(1, 0, 2),
        height=float(compiled.height(*args)),
    )


def compute_equilibrium_beta_from_length(length, parameters=DEFAULT_ROBOT):
    """Static beta with equal hip-torque sharing, body pitch fixed at zero."""
    length, _ = _lengths(length, length)
    lwp, x_com = leg_com(length, parameters)
    axial = parameters.body_mass * length / 2 + parameters.leg_mass * lwp
    offset = parameters.leg_mass * x_com
    lever = np.hypot(axial, offset)
    target = parameters.body_mass * parameters.body_com_x_from_hip / 2
    if abs(target) >= lever:
        raise ValueError(
            "body COM cannot be balanced at this leg length with zero pitch"
        )
    return float(np.arctan2(offset, axial) + np.arcsin(target / lever))


def linearize_static_trim(l1, l2, *, parameters=DEFAULT_ROBOT):
    l1, l2 = _lengths(l1, l2)
    if (
        min(l1, l2) < parameters.min_leg_length - 1e-12
        or max(l1, l2) > parameters.max_leg_length + 1e-12
    ):
        raise ValueError("L1/L2 are outside the configured leg-length range")
    q = np.array(
        [
            0.0,
            0.0,
            0.0,
            *(
                compute_equilibrium_beta_from_length(length, parameters)
                for length in (l1, l2)
            ),
        ]
    )
    mass = evaluate_mass(q, l1, l2, parameters=parameters)
    if np.min(np.linalg.eigvalsh(mass)) <= 0:
        raise ValueError("ten-state mass must be positive definite")
    inputs = evaluate_input_matrix(q, l1, l2, parameters=parameters)
    gravity = evaluate_gravity(q, l1, l2, parameters=parameters)
    u0, *_ = np.linalg.lstsq(inputs, gravity, rcond=None)
    residual = inputs @ u0 - gravity
    if np.max(np.abs(residual)) > 1e-8:
        raise ValueError("ten-state static equilibrium residual exceeds tolerance")
    compiled = _compile_model(parameters)
    stiffness = np.asarray(compiled.stiffness(*_arguments(q, l1, l2), *u0), dtype=float)
    A, B = np.zeros((N_X, N_X)), np.zeros((N_X, N_U))
    A[0::2, 1::2] = np.eye(N_Q)
    A[1::2, 0::2] = -np.linalg.solve(mass, stiffness)
    B[1::2] = np.linalg.solve(mass, inputs)
    return StaticLinearization(
        geometry=GeometryAtTrim(
            float(l1), float(l2), q, float(compiled.height(*_arguments(q, l1, l2)))
        ),
        equilibrium_input=u0,
        equilibrium_residual=residual,
        mass=mass,
        stiffness=stiffness,
        input_matrix=inputs,
        A=A,
        B=B,
    )


@dataclass(frozen=True)
class Ground10Model:
    parameters: RobotParameters = DEFAULT_ROBOT

    def linearize(self, l1, l2):
        return linearize_static_trim(l1, l2, parameters=self.parameters)

    def mass(self, q, l1, l2):
        return evaluate_mass(q, l1, l2, parameters=self.parameters)
