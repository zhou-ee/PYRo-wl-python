"""航向坐标系下七坐标模型的静态配平力学。

Heading-frame static-trim mechanics for the seven-coordinate model.  Builds the
symbolic mass / gravity / input matrices, then linearizes them around a static
trim to obtain the continuous-time ``A`` / ``B`` pair used by the gain schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import sympy as sp

from ...config import DEFAULT_ROBOT, RobotParameters
from ..leg_geometry import ground_inertia_com_coefficients
from .trim import compute_equilibrium_beta_from_length
from .config import N_Q, N_U, N_X


@dataclass(frozen=True)
class GeometryAtTrim:
    l1: float
    l2: float
    kappa_g: float
    q: np.ndarray
    leg_lengths: np.ndarray
    height: float
    feasible: bool


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
class _CompiledModel:
    mass: object
    gravity: object
    input_matrix: object
    lengths: object
    height: object
    wheel_positions_heading: object
    wheel_velocity_maps_heading: object


def _rotation_x(angle: sp.Expr) -> sp.Matrix:
    c, s = sp.cos(angle), sp.sin(angle)
    return sp.Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rotation_y(angle: sp.Expr) -> sp.Matrix:
    c, s = sp.cos(angle), sp.sin(angle)
    return sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])


@lru_cache(maxsize=8)
def _compile_model(parameters: RobotParameters = DEFAULT_ROBOT) -> _CompiledModel:
    p = parameters
    x, psi, theta, phi, l_bar, beta_1, beta_2, kappa = sp.symbols(
        "x psi theta phi l_bar beta_1 beta_2 kappa", real=True
    )
    q = sp.Matrix([x, psi, theta, phi, l_bar, beta_1, beta_2])
    body_to_heading = _rotation_y(theta) * _rotation_x(phi)

    rw = sp.Float(p.wheel_radius)
    ah = sp.Float(p.hip_half_track)
    aw = sp.Float(p.wheel_half_track)
    r_phi = ah * sp.cos(theta) * sp.sin(phi)
    geometry_offset = r_phi - kappa
    sec_beta_1 = 1 / sp.cos(beta_1)
    sec_beta_2 = 1 / sp.cos(beta_2)
    # Invert L_bar = (L1 + L2) / 2 to recover the vertical projection h.
    h = (
        2 * l_bar - geometry_offset * (sec_beta_1 - sec_beta_2)
    ) / (sec_beta_1 + sec_beta_2)
    lengths = (
        (h + geometry_offset) / sp.cos(beta_1),
        (h - geometry_offset) / sp.cos(beta_2),
    )
    betas = (beta_1, beta_2)
    sigmas = (sp.Integer(1), sp.Integer(-1))
    e1 = sp.Matrix([1, 0, 0])
    e3 = sp.Matrix([0, 0, 1])
    bar_s = sum(
        lengths[index] * sp.sin(betas[index]) for index in range(2)
    ) / 2

    # All geometry is expressed in the horizontal heading frame Y, whose
    # origin is the wheel-centre midpoint C.  Y rotates about world vertical
    # with yaw psi, so absolute velocity components expressed in Y are
    #   v_k^Y = e1 * dot(x) + dot(rho_k^Y)
    #           + dot(psi) * e3 x rho_k^Y.
    r_h_mid = sp.Matrix([-bar_s, 0, h])

    def heading_velocity_map(relative_position: sp.Matrix) -> sp.Matrix:
        velocity_map = relative_position.jacobian(q)
        velocity_map[:, 0] = e1
        velocity_map[:, 1] = (
            velocity_map[:, 1] + e3.cross(relative_position)
        )
        return velocity_map

    wheel_positions_heading: list[sp.Matrix] = []
    leg_positions_heading: list[sp.Matrix] = []
    wheel_velocity_maps_heading: list[sp.Matrix] = []
    leg_velocity_maps_heading: list[sp.Matrix] = []
    for sigma, length, beta in zip(sigmas, lengths, betas):
        hip_offset = sp.Matrix([0, sigma * ah, 0])
        hip = r_h_mid + body_to_heading * hip_offset
        direction = sp.Matrix([-sp.sin(beta), 0, sp.cos(beta)])
        # P_i is the lower endpoint of the virtual leg in the five-bar plane.
        # W_i is the wheel centre; they differ by the fixed lateral axle offset.
        lower_pivot = hip - length * direction

        # The five-bar plane is at a_h while the wheel centre is at a_w.
        # In Y, the fixed axle offset is a constant lateral vector.
        axle_offset = sp.Matrix([0, sigma * (aw - ah), 0])
        wheel = lower_pivot + axle_offset

        l_wp = (
            sp.Float(p.leg_com_lwp_poly[0])
            + sp.Float(p.leg_com_lwp_poly[1]) * length
            + sp.Float(p.leg_com_lwp_poly[2]) * length**2
            + sp.Float(p.leg_com_lwp_poly[3]) * length**3
        )
        x_com = (
            sp.Float(p.effective_leg_com_x_poly[0])
            + sp.Float(p.effective_leg_com_x_poly[1]) * length
            + sp.Float(p.effective_leg_com_x_poly[2]) * length**2
            + sp.Float(p.effective_leg_com_x_poly[3]) * length**3
        )
        normal_direction = sp.Matrix([sp.cos(beta), 0, sp.sin(beta)])
        leg_com = lower_pivot + l_wp * direction + x_com * normal_direction
        wheel_positions_heading.append(wheel)
        leg_positions_heading.append(leg_com)
        wheel_velocity_maps_heading.append(heading_velocity_map(wheel))
        leg_velocity_maps_heading.append(heading_velocity_map(leg_com))

    body_com_offset = sp.Matrix(
        [sp.Float(value) for value in p.body_com_from_hip]
    )
    body_position_heading = r_h_mid + body_to_heading * body_com_offset
    body_velocity_map_heading = heading_velocity_map(body_position_heading)

    mass = sp.zeros(N_Q, N_Q)
    for wheel_velocity_map in wheel_velocity_maps_heading:
        rolling_jacobian = e1.T * wheel_velocity_map
        mass += (
            sp.Float(p.wheel_mass)
            * wheel_velocity_map.T
            * wheel_velocity_map
        )
        mass += (
            sp.Float(p.wheel_rolling_inertia / p.wheel_radius**2)
            * rolling_jacobian.T
            * rolling_jacobian
        )
    for leg_velocity_map in leg_velocity_maps_heading:
        mass += (
            sp.Float(p.leg_mass) * leg_velocity_map.T * leg_velocity_map
        )
    mass += (
        sp.Float(p.body_mass)
        * body_velocity_map_heading.T
        * body_velocity_map_heading
    )

    for index, length in enumerate(lengths):
        l_ph = length - (
            sp.Float(ground_inertia_com_coefficients(p)[0])
            + sp.Float(ground_inertia_com_coefficients(p)[1]) * length
            + sp.Float(ground_inertia_com_coefficients(p)[2]) * length**2
            + sp.Float(ground_inertia_com_coefficients(p)[3]) * length**3
        )
        offset = length / 2 - l_ph
        leg_inertia = (
            sp.Float(p.leg_mass) * length**2 / 12
            + sp.Float(p.leg_mass) * offset**2
        )
        beta_selector = sp.zeros(N_Q, 1)
        beta_selector[5 + index] = 1
        mass += leg_inertia * beta_selector * beta_selector.T

    angular_map = sp.zeros(3, N_Q)
    angular_map[:, 1] = sp.Matrix(
        [-sp.sin(theta), sp.cos(theta) * sp.sin(phi), sp.cos(theta) * sp.cos(phi)]
    )
    angular_map[:, 2] = sp.Matrix([0, sp.cos(phi), -sp.sin(phi)])
    angular_map[:, 3] = sp.Matrix([1, 0, 0])
    body_inertia = sp.Matrix(
        [[sp.Float(value) for value in row] for row in p.body_inertia]
    )
    mass += angular_map.T * body_inertia * angular_map

    gravity = sp.Float(p.gravity)
    potential = (
        sp.Float(p.body_mass)
        * gravity
        * (rw + body_position_heading[2])
    )
    potential += sum(
        sp.Float(p.leg_mass) * gravity * (rw + position[2])
        for position in leg_positions_heading
    )
    potential += sum(
        sp.Float(p.wheel_mass) * gravity * (rw + position[2])
        for position in wheel_positions_heading
    )
    gravity_vector = sp.Matrix([sp.diff(potential, coordinate) for coordinate in q])

    input_matrix = sp.zeros(N_Q, N_U)
    for index, wheel_velocity_map in enumerate(wheel_velocity_maps_heading):
        wheel_column = wheel_velocity_map.T * e1 / rw
        wheel_column[5 + index] += 1
        input_matrix[:, index] = wheel_column
    for index in range(2):
        input_matrix[2, 2 + index] = 1
        input_matrix[5 + index, 2 + index] = 1
        input_matrix[:, 4 + index] = sp.Matrix(
            [sp.diff(lengths[index], coordinate) for coordinate in q]
        )

    arguments = (*q, kappa)
    modules = "numpy"
    return _CompiledModel(
        mass=sp.lambdify(arguments, mass, modules=modules),
        gravity=sp.lambdify(arguments, gravity_vector, modules=modules),
        input_matrix=sp.lambdify(arguments, input_matrix, modules=modules),
        lengths=sp.lambdify(arguments, sp.Matrix(lengths), modules=modules),
        height=sp.lambdify(arguments, h, modules=modules),
        wheel_positions_heading=sp.lambdify(
            arguments,
            sp.Matrix.hstack(*wheel_positions_heading),
            modules=modules,
        ),
        wheel_velocity_maps_heading=sp.lambdify(
            arguments,
            sp.Matrix.hstack(*wheel_velocity_maps_heading),
            modules=modules,
        ),
    )


def _arguments(q: np.ndarray, kappa_g: float, *, parameters: RobotParameters = DEFAULT_ROBOT) -> tuple[float, ...]:
    q = np.asarray(q, dtype=float).reshape(N_Q)
    return (*q.tolist(), float(kappa_g))


def evaluate_mass(q: np.ndarray, kappa_g: float, *, parameters: RobotParameters = DEFAULT_ROBOT) -> np.ndarray:
    value = np.asarray(_compile_model(parameters).mass(*_arguments(q, kappa_g)), dtype=float)
    return 0.5 * (value + value.T)


def evaluate_gravity(q: np.ndarray, kappa_g: float, *, parameters: RobotParameters = DEFAULT_ROBOT) -> np.ndarray:
    return np.asarray(
        _compile_model(parameters).gravity(*_arguments(q, kappa_g)), dtype=float
    ).reshape(N_Q)


def evaluate_input_matrix(q: np.ndarray, kappa_g: float, *, parameters: RobotParameters = DEFAULT_ROBOT) -> np.ndarray:
    return np.asarray(
        _compile_model(parameters).input_matrix(*_arguments(q, kappa_g)), dtype=float
    ).reshape(N_Q, N_U)


def evaluate_lengths(q: np.ndarray, kappa_g: float, *, parameters: RobotParameters = DEFAULT_ROBOT) -> np.ndarray:
    return np.asarray(
        _compile_model(parameters).lengths(*_arguments(q, kappa_g)), dtype=float
    ).reshape(2)


def evaluate_height(q: np.ndarray, kappa_g: float, *, parameters: RobotParameters = DEFAULT_ROBOT) -> float:
    """Return the derived vertical projection for the L_bar coordinate."""
    return float(_compile_model(parameters).height(*_arguments(q, kappa_g)))


def evaluate_wheel_positions_heading(
    q: np.ndarray,
    kappa_g: float,
    *, parameters: RobotParameters = DEFAULT_ROBOT,
) -> np.ndarray:
    """Return wheel positions in Y relative to wheel-centre midpoint C."""
    return np.asarray(
        _compile_model(parameters).wheel_positions_heading(*_arguments(q, kappa_g)),
        dtype=float,
    ).reshape(3, 2)


def evaluate_wheel_velocity_maps_heading(
    q: np.ndarray,
    kappa_g: float,
    *, parameters: RobotParameters = DEFAULT_ROBOT,
) -> np.ndarray:
    """Return absolute wheel velocity components expressed in heading frame Y."""
    value = np.asarray(
        _compile_model(parameters).wheel_velocity_maps_heading(*_arguments(q, kappa_g)),
        dtype=float,
    ).reshape(3, 2 * N_Q)
    return np.stack(
        [value[:, index * N_Q : (index + 1) * N_Q] for index in range(2)]
    )


def make_trim_geometry(
    l1: float,
    l2: float,
    *,
    parameters: RobotParameters = DEFAULT_ROBOT,
    beta_1: float | None = None,
    beta_2: float | None = None,
) -> GeometryAtTrim:
    """Build a static trim directly from the two actual leg lengths."""
    L1 = float(l1)
    L2 = float(l2)
    if L1 <= 0.0 or L2 <= 0.0:
        raise ValueError("leg lengths must be positive")
    b1 = (
        compute_equilibrium_beta_from_length(L1, parameters)
        if beta_1 is None
        else float(beta_1)
    )
    b2 = (
        compute_equilibrium_beta_from_length(L2, parameters)
        if beta_2 is None
        else float(beta_2)
    )
    l_bar = (L1 + L2) / 2.0
    c1, c2 = np.cos(b1), np.cos(b2)
    kappa = (L2 * c2 - L1 * c1) / 2.0

    q = np.array([0.0, 0.0, 0.0, 0.0, l_bar, b1, b2], dtype=float)
    lengths = evaluate_lengths(q, kappa, parameters=parameters)
    height_value = evaluate_height(q, kappa, parameters=parameters)

    p = parameters
    feasible = bool(
        np.all(lengths >= p.min_leg_length - 1e-12)
        and np.all(lengths <= p.max_leg_length + 1e-12)
    )
    return GeometryAtTrim(
        l1=L1,
        l2=L2,
        kappa_g=float(kappa),
        q=q,
        leg_lengths=lengths,
        height=height_value,
        feasible=feasible,
    )


def solve_equilibrium_input(
    q: np.ndarray,
    kappa_g: float,
    *, parameters: RobotParameters = DEFAULT_ROBOT,
) -> tuple[np.ndarray, np.ndarray]:
    """最小二乘求解静态配平输入并返回残差。

    Least-squares solve the static equilibrium input ``u = B^+ g`` and return
    both the input and the residual ``B u - g``.
    """
    gravity = evaluate_gravity(q, kappa_g, parameters=parameters)
    input_matrix = evaluate_input_matrix(q, kappa_g, parameters=parameters)
    equilibrium_input, *_ = np.linalg.lstsq(input_matrix, gravity, rcond=None)
    residual = input_matrix @ equilibrium_input - gravity
    return equilibrium_input, residual


def _interleaving_permutation() -> np.ndarray:
    indices = [item for index in range(N_Q) for item in (index, N_Q + index)]
    return np.eye(N_X, dtype=float)[indices]


def linearize_static_trim(
    l1: float,
    l2: float,
    *,
    parameters: RobotParameters = DEFAULT_ROBOT,
    beta_1: float | None = None,
    beta_2: float | None = None,
    difference_step: float = 1e-6,
    require_feasible: bool = True,
    equilibrium_tolerance: float = 1e-8,
) -> StaticLinearization:
    """围绕静态配平做数值线性化，得到 (A, B) 与配平输入。

    Numerically linearize the dynamics about a static trim using central
    finite differences on the stiffness, then assemble the 14-state ``A`` /
    6-input ``B`` pair (interleaved to the canonical state order).  The fifth
    generalized coordinate is ``L_bar``; vertical height is a derived quantity.
    """
    geometry = make_trim_geometry(
        l1,
        l2,
        beta_1=beta_1,
        beta_2=beta_2,
        parameters=parameters,
    )
    if require_feasible and not geometry.feasible:
        raise ValueError(
            f"leg lengths ({l1:.9f}, {l2:.9f}) are out of range: "
            f"{geometry.leg_lengths.tolist()}"
        )
    q0, kappa = geometry.q, geometry.kappa_g
    mass = evaluate_mass(q0, kappa, parameters=parameters)
    input_matrix = evaluate_input_matrix(q0, kappa, parameters=parameters)
    equilibrium_input, equilibrium_residual = solve_equilibrium_input(q0, kappa, parameters=parameters)
    if np.linalg.norm(equilibrium_residual, ord=np.inf) > equilibrium_tolerance:
        raise ValueError(
            "specified q0 is not a static trim: residual infinity norm="
            f"{np.linalg.norm(equilibrium_residual, ord=np.inf):.6e}"
        )

    stiffness = np.empty((N_Q, N_Q), dtype=float)
    for index in range(N_Q):
        step = difference_step * max(1.0, abs(q0[index]))
        plus = q0.copy()
        minus = q0.copy()
        plus[index] += step
        minus[index] -= step
        plus_residual = (
            evaluate_gravity(plus, kappa, parameters=parameters)
            - evaluate_input_matrix(plus, kappa, parameters=parameters) @ equilibrium_input
        )
        minus_residual = (
            evaluate_gravity(minus, kappa, parameters=parameters)
            - evaluate_input_matrix(minus, kappa, parameters=parameters) @ equilibrium_input
        )
        stiffness[:, index] = (plus_residual - minus_residual) / (2 * step)

    acceleration_stiffness = -np.linalg.solve(mass, stiffness)
    acceleration_input = np.linalg.solve(mass, input_matrix)
    grouped_a = np.block(
        [
            [np.zeros((N_Q, N_Q)), np.eye(N_Q)],
            [acceleration_stiffness, np.zeros((N_Q, N_Q))],
        ]
    )
    grouped_b = np.vstack([np.zeros((N_Q, N_U)), acceleration_input])
    permutation = _interleaving_permutation()
    return StaticLinearization(
        geometry=geometry,
        equilibrium_input=equilibrium_input,
        equilibrium_residual=equilibrium_residual,
        mass=mass,
        stiffness=stiffness,
        input_matrix=input_matrix,
        A=permutation @ grouped_a @ permutation.T,
        B=permutation @ grouped_b,
    )


@dataclass(frozen=True)
class GroundModel:
    parameters: RobotParameters = DEFAULT_ROBOT

    def linearize(self, l1, l2, **kwargs):
        return linearize_static_trim(l1, l2, parameters=self.parameters, **kwargs)

    def mass(self, q, kappa_g):
        return evaluate_mass(q, kappa_g, parameters=self.parameters)
