from __future__ import annotations
from ..models.ground.trim import compute_equilibrium_beta_from_length
from ..models.leg_geometry import compute_gas_spring_force
from ..config import DEFAULT_ROBOT, RobotParameters
from dataclasses import dataclass
import numpy as np
from ..models.ground import config
from ..models.ground.model import StaticLinearization, linearize_static_trim
from ..control.lqr import controllability_rank, discretize_state_space, solve_discrete_lqr
from .polynomial import evaluate_tensor_polynomial, fit_tensor_coefficients, tensor_design_matrix
from ..simulation.saturation import SaturationValidation, validate_saturated_closed_loop
from .domain import make_leg_length_grid


@dataclass(frozen=True)
class GainNode:
    l1: float
    l2: float
    height: float  # Derived vertical height; not an LQR state.
    linearization: StaticLinearization
    Ad: np.ndarray
    Bd: np.ndarray
    gain: np.ndarray
    closed_loop_poles: np.ndarray
    controllability_rank: int
    parameters: RobotParameters = DEFAULT_ROBOT
    settings: config.GroundSettings = config.DEFAULT_GROUND


@dataclass(frozen=True)
class GainScheduleFit:
    l1_grid: np.ndarray
    l2_grid: np.ndarray
    nodes: tuple[GainNode, ...]
    gain_coefficients: np.ndarray  # Shape: (INPUT_DIM, STATE_DIM, DEG1+1, DEG2+1)
    input_coefficients: np.ndarray  # Shape: (INPUT_DIM, DEG1+1, DEG2+1)
    fitted_gains: np.ndarray  # Shape: (N_NODES, INPUT_DIM, STATE_DIM)
    max_abs_gain_error: float
    rms_gain_error: float
    max_raw_spectral_radius: float
    max_fitted_spectral_radius: float
    saturation_validation: SaturationValidation
    saturation_validation_node_count: int


def _discretize(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return discretize_state_space(A, B, sample_time=config.SAMPLE_TIME)


def _controllability_rank(A: np.ndarray, B: np.ndarray) -> int:
    return controllability_rank(A, B)


def compute_gain_node(l1: float, l2: float, *, parameters: RobotParameters = DEFAULT_ROBOT, settings: config.GroundSettings | None = None) -> GainNode:
    """在单个 (L1, L2) 坐标处求解自平衡配平、线性化、离散化与离散 LQR。"""
    settings = settings or config.GroundSettings(q_diag=tuple(config.Q_DIAG), r_diag=tuple(config.R_DIAG))
    b1 = compute_equilibrium_beta_from_length(l1, parameters)
    b2 = compute_equilibrium_beta_from_length(l2, parameters)
    linearization = linearize_static_trim(
        l1,
        l2,
        beta_1=b1,
        beta_2=b2,
        parameters=parameters,
    )
    Ad, Bd = discretize_state_space(linearization.A, linearization.B, sample_time=settings.sample_time)
    gain, _, poles = solve_discrete_lqr(Ad, Bd, settings.make_q(), settings.make_r())
    return GainNode(
        l1=float(l1),
        l2=float(l2),
        height=float(linearization.geometry.height),
        linearization=linearization,
        Ad=Ad,
        Bd=Bd,
        gain=gain,
        closed_loop_poles=poles,
        controllability_rank=_controllability_rank(Ad, Bd),
        parameters=parameters, settings=settings,
    )


def compute_active_equilibrium_input(node: GainNode) -> np.ndarray:
    """Return motor-side trim input after passive gas-spring support.

    The linear plant and LQR gain continue to use total virtual ``F_L``.
    Only the exported motor feedforward subtracts the already-equivalent
    passive gas-spring force on each leg.
    """

    active_input = node.linearization.equilibrium_input.copy()
    active_input[4] -= compute_gas_spring_force(node.l1, node.parameters)
    active_input[5] -= compute_gas_spring_force(node.l2, node.parameters)
    return active_input


def _build_tensor_design_matrix(
    lam1: np.ndarray,
    lam2: np.ndarray,
    deg1: int = 3,
    deg2: int = 3,
) -> np.ndarray:
    """构建 L1/L2 张量积多项式设计矩阵 X，形状为 (N, (deg1+1)*(deg2+1))。"""
    return tensor_design_matrix(lam1, lam2, deg1, deg2)


def _fit_tensor_coefficients(
    lam1: np.ndarray,
    lam2: np.ndarray,
    values: np.ndarray,
    deg1: int = 3,
    deg2: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """拟合多通道 L1/L2 张量积多项式系数。

    values 形状: (N_SAMPLES, ...)
    返回 coefficients 形状: (..., deg1+1, deg2+1)
    fitted 形状: (N_SAMPLES, ...)
    """
    return fit_tensor_coefficients(lam1, lam2, values, deg1, deg2)


def evaluate_fitted_gain(
    coefficients: np.ndarray,
    l1: float,
    l2: float,
    l_nominal: float | None = None,
    l_half_range: float | None = None,
    deg1: int = 3,
    deg2: int = 3,
    *,
    parameters: RobotParameters = DEFAULT_ROBOT,
) -> np.ndarray:
    """使用嵌套秦九韶算法计算给定 (L1, L2) 处的 L1/L2 拟合增益矩阵。"""
    l_nominal = (parameters.min_leg_length + parameters.max_leg_length) / 2 if l_nominal is None else l_nominal
    l_half_range = (parameters.max_leg_length - parameters.min_leg_length) / 2 if l_half_range is None else l_half_range
    lam1 = np.clip((l1 - l_nominal) / l_half_range, -1.0, 1.0)
    lam2 = np.clip((l2 - l_nominal) / l_half_range, -1.0, 1.0)

    # coefficients: (INPUT_DIM, STATE_DIM, deg1+1, deg2+1)
    return evaluate_tensor_polynomial(coefficients, lam1, lam2)


def compute_gain_schedule(
    step: float = 0.01,
    deg1: int = 3,
    deg2: int = 3,
    *,
    exhaustive_validation: bool = False,
    parameters: RobotParameters = DEFAULT_ROBOT,
    settings: config.GroundSettings | None = None,
) -> GainScheduleFit:
    """在完整 (L1, L2) 网格上求解并拟合张量积增益调度矩阵。"""
    l1_grid = make_leg_length_grid(parameters.min_leg_length, parameters.max_leg_length, step)
    l2_grid = make_leg_length_grid(parameters.min_leg_length, parameters.max_leg_length, step)

    nodes_list: list[GainNode] = []
    lam1_list: list[float] = []
    lam2_list: list[float] = []

    l_nominal = (parameters.min_leg_length + parameters.max_leg_length) / 2
    l_half_range = (parameters.max_leg_length - parameters.min_leg_length) / 2

    for l1 in l1_grid:
        for l2 in l2_grid:
            node = compute_gain_node(l1, l2, parameters=parameters, settings=settings)
            nodes_list.append(node)
            lam1_list.append((l1 - l_nominal) / l_half_range)
            lam2_list.append((l2 - l_nominal) / l_half_range)

    nodes = tuple(nodes_list)
    lam1_arr = np.array(lam1_list, dtype=float)
    lam2_arr = np.array(lam2_list, dtype=float)

    raw_gains = np.array([node.gain for node in nodes])
    raw_inputs = np.array([compute_active_equilibrium_input(node) for node in nodes])

    gain_coefficients, fitted_gains = _fit_tensor_coefficients(
        lam1_arr, lam2_arr, raw_gains, deg1, deg2
    )
    input_coefficients, _ = _fit_tensor_coefficients(
        lam1_arr, lam2_arr, raw_inputs, deg1, deg2
    )

    fitted_radii = []
    for node, fitted_gain in zip(nodes, fitted_gains):
        closed_loop = node.Ad - node.Bd @ fitted_gain
        poles = np.linalg.eigvals(closed_loop)
        fitted_radii.append(float(np.max(np.abs(poles))))

    error = fitted_gains - raw_gains
    float32_fitted_gains = _evaluate_float32_fitted_gains(
        nodes, gain_coefficients, deg1, deg2
    )
    if exhaustive_validation:
        validation_indices = np.arange(len(nodes), dtype=int)
    else:
        validation_indices = _representative_validation_indices(l1_grid, l2_grid)
    validation_nodes = tuple(nodes[index] for index in validation_indices)
    validation_gains = float32_fitted_gains[validation_indices]
    saturation_validation = _validate_fitted_saturation(
        validation_nodes,
        validation_gains,
    )
    if not saturation_validation.passed:
        raise ValueError(
            "saturated closed-loop recovery failed at "
            f"{saturation_validation.worst_case}"
        )
    return GainScheduleFit(
        l1_grid=l1_grid,
        l2_grid=l2_grid,
        nodes=nodes,
        gain_coefficients=gain_coefficients,
        input_coefficients=input_coefficients,
        fitted_gains=fitted_gains,
        max_abs_gain_error=float(np.max(np.abs(error))),
        rms_gain_error=float(np.sqrt(np.mean(error**2))),
        max_raw_spectral_radius=max(
            float(np.max(np.abs(node.closed_loop_poles))) for node in nodes
        ),
        max_fitted_spectral_radius=max(fitted_radii),
        saturation_validation=saturation_validation,
        saturation_validation_node_count=len(validation_nodes),
    )


def _representative_validation_indices(
    l1_grid: np.ndarray,
    l2_grid: np.ndarray,
) -> np.ndarray:
    """Return a 5 x 5 spread of nodes for the slow saturation simulations.

    The selected coordinates include both operating-envelope boundaries and
    three interior slices.  This samples asymmetry in both L1 and L2 while
    keeping the default workflow substantially faster than exhaustive
    per-node simulation.
    """
    positions_1 = np.linspace(0, len(l1_grid) - 1, num=min(5, len(l1_grid)), dtype=int)
    positions_2 = np.linspace(0, len(l2_grid) - 1, num=min(5, len(l2_grid)), dtype=int)
    indices = {
        index_1 * len(l2_grid) + index_2
        for index_1 in positions_1
        for index_2 in positions_2
    }
    return np.asarray(sorted(indices), dtype=int)


def _validate_fitted_saturation(
    nodes: tuple[GainNode, ...], fitted_gains: np.ndarray
) -> SaturationValidation:
    """Validate the gain actually exported by the L1/L2 polynomial fit."""
    max_state_abs = 0.0
    max_final_state_abs = 0.0
    max_commanded_input_abs = 0.0
    total_saturation_steps = 0
    passed = True
    worst_case = "none"
    for node, gain in zip(nodes, fitted_gains):
        validation = validate_saturated_closed_loop(
            node.Ad,
            node.Bd,
            gain,
            node.linearization.equilibrium_input,
            cases=config.default_initial_errors(), sample_time=node.settings.sample_time,
            lower_limits=node.parameters.input_lower_limits, upper_limits=node.parameters.input_upper_limits,
        )
        max_state_abs = max(max_state_abs, validation.max_state_abs)
        max_final_state_abs = max(max_final_state_abs, validation.max_final_state_abs)
        max_commanded_input_abs = max(
            max_commanded_input_abs, validation.max_commanded_input_abs
        )
        total_saturation_steps += validation.saturation_steps
        if not validation.passed:
            passed = False
            worst_case = (
                f"L1={node.l1:.6f}, L2={node.l2:.6f}: "
                f"{validation.worst_case}"
            )
    return SaturationValidation(
        passed=passed,
        max_state_abs=max_state_abs,
        max_final_state_abs=max_final_state_abs,
        max_commanded_input_abs=max_commanded_input_abs,
        saturation_steps=total_saturation_steps,
        worst_case=worst_case,
    )


def _evaluate_float32_fitted_gains(
    nodes: tuple[GainNode, ...],
    coefficients: np.ndarray,
    deg1: int,
    deg2: int,
) -> np.ndarray:
    """Evaluate the same float32 tensor polynomial used by the header."""
    coefficients = np.asarray(coefficients, dtype=np.float32)
    gains = []
    for node in nodes:
        lam1 = np.float32(np.clip((node.l1 - (node.parameters.min_leg_length + node.parameters.max_leg_length) / 2) / ((node.parameters.max_leg_length - node.parameters.min_leg_length) / 2), -1.0, 1.0))
        lam2 = np.float32(np.clip((node.l2 - (node.parameters.min_leg_length + node.parameters.max_leg_length) / 2) / ((node.parameters.max_leg_length - node.parameters.min_leg_length) / 2), -1.0, 1.0))
        p_powers = np.array([lam1**p for p in range(deg1 + 1)], dtype=np.float32)
        q_powers = np.array([lam2**q for q in range(deg2 + 1)], dtype=np.float32)
        gain = np.einsum("ijpq,p,q->ij", coefficients, p_powers, q_powers)
        gains.append(gain.astype(float))
    return np.asarray(gains, dtype=float)


def _float32_fit_metrics(
    result: GainScheduleFit,
) -> tuple[float, float]:
    fitted_float = result.gain_coefficients.astype(np.float32)
    parameters = result.nodes[0].parameters
    l_nominal = (parameters.min_leg_length + parameters.max_leg_length) / 2
    l_half_range = (parameters.max_leg_length - parameters.min_leg_length) / 2
    deg1 = result.gain_coefficients.shape[2] - 1
    deg2 = result.gain_coefficients.shape[3] - 1

    radii = []
    gain_errors = []
    for node in result.nodes:
        lam1 = np.float32(
            np.clip((node.l1 - l_nominal) / l_half_range, -1.0, 1.0)
        )
        lam2 = np.float32(
            np.clip((node.l2 - l_nominal) / l_half_range, -1.0, 1.0)
        )
        p_pow = np.array([lam1**p for p in range(deg1 + 1)], dtype=np.float32)
        q_pow = np.array([lam2**q for q in range(deg2 + 1)], dtype=np.float32)
        gain_f32 = np.einsum("ijpq,p,q->ij", fitted_float, p_pow, q_pow)

        closed_loop = (
            node.Ad.astype(np.float32)
            - node.Bd.astype(np.float32) @ gain_f32
        )
        poles = np.linalg.eigvals(closed_loop)
        radii.append(float(np.max(np.abs(poles))))
        gain_errors.append(
            float(np.max(np.abs(gain_f32.astype(float) - node.gain)))
        )

    return max(radii), max(gain_errors)
