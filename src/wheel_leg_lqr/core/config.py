"""数值配置：固定平均腿长双轮腿 LQR 扫描的常量与物理参数。

Numeric configuration (constants and physical parameters) for the fixed-mean-
leg-length double-leg LQR sweep.  This module is the single source of truth for state /
input dimensions, weighting matrices, trim conditions, and scheduling bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# 状态/输入维度（由 14 维状态、6 维输入、7 维广义坐标收敛到此处）。
# State / input dimensions: 14 states, 6 inputs, 7 generalized coordinates.
N_Q = 7
N_X = 14
N_U = 6
STATE_NAMES = (
    "x",
    "dx",
    "psi",
    "dpsi",
    "theta",
    "dtheta",
    "phi",
    "dphi",
    "l_bar",
    "dl_bar",
    "beta_1",
    "dbeta_1",
    "beta_2",
    "dbeta_2",
)
INPUT_NAMES = ("T_w1", "T_w2", "T_p1", "T_p2", "f_l1", "f_l2")


@dataclass(frozen=True)
class DoubleLegParameters:
    """Physical parameters in SI units."""

    wheel_mass: float = 0.57
    wheel_rolling_inertia: float = 0.000596
    wheel_radius: float = 0.06
    leg_mass: float = 1.232
    body_mass: float = 18.9255  # 12.0
    hip_half_track: float = 0.158  # 0.147
    wheel_half_track: float = 0.20025
    body_com_from_hip: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.032], dtype=float)
    )
    body_inertia: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.418395663, 0.0, -0.026043600],
                [0.0, 0.582892610, 0.0],
                [-0.026043600, 0.0, 0.451416848],
            ],
            dtype=float,
        )
    )
    # body_inertia: np.ndarray = field(
    #     default_factory=lambda: np.array(
    #         [
    #             [0.173525859, 0.0, 0.011609366],
    #             [0.0, 0.256537589, 0.0],
    #             [0.011609366, 0.0, 0.347905209],
    #         ],
    #         dtype=float,
    #     )
    # )
    gravity: float = 9.81
    min_leg_length: float = 0.18
    max_leg_length: float = 0.38

    def __post_init__(self) -> None:
        positive = (
            self.wheel_mass,
            self.wheel_rolling_inertia,
            self.wheel_radius,
            self.leg_mass,
            self.body_mass,
            self.hip_half_track,
            self.wheel_half_track,
            self.gravity,
            self.min_leg_length,
            self.max_leg_length,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("physical scalar parameters must be positive")
        if self.max_leg_length <= self.min_leg_length:
            raise ValueError("max_leg_length must exceed min_leg_length")
        inertia = np.asarray(self.body_inertia, dtype=float)
        if inertia.shape != (3, 3) or not np.allclose(inertia, inertia.T):
            raise ValueError("body_inertia must be a symmetric 3-by-3 matrix")
        if np.min(np.linalg.eigvalsh(inertia)) <= 0.0:
            raise ValueError("body_inertia must be positive definite")
        if np.asarray(self.body_com_from_hip).shape != (3,):
            raise ValueError("body_com_from_hip must contain three components")


PARAMETERS = DoubleLegParameters()

# 腿部质心偏移多项式拟合系数 (来自清洗后重心.xlsx，升幂: c0 + c1*L + c2*L^2 + c3*L^3)
LEG_COM_LWP_POLY: tuple[float, float, float, float] = (
    0.05806884,
    0.37602793,
    0.79724260,
    -0.73809850,
)
LEG_COM_X_POLY: tuple[float, float, float, float] = (
    -0.07916397,
    0.03685243,
    -0.12886076,
    0.69820022,
)
LEG_COM_ALPHA_POLY: tuple[float, float, float, float] = (
    -0.92800828,
    3.48715945,
    -5.75430644,
    4.92807256,
)
LEG_COM_BETA_EQ_POLY: tuple[float, float, float, float] = (
    -0.20136658,
    1.12374336,
    -2.65906096,
    2.49879461,
)


def compute_equilibrium_beta_from_length(leg_length: float) -> float:
    """计算给定腿长 L 下的静力学自平衡摆角 beta_eq(L) (rad)。

    Static balance: tan(beta_eq) = (m_p * x_com) / (m_b * L / 2 + m_p * L_wp)
    """
    length = float(leg_length)
    lwp = (
        LEG_COM_LWP_POLY[0]
        + LEG_COM_LWP_POLY[1] * length
        + LEG_COM_LWP_POLY[2] * length**2
        + LEG_COM_LWP_POLY[3] * length**3
    )
    xcom = (
        LEG_COM_X_POLY[0]
        + LEG_COM_X_POLY[1] * length
        + LEG_COM_X_POLY[2] * length**2
        + LEG_COM_X_POLY[3] * length**3
    )
    mp = PARAMETERS.leg_mass
    mb = PARAMETERS.body_mass
    return float(np.arctan((mp * xcom) / (mb * length / 2.0 + mp * lwp)))


def compute_equilibrium_beta(height: float = 0.28) -> float:
    """计算给定目标竖直高度 h 下的自洽静力学平衡摆角 beta_eq (rad)。

    Coupled solve: L = h / cos(beta), tan(beta) = (m_p * x_com(L)) / (m_b * L / 2 + m_p * L_wp(L)).
    """
    h = float(height)
    # Fixed-point iteration (converges within 4-5 steps to machine precision)
    beta = compute_equilibrium_beta_from_length(h)
    for _ in range(10):
        length = h / np.cos(beta)
        beta_next = compute_equilibrium_beta_from_length(length)
        if abs(beta_next - beta) < 1e-13:
            break
        beta = beta_next
    return float(beta)


TRIM_HEIGHT = 0.28
# 基于实际非共线腿部质心计算标称高度 0.28m 下的静态自平衡摆角
TRIM_BETA_1 = compute_equilibrium_beta(TRIM_HEIGHT)
TRIM_BETA_2 = compute_equilibrium_beta(TRIM_HEIGHT)
# The controlled fifth generalized coordinate is the mean actual leg length.
# At the symmetric nominal trim, L_bar = h / cos(beta).
TRIM_L_BAR = float(TRIM_HEIGHT / np.cos(TRIM_BETA_1))
SAMPLE_TIME = 0.001

POLYNOMIAL_DEGREE = 3

# Interleaved state order: [x, dx, psi, dpsi, theta, dtheta, phi, dphi,
#                           L_bar, dL_bar, beta_1, dbeta_1, beta_2, dbeta_2].
# Q_DIAG = np.array(
#     [
#         60.0,
#         25.0,  # x, dx       (前向位移与速度：适度降低刚度，避免轮力矩超限)
#         40.0,
#         8.0,  # psi, dpsi   (航向偏航与角速度：保持平稳巡航)
#         400.0,
#         25.0,  # theta, dtheta (机体俯仰与角速度：大幅提高阻尼，死死按住抬头后仰)
#         800.0,
#         25.0,  # phi, dphi   (横滚与侧倾角速度：维持横坡高抗倾刚度)
#         200.0,
#         12.0,  # L_bar, dL_bar (平均腿长与速度：抑制腿长漂移)
#         120.0,
#         6.0,  # beta_1, dbeta_1 (左腿摆角与速度：适度软化，降低髋关节冲击)
#         120.0,
#         6.0,  # beta_2, dbeta_2 (右腿摆角与速度：同上)
#     ],
#     dtype=float,
# )


Q_DIAG = np.array(
    [
        60.0,
        25.0,  # x, dx       (前向位移与速度：适度降低刚度，避免轮力矩超限)
        40.0,
        8.0,  # psi, dpsi   (航向偏航与角速度：保持平稳巡航)
        400.0,
        25.0,  # theta, dtheta (机体俯仰与角速度：大幅提高阻尼，死死按住抬头后仰)
        1200.0,
        25.0,  # phi, dphi   (横滚与侧倾角速度：维持横坡高抗倾刚度)
        280.0,
        10.0,  # L_bar, dL_bar (平均腿长与速度：抑制腿长漂移)
        120.0,
        6.0,  # beta_1, dbeta_1 (左腿摆角与速度：适度软化，降低髋关节冲击)
        120.0,
        6.0,  # beta_2, dbeta_2 (右腿摆角与速度：同上)
    ],
    dtype=float,
)

R_DIAG = np.array(
    [
        2.8,
        2.8,  # T_w1, T_w2 (驱动轮力矩：小幅提高惩罚至 2.8，抑制饱和)
        0.1,
        0.1,  # T_p1, T_p2 (髋摆动力矩：保持 0.1，充分发挥大扭矩电机能力)
        0.006,
        0.006,  # f_l1, f_l2 (腿部伸缩推力：设为 0.03，平衡顶升刚度与能耗)
    ],
    dtype=float,
)


WHEEL_TORQUE_CONSTANT_NM_PER_A = 0.21777611
WHEEL_MAX_CURRENT_A = 15.0
WHEEL_TORQUE_LIMIT_NM = (
    WHEEL_TORQUE_CONSTANT_NM_PER_A * WHEEL_MAX_CURRENT_A
)

INPUT_LOWER_LIMITS = np.array(
    [
        -WHEEL_TORQUE_LIMIT_NM,
        -WHEEL_TORQUE_LIMIT_NM,
        -60.0,
        -60.0,
        -300.0,
        -300.0,
    ],
    dtype=float,
)
INPUT_UPPER_LIMITS = -INPUT_LOWER_LIMITS


def make_q() -> np.ndarray:
    return np.diag(Q_DIAG)


def make_r() -> np.ndarray:
    return np.diag(R_DIAG)
