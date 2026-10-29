"""Robot physical data in SI units; the only maintained hardware parameter set."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
import numpy as np

@dataclass(frozen=True)
class RobotParameters:
    wheel_mass: float = 0.57
    wheel_rolling_inertia: float = 0.000596
    wheel_radius: float = 0.06
    leg_mass: float = 1.232
    body_mass: float = 12.0 #18.9255
    hip_half_track: float = 0.158
    wheel_half_track: float = 0.20025
    body_com_from_hip: tuple[float, float, float] = (0.0, 0.0, 0.032)
    # 机体质量分布前后、左右对称；质心坐标轴沿对称方向。
    # CAD 导出的非对角惯量为误差，Ixy/Ixz/Iyz 均置零。
    # 模型导出后 * 1.5得到
    body_inertia: tuple[tuple[float, ...], ...] = (
        (0.173525859, 0.0, 0.0),
        (0.0, 0.256537589, 0.0),
        (0.0, 0.0, 0.347905209),
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
    leg_com_lwp_poly: tuple[float, ...] = (0.05806884, 0.37602793, 0.79724260, -0.73809850)
    leg_com_x_poly: tuple[float, ...] = (-0.07916397, 0.03685243, -0.12886076, 0.69820022)
    # 前后质心调度开关；False 时质心在腿部矢状平面的髋—轮连线上。
    # 沿腿方向的 leg_com_lwp_poly 仍随腿长变化，原始 x 曲线保留。
    leg_com_fore_aft_enabled: bool = True
    gas_spring_length_center: float = 0.28
    gas_spring_length_scale: float = 0.10
    gas_spring_force_poly: tuple[float, ...] = (
        119.867552440332, 37.1404981279233, -4.91246097241044, 2.70356625079196,
    )
    wheel_torque_constant_nm_per_a: float = 0.21777611
    wheel_max_current_a: float = 15.0
    hip_torque_limit_nm: float = 60.0
    leg_force_limit_n: float = 300.0

    def __post_init__(self):
        if not isinstance(self.leg_com_fore_aft_enabled, bool):
            raise ValueError('leg_com_fore_aft_enabled must be a bool')
        # Normalize sequence inputs before hashing/caching; frozen also applies to nested data.
        for name in ('body_com_from_hip', 'leg_com_lwp_poly', 'leg_com_x_poly', 'gas_spring_force_poly'):
            object.__setattr__(self, name, tuple(float(v) for v in getattr(self, name)))
        object.__setattr__(self, 'body_inertia', tuple(tuple(float(v) for v in row) for row in self.body_inertia))
        for name, value in asdict(self).items():
            if name == 'leg_com_fore_aft_enabled':
                continue
            if isinstance(value, (float, int)) and (not np.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be positive and finite')
        if self.max_leg_length <= self.min_leg_length:
            raise ValueError('max_leg_length must exceed min_leg_length')
        inertia = np.asarray(self.body_inertia)
        if inertia.shape != (3, 3) or not np.all(np.isfinite(inertia)) or not np.allclose(inertia, inertia.T):
            raise ValueError('body_inertia must be a finite symmetric 3-by-3 matrix')
        if np.min(np.linalg.eigvalsh(inertia)) <= 0:
            raise ValueError('body_inertia must be positive definite')
        if len(self.body_com_from_hip) != 3 or not np.all(np.isfinite(self.body_com_from_hip)):
            raise ValueError('body_com_from_hip must contain three finite components')
        for name in ('leg_com_lwp_poly', 'leg_com_x_poly', 'gas_spring_force_poly'):
            if len(getattr(self, name)) != 4 or not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f'{name} must contain four finite cubic coefficients')

    @property
    def effective_leg_com_x_poly(self) -> tuple[float, ...]:
        """Active fore-aft COM curve; retain the CAD curve when disabled."""
        return self.leg_com_x_poly if self.leg_com_fore_aft_enabled else (0.0,) * 4

    @property
    def body_pitch_inertia(self):
        return self.body_inertia[1][1]

    @property
    def body_com_x_from_hip(self):
        return self.body_com_from_hip[0]

    @property
    def body_com_z_from_hip(self):
        return self.body_com_from_hip[2]

    @property
    def wheel_torque_limit_nm(self):
        return self.wheel_torque_constant_nm_per_a * self.wheel_max_current_a

    @property
    def input_upper_limits(self):
        return np.array([self.wheel_torque_limit_nm] * 2 + [self.hip_torque_limit_nm] * 2 + [self.leg_force_limit_n] * 2)

    @property
    def input_lower_limits(self):
        return -self.input_upper_limits

    def snapshot(self):
        return asdict(self)

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.snapshot(), sort_keys=True, allow_nan=False).encode()).hexdigest()

DEFAULT_ROBOT = RobotParameters()
