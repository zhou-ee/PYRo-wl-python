"""Ground state layout and controller settings; hardware values live in root config."""
from dataclasses import dataclass
import numpy as np
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


SAMPLE_TIME = 0.001
TRIM_HEIGHT = 0.28
ANALYSIS_HEIGHT = 0.24
OPTIMIZER_INITIAL_SWING_ANGLE = -1.0

Q_DIAG = np.array(
    [80.0, 25.0,
     40.0, 8.0,
     400.0, 25.0,
     500.0, 15.0,
     3000.0, 12.0,
     120.0, 6.0,
     120.0, 6.0],
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


def make_q():
    return np.diag(Q_DIAG)

def make_r():
    return np.diag(R_DIAG)

def default_initial_errors():
    cases = []
    symmetric = np.zeros(N_X); symmetric[[10, 12]] = 0.5
    cases.append(('symmetric_leg_swing', symmetric))
    differential = np.zeros(N_X); differential[10] = 0.5; differential[12] = -0.5
    cases.append(('differential_leg_swing', differential))
    combined = np.zeros(N_X)
    combined[[0, 4, 6, 8, 10, 12]] = [0.10, np.deg2rad(5), np.deg2rad(5), 0.02, 0.25, -0.25]
    cases.append(('combined_body_and_leg_error', combined))
    return tuple(cases)

@dataclass(frozen=True)
class GroundSettings:
    sample_time: float = SAMPLE_TIME
    q_diag: tuple[float, ...] = tuple(Q_DIAG)
    r_diag: tuple[float, ...] = tuple(R_DIAG)

    def __post_init__(self):
        object.__setattr__(self, 'q_diag', tuple(self.q_diag))
        object.__setattr__(self, 'r_diag', tuple(self.r_diag))
        if not np.isfinite(self.sample_time) or self.sample_time <= 0:
            raise ValueError('sample_time must be positive and finite')
        for name, size in (('q_diag', N_X), ('r_diag', N_U)):
            value = np.asarray(getattr(self, name))
            if value.shape != (size,) or not np.all(np.isfinite(value)) or np.any(value <= 0):
                raise ValueError(f'{name} must contain {size} positive finite values')

    def make_q(self): return np.diag(self.q_diag)
    def make_r(self): return np.diag(self.r_diag)

DEFAULT_GROUND = GroundSettings()
