"""Ground equilibrium angles derived from the current physical parameters."""
import numpy as np
from ...config import DEFAULT_ROBOT, RobotParameters
from ..leg_geometry import leg_com
from . import config

def compute_equilibrium_beta_from_length(leg_length, parameters: RobotParameters = DEFAULT_ROBOT):
    length = float(leg_length)
    if not np.isfinite(length) or length <= 0:
        raise ValueError('leg_length must be positive and finite')
    lwp, xcom = leg_com(length, parameters)
    return float(np.arctan((parameters.leg_mass * xcom) /
                          (parameters.body_mass * length / 2 + parameters.leg_mass * lwp)))

def compute_equilibrium_beta(height=None, parameters: RobotParameters = DEFAULT_ROBOT):
    h = float(config.TRIM_HEIGHT if height is None else height)
    beta = compute_equilibrium_beta_from_length(h, parameters)
    for _ in range(10):
        length = h / np.cos(beta)
        next_beta = compute_equilibrium_beta_from_length(length, parameters)
        if abs(next_beta - beta) < 1e-13:
            break
        beta = next_beta
    return float(beta)

def fit_equilibrium_beta(parameters: RobotParameters = DEFAULT_ROBOT, degree=3):
    lengths = np.linspace(parameters.min_leg_length, parameters.max_leg_length, 201)
    values = [compute_equilibrium_beta_from_length(length, parameters) for length in lengths]
    return np.polynomial.polynomial.polyfit(lengths, values, degree)
