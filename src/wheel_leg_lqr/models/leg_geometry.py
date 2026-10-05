"""COM curves, gas support and total/active input conventions."""
import numpy as np
from ..config import DEFAULT_ROBOT, RobotParameters

def leg_com(length, parameters: RobotParameters = DEFAULT_ROBOT):
    def evaluate(coefficients):
        return (coefficients[0] + coefficients[1] * length
                + coefficients[2] * length**2 + coefficients[3] * length**3)
    return evaluate(parameters.leg_com_lwp_poly), evaluate(parameters.leg_com_x_poly)

def ground_inertia_com_coefficients(parameters: RobotParameters = DEFAULT_ROBOT):
    """Keep the original ground/contact rod-inertia approximation (4 decimal places)."""
    return tuple(round(value, 4) for value in parameters.leg_com_lwp_poly)

def compute_gas_spring_force(leg_length, parameters: RobotParameters = DEFAULT_ROBOT):
    normalized = np.clip((float(leg_length) - parameters.gas_spring_length_center)
                         / parameters.gas_spring_length_scale, -1.0, 1.0)
    return float(np.polynomial.polynomial.polyval(normalized, parameters.gas_spring_force_poly))

def active_equilibrium_input(total_input, lengths, parameters: RobotParameters = DEFAULT_ROBOT):
    active = np.asarray(total_input, dtype=float).copy()
    active[4:] -= [compute_gas_spring_force(length, parameters) for length in lengths]
    return active
