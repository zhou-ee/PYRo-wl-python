# User-editable configuration for the document-coordinate model.

from __future__ import annotations

import numpy as np

from .parameters import RobotParameters


class ConfigurationError(ValueError):
    """Raised when a required configuration value is missing or invalid."""


# Leg geometry. L is in metres.
L: float = 0.24

# L_ph(L) = p1*L^3 + p2*L^2 + p3*L + p4.
Lph_p1: float = 0.7381
Lph_p2: float = -0.7972
Lph_p3: float = 0.6240
Lph_p4: float = -0.0581

# Valid leg-length range used for fitting K(L). Edit these values if needed.
L_SWEEP_START: float = 0.12
L_SWEEP_STOP: float = 0.38
L_SWEEP_STEP: float = 0.01
K_POLY_DEGREE: int = 3


def calculate_l_ph(length: float) -> float:
    """Calculate the leg-COM-to-hip distance from the supplied cubic fit."""
    return (
        Lph_p1 * length**3
        + Lph_p2 * length**2
        + Lph_p3 * length
        + Lph_p4
    )


# Physical parameters. Lengths are in metres, masses in kg, and inertias in kg*m^2.
m_w: float = 0.57
I_w: float = 0.000596
R_w: float = 0.06
m_p: float = 1.232
m_b: float = 6.0
I_b: float = 0.086
L_c: float = 0.032
g: float = 9.81


def geometry_for_leg_length(length: float) -> tuple[float, float, float, float]:
    """Return L_ph, L_wp, COM offset d, and corrected leg inertia I_p."""
    length = float(length)
    if length <= 0.0:
        raise ConfigurationError("leg length L must be positive")

    l_ph = calculate_l_ph(length)
    l_wp = length - l_ph
    if l_ph <= 0.0 or l_wp <= 0.0:
        raise ConfigurationError(
            f"geometry fit is nonphysical at L={length:.9f}: "
            f"L_ph={l_ph:.9f}, L_wp={l_wp:.9f}"
        )

    d = 0.5 * length - l_ph
    i_p = m_p * length**2 / 12.0 + m_p * d**2
    return l_ph, l_wp, d, i_p


# Derived values at the nominal fixed length L.
L_ph, L_wp, d, I_p = geometry_for_leg_length(L)


def make_parameters_for_leg_length(length: float) -> RobotParameters:
    """Build model parameters for one variable leg length."""
    l_ph, l_wp, _, i_p = geometry_for_leg_length(length)
    return RobotParameters(
        mw=m_w,
        iw=I_w,
        rw=R_w,
        mp=m_p,
        ip=i_p,
        l_wp=l_wp,
        l_ph=l_ph,
        mb=m_b,
        ib=I_b,
        l_c=L_c,
        g=g,
    )


def make_parameters() -> RobotParameters:
    """Build model parameters at the nominal fixed length."""
    return make_parameters_for_leg_length(L)


# Linearization working point.
x_0: float = 0.0
dx_0: float = 0.0
beta_0: float = 0.1
dbeta_0: float = 0.0
gamma_0: float = 0.0
dgamma_0: float = 0.0
T_w_0: float = 0.0
T_p_0: float = 0.0
Ts: float = 0.001

# State order: [x, dx, beta, dbeta, gamma, dgamma].
Q_diag: list[float | None] = [80.0, 20.0, 500.0, 5.0, 100.0, 5.0]
R_diag: list[float | None] = [2.4, 0.1]


def _require(name: str, value: float | None) -> float:
    """Require one configured scalar."""
    if value is None:
        raise ConfigurationError(
            f"{name} is not configured; edit wheel_legged_lqr/config.py"
        )
    return float(value)


def _require_vector(
    name: str,
    values: list[float | None],
    size: int,
) -> np.ndarray:
    """Require a complete numeric vector."""
    if len(values) != size or any(value is None for value in values):
        raise ConfigurationError(
            f"{name} must contain {size} numbers; edit wheel_legged_lqr/config.py"
        )
    return np.asarray(values, dtype=float)


def make_working_state() -> np.ndarray:
    """Build the nominal state."""
    return np.array(
        [
            _require("x_0", x_0),
            _require("dx_0", dx_0),
            _require("beta_0", beta_0),
            _require("dbeta_0", dbeta_0),
            _require("gamma_0", gamma_0),
            _require("dgamma_0", dgamma_0),
        ],
        dtype=float,
    )


def make_working_input() -> np.ndarray:
    """Build the nominal input."""
    return np.array(
        [_require("T_w_0", T_w_0), _require("T_p_0", T_p_0)],
        dtype=float,
    )


def make_q() -> np.ndarray:
    """Build the state-weight matrix Q."""
    return np.diag(_require_vector("Q_diag", Q_diag, 6))


def make_r() -> np.ndarray:
    """Build the input-weight matrix R."""
    return np.diag(_require_vector("R_diag", R_diag, 2))



