"""RDOB deployment settings; hardware data remains in the root config."""

import numpy as np
from ..config import DEFAULT_ROBOT
from ..models.ground.config import SAMPLE_TIME

DEFAULT_RDOB_STEP_POLE = 0.992
_rate = (
    2.0
    * (1.0 - DEFAULT_RDOB_STEP_POLE)
    / (SAMPLE_TIME * (1.0 + DEFAULT_RDOB_STEP_POLE))
)
# Override individual channels here when different observer bandwidths are wanted.
DEFAULT_BANDWIDTH_HZ = tuple(float(_rate / (2.0 * np.pi)) for _ in range(6))
DEFAULT_DISTURBANCE_LIMIT_RATIO = 0.1
DEFAULT_DISTURBANCE_LIMITS = tuple(
    float(value)
    for value in DEFAULT_DISTURBANCE_LIMIT_RATIO * DEFAULT_ROBOT.input_upper_limits
)
