"""Shared inclusive grids and explicit schedule normalization."""
from dataclasses import dataclass
import numpy as np
from ..config import DEFAULT_ROBOT

def make_leg_length_grid(start=DEFAULT_ROBOT.min_leg_length, stop=DEFAULT_ROBOT.max_leg_length, step=0.01):
    if not all(np.isfinite(v) for v in (start, stop, step)) or start <= 0 or stop <= start or step <= 0:
        raise ValueError('grid requires finite 0 < start < stop and step > 0')
    count = int(round((stop - start) / step)) + 1
    if count < 2:
        raise ValueError('step is too large for the leg-length interval')
    return np.linspace(start, stop, count)

@dataclass(frozen=True)
class LengthDomain:
    lower: float
    upper: float
    def __post_init__(self):
        if not np.isfinite(self.lower) or not np.isfinite(self.upper) or not 0 < self.lower < self.upper:
            raise ValueError('domain requires finite 0 < lower < upper')
    @property
    def center(self): return (self.lower + self.upper) / 2
    @property
    def scale(self): return (self.upper - self.lower) / 2
    def normalize(self, length): return np.clip((length - self.center) / self.scale, -1, 1)
