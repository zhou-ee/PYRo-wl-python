# 物理参数数据类 / Physical-parameter data class

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class RobotParameters:
    """Physical parameters in SI units /  SI """

    mw: float
    iw: float
    rw: float
    mp: float
    ip: float
    l_wp: float
    l_ph: float
    mb: float
    ib: float
    l_c: float
    g: float

    def __post_init__(self) -> None:
        """Validate values / """
        values = (self.mw, self.iw, self.rw, self.mp, self.ip, self.l_wp, self.l_ph, self.mb, self.ib, self.l_c, self.g)
        if any(value <= 0.0 for value in values):
            raise ValueError("all physical parameters must be positive")

    @property
    def leg_length(self) -> float:
        """Return L_wp + L_ph / """
        return self.l_wp + self.l_ph
