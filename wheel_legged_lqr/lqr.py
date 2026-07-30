# 主计算流程 / Main calculation pipeline

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .control_tools import LQRResult, controllability_rank, discrete_lqr, discretize_zoh
from .linearization import linearize
from .parameters import RobotParameters

@dataclass(frozen=True)
class LQRCalculation:
    """All calculated matrices / """

    parameters: RobotParameters
    state: np.ndarray
    inputs: np.ndarray
    q: np.ndarray
    r: np.ndarray
    sample_time: float
    A: np.ndarray
    B: np.ndarray
    Ad: np.ndarray
    Bd: np.ndarray
    lqr: LQRResult
    controllability_rank: int

def compute_lqr(
    *,
    beta: float | None = None,
    gamma: float | None = None,
    sample_time: float | None = None,
    parameters: RobotParameters | None = None,
    state: np.ndarray | None = None,
    inputs: np.ndarray | None = None,
    q: np.ndarray | None = None,
    r: np.ndarray | None = None,
) -> LQRCalculation:
    """Compute the document-coordinate model / """
    physical = config.make_parameters() if parameters is None else parameters
    ts = config._require("Ts", config.Ts) if sample_time is None else sample_time
    q_value = config.make_q() if q is None else np.asarray(q, dtype=float)
    r_value = config.make_r() if r is None else np.asarray(r, dtype=float)
    state_value = config.make_working_state() if state is None else np.asarray(state, dtype=float).reshape(6)
    input_value = config.make_working_input() if inputs is None else np.asarray(inputs, dtype=float).reshape(2)
    if beta is not None:
        state_value[2] = beta
    if gamma is not None:
        state_value[4] = gamma

    # Calculate continuous model
    A, B = linearize(state_value, input_value, physical)
    # Use scipy cont2discrete /  scipy cont2discrete 
    Ad, Bd = discretize_zoh(A, B, ts)
    # Use python-control dlqr /  python-control dlqr  LQR
    lqr = discrete_lqr(Ad, Bd, q_value, r_value)
    rank = controllability_rank(Ad, Bd)
    return LQRCalculation(
        parameters=physical, state=state_value, inputs=input_value,
        q=q_value, r=r_value, sample_time=ts,
        A=A, B=B, Ad=Ad, Bd=Bd, lqr=lqr,
        controllability_rank=rank,
    )
