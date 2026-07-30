# 控制算法测试 / Control tests

import numpy as np

from wheel_legged_lqr.control_tools import controllability_rank, discrete_lqr, discretize_zoh
from wheel_legged_lqr.linearization import linearize
from wheel_legged_lqr.lqr import compute_lqr

from test_support import make_test_parameters

def test_linearization_dimensions_and_controllability():
    #  A/B  / Check A/B and controllability.
    params = make_test_parameters()
    A, B = linearize(np.zeros(6), np.zeros(2), params)
    assert A.shape == (6, 6)
    assert B.shape == (6, 2)
    assert controllability_rank(A, B) == 6

def test_library_discretization_and_lqr():
    #  scipy  python-control / Use scipy and python-control.
    params = make_test_parameters()
    A, B = linearize(np.zeros(6), np.zeros(2), params)
    Ad, Bd = discretize_zoh(A, B, 0.001)
    Q = np.diag([80.0, 20.0, 200.0, 5.0, 500.0, 5.0])
    R = np.diag([2.0, 0.1])
    result = discrete_lqr(Ad, Bd, Q, R)
    assert result.K.shape == (2, 6)
    assert np.all(np.isfinite(result.K))
    assert result.stable

def test_config_builds_configured_user_parameters():
    # The production configuration contains the current robot parameters.
    from wheel_legged_lqr.config import I_p, L_ph, L_wp, make_parameters

    params = make_parameters()
    assert params.mb == 6.0
    assert params.ib == 0.086
    assert params.ip == I_p
    assert abs(params.l_wp - L_wp) < 1e-12
    assert abs(params.l_ph - L_ph) < 1e-12

def test_compute_lqr_with_explicit_test_values():
    #  / Pipeline accepts explicit parameters.
    params = make_test_parameters()
    result = compute_lqr(
        beta=0.0, gamma=0.0, sample_time=0.001, parameters=params,
        state=np.zeros(6), inputs=np.zeros(2),
        q=np.diag([80.0, 20.0, 200.0, 5.0, 500.0, 5.0]),
        r=np.diag([2.0, 0.1]),
    )
    assert result.parameters.leg_length == 0.24
    assert result.controllability_rank == 6
    assert result.lqr.stable
