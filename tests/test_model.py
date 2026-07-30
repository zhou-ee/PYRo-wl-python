# 动力学测试 / Dynamics tests

import numpy as np

from wheel_legged_lqr.dynamics import accelerations, state_derivative

from test_support import make_test_parameters

def test_zero_state_zero_input_is_static():
    #  / Zero state and input remain static.
    params = make_test_parameters()
    derivative = state_derivative(np.zeros(6), np.zeros(2), params)
    np.testing.assert_allclose(derivative, np.zeros(6), atol=1e-12)

def test_acceleration_shape_and_finiteness():
    #  / Return three generalized accelerations.
    params = make_test_parameters()
    value = accelerations(np.zeros(6), np.zeros(2), params)
    assert value.shape == (3,)
    assert np.all(np.isfinite(value))


def test_corrected_tp_sign_in_leg_and_body_equations():
    # At the zero configuration, positive T_p enters both corrected equations
    # on the right-hand side, so the residual at zero acceleration is -T_p.
    from wheel_legged_lqr.dynamics import _residual

    params = make_test_parameters()
    residual = _residual(
        np.zeros(3),
        np.zeros(6),
        np.array([0.0, 1.0]),
        params,
    )
    np.testing.assert_allclose(residual[1:], [-1.0, -1.0], atol=1e-12)
