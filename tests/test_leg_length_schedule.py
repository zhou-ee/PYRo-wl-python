import numpy as np
from wheel_leg_lqr import numeric_config as config
from wheel_leg_lqr.gain_schedule import (
    _float32_fit_metrics,
    compute_gain_node,
    compute_gain_schedule,
    evaluate_fitted_gain,
    render_header,
)


def test_single_node_is_controllable_and_dlqr_stable():
    # Test symmetric nominal node
    node = compute_gain_node(0.28, 0.28)
    assert node.controllability_rank == 14
    assert np.max(np.abs(node.closed_loop_poles)) < 1.0
    np.testing.assert_allclose(
        np.mean(node.linearization.geometry.leg_lengths),
        node.linearization.geometry.q[4],
        atol=1e-12,
    )

    # Test asymmetric extreme node (L1 min, L2 max)
    extreme_node = compute_gain_node(0.18, 0.38)
    assert extreme_node.controllability_rank == 14
    assert np.max(np.abs(extreme_node.closed_loop_poles)) < 1.0
    np.testing.assert_allclose(
        np.mean(extreme_node.linearization.geometry.leg_lengths),
        extreme_node.linearization.geometry.q[4],
        atol=1e-12,
    )


def test_gain_schedule_fit_accuracy_and_float32_stability():
    result = compute_gain_schedule(step=0.02)  # 11 x 11 = 121 nodes
    assert len(result.nodes) == 121
    assert result.max_raw_spectral_radius < 1.0
    assert result.max_fitted_spectral_radius < 1.0
    assert result.rms_gain_error < 0.02
    assert result.saturation_validation_node_count == 25

    max_float_radius, max_float_gain_error = _float32_fit_metrics(result)
    assert max_float_radius < 1.0
    assert max_float_gain_error < 1.0


def test_left_right_symmetry_at_equal_leg_lengths():
    # When L1 == L2, wheel drive torques on longitudinal velocity must be identical
    # and on yaw velocity must be opposite
    K_sym = evaluate_fitted_gain(
        compute_gain_schedule(step=0.02).gain_coefficients,
        0.28,
        0.28,
    )
    # Tw1 vs Tw2 on state 'x' (index 0)
    np.testing.assert_allclose(K_sym[0, 0], K_sym[1, 0], rtol=1e-3, atol=1e-3)
    # Tw1 vs Tw2 on state 'psi' (index 2)
    np.testing.assert_allclose(K_sym[0, 2], -K_sym[1, 2], rtol=1e-3, atol=1e-3)


def test_header_rendering_format():
    result = compute_gain_schedule(step=0.04)  # coarse grid for quick test
    header_text = render_header(result)
    assert "#ifndef __LQR_COEF_H__" in header_text
    assert '#include "stdint.h"' in header_text
    assert "constexpr uint32_t K_POLY_DEGREE" in header_text
    assert "constexpr float K_POLY_COEF" in header_text
    assert "constexpr float FL_U0_POLY_COEF" in header_text
    assert "constexpr float BETA_TRIM_POLY_COEF" in header_text
    assert "namespace lqr_state_def" in header_text
    assert "namespace lqr_input_def" in header_text
