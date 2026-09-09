from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from wheel_leg_lqr.numeric_config import N_U, N_X, SAMPLE_TIME
from wheel_leg_lqr.scheduling.leg_length_schedule import (
    compute_gain_schedule,
    render_active_fl_u0_block,
)
from wheel_leg_lqr.scheduling.leso_schedule import (
    G_SCHEDULED_COLUMN_OFFSET,
    INPUT_EXCHANGE,
    STATE_EXCHANGE,
    _compensation_filter_stage_alpha,
    _compensation_filter_state_space,
    compute_leso_schedule,
    render_combined_coef_header,
    render_firmware_coefficients,
    render_firmware_fit_block,
    save_result,
)


@pytest.fixture(scope="module")
def coarse_schedule():
    return compute_leso_schedule(step=0.02, holdout_random_count=8)


def test_schedule_meets_float32_holdout_and_observer_budgets(coarse_schedule):
    result = coarse_schedule
    np.testing.assert_array_equal(
        result.state_bandwidths_hz,
        np.asarray([100.0, 100.0, 80.0, 80.0, 25.0, 25.0]),
    )
    np.testing.assert_array_equal(
        result.disturbance_bandwidths_hz,
        np.asarray([2.4, 2.4, 2.4, 2.4, 1.2, 1.2]),
    )
    assert result.state_poles[0] == pytest.approx(0.533488091)
    assert result.state_poles[2] == pytest.approx(0.604922562)
    assert result.state_poles[4] == pytest.approx(0.854635999)
    assert result.disturbance_poles[0] == pytest.approx(0.985033484)
    assert result.disturbance_poles[4] == pytest.approx(0.992488531)
    assert result.unmatched_state_pole == pytest.approx(0.854635999)
    assert result.disturbance_leak_bandwidth_hz == pytest.approx(0.2)
    assert result.disturbance_retention_pole == pytest.approx(0.998744152)
    np.testing.assert_allclose(
        result.disturbance_limits,
        np.asarray([0.32666417, 0.32666417, 6.0, 6.0, 30.0, 30.0]),
    )
    np.testing.assert_allclose(
        result.disturbance_maximum_increments,
        np.asarray(
            [
                0.004889024,
                0.004889024,
                0.089799099,
                0.089799099,
                0.225344077,
                0.225344077,
            ]
        ),
        rtol=2.0e-7,
    )
    np.testing.assert_allclose(
        result.constant_disturbance_tracking_fraction,
        np.asarray(
            [
                0.91362339,
                0.91362339,
                0.91317749,
                0.91317749,
                0.82561438,
                0.82561438,
            ]
        ),
        rtol=2.0e-7,
    )
    assert result.compensation_cutoff_hz == pytest.approx(15.0)
    assert result.fl_common_compensation_gain == pytest.approx(0.10)
    assert result.fl_differential_compensation_gain == pytest.approx(0.30)
    np.testing.assert_allclose(
        result.fl_compensation_matrix[4:, 4:],
        np.asarray([[0.2, -0.1], [-0.1, 0.2]]),
    )
    assert result.g_schedule.total_degree == 4
    assert result.h_schedule.total_degree == 4
    assert result.ld_schedule.total_degree == 4
    assert (result.g_schedule.even_rank, result.g_schedule.odd_rank) == (3, 3)
    assert (result.h_schedule.even_rank, result.h_schedule.odd_rank) == (3, 3)
    assert (result.ld_schedule.even_rank, result.ld_schedule.odd_rank) == (3, 3)
    assert result.max_fixed_structure_residual < 1.0e-10
    assert result.max_training_symmetry_residual < 2.0e-10
    assert result.max_holdout_symmetry_residual < 2.0e-10
    assert result.g_metrics.float32_holdout_max_relative_error < 1.0e-3
    assert result.h_metrics.float32_holdout_max_relative_error < 1.0e-3
    assert result.ld_metrics.float32_holdout_max_relative_error < 5.0e-3
    assert result.lx_metrics.float32_holdout_max_relative_error < 1.0e-2
    assert result.observer_metrics.float32_holdout_max_radius < 0.995
    assert result.observer_metrics.minimum_float32_margin > 0.005
    assert result.downstream_metrics.float32_train_max_radius < 0.9995
    assert result.downstream_metrics.float32_holdout_max_radius < 0.9995
    assert result.downstream_metrics.minimum_float32_margin > 5.0e-4


def test_third_order_compensation_filter_keeps_15_hz_cutoff():
    cutoff_hz = 15.0
    alpha = np.float32(_compensation_filter_stage_alpha(cutoff_hz))
    keep = np.float32(1.0) - alpha
    omega = 2.0 * np.pi * cutoff_hz * SAMPLE_TIME
    stage_magnitude = alpha / np.sqrt(
        1.0 + keep * keep - 2.0 * keep * np.cos(omega)
    )
    assert stage_magnitude**3 == pytest.approx(
        1.0 / np.sqrt(2.0), rel=2.0e-6
    )


def test_third_order_filter_model_matches_sequential_firmware_update():
    alpha = np.float32(_compensation_filter_stage_alpha(15.0))
    state_matrix, input_matrix, output_matrix, output_input = (
        _compensation_filter_state_space(alpha)
    )
    rng = np.random.default_rng(20260909)
    stages = rng.normal(size=(3, N_U))
    disturbance = rng.normal(size=N_U)
    expected = stages.copy()
    for stage in range(3):
        source = disturbance if stage == 0 else expected[stage - 1]
        expected[stage] += alpha * (source - expected[stage])

    flat_state = stages.reshape(-1)
    modeled = state_matrix @ flat_state + input_matrix @ disturbance
    output = output_matrix @ flat_state + output_input @ disturbance
    np.testing.assert_allclose(modeled.reshape(3, N_U), expected)
    np.testing.assert_allclose(output, expected[-1])


def test_schedule_preserves_fixed_structure_and_exchange_off_grid(
    coarse_schedule,
):
    result = coarse_schedule
    l1, l2 = 0.217, 0.361
    direct = result.matrices(l1, l2, dtype=np.float32, firmware_order=True)
    exchanged = result.matrices(l2, l1, dtype=np.float32, firmware_order=True)
    transforms = (
        (STATE_EXCHANGE, STATE_EXCHANGE),
        (STATE_EXCHANGE, INPUT_EXCHANGE),
        (STATE_EXCHANGE, STATE_EXCHANGE),
        (INPUT_EXCHANGE, STATE_EXCHANGE),
    )
    for direct_matrix, exchanged_matrix, (row_map, column_map) in zip(
        direct, exchanged, transforms
    ):
        np.testing.assert_allclose(
            exchanged_matrix,
            row_map @ direct_matrix @ column_map.T,
            atol=2.0e-6,
            rtol=2.0e-6,
        )

    state_matrix = direct[0]
    dynamic = state_matrix - np.eye(N_X, dtype=np.float32)
    expected_prefix = np.zeros(
        (N_X, G_SCHEDULED_COLUMN_OFFSET), dtype=np.float32
    )
    expected_prefix[0, 1] = SAMPLE_TIME
    expected_prefix[2, 3] = SAMPLE_TIME
    np.testing.assert_array_equal(
        dynamic[:, :G_SCHEDULED_COLUMN_OFFSET], expected_prefix
    )


def test_fused_heterogeneous_state_correction_matches_dense_lx(
    coarse_schedule,
):
    result = coarse_schedule
    state_matrix, input_matrix, state_gain, disturbance_gain = result.matrices(
        0.217, 0.361
    )
    rng = np.random.default_rng(20260909)
    equilibrium = rng.normal(scale=0.1, size=N_X)
    estimate = rng.normal(scale=0.1, size=N_X)
    residual = rng.normal(scale=0.01, size=N_X)
    applied_delta = rng.normal(scale=0.2, size=N_U)
    disturbance = rng.normal(scale=0.2, size=N_U)
    disturbance_residual = disturbance_gain @ residual

    dense = (
        equilibrium
        + state_matrix @ (estimate - equilibrium)
        + input_matrix @ (applied_delta + disturbance)
        + state_gain @ residual
    )
    fused = (
        equilibrium
        + state_matrix @ (estimate - equilibrium + residual)
        - result.unmatched_state_pole * residual
        + input_matrix
        @ (
            applied_delta
            + disturbance
            - result.matched_correction_scales * disturbance_residual
        )
    )
    np.testing.assert_allclose(fused, dense, atol=2.0e-13, rtol=2.0e-13)


def test_generated_firmware_block_and_artifacts_are_deterministic(
    coarse_schedule, tmp_path: Path
):
    coefficients = render_firmware_coefficients(coarse_schedule)
    runtime = render_firmware_fit_block(coarse_schedule)
    assert coefficients == render_firmware_coefficients(coarse_schedule)
    assert runtime == render_firmware_fit_block(coarse_schedule)
    assert "BEGIN GENERATED LESO POD-CHEBYSHEV COEFFICIENTS" in coefficients
    assert "LESO_CHEBYSHEV_DEGREE = 4" in coefficients
    assert "LESO_EVEN_TERM_COUNT = 9" in coefficients
    assert "LESO_ODD_TERM_COUNT = 6" in coefficients
    assert "LESO_POD_RANK = 3" in coefficients
    assert "LESO_STATE_BANDWIDTH_HZ[INPUT_DIM]" in coefficients
    assert "LESO_DISTURBANCE_BANDWIDTH_HZ[INPUT_DIM]" in coefficients
    assert "LESO_STATE_POLE[INPUT_DIM]" in coefficients
    assert "LESO_DISTURBANCE_POLE[INPUT_DIM]" in coefficients
    assert "LESO_MATCHED_ALPHA[INPUT_DIM]" in coefficients
    assert "LESO_DISTURBANCE_COUPLING[INPUT_DIM]" in coefficients
    assert "LESO_MATCHED_CORRECTION_SCALE[INPUT_DIM]" in coefficients
    assert "LESO_UNMATCHED_STATE_BANDWIDTH_HZ = 2.500000000e+01f" in coefficients
    assert "LESO_UNMATCHED_POLE = 8.546360135e-01f" in coefficients
    assert "LESO_DISTURBANCE_LEAK_BANDWIDTH_HZ = 2.000000030e-01f" in coefficients
    assert "LESO_DISTURBANCE_RETENTION_POLE = 9.987441301e-01f" in coefficients
    assert "LESO_DISTURBANCE_LIMIT[INPUT_DIM]" in coefficients
    assert "LESO_DISTURBANCE_MAX_INCREMENT[INPUT_DIM]" in coefficients
    assert "LESO_COMPENSATION_FILTER_ORDER = 3" in coefficients
    assert "LESO_COMPENSATION_FILTER_STAGE_ALPHA = 1.685073525e-01f" in coefficients
    assert "LESO_COMPENSATION_GAIN[INPUT_DIM]" in coefficients
    assert "LESO_FL_COMMON_COMPENSATION_GAIN = 1.000000015e-01f" in coefficients
    assert "LESO_FL_DIFFERENTIAL_COMPENSATION_GAIN = 3.000000119e-01f" in coefficients
    assert coefficients.count("3.000000119e-01f") == 1
    assert "static_assert(STATE_DIM == 14 && INPUT_DIM == 6" not in coefficients
    assert "LESO fit dimensions no longer match the model" not in coefficients
    assert "LESO_G_EVEN_RANK" not in coefficients + runtime
    assert "LESO_H_ODD_RANK" not in coefficients + runtime
    assert "LESO_LD_ODD_RANK" not in coefficients + runtime
    assert "LESO_TERMS[LESO_EVEN_TERM_COUNT][2]" in coefficients
    assert "LESO_EVEN_TERMS" not in coefficients + runtime
    assert "LESO_ODD_TERMS" not in coefficients + runtime
    assert "constexpr float LESO_G_MEAN" in coefficients
    assert (
        "LESO_G_EVEN_MODES[LESO_POD_RANK][STATE_DIM]"
        "[LESO_G_COLUMN_COUNT]" in coefficients
    )
    assert "LESO_H_ODD_MODES[LESO_POD_RANK]" in coefficients
    assert "LESO_LD_ODD_COEFFICIENTS[LESO_POD_RANK]" in coefficients
    assert "_ctx.data." not in coefficients
    assert "BEGIN GENERATED LESO POD-CHEBYSHEV RUNTIME FIT" in runtime
    assert "static constexpr" not in runtime
    assert runtime.count("for (uint32_t") == 19
    assert "static float" not in runtime
    assert "MatrixView" not in runtime
    assert "LESO_TERMS[term]" in runtime
    assert "uint32_t odd_term = 0" in runtime
    assert "for (uint32_t mode = 0; mode < LESO_POD_RANK; ++mode)" in runtime
    assert "_ctx.data.G(row, column)" in runtime
    assert "_ctx.data.H(row, column)" in runtime
    assert "_ctx.data.L_x" not in runtime
    assert "_ctx.data.L_d(row, column)" in runtime

    save_result(coarse_schedule, tmp_path)
    assert (tmp_path / "LESO_README.md").is_file()
    assert (tmp_path / "leso_coefficients.inc").read_text(
        encoding="utf-8"
    ) == (coefficients + "\n")
    assert (tmp_path / "fit_params_leso.inc").read_text(encoding="utf-8") == (
        runtime + "\n"
    )
    assert (tmp_path / "leso_schedule.npz").is_file()
    assert (tmp_path / "leso_schedule.json").is_file()
    with np.load(tmp_path / "leso_schedule.npz") as archive:
        assert float(archive["fl_common_compensation_gain"]) == pytest.approx(
            0.10
        )
        assert float(archive["fl_differential_compensation_gain"]) == pytest.approx(
            0.30
        )
        assert int(archive["compensation_filter_order"]) == 3
        assert float(archive["compensation_filter_stage_alpha"]) == pytest.approx(
            0.1685073539
        )
        assert float(archive["unmatched_state_bandwidth_hz"]) == pytest.approx(
            25.0
        )
        assert float(archive["disturbance_leak_bandwidth_hz"]) == pytest.approx(
            0.2
        )
        assert float(archive["disturbance_retention_pole"]) == pytest.approx(
            0.998744152
        )
        np.testing.assert_allclose(
            archive["disturbance_maximum_increments"],
            coarse_schedule.disturbance_maximum_increments,
        )
        np.testing.assert_allclose(
            archive["constant_disturbance_tracking_fraction"],
            coarse_schedule.constant_disturbance_tracking_fraction,
        )
        np.testing.assert_allclose(
            archive["compensation_matrix"][4:, 4:],
            np.asarray([[0.2, -0.1], [-0.1, 0.2]], dtype=np.float32),
        )


@pytest.mark.parametrize(
    "argument",
    [
        "fl_common_compensation_gain",
        "fl_differential_compensation_gain",
    ],
)
@pytest.mark.parametrize("gain", [-0.01, 1.01, np.nan])
def test_schedule_rejects_invalid_fl_mode_gains(argument, gain):
    with pytest.raises(ValueError, match=argument):
        compute_leso_schedule(step=0.04, **{argument: gain})


@pytest.mark.parametrize("bandwidth", [0.0, 500.0, np.nan])
def test_schedule_rejects_invalid_observer_bandwidth(bandwidth):
    with pytest.raises(ValueError, match="observer bandwidths"):
        compute_leso_schedule(
            step=0.04,
            wheel_state_bandwidth_hz=bandwidth,
        )


@pytest.mark.parametrize("ratio", [0.0, 1.01, np.nan])
def test_schedule_rejects_invalid_disturbance_limit_ratio(ratio):
    with pytest.raises(ValueError, match="disturbance_limit_ratio"):
        compute_leso_schedule(step=0.04, disturbance_limit_ratio=ratio)


def test_firmware_renderer_rejects_nonuniform_ranks(coarse_schedule):
    shortened_ld = replace(
        coarse_schedule.ld_schedule,
        odd_modes=coarse_schedule.ld_schedule.odd_modes[:-1],
        odd_coefficients=coarse_schedule.ld_schedule.odd_coefficients[:-1],
    )
    nonuniform = replace(coarse_schedule, ld_schedule=shortened_ld)
    with pytest.raises(ValueError, match="requires equal G/H/L_d even/odd ranks"):
        render_firmware_coefficients(nonuniform)


def test_combined_firmware_coef_contains_unchanged_lqr_and_leso(
    coarse_schedule, tmp_path: Path
):
    lqr_schedule = compute_gain_schedule(step=0.04)
    header = render_combined_coef_header(lqr_schedule, coarse_schedule)

    assert header.startswith("#ifndef __COEF_H__\n#define __COEF_H__")
    assert '#include "stdint.h"' in header
    assert "constexpr uint32_t EXT_STATE_DIM" in header
    assert "constexpr float K_POLY_COEF" in header
    assert "constexpr float FL_U0_POLY_COEF" in header
    assert "Active motor F_L feedforward" in header
    assert render_active_fl_u0_block(
        lqr_schedule.input_coefficients[4, :, 0]
    ) in header
    assert "namespace state_def" in header
    assert "namespace input_def" in header
    assert "namespace lqr_state_def" not in header
    assert "namespace lqr_input_def" not in header
    assert "BEGIN GENERATED LESO POD-CHEBYSHEV COEFFICIENTS" in header
    assert "constexpr uint32_t LESO_POD_RANK = 3" in header
    assert header.rstrip().endswith("#endif")

    save_result(coarse_schedule, tmp_path, lqr_result=lqr_schedule)
    assert (tmp_path / "coef.h").read_text(encoding="utf-8") == header
