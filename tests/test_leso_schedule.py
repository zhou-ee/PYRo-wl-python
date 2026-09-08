from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from wheel_leg_lqr.numeric_config import N_X, SAMPLE_TIME
from wheel_leg_lqr.scheduling.leg_length_schedule import (
    compute_gain_schedule,
    render_active_fl_u0_block,
)
from wheel_leg_lqr.scheduling.leso_schedule import (
    G_SCHEDULED_COLUMN_OFFSET,
    INPUT_EXCHANGE,
    STATE_EXCHANGE,
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
    assert result.state_pole == pytest.approx(0.80)
    assert result.disturbance_pole == pytest.approx(0.992)
    assert result.unmatched_state_pole == pytest.approx(0.792)
    assert result.compensation_cutoff_hz == pytest.approx(15.0)
    assert result.fl_compensation_gain == pytest.approx(0.30)
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
    assert result.observer_metrics.float32_holdout_max_radius < 0.995
    assert result.observer_metrics.minimum_float32_margin > 0.005
    assert result.downstream_metrics.float32_train_max_radius < 0.9995
    assert result.downstream_metrics.float32_holdout_max_radius < 0.9995
    assert result.downstream_metrics.minimum_float32_margin > 5.0e-4


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
    assert "LESO_STATE_POLE = 8.000000119e-01f" in coefficients
    assert "LESO_DISTURBANCE_POLE = 9.919999838e-01f" in coefficients
    assert "LESO_UNMATCHED_POLE = 7.919999957e-01f" in coefficients
    assert "LESO_COMPENSATION_FILTER_ALPHA" in coefficients
    assert "LESO_COMPENSATION_GAIN[INPUT_DIM]" in coefficients
    assert coefficients.count("3.000000119e-01f") == 2
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
    assert runtime.count("for (uint32_t") == 20
    assert "static float" not in runtime
    assert "MatrixView" not in runtime
    assert "LESO_TERMS[term]" in runtime
    assert "uint32_t odd_term = 0" in runtime
    assert "for (uint32_t mode = 0; mode < LESO_POD_RANK; ++mode)" in runtime
    assert "_ctx.data.G(row, column)" in runtime
    assert "_ctx.data.H(row, column)" in runtime
    assert "_ctx.data.L_x(row, column)" in runtime
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
        assert float(archive["fl_compensation_gain"]) == pytest.approx(0.30)
        np.testing.assert_array_equal(
            archive["compensation_gains"],
            np.asarray(
                [1.0, 1.0, 1.0, 1.0, 0.3, 0.3], dtype=np.float32
            ),
        )


@pytest.mark.parametrize("gain", [-0.01, 1.01, np.nan])
def test_schedule_rejects_invalid_fl_compensation_gain(gain):
    with pytest.raises(ValueError, match="fl_compensation_gain"):
        compute_leso_schedule(step=0.04, fl_compensation_gain=gain)


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
