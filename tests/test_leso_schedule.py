import json

import numpy as np
import pytest

from wheel_leg_lqr import numeric_config as config
from wheel_leg_lqr.control.leso import initialize_leso, step_leso_split
from wheel_leg_lqr.scheduling.leso_schedule import (
    compute_leso_schedule,
    refresh_combined_header_active_u0,
    replace_combined_header_active_u0,
    save_leso_result,
    step_scheduled_leso,
)


@pytest.fixture(scope="module")
def leso_schedule():
    return compute_leso_schedule(
        step=0.04,
        scheduled_ad_entries=20,
        scheduled_bd_entries=20,
    )


def test_sparse_schedule_is_accurate_and_float32_stable(leso_schedule):
    assert len(leso_schedule.ad_schedule.entries) >= 20
    assert len(leso_schedule.bd_schedule.entries) >= 20
    assert leso_schedule.max_ad_dynamic_relative_error < 0.01
    assert leso_schedule.max_bd_relative_error < 0.02
    assert leso_schedule.max_raw_observer_spectral_radius < 1.0
    assert leso_schedule.max_fitted_observer_spectral_radius < 1.0
    assert leso_schedule.max_float32_observer_spectral_radius < 1.0


def test_selected_entries_are_closed_under_left_right_mirroring(leso_schedule):
    state_mirror = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 10, 11)
    input_mirror = (1, 0, 3, 2, 5, 4)
    ad_entries = {tuple(entry) for entry in leso_schedule.ad_schedule.entries}
    bd_entries = {tuple(entry) for entry in leso_schedule.bd_schedule.entries}
    for row, column in ad_entries:
        assert (state_mirror[row], state_mirror[column]) in ad_entries
    for row, column in bd_entries:
        assert (state_mirror[row], input_mirror[column]) in bd_entries


def test_sparse_matvec_and_step_match_dense_schedule(leso_schedule):
    rng = np.random.default_rng(7)
    l1, l2 = 0.273, 0.347
    state_vector = rng.normal(scale=0.1, size=config.N_X)
    input_vector = rng.normal(scale=0.1, size=config.N_U)
    ad, bd = leso_schedule.matrices(l1, l2)
    np.testing.assert_allclose(
        leso_schedule.ad_schedule.matvec(l1, l2, state_vector),
        ad @ state_vector,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        leso_schedule.bd_schedule.matvec(l1, l2, input_vector),
        bd @ input_vector,
        atol=1e-12,
    )

    estimate = initialize_leso(rng.normal(scale=0.01, size=config.N_X))
    measured_state = rng.normal(scale=0.01, size=config.N_X)
    applied_delta = rng.normal(scale=0.1, size=config.N_U)
    sparse_result = step_scheduled_leso(
        leso_schedule,
        estimate,
        measured_state,
        applied_delta,
        l1,
        l2,
    )
    dense_result = step_leso_split(
        estimate,
        measured_state,
        applied_delta,
        ad,
        bd,
        leso_schedule.fixed_gain,
    )
    np.testing.assert_allclose(
        sparse_result.estimate.state,
        dense_result.estimate.state,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        sparse_result.estimate.disturbance,
        dense_result.estimate.disturbance,
        atol=1e-12,
    )


def test_python_artifacts_round_trip(leso_schedule, tmp_path):
    save_leso_result(leso_schedule, tmp_path)
    archive = np.load(tmp_path / "leso_schedule.npz")
    report = json.loads((tmp_path / "leso_schedule.json").read_text(encoding="utf-8"))

    assert archive["fixed_gain"].shape == (20, 14)
    assert archive["ad_entries"].shape[1] == 2
    assert archive["bd_entries"].shape[1] == 2
    assert "state_scales" not in archive.files
    assert "input_scales" not in archive.files
    assert report["max_float32_observer_spectral_radius"] < 1.0


def test_combined_header_u0_refresh_is_narrow_and_idempotent(tmp_path):
    stale_header = """#ifndef __COEF_H__
#define __COEF_H__

constexpr float K_POLY_COEF[] = {42.0f};

// Leg thrust static equilibrium feedforward polynomial.
// f_l_trim is the total force.
constexpr float FL_U0_POLY_COEF[U0_POLY_DEGREE + 1] = {
    1.0059390686e+02f, 4.6173437895e-01f, -2.8751243380e-01f,
    1.3612422702e-02f};

constexpr float LESO_G_MEAN[] = {7.0f};
#endif
"""
    active_coefficients = np.array(
        [-19.273645585, -36.678763749, 4.6249485386, -2.6899538281]
    )
    updated = replace_combined_header_active_u0(
        stale_header,
        active_coefficients,
    )
    assert "-1.9273645585e+01f" in updated
    assert "-3.6678763749e+01f" in updated
    assert "Active motor F_L feedforward" in updated
    assert "constexpr float K_POLY_COEF[] = {42.0f};" in updated
    assert "constexpr float LESO_G_MEAN[] = {7.0f};" in updated
    assert replace_combined_header_active_u0(updated, active_coefficients) == updated

    header_path = tmp_path / "coef.h"
    header_path.write_text(stale_header, encoding="utf-8")
    assert refresh_combined_header_active_u0(
        header_path,
        active_coefficients,
    )
    assert header_path.read_text(encoding="utf-8") == updated
    assert not refresh_combined_header_active_u0(
        tmp_path / "missing.h",
        active_coefficients,
    )
