from tests.reference_parameters import (
    REFERENCE_BANDWIDTH_HZ,
    REFERENCE_CONTACT,
    REFERENCE_GROUND,
    REFERENCE_ROBOT,
)
import numpy as np
import pytest

from wheel_leg_lqr.simulation.step_drop import (
    simulate_step_drop,
    target_lengths_for_step,
)
from wheel_leg_lqr.models.ground.model import make_trim_geometry


def test_target_lengths_keep_mean_and_match_018m_step():
    lengths = target_lengths_for_step(
        0.18, mean_leg_length=0.28, parameters=REFERENCE_ROBOT
    )

    assert np.mean(lengths) == pytest.approx(0.28)
    geometry = make_trim_geometry(*lengths, parameters=REFERENCE_ROBOT)
    assert 2.0 * abs(geometry.kappa_g) == pytest.approx(0.18, abs=1e-12)
    assert np.all(lengths >= 0.18)
    assert np.all(lengths <= 0.38)


def test_sudden_018m_drop_hits_lower_contact_but_does_not_recover():
    result = simulate_step_drop(
        step_height=0.18,
        horizon_s=0.2,
        parameters=REFERENCE_ROBOT,
        settings=REFERENCE_GROUND,
        bandwidth_hz=REFERENCE_BANDWIDTH_HZ,
        **REFERENCE_CONTACT,
    )

    assert not result.converged
    assert result.impact_time == pytest.approx(0.090, abs=0.010)
    assert result.impact_speed is not None and result.impact_speed < 0.0
    assert result.mechanical_stop_impulse[0] > 0.0
    assert result.failure_reason is not None
    assert result.saturation_steps > 0
