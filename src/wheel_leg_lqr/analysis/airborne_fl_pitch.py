"""Instantaneous FL-to-pitch sensitivity analysis, using the airborne plant."""

from ..config import DEFAULT_ROBOT
from ..models.airborne.model import compute_leg_force_pitch_effect


def analyze_leg_force_pitch(
    l1=None, l2=None, *, body_com_x_from_hip=None, parameters=DEFAULT_ROBOT, **angles
):
    nominal = (parameters.min_leg_length + parameters.max_leg_length) / 2
    l1 = nominal if l1 is None else l1
    l2 = nominal if l2 is None else l2
    return compute_leg_force_pitch_effect(
        l1, l2, body_com_x_from_hip=body_com_x_from_hip, parameters=parameters, **angles
    )
