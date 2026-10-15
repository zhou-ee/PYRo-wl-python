"""Fixed regression samples, deliberately independent of production defaults.

These values describe the tested reference robot, not the active robot config.
Changing config.py must not change this sample or its numerical expectations.

The historical sample deliberately retains the pre-correction nonzero Ixz;
it is not the symmetric-body hardware configuration used by DEFAULT_ROBOT.
"""

import numpy as np
from wheel_leg_lqr.config import RobotParameters
from wheel_leg_lqr.models.ground.config import GroundSettings

REFERENCE_ROBOT = RobotParameters(
    wheel_mass=0.57,
    wheel_rolling_inertia=0.000596,
    wheel_radius=0.06,
    leg_mass=1.232,
    body_mass=18.9255,
    hip_half_track=0.158,
    wheel_half_track=0.20025,
    body_com_from_hip=(0.0, 0.0, 0.032),
    body_inertia=(
        (0.418395663, 0.0, -0.026043600),
        (0.0, 0.582892610, 0.0),
        (-0.026043600, 0.0, 0.451416848),
    ),
    gravity=9.81,
    min_leg_length=0.18,
    max_leg_length=0.38,
    leg_com_lwp_poly=(0.05806884, 0.37602793, 0.79724260, -0.73809850),
    leg_com_x_poly=(-0.07916397, 0.03685243, -0.12886076, 0.69820022),
    leg_com_fore_aft_enabled=True,
    gas_spring_length_center=0.28,
    gas_spring_length_scale=0.10,
    gas_spring_force_poly=(
        119.867552440332,
        37.1404981279233,
        -4.91246097241044,
        2.70356625079196,
    ),
    wheel_torque_constant_nm_per_a=0.21777611,
    wheel_max_current_a=15.0,
    hip_torque_limit_nm=60.0,
    leg_force_limit_n=300.0,
)
REFERENCE_GROUND = GroundSettings(
    sample_time=0.001,
    q_diag=(
        80.0,
        25.0,
        40.0,
        8.0,
        400.0,
        25.0,
        600.0,
        25.0,
        3000.0,
        12.0,
        120.0,
        6.0,
        120.0,
        6.0,
    ),
    r_diag=(2.8, 2.8, 0.1, 0.1, 0.006, 0.006),
)
REFERENCE_AIRBORNE_Q = np.array([1200.0, 6.0, 1200.0, 6.0, 400.0, 25.0])
REFERENCE_AIRBORNE_R = np.array([0.1, 0.1])
REFERENCE_AIRBORNE_Q.flags.writeable = False
REFERENCE_AIRBORNE_R.flags.writeable = False
REFERENCE_BANDWIDTH_HZ = (1.2783529565614096,) * 6
REFERENCE_CONTACT = dict(
    integration_substeps=5,
    ground_stiffness=5.0e3,
    ground_damping=2.0e2,
    stop_stiffness=1.0e4,
    stop_damping=3.0e2,
)
