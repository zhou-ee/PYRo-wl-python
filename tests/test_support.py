# 测试专用参数 / Test-only parameters

from wheel_legged_lqr.parameters import RobotParameters

def make_test_parameters() -> RobotParameters:
    """Use synthetic values only for automated tests / """
    return RobotParameters(
        mw=0.5, iw=0.001, rw=0.06,
        mp=1.0, ip=0.0048, l_wp=0.12, l_ph=0.12,
        mb=5.0, ib=0.12, l_c=0.01, g=9.81,
    )
