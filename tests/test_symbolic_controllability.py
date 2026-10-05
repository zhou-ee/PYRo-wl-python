import sympy as sp

from wheel_leg_lqr.analysis.symbolic import build_nominal_input_matrix, generic_witness, second_order_pbh_matrix


def test_nominal_input_matrix_has_six_independent_columns():
    matrix = build_nominal_input_matrix()
    assert matrix.shape == (7, 6)
    assert matrix.rank() == 6


def test_generic_second_order_witness_is_controllable():
    result = generic_witness()
    assert result["A"].shape == (14, 14)
    assert result["B"].shape == (14, 6)
    assert result["controllability_matrix"].shape == (14, 84)
    assert result["controllability_rank"] == 14


def test_second_order_pbh_matrix_shape():
    result = generic_witness()
    lam = sp.Symbol("lambda")
    matrix = second_order_pbh_matrix(
        lam,
        result["M"],
        result["D"],
        result["K"],
        result["S"],
    )
    assert matrix.shape == (7, 13)
