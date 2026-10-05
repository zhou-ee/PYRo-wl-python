from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sympy as sp
from ..models.ground.config import N_Q, N_U


@dataclass(frozen=True)
class SymbolTable:
    """Symbols used by the local upright input mapping."""

    R_w: sp.Symbol
    a_w: sp.Symbol
    a_h: sp.Symbol
    g: sp.Symbol
    m_b: sp.Symbol
    m_p1: sp.Symbol
    m_p2: sp.Symbol
    m_w1: sp.Symbol
    m_w2: sp.Symbol
    I_bx: sp.Symbol
    I_by: sp.Symbol
    I_bz: sp.Symbol
    I_p1: sp.Symbol
    I_p2: sp.Symbol
    I_w1: sp.Symbol
    I_w2: sp.Symbol
    L_1: sp.Symbol
    L_2: sp.Symbol
    beta_1: sp.Symbol
    beta_2: sp.Symbol
    l_bar: sp.Symbol
    phi: sp.Symbol
    theta: sp.Symbol
    kappa_g: sp.Symbol
    s_w1_psi: sp.Symbol
    s_w2_psi: sp.Symbol


def build_symbols() -> SymbolTable:
    """Create the physical, geometry, and trim symbols."""
    return SymbolTable(
        R_w=sp.Symbol("R_w", positive=True, nonzero=True),
        a_w=sp.Symbol("a_w", positive=True, nonzero=True),
        a_h=sp.Symbol("a_h", positive=True, nonzero=True),
        g=sp.Symbol("g", positive=True),
        m_b=sp.Symbol("m_b", positive=True),
        m_p1=sp.Symbol("m_p1", positive=True),
        m_p2=sp.Symbol("m_p2", positive=True),
        m_w1=sp.Symbol("m_w1", positive=True),
        m_w2=sp.Symbol("m_w2", positive=True),
        I_bx=sp.Symbol("I_bx", positive=True),
        I_by=sp.Symbol("I_by", positive=True),
        I_bz=sp.Symbol("I_bz", positive=True),
        I_p1=sp.Symbol("I_p1", positive=True),
        I_p2=sp.Symbol("I_p2", positive=True),
        I_w1=sp.Symbol("I_w1", positive=True),
        I_w2=sp.Symbol("I_w2", positive=True),
        L_1=sp.Symbol("L_1", positive=True),
        L_2=sp.Symbol("L_2", positive=True),
        beta_1=sp.Symbol("beta_1", real=True),
        beta_2=sp.Symbol("beta_2", real=True),
        l_bar=sp.Symbol("L_bar", positive=True),
        phi=sp.Symbol("phi", real=True),
        theta=sp.Symbol("theta", real=True),
        kappa_g=sp.Symbol("kappa_g", real=True),
        s_w1_psi=sp.Symbol("s_w1_psi", real=True),
        s_w2_psi=sp.Symbol("s_w2_psi", real=True),
    )


def build_nominal_input_matrix() -> sp.Matrix:
    """Return the local upright 7-by-6 generalized-input matrix.

    The rows follow q = [x, psi, theta, phi, L_bar, beta_1, beta_2].
    The yaw row uses complete heading-frame wheel velocity-map symbols
    s_wi_psi.  They are evaluated from e1^T H_v_wi^Y[:, psi] / R_w at each
    trim point, including the yaw transport column e3 x rho_wi^Y.
    The remaining rows retain the local straight-line mapping for the first
    structural controllability pass.
    """
    symbols = build_symbols()
    rw, ah = symbols.R_w, symbols.a_h
    return sp.Matrix(
        [
            [1 / rw, 1 / rw, 0, 0, 0, 0],
            [symbols.s_w1_psi, symbols.s_w2_psi, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, ah, -ah],
            [0, 0, 0, 0, 1, 1],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
        ]
    )


def build_state_space(
    M: sp.MatrixBase,
    K: sp.MatrixBase,
    D: sp.MatrixBase,
    S: sp.MatrixBase,
) -> tuple[sp.Matrix, sp.Matrix]:
    """Build A and B from M qdd + D qd + K q = S u."""
    matrices = {
        "M": sp.Matrix(M),
        "K": sp.Matrix(K),
        "D": sp.Matrix(D),
        "S": sp.Matrix(S),
    }
    if matrices["M"].shape != (N_Q, N_Q):
        raise ValueError("M must have shape (7, 7)")
    if matrices["K"].shape != (N_Q, N_Q):
        raise ValueError("K must have shape (7, 7)")
    if matrices["D"].shape != (N_Q, N_Q):
        raise ValueError("D must have shape (7, 7)")
    if matrices["S"].shape != (N_Q, N_U):
        raise ValueError("S must have shape (7, 6)")

    mass_inverse = matrices["M"].inv()
    zero = sp.zeros(N_Q, N_Q)
    identity = sp.eye(N_Q)
    A = zero.row_join(identity).col_join(
        (-mass_inverse * matrices["K"]).row_join(-mass_inverse * matrices["D"])
    )
    B = sp.zeros(N_Q, N_U).col_join(mass_inverse * matrices["S"])
    return A, B


def controllability_matrix(A: sp.MatrixBase, B: sp.MatrixBase) -> sp.Matrix:
    """Construct [B, AB, ..., A^(n-1)B]."""
    A = sp.Matrix(A)
    B = sp.Matrix(B)
    if A.rows != A.cols or B.rows != A.rows:
        raise ValueError("A and B dimensions are incompatible")
    blocks = []
    current = B
    for _ in range(A.rows):
        blocks.append(current)
        current = A * current
    return sp.Matrix.hstack(*blocks)


def second_order_pbh_matrix(
    lam: sp.Symbol,
    M: sp.MatrixBase,
    D: sp.MatrixBase,
    K: sp.MatrixBase,
    S: sp.MatrixBase,
) -> sp.Matrix:
    """Return [lam^2 M + lam D + K, -S] for the second-order PBH test."""
    return (
        sp.Matrix(M) * lam**2 + sp.Matrix(D) * lam + sp.Matrix(K)
    ).row_join(-sp.Matrix(S))


def _witness_matrices() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix]:
    """Create one exact rational, generic mechanical witness."""
    symbols = build_symbols()
    S = build_nominal_input_matrix().subs(
        {
            symbols.R_w: sp.Rational(3, 50),
            symbols.a_h: sp.Rational(1, 10),
            symbols.s_w1_psi: sp.Rational(-5, 3),
            symbols.s_w2_psi: sp.Rational(5, 3),
        }
    )
    M = sp.diag(6, 2, 3, 2, 7, 1, 1)
    D = sp.diag(1, 2, 1, 2, 1, 1, 1)
    K = sp.diag(2, 3, 5, 4, 8, 6, 7)
    for i, j, value in (
        (0, 2, sp.Rational(1, 5)),
        (0, 5, sp.Rational(1, 4)),
        (2, 5, sp.Rational(1, 3)),
        (3, 4, sp.Rational(1, 2)),
        (3, 6, sp.Rational(2, 5)),
        (4, 5, sp.Rational(1, 6)),
    ):
        K[i, j] = value
        K[j, i] = value
    return M, K, D, S


def generic_witness() -> dict[str, Any]:
    """Evaluate a generic exact-rational witness for structural rank."""
    symbolic_input_matrix = build_nominal_input_matrix()
    M, K, D, S = _witness_matrices()
    A, B = build_state_space(M, K, D, S)
    matrix = controllability_matrix(A, B)
    return {
        "symbolic_input_matrix": symbolic_input_matrix,
        "symbolic_input_rank": symbolic_input_matrix.rank(),
        "M": M,
        "K": K,
        "D": D,
        "S": S,
        "A": A,
        "B": B,
        "controllability_matrix": matrix,
        "input_rank": S.rank(),
        "controllability_rank": matrix.rank(),
    }


def _write_matrix(path: Path, matrix: sp.MatrixBase) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(matrix), encoding="utf-8")
