"""腿部质心偏移多项式拟合与平衡角求解脚本。

Fit polynomials for leg center of mass (COM) coordinates and compute equilibrium
pitch angles beta_eq(L) based on CAD data in '清洗后重心.xlsx'.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from ..config import DEFAULT_ROBOT, RobotParameters
from ..artifacts.io import project_root


EXCEL_PATH = (project_root() or Path.cwd()) / "data/reference/清洗后重心.xlsx"


def fit_leg_com_data(excel_path: Path = EXCEL_PATH, parameters: RobotParameters = DEFAULT_ROBOT) -> dict:
    """读取清洗后重心 Excel 并执行 3 次多项式拟合。"""
    import pandas as pd
    df = pd.read_excel(excel_path)
    # 统一列名
    df.columns = ["L_mm", "com_X_mm", "com_Y_mm", "unnamed", "Lph_mm", "Lph_m", "L_m"]

    L = df["L_m"].to_numpy(dtype=float)
    L_ph = -df["com_Y_mm"].to_numpy(dtype=float) / 1000.0  # 沿腿连线距离髋关节的距离 (m)
    L_wp = L - L_ph                                       # 沿腿连线距离下支点的距离 (m)
    x_com = df["com_X_mm"].to_numpy(dtype=float) / 1000.0 # 垂直于腿连线的横向偏移 (m)

    # 偏转角 alpha_P: 下支点处质心相对于腿连线 (P->H) 的偏角 (rad)
    alpha_P = np.arctan2(x_com, L_wp)

    # 偏转角 alpha_H: 髋关节处质心相对于腿方向 (H->P) 的偏角 (rad)
    alpha_H = np.arctan2(x_com, -L_ph)

    # 3 次多项式拟合: y = c0 + c1*L + c2*L^2 + c3*L^3 (ascending order)
    poly_wp = np.polyfit(L, L_wp, 3)[::-1]
    poly_x = np.polyfit(L, x_com, 3)[::-1]
    poly_alpha_P = np.polyfit(L, alpha_P, 3)[::-1]
    poly_alpha_H = np.polyfit(L, alpha_H, 3)[::-1]

    # 拟合误差评估
    err_wp = np.abs(np.polyval(poly_wp[::-1], L) - L_wp)
    err_x = np.abs(np.polyval(poly_x[::-1], L) - x_com)
    err_alpha_P = np.abs(np.polyval(poly_alpha_P[::-1], L) - alpha_P)

    # 静力学平衡角 beta_eq(L):
    # tan(beta_eq) = (mp * x_com) / (mb * L / 2 + mp * L_wp)
    mb = parameters.body_mass
    mp = parameters.leg_mass
    beta_eq = np.arctan((mp * x_com) / (mb * L / 2.0 + mp * L_wp))
    poly_beta_eq = np.polyfit(L, beta_eq, 3)[::-1]
    err_beta_eq = np.abs(np.polyval(poly_beta_eq[::-1], L) - beta_eq)

    return {
        "L": L,
        "L_wp": L_wp,
        "x_com": x_com,
        "alpha_P": alpha_P,
        "alpha_H": alpha_H,
        "beta_eq": beta_eq,
        "poly_wp": poly_wp,
        "poly_x": poly_x,
        "poly_alpha_P": poly_alpha_P,
        "poly_alpha_H": poly_alpha_H,
        "poly_beta_eq": poly_beta_eq,
        "metrics": {
            "max_err_wp_mm": float(np.max(err_wp) * 1000),
            "rms_err_wp_mm": float(np.sqrt(np.mean(err_wp**2)) * 1000),
            "max_err_x_mm": float(np.max(err_x) * 1000),
            "rms_err_x_mm": float(np.sqrt(np.mean(err_x**2)) * 1000),
            "max_err_alpha_deg": float(np.degrees(np.max(err_alpha_P))),
            "rms_err_alpha_deg": float(np.degrees(np.sqrt(np.mean(err_alpha_P**2)))),
            "max_err_beta_eq_deg": float(np.degrees(np.max(err_beta_eq))),
            "rms_err_beta_eq_deg": float(np.degrees(np.sqrt(np.mean(err_beta_eq**2)))),
        },
    }
