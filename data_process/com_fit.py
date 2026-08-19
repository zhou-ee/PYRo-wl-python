"""腿部质心偏移多项式拟合与平衡角求解脚本。

Fit polynomials for leg center of mass (COM) coordinates and compute equilibrium
pitch angles beta_eq(L) based on CAD data in '清洗后重心.xlsx'.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


_ROOT = Path(__file__).resolve().parent
EXCEL_PATH = _ROOT / "清洗后重心.xlsx"


def fit_leg_com_data(excel_path: Path = EXCEL_PATH) -> dict:
    """读取清洗后重心 Excel 并执行 3 次多项式拟合。"""
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
    mb = 12.0
    mp = 1.232
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


def main() -> None:
    res = fit_leg_com_data()
    m = res["metrics"]
    
    print("=" * 80)
    print("              双轮足腿部质心偏移多项式拟合结果与评估报告")
    print("=" * 80)
    print("\n1. 拟合多项式系数 (升幂排列: c0 + c1*L + c2*L^2 + c3*L^3):")
    print(f"   L_wp(L)   [m]   = {res['poly_wp'][0]:.8f} + ({res['poly_wp'][1]:.8f})*L + ({res['poly_wp'][2]:.8f})*L^2 + ({res['poly_wp'][3]:.8f})*L^3")
    print(f"   x_com(L)  [m]   = {res['poly_x'][0]:.8f} + ({res['poly_x'][1]:.8f})*L + ({res['poly_x'][2]:.8f})*L^2 + ({res['poly_x'][3]:.8f})*L^3")
    print(f"   alpha_P(L)[rad] = {res['poly_alpha_P'][0]:.8f} + ({res['poly_alpha_P'][1]:.8f})*L + ({res['poly_alpha_P'][2]:.8f})*L^2 + ({res['poly_alpha_P'][3]:.8f})*L^3")
    print(f"   beta_eq(L)[rad] = {res['poly_beta_eq'][0]:.8f} + ({res['poly_beta_eq'][1]:.8f})*L + ({res['poly_beta_eq'][2]:.8f})*L^2 + ({res['poly_beta_eq'][3]:.8f})*L^3")
    
    print("\n2. 拟合精度指标 (Max / RMS Residuals):")
    print(f"   - L_wp(L)    最大残差: {m['max_err_wp_mm']:.4f} mm,  RMS 残差: {m['rms_err_wp_mm']:.4f} mm")
    print(f"   - x_com(L)   最大残差: {m['max_err_x_mm']:.4f} mm,  RMS 残差: {m['rms_err_x_mm']:.4f} mm")
    print(f"   - alpha_P(L) 最大残差: {m['max_err_alpha_deg']:.4f} deg, RMS 残差: {m['rms_err_alpha_deg']:.4f} deg")
    print(f"   - beta_eq(L) 最大残差: {m['max_err_beta_eq_deg']:.4f} deg, RMS 残差: {m['rms_err_beta_eq_deg']:.4f} deg")
    
    print("\n3. 典型腿长下的质心位置与单腿平衡角对照表:")
    print("-" * 80)
    print(f"{'L (m)':>8s} | {'L_wp (mm)':>10s} | {'x_com (mm)':>10s} | {'alpha_P (deg)':>14s} | {'beta_eq (deg)':>14s} | {'beta_eq (rad)':>14s}")
    print("-" * 80)
    
    test_Ls = [0.18, 0.20, 0.23, 0.25, 0.28, 0.30, 0.33, 0.35, 0.38]
    for L_val in test_Ls:
        lwp_val = np.polyval(res["poly_wp"][::-1], L_val)
        xcom_val = np.polyval(res["poly_x"][::-1], L_val)
        alpha_val = np.polyval(res["poly_alpha_P"][::-1], L_val)
        beta_val = np.arctan((1.232 * xcom_val) / (12.0 * L_val / 2.0 + 1.232 * lwp_val))
        print(
            f"{L_val:8.3f} | {lwp_val*1000:10.2f} | {xcom_val*1000:10.2f} | "
            f"{np.degrees(alpha_val):14.4f} | {np.degrees(beta_val):14.4f} | {beta_val:14.6f}"
        )
    print("-" * 80)


if __name__ == "__main__":
    main()
