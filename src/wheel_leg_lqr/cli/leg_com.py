import argparse
from pathlib import Path
from ..calibration.leg_com import EXCEL_PATH
import numpy as np
from ..config import DEFAULT_ROBOT
from ..calibration.leg_com import fit_leg_com_data
def main() -> None:
    parser = argparse.ArgumentParser(description="Fit CAD leg COM curves and equilibrium angles.")
    parser.add_argument("--excel", type=Path, default=EXCEL_PATH, help="CAD source workbook")
    args = parser.parse_args()
    res = fit_leg_com_data(args.excel)
    m = res["metrics"]

    print("=" * 80)
    print("              双轮足腿部质心偏移多项式拟合结果与评估报告")
    print("=" * 80)
    mode = "开启" if DEFAULT_ROBOT.leg_com_fore_aft_enabled else "关闭（模型 x_com = 0）"
    print(f"模型前后质心调度：{mode}；beta_eq 按此模式计算。")
    print("下列 x_com、alpha_P 为原始 CAD 拟合，关闭开关时仍保留供参考。")
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

    test_Ls = np.linspace(DEFAULT_ROBOT.min_leg_length, DEFAULT_ROBOT.max_leg_length, 9)
    for L_val in test_Ls:
        lwp_val = np.polyval(res["poly_wp"][::-1], L_val)
        xcom_val = np.polyval(res["poly_x"][::-1], L_val)
        alpha_val = np.polyval(res["poly_alpha_P"][::-1], L_val)
        model_xcom_val = xcom_val if DEFAULT_ROBOT.leg_com_fore_aft_enabled else 0.0
        beta_val = np.arctan((DEFAULT_ROBOT.leg_mass * model_xcom_val) / (DEFAULT_ROBOT.body_mass * L_val / 2.0 + DEFAULT_ROBOT.leg_mass * lwp_val))
        print(
            f"{L_val:8.3f} | {lwp_val*1000:10.2f} | {xcom_val*1000:10.2f} | "
            f"{np.degrees(alpha_val):14.4f} | {np.degrees(beta_val):14.4f} | {beta_val:14.6f}"
        )
    print("-" * 80)

if __name__ == "__main__":
    main()
