from __future__ import annotations
from ..models.ground.trim import compute_equilibrium_beta
import argparse
import time
import numpy as np
from ..analysis.qr_optimizer import optimize_qr_matrix
from ..models.ground import config
from ..config import DEFAULT_ROBOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=float, default=config.ANALYSIS_HEIGHT, help="Target height in meters (default: ground config).")
    parser.add_argument("--beta0", type=float, default=config.OPTIMIZER_INITIAL_SWING_ANGLE, help="Initial swing angle in rad (default: ground config).")
    parser.add_argument("--l0", type=float, default=(DEFAULT_ROBOT.min_leg_length + DEFAULT_ROBOT.max_leg_length) / 2, help="Initial leg length in meters (default: configured domain midpoint).")
    parser.add_argument("--max-iter", type=int, default=400, help="Max optimizer iterations (default: 400).")
    args = parser.parse_args()

    print("=================================================================")
    beta_target = compute_equilibrium_beta(args.height)
    l_bar_target = args.height / np.cos(beta_target)
    print(
        "  Automatic LQR QR Optimizer "
        f"(Target Mean Leg Length: {l_bar_target:.6f} m, Derived Height: {args.height} m)"
    )
    print("=================================================================\n")
    print(f"Scenario: Initial Beta={args.beta0:.2f} rad ({np.degrees(args.beta0):.1f} deg), Leg L={args.l0:.2f} m")
    print("Running continuous Nelder-Mead optimization with Balance-First Hierarchy & Damping Constraints...")

    t0 = time.perf_counter()
    opt_q, opt_r, base_m, opt_m, nit, nfev = optimize_qr_matrix(
        h_target=args.height,
        beta0=args.beta0,
        L0=args.l0,
        max_iter=args.max_iter,
    )
    elapsed = time.perf_counter() - t0

    print(f"\nOptimization converged in {elapsed:.2f}s! ({nit} iterations, {nfev} function evaluations)\n")
    print("-----------------------------------------------------------------")
    print("  Performance Metric Comparison: Baseline vs Continuous Optimum")
    print("-----------------------------------------------------------------")
    print(f"{'Metric':<35} | {'Baseline':<12} | {'Optimized':<12}")
    print("-" * 65)
    print(f"{'Wheel Torque Saturation Time':<35} | {base_m.wheel_sat_duration_s*1000:8.1f} ms  | {opt_m.wheel_sat_duration_s*1000:8.1f} ms")
    print(f"{'Max Chassis Pitch Angle':<35} | {base_m.max_pitch_deg:8.2f} deg | {opt_m.max_pitch_deg:8.2f} deg")
    print(f"{'Max Forward Rollout Displacement':<35} | {base_m.max_displacement_m:8.4f} m   | {opt_m.max_displacement_m:8.4f} m")
    print(f"{'Final Position Drift':<35} | {base_m.final_displacement_m:8.4f} m   | {opt_m.final_displacement_m:8.4f} m")
    print(f"{'Leg Swing Settling Time (<2 deg)':<35} | {base_m.beta_settling_time_s*1000:8.1f} ms  | {opt_m.beta_settling_time_s*1000:8.1f} ms")
    print(f"{'Mean Leg Length Settling (<5 mm)':<35} | {base_m.lbar_settling_time_s*1000:8.1f} ms  | {opt_m.lbar_settling_time_s*1000:8.1f} ms")
    print(f"{'Peak Initial Hip Torque Tp':<35} | {base_m.initial_peak_tp_nm:8.2f} N.m | {opt_m.initial_peak_tp_nm:8.2f} N.m")
    print(f"{'Max Discrete Spectral Radius rho':<35} | {base_m.max_spectral_radius:8.6f}     | {opt_m.max_spectral_radius:8.6f}")
    print(f"{'Composite Cost J':<35} | {base_m.total_cost:8.2f}     | {opt_m.total_cost:8.2f}")
    print("-----------------------------------------------------------------\n")

    print(">>> Continuous Optimal Q_DIAG:")
    print("np.array([")
    print(f"    {opt_q[0]:.6f}, {opt_q[1]:.6f},  # x, dx       (ratio Qx/Qtheta = {opt_q[0]/opt_q[4]:.4f})")
    print(f"    {opt_q[2]:.6f}, {opt_q[3]:.6f},  # psi, dpsi")
    print(f"    {opt_q[4]:.6f}, {opt_q[5]:.6f},  # theta, dtheta (ratio qdtheta/sqrt(qtheta) = {opt_q[5]/np.sqrt(opt_q[4]):.2f})")
    print(f"    {opt_q[6]:.6f}, {opt_q[7]:.6f},  # phi, dphi")
    print(f"    {opt_q[8]:.6f}, {opt_q[9]:.6f},  # L_bar, dL_bar")
    print(f"    {opt_q[10]:.6f}, {opt_q[11]:.6f},  # beta_1, dbeta_1")
    print(f"    {opt_q[12]:.6f}, {opt_q[13]:.6f},  # beta_2, dbeta_2")
    print("], dtype=float)")

    print("\n>>> Baseline R_DIAG:")
    print("np.array([")
    print(f"    {opt_r[0]:.6f}, {opt_r[1]:.6f},  # T_w1, T_w2")
    print(f"    {opt_r[2]:.6f}, {opt_r[3]:.6f},  # T_p1, T_p2")
    print(f"    {opt_r[4]:.6f}, {opt_r[5]:.6f},  # f_l1, f_l2")
    print("], dtype=float)\n")

if __name__ == "__main__":
    main()
