"""基于非线性时域闭环仿真与执行器饱和约束的自动 QR 参数寻优工具。

Automatic QR Matrix Optimizer based on nonlinear closed-loop step response simulation
under physical actuator torque saturation (wheel torque limit, hip torque limit, thrust limit).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

from ..core import config
from ..core.model import linearize_static_trim
from ..control.lqr import discretize_state_space, solve_discrete_lqr


@dataclass
class OptimizationMetrics:
    q_diag: np.ndarray
    r_diag: np.ndarray
    max_spectral_radius: float
    wheel_sat_duration_s: float
    max_pitch_deg: float
    max_displacement_m: float
    final_displacement_m: float
    beta_settling_time_s: float
    lbar_settling_time_s: float
    initial_peak_tp_nm: float
    initial_peak_fl_n: float
    total_cost: float


def run_closed_loop_simulation(
    Ad: np.ndarray,
    Bd: np.ndarray,
    K: np.ndarray,
    e0: np.ndarray,
    sim_time: float = 2.5,
    dt: float = 0.001,
    tw_max: float = config.WHEEL_TORQUE_LIMIT_NM,
    tp_max: float = 60.0,
    fl_max: float = 300.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """带有硬执行器限幅的时域闭环步进响应仿真。"""
    steps = int(sim_time / dt)
    e_hist = np.zeros((steps, 14), dtype=float)
    u_hist = np.zeros((steps, 6), dtype=float)
    e = e0.copy()
    sat_time = 0.0

    lower_lims = np.array([-tw_max, -tw_max, -tp_max, -tp_max, -fl_max, -fl_max], dtype=float)
    upper_lims = -lower_lims

    for t in range(steps):
        e_hist[t] = e
        u_raw = - K @ e
        u_clip = np.clip(u_raw, lower_lims, upper_lims)
        if abs(u_raw[0]) > tw_max or abs(u_raw[1]) > tw_max:
            sat_time += dt
        u_hist[t] = u_clip
        e = Ad @ e + Bd @ u_clip
        if np.any(np.isnan(e)) or np.max(np.abs(e)) > 20.0:
            return e_hist, u_hist, 999.0

    return e_hist, u_hist, sat_time


def evaluate_qr_candidate(
    q_diag: np.ndarray,
    r_diag: np.ndarray,
    Ad: np.ndarray,
    Bd: np.ndarray,
    e0: np.ndarray,
    h_target: float,
    beta_target: float,
    extra_penalty_cost: float = 0.0,
) -> OptimizationMetrics:
    """评估一组给定 Q/R 的综合动态响应指标与代价标量。"""
    Q = np.diag(q_diag)
    R = np.diag(r_diag)

    try:
        K, _, poles = solve_discrete_lqr(Ad, Bd, Q, R)
    except Exception:
        return OptimizationMetrics(
            q_diag=q_diag, r_diag=r_diag, max_spectral_radius=1.5,
            wheel_sat_duration_s=999.0, max_pitch_deg=999.0, max_displacement_m=999.0,
            final_displacement_m=999.0, beta_settling_time_s=999.0,
            lbar_settling_time_s=999.0, initial_peak_tp_nm=999.0, initial_peak_fl_n=999.0,
            total_cost=1e6
        )

    rho = float(np.max(np.abs(poles)))
    if rho >= 0.9999:
        cost = 1e6
    else:
        dt = 0.001
        sim_time = 2.5
        e_hist, u_hist, sat_time = run_closed_loop_simulation(Ad, Bd, K, e0, sim_time=sim_time, dt=dt)
        
        x_pos = e_hist[:, 0]
        max_disp = float(np.max(np.abs(x_pos)))
        final_disp = float(abs(x_pos[-1]))
        max_theta_deg = float(np.degrees(np.max(np.abs(e_hist[:, 4]))))
        
        # 摆角稳定时间 (< 2.0 deg)
        deg_err_beta = np.abs(np.degrees(e_hist[:, 10]))
        below_beta = np.where(deg_err_beta < 2.0)[0]
        beta_settle_s = float(below_beta[0] * dt) if len(below_beta) > 0 else sim_time

        # 平均腿长稳定时间 (< 5mm)
        lbar_err = np.abs(e_hist[:, 8])
        below_lbar = np.where(lbar_err < 0.005)[0]
        lbar_settle_s = float(below_lbar[0] * dt) if len(below_lbar) > 0 else sim_time

        cost = (
            120.0 * sat_time
            + 40.0 * (max_theta_deg / 10.0)
            + 15.0 * beta_settle_s
            + 2.0 * max_disp
            + 1.0 * final_disp
            + 5.0 * lbar_settle_s
            + extra_penalty_cost
        )

    return OptimizationMetrics(
        q_diag=q_diag,
        r_diag=r_diag,
        max_spectral_radius=rho,
        wheel_sat_duration_s=sat_time,
        max_pitch_deg=max_theta_deg,
        max_displacement_m=max_disp,
        final_displacement_m=final_disp,
        beta_settling_time_s=beta_settle_s,
        lbar_settling_time_s=lbar_settle_s,
        initial_peak_tp_nm=float(u_hist[0, 2]),
        initial_peak_fl_n=float(u_hist[0, 4] + 66.6),
        total_cost=cost,
    )


def optimize_qr_matrix(
    h_target: float = 0.24,
    beta0: float = -1.0,
    L0: float = 0.28,
    max_iter: int = 400,
) -> tuple[np.ndarray, np.ndarray, OptimizationMetrics, OptimizationMetrics, int, int]:
    """主优化算法：在倒立摆层级优先级约束与临界阻尼约束下求解真实连续实数最优解。"""
    beta_target = config.compute_equilibrium_beta(h_target)
    l_bar_target = h_target / np.cos(beta_target)
    lin = linearize_static_trim(
        l_bar_target,
        l_bar_target,
        beta_1=beta_target,
        beta_2=beta_target,
    )
    Ad, Bd = _discretize(lin.A, lin.B)

    # 初始状态误差 e0
    l_bar0 = L0
    e0 = np.zeros(14, dtype=float)
    e0[8] = l_bar0 - l_bar_target
    e0[10] = beta0 - beta_target
    e0[12] = beta0 - beta_target

    # 固定 R 矩阵基准
    r_diag_fixed = config.R_DIAG.copy()

    # 参数化: p = [log(q_theta), log(ratio_x), log(q_lbar), log(q_beta), eta_x, eta_theta, eta_lbar, eta_beta]
    # 其中 Q_x = ratio_x * Q_theta, 强制保证 Q_theta >> Q_x (姿态优先于位移)
    def decode_params(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        qtheta = 10.0**p[0]
        ratio_x = 10.0**p[1]
        qx = ratio_x * qtheta
        qlbar = 10.0**p[2]
        qbeta = 10.0**p[3]
        eta_x, eta_theta, eta_lbar, eta_beta = p[4:]
        
        qdx = eta_x * np.sqrt(qx)
        qdtheta = eta_theta * np.sqrt(qtheta)
        qdlbar = eta_lbar * np.sqrt(qlbar)
        qdbeta = eta_beta * np.sqrt(qbeta)

        q_diag = np.array([
            qx, qdx,
            40.0, 8.0,        # psi, dpsi
            qtheta, qdtheta,  # theta, dtheta
            800.0, 25.0,      # phi, dphi
            qlbar, qdlbar,    # L_bar, dL_bar
            qbeta, qdbeta,    # beta1, dbeta1
            qbeta, qdbeta,    # beta2, dbeta2
        ], dtype=float)

        # 优先级层级与阻尼比软约束:
        # ratio_x (Q_x / Q_theta) 期望在 0.10 ~ 0.15 附近
        ratio_penalty = (p[1] - np.log10(0.12))**2 * 20.0
        damp_penalty = (
            (eta_x - 1.25)**2 +
            (eta_theta - 1.25)**2 +
            (eta_lbar - 1.00)**2 +
            (eta_beta - 0.85)**2
        ) * 8.0
        return q_diag, r_diag_fixed, ratio_penalty + damp_penalty

    def objective(p: np.ndarray) -> float:
        q_diag, r_diag, extra_cost = decode_params(p)
        metrics = evaluate_qr_candidate(q_diag, r_diag, Ad, Bd, e0, h_target, beta_target, extra_penalty_cost=extra_cost)
        return metrics.total_cost

    baseline_q = config.Q_DIAG.copy()
    baseline_r = config.R_DIAG.copy()
    baseline_metrics = evaluate_qr_candidate(baseline_q, baseline_r, Ad, Bd, e0, h_target, beta_target)

    p0 = np.array([
        np.log10(400.0),
        np.log10(0.12),
        np.log10(200.0),
        np.log10(100.0),
        1.25,
        1.25,
        1.00,
        0.85,
    ])

    res = minimize(
        objective,
        p0,
        method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 1e-5, "fatol": 1e-5},
    )

    opt_q, opt_r, _ = decode_params(res.x)
    opt_metrics = evaluate_qr_candidate(opt_q, opt_r, Ad, Bd, e0, h_target, beta_target)
    return opt_q, opt_r, baseline_metrics, opt_metrics, res.nit, res.nfev


def _discretize(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return discretize_state_space(A, B)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=float, default=0.24, help="Target height in meters (default: 0.24).")
    parser.add_argument("--beta0", type=float, default=-1.0, help="Initial swing angle in rad (default: -1.0).")
    parser.add_argument("--l0", type=float, default=0.28, help="Initial leg length in meters (default: 0.28).")
    parser.add_argument("--max-iter", type=int, default=400, help="Max optimizer iterations (default: 400).")
    args = parser.parse_args()

    print("=================================================================")
    beta_target = config.compute_equilibrium_beta(args.height)
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
