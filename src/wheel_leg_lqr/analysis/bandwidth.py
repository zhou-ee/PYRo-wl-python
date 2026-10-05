from __future__ import annotations
from ..models.ground.trim import compute_equilibrium_beta
import numpy as np
from ..models.ground import config
from ..models.ground.model import linearize_static_trim
from ..control.lqr import discretize_state_space, solve_discrete_lqr


def analyze_controller_bandwidth(
    h_target: float = config.ANALYSIS_HEIGHT,
    q_diag: np.ndarray | None = None,
    r_diag: np.ndarray | None = None,
) -> dict:
    """计算 LQR 控制器在各物理通道上的模态极点与等效闭环带宽。"""
    if q_diag is None:
        q_diag = config.Q_DIAG
    if r_diag is None:
        r_diag = config.R_DIAG

    beta_target = compute_equilibrium_beta(h_target)
    l_bar_target = h_target / np.cos(beta_target)
    lin = linearize_static_trim(
        l_bar_target,
        l_bar_target,
        beta_1=beta_target,
        beta_2=beta_target,
    )
    A, B = lin.A, lin.B
    dt = config.SAMPLE_TIME
    Ad, Bd = discretize_state_space(A, B, sample_time=dt)

    Q = np.diag(q_diag)
    R = np.diag(r_diag)
    K, _, poles_d = solve_discrete_lqr(Ad, Bd, Q, R)

    # 离散极点映射为连续 s 平面等效极点: s = ln(z) / dt
    poles_s = np.log(poles_d) / dt

    # 模态右特征向量
    _, eig_vecs = np.linalg.eig(Ad - Bd @ K)

    state_names = [
        "x (位移)", "dx (速度)",
        "psi (偏航)", "dpsi (偏航角速度)",
        "theta (俯仰)", "dtheta (俯仰角速度)",
        "phi (横滚)", "dphi (横滚角速度)",
        "L_bar (平均腿长)", "dL_bar (平均腿长速度)",
        "beta1 (左腿摆角)", "dbeta1 (左腿角速度)",
        "beta2 (右腿摆角)", "dbeta2 (右腿角速度)",
    ]

    modal_info = []
    for i in range(len(poles_s)):
        p = poles_s[i]
        v = np.abs(eig_vecs[:, i])
        dom_idx = int(np.argmax(v))
        dom_state = state_names[dom_idx]
        wn = float(np.abs(p))
        zeta = float(-p.real / wn) if wn > 1e-6 else 0.0
        fn = wn / (2 * np.pi)

        # 二阶闭环系统 -3dB 截止带宽
        if zeta <= 1.0 / np.sqrt(2):
            wbw = wn * np.sqrt(1.0 - 2.0 * zeta**2 + np.sqrt(4.0 * zeta**4 - 4.0 * zeta**2 + 2.0))
        else:
            wbw = wn * (1.0 / (1.0 + zeta)) * 1.5
        fbw = wbw / (2 * np.pi)

        modal_info.append({
            "pole": p,
            "dom_state": dom_state,
            "dom_idx": dom_idx,
            "wn_rad_s": wn,
            "fn_hz": fn,
            "zeta": zeta,
            "bw_rad_s": wbw,
            "bw_hz": fbw,
        })

    # 按物理通道归纳聚合主导带宽
    channel_groups = {
        "髋关节摆腿环 (Leg Swing beta1/beta2)": [10, 11, 12, 13],
        "机身横滚平衡环 (Chassis Roll phi)": [6, 7],
        "机身俯仰平衡环 (Chassis Pitch theta)": [4, 5],
        "机身偏航转向环 (Chassis Yaw psi)": [2, 3],
        "平均腿长调节环 (Mean Leg Length L_bar)": [8, 9],
        "水平位移巡航环 (Position / Velocity x)": [0, 1],
    }

    summary = {}
    for ch_name, indices in channel_groups.items():
        matched = [m for m in modal_info if m["dom_idx"] in indices]
        if matched:
            fastest = max(matched, key=lambda m: m["wn_rad_s"])
            summary[ch_name] = fastest

    return {
        "modal_info": modal_info,
        "channel_summary": summary,
        "K": K,
        "poles_d": poles_d,
        "poles_s": poles_s,
    }
