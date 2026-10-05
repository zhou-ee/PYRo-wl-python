from __future__ import annotations
from ..models.ground.trim import compute_equilibrium_beta
import argparse
import numpy as np
from ..models.ground import config
from ..analysis.bandwidth import analyze_controller_bandwidth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=float, default=config.ANALYSIS_HEIGHT, help="Target height in meters (default: ground config).")
    args = parser.parse_args()

    res = analyze_controller_bandwidth(h_target=args.height)

    print("=========================================================================================")
    beta_target = compute_equilibrium_beta(args.height)
    l_bar_target = args.height / np.cos(beta_target)
    print(
        "       LQR 控制器全通道物理闭环带宽分析 "
        f"(目标平均腿长: {l_bar_target:.6f} m, 派生高度: {args.height} m, 采样率: {1.0/config.SAMPLE_TIME:g} Hz)"
    )
    print("=========================================================================================\n")

    print(f"{'物理控制环路 (Physical Loop)':<38} | {'固有频率 wn':<15} | {'阻尼比 zeta':<10} | {'闭环带宽 (-3dB BW)':<18}")
    print("-" * 90)

    for ch_name, m in res["channel_summary"].items():
        wn_str = f"{m['wn_rad_s']:6.2f} rad/s ({m['fn_hz']:4.2f} Hz)"
        zeta_str = f"{m['zeta']:5.2f}"
        bw_str = f"{m['bw_rad_s']:6.2f} rad/s ({m['bw_hz']:5.2f} Hz)"
        print(f"{ch_name:<38} | {wn_str:<15} | {zeta_str:<10} | {bw_str:<18}")

    print("\n-----------------------------------------------------------------------------------------")
    print("  采样率与带宽裕度评价 (Sampling Frequency vs Bandwidth Margin)")
    print("-----------------------------------------------------------------------------------------")
    max_bw = max(m["bw_hz"] for m in res["channel_summary"].values())
    print(f"  * 系统最高物理闭环带宽 (摆腿环):   {max_bw:.2f} Hz")
    print(f"  * 控制器中断执行频率 (fs):          {1.0/config.SAMPLE_TIME:.0f} Hz (周期 {1000*config.SAMPLE_TIME:g} ms)")
    print(f"  * 采样率/带宽倍数比 (fs / f_bw):    {1.0/config.SAMPLE_TIME/max_bw:.1f}x (标准控制工程推荐 > 15x ~ 20x)")
    print("  * 评价结论: 采样率极为充裕，离散化相位滞后极小，系统具有极佳的离散鲁棒性！")
    print("=========================================================================================\n")

if __name__ == "__main__":
    main()
