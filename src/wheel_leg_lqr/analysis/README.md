# 离线分析

`symbolic.py`：符号能控性结构和 PBH 分析。`bandwidth.py`：闭环模态与带宽。`qr_optimizer.py`：地面 Q/R 离线优化和候选时域评价。`airborne_fl_pitch.py`：腾空 FL 对 pitch 的瞬时影响。

分析使用公共硬件参数及对应模型控制设置，计算结果由 CLI 展示。QR 优化不自动回写默认权重；需要采用结果时明确更新地面配置。

## 依赖边界

依赖模型、控制算法及科学计算库；不导入 CLI。

## 使用示例

```python
from wheel_leg_lqr.analysis.airborne_fl_pitch import analyze_leg_force_pitch
print(analyze_leg_force_pitch().pitch_acceleration_per_newton)
```

## 文件与接口

| 文件 | 主要接口 |
|---|---|
| [airborne_fl_pitch.py](airborne_fl_pitch.py) | `analyze_leg_force_pitch` |
| [bandwidth.py](bandwidth.py) | `analyze_controller_bandwidth` |
| [qr_optimizer.py](qr_optimizer.py) | `OptimizationMetrics`, `run_closed_loop_simulation`, `evaluate_qr_candidate`, `optimize_qr_matrix` |
| [symbolic.py](symbolic.py) | `SymbolTable`, `build_symbols`, `build_nominal_input_matrix`, `build_state_space`, `controllability_matrix`, `second_order_pbh_matrix`, `generic_witness` |
