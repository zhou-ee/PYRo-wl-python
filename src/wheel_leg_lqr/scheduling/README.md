# 腿长调度与拟合

`domain.py` 提供网格与腿长归一化，`polynomial.py` 提供张量积/总阶三次基础运算。

`ground_lqr.py`：单点地面控制器与增益调度，结果节点保存物理参数和 `GroundSettings`，验证默认选取 25 个代表节点做饱和恢复。

`airborne_lqr.py`：形状 LQR、双腿长增益拟合、网格及单元中心 float32 验证。结果保存物理参数、实际 Q/R 和采样时间。独立系数求值可显式提供 `lower/upper`。

`config.py`：RDOB 默认极点、六通道带宽及扰动限幅比例。`rdob_node.py`：六通道单点设计；`rdob.py`：Gamma 张量三次、Bq 总阶三次、系数更新/重基准化和冻结点闭环检查。RDOB 的 q0/u0 由调用者提供，默认扰动限幅根据调度保存的硬件参数推导，避免与全局默认机器人混用。

各函数只计算与验证，不写文件、不解析 CLI。节点/调度结果交给 `artifacts` 导出。

## 依赖边界

依赖模型、控制算法、公共拟合及必要的饱和仿真；不反向依赖 CLI 或导出器。

## 使用示例

```python
from wheel_leg_lqr.scheduling.ground_lqr import compute_gain_node
from wheel_leg_lqr.scheduling.airborne_lqr import compute_airborne_gain_node
ground = compute_gain_node(0.24, 0.32)
air = compute_airborne_gain_node(0.24, 0.32)
print(ground.gain.shape, air.gain.shape)
```

## 文件与接口

| 文件 | 主要接口 |
|---|---|
| [config.py](config.py) | RDOB 默认极点、六通道带宽、扰动限幅比例 |
| [airborne_lqr.py](airborne_lqr.py) | `AirborneGainNode`, `AirborneScheduleFit`, `make_leg_length_grid`, `compute_airborne_gain_node`, `evaluate_airborne_gain`, `compute_airborne_schedule` |
| [domain.py](domain.py) | `make_leg_length_grid`, `LengthDomain` |
| [ground_lqr.py](ground_lqr.py) | `GainNode`, `GainScheduleFit`, `compute_gain_node`, `compute_active_equilibrium_input`, `evaluate_fitted_gain`, `compute_gain_schedule` |
| [polynomial.py](polynomial.py) | `fit_total_cubic_coefficients`, `evaluate_total_cubic_polynomial`, `tensor_design_matrix`, `fit_tensor_coefficients`, `evaluate_tensor_polynomial` |
| [rdob.py](rdob.py) | `DirectPolynomialFit`, `RdobSchedulePoint`, `RdobSchedule`, `frozen_closed_loop_matrix`, `compute_rdob_schedule`, `step_scheduled_rdob`, `runtime_accounting` |
| [rdob_node.py](rdob_node.py) | `RdobNode`, `compute_rdob_node`, `rdob_diagnostics` |
