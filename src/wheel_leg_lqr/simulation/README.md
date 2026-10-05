# 闭环与接触仿真

`saturation.py` 支持任意维数单例/批量饱和闭环时间推进，显式接收采样周期、执行器上下限和初始扰动用例。绝对输入先限幅，再减去配平输入进入扰动模型。

`step_drop.py` 的 `simulate_step_drop(parameters=...)` 组织下降腿释放、低台阶碰撞、机械限位和 LQR/RDOB 恢复；力学来自 `models/contact`，控制器来自调度层。`target_lengths_for_step()` 求给定台阶下保持平均腿长的双腿长度。

地面十四状态的默认扰动用例维护在 `models/ground/config.py`。

## 依赖边界

依赖模型、控制与调度，不导入命令行或文件导出。

## 使用示例

```python
from wheel_leg_lqr.simulation.step_drop import target_lengths_for_step
print(target_lengths_for_step(0.18))
```

## 文件与接口

| 文件 | 主要接口 |
|---|---|
| [saturation.py](saturation.py) | `SaturationValidation`, `simulate_saturated_closed_loop_batch`, `simulate_saturated_closed_loop`, `validate_saturated_closed_loop` |
| [step_drop.py](step_drop.py) | `StepDropResult`, `target_lengths_for_step`, `simulate_step_drop` |
