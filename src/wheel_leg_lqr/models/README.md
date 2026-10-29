# 物理模型

`ground/` 是双支撑七坐标模型，`airborne/` 是无接触平面模型，`contact/` 扩展为九坐标释放与碰撞模型。三者共享同一份 `RobotParameters`，按各自约束建立方程。

新增 [ground10/](ground10/README.md)：独立五坐标、十状态、四力矩输入模型，L1/L2 在能量求导前固定，无 roll/伸缩自由度。它采用平均髋高的局部近似，不读取十四状态几何、质量矩阵或 A/B；具体适用范围和耦合边界见其 README。统一物理配置与质心开关继续共用。

`leg_geometry.py` 提供 `leg_com()`、`compute_gas_spring_force()`、`active_equilibrium_input()`。地面/接触的杆惯量估计沿用原先四位小数的质心近似，但系数从公共曲线派生；腾空使用完整曲线精度，保持原数值行为。

气弹簧由下位机抵消，当前模型 A/B/K 继续使用总虚拟腿力，导出的主动前馈为 `total - gas`。

三类模型统一读取 `parameters.effective_leg_com_x_poly`。统一配置的 `leg_com_fore_aft_enabled=False` 时，前后偏移及其腿长导数均为零，保留沿腿质心曲线；腿质心在矢状平面的髋—轮连线上。此开关也改变动力学和控制增益，不能只替换平衡腿角。左右轮距及髋间距仍由各自几何参数定义。

## 依赖边界

依赖公共物理配置和 NumPy/SymPy，不导入调度、导出或命令行。

## 使用示例

```python
from wheel_leg_lqr.models.leg_geometry import compute_gas_spring_force
print(compute_gas_spring_force(0.28))
```

## 文件与接口

| 文件 | 主要接口 |
|---|---|
| [leg_geometry.py](leg_geometry.py) | `leg_com`, `ground_inertia_com_coefficients`, `compute_gas_spring_force`, `active_equilibrium_input` |
