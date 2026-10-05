# 地面双支撑模型

`config.py`：14 状态、6 输入、地面 Q/R、1 ms 采样和饱和恢复初始用例。`GroundSettings` 可保存一次实验的权重与采样周期。

`model.py`：几何、质量/重力/输入矩阵、静态配平和 A/B；`GroundModel(parameters)` 是实例入口，亦保留显式 `parameters=` 的函数接口。符号编译缓存以完整不可变物理参数为键。

`trim.py`：`compute_equilibrium_beta_from_length()`、高度配平角和 `fit_equilibrium_beta()`。固件平衡角曲线从当前物理参数重新拟合，避免与质量修改不同步。

状态：`x,dx,psi,dpsi,theta,dtheta,phi,dphi,l_bar,dl_bar,beta_1,dbeta_1,beta_2,dbeta_2`。输入：`T_w1,T_w2,T_p1,T_p2,f_l1,f_l2`。位移/腿长 m，角度 rad，力矩 N·m，腿力 N。

虚拟腿力是腿长坐标的共轭力，不等于整机地面支撑力；轮高度固定，腿质心随伸缩的运动比例由 CAD 曲线决定。

## 依赖边界

读取公共参数、状态定义与配平函数；物理模型不读取 Q/R，不依赖控制设计。

## 使用示例

```python
from wheel_leg_lqr.models.ground.model import GroundModel
plant = GroundModel().linearize(0.24, 0.32)
print(plant.mass.shape, plant.equilibrium_input)
```

## 文件与接口

| 文件 | 主要接口 |
|---|---|
| [config.py](config.py) | `make_q`, `make_r`, `default_initial_errors`, `GroundSettings` |
| [model.py](model.py) | `GeometryAtTrim`, `StaticLinearization`, `evaluate_mass`, `evaluate_gravity`, `evaluate_input_matrix`, `evaluate_lengths`, `evaluate_height`, `evaluate_wheel_positions_heading`, `evaluate_wheel_velocity_maps_heading`, `make_trim_geometry`, `solve_equilibrium_input`, `linearize_static_trim`, `GroundModel` |
| [trim.py](trim.py) | `compute_equilibrium_beta_from_length`, `compute_equilibrium_beta`, `fit_equilibrium_beta` |
