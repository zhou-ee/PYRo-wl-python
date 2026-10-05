# 腾空双髋模型

`config.py` 保存六状态顺序、两输入、独立 Q/R 与采样周期，`FULL_STATE_INDICES` 给出固件十四状态接线映射。

`model.py` 的 `linearize_airborne()` 只生成物理 `AirbornePlant`。`airborne_generalized_mass()` 支持含腿长自由度的五坐标无接触质量矩阵；`compute_leg_force_pitch_effect()` 保留 FL 对 pitch 的瞬时影响分析。

`reduction.py` 的 `build_shape_model()` 在物理模型上构造四维可控形状空间、角动量守恒约束和与 Q 相关的加权投影。共享质量、惯量和质心，不读取地面的 Q/R。

状态为 `beta_1,dbeta_1,beta_2,dbeta_2,theta,dtheta`，输入 `T_p1,T_p2`。完整六状态有两个中性转动模态，验证四维形状闭环稳定性。

## 依赖边界

依赖公共参数、NumPy和内部配置；地面模型矩阵不参与腾空推导。

## 使用示例

```python
from wheel_leg_lqr.models.airborne.model import linearize_airborne
from wheel_leg_lqr.models.airborne.reduction import build_shape_model
plant = linearize_airborne(0.24, 0.32)
shape = build_shape_model(0.24, 0.32)
print(plant.A.shape, shape.A_shape.shape)
```

## 文件与接口

| 文件 | 主要接口 |
|---|---|
| [config.py](config.py) | `make_q`, `make_r` |
| [model.py](model.py) | `airborne_generalized_mass`, `airborne_rotational_mass`, `LegForcePitchEffect`, `compute_leg_force_pitch_effect`, `AirbornePlant`, `linearize_airborne` |
| [reduction.py](reduction.py) | `AirborneLinearization`, `build_shape_model` |
