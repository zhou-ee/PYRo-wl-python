# 释放与碰撞接触模型

`config.py` 的 `ContactSettings` 保存地面/限位刚度、阻尼和积分子步默认值。硬件质量和腿长限位来自公共物理配置。

`model.py` 扩展七坐标为 `x,psi,theta,phi,l_bar,beta_1,beta_2,kappa,z_c`。`_mass_and_input()` 计算质量、重力、输入映射与运动学；`_compliant_acceleration()` 加入单边地面/机械限位弹簧阻尼力。这些内部接口由 `simulation/step_drop.py` 调用。

保留原模型的速度二次项缺省约定。接触系数仍需实物辨识，模型用于接触过渡筛查。

## 依赖边界

依赖公共参数、公共腿几何与地面输入顺序；不依赖控制器、调度器或仿真时间推进。

## 使用示例

```python
from wheel_leg_lqr.models.contact.config import DEFAULT_CONTACT
print(DEFAULT_CONTACT)
```

## 文件与接口

| 文件 | 主要接口 |
|---|---|
| [config.py](config.py) | `ContactSettings` |
| [model.py](model.py) |  |
