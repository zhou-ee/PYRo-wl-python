# 建模包与统一物理配置

`config.py` 是整机物理参数的唯一维护入口。`RobotParameters` 保存 SI 单位的质量、惯量、质心、轮腿几何、CAD 质心曲线、气弹簧和执行器极限；`DEFAULT_ROBOT` 是当前机器人的默认快照。包初始化只声明版本。

配置使用嵌套 tuple，避免修改 ndarray 导致缓存与参数漂移。新实验用 `dataclasses.replace(DEFAULT_ROBOT, body_mass=...)` 创建快照，再通过 `parameters=` 传给模型和调度。`snapshot()`/`fingerprint()` 用于产物追溯。地面和腾空共用一个惯量张量，腾空 Iyy 与质心 x/z 是派生属性。

实际质量及其他数值直接读取 `config.py`，文档不维护另一份活动参数。机体坐标轴延续原模型，pitch 为 y 轴；质心相对髋中心，惯量在模型中作为机体质心转动惯量使用。CAD 文件没有提供可自动验证惯量参考点的元数据，因此迁移保留原有参考点解释，修改 CAD 参数时须核对其来源。

当前机体质量分布前后、左右对称，机体坐标轴沿对称方向，因此质心惯量中的 `Ixy/Ixz/Iyz` 均为零。原 CAD 导出的非零 `Ixz` 已按导出误差移除。对角惯量仍读取配置；腿部质心曲线、非直立姿态的惯性耦合和关节运动学耦合由各模型正常计算，不能将整个质量矩阵的非对角项清零。重新运行计算入口后，导出自动使用修正后的参数快照和指纹；已生成的旧文件需重新生成。

Q/R 和状态顺序分别维护在 `models/ground/config.py`、`models/airborne/config.py`。公共物理配置不保存控制权重、RDOB 带宽和拟合阶数。

十状态地面模型的 Q/R 和采样周期独立维护在 `models/ground10/config.py`；物理参数及前后质心开关继续读取本层 `config.py`。新增入口和平均髋高局部近似见 [十状态组件 README](models/ground10/README.md)。

`leg_com_fore_aft_enabled=True` 启用前后质心曲线。改为 `False` 后，派生属性 `effective_leg_com_x_poly` 返回零系数，所有模型统一使用零前后偏移，原始 `leg_com_x_poly` 保留。沿腿的 `leg_com_lwp_poly` 不受开关影响，仍随腿长变化。开关参与参数快照、指纹和地面编译缓存键；切换后需重新启动计算、重新生成产物。

## 依赖边界

依赖 NumPy；下层组件读取此配置，配置不反向导入模型、调度或 CLI。

## 使用示例

```python
from dataclasses import replace
from wheel_leg_lqr.config import DEFAULT_ROBOT
from wheel_leg_lqr.models.ground.model import GroundModel
robot = replace(DEFAULT_ROBOT, body_mass=13.0)
plant = GroundModel(robot).linearize(0.24, 0.32)
print(plant.A.shape, plant.B.shape)
```

## 文件与接口

| 文件 | 主要接口 |
|---|---|
| [config.py](config.py) | `RobotParameters` |
