# 通用控制算法

`lqr.py`：`discretize_state_space(A,B,sample_time=...)`、`controllability_rank()`、`solve_discrete_lqr(Ad,Bd,Q,R)`。Q/R 和采样时间必须显式提供，支持不同状态与输入维数。

`rdob.py`：`design_rdob()`、初始化、扰动重构、连续导数与梯形离散步进。维数从质量/输入矩阵推断；标量速率求梯形常量时须提供 `channel_count`，向量速率直接按其长度处理。

RDOB 的输入是区间实际施加的总输入减配平总输入，更新使用两端 q/v 测量。不隐式插入补偿开关或限幅，六通道部署策略在调度层。

## 依赖边界

只依赖 NumPy、SciPy、python-control；不导入机器人配置或具体模型。

## 使用示例

```python
import numpy as np
from wheel_leg_lqr.control.lqr import discretize_state_space, solve_discrete_lqr
ad, bd = discretize_state_space(np.zeros((2, 2)), np.eye(2), sample_time=0.001)
gain, cost, poles = solve_discrete_lqr(ad, bd, np.eye(2), np.eye(2))
```

## 文件与接口

| 文件 | 主要接口 |
|---|---|
| [lqr.py](lqr.py) | `discretize_state_space`, `controllability_rank`, `solve_discrete_lqr` |
| [rdob.py](rdob.py) | `RdobUpdateCoefficients`, `trapezoidal_step_constants`, `RdobDesign`, `design_rdob`, `reconstruct_disturbance`, `initialize_rdob`, `rdob_derivative`, `step_rdob` |
