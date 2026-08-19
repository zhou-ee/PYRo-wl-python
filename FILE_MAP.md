# 重构映射

| 模块 | 职责 |
|---|---|
| `src/wheel_leg_lqr/core/config.py` | 物理参数、状态/输入维度、权重与配平计算 |
| `src/wheel_leg_lqr/core/model.py` | 直接基于 `L1/L2` 的几何、质量矩阵与静态线性化 |
| `src/wheel_leg_lqr/control/lqr.py` | 离散化、LQR 和数值可控性公共算法 |
| `src/wheel_leg_lqr/control/saturation.py` | 执行器饱和仿真和验证 |
| `src/wheel_leg_lqr/scheduling/leg_length_schedule.py` | 项目唯一的 `L1/L2` 张量积增益拟合与产物导出 |
| `src/wheel_leg_lqr/scheduling/polynomial.py` | L1/L2 归一化张量积多项式基础运算 |
| `src/wheel_leg_lqr/analysis/symbolic.py` | 符号能控性分析 |
| `src/wheel_leg_lqr/analysis/bandwidth.py` | 闭环模态与带宽分析 |
| `src/wheel_leg_lqr/analysis/qr_optimizer.py` | QR 权重离线寻优 |
| `src/wheel_leg_lqr/artifacts/io.py` | 生成物写盘工具 |

`src/wheel_leg_lqr/gain_schedule.py` 是正式公共入口；不再提供只基于腿长差值的调度模块，也不使用带有二维后缀的命名。
